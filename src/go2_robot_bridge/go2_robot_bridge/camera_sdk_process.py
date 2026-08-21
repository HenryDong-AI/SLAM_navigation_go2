"""Parent-side process manager for the non-ROS Unitree camera worker."""

import socket
import subprocess
import sys
import threading
import time
from typing import Dict, Optional, Tuple

from .camera_sdk_worker import PACKET_FRAME, PACKET_STATUS, receive_packet


class CameraSdkProcess:
    """Keep Unitree's CycloneDDS binding outside the ROS 2 process.

    The newest frame replaces older frames so a slow ROS publisher never builds
    an unbounded video queue.  All child data is bounded by the worker protocol.
    """

    def __init__(
        self,
        interface: str,
        sdk_python_path: str,
        cyclonedds_python_path: str,
        timeout_sec: float,
        rate_hz: float,
        reconnect_delay_sec: float,
        max_failures: int,
    ) -> None:
        self._arguments: Dict[str, object] = {
            "interface": str(interface),
            "sdk_python_path": str(sdk_python_path),
            "cyclonedds_python_path": str(cyclonedds_python_path),
            "timeout_sec": max(0.1, float(timeout_sec)),
            "rate_hz": max(0.2, min(float(rate_hz), 15.0)),
            "reconnect_delay_sec": max(0.1, float(reconnect_delay_sec)),
            "max_failures": max(1, int(max_failures)),
        }
        if not self._arguments["interface"]:
            raise ValueError("camera network interface must not be empty")
        self._lock = threading.Lock()
        self._process: Optional[subprocess.Popen] = None
        self._connection: Optional[socket.socket] = None
        self._reader: Optional[threading.Thread] = None
        self._sequence = 0
        self._latest_frame: Optional[Tuple[int, int, bytes, int]] = None
        self._status_sequence = 0
        self._latest_status: Optional[Tuple[int, int, str]] = None
        self._reader_error = ""
        self._generation = 0

    @property
    def running(self) -> bool:
        process = self._process
        return process is not None and process.poll() is None

    @property
    def returncode(self) -> Optional[int]:
        process = self._process
        return None if process is None else process.poll()

    @property
    def reader_error(self) -> str:
        with self._lock:
            return self._reader_error

    @property
    def reader_alive(self) -> bool:
        reader = self._reader
        return reader is not None and reader.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self.close()
        parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        child.set_inheritable(True)
        arguments = self._arguments
        command = [
            sys.executable,
            "-m",
            "go2_robot_bridge.camera_sdk_worker",
            "--ipc-fd",
            str(child.fileno()),
            "--interface",
            str(arguments["interface"]),
            "--sdk-python-path",
            str(arguments["sdk_python_path"]),
            "--cyclonedds-python-path",
            str(arguments["cyclonedds_python_path"]),
            "--timeout-sec",
            str(arguments["timeout_sec"]),
            "--rate-hz",
            str(arguments["rate_hz"]),
            "--reconnect-delay-sec",
            str(arguments["reconnect_delay_sec"]),
            "--max-failures",
            str(arguments["max_failures"]),
        ]
        try:
            process = subprocess.Popen(
                command,
                pass_fds=(child.fileno(),),
                close_fds=True,
                stdin=subprocess.DEVNULL,
            )
        except Exception:
            parent.close()
            child.close()
            raise
        child.close()
        self._process = process
        self._connection = parent
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._latest_frame = None
            self._latest_status = None
            self._reader_error = ""
        self._reader = threading.Thread(
            target=self._read_loop,
            args=(parent, generation),
            name="go2-camera-sdk-reader",
            daemon=True,
        )
        self._reader.start()

    def _read_loop(self, connection: socket.socket, generation: int) -> None:
        try:
            while True:
                packet = receive_packet(connection)
                if packet is None:
                    with self._lock:
                        if generation == self._generation:
                            self._reader_error = "camera worker socket closed"
                    return
                kind, capture_ns, payload = packet
                received_ns = time.monotonic_ns()
                with self._lock:
                    if generation != self._generation:
                        return
                    if kind == PACKET_FRAME:
                        self._sequence += 1
                        self._latest_frame = (
                            self._sequence,
                            int(capture_ns),
                            payload,
                            received_ns,
                        )
                    elif kind == PACKET_STATUS:
                        self._status_sequence += 1
                        self._latest_status = (
                            self._status_sequence,
                            received_ns,
                            payload.decode("utf-8", errors="replace"),
                        )
        except (OSError, ValueError) as error:
            with self._lock:
                if generation == self._generation:
                    self._reader_error = str(error)

    def latest_frame_after(
        self, sequence: int
    ) -> Optional[Tuple[int, int, bytes, int]]:
        with self._lock:
            frame = self._latest_frame
            if frame is None or frame[0] <= int(sequence):
                return None
            return frame

    def latest_status_after(
        self, sequence: int
    ) -> Optional[Tuple[int, int, str]]:
        with self._lock:
            status = self._latest_status
            if status is None or status[0] <= int(sequence):
                return None
            return status

    def close(self, timeout_sec: float = 1.0) -> None:
        with self._lock:
            # Invalidate the old reader before closing its socket. This also
            # prevents any pre-restart frame from being observed later.
            self._generation += 1
            self._latest_frame = None
            self._latest_status = None
            self._reader_error = ""
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                connection.close()
            except OSError:
                pass
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=max(0.0, float(timeout_sec)))
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=max(0.2, float(timeout_sec)))
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
        reader, self._reader = self._reader, None
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=max(0.0, float(timeout_sec)))


def ros_stamp_from_monotonic_capture(
    ros_now_ns: int,
    monotonic_now_ns: int,
    capture_monotonic_ns: int,
    offset_ns: int = 0,
) -> int:
    """Translate a child monotonic timestamp into the parent's ROS clock."""
    age_ns = int(monotonic_now_ns) - int(capture_monotonic_ns)
    return max(1, int(ros_now_ns) - max(0, age_ns) + int(offset_ns))
