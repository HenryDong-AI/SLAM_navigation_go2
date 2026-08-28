"""Private, bounded RPC transport for the isolated Unitree motion worker.

This module deliberately imports only the Python standard library.  In
particular, importing it in the ROS process cannot load CycloneDDS or the
Unitree SDK.
"""

import json
import math
import os
import site
import socket
import struct
import subprocess
import sys
import threading
from typing import Any, Callable, Dict, Optional, Tuple


MAX_FRAME_BYTES = 16 * 1024
_HEADER = struct.Struct("!I")

DEFAULT_SDK_PYTHON_PATH = "/home/unitree/Documents/demov1/unitree_sdk2_python"
DEFAULT_CYCLONEDDS_PYTHON_PATH = os.environ.get(
    "GO2_CYCLONEDDS_PYTHON", site.getusersitepackages()
)


class MotionRpcError(RuntimeError):
    """Base class for worker startup, transport, and remote SDK failures."""


class MotionProtocolError(MotionRpcError):
    """The private peer violated the length-framed JSON protocol."""


class MotionSdkError(MotionRpcError):
    """The worker reached the SDK but the requested operation failed."""

    def __init__(self, message: str, code: Optional[int] = None) -> None:
        super().__init__(message)
        self.code = code


def _json_bytes(payload: Dict[str, Any]) -> bytes:
    try:
        encoded = json.dumps(
            payload, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MotionProtocolError("payload is not finite JSON: {}".format(error))
    if not encoded or len(encoded) > MAX_FRAME_BYTES:
        raise MotionProtocolError(
            "JSON frame length {} is outside 1..{}".format(
                len(encoded), MAX_FRAME_BYTES
            )
        )
    return encoded


def send_frame(peer: socket.socket, payload: Dict[str, Any]) -> None:
    """Send one size-prefixed JSON object, rejecting unbounded payloads."""

    encoded = _json_bytes(payload)
    peer.sendall(_HEADER.pack(len(encoded)) + encoded)


def _recv_exact(peer: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = peer.recv(remaining)
        if not chunk:
            if remaining == size:
                raise EOFError("motion worker socket closed")
            raise MotionProtocolError("motion worker closed during a frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def recv_frame(peer: socket.socket) -> Dict[str, Any]:
    """Receive one bounded size-prefixed JSON object."""

    (length,) = _HEADER.unpack(_recv_exact(peer, _HEADER.size))
    if length < 1 or length > MAX_FRAME_BYTES:
        raise MotionProtocolError(
            "peer frame length {} is outside 1..{}".format(
                length, MAX_FRAME_BYTES
            )
        )
    encoded = _recv_exact(peer, length)
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise MotionProtocolError("peer sent invalid JSON: {}".format(error))
    if not isinstance(payload, dict):
        raise MotionProtocolError("peer JSON root must be an object")
    return payload


def validate_move_arguments(arguments: Any) -> Tuple[float, float, float]:
    """Validate the only numeric command accepted by the worker."""

    if not isinstance(arguments, list) or len(arguments) != 3:
        raise MotionProtocolError("Move requires exactly three arguments")
    values = []
    for value in arguments:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MotionProtocolError("Move arguments must be real numbers")
        converted = float(value)
        if not math.isfinite(converted):
            raise MotionProtocolError("Move arguments must be finite")
        values.append(converted)
    return values[0], values[1], values[2]


class MotionWorkerProxy:
    """Synchronous, fail-closed owner of one non-ROS worker subprocess."""

    def __init__(
        self,
        peer: socket.socket,
        process: Any,
        rpc_timeout_sec: float,
        reap_timeout_sec: float,
        controller: str = "unknown",
        posture_prepared: bool = False,
        body_height: float = 0.0,
        sport_mode: int = -1,
        gait_type: int = -1,
        sport_error_code: int = -1,
    ) -> None:
        self._peer: Optional[socket.socket] = peer
        self._process = process
        self._rpc_timeout = max(0.05, float(rpc_timeout_sec))
        self._reap_timeout = max(0.05, float(reap_timeout_sec))
        self._next_id = 1
        self._lock = threading.Lock()
        self._controller = str(controller)
        self._posture_prepared = bool(posture_prepared)
        self._body_height = float(body_height)
        self._sport_mode = int(sport_mode)
        self._gait_type = int(gait_type)
        self._sport_error_code = int(sport_error_code)

    @classmethod
    def start(
        cls,
        *,
        network_interface: str,
        sdk_timeout_sec: float,
        controller_rpc_timeout_sec: float = 5.0,
        controller_transition_timeout_sec: float = 15.0,
        auto_prepare_posture: bool = True,
        sport_state_timeout_sec: float = 5.0,
        standing_min_body_height: float = 0.25,
        startup_timeout_sec: float,
        rpc_timeout_sec: float,
        reap_timeout_sec: float,
        sdk_python_path: str = DEFAULT_SDK_PYTHON_PATH,
        cyclonedds_python_path: str = DEFAULT_CYCLONEDDS_PYTHON_PATH,
        require_noshm_runtime: bool = True,
        noshm_library_fragment: str = "install_noshm/lib",
        popen_factory: Callable[..., Any] = subprocess.Popen,
        socketpair_factory: Callable[[], Tuple[socket.socket, socket.socket]] = socket.socketpair,
    ) -> "MotionWorkerProxy":
        parent_peer, child_peer = socketpair_factory()
        process = None
        try:
            child_fd = child_peer.fileno()
            command = [
                sys.executable,
                "-m",
                "go2_robot_bridge.motion_worker",
                "--fd",
                str(child_fd),
                "--network-interface",
                str(network_interface),
                "--sdk-timeout-sec",
                str(max(0.05, float(sdk_timeout_sec))),
                "--controller-rpc-timeout-sec",
                str(max(0.1, float(controller_rpc_timeout_sec))),
                "--controller-transition-timeout-sec",
                str(max(0.1, float(controller_transition_timeout_sec))),
                "--sport-state-timeout-sec",
                str(max(0.1, float(sport_state_timeout_sec))),
                "--standing-min-body-height",
                str(max(0.01, float(standing_min_body_height))),
                "--sdk-python-path",
                str(sdk_python_path),
                "--cyclonedds-python-path",
                str(cyclonedds_python_path),
                "--noshm-library-fragment",
                str(noshm_library_fragment),
            ]
            if auto_prepare_posture:
                command.append("--auto-prepare-posture")
            if require_noshm_runtime:
                command.append("--require-noshm-runtime")
            process = popen_factory(
                command,
                close_fds=True,
                pass_fds=(child_fd,),
                stdin=subprocess.DEVNULL,
            )
            child_peer.close()
            parent_peer.settimeout(max(0.05, float(startup_timeout_sec)))
            ready = recv_frame(parent_peer)
            if ready.get("type") != "ready" or not isinstance(
                ready.get("ok"), bool
            ):
                raise MotionProtocolError("worker sent an invalid startup response")
            if not ready["ok"]:
                raise MotionRpcError(
                    "motion worker startup failed: {}".format(
                        str(ready.get("error", "unknown error"))[:512]
                    )
                )
            controller = ready.get("controller")
            if controller not in ("mcf", "sport_mode"):
                raise MotionProtocolError(
                    "worker sent an invalid motion controller"
                )
            posture_prepared = ready.get("posture_prepared")
            body_height = ready.get("body_height")
            sport_mode = ready.get("sport_mode")
            gait_type = ready.get("gait_type")
            sport_error_code = ready.get("sport_error_code")
            if not isinstance(posture_prepared, bool):
                raise MotionProtocolError(
                    "worker sent invalid posture-prepared status"
                )
            if (
                isinstance(body_height, bool)
                or not isinstance(body_height, (int, float))
                or not math.isfinite(float(body_height))
                or float(body_height) < 0.0
            ):
                raise MotionProtocolError("worker sent invalid body height")
            for name, value in (
                ("sport mode", sport_mode),
                ("gait type", gait_type),
                ("sport error code", sport_error_code),
            ):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise MotionProtocolError(
                        "worker sent invalid {}".format(name)
                    )
            parent_peer.settimeout(max(0.05, float(rpc_timeout_sec)))
            return cls(
                parent_peer,
                process,
                rpc_timeout_sec,
                reap_timeout_sec,
                controller=controller,
                posture_prepared=posture_prepared,
                body_height=float(body_height),
                sport_mode=sport_mode,
                gait_type=gait_type,
                sport_error_code=sport_error_code,
            )
        except Exception as error:
            try:
                child_peer.close()
            except OSError:
                pass
            try:
                parent_peer.close()
            except OSError:
                pass
            if process is not None:
                cls._reap_process(process, max(0.05, float(reap_timeout_sec)))
            if isinstance(error, MotionRpcError):
                raise
            raise MotionRpcError("motion worker did not start: {}".format(error))

    @property
    def alive(self) -> bool:
        return self._peer is not None and self._process.poll() is None

    @property
    def controller(self) -> str:
        return self._controller

    @property
    def posture_prepared(self) -> bool:
        return self._posture_prepared

    @property
    def body_height(self) -> float:
        return self._body_height

    @property
    def sport_mode(self) -> int:
        return self._sport_mode

    @property
    def gait_type(self) -> int:
        return self._gait_type

    @property
    def sport_error_code(self) -> int:
        return self._sport_error_code

    def _request(self, method: str, arguments: Any) -> int:
        with self._lock:
            if self._peer is None or self._process.poll() is not None:
                raise MotionRpcError("motion worker is not running")
            request_id = self._next_id
            self._next_id += 1
            request = {"id": request_id, "method": method, "args": arguments}
            try:
                self._peer.settimeout(self._rpc_timeout)
                send_frame(self._peer, request)
                response = recv_frame(self._peer)
            except socket.timeout as error:
                raise MotionRpcError("motion worker RPC timed out") from error
            except (OSError, EOFError, MotionProtocolError) as error:
                raise MotionRpcError("motion worker RPC failed: {}".format(error))
            if response.get("id") != request_id:
                raise MotionProtocolError("motion worker response ID mismatch")
            if not isinstance(response.get("ok"), bool):
                raise MotionProtocolError("motion worker response has no boolean ok")
            code = response.get("code")
            if isinstance(code, bool) or not isinstance(code, int):
                raise MotionProtocolError("motion worker response has invalid SDK code")
            if not response["ok"]:
                raise MotionSdkError(
                    str(response.get("error", "SDK operation failed"))[:512], code
                )
            if code != 0:
                raise MotionProtocolError("successful worker response has nonzero code")
            return code

    def move(self, linear_x: float, linear_y: float, angular_z: float) -> int:
        values = validate_move_arguments([linear_x, linear_y, angular_z])
        return self._request("Move", list(values))

    def stop_move(self) -> int:
        return self._request("StopMove", [])

    @staticmethod
    def _reap_process(process: Any, timeout_sec: float) -> None:
        try:
            process.wait(timeout=timeout_sec)
            return
        except subprocess.TimeoutExpired:
            pass
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=timeout_sec)
            return
        except subprocess.TimeoutExpired:
            pass
        if process.poll() is None:
            process.kill()
        try:
            process.wait(timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            pass

    def close(self) -> None:
        """Close the private channel and boundedly reap the worker.

        EOF is the worker's shutdown signal.  The worker makes its own final
        best-effort StopMove call before exiting.
        """

        with self._lock:
            peer = self._peer
            self._peer = None
            if peer is not None:
                try:
                    peer.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    peer.close()
                except OSError:
                    pass
        self._reap_process(self._process, self._reap_timeout)
