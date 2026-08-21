"""Non-ROS Unitree VideoClient worker.

Unitree's Python CycloneDDS binding and Foxy's rmw_cyclonedds cannot create
domains in one process on this Go2.  This module intentionally imports no ROS
code and sends bounded camera packets over a private inherited socket.
"""

import argparse
import socket
import struct
import time
from typing import Optional, Tuple

from .sdk_runtime import load_camera_sdk


PACKET_FRAME = 1
PACKET_STATUS = 2
MAX_PAYLOAD_BYTES = 16 * 1024 * 1024
PACKET_HEADER = struct.Struct("!BQI")


def send_packet(
    connection: socket.socket, kind: int, capture_monotonic_ns: int, payload: bytes
) -> None:
    payload = bytes(payload)
    if kind not in (PACKET_FRAME, PACKET_STATUS):
        raise ValueError("unknown camera packet kind")
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("camera packet exceeds the protocol limit")
    header = PACKET_HEADER.pack(
        int(kind), int(capture_monotonic_ns), len(payload)
    )
    connection.sendall(header)
    if payload:
        connection.sendall(payload)


def _receive_exact(
    connection: socket.socket, length: int
) -> Optional[bytes]:
    chunks = bytearray()
    while len(chunks) < int(length):
        block = connection.recv(int(length) - len(chunks))
        if not block:
            return None
        chunks.extend(block)
    return bytes(chunks)


def receive_packet(
    connection: socket.socket,
) -> Optional[Tuple[int, int, bytes]]:
    header = _receive_exact(connection, PACKET_HEADER.size)
    if header is None:
        return None
    kind, capture_monotonic_ns, length = PACKET_HEADER.unpack(header)
    if kind not in (PACKET_FRAME, PACKET_STATUS):
        raise ValueError("camera worker sent an unknown packet kind")
    if length > MAX_PAYLOAD_BYTES:
        raise ValueError("camera worker payload exceeds the protocol limit")
    payload = _receive_exact(connection, length)
    if payload is None:
        return None
    return kind, capture_monotonic_ns, payload


def _status(connection: socket.socket, message: str) -> None:
    send_packet(
        connection,
        PACKET_STATUS,
        time.monotonic_ns(),
        str(message).encode("utf-8", errors="replace")[:4096],
    )


def run_worker(arguments) -> int:
    connection = socket.socket(fileno=int(arguments.ipc_fd))
    client = None
    factory_initialized = False
    failure_count = 0
    period = 1.0 / max(0.2, min(float(arguments.rate_hz), 15.0))
    next_sample_at = time.monotonic()
    try:
        factory_initialize, video_client_type = load_camera_sdk(
            arguments.sdk_python_path, arguments.cyclonedds_python_path
        )
        while True:
            if client is None:
                try:
                    if not factory_initialized:
                        factory_initialize(0, arguments.interface)
                        factory_initialized = True
                    candidate = video_client_type()
                    candidate.SetTimeout(float(arguments.timeout_sec))
                    candidate.Init()
                    client = candidate
                    failure_count = 0
                    _status(connection, "ready")
                except Exception as error:
                    _status(connection, "connect_error: {}".format(error))
                    time.sleep(max(0.1, float(arguments.reconnect_delay_sec)))
                    continue

            started_ns = time.monotonic_ns()
            try:
                code, data = client.GetImageSample()
                finished_ns = time.monotonic_ns()
                if int(code) != 0 or not data:
                    raise RuntimeError(
                        "GetImageSample returned SDK code {}".format(code)
                    )
                jpeg = bytes(data)
                if len(jpeg) > MAX_PAYLOAD_BYTES:
                    raise RuntimeError("camera JPEG exceeds protocol limit")
                send_packet(
                    connection,
                    PACKET_FRAME,
                    (started_ns + finished_ns) // 2,
                    jpeg,
                )
                failure_count = 0
            except (BrokenPipeError, ConnectionError):
                return 0
            except Exception as error:
                failure_count += 1
                _status(connection, "sample_error: {}".format(error))
                if failure_count >= max(1, int(arguments.max_failures)):
                    client = None
                    failure_count = 0

            next_sample_at += period
            delay = next_sample_at - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
            elif delay < -period:
                next_sample_at = time.monotonic()
    except (BrokenPipeError, ConnectionError):
        return 0
    except Exception as error:
        try:
            _status(connection, "fatal_error: {}".format(error))
        except (BrokenPipeError, ConnectionError, OSError):
            pass
        return 2
    finally:
        try:
            connection.close()
        except OSError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipc-fd", required=True, type=int)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--sdk-python-path", required=True)
    parser.add_argument("--cyclonedds-python-path", required=True)
    parser.add_argument("--timeout-sec", required=True, type=float)
    parser.add_argument("--rate-hz", required=True, type=float)
    parser.add_argument("--reconnect-delay-sec", required=True, type=float)
    parser.add_argument("--max-failures", required=True, type=int)
    return parser


def main(argv=None) -> int:
    try:
        return run_worker(_parser().parse_args(argv))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
