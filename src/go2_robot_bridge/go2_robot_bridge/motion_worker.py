"""Non-ROS process that exclusively owns Unitree SportClient.

The private request dispatcher intentionally exposes only ``Move`` and
``StopMove``.  No posture, standing, or mode-changing API is reachable through
this process.
"""

import argparse
import math
import socket
from typing import Any, Dict, Optional, Sequence

from .motion_rpc import MotionProtocolError, recv_frame, send_frame, validate_move_arguments


def _safe_error(error: Any) -> str:
    return "{}: {}".format(type(error).__name__, error)[:512]


def _sdk_result_code(result: Any) -> int:
    if result is None:
        return 0
    if isinstance(result, tuple):
        result = result[0] if result else 0
    if isinstance(result, bool):
        raise RuntimeError("SDK returned a boolean instead of an integer code")
    return int(result)


def dispatch_request(sport: Any, request: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and dispatch one request without exposing other SDK methods."""

    request_id = request.get("id")
    response: Dict[str, Any] = {"id": request_id, "ok": False, "code": -1}
    try:
        if set(request) != {"id", "method", "args"}:
            raise MotionProtocolError("request fields must be id, method, and args")
        if isinstance(request_id, bool) or not isinstance(request_id, int) or request_id < 1:
            raise MotionProtocolError("request id must be a positive integer")
        method = request["method"]
        arguments = request["args"]
        if method == "Move":
            linear_x, linear_y, angular_z = validate_move_arguments(arguments)
            result = sport.Move(linear_x, linear_y, angular_z)
        elif method == "StopMove":
            if arguments != []:
                raise MotionProtocolError("StopMove does not accept arguments")
            result = sport.StopMove()
        else:
            raise MotionProtocolError("unsupported motion method")
        code = _sdk_result_code(result)
        response["code"] = code
        if code != 0:
            response["error"] = "SDK returned code {}".format(code)
            return response
        response["ok"] = True
        return response
    except Exception as error:
        response["error"] = _safe_error(error)
        return response


def _best_effort_stop(sport: Any) -> None:
    if sport is None:
        return
    try:
        sport.StopMove()
    except Exception:
        pass


def serve_requests(peer: socket.socket, sport: Any) -> int:
    """Serve requests until clean EOF and always make one final stop attempt."""

    try:
        while True:
            try:
                request = recv_frame(peer)
            except EOFError:
                return 0
            response = dispatch_request(sport, request)
            send_frame(peer, response)
    finally:
        _best_effort_stop(sport)


def run_worker(
    peer: socket.socket,
    *,
    network_interface: str,
    sdk_timeout_sec: float,
    sdk_python_path: str,
    cyclonedds_python_path: str,
    require_noshm_runtime: bool,
    noshm_library_fragment: str,
) -> int:
    """Initialize the SDK in this process and serve the private socket."""

    sport: Optional[Any] = None
    request_loop_owns_stop = False
    try:
        # Deliberately late and child-only: this imports the Unitree Python
        # CycloneDDS binding only after the ROS parent has exec'd this worker.
        from .sdk_runtime import load_motion_sdk, no_shm_runtime_present

        if require_noshm_runtime and not no_shm_runtime_present(
            noshm_library_fragment
        ):
            raise RuntimeError(
                "no-SHM CycloneDDS directory is absent from LD_LIBRARY_PATH"
            )
        factory_initialize, sport_client_type = load_motion_sdk(
            sdk_python_path, cyclonedds_python_path
        )
        factory_initialize(0, network_interface)
        candidate = sport_client_type()
        candidate.SetTimeout(max(0.05, float(sdk_timeout_sec)))
        candidate.Init()
        sport = candidate
        send_frame(peer, {"type": "ready", "ok": True})
        request_loop_owns_stop = True
        return serve_requests(peer, sport)
    except Exception as error:
        if sport is None:
            try:
                send_frame(
                    peer,
                    {"type": "ready", "ok": False, "error": _safe_error(error)},
                )
            except Exception:
                pass
        return 2
    finally:
        # EOF, a malformed frame, a broken parent, and ordinary shutdown all
        # converge on the same last-resort stop path.
        if not request_loop_owns_stop:
            _best_effort_stop(sport)
        try:
            peer.close()
        except OSError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fd", required=True, type=int)
    parser.add_argument("--network-interface", required=True)
    parser.add_argument("--sdk-timeout-sec", required=True, type=float)
    parser.add_argument("--sdk-python-path", required=True)
    parser.add_argument("--cyclonedds-python-path", required=True)
    parser.add_argument("--require-noshm-runtime", action="store_true")
    parser.add_argument("--noshm-library-fragment", required=True)
    return parser


def main(arguments: Optional[Sequence[str]] = None) -> int:
    options = _parser().parse_args(arguments)
    if options.fd < 0 or not math.isfinite(options.sdk_timeout_sec):
        return 2
    peer = socket.socket(fileno=options.fd)
    peer.settimeout(None)
    try:
        return run_worker(
            peer,
            network_interface=options.network_interface,
            sdk_timeout_sec=options.sdk_timeout_sec,
            sdk_python_path=options.sdk_python_path,
            cyclonedds_python_path=options.cyclonedds_python_path,
            require_noshm_runtime=options.require_noshm_runtime,
            noshm_library_fragment=options.noshm_library_fragment,
        )
    except KeyboardInterrupt:
        # run_worker has already executed its StopMove/socket finalizer.
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
