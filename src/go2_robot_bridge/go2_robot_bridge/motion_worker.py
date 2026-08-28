"""Non-ROS process that exclusively owns Unitree motion SDK clients.

The private request dispatcher intentionally exposes only ``Move`` and
``StopMove``. During explicitly armed worker startup, this process detects and
prepares the firmware-appropriate high-level controller (``mcf`` on current
firmware, ``sport_mode`` on legacy firmware). No posture or mode-changing API
is reachable through the private request dispatcher.
"""

import argparse
import math
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

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


def _require_sdk_success(result: Any, action: str) -> None:
    code = _sdk_result_code(result)
    if code != 0:
        raise RuntimeError("{} failed with SDK code {}".format(action, code))


@dataclass(frozen=True)
class SportStateSnapshot:
    sequence: int
    mode: int
    gait_type: int
    body_height: float
    error_code: int


class SportStateMonitor:
    """Thread-safe boundary around the SDK's sport-state callback."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._sequence = 0
        self._snapshot: Optional[SportStateSnapshot] = None

    def callback(self, message: Any) -> None:
        try:
            mode = int(message.mode)
            gait_type = int(message.gait_type)
            body_height = float(message.body_height)
            error_code = int(message.error_code)
            if not math.isfinite(body_height):
                return
        except (AttributeError, TypeError, ValueError, OverflowError):
            return
        with self._condition:
            self._sequence += 1
            self._snapshot = SportStateSnapshot(
                sequence=self._sequence,
                mode=mode,
                gait_type=gait_type,
                body_height=body_height,
                error_code=error_code,
            )
            self._condition.notify_all()

    def wait_for_sample(
        self, timeout_sec: float, *, after_sequence: int = -1
    ) -> SportStateSnapshot:
        deadline = time.monotonic() + max(0.05, float(timeout_sec))
        with self._condition:
            while (
                self._snapshot is None
                or self._snapshot.sequence <= int(after_sequence)
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError("timed out waiting for fresh sport state")
                self._condition.wait(remaining)
            return self._snapshot


def prepare_robot_posture(
    sport: Any,
    monitor: Any,
    *,
    auto_prepare: bool,
    standing_min_body_height: float,
    state_timeout_sec: float,
    sleep: Callable[[float], None] = time.sleep,
) -> Tuple[SportStateSnapshot, bool]:
    """Verify standing height and recover to BalanceStand when authorized."""

    minimum_height = max(0.01, float(standing_min_body_height))
    initial = monitor.wait_for_sample(state_timeout_sec)
    if initial.body_height >= minimum_height:
        return initial, False
    if not auto_prepare:
        raise RuntimeError(
            "robot body height {:.3f} m is below standing minimum {:.3f} m; "
            "automatic posture preparation is disabled".format(
                initial.body_height, minimum_height
            )
        )

    _require_sdk_success(sport.Damp(), "enter damp before recovery stand")
    sleep(1.0)
    _require_sdk_success(sport.RecoveryStand(), "recover stand")
    sleep(4.0)
    _require_sdk_success(sport.BalanceStand(), "enter balance stand")
    sleep(1.5)

    final = monitor.wait_for_sample(
        state_timeout_sec, after_sequence=initial.sequence
    )
    if final.body_height < minimum_height:
        raise RuntimeError(
            "posture preparation finished but body height {:.3f} m remains "
            "below {:.3f} m (sport error code {})".format(
                final.body_height, minimum_height, final.error_code
            )
        )
    return final, True


def _service_snapshot(robot_state: Any) -> Dict[str, Any]:
    result = robot_state.ServiceList()
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("ServiceList returned an invalid result")
    _require_sdk_success(result[0], "list robot services")
    items = result[1]
    if not isinstance(items, (list, tuple)):
        raise RuntimeError("ServiceList returned no service list")
    services: Dict[str, Any] = {}
    for item in items:
        name = getattr(item, "name", None)
        status = getattr(item, "status", None)
        if not isinstance(name, str) or not name:
            raise RuntimeError("ServiceList contained an invalid service name")
        if isinstance(status, bool) or not isinstance(status, int):
            raise RuntimeError(
                "service {!r} has invalid status {!r}".format(name, status)
            )
        services[name] = item
    return services


def _selected_mode_name(motion_switcher: Any) -> str:
    result = motion_switcher.CheckMode()
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("CheckMode returned an invalid result")
    _require_sdk_success(result[0], "check motion mode")
    selected = result[1]
    if not isinstance(selected, dict):
        raise RuntimeError("CheckMode returned no selected-mode object")
    name = selected.get("name", "")
    if not isinstance(name, str):
        raise RuntimeError("CheckMode returned an invalid mode name")
    return name


def prepare_motion_controller(
    robot_state: Any,
    motion_switcher_type: Any,
    *,
    controller_rpc_timeout_sec: float,
    transition_timeout_sec: float,
    poll_interval_sec: float = 0.25,
) -> str:
    """Prepare the firmware controller without changing robot posture.

    RobotState reports status 0 for a running service. New firmware exposes
    ``mcf`` while older firmware exposes ``sport_mode``. ServiceList and
    CheckMode are treated as authoritative because ServiceSwitch can return a
    transitional wrapper code even when the service subsequently starts.
    """

    timeout = max(0.1, float(transition_timeout_sec))
    poll_interval = max(0.01, float(poll_interval_sec))
    services = _service_snapshot(robot_state)
    controller = (
        "mcf"
        if "mcf" in services
        else "sport_mode"
        if "sport_mode" in services
        else None
    )
    if controller is None:
        raise RuntimeError("robot exposes neither mcf nor sport_mode")

    if services[controller].status != 0:
        switch_code = _sdk_result_code(robot_state.ServiceSwitch(controller, True))
        deadline = time.monotonic() + timeout
        while True:
            services = _service_snapshot(robot_state)
            if controller in services and services[controller].status == 0:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "high-level controller {!r} did not start "
                    "(ServiceSwitch SDK code {})".format(controller, switch_code)
                )
            time.sleep(poll_interval)

    if controller == "mcf":
        motion_switcher = motion_switcher_type()
        motion_switcher.SetTimeout(max(0.1, float(controller_rpc_timeout_sec)))
        motion_switcher.Init()
        if _selected_mode_name(motion_switcher) != "mcf":
            _require_sdk_success(
                motion_switcher.SelectMode("mcf"), "select mcf motion mode"
            )
            deadline = time.monotonic() + timeout
            while _selected_mode_name(motion_switcher) != "mcf":
                if time.monotonic() >= deadline:
                    raise RuntimeError("mcf motion mode did not become active")
                time.sleep(poll_interval)

    return controller


def prepare_velocity_client(
    sport: Any,
    obstacles_avoid_client_type: Any,
    *,
    rpc_timeout_sec: float,
    transition_timeout_sec: float = 3.0,
    poll_interval_sec: float = 0.1,
    sleep: Callable[[float], None] = time.sleep,
) -> Tuple[Any, Optional[Any], str]:
    """Prepare the direct SportClient velocity path used by demov2 navigation.

    Current MCF firmware can acknowledge ``ObstaclesAvoidClient.Move`` while
    silently suppressing locomotion. This project already has a fail-closed
    LiDAR obstacle gate, so follow demov2's navigation path: safely acquire and
    stop the avoidance command owner, release it, disable the firmware service,
    verify the transition, and send velocity through ``SportClient``.

    The returned avoidance client is non-None only when its original enabled
    state should be restored during worker shutdown.
    """

    avoidance = obstacles_avoid_client_type()
    avoidance.SetTimeout(max(0.1, float(rpc_timeout_sec)))
    avoidance.Init()
    result = avoidance.SwitchGet()
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError("obstacle-avoidance SwitchGet returned an invalid result")
    _require_sdk_success(result[0], "read obstacle-avoidance state")
    enabled = result[1]
    if not isinstance(enabled, (bool, int)):
        raise RuntimeError("obstacle-avoidance state is not boolean")
    if not bool(enabled):
        return sport, None, "sport"

    _require_sdk_success(
        avoidance.UseRemoteCommandFromApi(True),
        "acquire obstacle-avoidance control before disabling it",
    )
    # Match Unitree's and demov2's command-input handoff delay, then explicitly
    # stop before changing the service owner.
    sleep(0.5)
    _require_sdk_success(
        avoidance.Move(0.0, 0.0, 0.0),
        "stop obstacle-avoidance velocity owner",
    )
    _require_sdk_success(
        avoidance.UseRemoteCommandFromApi(False),
        "release obstacle-avoidance API command control",
    )
    _require_sdk_success(
        avoidance.SwitchSet(False),
        "disable built-in obstacle avoidance for direct SportClient control",
    )

    timeout = max(0.1, float(transition_timeout_sec))
    poll_interval = max(0.01, float(poll_interval_sec))
    deadline = time.monotonic() + timeout
    while True:
        result = avoidance.SwitchGet()
        if not isinstance(result, tuple) or len(result) != 2:
            raise RuntimeError(
                "obstacle-avoidance SwitchGet returned an invalid result"
            )
        _require_sdk_success(result[0], "verify obstacle-avoidance state")
        if not bool(result[1]):
            break
        if time.monotonic() >= deadline:
            raise RuntimeError("built-in obstacle avoidance did not disable")
        sleep(poll_interval)

    # Give MCF time to return velocity ownership to SportClient.
    sleep(0.5)
    return sport, avoidance, "sport_direct"


def _restore_obstacle_avoidance(avoidance: Any) -> None:
    """Best-effort restoration of the avoidance state owned before arming."""

    avoidance.UseRemoteCommandFromApi(False)
    result = avoidance.SwitchGet()
    if (
        isinstance(result, tuple)
        and len(result) == 2
        and _sdk_result_code(result[0]) == 0
        and not bool(result[1])
    ):
        avoidance.SwitchSet(True)


def _stop_motion(client: Any, controller: str, motion_backend: str = "sport") -> Any:
    """Issue the stop operation supported by the active firmware controller."""

    if motion_backend == "obstacle_avoidance":
        return client.Move(0.0, 0.0, 0.0)
    if controller == "mcf":
        # Current MCF firmware rejects SportClient.StopMove() with SDK code -1.
        # Its supported velocity stop is the same zero Move used by demov2.
        return client.Move(0.0, 0.0, 0.0)
    if controller == "sport_mode":
        return client.StopMove()
    raise RuntimeError("unsupported motion controller {!r}".format(controller))


def dispatch_request(
    client: Any,
    request: Dict[str, Any],
    *,
    controller: str,
    motion_backend: str = "sport",
) -> Dict[str, Any]:
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
            result = client.Move(linear_x, linear_y, angular_z)
        elif method == "StopMove":
            if arguments != []:
                raise MotionProtocolError("StopMove does not accept arguments")
            result = _stop_motion(client, controller, motion_backend)
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


def _best_effort_stop(
    client: Any, controller: Optional[str], motion_backend: str = "sport"
) -> None:
    if client is None:
        return
    try:
        _stop_motion(client, str(controller), motion_backend)
    except Exception:
        pass


def serve_requests(
    peer: socket.socket,
    client: Any,
    *,
    controller: str,
    motion_backend: str = "sport",
) -> int:
    """Serve requests until clean EOF and always make one final stop attempt."""

    try:
        while True:
            try:
                request = recv_frame(peer)
            except EOFError:
                return 0
            response = dispatch_request(
                client,
                request,
                controller=controller,
                motion_backend=motion_backend,
            )
            send_frame(peer, response)
    finally:
        _best_effort_stop(client, controller, motion_backend)


def run_worker(
    peer: socket.socket,
    *,
    network_interface: str,
    sdk_timeout_sec: float,
    controller_rpc_timeout_sec: float,
    controller_transition_timeout_sec: float,
    auto_prepare_posture: bool,
    sport_state_timeout_sec: float,
    standing_min_body_height: float,
    sdk_python_path: str,
    cyclonedds_python_path: str,
    require_noshm_runtime: bool,
    noshm_library_fragment: str,
) -> int:
    """Initialize the SDK in this process and serve the private socket."""

    sport: Optional[Any] = None
    motion_client: Optional[Any] = None
    avoidance_owner: Optional[Any] = None
    motion_backend = "sport"
    controller: Optional[str] = None
    ready_sent = False
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
        (
            factory_initialize,
            channel_subscriber_type,
            sport_client_type,
            robot_state_client_type,
            motion_switcher_client_type,
            obstacles_avoid_client_type,
            sport_mode_state_type,
        ) = load_motion_sdk(sdk_python_path, cyclonedds_python_path)
        factory_initialize(0, network_interface)
        robot_state = robot_state_client_type()
        robot_state.SetTimeout(max(0.1, float(controller_rpc_timeout_sec)))
        robot_state.Init()
        controller = prepare_motion_controller(
            robot_state,
            motion_switcher_client_type,
            controller_rpc_timeout_sec=controller_rpc_timeout_sec,
            transition_timeout_sec=controller_transition_timeout_sec,
        )
        candidate = sport_client_type()
        candidate.SetTimeout(max(0.1, float(controller_rpc_timeout_sec)))
        candidate.Init()
        sport = candidate
        monitor = SportStateMonitor()
        sport_state_subscriber = channel_subscriber_type(
            "rt/sportmodestate", sport_mode_state_type
        )
        sport_state_subscriber.Init(monitor.callback, 10)
        posture, posture_prepared = prepare_robot_posture(
            sport,
            monitor,
            auto_prepare=auto_prepare_posture,
            standing_min_body_height=standing_min_body_height,
            state_timeout_sec=sport_state_timeout_sec,
        )
        motion_client, avoidance_owner, motion_backend = prepare_velocity_client(
            sport,
            obstacles_avoid_client_type,
            rpc_timeout_sec=controller_rpc_timeout_sec,
        )
        candidate.SetTimeout(max(0.05, float(sdk_timeout_sec)))
        if motion_client is not sport:
            motion_client.SetTimeout(max(0.05, float(sdk_timeout_sec)))
        send_frame(
            peer,
            {
                "type": "ready",
                "ok": True,
                "controller": controller,
                "posture_prepared": posture_prepared,
                "body_height": posture.body_height,
                "sport_mode": posture.mode,
                "gait_type": posture.gait_type,
                "sport_error_code": posture.error_code,
            },
        )
        ready_sent = True
        request_loop_owns_stop = True
        return serve_requests(
            peer,
            motion_client,
            controller=controller,
            motion_backend=motion_backend,
        )
    except Exception as error:
        if not ready_sent:
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
            _best_effort_stop(motion_client or sport, controller, motion_backend)
        if avoidance_owner is not None:
            try:
                _restore_obstacle_avoidance(avoidance_owner)
            except Exception:
                pass
        try:
            peer.close()
        except OSError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--fd", required=True, type=int)
    parser.add_argument("--network-interface", required=True)
    parser.add_argument("--sdk-timeout-sec", required=True, type=float)
    parser.add_argument("--controller-rpc-timeout-sec", required=True, type=float)
    parser.add_argument(
        "--controller-transition-timeout-sec", required=True, type=float
    )
    parser.add_argument("--auto-prepare-posture", action="store_true")
    parser.add_argument("--sport-state-timeout-sec", required=True, type=float)
    parser.add_argument("--standing-min-body-height", required=True, type=float)
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
            controller_rpc_timeout_sec=options.controller_rpc_timeout_sec,
            controller_transition_timeout_sec=(
                options.controller_transition_timeout_sec
            ),
            auto_prepare_posture=options.auto_prepare_posture,
            sport_state_timeout_sec=options.sport_state_timeout_sec,
            standing_min_body_height=options.standing_min_body_height,
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
