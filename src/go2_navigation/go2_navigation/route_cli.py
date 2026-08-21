"""Validate, record, and execute named waypoint routes through Nav2.

Copyright (c) 2026 Go2 SLAM Navigation Maintainers. MIT License.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

from .route_io import Route, Waypoint, atomic_write_routes, load_routes


_TERMINAL_ACTION_STATUSES = frozenset((4, 5, 6))


def _is_terminal_action_status(status) -> bool:
    """Return whether an action status is one of the three terminal states."""

    try:
        return int(status) in _TERMINAL_ACTION_STATUSES
    except (TypeError, ValueError):
        return False


def _pose_values(position, orientation):
    """Validate a ROS-like pose and return finite planar route coordinates."""

    components = (
        position.x,
        position.y,
        position.z,
        orientation.x,
        orientation.y,
        orientation.z,
        orientation.w,
    )
    try:
        values = tuple(float(value) for value in components)
    except (TypeError, ValueError) as exc:
        raise ValueError("recorded pose components must be numeric") from exc
    if not all(math.isfinite(value) for value in values):
        raise ValueError("recorded pose components must be finite")
    qx, qy, qz, qw = values[3:]
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm < 1.0e-6:
        raise ValueError("recorded pose quaternion has near-zero norm")
    qx, qy, qz, qw = (value / norm for value in (qx, qy, qz, qw))
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny, cosy)
    if not math.isfinite(yaw):
        raise ValueError("recorded pose produced a non-finite yaw")
    return values[0], values[1], yaw


def _spin_for_future(rclpy_module, node, future, timeout: float) -> bool:
    """Spin briefly until a future completes or its bound expires."""

    if future is None:
        return False
    deadline = time.monotonic() + max(0.0, float(timeout))
    while not future.done() and rclpy_module.ok():
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        rclpy_module.spin_once(node, timeout_sec=min(0.1, remaining))
    return bool(future.done())


def _future_terminal_status(future):
    """Read an action result future without raising; None means unconfirmed."""

    if future is None or not future.done():
        return None
    try:
        wrapped = future.result()
    except Exception:
        return None
    status = None if wrapped is None else getattr(wrapped, "status", None)
    if not _is_terminal_action_status(status):
        return None
    return int(wrapped.status)


def _cancel_response_ok(response) -> bool:
    """The CancelGoal service uses return_code zero for an accepted request."""

    return response is not None and getattr(response, "return_code", None) == 0


def _request_latched_stop(
    rclpy_module, node, stop_client, trigger_type
) -> bool:
    """Call the motion bridge's latching stop service with bounded waits."""

    try:
        if not stop_client.wait_for_service(timeout_sec=1.0):
            node.get_logger().error(
                "Latching stop service /go2/motion/stop is unavailable"
            )
            return False
        future = stop_client.call_async(trigger_type.Request())
        if not _spin_for_future(rclpy_module, node, future, 2.0):
            node.get_logger().error(
                "Latching stop request was not acknowledged"
            )
            return False
        response = future.result()
        if response is None or not bool(getattr(response, "success", False)):
            detail = (
                ""
                if response is None
                else str(getattr(response, "message", ""))
            )
            node.get_logger().error(f"Latching stop request failed: {detail}")
            return False
        node.get_logger().warn(
            "Motion stop latched; explicit re-enable is required"
        )
        return True
    except BaseException as error:
        node.get_logger().error(f"Latching stop request raised: {error}")
        return False


def _request_cancel_all(
    rclpy_module, node, cancel_client, cancel_type
) -> bool:
    """Send the action protocol's zero-ID request to cancel all Nav2 goals."""

    try:
        if not cancel_client.wait_for_service(timeout_sec=1.0):
            node.get_logger().error(
                "NavigateToPose cancel-all service is unavailable"
            )
            return False
        future = cancel_client.call_async(cancel_type.Request())
        if not _spin_for_future(rclpy_module, node, future, 3.0):
            node.get_logger().error(
                "NavigateToPose cancel-all request timed out"
            )
            return False
        response = future.result()
        if not _cancel_response_ok(response):
            code = (
                None
                if response is None
                else getattr(response, "return_code", None)
            )
            node.get_logger().error(
                f"NavigateToPose cancel-all was rejected (code={code})"
            )
            return False
        node.get_logger().warn(
            "NavigateToPose cancel-all request was accepted"
        )
        return True
    except BaseException as error:
        node.get_logger().error(
            f"NavigateToPose cancel-all request raised: {error}"
        )
        return False


def _fail_safe_abort(
    rclpy_module,
    node,
    handle,
    result_future,
    cancel_client,
    cancel_type,
    stop_client,
    trigger_type,
) -> bool:
    """Latch stop, cancel navigation, and confirm a terminal result."""

    stop_confirmed = _request_latched_stop(
        rclpy_module, node, stop_client, trigger_type
    )
    terminal = _future_terminal_status(result_future)
    if handle is not None and result_future is None:
        try:
            result_future = handle.get_result_async()
        except BaseException as error:
            node.get_logger().error(
                f"Could not request active goal result: {error}"
            )

    if handle is not None and terminal is None:
        try:
            cancel_future = handle.cancel_goal_async()
            if _spin_for_future(rclpy_module, node, cancel_future, 3.0):
                response = cancel_future.result()
                if not _cancel_response_ok(response):
                    code = (
                        None
                        if response is None
                        else getattr(response, "return_code", None)
                    )
                    node.get_logger().error(
                        f"Active-goal cancellation was rejected (code={code})"
                    )
            else:
                node.get_logger().error(
                    "Active-goal cancellation request timed out"
                )
        except BaseException as error:
            node.get_logger().error(
                f"Active-goal cancellation raised: {error}"
            )
        if result_future is not None:
            _spin_for_future(rclpy_module, node, result_future, 5.0)
            terminal = _future_terminal_status(result_future)

    if terminal is None:
        _request_cancel_all(
            rclpy_module, node, cancel_client, cancel_type
        )
        if result_future is not None:
            _spin_for_future(rclpy_module, node, result_future, 5.0)
            terminal = _future_terminal_status(result_future)

    if terminal is None:
        node.get_logger().error(
            "Could not confirm a terminal NavigateToPose result; "
            "motion remains stop-latched"
        )
    else:
        node.get_logger().warn(
            f"NavigateToPose reached terminal action status {terminal}"
        )

    if not stop_confirmed:
        stop_confirmed = _request_latched_stop(
            rclpy_module, node, stop_client, trigger_type
        )
    return terminal is not None and stop_confirmed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="go2_route")
    parser.add_argument(
        "--file", default="routes.yaml", help="route YAML file"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list and validate routes")
    validate = subparsers.add_parser("validate", help="validate a route file")
    validate.add_argument("route", nargs="?")
    run = subparsers.add_parser(
        "run", help="execute a route using NavigateToPose"
    )
    run.add_argument("route")
    run.add_argument(
        "--loops",
        type=int,
        default=1,
        help="finite loop count (default: 1)",
    )
    record = subparsers.add_parser(
        "record", help="append the current odometry pose"
    )
    record.add_argument("route")
    record.add_argument("--name", default="")
    record.add_argument("--frame", default="odom")
    record.add_argument("--odom-topic", default="/go2/odom")
    record.add_argument("--wait", type=float, default=0.0)
    record.add_argument("--timeout", type=float, default=120.0)
    return parser


def _record(arguments) -> int:
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener

    rclpy.init()
    node = Node("go2_route_recorder")
    received = []
    qos = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
    node.create_subscription(
        Odometry, arguments.odom_topic, received.append, qos
    )
    tf_buffer = Buffer(node=node)
    tf_listener = TransformListener(tf_buffer, node)
    deadline = time.monotonic() + 5.0
    try:
        while rclpy.ok() and not received and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        if not received:
            print(
                f"No odometry received on {arguments.odom_topic} "
                "within 5 seconds",
                file=sys.stderr,
            )
            return 2
        message = received[-1]
        odom_frame = str(message.header.frame_id).strip().lstrip("/")
        requested_frame = str(arguments.frame).strip().lstrip("/")
        if not requested_frame:
            print("Route frame must not be empty", file=sys.stderr)
            return 2

        if odom_frame == requested_frame:
            position = message.pose.pose.position
            orientation = message.pose.pose.orientation
        else:
            base_frame = (
                str(message.child_frame_id).strip().lstrip("/")
                or "base_link"
            )
            transform = None
            tf_deadline = time.monotonic() + 5.0
            while (
                rclpy.ok()
                and transform is None
                and time.monotonic() < tf_deadline
            ):
                rclpy.spin_once(node, timeout_sec=0.1)
                try:
                    transform = tf_buffer.lookup_transform(
                        requested_frame,
                        base_frame,
                        Time(),
                        timeout=Duration(seconds=0.1),
                    )
                except Exception:
                    transform = None
            if transform is None:
                print(
                    f"No TF from {requested_frame!r} to "
                    f"{base_frame!r} within 5 seconds",
                    file=sys.stderr,
                )
                return 2
            position = transform.transform.translation
            orientation = transform.transform.rotation
    finally:
        del tf_listener
        node.destroy_node()
        rclpy.shutdown()
    try:
        x, y, yaw = _pose_values(position, orientation)
    except ValueError as error:
        print(
            f"Refusing to record invalid robot pose: {error}",
            file=sys.stderr,
        )
        return 2
    point = Waypoint(
        x=x,
        y=y,
        yaw=yaw,
        wait=max(0.0, arguments.wait),
        timeout=max(1.0, arguments.timeout),
        name=arguments.name or f"waypoint_{int(time.time())}",
    )
    path = Path(arguments.file).expanduser()
    if path.exists():
        routes = load_routes(path)
    else:
        routes = {}
    old = routes.get(arguments.route)
    if old is None:
        routes[arguments.route] = Route(
            arguments.route, requested_frame, False, [point]
        )
    else:
        if old.frame_id != requested_frame:
            print(
                "Cannot append a waypoint with a different frame",
                file=sys.stderr,
            )
            return 2
        routes[arguments.route] = Route(
            old.name,
            old.frame_id,
            old.loop,
            old.waypoints + [point],
        )
    atomic_write_routes(path, routes.values())
    print(
        f"Recorded {point.name} at x={point.x:.3f}, "
        f"y={point.y:.3f}, yaw={point.yaw:.3f}"
    )
    return 0


def _execute(arguments, route: Route) -> int:
    import rclpy
    from action_msgs.msg import GoalStatus
    from action_msgs.srv import CancelGoal
    from geometry_msgs.msg import PoseStamped
    from nav2_msgs.action import NavigateToPose
    from rclpy.action import ActionClient
    from rclpy.node import Node
    from std_srvs.srv import Trigger

    rclpy.init()
    node = Node("go2_route_executor")
    client = ActionClient(node, NavigateToPose, "navigate_to_pose")
    cancel_client = node.create_client(
        CancelGoal, "navigate_to_pose/_action/cancel_goal"
    )
    stop_client = node.create_client(Trigger, "/go2/motion/stop")
    loop_count = max(1, arguments.loops)
    handle = None
    result_future = None
    abort_invoked = False
    route_completed = False

    def abort_once():
        nonlocal abort_invoked
        if abort_invoked:
            return False
        abort_invoked = True
        return _fail_safe_abort(
            rclpy,
            node,
            handle,
            result_future,
            cancel_client,
            CancelGoal,
            stop_client,
            Trigger,
        )

    if route.loop and arguments.loops == 1:
        node.get_logger().warn(
            "Route is marked loop=true; CLI still executes one loop "
            "unless --loops is set"
        )
    try:
        if not client.wait_for_server(timeout_sec=15.0):
            node.get_logger().error(
                "NavigateToPose action server is unavailable"
            )
            abort_once()
            return 3
        for loop_index in range(loop_count):
            for index, point in enumerate(route.waypoints):
                goal = NavigateToPose.Goal()
                pose = PoseStamped()
                pose.header.frame_id = route.frame_id
                pose.header.stamp = node.get_clock().now().to_msg()
                pose.pose.position.x = point.x
                pose.pose.position.y = point.y
                pose.pose.orientation.z = math.sin(point.yaw / 2.0)
                pose.pose.orientation.w = math.cos(point.yaw / 2.0)
                goal.pose = pose
                label = point.name or str(index + 1)
                node.get_logger().info(
                    f"Loop {loop_index + 1}/{loop_count}: "
                    f"navigating to {label} "
                    f"({point.x:.2f}, {point.y:.2f}, {point.yaw:.2f})"
                )
                future = client.send_goal_async(goal)
                if not _spin_for_future(rclpy, node, future, 10.0):
                    node.get_logger().error(
                        f"Waypoint {label} goal response timed out; "
                        "goal ownership is unknown"
                    )
                    abort_once()
                    return 4
                handle = future.result()
                if handle is None:
                    node.get_logger().error(
                        f"Waypoint {label} returned no goal handle"
                    )
                    abort_once()
                    return 4
                if not handle.accepted:
                    node.get_logger().error(f"Waypoint {label} was rejected")
                    handle = None
                    abort_once()
                    return 4
                result_future = handle.get_result_async()
                deadline = time.monotonic() + point.timeout
                while (
                    rclpy.ok()
                    and not result_future.done()
                    and time.monotonic() < deadline
                ):
                    rclpy.spin_once(node, timeout_sec=0.2)
                if not result_future.done():
                    node.get_logger().error(
                        f"Waypoint {label} timed out; entering fail-safe abort"
                    )
                    abort_once()
                    return 5
                wrapped = result_future.result()
                if (
                    wrapped is None
                    or wrapped.status != GoalStatus.STATUS_SUCCEEDED
                ):
                    status = None if wrapped is None else wrapped.status
                    node.get_logger().error(
                        f"Waypoint {label} failed with action status {status}"
                    )
                    abort_once()
                    return 6
                handle = None
                result_future = None
                if point.wait > 0.0:
                    end = time.monotonic() + point.wait
                    while rclpy.ok() and time.monotonic() < end:
                        rclpy.spin_once(
                            node,
                            timeout_sec=min(0.2, end - time.monotonic()),
                        )
                    if not rclpy.ok():
                        node.get_logger().error(
                            "ROS shutdown while waiting between waypoints"
                        )
                        abort_once()
                        return 7
        route_completed = True
        node.get_logger().info("Route completed")
        return 0
    except KeyboardInterrupt:
        node.get_logger().warn("Interrupted; entering fail-safe abort")
        abort_once()
        return 130
    except Exception as error:
        node.get_logger().error(
            f"Route execution raised {type(error).__name__}: {error}"
        )
        abort_once()
        return 7
    finally:
        if not route_completed and not abort_invoked:
            abort_once()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def main(argv=None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "record":
        return _record(arguments)
    try:
        routes = load_routes(arguments.file)
    except (OSError, ValueError) as error:
        print(f"Route file error: {error}", file=sys.stderr)
        return 2
    if arguments.command == "list":
        for route in routes.values():
            print(
                f"{route.name}: {len(route.waypoints)} waypoint(s), "
                f"frame={route.frame_id}, loop={route.loop}"
            )
        return 0
    if arguments.command == "validate":
        if arguments.route and arguments.route not in routes:
            print(f"Unknown route {arguments.route!r}", file=sys.stderr)
            return 2
        print(f"Validated {len(routes)} route(s)")
        return 0
    route = routes.get(arguments.route)
    if route is None:
        print(f"Unknown route {arguments.route!r}", file=sys.stderr)
        return 2
    return _execute(arguments, route)


if __name__ == "__main__":
    raise SystemExit(main())
