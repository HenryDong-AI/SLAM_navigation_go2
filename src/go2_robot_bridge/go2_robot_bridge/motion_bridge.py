"""Safety-gated ROS 2 Twist to Unitree SportClient bridge."""

import json
import math
import threading
import time
from typing import Any, Dict, Optional

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from .safety import (
    GateDecision,
    PermanentFaultLatch,
    TimeSyncInterlock,
    TwistCommand,
    assess_front_cloud,
    clamp_twist,
    evaluate_motion_gate,
    iter_xyz,
    source_timestamp_is_fresh,
    slew_twist,
)
from .motion_rpc import (
    DEFAULT_CYCLONEDDS_PYTHON_PATH,
    DEFAULT_SDK_PYTHON_PATH,
    MotionRpcError,
    MotionWorkerProxy,
)
from .periodic_worker import PeriodicScheduler


class MotionBridge(Node):
    """Forward bounded velocity commands only while every interlock is healthy."""

    def __init__(self) -> None:
        super().__init__("motion_bridge")
        self._state_lock = threading.RLock()

        self.declare_parameter("network_interface", "eth0")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("odom_topic", "/go2/odom")
        self.declare_parameter("front_cloud_topic", "/go2/lidar/cloud_base")
        self.declare_parameter("time_sync_status_topic", "/go2/time_sync/status")
        self.declare_parameter("require_time_sync", True)
        self.declare_parameter("time_sync_status_timeout_sec", 1.0)
        self.declare_parameter("enable_service", "/go2/motion/enable")
        self.declare_parameter("stop_service", "/go2/motion/stop")
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("sdk_timeout_sec", 1.0)
        self.declare_parameter("sdk_reconnect_delay_sec", 2.0)
        self.declare_parameter("worker_startup_timeout_sec", 5.0)
        self.declare_parameter("worker_rpc_timeout_sec", 1.5)
        self.declare_parameter("worker_reap_timeout_sec", 2.0)
        self.declare_parameter("post_worker_sensor_refresh_timeout_sec", 2.0)
        self.declare_parameter("sdk_python_path", DEFAULT_SDK_PYTHON_PATH)
        self.declare_parameter(
            "cyclonedds_python_path", DEFAULT_CYCLONEDDS_PYTHON_PATH
        )
        self.declare_parameter("require_noshm_runtime", True)
        self.declare_parameter("noshm_library_fragment", "install_noshm/lib")

        self.declare_parameter("max_linear_x", 0.40)
        self.declare_parameter("max_linear_y", 0.25)
        self.declare_parameter("max_angular_z", 0.60)
        self.declare_parameter("linear_slew_rate", 0.60)
        self.declare_parameter("angular_slew_rate", 1.20)
        self.declare_parameter("command_timeout_sec", 0.50)
        self.declare_parameter("require_fresh_odom", True)
        self.declare_parameter("odom_timeout_sec", 0.50)
        self.declare_parameter("expected_odom_frame", "odom")
        self.declare_parameter("expected_base_frame", "base_link")
        self.declare_parameter("expected_cloud_frame", "base_link")
        self.declare_parameter("sensor_future_tolerance_sec", 0.10)
        self.declare_parameter("stop_repeat_sec", 0.20)
        self.declare_parameter("shutdown_stop_repetitions", 3)
        self.declare_parameter("shutdown_stop_interval_sec", 0.05)

        self.declare_parameter("front_obstacle_guard_enabled", True)
        self.declare_parameter("front_cloud_timeout_sec", 0.50)
        self.declare_parameter("fail_closed_on_cloud_timeout", True)
        self.declare_parameter("require_fresh_cloud_to_arm", True)
        self.declare_parameter("front_min_x", 0.15)
        self.declare_parameter("front_max_x", 0.80)
        self.declare_parameter("front_half_width", 0.35)
        # Normalized cloud_base is REP-103 Z-up; exclude the floor below it.
        self.declare_parameter("front_min_z", -0.22)
        self.declare_parameter("front_max_z", 0.50)
        self.declare_parameter("front_min_points", 5)
        self.declare_parameter("front_cloud_sample_stride", 1)
        self.declare_parameter("front_cloud_health_min_points", 20)
        self.declare_parameter("front_cloud_health_min_range", 0.10)
        self.declare_parameter("front_cloud_health_max_range", 30.0)

        self._interface = str(self.get_parameter("network_interface").value)
        self._sdk_path = str(self.get_parameter("sdk_python_path").value)
        self._cyclonedds_python_path = str(
            self.get_parameter("cyclonedds_python_path").value
        )
        self._sdk_timeout = max(0.1, float(self.get_parameter("sdk_timeout_sec").value))
        self._reconnect_delay = max(
            0.1, float(self.get_parameter("sdk_reconnect_delay_sec").value)
        )
        self._worker_startup_timeout = max(
            0.1, float(self.get_parameter("worker_startup_timeout_sec").value)
        )
        self._worker_rpc_timeout = max(
            0.1, float(self.get_parameter("worker_rpc_timeout_sec").value)
        )
        self._worker_reap_timeout = max(
            0.1, float(self.get_parameter("worker_reap_timeout_sec").value)
        )
        self._post_worker_refresh_timeout = max(
            0.1,
            float(
                self.get_parameter(
                    "post_worker_sensor_refresh_timeout_sec"
                ).value
            ),
        )
        self._control_rate = max(
            1.0, float(self.get_parameter("control_rate_hz").value)
        )

        self._max_linear_x = max(0.0, float(self.get_parameter("max_linear_x").value))
        self._max_linear_y = max(0.0, float(self.get_parameter("max_linear_y").value))
        self._max_angular_z = max(
            0.0, float(self.get_parameter("max_angular_z").value)
        )
        self._linear_slew = max(
            0.0, float(self.get_parameter("linear_slew_rate").value)
        )
        self._angular_slew = max(
            0.0, float(self.get_parameter("angular_slew_rate").value)
        )
        self._command_timeout = max(
            0.05, float(self.get_parameter("command_timeout_sec").value)
        )
        self._require_odom = bool(self.get_parameter("require_fresh_odom").value)
        self._odom_timeout = max(
            0.05, float(self.get_parameter("odom_timeout_sec").value)
        )
        self._expected_odom_frame = str(
            self.get_parameter("expected_odom_frame").value
        ).strip().lstrip("/")
        self._expected_base_frame = str(
            self.get_parameter("expected_base_frame").value
        ).strip().lstrip("/")
        self._expected_cloud_frame = str(
            self.get_parameter("expected_cloud_frame").value
        ).strip().lstrip("/")
        self._sensor_future_tolerance = max(
            0.0, float(self.get_parameter("sensor_future_tolerance_sec").value)
        )
        self._require_time_sync = bool(
            self.get_parameter("require_time_sync").value
        )
        self._time_sync_timeout = max(
            0.1,
            float(self.get_parameter("time_sync_status_timeout_sec").value),
        )
        self._stop_repeat = max(
            0.02, float(self.get_parameter("stop_repeat_sec").value)
        )

        self._front_guard = bool(
            self.get_parameter("front_obstacle_guard_enabled").value
        )
        self._cloud_timeout = max(
            0.05, float(self.get_parameter("front_cloud_timeout_sec").value)
        )
        self._cloud_fail_closed = bool(
            self.get_parameter("fail_closed_on_cloud_timeout").value
        )
        self._require_cloud_to_arm = bool(
            self.get_parameter("require_fresh_cloud_to_arm").value
        )
        self._front_min_x, self._front_max_x = sorted(
            (
                float(self.get_parameter("front_min_x").value),
                float(self.get_parameter("front_max_x").value),
            )
        )
        self._front_half_width = abs(
            float(self.get_parameter("front_half_width").value)
        )
        self._front_min_z, self._front_max_z = sorted(
            (
                float(self.get_parameter("front_min_z").value),
                float(self.get_parameter("front_max_z").value),
            )
        )
        self._front_min_points = max(
            1, int(self.get_parameter("front_min_points").value)
        )
        self._front_sample_stride = max(
            1, int(self.get_parameter("front_cloud_sample_stride").value)
        )
        self._cloud_health_min_points = max(
            1, int(self.get_parameter("front_cloud_health_min_points").value)
        )
        self._cloud_health_min_range = max(
            0.0,
            float(self.get_parameter("front_cloud_health_min_range").value),
        )
        self._cloud_health_max_range = max(
            self._cloud_health_min_range,
            float(self.get_parameter("front_cloud_health_max_range").value),
        )

        self._require_noshm = bool(
            self.get_parameter("require_noshm_runtime").value
        )
        self._noshm_fragment = str(
            self.get_parameter("noshm_library_fragment").value
        )

        self._armed = False
        self._target = TwistCommand()
        self._output = TwistCommand()
        self._last_command_at: Optional[float] = None
        self._last_odom_at: Optional[float] = None
        self._last_cloud_at: Optional[float] = None
        self._front_blocked = False
        self._last_control_at = time.monotonic()
        self._last_stop_at = -1.0e9
        self._last_gate_reason = "startup"
        self._control_deadline_announced = False
        self._log_times: Dict[str, float] = {}
        self._shutdown_started = False
        self._worker_was_armed = False
        self._time_sync_interlock = TimeSyncInterlock()
        self._safety_fault = PermanentFaultLatch()
        self._last_time_sync_status_at: Optional[float] = None
        self._post_worker_refresh_after: Optional[float] = None
        self._post_worker_refresh_deadline: Optional[float] = None

        # The ROS process never imports Unitree's Python SDK.  This proxy owns
        # a private socket to a freshly exec'd, non-ROS worker process.
        self._motion_worker: Optional[MotionWorkerProxy] = None
        self._next_connect_at = 0.0

        self.create_subscription(
            Twist,
            str(self.get_parameter("cmd_vel_topic").value),
            self._command_callback,
            10,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self._odom_callback,
            qos_profile_sensor_data,
        )
        if self._front_guard:
            self.create_subscription(
                PointCloud2,
                str(self.get_parameter("front_cloud_topic").value),
                self._cloud_callback,
                qos_profile_sensor_data,
            )
        sync_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("time_sync_status_topic").value),
            self._time_sync_callback,
            sync_qos,
        )
        self.create_service(
            SetBool,
            str(self.get_parameter("enable_service").value),
            self._enable_callback,
        )
        self.create_service(
            Trigger,
            str(self.get_parameter("stop_service").value),
            self._stop_callback,
        )
        self.get_logger().warning(
            "motion bridge started DISARMED; it never stands the robot automatically"
        )

    def _warn_limited(self, key: str, message: str, period: float = 5.0) -> None:
        now = time.monotonic()
        if now - self._log_times.get(key, -1.0e9) >= period:
            self._log_times[key] = now
            self.get_logger().warning(message)

    def _disarm_for_fault(self, reason: str, permanent: bool = False) -> None:
        if permanent:
            reason = self._safety_fault.latch(reason)
        was_armed = self._armed
        self._armed = False
        self._post_worker_refresh_after = None
        self._post_worker_refresh_deadline = None
        self._target = TwistCommand()
        self._output = TwistCommand()
        self._last_command_at = None
        sent = self._send_stop(force=True)
        self._close_worker()
        qualifier = "LATCHED " if permanent else ""
        if was_armed or permanent:
            self.get_logger().error(
                "motion DISARMED by %ssafety fault: %s" % (qualifier, reason)
            )
        if was_armed and not sent:
            self.get_logger().error(
                "StopMove could not be confirmed while disarming"
            )

    def _time_sync_callback(self, message: String) -> None:
        with self._state_lock:
            try:
                payload = json.loads(message.data)
            except (TypeError, ValueError) as error:
                reason = self._time_sync_interlock.latch(
                    "invalid time-sync status JSON: {}".format(error)
                )
            else:
                reason = self._time_sync_interlock.update(payload)
            self._last_time_sync_status_at = time.monotonic()
            if reason:
                # Set _armed false before the blocking SDK StopMove RPC.
                self._disarm_for_fault(reason, permanent=True)

    def _time_sync_is_fresh_and_locked(self, now: float) -> bool:
        if not self._require_time_sync:
            return True
        return (
            self._time_sync_interlock.ready
            and self._last_time_sync_status_at is not None
            and -0.1
            <= now - self._last_time_sync_status_at
            <= self._time_sync_timeout
        )

    def _close_worker(self) -> None:
        worker = self._motion_worker
        worker_was_armed = self._worker_was_armed or self._armed
        self._motion_worker = None
        self._worker_was_armed = False
        if worker is None:
            return
        try:
            worker.close()
        except Exception as exc:
            if worker_was_armed:
                self._safety_fault.latch(
                    "motion worker could not be reaped: {}".format(exc)
                )
            self.get_logger().error("motion worker reap failed: %s" % exc)

    def _start_worker_if_due(self, force: bool = False) -> bool:
        if self._motion_worker is not None and self._motion_worker.alive:
            return True
        if self._motion_worker is not None:
            self._close_worker()
        now = time.monotonic()
        if not force and now < self._next_connect_at:
            return False
        try:
            self._motion_worker = MotionWorkerProxy.start(
                network_interface=self._interface,
                sdk_timeout_sec=self._sdk_timeout,
                startup_timeout_sec=self._worker_startup_timeout,
                rpc_timeout_sec=self._worker_rpc_timeout,
                reap_timeout_sec=self._worker_reap_timeout,
                sdk_python_path=self._sdk_path,
                cyclonedds_python_path=self._cyclonedds_python_path,
                require_noshm_runtime=self._require_noshm,
                noshm_library_fragment=self._noshm_fragment,
            )
            self._worker_was_armed = False
            self.get_logger().info(
                "isolated SportClient worker is ready; motion remains DISARMED"
            )
            return True
        except Exception as exc:
            self._close_worker()
            self._next_connect_at = now + self._reconnect_delay
            self._warn_limited(
                "sdk_connect", "isolated SportClient worker failed to start: %s" % exc
            )
            return False

    def _worker_failure(
        self, operation: str, error: Any, try_best_effort_stop: bool
    ) -> None:
        worker = self._motion_worker
        post_arm = self._worker_was_armed or self._armed
        self._armed = False
        self._post_worker_refresh_after = None
        self._post_worker_refresh_deadline = None
        self._target = TwistCommand()
        self._output = TwistCommand()
        self._last_command_at = None
        stop_error: Optional[Exception] = None
        if try_best_effort_stop and worker is not None:
            try:
                worker.stop_move()
            except Exception as exc:
                stop_error = exc
        self._close_worker()
        self._next_connect_at = time.monotonic() + self._reconnect_delay
        detail = "{} failed ({})".format(operation, error)
        if stop_error is not None:
            detail += "; best-effort StopMove was unconfirmed ({})".format(stop_error)
        # StopMove must always be confirmed once a worker exists.  Any RPC/SDK
        # fault after arming also requires a complete-stack restart.
        restart_required = post_arm or operation == "StopMove"
        if restart_required:
            latched = self._safety_fault.latch(detail)
            self.get_logger().error(
                "%s; bridge DISARMED with LATCHED fault: %s"
                % (detail, latched)
            )
        else:
            self.get_logger().error("%s; bridge remains DISARMED" % detail)

    def _send_stop(self, force: bool = False) -> bool:
        worker = self._motion_worker
        if worker is None:
            if self._armed or self._worker_was_armed:
                self._worker_failure(
                    "StopMove",
                    MotionRpcError("motion worker is unavailable"),
                    try_best_effort_stop=False,
                )
            return False
        now = time.monotonic()
        if not force and now - self._last_stop_at < self._stop_repeat:
            return True
        try:
            worker.stop_move()
            self._last_stop_at = now
            return True
        except Exception as exc:
            self._worker_failure(
                "StopMove", exc, try_best_effort_stop=False
            )
            return False

    def _send_move(self, command: TwistCommand) -> bool:
        worker = self._motion_worker
        if worker is None:
            self._worker_failure(
                "Move",
                MotionRpcError("motion worker is unavailable"),
                try_best_effort_stop=False,
            )
            return False
        try:
            worker.move(command.linear_x, command.linear_y, command.angular_z)
            return True
        except Exception as exc:
            # A nonzero Move response can leave the private channel usable, so
            # attempt StopMove once before closing it.  Transport failures
            # still converge on worker EOF, whose child-side finalizer stops.
            self._worker_failure(
                "Move", exc, try_best_effort_stop=True
            )
            return False

    def _command_callback(self, message: Twist) -> None:
        with self._state_lock:
            # Commands received while disarmed are discarded so arming can
            # never activate a stale, latched velocity.
            if not self._armed or self._post_worker_refresh_after is not None:
                return
            command = TwistCommand(
                message.linear.x, message.linear.y, message.angular.z
            )
            self._target = clamp_twist(
                command,
                self._max_linear_x,
                self._max_linear_y,
                self._max_angular_z,
            )
            self._last_command_at = time.monotonic()

    def _header_stamp_fresh(self, message: Any, timeout_sec: float) -> bool:
        return source_timestamp_is_fresh(
            now_nanoseconds=self.get_clock().now().nanoseconds,
            stamp_sec=message.header.stamp.sec,
            stamp_nanosec=message.header.stamp.nanosec,
            timeout_sec=timeout_sec,
            future_tolerance_sec=self._sensor_future_tolerance,
        )

    def _odom_callback(self, message: Odometry) -> None:
        parent = str(message.header.frame_id).strip().lstrip("/")
        child = str(message.child_frame_id).strip().lstrip("/")
        pose = message.pose.pose
        values = (
            pose.position.x,
            pose.position.y,
            pose.position.z,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        quaternion_norm = sum(float(value) * float(value) for value in values[3:])
        if (
            parent != self._expected_odom_frame
            or child != self._expected_base_frame
            or not all(math.isfinite(float(value)) for value in values)
            or quaternion_norm < 1.0e-12
            or not self._header_stamp_fresh(message, self._odom_timeout)
        ):
            with self._state_lock:
                self._warn_limited(
                    "odom_invalid", "odometry safety sample rejected"
                )
            return
        with self._state_lock:
            self._last_odom_at = time.monotonic()

    def _cloud_callback(self, message: PointCloud2) -> None:
        try:
            frame = str(message.header.frame_id).strip().lstrip("/")
            if frame != self._expected_cloud_frame:
                raise ValueError("unexpected cloud frame %r" % frame)
            if not self._header_stamp_fresh(message, self._cloud_timeout):
                raise ValueError("stale, zero-stamped, or future cloud")
            points = iter_xyz(
                data=message.data,
                width=message.width,
                height=message.height,
                point_step=message.point_step,
                row_step=message.row_step,
                is_bigendian=message.is_bigendian,
                fields=message.fields,
                sample_stride=self._front_sample_stride,
            )
            observation = assess_front_cloud(
                points,
                min_x=self._front_min_x,
                max_x=self._front_max_x,
                half_width=self._front_half_width,
                min_z=self._front_min_z,
                max_z=self._front_max_z,
                obstacle_min_points=self._front_min_points,
                health_min_points=self._cloud_health_min_points,
                health_min_range=self._cloud_health_min_range,
                health_max_range=self._cloud_health_max_range,
            )
            if not observation.healthy:
                raise ValueError(
                    "only {} plausible returns; at least {} required".format(
                        observation.plausible_points,
                        self._cloud_health_min_points,
                    )
                )
            with self._state_lock:
                self._front_blocked = observation.front_blocked
                self._last_cloud_at = time.monotonic()
        except Exception as exc:
            # Do not refresh the timestamp: a persistent decode failure becomes
            # a stale-cloud stop when fail-closed behavior is selected.
            with self._state_lock:
                self._warn_limited(
                    "cloud_decode", "front cloud rejected: %s" % exc
                )

    def _odom_is_fresh(self, now: float) -> bool:
        return self._last_odom_at is not None and (
            -0.1 <= now - self._last_odom_at <= self._odom_timeout
        )

    def _cloud_is_fresh(self, now: float) -> bool:
        return self._last_cloud_at is not None and (
            -0.1 <= now - self._last_cloud_at <= self._cloud_timeout
        )

    def _post_worker_sensors_refreshed(self, now: float) -> bool:
        """Require callbacks received after blocking SportClient startup."""

        after = self._post_worker_refresh_after
        if after is None:
            return True
        time_ready = (
            not self._require_time_sync
            or (
                self._last_time_sync_status_at is not None
                and self._last_time_sync_status_at >= after
                and self._time_sync_is_fresh_and_locked(now)
            )
        )
        odom_ready = (
            not self._require_odom
            or (
                self._last_odom_at is not None
                and self._last_odom_at >= after
                and self._odom_is_fresh(now)
            )
        )
        cloud_required = self._front_guard and self._cloud_fail_closed
        cloud_ready = (
            not cloud_required
            or (
                self._last_cloud_at is not None
                and self._last_cloud_at >= after
                and self._cloud_is_fresh(now)
            )
        )
        return time_ready and odom_ready and cloud_ready

    def _enable_callback(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        with self._state_lock:
            return self._enable_callback_locked(request, response)

    def _enable_callback_locked(
        self, request: SetBool.Request, response: SetBool.Response
    ) -> SetBool.Response:
        if not request.data:
            was_armed = self._armed or self._worker_was_armed
            had_worker = self._motion_worker is not None
            self._armed = False
            self._post_worker_refresh_after = None
            self._post_worker_refresh_deadline = None
            self._target = TwistCommand()
            self._output = TwistCommand()
            self._last_command_at = None
            sent = self._send_stop(force=True) if had_worker or was_armed else True
            self._close_worker()
            response.success = sent
            response.message = (
                "motion disabled, StopMove confirmed, and worker reaped"
                if sent and had_worker
                else (
                    "motion already disabled; no SDK worker was running"
                    if sent
                    else "motion disabled but StopMove was unconfirmed; restart required"
                )
            )
            self.get_logger().warning("motion DISARMED by service request")
            return response

        if self._safety_fault.faulted:
            response.success = False
            response.message = (
                "motion safety fault is latched ({}); restart the complete "
                "stack".format(self._safety_fault.reason)
            )
            return response
        if self._time_sync_interlock.fault_reason:
            response.success = False
            response.message = (
                "time-sync fault is latched; restart the complete stack"
            )
            return response
        now = time.monotonic()
        if not self._time_sync_is_fresh_and_locked(now):
            response.success = False
            response.message = (
                "fresh locked time-sync status is required; bridge remains disarmed"
            )
            return response
        if self._require_odom and not self._odom_is_fresh(now):
            response.success = False
            response.message = "fresh odometry is required; bridge remains disarmed"
            return response
        if (
            self._front_guard
            and self._cloud_fail_closed
            and self._require_cloud_to_arm
            and not self._cloud_is_fresh(now)
        ):
            response.success = False
            response.message = "fresh valid LiDAR is required; bridge remains disarmed"
            return response

        # This is the only creation point for a motion worker.  Failed guard
        # checks above therefore cannot load or initialize the Unitree SDK.
        if not self._start_worker_if_due(force=True):
            response.success = False
            response.message = (
                "isolated SportClient worker is unavailable; bridge remains disarmed"
            )
            return response
        self._target = TwistCommand()
        self._output = TwistCommand()
        self._last_command_at = None
        if not self._send_stop(force=True):
            response.success = False
            response.message = (
                "initial StopMove was unconfirmed; restart the complete stack"
            )
            return response
        # Worker startup and the initial RPC block this node's single-threaded
        # executor. Samples that were fresh before startup can therefore look
        # stale at the next control tick. Keep StopMove asserted and discard
        # commands until every required callback runs after this point.
        refresh_started = time.monotonic()
        self._post_worker_refresh_after = refresh_started
        self._post_worker_refresh_deadline = (
            refresh_started + self._post_worker_refresh_timeout
        )
        self._armed = True
        self._worker_was_armed = True
        response.success = True
        response.message = (
            "motion enabled; waiting for fresh post-start sensors and a new cmd_vel"
        )
        self.get_logger().warning(
            "motion ARMED; the operator remains responsible for the physical E-stop"
        )
        return response

    def _stop_callback(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        with self._state_lock:
            return self._stop_callback_locked(_request, response)

    def _stop_callback_locked(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        # Stop is deliberately latching: a separate enable call is required.
        was_armed = self._armed or self._worker_was_armed
        had_worker = self._motion_worker is not None
        self._armed = False
        self._post_worker_refresh_after = None
        self._post_worker_refresh_deadline = None
        self._target = TwistCommand()
        self._output = TwistCommand()
        self._last_command_at = None
        sent = self._send_stop(force=True) if had_worker or was_armed else True
        self._close_worker()
        response.success = sent
        response.message = (
            "motion DISARMED, StopMove confirmed, and worker reaped"
            if sent and had_worker
            else (
                "motion already DISARMED; no SDK worker was running"
                if sent
                else "motion DISARMED but StopMove was unconfirmed; restart required"
            )
        )
        self.get_logger().warning("latched motion stop requested")
        return response

    def _control_tick(self) -> None:
        with self._state_lock:
            self._control_tick_locked()

    def _control_tick_locked(self) -> None:
        if self._shutdown_started:
            return
        if not self._control_deadline_announced:
            self.get_logger().info(
                "motion safety control deadline active at %.2f Hz"
                % self._control_rate
            )
            self._control_deadline_announced = True
        now = time.monotonic()
        dt = min(0.25, max(0.0, now - self._last_control_at))
        self._last_control_at = now

        if self._armed and self._post_worker_refresh_after is not None:
            if self._post_worker_sensors_refreshed(now):
                self._post_worker_refresh_after = None
                self._post_worker_refresh_deadline = None
                self._target = TwistCommand()
                self._output = TwistCommand()
                self._last_command_at = None
                self.get_logger().info(
                    "fresh post-start time, odometry, and LiDAR received; "
                    "motion command gate is ready"
                )
            elif (
                self._post_worker_refresh_deadline is not None
                and now > self._post_worker_refresh_deadline
            ):
                self._disarm_for_fault(
                    "fresh sensors were not received after SportClient startup",
                    permanent=True,
                )
                return
            else:
                self._target = TwistCommand()
                self._output = TwistCommand()
                self._last_command_at = None
                self._send_stop()
                if self._last_gate_reason != "post_start_sensor_refresh":
                    self.get_logger().info(
                        "motion gate: %s -> post_start_sensor_refresh"
                        % self._last_gate_reason
                    )
                    self._last_gate_reason = "post_start_sensor_refresh"
                return

        if self._armed and not self._time_sync_is_fresh_and_locked(now):
            self._disarm_for_fault(
                "time-sync status is stale or unlocked", permanent=True
            )
        elif self._armed and self._require_odom and not self._odom_is_fresh(now):
            self._disarm_for_fault("odometry became stale", permanent=True)
        elif (
            self._armed
            and self._front_guard
            and self._cloud_fail_closed
            and not self._cloud_is_fresh(now)
        ):
            self._disarm_for_fault("front LiDAR became stale", permanent=True)

        decision = evaluate_motion_gate(
            armed=self._armed,
            command=self._target,
            now=now,
            last_command_at=self._last_command_at,
            command_timeout=self._command_timeout,
            require_fresh_odom=self._require_odom,
            last_odom_at=self._last_odom_at,
            odom_timeout=self._odom_timeout,
            obstacle_guard_enabled=self._front_guard,
            front_blocked=self._front_blocked,
            last_cloud_at=self._last_cloud_at,
            cloud_timeout=self._cloud_timeout,
            fail_closed_on_cloud_timeout=self._cloud_fail_closed,
        )

        # If a reverse command arrives while forward velocity is still being
        # slewed down, an active/stale front guard must stop that residual
        # forward motion immediately. Reverse can begin on the following tick.
        if (
            decision.allowed
            and self._front_guard
            and self._output.linear_x > 1.0e-4
            and (
                self._front_blocked
                or (
                    self._cloud_fail_closed
                    and not self._cloud_is_fresh(now)
                )
            )
        ):
            decision = GateDecision(
                False,
                "front_obstacle"
                if self._front_blocked
                else "front_cloud_stale",
            )

        if not decision.allowed:
            self._output = TwistCommand()
            self._send_stop()
        else:
            self._output = slew_twist(
                self._output,
                self._target,
                self._linear_slew,
                self._angular_slew,
                dt,
            )
            if self._output.is_zero():
                self._send_stop()
            else:
                self._send_move(self._output)

        if decision.reason != self._last_gate_reason:
            previous = self._last_gate_reason
            self._last_gate_reason = decision.reason
            self.get_logger().info(
                "motion gate: %s -> %s" % (previous, decision.reason)
            )

    def stop_for_shutdown(self) -> None:
        with self._state_lock:
            self._stop_for_shutdown_locked()

    def _stop_for_shutdown_locked(self) -> None:
        if self._shutdown_started:
            return
        self._shutdown_started = True
        self._armed = False
        self._post_worker_refresh_after = None
        self._post_worker_refresh_deadline = None
        self._target = TwistCommand()
        self._output = TwistCommand()
        if self._motion_worker is None:
            return
        repetitions = max(
            1, int(self.get_parameter("shutdown_stop_repetitions").value)
        )
        interval = max(
            0.0, float(self.get_parameter("shutdown_stop_interval_sec").value)
        )
        for index in range(repetitions):
            if not self._send_stop(force=True):
                break
            if index + 1 < repetitions and interval > 0.0:
                time.sleep(interval)
        self._close_worker()


def main(args: Optional[Any] = None) -> None:
    rclpy.init(args=args)
    node: Optional[MotionBridge] = None
    executor: Optional[SingleThreadedExecutor] = None
    periodic: Optional[PeriodicScheduler] = None
    try:
        node = MotionBridge()
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        periodic = PeriodicScheduler(
            period_sec=1.0 / node._control_rate,
            schedule=executor.create_task,
            work=node._control_tick,
            keep_running=rclpy.ok,
            on_failure=rclpy.shutdown,
            name="motion-safety-control",
        )
        periodic.start()
        executor.spin()
        periodic.raise_if_failed()
    except KeyboardInterrupt:
        pass
    finally:
        if periodic is not None:
            periodic.stop()
        if node is not None:
            node.stop_for_shutdown()
            if executor is not None:
                executor.remove_node(node)
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
