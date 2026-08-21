"""ROS 2 boundary that maps native Go2 sensor time onto the host clock."""

import json
import threading
import time
from typing import Dict
import uuid

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String

from .coordinates import (
    AXIS_SIGNS,
    TRANSFORM_NAME,
    transform_odometry_in_place,
    transform_pointcloud_in_place,
)
from .periodic_worker import PeriodicScheduler
from .time_sync import OdomPoseGuard, SharedTimeEstimator, TimeSyncReset


NANOSECONDS = 1000000000


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * NANOSECONDS + int(stamp.nanosec)


def assign_stamp(stamp, value_ns: int) -> None:
    """Assign a positive timestamp after checking ROS Time's int32 seconds."""

    value_ns = int(value_ns)
    seconds, nanoseconds = divmod(value_ns, NANOSECONDS)
    if seconds < 0 or seconds > 2147483647:
        raise ValueError("normalized timestamp is outside ROS Time range")
    stamp.sec = seconds
    stamp.nanosec = nanoseconds


class SensorTimeBridge(Node):
    """Normalize odometry and both LiDAR clouds with one fail-closed clock."""

    def __init__(self) -> None:
        super().__init__("sensor_time_bridge")
        self._lock = threading.RLock()
        self._last_log_monotonic = 0.0
        self._last_event = "waiting for odometry clock warmup"
        self._status_deadline_announced = False
        self._fault_latched = False
        self._fault_reason = ""
        self._instance_id = uuid.uuid4().hex

        self._declare("raw_odom_topic", "/utlidar/robot_odom")
        self._declare("raw_cloud_base_topic", "/utlidar/cloud_base")
        self._declare("raw_cloud_deskewed_topic", "/utlidar/cloud_deskewed")
        self._declare("odom_topic", "/go2/odom")
        self._declare("cloud_base_topic", "/go2/lidar/cloud_base")
        self._declare(
            "cloud_deskewed_topic", "/go2/lidar/cloud_deskewed"
        )
        self._declare("status_topic", "/go2/time_sync/status")
        self._declare("warmup_samples", 30)
        self._declare("rolling_window_samples", 200)
        self._declare("clock_jump_threshold_sec", 1.0)
        self._declare("future_tolerance_sec", 0.25)
        self._declare("max_output_age_sec", 5.0)
        self._declare("status_publish_rate", 2.0)
        self._declare("expected_odom_frame", "odom")
        self._declare("expected_base_frame", "base_link")
        self._declare("expected_cloud_base_frame", "base_link")
        self._declare("expected_cloud_deskewed_frame", "odom")
        self._declare("max_odom_translation_step", 0.75)
        self._declare("max_odom_translation_speed", 3.0)
        self._declare("max_odom_angular_step", 1.5707963267948966)
        self._declare("max_odom_angular_speed", 8.0)

        self._estimator = SharedTimeEstimator(
            warmup_samples=int(self._value("warmup_samples")),
            window_size=int(self._value("rolling_window_samples")),
            clock_jump_threshold_ns=self._seconds_ns(
                "clock_jump_threshold_sec"
            ),
            future_tolerance_ns=self._seconds_ns("future_tolerance_sec"),
            max_output_age_ns=self._seconds_ns("max_output_age_sec"),
        )
        self._pose_guard = OdomPoseGuard(
            max_translation_step=float(
                self._value("max_odom_translation_step")
            ),
            max_translation_speed=float(
                self._value("max_odom_translation_speed")
            ),
            max_angular_step=float(self._value("max_odom_angular_step")),
            max_angular_speed=float(self._value("max_odom_angular_speed")),
        )
        status_rate = float(self._value("status_publish_rate"))
        if status_rate <= 0.0:
            raise ValueError("status_publish_rate must be positive")
        self._status_publish_rate = status_rate

        raw_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        normalized_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        status_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self._odom_publisher = self.create_publisher(
            Odometry, str(self._value("odom_topic")), normalized_qos
        )
        self._base_publisher = self.create_publisher(
            PointCloud2, str(self._value("cloud_base_topic")), normalized_qos
        )
        self._deskewed_publisher = self.create_publisher(
            PointCloud2,
            str(self._value("cloud_deskewed_topic")),
            normalized_qos,
        )
        self._status_publisher = self.create_publisher(
            String, str(self._value("status_topic")), status_qos
        )

        self._odom_subscription = self.create_subscription(
            Odometry,
            str(self._value("raw_odom_topic")),
            self._odom_callback,
            raw_qos,
        )
        self._base_subscription = self.create_subscription(
            PointCloud2,
            str(self._value("raw_cloud_base_topic")),
            self._base_callback,
            raw_qos,
        )
        self._deskewed_subscription = self.create_subscription(
            PointCloud2,
            str(self._value("raw_cloud_deskewed_topic")),
            self._deskewed_callback,
            raw_qos,
        )
        self._counters: Dict[str, int] = {
            "received_odom": 0,
            "received_cloud_base": 0,
            "received_cloud_deskewed": 0,
            "published_odom": 0,
            "published_cloud_base": 0,
            "published_cloud_deskewed": 0,
            "dropped_warmup": 0,
            "dropped_invalid": 0,
            "dropped_fault_latched": 0,
        }
        self.get_logger().info(
            "native sensor clock boundary warming with %d odometry samples"
            % self._estimator.warmup_samples
        )

    def _declare(self, name: str, default) -> None:
        self.declare_parameter(name, default)

    def _value(self, name: str):
        return self.get_parameter(name).value

    def _seconds_ns(self, name: str) -> int:
        value = float(self._value(name))
        if value < 0.0:
            raise ValueError("{} must not be negative".format(name))
        return int(round(value * NANOSECONDS))

    def _receipt_ns(self) -> int:
        return int(self.get_clock().now().nanoseconds)

    def _rejected(self, reason: str) -> None:
        self._counters["dropped_invalid"] += 1
        self._last_event = reason
        now = time.monotonic()
        if now - self._last_log_monotonic >= 5.0:
            self.get_logger().warning(reason)
            self._last_log_monotonic = now

    @staticmethod
    def _frame(value: str) -> str:
        return str(value).strip().lstrip("/")

    def _latch_fault(self, reason: str) -> None:
        self._fault_latched = True
        self._fault_reason = str(reason)
        self._last_event = "FAULT LATCHED: {}".format(reason)
        self._counters["dropped_invalid"] += 1
        self.get_logger().error(
            "SENSOR TIME FAULT LATCHED: %s. No normalized sensor output will "
            "resume in this process; stop and restart the complete stack."
            % reason
        )
        # Publish the reliable fault status immediately; waiting for the 2 Hz
        # timer would extend the physical stop latency by up to 0.5 seconds.
        self._publish_status()

    def _drop_if_fault_latched(self) -> bool:
        if not self._fault_latched:
            return False
        self._counters["dropped_fault_latched"] += 1
        return True

    def _odom_callback(self, message: Odometry) -> None:
        receipt_ns = self._receipt_ns()
        source_ns = stamp_to_ns(message.header.stamp)
        with self._lock:
            self._counters["received_odom"] += 1
            if self._drop_if_fault_latched():
                return
            parent = self._frame(message.header.frame_id)
            child = self._frame(message.child_frame_id)
            expected_parent = self._frame(self._value("expected_odom_frame"))
            expected_child = self._frame(self._value("expected_base_frame"))
            if parent != expected_parent or child != expected_child:
                self._estimator.reset("unexpected odometry frames")
                self._latch_fault(
                    "odometry frames '{} -> {}' do not match '{} -> {}'".format(
                        parent, child, expected_parent, expected_child
                    )
                )
                return
            pose = message.pose.pose
            try:
                self._pose_guard.observe(
                    source_ns,
                    (pose.position.x, pose.position.y, pose.position.z),
                    (
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ),
                )
            except ValueError as error:
                self._estimator.reset(str(error))
                self._latch_fault(str(error))
                return
            try:
                normalized_ns = self._estimator.process_odometry(
                    source_ns, receipt_ns
                )
            except TimeSyncReset as error:
                self._latch_fault(str(error))
                return
            if normalized_ns is None:
                self._counters["dropped_warmup"] += 1
                self._last_event = "odometry clock warmup"
                return
            try:
                assign_stamp(message.header.stamp, normalized_ns)
                transform_odometry_in_place(message)
            except ValueError as error:
                self._estimator.reset(str(error))
                self._latch_fault(str(error))
                return
            self._odom_publisher.publish(message)
            self._counters["published_odom"] += 1
            self._last_event = "clock locked"

    def _base_callback(self, message: PointCloud2) -> None:
        self._cloud_callback(
            "cloud_base",
            "received_cloud_base",
            "published_cloud_base",
            self._base_publisher,
            message,
        )

    def _deskewed_callback(self, message: PointCloud2) -> None:
        self._cloud_callback(
            "cloud_deskewed",
            "received_cloud_deskewed",
            "published_cloud_deskewed",
            self._deskewed_publisher,
            message,
        )

    def _cloud_callback(
        self,
        stream: str,
        received_counter: str,
        published_counter: str,
        publisher,
        message: PointCloud2,
    ) -> None:
        receipt_ns = self._receipt_ns()
        source_ns = stamp_to_ns(message.header.stamp)
        with self._lock:
            self._counters[received_counter] += 1
            if self._drop_if_fault_latched():
                return
            expected_parameter = (
                "expected_cloud_base_frame"
                if stream == "cloud_base"
                else "expected_cloud_deskewed_frame"
            )
            actual_frame = self._frame(message.header.frame_id)
            expected_frame = self._frame(self._value(expected_parameter))
            if actual_frame != expected_frame:
                self._estimator.reset("unexpected {} frame".format(stream))
                self._latch_fault(
                    "{} frame '{}' does not match '{}'".format(
                        stream, actual_frame, expected_frame
                    )
                )
                return
            try:
                normalized_ns = self._estimator.process_sensor(
                    stream, source_ns, receipt_ns
                )
            except TimeSyncReset as error:
                self._latch_fault(str(error))
                return
            if normalized_ns is None:
                self._counters["dropped_warmup"] += 1
                self._last_event = "{} dropped during clock warmup".format(stream)
                return
            try:
                assign_stamp(message.header.stamp, normalized_ns)
                transform_pointcloud_in_place(message)
            except ValueError as error:
                self._estimator.reset(str(error))
                self._latch_fault(str(error))
                return
            publisher.publish(message)
            self._counters[published_counter] += 1

    def _publish_status(self) -> None:
        if not self._status_deadline_announced:
            self.get_logger().info(
                "time-sync status deadline active at %.2f Hz"
                % self._status_publish_rate
            )
            self._status_deadline_announced = True
        with self._lock:
            payload = self._estimator.status()
            payload["instance_id"] = self._instance_id
            if self._fault_latched:
                payload["state"] = "fault_latched"
            payload["fault_reason"] = self._fault_reason
            payload["last_event"] = self._last_event
            payload["coordinate_transform"] = TRANSFORM_NAME
            payload["coordinate_axis_signs"] = list(AXIS_SIGNS)
            payload["counters"] = dict(self._counters)
        message = String()
        message.data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self._status_publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SensorTimeBridge()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    periodic = PeriodicScheduler(
        period_sec=1.0 / node._status_publish_rate,
        schedule=executor.create_task,
        work=node._publish_status,
        keep_running=rclpy.ok,
        on_failure=rclpy.shutdown,
        name="sensor-time-status",
    )
    try:
        periodic.start()
        executor.spin()
        periodic.raise_if_failed()
    except KeyboardInterrupt:
        pass
    finally:
        periodic.stop()
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
