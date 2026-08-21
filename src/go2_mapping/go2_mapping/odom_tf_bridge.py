"""Publish one normalized odometry-to-base transform from a Go2 Odometry stream."""

import json
import math
import time

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from tf2_ros import TransformBroadcaster
from std_msgs.msg import String

from .time_sync_guard import TimeSyncStatusGuard


def _normal_frame(frame: str) -> str:
    return str(frame).strip().lstrip("/")


class OdomTfBridge(Node):
    """The package's sole TF authority; the mapper intentionally publishes no TF."""

    def __init__(self) -> None:
        super().__init__("go2_odom_tf_bridge")
        self.declare_parameter("odom_topic", "/go2/odom")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("require_child_frame_match", True)
        self.declare_parameter("require_time_sync_status", True)
        self.declare_parameter(
            "time_sync_status_topic", "/go2/time_sync/status"
        )
        self.declare_parameter("max_message_age_sec", 2.0)
        self.declare_parameter("future_tolerance_sec", 1.0)
        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.odom_frame = _normal_frame(self.get_parameter("odom_frame").value)
        self.base_frame = _normal_frame(self.get_parameter("base_frame").value)
        self.require_child_frame_match = bool(
            self.get_parameter("require_child_frame_match").value
        )
        self.max_message_age_sec = float(
            self.get_parameter("max_message_age_sec").value
        )
        self.future_tolerance_sec = float(
            self.get_parameter("future_tolerance_sec").value
        )
        self._time_sync_guard = TimeSyncStatusGuard(
            required=bool(self.get_parameter("require_time_sync_status").value)
        )
        if not self.odom_frame or not self.base_frame:
            raise ValueError("odom_frame and base_frame must not be empty")
        if self.odom_frame == self.base_frame:
            raise ValueError("odom_frame and base_frame must be different")
        if self.max_message_age_sec <= 0.0 or self.future_tolerance_sec < 0.0:
            raise ValueError("timestamp limits are invalid")

        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._broadcaster = TransformBroadcaster(self)
        self._subscription = self.create_subscription(
            Odometry, self.odom_topic, self._callback, sensor_qos
        )
        status_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._time_sync_subscription = self.create_subscription(
            String,
            str(self.get_parameter("time_sync_status_topic").value),
            self._time_sync_callback,
            status_qos,
        )
        self._last_warning = 0.0
        self.get_logger().info(
            "bridging %s to TF %s -> %s"
            % (self.odom_topic, self.odom_frame, self.base_frame)
        )

    def _warn_throttled(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_warning >= 5.0:
            self.get_logger().warning(message)
            self._last_warning = now

    def _time_sync_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError) as error:
            reason = self._time_sync_guard.latch(
                "invalid time-sync status JSON: {}".format(error)
            )
        else:
            reason = self._time_sync_guard.update(payload)
        if reason:
            self._warn_throttled(
                "TF time-sync fault latched: {}; restart the complete stack".format(
                    reason
                )
            )

    def _callback(self, message: Odometry) -> None:
        if not self._time_sync_guard.ready:
            return
        parent = _normal_frame(message.header.frame_id)
        child = _normal_frame(message.child_frame_id)
        if parent != self.odom_frame:
            self._warn_throttled(
                "odometry parent frame {!r} does not match {!r}".format(
                    parent, self.odom_frame
                )
            )
            return
        if self.require_child_frame_match and child != self.base_frame:
            self._warn_throttled(
                "odometry child frame {!r} does not match {!r}".format(
                    child, self.base_frame
                )
            )
            return

        stamp_ns = (
            int(message.header.stamp.sec) * 1000000000
            + int(message.header.stamp.nanosec)
        )
        now_ns = self.get_clock().now().nanoseconds
        if stamp_ns <= 0:
            self._warn_throttled("odometry with a zero timestamp was rejected")
            return
        if now_ns > 0:
            age_sec = (now_ns - stamp_ns) / 1.0e9
            if (
                age_sec > self.max_message_age_sec
                or age_sec < -self.future_tolerance_sec
            ):
                self._warn_throttled("stale or future-dated odometry TF was rejected")
                return

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
        if not all(math.isfinite(value) for value in values):
            self._warn_throttled("non-finite odometry pose was rejected")
            return
        quaternion_norm = math.sqrt(sum(value * value for value in values[3:]))
        if quaternion_norm < 1.0e-6:
            self._warn_throttled("zero-length odometry quaternion was rejected")
            return

        transform = TransformStamped()
        # Preserve the sensor/LIO timestamp exactly; do not restamp at receipt.
        transform.header.stamp = message.header.stamp
        transform.header.frame_id = self.odom_frame
        transform.child_frame_id = self.base_frame
        transform.transform.translation.x = pose.position.x
        transform.transform.translation.y = pose.position.y
        transform.transform.translation.z = pose.position.z
        transform.transform.rotation.x = pose.orientation.x / quaternion_norm
        transform.transform.rotation.y = pose.orientation.y / quaternion_norm
        transform.transform.rotation.z = pose.orientation.z / quaternion_norm
        transform.transform.rotation.w = pose.orientation.w / quaternion_norm
        self._broadcaster.sendTransform(transform)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OdomTfBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
