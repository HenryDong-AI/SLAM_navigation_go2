"""Convert the Go2 base-frame point cloud into a planar LaserScan.

Copyright (c) 2026 Go2 SLAM Navigation Maintainers. MIT License.
"""

from __future__ import annotations

from array import array
import math
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan, PointCloud2

from .cloud_math import normalize_frame_id, points_to_ranges, xyz_from_buffer


class CloudToScan(Node):
    def __init__(self) -> None:
        super().__init__("go2_cloud_to_scan")
        self.declare_parameter("cloud_topic", "/go2/lidar/cloud_base")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("output_frame", "base_link")
        self.declare_parameter("angle_min", -math.pi)
        self.declare_parameter("angle_max", math.pi)
        self.declare_parameter("angle_increment", math.radians(0.5))
        self.declare_parameter("range_min", 0.15)
        self.declare_parameter("range_max", 12.0)
        # The sensor boundary publishes REP-103 Z-up. This band is the exact
        # rotated equivalent of the validated native -0.20..+0.22 m band.
        self.declare_parameter("min_height", -0.22)
        self.declare_parameter("max_height", 0.20)
        self.declare_parameter("max_publish_rate", 15.0)

        self._angle_min = float(self.get_parameter("angle_min").value)
        self._angle_max = float(self.get_parameter("angle_max").value)
        self._angle_increment = float(self.get_parameter("angle_increment").value)
        self._range_min = float(self.get_parameter("range_min").value)
        self._range_max = float(self.get_parameter("range_max").value)
        self._min_height = float(self.get_parameter("min_height").value)
        self._max_height = float(self.get_parameter("max_height").value)
        self._frame = normalize_frame_id(
            self.get_parameter("output_frame").value
        )
        rate = max(0.1, float(self.get_parameter("max_publish_rate").value))
        self._minimum_period_ns = int(1e9 / rate)
        self._last_publish_ns = 0
        self._lock = threading.Lock()

        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._publisher = self.create_publisher(
            LaserScan, str(self.get_parameter("scan_topic").value), qos
        )
        self._subscription = self.create_subscription(
            PointCloud2,
            str(self.get_parameter("cloud_topic").value),
            self._cloud_callback,
            qos,
        )
        self.get_logger().info(
            "Converting %s to %s (height %.2f..%.2f m)"
            % (
                self.get_parameter("cloud_topic").value,
                self.get_parameter("scan_topic").value,
                self._min_height,
                self._max_height,
            )
        )

    def _cloud_callback(self, message: PointCloud2) -> None:
        now_ns = self.get_clock().now().nanoseconds
        with self._lock:
            if now_ns - self._last_publish_ns < self._minimum_period_ns:
                return
            self._last_publish_ns = now_ns
        input_frame = normalize_frame_id(message.header.frame_id)
        if self._frame and input_frame != self._frame:
            self.get_logger().warn(
                "Expected cloud frame %s, received %s; refusing an untransformed scan"
                % (self._frame, message.header.frame_id),
                throttle_duration_sec=5.0,
            )
            return
        try:
            points = xyz_from_buffer(
                bytes(message.data),
                int(message.point_step),
                message.fields,
                bigendian=bool(message.is_bigendian),
                width=int(message.width),
                height=int(message.height),
                row_step=int(message.row_step),
            )
            ranges = points_to_ranges(
                points,
                angle_min=self._angle_min,
                angle_max=self._angle_max,
                angle_increment=self._angle_increment,
                range_min=self._range_min,
                range_max=self._range_max,
                min_height=self._min_height,
                max_height=self._max_height,
            )
        except (ValueError, TypeError) as error:
            self.get_logger().error(f"Point cloud rejected: {error}")
            return

        scan = LaserScan()
        scan.header.stamp = message.header.stamp
        scan.header.frame_id = self._frame or input_frame
        scan.angle_min = self._angle_min
        scan.angle_increment = self._angle_increment
        scan.angle_max = self._angle_min + (len(ranges) - 1) * self._angle_increment
        scan.time_increment = 0.0
        scan.scan_time = self._minimum_period_ns / 1e9
        scan.range_min = self._range_min
        scan.range_max = self._range_max
        payload = array("f")
        payload.frombytes(
            np.ascontiguousarray(ranges, dtype=np.float32).tobytes()
        )
        scan.ranges = payload
        self._publisher.publish(scan)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CloudToScan()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
