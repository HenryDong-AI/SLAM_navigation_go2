#!/usr/bin/env python3
"""Fuse aligned D435i depth images into the Go2 voxel and occupancy maps."""

import json
import math
import os
import threading
import time
from array import array
from collections import OrderedDict, deque

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import String
from std_srvs.srv import Trigger

from go2_mapping.grid_map import LogOddsGrid
from go2_mapping.pointcloud import xyzrgb_to_float32_bytes
from go2_mapping.state_io import load_snapshot, save_snapshot
from go2_mapping.time_sync_guard import TimeSyncStatusGuard
from go2_mapping.voxel_map import VoxelAccumulator

from .geometry import (
    decode_color_image,
    decode_depth_image,
    depth_to_camera_points_rgb,
    pose_matrix,
)
from .geometry import rigid_transform, transform_points


def _stamp_ns(message):
    return int(message.header.stamp.sec) * 1_000_000_000 + int(
        message.header.stamp.nanosec
    )


class DepthMappingNode(Node):
    """Build the normal Go2 map products from a registered depth stream."""

    def __init__(self):
        super().__init__("go2_mapping_depthcam")
        self._lock = threading.RLock()
        self._warn_times = {}

        self._depth_topic = self._param(
            "depth_topic", "/go2/depth_camera/aligned_depth/image_raw"
        )
        self._color_topic = self._param(
            "color_topic", "/go2/depth_camera/color/image_raw"
        )
        self._camera_info_topic = self._param(
            "camera_info_topic", "/go2/depth_camera/aligned_depth/camera_info"
        )
        self._odom_topic = self._param("odom_topic", "/go2/odom")
        self._time_sync_topic = self._param(
            "time_sync_topic", "/go2/time_sync/status"
        )
        self._map_cloud_topic = self._param("map_cloud_topic", "/go2/map/cloud")
        self._map_topic = self._param("map_topic", "/map")
        self._status_topic = self._param(
            "mapping_status_topic", "/go2/mapping/status"
        )
        self._world_frame = self._param("world_frame", "odom")
        self._base_frame = self._param("base_frame", "base_link")
        self._expected_depth_frame = self._param(
            "expected_depth_frame", "d435i_color_optical_frame"
        )

        self._extrinsics_confirmed = bool(
            self._param("extrinsics_confirmed", False)
        )
        transform_values = self._param(
            "base_from_camera_optical",
            [
                0.0,
                0.0,
                1.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        )
        try:
            self._base_from_camera = rigid_transform(transform_values)
        except ValueError as exc:
            raise RuntimeError(
                "base_from_camera_optical is not a valid rigid transform"
            ) from exc

        self._processing_rate_hz = float(self._param("processing_rate_hz", 5.0))
        self._pixel_stride = max(1, int(self._param("pixel_stride", 4)))
        self._min_depth = float(self._param("min_depth", 0.20))
        self._max_depth = float(self._param("max_depth", 5.0))
        self._max_points = max(1, int(self._param("max_points_per_frame", 20000)))
        self._max_depth_age = float(self._param("max_depth_age_sec", 0.50))
        self._max_odom_delta = float(self._param("max_odom_delta_sec", 0.15))
        odom_buffer_size = max(2, int(self._param("odom_buffer_size", 100)))
        self._odom_buffer = deque(maxlen=odom_buffer_size)
        self._pair_buffer_size = max(
            2, int(self._param("rgb_pair_buffer_size", 4))
        )
        self._pending_depth = OrderedDict()
        self._pending_color = OrderedDict()

        self._voxel_size = float(self._param("voxel_size", 0.08))
        self._min_point_range = float(self._param("min_point_range", 0.20))
        self._max_point_range = float(self._param("max_point_range", 5.0))
        self._relative_min_z = float(self._param("min_relative_z", -0.50))
        self._relative_max_z = float(self._param("max_relative_z", 2.00))
        self._max_voxels = int(self._param("max_voxels", 300000))
        self._retention_radius = float(self._param("retention_radius", 0.0))
        self._grid_resolution = float(self._param("grid_resolution", 0.10))
        self._max_dense_grid_cells = int(
            self._param("max_dense_grid_cells", 2_000_000)
        )
        self._grid_max_cells = int(self._param("max_grid_cells", 500000))
        self._occupied_log_odds = float(self._param("hit_log_odds", 0.85))
        self._free_log_odds = float(self._param("miss_log_odds", -0.40))
        self._min_log_odds = float(self._param("min_log_odds", -4.0))
        self._max_log_odds = float(self._param("max_log_odds", 4.0))
        self._obstacle_min_height = float(
            self._param("obstacle_min_height", 0.12)
        )
        self._obstacle_max_height = float(
            self._param("obstacle_max_height", 1.50)
        )
        self._max_rays_per_update = int(
            self._param("max_rays_per_update", 4000)
        )
        self._max_ray_cells = int(self._param("max_ray_cells", 4096))
        self._publish_rate_hz = float(self._param("publish_rate_hz", 1.0))

        self._output_directory = os.path.expanduser(
            str(self._param("output_directory", "maps_depth_camera"))
        )
        self._auto_save_period = float(self._param("auto_save_period_sec", 60.0))
        self._load_state_path = str(self._param("load_state_path", ""))
        self._free_threshold = int(self._param("save_free_threshold", 25))
        self._occupied_threshold = int(
            self._param("save_occupied_threshold", 65)
        )
        time_sync_required = bool(self._param("time_sync_required", True))

        self._voxels = VoxelAccumulator(
            voxel_size=self._voxel_size,
            min_point_range=self._min_point_range,
            max_point_range=self._max_point_range,
            min_relative_z=self._relative_min_z,
            max_relative_z=self._relative_max_z,
            max_voxels=self._max_voxels,
            retention_radius=self._retention_radius,
        )
        self._grid = LogOddsGrid(
            resolution=self._grid_resolution,
            hit_log_odds=self._occupied_log_odds,
            miss_log_odds=self._free_log_odds,
            min_log_odds=self._min_log_odds,
            max_log_odds=self._max_log_odds,
            obstacle_min_height=self._obstacle_min_height,
            obstacle_max_height=self._obstacle_max_height,
            max_ray_range=self._max_point_range,
            max_cells=self._grid_max_cells,
            max_rays_per_update=self._max_rays_per_update,
            max_ray_cells=self._max_ray_cells,
        )
        self._time_guard = TimeSyncStatusGuard(required=time_sync_required)
        self._camera_info = None
        self._last_depth_stamp_ns = -1
        self._last_process_monotonic = 0.0
        self._last_sensor_stamp_ns = 0
        self._frames_received = 0
        self._frames_fused = 0
        self._frames_dropped = 0
        self._last_error = ""

        if self._load_state_path:
            self._load_snapshot(self._load_state_path)

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        transient_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        reliable_qos = QoSProfile(depth=10)

        self._depth_sub = self.create_subscription(
            Image, self._depth_topic, self._on_depth, sensor_qos
        )
        self._color_sub = self.create_subscription(
            Image, self._color_topic, self._on_color, sensor_qos
        )
        self._info_sub = self.create_subscription(
            CameraInfo, self._camera_info_topic, self._on_camera_info, sensor_qos
        )
        self._odom_sub = self.create_subscription(
            Odometry, self._odom_topic, self._on_odometry, sensor_qos
        )
        self._time_sub = self.create_subscription(
            String, self._time_sync_topic, self._on_time_sync, reliable_qos
        )
        self._cloud_pub = self.create_publisher(
            PointCloud2, self._map_cloud_topic, transient_qos
        )
        self._grid_pub = self.create_publisher(
            OccupancyGrid, self._map_topic, transient_qos
        )
        self._status_pub = self.create_publisher(
            String, self._status_topic, reliable_qos
        )
        self._save_service = self.create_service(
            Trigger, "/go2/map/save", self._on_save
        )
        self._reset_service = self.create_service(
            Trigger, "/go2/map/reset", self._on_reset
        )

        publish_period = 1.0 / max(0.1, self._publish_rate_hz)
        self._publish_timer = self.create_timer(publish_period, self._publish_maps)
        self._status_timer = self.create_timer(1.0, self._publish_status)
        self._autosave_timer = None
        if self._auto_save_period > 0.0:
            self._autosave_timer = self.create_timer(
                self._auto_save_period, self._autosave
            )

        self.get_logger().info(
            "D435i RGB-D mapper configured: color=%s depth=%s odom=%s output=%s"
            % (
                self._color_topic,
                self._depth_topic,
                self._odom_topic,
                self._output_directory,
            )
        )
        if not self._extrinsics_confirmed:
            self.get_logger().error(
                "CALIBRATION REQUIRED: depth fusion is disabled. Measure "
                "base_from_camera_optical and set extrinsics_confirmed:=true."
            )

    def _param(self, name, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _warn_throttled(self, key, text, period=5.0):
        now = time.monotonic()
        previous = self._warn_times.get(key, 0.0)
        if now - previous >= period:
            self.get_logger().warning(text)
            self._warn_times[key] = now

    def _on_camera_info(self, message):
        if len(message.k) < 9:
            self._warn_throttled("camera_info_k", "CameraInfo K matrix is incomplete")
            return
        fx = float(message.k[0])
        fy = float(message.k[4])
        cx = float(message.k[2])
        cy = float(message.k[5])
        values = (fx, fy, cx, cy)
        if not all(math.isfinite(value) for value in values) or fx <= 0.0 or fy <= 0.0:
            self._warn_throttled("camera_info_intrinsics", "invalid depth intrinsics")
            return
        with self._lock:
            self._camera_info = {
                "width": int(message.width),
                "height": int(message.height),
                "frame": str(message.header.frame_id),
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
            }

    def _on_odometry(self, message):
        if message.header.frame_id and message.header.frame_id != self._world_frame:
            self._warn_throttled(
                "odom_frame",
                "ignoring odometry in frame %s (expected %s)"
                % (message.header.frame_id, self._world_frame),
            )
            return
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation
        pose_values = np.array(
            [
                position.x,
                position.y,
                position.z,
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(pose_values)):
            self._warn_throttled("odom_finite", "ignoring non-finite odometry")
            return
        try:
            transform = pose_matrix(pose_values[:3], pose_values[3:])
        except ValueError:
            self._warn_throttled("odom_quaternion", "ignoring invalid odometry pose")
            return
        with self._lock:
            if not self._time_guard.ready:
                return
            self._odom_buffer.append((_stamp_ns(message), transform))

    def _on_time_sync(self, message):
        became_faulted = False
        with self._lock:
            was_faulted = bool(self._time_guard.fault_reason)
            try:
                payload = json.loads(message.data)
            except (TypeError, ValueError) as exc:
                self._time_guard.latch("invalid time-sync status JSON: %s" % exc)
            else:
                self._time_guard.update(payload)
            became_faulted = bool(self._time_guard.fault_reason) and not was_faulted
            if became_faulted:
                self._voxels.clear()
                self._grid.clear()
                self._odom_buffer.clear()
                self._pending_depth.clear()
                self._pending_color.clear()
                self._last_depth_stamp_ns = -1
                self._last_error = self._time_guard.fault_reason
        if became_faulted:
            self.get_logger().error(
                "mapping time-sync fault latched: %s; restart the complete stack"
                % self._time_guard.fault_reason
            )

    def _nearest_odometry(self, stamp_ns):
        if not self._odom_buffer:
            return None, None
        sample_stamp, transform = min(
            self._odom_buffer, key=lambda item: abs(item[0] - stamp_ns)
        )
        delta_sec = abs(sample_stamp - stamp_ns) * 1.0e-9
        return transform, delta_sec

    def _on_depth(self, message):
        self._frames_received += 1
        stamp_ns = _stamp_ns(message)
        with self._lock:
            color_message = self._pending_color.pop(stamp_ns, None)
            if color_message is None:
                self._pending_depth[stamp_ns] = message
                while len(self._pending_depth) > self._pair_buffer_size:
                    self._pending_depth.popitem(last=False)
                    self._frames_dropped += 1
                return
        self._process_rgbd(message, color_message)

    def _on_color(self, message):
        stamp_ns = _stamp_ns(message)
        with self._lock:
            depth_message = self._pending_depth.pop(stamp_ns, None)
            if depth_message is None:
                self._pending_color[stamp_ns] = message
                while len(self._pending_color) > self._pair_buffer_size:
                    self._pending_color.popitem(last=False)
                return
        self._process_rgbd(depth_message, message)

    def _process_rgbd(self, message, color_message):
        if not self._extrinsics_confirmed:
            self._frames_dropped += 1
            return
        now_mono = time.monotonic()
        if self._processing_rate_hz > 0.0:
            period = 1.0 / self._processing_rate_hz
            if now_mono - self._last_process_monotonic < period:
                return

        stamp_ns = _stamp_ns(message)
        now_ns = self.get_clock().now().nanoseconds
        if stamp_ns <= 0 or stamp_ns <= self._last_depth_stamp_ns:
            self._frames_dropped += 1
            self._warn_throttled("depth_order", "depth timestamps are not increasing")
            return
        age_sec = abs(now_ns - stamp_ns) * 1.0e-9
        if age_sec > self._max_depth_age:
            self._frames_dropped += 1
            self._warn_throttled("depth_age", "depth image timestamp is stale")
            return

        with self._lock:
            if not self._time_guard.ready:
                self._frames_dropped += 1
                self._warn_throttled(
                    "time_sync", "waiting for a locked sensor time boundary"
                )
                return
            camera_info = dict(self._camera_info) if self._camera_info else None
            odom_from_base, odom_delta = self._nearest_odometry(stamp_ns)

        if camera_info is None:
            self._frames_dropped += 1
            self._warn_throttled("camera_info", "waiting for aligned CameraInfo")
            return
        if odom_from_base is None:
            self._frames_dropped += 1
            self._warn_throttled("odometry", "waiting for odometry")
            return
        if odom_delta > self._max_odom_delta:
            self._frames_dropped += 1
            self._warn_throttled(
                "odom_delta", "depth and nearest odometry timestamps are too far apart"
            )
            return
        if message.header.frame_id != self._expected_depth_frame:
            self._frames_dropped += 1
            self._warn_throttled(
                "depth_frame",
                "ignoring depth frame %s (expected %s)"
                % (message.header.frame_id, self._expected_depth_frame),
            )
            return
        if camera_info["frame"] != message.header.frame_id:
            self._frames_dropped += 1
            self._warn_throttled(
                "info_frame", "depth Image and CameraInfo frame IDs differ"
            )
            return
        if (
            int(message.width) != camera_info["width"]
            or int(message.height) != camera_info["height"]
        ):
            self._frames_dropped += 1
            self._warn_throttled(
                "image_size", "depth Image and CameraInfo dimensions differ"
            )
            return
        if (
            color_message.header.frame_id != message.header.frame_id
            or _stamp_ns(color_message) != stamp_ns
        ):
            self._frames_dropped += 1
            self._warn_throttled(
                "rgb_pair_stamp",
                "aligned RGB and depth frame IDs or timestamps differ",
            )
            return
        if (
            int(color_message.width) != int(message.width)
            or int(color_message.height) != int(message.height)
        ):
            self._frames_dropped += 1
            self._warn_throttled(
                "rgb_pair_size", "aligned RGB and depth image sizes differ"
            )
            return

        try:
            depth = decode_depth_image(message)
            color_bgr = decode_color_image(color_message)
            points_camera, colors_rgb = depth_to_camera_points_rgb(
                depth,
                color_bgr,
                camera_info["fx"],
                camera_info["fy"],
                camera_info["cx"],
                camera_info["cy"],
                pixel_stride=self._pixel_stride,
                min_depth=self._min_depth,
                max_depth=self._max_depth,
                max_points=self._max_points,
            )
        except ValueError as exc:
            self._frames_dropped += 1
            self._last_error = str(exc)
            self._warn_throttled("rgbd_decode", "invalid RGB-D pair: %s" % exc)
            return
        if points_camera.shape[0] == 0:
            self._frames_dropped += 1
            return

        odom_from_camera = odom_from_base @ self._base_from_camera
        points_world = transform_points(points_camera, odom_from_camera)
        robot_position = odom_from_base[:3, 3]
        accepted_mask = self._voxels.filter_mask(
            points_world, robot_position
        )
        accepted = points_world[accepted_mask]
        accepted_colors = colors_rgb[accepted_mask]
        if accepted.shape[0] == 0:
            self._frames_dropped += 1
            return

        with self._lock:
            self._voxels.fuse_filtered(
                accepted, robot_position, stamp_ns, accepted_colors
            )
            self._grid.update(accepted, robot_position, stamp_ns)
            self._last_depth_stamp_ns = stamp_ns
            self._last_sensor_stamp_ns = stamp_ns
            self._frames_fused += 1
            self._last_error = ""
        self._last_process_monotonic = now_mono

    def _point_cloud_message(self, points, colors_rgb, stamp_ns):
        message = PointCloud2()
        message.header.frame_id = self._world_frame
        message.header.stamp.sec = int(stamp_ns // 1_000_000_000)
        message.header.stamp.nanosec = int(stamp_ns % 1_000_000_000)
        message.height = 1
        message.width = int(points.shape[0])
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(
                name="rgb",
                offset=12,
                datatype=PointField.FLOAT32,
                count=1,
            ),
        ]
        message.is_bigendian = False
        message.point_step = 16
        message.row_step = 16 * message.width
        message.is_dense = True
        payload = array("B")
        payload.frombytes(xyzrgb_to_float32_bytes(points, colors_rgb))
        message.data = payload
        return message

    def _occupancy_message(self, stamp_ns):
        data, origin_x, origin_y, _cropped = self._grid.to_dense(
            self._max_dense_grid_cells
        )
        message = OccupancyGrid()
        message.header.frame_id = self._world_frame
        message.header.stamp.sec = int(stamp_ns // 1_000_000_000)
        message.header.stamp.nanosec = int(stamp_ns % 1_000_000_000)
        message.info.map_load_time = message.header.stamp
        message.info.resolution = float(self._grid_resolution)
        message.info.width = int(data.shape[1])
        message.info.height = int(data.shape[0])
        message.info.origin.position.x = float(origin_x)
        message.info.origin.position.y = float(origin_y)
        message.info.origin.orientation.w = 1.0
        payload = array("b")
        payload.frombytes(np.ascontiguousarray(data, dtype=np.int8).tobytes())
        message.data = payload
        return message

    def _publish_maps(self):
        with self._lock:
            stamp_ns = self._last_sensor_stamp_ns or self.get_clock().now().nanoseconds
            points, colors_rgb = self._voxels.points_with_colors()
            occupancy = self._occupancy_message(stamp_ns)
        if points.shape[0] > 0:
            self._cloud_pub.publish(
                self._point_cloud_message(points, colors_rgb, stamp_ns)
            )
        if occupancy is not None:
            self._grid_pub.publish(occupancy)

    def _state(self):
        if not self._extrinsics_confirmed:
            return "calibration_required"
        if self._time_guard.fault_reason:
            return "time_sync_fault"
        if not self._time_guard.ready:
            return "waiting_for_time_sync"
        if self._camera_info is None:
            return "waiting_for_camera_info"
        if not self._odom_buffer:
            return "waiting_for_odometry"
        if self._frames_fused == 0:
            return "waiting_for_depth"
        return "mapping"

    def _publish_status(self):
        with self._lock:
            status = {
                "node": "go2_mapping_depthcam",
                "backend": "depth_camera",
                "state": self._state(),
                "extrinsics_confirmed": self._extrinsics_confirmed,
                "time_sync": {
                    "required": self._time_guard.required,
                    "state": self._time_guard.state,
                    "instance_id": self._time_guard.instance_id,
                    "epoch": self._time_guard.epoch,
                    "fault_reason": self._time_guard.fault_reason,
                },
                "frames_received": self._frames_received,
                "frames_fused": self._frames_fused,
                "frames_dropped": self._frames_dropped,
                "voxel_count": len(self._voxels),
                "colorized_voxel_count": self._voxels.colorized_count(),
                "rgb_fusion": True,
                "pending_depth_frames": len(self._pending_depth),
                "pending_color_frames": len(self._pending_color),
                "grid_cell_count": len(self._grid),
                "last_error": self._last_error,
                "output_directory": self._output_directory,
            }
        message = String()
        message.data = json.dumps(status, sort_keys=True)
        self._status_pub.publish(message)

    def _snapshot_metadata(self):
        return {
            "producer": "go2_mapping_depthcam",
            "source": "Intel RealSense aligned RGB-D",
            "has_rgb": True,
            "world_frame": self._world_frame,
            "base_frame": self._base_frame,
            "depth_frame": self._expected_depth_frame,
            "voxel_size": self._voxel_size,
            "grid_resolution": self._grid_resolution,
            "base_from_camera_optical": self._base_from_camera.reshape(-1).tolist(),
            "extrinsics_confirmed": self._extrinsics_confirmed,
            "saved_at_unix": time.time(),
        }

    def _save_snapshot(self):
        with self._lock:
            if not self._time_guard.ready:
                raise RuntimeError(
                    "map is not saveable without a locked, fault-free "
                    "sensor-time boundary"
                )
            voxel_state = self._voxels.state()
            grid_state = self._grid.state()
            grid, origin_x, origin_y, cropped = self._grid.to_dense(
                self._max_dense_grid_cells
            )
            metadata = self._snapshot_metadata()
            metadata["dense_grid_cropped"] = bool(cropped)
        if voxel_state["voxel_centroids"].shape[0] == 0:
            raise RuntimeError("the fused 3D voxel map is empty")
        return save_snapshot(
            output_dir=self._output_directory,
            voxel_state=voxel_state,
            grid_state=grid_state,
            occupancy=grid,
            origin_x=origin_x,
            origin_y=origin_y,
            resolution=self._grid.resolution,
            metadata=metadata,
            free_threshold=self._free_threshold,
            occupied_threshold=self._occupied_threshold,
        )

    def _load_snapshot(self, path):
        try:
            arrays, metadata = load_snapshot(path)
        except FileNotFoundError:
            self.get_logger().info("no existing depth-camera map snapshot found")
            return
        voxel_size = float(metadata.get("voxel_size", self._voxel_size))
        grid_resolution = float(
            metadata.get("grid_resolution", self._grid_resolution)
        )
        if not math.isclose(voxel_size, self._voxel_size, rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError("saved map voxel size differs from this configuration")
        if not math.isclose(
            grid_resolution, self._grid_resolution, rel_tol=0.0, abs_tol=1e-9
        ):
            raise RuntimeError("saved map grid resolution differs from configuration")
        self._voxels.restore(arrays)
        self._grid.restore(arrays)
        self.get_logger().info(
            "loaded depth-camera snapshot with %d voxels" % len(self._voxels)
        )

    def _on_save(self, request, response):
        del request
        try:
            paths = self._save_snapshot()
            response.success = True
            response.message = "saved atomic snapshot to %s" % paths
        except Exception as exc:  # service boundary
            response.success = False
            response.message = str(exc)
        return response

    def _on_reset(self, request, response):
        del request
        with self._lock:
            self._voxels.clear()
            self._grid.clear()
            self._pending_depth.clear()
            self._pending_color.clear()
            self._last_sensor_stamp_ns = 0
            self._frames_fused = 0
        response.success = True
        response.message = "depth-camera map cleared"
        return response

    def _autosave(self):
        with self._lock:
            has_map = len(self._voxels) > 0
            ready = self._time_guard.ready
        if not has_map or not ready:
            return
        try:
            self._save_snapshot()
        except Exception as exc:  # timer boundary
            self.get_logger().error("automatic map save failed: %s" % exc)

    def destroy_node(self):
        with self._lock:
            has_map = len(self._voxels) > 0
        if has_map:
            try:
                self._save_snapshot()
                self.get_logger().info("saved the final depth-camera map snapshot")
            except Exception as exc:  # shutdown boundary
                self.get_logger().error("final map save failed: %s" % exc)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DepthMappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
