#!/usr/bin/env python3
"""Fuse atomic D435i XYZRGB clouds into Go2 voxel and occupancy maps."""

import json
import math
import os
import threading
import time
from array import array
from collections import deque

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
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String
from std_srvs.srv import Trigger

from go2_mapping.grid_map import LogOddsGrid
from go2_mapping.pointcloud import (
    PointCloudFormatError,
    read_xyzrgb,
    xyzrgb_to_float32_bytes,
)
from go2_mapping.state_io import load_snapshot, save_snapshot
from go2_mapping.time_sync_guard import TimeSyncStatusGuard
from go2_mapping.voxel_map import VoxelAccumulator

from .geometry import pose_matrix, rigid_transform, transform_points
from .registration import (
    planar_rotation_degrees,
    register_planar_scan,
    scale_planar_transform,
    transform_points_fast,
    voxel_downsample,
)


def _stamp_ns(message):
    return int(message.header.stamp.sec) * 1_000_000_000 + int(
        message.header.stamp.nanosec
    )


class DepthMappingNode(Node):
    """Build the normal Go2 map products from a registered depth stream."""

    def __init__(self):
        super().__init__("go2_mapping_depthcam")
        # Sensor/mailbox state is independent of all O(map size) work.
        self._lock = threading.RLock()
        self._map_lock = threading.RLock()
        self._warn_times = {}

        self._camera_cloud_topic = self._param(
            "camera_cloud_topic", "/go2/depth_camera/points"
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
        self._expected_camera_frame = self._param(
            "expected_camera_frame", "d435i_color_optical_frame"
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
        self._max_camera_points = max(
            1, int(self._param("max_camera_points", 60000))
        )
        self._max_camera_age = float(
            self._param("max_camera_age_sec", 0.50)
        )
        self._max_odom_delta = float(self._param("max_odom_delta_sec", 0.15))
        odom_buffer_size = max(2, int(self._param("odom_buffer_size", 100)))
        self._odom_buffer = deque(maxlen=odom_buffer_size)
        # The bridge publishes geometry and RGB in one message, so no
        # downstream approximate/exact image synchronization can deadlock.
        # The callback only replaces this latest-only mailbox; all heavy map
        # work stays on the fusion worker.
        self._latest_camera_cloud = None
        self._processing_wake = threading.Event()
        self._processing_stop = threading.Event()
        self._processing_lock = threading.Lock()
        self._processing_active = False

        # Full-map serialization grows with map size. Timers only set these
        # coalescing flags; a dedicated output worker performs publication and
        # autosave so sensor callbacks never execute O(map size) work.
        self._output_request_lock = threading.Lock()
        self._output_wake = threading.Event()
        self._output_stop = threading.Event()
        self._map_publish_requested = False
        self._autosave_requested = False
        self._output_active = "idle"
        self._map_publish_count = 0
        self._map_publish_coalesced = 0
        self._last_map_publish_duration_sec = 0.0
        self._autosave_count = 0
        self._autosave_coalesced = 0
        self._last_autosave_duration_sec = 0.0
        self._map_generation = 0

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

        # Go2 odometry remains the initial pose estimate. Bounded planar ICP
        # aligns each moving RGB-D scan to a short recent submap before the
        # permanent voxel map is updated, reducing duplicated object edges.
        self._registration_enabled = bool(
            self._param("registration_enabled", True)
        )
        self._registration_rate_hz = float(
            self._param("registration_rate_hz", 2.0)
        )
        self._registration_voxel_size = float(
            self._param("registration_voxel_size", 0.06)
        )
        self._registration_submap_frames = max(
            2, int(self._param("registration_submap_frames", 5))
        )
        self._registration_max_source_points = max(
            100, int(self._param("registration_max_source_points", 5000))
        )
        self._registration_max_target_points = max(
            100, int(self._param("registration_max_target_points", 12000))
        )
        self._registration_max_correspondence = float(
            self._param("registration_max_correspondence", 0.14)
        )
        self._registration_iterations = max(
            1, int(self._param("registration_iterations", 8))
        )
        self._registration_trim_fraction = float(
            self._param("registration_trim_fraction", 0.75)
        )
        self._registration_min_correspondences = max(
            20, int(self._param("registration_min_correspondences", 250))
        )
        self._registration_min_overlap = float(
            self._param("registration_min_overlap", 0.35)
        )
        self._registration_max_rmse = float(
            self._param("registration_max_rmse", 0.07)
        )
        self._registration_max_translation = float(
            self._param("registration_max_translation", 0.10)
        )
        self._registration_max_rotation_deg = float(
            self._param("registration_max_rotation_deg", 4.0)
        )
        self._registration_gain = float(
            self._param("registration_gain", 0.60)
        )
        self._registration_min_motion_translation = float(
            self._param("registration_min_motion_translation", 0.015)
        )
        self._registration_min_motion_rotation_deg = float(
            self._param("registration_min_motion_rotation_deg", 0.50)
        )
        self._registration_reseed_after_rejections = max(
            1,
            int(self._param("registration_reseed_after_rejections", 3)),
        )
        if self._registration_enabled:
            if self._registration_rate_hz <= 0.0:
                raise ValueError("registration_rate_hz must be positive")
            if self._registration_voxel_size <= 0.0:
                raise ValueError("registration_voxel_size must be positive")
            if self._registration_max_correspondence <= 0.0:
                raise ValueError(
                    "registration_max_correspondence must be positive"
                )
            if not 0.25 <= self._registration_trim_fraction <= 1.0:
                raise ValueError(
                    "registration_trim_fraction must be in [0.25, 1]"
                )
            if not 0.0 < self._registration_min_overlap <= 1.0:
                raise ValueError("registration_min_overlap must be in (0, 1]")
            if self._registration_max_rmse <= 0.0:
                raise ValueError("registration_max_rmse must be positive")
            if self._registration_max_translation <= 0.0:
                raise ValueError(
                    "registration_max_translation must be positive"
                )
            if self._registration_max_rotation_deg <= 0.0:
                raise ValueError(
                    "registration_max_rotation_deg must be positive"
                )
            if not 0.0 < self._registration_gain <= 1.0:
                raise ValueError("registration_gain must be in (0, 1]")
            if (
                self._registration_max_source_points
                < self._registration_min_correspondences
                or self._registration_max_target_points
                < self._registration_min_correspondences
            ):
                raise ValueError(
                    "registration point caps must not be below the minimum "
                    "correspondence count"
                )
            if (
                self._registration_min_motion_translation < 0.0
                or self._registration_min_motion_rotation_deg < 0.0
            ):
                raise ValueError(
                    "registration motion thresholds must not be negative"
                )

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
        self._last_camera_stamp_ns = -1
        self._last_process_monotonic = 0.0
        self._last_sensor_stamp_ns = 0
        self._frames_received = 0
        self._frames_fused = 0
        self._frames_dropped = 0
        self._frames_superseded = 0
        self._last_input_monotonic = 0.0
        self._last_fused_monotonic = 0.0
        self._last_error = ""
        self._last_registration_correction = np.eye(4, dtype=np.float64)
        self._registration_submap = deque(
            maxlen=self._registration_submap_frames
        )
        self._last_registration_monotonic = 0.0
        self._last_registration_raw_pose = None
        self._registration_attempts = 0
        self._registration_accepted = 0
        self._registration_rejected = 0
        self._registration_consecutive_rejections = 0
        self._registration_submap_reseeds = 0
        self._registration_state = (
            "initializing" if self._registration_enabled else "disabled"
        )
        self._registration_last_rmse = None
        self._registration_last_overlap = None
        self._registration_last_translation = None
        self._registration_last_rotation_deg = None
        self._registration_last_reason = ""

        if self._load_state_path:
            self._load_snapshot(self._load_state_path)

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
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

        self._camera_cloud_sub = self.create_subscription(
            PointCloud2,
            self._camera_cloud_topic,
            self._on_camera_cloud,
            sensor_qos,
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
        self._publish_timer = self.create_timer(
            publish_period, self._request_map_publish
        )
        self._status_timer = self.create_timer(1.0, self._publish_status)
        self._autosave_timer = None
        if self._auto_save_period > 0.0:
            self._autosave_timer = self.create_timer(
                self._auto_save_period, self._request_autosave
            )

        self.get_logger().info(
            "D435i atomic-cloud mapper configured: cloud=%s odom=%s "
            "voxel=%.3f m planar_registration=%s output=%s"
            % (
                self._camera_cloud_topic,
                self._odom_topic,
                self._voxel_size,
                "on" if self._registration_enabled else "off",
                self._output_directory,
            )
        )
        if not self._extrinsics_confirmed:
            self.get_logger().error(
                "CALIBRATION REQUIRED: depth fusion is disabled. Measure "
                "base_from_camera_optical and set extrinsics_confirmed:=true."
            )
        self._processing_thread = threading.Thread(
            target=self._processing_loop,
            name="go2-atomic-cloud-fusion",
            daemon=True,
        )
        self._output_thread = threading.Thread(
            target=self._output_loop,
            name="go2-map-output",
            daemon=True,
        )
        self._processing_thread.start()
        self._output_thread.start()

    def _param(self, name, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _warn_throttled(self, key, text, period=5.0):
        now = time.monotonic()
        previous = self._warn_times.get(key, 0.0)
        if now - previous >= period:
            self.get_logger().warning(text)
            self._warn_times[key] = now

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
        fault_reason = ""
        with self._lock:
            was_faulted = bool(self._time_guard.fault_reason)
            try:
                payload = json.loads(message.data)
            except (TypeError, ValueError) as exc:
                self._time_guard.latch("invalid time-sync status JSON: %s" % exc)
            else:
                self._time_guard.update(payload)
            became_faulted = bool(self._time_guard.fault_reason) and not was_faulted
            fault_reason = self._time_guard.fault_reason
        if became_faulted:
            # Wait for any in-flight fusion to finish, then atomically clear
            # both the permanent map and every queued pre-fault sensor pair.
            with self._processing_lock:
                # Wait for map output without holding the sensor lock.
                with self._map_lock:
                    self._map_generation += 1
                    self._voxels.clear()
                    self._grid.clear()
                    self._registration_submap.clear()
                    self._last_registration_correction = np.eye(
                        4, dtype=np.float64
                    )
                    self._last_registration_raw_pose = None
                    self._registration_consecutive_rejections = 0
                    self._registration_state = (
                        "faulted" if self._registration_enabled else "disabled"
                    )
                with self._lock:
                    self._odom_buffer.clear()
                    self._latest_camera_cloud = None
                    self._processing_wake.clear()
                    self._last_camera_stamp_ns = -1
                    self._last_fused_monotonic = 0.0
                    self._last_error = fault_reason
            self.get_logger().error(
                "mapping time-sync fault latched: %s; restart the complete stack"
                % fault_reason
            )

    def _nearest_odometry(self, stamp_ns):
        if not self._odom_buffer:
            return None, None
        sample_stamp, transform = min(
            self._odom_buffer, key=lambda item: abs(item[0] - stamp_ns)
        )
        delta_sec = abs(sample_stamp - stamp_ns) * 1.0e-9
        return transform, delta_sec

    @staticmethod
    def _planar_pose_delta(previous, current):
        relative = np.linalg.inv(previous) @ current
        translation = float(np.linalg.norm(relative[:2, 3]))
        rotation = planar_rotation_degrees(relative)
        return translation, rotation

    def _registration_target(self):
        with self._map_lock:
            frames = [frame.copy() for frame in self._registration_submap]
        if not frames:
            return np.empty((0, 3), dtype=np.float64)
        return voxel_downsample(
            np.vstack(frames),
            self._registration_voxel_size,
            self._registration_max_target_points,
        )

    def _maybe_register(self, points_world, raw_odom_from_base):
        """Correct only a predicted scan against the recent RGB-D submap."""

        with self._map_lock:
            previous_raw_pose = (
                self._last_registration_raw_pose.copy()
                if self._last_registration_raw_pose is not None
                else None
            )
        # Registration is deliberately map-only. It never changes /go2/odom
        # or the odom -> base_link TF consumed by Nav2 and the motion gate.
        if not self._registration_enabled:
            return points_world, False

        target = self._registration_target()
        if target.shape[0] < self._registration_min_correspondences:
            with self._map_lock:
                self._registration_state = "initializing"
                self._last_registration_raw_pose = raw_odom_from_base.copy()
                self._registration_consecutive_rejections = 0
            return points_world, True

        now_mono = time.monotonic()
        period = 1.0 / self._registration_rate_hz
        moved = 0.0
        turned = 0.0
        if previous_raw_pose is not None:
            moved, turned = self._planar_pose_delta(
                previous_raw_pose, raw_odom_from_base
            )
            if (
                moved < self._registration_min_motion_translation
                and turned < self._registration_min_motion_rotation_deg
            ):
                with self._map_lock:
                    self._registration_state = "stationary"
                return points_world, True
        with self._map_lock:
            registration_due = (
                now_mono - self._last_registration_monotonic >= period
            )
            last_delta = self._last_registration_correction.copy()
        if not registration_due:
            # Preserve the last confirmed local target while the robot moves;
            # the next rate-limited attempt will align against that target.
            # Reuse the last accepted bounded correction so intermediate
            # moving frames do not reintroduce raw-odometry edge smear.
            return transform_points_fast(points_world, last_delta), False

        source = voxel_downsample(
            points_world,
            self._registration_voxel_size,
            self._registration_max_source_points,
        )
        with self._map_lock:
            self._last_registration_monotonic = now_mono
            self._registration_attempts += 1
        try:
            delta, rmse, overlap, _ = register_planar_scan(
                source,
                target,
                self._registration_max_correspondence,
                self._registration_iterations,
                self._registration_trim_fraction,
                self._registration_min_correspondences,
            )
            translation = float(np.linalg.norm(delta[:2, 3]))
            rotation = planar_rotation_degrees(delta)
            rejection = ""
            if overlap < self._registration_min_overlap:
                rejection = "overlap {:.1%} is below {:.1%}".format(
                    overlap, self._registration_min_overlap
                )
            elif rmse > self._registration_max_rmse:
                rejection = "RMSE {:.3f} m exceeds {:.3f} m".format(
                    rmse, self._registration_max_rmse
                )
            elif translation > self._registration_max_translation:
                rejection = "translation {:.3f} m exceeds {:.3f} m".format(
                    translation, self._registration_max_translation
                )
            elif rotation > self._registration_max_rotation_deg:
                rejection = "rotation {:.2f} deg exceeds {:.2f} deg".format(
                    rotation, self._registration_max_rotation_deg
                )
        except Exception as exc:  # OpenCV/registration safety boundary
            delta = None
            rmse = None
            overlap = None
            translation = None
            rotation = None
            rejection = str(exc)

        with self._map_lock:
            self._registration_last_rmse = rmse
            self._registration_last_overlap = overlap
            self._registration_last_translation = translation
            self._registration_last_rotation_deg = rotation
        if rejection:
            reseeded = False
            with self._map_lock:
                self._registration_rejected += 1
                self._registration_consecutive_rejections += 1
                self._registration_state = "rejected"
                self._registration_last_reason = rejection
                self._last_registration_correction = np.eye(
                    4, dtype=np.float64
                )
                if (
                    self._registration_consecutive_rejections
                    >= self._registration_reseed_after_rejections
                    and source.shape[0]
                    >= self._registration_min_correspondences
                ):
                    rejected_count = self._registration_consecutive_rejections
                    self._registration_submap.clear()
                    self._registration_submap.append(source.copy())
                    self._last_registration_raw_pose = (
                        raw_odom_from_base.copy()
                    )
                    self._registration_consecutive_rejections = 0
                    self._registration_submap_reseeds += 1
                    self._registration_state = "reinitializing"
                    self._registration_last_reason = (
                        "local submap reseeded after %d consecutive "
                        "rejections; last rejection: %s"
                        % (rejected_count, rejection)
                    )
                    reseed_count = self._registration_submap_reseeds
                    reseeded = True
            if reseeded:
                self.get_logger().warning(
                    "RGB-D registration reseeded local submap after %d "
                    "consecutive rejections (reseed #%d): %s"
                    % (rejected_count, reseed_count, rejection)
                )
            else:
                self._warn_throttled(
                    "rgbd_registration",
                    "RGB-D registration rejected: %s; using guarded Go2 "
                    "odometry" % rejection,
                )
            # Do not put this unconfirmed moving scan into the registration
            # submap unless the explicit recovery above has just replaced the
            # stale target. Geometry follows the guarded Go2 odometry pose.
            return points_world, False

        applied_delta = scale_planar_transform(
            delta, self._registration_gain
        )
        corrected_points = transform_points_fast(points_world, applied_delta)
        with self._map_lock:
            self._last_registration_correction = applied_delta.copy()
            self._last_registration_raw_pose = raw_odom_from_base.copy()
            self._registration_accepted += 1
            self._registration_consecutive_rejections = 0
            self._registration_state = "tracking"
            self._registration_last_reason = ""
            accepted_count = self._registration_accepted
        if accepted_count == 1 or accepted_count % 20 == 0:
            self.get_logger().info(
                "RGB-D registration tracking: accepted=%d rmse=%.3f m "
                "overlap=%.1f%% correction=%.3f m/%.2f deg"
                % (
                    accepted_count,
                    rmse,
                    overlap * 100.0,
                    translation,
                    rotation,
                )
            )
        return corrected_points, True

    def _update_registration_submap(self, points_world):
        if not self._registration_enabled:
            return
        frame = voxel_downsample(
            points_world,
            self._registration_voxel_size,
            self._registration_max_source_points,
        )
        if frame.shape[0] < self._registration_min_correspondences:
            return
        with self._map_lock:
            self._registration_submap.append(frame)

    def _record_drop(self, error=""):
        with self._lock:
            self._frames_dropped += 1
            if error:
                self._last_error = str(error)

    def _queue_latest_camera_cloud(self, message):
        # Replace unprocessed input atomically; stale work never forms a queue.
        with self._lock:
            self._frames_received += 1
            self._last_input_monotonic = time.monotonic()
            if self._latest_camera_cloud is not None:
                self._frames_dropped += 1
                self._frames_superseded += 1
            self._latest_camera_cloud = message
            self._processing_wake.set()

    def _take_latest_camera_cloud(self):
        with self._lock:
            message = self._latest_camera_cloud
            self._latest_camera_cloud = None
            self._processing_wake.clear()
            return message

    def _on_camera_cloud(self, message):
        self._queue_latest_camera_cloud(message)

    def _processing_loop(self):
        # Fuse only the freshest atomic cloud without blocking ROS callbacks.
        period = (
            1.0 / self._processing_rate_hz
            if self._processing_rate_hz > 0.0
            else 0.0
        )
        while not self._processing_stop.is_set():
            if not self._processing_wake.wait(timeout=0.25):
                continue
            if self._processing_stop.is_set():
                break

            if period > 0.0 and self._last_process_monotonic > 0.0:
                delay = (
                    self._last_process_monotonic + period - time.monotonic()
                )
                if delay > 0.0:
                    self._processing_stop.wait(delay)
                    continue

            message = self._take_latest_camera_cloud()
            if message is None:
                continue
            with self._lock:
                self._processing_active = True
            try:
                # Serialize fusion with reset and time-fault map clearing.
                with self._processing_lock:
                    if not self._processing_stop.is_set():
                        self._process_camera_cloud(message)
            except Exception as exc:  # worker safety boundary
                self._record_drop(exc)
                self._warn_throttled(
                    "camera_cloud_worker",
                    "atomic camera-cloud fusion worker failed: %s" % exc,
                )
            finally:
                with self._lock:
                    self._processing_active = False

    def _process_camera_cloud(self, message):
        if not self._extrinsics_confirmed:
            self._record_drop()
            return
        now_mono = time.monotonic()
        stamp_ns = _stamp_ns(message)
        now_ns = self.get_clock().now().nanoseconds
        with self._lock:
            last_stamp_ns = self._last_camera_stamp_ns
        if stamp_ns <= 0 or stamp_ns <= last_stamp_ns:
            self._record_drop()
            self._warn_throttled(
                "camera_cloud_order",
                "camera-cloud timestamps are not increasing",
            )
            return
        age_sec = abs(now_ns - stamp_ns) * 1.0e-9
        if age_sec > self._max_camera_age:
            self._record_drop()
            self._warn_throttled(
                "camera_cloud_age", "camera-cloud timestamp is stale"
            )
            return

        with self._lock:
            if not self._time_guard.ready:
                self._frames_dropped += 1
                self._warn_throttled(
                    "time_sync", "waiting for a locked sensor time boundary"
                )
                return
            odom_from_base, odom_delta = self._nearest_odometry(stamp_ns)

        if odom_from_base is None:
            self._record_drop()
            self._warn_throttled("odometry", "waiting for odometry")
            return
        if odom_delta > self._max_odom_delta:
            self._record_drop()
            self._warn_throttled(
                "odom_delta",
                "camera cloud and nearest odometry are too far apart",
            )
            return
        if message.header.frame_id != self._expected_camera_frame:
            self._record_drop()
            self._warn_throttled(
                "camera_cloud_frame",
                "ignoring camera cloud in frame %s (expected %s)"
                % (message.header.frame_id, self._expected_camera_frame),
            )
            return

        try:
            points_camera, colors_rgb = read_xyzrgb(
                message, max_points=self._max_camera_points
            )
        except (PointCloudFormatError, ValueError) as exc:
            self._record_drop(exc)
            self._warn_throttled(
                "camera_cloud_decode", "invalid atomic camera cloud: %s" % exc
            )
            return
        finite = np.isfinite(points_camera).all(axis=1)
        points_camera = points_camera[finite]
        colors_rgb = colors_rgb[finite]
        if points_camera.shape[0] == 0:
            self._record_drop()
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
            self._record_drop()
            return

        accepted, update_submap = self._maybe_register(
            accepted, odom_from_base
        )
        corrected_mask = self._voxels.filter_mask(accepted, robot_position)
        accepted = accepted[corrected_mask]
        accepted_colors = accepted_colors[corrected_mask]
        if accepted.shape[0] == 0:
            self._record_drop()
            return

        with self._map_lock:
            self._voxels.fuse_filtered(
                accepted, robot_position, stamp_ns, accepted_colors
            )
            self._grid.update(accepted, robot_position, stamp_ns)
        with self._lock:
            self._last_camera_stamp_ns = stamp_ns
            self._last_sensor_stamp_ns = stamp_ns
            self._frames_fused += 1
            self._last_fused_monotonic = time.monotonic()
            self._last_error = ""
        if update_submap:
            self._update_registration_submap(accepted)
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

    def _request_map_publish(self):
        with self._output_request_lock:
            if self._map_publish_requested:
                self._map_publish_coalesced += 1
            self._map_publish_requested = True
            self._output_wake.set()

    def _request_autosave(self):
        with self._output_request_lock:
            if self._autosave_requested:
                self._autosave_coalesced += 1
            self._autosave_requested = True
            self._output_wake.set()

    def _take_output_requests(self):
        with self._output_request_lock:
            publish = self._map_publish_requested
            autosave = self._autosave_requested
            self._map_publish_requested = False
            self._autosave_requested = False
            self._output_wake.clear()
            return publish, autosave

    def _output_loop(self):
        while not self._output_stop.is_set():
            if not self._output_wake.wait(timeout=0.25):
                continue
            if self._output_stop.is_set():
                break
            publish, autosave = self._take_output_requests()
            if publish:
                started = time.monotonic()
                with self._output_request_lock:
                    self._output_active = "publishing"
                try:
                    self._publish_maps_now()
                except Exception as exc:  # output worker safety boundary
                    with self._lock:
                        self._last_error = "map publication failed: %s" % exc
                    self.get_logger().error(
                        "background map publication failed: %s" % exc
                    )
                finally:
                    duration = time.monotonic() - started
                    with self._output_request_lock:
                        self._map_publish_count += 1
                        self._last_map_publish_duration_sec = duration
                        self._output_active = "idle"
            if autosave:
                started = time.monotonic()
                with self._output_request_lock:
                    self._output_active = "autosaving"
                try:
                    self._autosave_now()
                except Exception as exc:  # output worker safety boundary
                    self.get_logger().error(
                        "automatic map save failed: %s" % exc
                    )
                finally:
                    duration = time.monotonic() - started
                    with self._output_request_lock:
                        self._autosave_count += 1
                        self._last_autosave_duration_sec = duration
                        self._output_active = "idle"

    def _publish_maps_now(self):
        with self._lock:
            stamp_ns = (
                self._last_sensor_stamp_ns
                or self.get_clock().now().nanoseconds
            )
        with self._map_lock:
            generation = self._map_generation
            points, colors_rgb = self._voxels.points_with_colors()
            occupancy = self._occupancy_message(stamp_ns)
        cloud = None
        if points.shape[0] > 0:
            cloud = self._point_cloud_message(
                points, colors_rgb, stamp_ns
            )
        # A reset/time fault invalidates snapshots already being serialized.
        with self._map_lock:
            if generation != self._map_generation:
                return
        if cloud is not None:
            self._cloud_pub.publish(cloud)
        if occupancy is not None:
            self._grid_pub.publish(occupancy)

    def _state(self):
        if not self._extrinsics_confirmed:
            return "calibration_required"
        if self._time_guard.fault_reason:
            return "time_sync_fault"
        if not self._time_guard.ready:
            return "waiting_for_time_sync"
        if not self._odom_buffer:
            return "waiting_for_odometry"
        if self._frames_fused == 0:
            return "waiting_for_camera_cloud"
        return "mapping"

    def _publish_status(self):
        with self._output_request_lock:
            output_status = {
                "worker_alive": self._output_thread.is_alive(),
                "active": self._output_active,
                "publish_requested": self._map_publish_requested,
                "publish_count": self._map_publish_count,
                "publish_coalesced": self._map_publish_coalesced,
                "last_publish_duration_sec": (
                    self._last_map_publish_duration_sec
                ),
                "autosave_requested": self._autosave_requested,
                "autosave_count": self._autosave_count,
                "autosave_coalesced": self._autosave_coalesced,
                "last_autosave_duration_sec": (
                    self._last_autosave_duration_sec
                ),
            }
        now_monotonic = time.monotonic()
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
                "frames_superseded": self._frames_superseded,
                "input_age_sec": (
                    max(0.0, now_monotonic - self._last_input_monotonic)
                    if self._last_input_monotonic
                    else None
                ),
                "fusion_age_sec": (
                    max(0.0, now_monotonic - self._last_fused_monotonic)
                    if self._last_fused_monotonic
                    else None
                ),
                "input_type": "atomic_xyzrgb_pointcloud2",
                "input_topic": self._camera_cloud_topic,
                "latest_cloud_pending": self._latest_camera_cloud is not None,
                "processing_active": self._processing_active,
                "processing_worker_alive": self._processing_thread.is_alive(),
                "output": output_status,
                "rgb_fusion": True,
                "last_error": self._last_error,
                "output_directory": self._output_directory,
            }
        with self._map_lock:
            status.update(
                {
                    "voxel_count": len(self._voxels),
                    "colorized_voxel_count": self._voxels.colorized_count(),
                    "registration": {
                        "enabled": self._registration_enabled,
                        "type": "planar_scan_to_local_submap",
                        "state": self._registration_state,
                        "attempts": self._registration_attempts,
                        "accepted": self._registration_accepted,
                        "rejected": self._registration_rejected,
                        "consecutive_rejections": (
                            self._registration_consecutive_rejections
                        ),
                        "submap_reseeds": self._registration_submap_reseeds,
                        "last_rmse_m": self._registration_last_rmse,
                        "last_overlap": self._registration_last_overlap,
                        "last_translation_m": (
                            self._registration_last_translation
                        ),
                        "last_rotation_deg": (
                            self._registration_last_rotation_deg
                        ),
                        "last_rejection": self._registration_last_reason,
                        "submap_frames": len(self._registration_submap),
                        "last_scan_correction": (
                            self._last_registration_correction.reshape(-1).tolist()
                        ),
                    },
                    "grid_cell_count": len(self._grid),
                }
            )
        message = String()
        message.data = json.dumps(status, sort_keys=True)
        self._status_pub.publish(message)

    def _snapshot_metadata(self):
        return {
            "producer": "go2_mapping_depthcam",
            "source": "Intel RealSense atomic aligned XYZRGB cloud",
            "has_rgb": True,
            "world_frame": self._world_frame,
            "base_frame": self._base_frame,
            "camera_frame": self._expected_camera_frame,
            "camera_cloud_topic": self._camera_cloud_topic,
            "voxel_size": self._voxel_size,
            "grid_resolution": self._grid_resolution,
            "base_from_camera_optical": self._base_from_camera.reshape(-1).tolist(),
            "extrinsics_confirmed": self._extrinsics_confirmed,
            "registration_enabled": self._registration_enabled,
            "registration_type": "planar_scan_to_local_submap",
            "registration_attempts": self._registration_attempts,
            "registration_accepted": self._registration_accepted,
            "registration_rejected": self._registration_rejected,
            "registration_submap_reseeds": (
                self._registration_submap_reseeds
            ),
            "last_scan_correction": (
                self._last_registration_correction.reshape(-1).tolist()
            ),
            "saved_at_unix": time.time(),
        }

    def _save_snapshot(self):
        with self._lock:
            if not self._time_guard.ready:
                raise RuntimeError(
                    "map is not saveable without a locked, fault-free "
                    "sensor-time boundary"
                )
        with self._map_lock:
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
        with self._processing_lock:
            # Map output may be busy, but sensor intake stays lock-independent.
            with self._map_lock:
                self._map_generation += 1
                self._voxels.clear()
                self._grid.clear()
                self._registration_submap.clear()
                self._last_registration_correction = np.eye(
                    4, dtype=np.float64
                )
                self._last_registration_monotonic = 0.0
                self._last_registration_raw_pose = None
                self._registration_attempts = 0
                self._registration_accepted = 0
                self._registration_rejected = 0
                self._registration_consecutive_rejections = 0
                self._registration_submap_reseeds = 0
                self._registration_state = (
                    "initializing" if self._registration_enabled else "disabled"
                )
                self._registration_last_rmse = None
                self._registration_last_overlap = None
                self._registration_last_translation = None
                self._registration_last_rotation_deg = None
                self._registration_last_reason = ""
            with self._lock:
                self._latest_camera_cloud = None
                self._processing_wake.clear()
                self._last_sensor_stamp_ns = 0
                self._last_camera_stamp_ns = -1
                self._last_fused_monotonic = 0.0
                self._frames_fused = 0
        response.success = True
        response.message = "depth-camera map cleared"
        return response

    def _autosave_now(self):
        with self._lock:
            ready = self._time_guard.ready
        with self._map_lock:
            has_map = len(self._voxels) > 0
        if has_map and ready:
            self._save_snapshot()

    def destroy_node(self):
        self._processing_stop.set()
        self._processing_wake.set()
        if self._processing_thread.is_alive():
            self._processing_thread.join(timeout=10.0)
        if self._processing_thread.is_alive():
            self.get_logger().error(
                "atomic-cloud fusion worker did not stop before shutdown"
            )
        self._output_stop.set()
        self._output_wake.set()
        if self._output_thread.is_alive():
            self._output_thread.join(timeout=20.0)
        if self._output_thread.is_alive():
            self.get_logger().error(
                "map output worker did not stop before node shutdown"
            )
        with self._map_lock:
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
