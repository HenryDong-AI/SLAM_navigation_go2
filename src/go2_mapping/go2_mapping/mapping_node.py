"""ROS 2 node for bounded 3D mapping and 2D occupancy projection."""

from array import array
import json
import math
import threading
import time
from typing import Dict, Optional, Tuple

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .grid_map import LogOddsGrid
from .pointcloud import (
    PointCloudFormatError,
    read_xyz,
    xyzrgb_to_float32_bytes,
)
from .state_io import load_snapshot, save_snapshot
from .time_sync_guard import TimeSyncStatusGuard
from .voxel_map import VoxelAccumulator


def _normal_frame(frame: str) -> str:
    return str(frame).strip().lstrip("/")


def _stamp_ns(header) -> int:
    return int(header.stamp.sec) * 1000000000 + int(header.stamp.nanosec)


class MappingNode(Node):
    """Accumulate an already world-registered PointCloud2 stream."""

    def __init__(self) -> None:
        super().__init__("go2_mapping")
        self._lock = threading.RLock()
        self._started_monotonic = time.monotonic()
        self._last_warning_monotonic = 0.0
        self._last_warning = ""

        self.cloud_topic = str(
            self._parameter("cloud_topic", "/go2/lidar/cloud_deskewed")
        )
        self.odom_topic = str(self._parameter("odom_topic", "/go2/odom"))
        self.map_cloud_topic = str(
            self._parameter("map_cloud_topic", "/go2/map/cloud")
        )
        self.occupancy_topic = str(self._parameter("occupancy_topic", "/map"))
        self.status_topic = str(
            self._parameter("status_topic", "/go2/mapping/status")
        )
        self.time_sync_status_topic = str(
            self._parameter("time_sync_status_topic", "/go2/time_sync/status")
        )
        self._time_sync_guard = TimeSyncStatusGuard(
            required=bool(self._parameter("require_time_sync_status", True))
        )
        self.world_frame = _normal_frame(self._parameter("world_frame", "odom"))
        if not self.world_frame:
            raise ValueError("world_frame must not be empty")

        self.max_cloud_points = int(self._parameter("max_cloud_points", 300000))
        self.max_cloud_age_sec = float(
            self._parameter("max_cloud_age_sec", 2.0)
        )
        self.max_odom_age_sec = float(self._parameter("max_odom_age_sec", 0.5))
        self.out_of_order_tolerance_sec = float(
            self._parameter("out_of_order_tolerance_sec", 0.05)
        )
        self.future_tolerance_sec = float(
            self._parameter("future_tolerance_sec", 1.0)
        )
        self.max_dense_grid_cells = int(
            self._parameter("max_dense_grid_cells", 4000000)
        )
        self.output_dir = str(self._parameter("output_dir", "~/go2_maps"))
        self.load_state_path = str(self._parameter("load_state_path", ""))
        self.autosave_interval_sec = float(
            self._parameter("autosave_interval_sec", 60.0)
        )
        self.free_threshold = int(self._parameter("save_free_threshold", 25))
        self.occupied_threshold = int(
            self._parameter("save_occupied_threshold", 65)
        )

        if self.max_cloud_points <= 0 or self.max_dense_grid_cells <= 0:
            raise ValueError("cloud and dense grid limits must be positive")
        if self.max_cloud_age_sec <= 0.0 or self.max_odom_age_sec <= 0.0:
            raise ValueError("message age limits must be positive")
        if self.autosave_interval_sec < 0.0:
            raise ValueError("autosave_interval_sec must be non-negative")
        self._last_autosave_integrated_clouds = -1

        voxel_size = float(self._parameter("voxel_size", 0.10))
        min_point_range = float(self._parameter("min_point_range", 0.15))
        max_point_range = float(self._parameter("max_point_range", 15.0))
        min_relative_z = float(self._parameter("min_relative_z", -1.0))
        max_relative_z = float(self._parameter("max_relative_z", 3.0))
        max_voxels = int(self._parameter("max_voxels", 300000))
        retention_radius = float(self._parameter("retention_radius", 0.0))
        self._voxels = VoxelAccumulator(
            voxel_size=voxel_size,
            min_point_range=min_point_range,
            max_point_range=max_point_range,
            min_relative_z=min_relative_z,
            max_relative_z=max_relative_z,
            max_voxels=max_voxels,
            retention_radius=retention_radius,
        )

        grid_resolution = float(self._parameter("grid_resolution", 0.10))
        self._grid = LogOddsGrid(
            resolution=grid_resolution,
            hit_log_odds=float(self._parameter("hit_log_odds", 0.85)),
            miss_log_odds=float(self._parameter("miss_log_odds", -0.40)),
            min_log_odds=float(self._parameter("min_log_odds", -2.0)),
            max_log_odds=float(self._parameter("max_log_odds", 3.5)),
            obstacle_min_height=float(
                self._parameter("obstacle_min_height", 0.12)
            ),
            obstacle_max_height=float(
                self._parameter("obstacle_max_height", 1.50)
            ),
            max_ray_range=max_point_range,
            max_cells=int(self._parameter("max_grid_cells", 500000)),
            max_rays_per_update=int(
                self._parameter("max_rays_per_cloud", 4000)
            ),
            max_ray_cells=int(self._parameter("max_ray_cells", 4096)),
        )

        self._robot_position: Optional[np.ndarray] = None
        self._last_odom_stamp_ns = 0
        self._last_odom_receive_ns = 0
        self._last_cloud_stamp_ns = 0
        self._last_cloud_receive_ns = 0
        self._last_dense_cropped = False
        self._counts: Dict[str, int] = {
            "cloud_messages": 0,
            "integrated_clouds": 0,
            "dropped_clouds": 0,
            "odom_messages": 0,
            "dropped_odometry": 0,
            "input_points": 0,
            "accepted_points": 0,
        }

        if self.load_state_path:
            self._load_initial_state(self.load_state_path)

        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        map_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        status_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self._cloud_publisher = self.create_publisher(
            PointCloud2, self.map_cloud_topic, map_qos
        )
        self._occupancy_publisher = self.create_publisher(
            OccupancyGrid, self.occupancy_topic, map_qos
        )
        self._status_publisher = self.create_publisher(
            String, self.status_topic, status_qos
        )
        self._cloud_subscription = self.create_subscription(
            PointCloud2, self.cloud_topic, self._cloud_callback, sensor_qos
        )
        self._odom_subscription = self.create_subscription(
            Odometry, self.odom_topic, self._odom_callback, sensor_qos
        )
        self._time_sync_subscription = self.create_subscription(
            String,
            self.time_sync_status_topic,
            self._time_sync_callback,
            status_qos,
        )

        self._save_service = self.create_service(
            Trigger, "/go2/map/save", self._save_callback
        )
        self._reset_service = self.create_service(
            Trigger, "/go2/map/reset", self._reset_callback
        )

        cloud_rate = float(self._parameter("cloud_publish_rate", 0.5))
        occupancy_rate = float(self._parameter("occupancy_publish_rate", 1.0))
        status_rate = float(self._parameter("status_publish_rate", 1.0))
        if min(cloud_rate, occupancy_rate, status_rate) <= 0.0:
            raise ValueError("publish rates must be positive")
        self._cloud_timer = self.create_timer(
            1.0 / cloud_rate, self._publish_map_cloud
        )
        self._occupancy_timer = self.create_timer(
            1.0 / occupancy_rate, self._publish_occupancy
        )
        self._status_timer = self.create_timer(1.0 / status_rate, self._publish_status)

        self._autosave_timer = None
        if self.autosave_interval_sec > 0.0:
            self._autosave_timer = self.create_timer(
                self.autosave_interval_sec, self._autosave_callback
            )

        self.get_logger().info(
            "mapping world-frame cloud %s with odometry %s in frame %s"
            % (self.cloud_topic, self.odom_topic, self.world_frame)
        )
        if self.autosave_interval_sec > 0.0:
            self.get_logger().info(
                "automatic map snapshots every %.1f s below %s"
                % (self.autosave_interval_sec, self.output_dir)
            )

    def _parameter(self, name: str, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _warn_drop(self, reason: str) -> None:
        self._last_warning = reason
        now = time.monotonic()
        if now - self._last_warning_monotonic >= 5.0:
            self.get_logger().warning(reason)
            self._last_warning_monotonic = now

    def _time_sync_callback(self, message: String) -> None:
        with self._lock:
            first_fault = not bool(self._time_sync_guard.fault_reason)
            try:
                payload = json.loads(message.data)
            except (TypeError, ValueError) as error:
                reason = self._time_sync_guard.latch(
                    "invalid time-sync status JSON: {}".format(error)
                )
            else:
                reason = self._time_sync_guard.update(payload)
            if reason:
                if first_fault:
                    self._invalidate_for_time_fault_locked()
                self._warn_drop(
                    "mapping time-sync fault latched: {}; restart the complete stack".format(
                        reason
                    )
                )

    def _invalidate_for_time_fault_locked(self) -> None:
        """Discard every value that could have crossed a sensor-clock epoch."""
        self._voxels.clear()
        self._grid.clear()
        self._robot_position = None
        self._last_odom_stamp_ns = 0
        self._last_odom_receive_ns = 0
        self._last_cloud_stamp_ns = 0
        self._last_cloud_receive_ns = 0
        self._last_dense_cropped = False

    def _odom_callback(self, message: Odometry) -> None:
        now_ns = self.get_clock().now().nanoseconds
        with self._lock:
            if not self._time_sync_guard.ready:
                self._counts["dropped_odometry"] += 1
                return
        frame = _normal_frame(message.header.frame_id)
        if frame != self.world_frame:
            with self._lock:
                self._counts["dropped_odometry"] += 1
                self._warn_drop(
                    "odometry frame {!r} does not match world_frame {!r}".format(
                        frame, self.world_frame
                    )
                )
            return
        position = message.pose.pose.position
        robot = np.asarray([position.x, position.y, position.z], dtype=np.float64)
        orientation = message.pose.pose.orientation
        quaternion = np.asarray(
            [orientation.x, orientation.y, orientation.z, orientation.w],
            dtype=np.float64,
        )
        if not np.isfinite(robot).all() or not np.isfinite(quaternion).all():
            with self._lock:
                self._counts["dropped_odometry"] += 1
                self._warn_drop("odometry contains a non-finite pose")
            return
        norm = float(np.linalg.norm(quaternion))
        if norm < 1.0e-6:
            with self._lock:
                self._counts["dropped_odometry"] += 1
                self._warn_drop("odometry contains a zero-length quaternion")
            return

        message_stamp_ns = _stamp_ns(message.header)
        if message_stamp_ns <= 0:
            with self._lock:
                self._counts["dropped_odometry"] += 1
                self._warn_drop("zero-stamped odometry was rejected")
            return
        if self._is_stale(message_stamp_ns, now_ns, self.max_odom_age_sec):
            with self._lock:
                self._counts["dropped_odometry"] += 1
                self._warn_drop("stale or future-dated odometry was rejected")
            return
        with self._lock:
            self._robot_position = robot
            self._last_odom_stamp_ns = message_stamp_ns
            self._last_odom_receive_ns = now_ns
            self._counts["odom_messages"] += 1

    def _is_stale(self, stamp_ns: int, now_ns: int, max_age_sec: float) -> bool:
        if stamp_ns <= 0 or now_ns <= 0:
            return False
        delta_sec = (now_ns - stamp_ns) / 1.0e9
        return delta_sec > max_age_sec or delta_sec < -self.future_tolerance_sec

    def _cloud_callback(self, message: PointCloud2) -> None:
        now_ns = self.get_clock().now().nanoseconds
        message_stamp_ns = _stamp_ns(message.header)
        with self._lock:
            self._counts["cloud_messages"] += 1
            if not self._time_sync_guard.ready:
                self._drop_cloud("waiting for locked sensor time synchronization")
                return
        if message_stamp_ns <= 0:
            self._drop_cloud("zero-stamped cloud was rejected")
            return

        frame = _normal_frame(message.header.frame_id)
        if frame != self.world_frame:
            self._drop_cloud(
                "cloud frame {!r} does not match world_frame {!r}".format(
                    frame, self.world_frame
                )
            )
            return
        if self._is_stale(message_stamp_ns, now_ns, self.max_cloud_age_sec):
            self._drop_cloud("stale or future-dated cloud was rejected")
            return
        with self._lock:
            if (
                self._last_cloud_stamp_ns
                and message_stamp_ns < self._last_cloud_stamp_ns
                - int(self.out_of_order_tolerance_sec * 1.0e9)
            ):
                self._drop_cloud("out-of-order cloud was rejected")
                return
            if self._robot_position is None:
                self._drop_cloud("cloud received before valid odometry")
                return
            odom_stamp_ns = self._last_odom_stamp_ns
            robot = self._robot_position.copy()
        if (
            odom_stamp_ns
            and abs(message_stamp_ns - odom_stamp_ns)
            > int(self.max_odom_age_sec * 1.0e9)
        ):
            self._drop_cloud("cloud and latest odometry timestamps are too far apart")
            return

        try:
            points = read_xyz(message, max_points=self.max_cloud_points)
        except (PointCloudFormatError, TypeError, ValueError) as error:
            self._drop_cloud("invalid PointCloud2: {}".format(error))
            return

        with self._lock:
            accepted = self._voxels.filter_points(points, robot)
            self._voxels.fuse_filtered(accepted, robot, message_stamp_ns)
            self._grid.update(accepted, robot, message_stamp_ns)
            self._last_cloud_stamp_ns = message_stamp_ns
            self._last_cloud_receive_ns = now_ns
            self._counts["input_points"] += int(points.shape[0])
            self._counts["accepted_points"] += int(accepted.shape[0])
            self._counts["integrated_clouds"] += 1

    def _drop_cloud(self, reason: str) -> None:
        with self._lock:
            self._counts["dropped_clouds"] += 1
            self._warn_drop(reason)

    def _publish_map_cloud(self) -> None:
        with self._lock:
            points = self._voxels.points()
        # Keep the public map-cloud schema identical for both mapping backends.
        # The RGB-D backend publishes fused camera colors; the LiDAR-only
        # backend has no color sensor correspondence, so it publishes neutral
        # gray while retaining the standard packed PCL ``rgb`` field.
        colors_rgb = np.full(points.shape, 180, dtype=np.uint8)
        now = self.get_clock().now().to_msg()
        message = PointCloud2()
        message.header.stamp = now
        message.header.frame_id = self.world_frame
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
        message.row_step = message.point_step * message.width
        payload = array("B")
        payload.frombytes(xyzrgb_to_float32_bytes(points, colors_rgb))
        message.data = payload
        message.is_dense = True
        self._cloud_publisher.publish(message)

    def _dense_grid(self) -> Tuple[np.ndarray, float, float, bool]:
        with self._lock:
            return self._grid.to_dense(self.max_dense_grid_cells)

    def _publish_occupancy(self) -> None:
        grid, origin_x, origin_y, cropped = self._dense_grid()
        with self._lock:
            self._last_dense_cropped = cropped
            if cropped:
                self._last_warning = (
                    "dense occupancy output was cropped around the robot to honor "
                    "max_dense_grid_cells"
                )
        now = self.get_clock().now().to_msg()
        message = OccupancyGrid()
        message.header.stamp = now
        message.header.frame_id = self.world_frame
        message.info.map_load_time = now
        message.info.resolution = float(self._grid.resolution)
        message.info.width = int(grid.shape[1])
        message.info.height = int(grid.shape[0])
        message.info.origin.position.x = float(origin_x)
        message.info.origin.position.y = float(origin_y)
        message.info.origin.position.z = 0.0
        message.info.origin.orientation.w = 1.0
        payload = array("b")
        payload.frombytes(
            np.ascontiguousarray(grid, dtype=np.int8).tobytes(order="C")
        )
        message.data = payload
        self._occupancy_publisher.publish(message)

    def _metadata(self, dense_cropped: bool) -> Dict[str, object]:
        return {
            "producer": "go2_mapping",
            "world_frame": self.world_frame,
            "voxel_size": self._voxels.voxel_size,
            "grid_resolution": self._grid.resolution,
            "dense_grid_cropped": bool(dense_cropped),
            "voxel_count": len(self._voxels),
            "grid_cell_count": len(self._grid),
        }

    def _save_callback(self, _request, response):
        try:
            with self._lock:
                if not self._time_sync_guard.ready:
                    raise RuntimeError(
                        "map is not saveable without a locked, fault-free "
                        "sensor-time boundary"
                    )
                voxel_state = self._voxels.state()
                grid_state = self._grid.state()
                grid, origin_x, origin_y, cropped = self._grid.to_dense(
                    self.max_dense_grid_cells
                )
                metadata = self._metadata(cropped)
            destination = save_snapshot(
                output_dir=self.output_dir,
                voxel_state=voxel_state,
                grid_state=grid_state,
                occupancy=grid,
                origin_x=origin_x,
                origin_y=origin_y,
                resolution=self._grid.resolution,
                metadata=metadata,
                free_threshold=self.free_threshold,
                occupied_threshold=self.occupied_threshold,
            )
            response.success = True
            response.message = "saved atomic snapshot to {}".format(destination)
        except Exception as error:  # Service must report filesystem errors to caller.
            self.get_logger().error("map save failed: {}".format(error))
            response.success = False
            response.message = "map save failed: {}".format(error)
        return response

    def _autosave_callback(self) -> None:
        if self.autosave_interval_sec <= 0.0:
            return
        with self._lock:
            revision = self._counts["integrated_clouds"]
            ready = self._time_sync_guard.ready
            nonempty = len(self._voxels) > 0
        if (
            not ready
            or not nonempty
            or revision <= self._last_autosave_integrated_clouds
        ):
            return
        response = Trigger.Response()
        self._save_callback(None, response)
        if response.success:
            self._last_autosave_integrated_clouds = revision
            self.get_logger().info("automatic map save: %s" % response.message)

    def _reset_callback(self, _request, response):
        with self._lock:
            self._voxels.clear()
            self._grid.clear()
            self._last_cloud_stamp_ns = 0
            self._last_cloud_receive_ns = 0
            self._last_dense_cropped = False
        response.success = True
        response.message = "3D voxel map and 2D occupancy evidence reset"
        return response

    def _load_initial_state(self, path: str) -> None:
        arrays, metadata = load_snapshot(path)
        if _normal_frame(metadata.get("world_frame", "")) != self.world_frame:
            raise ValueError("snapshot world_frame does not match configured world_frame")
        if not math.isclose(
            float(metadata.get("voxel_size", -1.0)),
            self._voxels.voxel_size,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("snapshot voxel_size does not match configuration")
        if not math.isclose(
            float(metadata.get("grid_resolution", -1.0)),
            self._grid.resolution,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("snapshot grid_resolution does not match configuration")
        self._voxels.restore(arrays)
        self._grid.restore(arrays)
        self.get_logger().info(
            "loaded mapping state from %s (%d voxels, %d grid cells)"
            % (path, len(self._voxels), len(self._grid))
        )

    def _publish_status(self) -> None:
        now_ns = self.get_clock().now().nanoseconds
        with self._lock:
            if self._time_sync_guard.fault_reason:
                state = "fault_latched"
            elif not self._time_sync_guard.ready:
                state = "waiting_for_time_sync"
            else:
                state = (
                    "mapping"
                    if self._robot_position is not None
                    else "waiting_for_odometry"
                )
            cloud_age = (
                None
                if not self._last_cloud_receive_ns
                else max(0.0, (now_ns - self._last_cloud_receive_ns) / 1.0e9)
            )
            payload = {
                "state": state,
                "world_frame": self.world_frame,
                "cloud_topic": self.cloud_topic,
                "odom_topic": self.odom_topic,
                "uptime_sec": round(time.monotonic() - self._started_monotonic, 3),
                "last_cloud_receive_age_sec": cloud_age,
                "voxel_count": len(self._voxels),
                "grid_cell_count": len(self._grid),
                "dense_grid_cropped": self._last_dense_cropped,
                "last_warning": self._last_warning,
                "time_sync_state": self._time_sync_guard.state,
                "time_sync_fault_reason": self._time_sync_guard.fault_reason,
                "counters": dict(self._counts),
            }
        message = String()
        message.data = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        self._status_publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MappingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._autosave_callback()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
