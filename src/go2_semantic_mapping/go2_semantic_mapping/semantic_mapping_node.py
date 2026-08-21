# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Go2 Semantic Mapping contributors

"""ROS 2 Foxy node for persistent Go2 RGB/LiDAR semantic voxel mapping.

This node is perception-only. It does not publish velocity, sport, posture, or
other robot motion commands.
"""

from __future__ import annotations

from array import array
import colorsys
import hashlib
import json
import math
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import Header, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from go2_mapping.pointcloud import read_xyz
from go2_mapping.time_sync_guard import TimeSyncStatusGuard

from .core import (
    Detection,
    SemanticVoxelMap,
    as_transform,
    associate_detections,
    nearest_stamped_sample,
    pose_matrix,
    project_base_points,
    sample_image_rgb,
    save_snapshot_bundle_atomic,
    transform_points,
)


class LazyUltralyticsDetector:
    """Load Ultralytics only on the first enabled inference request."""

    def __init__(
        self,
        enabled: bool,
        model_path: str,
        confidence: float,
        iou: float,
        device: str,
        class_filter: Sequence[int],
        error_callback,
    ) -> None:
        self.enabled = bool(enabled)
        self.model_path = str(model_path).strip()
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.device = str(device).strip()
        self.class_filter = list(class_filter)
        self.error_callback = error_callback
        self._model = None
        self._load_attempted = False
        self._permanent_error = ""

    @property
    def status(self) -> str:
        if not self.enabled:
            return "detector disabled"
        if not self.model_path:
            return "no detector model configured"
        if self._permanent_error:
            return "detector unavailable: {}".format(self._permanent_error)
        if self._model is None:
            return "detector waiting for lazy load"
        return "detector ready"

    def _load(self) -> bool:
        if not self.enabled or not self.model_path:
            return False
        if self._model is not None:
            return True
        if self._load_attempted:
            return False
        self._load_attempted = True
        try:
            from ultralytics import YOLO  # pylint: disable=import-outside-toplevel

            self._model = YOLO(self.model_path)
            return True
        except Exception as exc:  # optional dependency or invalid model
            self._permanent_error = str(exc)
            self.error_callback(
                "Ultralytics model could not be loaded; continuing with unknown-only geometry: {}".format(
                    exc
                )
            )
            return False

    def detect(self, image_bgr: np.ndarray) -> List[Detection]:
        if not self._load():
            return []
        arguments = {
            "source": image_bgr,
            "conf": self.confidence,
            "iou": self.iou,
            "verbose": False,
        }
        if self.device:
            arguments["device"] = self.device
        if self.class_filter:
            arguments["classes"] = self.class_filter

        try:
            results = self._model.predict(**arguments)
            if not results:
                return []
            result = results[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                return []

            xyxy = boxes.xyxy.detach().cpu().numpy()
            scores = boxes.conf.detach().cpu().numpy()
            raw_labels = boxes.cls.detach().cpu().numpy().astype(np.int64)
            names = getattr(result, "names", {})
            mask_rows = None
            result_masks = getattr(result, "masks", None)
            if result_masks is not None and getattr(result_masks, "data", None) is not None:
                mask_rows = result_masks.data.detach().cpu().numpy()

            detections: List[Detection] = []
            for index, (box, score, raw_label) in enumerate(zip(xyxy, scores, raw_labels)):
                if isinstance(names, dict):
                    name = str(names.get(int(raw_label), "class_{}".format(int(raw_label))))
                elif int(raw_label) < len(names):
                    name = str(names[int(raw_label)])
                else:
                    name = "class_{}".format(int(raw_label))

                mask = None
                if mask_rows is not None and index < len(mask_rows):
                    mask = mask_rows[index]
                    if mask.shape != image_bgr.shape[:2]:
                        mask = cv2.resize(
                            mask.astype(np.float32),
                            (image_bgr.shape[1], image_bgr.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    mask = mask > 0.5

                # Label 0 is reserved for unknown map geometry.
                semantic_label = int(raw_label) + 1
                detections.append(
                    Detection(
                        label_id=semantic_label,
                        label_name=name,
                        confidence=float(score),
                        xyxy=tuple(float(value) for value in box),
                        mask=mask,
                    )
                )
            return detections
        except Exception as exc:
            self.error_callback("Ultralytics inference failed for this frame: {}".format(exc))
            return []


class Go2SemanticMappingNode(Node):
    """Fuse synchronized camera detections and base-frame LiDAR into odom voxels."""

    def __init__(self) -> None:
        super().__init__("semantic_mapping")
        self._declare_parameters()
        self.add_on_set_parameters_callback(self._reject_live_parameter_updates)

        self._bridge = CvBridge()
        self._cache_lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._time_sync_lock = threading.RLock()
        self._time_sync_guard = TimeSyncStatusGuard(
            required=bool(self._parameter("require_time_sync_status"))
        )
        self._time_sync_fault_handled = False
        self._last_time_sync_status_monotonic: Optional[float] = None
        self._cloud_samples = deque(maxlen=int(self._parameter("cloud_buffer_size")))
        self._odom_samples = deque(maxlen=int(self._parameter("odom_buffer_size")))
        self._last_inference_start = -math.inf
        self._inference_busy = False
        self._map_generation = 0
        self._worker_thread: Optional[threading.Thread] = None
        self._shutting_down = False
        self._warning_times: Dict[str, float] = {}

        self._camera_info_valid = False
        self._camera_info_metadata: Dict[str, object] = {}
        self._camera_from_base = self._load_extrinsic()
        intrinsics = np.asarray(
            [
                self._parameter("fx"),
                self._parameter("fy"),
                self._parameter("cx"),
                self._parameter("cy"),
            ],
            dtype=np.float64,
        )
        expected_width = int(self._parameter("expected_image_width"))
        expected_height = int(self._parameter("expected_image_height"))
        self._intrinsics_valid = bool(
            np.isfinite(intrinsics).all()
            and intrinsics[0] > 0.0
            and intrinsics[1] > 0.0
            and 0.0 <= intrinsics[2] < expected_width
            and 0.0 <= intrinsics[3] < expected_height
        )
        self._configured_calibration_ready = self._compute_calibration_ready()
        self._calibration_ready = self._configured_calibration_ready and not bool(
            self._parameter("require_matching_camera_info")
        )
        self._map = SemanticVoxelMap(
            voxel_size=float(self._parameter("voxel_size")),
            max_voxels=int(self._parameter("max_voxels")),
            default_color=self._parameter("default_rgb"),
        )

        class_filter = self._parse_class_filter(str(self._parameter("detector_classes_csv")))
        self._detector = LazyUltralyticsDetector(
            enabled=bool(self._parameter("detector_enabled")),
            model_path=str(self._parameter("detector_model")),
            confidence=float(self._parameter("detector_confidence")),
            iou=float(self._parameter("detector_iou")),
            device=str(self._parameter("detector_device")),
            class_filter=class_filter,
            error_callback=lambda message: self._warn_throttled("detector", message, 10.0),
        )
        self._detector_model_sha256 = self._hash_file(
            str(self._parameter("detector_model"))
        )

        output_qos = QoSProfile(depth=1)
        output_qos.reliability = ReliabilityPolicy.RELIABLE
        output_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._cloud_publisher = self.create_publisher(
            PointCloud2, str(self._parameter("semantic_cloud_topic")), output_qos
        )
        self._marker_publisher = self.create_publisher(
            MarkerArray, str(self._parameter("semantic_marker_topic")), output_qos
        )
        self._image_publisher = self.create_publisher(
            Image, str(self._parameter("annotated_image_topic")), 2
        )

        self._image_subscription = self.create_subscription(
            Image,
            str(self._parameter("image_topic")),
            self._image_callback,
            qos_profile_sensor_data,
        )
        self._cloud_subscription = self.create_subscription(
            PointCloud2,
            str(self._parameter("cloud_topic")),
            self._cloud_callback,
            qos_profile_sensor_data,
        )
        self._odom_subscription = self.create_subscription(
            Odometry,
            str(self._parameter("odom_topic")),
            self._odom_callback,
            qos_profile_sensor_data,
        )
        self._camera_info_subscription = self.create_subscription(
            CameraInfo,
            str(self._parameter("camera_info_topic")),
            self._camera_info_callback,
            qos_profile_sensor_data,
        )
        status_qos = QoSProfile(depth=10)
        status_qos.reliability = ReliabilityPolicy.RELIABLE
        self._time_sync_subscription = self.create_subscription(
            String,
            str(self._parameter("time_sync_status_topic")),
            self._time_sync_callback,
            status_qos,
        )

        self._save_service = self.create_service(
            Trigger, str(self._parameter("save_service")), self._save_callback
        )
        self._reset_service = self.create_service(
            Trigger, str(self._parameter("reset_service")), self._reset_callback
        )
        publish_period = 1.0 / max(float(self._parameter("publish_rate_hz")), 0.01)
        self._publish_timer = self.create_timer(publish_period, self._publish_map)
        self._calibration_timer = self.create_timer(30.0, self._calibration_warning)
        watchdog_timeout = max(
            float(self._parameter("time_sync_status_timeout_sec")), 0.1
        )
        self._time_sync_watchdog_timer = self.create_timer(
            min(max(watchdog_timeout / 4.0, 0.05), 0.25),
            self._time_sync_watchdog,
        )

        self.get_logger().info(
            "Semantic mapping is perception-only; no motion command publishers or services are created."
        )
        self.get_logger().info(
            "Inputs: image={} cloud={} odom={}".format(
                str(self._parameter("image_topic")),
                str(self._parameter("cloud_topic")),
                str(self._parameter("odom_topic")),
            )
        )
        self.get_logger().info("Detector state: {}".format(self._detector.status))
        self._calibration_warning()

    def _declare_parameters(self) -> None:
        self.declare_parameter("image_topic", "/go2/camera/image_rect")
        self.declare_parameter("cloud_topic", "/go2/lidar/cloud_base")
        self.declare_parameter("odom_topic", "/go2/odom")
        self.declare_parameter("camera_info_topic", "/go2/camera/camera_info")
        self.declare_parameter("semantic_cloud_topic", "/go2/semantic/cloud")
        self.declare_parameter("semantic_marker_topic", "/go2/semantic/markers")
        self.declare_parameter("annotated_image_topic", "/go2/semantic/annotated_image")
        self.declare_parameter("save_service", "/go2/semantic/save")
        self.declare_parameter("reset_service", "/go2/semantic/reset")
        self.declare_parameter("map_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("expected_cloud_frame", "base_link")
        self.declare_parameter("reject_unexpected_cloud_frame", True)
        self.declare_parameter("reject_unexpected_odom_frames", True)
        self.declare_parameter("time_sync_status_topic", "/go2/time_sync/status")
        self.declare_parameter("require_time_sync_status", True)
        self.declare_parameter("time_sync_status_timeout_sec", 1.0)

        self.declare_parameter("calibration_confirmed", False)
        self.declare_parameter("require_calibration_confirmation", True)
        self.declare_parameter("fx", 0.0)
        self.declare_parameter("fy", 0.0)
        self.declare_parameter("cx", 0.0)
        self.declare_parameter("cy", 0.0)
        self.declare_parameter("expected_image_width", 1920)
        self.declare_parameter("expected_image_height", 1080)
        self.declare_parameter("expected_camera_frame", "go2_front_camera_optical_frame")
        self.declare_parameter("require_matching_camera_info", True)
        self.declare_parameter("camera_info_intrinsics_tolerance", 1.0e-3)
        self.declare_parameter(
            "base_to_camera_optical",
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        )

        self.declare_parameter("detector_enabled", False)
        self.declare_parameter("detector_model", "")
        self.declare_parameter("detector_device", "")
        self.declare_parameter("detector_classes_csv", "")
        self.declare_parameter("detector_confidence", 0.35)
        self.declare_parameter("detector_iou", 0.45)
        self.declare_parameter("inference_rate_hz", 2.0)
        self.declare_parameter("max_processing_latency_sec", 5.0)

        self.declare_parameter("max_image_cloud_delta_sec", 0.15)
        self.declare_parameter("camera_time_offset_sec", 0.0)
        self.declare_parameter("cloud_buffer_size", 30)
        self.declare_parameter("max_cached_cloud_age_sec", 0.5)
        self.declare_parameter("max_odom_delta_sec", 0.1)
        self.declare_parameter("odom_buffer_size", 250)
        self.declare_parameter("max_cloud_points_per_fusion", 50000)
        self.declare_parameter("projection_min_depth", 0.15)
        self.declare_parameter("projection_max_depth", 15.0)
        self.declare_parameter("minimum_detection_points", 3)
        self.declare_parameter("absolute_depth_gate", 0.25)
        self.declare_parameter("depth_mad_scale", 3.0)
        self.declare_parameter("draw_projected_points", True)
        self.declare_parameter("max_projected_points_overlay", 1200)
        self.declare_parameter("projected_point_radius", 1)

        self.declare_parameter("voxel_size", 0.10)
        self.declare_parameter("max_voxels", 100000)
        self.declare_parameter("default_rgb", [128, 128, 128])
        self.declare_parameter("publish_rate_hz", 1.0)
        self.declare_parameter("max_label_markers", 250)
        self.declare_parameter("marker_text_height", 0.12)
        self.declare_parameter("marker_z_offset", 0.12)
        self.declare_parameter("save_directory", "~/go2_semantic_maps")
        self.declare_parameter("save_prefix", "semantic_map")

    def _parameter(self, name: str):
        return self.get_parameter(name).value

    @staticmethod
    def _reject_live_parameter_updates(_parameters) -> SetParametersResult:
        return SetParametersResult(
            successful=False,
            reason="semantic mapping parameters are restart-required for consistency",
        )

    @staticmethod
    def _hash_file(path: str) -> str:
        source = os.path.abspath(os.path.expanduser(str(path)))
        if not source or not os.path.isfile(source):
            return ""
        digest = hashlib.sha256()
        with open(source, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _parse_class_filter(value: str) -> List[int]:
        if not value.strip():
            return []
        result = []
        for token in value.split(","):
            result.append(int(token.strip()))
        return result

    @staticmethod
    def _normalize_frame(frame_id: str) -> str:
        return str(frame_id).strip().lstrip("/")

    @staticmethod
    def _message_stamp(stamp: Time) -> Optional[float]:
        seconds = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        return seconds if seconds > 0.0 else None

    def _load_extrinsic(self) -> Optional[np.ndarray]:
        try:
            return as_transform(self._parameter("base_to_camera_optical"))
        except Exception as exc:
            self.get_logger().error("Invalid base_to_camera_optical matrix: {}".format(exc))
            return None

    def _compute_calibration_ready(self) -> bool:
        if self._camera_from_base is None or not self._intrinsics_valid:
            return False
        if bool(self._parameter("require_calibration_confirmation")):
            return bool(self._parameter("calibration_confirmed"))
        return True

    def _calibration_warning(self) -> None:
        if self._calibration_ready:
            return
        self.get_logger().error(
            "CALIBRATION REQUIRED: semantic projection is disabled. Set measured fx/fy/cx/cy and "
            "the row-major camera_from_base optical transform, then set calibration_confirmed:=true. "
            "The identity matrix is only a placeholder. Unknown geometry may still be fused."
        )

    def _time_sync_ready(self) -> bool:
        with self._time_sync_lock:
            return self._time_sync_guard.ready

    def _latch_time_sync_fault(self, reason: str) -> None:
        with self._time_sync_lock:
            latched_reason = self._time_sync_guard.latch(reason)
        self._invalidate_for_time_sync_fault(latched_reason)

    def _invalidate_for_time_sync_fault(self, latched_reason: str) -> None:
        with self._time_sync_lock:
            if self._time_sync_fault_handled:
                return
            self._time_sync_fault_handled = True
        if self._shutting_down:
            return

        # The guard becomes non-ready before waiting for the inference lock.
        # A worker already at its commit boundary therefore discards its result.
        with self._inference_lock:
            self._map_generation += 1
            removed = self._map.clear()
        with self._cache_lock:
            self._cloud_samples.clear()
            self._odom_samples.clear()
        self.get_logger().error(
            "SEMANTIC TIME FAULT LATCHED: {}. Cleared {} voxels and invalidated "
            "in-flight inference. Restart the complete stack before recording "
            "another semantic map.".format(latched_reason, removed)
        )
        self._publish_map()

    def _time_sync_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            self._latch_time_sync_fault(
                "time-sync status JSON is invalid: {}".format(error)
            )
            return
        with self._time_sync_lock:
            self._last_time_sync_status_monotonic = time.monotonic()
            reason = self._time_sync_guard.update(payload)
        if reason:
            self._invalidate_for_time_sync_fault(reason)

    def _time_sync_watchdog(self) -> None:
        if not self._time_sync_guard.required:
            return
        with self._time_sync_lock:
            last_status = self._last_time_sync_status_monotonic
            ever_locked = self._time_sync_guard.ever_locked
            faulted = bool(self._time_sync_guard.fault_reason)
        if not ever_locked or faulted or last_status is None:
            return
        age = time.monotonic() - last_status
        timeout = max(float(self._parameter("time_sync_status_timeout_sec")), 0.1)
        if age > timeout:
            self._latch_time_sync_fault(
                "time-sync status is stale ({:.3f}s > {:.3f}s)".format(
                    age, timeout
                )
            )

    def _camera_info_callback(self, message: CameraInfo) -> None:
        expected_width = int(self._parameter("expected_image_width"))
        expected_height = int(self._parameter("expected_image_height"))
        expected_frame = self._normalize_frame(self._parameter("expected_camera_frame"))
        actual_frame = self._normalize_frame(message.header.frame_id)
        k = np.asarray(message.k, dtype=np.float64)
        d = np.asarray(message.d, dtype=np.float64)
        r = np.asarray(message.r, dtype=np.float64)
        p = np.asarray(message.p, dtype=np.float64)
        configured = np.asarray(
            [
                self._parameter("fx"),
                self._parameter("fy"),
                self._parameter("cx"),
                self._parameter("cy"),
            ],
            dtype=np.float64,
        )
        rectified = (
            np.asarray([p[0], p[5], p[2], p[6]], dtype=np.float64)
            if p.size == 12
            else np.full(4, np.nan, dtype=np.float64)
        )
        tolerance = max(
            float(self._parameter("camera_info_intrinsics_tolerance")), 0.0
        )
        valid = bool(
            int(message.width) == expected_width
            and int(message.height) == expected_height
            and actual_frame == expected_frame
            and k.size == 9
            and r.size == 9
            and p.size == 12
            and np.isfinite(k).all()
            and np.isfinite(d).all()
            and np.isfinite(r).all()
            and np.isfinite(p).all()
            and k[0] > 0.0
            and k[4] > 0.0
            and rectified[0] > 0.0
            and rectified[1] > 0.0
            and np.allclose(configured, rectified, rtol=0.0, atol=tolerance)
            and message.distortion_model
            in ("plumb_bob", "rational_polynomial", "equidistant")
        )
        metadata = {
            "frame_id": actual_frame,
            "width": int(message.width),
            "height": int(message.height),
            "distortion_model": str(message.distortion_model),
            "D": d.tolist(),
            "K": k.tolist(),
            "R": r.tolist(),
            "P": p.tolist(),
        }
        with self._cache_lock:
            changed = valid != self._camera_info_valid
            self._camera_info_valid = valid
            self._camera_info_metadata = metadata
            self._calibration_ready = self._configured_calibration_ready and (
                valid or not bool(self._parameter("require_matching_camera_info"))
            )
        if changed:
            if valid:
                self.get_logger().info(
                    "CameraInfo matches configured rectified intrinsics; semantic projection enabled."
                )
            else:
                self._warn_throttled(
                    "camera_info_invalid",
                    "CameraInfo does not match frame, resolution, or rectified intrinsics; projection disabled.",
                )

    def _warn_throttled(self, key: str, message: str, period: float = 5.0) -> None:
        now = time.monotonic()
        previous = self._warning_times.get(key, -math.inf)
        if now - previous >= period:
            self._warning_times[key] = now
            self.get_logger().warning(message)

    def _cloud_callback(self, message: PointCloud2) -> None:
        if not self._time_sync_ready():
            self._warn_throttled(
                "time_sync_cloud", "Waiting for a locked sensor time boundary."
            )
            return
        receive_time = time.monotonic()
        expected = self._normalize_frame(self._parameter("expected_cloud_frame"))
        actual = self._normalize_frame(message.header.frame_id)
        if expected and actual != expected:
            self._warn_throttled(
                "cloud_frame",
                "Cloud frame '{}' does not match expected base frame '{}'.".format(actual, expected),
            )
            if bool(self._parameter("reject_unexpected_cloud_frame")):
                return
        stamp = self._message_stamp(message.header.stamp)
        if stamp is None:
            self._warn_throttled("cloud_stamp", "Zero-stamped LiDAR cloud was rejected.")
            return
        with self._cache_lock:
            self._cloud_samples.append((stamp, (message, receive_time)))

    def _odom_callback(self, message: Odometry) -> None:
        if not self._time_sync_ready():
            self._warn_throttled(
                "time_sync_odom", "Waiting for a locked sensor time boundary."
            )
            return
        receive_time = time.monotonic()
        expected_parent = self._normalize_frame(self._parameter("map_frame"))
        expected_child = self._normalize_frame(self._parameter("base_frame"))
        actual_parent = self._normalize_frame(message.header.frame_id)
        actual_child = self._normalize_frame(message.child_frame_id)
        if actual_parent != expected_parent or actual_child != expected_child:
            self._warn_throttled(
                "odom_frames",
                "Odometry frames '{} -> {}' do not match expected '{} -> {}'.".format(
                    actual_parent, actual_child, expected_parent, expected_child
                ),
            )
            if bool(self._parameter("reject_unexpected_odom_frames")):
                return

        pose = message.pose.pose
        try:
            odom_from_base = pose_matrix(
                [pose.position.x, pose.position.y, pose.position.z],
                [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
            )
        except ValueError as exc:
            self._warn_throttled("odom_pose", "Rejected invalid odometry pose: {}".format(exc))
            return
        stamp = self._message_stamp(message.header.stamp)
        if stamp is None:
            self._warn_throttled("odom_stamp", "Zero-stamped odometry was rejected.")
            return
        with self._cache_lock:
            self._odom_samples.append((stamp, odom_from_base))

    def _image_callback(self, message: Image) -> None:
        if not self._time_sync_ready():
            self._warn_throttled(
                "time_sync_image", "Waiting for a locked sensor time boundary."
            )
            return
        receive_time = time.monotonic()
        inference_rate = max(float(self._parameter("inference_rate_hz")), 0.01)
        with self._inference_lock:
            if self._inference_busy or receive_time - self._last_inference_start < 1.0 / inference_rate:
                return

        with self._cache_lock:
            cloud_samples = list(self._cloud_samples)
            odom_samples = list(self._odom_samples)
        if not cloud_samples:
            self._warn_throttled("no_cloud", "Waiting for a base-frame LiDAR cloud.")
            return

        image_stamp = self._message_stamp(message.header.stamp)
        if image_stamp is None:
            self._warn_throttled("image_stamp", "Zero-stamped camera image was rejected.")
            return
        image_stamp += float(self._parameter("camera_time_offset_sec"))
        cloud_pair, image_cloud_delta = nearest_stamped_sample(
            cloud_samples,
            image_stamp,
            float(self._parameter("max_image_cloud_delta_sec")),
        )
        if cloud_pair is None:
            self._warn_throttled(
                "image_cloud_delta",
                "No LiDAR cloud is close enough to the image stamp (nearest {:.3f}s).".format(
                    image_cloud_delta
                ),
            )
            return
        cloud_message, cloud_receive_time = cloud_pair
        cloud_stamp = self._message_stamp(cloud_message.header.stamp)
        if cloud_stamp is None:
            return
        if receive_time - cloud_receive_time > float(self._parameter("max_cached_cloud_age_sec")):
            self._warn_throttled("stale_cloud", "Latest LiDAR cloud is stale; image fusion skipped.")
            return

        odom_from_base, odom_delta = nearest_stamped_sample(
            odom_samples, cloud_stamp, float(self._parameter("max_odom_delta_sec"))
        )
        if odom_from_base is None:
            self._warn_throttled(
                "odom_delta",
                "No odometry sample close enough to cloud stamp (nearest {:.3f}s).".format(odom_delta),
            )
            return

        try:
            image_bgr = self._bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception as exc:
            self._warn_throttled("image_decode", "Could not decode camera image: {}".format(exc))
            return
        if self._normalize_frame(message.header.frame_id) != self._normalize_frame(
            self._parameter("expected_camera_frame")
        ):
            self._warn_throttled(
                "image_frame", "Camera image frame does not match expected optical frame."
            )
            return
        expected_size = (
            int(self._parameter("expected_image_height")),
            int(self._parameter("expected_image_width")),
        )
        if image_bgr.shape[:2] != expected_size:
            self._warn_throttled(
                "image_dimensions",
                "Image is {}x{}, expected {}x{}; fusion skipped.".format(
                    image_bgr.shape[1],
                    image_bgr.shape[0],
                    expected_size[1],
                    expected_size[0],
                ),
            )
            return

        with self._inference_lock:
            if self._inference_busy:
                return
            self._inference_busy = True
            self._last_inference_start = receive_time
            map_generation = self._map_generation
        self._worker_thread = threading.Thread(
            target=self._process_frame,
            args=(
                np.asarray(image_bgr).copy(),
                message.header,
                cloud_message,
                np.asarray(odom_from_base, dtype=np.float64).copy(),
                receive_time,
                map_generation,
            ),
            daemon=False,
            name="go2-semantic-inference",
        )
        self._worker_thread.start()

    @staticmethod
    def _cloud_xyz(message: PointCloud2, maximum_points: int) -> np.ndarray:
        points = read_xyz(message, max_points=max(int(maximum_points), 1))
        finite = points[np.isfinite(points).all(axis=1)]
        return np.asarray(finite, dtype=np.float32)

    def _process_frame(
        self,
        image_bgr: np.ndarray,
        image_header: Header,
        cloud_message: PointCloud2,
        odom_from_base: np.ndarray,
        start_monotonic: float,
        map_generation: int,
    ) -> None:
        detections: List[Detection] = []
        status = self._detector.status
        projected_uv = np.empty((0, 2), dtype=np.float64)
        projected_depth = np.empty(0, dtype=np.float64)
        try:
            points_base = self._cloud_xyz(
                cloud_message, max(int(self._parameter("max_cloud_points_per_fusion")), 1)
            )
            if len(points_base) == 0:
                self._warn_throttled("empty_cloud", "LiDAR cloud contains no finite XYZ points.")
                self._publish_annotated(image_bgr, image_header, detections, "empty cloud")
                return

            if self._calibration_ready:
                detections = self._detector.detect(image_bgr)
                status = self._detector.status

            processing_age = time.monotonic() - start_monotonic
            if processing_age > float(self._parameter("max_processing_latency_sec")):
                self._warn_throttled(
                    "processing_latency",
                    "Frame processing took {:.2f}s and exceeded the freshness guard; fusion skipped.".format(
                        processing_age
                    ),
                )
                self._publish_annotated(image_bgr, image_header, detections, "stale processing result")
                return

            colors = np.zeros((len(points_base), 3), dtype=np.uint8)
            color_valid = np.zeros(len(points_base), dtype=bool)
            labels = np.zeros(len(points_base), dtype=np.uint32)
            confidences = np.zeros(len(points_base), dtype=np.float32)
            class_names = {detection.label_id: detection.label_name for detection in detections}

            if self._calibration_ready and self._camera_from_base is not None:
                projection = project_base_points(
                    points_base,
                    self._camera_from_base,
                    float(self._parameter("fx")),
                    float(self._parameter("fy")),
                    float(self._parameter("cx")),
                    float(self._parameter("cy")),
                    image_bgr.shape[1],
                    image_bgr.shape[0],
                    float(self._parameter("projection_min_depth")),
                    float(self._parameter("projection_max_depth")),
                )
                if len(projection.source_indices):
                    projected_uv = projection.uv
                    projected_depth = projection.depth
                    colors[projection.source_indices] = sample_image_rgb(image_bgr, projection.uv)
                    color_valid[projection.source_indices] = True
                    projected_labels, projected_confidences = associate_detections(
                        projection.uv,
                        projection.depth,
                        detections,
                        min_points=int(self._parameter("minimum_detection_points")),
                        absolute_depth_gate=float(self._parameter("absolute_depth_gate")),
                        mad_scale=float(self._parameter("depth_mad_scale")),
                    )
                    labels[projection.source_indices] = projected_labels
                    confidences[projection.source_indices] = projected_confidences

            points_odom = transform_points(points_base, odom_from_base)
            # Serialize the short fusion commit with reset. A reset increments the
            # generation, so a result computed from a pre-reset frame is discarded.
            with self._inference_lock:
                if (
                    map_generation != self._map_generation
                    or not self._time_sync_ready()
                ):
                    self._publish_annotated(
                        image_bgr,
                        image_header,
                        detections,
                        "discarded by map/time reset",
                    )
                    return
                self._map.update(
                    points_odom,
                    colors_rgb=colors,
                    color_valid=color_valid,
                    labels=labels,
                    confidences=confidences,
                    class_names=class_names,
                    observed_ns=time.time_ns(),
                )
            if self._calibration_ready:
                status = "{} | projected lidar: {}".format(status, len(projected_uv))
            else:
                status = "calibration required | geometry-only"
            self._publish_annotated(
                image_bgr,
                image_header,
                detections,
                status,
                projected_uv=projected_uv,
                projected_depth=projected_depth,
            )
        except Exception as exc:
            self._warn_throttled("frame_processing", "Semantic frame processing failed: {}".format(exc))
            try:
                self._publish_annotated(image_bgr, image_header, detections, "processing error")
            except Exception:
                pass
        finally:
            with self._inference_lock:
                self._inference_busy = False

    @staticmethod
    def _label_bgr(label: int) -> Tuple[int, int, int]:
        hue = (float(label) * 0.61803398875) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.80, 1.0)
        return int(blue * 255), int(green * 255), int(red * 255)

    def _publish_annotated(
        self,
        image_bgr: np.ndarray,
        source_header: Header,
        detections: Sequence[Detection],
        status: str,
        projected_uv: Optional[np.ndarray] = None,
        projected_depth: Optional[np.ndarray] = None,
    ) -> None:
        if self._shutting_down:
            return
        annotated = image_bgr.copy()
        if (
            bool(self._parameter("draw_projected_points"))
            and projected_uv is not None
            and projected_depth is not None
            and len(projected_uv)
        ):
            pixels = np.asarray(projected_uv, dtype=np.float64)
            depths = np.asarray(projected_depth, dtype=np.float64)
            maximum = max(int(self._parameter("max_projected_points_overlay")), 1)
            stride = max(1, int(math.ceil(float(len(pixels)) / maximum)))
            pixels = pixels[::stride][:maximum]
            depths = depths[::stride][:maximum]
            finite_depths = depths[np.isfinite(depths)]
            if len(finite_depths):
                near = float(np.percentile(finite_depths, 5.0))
                far = float(np.percentile(finite_depths, 95.0))
                span = max(far - near, 1e-6)
                normalized = np.clip((depths - near) / span, 0.0, 1.0)
                # JET is available in the OpenCV version shipped with ROS 2 Foxy.
                color_input = np.rint((1.0 - normalized) * 255.0).astype(np.uint8).reshape(-1, 1)
                point_colors = cv2.applyColorMap(color_input, cv2.COLORMAP_JET).reshape(-1, 3)
                radius = max(int(self._parameter("projected_point_radius")), 1)
                for pixel, color in zip(pixels, point_colors):
                    cv2.circle(
                        annotated,
                        (int(round(pixel[0])), int(round(pixel[1]))),
                        radius,
                        (int(color[0]), int(color[1]), int(color[2])),
                        -1,
                        cv2.LINE_AA,
                    )
        for detection in detections:
            color = self._label_bgr(detection.label_id)
            if detection.mask is not None and detection.mask.shape == annotated.shape[:2]:
                mask = detection.mask.astype(bool)
                layer = np.empty_like(annotated)
                layer[:] = color
                annotated[mask] = cv2.addWeighted(annotated[mask], 0.55, layer[mask], 0.45, 0.0)
            x1, y1, x2, y2 = [int(round(value)) for value in detection.xyxy]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            text = "{} {:.0f}%".format(detection.label_name, detection.confidence * 100.0)
            cv2.putText(
                annotated,
                text,
                (max(x1, 0), max(y1 - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            annotated,
            status,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        output = Image()
        output.header = source_header
        output.height = int(annotated.shape[0])
        output.width = int(annotated.shape[1])
        output.encoding = "bgr8"
        output.is_bigendian = False
        output.step = output.width * 3
        payload = array("B")
        payload.frombytes(np.ascontiguousarray(annotated).tobytes())
        output.data = payload
        self._image_publisher.publish(output)

    def _snapshot_cloud(self, snapshot: Dict[str, object]) -> PointCloud2:
        points = np.asarray(snapshot["points"], dtype=np.float32)
        colors = np.asarray(snapshot["colors"], dtype=np.uint8)
        labels = np.asarray(snapshot["labels"], dtype=np.uint32)
        confidences = np.asarray(snapshot["confidences"], dtype=np.float32)
        packed_rgb = (
            colors[:, 0].astype(np.uint32) << 16
            | colors[:, 1].astype(np.uint32) << 8
            | colors[:, 2].astype(np.uint32)
        )
        dtype = np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("rgb", "<u4"),
                ("label", "<u4"),
                ("confidence", "<f4"),
            ]
        )
        rows = np.zeros(len(points), dtype=dtype)
        if len(points):
            rows["x"], rows["y"], rows["z"] = points[:, 0], points[:, 1], points[:, 2]
            rows["rgb"] = packed_rgb
            rows["label"] = labels
            rows["confidence"] = confidences

        message = PointCloud2()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = str(self._parameter("map_frame"))
        message.height = 1
        message.width = len(rows)
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
            PointField(name="label", offset=16, datatype=PointField.UINT32, count=1),
            PointField(name="confidence", offset=20, datatype=PointField.FLOAT32, count=1),
        ]
        message.is_bigendian = False
        message.point_step = dtype.itemsize
        message.row_step = message.point_step * message.width
        payload = array("B")
        payload.frombytes(rows.tobytes())
        message.data = payload
        message.is_dense = True
        return message

    def _snapshot_markers(self, snapshot: Dict[str, object]) -> MarkerArray:
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        points = np.asarray(snapshot["points"])
        labels = np.asarray(snapshot["labels"])
        confidences = np.asarray(snapshot["confidences"])
        semantic_observations = np.asarray(snapshot["semantic_observations"])
        class_names = dict(snapshot["class_names"])
        semantic_rows = np.flatnonzero(labels > 0)
        ranking = sorted(
            semantic_rows.tolist(),
            key=lambda row: (float(confidences[row]), int(semantic_observations[row])),
            reverse=True,
        )[: max(int(self._parameter("max_label_markers")), 0)]

        stamp = self.get_clock().now().to_msg()
        frame_id = str(self._parameter("map_frame"))
        for marker_id, row in enumerate(ranking):
            label = int(labels[row])
            blue, green, red = self._label_bgr(label)
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = frame_id
            marker.ns = "semantic_voxel_labels"
            marker.id = marker_id
            marker.type = Marker.TEXT_VIEW_FACING
            marker.action = Marker.ADD
            marker.pose.position.x = float(points[row, 0])
            marker.pose.position.y = float(points[row, 1])
            marker.pose.position.z = float(points[row, 2]) + float(self._parameter("marker_z_offset"))
            marker.pose.orientation.w = 1.0
            marker.scale.z = float(self._parameter("marker_text_height"))
            marker.color.r = red / 255.0
            marker.color.g = green / 255.0
            marker.color.b = blue / 255.0
            marker.color.a = 1.0
            marker.text = "{} {:.0f}%".format(
                class_names.get(label, "class_{}".format(label)), float(confidences[row]) * 100.0
            )
            markers.markers.append(marker)
        return markers

    def _publish_map(self) -> None:
        if self._shutting_down:
            return
        snapshot = self._map.snapshot(include_persistence_details=False)
        self._cloud_publisher.publish(self._snapshot_cloud(snapshot))
        self._marker_publisher.publish(self._snapshot_markers(snapshot))

    def _save_callback(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if not self._time_sync_ready():
            response.success = False
            response.message = (
                "Save rejected: sensor time is not locked or a time fault is latched"
            )
            return response
        try:
            snapshot = self._map.snapshot()
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            stem = "{}_{}".format(str(self._parameter("save_prefix")), timestamp)
            bundle = save_snapshot_bundle_atomic(
                snapshot,
                os.path.expanduser(str(self._parameter("save_directory"))),
                stem,
                str(self._parameter("map_frame")),
                metadata={
                    "calibration_confirmed": bool(self._parameter("calibration_confirmed")),
                    "effective_calibration_ready": bool(self._calibration_ready),
                    "projection_intrinsics": {
                        "fx": float(self._parameter("fx")),
                        "fy": float(self._parameter("fy")),
                        "cx": float(self._parameter("cx")),
                        "cy": float(self._parameter("cy")),
                        "width": int(self._parameter("expected_image_width")),
                        "height": int(self._parameter("expected_image_height")),
                    },
                    "base_to_camera_optical": list(
                        self._parameter("base_to_camera_optical")
                    ),
                    "camera_info": dict(self._camera_info_metadata),
                    "detector_status": self._detector.status,
                    "detector_model": str(self._parameter("detector_model")),
                    "detector_model_sha256": self._detector_model_sha256,
                    "source_cloud_topic": str(self._parameter("cloud_topic")),
                    "source_image_topic": str(self._parameter("image_topic")),
                    "source_odom_topic": str(self._parameter("odom_topic")),
                    "camera_time_offset_sec": float(
                        self._parameter("camera_time_offset_sec")
                    ),
                    "coordinate_convention": "REP-103 x-forward y-left z-up",
                    "time_sync_instance_id": self._time_sync_guard.instance_id,
                    "time_sync_epoch": self._time_sync_guard.epoch,
                },
            )
            response.success = True
            response.message = "Saved {} voxels atomically to {}".format(len(snapshot["points"]), bundle)
        except Exception as exc:
            response.success = False
            response.message = "Save failed: {}".format(exc)
            self.get_logger().error(response.message)
        return response

    def _reset_callback(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        with self._inference_lock:
            self._map_generation += 1
            removed = self._map.clear()
        response.success = True
        response.message = "Reset semantic map; removed {} voxels".format(removed)
        self._publish_map()
        return response

    def prepare_shutdown(self) -> None:
        self._shutting_down = True
        self._publish_timer.cancel()
        self._calibration_timer.cancel()
        self._time_sync_watchdog_timer.cancel()
        worker = self._worker_thread
        if worker is not None and worker.is_alive():
            worker.join()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Go2SemanticMappingNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.prepare_shutdown()
        executor.shutdown(timeout_sec=5.0)
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
