#!/usr/bin/env python3
"""Estimate the D435i mount transform from overlapping depth and LiDAR data."""

import math
import time

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image, PointCloud2

from go2_mapping.pointcloud import PointCloudFormatError, read_xyz

from .geometry import (
    decode_depth_image,
    depth_to_camera_points,
    rigid_transform,
)


_DEFAULT_BASE_FROM_OPTICAL = [
    0.0, 0.0, 1.0, 0.0,
    -1.0, 0.0, 0.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
]


def _stamp_ns(message):
    return (
        int(message.header.stamp.sec) * 1_000_000_000
        + int(message.header.stamp.nanosec)
    )


def voxel_downsample(points, voxel_size):
    """Keep one centroid per occupied voxel."""
    points = np.asarray(points, dtype=np.float64)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] == 0:
        return points.reshape(0, 3)
    keys = np.floor(points / float(voxel_size)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    result = np.column_stack(
        [
            np.bincount(inverse, weights=points[:, axis]) / counts
            for axis in range(3)
        ]
    )
    return result


def best_fit_transform(source, target):
    """Return the rigid transform mapping paired source points to target."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    invalid_shape = source.ndim != 2 or source.shape[1] != 3
    if source.shape != target.shape or invalid_shape:
        raise ValueError("paired point arrays must both have shape (N, 3)")
    if source.shape[0] < 3:
        raise ValueError("at least three point pairs are required")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    covariance = (source - source_mean).T @ (target - target_mean)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_mean - rotation @ source_mean
    return transform


def transform_points_fast(points, transform):
    return points @ transform[:3, :3].T + transform[:3, 3]


def rotation_angle_degrees(rotation):
    cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


class NearestIndex:
    """Small OpenCV FLANN wrapper returning Euclidean nearest distances."""

    def __init__(self, points):
        self.points = np.ascontiguousarray(points, dtype=np.float32)
        self.index = cv2.flann_Index(
            self.points, {"algorithm": 1, "trees": 8}
        )

    def query(self, points):
        query = np.ascontiguousarray(points, dtype=np.float32)
        indices, squared_distances = self.index.knnSearch(
            query, 1, params={"checks": 64}
        )
        return (
            np.sqrt(np.maximum(squared_distances[:, 0], 0.0)),
            indices[:, 0],
        )


def iterative_closest_point(source, target):
    """Register source to target with trimmed, coarse-to-fine point ICP."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape[0] < 100 or target.shape[0] < 100:
        raise ValueError("not enough points for calibration")
    tree = NearestIndex(target)
    total = np.eye(4, dtype=np.float64)
    previous_rmse = float("inf")
    stages = ((0.50, 30), (0.30, 30), (0.18, 35), (0.10, 40))
    for max_distance, iterations in stages:
        for _ in range(iterations):
            moved = transform_points_fast(source, total)
            distances, indices = tree.query(moved)
            valid = np.isfinite(distances) & (distances < max_distance)
            if int(valid.sum()) < 100:
                raise ValueError(
                    "too little LiDAR/depth overlap at {:.2f} m".format(
                        max_distance
                    )
                )
            valid_indices = np.flatnonzero(valid)
            valid_distances = distances[valid]
            # Reject the longest 20% of otherwise valid correspondences.
            cutoff = np.percentile(valid_distances, 80.0)
            valid_indices = valid_indices[valid_distances <= cutoff]
            increment = best_fit_transform(
                moved[valid_indices], target[indices[valid_indices]]
            )
            total = increment @ total
            rmse = float(
                np.sqrt(np.mean(np.square(distances[valid_indices])))
            )
            step_translation = float(np.linalg.norm(increment[:3, 3]))
            step_rotation = rotation_angle_degrees(increment[:3, :3])
            if (
                abs(previous_rmse - rmse) < 1.0e-5
                and step_translation < 1.0e-4
                and step_rotation < 0.01
            ):
                break
            previous_rmse = rmse

    moved = transform_points_fast(source, total)
    distances, _ = tree.query(moved)
    inliers = distances < 0.10
    overlap = float(np.mean(inliers))
    rmse = (
        float(np.sqrt(np.mean(np.square(distances[inliers]))))
        if inliers.any()
        else float("inf")
    )
    return total, rmse, overlap, int(inliers.sum())


def crop_lidar_to_view(
    points_base, base_from_optical, intrinsics, margin=0.15
):
    """Keep LiDAR points inside an expanded D435i color-camera frustum."""
    optical_from_base = np.linalg.inv(base_from_optical)
    points_optical = transform_points_fast(points_base, optical_from_base)
    z = points_optical[:, 2]
    valid = np.isfinite(points_optical).all(axis=1) & (z > 0.20) & (z < 5.0)
    safe_z = np.where(valid, z, 1.0)
    u = intrinsics["fx"] * points_optical[:, 0] / safe_z + intrinsics["cx"]
    v = intrinsics["fy"] * points_optical[:, 1] / safe_z + intrinsics["cy"]
    x_margin = intrinsics["width"] * margin
    y_margin = intrinsics["height"] * margin
    valid &= (u >= -x_margin) & (u < intrinsics["width"] + x_margin)
    valid &= (v >= -y_margin) & (v < intrinsics["height"] + y_margin)
    return points_base[valid]


class ExtrinsicCalibrator(Node):
    """Collect synchronized stationary depth/LiDAR observations."""

    def __init__(self):
        super().__init__("go2_depth_extrinsic_calibrator")
        self.declare_parameter(
            "base_from_camera_optical", _DEFAULT_BASE_FROM_OPTICAL
        )
        self.declare_parameter("sample_count", 15)
        self.declare_parameter("max_sync_delta_sec", 0.20)
        self.declare_parameter("voxel_size", 0.05)
        self.declare_parameter(
            "depth_topic", "/go2/depth_camera/aligned_depth/image_raw"
        )
        self.declare_parameter(
            "camera_info_topic", "/go2/depth_camera/aligned_depth/camera_info"
        )
        self.declare_parameter(
            "lidar_topic", "/go2/lidar/cloud_calibration"
        )

        self.nominal = rigid_transform(
            self.get_parameter("base_from_camera_optical").value
        )
        self.sample_count = int(self.get_parameter("sample_count").value)
        self.max_sync_delta_ns = int(
            float(self.get_parameter("max_sync_delta_sec").value) * 1.0e9
        )
        self.voxel_size = float(self.get_parameter("voxel_size").value)
        self.info = None
        self.latest_lidar = None
        self.camera_samples = []
        self.lidar_samples = []
        self.complete = False
        self.last_depth_stamp = 0

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            self._on_info,
            sensor_qos,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("lidar_topic").value),
            self._on_lidar,
            sensor_qos,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("depth_topic").value),
            self._on_depth,
            sensor_qos,
        )
        self.get_logger().info(
            "keep the robot stationary; collecting %d depth/LiDAR pairs"
            % self.sample_count
        )

    def _on_info(self, message):
        if len(message.k) < 6:
            return
        self.info = {
            "fx": float(message.k[0]),
            "fy": float(message.k[4]),
            "cx": float(message.k[2]),
            "cy": float(message.k[5]),
            "width": int(message.width),
            "height": int(message.height),
        }

    def _on_lidar(self, message):
        try:
            points = read_xyz(message, max_points=1_000_000)
        except (PointCloudFormatError, ValueError) as error:
            self.get_logger().warning("invalid LiDAR cloud: %s" % error)
            return
        points = points[np.isfinite(points).all(axis=1)]
        if points.shape[0] >= 100:
            self.latest_lidar = (_stamp_ns(message), points)

    def _on_depth(self, message):
        if self.complete or self.info is None or self.latest_lidar is None:
            return
        stamp = _stamp_ns(message)
        if stamp <= self.last_depth_stamp:
            return
        lidar_stamp, lidar_points = self.latest_lidar
        if abs(stamp - lidar_stamp) > self.max_sync_delta_ns:
            return
        try:
            depth = decode_depth_image(message)
            camera_points = depth_to_camera_points(
                depth,
                self.info["fx"],
                self.info["fy"],
                self.info["cx"],
                self.info["cy"],
                pixel_stride=4,
                min_depth=0.25,
                max_depth=5.0,
                max_points=25_000,
            )
        except ValueError as error:
            self.get_logger().warning("invalid depth image: %s" % error)
            return
        if camera_points.shape[0] < 200:
            return
        lidar_in_view = crop_lidar_to_view(
            lidar_points, self.nominal, self.info, margin=0.25
        )
        if lidar_in_view.shape[0] < 100:
            return
        self.camera_samples.append(camera_points)
        self.lidar_samples.append(lidar_in_view)
        self.last_depth_stamp = stamp
        self.get_logger().info(
            "calibration sample %d/%d: depth=%d lidar=%d sync=%.3f s"
            % (
                len(self.camera_samples),
                self.sample_count,
                camera_points.shape[0],
                lidar_in_view.shape[0],
                abs(stamp - lidar_stamp) * 1.0e-9,
            )
        )
        if len(self.camera_samples) >= self.sample_count:
            self.complete = True

    def estimate(self):
        camera = voxel_downsample(
            np.vstack(self.camera_samples), self.voxel_size
        )
        lidar = voxel_downsample(
            np.vstack(self.lidar_samples), self.voxel_size
        )
        source_base = transform_points_fast(camera, self.nominal)
        correction, rmse, overlap, inliers = iterative_closest_point(
            source_base, lidar
        )
        calibrated = correction @ self.nominal
        correction_translation = float(np.linalg.norm(correction[:3, 3]))
        correction_rotation = rotation_angle_degrees(correction[:3, :3])
        credible = (
            inliers >= 300
            and overlap >= 0.20
            and rmse <= 0.075
            and correction_translation <= 0.60
            and correction_rotation <= 25.0
        )
        return {
            "camera_points": camera.shape[0],
            "lidar_points": lidar.shape[0],
            "rmse": rmse,
            "overlap": overlap,
            "inliers": inliers,
            "correction_translation": correction_translation,
            "correction_rotation": correction_rotation,
            "credible": credible,
            "transform": calibrated,
        }


def main(args=None):
    rclpy.init(args=args)
    node = ExtrinsicCalibrator()
    deadline = time.monotonic() + 45.0
    try:
        while rclpy.ok() and not node.complete and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.20)
        if not node.complete:
            node.get_logger().error(
                "calibration timed out; keep both sensors publishing "
                "and the robot still"
            )
            return
        result = node.estimate()
        matrix = result["transform"].reshape(-1)
        print("CALIBRATION_RESULT")
        print(
            "quality: rmse={:.4f}m overlap={:.1%} inliers={} "
            "correction_translation={:.3f}m "
            "correction_rotation={:.2f}deg".format(
                result["rmse"],
                result["overlap"],
                result["inliers"],
                result["correction_translation"],
                result["correction_rotation"],
            )
        )
        print("credible: {}".format(str(result["credible"]).lower()))
        print("base_from_camera_optical:")
        for row in matrix.reshape(4, 4):
            print("  [{: .8f}, {: .8f}, {: .8f}, {: .8f}]".format(*row))
    except (ValueError, np.linalg.LinAlgError) as error:
        node.get_logger().error("calibration failed: %s" % error)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
