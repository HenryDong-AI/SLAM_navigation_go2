"""Lightweight planar scan registration for the Go2 RGB-D mapper."""

import math

import cv2
import numpy as np


def transform_points_fast(points, transform):
    """Apply a known-valid homogeneous transform without revalidating it."""

    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def voxel_downsample(points, voxel_size, max_points=0):
    """Return voxel centroids with an optional deterministic point cap."""

    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    if not math.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("voxel_size must be finite and positive")
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)
    keys = np.floor(points / float(voxel_size)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    counts = np.bincount(inverse)
    result = np.column_stack(
        [
            np.bincount(inverse, weights=points[:, axis]) / counts
            for axis in range(3)
        ]
    )
    limit = int(max_points)
    if limit > 0 and result.shape[0] > limit:
        indices = np.linspace(
            0, result.shape[0] - 1, limit, dtype=np.int64
        )
        result = result[indices]
    return result


def best_fit_planar_transform(source, target):
    """Estimate the yaw and XY translation mapping paired source to target."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    invalid_shape = source.ndim != 2 or source.shape[1] != 3
    if invalid_shape or source.shape != target.shape:
        raise ValueError("paired point arrays must both have shape (N, 3)")
    if source.shape[0] < 3:
        raise ValueError("at least three point pairs are required")

    source_xy = source[:, :2]
    target_xy = target[:, :2]
    source_mean = source_xy.mean(axis=0)
    target_mean = target_xy.mean(axis=0)
    covariance = (source_xy - source_mean).T @ (target_xy - target_mean)
    u, _, vt = np.linalg.svd(covariance)
    rotation_xy = vt.T @ u.T
    if np.linalg.det(rotation_xy) < 0.0:
        vt[-1, :] *= -1.0
        rotation_xy = vt.T @ u.T

    transform = np.eye(4, dtype=np.float64)
    transform[:2, :2] = rotation_xy
    transform[:2, 3] = target_mean - rotation_xy @ source_mean
    return transform


def planar_rotation_degrees(transform):
    """Return the absolute yaw magnitude of a planar transform."""

    return abs(
        math.degrees(
            math.atan2(float(transform[1, 0]), float(transform[0, 0]))
        )
    )


def scale_planar_transform(transform, gain):
    """Scale a planar correction in SE(2) to reduce frame-to-frame jitter."""

    gain = float(gain)
    if not math.isfinite(gain) or gain <= 0.0 or gain > 1.0:
        raise ValueError("registration gain must be in (0, 1]")
    yaw = math.atan2(float(transform[1, 0]), float(transform[0, 0])) * gain
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    result = np.eye(4, dtype=np.float64)
    result[0, 0] = cosine
    result[0, 1] = -sine
    result[1, 0] = sine
    result[1, 1] = cosine
    result[:2, 3] = transform[:2, 3] * gain
    return result


class NearestIndex:
    """OpenCV FLANN nearest-neighbour index for three-dimensional points."""

    def __init__(self, points):
        self.points = np.ascontiguousarray(points, dtype=np.float32)
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError("target points must have shape (N, 3)")
        if self.points.shape[0] < 3:
            raise ValueError("target must contain at least three points")
        self.index = cv2.flann_Index(
            self.points, {"algorithm": 1, "trees": 4}
        )

    def query(self, points):
        query = np.ascontiguousarray(points, dtype=np.float32)
        indices, squared_distances = self.index.knnSearch(
            query, 1, params={"checks": 32}
        )
        distances = np.sqrt(np.maximum(squared_distances[:, 0], 0.0))
        return distances, indices[:, 0]

    def query_k(self, points, neighbours):
        query = np.ascontiguousarray(points, dtype=np.float32)
        count = min(max(1, int(neighbours)), self.points.shape[0])
        indices, squared_distances = self.index.knnSearch(
            query, count, params={"checks": 64}
        )
        distances = np.sqrt(np.maximum(squared_distances, 0.0))
        return distances, indices


def register_planar_scan(
    source,
    target,
    max_correspondence,
    max_iterations,
    trim_fraction,
    min_correspondences,
):
    """Align one predicted world-frame scan to a recent local RGB-D submap."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source points must have shape (N, 3)")
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError("target points must have shape (N, 3)")
    minimum = max(3, int(min_correspondences))
    if source.shape[0] < minimum or target.shape[0] < minimum:
        raise ValueError("not enough points for RGB-D registration")
    if not math.isfinite(max_correspondence) or max_correspondence <= 0.0:
        raise ValueError("max_correspondence must be finite and positive")
    if not 0.25 <= float(trim_fraction) <= 1.0:
        raise ValueError("trim_fraction must be in [0.25, 1]")

    tree = NearestIndex(target)
    total = np.eye(4, dtype=np.float64)
    previous_rmse = float("inf")
    for _ in range(max(1, int(max_iterations))):
        moved = transform_points_fast(source, total)
        distances, indices = tree.query(moved)
        valid = np.isfinite(distances) & (distances < max_correspondence)
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size < minimum:
            raise ValueError("too little scan-to-submap overlap")
        valid_distances = distances[valid_indices]
        if trim_fraction < 1.0:
            cutoff = np.percentile(valid_distances, trim_fraction * 100.0)
            valid_indices = valid_indices[valid_distances <= cutoff]
        if valid_indices.size < minimum:
            raise ValueError("too few trimmed RGB-D correspondences")

        increment = best_fit_planar_transform(
            moved[valid_indices], target[indices[valid_indices]]
        )
        total = increment @ total
        rmse = float(np.sqrt(np.mean(np.square(distances[valid_indices]))))
        step_translation = float(np.linalg.norm(increment[:2, 3]))
        step_rotation = planar_rotation_degrees(increment)
        if (
            abs(previous_rmse - rmse) < 1.0e-5
            and step_translation < 1.0e-4
            and step_rotation < 0.01
        ):
            break
        previous_rmse = rmse

    moved = transform_points_fast(source, total)
    distances, _ = tree.query(moved)
    inliers = np.isfinite(distances) & (distances < max_correspondence)
    inlier_count = int(inliers.sum())
    overlap = float(inlier_count) / float(source.shape[0])
    rmse = (
        float(np.sqrt(np.mean(np.square(distances[inliers]))))
        if inlier_count
        else float("inf")
    )
    return total, rmse, overlap, inlier_count


def best_fit_rigid_transform(source, target):
    """Estimate the full SE(3) transform mapping paired source to target."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    invalid = source.ndim != 2 or source.shape[1:] != (3,)
    if invalid or source.shape != target.shape or source.shape[0] < 3:
        raise ValueError("paired point arrays must both have shape (N, 3)")
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


def rotation_degrees(transform):
    """Return the magnitude of a full 3D rotation."""

    rotation = np.asarray(transform, dtype=np.float64)[:3, :3]
    cosine = (float(np.trace(rotation)) - 1.0) * 0.5
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


def scale_rigid_transform(transform, gain):
    """Scale an SE(3) correction without reducing it to a planar transform."""

    gain = float(gain)
    if not math.isfinite(gain) or gain <= 0.0 or gain > 1.0:
        raise ValueError("registration gain must be in (0, 1]")
    matrix = np.asarray(transform, dtype=np.float64)
    rotation_vector, _ = cv2.Rodrigues(matrix[:3, :3])
    rotation, _ = cv2.Rodrigues(rotation_vector * gain)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = matrix[:3, 3] * gain
    return result


def estimate_normals(points, neighbours=12, max_curvature=0.20):
    """Estimate local PCA normals and reject non-surface neighbourhoods."""

    cloud = np.asarray(points, dtype=np.float64)
    if cloud.ndim != 2 or cloud.shape[1:] != (3,) or cloud.shape[0] < 4:
        raise ValueError("normal input must have shape (N, 3) with N >= 4")
    count = min(max(4, int(neighbours)), cloud.shape[0])
    tree = NearestIndex(cloud)
    _, indices = tree.query_k(cloud, count)
    neighbourhoods = cloud[indices]
    centered = neighbourhoods - neighbourhoods.mean(axis=1, keepdims=True)
    covariance = np.einsum(
        "nki,nkj->nij", centered, centered
    ) / float(max(1, count - 1))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]
    spread = np.maximum(eigenvalues.sum(axis=1), 1.0e-12)
    curvature = np.maximum(eigenvalues[:, 0], 0.0) / spread
    valid = (
        np.isfinite(normals).all(axis=1)
        & np.isfinite(curvature)
        & (spread > 1.0e-8)
        & (curvature <= float(max_curvature))
    )
    return normals, valid


def _increment_from_twist(twist):
    vector = np.asarray(twist, dtype=np.float64).reshape(6)
    rotation, _ = cv2.Rodrigues(vector[:3].reshape(3, 1))
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = vector[3:]
    return result


def register_rigid_scan(
    source,
    target,
    max_correspondence,
    max_iterations,
    trim_fraction,
    min_correspondences,
    normal_neighbours=12,
    huber_delta=0.03,
    damping=1.0e-5,
):
    """Align a scan to a local submap with robust full-SE(3) point-to-plane ICP."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.ndim != 2 or source.shape[1:] != (3,):
        raise ValueError("source points must have shape (N, 3)")
    if target.ndim != 2 or target.shape[1:] != (3,):
        raise ValueError("target points must have shape (N, 3)")
    minimum = max(6, int(min_correspondences))
    if source.shape[0] < minimum or target.shape[0] < minimum:
        raise ValueError("not enough points for RGB-D registration")
    if not math.isfinite(max_correspondence) or max_correspondence <= 0.0:
        raise ValueError("max_correspondence must be finite and positive")
    if not 0.25 <= float(trim_fraction) <= 1.0:
        raise ValueError("trim_fraction must be in [0.25, 1]")
    if not math.isfinite(huber_delta) or huber_delta <= 0.0:
        raise ValueError("huber_delta must be finite and positive")
    if not math.isfinite(damping) or damping < 0.0:
        raise ValueError("damping must be finite and non-negative")

    tree = NearestIndex(target)
    normals, normal_valid = estimate_normals(
        target, neighbours=normal_neighbours
    )
    total = np.eye(4, dtype=np.float64)
    previous_rmse = float("inf")
    for _ in range(max(1, int(max_iterations))):
        moved = transform_points_fast(source, total)
        distances, indices = tree.query(moved)
        valid = (
            np.isfinite(distances)
            & (distances < max_correspondence)
            & normal_valid[indices]
        )
        selected = np.flatnonzero(valid)
        if selected.size < minimum:
            raise ValueError("too little scan-to-submap surface overlap")
        selected_distances = distances[selected]
        if trim_fraction < 1.0:
            cutoff = np.percentile(
                selected_distances, float(trim_fraction) * 100.0
            )
            selected = selected[selected_distances <= cutoff]
        if selected.size < minimum:
            raise ValueError("too few trimmed RGB-D surface correspondences")

        moved_selected = moved[selected]
        target_selected = target[indices[selected]]
        normals_selected = normals[indices[selected]]
        residuals = np.einsum(
            "ij,ij->i", normals_selected, moved_selected - target_selected
        )
        absolute = np.abs(residuals)
        weights = np.ones_like(residuals)
        robust = absolute > huber_delta
        weights[robust] = huber_delta / absolute[robust]
        jacobian = np.column_stack(
            (
                np.cross(moved_selected, normals_selected),
                normals_selected,
            )
        )
        square_root_weight = np.sqrt(weights)
        weighted_jacobian = jacobian * square_root_weight[:, None]
        weighted_residual = residuals * square_root_weight
        hessian = weighted_jacobian.T @ weighted_jacobian
        gradient = weighted_jacobian.T @ weighted_residual
        diagonal = np.maximum(np.diag(hessian), 1.0)
        hessian += float(damping) * np.diag(diagonal)
        try:
            twist = np.linalg.solve(hessian, -gradient)
        except np.linalg.LinAlgError:
            increment = best_fit_rigid_transform(
                moved_selected, target_selected
            )
        else:
            rotation_norm = float(np.linalg.norm(twist[:3]))
            maximum_rotation = math.radians(5.0)
            if rotation_norm > maximum_rotation:
                twist[:3] *= maximum_rotation / rotation_norm
            translation_norm = float(np.linalg.norm(twist[3:]))
            maximum_translation = 0.5 * float(max_correspondence)
            if translation_norm > maximum_translation:
                twist[3:] *= maximum_translation / translation_norm
            increment = _increment_from_twist(twist)
        total = increment @ total
        rmse = float(np.sqrt(np.mean(np.square(residuals))))
        step_translation = float(np.linalg.norm(increment[:3, 3]))
        step_rotation = rotation_degrees(increment)
        if (
            abs(previous_rmse - rmse) < 1.0e-5
            and step_translation < 1.0e-4
            and step_rotation < 0.01
        ):
            break
        previous_rmse = rmse

    moved = transform_points_fast(source, total)
    distances, indices = tree.query(moved)
    inliers = (
        np.isfinite(distances)
        & (distances < max_correspondence)
        & normal_valid[indices]
    )
    inlier_count = int(inliers.sum())
    overlap = float(inlier_count) / float(source.shape[0])
    rmse = (
        float(np.sqrt(np.mean(np.square(distances[inliers]))))
        if inlier_count
        else float("inf")
    )
    return total, rmse, overlap, inlier_count
