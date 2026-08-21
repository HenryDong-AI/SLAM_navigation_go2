# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Go2 Semantic Mapping contributors

"""ROS-independent geometry, association, voxel fusion, and persistence.

The module intentionally imports only NumPy and the Python standard library so
its calibration math and map behavior can be tested without a ROS installation.
Label value 0 is reserved for unknown geometry; detector labels must be positive.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


UNKNOWN_LABEL = 0


@dataclass(frozen=True)
class Detection:
    """One 2D detection in image pixel coordinates.

    ``mask`` is optional and, when present, must be a two-dimensional boolean or
    numeric array in the same pixel geometry as the input image.
    """

    label_id: int
    label_name: str
    confidence: float
    xyxy: Tuple[float, float, float, float]
    mask: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.label_id <= UNKNOWN_LABEL:
            raise ValueError("detection label_id must be positive; 0 is unknown")
        if len(self.xyxy) != 4:
            raise ValueError("xyxy must contain exactly four values")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("detection confidence must be in [0, 1]")


@dataclass(frozen=True)
class ProjectionResult:
    """Projection values aligned by row."""

    source_indices: np.ndarray
    uv: np.ndarray
    depth: np.ndarray
    camera_points: np.ndarray


def as_transform(matrix_values: Sequence[float]) -> np.ndarray:
    """Validate and return a finite homogeneous 4x4 transform."""

    matrix = np.asarray(matrix_values, dtype=np.float64)
    if matrix.size != 16:
        raise ValueError("extrinsic must contain 16 row-major values")
    matrix = matrix.reshape(4, 4)
    if not np.isfinite(matrix).all():
        raise ValueError("extrinsic contains non-finite values")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-6):
        raise ValueError("extrinsic last row must be [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    determinant = float(np.linalg.det(rotation))
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-4):
        raise ValueError("extrinsic rotation must be orthonormal")
    if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1.0e-4):
        raise ValueError("extrinsic rotation must be proper (determinant +1)")
    return matrix


def quaternion_xyzw_to_matrix(quaternion: Sequence[float]) -> np.ndarray:
    """Convert an ``x, y, z, w`` quaternion to a 3x3 rotation matrix."""

    q = np.asarray(quaternion, dtype=np.float64)
    if q.size != 4 or not np.isfinite(q).all():
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("quaternion norm is zero")
    x, y, z, w = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose_matrix(position_xyz: Sequence[float], quaternion_xyzw: Sequence[float]) -> np.ndarray:
    """Build ``T_parent_from_child`` from a pose."""

    position = np.asarray(position_xyz, dtype=np.float64)
    if position.size != 3 or not np.isfinite(position).all():
        raise ValueError("position must contain three finite values")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = quaternion_xyzw_to_matrix(quaternion_xyzw)
    result[:3, 3] = position
    return result


def transform_points(points_xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a homogeneous transform to an Nx3 point array."""

    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_xyz must have shape (N, 3)")
    transform = as_transform(np.asarray(transform).reshape(-1))
    if len(points) == 0:
        return np.empty((0, 3), dtype=np.float64)
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    return (transform @ homogeneous.T).T[:, :3]


def project_base_points(
    points_base: np.ndarray,
    camera_from_base: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    image_width: int,
    image_height: int,
    min_depth: float = 0.1,
    max_depth: float = 30.0,
) -> ProjectionResult:
    """Project base-frame points into an optical camera image.

    The optical convention is x-right, y-down, z-forward. ``camera_from_base``
    must map base-frame coordinates into that optical frame.
    """

    points = np.asarray(points_base, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_base must have shape (N, 3)")
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("fx and fy must be positive")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if min_depth < 0.0 or max_depth <= min_depth:
        raise ValueError("invalid depth interval")

    finite_indices = np.flatnonzero(np.isfinite(points).all(axis=1))
    if finite_indices.size == 0:
        return ProjectionResult(
            source_indices=np.empty(0, dtype=np.int64),
            uv=np.empty((0, 2), dtype=np.float64),
            depth=np.empty(0, dtype=np.float64),
            camera_points=np.empty((0, 3), dtype=np.float64),
        )

    camera_points = transform_points(points[finite_indices], camera_from_base)
    depth = camera_points[:, 2]
    depth_valid = np.isfinite(camera_points).all(axis=1)
    depth_valid &= (depth >= min_depth) & (depth <= max_depth)
    camera_points = camera_points[depth_valid]
    depth = depth[depth_valid]
    source_indices = finite_indices[depth_valid]
    if depth.size == 0:
        return ProjectionResult(
            source_indices=np.empty(0, dtype=np.int64),
            uv=np.empty((0, 2), dtype=np.float64),
            depth=np.empty(0, dtype=np.float64),
            camera_points=np.empty((0, 3), dtype=np.float64),
        )

    u = fx * camera_points[:, 0] / depth + cx
    v = fy * camera_points[:, 1] / depth + cy
    inside = (u >= 0.0) & (u < image_width) & (v >= 0.0) & (v < image_height)
    return ProjectionResult(
        source_indices=source_indices[inside].astype(np.int64, copy=False),
        uv=np.column_stack((u[inside], v[inside])),
        depth=depth[inside],
        camera_points=camera_points[inside],
    )


def sample_image_rgb(image_bgr: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Sample BGR image pixels at projected coordinates and return RGB bytes."""

    image = np.asarray(image_bgr)
    pixels = np.asarray(uv, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("image_bgr must have shape (H, W, >=3)")
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("uv must have shape (N, 2)")
    if len(pixels) == 0:
        return np.empty((0, 3), dtype=np.uint8)
    x = np.clip(np.rint(pixels[:, 0]).astype(np.int64), 0, image.shape[1] - 1)
    y = np.clip(np.rint(pixels[:, 1]).astype(np.int64), 0, image.shape[0] - 1)
    return np.asarray(image[y, x, :3][:, ::-1], dtype=np.uint8)


def associate_detections(
    uv: np.ndarray,
    depth: np.ndarray,
    detections: Sequence[Detection],
    min_points: int = 3,
    absolute_depth_gate: float = 0.25,
    mad_scale: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign projected points to detections with robust depth gating.

    Higher-confidence detections claim overlapping points first. For each box or
    mask, a median/MAD surface gate rejects foreground/background depth outliers.
    Returned arrays are aligned with ``uv``.
    """

    pixels = np.asarray(uv, dtype=np.float64)
    depths = np.asarray(depth, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2 or depths.shape != (len(pixels),):
        raise ValueError("uv and depth shapes do not align")
    if min_points <= 0:
        raise ValueError("min_points must be positive")
    if absolute_depth_gate <= 0.0 or mad_scale < 0.0:
        raise ValueError("depth gate values must be non-negative")

    labels = np.zeros(len(pixels), dtype=np.uint32)
    confidences = np.zeros(len(pixels), dtype=np.float32)
    if len(pixels) == 0:
        return labels, confidences

    px = np.floor(pixels[:, 0]).astype(np.int64)
    py = np.floor(pixels[:, 1]).astype(np.int64)
    unclaimed = np.ones(len(pixels), dtype=bool)

    for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
        x1, y1, x2, y2 = detection.xyxy
        candidate = unclaimed & (pixels[:, 0] >= x1) & (pixels[:, 0] < x2)
        candidate &= (pixels[:, 1] >= y1) & (pixels[:, 1] < y2)

        if detection.mask is not None:
            mask = np.asarray(detection.mask)
            if mask.ndim != 2:
                raise ValueError("detection mask must be two-dimensional")
            in_mask_bounds = (px >= 0) & (px < mask.shape[1]) & (py >= 0) & (py < mask.shape[0])
            mask_membership = np.zeros(len(pixels), dtype=bool)
            valid_rows = np.flatnonzero(in_mask_bounds)
            mask_membership[valid_rows] = mask[py[valid_rows], px[valid_rows]] > 0
            candidate &= mask_membership

        candidate_indices = np.flatnonzero(candidate)
        if candidate_indices.size < min_points:
            continue
        candidate_depths = depths[candidate_indices]
        finite = np.isfinite(candidate_depths)
        candidate_indices = candidate_indices[finite]
        candidate_depths = candidate_depths[finite]
        if candidate_indices.size < min_points:
            continue

        median = float(np.median(candidate_depths))
        mad = float(np.median(np.abs(candidate_depths - median)))
        robust_sigma = 1.4826 * mad
        gate = max(float(absolute_depth_gate), float(mad_scale) * robust_sigma)
        accepted = candidate_indices[np.abs(candidate_depths - median) <= gate]
        if accepted.size < min_points:
            continue

        labels[accepted] = np.uint32(detection.label_id)
        confidences[accepted] = np.float32(detection.confidence)
        unclaimed[accepted] = False

    return labels, confidences


def nearest_stamped_sample(
    samples: Sequence[Tuple[float, object]], target_stamp: float, max_delta: float
) -> Tuple[Optional[object], float]:
    """Return the nearest timestamped payload and its absolute time delta."""

    if max_delta < 0.0:
        raise ValueError("max_delta must be non-negative")
    if not samples:
        return None, math.inf
    stamp, payload = min(samples, key=lambda item: abs(float(item[0]) - target_stamp))
    delta = abs(float(stamp) - float(target_stamp))
    if delta > max_delta:
        return None, delta
    return payload, delta


@dataclass
class _VoxelAccumulator:
    position_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    position_observations: int = 0
    color_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    color_observations: int = 0
    class_votes: MutableMapping[int, float] = field(default_factory=dict)
    class_observations: MutableMapping[int, int] = field(default_factory=dict)
    semantic_observations: int = 0
    last_seen_ns: int = 0

    def label_and_confidence(self) -> Tuple[int, float]:
        if not self.class_votes:
            return UNKNOWN_LABEL, 0.0
        label = max(self.class_votes, key=self.class_votes.get)
        total_vote = max(sum(self.class_votes.values()), 1e-12)
        winning_vote = self.class_votes[label]
        winning_count = max(self.class_observations.get(label, 1), 1)
        mean_detector_confidence = winning_vote / winning_count
        consensus = winning_vote / total_vote
        return int(label), float(np.clip(mean_detector_confidence * consensus, 0.0, 1.0))


class SemanticVoxelMap:
    """Thread-safe, bounded 3D voxel map with per-class evidence fusion.

    One geometric and one class vote per observed voxel per fusion frame avoids
    allowing denser point clouds to dominate repeated temporal evidence.
    Least-recently-observed voxels are evicted when ``max_voxels`` is exceeded.
    """

    def __init__(
        self,
        voxel_size: float,
        max_voxels: int,
        default_color: Sequence[int] = (128, 128, 128),
    ) -> None:
        if voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive")
        if max_voxels <= 0:
            raise ValueError("max_voxels must be positive")
        default = np.asarray(default_color, dtype=np.int64)
        if default.size != 3 or np.any(default < 0) or np.any(default > 255):
            raise ValueError("default_color must contain three bytes")
        self.voxel_size = float(voxel_size)
        self.max_voxels = int(max_voxels)
        self.default_color = default.astype(np.uint8)
        self._voxels: "OrderedDict[Tuple[int, int, int], _VoxelAccumulator]" = OrderedDict()
        self._class_names: Dict[int, str] = {}
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._voxels)

    def clear(self) -> int:
        with self._lock:
            previous = len(self._voxels)
            self._voxels.clear()
            self._class_names.clear()
            return previous

    def update(
        self,
        points_xyz: np.ndarray,
        colors_rgb: Optional[np.ndarray] = None,
        color_valid: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        confidences: Optional[np.ndarray] = None,
        class_names: Optional[Mapping[int, str]] = None,
        observed_ns: Optional[int] = None,
    ) -> int:
        points = np.asarray(points_xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_xyz must have shape (N, 3)")
        finite = np.isfinite(points).all(axis=1)
        points = points[finite]
        original_indices = np.flatnonzero(finite)
        if len(points) == 0:
            return 0

        if colors_rgb is None:
            colors = np.zeros((len(original_indices), 3), dtype=np.uint8)
            valid_color = np.zeros(len(original_indices), dtype=bool)
        else:
            all_colors = np.asarray(colors_rgb)
            if all_colors.shape != (len(finite), 3):
                raise ValueError("colors_rgb must align with points_xyz")
            colors = np.clip(all_colors[original_indices], 0, 255).astype(np.uint8)
            if color_valid is None:
                valid_color = np.ones(len(original_indices), dtype=bool)
            else:
                all_valid = np.asarray(color_valid, dtype=bool)
                if all_valid.shape != (len(finite),):
                    raise ValueError("color_valid must align with points_xyz")
                valid_color = all_valid[original_indices]

        if labels is None:
            frame_labels = np.zeros(len(original_indices), dtype=np.uint32)
        else:
            all_labels = np.asarray(labels)
            if all_labels.shape != (len(finite),):
                raise ValueError("labels must align with points_xyz")
            frame_labels = all_labels[original_indices].astype(np.uint32)

        if confidences is None:
            frame_confidences = np.zeros(len(original_indices), dtype=np.float32)
        else:
            all_confidences = np.asarray(confidences, dtype=np.float64)
            if all_confidences.shape != (len(finite),):
                raise ValueError("confidences must align with points_xyz")
            frame_confidences = np.clip(all_confidences[original_indices], 0.0, 1.0)

        keys = np.floor(points / self.voxel_size).astype(np.int64)
        grouped: Dict[Tuple[int, int, int], List[int]] = {}
        for row, key_values in enumerate(keys):
            key = (int(key_values[0]), int(key_values[1]), int(key_values[2]))
            grouped.setdefault(key, []).append(row)

        timestamp_ns = int(time.time_ns() if observed_ns is None else observed_ns)
        with self._lock:
            if class_names:
                for label, name in class_names.items():
                    if int(label) > UNKNOWN_LABEL:
                        self._class_names[int(label)] = str(name)

            for key, row_list in grouped.items():
                rows = np.asarray(row_list, dtype=np.int64)
                accumulator = self._voxels.get(key)
                if accumulator is None:
                    accumulator = _VoxelAccumulator()
                    self._voxels[key] = accumulator

                accumulator.position_sum += np.mean(points[rows], axis=0)
                accumulator.position_observations += 1
                visible_rows = rows[valid_color[rows]]
                if visible_rows.size:
                    accumulator.color_sum += np.mean(colors[visible_rows], axis=0)
                    accumulator.color_observations += 1

                positive_labels = np.unique(frame_labels[rows][frame_labels[rows] > UNKNOWN_LABEL])
                for label_value in positive_labels:
                    label = int(label_value)
                    label_rows = rows[frame_labels[rows] == label]
                    vote = float(np.max(frame_confidences[label_rows]))
                    accumulator.class_votes[label] = accumulator.class_votes.get(label, 0.0) + vote
                    accumulator.class_observations[label] = accumulator.class_observations.get(label, 0) + 1
                    accumulator.semantic_observations += 1

                accumulator.last_seen_ns = timestamp_ns
                self._voxels.move_to_end(key)

            while len(self._voxels) > self.max_voxels:
                self._voxels.popitem(last=False)

        return len(grouped)

    def snapshot(self, include_persistence_details: bool = True) -> Dict[str, object]:
        with self._lock:
            items = list(self._voxels.items())
            count = len(items)
            points = np.empty((count, 3), dtype=np.float32)
            colors = np.empty((count, 3), dtype=np.uint8)
            labels = np.empty(count, dtype=np.uint32)
            confidences = np.empty(count, dtype=np.float32)
            semantic_observations = np.empty(count, dtype=np.uint32)
            if include_persistence_details:
                observations = np.empty(count, dtype=np.uint32)
                last_seen_ns = np.empty(count, dtype=np.int64)
                keys = np.empty((count, 3), dtype=np.int64)
                votes: List[Dict[int, float]] = []

            for index, (key, accumulator) in enumerate(items):
                points[index] = accumulator.position_sum / max(accumulator.position_observations, 1)
                if accumulator.color_observations:
                    colors[index] = np.clip(
                        np.rint(accumulator.color_sum / accumulator.color_observations), 0, 255
                    ).astype(np.uint8)
                else:
                    colors[index] = self.default_color
                label, confidence = accumulator.label_and_confidence()
                labels[index] = label
                confidences[index] = confidence
                semantic_observations[index] = accumulator.semantic_observations
                if include_persistence_details:
                    observations[index] = accumulator.position_observations
                    last_seen_ns[index] = accumulator.last_seen_ns
                    keys[index] = key
                    votes.append(dict(accumulator.class_votes))

            result = {
                "points": points,
                "colors": colors,
                "labels": labels,
                "confidences": confidences,
                "semantic_observations": semantic_observations,
                "class_names": dict(self._class_names),
                "voxel_size": self.voxel_size,
            }
            if include_persistence_details:
                result.update(
                    {
                        "observations": observations,
                        "last_seen_ns": last_seen_ns,
                        "voxel_keys": keys,
                        "votes": votes,
                    }
                )
            return result


def _fsync_file(file_object: object) -> None:
    file_object.flush()
    os.fsync(file_object.fileno())


def _write_ply(path: Path, snapshot: Mapping[str, object]) -> None:
    points = np.asarray(snapshot["points"], dtype=np.float64)
    colors = np.asarray(snapshot["colors"], dtype=np.uint8)
    labels = np.asarray(snapshot["labels"], dtype=np.uint32)
    confidences = np.asarray(snapshot["confidences"], dtype=np.float64)
    if colors.shape != (len(points), 3) or labels.shape != (len(points),):
        raise ValueError("snapshot arrays do not align")

    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("ply\n")
        stream.write("format ascii 1.0\n")
        stream.write("comment semantic voxel map; label 0 means unknown\n")
        stream.write("element vertex {}\n".format(len(points)))
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        stream.write("property uint label\nproperty float confidence\n")
        stream.write("end_header\n")
        for point, color, label, confidence in zip(points, colors, labels, confidences):
            stream.write(
                "{:.7g} {:.7g} {:.7g} {} {} {} {} {:.6g}\n".format(
                    point[0], point[1], point[2], int(color[0]), int(color[1]), int(color[2]),
                    int(label), float(confidence)
                )
            )
        _fsync_file(stream)


def _write_json_stream(
    path: Path,
    snapshot: Mapping[str, object],
    frame_id: str,
    metadata: Optional[Mapping[str, object]],
) -> None:
    """Write voxel records incrementally to avoid a second full Python object graph."""
    points = np.asarray(snapshot["points"])
    colors = np.asarray(snapshot["colors"])
    labels = np.asarray(snapshot["labels"])
    confidences = np.asarray(snapshot["confidences"])
    observations = np.asarray(snapshot["observations"])
    semantic_observations = np.asarray(snapshot["semantic_observations"])
    last_seen = np.asarray(snapshot["last_seen_ns"])
    keys = np.asarray(snapshot["voxel_keys"])
    votes = list(snapshot["votes"])

    header: Dict[str, object] = {
        "format": "go2_semantic_voxel_map",
        "format_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "frame_id": frame_id,
        "voxel_size": float(snapshot["voxel_size"]),
        "voxel_count": len(points),
        "unknown_label": UNKNOWN_LABEL,
        "class_names": {str(key): value for key, value in dict(snapshot["class_names"]).items()},
        "ply_file": "semantic_map.ply",
    }
    if metadata:
        header["metadata"] = dict(metadata)

    with path.open("w", encoding="utf-8") as stream:
        stream.write("{\n")
        for index, (key, value) in enumerate(header.items()):
            stream.write("  ")
            json.dump(str(key), stream)
            stream.write(": ")
            json.dump(value, stream, sort_keys=True)
            stream.write(",\n")
        stream.write('  "voxels": [\n')
        for index in range(len(points)):
            record = {
                "key": [int(value) for value in keys[index]],
                "position": [float(value) for value in points[index]],
                "rgb": [int(value) for value in colors[index]],
                "label": int(labels[index]),
                "confidence": float(confidences[index]),
                "observations": int(observations[index]),
                "semantic_observations": int(semantic_observations[index]),
                "last_seen_ns": int(last_seen[index]),
                "class_votes": {str(label): float(vote) for label, vote in votes[index].items()},
            }
            stream.write("    ")
            json.dump(record, stream, sort_keys=True, separators=(",", ":"))
            stream.write(",\n" if index + 1 < len(points) else "\n")
        stream.write("  ]\n}\n")
        _fsync_file(stream)


def save_snapshot_bundle_atomic(
    snapshot: Mapping[str, object],
    output_directory: str,
    stem: str,
    frame_id: str,
    metadata: Optional[Mapping[str, object]] = None,
) -> Path:
    """Atomically publish a directory containing matching PLY and JSON files.

    Both files are written and fsynced in a hidden temporary directory. A single
    directory rename then makes the complete pair visible, preventing readers
    from observing a mixed or half-written bundle.
    """

    if not stem or Path(stem).name != stem:
        raise ValueError("stem must be a non-empty filename component")
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / stem
    if final_path.exists():
        final_path = output / "{}-{}".format(stem, uuid.uuid4().hex[:8])

    temporary_path = Path(tempfile.mkdtemp(prefix=".{}-tmp-".format(stem), dir=str(output)))
    try:
        _write_ply(temporary_path / "semantic_map.ply", snapshot)
        _write_json_stream(
            temporary_path / "semantic_map.json", snapshot, frame_id, metadata
        )

        directory_fd = os.open(str(temporary_path), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(str(temporary_path), str(final_path))
        parent_fd = os.open(str(output), os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception:
        shutil.rmtree(str(temporary_path), ignore_errors=True)
        raise
    return final_path
