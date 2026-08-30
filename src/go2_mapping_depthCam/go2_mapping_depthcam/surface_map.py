"""Confidence-weighted RGB-D surface fusion for the depth-camera backend."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from go2_mapping.voxel_map import VoxelAccumulator


VoxelKey = Tuple[int, int, int]


@dataclass
class _SurfaceVoxel:
    centroid: np.ndarray
    observations: int
    last_seen_ns: int
    color: Optional[np.ndarray]
    color_observations: int
    variance: float


class RgbdSurfaceAccumulator:
    """Fuse one equal-weight surface observation per voxel and camera frame.

    Raw pixel-count weighting makes close views dominate the map and preserves
    single-frame depth speckle. This accumulator first reduces every frame to
    one RGB surface sample per voxel, then applies bounded temporal confidence.
    Only repeatedly observed, geometrically stable voxels are published/saved.
    """

    def __init__(
        self,
        voxel_size,
        min_point_range,
        max_point_range,
        min_relative_z,
        max_relative_z,
        max_voxels,
        retention_radius=0.0,
        max_observation_weight=32,
        min_observations=2,
        max_surface_variance=0.0016,
    ):
        if max_observation_weight <= 0:
            raise ValueError("max_observation_weight must be positive")
        if min_observations <= 0:
            raise ValueError("min_observations must be positive")
        if max_surface_variance < 0.0:
            raise ValueError("max_surface_variance must not be negative")
        self.voxel_size = float(voxel_size)
        self.max_voxels = int(max_voxels)
        self.retention_radius = float(retention_radius)
        self.max_observation_weight = int(max_observation_weight)
        self.min_observations = int(min_observations)
        self.max_surface_variance = float(max_surface_variance)
        self._filter = VoxelAccumulator(
            voxel_size=voxel_size,
            min_point_range=min_point_range,
            max_point_range=max_point_range,
            min_relative_z=min_relative_z,
            max_relative_z=max_relative_z,
            max_voxels=max_voxels,
            retention_radius=retention_radius,
        )
        self._voxels: Dict[VoxelKey, _SurfaceVoxel] = {}
        self._fusion_count = 0
        self._surface_count = 0
        self._colorized_count = 0

    def __len__(self):
        return len(self._voxels)

    def clear(self):
        self._voxels.clear()
        self._fusion_count = 0
        self._surface_count = 0
        self._colorized_count = 0

    def filter_mask(self, points, robot_position):
        return self._filter.filter_mask(points, robot_position)

    def _stable(self, record):
        return (
            record.observations >= self.min_observations
            and record.variance <= self.max_surface_variance
        )

    def surface_count(self):
        return self._surface_count

    def colorized_count(self):
        return self._colorized_count

    def _remove(self, key):
        record = self._voxels.pop(key)
        if self._stable(record):
            self._surface_count -= 1
            if record.color_observations > 0:
                self._colorized_count -= 1

    def fuse_filtered(self, points, robot_position, stamp_ns, colors_rgb=None):
        array = np.asarray(points, dtype=np.float64)
        if array.ndim != 2 or array.shape[1:] != (3,):
            raise ValueError("points must have shape (N, 3)")
        colors = None
        if colors_rgb is not None:
            colors = np.asarray(colors_rgb, dtype=np.float64)
            if colors.shape != array.shape:
                raise ValueError("colors_rgb must have shape (N, 3)")
            if (
                not np.isfinite(colors).all()
                or (colors < 0.0).any()
                or (colors > 255.0).any()
            ):
                raise ValueError("RGB channels must be finite and in [0, 255]")
        robot = self._filter._as_robot_position(robot_position)
        stamp_ns = int(stamp_ns)
        self._fusion_count += 1

        if array.shape[0]:
            keys = np.floor(array / self.voxel_size).astype(np.int64)
            unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
            pixel_counts = np.bincount(inverse).astype(np.float64)
            point_sums = np.zeros((unique_keys.shape[0], 3), dtype=np.float64)
            np.add.at(point_sums, inverse, array)
            frame_centroids = point_sums / pixel_counts[:, None]
            frame_colors = None
            if colors is not None:
                color_sums = np.zeros_like(point_sums)
                np.add.at(color_sums, inverse, colors)
                frame_colors = color_sums / pixel_counts[:, None]

            for index, key_array in enumerate(unique_keys):
                key = tuple(int(value) for value in key_array)
                observation = frame_centroids[index]
                color = (
                    frame_colors[index] if frame_colors is not None else None
                )
                existing = self._voxels.get(key)
                if existing is None:
                    record = _SurfaceVoxel(
                        centroid=observation.copy(),
                        observations=1,
                        last_seen_ns=stamp_ns,
                        color=color.copy() if color is not None else None,
                        color_observations=1 if color is not None else 0,
                        variance=0.0,
                    )
                    self._voxels[key] = record
                    if self._stable(record):
                        self._surface_count += 1
                        if record.color_observations > 0:
                            self._colorized_count += 1
                    continue

                was_stable = self._stable(existing)
                was_colorized = (
                    was_stable and existing.color_observations > 0
                )
                weight = min(
                    existing.observations, self.max_observation_weight
                )
                difference = observation - existing.centroid
                new_weight = float(weight + 1)
                existing.centroid = (
                    existing.centroid * weight + observation
                ) / new_weight
                squared_error = float(np.dot(difference, difference))
                existing.variance = (
                    existing.variance * weight + squared_error
                ) / new_weight
                existing.observations = min(
                    existing.observations + 1, 2147483647
                )
                existing.last_seen_ns = max(
                    existing.last_seen_ns, stamp_ns
                )
                if color is not None:
                    color_weight = min(
                        existing.color_observations,
                        self.max_observation_weight,
                    )
                    if existing.color is None or color_weight == 0:
                        existing.color = color.copy()
                    else:
                        existing.color = (
                            existing.color * color_weight + color
                        ) / float(color_weight + 1)
                    existing.color_observations = min(
                        existing.color_observations + 1, 2147483647
                    )
                is_stable = self._stable(existing)
                is_colorized = (
                    is_stable and existing.color_observations > 0
                )
                self._surface_count += int(is_stable) - int(was_stable)
                self._colorized_count += (
                    int(is_colorized) - int(was_colorized)
                )

        if self.retention_radius > 0.0 and self._fusion_count % 10 == 0:
            radius_squared = self.retention_radius * self.retention_radius
            stale = [
                key
                for key, record in self._voxels.items()
                if float(np.sum((record.centroid - robot)[:2] ** 2))
                > radius_squared
            ]
            for key in stale:
                self._remove(key)
        self._enforce_limit()

    def _enforce_limit(self):
        if len(self._voxels) <= self.max_voxels:
            return
        target = max(1, int(self.max_voxels * 0.95))
        remove_count = len(self._voxels) - target
        oldest = sorted(
            self._voxels.items(), key=lambda item: item[1].last_seen_ns
        )[:remove_count]
        for key, _ in oldest:
            self._remove(key)

    def _ordered_stable(self):
        return [
            (key, record)
            for key, record in self._voxels.items()
            if self._stable(record)
        ]

    def points_with_colors(self, default_color=(180, 180, 180)):
        ordered = self._ordered_stable()
        if not ordered:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.uint8),
            )
        default = np.asarray(default_color, dtype=np.float64)
        points = np.vstack([record.centroid for _, record in ordered])
        colors = np.vstack(
            [
                record.color
                if record.color is not None
                else default
                for _, record in ordered
            ]
        )
        return points.copy(), np.rint(colors).clip(0, 255).astype(np.uint8)

    def state(self):
        ordered = self._ordered_stable()
        if not ordered:
            return {
                "voxel_keys": np.empty((0, 3), dtype=np.int64),
                "voxel_centroids": np.empty((0, 3), dtype=np.float64),
                "voxel_counts": np.empty((0,), dtype=np.int64),
                "voxel_last_seen_ns": np.empty((0,), dtype=np.int64),
            }
        return {
            "voxel_keys": np.asarray(
                [key for key, _ in ordered], dtype=np.int64
            ),
            "voxel_centroids": np.vstack(
                [record.centroid for _, record in ordered]
            ).astype(np.float64, copy=False),
            "voxel_counts": np.asarray(
                [record.observations for _, record in ordered], dtype=np.int64
            ),
            "voxel_last_seen_ns": np.asarray(
                [record.last_seen_ns for _, record in ordered], dtype=np.int64
            ),
            "voxel_colors": np.vstack(
                [
                    record.color
                    if record.color is not None
                    else np.zeros(3, dtype=np.float64)
                    for _, record in ordered
                ]
            ).astype(np.float64, copy=False),
            "voxel_color_counts": np.asarray(
                [record.color_observations for _, record in ordered],
                dtype=np.int64,
            ),
            "voxel_variances": np.asarray(
                [record.variance for _, record in ordered], dtype=np.float64
            ),
        }

    def restore(self, state):
        keys = np.asarray(state["voxel_keys"], dtype=np.int64)
        centroids = np.asarray(state["voxel_centroids"], dtype=np.float64)
        counts = np.asarray(state["voxel_counts"], dtype=np.int64)
        seen = np.asarray(state["voxel_last_seen_ns"], dtype=np.int64)
        colors = np.asarray(
            state.get("voxel_colors", np.zeros_like(centroids)),
            dtype=np.float64,
        )
        color_counts = np.asarray(
            state.get("voxel_color_counts", np.zeros_like(counts)),
            dtype=np.int64,
        )
        variances = np.asarray(
            state.get("voxel_variances", np.zeros_like(counts, dtype=float)),
            dtype=np.float64,
        )
        size = keys.shape[0] if keys.ndim == 2 else -1
        if (
            keys.shape != (size, 3)
            or centroids.shape != (size, 3)
            or counts.shape != (size,)
            or seen.shape != (size,)
            or colors.shape != (size, 3)
            or color_counts.shape != (size,)
            or variances.shape != (size,)
        ):
            raise ValueError("invalid RGB-D surface state array shapes")
        if (
            not np.isfinite(centroids).all()
            or not np.isfinite(colors).all()
            or not np.isfinite(variances).all()
            or (counts <= 0).any()
            or (color_counts < 0).any()
            or (variances < 0.0).any()
        ):
            raise ValueError("invalid RGB-D surface state values")
        expected_keys = np.floor(centroids / self.voxel_size).astype(np.int64)
        if not np.array_equal(keys, expected_keys):
            raise ValueError("surface state does not match configured voxel_size")
        order = np.argsort(seen)[-self.max_voxels:]
        restored = {}
        for index in order:
            key = tuple(int(value) for value in keys[index])
            restored[key] = _SurfaceVoxel(
                centroid=centroids[index].copy(),
                observations=min(int(counts[index]), 2147483647),
                last_seen_ns=int(seen[index]),
                color=(
                    colors[index].copy()
                    if color_counts[index] > 0 else None
                ),
                color_observations=min(
                    int(color_counts[index]), 2147483647
                ),
                variance=float(variances[index]),
            )
        self._voxels = restored
        self._fusion_count = 0
        self._surface_count = sum(
            1 for record in restored.values() if self._stable(record)
        )
        self._colorized_count = sum(
            1
            for record in restored.values()
            if self._stable(record) and record.color_observations > 0
        )
