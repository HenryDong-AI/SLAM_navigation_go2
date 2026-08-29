"""Bounded voxel fusion for a registered three-dimensional point stream."""

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np


VoxelKey = Tuple[int, int, int]


@dataclass
class _Voxel:
    centroid: np.ndarray
    count: int
    last_seen_ns: int
    color: Optional[np.ndarray] = None
    color_count: int = 0


class VoxelAccumulator:
    """Fuse points into voxel centroids while enforcing explicit memory limits."""

    def __init__(
        self,
        voxel_size: float,
        min_point_range: float,
        max_point_range: float,
        min_relative_z: float,
        max_relative_z: float,
        max_voxels: int,
        retention_radius: float = 0.0,
        coordinate_limit: float = 1000000.0,
    ) -> None:
        if voxel_size <= 0.0:
            raise ValueError("voxel_size must be positive")
        if min_point_range < 0.0 or max_point_range <= min_point_range:
            raise ValueError("point range bounds are invalid")
        if min_relative_z >= max_relative_z:
            raise ValueError("min_relative_z must be below max_relative_z")
        if max_voxels <= 0:
            raise ValueError("max_voxels must be positive")
        if retention_radius < 0.0:
            raise ValueError("retention_radius must not be negative")
        if coordinate_limit <= 0.0:
            raise ValueError("coordinate_limit must be positive")

        self.voxel_size = float(voxel_size)
        self.min_point_range = float(min_point_range)
        self.max_point_range = float(max_point_range)
        self.min_relative_z = float(min_relative_z)
        self.max_relative_z = float(max_relative_z)
        self.max_voxels = int(max_voxels)
        self.retention_radius = float(retention_radius)
        self.coordinate_limit = float(coordinate_limit)
        self._voxels: Dict[VoxelKey, _Voxel] = {}
        self._fusion_count = 0

    def __len__(self) -> int:
        return len(self._voxels)

    def colorized_count(self) -> int:
        """Return how many voxels contain at least one RGB observation."""

        return sum(
            1 for voxel in self._voxels.values() if voxel.color_count > 0
        )

    def clear(self) -> None:
        self._voxels.clear()
        self._fusion_count = 0

    @staticmethod
    def _as_robot_position(robot_position: Iterable[float]) -> np.ndarray:
        robot = np.asarray(tuple(robot_position), dtype=np.float64)
        if robot.shape != (3,) or not np.isfinite(robot).all():
            raise ValueError("robot_position must be a finite XYZ vector")
        return robot

    def filter_points(
        self, points: np.ndarray, robot_position: Iterable[float]
    ) -> np.ndarray:
        """Apply finite-value, coordinate, range, and relative-height guards."""

        array = np.asarray(points, dtype=np.float64)
        return array[self.filter_mask(array, robot_position)]

    def filter_mask(
        self, points: np.ndarray, robot_position: Iterable[float]
    ) -> np.ndarray:
        """Return the accepted-row mask so aligned attributes stay attached."""

        array = np.asarray(points, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if array.size == 0:
            return np.zeros((array.shape[0],), dtype=bool)
        robot = self._as_robot_position(robot_position)
        mask = np.isfinite(array).all(axis=1)
        safe = np.where(mask[:, None], array, robot)
        relative = safe - robot
        squared_range = np.einsum("ij,ij->i", relative, relative)
        # The firmware registered cloud contains thousands of exact (0,0,0)
        # padding records. Reject them in world coordinates even after the
        # robot has moved away from the odometry origin.
        absolute_squared = np.einsum("ij,ij->i", safe, safe)
        mask &= absolute_squared > 1.0e-12
        mask &= np.max(np.abs(safe), axis=1) <= self.coordinate_limit
        mask &= squared_range >= self.min_point_range * self.min_point_range
        mask &= squared_range <= self.max_point_range * self.max_point_range
        mask &= relative[:, 2] >= self.min_relative_z
        mask &= relative[:, 2] <= self.max_relative_z
        return mask

    def fuse(
        self,
        points: np.ndarray,
        robot_position: Iterable[float],
        stamp_ns: int,
    ) -> int:
        """Filter and fuse one cloud, returning the accepted point count."""

        filtered = self.filter_points(points, robot_position)
        self.fuse_filtered(filtered, robot_position, stamp_ns)
        return int(filtered.shape[0])

    def fuse_filtered(
        self,
        points: np.ndarray,
        robot_position: Iterable[float],
        stamp_ns: int,
        colors_rgb: Optional[np.ndarray] = None,
    ) -> None:
        """Fuse points which have already passed :meth:`filter_points`."""

        array = np.asarray(points, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3:
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
        robot = self._as_robot_position(robot_position)
        stamp_ns = int(stamp_ns)
        self._fusion_count += 1

        if array.shape[0]:
            keys = np.floor(array / self.voxel_size).astype(np.int64)
            unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
            counts = np.bincount(inverse).astype(np.int64)
            sums = np.zeros((unique_keys.shape[0], 3), dtype=np.float64)
            np.add.at(sums, inverse, array)
            color_sums = None
            if colors is not None:
                color_sums = np.zeros(
                    (unique_keys.shape[0], 3), dtype=np.float64
                )
                np.add.at(color_sums, inverse, colors)

            for index, key_array in enumerate(unique_keys):
                key = tuple(int(value) for value in key_array)
                batch_count = int(counts[index])
                batch_centroid = sums[index] / float(batch_count)
                batch_color = (
                    color_sums[index] / float(batch_count)
                    if color_sums is not None
                    else None
                )
                existing = self._voxels.get(key)
                if existing is None:
                    self._voxels[key] = _Voxel(
                        centroid=batch_centroid,
                        count=batch_count,
                        last_seen_ns=stamp_ns,
                        color=batch_color,
                        color_count=batch_count if batch_color is not None else 0,
                    )
                    continue

                # Keep the stored weight in int64 range and prevent ancient data
                # from becoming impossible to update.
                old_weight = min(existing.count, 1000000)
                new_weight = min(batch_count, 1000000)
                total_weight = old_weight + new_weight
                existing.centroid = (
                    existing.centroid * old_weight + batch_centroid * new_weight
                ) / float(total_weight)
                existing.count = min(existing.count + batch_count, 2147483647)
                existing.last_seen_ns = max(existing.last_seen_ns, stamp_ns)
                if batch_color is not None:
                    old_color_weight = min(existing.color_count, 1000000)
                    new_color_weight = min(batch_count, 1000000)
                    if existing.color is None or old_color_weight == 0:
                        existing.color = batch_color
                    else:
                        existing.color = (
                            existing.color * old_color_weight
                            + batch_color * new_color_weight
                        ) / float(old_color_weight + new_color_weight)
                    existing.color_count = min(
                        existing.color_count + batch_count, 2147483647
                    )

        if self.retention_radius > 0.0 and self._fusion_count % 10 == 0:
            radius_squared = self.retention_radius * self.retention_radius
            stale_keys = [
                key
                for key, voxel in self._voxels.items()
                if float(np.sum((voxel.centroid - robot)[:2] ** 2)) > radius_squared
            ]
            for key in stale_keys:
                del self._voxels[key]

        self._enforce_limit()

    def _enforce_limit(self) -> None:
        if len(self._voxels) <= self.max_voxels:
            return
        target = max(1, int(self.max_voxels * 0.95))
        remove_count = len(self._voxels) - target
        oldest = sorted(
            self._voxels.items(), key=lambda item: item[1].last_seen_ns
        )[:remove_count]
        for key, _ in oldest:
            del self._voxels[key]

    def points(self) -> np.ndarray:
        """Return a stable copy of the current voxel centroids."""

        if not self._voxels:
            return np.empty((0, 3), dtype=np.float64)
        ordered = sorted(self._voxels.items())
        return np.vstack([record.centroid for _, record in ordered]).copy()

    def points_with_colors(
        self, default_color=(180, 180, 180)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return aligned centroids and RGB colors in stable voxel-key order."""

        ordered = sorted(self._voxels.items())
        if not ordered:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 3), dtype=np.uint8),
            )
        default = np.asarray(default_color, dtype=np.float64)
        if (
            default.shape != (3,)
            or not np.isfinite(default).all()
            or (default < 0.0).any()
            or (default > 255.0).any()
        ):
            raise ValueError("default_color must contain three channels")
        points = np.vstack([record.centroid for _, record in ordered]).copy()
        colors = np.vstack(
            [
                record.color if record.color is not None else default
                for _, record in ordered
            ]
        )
        return points, np.rint(colors).clip(0, 255).astype(np.uint8)

    def state(self) -> Mapping[str, np.ndarray]:
        """Return plain arrays suitable for a safe, pickle-free NPZ file."""

        ordered = sorted(self._voxels.items())
        if not ordered:
            return {
                "voxel_keys": np.empty((0, 3), dtype=np.int64),
                "voxel_centroids": np.empty((0, 3), dtype=np.float64),
                "voxel_counts": np.empty((0,), dtype=np.int64),
                "voxel_last_seen_ns": np.empty((0,), dtype=np.int64),
            }
        state = {
            "voxel_keys": np.asarray([key for key, _ in ordered], dtype=np.int64),
            "voxel_centroids": np.vstack(
                [record.centroid for _, record in ordered]
            ).astype(np.float64, copy=False),
            "voxel_counts": np.asarray(
                [record.count for _, record in ordered], dtype=np.int64
            ),
            "voxel_last_seen_ns": np.asarray(
                [record.last_seen_ns for _, record in ordered], dtype=np.int64
            ),
        }
        if any(record.color is not None for _, record in ordered):
            state["voxel_colors"] = np.vstack(
                [
                    record.color
                    if record.color is not None
                    else np.zeros(3, dtype=np.float64)
                    for _, record in ordered
                ]
            ).astype(np.float64, copy=False)
            state["voxel_color_counts"] = np.asarray(
                [record.color_count for _, record in ordered], dtype=np.int64
            )
        return state

    def restore(self, state: Mapping[str, np.ndarray]) -> None:
        """Replace the map with a validated state snapshot."""

        keys = np.asarray(state["voxel_keys"], dtype=np.int64)
        centroids = np.asarray(state["voxel_centroids"], dtype=np.float64)
        counts = np.asarray(state["voxel_counts"], dtype=np.int64)
        seen = np.asarray(state["voxel_last_seen_ns"], dtype=np.int64)
        has_colors = "voxel_colors" in state or "voxel_color_counts" in state
        if has_colors and not (
            "voxel_colors" in state and "voxel_color_counts" in state
        ):
            raise ValueError("voxel color state arrays must appear together")
        colors = None
        color_counts = None
        if has_colors:
            colors = np.asarray(state["voxel_colors"], dtype=np.float64)
            color_counts = np.asarray(
                state["voxel_color_counts"], dtype=np.int64
            )
        size = keys.shape[0] if keys.ndim == 2 else -1
        if (
            keys.ndim != 2
            or keys.shape[1:] != (3,)
            or centroids.shape != (size, 3)
            or counts.shape != (size,)
            or seen.shape != (size,)
            or (has_colors and colors.shape != (size, 3))
            or (has_colors and color_counts.shape != (size,))
        ):
            raise ValueError("invalid voxel state array shapes")
        if not np.isfinite(centroids).all() or (counts <= 0).any():
            raise ValueError("invalid voxel state values")
        if has_colors and (
            not np.isfinite(colors).all()
            or (colors < 0.0).any()
            or (colors > 255.0).any()
            or (color_counts < 0).any()
        ):
            raise ValueError("invalid voxel color state values")
        expected_keys = np.floor(centroids / self.voxel_size).astype(np.int64)
        if not np.array_equal(keys, expected_keys):
            raise ValueError("voxel state does not match configured voxel_size")

        order = np.argsort(seen)[-self.max_voxels :]
        restored: Dict[VoxelKey, _Voxel] = {}
        for index in order:
            key = tuple(int(value) for value in keys[index])
            restored[key] = _Voxel(
                centroid=centroids[index].copy(),
                count=min(int(counts[index]), 2147483647),
                last_seen_ns=int(seen[index]),
                color=(
                    colors[index].copy()
                    if has_colors and color_counts[index] > 0
                    else None
                ),
                color_count=(
                    min(int(color_counts[index]), 2147483647)
                    if has_colors
                    else 0
                ),
            )
        self._voxels = restored
        self._fusion_count = 0
