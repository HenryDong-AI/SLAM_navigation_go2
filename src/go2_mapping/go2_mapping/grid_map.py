"""Sparse log-odds occupancy projection and deterministic grid encoders."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np


GridKey = Tuple[int, int]


def bresenham_cells(start: GridKey, end: GridKey) -> List[GridKey]:
    """Return all integer cells on a line, including both endpoints."""

    x0, y0 = int(start[0]), int(start[1])
    x1, y1 = int(end[0]), int(end[1])
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    error = dx + dy
    cells: List[GridKey] = []
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            return cells
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x0 += sx
        if doubled <= dx:
            error += dx
            y0 += sy


def occupancy_to_pgm(
    occupancy: np.ndarray,
    free_threshold: int = 25,
    occupied_threshold: int = 65,
) -> bytes:
    """Encode an OccupancyGrid-style array as a ROS map-server PGM image."""

    grid = np.asarray(occupancy)
    if grid.ndim != 2 or grid.size == 0:
        raise ValueError("occupancy must be a non-empty two-dimensional array")
    if not 0 <= free_threshold < occupied_threshold <= 100:
        raise ValueError("invalid occupancy thresholds")
    if ((grid < -1) | (grid > 100)).any():
        raise ValueError("occupancy values must be -1 or between 0 and 100")

    image = np.full(grid.shape, 205, dtype=np.uint8)
    image[(grid >= 0) & (grid <= free_threshold)] = 254
    image[grid >= occupied_threshold] = 0
    # OccupancyGrid row zero is the lower edge; image row zero is the top edge.
    image = np.flipud(image)
    header = "P5\n# go2_mapping occupancy map\n{} {}\n255\n".format(
        grid.shape[1], grid.shape[0]
    ).encode("ascii")
    return header + image.tobytes()


@dataclass
class _Cell:
    log_odds: float
    last_seen_ns: int


class LogOddsGrid:
    """Sparse bounded occupancy grid updated with endpoint hits and ray misses."""

    def __init__(
        self,
        resolution: float,
        hit_log_odds: float,
        miss_log_odds: float,
        min_log_odds: float,
        max_log_odds: float,
        obstacle_min_height: float,
        obstacle_max_height: float,
        max_ray_range: float,
        max_cells: int,
        max_rays_per_update: int,
        max_ray_cells: int = 4096,
    ) -> None:
        if resolution <= 0.0:
            raise ValueError("resolution must be positive")
        if hit_log_odds <= 0.0 or miss_log_odds >= 0.0:
            raise ValueError("hit_log_odds must be positive and miss_log_odds negative")
        if min_log_odds >= max_log_odds:
            raise ValueError("min_log_odds must be below max_log_odds")
        if obstacle_min_height >= obstacle_max_height:
            raise ValueError("obstacle height limits are invalid")
        if max_ray_range <= 0.0 or max_cells <= 0:
            raise ValueError("range and cell limits must be positive")
        if max_rays_per_update <= 0 or max_ray_cells <= 0:
            raise ValueError("ray limits must be positive")

        self.resolution = float(resolution)
        self.hit_log_odds = float(hit_log_odds)
        self.miss_log_odds = float(miss_log_odds)
        self.min_log_odds = float(min_log_odds)
        self.max_log_odds = float(max_log_odds)
        self.obstacle_min_height = float(obstacle_min_height)
        self.obstacle_max_height = float(obstacle_max_height)
        self.max_ray_range = float(max_ray_range)
        self.max_cells = int(max_cells)
        self.max_rays_per_update = int(max_rays_per_update)
        self.max_ray_cells = int(max_ray_cells)
        self._cells: Dict[GridKey, _Cell] = {}
        self._last_robot_cell: GridKey = (0, 0)

    def __len__(self) -> int:
        return len(self._cells)

    def clear(self) -> None:
        self._cells.clear()

    def _cell_for_xy(self, xy: np.ndarray) -> GridKey:
        cell = np.floor(np.asarray(xy, dtype=np.float64) / self.resolution)
        return int(cell[0]), int(cell[1])

    def update(
        self,
        points: np.ndarray,
        robot_position: Iterable[float],
        stamp_ns: int,
    ) -> Tuple[int, int]:
        """Apply at most one miss/hit per cell for this scan.

        A cell observed as an endpoint wins over a clearing ray in the same scan,
        avoiding point-density-dependent log-odds and self-erasing obstacles.
        """

        array = np.asarray(points, dtype=np.float64)
        robot = np.asarray(tuple(robot_position), dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if robot.shape != (3,) or not np.isfinite(robot).all():
            raise ValueError("robot_position must be a finite XYZ vector")
        stamp_ns = int(stamp_ns)
        start = self._cell_for_xy(robot[:2])
        self._last_robot_cell = start
        if array.shape[0] == 0:
            return 0, 0

        relative = array - robot
        horizontal_range = np.linalg.norm(relative[:, :2], axis=1)
        ray_mask = np.isfinite(array).all(axis=1)
        ray_mask &= horizontal_range <= self.max_ray_range
        ray_points = array[ray_mask]
        ray_relative_z = relative[ray_mask, 2]
        if ray_points.shape[0] > self.max_rays_per_update:
            stride = int(np.ceil(ray_points.shape[0] / self.max_rays_per_update))
            ray_points = ray_points[::stride][: self.max_rays_per_update]
            ray_relative_z = ray_relative_z[::stride][: self.max_rays_per_update]

        obstacle_rows = (
            (ray_relative_z >= self.obstacle_min_height)
            & (ray_relative_z <= self.obstacle_max_height)
        )
        hit_cells: Set[GridKey] = set()
        free_cells: Set[GridKey] = set()
        for point, is_obstacle in zip(ray_points, obstacle_rows):
            endpoint = self._cell_for_xy(point[:2])
            if max(abs(endpoint[0] - start[0]), abs(endpoint[1] - start[1])) > (
                self.max_ray_cells - 1
            ):
                continue
            cells = bresenham_cells(start, endpoint)
            if bool(is_obstacle):
                hit_cells.add(endpoint)
                free_cells.update(cells[:-1])
            else:
                # A floor/ceiling return is not an obstacle, but the complete
                # unobstructed ray (including its endpoint) is free evidence.
                free_cells.update(cells)
        free_cells.difference_update(hit_cells)

        for key in free_cells:
            existing = self._cells.get(key)
            value = 0.0 if existing is None else existing.log_odds
            self._cells[key] = _Cell(
                log_odds=max(self.min_log_odds, value + self.miss_log_odds),
                last_seen_ns=stamp_ns,
            )
        for key in hit_cells:
            existing = self._cells.get(key)
            value = 0.0 if existing is None else existing.log_odds
            self._cells[key] = _Cell(
                log_odds=min(self.max_log_odds, value + self.hit_log_odds),
                last_seen_ns=stamp_ns,
            )

        self._enforce_limit()
        return len(free_cells), len(hit_cells)

    def _enforce_limit(self) -> None:
        if len(self._cells) <= self.max_cells:
            return
        target = max(1, int(self.max_cells * 0.95))
        remove_count = len(self._cells) - target
        oldest = sorted(
            self._cells.items(), key=lambda item: item[1].last_seen_ns
        )[:remove_count]
        for key, _ in oldest:
            del self._cells[key]

    def log_odds_at(self, key: GridKey) -> Optional[float]:
        cell = self._cells.get((int(key[0]), int(key[1])))
        return None if cell is None else float(cell.log_odds)

    def to_dense(
        self,
        max_dense_cells: int,
        padding: int = 1,
    ) -> Tuple[np.ndarray, float, float, bool]:
        """Return data, metric origin, and whether memory-safe cropping occurred."""

        if max_dense_cells <= 0 or padding < 0:
            raise ValueError("dense-grid limits are invalid")
        if not self._cells:
            x_min, y_min = self._last_robot_cell
            return (
                np.full((1, 1), -1, dtype=np.int8),
                x_min * self.resolution,
                y_min * self.resolution,
                False,
            )

        keys = np.asarray(list(self._cells.keys()), dtype=np.int64)
        x_min = int(keys[:, 0].min()) - padding
        x_max = int(keys[:, 0].max()) + padding
        y_min = int(keys[:, 1].min()) - padding
        y_max = int(keys[:, 1].max()) + padding
        width = x_max - x_min + 1
        height = y_max - y_min + 1
        cropped = width * height > max_dense_cells
        if cropped:
            side = max(1, int(np.floor(np.sqrt(max_dense_cells))))
            x_min = self._last_robot_cell[0] - side // 2
            y_min = self._last_robot_cell[1] - side // 2
            width = side
            height = side
            x_max = x_min + width - 1
            y_max = y_min + height - 1

        dense = np.full((height, width), -1, dtype=np.int8)
        for (x_cell, y_cell), cell in self._cells.items():
            if x_min <= x_cell <= x_max and y_min <= y_cell <= y_max:
                probability = 100.0 / (1.0 + np.exp(-cell.log_odds))
                dense[y_cell - y_min, x_cell - x_min] = np.int8(
                    int(np.clip(np.rint(probability), 0, 100))
                )
        return (
            dense,
            x_min * self.resolution,
            y_min * self.resolution,
            cropped,
        )

    def state(self) -> Mapping[str, np.ndarray]:
        ordered = sorted(self._cells.items())
        if not ordered:
            return {
                "grid_keys": np.empty((0, 2), dtype=np.int64),
                "grid_log_odds": np.empty((0,), dtype=np.float64),
                "grid_last_seen_ns": np.empty((0,), dtype=np.int64),
                "grid_last_robot_cell": np.asarray(
                    self._last_robot_cell, dtype=np.int64
                ),
            }
        return {
            "grid_keys": np.asarray([key for key, _ in ordered], dtype=np.int64),
            "grid_log_odds": np.asarray(
                [cell.log_odds for _, cell in ordered], dtype=np.float64
            ),
            "grid_last_seen_ns": np.asarray(
                [cell.last_seen_ns for _, cell in ordered], dtype=np.int64
            ),
            "grid_last_robot_cell": np.asarray(
                self._last_robot_cell, dtype=np.int64
            ),
        }

    def restore(self, state: Mapping[str, np.ndarray]) -> None:
        keys = np.asarray(state["grid_keys"], dtype=np.int64)
        values = np.asarray(state["grid_log_odds"], dtype=np.float64)
        seen = np.asarray(state["grid_last_seen_ns"], dtype=np.int64)
        robot_cell = np.asarray(state["grid_last_robot_cell"], dtype=np.int64)
        size = keys.shape[0] if keys.ndim == 2 else -1
        if (
            keys.ndim != 2
            or keys.shape[1:] != (2,)
            or values.shape != (size,)
            or seen.shape != (size,)
            or robot_cell.shape != (2,)
        ):
            raise ValueError("invalid grid state array shapes")
        if not np.isfinite(values).all():
            raise ValueError("grid state contains non-finite log odds")
        if ((values < self.min_log_odds) | (values > self.max_log_odds)).any():
            raise ValueError("grid state exceeds configured log-odds limits")

        order = np.argsort(seen)[-self.max_cells :]
        restored: Dict[GridKey, _Cell] = {}
        for index in order:
            key = int(keys[index, 0]), int(keys[index, 1])
            restored[key] = _Cell(float(values[index]), int(seen[index]))
        self._cells = restored
        self._last_robot_cell = int(robot_cell[0]), int(robot_cell[1])
