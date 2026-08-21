"""PointCloud2 decoding and polar scan reduction without ROS dependencies.

Copyright (c) 2026 Go2 SLAM Navigation Maintainers. MIT License.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Optional

import numpy as np


FLOAT32 = 7


def normalize_frame_id(frame_id: str) -> str:
    return str(frame_id).strip().lstrip("/")


def xyz_from_buffer(
    data: bytes,
    point_step: int,
    fields: Iterable,
    *,
    bigendian: bool = False,
    width: Optional[int] = None,
    height: int = 1,
    row_step: Optional[int] = None,
) -> np.ndarray:
    """Return an N x 3 float32 view/copy from a PointCloud2-like byte buffer."""
    if point_step <= 0:
        raise ValueError("invalid PointCloud2 point_step")
    field_list = list(fields)
    by_name: Mapping[str, object] = {field.name: field for field in field_list}
    if len(by_name) != len(field_list):
        raise ValueError("duplicate PointCloud2 field name")
    for axis in ("x", "y", "z"):
        field = by_name.get(axis)
        if field is None or int(field.datatype) != FLOAT32 or int(field.count) != 1:
            raise ValueError("point cloud requires scalar FLOAT32 x, y, z fields")
        if int(field.offset) < 0 or int(field.offset) + 4 > point_step:
            raise ValueError(f"invalid {axis} field offset")
    endian = ">" if bigendian else "<"
    dtype = np.dtype({
        "names": ["x", "y", "z"],
        "formats": [endian + "f4", endian + "f4", endian + "f4"],
        "offsets": [int(by_name[a].offset) for a in ("x", "y", "z")],
        "itemsize": point_step,
    })
    if width is None:
        if len(data) % point_step:
            raise ValueError("invalid PointCloud2 data length")
        structured = np.frombuffer(data, dtype=dtype)
    else:
        width = int(width)
        height = int(height)
        if width < 0 or height < 1:
            raise ValueError("invalid PointCloud2 dimensions")
        packed_row_bytes = width * point_step
        row_step = packed_row_bytes if row_step is None else int(row_step)
        if row_step < packed_row_bytes:
            raise ValueError("PointCloud2 row_step is smaller than its packed row")
        required = row_step * height
        if len(data) < required:
            raise ValueError("PointCloud2 data is shorter than row_step * height")
        if width == 0:
            return np.empty((0, 3), dtype=np.float32)
        if row_step == packed_row_bytes:
            structured = np.frombuffer(
                data, dtype=dtype, count=width * height
            )
        else:
            rows = [
                np.frombuffer(
                    memoryview(data)[
                        row * row_step:row * row_step + packed_row_bytes
                    ],
                    dtype=dtype,
                    count=width,
                )
                for row in range(height)
            ]
            structured = np.concatenate(rows)
    return np.column_stack(
        (structured["x"], structured["y"], structured["z"])
    ).astype(np.float32, copy=False)


def points_to_ranges(
    points: np.ndarray,
    *,
    angle_min: float = -math.pi,
    angle_max: float = math.pi,
    angle_increment: float = math.radians(0.5),
    range_min: float = 0.15,
    range_max: float = 12.0,
    min_height: float = -0.25,
    max_height: float = 0.60,
) -> np.ndarray:
    """Reduce XYZ points in base_link coordinates to closest polar ranges."""
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be shaped N x 3")
    if not (angle_increment > 0.0 and angle_max > angle_min and range_max > range_min):
        raise ValueError("invalid scan bounds")
    count = int(math.ceil((angle_max - angle_min) / angle_increment))
    output = np.full(count, np.inf, dtype=np.float32)
    if points.size == 0:
        return output
    finite = np.isfinite(points).all(axis=1)
    planar = np.hypot(points[:, 0], points[:, 1])
    angle = np.arctan2(points[:, 1], points[:, 0])
    with np.errstate(invalid="ignore"):
        keep = (
            finite
            & (points[:, 2] >= min_height)
            & (points[:, 2] <= max_height)
            & (planar >= range_min)
            & (planar <= range_max)
            & (angle >= angle_min)
            & (angle < angle_max)
        )
    if not np.any(keep):
        return output
    indices = ((angle[keep] - angle_min) / angle_increment).astype(np.int64)
    np.minimum.at(output, indices, planar[keep].astype(np.float32))
    return output
