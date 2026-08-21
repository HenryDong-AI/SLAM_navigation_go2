"""Small, dependency-light PointCloud2 helpers.

The parser deliberately accepts a duck-typed message so it can be tested without
a ROS installation.  Only the standard PointField datatype identifiers are used.
"""

from typing import Dict, Optional, Tuple

import numpy as np


class PointCloudFormatError(ValueError):
    """Raised when a PointCloud2 buffer or field layout is unsafe to consume."""


# sensor_msgs/msg/PointField constants, duplicated as values to keep this module
# importable outside ROS.
_FIELD_TYPES: Dict[int, Tuple[str, int]] = {
    1: ("i1", 1),
    2: ("u1", 1),
    3: ("i2", 2),
    4: ("u2", 2),
    5: ("i4", 4),
    6: ("u4", 4),
    7: ("f4", 4),
    8: ("f8", 8),
}


def _xyz_dtype(message) -> np.dtype:
    point_step = int(message.point_step)
    if point_step <= 0:
        raise PointCloudFormatError("point_step must be positive")

    by_name = {str(field.name): field for field in message.fields}
    missing = [name for name in ("x", "y", "z") if name not in by_name]
    if missing:
        raise PointCloudFormatError("missing PointCloud2 fields: " + ", ".join(missing))

    byte_order = ">" if bool(message.is_bigendian) else "<"
    formats = []
    offsets = []
    for name in ("x", "y", "z"):
        field = by_name[name]
        datatype = int(field.datatype)
        if datatype not in _FIELD_TYPES:
            raise PointCloudFormatError(
                "unsupported datatype {} for field {}".format(datatype, name)
            )
        if int(getattr(field, "count", 1)) != 1:
            raise PointCloudFormatError("field {} must be scalar".format(name))
        code, size = _FIELD_TYPES[datatype]
        offset = int(field.offset)
        if offset < 0 or offset + size > point_step:
            raise PointCloudFormatError(
                "field {} extends beyond point_step".format(name)
            )
        formats.append(byte_order + code)
        offsets.append(offset)

    return np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": formats,
            "offsets": offsets,
            "itemsize": point_step,
        }
    )


def read_xyz(message, max_points: Optional[int] = None) -> np.ndarray:
    """Return the x/y/z columns of a PointCloud2-like message as float64.

    Organized clouds with row padding and both endian layouts are supported.
    The declared layout is fully checked before NumPy is allowed to view it.
    NaN and infinite values are retained here and filtered by the mapping layer.
    """

    width = int(message.width)
    height = int(message.height)
    if width < 0 or height < 0:
        raise PointCloudFormatError("width and height must not be negative")
    count = width * height
    if max_points is not None and count > int(max_points):
        raise PointCloudFormatError(
            "cloud declares {} points, above the configured limit {}".format(
                count, int(max_points)
            )
        )
    if count == 0:
        return np.empty((0, 3), dtype=np.float64)

    dtype = _xyz_dtype(message)
    point_step = int(message.point_step)
    minimum_row_step = width * point_step
    row_step = int(message.row_step)
    if row_step < minimum_row_step:
        raise PointCloudFormatError("row_step is smaller than width * point_step")

    data = memoryview(message.data)
    required_bytes = (height - 1) * row_step + minimum_row_step
    if len(data) < required_bytes:
        raise PointCloudFormatError(
            "PointCloud2 data buffer is shorter than its declared layout"
        )

    if row_step == minimum_row_step:
        records = np.frombuffer(data[: required_bytes], dtype=dtype, count=count)
    else:
        rows = []
        for row_index in range(height):
            begin = row_index * row_step
            end = begin + minimum_row_step
            rows.append(np.frombuffer(data[begin:end], dtype=dtype, count=width))
        records = np.concatenate(rows)

    # astype also detaches the result from a potentially mutable ROS buffer.
    return np.column_stack(
        (
            records["x"].astype(np.float64, copy=False),
            records["y"].astype(np.float64, copy=False),
            records["z"].astype(np.float64, copy=False),
        )
    ).copy()


def xyz_to_float32_bytes(points: np.ndarray) -> bytes:
    """Encode an N-by-3 array in the conventional little-endian XYZ layout."""

    array = np.asarray(points)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return np.ascontiguousarray(array, dtype="<f4").tobytes()
