import math
from dataclasses import dataclass

import numpy as np
import pytest

from go2_navigation.cloud_math import (
    normalize_frame_id,
    points_to_ranges,
    xyz_from_buffer,
)


@dataclass
class Field:
    name: str
    offset: int
    datatype: int = 7
    count: int = 1


def test_xyz_from_padded_little_endian_buffer():
    dtype = np.dtype({
        "names": ["x", "y", "z", "pad"],
        "formats": ["<f4", "<f4", "<f4", "V20"],
        "offsets": [0, 4, 8, 12],
        "itemsize": 32,
    })
    data = np.zeros(2, dtype=dtype)
    data["x"] = [1.0, 2.0]
    data["y"] = [3.0, 4.0]
    data["z"] = [5.0, 6.0]
    result = xyz_from_buffer(
        data.tobytes(), 32, [Field("x", 0), Field("y", 4), Field("z", 8)]
    )
    np.testing.assert_allclose(result, [[1, 3, 5], [2, 4, 6]])


def test_xyz_rejects_wrong_field_type():
    with pytest.raises(ValueError):
        xyz_from_buffer(bytes(32), 32, [Field("x", 0, 2), Field("y", 4), Field("z", 8)])


def test_xyz_reads_organized_rows_without_consuming_row_padding():
    point_step = 16
    width = 2
    height = 2
    row_step = 40
    data = bytearray(row_step * height)
    values = ((1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12))
    point_dtype = np.dtype(
        {
            "names": ["x", "y", "z"],
            "formats": ["<f4", "<f4", "<f4"],
            "offsets": [0, 4, 8],
            "itemsize": point_step,
        }
    )
    for row in range(height):
        row_values = np.zeros(width, dtype=point_dtype)
        for column in range(width):
            value = values[row * width + column]
            row_values[column] = value
        start = row * row_step
        data[start:start + width * point_step] = row_values.tobytes()
        data[start + width * point_step:(row + 1) * row_step] = b"padding!"
    result = xyz_from_buffer(
        bytes(data),
        point_step,
        [Field("x", 0), Field("y", 4), Field("z", 8)],
        width=width,
        height=height,
        row_step=row_step,
    )
    np.testing.assert_allclose(result, values)


def test_xyz_rejects_bad_row_layout_and_duplicate_fields():
    fields = [Field("x", 0), Field("y", 4), Field("z", 8)]
    with pytest.raises(ValueError):
        xyz_from_buffer(bytes(32), 16, fields, width=2, height=2, row_step=16)
    with pytest.raises(ValueError):
        xyz_from_buffer(
            bytes(32), 16, fields, width=2, height=2, row_step=32
        )
    with pytest.raises(ValueError):
        xyz_from_buffer(
            bytes(16),
            16,
            fields + [Field("x", 12)],
            width=1,
            height=1,
            row_step=16,
        )


def test_frame_ids_are_canonicalized():
    assert normalize_frame_id(" /base_link ") == "base_link"


def test_points_to_ranges_selects_nearest_and_height_band():
    points = np.array(
        [
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.1],
            [0.5, 0.0, 1.0],
            [0.0, 2.0, 0.0],
            [np.nan, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    ranges = points_to_ranges(
        points,
        angle_min=-math.pi,
        angle_max=math.pi,
        angle_increment=math.pi / 2,
        min_height=-0.2,
        max_height=0.5,
    )
    assert ranges[2] == pytest.approx(1.0)
    assert ranges[3] == pytest.approx(2.0)
    assert np.isinf(ranges[0])
