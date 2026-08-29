import struct
import unittest
from types import SimpleNamespace

import numpy as np

from go2_mapping.pointcloud import (
    PointCloudFormatError,
    read_xyz,
    xyzrgb_to_float32_bytes,
)


def field(name, offset, datatype=7):
    return SimpleNamespace(name=name, offset=offset, datatype=datatype, count=1)


class PointCloudParsingTest(unittest.TestCase):
    def test_xyzrgb_encoder_uses_pcl_packed_rgb_layout(self):
        encoded = xyzrgb_to_float32_bytes(
            np.asarray([[1.0, 2.0, 3.0]]),
            np.asarray([[12, 34, 56]], dtype=np.uint8),
        )
        x, y, z, packed = struct.unpack("<fffI", encoded)
        self.assertEqual((x, y, z), (1.0, 2.0, 3.0))
        self.assertEqual(packed, 0x000C2238)

    def test_organized_float_cloud_with_row_padding(self):
        first_row = struct.pack("<ffffff", 1, 2, 3, 4, 5, 6) + b"PAD!"
        second_row = struct.pack("<ffffff", 7, 8, 9, 10, 11, 12) + b"PAD!"
        message = SimpleNamespace(
            width=2,
            height=2,
            fields=[field("x", 0), field("y", 4), field("z", 8)],
            is_bigendian=False,
            point_step=12,
            row_step=28,
            data=first_row + second_row,
        )
        points = read_xyz(message)
        np.testing.assert_allclose(
            points,
            [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        )

    def test_big_endian_float64(self):
        message = SimpleNamespace(
            width=1,
            height=1,
            fields=[
                field("x", 0, 8),
                field("y", 8, 8),
                field("z", 16, 8),
            ],
            is_bigendian=True,
            point_step=24,
            row_step=24,
            data=struct.pack(">ddd", 1.25, -2.5, 3.75),
        )
        np.testing.assert_allclose(read_xyz(message), [[1.25, -2.5, 3.75]])

    def test_rejects_unsafe_layout_and_point_limit(self):
        message = SimpleNamespace(
            width=2,
            height=1,
            fields=[field("x", 0), field("y", 4), field("z", 10)],
            is_bigendian=False,
            point_step=12,
            row_step=24,
            data=bytes(24),
        )
        with self.assertRaises(PointCloudFormatError):
            read_xyz(message)
        message.fields = [field("x", 0), field("y", 4), field("z", 8)]
        with self.assertRaises(PointCloudFormatError):
            read_xyz(message, max_points=1)


if __name__ == "__main__":
    unittest.main()
