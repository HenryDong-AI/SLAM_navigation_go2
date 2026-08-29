import unittest
from types import SimpleNamespace

import numpy as np

from go2_mapping_depthcam.geometry import (
    decode_color_image,
    decode_depth_image,
    depth_to_camera_points,
    depth_to_camera_points_rgb,
    pose_matrix,
    rigid_transform,
    transform_points,
)


class _DepthMessage:
    width = 2
    height = 2
    step = 6
    encoding = "16UC1"
    is_bigendian = False
    data = (
        np.asarray([1000, 2000], dtype="<u2").tobytes()
        + b"xx"
        + np.asarray([0, 3000], dtype="<u2").tobytes()
    )


class GeometryTest(unittest.TestCase):
    def test_decode_bgr_with_padded_rows(self):
        message = SimpleNamespace(
            width=2,
            height=2,
            step=8,
            encoding="bgr8",
            data=bytes([1, 2, 3, 4, 5, 6, 99, 99,
                        7, 8, 9, 10, 11, 12]),
        )
        image = decode_color_image(message)
        np.testing.assert_array_equal(
            image,
            np.array(
                [[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
                dtype=np.uint8,
            ),
        )

    def test_depth_decode_handles_row_padding_and_scale(self):
        actual = decode_depth_image(_DepthMessage(), raw_depth_scale=0.001)
        np.testing.assert_allclose(actual, [[1.0, 2.0], [0.0, 3.0]])

    def test_depth_deprojection_uses_optical_axes(self):
        depth = np.asarray([[1.0, 2.0], [0.0, np.nan]], dtype=np.float32)
        points = depth_to_camera_points(
            depth,
            fx=1.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
            pixel_stride=1,
            min_depth=0.1,
            max_depth=3.0,
            max_points=10,
        )
        np.testing.assert_allclose(points, [[0.0, 0.0, 1.0], [2.0, 0.0, 2.0]])

    def test_depth_deprojection_keeps_aligned_rgb(self):
        depth = np.asarray([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32)
        bgr = np.asarray(
            [
                [[30, 20, 10], [0, 0, 0]],
                [[60, 50, 40], [90, 80, 70]],
            ],
            dtype=np.uint8,
        )
        points, rgb = depth_to_camera_points_rgb(
            depth,
            bgr,
            fx=1.0,
            fy=1.0,
            cx=0.0,
            cy=0.0,
            pixel_stride=1,
            min_depth=0.1,
            max_depth=2.5,
            max_points=10,
        )
        np.testing.assert_allclose(points, [[0, 0, 1], [0, 2, 2]])
        np.testing.assert_array_equal(rgb, [[10, 20, 30], [40, 50, 60]])

    def test_optical_to_base_and_odom_pose(self):
        base_from_optical = rigid_transform(
            [
                0.0, 0.0, 1.0, 0.2,
                -1.0, 0.0, 0.0, 0.0,
                0.0, -1.0, 0.0, 0.3,
                0.0, 0.0, 0.0, 1.0,
            ]
        )
        odom_from_base = pose_matrix([1.0, 2.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        point = np.asarray([[0.0, 0.0, 2.0]])
        actual = transform_points(point, odom_from_base @ base_from_optical)
        np.testing.assert_allclose(actual, [[3.2, 2.0, 0.3]])

    def test_rejects_non_rigid_transform(self):
        with self.assertRaises(ValueError):
            rigid_transform(np.diag([1.0, 1.0, 2.0, 1.0]).reshape(-1))


if __name__ == "__main__":
    unittest.main()
