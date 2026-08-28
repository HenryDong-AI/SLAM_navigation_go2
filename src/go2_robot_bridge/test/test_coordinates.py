import math
import struct
from types import SimpleNamespace
import unittest

from go2_robot_bridge.coordinates import (
    sample_pointcloud_xyz,
    transform_covariance,
    transform_odometry_in_place,
    transform_pointcloud_data,
    transform_pointcloud_in_place,
    transform_quaternion,
    transform_xyz,
)


def vector(x, y, z):
    return SimpleNamespace(x=x, y=y, z=z)


def field(name, offset):
    return SimpleNamespace(name=name, offset=offset, datatype=7, count=1)


class CoordinateConversionTest(unittest.TestCase):
    def test_bounded_xyz_sampling_handles_organized_padding(self):
        point_step = 16
        row_step = 36
        raw = bytearray(b"\xA5" * (2 * row_step))
        points = (
            (1.0, 2.0, 3.0, 10.0),
            (4.0, 5.0, 6.0, 20.0),
            (7.0, 8.0, 9.0, 30.0),
            (10.0, 11.0, 12.0, 40.0),
        )
        for index, point in enumerate(points):
            row, column = divmod(index, 2)
            struct.pack_into(
                "<ffff",
                raw,
                row * row_step + column * point_step,
                *point
            )
        message = SimpleNamespace(
            width=2,
            height=2,
            point_step=point_step,
            row_step=row_step,
            is_bigendian=False,
            fields=[
                field("x", 0),
                field("y", 4),
                field("z", 8),
                field("intensity", 12),
            ],
            data=bytes(raw),
        )
        sampled = sample_pointcloud_xyz(message, max_points=2)
        self.assertEqual(sampled.tolist(), [[1.0, 2.0, 3.0],
                                            [10.0, 11.0, 12.0]])

    def test_vectors_quaternion_and_covariance_use_complete_basis_change(self):
        self.assertEqual(transform_xyz((1.0, 2.0, -3.0)), (1.0, -2.0, 3.0))

        result = transform_quaternion((0.1, 0.2, 0.3, 0.9))
        norm = math.sqrt(0.95)
        expected = (0.1 / norm, -0.2 / norm, -0.3 / norm, 0.9 / norm)
        for actual, wanted in zip(result, expected):
            self.assertAlmostEqual(actual, wanted)

        source = tuple(float(index + 1) for index in range(36))
        converted = transform_covariance(source)
        signs = (1.0, -1.0, -1.0, 1.0, -1.0, -1.0)
        for row in range(6):
            for column in range(6):
                self.assertEqual(
                    converted[row * 6 + column],
                    signs[row] * source[row * 6 + column] * signs[column],
                )

    def test_organized_cloud_preserves_other_fields_padding_and_trailer(self):
        point_step = 16
        row_step = 36
        raw = bytearray(b"\xA5" * (2 * row_step + 3))
        points = (
            (1.0, 2.0, 3.0, 10.0),
            (4.0, -5.0, 6.0, 20.0),
            (-7.0, 8.0, -9.0, 30.0),
            (10.0, 11.0, 12.0, 40.0),
        )
        for index, point in enumerate(points):
            row, column = divmod(index, 2)
            struct.pack_into(
                "<ffff",
                raw,
                row * row_step + column * point_step,
                *point
            )
        message = SimpleNamespace(
            width=2,
            height=2,
            point_step=point_step,
            row_step=row_step,
            is_bigendian=False,
            fields=[
                field("x", 0),
                field("y", 4),
                field("z", 8),
                field("intensity", 12),
            ],
            data=bytes(raw),
        )

        converted = transform_pointcloud_data(message)
        for index, source in enumerate(points):
            row, column = divmod(index, 2)
            actual = struct.unpack_from(
                "<ffff", converted, row * row_step + column * point_step
            )
            self.assertEqual(
                actual, (source[0], -source[1], -source[2], source[3])
            )
        self.assertEqual(converted[32:36], raw[32:36])
        self.assertEqual(converted[68:72], raw[68:72])
        self.assertEqual(converted[72:], raw[72:])

        runtime_buffer = bytearray(raw)
        message.data = runtime_buffer
        transform_pointcloud_in_place(message)
        self.assertIs(message.data, runtime_buffer)
        self.assertEqual(
            struct.unpack_from("<ffff", runtime_buffer, 0),
            (1.0, -2.0, -3.0, 10.0),
        )

    def test_big_endian_cloud_and_invalid_layout(self):
        raw = struct.pack(">fff", 1.25, -2.5, 3.75)
        message = SimpleNamespace(
            width=1,
            height=1,
            point_step=12,
            row_step=12,
            is_bigendian=True,
            fields=[field("x", 0), field("y", 4), field("z", 8)],
            data=raw,
        )
        converted = transform_pointcloud_data(message)
        self.assertEqual(
            struct.unpack(">fff", converted),
            (1.25, 2.5, -3.75),
        )
        message.row_step = 11
        with self.assertRaisesRegex(ValueError, "row_step"):
            transform_pointcloud_data(message)

        message.row_step = 12
        message.fields = [field("x", 0), field("y", 2), field("z", 8)]
        with self.assertRaisesRegex(ValueError, "overlap"):
            transform_pointcloud_data(message)

        message.fields = [
            field("x", 0),
            field("x", 4),
            field("y", 4),
            field("z", 8),
        ]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            transform_pointcloud_data(message)

    def test_odometry_pose_twist_and_covariances_all_convert(self):
        pose_covariance = [0.0] * 36
        pose_covariance[1] = 2.0
        twist_covariance = [0.0] * 36
        twist_covariance[4] = 3.0
        message = SimpleNamespace(
            pose=SimpleNamespace(
                pose=SimpleNamespace(
                    position=vector(1.0, 2.0, 3.0),
                    orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=2.0),
                ),
                covariance=pose_covariance,
            ),
            twist=SimpleNamespace(
                twist=SimpleNamespace(
                    linear=vector(4.0, 5.0, 6.0),
                    angular=vector(7.0, 8.0, 9.0),
                ),
                covariance=twist_covariance,
            ),
        )

        transform_odometry_in_place(message)
        self.assertEqual(
            (message.pose.pose.position.x, message.pose.pose.position.y,
             message.pose.pose.position.z),
            (1.0, -2.0, -3.0),
        )
        self.assertEqual(
            (message.pose.pose.orientation.x, message.pose.pose.orientation.y,
             message.pose.pose.orientation.z, message.pose.pose.orientation.w),
            (0.0, -0.0, -0.0, 1.0),
        )
        self.assertEqual(
            (message.twist.twist.linear.x, message.twist.twist.linear.y,
             message.twist.twist.linear.z),
            (4.0, -5.0, -6.0),
        )
        self.assertEqual(
            (message.twist.twist.angular.x, message.twist.twist.angular.y,
             message.twist.twist.angular.z),
            (7.0, -8.0, -9.0),
        )
        self.assertEqual(message.pose.covariance[1], -2.0)
        self.assertEqual(message.twist.covariance[4], -3.0)


if __name__ == "__main__":
    unittest.main()
