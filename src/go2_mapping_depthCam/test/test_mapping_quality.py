import unittest

import cv2
import numpy as np

from go2_mapping_depthcam.geometry import (
    DeviceClockSynchronizer,
    interpolate_pose,
    pose_matrix,
)
from go2_mapping_depthcam.registration import (
    register_rigid_scan,
    rotation_degrees,
    transform_points_fast,
)
from go2_mapping_depthcam.surface_map import RgbdSurfaceAccumulator


class MappingQualityTest(unittest.TestCase):
    def test_camera_clock_uses_capture_cadence_not_callback_time(self):
        clock = DeviceClockSynchronizer(max_slew_ns=200000)
        first = clock.to_ros_ns(1000.0, 10000000000)
        second = clock.to_ros_ns(1066.0, 10076000000)

        self.assertEqual(first, 10000000000)
        self.assertEqual(second, 10066200000)
        self.assertLess(second, 10076000000)

    def test_pose_interpolation_uses_translation_and_slerp(self):
        first = pose_matrix([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        yaw = np.deg2rad(20.0)
        second = pose_matrix(
            [2.0, -1.0, 0.4],
            [0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)],
        )
        middle = interpolate_pose(first, second, 0.5)

        np.testing.assert_allclose(middle[:3, 3], [1.0, -0.5, 0.2])
        self.assertAlmostEqual(rotation_degrees(middle), 10.0, places=6)

    def test_full_se3_registration_recovers_indoor_pose_error(self):
        horizontal = np.linspace(-1.5, 1.5, 35)
        vertical = np.linspace(0.0, 1.8, 25)
        xx, yy = np.meshgrid(horizontal, horizontal)
        floor = np.column_stack(
            (xx.ravel(), yy.ravel(), np.zeros(xx.size))
        )
        yy, zz = np.meshgrid(horizontal, vertical)
        wall_x = np.column_stack(
            (np.full(yy.size, 1.5), yy.ravel(), zz.ravel())
        )
        xx, zz = np.meshgrid(horizontal, vertical)
        wall_y = np.column_stack(
            (xx.ravel(), np.full(xx.size, -1.5), zz.ravel())
        )
        target = np.vstack((floor, wall_x, wall_y))

        rotation, _ = cv2.Rodrigues(
            np.deg2rad([1.0, -1.5, 2.0]).reshape(3, 1)
        )
        expected = np.eye(4)
        expected[:3, :3] = rotation
        expected[:3, 3] = [0.035, -0.025, 0.018]
        source = transform_points_fast(target, np.linalg.inv(expected))

        actual, rmse, overlap, _ = register_rigid_scan(
            source,
            target,
            max_correspondence=0.20,
            max_iterations=15,
            trim_fraction=0.9,
            min_correspondences=200,
        )

        self.assertLess(rmse, 0.002)
        self.assertGreater(overlap, 0.98)
        np.testing.assert_allclose(
            actual[:3, 3], expected[:3, 3], atol=0.002
        )
        self.assertLess(
            rotation_degrees(np.linalg.inv(expected) @ actual), 0.1
        )

    @staticmethod
    def _surface_map(max_variance=0.0016):
        return RgbdSurfaceAccumulator(
            voxel_size=0.10,
            min_point_range=0.0,
            max_point_range=5.0,
            min_relative_z=-1.0,
            max_relative_z=2.0,
            max_voxels=100,
            min_observations=2,
            max_surface_variance=max_variance,
        )

    def test_surface_fusion_counts_frames_not_pixels(self):
        surface = self._surface_map()
        robot = np.zeros(3)
        first = np.repeat([[0.12, 0.02, 0.02]], 100, axis=0)
        second = np.repeat([[0.14, 0.02, 0.02]], 2, axis=0)
        colors = np.repeat([[20, 40, 60]], first.shape[0], axis=0)

        surface.fuse_filtered(first, robot, 1, colors)
        self.assertEqual(surface.surface_count(), 0)
        surface.fuse_filtered(
            second, robot, 2, np.repeat([[40, 60, 80]], 2, axis=0)
        )

        self.assertEqual(surface.surface_count(), 1)
        state = surface.state()
        self.assertEqual(int(state["voxel_counts"][0]), 2)
        np.testing.assert_allclose(
            state["voxel_centroids"][0], [0.13, 0.02, 0.02]
        )

    def test_unstable_surface_is_not_published(self):
        surface = self._surface_map(max_variance=0.0001)
        robot = np.zeros(3)
        surface.fuse_filtered(
            np.asarray([[0.11, 0.02, 0.02]]), robot, 1
        )
        surface.fuse_filtered(
            np.asarray([[0.19, 0.02, 0.02]]), robot, 2
        )

        self.assertEqual(len(surface), 1)
        self.assertEqual(surface.surface_count(), 0)
        self.assertEqual(surface.points_with_colors()[0].shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
