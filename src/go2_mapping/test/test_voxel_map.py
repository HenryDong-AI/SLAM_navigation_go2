import unittest

import numpy as np

from go2_mapping.voxel_map import VoxelAccumulator


def make_map(max_voxels=10):
    return VoxelAccumulator(
        voxel_size=0.1,
        min_point_range=0.0,
        max_point_range=5.0,
        min_relative_z=-1.0,
        max_relative_z=2.0,
        max_voxels=max_voxels,
    )


class VoxelAccumulatorTest(unittest.TestCase):
    def test_filters_and_fuses_centroid(self):
        accumulator = make_map()
        points = np.asarray(
            [
                [0.01, 0.02, 0.03],
                [0.09, 0.08, 0.07],
                [9.0, 0.0, 0.5],
                [np.nan, 0.0, 0.5],
            ]
        )
        accepted = accumulator.fuse(points, [0, 0, 0], stamp_ns=100)
        self.assertEqual(accepted, 2)
        self.assertEqual(len(accumulator), 1)
        np.testing.assert_allclose(accumulator.points(), [[0.05, 0.05, 0.05]])

        accumulator.fuse(np.asarray([[0.07, 0.05, 0.06]]), [0, 0, 0], 200)
        np.testing.assert_allclose(
            accumulator.points(), [[0.0566666667, 0.05, 0.0533333333]], rtol=1e-7
        )

    def test_memory_bound_and_state_round_trip(self):
        accumulator = make_map(max_voxels=3)
        accumulator.fuse(
            np.asarray([[0.0, 0, 0], [0.2, 0, 0], [0.4, 0, 0], [0.6, 0, 0]]),
            [0, 0, 0],
            10,
        )
        self.assertLessEqual(len(accumulator), 3)
        state = accumulator.state()
        restored = make_map(max_voxels=3)
        restored.restore(state)
        np.testing.assert_allclose(restored.points(), accumulator.points())

    def test_world_origin_padding_is_rejected_after_robot_moves(self):
        accumulator = make_map()
        accepted = accumulator.fuse(
            np.asarray([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]]),
            [1.0, 0.0, 0.0],
            10,
        )
        self.assertEqual(accepted, 1)
        np.testing.assert_allclose(accumulator.points(), [[1.2, 0.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
