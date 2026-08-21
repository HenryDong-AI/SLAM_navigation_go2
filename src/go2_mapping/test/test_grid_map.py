import unittest

import numpy as np

from go2_mapping.grid_map import LogOddsGrid, bresenham_cells, occupancy_to_pgm


def make_grid():
    return LogOddsGrid(
        resolution=1.0,
        hit_log_odds=0.8,
        miss_log_odds=-0.4,
        min_log_odds=-2.0,
        max_log_odds=3.0,
        obstacle_min_height=0.2,
        obstacle_max_height=2.0,
        max_ray_range=10.0,
        max_cells=100,
        max_rays_per_update=10,
    )


class GridMapTest(unittest.TestCase):
    def test_bresenham_all_octants_and_endpoints(self):
        self.assertEqual(
            bresenham_cells((0, 0), (3, 1)),
            [(0, 0), (1, 0), (2, 1), (3, 1)],
        )
        self.assertEqual(
            bresenham_cells((3, 1), (0, 0)),
            [(3, 1), (2, 1), (1, 0), (0, 0)],
        )

    def test_ray_clearing_and_obstacle_height(self):
        grid = make_grid()
        free_count, hit_count = grid.update(
            np.asarray([[3.2, 0.2, 1.0], [2.0, 2.0, 0.05]]),
            [0.0, 0.0, 0.0],
            stamp_ns=100,
        )
        self.assertEqual(hit_count, 1)
        self.assertEqual(free_count, 5)
        self.assertAlmostEqual(grid.log_odds_at((3, 0)), 0.8)
        self.assertAlmostEqual(grid.log_odds_at((0, 0)), -0.4)
        self.assertAlmostEqual(grid.log_odds_at((2, 2)), -0.4)

    def test_hit_wins_over_free_ray_in_same_scan(self):
        grid = make_grid()
        grid.update(
            np.asarray([[2.2, 0.2, 1.0], [4.2, 0.2, 1.0]]),
            [0.0, 0.0, 0.0],
            stamp_ns=100,
        )
        self.assertAlmostEqual(grid.log_odds_at((2, 0)), 0.8)

    def test_dense_probability_and_trinary_pgm_encoding(self):
        occupancy = np.asarray([[-1, 0, 50], [100, 25, 65]], dtype=np.int8)
        encoded = occupancy_to_pgm(occupancy)
        raster = encoded.split(b"\n", 4)[-1]
        # PGM is vertically flipped: second occupancy row precedes the first.
        self.assertEqual(list(raster), [0, 254, 0, 205, 254, 205])

        grid = make_grid()
        grid.update(np.asarray([[2.2, 0.2, 1.0]]), [0, 0, 0], 100)
        dense, origin_x, origin_y, cropped = grid.to_dense(max_dense_cells=100)
        self.assertFalse(cropped)
        self.assertEqual((origin_x, origin_y), (-1.0, -1.0))
        self.assertTrue(((dense == -1) | ((dense >= 0) & (dense <= 100))).all())


if __name__ == "__main__":
    unittest.main()
