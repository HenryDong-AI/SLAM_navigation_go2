import unittest

import numpy as np

from go2_mapping.pointcloud import read_xyzrgb
from go2_mapping_depthcam.depth_camera_bridge import DepthCameraBridge


class AtomicCameraCloudTest(unittest.TestCase):
    def test_bridge_cloud_preserves_aligned_xyz_and_rgb(self):
        points = np.asarray(
            [[0.1, -0.2, 1.5], [0.4, 0.5, 2.0]], dtype=np.float64
        )
        colors = np.asarray([[255, 20, 3], [4, 50, 200]], dtype=np.uint8)

        message = DepthCameraBridge._point_cloud_message(points, colors)
        decoded_points, decoded_colors = read_xyzrgb(message)

        self.assertEqual(message.width, 2)
        self.assertEqual(message.point_step, 16)
        np.testing.assert_allclose(decoded_points, points, rtol=0.0, atol=1e-6)
        np.testing.assert_array_equal(decoded_colors, colors)


if __name__ == "__main__":
    unittest.main()
