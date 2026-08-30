import threading
import unittest
from collections import deque
from unittest import mock

import numpy as np

from go2_mapping_depthcam.depth_mapping_node import DepthMappingNode


class _Logger:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)


def _pose(x):
    result = np.eye(4, dtype=np.float64)
    result[0, 3] = float(x)
    return result


class RegistrationRecoveryTest(unittest.TestCase):
    def _node(self, points):
        node = DepthMappingNode.__new__(DepthMappingNode)
        node._lock = threading.RLock()
        node._map_lock = threading.RLock()
        node._registration_enabled = True
        node._registration_rate_hz = 2.0
        node._registration_voxel_size = 0.02
        node._registration_submap_frames = 5
        node._registration_max_source_points = 500
        node._registration_max_target_points = 500
        node._registration_max_correspondence = 0.14
        node._registration_iterations = 8
        node._registration_trim_fraction = 0.75
        node._registration_min_correspondences = 20
        node._registration_min_overlap = 0.35
        node._registration_max_rmse = 0.07
        node._registration_max_translation = 0.10
        node._registration_max_rotation_deg = 4.0
        node._registration_gain = 0.60
        node._registration_min_motion_translation = 0.015
        node._registration_min_motion_rotation_deg = 0.50
        node._registration_reseed_after_rejections = 3
        node._registration_normal_neighbours = 12
        node._registration_huber_delta = 0.03
        node._registration_damping = 1.0e-5
        node._registration_submap = deque(
            [points.copy()], maxlen=node._registration_submap_frames
        )
        node._last_registration_monotonic = 0.0
        node._last_registration_raw_pose = _pose(0.0)
        node._last_registration_correction = np.eye(4, dtype=np.float64)
        node._registration_attempts = 0
        node._registration_accepted = 0
        node._registration_rejected = 0
        node._registration_consecutive_rejections = 0
        node._registration_submap_reseeds = 0
        node._registration_state = "tracking"
        node._registration_last_rmse = None
        node._registration_last_overlap = None
        node._registration_last_translation = None
        node._registration_last_rotation_deg = None
        node._registration_last_reason = ""
        node._warn_throttled = mock.Mock()
        node.get_logger = mock.Mock(return_value=_Logger())
        return node

    def test_three_failures_reseed_stale_local_submap(self):
        rng = np.random.RandomState(12)
        points = rng.uniform([-1.0, -1.0, 0.0], [1.0, 1.0, 1.5], (120, 3))
        node = self._node(points)

        with mock.patch(
            "go2_mapping_depthcam.depth_mapping_node.register_rigid_scan",
            side_effect=ValueError("too little scan-to-submap overlap"),
        ):
            for index in range(1, 4):
                node._last_registration_monotonic = 0.0
                odom_positioned = points + [0.1 * index, 0.0, 0.0]
                fused_points, update_submap, fusion_allowed = node._maybe_register(
                    odom_positioned, _pose(0.1 * index)
                )
                self.assertFalse(update_submap)
                self.assertFalse(fusion_allowed)
                np.testing.assert_allclose(fused_points, odom_positioned)

        self.assertEqual(node._registration_rejected, 3)
        self.assertEqual(node._registration_consecutive_rejections, 0)
        self.assertEqual(node._registration_submap_reseeds, 1)
        self.assertEqual(node._registration_state, "reinitializing")
        self.assertEqual(len(node._registration_submap), 1)
        np.testing.assert_allclose(
            node._last_registration_raw_pose, _pose(0.3)
        )
        self.assertIn("submap reseeded", node._registration_last_reason)
        self.assertEqual(len(node.get_logger().warnings), 1)


if __name__ == "__main__":
    unittest.main()
