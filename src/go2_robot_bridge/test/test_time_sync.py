import unittest

from go2_robot_bridge.time_sync import (
    OdomPoseGuard,
    SharedTimeEstimator,
    TimeSyncReset,
)


class SharedTimeEstimatorTest(unittest.TestCase):
    def make_estimator(self, warmup=3, jump=1000):
        return SharedTimeEstimator(
            warmup_samples=warmup,
            window_size=max(5, warmup),
            clock_jump_threshold_ns=jump,
            future_tolerance_ns=1000,
            max_output_age_ns=10000,
        )

    def test_warmup_minimum_is_shared_and_preserves_relative_time(self):
        estimator = self.make_estimator()
        self.assertIsNone(estimator.process_odometry(1000, 1120))
        self.assertIsNone(estimator.process_odometry(1100, 1180))
        self.assertEqual(estimator.process_odometry(1200, 1310), 1280)
        self.assertTrue(estimator.ready)
        self.assertEqual(estimator.offset_ns, 80)

        base_stamp = estimator.process_sensor("cloud_base", 1150, 1300)
        deskewed_stamp = estimator.process_sensor(
            "cloud_deskewed", 1170, 1320
        )
        self.assertEqual(base_stamp, 1230)
        self.assertEqual(deskewed_stamp, 1250)
        self.assertEqual(deskewed_stamp - base_stamp, 20)

    def test_rolling_minimum_updates_from_odometry_only(self):
        estimator = self.make_estimator(warmup=2)
        estimator.process_odometry(1000, 1100)
        self.assertEqual(estimator.process_odometry(1100, 1180), 1180)
        self.assertEqual(estimator.offset_ns, 80)
        self.assertEqual(estimator.process_sensor("cloud_base", 1150, 1250), 1230)
        self.assertEqual(estimator.process_odometry(1300, 1360), 1360)
        self.assertEqual(estimator.offset_ns, 60)

    def test_regression_rejects_message_and_restarts_warmup(self):
        estimator = self.make_estimator(warmup=2)
        estimator.process_odometry(1000, 1100)
        estimator.process_odometry(1100, 1200)
        with self.assertRaises(TimeSyncReset):
            estimator.process_odometry(1100, 1210)
        self.assertFalse(estimator.ready)
        self.assertEqual(estimator.reset_count, 1)
        self.assertEqual(estimator.warmup_collected, 0)
        self.assertIsNone(estimator.process_odometry(2000, 2100))
        self.assertEqual(estimator.process_odometry(2100, 2200), 2200)

    def test_zero_stamp_and_clock_step_fail_closed(self):
        estimator = self.make_estimator(warmup=1, jump=500)
        self.assertEqual(estimator.process_odometry(1000, 1100), 1100)
        with self.assertRaises(TimeSyncReset):
            estimator.process_sensor("cloud_base", 0, 1150)
        self.assertFalse(estimator.ready)

        self.assertEqual(estimator.process_odometry(2000, 2100), 2100)
        with self.assertRaises(TimeSyncReset):
            estimator.process_odometry(3000, 2200)
        self.assertFalse(estimator.ready)
        self.assertEqual(estimator.reset_count, 2)

    def test_offset_change_cannot_make_output_non_monotonic(self):
        estimator = self.make_estimator(warmup=1, jump=2000)
        estimator.process_odometry(1000, 1100)
        self.assertEqual(estimator.process_sensor("cloud_base", 1000, 1200), 1100)
        # A lower rolling-minimum offset remains acceptable for odometry, but it
        # would move the next cloud behind its prior normalized output.
        estimator.process_odometry(2000, 2000)
        with self.assertRaises(TimeSyncReset):
            estimator.process_sensor("cloud_base", 1050, 2050)
        self.assertFalse(estimator.ready)
        self.assertIn("monotonic", estimator.last_reset_reason)


class OdomPoseGuardTest(unittest.TestCase):
    def test_normal_motion_and_quaternion_sign_flip_are_accepted(self):
        guard = OdomPoseGuard()
        guard.observe(1_000_000_000, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        guard.observe(
            1_100_000_000,
            (0.1, 0.0, 0.0),
            (0.0, 0.0, 0.0, -1.0),
        )

    def test_translation_jump_and_rate_are_rejected(self):
        guard = OdomPoseGuard()
        guard.observe(1_000_000_000, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        with self.assertRaisesRegex(ValueError, "translation"):
            guard.observe(
                1_010_000_000,
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )

    def test_angular_jump_is_rejected_geodesically(self):
        guard = OdomPoseGuard(max_angular_step=0.5)
        guard.observe(1_000_000_000, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        with self.assertRaisesRegex(ValueError, "angular"):
            guard.observe(
                1_100_000_000,
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.5, 0.866025403784),
            )


if __name__ == "__main__":
    unittest.main()
