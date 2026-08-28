import math
import struct
import unittest

from go2_robot_bridge.safety import (
    PermanentFaultLatch,
    TimeSyncInterlock,
    TwistCommand,
    assess_front_cloud,
    clamp_twist,
    evaluate_motion_gate,
    front_sector_blocked,
    iter_xyz,
    source_timestamp_is_fresh,
    slew_twist,
)


FIELDS = [
    {"name": "x", "offset": 0, "datatype": 7, "count": 1},
    {"name": "y", "offset": 4, "datatype": 7, "count": 1},
    {"name": "z", "offset": 8, "datatype": 7, "count": 1},
    {"name": "intensity", "offset": 16, "datatype": 7, "count": 1},
    {"name": "ring", "offset": 20, "datatype": 4, "count": 1},
    {"name": "time", "offset": 24, "datatype": 7, "count": 1},
]


def make_cloud(points, *, bigendian=False, row_padding=0):
    point_step = 32
    row_step = len(points) * point_step + row_padding
    data = bytearray(row_step)
    prefix = ">" if bigendian else "<"
    for index, (x, y, z) in enumerate(points):
        struct.pack_into(prefix + "fff", data, index * point_step, x, y, z)
    return {
        "data": data,
        "width": len(points),
        "height": 1,
        "point_step": point_step,
        "row_step": row_step,
        "is_bigendian": bigendian,
        "fields": FIELDS,
    }


class VelocityHelperTests(unittest.TestCase):
    def test_clamp_rejects_nonfinite_values(self):
        result = clamp_twist(
            TwistCommand(math.nan, 2.0, -4.0),
            max_linear_x=0.4,
            max_linear_y=0.25,
            max_angular_z=0.6,
        )
        self.assertEqual(result, TwistCommand(0.0, 0.25, -0.6))

    def test_slew_limits_each_axis(self):
        result = slew_twist(
            TwistCommand(),
            TwistCommand(1.0, -1.0, 2.0),
            linear_rate=0.5,
            angular_rate=1.0,
            dt=0.2,
        )
        self.assertAlmostEqual(result.linear_x, 0.1)
        self.assertAlmostEqual(result.linear_y, -0.1)
        self.assertAlmostEqual(result.angular_z, 0.2)


class MotionGateTests(unittest.TestCase):
    def decision(self, **overrides):
        arguments = dict(
            armed=True,
            command=TwistCommand(0.1, 0.0, 0.0),
            now=10.0,
            last_command_at=9.9,
            command_timeout=0.5,
            require_fresh_odom=True,
            last_odom_at=9.9,
            odom_timeout=0.5,
            obstacle_guard_enabled=True,
            front_blocked=False,
            last_cloud_at=9.9,
            cloud_timeout=0.5,
            fail_closed_on_cloud_timeout=True,
        )
        arguments.update(overrides)
        return evaluate_motion_gate(**arguments)

    def test_disarmed_is_first_interlock(self):
        result = self.decision(armed=False, last_odom_at=None)
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "disarmed")

    def test_command_timeout_stops(self):
        result = self.decision(last_command_at=9.0)
        self.assertEqual(result.reason, "command_timeout")

    def test_stale_odometry_stops(self):
        result = self.decision(last_odom_at=9.0)
        self.assertEqual(result.reason, "odometry_stale")

    def test_front_obstacle_stops_forward(self):
        result = self.decision(front_blocked=True)
        self.assertEqual(result.reason, "front_obstacle")

    def test_stale_cloud_fails_closed_for_forward_only(self):
        result = self.decision(last_cloud_at=None)
        self.assertEqual(result.reason, "front_cloud_stale")
        reverse = self.decision(
            command=TwistCommand(-0.1, 0.0, 0.0),
            front_blocked=True,
            last_cloud_at=None,
        )
        self.assertFalse(reverse.allowed)

    def test_fresh_cloud_allows_reverse_escape_from_front_obstacle(self):
        reverse = self.decision(
            command=TwistCommand(-0.1, 0.0, 0.0),
            front_blocked=True,
            last_cloud_at=9.9,
        )
        self.assertTrue(reverse.allowed)

    def test_source_timestamp_rejects_replay_and_future(self):
        self.assertTrue(
            source_timestamp_is_fresh(
                now_nanoseconds=10_000_000_000,
                stamp_sec=9,
                stamp_nanosec=900_000_000,
                timeout_sec=0.5,
            )
        )
        self.assertFalse(
            source_timestamp_is_fresh(
                now_nanoseconds=10_000_000_000,
                stamp_sec=8,
                stamp_nanosec=0,
                timeout_sec=0.5,
            )
        )
        self.assertFalse(
            source_timestamp_is_fresh(
                now_nanoseconds=10_000_000_000,
                stamp_sec=11,
                stamp_nanosec=0,
                timeout_sec=0.5,
            )
        )

    def test_zero_command_uses_stop_path(self):
        result = self.decision(command=TwistCommand())
        self.assertEqual(result.reason, "zero_command")


class TimeSyncInterlockTests(unittest.TestCase):
    def status(self, state="locked", instance_id="one", epoch=0, **extra):
        payload = {
            "state": state,
            "instance_id": instance_id,
            "epoch": epoch,
        }
        payload.update(extra)
        return payload

    def test_warmup_then_locked_is_healthy(self):
        interlock = TimeSyncInterlock()
        self.assertEqual(interlock.update(self.status(state="warming")), "")
        self.assertFalse(interlock.ready)
        self.assertEqual(interlock.update(self.status()), "")
        self.assertTrue(interlock.ready)

    def test_fault_epoch_or_process_change_latches(self):
        for changed in (
            self.status(state="fault_latched", fault_reason="clock reset"),
            self.status(epoch=1),
            self.status(instance_id="two"),
        ):
            interlock = TimeSyncInterlock()
            interlock.update(self.status())
            self.assertTrue(interlock.update(changed))
            self.assertFalse(interlock.ready)
            self.assertTrue(interlock.fault_reason)

    def test_bridge_fault_reason_wins_when_epoch_changes_too(self):
        interlock = TimeSyncInterlock()
        interlock.update(self.status())
        reason = interlock.update(
            self.status(
                state="fault_latched",
                epoch=1,
                fault_reason="odometry translation step 0.800m exceeds 0.750m",
            )
        )
        self.assertEqual(
            reason, "odometry translation step 0.800m exceeds 0.750m"
        )

    def test_leaving_locked_state_and_bad_status_latch(self):
        interlock = TimeSyncInterlock()
        interlock.update(self.status())
        self.assertTrue(interlock.update(self.status(state="warming")))
        self.assertTrue(interlock.fault_reason)

        invalid = TimeSyncInterlock()
        self.assertTrue(invalid.update({"state": "locked"}))

    def test_permanent_motion_fault_keeps_first_reason(self):
        latch = PermanentFaultLatch()
        self.assertFalse(latch.faulted)
        self.assertEqual(latch.latch("odometry became stale"), "odometry became stale")
        self.assertTrue(latch.faulted)
        self.assertEqual(latch.latch("cloud became stale"), "odometry became stale")


class PointCloudTests(unittest.TestCase):
    def test_unitree_raw_layout_and_row_padding(self):
        cloud = make_cloud([(0.2, -0.1, 0.0), (1.0, 0.2, 0.3)], row_padding=16)
        points = list(iter_xyz(**cloud))
        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(points[0][0], 0.2)
        self.assertAlmostEqual(points[1][2], 0.3)

    def test_big_endian_cloud(self):
        cloud = make_cloud([(0.4, 0.1, -0.2)], bigendian=True)
        point = next(iter_xyz(**cloud))
        self.assertAlmostEqual(point[0], 0.4)
        self.assertAlmostEqual(point[2], -0.2)

    def test_rep103_floor_below_sensor_is_excluded(self):
        # Five clustered returns are obstacles. Floor returns at negative
        # z=-0.50, returns behind the robot, and wide returns must be ignored.
        points = [
            (0.40, 0.00, 0.05),
            (0.42, 0.05, 0.08),
            (0.44, -0.05, 0.10),
            (0.46, 0.10, -0.05),
            (0.48, -0.10, 0.15),
            (0.35, 0.00, -0.50),
            (-0.30, 0.00, 0.00),
            (0.40, 0.80, 0.00),
        ]
        cloud = make_cloud(points)
        blocked = front_sector_blocked(
            iter_xyz(**cloud),
            min_x=0.15,
            max_x=0.80,
            half_width=0.35,
            min_z=-0.22,
            max_z=0.50,
            min_points=5,
        )
        self.assertTrue(blocked)

    def test_floor_only_is_clear(self):
        cloud = make_cloud([(0.3, 0.0, -0.45)] * 10)
        blocked = front_sector_blocked(
            iter_xyz(**cloud),
            min_x=0.15,
            max_x=0.80,
            half_width=0.35,
            min_z=-0.22,
            max_z=0.50,
            min_points=5,
        )
        self.assertFalse(blocked)

    def _assess(self, points, health_min_points=3):
        cloud = make_cloud(points)
        return assess_front_cloud(
            iter_xyz(**cloud),
            min_x=0.15,
            max_x=0.80,
            half_width=0.35,
            min_z=-0.22,
            max_z=0.50,
            obstacle_min_points=5,
            health_min_points=health_min_points,
            health_min_range=0.10,
            health_max_range=30.0,
        )

    def test_open_but_plausible_cloud_is_healthy(self):
        observation = self._assess(
            [(1.5, -1.0, -0.4), (2.0, 0.8, 0.2), (-1.0, 0.0, -0.4)]
        )
        self.assertTrue(observation.healthy)
        self.assertFalse(observation.front_blocked)
        self.assertEqual(observation.plausible_points, 3)

    def test_empty_zero_nan_and_out_of_range_clouds_are_unhealthy(self):
        for points in (
            [],
            [(0.0, 0.0, 0.0)] * 20,
            [(math.nan, math.nan, math.nan)] * 20,
            [(100.0, 0.0, 0.0)] * 20,
        ):
            observation = self._assess(points)
            self.assertFalse(observation.healthy)
            self.assertFalse(observation.front_blocked)

    def test_wrong_xyz_type_is_rejected(self):
        cloud = make_cloud([(0.2, 0.0, 0.0)])
        cloud["fields"] = [dict(field) for field in FIELDS]
        cloud["fields"][0]["datatype"] = 4
        with self.assertRaises(ValueError):
            list(iter_xyz(**cloud))

    def test_vector_xyz_field_is_rejected(self):
        cloud = make_cloud([(0.2, 0.0, 0.0)])
        cloud["fields"] = [dict(field) for field in FIELDS]
        cloud["fields"][0]["count"] = 2
        with self.assertRaises(ValueError):
            list(iter_xyz(**cloud))

    def test_truncated_buffer_is_rejected(self):
        cloud = make_cloud([(0.2, 0.0, 0.0), (0.3, 0.0, 0.0)])
        cloud["data"] = cloud["data"][:-1]
        with self.assertRaises(ValueError):
            list(iter_xyz(**cloud))

    def test_short_row_step_is_rejected(self):
        cloud = make_cloud([(0.2, 0.0, 0.0), (0.3, 0.0, 0.0)])
        cloud["row_step"] = cloud["point_step"]
        with self.assertRaises(ValueError):
            list(iter_xyz(**cloud))


if __name__ == "__main__":
    unittest.main()
