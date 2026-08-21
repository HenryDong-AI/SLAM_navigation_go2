import time
import unittest
from types import SimpleNamespace

from go2_robot_bridge.motion_bridge import MotionBridge
from go2_robot_bridge.motion_rpc import MotionRpcError
from go2_robot_bridge.safety import PermanentFaultLatch, TwistCommand


class NullLogger:
    def __init__(self):
        self.errors = []

    def error(self, message):
        self.errors.append(message)

    def warning(self, _message):
        pass

    def info(self, _message):
        pass


class FailingStopWorker:
    alive = True

    def __init__(self):
        self.closed = False
        self.stop_calls = 0

    def stop_move(self):
        self.stop_calls += 1
        raise MotionRpcError("unconfirmed")

    def close(self):
        self.closed = True


class FailureHarness:
    _send_stop = MotionBridge._send_stop
    _worker_failure = MotionBridge._worker_failure
    _close_worker = MotionBridge._close_worker

    def __init__(self, worker):
        self._motion_worker = worker
        self._worker_was_armed = True
        self._armed = True
        self._target = TwistCommand(0.1, 0.0, 0.0)
        self._output = TwistCommand(0.1, 0.0, 0.0)
        self._last_command_at = time.monotonic()
        self._last_stop_at = -1.0e9
        self._stop_repeat = 0.2
        self._reconnect_delay = 2.0
        self._next_connect_at = 0.0
        self._safety_fault = PermanentFaultLatch()
        self.logger = NullLogger()

    def get_logger(self):
        return self.logger


class EnableHarness:
    _enable_callback = MotionBridge._enable_callback
    _enable_callback_locked = MotionBridge._enable_callback_locked

    def __init__(self, odom_ready):
        self._state_lock = __import__("threading").RLock()
        self._armed = False
        self._worker_was_armed = False
        self._motion_worker = None
        self._target = TwistCommand()
        self._output = TwistCommand()
        self._last_command_at = None
        self._post_worker_refresh_after = None
        self._post_worker_refresh_deadline = None
        self._post_worker_refresh_timeout = 2.0
        self._safety_fault = PermanentFaultLatch()
        self._time_sync_interlock = SimpleNamespace(fault_reason="")
        self._require_odom = True
        self._front_guard = False
        self._cloud_fail_closed = True
        self._require_cloud_to_arm = True
        self.odom_ready = odom_ready
        self.start_calls = 0
        self.stop_calls = 0
        self.logger = NullLogger()

    def _time_sync_is_fresh_and_locked(self, _now):
        return True

    def _odom_is_fresh(self, _now):
        return self.odom_ready

    def _cloud_is_fresh(self, _now):
        return True

    def _start_worker_if_due(self, force=False):
        self.start_calls += 1
        return force

    def _send_stop(self, force=False):
        self.stop_calls += 1
        return force

    def get_logger(self):
        return self.logger


class RefreshGateHarness:
    _control_tick = MotionBridge._control_tick
    _control_tick_locked = MotionBridge._control_tick_locked

    def __init__(self):
        self._state_lock = __import__("threading").RLock()
        now = time.monotonic()
        self._shutdown_started = False
        self._control_deadline_announced = False
        self._control_rate = 20.0
        self._last_control_at = now
        self._armed = True
        self._post_worker_refresh_after = now - 0.25
        self._post_worker_refresh_deadline = now + 1.0
        self._target = TwistCommand(0.1, 0.0, 0.0)
        self._output = TwistCommand(0.1, 0.0, 0.0)
        self._last_command_at = now
        self._last_gate_reason = "startup"
        self.stop_calls = 0
        self.faults = []
        self.logger = NullLogger()

    def _post_worker_sensors_refreshed(self, _now):
        return False

    def _send_stop(self, force=False):
        self.stop_calls += 1
        return not force

    def _disarm_for_fault(self, reason, permanent=False):
        self._armed = False
        self.faults.append((reason, permanent))

    def get_logger(self):
        return self.logger


class MotionBridgeIsolationTests(unittest.TestCase):
    def test_unconfirmed_stop_after_arm_latches_and_reaps(self):
        worker = FailingStopWorker()
        bridge = FailureHarness(worker)

        self.assertFalse(bridge._send_stop(force=True))
        self.assertFalse(bridge._armed)
        self.assertTrue(bridge._safety_fault.faulted)
        self.assertIn("StopMove failed", bridge._safety_fault.reason)
        self.assertIsNone(bridge._motion_worker)
        self.assertTrue(worker.closed)

    def test_worker_start_occurs_only_after_enable_guards_pass(self):
        request = SimpleNamespace(data=True)

        blocked = EnableHarness(odom_ready=False)
        blocked_response = SimpleNamespace(success=None, message="")
        blocked._enable_callback(request, blocked_response)
        self.assertFalse(blocked_response.success)
        self.assertEqual(blocked.start_calls, 0)
        self.assertEqual(blocked.stop_calls, 0)

        ready = EnableHarness(odom_ready=True)
        ready_response = SimpleNamespace(success=None, message="")
        ready._enable_callback(request, ready_response)
        self.assertTrue(ready_response.success)
        self.assertEqual(ready.start_calls, 1)
        self.assertEqual(ready.stop_calls, 1)
        self.assertTrue(ready._armed)
        self.assertTrue(ready._worker_was_armed)
        self.assertIsNotNone(ready._post_worker_refresh_after)
        self.assertIn("fresh post-start sensors", ready_response.message)

    def test_post_worker_refresh_rejects_pre_start_samples(self):
        now = time.monotonic()
        bridge = SimpleNamespace(
            _post_worker_refresh_after=now - 0.25,
            _require_time_sync=True,
            _last_time_sync_status_at=now - 0.50,
            _require_odom=True,
            _last_odom_at=now,
            _front_guard=True,
            _cloud_fail_closed=True,
            _last_cloud_at=now,
            _time_sync_is_fresh_and_locked=lambda _now: True,
            _odom_is_fresh=lambda _now: True,
            _cloud_is_fresh=lambda _now: True,
        )
        refreshed = MotionBridge._post_worker_sensors_refreshed(bridge, now)
        self.assertFalse(refreshed)

        bridge._last_time_sync_status_at = now
        refreshed = MotionBridge._post_worker_sensors_refreshed(bridge, now)
        self.assertTrue(refreshed)

    def test_post_worker_refresh_holds_stop_then_fails_closed(self):
        bridge = RefreshGateHarness()

        bridge._control_tick()
        self.assertTrue(bridge._armed)
        self.assertEqual(bridge.stop_calls, 1)
        self.assertEqual(bridge.faults, [])
        self.assertTrue(bridge._target.is_zero())
        self.assertIsNone(bridge._last_command_at)

        bridge._post_worker_refresh_deadline = time.monotonic() - 0.1
        bridge._control_tick()
        self.assertFalse(bridge._armed)
        self.assertEqual(len(bridge.faults), 1)
        self.assertTrue(bridge.faults[0][1])


if __name__ == "__main__":
    unittest.main()
