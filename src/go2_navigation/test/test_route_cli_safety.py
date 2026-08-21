import math
from types import SimpleNamespace

import pytest

from go2_navigation.route_cli import (
    _cancel_response_ok,
    _fail_safe_abort,
    _is_terminal_action_status,
    _pose_values,
)


class _Point:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class _Quaternion:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.w = w


@pytest.mark.parametrize("status", [4, 5, 6])
def test_only_terminal_action_statuses_are_accepted(status):
    assert _is_terminal_action_status(status)


@pytest.mark.parametrize("status", [None, "bad", 0, 1, 2, 3, 7])
def test_nonterminal_action_statuses_are_rejected(status):
    assert not _is_terminal_action_status(status)


def test_pose_values_normalize_quaternion_before_yaw():
    yaw = 1.2
    quaternion = _Quaternion(
        z=3.0 * math.sin(yaw / 2.0),
        w=3.0 * math.cos(yaw / 2.0),
    )
    x, y, actual_yaw = _pose_values(_Point(1.0, -2.0, 0.4), quaternion)
    assert x == 1.0
    assert y == -2.0
    assert actual_yaw == pytest.approx(yaw)


@pytest.mark.parametrize(
    ("point", "quaternion"),
    [
        (_Point(float("nan"), 0.0, 0.0), _Quaternion()),
        (_Point(0.0, 0.0, float("inf")), _Quaternion()),
        (_Point(), _Quaternion(w=float("nan"))),
        (_Point(), _Quaternion(w=0.0)),
    ],
)
def test_invalid_recorded_poses_are_rejected(point, quaternion):
    with pytest.raises(ValueError):
        _pose_values(point, quaternion)


def test_cancel_response_requires_success_code():
    assert _cancel_response_ok(SimpleNamespace(return_code=0))
    assert not _cancel_response_ok(SimpleNamespace(return_code=1))
    assert not _cancel_response_ok(None)


class _Logger:
    def __init__(self):
        self.messages = []

    def error(self, message):
        self.messages.append(("error", message))

    def warn(self, message):
        self.messages.append(("warn", message))


class _Future:
    def __init__(self, value=None, ready=True):
        self.value = value
        self.ready = ready

    def done(self):
        return self.ready

    def result(self):
        if not self.ready:
            raise RuntimeError("future is not ready")
        return self.value


class _Node:
    def __init__(self):
        self.logger = _Logger()
        self.pending = []

    def get_logger(self):
        return self.logger


class _Rclpy:
    @staticmethod
    def ok():
        return True

    @staticmethod
    def spin_once(node, timeout_sec):
        del timeout_sec
        for future in node.pending:
            future.ready = True


class _ServiceType:
    class Request:
        pass


class _Client:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    @staticmethod
    def wait_for_service(timeout_sec):
        del timeout_sec
        return True

    def call_async(self, request):
        del request
        self.calls += 1
        return _Future(self.response)


class _Handle:
    def __init__(self, result_future, cancel_future):
        self.result_future = result_future
        self.cancel_future = cancel_future
        self.cancel_calls = 0

    def get_result_async(self):
        return self.result_future

    def cancel_goal_async(self):
        self.cancel_calls += 1
        return self.cancel_future


def _successful_stop_client():
    return _Client(SimpleNamespace(success=True, message="stopped"))


def _successful_cancel_client():
    return _Client(SimpleNamespace(return_code=0, goals_canceling=[]))


def test_fail_safe_abort_confirms_specific_goal_terminal_result():
    node = _Node()
    result_future = _Future(SimpleNamespace(status=5), ready=False)
    node.pending.append(result_future)
    handle = _Handle(
        result_future,
        _Future(SimpleNamespace(return_code=0, goals_canceling=[object()])),
    )
    cancel_all = _successful_cancel_client()
    stop = _successful_stop_client()

    confirmed = _fail_safe_abort(
        _Rclpy,
        node,
        handle,
        result_future,
        cancel_all,
        _ServiceType,
        stop,
        _ServiceType,
    )

    assert confirmed
    assert handle.cancel_calls == 1
    assert cancel_all.calls == 0
    assert stop.calls == 1


class _BrokenHandle:
    @staticmethod
    def get_result_async():
        raise RuntimeError("unknown goal")

    @staticmethod
    def cancel_goal_async():
        raise RuntimeError("unknown goal")


def test_fail_safe_abort_falls_back_to_cancel_all_and_still_latches_stop():
    node = _Node()
    cancel_all = _successful_cancel_client()
    stop = _successful_stop_client()

    confirmed = _fail_safe_abort(
        _Rclpy,
        node,
        _BrokenHandle(),
        None,
        cancel_all,
        _ServiceType,
        stop,
        _ServiceType,
    )

    assert not confirmed
    assert cancel_all.calls == 1
    assert stop.calls == 1
    assert any(
        "Could not confirm" in message
        for _, message in node.logger.messages
    )
