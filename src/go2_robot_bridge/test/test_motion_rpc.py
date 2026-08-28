import os
import socket
import struct
import subprocess
import threading
import unittest
from types import SimpleNamespace

from go2_robot_bridge.motion_rpc import (
    MAX_FRAME_BYTES,
    MotionProtocolError,
    MotionRpcError,
    MotionSdkError,
    MotionWorkerProxy,
    recv_frame,
    send_frame,
    validate_move_arguments,
)
from go2_robot_bridge.motion_worker import (
    SportStateSnapshot,
    dispatch_request,
    prepare_motion_controller,
    prepare_robot_posture,
    prepare_velocity_client,
    serve_requests,
)


class FakeSport:
    def __init__(
        self,
        move_result=0,
        stop_result=0,
        damp_result=0,
        recovery_result=0,
        balance_result=0,
    ):
        self.move_result = move_result
        self.stop_result = stop_result
        self.damp_result = damp_result
        self.recovery_result = recovery_result
        self.balance_result = balance_result
        self.moves = []
        self.stop_calls = 0
        self.posture_calls = []

    def Move(self, linear_x, linear_y, angular_z):
        self.moves.append((linear_x, linear_y, angular_z))
        return self.move_result

    def StopMove(self):
        self.stop_calls += 1
        return self.stop_result

    def Damp(self):
        self.posture_calls.append("Damp")
        return self.damp_result

    def RecoveryStand(self):
        self.posture_calls.append("RecoveryStand")
        return self.recovery_result

    def BalanceStand(self):
        self.posture_calls.append("BalanceStand")
        return self.balance_result


class FakeAvoidance:
    states = [False]
    instances = []

    def __init__(self):
        self.timeout = None
        self.remote_calls = []
        self.switch_calls = []
        self.moves = []
        self._states = list(type(self).states)
        type(self).instances.append(self)

    def SetTimeout(self, timeout):
        self.timeout = timeout

    def Init(self):
        pass

    def SwitchGet(self):
        state = self._states.pop(0) if len(self._states) > 1 else self._states[0]
        return 0, state

    def SwitchSet(self, enabled):
        self.switch_calls.append(bool(enabled))
        return 0

    def UseRemoteCommandFromApi(self, enabled):
        self.remote_calls.append(bool(enabled))
        return 0

    def Move(self, linear_x, linear_y, angular_z):
        self.moves.append((linear_x, linear_y, angular_z))
        return 0


class FakeProcess:
    def __init__(self):
        self.running = True
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self.running else 0

    def wait(self, timeout):
        if self.running:
            raise subprocess.TimeoutExpired("fake-motion-worker", timeout)
        return 0

    def terminate(self):
        self.terminated = True
        self.running = False

    def kill(self):
        self.killed = True
        self.running = False


class FakeRobotState:
    def __init__(self, snapshots, switch_result=0):
        self.snapshots = list(snapshots)
        self.switch_result = switch_result
        self.switches = []

    def ServiceList(self):
        snapshot = (
            self.snapshots.pop(0)
            if len(self.snapshots) > 1
            else self.snapshots[0]
        )
        return 0, [
            SimpleNamespace(name=name, status=status) for name, status in snapshot
        ]

    def ServiceSwitch(self, name, enabled):
        self.switches.append((name, enabled))
        return self.switch_result


class FakeMotionSwitcher:
    selected_names = []
    instances = []

    def __init__(self):
        self.timeout = None
        self.select_calls = []
        self.names = list(type(self).selected_names)
        type(self).instances.append(self)

    def SetTimeout(self, timeout):
        self.timeout = timeout

    def Init(self):
        pass

    def CheckMode(self):
        name = self.names.pop(0) if len(self.names) > 1 else self.names[0]
        return 0, {"name": name}

    def SelectMode(self, name):
        self.select_calls.append(name)
        return 0, None


class FakeSportStateMonitor:
    def __init__(self, samples):
        self.samples = list(samples)
        self.after_sequences = []

    def wait_for_sample(self, _timeout_sec, *, after_sequence=-1):
        self.after_sequences.append(after_sequence)
        if not self.samples:
            raise RuntimeError("no fake sport state")
        return self.samples.pop(0)


class FrameProtocolTests(unittest.TestCase):
    def test_round_trip_uses_one_bounded_json_object(self):
        receiver, sender = socket.socketpair()
        try:
            payload = {"id": 7, "method": "StopMove", "args": []}
            send_frame(sender, payload)
            self.assertEqual(recv_frame(receiver), payload)
        finally:
            receiver.close()
            sender.close()

    def test_oversized_declared_frame_is_rejected_before_body_read(self):
        receiver, sender = socket.socketpair()
        try:
            sender.sendall(struct.pack("!I", MAX_FRAME_BYTES + 1))
            with self.assertRaises(MotionProtocolError):
                recv_frame(receiver)
        finally:
            receiver.close()
            sender.close()

    def test_nonfinite_and_boolean_move_values_are_rejected(self):
        self.assertEqual(validate_move_arguments([1, -0.2, 0.3]), (1.0, -0.2, 0.3))
        for arguments in ([float("nan"), 0.0, 0.0], [True, 0.0, 0.0], [0.0]):
            with self.assertRaises(MotionProtocolError):
                validate_move_arguments(arguments)


class WorkerDispatchTests(unittest.TestCase):
    def request(self, method, arguments):
        return {"id": 1, "method": method, "args": arguments}

    def test_dispatcher_exposes_only_move_and_stop_move(self):
        sport = FakeSport()
        moved = dispatch_request(
            sport,
            self.request("Move", [0.1, -0.2, 0.3]),
            controller="sport_mode",
        )
        stopped = dispatch_request(
            sport, self.request("StopMove", []), controller="sport_mode"
        )
        forbidden = dispatch_request(
            sport, self.request("StandUp", []), controller="sport_mode"
        )

        self.assertEqual(moved, {"id": 1, "ok": True, "code": 0})
        self.assertEqual(stopped, {"id": 1, "ok": True, "code": 0})
        self.assertFalse(forbidden["ok"])
        self.assertIn("unsupported motion method", forbidden["error"])
        self.assertEqual(sport.moves, [(0.1, -0.2, 0.3)])
        self.assertEqual(sport.stop_calls, 1)

    def test_mcf_stop_uses_zero_velocity_move(self):
        sport = FakeSport()

        stopped = dispatch_request(
            sport, self.request("StopMove", []), controller="mcf"
        )

        self.assertEqual(stopped, {"id": 1, "ok": True, "code": 0})
        self.assertEqual(sport.moves, [(0.0, 0.0, 0.0)])
        self.assertEqual(sport.stop_calls, 0)

    def test_invalid_request_never_reaches_sdk(self):
        sport = FakeSport()
        invalid = [
            self.request("Move", [float("inf"), 0.0, 0.0]),
            self.request("StopMove", [1]),
            {"id": 1, "method": "Move", "args": [0.0, 0.0, 0.0], "extra": 1},
            {"id": True, "method": "StopMove", "args": []},
        ]
        for request in invalid:
            response = dispatch_request(
                sport, request, controller="sport_mode"
            )
            self.assertFalse(response["ok"])
        self.assertEqual(sport.moves, [])
        self.assertEqual(sport.stop_calls, 0)

    def test_nonzero_sdk_code_is_an_explicit_error(self):
        response = dispatch_request(
            FakeSport(move_result=(42, "detail")),
            self.request("Move", [0.0, 0.0, 0.1]),
            controller="sport_mode",
        )
        self.assertFalse(response["ok"])
        self.assertEqual(response["code"], 42)

    def test_clean_parent_eof_triggers_best_effort_stop(self):
        parent, worker = socket.socketpair()
        sport = FakeSport()
        result = []
        thread = threading.Thread(
            target=lambda: result.append(
                serve_requests(worker, sport, controller="sport_mode")
            )
        )
        thread.start()
        parent.close()
        thread.join(timeout=1.0)
        worker.close()
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [0])
        self.assertEqual(sport.stop_calls, 1)


class FirmwareControllerTests(unittest.TestCase):
    def setUp(self):
        FakeMotionSwitcher.instances = []

    def test_running_mcf_is_preserved_and_verified(self):
        robot_state = FakeRobotState([[("sport_mode", 0), ("mcf", 0)]])
        FakeMotionSwitcher.selected_names = ["mcf"]

        controller = prepare_motion_controller(
            robot_state,
            FakeMotionSwitcher,
            controller_rpc_timeout_sec=5.0,
            transition_timeout_sec=0.1,
        )

        self.assertEqual(controller, "mcf")
        self.assertEqual(robot_state.switches, [])
        self.assertEqual(FakeMotionSwitcher.instances[0].select_calls, [])

    def test_stopped_mcf_is_started_and_selected(self):
        robot_state = FakeRobotState(
            [[("sport_mode", 0), ("mcf", 1)], [("sport_mode", 0), ("mcf", 0)]],
            switch_result=5201,
        )
        FakeMotionSwitcher.selected_names = ["legacy", "mcf"]

        controller = prepare_motion_controller(
            robot_state,
            FakeMotionSwitcher,
            controller_rpc_timeout_sec=5.0,
            transition_timeout_sec=0.1,
            poll_interval_sec=0.01,
        )

        self.assertEqual(controller, "mcf")
        self.assertEqual(robot_state.switches, [("mcf", True)])
        self.assertEqual(FakeMotionSwitcher.instances[0].select_calls, ["mcf"])

    def test_legacy_firmware_uses_sport_mode_without_motion_switcher(self):
        robot_state = FakeRobotState([[("sport_mode", 0)]])
        FakeMotionSwitcher.selected_names = ["unused"]

        controller = prepare_motion_controller(
            robot_state,
            FakeMotionSwitcher,
            controller_rpc_timeout_sec=5.0,
            transition_timeout_sec=0.1,
        )

        self.assertEqual(controller, "sport_mode")
        self.assertEqual(FakeMotionSwitcher.instances, [])


class VelocityClientTests(unittest.TestCase):
    def setUp(self):
        FakeAvoidance.instances = []

    def test_disabled_avoidance_keeps_direct_sport_client(self):
        FakeAvoidance.states = [False]
        sport = FakeSport()

        client, owner, backend = prepare_velocity_client(
            sport,
            FakeAvoidance,
            rpc_timeout_sec=1.0,
            sleep=lambda _seconds: None,
        )

        self.assertIs(client, sport)
        self.assertIsNone(owner)
        self.assertEqual(backend, "sport")
        self.assertEqual(FakeAvoidance.instances[0].switch_calls, [])

    def test_enabled_avoidance_is_stopped_disabled_and_sport_is_selected(self):
        FakeAvoidance.states = [True, True, False]
        sport = FakeSport()
        sleeps = []

        client, owner, backend = prepare_velocity_client(
            sport,
            FakeAvoidance,
            rpc_timeout_sec=1.0,
            transition_timeout_sec=1.0,
            poll_interval_sec=0.1,
            sleep=sleeps.append,
        )

        avoidance = FakeAvoidance.instances[0]
        self.assertIs(client, sport)
        self.assertIs(owner, avoidance)
        self.assertEqual(backend, "sport_direct")
        self.assertEqual(avoidance.remote_calls, [True, False])
        self.assertEqual(avoidance.moves, [(0.0, 0.0, 0.0)])
        self.assertEqual(avoidance.switch_calls, [False])
        self.assertEqual(sleeps, [0.5, 0.1, 0.5])


class PosturePreparationTests(unittest.TestCase):
    @staticmethod
    def sample(sequence, height, error_code=0):
        return SportStateSnapshot(
            sequence=sequence,
            mode=0,
            gait_type=0,
            body_height=height,
            error_code=error_code,
        )

    def test_standing_robot_is_verified_without_posture_commands(self):
        sport = FakeSport()
        monitor = FakeSportStateMonitor([self.sample(7, 0.322, 100)])

        final, prepared = prepare_robot_posture(
            sport,
            monitor,
            auto_prepare=True,
            standing_min_body_height=0.25,
            state_timeout_sec=1.0,
            sleep=lambda _seconds: None,
        )

        self.assertFalse(prepared)
        self.assertAlmostEqual(final.body_height, 0.322)
        self.assertEqual(sport.posture_calls, [])

    def test_low_robot_runs_proven_recovery_sequence_and_verifies_height(self):
        sport = FakeSport()
        monitor = FakeSportStateMonitor(
            [self.sample(4, 0.107, 1001), self.sample(9, 0.321, 100)]
        )
        sleeps = []

        final, prepared = prepare_robot_posture(
            sport,
            monitor,
            auto_prepare=True,
            standing_min_body_height=0.25,
            state_timeout_sec=1.0,
            sleep=sleeps.append,
        )

        self.assertTrue(prepared)
        self.assertAlmostEqual(final.body_height, 0.321)
        self.assertEqual(
            sport.posture_calls,
            ["Damp", "RecoveryStand", "BalanceStand"],
        )
        self.assertEqual(sleeps, [1.0, 4.0, 1.5])
        self.assertEqual(monitor.after_sequences, [-1, 4])

    def test_low_robot_fails_closed_when_auto_prepare_is_disabled(self):
        sport = FakeSport()
        monitor = FakeSportStateMonitor([self.sample(1, 0.10, 1001)])

        with self.assertRaisesRegex(RuntimeError, "below standing minimum"):
            prepare_robot_posture(
                sport,
                monitor,
                auto_prepare=False,
                standing_min_body_height=0.25,
                state_timeout_sec=1.0,
                sleep=lambda _seconds: None,
            )

        self.assertEqual(sport.posture_calls, [])

    def test_failed_height_verification_rejects_worker_startup(self):
        sport = FakeSport()
        monitor = FakeSportStateMonitor(
            [self.sample(1, 0.10, 1001), self.sample(2, 0.12, 1001)]
        )

        with self.assertRaisesRegex(RuntimeError, "remains below"):
            prepare_robot_posture(
                sport,
                monitor,
                auto_prepare=True,
                standing_min_body_height=0.25,
                state_timeout_sec=1.0,
                sleep=lambda _seconds: None,
            )


class ProxyTests(unittest.TestCase):
    def test_proxy_round_trips_move_and_stop(self):
        client, server = socket.socketpair()
        process = FakeProcess()
        observed = []

        def respond():
            for _ in range(2):
                request = recv_frame(server)
                observed.append(request)
                send_frame(
                    server,
                    {"id": request["id"], "ok": True, "code": 0},
                )

        thread = threading.Thread(target=respond)
        thread.start()
        proxy = MotionWorkerProxy(client, process, 0.2, 0.01)
        try:
            self.assertEqual(proxy.move(0.1, -0.1, 0.2), 0)
            self.assertEqual(proxy.stop_move(), 0)
        finally:
            proxy.close()
            server.close()
            thread.join(timeout=1.0)

        self.assertEqual([item["method"] for item in observed], ["Move", "StopMove"])
        self.assertTrue(process.terminated)

    def test_proxy_rejects_sdk_error_and_response_id_mismatch(self):
        for response_builder, expected in (
            (
                lambda request: {
                    "id": request["id"],
                    "ok": False,
                    "code": 9,
                    "error": "SDK rejected command",
                },
                MotionSdkError,
            ),
            (
                lambda request: {"id": request["id"] + 1, "ok": True, "code": 0},
                MotionProtocolError,
            ),
            (
                lambda request: {"id": request["id"], "ok": True, "code": True},
                MotionProtocolError,
            ),
        ):
            client, server = socket.socketpair()
            process = FakeProcess()

            def respond_once():
                request = recv_frame(server)
                send_frame(server, response_builder(request))

            thread = threading.Thread(target=respond_once)
            thread.start()
            proxy = MotionWorkerProxy(client, process, 0.2, 0.01)
            try:
                with self.assertRaises(expected):
                    proxy.stop_move()
            finally:
                proxy.close()
                server.close()
                thread.join(timeout=1.0)

    def test_start_uses_module_exec_and_one_inherited_private_fd(self):
        captured = {}

        def fake_popen(command, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            duplicate = socket.socket(fileno=os.dup(kwargs["pass_fds"][0]))
            send_frame(
                duplicate,
                {
                    "type": "ready",
                    "ok": True,
                    "controller": "mcf",
                    "posture_prepared": False,
                    "body_height": 0.322,
                    "sport_mode": 0,
                    "gait_type": 0,
                    "sport_error_code": 100,
                },
            )
            duplicate.close()
            process = FakeProcess()
            captured["process"] = process
            return process

        proxy = MotionWorkerProxy.start(
            network_interface="eth0",
            sdk_timeout_sec=1.0,
            startup_timeout_sec=0.2,
            rpc_timeout_sec=0.2,
            reap_timeout_sec=0.01,
            sdk_python_path="/sdk",
            cyclonedds_python_path="/dds",
            popen_factory=fake_popen,
        )
        try:
            command = captured["command"]
            self.assertIn("-m", command)
            self.assertIn("go2_robot_bridge.motion_worker", command)
            self.assertNotIn("StandUp", " ".join(command))
            self.assertIn("--controller-rpc-timeout-sec", command)
            self.assertIn("--controller-transition-timeout-sec", command)
            self.assertIn("--auto-prepare-posture", command)
            self.assertIn("--sport-state-timeout-sec", command)
            self.assertIn("--standing-min-body-height", command)
            self.assertEqual(proxy.controller, "mcf")
            self.assertFalse(proxy.posture_prepared)
            self.assertAlmostEqual(proxy.body_height, 0.322)
            self.assertEqual(proxy.sport_error_code, 100)
            self.assertEqual(len(captured["kwargs"]["pass_fds"]), 1)
            self.assertTrue(captured["kwargs"]["close_fds"])
            self.assertNotIn("shell", captured["kwargs"])
        finally:
            proxy.close()
        self.assertTrue(captured["process"].terminated)

    def test_startup_failure_closes_socket_and_reaps_process(self):
        captured = {}

        def fake_popen(_command, **kwargs):
            duplicate = socket.socket(fileno=os.dup(kwargs["pass_fds"][0]))
            send_frame(
                duplicate,
                {"type": "ready", "ok": False, "error": "SDK init failed"},
            )
            duplicate.close()
            process = FakeProcess()
            captured["process"] = process
            return process

        with self.assertRaisesRegex(MotionRpcError, "SDK init failed"):
            MotionWorkerProxy.start(
                network_interface="eth0",
                sdk_timeout_sec=1.0,
                startup_timeout_sec=0.2,
                rpc_timeout_sec=0.2,
                reap_timeout_sec=0.01,
                popen_factory=fake_popen,
            )
        self.assertTrue(captured["process"].terminated)


if __name__ == "__main__":
    unittest.main()
