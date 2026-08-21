import os
import socket
import struct
import subprocess
import threading
import unittest

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
from go2_robot_bridge.motion_worker import dispatch_request, serve_requests


class FakeSport:
    def __init__(self, move_result=0, stop_result=0):
        self.move_result = move_result
        self.stop_result = stop_result
        self.moves = []
        self.stop_calls = 0

    def Move(self, linear_x, linear_y, angular_z):
        self.moves.append((linear_x, linear_y, angular_z))
        return self.move_result

    def StopMove(self):
        self.stop_calls += 1
        return self.stop_result


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
        moved = dispatch_request(sport, self.request("Move", [0.1, -0.2, 0.3]))
        stopped = dispatch_request(sport, self.request("StopMove", []))
        forbidden = dispatch_request(sport, self.request("StandUp", []))

        self.assertEqual(moved, {"id": 1, "ok": True, "code": 0})
        self.assertEqual(stopped, {"id": 1, "ok": True, "code": 0})
        self.assertFalse(forbidden["ok"])
        self.assertIn("unsupported motion method", forbidden["error"])
        self.assertEqual(sport.moves, [(0.1, -0.2, 0.3)])
        self.assertEqual(sport.stop_calls, 1)

    def test_invalid_request_never_reaches_sdk(self):
        sport = FakeSport()
        invalid = [
            self.request("Move", [float("inf"), 0.0, 0.0]),
            self.request("StopMove", [1]),
            {"id": 1, "method": "Move", "args": [0.0, 0.0, 0.0], "extra": 1},
            {"id": True, "method": "StopMove", "args": []},
        ]
        for request in invalid:
            response = dispatch_request(sport, request)
            self.assertFalse(response["ok"])
        self.assertEqual(sport.moves, [])
        self.assertEqual(sport.stop_calls, 0)

    def test_nonzero_sdk_code_is_an_explicit_error(self):
        response = dispatch_request(
            FakeSport(move_result=(42, "detail")),
            self.request("Move", [0.0, 0.0, 0.1]),
        )
        self.assertFalse(response["ok"])
        self.assertEqual(response["code"], 42)

    def test_clean_parent_eof_triggers_best_effort_stop(self):
        parent, worker = socket.socketpair()
        sport = FakeSport()
        result = []
        thread = threading.Thread(
            target=lambda: result.append(serve_requests(worker, sport))
        )
        thread.start()
        parent.close()
        thread.join(timeout=1.0)
        worker.close()
        self.assertFalse(thread.is_alive())
        self.assertEqual(result, [0])
        self.assertEqual(sport.stop_calls, 1)


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
            send_frame(duplicate, {"type": "ready", "ok": True})
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
