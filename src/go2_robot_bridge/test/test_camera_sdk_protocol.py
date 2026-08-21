import socket
import threading
import unittest

from go2_robot_bridge.camera_sdk_process import ros_stamp_from_monotonic_capture
from go2_robot_bridge.camera_sdk_process import CameraSdkProcess
from go2_robot_bridge.camera_sdk_worker import (
    MAX_PAYLOAD_BYTES,
    PACKET_FRAME,
    PACKET_HEADER,
    PACKET_STATUS,
    receive_packet,
    send_packet,
)


class CameraSdkProtocolTest(unittest.TestCase):
    def test_frame_and_status_round_trip(self):
        left, right = socket.socketpair()
        try:
            send_packet(left, PACKET_FRAME, 1234, b"jpeg")
            self.assertEqual(receive_packet(right), (PACKET_FRAME, 1234, b"jpeg"))
            send_packet(left, PACKET_STATUS, 2345, b"ready")
            self.assertEqual(
                receive_packet(right), (PACKET_STATUS, 2345, b"ready")
            )
        finally:
            left.close()
            right.close()

    def test_fragmented_packet_is_reassembled(self):
        left, right = socket.socketpair()
        packet = PACKET_HEADER.pack(PACKET_FRAME, 99, 6) + b"abcdef"

        def writer():
            try:
                for byte in packet:
                    left.sendall(bytes((byte,)))
            finally:
                left.close()

        thread = threading.Thread(target=writer)
        thread.start()
        try:
            self.assertEqual(receive_packet(right), (PACKET_FRAME, 99, b"abcdef"))
        finally:
            right.close()
            thread.join()

    def test_invalid_kind_and_oversize_are_rejected(self):
        left, right = socket.socketpair()
        try:
            left.sendall(PACKET_HEADER.pack(255, 0, 0))
            with self.assertRaisesRegex(ValueError, "unknown"):
                receive_packet(right)
        finally:
            left.close()
            right.close()

        left, right = socket.socketpair()
        try:
            left.sendall(PACKET_HEADER.pack(PACKET_FRAME, 0, MAX_PAYLOAD_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "exceeds"):
                receive_packet(right)
        finally:
            left.close()
            right.close()

    def test_truncated_payload_returns_eof(self):
        left, right = socket.socketpair()
        left.sendall(PACKET_HEADER.pack(PACKET_FRAME, 1, 5) + b"xx")
        left.close()
        try:
            self.assertIsNone(receive_packet(right))
        finally:
            right.close()

    def test_monotonic_capture_translation(self):
        self.assertEqual(
            ros_stamp_from_monotonic_capture(5_000, 2_000, 1_500, -100),
            4_400,
        )
        self.assertEqual(
            ros_stamp_from_monotonic_capture(5_000, 2_000, 2_050, 0),
            5_000,
        )

    def test_process_close_clears_cached_generation(self):
        process = CameraSdkProcess("eth0", "/sdk", "/dds", 1.0, 5.0, 1.0, 3)
        process._latest_frame = (1, 2, b"old", 3)
        process._latest_status = (1, 3, "old")
        process._reader_error = "old"
        process.close()
        self.assertIsNone(process.latest_frame_after(0))
        self.assertIsNone(process.latest_status_after(0))
        self.assertEqual(process.reader_error, "")
        self.assertFalse(process.reader_alive)


if __name__ == "__main__":
    unittest.main()
