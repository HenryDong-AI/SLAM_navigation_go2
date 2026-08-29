import threading
import unittest

from go2_mapping_depthcam.depth_mapping_node import DepthMappingNode


class LatestFrameProcessingTest(unittest.TestCase):
    def _node(self):
        node = DepthMappingNode.__new__(DepthMappingNode)
        node._lock = threading.RLock()
        node._map_lock = threading.RLock()
        node._latest_camera_cloud = None
        node._processing_wake = threading.Event()
        node._processing_stop = threading.Event()
        node._processing_lock = threading.Lock()
        node._processing_active = False
        node._processing_rate_hz = 0.0
        node._last_process_monotonic = 0.0
        node._frames_received = 0
        node._frames_dropped = 0
        node._frames_superseded = 0
        node._last_error = ""
        return node

    def test_new_cloud_replaces_unprocessed_cloud(self):
        node = self._node()
        first = object()
        newest = object()

        node._queue_latest_camera_cloud(first)
        node._queue_latest_camera_cloud(newest)

        self.assertTrue(node._processing_wake.is_set())
        self.assertEqual(node._frames_received, 2)
        self.assertEqual(node._frames_dropped, 1)
        self.assertEqual(node._frames_superseded, 1)
        self.assertIs(node._take_latest_camera_cloud(), newest)
        self.assertFalse(node._processing_wake.is_set())

    def test_taken_cloud_does_not_make_next_cloud_a_drop(self):
        node = self._node()
        first = object()
        second = object()

        node._queue_latest_camera_cloud(first)
        self.assertIs(node._take_latest_camera_cloud(), first)
        node._queue_latest_camera_cloud(second)

        self.assertEqual(node._frames_dropped, 0)
        self.assertEqual(node._frames_superseded, 0)
        self.assertIs(node._take_latest_camera_cloud(), second)

    def test_map_lock_cannot_block_camera_intake(self):
        node = self._node()
        map_locked = threading.Event()
        release_map = threading.Event()
        intake_done = threading.Event()

        def hold_map_lock():
            with node._map_lock:
                map_locked.set()
                release_map.wait(timeout=1.0)

        holder = threading.Thread(target=hold_map_lock)
        holder.start()
        self.assertTrue(map_locked.wait(timeout=1.0))
        intake = threading.Thread(
            target=lambda: (
                node._queue_latest_camera_cloud(object()),
                intake_done.set(),
            )
        )
        intake.start()
        self.assertTrue(intake_done.wait(timeout=0.2))
        release_map.set()
        intake.join(timeout=1.0)
        holder.join(timeout=1.0)
        self.assertFalse(intake.is_alive())
        self.assertFalse(holder.is_alive())

    def test_worker_consumes_only_newest_atomic_cloud(self):
        node = self._node()
        first = object()
        newest = object()
        processed = []
        processed_event = threading.Event()

        def process(message):
            processed.append(message)
            processed_event.set()

        node._process_camera_cloud = process
        node._warn_throttled = lambda *args, **kwargs: None
        node._queue_latest_camera_cloud(first)
        node._queue_latest_camera_cloud(newest)

        worker = threading.Thread(target=node._processing_loop)
        worker.start()
        self.assertTrue(processed_event.wait(timeout=1.0))
        node._processing_stop.set()
        node._processing_wake.set()
        worker.join(timeout=1.0)

        self.assertFalse(worker.is_alive())
        self.assertEqual(processed, [newest])


if __name__ == "__main__":
    unittest.main()
