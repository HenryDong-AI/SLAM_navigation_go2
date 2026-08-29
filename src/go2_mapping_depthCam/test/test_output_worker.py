import threading
import unittest

from go2_mapping_depthcam.depth_mapping_node import DepthMappingNode


class OutputWorkerTest(unittest.TestCase):
    def _node(self):
        node = DepthMappingNode.__new__(DepthMappingNode)
        node._output_request_lock = threading.Lock()
        node._output_wake = threading.Event()
        node._output_stop = threading.Event()
        node._map_publish_requested = False
        node._autosave_requested = False
        node._map_publish_coalesced = 0
        node._autosave_coalesced = 0
        return node

    def test_repeated_timer_requests_coalesce_to_one_publish(self):
        node = self._node()
        node._request_map_publish()
        node._request_map_publish()
        publish, autosave = node._take_output_requests()

        self.assertTrue(publish)
        self.assertFalse(autosave)
        self.assertEqual(node._map_publish_coalesced, 1)
        self.assertFalse(node._output_wake.is_set())

    def test_publish_and_autosave_requests_share_one_wakeup(self):
        node = self._node()
        node._request_map_publish()
        node._request_autosave()
        publish, autosave = node._take_output_requests()

        self.assertTrue(publish)
        self.assertTrue(autosave)
        self.assertEqual(node._map_publish_coalesced, 0)
        self.assertEqual(node._autosave_coalesced, 0)


if __name__ == "__main__":
    unittest.main()
