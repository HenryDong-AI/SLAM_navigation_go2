import tempfile
import unittest
from pathlib import Path

import numpy as np

from go2_mapping.state_io import load_snapshot, save_snapshot


class StateIoTest(unittest.TestCase):
    @staticmethod
    def _read_pgm(path):
        data = Path(path).read_bytes()
        parts = data.split(b"\n", 4)
        if len(parts) != 5 or parts[0] != b"P5" or parts[3] != b"255":
            raise AssertionError("unexpected PGM encoding")
        width, height = (int(value) for value in parts[2].split())
        return np.frombuffer(parts[4], dtype=np.uint8).reshape(height, width)

    @staticmethod
    def _yaml_scalars(path):
        values = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                values[key.strip()] = value.strip()
        return values

    def test_atomic_snapshot_and_pickle_free_load(self):
        voxel_state = {
            "voxel_keys": np.asarray([[0, 0, 0]], dtype=np.int64),
            "voxel_centroids": np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64),
            "voxel_counts": np.asarray([2], dtype=np.int64),
            "voxel_last_seen_ns": np.asarray([10], dtype=np.int64),
        }
        grid_state = {
            "grid_keys": np.asarray([[0, 0]], dtype=np.int64),
            "grid_log_odds": np.asarray([0.8], dtype=np.float64),
            "grid_last_seen_ns": np.asarray([10], dtype=np.int64),
            "grid_last_robot_cell": np.asarray([0, 0], dtype=np.int64),
        }
        with tempfile.TemporaryDirectory() as temporary:
            destination = save_snapshot(
                output_dir=temporary,
                voxel_state=voxel_state,
                grid_state=grid_state,
                occupancy=np.asarray([[80]], dtype=np.int8),
                origin_x=0.0,
                origin_y=0.0,
                resolution=0.1,
                metadata={
                    "world_frame": "odom",
                    "voxel_size": 0.1,
                    "grid_resolution": 0.1,
                },
            )
            self.assertEqual(
                {path.name for path in Path(destination).iterdir()},
                {"map.ply", "map.pgm", "map.yaml", "state.npz"},
            )
            self.assertFalse(
                any(path.name.startswith(".go2_map_tmp_") for path in Path(temporary).iterdir())
            )
            arrays, metadata = load_snapshot(str(destination))
            np.testing.assert_array_equal(arrays["voxel_counts"], [2])
            self.assertEqual(metadata["world_frame"], "odom")
            self.assertEqual(metadata["schema_version"], 1)

    def test_saved_unknown_cells_survive_foxy_map_server_trinary_reload(self):
        voxel_state = {
            "voxel_keys": np.empty((0, 3), dtype=np.int64),
            "voxel_centroids": np.empty((0, 3), dtype=np.float64),
            "voxel_counts": np.empty((0,), dtype=np.int64),
            "voxel_last_seen_ns": np.empty((0,), dtype=np.int64),
        }
        grid_state = {
            "grid_keys": np.empty((0, 2), dtype=np.int64),
            "grid_log_odds": np.empty((0,), dtype=np.float64),
            "grid_last_seen_ns": np.empty((0,), dtype=np.int64),
            "grid_last_robot_cell": np.asarray([0, 0], dtype=np.int64),
        }
        source = np.asarray([[-1, 0, 50, 100]], dtype=np.int8)
        with tempfile.TemporaryDirectory() as temporary:
            destination = save_snapshot(
                output_dir=temporary,
                voxel_state=voxel_state,
                grid_state=grid_state,
                occupancy=source,
                origin_x=0.0,
                origin_y=0.0,
                resolution=0.1,
                metadata={
                    "world_frame": "odom",
                    "voxel_size": 0.1,
                    "grid_resolution": 0.1,
                },
            )
            scalars = self._yaml_scalars(destination / "map.yaml")
            free_threshold = float(scalars["free_thresh"])
            occupied_threshold = float(scalars["occupied_thresh"])
            pixels = np.flipud(self._read_pgm(destination / "map.pgm"))
            probability = (255.0 - pixels.astype(np.float64)) / 255.0
            decoded = np.full(pixels.shape, -1, dtype=np.int8)
            decoded[probability < free_threshold] = 0
            decoded[probability > occupied_threshold] = 100
            np.testing.assert_array_equal(decoded, [[-1, 0, -1, 100]])
            self.assertGreater((255.0 - 205.0) / 255.0, free_threshold)


if __name__ == "__main__":
    unittest.main()
