import unittest

from go2_robot_bridge.calibration import calibration_from_mapping


class CameraCalibrationTests(unittest.TestCase):
    def test_ros_camera_yaml_mapping(self):
        document = {
            "camera_name": "front",
            "image_width": 1920,
            "image_height": 1080,
            "distortion_model": "plumb_bob",
            "distortion_coefficients": {"data": [0.1, 0.0, 0.0, 0.0, 0.0]},
            "camera_matrix": {
                "data": [900.0, 0.0, 960.0, 0.0, 900.0, 540.0, 0.0, 0.0, 1.0]
            },
            "rectification_matrix": {
                "data": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            },
            "projection_matrix": {
                "data": [
                    900.0,
                    0.0,
                    960.0,
                    0.0,
                    0.0,
                    900.0,
                    540.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ]
            },
        }
        calibration = calibration_from_mapping(document)
        self.assertTrue(calibration.calibrated)
        self.assertEqual(calibration.width, 1920)
        self.assertEqual(calibration.k[2], 960.0)
        self.assertEqual(len(calibration.p), 12)

    def test_missing_matrices_become_uncalibrated_zeros(self):
        calibration = calibration_from_mapping({})
        self.assertFalse(calibration.calibrated)
        self.assertEqual(calibration.k, (0.0,) * 9)
        self.assertEqual(len(calibration.r), 9)

    def test_wrong_matrix_length_is_rejected(self):
        with self.assertRaises(ValueError):
            calibration_from_mapping({"camera_matrix": {"data": [1.0, 2.0]}})


if __name__ == "__main__":
    unittest.main()
