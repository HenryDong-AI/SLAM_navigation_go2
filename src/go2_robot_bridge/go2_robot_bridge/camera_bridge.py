"""Publish the Go2 front camera through standard ROS 2 image messages."""

from array import array
import os
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Header

from .calibration import CameraCalibration, load_camera_calibration
from .camera_sdk_process import CameraSdkProcess, ros_stamp_from_monotonic_capture
from .sdk_runtime import (
    DEFAULT_CYCLONEDDS_PYTHON_PATH,
    DEFAULT_SDK_PYTHON_PATH,
    no_shm_runtime_present,
)


def _default_calibration_file() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory

        return os.path.join(
            get_package_share_directory("go2_robot_bridge"), "config", "camera_info.yaml"
        )
    except Exception:
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, "config", "camera_info.yaml")
        )


class CameraBridge(Node):
    def __init__(self) -> None:
        super().__init__("camera_bridge")

        self.declare_parameter("network_interface", "eth0")
        # Five full-HD frames/s is the validated Jetson operating point and is
        # sufficient for the semantic mapper's default 2 Hz inference loop.
        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("image_topic", "/go2/camera/image_raw")
        self.declare_parameter(
            "compressed_topic", "/go2/camera/image_raw/compressed"
        )
        self.declare_parameter("rectified_topic", "/go2/camera/image_rect")
        self.declare_parameter("camera_info_topic", "/go2/camera/camera_info")
        self.declare_parameter("frame_id", "go2_front_camera_optical_frame")
        self.declare_parameter("calibration_file", _default_calibration_file())
        self.declare_parameter("publish_decoded_image", False)
        self.declare_parameter("publish_rectified_image", True)
        self.declare_parameter("publish_compressed_image", True)
        self.declare_parameter("publish_camera_info", True)
        self.declare_parameter("sdk_timeout_sec", 3.0)
        # Unitree's RPC has no acquisition stamp. Timestamp the request
        # midpoint, then apply a measured (usually negative) pipeline offset.
        self.declare_parameter("timestamp_offset_sec", 0.0)
        self.declare_parameter("reconnect_delay_sec", 2.0)
        self.declare_parameter("max_consecutive_failures", 3)
        self.declare_parameter("max_frame_age_sec", 3.0)
        self.declare_parameter("worker_watchdog_sec", 8.0)
        self.declare_parameter("sdk_python_path", DEFAULT_SDK_PYTHON_PATH)
        self.declare_parameter(
            "cyclonedds_python_path", DEFAULT_CYCLONEDDS_PYTHON_PATH
        )
        self.declare_parameter("require_noshm_runtime", True)
        self.declare_parameter("noshm_library_fragment", "install_noshm/lib")

        self._interface = str(self.get_parameter("network_interface").value)
        self._frame_id = str(self.get_parameter("frame_id").value)
        self._sdk_path = str(self.get_parameter("sdk_python_path").value)
        self._cyclonedds_python_path = str(
            self.get_parameter("cyclonedds_python_path").value
        )
        self._sdk_timeout = max(0.1, float(self.get_parameter("sdk_timeout_sec").value))
        self._timestamp_offset_ns = int(
            float(self.get_parameter("timestamp_offset_sec").value) * 1.0e9
        )
        self._reconnect_delay = max(
            0.1, float(self.get_parameter("reconnect_delay_sec").value)
        )
        self._max_failures = max(
            1, int(self.get_parameter("max_consecutive_failures").value)
        )
        self._max_frame_age_ns = int(
            max(0.1, float(self.get_parameter("max_frame_age_sec").value)) * 1.0e9
        )
        self._worker_watchdog_ns = int(
            max(
                self._sdk_timeout + 2.0,
                float(self.get_parameter("worker_watchdog_sec").value),
            )
            * 1.0e9
        )
        self._publish_raw = bool(self.get_parameter("publish_decoded_image").value)
        self._publish_compressed = bool(
            self.get_parameter("publish_compressed_image").value
        )
        self._publish_rectified = bool(
            self.get_parameter("publish_rectified_image").value
        )
        self._publish_info = bool(self.get_parameter("publish_camera_info").value)

        require_noshm = bool(self.get_parameter("require_noshm_runtime").value)
        noshm_fragment = str(self.get_parameter("noshm_library_fragment").value)
        if require_noshm and not no_shm_runtime_present(noshm_fragment):
            raise RuntimeError(
                "the no-SHM CycloneDDS directory is not present in LD_LIBRARY_PATH; "
                "refusing to call VideoClient because the system SHM build is unsafe"
            )

        self._raw_publisher = self.create_publisher(
            Image, str(self.get_parameter("image_topic").value), qos_profile_sensor_data
        )
        self._compressed_publisher = self.create_publisher(
            CompressedImage,
            str(self.get_parameter("compressed_topic").value),
            qos_profile_sensor_data,
        )
        self._rectified_publisher = self.create_publisher(
            Image,
            str(self.get_parameter("rectified_topic").value),
            qos_profile_sensor_data,
        )
        self._info_publisher = self.create_publisher(
            CameraInfo,
            str(self.get_parameter("camera_info_topic").value),
            qos_profile_sensor_data,
        )

        self._calibration = self._read_calibration(
            str(self.get_parameter("calibration_file").value)
        )
        rate = max(
            0.2, min(float(self.get_parameter("publish_rate_hz").value), 15.0)
        )
        self._worker = CameraSdkProcess(
            interface=self._interface,
            sdk_python_path=self._sdk_path,
            cyclonedds_python_path=self._cyclonedds_python_path,
            timeout_sec=self._sdk_timeout,
            rate_hz=rate,
            reconnect_delay_sec=self._reconnect_delay,
            max_failures=self._max_failures,
        )
        self._next_connect_at = 0.0
        self._last_frame_sequence = 0
        self._last_status_sequence = 0
        self._worker_started_ns = 0
        self._last_worker_progress_ns = 0
        self._failure_count = 0
        self._log_times: Dict[str, float] = {}
        self._dimension_warning_emitted = False
        self._rectification_maps = None
        self._rectification_size = None

        self._timer = self.create_timer(1.0 / rate, self._tick)
        self.get_logger().info(
            "Go2 camera bridge configured for %s at %.2f Hz" % (self._interface, rate)
        )

    def _read_calibration(self, path: str) -> CameraCalibration:
        try:
            calibration = load_camera_calibration(path)
        except Exception as exc:
            raise RuntimeError("unable to load camera calibration %s: %s" % (path, exc))
        if not calibration.calibrated:
            self.get_logger().warning(
                "camera_info.yaml is an uncalibrated placeholder; calibrate the "
                "camera before projecting semantic labels into the LiDAR map"
            )
        return calibration

    def _warn_limited(self, key: str, message: str, period: float = 5.0) -> None:
        now = time.monotonic()
        if now - self._log_times.get(key, -1.0e9) >= period:
            self._log_times[key] = now
            self.get_logger().warning(message)

    def _start_worker_if_due(self) -> None:
        if self._worker.running or time.monotonic() < self._next_connect_at:
            return
        try:
            self._worker.start()
            self._worker_started_ns = time.monotonic_ns()
            self._last_worker_progress_ns = self._worker_started_ns
            self._failure_count = 0
            self.get_logger().info("started isolated Go2 VideoClient worker")
        except Exception as exc:
            self._next_connect_at = time.monotonic() + self._reconnect_delay
            self._warn_limited("connect", "camera worker start failed: %s" % exc)

    def _restart_worker(self) -> None:
        self._worker.close()
        self._worker_started_ns = 0
        self._last_worker_progress_ns = 0
        self._next_connect_at = time.monotonic() + self._reconnect_delay
        self._failure_count = 0
        self.get_logger().warning("camera worker reset; restart is scheduled")

    def _record_failure(self, message: str) -> None:
        self._failure_count += 1
        self._warn_limited("sample", message)
        if self._failure_count >= self._max_failures:
            self._restart_worker()

    def _camera_info(self, header: Header, width: int, height: int) -> CameraInfo:
        info = CameraInfo()
        info.header = header
        info.width = int(width)
        info.height = int(height)
        info.distortion_model = self._calibration.distortion_model
        info.d = list(self._calibration.d)
        info.k = list(self._calibration.k)
        info.r = list(self._calibration.r)
        info.p = list(self._calibration.p)
        if (
            not self._dimension_warning_emitted
            and self._calibration.width > 0
            and self._calibration.height > 0
            and (self._calibration.width != width or self._calibration.height != height)
        ):
            self._dimension_warning_emitted = True
            self.get_logger().warning(
                "calibration is %dx%d but camera returned %dx%d"
                % (
                    self._calibration.width,
                    self._calibration.height,
                    width,
                    height,
                )
            )
        return info

    @staticmethod
    def _image_message(frame: np.ndarray, header: Header) -> Image:
        image = Image()
        image.header = header
        image.height = int(frame.shape[0])
        image.width = int(frame.shape[1])
        image.encoding = "bgr8"
        image.is_bigendian = False
        image.step = image.width * 3
        payload = array("B")
        payload.frombytes(np.ascontiguousarray(frame).tobytes())
        image.data = payload
        return image

    def _rectify(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Rectify with calibrated K/D/R/P; pass through only while uncalibrated."""
        if not self._calibration.calibrated:
            return frame
        height, width = int(frame.shape[0]), int(frame.shape[1])
        if (width, height) != (self._calibration.width, self._calibration.height):
            self._warn_limited(
                "rectify_dimensions",
                "refusing rectification because image and calibration dimensions differ",
            )
            return None
        try:
            if self._rectification_maps is None or self._rectification_size != (width, height):
                camera_matrix = np.asarray(self._calibration.k, dtype=np.float64).reshape(3, 3)
                distortion = np.asarray(self._calibration.d, dtype=np.float64)
                rectification = np.asarray(self._calibration.r, dtype=np.float64).reshape(3, 3)
                projection = np.asarray(self._calibration.p, dtype=np.float64).reshape(3, 4)
                new_camera_matrix = projection[:, :3]
                if new_camera_matrix[0, 0] <= 0.0 or new_camera_matrix[1, 1] <= 0.0:
                    new_camera_matrix = camera_matrix
                model = self._calibration.distortion_model
                if model == "equidistant":
                    if distortion.size != 4:
                        raise ValueError("equidistant calibration requires four coefficients")
                    maps = cv2.fisheye.initUndistortRectifyMap(
                        camera_matrix,
                        distortion.reshape(4, 1),
                        rectification,
                        new_camera_matrix,
                        (width, height),
                        cv2.CV_32FC1,
                    )
                elif model in ("plumb_bob", "rational_polynomial"):
                    maps = cv2.initUndistortRectifyMap(
                        camera_matrix,
                        distortion,
                        rectification,
                        new_camera_matrix,
                        (width, height),
                        cv2.CV_32FC1,
                    )
                else:
                    raise ValueError("unsupported distortion model %r" % model)
                self._rectification_maps = maps
                self._rectification_size = (width, height)
            return cv2.remap(
                frame,
                self._rectification_maps[0],
                self._rectification_maps[1],
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
            )
        except Exception as exc:
            self._warn_limited("rectify", "camera rectification failed: %s" % exc)
            return None

    def _tick(self) -> None:
        self._start_worker_if_due()
        if not self._worker.running:
            return_code = self._worker.returncode
            if return_code is None:
                return
            reader_error = self._worker.reader_error
            self._worker.close()
            self._next_connect_at = time.monotonic() + self._reconnect_delay
            detail = "" if not reader_error else "; protocol: %s" % reader_error
            self._warn_limited(
                "worker_exit",
                "camera worker exited with code %s%s; restart is scheduled"
                % (return_code, detail),
            )
            return

        reader_error = self._worker.reader_error
        if reader_error or not self._worker.reader_alive:
            self._warn_limited(
                "reader_failure",
                "camera protocol reader stopped: %s"
                % (reader_error or "reader thread exited"),
            )
            self._restart_worker()
            return

        status = self._worker.latest_status_after(self._last_status_sequence)
        if status is not None:
            self._last_status_sequence, received_ns, status_text = status
            self._last_worker_progress_ns = max(
                self._last_worker_progress_ns, received_ns
            )
            if status_text == "ready":
                self.get_logger().info("isolated Go2 VideoClient is ready")
            else:
                self._warn_limited("worker_status", "camera worker: %s" % status_text)

        sample = self._worker.latest_frame_after(self._last_frame_sequence)
        if sample is None:
            now_ns = time.monotonic_ns()
            if (
                self._last_worker_progress_ns
                and now_ns - self._last_worker_progress_ns
                > self._worker_watchdog_ns
            ):
                self._warn_limited(
                    "worker_stalled",
                    "camera worker made no IPC progress for %.2f s"
                    % ((now_ns - self._last_worker_progress_ns) / 1.0e9),
                )
                self._restart_worker()
            return
        sequence, capture_monotonic_ns, jpeg, received_monotonic_ns = sample
        self._last_frame_sequence = sequence
        self._last_worker_progress_ns = max(
            self._last_worker_progress_ns, received_monotonic_ns
        )
        monotonic_now_ns = time.monotonic_ns()
        age_ns = monotonic_now_ns - capture_monotonic_ns
        if age_ns < -100000000 or age_ns > self._max_frame_age_ns:
            self._record_failure(
                "camera frame has invalid monotonic age %.3f s" % (age_ns / 1.0e9)
            )
            return

        header = Header()
        stamp_ns = ros_stamp_from_monotonic_capture(
            self.get_clock().now().nanoseconds,
            monotonic_now_ns,
            capture_monotonic_ns,
            self._timestamp_offset_ns,
        )
        header.stamp.sec = int(stamp_ns // 1000000000)
        header.stamp.nanosec = int(stamp_ns % 1000000000)
        header.frame_id = self._frame_id

        if self._publish_compressed:
            compressed = CompressedImage()
            compressed.header = header
            compressed.format = "jpeg"
            payload = array("B")
            payload.frombytes(jpeg)
            compressed.data = payload
            self._compressed_publisher.publish(compressed)

        # Decode exactly once for both Image and CameraInfo dimensions.
        encoded = np.frombuffer(jpeg, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            self._record_failure("received an invalid JPEG camera sample")
            return
        frame = np.ascontiguousarray(frame)
        height, width = int(frame.shape[0]), int(frame.shape[1])

        if self._publish_raw:
            self._raw_publisher.publish(self._image_message(frame, header))

        if self._publish_rectified:
            rectified = self._rectify(frame)
            if rectified is not None:
                self._rectified_publisher.publish(self._image_message(rectified, header))

        if self._publish_info:
            self._info_publisher.publish(self._camera_info(header, width, height))
        self._failure_count = 0

    def shutdown_worker(self) -> None:
        self._worker.close()


def main(args: Optional[Any] = None) -> None:
    rclpy.init(args=args)
    node: Optional[CameraBridge] = None
    try:
        node = CameraBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown_worker()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
