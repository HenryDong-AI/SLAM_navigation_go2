"""Capture a RealSense and publish synchronized ROS 2 images."""

import json
import sys
import time
from array import array
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


class DepthCameraBridge(Node):
    """Own the RealSense pipeline and publish aligned RGB-D frames."""

    def __init__(self) -> None:
        super().__init__("depth_camera_bridge")
        self._pipeline = None
        self._align = None
        self._rs = None
        self._last_publish_monotonic = 0.0
        self._last_frame_monotonic = 0.0
        self._last_error_monotonic = 0.0
        self._frames_published = 0
        self._capture_errors = 0
        self._recovery_count = 0
        self._last_publish_duration_sec = 0.0

        self.serial_number = str(self._parameter("serial_number", "")).strip()
        self.width = int(self._parameter("width", 640))
        self.height = int(self._parameter("height", 480))
        self.fps = int(self._parameter("fps", 15))
        self.publish_rate_hz = float(self._parameter("publish_rate_hz", 15.0))
        self.no_frame_timeout_sec = float(
            self._parameter("no_frame_timeout_sec", 3.0)
        )
        self.auto_hardware_reset = bool(
            self._parameter("auto_hardware_reset", True)
        )
        self.hardware_reset_wait_sec = float(
            self._parameter("hardware_reset_wait_sec", 5.0)
        )
        self.frame_id = str(
            self._parameter("frame_id", "d435i_color_optical_frame")
        ).strip()
        self.color_topic = str(
            self._parameter(
                "color_topic", "/go2/depth_camera/color/image_raw"
            )
        )
        self.depth_topic = str(
            self._parameter(
                "depth_topic", "/go2/depth_camera/aligned_depth/image_raw"
            )
        )
        self.color_info_topic = str(
            self._parameter(
                "color_info_topic", "/go2/depth_camera/color/camera_info"
            )
        )
        self.depth_info_topic = str(
            self._parameter(
                "depth_info_topic",
                "/go2/depth_camera/aligned_depth/camera_info",
            )
        )
        self.status_topic = str(
            self._parameter("status_topic", "/go2/depth_camera/status")
        )

        if min(self.width, self.height, self.fps) <= 0:
            raise ValueError("camera dimensions and fps must be positive")
        if self.publish_rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be positive")
        invalid_recovery_timing = (
            self.no_frame_timeout_sec <= 0.0
            or self.hardware_reset_wait_sec < 0.0
        )
        if invalid_recovery_timing:
            raise ValueError("camera recovery timing is invalid")
        if not self.frame_id:
            raise ValueError("frame_id must not be empty")

        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        status_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self._color_publisher = self.create_publisher(
            Image, self.color_topic, sensor_qos
        )
        self._depth_publisher = self.create_publisher(
            Image, self.depth_topic, sensor_qos
        )
        self._color_info_publisher = self.create_publisher(
            CameraInfo, self.color_info_topic, sensor_qos
        )
        self._depth_info_publisher = self.create_publisher(
            CameraInfo, self.depth_info_topic, sensor_qos
        )
        self._status_publisher = self.create_publisher(
            String, self.status_topic, status_qos
        )

        self._profile = self._start_pipeline()
        self._camera_info_template = self._make_camera_info(self._profile)
        self._pipeline_started_monotonic = time.monotonic()
        self._poll_timer = self.create_timer(0.005, self._poll_camera)
        self._status_timer = self.create_timer(1.0, self._publish_status)
        self.get_logger().info(
            "direct D435i capture ready: serial=%s %dx%d@%d Hz"
            % (
                self._device_serial or "automatic",
                self.width,
                self.height,
                self.fps,
            )
        )

    def _parameter(self, name: str, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _start_pipeline(self):
        try:
            import pyrealsense2 as rs
        except ImportError as error:
            raise RuntimeError(
                "pyrealsense2 is missing from the selected Python environment"
            ) from error

        self._rs = rs
        context = rs.context()
        devices = context.query_devices()
        if len(devices) == 0:
            raise RuntimeError("no Intel RealSense device was detected")
        if self.serial_number:
            matching = [
                device
                for device in devices
                if device.get_info(rs.camera_info.serial_number)
                == self.serial_number
            ]
            if not matching:
                raise RuntimeError(
                    "configured RealSense serial {} was not detected".format(
                        self.serial_number
                    )
                )
            device = matching[0]
        else:
            device = devices[0]
        self._device_serial = device.get_info(rs.camera_info.serial_number)
        self._device_name = device.get_info(rs.camera_info.name)
        self._firmware = device.get_info(rs.camera_info.firmware_version)
        self._usb_type = device.get_info(rs.camera_info.usb_type_descriptor)

        pipeline = rs.pipeline(context)
        config = rs.config()
        config.enable_device(self._device_serial)
        config.enable_stream(
            rs.stream.depth,
            self.width,
            self.height,
            rs.format.z16,
            self.fps,
        )
        config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps,
        )
        try:
            profile = pipeline.start(config)
        except Exception as error:
            raise RuntimeError(
                "RealSense pipeline failed to start: {}".format(error)
            ) from error
        self._pipeline = pipeline
        self._align = rs.align(rs.stream.color)
        self._depth_scale = float(
            profile.get_device().first_depth_sensor().get_depth_scale()
        )
        return profile

    def _make_camera_info(self, profile) -> CameraInfo:
        stream = profile.get_stream(self._rs.stream.color)
        intrinsics = stream.as_video_stream_profile().get_intrinsics()
        info = CameraInfo()
        info.width = int(intrinsics.width)
        info.height = int(intrinsics.height)
        info.distortion_model = "plumb_bob"
        info.d = [float(value) for value in intrinsics.coeffs[:5]]
        info.k = [
            float(intrinsics.fx),
            0.0,
            float(intrinsics.ppx),
            0.0,
            float(intrinsics.fy),
            float(intrinsics.ppy),
            0.0,
            0.0,
            1.0,
        ]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [
            float(intrinsics.fx),
            0.0,
            float(intrinsics.ppx),
            0.0,
            0.0,
            float(intrinsics.fy),
            float(intrinsics.ppy),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        return info

    def _camera_info(self, stamp) -> CameraInfo:
        source = self._camera_info_template
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = self.frame_id
        info.width = source.width
        info.height = source.height
        info.distortion_model = source.distortion_model
        info.d = list(source.d)
        info.k = list(source.k)
        info.r = list(source.r)
        info.p = list(source.p)
        info.binning_x = source.binning_x
        info.binning_y = source.binning_y
        info.roi = source.roi
        return info

    def _poll_camera(self) -> None:
        now_monotonic = time.monotonic()
        if now_monotonic - self._last_publish_monotonic < (
            1.0 / self.publish_rate_hz
        ):
            return
        if self._pipeline is None:
            if (
                self.auto_hardware_reset
                and now_monotonic - self._pipeline_started_monotonic
                >= self.no_frame_timeout_sec
            ):
                self._recover_camera()
            return
        try:
            frames = self._pipeline.poll_for_frames()
            if not frames:
                last_activity = (
                    self._last_frame_monotonic
                    or self._pipeline_started_monotonic
                )
                if (
                    self.auto_hardware_reset
                    and now_monotonic - last_activity
                    >= self.no_frame_timeout_sec
                ):
                    self._recover_camera()
                return
            aligned = self._align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            if not depth_frame or not color_frame:
                return
            color = np.asanyarray(color_frame.get_data())
            raw_depth = np.asanyarray(depth_frame.get_data())
            depth_metres = raw_depth.astype(np.float32) * self._depth_scale

            color_message = self._image_message(color, "bgr8")
            depth_message = self._image_message(depth_metres, "32FC1")
            # Stamp after both expensive NumPy-to-ROS copies. Publish depth
            # first so the mapper sees a host-clock timestamp close to its
            # actual DDS delivery time instead of waiting behind RGB.
            publish_started = time.monotonic()
            stamp = self.get_clock().now().to_msg()
            for message in (color_message, depth_message):
                message.header.stamp = stamp
                message.header.frame_id = self.frame_id
            self._depth_publisher.publish(depth_message)
            self._depth_info_publisher.publish(self._camera_info(stamp))
            self._color_publisher.publish(color_message)
            self._color_info_publisher.publish(self._camera_info(stamp))
            self._last_publish_duration_sec = (
                time.monotonic() - publish_started
            )
            self._last_publish_monotonic = now_monotonic
            self._last_frame_monotonic = now_monotonic
            self._frames_published += 1
        except Exception as error:
            self._capture_errors += 1
            if now_monotonic - self._last_error_monotonic >= 2.0:
                self._last_error_monotonic = now_monotonic
                self.get_logger().error(
                    "RealSense capture failed: {}".format(error)
                )

    def _recover_camera(self) -> None:
        """Reset a D435i which enumerates but stops delivering frames."""
        self._pipeline_started_monotonic = time.monotonic()
        self.get_logger().warning(
            "D435i produced no frames for %.1f s; resetting only the camera"
            % self.no_frame_timeout_sec
        )
        try:
            if self._pipeline is not None:
                try:
                    self._pipeline.stop()
                except Exception as error:
                    self.get_logger().warning(
                        "stalled D435i pipeline stop failed: {}".format(error)
                    )
            self._pipeline = None
            context = self._rs.context()
            matching = [
                device
                for device in context.query_devices()
                if device.get_info(self._rs.camera_info.serial_number)
                == self._device_serial
            ]
            if not matching:
                raise RuntimeError("the configured D435i disappeared from USB")
            matching[0].hardware_reset()
            time.sleep(self.hardware_reset_wait_sec)
            self._profile = self._start_pipeline()
            self._camera_info_template = self._make_camera_info(self._profile)
            self._pipeline_started_monotonic = time.monotonic()
            self._last_frame_monotonic = 0.0
            self._recovery_count += 1
            self.get_logger().info("D435i recovery restart completed")
        except Exception as error:
            self._capture_errors += 1
            self._pipeline_started_monotonic = time.monotonic()
            self.get_logger().error("D435i recovery failed: {}".format(error))

    @staticmethod
    def _image_message(image: np.ndarray, encoding: str) -> Image:
        contiguous = np.ascontiguousarray(image)
        message = Image()
        message.height = int(contiguous.shape[0])
        message.width = int(contiguous.shape[1])
        message.encoding = encoding
        message.is_bigendian = sys.byteorder == "big"
        message.step = int(contiguous.strides[0])
        payload = array("B")
        payload.frombytes(contiguous.tobytes(order="C"))
        message.data = payload
        return message

    def _publish_status(self) -> None:
        age: Optional[float]
        if self._last_frame_monotonic:
            age = max(0.0, time.monotonic() - self._last_frame_monotonic)
        else:
            age = None
        payload = {
            "state": (
                "streaming"
                if age is not None and age < 1.0
                else "waiting"
            ),
            "device": self._device_name,
            "serial": self._device_serial,
            "firmware": self._firmware,
            "usb_type": self._usb_type,
            "depth_scale": self._depth_scale,
            "frame_age_sec": age,
            "frames_published": self._frames_published,
            "capture_errors": self._capture_errors,
            "recovery_count": self._recovery_count,
            "publish_duration_sec": self._last_publish_duration_sec,
        }
        message = String()
        message.data = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        )
        self._status_publisher.publish(message)

    def destroy_node(self):
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except (Exception, KeyboardInterrupt) as error:
                self.get_logger().warning(
                    "RealSense pipeline stop failed: {}".format(error)
                )
            self._pipeline = None
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DepthCameraBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
