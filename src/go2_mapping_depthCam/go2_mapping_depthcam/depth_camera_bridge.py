"""Capture RealSense frames and publish atomic XYZRGB clouds plus images."""

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
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_msgs.msg import String

from go2_mapping.pointcloud import xyzrgb_to_float32_bytes

from .geometry import DeviceClockSynchronizer, depth_to_camera_points_rgb


class DepthCameraBridge(Node):
    """Own the RealSense pipeline and publish aligned RGB-D frames."""

    def __init__(self) -> None:
        super().__init__("depth_camera_bridge")
        self._pipeline = None
        self._align = None
        self._rs = None
        self._depth_filters = []
        self._capture_clock = DeviceClockSynchronizer()
        self._capture_timestamp_domain = "unknown"
        self._last_capture_latency_sec = None
        self._last_capture_stamp_ns = 0
        self._last_publish_monotonic = 0.0
        self._last_frame_monotonic = 0.0
        self._last_error_monotonic = 0.0
        self._frames_published = 0
        self._capture_errors = 0
        self._recovery_count = 0
        self._last_publish_duration_sec = 0.0
        self._last_frame_processing_duration_sec = 0.0
        self._last_cloud_point_count = 0

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
        self.pointcloud_topic = str(
            self._parameter("pointcloud_topic", "/go2/depth_camera/points")
        )
        self.pointcloud_pixel_stride = max(
            1, int(self._parameter("pointcloud_pixel_stride", 2))
        )
        self.pointcloud_min_depth = float(
            self._parameter("pointcloud_min_depth", 0.25)
        )
        self.pointcloud_max_depth = float(
            self._parameter("pointcloud_max_depth", 3.0)
        )
        self.pointcloud_max_points = max(
            1, int(self._parameter("pointcloud_max_points", 60000))
        )
        self.depth_spatial_filter = bool(
            self._parameter("depth_spatial_filter", True)
        )
        self.depth_spatial_magnitude = int(
            self._parameter("depth_spatial_magnitude", 2)
        )
        self.depth_spatial_alpha = float(
            self._parameter("depth_spatial_alpha", 0.50)
        )
        self.depth_spatial_delta = float(
            self._parameter("depth_spatial_delta", 20.0)
        )
        self.depth_spatial_holes_fill = int(
            self._parameter("depth_spatial_holes_fill", 0)
        )
        self.depth_temporal_filter = bool(
            self._parameter("depth_temporal_filter", False)
        )
        self.depth_temporal_alpha = float(
            self._parameter("depth_temporal_alpha", 0.40)
        )
        self.depth_temporal_delta = float(
            self._parameter("depth_temporal_delta", 20.0)
        )
        self.depth_hole_filling_mode = int(
            self._parameter("depth_hole_filling_mode", -1)
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
        if (
            self.pointcloud_min_depth <= 0.0
            or self.pointcloud_max_depth <= self.pointcloud_min_depth
        ):
            raise ValueError("point-cloud depth bounds are invalid")

        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        cloud_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
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
        self._pointcloud_publisher = self.create_publisher(
            PointCloud2, self.pointcloud_topic, cloud_qos
        )

        self._profile = self._start_pipeline()
        self._camera_info_template = self._make_camera_info(self._profile)
        self._pipeline_started_monotonic = time.monotonic()
        self._poll_timer = self.create_timer(0.005, self._poll_camera)
        self._status_timer = self.create_timer(1.0, self._publish_status)
        self.get_logger().info(
            "direct D435i capture ready: serial=%s %dx%d@%d Hz atomic_cloud=%s"
            % (
                self._device_serial or "automatic",
                self.width,
                self.height,
                self.fps,
                self.pointcloud_topic,
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
        self._configure_depth_filters()
        return profile

    @staticmethod
    def _set_filter_option(processing_filter, option, value):
        try:
            processing_filter.set_option(option, float(value))
        except RuntimeError:
            # Older librealsense builds omit a few optional filter controls.
            pass

    def _configure_depth_filters(self):
        self._depth_filters = []
        if not (self.depth_spatial_filter or self.depth_temporal_filter):
            return
        self._depth_filters.append(self._rs.disparity_transform(True))
        if self.depth_spatial_filter:
            spatial = self._rs.spatial_filter()
            self._set_filter_option(
                spatial, self._rs.option.filter_magnitude,
                self.depth_spatial_magnitude,
            )
            self._set_filter_option(
                spatial, self._rs.option.filter_smooth_alpha,
                self.depth_spatial_alpha,
            )
            self._set_filter_option(
                spatial, self._rs.option.filter_smooth_delta,
                self.depth_spatial_delta,
            )
            self._set_filter_option(
                spatial, self._rs.option.holes_fill,
                self.depth_spatial_holes_fill,
            )
            self._depth_filters.append(spatial)
        if self.depth_temporal_filter:
            temporal = self._rs.temporal_filter()
            self._set_filter_option(
                temporal, self._rs.option.filter_smooth_alpha,
                self.depth_temporal_alpha,
            )
            self._set_filter_option(
                temporal, self._rs.option.filter_smooth_delta,
                self.depth_temporal_delta,
            )
            self._depth_filters.append(temporal)
        self._depth_filters.append(self._rs.disparity_transform(False))
        if self.depth_hole_filling_mode >= 0:
            self._depth_filters.append(
                self._rs.hole_filling_filter(self.depth_hole_filling_mode)
            )

    def _filtered_depth_frame(self, depth_frame):
        filtered = depth_frame
        for processing_filter in self._depth_filters:
            filtered = processing_filter.process(filtered)
        return filtered.as_depth_frame()

    @staticmethod
    def _time_message(stamp_ns, template):
        template.sec = int(stamp_ns // 1000000000)
        template.nanosec = int(stamp_ns % 1000000000)
        return template

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
            arrival_ros_ns = self.get_clock().now().nanoseconds
            device_timestamp_ms = float(color_frame.get_timestamp())
            stamp_ns = self._capture_clock.to_ros_ns(
                device_timestamp_ms, arrival_ros_ns
            )
            self._capture_timestamp_domain = str(
                color_frame.get_frame_timestamp_domain()
            )
            depth_frame = self._filtered_depth_frame(depth_frame)
            frame_processing_started = time.monotonic()
            color = np.asanyarray(color_frame.get_data())
            raw_depth = np.asanyarray(depth_frame.get_data())
            depth_metres = raw_depth.astype(np.float32) * self._depth_scale

            intrinsics = self._camera_info_template.k
            points_camera, colors_rgb = depth_to_camera_points_rgb(
                depth_metres,
                color,
                float(intrinsics[0]),
                float(intrinsics[4]),
                float(intrinsics[2]),
                float(intrinsics[5]),
                pixel_stride=self.pointcloud_pixel_stride,
                min_depth=self.pointcloud_min_depth,
                max_depth=self.pointcloud_max_depth,
                max_points=self.pointcloud_max_points,
            )
            cloud_message = self._point_cloud_message(
                points_camera, colors_rgb
            )
            color_message = self._image_message(color, "bgr8")
            depth_message = self._image_message(depth_metres, "32FC1")
            # All geometry and color for mapping are now one atomic message.
            # The viewer images retain the exact same capture timestamp but
            # are no longer synchronized downstream by independent DDS queues.
            publish_started = time.monotonic()
            stamp = self._time_message(stamp_ns, self.get_clock().now().to_msg())
            for message in (cloud_message, color_message, depth_message):
                message.header.stamp = stamp
                message.header.frame_id = self.frame_id
            self._pointcloud_publisher.publish(cloud_message)
            self._depth_publisher.publish(depth_message)
            self._depth_info_publisher.publish(self._camera_info(stamp))
            self._last_capture_stamp_ns = stamp_ns
            self._last_capture_latency_sec = max(
                0.0, (arrival_ros_ns - stamp_ns) * 1.0e-9
            )
            self._color_publisher.publish(color_message)
            self._color_info_publisher.publish(self._camera_info(stamp))
            self._last_cloud_point_count = int(points_camera.shape[0])
            self._last_publish_duration_sec = (
                time.monotonic() - publish_started
            )
            self._last_frame_processing_duration_sec = (
                time.monotonic() - frame_processing_started
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
            self._capture_clock.reset()
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

    @staticmethod
    def _point_cloud_message(points, colors_rgb) -> PointCloud2:
        message = PointCloud2()
        message.height = 1
        message.width = int(points.shape[0])
        message.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(
                name="rgb",
                offset=12,
                datatype=PointField.FLOAT32,
                count=1,
            ),
        ]
        message.is_bigendian = False
        message.point_step = 16
        message.row_step = message.point_step * message.width
        message.is_dense = True
        payload = array("B")
        payload.frombytes(xyzrgb_to_float32_bytes(points, colors_rgb))
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
            "frame_processing_duration_sec": (
                self._last_frame_processing_duration_sec
            ),
            "atomic_pointcloud_topic": self.pointcloud_topic,
            "atomic_pointcloud_points": self._last_cloud_point_count,
            "pointcloud_pixel_stride": self.pointcloud_pixel_stride,
            "capture_timestamp_domain": self._capture_timestamp_domain,
            "capture_latency_sec": self._last_capture_latency_sec,
            "capture_clock_reset_count": self._capture_clock.reset_count,
            "depth_filtering": {
                "spatial": self.depth_spatial_filter,
                "temporal": self.depth_temporal_filter,
                "hole_filling_mode": self.depth_hole_filling_mode,
            },
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
