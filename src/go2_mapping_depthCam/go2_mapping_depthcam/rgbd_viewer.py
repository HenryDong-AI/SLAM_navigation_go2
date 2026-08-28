"""OpenCV side-by-side RGB and RGB-D live viewer."""

import threading
from typing import Optional

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image

from .geometry import decode_color_image, decode_depth_image


class RgbdViewer(Node):
    """Render RGB beside a depth-colorized RGB overlay."""

    def __init__(self) -> None:
        super().__init__("rgbd_viewer")
        self._lock = threading.Lock()
        self._color: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None
        self._window_name = str(self._parameter("window_name", "Go2 D435i RGB | RGB-D"))
        self._max_visual_depth = float(self._parameter("max_visual_depth", 5.0))
        self._window_ready = False

        color_topic = str(
            self._parameter("color_topic", "/go2/depth_camera/color/image_raw")
        )
        depth_topic = str(
            self._parameter(
                "depth_topic", "/go2/depth_camera/aligned_depth/image_raw"
            )
        )
        sensor_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        self.create_subscription(Image, color_topic, self._color_callback, sensor_qos)
        self.create_subscription(Image, depth_topic, self._depth_callback, sensor_qos)
        self._display_timer = self.create_timer(0.05, self._display)
        self.get_logger().info(
            "RGB/RGB-D viewer waiting for %s and %s" % (color_topic, depth_topic)
        )

    def _parameter(self, name: str, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _color_callback(self, message: Image) -> None:
        try:
            color = decode_color_image(message)
        except Exception as error:
            self.get_logger().warning("invalid color image: {}".format(error))
            return
        with self._lock:
            self._color = np.asarray(color).copy()

    def _depth_callback(self, message: Image) -> None:
        try:
            depth = decode_depth_image(message)
        except Exception as error:
            self.get_logger().warning("invalid depth image: {}".format(error))
            return
        with self._lock:
            self._depth = np.asarray(depth, dtype=np.float32).copy()

    def _display(self) -> None:
        with self._lock:
            if self._color is None or self._depth is None:
                return
            color = self._color.copy()
            depth = self._depth.copy()
        if depth.shape != color.shape[:2]:
            self.get_logger().warning("RGB and aligned-depth sizes do not match")
            return

        valid = np.isfinite(depth) & (depth > 0.0)
        clipped = np.clip(depth, 0.0, self._max_visual_depth)
        normalized = np.zeros(depth.shape, dtype=np.uint8)
        normalized[valid] = np.asarray(
            255.0 * (1.0 - clipped[valid] / self._max_visual_depth),
            dtype=np.uint8,
        )
        depth_color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
        depth_color[~valid] = 0
        rgbd = cv2.addWeighted(color, 0.55, depth_color, 0.45, 0.0)
        cv2.putText(color, "RGB", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(rgbd, "RGB-D", (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        canvas = np.hstack((color, rgbd))
        try:
            if not self._window_ready:
                cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
                self._window_ready = True
            cv2.imshow(self._window_name, canvas)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                rclpy.shutdown()
        except cv2.error as error:
            raise RuntimeError(
                "OpenCV viewer requires a graphical DISPLAY: {}".format(error)
            ) from error

    def destroy_node(self):
        if self._window_ready:
            cv2.destroyWindow(self._window_name)
            cv2.waitKey(1)
            self._window_ready = False
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RgbdViewer()
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
