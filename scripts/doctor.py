#!/usr/bin/env python3
"""Read-only preflight checks for the Go2 SLAM/navigation workspace."""

import os
import pathlib
import shutil
import statistics
import subprocess
import sys
import time


REQUIRED_PACKAGES = (
    "rclpy",
    "sensor_msgs",
    "nav2_controller",
    "nav2_planner",
    "nav2_bt_navigator",
    "nav2_amcl",
    "nav2_map_server",
    "nav2_rviz_plugins",
)
REQUIRED_TOPICS = {
    "/utlidar/cloud",
    "/utlidar/cloud_base",
    "/utlidar/cloud_deskewed",
    "/utlidar/robot_odom",
}


def check(condition, ok, bad, fatal=False):
    prefix = "OK  " if condition else ("FAIL" if fatal else "WARN")
    print(f"[{prefix}] {ok if condition else bad}")
    return condition or not fatal


def measure_raw_clock_offset(topic="/utlidar/robot_odom", sample_count=12):
    """Return host-minus-sensor offsets and source stamps from live odometry."""

    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.qos import (
        QoSDurabilityPolicy,
        QoSHistoryPolicy,
        QoSProfile,
        QoSReliabilityPolicy,
    )

    offsets = []
    stamps = []
    rclpy.init(args=None)
    node = rclpy.create_node("go2_slam_nav_clock_doctor")
    qos = QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=20,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
        durability=QoSDurabilityPolicy.VOLATILE,
    )

    def callback(message):
        source_ns = (
            int(message.header.stamp.sec) * 1000000000
            + int(message.header.stamp.nanosec)
        )
        receipt_ns = int(node.get_clock().now().nanoseconds)
        stamps.append(source_ns)
        offsets.append((receipt_ns - source_ns) / 1.0e9)

    subscription = node.create_subscription(Odometry, topic, callback, qos)
    deadline = time.monotonic() + 4.0
    try:
        while (
            rclpy.ok()
            and len(offsets) < int(sample_count)
            and time.monotonic() < deadline
        ):
            rclpy.spin_once(node, timeout_sec=0.2)
    finally:
        del subscription
        node.destroy_node()
        rclpy.shutdown()
    return offsets, stamps


def main():
    healthy = True
    healthy &= check(os.environ.get("ROS_DISTRO") == "foxy", "ROS_DISTRO=foxy", "source scripts/env.sh first", True)
    healthy &= check(
        os.environ.get("RMW_IMPLEMENTATION") == "rmw_cyclonedds_cpp",
        "CycloneDDS RMW selected",
        "RMW_IMPLEMENTATION is not rmw_cyclonedds_cpp",
        True,
    )
    uri = os.environ.get("CYCLONEDDS_URI", "")
    check(bool(uri), f"CycloneDDS config: {uri}", "CYCLONEDDS_URI is unset")
    sdk = pathlib.Path(os.environ.get("GO2_SDK_PYTHON", ""))
    healthy &= check((sdk / "unitree_sdk2py").is_dir(), f"Unitree Python SDK: {sdk}", "Unitree Python SDK not found", True)
    no_shm = sdk / "cyclonedds" / "install_noshm" / "lib"
    healthy &= check(str(no_shm) in os.environ.get("LD_LIBRARY_PATH", ""), "no-SHM CycloneDDS library is first-class", "no-SHM SDK DDS library missing from LD_LIBRARY_PATH", True)
    cyclonedds_python = pathlib.Path(
        os.environ.get("GO2_CYCLONEDDS_PYTHON", "")
    )
    healthy &= check(
        (cyclonedds_python / "cyclonedds").is_dir(),
        f"runtime-user CycloneDDS Python package: {cyclonedds_python}",
        f"CycloneDDS Python package missing under {cyclonedds_python}",
        True,
    )
    healthy &= check(shutil.which("ros2") is not None, "ros2 CLI found", "ros2 CLI not found", True)
    detector_python = pathlib.Path(
        os.environ.get(
            "GO2_SEMANTIC_PYTHON",
            "/home/unitree/Documents/demov1/venv-yolo/bin/python",
        )
    )
    detector_model = pathlib.Path("/home/unitree/Documents/demov1/yolov8n.pt")
    healthy &= check(
        detector_python.is_file(),
        f"semantic detector interpreter: {detector_python}",
        f"missing semantic detector interpreter: {detector_python}",
        True,
    )
    healthy &= check(
        detector_model.is_file(),
        f"local semantic detector model: {detector_model}",
        f"missing local semantic detector model: {detector_model}",
        True,
    )
    if detector_python.is_file():
        detector_check = subprocess.run(
            [
                str(detector_python),
                "-c",
                # cv_bridge's Foxy extension must see OpenCV/NumPy initialized
                # first when running under the detector venv on this Jetson.
                "import numpy, cv2, cv_bridge, rclpy, torch, ultralytics",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
        healthy &= check(
            detector_check.returncode == 0,
            "Foxy + Jetson Torch + Ultralytics import together",
            "semantic detector environment is incompatible with sourced ROS",
            True,
        )

    for package in REQUIRED_PACKAGES:
        result = subprocess.run(
            ["ros2", "pkg", "prefix", package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        healthy &= check(result.returncode == 0, f"ROS package {package}", f"missing ROS package {package}", True)

    try:
        result = subprocess.run(
            ["ros2", "topic", "list"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        topics = set(result.stdout.splitlines())
        missing = sorted(REQUIRED_TOPICS - topics)
        healthy &= check(not missing, "all required Go2 sensor topics are visible", "missing topics: " + ", ".join(missing), True)
    except subprocess.TimeoutExpired:
        healthy = False
        print("[FAIL] timed out discovering DDS topics")

    try:
        offsets, stamps = measure_raw_clock_offset()
        progressing = (
            len(stamps) >= 3
            and all(stamp > 0 for stamp in stamps)
            and all(newer > older for older, newer in zip(stamps, stamps[1:]))
        )
        healthy &= check(
            progressing,
            "raw Go2 odometry timestamps are nonzero and progressing",
            "raw Go2 odometry timestamps are missing, repeated, or regressing",
            True,
        )
        if offsets:
            median_offset = statistics.median(offsets)
            offset_span = max(offsets) - min(offsets)
            print(
                "[INFO] raw host-minus-Go2 clock offset: "
                f"{median_offset:.3f}s (sample span {offset_span:.3f}s)"
            )
            if abs(median_offset) > 1.0:
                print(
                    "[INFO] large native offset is expected on this unit; "
                    "sensor_time_bridge must be running for all /go2 sensor topics"
                )
    except Exception as error:
        healthy = False
        print(f"[FAIL] could not measure raw sensor clock: {error}")

    iface = os.environ.get("GO2_NETWORK_INTERFACE", "eth0")
    healthy &= check(pathlib.Path("/sys/class/net", iface).exists(), f"network interface {iface}", f"interface {iface} does not exist", True)
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
        import yaml  # noqa: F401
        check(True, "OpenCV, NumPy, and PyYAML import", "")
    except ImportError as error:
        healthy = False
        print(f"[FAIL] Python dependency: {error}")

    print("\nPreflight passed." if healthy else "\nPreflight failed; do not enable motion.")
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
