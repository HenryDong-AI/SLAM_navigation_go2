"""Launch Go2 time normalization, camera, and safety-gated motion bridges."""

import os
import site

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory("go2_robot_bridge")
    safety_file = os.path.join(share, "config", "safety.yaml")
    calibration_file = os.path.join(share, "config", "camera_info.yaml")

    network_interface = LaunchConfiguration("network_interface")
    sdk_python_path = LaunchConfiguration("sdk_python_path")
    cyclonedds_python_path = LaunchConfiguration("cyclonedds_python_path")
    raw_odom_topic = LaunchConfiguration("raw_odom_topic")
    raw_cloud_base_topic = LaunchConfiguration("raw_cloud_base_topic")
    raw_cloud_deskewed_topic = LaunchConfiguration("raw_cloud_deskewed_topic")
    camera_rate_hz = LaunchConfiguration("camera_rate_hz")

    return LaunchDescription(
        [
            DeclareLaunchArgument("network_interface", default_value="eth0"),
            DeclareLaunchArgument(
                "sdk_python_path",
                default_value="/home/unitree/Documents/demov1/unitree_sdk2_python",
            ),
            DeclareLaunchArgument(
                "cyclonedds_python_path",
                default_value=os.environ.get(
                    "GO2_CYCLONEDDS_PYTHON",
                    site.getusersitepackages(),
                ),
            ),
            DeclareLaunchArgument("start_camera", default_value="true"),
            DeclareLaunchArgument("start_motion", default_value="true"),
            DeclareLaunchArgument("camera_rate_hz", default_value="5.0"),
            DeclareLaunchArgument(
                "raw_odom_topic", default_value="/utlidar/robot_odom"
            ),
            DeclareLaunchArgument(
                "raw_cloud_base_topic", default_value="/utlidar/cloud_base"
            ),
            DeclareLaunchArgument(
                "raw_cloud_deskewed_topic",
                default_value="/utlidar/cloud_deskewed",
            ),
            # This node is deliberately unconditional: every downstream sensor
            # consumer gets one host-clock boundary unless it is launched alone.
            Node(
                package="go2_robot_bridge",
                executable="sensor_time_bridge",
                name="sensor_time_bridge",
                output="screen",
                emulate_tty=True,
                parameters=[
                    {
                        "raw_odom_topic": raw_odom_topic,
                        "raw_cloud_base_topic": raw_cloud_base_topic,
                        "raw_cloud_deskewed_topic": raw_cloud_deskewed_topic,
                    }
                ],
            ),
            Node(
                package="go2_robot_bridge",
                executable="camera_bridge",
                name="camera_bridge",
                output="screen",
                emulate_tty=True,
                condition=IfCondition(LaunchConfiguration("start_camera")),
                parameters=[
                    {
                        "network_interface": network_interface,
                        "sdk_python_path": sdk_python_path,
                        "cyclonedds_python_path": cyclonedds_python_path,
                        "calibration_file": calibration_file,
                        "publish_rate_hz": camera_rate_hz,
                    }
                ],
            ),
            Node(
                package="go2_robot_bridge",
                executable="motion_bridge",
                name="motion_bridge",
                output="screen",
                emulate_tty=True,
                condition=IfCondition(LaunchConfiguration("start_motion")),
                parameters=[
                    safety_file,
                    {
                        "network_interface": network_interface,
                        "sdk_python_path": sdk_python_path,
                        "cyclonedds_python_path": cyclonedds_python_path,
                    },
                ],
            ),
        ]
    )
