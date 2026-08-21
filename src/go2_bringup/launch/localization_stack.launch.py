"""Bring up saved-map AMCL navigation and a disarmed Go2 motion bridge."""

import os
import site

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _source(package, filename):
    return PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory(package), "launch", filename)
    )


def generate_launch_description():
    navigation_share = get_package_share_directory("go2_navigation")
    map_yaml = LaunchConfiguration("map")
    use_rviz = LaunchConfiguration("use_rviz")
    use_sim_time = LaunchConfiguration("use_sim_time")
    start_camera = LaunchConfiguration("start_camera")
    start_motion_bridge = LaunchConfiguration("start_motion_bridge")
    network_interface = LaunchConfiguration("network_interface")
    sdk_python_path = LaunchConfiguration("sdk_python_path")
    cyclonedds_python_path = LaunchConfiguration("cyclonedds_python_path")

    return LaunchDescription(
        [
            DeclareLaunchArgument("map", description="Absolute path to a saved map YAML"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("start_camera", default_value="false"),
            DeclareLaunchArgument("start_motion_bridge", default_value="true"),
            DeclareLaunchArgument(
                "network_interface", default_value=os.environ.get("GO2_NETWORK_INTERFACE", "eth0")
            ),
            DeclareLaunchArgument(
                "sdk_python_path",
                default_value=os.environ.get(
                    "GO2_SDK_PYTHON", "/home/unitree/Documents/demov1/unitree_sdk2_python"
                ),
            ),
            DeclareLaunchArgument(
                "cyclonedds_python_path",
                default_value=os.environ.get(
                    "GO2_CYCLONEDDS_PYTHON",
                    site.getusersitepackages(),
                ),
            ),
            IncludeLaunchDescription(
                _source("go2_robot_bridge", "robot_bridge.launch.py"),
                launch_arguments={
                    "network_interface": network_interface,
                    "sdk_python_path": sdk_python_path,
                    "cyclonedds_python_path": cyclonedds_python_path,
                    "start_camera": start_camera,
                    "start_motion": start_motion_bridge,
                }.items(),
            ),
            IncludeLaunchDescription(
                _source("go2_navigation", "localization.launch.py"),
                launch_arguments={
                    "map": map_yaml,
                    "start_sensor_bridge": "false",
                    "use_rviz": "false",
                    "use_sim_time": use_sim_time,
                }.items(),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=[
                    "-d",
                    os.path.join(navigation_share, "rviz", "go2_navigation.rviz"),
                    "-f",
                    "map",
                ],
                parameters=[{"use_sim_time": use_sim_time}],
                condition=IfCondition(use_rviz),
                output="screen",
            ),
        ]
    )
