"""Bring up live Go2 mapping, semantic perception, Nav2, and disarmed motion."""

import os
import site

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _include(package, filename, arguments, condition=None):
    source = PythonLaunchDescriptionSource(
        os.path.join(get_package_share_directory(package), "launch", filename)
    )
    return IncludeLaunchDescription(
        source,
        launch_arguments=arguments.items(),
        condition=condition,
    )


def generate_launch_description():
    mapping_share = get_package_share_directory("go2_mapping")
    semantic_share = get_package_share_directory("go2_semantic_mapping")
    navigation_share = get_package_share_directory("go2_navigation")

    network_interface = LaunchConfiguration("network_interface")
    sdk_python_path = LaunchConfiguration("sdk_python_path")
    cyclonedds_python_path = LaunchConfiguration("cyclonedds_python_path")
    start_camera = LaunchConfiguration("start_camera")
    camera_rate_hz = LaunchConfiguration("camera_rate_hz")
    start_motion_bridge = LaunchConfiguration("start_motion_bridge")
    start_semantics = LaunchConfiguration("start_semantics")
    start_navigation = LaunchConfiguration("start_navigation")
    use_rviz = LaunchConfiguration("use_rviz")
    use_sim_time = LaunchConfiguration("use_sim_time")
    mapping_config = LaunchConfiguration("mapping_config")
    semantic_config = LaunchConfiguration("semantic_config")
    navigation_config = LaunchConfiguration("navigation_config")
    map_output_dir = LaunchConfiguration("map_output_dir")
    semantic_output_dir = LaunchConfiguration("semantic_output_dir")

    return LaunchDescription(
        [
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
            DeclareLaunchArgument("start_camera", default_value="true"),
            DeclareLaunchArgument("camera_rate_hz", default_value="5.0"),
            DeclareLaunchArgument("start_motion_bridge", default_value="true"),
            DeclareLaunchArgument("start_semantics", default_value="true"),
            DeclareLaunchArgument("start_navigation", default_value="true"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument(
                "mapping_config",
                default_value=os.path.join(mapping_share, "config", "mapping.yaml"),
            ),
            DeclareLaunchArgument(
                "semantic_config",
                default_value=os.path.join(semantic_share, "config", "semantic_mapping.yaml"),
            ),
            DeclareLaunchArgument(
                "navigation_config",
                default_value=os.path.join(navigation_share, "config", "nav2_live.yaml"),
            ),
            DeclareLaunchArgument(
                "map_output_dir", default_value="/home/unitree/SLAM_nav/maps"
            ),
            DeclareLaunchArgument(
                "semantic_output_dir", default_value="/home/unitree/SLAM_nav/semantic_maps"
            ),
            _include(
                "go2_robot_bridge",
                "robot_bridge.launch.py",
                {
                    "network_interface": network_interface,
                    "sdk_python_path": sdk_python_path,
                    "cyclonedds_python_path": cyclonedds_python_path,
                    "start_camera": start_camera,
                    "camera_rate_hz": camera_rate_hz,
                    "start_motion": start_motion_bridge,
                },
            ),
            _include(
                "go2_mapping",
                "mapping.launch.py",
                {
                    "config_file": mapping_config,
                    "output_dir": map_output_dir,
                    "publish_odom_tf": "true",
                },
            ),
            _include(
                "go2_semantic_mapping",
                "semantic_mapping.launch.py",
                {
                    "params_file": semantic_config,
                    "image_topic": "/go2/camera/image_rect",
                    "cloud_topic": "/go2/lidar/cloud_base",
                    "odom_topic": "/go2/odom",
                    "save_directory": semantic_output_dir,
                },
                condition=IfCondition(start_semantics),
            ),
            _include(
                "go2_navigation",
                "navigation.launch.py",
                {
                    "params_file": navigation_config,
                    "use_rviz": "false",
                    "use_sim_time": use_sim_time,
                },
                condition=IfCondition(start_navigation),
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=[
                    "-d",
                    os.path.join(navigation_share, "rviz", "go2_navigation.rviz"),
                ],
                parameters=[{"use_sim_time": use_sim_time}],
                condition=IfCondition(use_rviz),
                output="screen",
            ),
        ]
    )
