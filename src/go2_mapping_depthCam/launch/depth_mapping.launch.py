"""Launch direct D435i capture and the optional depth mapping/viewer nodes."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    python_executable = LaunchConfiguration("python_executable")
    start_camera = LaunchConfiguration("start_camera")
    start_mapper = LaunchConfiguration("start_mapper")
    use_viewer = LaunchConfiguration("use_viewer")
    publish_odom_tf = LaunchConfiguration("publish_odom_tf")
    odom_tf_config = LaunchConfiguration("odom_tf_config")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("go2_mapping_depthcam"),
                        "config",
                        "depth_mapping.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument(
                "python_executable",
                default_value=os.environ.get(
                    "GO2_DEPTH_PYTHON",
                    "/home/unitree/SLAM_nav/.conda/envs/slam_nav/bin/python",
                ),
            ),
            DeclareLaunchArgument("start_camera", default_value="true"),
            DeclareLaunchArgument("start_mapper", default_value="true"),
            DeclareLaunchArgument("use_viewer", default_value="false"),
            DeclareLaunchArgument("publish_odom_tf", default_value="true"),
            DeclareLaunchArgument(
                "odom_tf_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("go2_mapping"), "config", "mapping.yaml"]
                ),
            ),
            Node(
                package="go2_mapping_depthcam",
                executable="depth_camera_bridge",
                name="depth_camera_bridge",
                prefix=[python_executable],
                parameters=[params_file],
                output="screen",
                condition=IfCondition(start_camera),
            ),
            Node(
                package="go2_mapping_depthcam",
                executable="depth_mapping_node",
                name="go2_mapping_depthcam",
                prefix=[python_executable],
                parameters=[params_file],
                output="screen",
                condition=IfCondition(start_mapper),
            ),
            Node(
                package="go2_mapping_depthcam",
                executable="rgbd_viewer",
                name="rgbd_viewer",
                prefix=[python_executable],
                parameters=[params_file],
                output="screen",
                condition=IfCondition(use_viewer),
            ),
            # The mapper publishes no TF. Keep the same single odom->base_link
            # authority used by the LiDAR backend.
            Node(
                package="go2_mapping",
                executable="odom_tf_bridge",
                name="go2_odom_tf_bridge",
                parameters=[odom_tf_config],
                output="screen",
                condition=IfCondition(publish_odom_tf),
            ),
        ]
    )
