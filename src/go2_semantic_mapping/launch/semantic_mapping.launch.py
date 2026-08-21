# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Go2 Semantic Mapping contributors

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_parameters = PathJoinSubstitution(
        [FindPackageShare("go2_semantic_mapping"), "config", "semantic_mapping.yaml"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_parameters),
            DeclareLaunchArgument(
                "python_executable",
                default_value=os.environ.get(
                    "GO2_SEMANTIC_PYTHON",
                    "/home/unitree/Documents/demov1/venv-yolo/bin/python",
                ),
            ),
            DeclareLaunchArgument("image_topic", default_value="/go2/camera/image_rect"),
            DeclareLaunchArgument("cloud_topic", default_value="/go2/lidar/cloud_base"),
            DeclareLaunchArgument("odom_topic", default_value="/go2/odom"),
            DeclareLaunchArgument("save_directory", default_value="~/go2_semantic_maps"),
            Node(
                package="go2_semantic_mapping",
                executable="semantic_mapping_node",
                name="semantic_mapping",
                # The selected environment provides Jetson Torch/Ultralytics.
                # It can also import the sourced Foxy rclpy/cv_bridge modules.
                prefix=[LaunchConfiguration("python_executable")],
                output="screen",
                parameters=[
                    LaunchConfiguration("params_file"),
                    {
                        "image_topic": LaunchConfiguration("image_topic"),
                        "cloud_topic": LaunchConfiguration("cloud_topic"),
                        "odom_topic": LaunchConfiguration("odom_topic"),
                        "save_directory": LaunchConfiguration("save_directory"),
                    },
                ],
            ),
        ]
    )
