"""Saved-map localization and Nav2 bringup for the Go2 EDU."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    share = get_package_share_directory("go2_navigation")
    nav_params = LaunchConfiguration("params_file")
    localization_params = LaunchConfiguration("localization_params_file")
    map_yaml = LaunchConfiguration("map")
    use_rviz = LaunchConfiguration("use_rviz")
    use_sim_time = LaunchConfiguration("use_sim_time")
    cloud_topic = LaunchConfiguration("cloud_topic")
    start_sensor_bridge = LaunchConfiguration("start_sensor_bridge")

    configured_nav_params = RewrittenYaml(
        source_file=nav_params,
        param_rewrites={
            "bt_navigator.ros__parameters.global_frame": "map",
            "global_costmap.global_costmap.ros__parameters.global_frame": "map",
            "use_sim_time": use_sim_time,
        },
        convert_types=True,
    )
    nav_common = [configured_nav_params]
    localization_common = [localization_params, {"use_sim_time": use_sim_time}]
    rviz_config = os.path.join(share, "rviz", "go2_navigation.rviz")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file", default_value=os.path.join(share, "config", "nav2_live.yaml")
            ),
            DeclareLaunchArgument(
                "localization_params_file",
                default_value=os.path.join(share, "config", "localization.yaml"),
            ),
            DeclareLaunchArgument("map", description="Absolute path to a saved map YAML file"),
            DeclareLaunchArgument("cloud_topic", default_value="/go2/lidar/cloud_base"),
            DeclareLaunchArgument("start_sensor_bridge", default_value="true"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(
                        get_package_share_directory("go2_robot_bridge"),
                        "launch",
                        "robot_bridge.launch.py",
                    )
                ),
                launch_arguments={
                    "start_camera": "false",
                    "start_motion": "false",
                }.items(),
                condition=IfCondition(start_sensor_bridge),
            ),
            Node(
                package="go2_mapping",
                executable="odom_tf_bridge",
                name="odom_tf_bridge",
                output="screen",
                parameters=[{"odom_topic": "/go2/odom"}],
            ),
            Node(
                package="go2_navigation",
                executable="cloud_to_scan",
                name="cloud_to_scan",
                output="screen",
                parameters=[{"cloud_topic": cloud_topic, "output_frame": "base_link"}],
            ),
            Node(
                package="nav2_map_server",
                executable="map_server",
                name="map_server",
                output="screen",
                parameters=localization_common + [{"yaml_filename": map_yaml}],
            ),
            Node(
                package="nav2_amcl",
                executable="amcl",
                name="amcl",
                output="screen",
                parameters=localization_common,
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_localization",
                output="screen",
                parameters=localization_common,
            ),
            Node(
                package="nav2_controller",
                executable="controller_server",
                name="controller_server",
                output="screen",
                parameters=nav_common,
            ),
            Node(
                package="nav2_planner",
                executable="planner_server",
                name="planner_server",
                output="screen",
                parameters=nav_common,
            ),
            Node(
                package="nav2_recoveries",
                executable="recoveries_server",
                name="recoveries_server",
                output="screen",
                parameters=nav_common,
            ),
            Node(
                package="nav2_bt_navigator",
                executable="bt_navigator",
                name="bt_navigator",
                output="screen",
                parameters=nav_common,
            ),
            Node(
                package="nav2_waypoint_follower",
                executable="waypoint_follower",
                name="waypoint_follower",
                output="screen",
                parameters=nav_common,
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="lifecycle_manager_navigation",
                output="screen",
                parameters=nav_common,
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                arguments=["-d", rviz_config, "-f", "map"],
                parameters=[{"use_sim_time": use_sim_time}],
                condition=IfCondition(use_rviz),
                output="screen",
            ),
        ]
    )
