"""Launch bounded Go2 mapping and its single odometry TF authority."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = LaunchConfiguration("config_file")
    cloud_topic = LaunchConfiguration("cloud_topic")
    odom_topic = LaunchConfiguration("odom_topic")
    world_frame = LaunchConfiguration("world_frame")
    base_frame = LaunchConfiguration("base_frame")
    output_dir = LaunchConfiguration("output_dir")
    load_state_path = LaunchConfiguration("load_state_path")
    publish_odom_tf = LaunchConfiguration("publish_odom_tf")
    require_time_sync_status = LaunchConfiguration("require_time_sync_status")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("go2_mapping"), "config", "mapping.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "cloud_topic", default_value="/go2/lidar/cloud_deskewed"
            ),
            DeclareLaunchArgument(
                "odom_topic", default_value="/go2/odom"
            ),
            DeclareLaunchArgument("world_frame", default_value="odom"),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument("output_dir", default_value="~/go2_maps"),
            DeclareLaunchArgument("load_state_path", default_value=""),
            DeclareLaunchArgument("publish_odom_tf", default_value="true"),
            DeclareLaunchArgument(
                "require_time_sync_status", default_value="true"
            ),
            Node(
                package="go2_mapping",
                executable="go2_mapping_node",
                name="go2_mapping",
                output="screen",
                parameters=[
                    config_file,
                    {
                        "cloud_topic": cloud_topic,
                        "odom_topic": odom_topic,
                        "world_frame": world_frame,
                        "output_dir": output_dir,
                        "load_state_path": load_state_path,
                        "require_time_sync_status": require_time_sync_status,
                    },
                ],
            ),
            # MappingNode intentionally sends no TF.  Keeping this bridge as the
            # sole authority prevents two odom->base transforms in one launch.
            Node(
                package="go2_mapping",
                executable="odom_tf_bridge",
                name="go2_odom_tf_bridge",
                output="screen",
                condition=IfCondition(publish_odom_tf),
                parameters=[
                    config_file,
                    {
                        "odom_topic": odom_topic,
                        "odom_frame": world_frame,
                        "base_frame": base_frame,
                        "require_time_sync_status": require_time_sync_status,
                    },
                ],
            ),
        ]
    )
