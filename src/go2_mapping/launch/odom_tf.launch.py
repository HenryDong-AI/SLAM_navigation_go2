"""Run only the odometry TF bridge, for example with saved-map localization."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = LaunchConfiguration("config_file")
    odom_topic = LaunchConfiguration("odom_topic")
    odom_frame = LaunchConfiguration("odom_frame")
    base_frame = LaunchConfiguration("base_frame")
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
                "odom_topic", default_value="/go2/odom"
            ),
            DeclareLaunchArgument("odom_frame", default_value="odom"),
            DeclareLaunchArgument("base_frame", default_value="base_link"),
            DeclareLaunchArgument(
                "require_time_sync_status", default_value="true"
            ),
            Node(
                package="go2_mapping",
                executable="odom_tf_bridge",
                name="go2_odom_tf_bridge",
                output="screen",
                parameters=[
                    config_file,
                    {
                        "odom_topic": odom_topic,
                        "odom_frame": odom_frame,
                        "base_frame": base_frame,
                        "require_time_sync_status": require_time_sync_status,
                    },
                ],
            ),
        ]
    )
