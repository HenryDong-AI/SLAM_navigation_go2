from glob import glob
from setuptools import find_packages, setup


package_name = "go2_robot_bridge"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md", "LICENSE"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="SLAM Nav Maintainers",
    maintainer_email="maintainer@example.com",
    description="Safety-oriented Unitree Go2 ROS 2 hardware bridges.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "camera_bridge = go2_robot_bridge.camera_bridge:main",
            "motion_bridge = go2_robot_bridge.motion_bridge:main",
            "sensor_time_bridge = go2_robot_bridge.sensor_time_bridge:main",
        ],
    },
)
