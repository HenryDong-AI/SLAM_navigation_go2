from glob import glob
from setuptools import find_packages, setup


package_name = "go2_navigation"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools", "PyYAML", "numpy"],
    zip_safe=True,
    maintainer="Go2 SLAM Navigation Maintainers",
    maintainer_email="robotics@example.com",
    description="Safe Nav2 route execution and cloud-to-scan conversion for Go2 EDU.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "cloud_to_scan = go2_navigation.cloud_to_scan:main",
            "go2_route = go2_navigation.route_cli:main",
        ],
    },
)
