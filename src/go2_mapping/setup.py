from glob import glob
from setuptools import find_packages, setup


package_name = "go2_mapping"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="SLAM Nav Maintainers",
    maintainer_email="maintainer@example.com",
    description=(
        "Bounded 3D voxel mapping and 2D occupancy projection for the Unitree Go2"
    ),
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "go2_mapping_node = go2_mapping.mapping_node:main",
            "odom_tf_bridge = go2_mapping.odom_tf_bridge:main",
        ],
    },
)
