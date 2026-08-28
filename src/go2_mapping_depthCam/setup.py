from glob import glob

from setuptools import find_packages, setup


package_name = "go2_mapping_depthcam"


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
    description="Direct RealSense RGB-D bridge and voxel mapping backend for Go2",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "depth_camera_bridge = go2_mapping_depthcam.depth_camera_bridge:main",
            "extrinsic_calibrator = go2_mapping_depthcam.extrinsic_calibrator:main",
            "depth_mapping_node = go2_mapping_depthcam.depth_mapping_node:main",
            "rgbd_viewer = go2_mapping_depthcam.rgbd_viewer:main",
        ],
    },
)
