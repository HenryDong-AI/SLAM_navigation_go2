# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Go2 Semantic Mapping contributors

from setuptools import find_packages, setup


package_name = "go2_semantic_mapping"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md", "LICENSE"]),
        ("share/" + package_name + "/config", ["config/semantic_mapping.yaml"]),
        ("share/" + package_name + "/launch", ["launch/semantic_mapping.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Go2 Semantic Mapping contributors",
    maintainer_email="maintainer@example.com",
    description="Persistent RGB and LiDAR semantic voxel mapping for the Unitree Go2.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "semantic_mapping_node = go2_semantic_mapping.semantic_mapping_node:main",
        ],
    },
)
