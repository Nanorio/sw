import os
from glob import glob

from setuptools import find_packages, setup


package_name = "cap_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="CAP",
    maintainer_email="cap@example.com",
    description="ROS2 Python nodes for CAP stereo detection.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "camera_node = cap_ros2.camera_node:main",
            "yolo_node = cap_ros2.yolo_node:main",
            "sgbm_node = cap_ros2.sgbm_node:main",
            "pose_node = cap_ros2.pose_node:main",
            "display_node = cap_ros2.display_node:main",
        ],
    },
)
