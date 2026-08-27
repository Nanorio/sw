"""Launch the complete CAP stereo detection pipeline."""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    project_candidates = [
        Path(__file__).resolve().parents[4],
        Path(__file__).resolve().parents[3].parent,
        Path.cwd().parent,
    ]
    default_project_root = None
    for candidate in project_candidates:
        if (candidate / "weights").exists():
            default_project_root = str(candidate)
            break
    if default_project_root is None:
        default_project_root = os.environ.get(
            "CAP2_ROOT", str(Path.cwd().parent)
        )
    config_dir = os.path.join(get_package_share_directory("cap_ros2"), "config")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "project_root",
                default_value=default_project_root,
                description="Parent project directory containing weights/outcome.",
            ),
            DeclareLaunchArgument(
                "use_display",
                default_value="true",
                description="Launch the OpenCV display node.",
            ),
            DeclareLaunchArgument(
                "model_path",
                default_value="",
                description="Optional absolute YOLO weights path.",
            ),
            DeclareLaunchArgument(
                "camera_id",
                default_value="-1",
                description="Camera device id; -1 uses capture.yaml.",
            ),
            Node(
                package="cap_ros2",
                executable="camera_node",
                name="camera_node",
                output="screen",
                parameters=[
                    {
                        "config_dir": config_dir,
                        "project_root": LaunchConfiguration("project_root"),
                        "camera_id": LaunchConfiguration("camera_id"),
                    }
                ],
            ),
            Node(
                package="cap_ros2",
                executable="sgbm_node",
                name="sgbm_node",
                output="screen",
                parameters=[{"config_dir": config_dir}],
            ),
            Node(
                package="cap_ros2",
                executable="yolo_node",
                name="yolo_node",
                output="screen",
                parameters=[
                    {
                        "config_dir": config_dir,
                        "project_root": LaunchConfiguration("project_root"),
                        "model_path": LaunchConfiguration("model_path"),
                    }
                ],
            ),
            Node(
                package="cap_ros2",
                executable="pose_node",
                name="pose_node",
                output="screen",
                parameters=[
                    {
                        "config_dir": config_dir,
                        "project_root": LaunchConfiguration("project_root"),
                    }
                ],
            ),
            Node(
                package="cap_ros2",
                executable="display_node",
                name="display_node",
                output="screen",
                condition=IfCondition(LaunchConfiguration("use_display")),
            ),
        ]
    )
