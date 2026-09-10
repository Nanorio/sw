"""Launch the complete CAP stereo detection pipeline."""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _int_parameter(name: str):
    return ParameterValue(LaunchConfiguration(name), value_type=int)


def _float_parameter(name: str):
    return ParameterValue(LaunchConfiguration(name), value_type=float)


def _bool_parameter(name: str):
    return ParameterValue(LaunchConfiguration(name), value_type=bool)


def _find_default_project_root() -> str:
    env_root = os.environ.get("CAP2_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.append(Path.cwd())
    candidates.append(Path.cwd().parent)
    candidates.extend(Path(__file__).resolve().parents)

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if (resolved / "weights").exists():
            return str(resolved)
    return str(Path.cwd().resolve())


def generate_launch_description():
    default_project_root = _find_default_project_root()
    config_dir = os.path.join(get_package_share_directory("cap_ros2"), "config")

    common_parameters = {
        "config_dir": LaunchConfiguration("config_dir"),
    }

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "project_root",
                default_value=default_project_root,
                description="Parent project directory containing weights/outcome.",
            ),
            DeclareLaunchArgument(
                "config_dir",
                default_value=config_dir,
                description="CAP configuration directory.",
            ),
            DeclareLaunchArgument(
                "use_display",
                default_value="true",
                description="Launch the OpenCV display node.",
            ),
            DeclareLaunchArgument(
                "use_window",
                default_value="true",
                description="Show OpenCV display windows when display_node is launched.",
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
            DeclareLaunchArgument("jpeg_quality", default_value="88"),
            DeclareLaunchArgument(
                "yolo_conf",
                default_value="-1.0",
                description="YOLO confidence override; -1 uses yolo_params.yaml.",
            ),
            DeclareLaunchArgument(
                "device",
                default_value="",
                description="YOLO device override, for example 0 or cpu.",
            ),
            DeclareLaunchArgument("roi_margin", default_value="40"),
            DeclareLaunchArgument("min_roi_size", default_value="64"),
            DeclareLaunchArgument("filter_alpha", default_value="0.35"),
            DeclareLaunchArgument("max_trajectory_pts", default_value="200"),
            Node(
                package="cap_ros2",
                executable="camera_node",
                name="camera_node",
                output="screen",
                emulate_tty=True,
                parameters=[
                    common_parameters,
                    {
                        "project_root": LaunchConfiguration("project_root"),
                        "camera_id": _int_parameter("camera_id"),
                        "jpeg_quality": _int_parameter("jpeg_quality"),
                    }
                ],
            ),
            Node(
                package="cap_ros2",
                executable="sgbm_node",
                name="sgbm_node",
                output="screen",
                emulate_tty=True,
                parameters=[
                    common_parameters,
                    {
                        "roi_margin": _int_parameter("roi_margin"),
                        "min_roi_size": _int_parameter("min_roi_size"),
                    },
                ],
            ),
            Node(
                package="cap_ros2",
                executable="yolo_node",
                name="yolo_node",
                output="screen",
                emulate_tty=True,
                parameters=[
                    common_parameters,
                    {
                        "project_root": LaunchConfiguration("project_root"),
                        "model_path": LaunchConfiguration("model_path"),
                        "conf": _float_parameter("yolo_conf"),
                        "device": LaunchConfiguration("device"),
                        "jpeg_quality": _int_parameter("jpeg_quality"),
                    }
                ],
            ),
            Node(
                package="cap_ros2",
                executable="pose_node",
                name="pose_node",
                output="screen",
                emulate_tty=True,
                parameters=[
                    common_parameters,
                    {
                        "project_root": LaunchConfiguration("project_root"),
                        "filter_alpha": _float_parameter("filter_alpha"),
                        "max_trajectory_pts": _int_parameter("max_trajectory_pts"),
                    }
                ],
            ),
            Node(
                package="cap_ros2",
                executable="display_node",
                name="display_node",
                output="screen",
                condition=IfCondition(LaunchConfiguration("use_display")),
                emulate_tty=True,
                parameters=[
                    {
                        "use_window": _bool_parameter("use_window"),
                    }
                ],
            ),
        ]
    )
