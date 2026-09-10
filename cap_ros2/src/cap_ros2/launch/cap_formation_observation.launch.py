"""Launch the first-level formation observation node for RViz."""

from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _load_node_defaults(params_file: Path) -> dict:
    try:
        with params_file.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
    except Exception:
        return {}
    node_config = config.get("formation_observation_node", {})
    if not isinstance(node_config, dict):
        return {}
    parameters = node_config.get("ros__parameters", {})
    return parameters if isinstance(parameters, dict) else {}


def _launch_default(parameters: dict, name: str, fallback) -> str:
    value = parameters.get(name, fallback)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def generate_launch_description():
    package_share = Path(get_package_share_directory("cap_ros2"))
    config_dir = package_share / "config"
    params_file = config_dir / "formation_node_params.yaml"
    defaults = _load_node_defaults(params_file)

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "self_slot_id",
                default_value=_launch_default(defaults, "self_slot_id", 1),
            ),
            DeclareLaunchArgument(
                "formation_id",
                default_value=_launch_default(defaults, "formation_id", 0),
            ),
            DeclareLaunchArgument(
                "formation_id_topic",
                default_value=_launch_default(
                    defaults,
                    "formation_id_topic",
                    "/cap/formation/id",
                ),
            ),
            DeclareLaunchArgument(
                "legacy_details_topic",
                default_value=_launch_default(
                    defaults,
                    "legacy_details_topic",
                    "",
                ),
            ),
            DeclareLaunchArgument(
                "legacy_details_camera_id",
                default_value=_launch_default(
                    defaults,
                    "legacy_details_camera_id",
                    0,
                ),
            ),
            DeclareLaunchArgument(
                "publish_tf",
                default_value=_launch_default(defaults, "publish_tf", True),
            ),
            Node(
                package="cap_ros2",
                executable="formation_observation_node",
                name="formation_observation_node",
                output="screen",
                parameters=[
                    str(params_file),
                    {
                        "config_dir": str(config_dir),
                        "self_slot_id": ParameterValue(
                            LaunchConfiguration("self_slot_id"),
                            value_type=int,
                        ),
                        "formation_id": ParameterValue(
                            LaunchConfiguration("formation_id"),
                            value_type=int,
                        ),
                        "formation_id_topic": LaunchConfiguration(
                            "formation_id_topic"
                        ),
                        "legacy_details_topic": LaunchConfiguration(
                            "legacy_details_topic"
                        ),
                        "legacy_details_camera_id": ParameterValue(
                            LaunchConfiguration("legacy_details_camera_id"),
                            value_type=int,
                        ),
                        "publish_tf": ParameterValue(
                            LaunchConfiguration("publish_tf"),
                            value_type=bool,
                        ),
                    },
                ],
            ),
        ]
    )
