"""First-level formation observation and RViz visualization node.

The node keeps the algorithm deliberately limited to geometry:

1. Receive target points in each OpenCV camera frame.
2. Transform each point into the robot body frame.
3. Convert the selected formation template into offsets relative to the
   local robot slot.
4. Publish body-frame observations, expected slots, TF and RViz markers.

All algorithm coordinates are millimeters. ROS visualization messages use
meters at the publishing boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PointStamped, Pose, PoseArray, TransformStamped
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Int32, MultiArrayDimension
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from .common import get_config_dir
from .formation_geometry import (
    CameraExtrinsics,
    FormationTemplates,
    quaternion_from_rotation_matrix,
)


@dataclass
class _Observation:
    point_camera_mm: np.ndarray
    received_ns: int


class FormationObservationNode(Node):
    def __init__(self):
        super().__init__("formation_observation_node")

        self.declare_parameter("config_dir", "")
        self.declare_parameter("camera_extrinsics_path", "")
        self.declare_parameter("formation_templates_path", "")
        self.declare_parameter("self_slot_id", 1)
        self.declare_parameter("formation_id", 0)
        self.declare_parameter("formation_id_topic", "/cap/formation/id")
        self.declare_parameter(
            "camera_point_topics",
            [
                "/cap/formation/camera_0/point",
                "/cap/formation/camera_1/point",
                "/cap/formation/camera_2/point",
            ],
        )
        self.declare_parameter(
            "legacy_details_topics",
            ["", "", ""],
        )
        self.declare_parameter("legacy_details_topic", "")
        self.declare_parameter("legacy_details_camera_id", 0)
        self.declare_parameter("stale_timeout_sec", 0.8)
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("axis_length_mm", 500.0)
        self.declare_parameter("camera_axis_length_mm", 180.0)

        config_dir = self._resolve_config_dir(
            str(self.get_parameter("config_dir").value)
        )
        extrinsics_path = self._resolve_config_path(
            str(self.get_parameter("camera_extrinsics_path").value),
            config_dir,
            "camera_extrinsics.yaml",
        )
        formations_path = self._resolve_config_path(
            str(self.get_parameter("formation_templates_path").value),
            config_dir,
            "formation_templates.yaml",
        )

        self.extrinsics = CameraExtrinsics.load(extrinsics_path)
        self.formations = FormationTemplates.load(formations_path)
        self.robot_frame = self.extrinsics.robot_frame
        self.self_slot_id = int(self.get_parameter("self_slot_id").value)
        self.formation_id = int(self.get_parameter("formation_id").value)
        self.stale_timeout_sec = max(
            0.05, float(self.get_parameter("stale_timeout_sec").value)
        )
        self.axis_length_mm = max(
            1.0, float(self.get_parameter("axis_length_mm").value)
        )
        self.camera_axis_length_mm = max(
            1.0, float(self.get_parameter("camera_axis_length_mm").value)
        )

        self._validate_formation(self.formation_id)
        self._observations: dict[int, _Observation] = {}

        self.observations_pub = self.create_publisher(
            PoseArray, "/cap/formation/observations", 10
        )
        self.expected_slots_pub = self.create_publisher(
            PoseArray, "/cap/formation/expected_slots", 10
        )
        self.observations_mm_pub = self.create_publisher(
            Float64MultiArray, "/cap/formation/observations_mm", 10
        )
        self.expected_slots_mm_pub = self.create_publisher(
            Float64MultiArray, "/cap/formation/expected_slots_mm", 10
        )
        self.state_pub = self.create_publisher(
            Float64MultiArray, "/cap/formation/state", 10
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, "/cap/formation/markers", 10
        )

        self._point_subscriptions = []
        point_topics = self._string_list(
            self.get_parameter("camera_point_topics").value
        )
        for camera_id, topic in enumerate(point_topics[:3]):
            if topic:
                self._point_subscriptions.append(
                    self.create_subscription(
                        PointStamped,
                        topic,
                        lambda msg, cid=camera_id: self._on_point(cid, msg),
                        10,
                    )
                )

        details_mappings = self._legacy_details_mappings()
        for camera_id, topic in details_mappings:
            self._point_subscriptions.append(
                self.create_subscription(
                    Float64MultiArray,
                    topic,
                    lambda msg, cid=camera_id: self._on_legacy_details(
                        cid, msg
                    ),
                    10,
                )
            )

        formation_id_topic = str(
            self.get_parameter("formation_id_topic").value
        ).strip()
        self.formation_id_sub = None
        if formation_id_topic:
            self.formation_id_sub = self.create_subscription(
                Int32,
                formation_id_topic,
                self._on_formation_id,
                10,
            )

        self.timer = self.create_timer(0.1, self._on_timer)
        self.tf_broadcaster: Optional[StaticTransformBroadcaster] = None
        if bool(self.get_parameter("publish_tf").value):
            self.tf_broadcaster = StaticTransformBroadcaster(self)
            self._publish_static_camera_tf()

        self.get_logger().info(
            "formation_observation_node ready: "
            f"robot_frame={self.robot_frame}, "
            f"self_slot_id={self.self_slot_id}, "
            f"formation_id={self.formation_id}"
        )
        self.get_logger().info(
            "inputs: "
            f"point_topics={point_topics[:3]}, "
            f"legacy_details_mappings={details_mappings}"
        )

    @staticmethod
    def _string_list(value) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value]
        return []

    @staticmethod
    def _resolve_config_dir(value: str) -> Path:
        candidates = []
        if value:
            candidates.append(Path(value).expanduser())

        try:
            candidates.append(Path(get_config_dir()))
        except Exception:
            pass

        module_dir = Path(__file__).resolve().parent
        candidates.extend(
            [
                module_dir / "config",
                module_dir.parent / "config",
                module_dir.parent.parent / "config",
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        return candidates[0].resolve() if candidates else Path.cwd().resolve()

    @staticmethod
    def _resolve_config_path(value: str, config_dir: Path, filename: str) -> Path:
        if value:
            path = Path(value).expanduser()
            if path.is_absolute():
                return path
            config_path = config_dir / path
            if config_path.exists():
                return config_path
            return Path.cwd() / path
        return config_dir / filename

    def _legacy_details_mappings(self) -> list[tuple[int, str]]:
        topic = str(self.get_parameter("legacy_details_topic").value).strip()
        if topic:
            camera_id = int(
                self.get_parameter("legacy_details_camera_id").value
            )
            mappings = [(camera_id, topic)]
        else:
            details_topics = self._string_list(
                self.get_parameter("legacy_details_topics").value
            )
            mappings = [
                (camera_id, topic)
                for camera_id, topic in enumerate(details_topics[:3])
                if topic
            ]

        valid_mappings = []
        for camera_id, topic in mappings:
            try:
                self.extrinsics.get(camera_id)
            except ValueError as exc:
                self.get_logger().warning(
                    f"ignore legacy details mapping {topic!r}: {exc}"
                )
                continue
            valid_mappings.append((camera_id, topic))
        return valid_mappings

    def _validate_formation(self, formation_id: int) -> None:
        self.formations.get_slots(formation_id)
        if self.self_slot_id not in self.formations.get_slots(formation_id):
            raise ValueError(
                f"self_slot_id={self.self_slot_id} is not present in "
                f"formation_id={formation_id}"
            )

    @staticmethod
    def _now_ns(node: Node) -> int:
        return int(node.get_clock().now().nanoseconds)

    @staticmethod
    def _stamp_ns(stamp, fallback_ns: int) -> int:
        value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        return value if value > 0 else fallback_ns

    @staticmethod
    def _valid_point(point: Iterable[float]) -> Optional[np.ndarray]:
        array = np.asarray(point, dtype=float).reshape(-1)
        if array.size != 3 or not np.all(np.isfinite(array)):
            return None
        if array[2] <= 0.0:
            return None
        return array

    def _on_point(self, camera_id: int, msg: PointStamped) -> None:
        point = self._valid_point(
            [msg.point.x, msg.point.y, msg.point.z]
        )
        if point is None:
            self._observations.pop(camera_id, None)
            return
        now_ns = self._now_ns(self)
        self._observations[camera_id] = _Observation(
            point_camera_mm=point,
            received_ns=now_ns,
        )

    def _on_legacy_details(
        self,
        camera_id: int,
        msg: Float64MultiArray,
    ) -> None:
        data = list(msg.data)
        if len(data) < 6:
            self.get_logger().warning(
                f"legacy details for camera {camera_id} has fewer than 6 values"
            )
            return
        if len(data) >= 11 and float(data[10]) <= 0.5:
            self._observations.pop(camera_id, None)
            return

        point = self._valid_point(data[3:6])
        if point is None:
            self._observations.pop(camera_id, None)
            return
        self._observations[camera_id] = _Observation(
            point_camera_mm=point,
            received_ns=self._now_ns(self),
        )

    def _on_formation_id(self, msg: Int32) -> None:
        formation_id = int(msg.data)
        try:
            self._validate_formation(formation_id)
        except ValueError as exc:
            self.get_logger().warning(str(exc))
            return
        if formation_id != self.formation_id:
            self.formation_id = formation_id
            self.get_logger().info(f"formation_id changed to {formation_id}")
            self._publish()

    def _on_timer(self) -> None:
        now_ns = self._now_ns(self)
        timeout_ns = int(self.stale_timeout_sec * 1_000_000_000)
        for camera_id, observation in list(self._observations.items()):
            if now_ns - observation.received_ns > timeout_ns:
                del self._observations[camera_id]
        self._publish()

    def _transformed_observations(self):
        transformed = []
        for camera_id, observation in sorted(self._observations.items()):
            camera = self.extrinsics.get(camera_id)
            point_robot_mm = camera.transform_point(observation.point_camera_mm)
            transformed.append((camera_id, observation, point_robot_mm))
        return transformed

    @staticmethod
    def _pose_array(
        stamp,
        frame_id: str,
        points_mm: Iterable[np.ndarray],
    ) -> PoseArray:
        message = PoseArray()
        message.header.stamp = stamp
        message.header.frame_id = frame_id
        for point_mm in points_mm:
            pose = Pose()
            pose.position.x = float(point_mm[0]) / 1000.0
            pose.position.y = float(point_mm[1]) / 1000.0
            pose.position.z = float(point_mm[2]) / 1000.0
            pose.orientation.w = 1.0
            message.poses.append(pose)
        return message

    @staticmethod
    def _array_message(data: list[float], label: str) -> Float64MultiArray:
        message = Float64MultiArray()
        message.layout.dim.append(MultiArrayDimension())
        message.layout.dim[0].label = label
        message.layout.dim[0].size = len(data)
        message.layout.dim[0].stride = len(data)
        message.data = data
        return message

    def _publish(self) -> None:
        now = self.get_clock().now().to_msg()
        transformed = self._transformed_observations()
        relative_slots = self.formations.relative_slots(
            self.formation_id,
            self.self_slot_id,
        )

        observation_points = [
            point_robot_mm for _, _, point_robot_mm in transformed
        ]
        expected_points = [
            relative_slots[slot_id] for slot_id in sorted(relative_slots)
        ]
        self.observations_pub.publish(
            self._pose_array(now, self.robot_frame, observation_points)
        )
        self.expected_slots_pub.publish(
            self._pose_array(now, self.robot_frame, expected_points)
        )

        observation_data = [float(len(transformed))]
        for camera_id, observation, point_robot_mm in transformed:
            observation_data.extend(
                [
                    float(camera_id),
                    float(observation.point_camera_mm[0]),
                    float(observation.point_camera_mm[1]),
                    float(observation.point_camera_mm[2]),
                    float(point_robot_mm[0]),
                    float(point_robot_mm[1]),
                    float(point_robot_mm[2]),
                ]
            )
        self.observations_mm_pub.publish(
            self._array_message(observation_data, "camera_observations_mm")
        )

        expected_data = [
            float(self.formation_id),
            float(self.self_slot_id),
            float(len(relative_slots)),
        ]
        for slot_id in sorted(relative_slots):
            expected_data.extend(
                [
                    float(slot_id),
                    float(relative_slots[slot_id][0]),
                    float(relative_slots[slot_id][1]),
                    float(relative_slots[slot_id][2]),
                ]
            )
        self.expected_slots_mm_pub.publish(
            self._array_message(expected_data, "expected_slots_mm")
        )
        self.state_pub.publish(
            self._array_message(
                [float(self.formation_id), float(self.self_slot_id)],
                "formation_state",
            )
        )
        self.marker_pub.publish(
            self._build_markers(now, transformed, relative_slots)
        )

    def _publish_static_camera_tf(self) -> None:
        if self.tf_broadcaster is None:
            return
        stamp = self.get_clock().now().to_msg()
        transforms = []
        for camera_id in sorted(self.extrinsics.cameras):
            camera = self.extrinsics.get(camera_id)
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = self.robot_frame
            transform.child_frame_id = f"cap_camera_{camera_id}"
            transform.transform.translation.x = (
                float(camera.position_mm[0]) / 1000.0
            )
            transform.transform.translation.y = (
                float(camera.position_mm[1]) / 1000.0
            )
            transform.transform.translation.z = (
                float(camera.position_mm[2]) / 1000.0
            )
            qx, qy, qz, qw = quaternion_from_rotation_matrix(
                camera.rotation_camera_to_robot
            )
            transform.transform.rotation.x = qx
            transform.transform.rotation.y = qy
            transform.transform.rotation.z = qz
            transform.transform.rotation.w = qw
            transforms.append(transform)
        self.tf_broadcaster.sendTransform(transforms)

    @staticmethod
    def _set_color(marker: Marker, rgb: tuple[float, float, float], alpha=1.0):
        marker.color.r = float(rgb[0])
        marker.color.g = float(rgb[1])
        marker.color.b = float(rgb[2])
        marker.color.a = float(alpha)

    def _make_arrow(
        self,
        stamp,
        namespace: str,
        marker_id: int,
        start_mm: Iterable[float],
        end_mm: Iterable[float],
        rgb: tuple[float, float, float],
        width_m: float,
    ) -> Marker:
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.robot_frame
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        start = np.asarray(start_mm, dtype=float) / 1000.0
        end = np.asarray(end_mm, dtype=float) / 1000.0
        marker.points = [Point(), Point()]
        marker.points[0].x = float(start[0])
        marker.points[0].y = float(start[1])
        marker.points[0].z = float(start[2])
        marker.points[1].x = float(end[0])
        marker.points[1].y = float(end[1])
        marker.points[1].z = float(end[2])
        marker.scale.x = width_m
        marker.scale.y = width_m * 2.0
        marker.scale.z = width_m * 3.0
        self._set_color(marker, rgb)
        return marker

    def _make_sphere(
        self,
        stamp,
        namespace: str,
        marker_id: int,
        point_mm: Iterable[float],
        rgb: tuple[float, float, float],
        diameter_m: float,
    ) -> Marker:
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.robot_frame
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        point = np.asarray(point_mm, dtype=float) / 1000.0
        marker.pose.position.x = float(point[0])
        marker.pose.position.y = float(point[1])
        marker.pose.position.z = float(point[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = diameter_m
        marker.scale.y = diameter_m
        marker.scale.z = diameter_m
        self._set_color(marker, rgb)
        return marker

    def _make_text(
        self,
        stamp,
        namespace: str,
        marker_id: int,
        point_mm: Iterable[float],
        text: str,
        rgb: tuple[float, float, float],
    ) -> Marker:
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.robot_frame
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        point = np.asarray(point_mm, dtype=float) / 1000.0
        marker.pose.position.x = float(point[0])
        marker.pose.position.y = float(point[1])
        marker.pose.position.z = float(point[2])
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.06
        marker.text = text
        self._set_color(marker, rgb)
        return marker

    def _make_delete(
        self,
        stamp,
        namespace: str,
        marker_id: int,
    ) -> Marker:
        marker = Marker()
        marker.header.stamp = stamp
        marker.header.frame_id = self.robot_frame
        marker.ns = namespace
        marker.id = marker_id
        marker.action = Marker.DELETE
        marker.pose.orientation.w = 1.0
        return marker

    def _delete_stale_observation_markers(
        self,
        stamp,
        active_count: int,
    ) -> list[Marker]:
        markers = []
        for marker_id in range(active_count, 3):
            for namespace in (
                "observations",
                "observation_labels",
                "observation_rays",
            ):
                markers.append(self._make_delete(stamp, namespace, marker_id))
        return markers

    def _build_markers(self, stamp, transformed, relative_slots) -> MarkerArray:
        marker_array = MarkerArray()
        marker_array.markers.extend(
            self._delete_stale_observation_markers(stamp, len(transformed))
        )

        origin = np.zeros(3, dtype=float)
        axis_mm = self.axis_length_mm
        marker_array.markers.extend(
            [
                self._make_arrow(
                    stamp,
                    "robot_axes",
                    0,
                    origin,
                    [axis_mm, 0.0, 0.0],
                    (1.0, 0.0, 0.0),
                    0.012,
                ),
                self._make_arrow(
                    stamp,
                    "robot_axes",
                    1,
                    origin,
                    [0.0, axis_mm, 0.0],
                    (0.0, 1.0, 0.0),
                    0.012,
                ),
                self._make_arrow(
                    stamp,
                    "robot_axes",
                    2,
                    origin,
                    [0.0, 0.0, axis_mm],
                    (0.0, 0.0, 1.0),
                    0.012,
                ),
            ]
        )
        marker_array.markers.append(
            self._make_sphere(
                stamp,
                "robot",
                0,
                origin,
                (0.7, 0.7, 0.7),
                0.10,
            )
        )
        marker_array.markers.append(
            self._make_text(
                stamp,
                "robot",
                1,
                [0.0, 0.0, 80.0],
                self.robot_frame,
                (1.0, 1.0, 1.0),
            )
        )

        for camera_id in sorted(self.extrinsics.cameras):
            camera = self.extrinsics.get(camera_id)
            camera_pos = camera.position_mm
            marker_array.markers.append(
                self._make_sphere(
                    stamp,
                    "cameras",
                    camera_id,
                    camera_pos,
                    (1.0, 0.55, 0.0),
                    0.08,
                )
            )
            marker_array.markers.append(
                self._make_text(
                    stamp,
                    "camera_labels",
                    camera_id,
                    camera_pos + np.array([0.0, 0.0, 100.0]),
                    f"cam{camera_id} {camera.name}",
                    (1.0, 0.7, 0.1),
                )
            )
            for axis_id, color in enumerate(
                (
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                    (0.0, 0.0, 1.0),
                )
            ):
                endpoint = camera_pos + (
                    camera.rotation_camera_to_robot[:, axis_id]
                    * self.camera_axis_length_mm
                )
                marker_array.markers.append(
                    self._make_arrow(
                        stamp,
                        f"camera_{camera_id}_axes",
                        axis_id,
                        camera_pos,
                        endpoint,
                        color,
                        0.008,
                    )
                )

        slot_colors = {
            0: (1.0, 0.55, 0.0),
            1: (1.0, 0.0, 1.0),
            2: (0.0, 0.9, 1.0),
        }
        expected_positions = []
        for slot_id in sorted(relative_slots):
            position = relative_slots[slot_id]
            expected_positions.append(position)
            color = slot_colors.get(slot_id, (1.0, 1.0, 0.0))
            if slot_id == self.self_slot_id:
                color = (0.8, 0.8, 0.8)
            marker_array.markers.append(
                self._make_sphere(
                    stamp,
                    "expected_slots",
                    slot_id,
                    position,
                    color,
                    0.12 if slot_id == self.self_slot_id else 0.09,
                )
            )
            marker_array.markers.append(
                self._make_text(
                    stamp,
                    "expected_slot_labels",
                    slot_id,
                    position + np.array([0.0, 0.0, 90.0]),
                    f"slot{slot_id}",
                    color,
                )
            )

        if len(expected_positions) >= 2:
            edges = Marker()
            edges.header.stamp = stamp
            edges.header.frame_id = self.robot_frame
            edges.ns = "formation_edges"
            edges.id = 0
            edges.type = Marker.LINE_LIST
            edges.action = Marker.ADD
            edges.scale.x = 0.008
            self._set_color(edges, (1.0, 1.0, 0.0), 0.75)
            slot_ids = sorted(relative_slots)
            for index, slot_id in enumerate(slot_ids):
                next_slot = slot_ids[(index + 1) % len(slot_ids)]
                start = relative_slots[slot_id] / 1000.0
                end = relative_slots[next_slot] / 1000.0
                start_point = Point()
                start_point.x = float(start[0])
                start_point.y = float(start[1])
                start_point.z = float(start[2])
                end_point = Point()
                end_point.x = float(end[0])
                end_point.y = float(end[1])
                end_point.z = float(end[2])
                edges.points.extend([start_point, end_point])
            marker_array.markers.append(edges)

        for index, (camera_id, observation, point_robot_mm) in enumerate(
            transformed
        ):
            color = slot_colors.get(camera_id, (0.2, 1.0, 0.2))
            marker_array.markers.append(
                self._make_sphere(
                    stamp,
                    "observations",
                    index,
                    point_robot_mm,
                    color,
                    0.10,
                )
            )
            marker_array.markers.append(
                self._make_text(
                    stamp,
                    "observation_labels",
                    index,
                    point_robot_mm + np.array([0.0, 0.0, 100.0]),
                    f"cam{camera_id} obs",
                    color,
                )
            )

            line = Marker()
            line.header.stamp = stamp
            line.header.frame_id = self.robot_frame
            line.ns = "observation_rays"
            line.id = index
            line.type = Marker.LINE_LIST
            line.action = Marker.ADD
            line.scale.x = 0.005
            self._set_color(line, color, 0.65)
            camera_pos = self.extrinsics.get(camera_id).position_mm / 1000.0
            target_pos = point_robot_mm / 1000.0
            start_point = Point()
            start_point.x = float(camera_pos[0])
            start_point.y = float(camera_pos[1])
            start_point.z = float(camera_pos[2])
            end_point = Point()
            end_point.x = float(target_pos[0])
            end_point.y = float(target_pos[1])
            end_point.z = float(target_pos[2])
            line.points = [start_point, end_point]
            marker_array.markers.append(line)

        marker_array.markers.append(
            self._make_text(
                stamp,
                "formation_status",
                0,
                [0.0, 0.0, axis_mm + 120.0],
                f"formation{self.formation_id} self{self.self_slot_id}",
                (1.0, 1.0, 1.0),
            )
        )
        return marker_array


def main(args=None):
    rclpy.init(args=args)
    node = FormationObservationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
