"""Pose solver node: YOLO detections + disparity -> 3D target pose."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray, Int32MultiArray, MultiArrayDimension
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from vision_msgs.msg import Detection2DArray

from .common import (
    build_rectify_maps,
    decode_float32_image,
    get_config_dir,
    load_yaml,
    resolve_project_root,
    robust_depth_from_xyz,
)
from .robot_coord import RobotCoord


@dataclass
class _TargetPoint:
    cx: float
    cy: float
    class_id: int
    pt_3d: np.ndarray


class PoseNode(Node):
    def __init__(self):
        super().__init__("pose_node")

        self.declare_parameter("config_dir", get_config_dir())
        self.declare_parameter("project_root", str(Path.cwd()))
        self.declare_parameter("enabled", True)

        config_dir = Path(self.get_parameter("config_dir").value)
        self.project_root = resolve_project_root(
            self.get_parameter("project_root").value
        )
        self.enabled = bool(self.get_parameter("enabled").value)
        self.declare_parameter("filter_alpha", 0.35)
        self.filter_alpha = float(self.get_parameter("filter_alpha").value)
        self._filtered = None
        self._filtered_timestamp = None

        self.calib = load_yaml(config_dir / "camera_params.yaml")
        self.rectify = build_rectify_maps(self.calib)
        self.Q = self.rectify["Q"]
        self.depth_cor_factor = float(
            load_yaml(config_dir / "SGBM_params.yaml").get("depthCorFactor", 1.0)
        )
        class_map = load_yaml(config_dir / "class_map.yaml")
        self.blue_class_id = int(class_map.get("blue_light", 0))
        self.green_class_id = int(class_map.get("green_light", 1))
        self.robot = RobotCoord(config_dir / "camera_setup.yaml")
        self.active_cam_id = self.robot.active_cam_id

        self._detections = {}
        self._disparities = {}
        self._rois = {}
        self._xyz = None
        self._last_detected = False
        self._last_details = None
        self._pose_counter = 0
        self._trajectory = []
        self._max_trajectory_pts = int(self.declare_parameter("max_trajectory_pts", 200).value)
        self._pair_timeout_ns = int(0.6 * 1e9)

        sensor_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.det_sub = self.create_subscription(
            Detection2DArray, "/cap/yolo/detections", self._on_detections, 10
        )
        self.disp_sub = self.create_subscription(
            Image, "/cap/sgbm/disparity", self._on_disparity, sensor_qos
        )
        self.roi_sub = self.create_subscription(
            Int32MultiArray, "/cap/sgbm/disparity_roi", self._on_roi, 10
        )
        self.pose_pub = self.create_publisher(
            PoseStamped, "/cap/pose/target_pose", 10
        )
        self.details_pub = self.create_publisher(
            Float64MultiArray, "/cap/pose/target_pose_details", 10
        )
        self.pose_srv = self.create_service(
            Trigger, "/cap/pose/get_target_pose", self._on_get_pose
        )
        self.marker_pub = self.create_publisher(
            MarkerArray, "/cap/pose/markers", 10
        )
        self.get_logger().info(
            f"pose_node ready, active camera {self.active_cam_id}"
        )

    @staticmethod
    def _stamp_key(stamp):
        return (stamp.sec, stamp.nanosec)

    def _prune(self):
        now_ns = self.get_clock().now().nanoseconds
        for store in (self._detections, self._disparities, self._rois):
            for key in list(store):
                stamp_ns = key[0] * 1_000_000_000 + key[1]
                if now_ns - stamp_ns > self._pair_timeout_ns:
                    del store[key]

    def _on_detections(self, msg):
        if not self.enabled:
            return
        key = self._stamp_key(msg.header.stamp)
        self._detections[key] = msg
        self._prune()
        disparity = self._disparities.pop(key, None)
        if disparity is not None:
            self._detections.pop(key, None)
            self._solve(msg, disparity, self._rois.pop(key, None))

    def _on_roi(self, msg):
        key = self._stamp_key(msg.header.stamp) if hasattr(msg, "header") else None
        if key is None:
            return
        if len(msg.data) < 6:
            return
        self._rois[key] = list(msg.data)
        self._prune()
        disparity = self._disparities.pop(key, None)
        detections = self._detections.pop(key, None)
        roi = self._rois.pop(key, None)
        if disparity is not None and detections is not None:
            self._solve(detections, disparity, roi)

    def _on_disparity(self, msg):
        if not self.enabled:
            return
        key = self._stamp_key(msg.header.stamp)
        try:
            disparity = decode_float32_image(msg)
        except Exception as exc:
            self.get_logger().error(f"Disparity decode failed: {exc}")
            return
        self._disparities[key] = disparity
        self._prune()
        detections = self._detections.pop(key, None)
        roi = self._rois.pop(key, None)
        if detections is not None:
            self._disparities.pop(key, None)
            self._solve(detections, disparity, roi)

    def _solve(self, det_msg, disparity, roi=None):
        try:
            self._xyz = cv2.reprojectImageTo3D(
                disparity, self.Q, handleMissingValues=True
            )
        except Exception as exc:
            self.get_logger().error(f"reprojectImageTo3D failed: {exc}")
            return

        blue_3d = []
        green_3d = []
        other_3d = []
        roi_x, roi_y = 0, 0
        if roi and len(roi) >= 4:
            roi_x = int(roi[0])
            roi_y = int(roi[1])
        height, width = disparity.shape[:2]
        for det in det_msg.detections:
            if not det.results:
                continue
            try:
                class_id = int(det.results[0].hypothesis.class_id)
            except (TypeError, ValueError):
                class_id = -1

            center_x = float(det.bbox.center.position.x) - roi_x
            center_y = float(det.bbox.center.position.y) - roi_y
            box_w = float(det.bbox.size_x)
            box_h = float(det.bbox.size_y)
            x1 = int(max(0, min(width - 1, center_x - box_w / 2.0)))
            y1 = int(max(0, min(height - 1, center_y - box_h / 2.0)))
            x2 = int(max(x1 + 1, min(width, center_x + box_w / 2.0)))
            y2 = int(max(y1 + 1, min(height, center_y + box_h / 2.0)))

            pt_3d = robust_depth_from_xyz(
                self._xyz,
                x1,
                y1,
                x2,
                y2,
                self.depth_cor_factor,
            )
            if pt_3d is None:
                continue
            point = _TargetPoint(
                cx=center_x,
                cy=center_y,
                class_id=class_id,
                pt_3d=pt_3d,
            )
            if class_id == self.blue_class_id:
                blue_3d.append(point)
            elif class_id == self.green_class_id:
                green_3d.append(point)
            else:
                other_3d.append(point)

        all_points = blue_3d + green_3d + other_3d
        if not all_points:
            self._publish_empty(det_msg.header.stamp)
            return

        center = np.mean([p.pt_3d for p in all_points], axis=0)
        target_yaw = math.degrees(math.atan2(center[0], center[2]))
        target_pitch = math.degrees(math.atan2(-center[1], center[2]))
        target_roll = self._solve_roll(blue_3d, green_3d)
        abs_pt = self.robot.camera_to_abs(self.active_cam_id, center)

        self._publish_pose(
            det_msg.header,
            abs_pt,
            center,
            target_yaw,
            target_pitch,
            target_roll,
            len(all_points),
        )

        self._pose_counter += 1
        if self._pose_counter % 30 == 0:
            self.get_logger().info(
                f"target xyz=({abs_pt[0]:.0f},{abs_pt[1]:.0f},{abs_pt[2]:.0f}) "
                f"yaw={target_yaw:.1f} pitch={target_pitch:.1f} "
                f"roll={target_roll if target_roll is not None else 0.0:.1f}"
            )

    def _publish_pose(
        self,
        header,
        abs_pt,
        center,
        yaw,
        pitch,
        roll,
        detection_count,
    ):
        roll_value = roll if roll is not None else 0.0
        raw = [
            float(abs_pt[0]),
            float(abs_pt[1]),
            float(abs_pt[2]),
            float(center[0]),
            float(center[1]),
            float(center[2]),
            float(yaw),
            float(pitch),
            float(roll_value),
        ]
        if self._filtered is None:
            self._filtered = list(raw)
        else:
            alpha = self.filter_alpha
            self._filtered = [
                alpha * new + (1.0 - alpha) * old
                for new, old in zip(raw, self._filtered)
            ]
        abs_pt_f = np.array(self._filtered[:3])
        center_f = np.array(self._filtered[3:6])
        yaw_f = self._filtered[6]
        pitch_f = self._filtered[7]
        roll_f = self._filtered[8]
        self._last_detected = True

        pose = PoseStamped()
        pose.header = header
        pose.header.frame_id = f"cap_camera_{self.active_cam_id}"
        pose.pose.position.x = float(abs_pt_f[0]) / 1000.0
        pose.pose.position.y = float(abs_pt_f[1]) / 1000.0
        pose.pose.position.z = float(abs_pt_f[2]) / 1000.0
        qx, qy, qz, qw = self._euler_to_quaternion(yaw_f, pitch_f, roll_f)
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        details = Float64MultiArray()
        details.layout.dim.append(MultiArrayDimension())
        details.layout.dim[0].label = "pose"
        details.layout.dim[0].size = 11
        details.layout.dim[0].stride = 11
        details.data = [
            float(abs_pt_f[0]),
            float(abs_pt_f[1]),
            float(abs_pt_f[2]),
            float(center_f[0]),
            float(center_f[1]),
            float(center_f[2]),
            float(yaw_f),
            float(pitch_f),
            float(roll_f),
            float(detection_count),
            1.0,
        ]
        self._last_details = details
        self.pose_pub.publish(pose)
        self.details_pub.publish(details)
        self._trajectory.append([
            float(abs_pt_f[0]) / 1000.0,
            float(abs_pt_f[1]) / 1000.0,
            float(abs_pt_f[2]) / 1000.0,
        ])
        if len(self._trajectory) > self._max_trajectory_pts:
            self._trajectory = self._trajectory[-self._max_trajectory_pts:]
        self.marker_pub.publish(self._build_markers(header.stamp))

    @staticmethod
    def _euler_to_quaternion(yaw, pitch, roll):
        yaw_r = math.radians(yaw)
        pitch_r = math.radians(pitch)
        roll_r = math.radians(roll)
        cy = math.cos(yaw_r * 0.5)
        sy = math.sin(yaw_r * 0.5)
        cp = math.cos(pitch_r * 0.5)
        sp = math.sin(pitch_r * 0.5)
        cr = math.cos(roll_r * 0.5)
        sr = math.sin(roll_r * 0.5)
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        qw = cr * cp * cy + sr * sp * sy
        return qx, qy, qz, qw

    @staticmethod
    def _solve_roll(blue_3d, green_3d):
        if len(blue_3d) < 2 or not green_3d:
            return None
        blue_sorted = sorted(blue_3d, key=lambda p: p.cx)
        left, right = blue_sorted[0], blue_sorted[-1]
        dx = right.cx - left.cx
        dy = right.cy - left.cy
        if abs(dx) <= 1 and abs(dy) <= 1:
            return None
        p1 = (dy, -dx)
        p2 = (-dy, dx)
        mid_x = (left.cx + right.cx) / 2.0
        mid_y = (left.cy + right.cy) / 2.0
        green = green_3d[0]
        to_green = (green.cx - mid_x, green.cy - mid_y)
        aux = p1 if (p1[0] * to_green[0] + p1[1] * to_green[1]) < 0 else p2
        return -math.degrees(math.atan2(aux[0], -aux[1]))

    def _publish_empty(self, stamp):
        self._last_detected = False
        self._filtered = None
        self._filtered_timestamp = None

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = f"cap_camera_{self.active_cam_id}"
        pose.pose.orientation.w = 1.0

        details = Float64MultiArray()
        details.layout.dim.append(MultiArrayDimension())
        details.layout.dim[0].label = "pose"
        details.layout.dim[0].size = 11
        details.layout.dim[0].stride = 11
        details.data = [0.0] * 11
        self._last_details = details
        self.pose_pub.publish(pose)
        self.details_pub.publish(details)

    def _build_markers(self, stamp):
        markers = MarkerArray()
        if self._trajectory:
            line = Marker()
            line.header.stamp = stamp
            line.header.frame_id = "map"
            line.ns = "target_path"
            line.id = 0
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.pose.orientation.w = 1.0
            line.scale.x = 0.02
            line.color.a = 1.0
            line.color.r = 1.0
            line.color.g = 0.7
            line.color.b = 0.0
            for pt in self._trajectory:
                from geometry_msgs.msg import Point
                p = Point()
                p.x = float(pt[0])
                p.y = float(pt[1])
                p.z = float(pt[2])
                line.points.append(p)
            markers.markers.append(line)

            sphere = Marker()
            sphere.header.stamp = stamp
            sphere.header.frame_id = "map"
            sphere.ns = "target"
            sphere.id = 1
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            last = self._trajectory[-1]
            sphere.pose.position.x = float(last[0])
            sphere.pose.position.y = float(last[1])
            sphere.pose.position.z = float(last[2])
            sphere.pose.orientation.w = 1.0
            sphere.scale.x = 0.08
            sphere.scale.y = 0.08
            sphere.scale.z = 0.08
            sphere.color.a = 1.0
            sphere.color.r = 1.0
            sphere.color.g = 0.0
            sphere.color.b = 0.0
            markers.markers.append(sphere)
        return markers

    def _on_get_pose(self, request, response):
        if not self._last_detected or self._last_details is None:
            response.success = False
            response.message = json.dumps(
                {"success": False, "message": "no pose available yet"}
            )
            return response
        data = self._last_details.data
        response.success = True
        response.message = json.dumps(
            {
                "success": True,
                "x_mm": data[0],
                "y_mm": data[1],
                "z_mm": data[2],
                "cam_x_mm": data[3],
                "cam_y_mm": data[4],
                "cam_z_mm": data[5],
                "yaw_deg": data[6],
                "pitch_deg": data[7],
                "roll_deg": data[8],
                "detection_count": int(data[9]),
            }
        )
        return response


def main(args=None):
    rclpy.init(args=args)
    node = PoseNode()
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
