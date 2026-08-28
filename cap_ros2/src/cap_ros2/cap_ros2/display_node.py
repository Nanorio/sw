"""Display node: annotated image, disparity image, raw stereo and pose overlay."""

from __future__ import annotations

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64MultiArray
from vision_msgs.msg import Detection2DArray

from .common import decode_compressed


class DisplayNode(Node):
    def __init__(self):
        super().__init__("display_node")

        self.declare_parameter("use_window", True)
        self.use_window = bool(self.get_parameter("use_window").value)
        self._annotated = None
        self._left = None
        self._detections = None
        self._depth = None
        self._stereo = None
        self._pose_stamped = None
        self._pose_details = None
        self._log_counter = 0

        sensor_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.left_sub = self.create_subscription(
            CompressedImage, "/cap/camera/left", self._on_left, sensor_qos
        )
        self.det_sub = self.create_subscription(
            Detection2DArray, "/cap/yolo/detections", self._on_detections, 10
        )
        self.depth_sub = self.create_subscription(
            CompressedImage,
            "/cap/sgbm/disparity_visual",
            self._on_depth,
            sensor_qos,
        )
        self.stereo_sub = self.create_subscription(
            CompressedImage,
            "/cap/camera/stereo_raw",
            self._on_stereo,
            sensor_qos,
        )
        self.pose_sub = self.create_subscription(
            PoseStamped, "/cap/pose/target_pose", self._on_pose, 10
        )
        self.details_sub = self.create_subscription(
            Float64MultiArray,
            "/cap/pose/target_pose_details",
            self._on_pose_details,
            10,
        )
        self.timer = self.create_timer(1.0 / 30.0, self.render)
        self.get_logger().info(f"display_node ready, use_window={self.use_window}")

    def _on_left(self, msg):
        try:
            self._left = decode_compressed(msg)
        except Exception as exc:
            self.get_logger().error(f"Left image decode failed: {exc}")

    def _on_detections(self, msg):
        self._detections = msg

    def _on_depth(self, msg):
        try:
            self._depth = decode_compressed(msg)
        except Exception as exc:
            self.get_logger().error(f"Depth decode failed: {exc}")

    def _on_stereo(self, msg):
        try:
            self._stereo = decode_compressed(msg)
        except Exception as exc:
            self.get_logger().error(f"Stereo decode failed: {exc}")

    def _on_pose(self, msg):
        self._pose_stamped = msg

    def _on_pose_details(self, msg):
        self._pose_details = msg

    def render(self):
        if not self.use_window:
            self._log_pose_headless()
            return
        try:
            frame = self._build_main_frame()
            if frame is not None:
                cv2.imshow("CAP Detection", frame)
            if self._depth is not None:
                cv2.imshow("CAP SGBM Depth", self._depth)
            if self._stereo is not None:
                cv2.imshow("CAP Stereo Raw", self._stereo)
            cv2.waitKey(1)
        except cv2.error as exc:
            self.use_window = False
            self.get_logger().warning(
                f"GUI unavailable, disabling windows: {exc}"
            )

    @staticmethod
    def _overlay_pose(frame, pose):
        data = pose.data
        if len(data) < 11:
            return
        lines = [
            f"Yaw:{data[6]:+.1f}  Pitch:{data[7]:+.1f}",
            f"Z:{data[5]:.0f}mm  X:{data[3]:.0f}  Y:{data[4]:.0f}",
        ]
        if data[9] > 0.0:
            lines.append(f"Roll:{data[8]:+.1f}")
        lines.append(f"ROBOT XYZ:{data[0]:.0f} {data[1]:.0f} {data[2]:.0f} mm")
        for idx, line in enumerate(lines):
            y = 70 + idx * 26
            cv2.putText(
                frame,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

    def _build_main_frame(self):
        if self._left is None and self._annotated is not None:
            frame = self._annotated.copy()
        elif self._left is not None:
            frame = self._left.copy()
        else:
            frame = None
        if frame is not None and self._detections is not None:
            details_for_draw = self._pose_details if self._pose_details is not None else None
            self._draw_detections(frame, self._detections, details_for_draw)
        if frame is not None and self._pose_details is not None and self._is_detected(
            self._pose_details
        ):
            self._overlay_pose(frame, self._pose_details)
        return frame

    def _draw_detections(self, frame, det_msg, details=None):
        h_i, w_i = frame.shape[:2]
        cx_i, cy_i = 331, 234
        cv2.line(frame, (cx_i, 0), (cx_i, h_i - 1), (0, 255, 0), 1)
        cv2.line(frame, (0, cy_i), (w_i - 1, cy_i), (0, 255, 0), 1)
        cv2.circle(frame, (cx_i, cy_i), 5, (0, 255, 0), 2)

        blue_pts = []
        green_pts = []
        for det in det_msg.detections:
            cx = float(det.bbox.center.position.x)
            cy = float(det.bbox.center.position.y)
            center = (int(cx), int(cy))
            class_id = -1
            if det.results:
                try:
                    class_id = int(det.results[0].hypothesis.class_id)
                except (TypeError, ValueError):
                    pass
            if class_id == 0:
                blue_pts.append(center)
            else:
                green_pts.append(center)
            x1 = int(det.bbox.center.position.x - det.bbox.size_x / 2.0)
            y1 = int(det.bbox.center.position.y - det.bbox.size_y / 2.0)
            x2 = int(det.bbox.center.position.x + det.bbox.size_x / 2.0)
            y2 = int(det.bbox.center.position.y + det.bbox.size_y / 2.0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

        for pt in blue_pts:
            cv2.circle(frame, pt, 6, (0, 0, 255), -1)
            cv2.putText(frame, "BlueLight", (pt[0] + 10, pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        for pt in green_pts:
            cv2.circle(frame, pt, 6, (255, 0, 255), -1)
            cv2.putText(frame, "GreenLight", (pt[0] + 10, pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        self._draw_per_box_labels(frame, details)

        for g_pt in green_pts:
            for b_pt in blue_pts:
                cv2.line(frame, g_pt, b_pt, (255, 255, 255), 2)

        for i in range(len(blue_pts)):
            for j in range(i + 1, len(blue_pts)):
                cv2.line(frame, blue_pts[i], blue_pts[j], (0, 0, 0), 2)

        all_pts = blue_pts + green_pts
        if len(all_pts) >= 3 and len(blue_pts) >= 2 and len(green_pts) >= 1:
            sy = sorted(all_pts, key=lambda p: p[1])
            bot = sy[-1]
            top2 = sorted(sy[:2], key=lambda p: p[0])
            lpt, rpt = top2[0], top2[1]
            cv2.line(frame, lpt, rpt, (255, 255, 0), 2)
            cv2.line(frame, lpt, bot, (255, 255, 0), 2)
            cv2.line(frame, rpt, bot, (255, 255, 0), 2)
            cv2.circle(frame, lpt, 4, (0, 255, 255), -1)
            cv2.circle(frame, rpt, 4, (0, 255, 255), -1)
            cv2.circle(frame, bot, 4, (0, 255, 255), -1)

    def _draw_per_box_labels(self, frame, details):
        if details is None or len(details.data) < 13:
            return
        data = details.data
        count = int(data[11]) if len(data) > 11 else 0
        if count <= 0:
            return
        idx = 12
        for _ in range(count):
            if idx + 4 >= len(data):
                break
            cx = data[idx]
            cy = data[idx + 1]
            yaw = data[idx + 2]
            pitch = data[idx + 3]
            z_mm = data[idx + 4]
            idx += 5
            cv2.putText(frame, f"Yaw:{yaw:+.1f} Pitch:{pitch:+.1f}",
                        (int(cx) + 10, int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.putText(frame, f"Z:{z_mm:.0f}mm",
                        (int(cx) + 10, int(cy) + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    def _log_pose_headless(self):
        self._log_counter += 1
        if self._pose_details is None or not self._is_detected(
            self._pose_details
        ):
            if self._log_counter % 60 == 0:
                self.get_logger().info("no target")
            return
        if self._log_counter % 30 == 0:
            data = self._pose_details.data
            self.get_logger().info(
                f"target xyz=({data[0]:.0f},{data[1]:.0f},"
                f"{data[2]:.0f}) yaw={data[6]:.1f}"
            )

    @staticmethod
    def _is_detected(msg):
        return len(msg.data) >= 11 and msg.data[10] > 0.5

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DisplayNode()
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
