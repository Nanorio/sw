"""Display node: annotated image, disparity image, raw stereo and pose overlay."""

from __future__ import annotations

import cv2
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float64MultiArray

from .common import decode_compressed


class DisplayNode(Node):
    def __init__(self):
        super().__init__("display_node")

        self.declare_parameter("use_window", True)
        self.use_window = bool(self.get_parameter("use_window").value)
        self._annotated = None
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
        self.annotated_sub = self.create_subscription(
            CompressedImage, "/cap/yolo/annotated_image", self._on_annotated, sensor_qos
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

    def _on_annotated(self, msg):
        try:
            self._annotated = decode_compressed(msg)
        except Exception as exc:
            self.get_logger().error(f"Annotated decode failed: {exc}")

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
            if self._annotated is not None:
                frame = self._annotated.copy()
                if self._pose_details is not None and self._is_detected(
                    self._pose_details
                ):
                    self._overlay_pose(frame, self._pose_details)
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
            f"XYZ: {data[0]:.0f} {data[1]:.0f} {data[2]:.0f} mm",
            f"Yaw: {data[6]:+.1f}  Pitch: {data[7]:+.1f}",
        ]
        if data[9] > 0.0:
            lines.append(f"Roll: {data[8]:+.1f}")
        for idx, line in enumerate(lines):
            y = 40 + idx * 30
            cv2.putText(
                frame,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

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
