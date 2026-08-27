"""Camera distribution node: stereo camera -> compressed ROS2 image topics."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import SetBool, Trigger

from .common import (
    build_rectify_maps,
    compressed_image_msg,
    get_config_dir,
    load_yaml,
    resolve_project_root,
    resolve_path,
)


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")

        self.declare_parameter("config_dir", get_config_dir())
        self.declare_parameter("project_root", str(Path.cwd()))
        self.declare_parameter("camera_id", -1)
        self.declare_parameter("jpeg_quality", 88)
        self.declare_parameter("enabled", True)

        config_dir = Path(self.get_parameter("config_dir").value)
        self.project_root = resolve_project_root(
            self.get_parameter("project_root").value
        )
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.enabled = bool(self.get_parameter("enabled").value)

        self.capture_cfg = load_yaml(config_dir / "capture.yaml")
        self.calib = load_yaml(config_dir / "camera_params.yaml")
        self.rectify = build_rectify_maps(self.calib)
        self.image_size = self.rectify["image_size"]
        self._last_raw = None
        self._last_rect_left = None
        self._last_rect_right = None

        self._open_capture()
        self._create_publishers()
        self._create_services()

        fps = float(self.capture_cfg.get("fps", 30.0))
        self.timer = self.create_timer(1.0 / max(1.0, fps), self.tick)
        self.get_logger().info(
            "camera_node ready, publishing to /cap/camera/{left,right,stereo_raw}"
        )

    def _open_capture(self):
        mode = self.capture_cfg.get("captureMode", "camera")
        if mode == "one image":
            image_path = self.capture_cfg.get(
                "imagePath", "./staticSource/image/test.jpg"
            )
            path = resolve_path(image_path, self.project_root)
            frame = cv2.imread(str(path))
            if frame is None:
                raise FileNotFoundError(f"Static image not readable: {path}")
            self._static_frame = frame
            self.cap = None
            self.get_logger().info(f"Static image mode: {path}")
            return

        if mode == "one video":
            video_path = self.capture_cfg.get(
                "videoPath", "./staticSource/video/test.mp4"
            )
            path = resolve_path(video_path, self.project_root)
            self._static_frame = None
            self.cap = cv2.VideoCapture(str(path))
            self.get_logger().info(f"Video file mode: {path}")
            return

        override_id = int(self.get_parameter("camera_id").value)
        if override_id >= 0:
            camera_id = override_id
        else:
            camera_id = int(
                self.capture_cfg.get("v4l2", {}).get(
                    "realDeviceId",
                    self.capture_cfg.get("cameraId", 0),
                )
            )

        flags = cv2.CAP_V4L2 if sys.platform.startswith("linux") else 0
        self.cap = cv2.VideoCapture(camera_id, flags)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera /dev/video{camera_id}; "
                "start camera_server or set camera_id:=0"
            )
        width, height = self.image_size
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width * 2)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._static_frame = None
        self.get_logger().info(f"Camera opened: /dev/video{camera_id}")

    def _create_publishers(self):
        sensor_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.left_pub = self.create_publisher(
            CompressedImage, "/cap/camera/left", sensor_qos
        )
        self.right_pub = self.create_publisher(
            CompressedImage, "/cap/camera/right", sensor_qos
        )
        self.stereo_pub = self.create_publisher(
            CompressedImage, "/cap/camera/stereo_raw", sensor_qos
        )

    def _create_services(self):
        self.enable_srv = self.create_service(
            SetBool, "/cap/camera/enable", self._on_enable
        )
        self.capture_srv = self.create_service(
            Trigger, "/cap/camera/capture", self._on_capture
        )

    def tick(self):
        if not self.enabled:
            return
        now = self.get_clock().now().to_msg()

        if self._static_frame is not None:
            frame = self._static_frame.copy()
        elif self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                self.get_logger().warning("Camera read failed, skipping frame")
                return
        else:
            return

        width, height = self.image_size
        if frame.shape[1] != width * 2 or frame.shape[0] != height:
            frame = cv2.resize(frame, (width * 2, height))

        left_raw = frame[:, :width].copy()
        right_raw = frame[:, width:].copy()
        rect_left = cv2.remap(
            left_raw,
            self.rectify["left_map1"],
            self.rectify["left_map2"],
            cv2.INTER_LINEAR,
        )
        rect_right = cv2.remap(
            right_raw,
            self.rectify["right_map1"],
            self.rectify["right_map2"],
            cv2.INTER_LINEAR,
        )

        self._last_raw = frame.copy()
        self._last_rect_left = rect_left
        self._last_rect_right = rect_right

        self.left_pub.publish(
            compressed_image_msg(
                rect_left, now, "cap_left", self.jpeg_quality
            )
        )
        self.right_pub.publish(
            compressed_image_msg(
                rect_right, now, "cap_right", self.jpeg_quality
            )
        )
        self.stereo_pub.publish(
            compressed_image_msg(frame, now, "cap_stereo", self.jpeg_quality)
        )

    def _on_enable(self, request, response):
        self.enabled = bool(request.data)
        response.success = True
        response.message = (
            "enabled" if self.enabled else "paused"
        )
        return response

    def _on_capture(self, request, response):
        if self._last_rect_left is None:
            response.success = False
            response.message = "no frame captured yet"
            return response
        save_dir = self.project_root / "outcome" / "ros2_captures"
        save_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(str(save_dir / f"left_{ts}.jpg"), self._last_rect_left)
        cv2.imwrite(str(save_dir / f"right_{ts}.jpg"), self._last_rect_right)
        if self._last_raw is not None:
            cv2.imwrite(str(save_dir / f"stereo_{ts}.jpg"), self._last_raw)
        response.success = True
        response.message = str(save_dir)
        return response

    def destroy_node(self):
        if getattr(self, "cap", None) is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
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
