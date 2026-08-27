"""SGBM stereo node: rectified left/right images -> disparity topics."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_srvs.srv import SetBool

from .common import (
    compressed_image_msg,
    create_sgbm,
    decode_compressed,
    float32_image_msg,
    get_config_dir,
    load_yaml,
    make_disparity_visual,
)


class SgbmNode(Node):
    def __init__(self):
        super().__init__("sgbm_node")

        self.declare_parameter("config_dir", get_config_dir())
        self.declare_parameter("enabled", True)

        config_dir = Path(self.get_parameter("config_dir").value)
        self.enabled = bool(self.get_parameter("enabled").value)
        sgbm_cfg = load_yaml(config_dir / "SGBM_params.yaml")
        self.stereo, self.depth_cor_factor = create_sgbm(sgbm_cfg)

        self._left_frames = {}
        self._right_frames = {}
        self._last_process_time = time.monotonic()
        self._pair_timeout_ns = int(0.5 * 1e9)

        sensor_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.left_sub = self.create_subscription(
            CompressedImage, "/cap/camera/left", self._on_left, sensor_qos
        )
        self.right_sub = self.create_subscription(
            CompressedImage, "/cap/camera/right", self._on_right, sensor_qos
        )
        self.disparity_pub = self.create_publisher(
            Image, "/cap/sgbm/disparity", sensor_qos
        )
        self.visual_pub = self.create_publisher(
            CompressedImage, "/cap/sgbm/disparity_visual", sensor_qos
        )
        self.enable_srv = self.create_service(
            SetBool, "/cap/sgbm/enable", self._on_enable
        )
        self.get_logger().info("sgbm_node ready")

    @staticmethod
    def _stamp_key(stamp):
        return (stamp.sec, stamp.nanosec)

    def _prune(self):
        now_ns = self.get_clock().now().nanoseconds
        for store in (self._left_frames, self._right_frames):
            for key in list(store):
                stamp_ns = key[0] * 1_000_000_000 + key[1]
                if now_ns - stamp_ns > self._pair_timeout_ns:
                    del store[key]

    def _on_left(self, msg):
        if not self.enabled:
            return
        key = self._stamp_key(msg.header.stamp)
        self._left_frames[key] = msg
        self._prune()
        right = self._right_frames.pop(key, None)
        if right is not None:
            self._left_frames.pop(key, None)
            self._process(msg, right)

    def _on_right(self, msg):
        if not self.enabled:
            return
        key = self._stamp_key(msg.header.stamp)
        self._right_frames[key] = msg
        self._prune()
        left = self._left_frames.pop(key, None)
        if left is not None:
            self._right_frames.pop(key, None)
            self._process(left, msg)

    def _process(self, left_msg, right_msg):
        try:
            left_bgr = decode_compressed(left_msg)
            right_bgr = decode_compressed(right_msg)
            gray_left = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY)
            disp_raw = self.stereo.compute(gray_left, gray_right)
            disp_f32 = disp_raw.astype(np.float32) / 16.0

            now = time.monotonic()
            elapsed = now - self._last_process_time
            self._last_process_time = now
            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            self.disparity_pub.publish(
                float32_image_msg(
                    disp_f32, left_msg.header.stamp, "cap_left"
                )
            )
            visual = make_disparity_visual(disp_raw, disp_f32, fps)
            self.visual_pub.publish(
                compressed_image_msg(
                    visual, left_msg.header.stamp, "cap_depth", 88
                )
            )
        except Exception as exc:
            self.get_logger().error(f"SGBM frame failed: {exc}")

    def _on_enable(self, request, response):
        self.enabled = bool(request.data)
        response.success = True
        response.message = "enabled" if self.enabled else "paused"
        return response


def main(args=None):
    rclpy.init(args=args)
    node = SgbmNode()
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
