"""YOLO detection node: rectified left image -> detections and annotated image."""

from __future__ import annotations

from pathlib import Path

import traceback
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_srvs.srv import SetBool, Trigger
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose

from .common import (
    compressed_image_msg,
    decode_compressed,
    get_config_dir,
    load_yaml,
    resolve_project_root,
)


class YoloNode(Node):
    def __init__(self):
        super().__init__("yolo_node")

        self.declare_parameter("config_dir", get_config_dir())
        self.declare_parameter("project_root", str(Path.cwd()))
        self.declare_parameter("model_path", "")
        self.declare_parameter("conf", -1.0)
        self.declare_parameter("device", "")
        self.declare_parameter("jpeg_quality", 88)
        self.declare_parameter("enabled", True)

        config_dir = Path(self.get_parameter("config_dir").value)
        self._config_dir = config_dir
        self.project_root = resolve_project_root(
            self.get_parameter("project_root").value
        )
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.enabled = bool(self.get_parameter("enabled").value)

        params = load_yaml(config_dir / "yolo_params.yaml")
        self._weight_name = str(params.get("weightName", "pjhdBG.engine"))
        conf_param = float(self.get_parameter("conf").value)
        self.conf = conf_param if conf_param >= 0 else float(
            params.get("conf", 0.30)
        )
        device_param = self.get_parameter("device").value
        self.device = device_param if device_param else str(
            params.get("device", "0")
        )

        model_path = self._resolve_model_path()
        self.get_logger().info(f"Loading YOLO model: {model_path}")
        from ultralytics import YOLO

        self.model = YOLO(str(model_path), task="detect")
        self._names = self._get_names()
        self.get_logger().info(f"YOLO classes: {self._names}")

        sensor_qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self.left_sub = self.create_subscription(
            CompressedImage, "/cap/camera/left", self._on_left, sensor_qos
        )
        self.det_pub = self.create_publisher(
            Detection2DArray, "/cap/yolo/detections", 10
        )
        self.enable_srv = self.create_service(
            SetBool, "/cap/yolo/enable", self._on_enable
        )
        self.reload_srv = self.create_service(
            Trigger, "/cap/yolo/reload", self._on_reload
        )

    def _resolve_model_path(self) -> Path:
        param_value = self.get_parameter("model_path").value
        weight_name = self._weight_name
        candidates = []
        if param_value:
            path = Path(param_value).expanduser()
            if not path.is_absolute():
                path = self.project_root / path
            candidates.append(path)
        candidates += [
            self.project_root / "weights" / weight_name,
            Path.cwd() / "weights" / weight_name,
            Path(__file__).resolve().parents[4] / "weights" / weight_name,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"YOLO weights not found for {weight_name}; set model_path or symlink weights"
        )

    def _get_names(self):
        names = getattr(self.model, "names", {})
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        if isinstance(names, list):
            return {i: str(name) for i, name in enumerate(names)}
        return {}

    def _on_left(self, msg):
        if not self.enabled:
            return
        try:
            image = decode_compressed(msg)
            results = self.model.predict(
                source=image,
                conf=self.conf,
                device=self.device,
                save=False,
                show=False,
                stream=False,
                verbose=False,
            )
            result = results[0]
            detections = self._build_detections(result, msg.header)
            self.det_pub.publish(detections)
        except Exception:
            self.get_logger().error(f"YOLO frame failed: {traceback.format_exc()}")

    def _build_detections(self, result, header):
        array = Detection2DArray()
        array.header = header
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return array
        for index, box in enumerate(boxes):
            detection = Detection2D()
            class_id = int(box.cls[0])
            class_name = self._names.get(class_id, str(class_id))
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(class_id)
            hypothesis.hypothesis.score = float(box.conf[0])
            detection.results.append(hypothesis)
            detection.bbox.center.position.x = float(center_x)
            detection.bbox.center.position.y = float(center_y)
            detection.bbox.size_x = float(x2 - x1)
            detection.bbox.size_y = float(y2 - y1)
            detection.id = f"{class_name}_{index}"
            array.detections.append(detection)
        return array

    def _on_enable(self, request, response):
        self.enabled = bool(request.data)
        response.success = True
        response.message = "enabled" if self.enabled else "paused"
        return response

    def _on_reload(self, request, response):
        try:
            model_path = self._resolve_model_path()
            from ultralytics import YOLO

            self.model = YOLO(str(model_path), task="detect")
            self._names = self._get_names()
            response.success = True
            response.message = f"reloaded {model_path}"
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = YoloNode()
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
