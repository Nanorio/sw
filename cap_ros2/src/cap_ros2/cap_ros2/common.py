"""Shared helpers for the CAP ROS2 nodes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import yaml


def load_yaml(path: Union[str, Path]) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def get_config_dir(package_name: str = "cap_ros2") -> str:
    """Return the installed share config dir, falling back to source layout."""
    try:
        from ament_index_python.packages import get_package_share_directory

        share = Path(get_package_share_directory(package_name))
        config = share / "config"
        if config.exists():
            return str(config)
    except Exception:
        pass
    return str(Path(__file__).resolve().parents[1] / "config")


def resolve_project_root(value: Optional[str]) -> Path:
    if not value:
        return Path.cwd().resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def build_rectify_maps(calib: dict) -> dict:
    image_size = tuple(int(v) for v in calib["imageSize"])
    left_matrix = np.array(calib["left"]["matrix"], dtype=float).T
    left_dist = np.array(calib["left"]["distortion"], dtype=float)
    right_matrix = np.array(calib["right"]["matrix"], dtype=float).T
    right_dist = np.array(calib["right"]["distortion"], dtype=float)
    rotation = np.array(calib["stereo"]["Rotation"], dtype=float).T
    translation = np.array(calib["stereo"]["Translation"], dtype=float).reshape(3, 1)

    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
        left_matrix,
        left_dist,
        right_matrix,
        right_dist,
        image_size,
        rotation,
        translation,
    )
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        left_matrix, left_dist, r1, p1, image_size, cv2.CV_16SC2
    )
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        right_matrix, right_dist, r2, p2, image_size, cv2.CV_16SC2
    )
    return {
        "image_size": image_size,
        "R1": r1,
        "R2": r2,
        "P1": p1,
        "P2": p2,
        "Q": q,
        "left_map1": left_map1,
        "left_map2": left_map2,
        "right_map1": right_map1,
        "right_map2": right_map2,
    }


def create_sgbm(cfg: dict):
    mode_name = cfg.get("mode", "STEREO_SGBM_MODE_SGBM")
    if not hasattr(cv2, mode_name):
        raise ValueError(f"Unsupported SGBM mode: {mode_name}")
    stereo = cv2.StereoSGBM_create(
        minDisparity=int(cfg["minDisparity"]),
        numDisparities=int(cfg["numDisparities"]),
        blockSize=int(cfg["blockSize"]),
        P1=int(cfg["P1_factor"]) * int(cfg["imgChannels"]) * int(cfg["blockSize"]) ** 2,
        P2=int(cfg["P2_factor"]) * int(cfg["imgChannels"]) * int(cfg["blockSize"]) ** 2,
        disp12MaxDiff=int(cfg["disp12MaxDiff"]),
        preFilterCap=int(cfg["preFilterCap"]),
        uniquenessRatio=int(cfg["uniquenessRatio"]),
        speckleWindowSize=int(cfg["speckleWindowSize"]),
        speckleRange=int(cfg["speckleRange"]),
        mode=getattr(cv2, mode_name),
    )
    return stereo, float(cfg.get("depthCorFactor", 1.0))


def compressed_image_msg(bgr: np.ndarray, stamp, frame_id: str, quality: int = 88):
    from sensor_msgs.msg import CompressedImage

    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    msg = CompressedImage()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.format = "jpeg"
    msg.data = buf.tobytes()
    return msg


def decode_compressed(msg) -> np.ndarray:
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode compressed image")
    return image


def float32_image_msg(image: np.ndarray, stamp, frame_id: str):
    from sensor_msgs.msg import Image

    img = np.ascontiguousarray(image, dtype=np.float32)
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(img.shape[0])
    msg.width = int(img.shape[1])
    msg.encoding = "32FC1"
    msg.is_bigendian = 0
    msg.step = int(img.shape[1] * 4)
    msg.data = img.tobytes()
    return msg


def decode_float32_image(msg) -> np.ndarray:
    if msg.encoding != "32FC1":
        raise ValueError(f"Expected 32FC1 image, got {msg.encoding}")
    array = np.frombuffer(msg.data, dtype=np.float32)
    return array.reshape(msg.height, msg.width)


def make_disparity_visual(disp_raw: np.ndarray, disp_f32: np.ndarray, fps: float = 0.0):
    mask_valid = cv2.compare(disp_raw, 1, cv2.CMP_GE)
    valid_vals = disp_f32[mask_valid > 0]
    if len(valid_vals) > 0:
        clipped = np.clip(disp_f32, 0, np.percentile(valid_vals, 99))
    else:
        clipped = disp_f32
    normalized = cv2.normalize(
        clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    color = cv2.bitwise_and(color, color, mask=mask_valid)
    if fps > 0.0:
        cv2.putText(
            color,
            f"SGBM FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    return color


def robust_depth_from_xyz(
    xyz_matrix: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    depth_cor_factor: float = 1.0,
    roi_x: float = 0.0,
    roi_y: float = 0.0,
    fx: float = 1.0,
    fy: float = 1.0,
):
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = max(x1 + 1, int(x2)), max(y1 + 1, int(y2))
    box_w, box_h = x2 - x1, y2 - y1
    margin_x = int(box_w * 0.2)
    margin_y = int(box_h * 0.2)
    roi = xyz_matrix[y1 + margin_y : y2 - margin_y, x1 + margin_x : x2 - margin_x]
    if roi.size == 0:
        return None

    z_vals = roi[:, :, 2]
    valid = (
        np.isfinite(z_vals)
        & (z_vals > 0.0)
        & (z_vals < 10000.0)
    )
    valid_pts = roi[valid]
    if len(valid_pts) < 5:
        return None

    median_x = float(np.median(valid_pts[:, 0]))
    median_y = float(np.median(valid_pts[:, 1]))
    median_z = float(np.median(valid_pts[:, 2])) * depth_cor_factor
    fx = fx if fx and fx > 0 else 1.0
    fy = fy if fy and fy > 0 else 1.0
    return np.array(
        [
            median_x + roi_x * median_z / fx,
            median_y + roi_y * median_z / fy,
            median_z,
        ],
        dtype=float,
    )
