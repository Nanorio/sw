#!/usr/bin/env python3
"""相机帧发布服务 —— 读真实相机 → v4l2loopback 虚拟设备"""

import cv2
import subprocess
import signal
import sys
import yaml
import time
from pathlib import Path


def main():
    cfg_path = Path("./config/capture.yaml")
    if not cfg_path.exists():
        print("[camera_server] 找不到配置文件"); sys.exit(1)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    calib_path = Path("./config/camera_params.yaml")
    with open(calib_path, "r", encoding="utf-8") as f:
        calib = yaml.safe_load(f)
    img_w, img_h = calib["imageSize"]
    real_id = cfg.get("v4l2", {}).get("realDeviceId", 0)
    virt_id = cfg.get("v4l2", {}).get("virtualDeviceId", 2)
    virt_dev = f"/dev/video{virt_id}"

    print(f"[camera_server] 打开 /dev/video{real_id}")
    cap = cv2.VideoCapture(real_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("[camera_server] 相机打开失败！"); sys.exit(1)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, img_w * 2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_h)
    exp = cfg.get("exposure", {})
    if exp.get("mode") == "manual":
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
        cap.set(cv2.CAP_PROP_EXPOSURE, exp.get("manualValue", 50))
        print(f"[camera_server] 手动曝光")
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        print(f"[camera_server] 自动曝光")
    ret, sample = cap.read()
    if not ret:
        print("[camera_server] 无法获取第一帧"); sys.exit(1)
    cap_w, cap_h = sample.shape[1], sample.shape[0]
    print(f"[camera_server] 帧尺寸: {cap_w}x{cap_h}")

    print(f"[camera_server] 启动 ffmpeg → {virt_dev}")
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-f", "rawvideo", "-pixel_format", "bgr24",
         "-video_size", f"{cap_w}x{cap_h}", "-framerate", "30",
         "-i", "pipe:", "-f", "v4l2", "-pix_fmt", "yuyv422", virt_dev],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    print(f"[camera_server] 已运行, {virt_dev} YUYV 422")

    running = True
    def shutdown(sig, frame):
        nonlocal running; running = False
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while running:
        ret, frame = cap.read()
        if not ret:
            print("[camera_server] 读帧失败"); cv2.waitKey(100); continue
        try:
            ffmpeg.stdin.write(frame.tobytes())
        except BrokenPipeError:
            print("[camera_server] ffmpeg 崩溃")
            print("  需 sudo modprobe v4l2loopback video_nr=2 exclusive_caps=1")
            break

    ffmpeg.stdin.close(); ffmpeg.wait(); cap.release()
    print("[camera_server] 已停止")


if __name__ == "__main__":
    main()
