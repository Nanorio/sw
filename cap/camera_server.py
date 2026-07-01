#!/usr/bin/env python3
"""
相机帧发布服务 —— 读真实相机 → v4l2loopback 虚拟设备。
任何支持 V4L2 的软件打开 /dev/video2 即可获取实时画面。

前置安装（边缘端跑一次）:
  sudo apt install ffmpeg v4l2loopback-dkms
  sudo modprobe v4l2loopback video_nr=2 card_label="CAM_virtual" exclusive_caps=1

启动:
  python3 camera_server.py
"""

import cv2
import subprocess
import signal
import sys
import yaml
import time
from pathlib import Path


def main():
    # ========== 读取配置 ==========
    cfg_path = Path("./config/capture.yaml")
    if not cfg_path.exists():
        print(f"[camera_server] 找不到 {cfg_path}")
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    calib_path = Path("./config/camera_params.yaml")
    with open(calib_path, "r", encoding="utf-8") as f:
        calib = yaml.safe_load(f)

    img_w, img_h = calib["imageSize"]                     # 单目尺寸
    real_id      = cfg.get("v4l2", {}).get("realDeviceId", 0)
    virt_id      = cfg.get("v4l2", {}).get("virtualDeviceId", 2)
    virt_dev     = f"/dev/video{virt_id}"

    # ========== 打开真实相机 ==========
    print(f"[camera_server] 打开 /dev/video{real_id} ...")
    cap = cv2.VideoCapture(real_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("[camera_server] 相机打开失败！")
        sys.exit(1)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  img_w * 2)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, img_h)

    exp_cfg = cfg.get("exposure", {})
    if exp_cfg.get("mode") == "manual":
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
        cap.set(cv2.CAP_PROP_EXPOSURE, exp_cfg.get("manualValue", 50))
        print(f"[camera_server] 手动曝光 | 值={exp_cfg.get('manualValue', 50)}")
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        print("[camera_server] 自动曝光")

    # 读第一帧确认实际尺寸
    ret, sample = cap.read()
    if not ret:
        print("[camera_server] 无法获取第一帧，退出")
        sys.exit(1)
    cap_w = sample.shape[1]
    cap_h = sample.shape[0]
    print(f"[camera_server] 帧尺寸: {cap_w} x {cap_h}")

    # ========== 启动 ffmpeg → 虚拟设备 ==========
    print(f"[camera_server] 启动 ffmpeg → {virt_dev} ...")
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y",
         "-f", "rawvideo",
         "-pixel_format", "bgr24",
         "-video_size", f"{cap_w}x{cap_h}",
         "-framerate", "30",
         "-i", "pipe:",
         "-f", "v4l2",
         "-pix_fmt", "yuv420p",
         virt_dev],
        stdin=subprocess.PIPE,
        stderr=subprocess.DEVNULL   # 设为 sys.stderr 可看 ffmpeg 日志
    )
    print(f"[camera_server] 已运行，{virt_dev} 可被任意 V4L2 软件打开")

    # ========== 信号处理 ==========
    running = True
    def shutdown(sig, frame):
        nonlocal running
        print("\n[camera_server] 关闭...")
        running = False
    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ========== 主循环 ==========
    count = 0
    last_log = time.time()

    while running:
        ret, frame = cap.read()
        if not ret:
            print("[camera_server] 读帧失败，重试...")
            cv2.waitKey(100)
            continue

        ffmpeg.stdin.write(frame.tobytes())
        count += 1

        if time.time() - last_log >= 10.0:
            print(f"[camera_server] 已转发 {count} 帧")
            last_log = time.time()

    # ========== 清理 ==========
    ffmpeg.stdin.close()
    ffmpeg.wait()
    cap.release()
    print("[camera_server] 已停止")


if __name__ == "__main__":
    main()
