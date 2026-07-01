#!/usr/bin/env python3
"""
测试用 —— 打开 /dev/video2 显示相机画面（v4l2loopback 虚拟设备）。
camera_server 在后台时，任何软件开 /dev/video2 都能收到实时帧。
"""

import cv2
import yaml
from pathlib import Path


def main():
    cfg_path = Path("./config/capture.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    camera_id = cfg.get("v4l2", {}).get("virtualDeviceId", 2)

    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[test_viewer] 无法打开 /dev/video{camera_id}，camera_server 是否在运行？")
        return

    print(f"[test_viewer] 已连接 /dev/video{camera_id}，按 q 退出")

    cv2.namedWindow("test_viewer", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[test_viewer] 读帧失败")
            break

        cv2.imshow("test_viewer", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[test_viewer] 已退出")


if __name__ == "__main__":
    main()
