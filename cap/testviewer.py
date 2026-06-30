#!/usr/bin/env python3
"""
测试用 — 独立消费者进程，验证 camera_server 共享内存分发。
不依赖 cap.py 的任何代码，仅展示相机画面。
"""

import cv2
import numpy
import zmq
import struct
import yaml
from pathlib import Path
from multiprocessing import shared_memory


def main():
    # 读取配置（端口、图像尺寸）
    cfg_path = Path("./config/capture.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    calib_path = Path("./config/camera_params.yaml")
    with open(calib_path, "r", encoding="utf-8") as f:
        calib = yaml.safe_load(f)

    img_w, img_h = calib["imageSize"]
    zmq_port = cfg.get("zmqPort", 5555)

    # 1. ZMQ SUB — 收通知
    context = zmq.Context()
    sub = context.socket(zmq.SUB)
    sub.setsockopt(zmq.RCVHWM, 4)
    sub.setsockopt_string(zmq.SUBSCRIBE, "")
    sub.connect(f"tcp://localhost:{zmq_port}")

    # 2. 共享内存 — 读帧（零拷贝）
    cap_w, cap_h = img_w * 2, img_h
    frame_sz = cap_w * cap_h * 3
    shm = shared_memory.SharedMemory(name="cap_shm")
    counter_view = numpy.ndarray((1,), dtype=numpy.int64, buffer=shm.buf[:8])
    buf0 = numpy.ndarray((cap_h, cap_w, 3), dtype=numpy.uint8,
                          buffer=shm.buf[8:8+frame_sz])
    buf1 = numpy.ndarray((cap_h, cap_w, 3), dtype=numpy.uint8,
                          buffer=shm.buf[8+frame_sz:])

    last_counter = 0
    print("[test_viewer] 已连接，等待帧...")
    print("[test_viewer] 按 q 退出\n")

    cv2.namedWindow("test_viewer", cv2.WINDOW_NORMAL)

    while True:
        if sub.poll(timeout=3000):
            msg = sub.recv()
            counter = struct.unpack("!q", msg)[0]

            # camera_server 写入的是 buffer[(counter-1) % 2]
            safe = buf0 if (counter - 1) % 2 == 0 else buf1
            frame = safe.copy()  # 必须 copy（显示用）

            cv2.imshow("test_viewer", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        else:
            print("[test_viewer] ZMQ 超时，camera_server 是否在运行？")
            break

    sub.close()
    context.term()
    shm.close()
    cv2.destroyAllWindows()
    print("[test_viewer] 已退出")


if __name__ == "__main__":
    main()
