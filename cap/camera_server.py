#!/usr/bin/env python3
"""
相机帧发布服务 — 共享内存 + ZMQ 通知，零拷贝帧分发。
启动:  python3 camera_server.py
"""

import cv2
import numpy
import zmq
import struct
import yaml
import signal
import sys
from pathlib import Path
from multiprocessing import shared_memory


def main():
    # ========== 读取配置 ==========
    cfg_path = Path("./config/capture.yaml")
    if not cfg_path.exists():
        print(f"[camera_server] 找不到配置文件: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    calib_path = Path("./config/camera_params.yaml")
    with open(calib_path, "r", encoding="utf-8") as f:
        calib = yaml.safe_load(f)

    img_w, img_h = calib["imageSize"]
    camera_id    = cfg.get("cameraId", 0)
    zmq_port     = cfg.get("zmqPort", 5555)

    # ========== 打开相机 ==========
    print(f"[camera_server] 打开相机 /dev/video{camera_id} ...")
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
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

    # ========== 共享内存（双缓冲） ==========
    cap_w     = img_w * 2
    cap_h     = img_h
    frame_sz  = cap_w * cap_h * 3          # 单帧字节数
    head_sz   = 8                           # int64 counter
    shm_size  = head_sz + frame_sz * 2      # 双缓冲总大小

    SHM_NAME = "cap_shm"
    try:
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=shm_size)
    except FileExistsError:
        # 上次残留，清理后重建
        old = shared_memory.SharedMemory(name=SHM_NAME)
        old.close()
        old.unlink()
        shm = shared_memory.SharedMemory(name=SHM_NAME, create=True, size=shm_size)

    # numpy 视图
    counter = numpy.ndarray((1,), dtype=numpy.int64, buffer=shm.buf[:head_sz])
    buf0    = numpy.ndarray((cap_h, cap_w, 3), dtype=numpy.uint8,
                            buffer=shm.buf[head_sz:head_sz+frame_sz])
    buf1    = numpy.ndarray((cap_h, cap_w, 3), dtype=numpy.uint8,
                            buffer=shm.buf[head_sz+frame_sz:])

    counter[0] = 0
    write_idx  = 0
    print(f"[camera_server] 共享内存: {SHM_NAME} ({shm_size} bytes)")
    print(f"[camera_server] 帧尺寸: {cap_w} x {cap_h}")

    # ========== ZMQ 通知通道 ==========
    context = zmq.Context()
    pub = context.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 4)
    addr = f"tcp://*:{zmq_port}"
    pub.bind(addr)
    print(f"[camera_server] ZMQ 通知: {addr} (每帧仅 8 字节)")

    # ========== 信号处理 ==========
    running = True
    def shutdown(sig, frame):
        nonlocal running
        print("\n[camera_server] 收到退出信号，关闭服务...")
        running = False
    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ========== 主循环 ==========
    count = 0
    while running:
        ret, frame = cap.read()
        if not ret:
            print("[camera_server] 读帧失败，重试中...")
            cv2.waitKey(100)
            continue

        # 写入当前缓冲
        target = buf0 if write_idx == 0 else buf1
        numpy.copyto(target, frame)

        # 翻写缓冲索引后更新共享计数器
        write_idx = 1 - write_idx
        counter[0] = numpy.int64(count + 1)

        # ZMQ 通知（仅 8 字节帧号，远轻于发整帧）
        pub.send(struct.pack("!q", counter[0]))

        count += 1
        if count % 200 == 0:
            print(f"[camera_server] 已发布 {count} 帧")

    # ========== 清理 ==========
    cap.release()
    pub.close()
    context.term()
    shm.close()
    shm.unlink()
    print("[camera_server] 已停止")


if __name__ == "__main__":
    main()
