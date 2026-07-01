#>>> 库导入
import os
import sys
import datetime
import subprocess
import atexit
import signal
import time
import cv2
from cap_init import StereoCamera
from yolo_detector import YOLOV8
from outcome import outcome_action
#<<<


def main():
    # ========== 1. 启动相机服务 ==========
    print(">>> 启动 camera_server ...")
    camera_proc = subprocess.Popen(
        [sys.executable, "camera_server.py"],
        # stdout/stderr 继承终端，能看到 camera_server 日志
    )
    atexit.register(lambda: (
        camera_proc.terminate(),
        camera_proc.wait()
    ) if camera_proc.poll() is None else None)

    # 等 camera_server 初始化（v4l2loopback + ffmpeg 就绪）
    time.sleep(2)

    # ========== 2. 初始化各模块 ==========
    print(">>> 初始化 StereoCamera ...")
    cam = StereoCamera()
    print(">>> 初始化 YOLOV8 ...")
    yolo = YOLOV8()
    print("\n\n————————————————————\n开始检测\n————————————————————")

    # ========== 3. 主循环 ==========
    win_name = f"CAP Detection ({os.getpid()})"
    retry = 0

    try:
        while True:
            loop_start = datetime.datetime.now()

            # 检查 camera_server 是否还活着
            if camera_proc.poll() is not None:
                print("\n[camera_server] 已退出，无法获取画面")
                print("请检查:")
                print("  1. 是否已加载 v4l2loopback:")
                print('     sudo modprobe v4l2loopback video_nr=2 card_label="CAM_virtual"')
                print("  2. 相机是否连接正确")
                break

            if not cam.get_split_img():
                retry += 1
                if retry > 60:
                    print("相机持续无画面输入，退出检测")
                    break
                print(f"相机无画面输入，重试第 {retry} 次...")
                cv2.waitKey(500)
                continue
            retry = 0

            try:
                cam.rectify_images()
                yolo.predict(cam.rectify_bgr_left)
                cam.cpt_disparity()
                cam.cpt_xyz()

                display_img = outcome_action(
                    yolo.results, cam.xyz, cam.rectify_bgr_left
                )
                cv2.imshow(win_name, display_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("用户按下 q，退出检测")
                    break

            except Exception as e:
                print(f"处理这一帧时发生异常: {e}")
                continue

            loop_end = datetime.datetime.now()
            duration = loop_end - loop_start
            total_seconds = duration.total_seconds()
            hz = 1.0 / total_seconds if total_seconds > 0 else 0
            print(f">>> 单次循环耗时: {total_seconds:.4f}s | 速率: {hz:.2f} Hz")
            print("-" * 50)

    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，退出检测")

    finally:
        # 清理
        cam.captureData.release()
        cv2.destroyAllWindows()
        camera_proc.terminate()
        camera_proc.wait()
        print("程序退出")


if __name__ == "__main__":
    main()
