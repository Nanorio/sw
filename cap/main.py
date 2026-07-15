#>>> 库导入
import os
import sys
import datetime
import subprocess
import atexit
import time
import cv2
import numpy
from functions.cap_init import StereoCamera
from functions.yolo_detector import YOLOV8
from functions.outcome import outcome_action
from functions.robot_coord import RobotCoord
from functions.vispy_viewer import Robot3DViewer
#<<<


def main():
    procs = []  # [camera_server, web_viewer]

    # ========== 0. 设置 v4l2loopback ==========
    print(">>> 设置 v4l2loopback ...")
    r = subprocess.run([sys.executable, "functions/setup_v4l2.py"])
    if r.returncode != 0:
        print("v4l2loopback 设置失败，请手动:")
        print("  sudo python3 functions/setup_v4l2.py")
        sys.exit(1)

    # ========== 1. 启动相机服务（守护进程）==========
    print(">>> 启动 camera_server ...")
    procs.append(subprocess.Popen([sys.executable, "functions/camera_server.py"]))
    time.sleep(8)

    # ========== 2. 启动网页预览服务（守护进程）==========
    print(">>> 启动 web_viewer ...")
    procs.append(subprocess.Popen([sys.executable, "functions/web_viewer.py"]))

    # ========== 退出时统一清理子进程 ==========
    def _cleanup():
        for p in procs:
            if p.poll() is None:
                try: p.terminate()
                except: pass
        for p in procs:
            try: p.wait(timeout=5)
            except: pass
    atexit.register(_cleanup)

    # ========== 3. 初始化各模块 ==========
    print(">>> 初始化 StereoCamera ...")
    cam = StereoCamera()
    print(">>> 初始化 YOLOV8 ...")
    yolo = YOLOV8()
    # ========== 加载机器人坐标系 ==========
    robot = RobotCoord()
    _cam_id = robot.active_cam_id

    # ========== 启动 3D 视图 ==========
    import yaml
    with open("./config/camera_params.yaml", "r", encoding="utf-8") as _fh:
        _cal = yaml.safe_load(_fh)
    _fx = _cal["left"]["matrix"][0][0]
    _fy = _cal["left"]["matrix"][1][1]
    viewer = Robot3DViewer(robot, _fx, _fy)

    print("\n\n————————————————————\n开始检测\n————————————————————")

    # ========== 4. 主循环 ==========
    win_name = f"CAP Detection ({os.getpid()})"
    retry = 0
    try:
        while True:
            loop_start = datetime.datetime.now()
            if procs[0].poll() is not None:
                print("[camera_server] 已退出，无法获取画面")
                print("请检查 v4l2loopback 是否加载正确"); break
            if not cam.get_split_img():
                retry += 1
                if retry > 60:
                    print("相机持续无画面输入，退出检测"); break
                print(f"相机无画面输入，重试第 {retry} 次..."); cv2.waitKey(500); continue
            retry = 0
            try:
                cam.rectify_images(); yolo.predict(cam.rectify_bgr_left)
                cam.cpt_disparity(); cam.cpt_xyz()
                _le = datetime.datetime.now()
                current_fps = 1.0 / (_le - loop_start).total_seconds() if (_le - loop_start).total_seconds() > 0 else 0

                mask_valid = cv2.compare(cam.disparity, 1, cv2.CMP_GE)
                disp_f32 = cam.disparity.astype(numpy.float32)
                valid_vals = disp_f32[mask_valid > 0]
                if len(valid_vals) > 0:
                    disp_f32 = numpy.clip(disp_f32, 0, numpy.percentile(valid_vals, 99))
                disp_norm = cv2.normalize(disp_f32, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
                disp_color = cv2.bitwise_and(disp_color, disp_color, mask=mask_valid)
                cv2.putText(disp_color, f"FPS: {current_fps:.1f}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("SGBM Depth", disp_color)

                display_img = outcome_action(yolo.results, cam.xyz, cam.rectify_bgr_left, current_fps)
                cv2.imshow(win_name, display_img)

                # ===== 3D 视图 + 终端输出 =====
                if hasattr(outcome_action, 'last_center_cam'):
                    _cam_pt = outcome_action.last_center_cam
                    _abs_pt = robot.camera_to_abs(_cam_id, _cam_pt)
                    # 独立保护 3D 更新，不影响主线程
                    try:
                        viewer.update_target(_abs_pt)
                    except Exception as _e3d:
                        print(f"3D更新异常: {_e3d}")
                    _y = outcome_action.last_yaw
                    _p = outcome_action.last_pitch
                    _r = outcome_action.last_roll
                    _loc = f"{_abs_pt[0]:.0f},{_abs_pt[1]:.0f},{_abs_pt[2]:.0f}"
                    if _r is not None:
                        print(f"[Cam {_cam_id}]目标 → Yaw:{_y:+.1f} Pitch:{_p:+.1f} Roll:{_r:+.1f} Depth:{_cam_pt[2]:.0f}mm Location:{_loc}")
                    else:
                        print(f"[Cam {_cam_id}]目标 → Yaw:{_y:+.1f} Pitch:{_p:+.1f} Depth:{_cam_pt[2]:.0f}mm Location:{_loc}")
                else:
                    try:
                        viewer.update_target(None)
                    except Exception as _e3d:
                        print(f"3D更新异常: {_e3d}")
                try:
                    Robot3DViewer.process_events()
                except Exception as _e3d:
                    print(f"3D事件异常: {_e3d}")

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("用户按下 q，退出检测"); break
            except Exception as e:
                import traceback
                print(f"处理这一帧时发生异常: {e}\n{traceback.format_exc()}"); continue
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，退出检测")
    finally:
        cam.captureData.release(); cv2.destroyAllWindows()
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try: p.wait(timeout=5)
            except: pass
        print("程序退出")


if __name__ == "__main__":
    main()