#>>> 库导入
import os
import yaml
import numpy as np
from pathlib import Path
import cv2
import time
import threading
import queue
from ultralytics import YOLO  # 导入 YOLO
#<<<

# ==========================================
# 模块 1：光信号时序解码器
# ==========================================
class LightSignalDecoder:
    def __init__(self):
        # 防抖参数
        self.last_seen_time = 0.0     
        self.debounce_window = 0.15   # 防抖窗口：0.15秒内漏检依然认为灯是亮的
        self.is_logically_on = False  
        
        # 序列统计参数
        self.flash_count = 0          
        self.last_flash_time = 0.0    
        self.timeout_threshold = 1.2  # 超时结算：超过1.2秒没亮，视为这组信号发完了

    def update(self, yolo_detected_light: bool):
        current_time = time.time()

        # 1. 防抖处理
        if yolo_detected_light:
            self.last_seen_time = current_time
        
        current_logic_state = (current_time - self.last_seen_time) < self.debounce_window

        # 2. 边缘检测 (捕捉 "灭 -> 亮" 的瞬间)
        if current_logic_state and not self.is_logically_on:
            self.flash_count += 1
            self.last_flash_time = current_time
            
        self.is_logically_on = current_logic_state

        # 3. 超时结算
        if self.flash_count > 0 and (current_time - self.last_flash_time) > self.timeout_threshold:
            command = self._decode_pattern(self.flash_count)
            final_count = self.flash_count
            self.flash_count = 0 
            return command, final_count

        return None, 0

    def _decode_pattern(self, count):
        if count == 2: return "直行 (Go Straight)"
        elif count == 4: return "左转弯 (Left Turn)"
        elif count == 5: return "警告 (Warning)"
        elif count == 6: return "右转弯 (Right Turn)"
        else: return f"未知信号 (Unknown: {count} flashes)"

# ==========================================
# 模块 2：多线程双目 SGBM 引擎
# ==========================================
class StereoSGBMTester:
    def __init__(self):
        self.clicked_point = None
        self.is_running = True
        
        # 多线程通道
        self.raw_queue = queue.Queue(maxsize=2)
        self.processed_queue = queue.Queue(maxsize=2)
        
        self._load_camera_configuration()
        self._rectify_maps_init()
        self._SGBM_param_init()
        self._capture_init()
        
        if hasattr(self, 'captureData') and self.captureData.isOpened():
            threading.Thread(target=self._capture_worker, daemon=True).start()
            threading.Thread(target=self._process_worker, daemon=True).start()
            print(">>> 引擎启动：双目抓图与算力线程已在后台运行 <<<")

    def _load_camera_configuration(self):
        config_path = "./config/camera_params.yaml"
        if not Path(config_path).exists(): raise FileNotFoundError(config_path)
        with open(config_path, 'r', encoding='utf-8') as f: params = yaml.safe_load(f)
        self.left_matrix = np.array(params['left']['matrix']).T
        self.left_distortion = np.array(params['left']['distortion'])
        self.right_matrix = np.array(params['right']['matrix']).T
        self.right_distortion = np.array(params['right']['distortion'])
        self.imageSize = tuple(params['imageSize'])
        self.Q = np.array(params['stereo']['Q'])
        self.Rotation = np.array(params['stereo']['Rotation']).T
        self.Translation = np.array(params['stereo']['Translation']).reshape(3, 1)
    
    def _rectify_maps_init(self):
        self.R1, self.R2, self.P1, self.P2, self.Q, _, _ = cv2.stereoRectify(
            self.left_matrix, self.left_distortion, self.right_matrix, self.right_distortion,
            self.imageSize, self.Rotation, self.Translation)
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.left_matrix, self.left_distortion, self.R1, self.P1, self.imageSize, cv2.CV_16SC2)
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.right_matrix, self.right_distortion, self.R2, self.P2, self.imageSize, cv2.CV_16SC2)

    def _SGBM_param_init(self):
        config_path = "./config/SGBM_params.yaml"
        if not Path(config_path).exists(): raise FileNotFoundError(config_path)
        with open(config_path, 'r', encoding='utf-8') as f: params = yaml.safe_load(f)
        self.stereo = cv2.StereoSGBM_create(
            minDisparity = params.get('minDisparity', 0),
            numDisparities = params.get('numDisparities', 64),
            blockSize = params.get('blockSize', 7),
            P1 = params.get('P1_factor', 8) * params.get('imgChannels', 3) * params.get('blockSize', 7) ** 2,
            P2 = params.get('P2_factor', 32) * params.get('imgChannels', 3) * params.get('blockSize', 7) ** 2,
            disp12MaxDiff = params.get('disp12MaxDiff', 2),
            preFilterCap = params.get('preFilterCap', 63),
            uniquenessRatio = params.get('uniquenessRatio', 10),
            speckleWindowSize = params.get('speckleWindowSize', 100),
            speckleRange = params.get('speckleRange', 2),
            mode = getattr(cv2, params.get('mode', 'STEREO_SGBM_MODE_SGBM'))
        )
        
    def _capture_init(self):
        config_path = "./config/capture.yaml"
        if not Path(config_path).exists(): raise FileNotFoundError(config_path)
        with open(config_path, 'r', encoding='utf-8') as f: params = yaml.safe_load(f)
        if params['captureMode'] == 'camera':
            self.captureData = cv2.VideoCapture(params['cameraId'], cv2.CAP_V4L2)
            self.captureData.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.captureData.set(cv2.CAP_PROP_FRAME_WIDTH, self.imageSize[0] * 2)
            self.captureData.set(cv2.CAP_PROP_FRAME_HEIGHT, self.imageSize[1])
            if params.get('exposure', {}).get('mode', 'auto') == 'manual':
                self.captureData.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
                self.captureData.set(cv2.CAP_PROP_EXPOSURE, params.get('exposure', {}).get('manualValue', 50))
            else:
                self.captureData.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

    def _capture_worker(self):
        while self.is_running:
            retval, frame = self.captureData.read()
            if retval:
                if self.raw_queue.full():
                    try: self.raw_queue.get_nowait() 
                    except queue.Empty: pass
                self.raw_queue.put(frame)
            else:
                time.sleep(0.01)

    def _process_worker(self):
        while self.is_running:
            try: frame = self.raw_queue.get(timeout=0.1) 
            except queue.Empty: continue
                
            bgr_left = frame[:, :self.imageSize[0]]
            bgr_right = frame[:, self.imageSize[0]:]
            
            rect_bgr_left = cv2.remap(bgr_left, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
            rect_bgr_right = cv2.remap(bgr_right, self.right_map1, self.right_map2, cv2.INTER_LINEAR)
            rect_gray_left = cv2.cvtColor(rect_bgr_left, cv2.COLOR_BGR2GRAY)
            rect_gray_right = cv2.cvtColor(rect_bgr_right, cv2.COLOR_BGR2GRAY)
            
            disparity = self.stereo.compute(rect_gray_left, rect_gray_right)
            
            disp_vis = np.maximum(disparity, 0)
            disp_vis = cv2.normalize(disp_vis, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
            
            if self.processed_queue.full():
                try: self.processed_queue.get_nowait()
                except queue.Empty: pass
            
            self.processed_queue.put((rect_bgr_left, disp_color, disparity))

    def get_single_3d_point(self, x, y, disparity_matrix):
        d = disparity_matrix[y, x].astype(np.float32) / 16.0
        if d <= 0: return None
        point_2d_disp = np.array([[[x, y, d]]], dtype=np.float32)
        return cv2.perspectiveTransform(point_2d_disp, self.Q)[0][0]

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = (x, y)

    def cleanup(self):
        self.is_running = False
        time.sleep(0.5)
        if hasattr(self, 'captureData'): self.captureData.release()
        print("相机资源已安全释放。")


# ==========================================
# 模块 3：主程序 (GUI + YOLO + 信号解码)
# ==========================================
if __name__ == '__main__':
    print("\n==========================================")
    print("🚀 启动大模型驱动的异构机电协同视觉平台")
    print("==========================================\n")
    
    # 1. 初始化双目相机与 SGBM 引擎
    tester = StereoSGBMTester()
    
    # 2. 初始化 YOLOv8 蓝灯检测模型
    print(">>> 正在加载 YOLOv8 蓝灯模型...")
    yolo_weight_path = "./weights/best.pt"
    if not Path(yolo_weight_path).exists():
        print(f"⚠️ 找不到 YOLO 权重文件: {yolo_weight_path}，请检查路径！")
        yolo_model = None
    else:
        yolo_model = YOLO(yolo_weight_path, task="detect")
    
    # 3. 初始化光信号解码器
    decoder = LightSignalDecoder()
    
    window_name = "AI Coordinated Vision System"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, tester.on_mouse_click)
    
    last_frame_time = time.time()
    active_command = "None"
    command_display_timer = 0
    
    while True:
        try:
            # 尝试从队列拿到双目处理完毕的最新帧
            rect_bgr_left, disp_color, raw_disparity = tester.processed_queue.get(timeout=0.01)
        except queue.Empty:
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue

        # 计算真实系统处理 FPS
        current_time = time.time()
        fps = 1.0 / (current_time - last_frame_time)
        last_frame_time = current_time

        # ---------------------------------------------------------
        # A. 执行 YOLO 目标检测与信号解码
        # ---------------------------------------------------------
        blue_light_detected = False
        
        if yolo_model is not None:
            # 必须设置 stream=False 防止返回生成器崩溃
            results = yolo_model.predict(source=rect_bgr_left, conf=0.5, verbose=False, stream=False)
            boxes = results[0].boxes
            
            # 遍历识别到的框
            for box in boxes:
                cls_id = int(box.cls[0])
                # 假设 BlueLight 的类别 ID 为 0
                if cls_id == 0:
                    blue_light_detected = True
                    
                    # 画出蓝灯的 Bounding Box 方便观测
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(rect_bgr_left, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(rect_bgr_left, f"Lamp {box.conf[0]:.2f}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 送入解码器更新状态
        command, final_count = decoder.update(blue_light_detected)
        
        # 捕捉到有效的完整信号
        if command:
            if "未知" not in command:
                print(f"\n[指令下发] 捕获时序信号 -> {command}")
                active_command = command
                command_display_timer = 30  # 在屏幕上持续显示 30 帧
            else:
                print(f"⚠️ [环境干扰] 收到未知闪烁序列: {final_count} 次")

        # ---------------------------------------------------------
        # B. 交互功能：鼠标单点测距
        # ---------------------------------------------------------
        if tester.clicked_point is not None:
            cx, cy = tester.clicked_point
            pt_3d = tester.get_single_3d_point(cx, cy, raw_disparity)
            
            if pt_3d is not None:
                z_val = pt_3d[2] 
                if 0 < z_val < 10000:
                    text, color = f"Z: {z_val:.0f}mm", (0, 255, 0)
                else:
                    text, color = "Z: Out Range", (0, 165, 255)
            else:
                text, color = "Z: Invalid", (0, 0, 255)
            
            cv2.circle(rect_bgr_left, (cx, cy), 5, color, -1)
            cv2.putText(rect_bgr_left, text, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # ---------------------------------------------------------
        # C. GUI 状态渲染
        # ---------------------------------------------------------
        # 显示基础系统数据
        cv2.putText(rect_bgr_left, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(rect_bgr_left, f"Flash Buffer: {decoder.flash_count}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # 如果检测到灯是亮的，在左上角打个标记
        if decoder.is_logically_on:
            cv2.circle(rect_bgr_left, (250, 60), 10, (255, 255, 0), -1)
            
        # 屏幕中心大字显示当前解码的指令
        if command_display_timer > 0:
            cv2.putText(rect_bgr_left, f"CMD: {active_command}", (rect_bgr_left.shape[1]//2 - 150, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            command_display_timer -= 1
        
        # 渲染窗口
        cv2.imshow(window_name, rect_bgr_left)
        cv2.imshow("Disparity Heatmap", disp_color)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n>>> 收到退出指令，准备结束程序...")
            break

    tester.cleanup()
    cv2.destroyAllWindows()