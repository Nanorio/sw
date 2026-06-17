#>>> 库导入
import os
import yaml
import numpy
from pathlib import Path
import cv2
import time
#<<<

class StereoSGBMTester:

    def __init__(self):
        """双目SGBM独立测试类初始化"""
        self.clicked_point = None  # [新增] 存储鼠标点击的 (x, y) 坐标
        
        self._load_camera_configuration()
        self._rectify_maps_init()
        self._SGBM_param_init()
        self._capture_init()

    def _load_camera_configuration(self):
        """初始化双目相机参数配置""" 
        print(">>>\n正在读取相机内参配置文件...\n...")
        camera_configuration = "./config/camera_params.yaml"

        if not Path(camera_configuration).exists():
            raise FileNotFoundError(f"找不到相机内参配置文件: {camera_configuration}")
        
        with open(camera_configuration, 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f)

            self.left_matrix = numpy.array(params['left']['matrix']).T
            self.left_distortion = numpy.array(params['left']['distortion'])
            self.right_matrix = numpy.array(params['right']['matrix']).T
            self.right_distortion = numpy.array(params['right']['distortion'])
            
            self.imageSize = tuple(params['imageSize'])
            self.Q = numpy.array(params['stereo']['Q'])
            self.Rotation = numpy.array(params['stereo']['Rotation']).T
            self.Translation = numpy.array(params['stereo']['Translation']).reshape(3, 1)
        
        print("相机内参读取完毕!\n<<<\n————————————————————")
    
    def _rectify_maps_init(self):
        """初始化双目相机映射表"""
        self.R1, self.R2, self.P1, self.P2, self.Q, self.validPixROI1, self.validPixROI2 = cv2.stereoRectify(
            self.left_matrix, self.left_distortion,
            self.right_matrix, self.right_distortion,
            self.imageSize, self.Rotation, self.Translation
        )

        print(">>>\n生成映射表中...\n...")
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.left_matrix, self.left_distortion, self.R1, self.P1, self.imageSize, cv2.CV_16SC2)
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.right_matrix, self.right_distortion, self.R2, self.P2, self.imageSize, cv2.CV_16SC2)
        print("映射表生成完毕!\n<<<\n————————————————————")

    def _SGBM_param_init(self):
        """初始化SGBM参数"""
        print(">>>\n正在读取SGBM配置文件...\n...")
        SGBM_Configuration = "./config/SGBM_params.yaml"

        if not Path(SGBM_Configuration).exists():
            raise FileNotFoundError(f"找不到SGBM配置文件: {SGBM_Configuration}")
        
        with open(SGBM_Configuration, 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f)

            self.blockSize = params.get('blockSize', 7)
            self.imgChannels = params.get('imgChannels', 3)
            self.minDisparity = params.get('minDisparity', 0)
            self.numDisparities = params.get('numDisparities', 64)
            self.P1_factor = params.get('P1_factor', 8)
            self.P2_factor = params.get('P2_factor', 32)
            self.disp12MaxDiff = params.get('disp12MaxDiff', 2)
            self.preFilterCap = params.get('preFilterCap', 63)
            self.uniquenessRatio = params.get('uniquenessRatio', 10)
            self.speckleWindowSize = params.get('speckleWindowSize', 100)
            self.speckleRange = params.get('speckleRange', 2)
            mode_str = params.get('mode', 'STEREO_SGBM_MODE_SGBM')
            self.mode = getattr(cv2, mode_str)

        self.stereo = cv2.StereoSGBM_create(minDisparity = self.minDisparity,
                                   numDisparities = self.numDisparities,
                                   blockSize = self.blockSize,
                                   P1 = self.P1_factor * self.imgChannels * self.blockSize ** 2,
                                   P2 = self.P2_factor * self.imgChannels * self.blockSize ** 2,
                                   disp12MaxDiff = self.disp12MaxDiff,
                                   preFilterCap = self.preFilterCap,
                                   uniquenessRatio = self.uniquenessRatio,
                                   speckleWindowSize = self.speckleWindowSize,
                                   speckleRange = self.speckleRange,
                                   mode = self.mode)
        print("初始化SGBM配置完毕!\n<<<\n————————————————————")
        
    def _capture_init(self):
        print(">>>\n正在读取捕捉配置文件...\n...")
        self.capture_configuration = "./config/capture.yaml"

        if not Path(self.capture_configuration).exists():
            raise FileNotFoundError(f"找不到捕捉配置文件: {self.capture_configuration}")
        
        with open(self.capture_configuration, 'r', encoding = 'utf-8') as f:
            params = yaml.safe_load(f)

            if params['captureMode'] == 'camera':
                print(">>> 当前输入源为相机 <<<")
                self.captureData = cv2.VideoCapture(params['cameraId'], cv2.CAP_V4L2)
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.captureData.set(cv2.CAP_PROP_FOURCC, fourcc)
                self.captureData.set(cv2.CAP_PROP_FRAME_WIDTH, self.imageSize[0] * 2)
                self.captureData.set(cv2.CAP_PROP_FRAME_HEIGHT, self.imageSize[1])
                
                exp_cfg = params.get('exposure', {})
                exp_mode = exp_cfg.get('mode', 'auto')
                exp_manual_val = exp_cfg.get('manualValue', 50)

                exp_range = exp_cfg.get('cameraRange', [])
                if not exp_range:
                    exp_range = self._detect_exposure_range(params['cameraId'])
                    params.setdefault('exposure', {})['cameraRange'] = exp_range
                    with open(self.capture_configuration, 'w', encoding='utf-8') as fw:
                        yaml.dump(params, fw, default_flow_style=False, allow_unicode=True)

                if exp_mode == 'manual':
                    self.captureData.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1.0)
                    self.captureData.set(cv2.CAP_PROP_EXPOSURE, exp_manual_val)
                    print(f">>> 手动曝光 | 值={exp_manual_val} <<<")
                else:
                    self.captureData.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                    print(">>> 自动曝光 <<<")
            else:
                 # 处理静态图片或视频逻辑...
                 pass 
            
        print("读取捕捉配置文件成功!\n<<<\n————————————————————")

    def _detect_exposure_range(self, camera_id):
        # ... 保持不变 ...
        return [0, 10000]

    def get_split_img(self):
        retval, frame = self.captureData.read()
        if not retval:
            return False
        self.bgr_left = frame[:, :self.imageSize[0]]
        self.bgr_right = frame[:, self.imageSize[0]:]
        return True

    def rectify_images(self):
        self.rectify_bgr_left = cv2.remap(self.bgr_left, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
        self.rectify_bgr_right = cv2.remap(self.bgr_right, self.right_map1, self.right_map2, cv2.INTER_LINEAR)
        self.rectify_gray_left = cv2.cvtColor(self.rectify_bgr_left, cv2.COLOR_BGR2GRAY)
        self.rectify_gray_right = cv2.cvtColor(self.rectify_bgr_right, cv2.COLOR_BGR2GRAY)

    def cpt_disparity(self):
        self.disparity = self.stereo.compute(self.rectify_gray_left, self.rectify_gray_right)

    # [新增] 把转换 3D 点云的函数加回来
    def cpt_xyz(self):
        """计算视差对应的 3D 坐标矩阵"""
        disparity_float = self.disparity.astype(numpy.float32) / 16.0
        self.xyz = cv2.reprojectImageTo3D(disparity_float, self.Q, handleMissingValues=True)

    def get_visual_disparity(self):
        disp_vis = numpy.maximum(self.disparity, 0)
        disp_vis = cv2.normalize(disp_vis, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    # [新增] 鼠标点击回调函数
    def on_mouse_click(self, event, x, y, flags, param):
        """监听鼠标左键点击，记录点击坐标"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = (x, y)
            # 在终端同步输出一次信息
            if hasattr(self, 'xyz'):
                z = -self.xyz[y, x, 2]*0.71
                print(f"[UI 探测] 像素坐标: ({x}, {y}) | 对应深度 Z: {z:.2f} mm")


if __name__ == '__main__':

    tester = StereoSGBMTester()
    print("\n\n————————————————————")
    print("开始 SGBM 视差图实时测试")
    print("👉 请在 [Rectified Left RGB] 窗口中用鼠标【左键点击】探测深度")
    print("按 'q' 键退出")
    print("————————————————————\n")
    
    # [核心修改 1] 必须先创建窗口，才能把鼠标事件绑定上去
    window_name = "Rectified Left RGB"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, tester.on_mouse_click)
    
    while True:
        start_time = time.time()

        if not tester.get_split_img():
            print("未能获取到图像画面，重试中...")
            cv2.waitKey(500)
            continue

        try:
            tester.rectify_images() 
            tester.cpt_disparity() 
            # [核心修改 2] 在主循环中计算 3D 坐标
            tester.cpt_xyz()
            
            disparity_color_map = tester.get_visual_disparity()
            
            # [核心修改 3] 如果用户点击了某个点，在画面上实时追踪显示它的深度
            if tester.clicked_point is not None:
                cx, cy = tester.clicked_point
                # 取出该点的 Z 轴距离
                z_val = tester.xyz[cy, cx, 2]
                
                # 判定深度是否有效 (过滤无效的极大值或负数)
                if 0 < z_val < 10000:
                    text = f"Z: {z_val:.0f}mm"
                    color = (0, 255, 0) # 绿色代表数据健康
                else:
                    text = "Z: Invalid"
                    color = (0, 0, 255) # 红色代表无效空洞
                
                # 在点击位置画一个实心点，并打印数据
                cv2.circle(tester.rectify_bgr_left, (cx, cy), 5, color, -1)
                cv2.putText(tester.rectify_bgr_left, text, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(tester.rectify_bgr_left, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(disparity_color_map, "Disparity (Jet Colormap)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 显示窗口
            cv2.imshow(window_name, tester.rectify_bgr_left)
            cv2.imshow("Real-time Disparity Map", disparity_color_map)

        except Exception as e:
            print(f"处理发生异常: {e}")
            continue

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n>>> 检测到退出指令，正在释放资源...")
            break

    tester.captureData.release()
    cv2.destroyAllWindows()