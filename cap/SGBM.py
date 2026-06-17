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

            # 注意：此处保留了对 MATLAB 内参矩阵的转置 (.T) 和平移向量的形变 (.reshape)
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
            self.left_matrix, 
            self.left_distortion,
            self.right_matrix, 
            self.right_distortion,
            self.imageSize, 
            self.Rotation, 
            self.Translation
        )

        print(">>>\n生成左眼映射表中...\n...")
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.left_matrix, 
            self.left_distortion, 
            self.R1, 
            self.P1, 
            self.imageSize, 
            cv2.CV_16SC2
        )
        print("左眼映射表生成完毕!\n<<<\n————————————————————")

        print(">>>\n生成右眼映射表中...\n...")
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.right_matrix, 
            self.right_distortion, 
            self.R2, 
            self.P2, 
            self.imageSize, 
            cv2.CV_16SC2
        )
        print("右眼映射表生成完毕!\n<<<\n————————————————————")

    def _SGBM_param_init(self):
        """初始化SGBM参数"""
        print(">>>\n正在读取SGBM配置文件...\n...")
        SGBM_Configuration = "./config/SGBM_params.yaml"

        if not Path(SGBM_Configuration).exists():
            raise FileNotFoundError(f"找不到SGBM配置文件: {SGBM_Configuration}")
        
        with open(SGBM_Configuration, 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f)

            self.blockSize = params['blockSize']
            self.imgChannels = params['imgChannels']
            self.minDisparity = params['minDisparity']
            self.numDisparities = params['numDisparities']
            self.P1_factor = params['P1_factor']
            self.P2_factor = params['P2_factor']
            self.disp12MaxDiff = params['disp12MaxDiff']
            self.preFilterCap = params['preFilterCap']
            self.uniquenessRatio = params['uniquenessRatio']
            self.speckleWindowSize = params['speckleWindowSize']
            self.speckleRange = params['speckleRange']
            self.mode = getattr(cv2, params['mode'])

        print("SGBM配置读取完毕!")

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
                    print(f">>> 手动曝光 | 值={exp_manual_val} | 范围=[{exp_range[0]}, {exp_range[1]}] <<<")
                else:
                    self.captureData.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                    print(">>> 自动曝光 <<<")
            elif params['captureMode'] == 'one image':
                print(">>> 当前输入源为静态图片 <<<")
                self.captureData = cv2.VideoCapture('./staticSource/image/test.jpg')
            elif params['captureMode'] == 'one video':
                print(">>> 当前输入源为mp4媒体文件 <<<")
                self.captureData = cv2.VideoCapture('./staticSource/video/test.mp4')    
            
        print("读取捕捉配置文件成功!\n<<<\n————————————————————")

    def _detect_exposure_range(self, camera_id):
        """尝试读取相机驱动的默认曝光范围, 失败返回 [0, 10000]"""
        import platform
        if platform.system() != 'Linux':
            return [0, 10000]
        try:
            import subprocess
            result = subprocess.run(
                ['v4l2-ctl', '-d', f'/dev/video{camera_id}', '-L'],
                capture_output=True, text=True, timeout=3
            )
            for line in result.stdout.split('\n'):
                if 'exposure' in line.lower() and 'absolute' in line.lower():
                    import re
                    match = re.search(r'min=(\d+)\s+max=(\d+)', line)
                    if match:
                        return [int(match.group(1)), int(match.group(2))]
        except:
            pass
        return [0, 10000]

    def get_split_img(self):
        """从相机流中读取最新的一帧画面，并将其从中间切割为左右两张独立的图像"""
        retval, frame = self.captureData.read()

        if not retval:
            return False
        
        self.bgr_left = frame[:, :self.imageSize[0]]
        self.bgr_right = frame[:, self.imageSize[0]:]
        return True

    def rectify_images(self):
        """传入原始左右图像，返回极线校正且去畸变后的左右图像"""
        self.rectify_bgr_left = cv2.remap(self.bgr_left, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
        self.rectify_bgr_right = cv2.remap(self.bgr_right, self.right_map1, self.right_map2, cv2.INTER_LINEAR)
        self.rectify_gray_left = cv2.cvtColor(self.rectify_bgr_left, cv2.COLOR_BGR2GRAY)
        self.rectify_gray_right = cv2.cvtColor(self.rectify_bgr_right, cv2.COLOR_BGR2GRAY)

    def cpt_disparity(self):
        """计算视差图"""
        self.disparity = self.stereo.compute(self.rectify_gray_left, self.rectify_gray_right)

    def get_visual_disparity(self):
        """将底层计算出的视差图转换为可供人类直观观测的伪彩色 RGB 图像"""
        # 将无效的负数视差过滤掉（SGBM匹配失败的区域）
        disp_vis = numpy.maximum(self.disparity, 0)
        # 将 16 位整型归一化到 0~255 的 8 位图像
        disp_vis = cv2.normalize(disp_vis, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # 叠加 Jet 伪彩色滤镜 (红色表示近距离/高视差，蓝色表示远距离/低视差，黑色表示匹配失败)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)
        return disp_color

if __name__ == '__main__':

    tester = StereoSGBMTester()
    print("\n\n————————————————————")
    print("开始 SGBM 视差图实时测试")
    print("在弹出的图像窗口中按 'q' 键退出")
    print("————————————————————\n")
    
    while True:
        start_time = time.time()

        if not tester.get_split_img():
            print("未能获取到图像画面，重试中...")
            cv2.waitKey(500)
            continue

        try:
            # 运行核心流水线：极线校正 -> 计算视差
            tester.rectify_images() 
            tester.cpt_disparity() 
            
            # 获取伪彩色视差图
            disparity_color_map = tester.get_visual_disparity()
            
            # 计算帧率
            fps = 1.0 / (time.time() - start_time)
            
            # 在画面上打上英文水印 (避免中文字符乱码)
            cv2.putText(tester.rectify_bgr_left, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(disparity_color_map, "Disparity (Jet Colormap)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # 英文命名 GUI 窗口
            cv2.imshow("Rectified Left RGB", tester.rectify_bgr_left)
            cv2.imshow("Real-time Disparity Map", disparity_color_map)

        except Exception as e:
            print(f"处理发生异常: {e}")
            continue

        # 按下 'q' 键退出循环
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n>>> 检测到退出指令，正在释放资源...")
            break

    tester.captureData.release()
    cv2.destroyAllWindows()