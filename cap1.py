# -*- coding: utf-8 -*-

#>>> 库导入
import os
import datetime
import yaml
import numpy
from pathlib import Path
import cv2
from ultralytics import YOLO
from pnp_solver import solve_and_visualize
#<<<

class StereoCamera:

    def __init__(self):
        """双目相机总初始化"""
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

            self.left_matrix = numpy.array(params['left']['matrix'])
            self.left_distortion = numpy.array(params['left']['distortion'])
            self.right_matrix = numpy.array(params['right']['matrix'])
            self.right_distortion = numpy.array(params['right']['distortion'])
            
            self.imageSize = tuple(params['imageSize'])
            self.Q = numpy.array(params['stereo']['Q'])
            self.Rotation = numpy.array(params['stereo']['Rotation'])
            self.Translation = numpy.array(params['stereo']['Translation'])
        
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
            -self.Translation
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
            raise FileNotFoundError(f"找不到SGBM配置文件: {self.SGBM_Configuration}")
        
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
        
        # print(f">>>\n正在启动底层相机硬件[Camera ID: {camera_id}]...\n...")
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
            elif params['captureMode'] == 'one image':
                print(">>> 当前输入源为静态图片 <<<")
                self.captureData = cv2.VideoCapture('./staticSource/image/test.jpg')
            elif params['captureMode'] == 'one video':
                print(">>> 当前输入源为mp4媒体文件 <<<")
                self.captureData = cv2.VideoCapture('./staticSource/video/test.mp4')    
            
        print("读取捕捉配置文件成功!\n<<<\n————————————————————")

    def get_split_img(self):
        """
        从相机流中读取最新的一帧画面，并将其从中间切割为左右两张独立的图像
        """
        retval, frame = self.captureData.read()

        if not retval:
            print("未能获取到图像画面！")
            self.bgr_left = None
            self.bgr_right = None
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
        self.disparity = self.stereo.compute(self.rectify_gray_left, self.rectify_gray_right)

    def cpt_xyz(self):
        disparity_float = self.disparity.astype(numpy.float32) / 16.0
        self.xyz = cv2.reprojectImageTo3D(disparity_float, self.Q, handleMissingValues=True)

class YOLOInit():

    def __init__(self):
        self._predict_init()
    
    def _predict_init(self):

        print(">>>\n正在读取yolo配置文件...\n...")

        yolo_configuration = "./config/yolo_params.yaml"
        if not Path(yolo_configuration).exists():
            raise FileNotFoundError(f"找不到yolo配置文件: {yolo_configuration}")
      
        with open(yolo_configuration, 'r', encoding='utf-8') as f:
            params = yaml.safe_load(f)
            self.save = params["save"]
            self.conf = params["conf"]
            self.show = params["show"]
            self.stream = params["stream"]
            self.verbose = params["verbose"]
            self.device = params["device"]
            yolo_weight = Path("./weights") / params['weightName']
            if not Path(yolo_weight).exists():
                raise FileNotFoundError(f"找不到yolo权重文件: {yolo_weight}")
            self.model = YOLO(yolo_weight, task="detect")

        print("yolo配置初始化完毕!\n<<<\n————————————————————")
    
    def predict(self, image):
        self.results = self.model.predict(source = image, 
							    save = self.save, 
							    conf = self.conf, 
							    show = self.show, 
							    device = self.device, 
							    stream = self.stream,
                                verbose = self.verbose)

mouse_points = []
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(mouse_points) < 3:
            mouse_points.append((x, y))
            print(f"已选取点 {len(mouse_points)}: ({x}, {y})")
            if len(mouse_points) == 3:
                print("三个点已选完，开始 PnP 解算...")


def outcome_action(results, xyz_matrix, rectify_bgr_img):
    if not hasattr(outcome_action, "seq_count"):
        outcome_action.seq_count = 1
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        
        # 自动寻找并创建当天的新序号文件夹 (例如: ./outcome/20260331_1)
        run_idx = 1
        while True:
            dir_name = f"./outcome/{date_str}_{run_idx}"
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
                outcome_action.save_dir = dir_name
                break
            run_idx += 1

    save_img = rectify_bgr_img.copy()

    if len(results[0].boxes) > 0:
        for target in results[0].boxes:
            # 1. 算中心点坐标
            x1, y1, x2, y2 = target.xyxy[0]
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # 顺便获取一下类别名称，方便你在终端看出来识别到了什么
            cls_id = int(target.cls[0])
            cls_name = results[0].names[cls_id]

            # 2. 提取该点的深度 Z
            if 0 <= cy < xyz_matrix.shape[0] and 0 <= cx < xyz_matrix.shape[1]:
                depth = xyz_matrix[cy, cx, 2] / 1000.0
            else:
                depth = -1.0

            print(f"深度：{depth:.2f}")
            # 3. 纯终端打印输出
            if 0 < depth < 1000000:  
                print(f"发现目标 [{cls_name}] | 中心像素点: ({cx}, {cy}) | 真实距离: {depth:.2f}m")
            else:
                print(f"发现目标 [{cls_name}] | 中心像素点: ({cx}, {cy}) | 真实距离: N/A (计算失败)")

            cv2.circle(save_img, (cx, cy), 3, (0, 0, 255), -1)

    save_path = f"{outcome_action.save_dir}/{outcome_action.seq_count}.jpg"
    cv2.imwrite(save_path, save_img)

    outcome_action.seq_count += 1    


if __name__ == '__main__':

    cam = StereoCamera()
    yolo = YOLOInit()
    print("\n\n————————————————————\n开始检测\n————————————————————")
    while True:

        loop_start = datetime.datetime.now()

        cam.get_split_img()  # 0.5s
        cam.rectify_images() # 0.15s
        yolo.predict(cam.rectify_bgr_left) # 6s
        cam.cpt_disparity() # 2
        cam.cpt_xyz() # 0.05s
        outcome_action(yolo.results, cam.xyz, cam.rectify_bgr_left)

        loop_end = datetime.datetime.now()
        duration = loop_end - loop_start
        total_seconds = duration.total_seconds()

        hz = 1.0 / total_seconds if total_seconds > 0 else 0
        print(f">>> 单次循环耗时: {total_seconds:.4f}s | 速率: {hz:.2f} Hz")
        print("-" * 30)
        
