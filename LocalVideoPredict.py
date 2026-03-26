# -*- coding: utf-8 -*-
import numpy as np
import cv2
import math
import time
from ultralytics import YOLO  # 导入 YOLO
from pathlib import Path   # 文件路径处理

# -------------------------------------------------------
# 1.定义相机内参
# ---------------------------------------------------------
def Camara_Param_Init():
    # 左相机内参
    left_camera_matrix = np.array([[627.1991, 0.0, 941.6919],
                                [0.0, 624.4099, 538.9668],
                                [0.0, 0.0, 1.0]])

    # 左相机畸变系数:[k1, k2, p1, p2, k3]
    left_distortion = np.array([[0.0542, -0.0450, -0.0023, -0.0046, 0.0]])

    # 右相机内参
    right_camera_matrix = np.array([[630.8504, 0.0, 947.8281],
                                    [0.0, 627.3688, 539.7096],
                                    [0.0, 0.0, 1.0]])

    # 右相机畸变系数:[k1, k2, p1, p2, k3]                                          
    right_distortion = np.array([[0.0560, -0.0449, -0.0022, -0.0045, 0.0]])

    # 旋转矩阵
    R = np.array([[1.0000, 0.00048566, -0.0029],
                [ -0.00049732, 1.0000, -0.0041],
                [ 0.0029, 0.0041, 1.0000]])

    # 平移向量
    Translation = np.array([[61.9624], [-0.8920], [-0.4837]])

    # 单个相机分辨率大小，测试相机最高为1920*1080(640x240@60fps;1280x480@60fps;2560x720@60fps;2560x960@60fps)
    size = (1280, 720)  

    # R1:左摄像机旋转矩阵, P1:左摄像机投影矩阵, Q:重投影矩阵
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, 
                                                                    left_distortion,
                                                                    right_camera_matrix, 
                                                                    right_distortion, 
                                                                    size, 
                                                                    R, 
                                                                    Translation)

    # 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
    left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, 
                                                    left_distortion, 
                                                    R1, 
                                                    P1, 
                                                    size, 
                                                    cv2.CV_16SC2)

    right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, 
                                                        right_distortion, 
                                                        R2, 
                                                        P2, 
                                                        size, 
                                                        cv2.CV_16SC2)

    param_dict = {
        "left_map1": left_map1,
        "left_map2": left_map2,
        "right_map1": right_map1,
        "right_map2": right_map2,
        "Q": Q,
        "size": size
    }
    return param_dict

# -------------------------------------------------------
# 2.yolov8初始化
# ---------------------------------------------------------
def Yolov8_Init():

    # 引用train.py的权重文件，pt是原权重,onnx是通用权重文件，engine是Jetson专用权重文件,tflite是精简权重(适用于arm架构)
    weight ={"pt":"./weights/best.pt", 
		  	 "onnx":"./weights/best.onnx", 
			 "engine":"./weights/best.engine",
			 "tflite":"./weights/best.tflite" 	
		   	}

    Detect_Source = {"Local_Camara":0, 
					 "External_Camera":1, 
					 "Screen":"screen", 
					 "Static_image":"./BePredicted"
					}
	
    Detect_Source_Choice = "Local_Camara"
	
    # 加载训练权重文件
    print("加载YOLO模型中....") 
    try:
        model = YOLO(weight.get("pt"))
        return model
    except:
        print("模型加载失败，请排查原因，进程退出")
        return None

# -------------------------------------------------------
# 3.双目SGBM参数设置
# -------------------------------------------------------
def SGBM_Perem_Init():
    blockSize = 3 # 0~10
    img_channels = 3 # 固定为3

    '''
    stereo = cv2.StereoSGBM_create(minDisparity = 1,
                                   numDisparities = 64,
                                   blockSize = blockSize,
                                   P1 = 8 * img_channels * blockSize * blockSize,
                                   P2 = 32 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff = -1,
                                   preFilterCap = 1,
                                   uniquenessRatio = 10,
                                   speckleWindowSize = 100,
                                   speckleRange = 100,
                                   mode = cv2.STEREO_SGBM_MODE_HH4) # cv2.STEREO_SGBM_MODE_SGBM / cv2.STEREO_SGBM_MODE_HH4 / cv2.STEREO_SGBM_MODE_HH
    '''
    stereo = cv2.StereoSGBM_create(minDisparity = 1,
                                   numDisparities = 64,
                                   blockSize = blockSize,
                                   P1 = 8 * img_channels * blockSize * blockSize,
                                   P2 = 32 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff = 15,
                                   preFilterCap = 63,
                                   uniquenessRatio = 10,
                                   speckleWindowSize = 100,
                                   speckleRange = 10,
                                   mode = cv2.STEREO_SGBM_MODE_HH4) # cv2.STEREO_SGBM_MODE_SGBM / cv2.STEREO_SGBM_MODE_HH4 / cv2.STEREO_SGBM_MODE_HH
    
    return stereo

# -------------------------------------------------------
# 4.摄像头与窗口初始化
# ---------------------------------------------------------   
def Camera_View_Init(param, camera_id=0):
    # 启动相机并捕获视频数据给变量capture,并设置捕捉分辨率
    capture = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    capture.set(3, param["size"][0]*2)  #图像帧像素宽度
    capture.set(4, param["size"][1])    #图像帧像素高度，打开并设置摄像头

    if not capture.isOpened():
        print("无法打开摄像头！")
        return None
    
    return capture

# -------------------------------------------------------
# 5.预测部分
# ---------------------------------------------------------   
if __name__ == "__main__":
    # 1.获取参数字典
    params = Camara_Param_Init()
    size = params["size"]
    Q = params["Q"]
    left_map1 = params["left_map1"]
    left_map2 = params["left_map2"]
    right_map1 = params["right_map1"]
    right_map2 = params["right_map2"]

    # 2.初始化模型
    model = Yolov8_Init()

    # 3.初始化SGBM
    stereo = SGBM_Perem_Init()

    # 本地图像路径
    video_path = "./staticSource/video/test.mp4"

    # 记录日志
    log_filename = "./log/LocalVideoPredict.log"

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print(f"无法打开视频文件，请检查路径: {video_path}")
        exit()
    
    # 5. 进入主循环
    print("---预测开始---")
    circle=1
    while (circle):
        t1 = time.time()    # 用于计算FPS
        print('tag1')

        ret, frame = capture.read()
        if not ret:
            print("视频读取完毕！")
            break

        print('tag2')

        # 双目图像分割成左右2张
        frame1 = frame[0:size[1], 0:size[0]]
        frame2 = frame[0:size[1], size[0]:size[0]*2]  

        print('tag3')

        imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # 将BGR格式转换成灰度图片
        imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
 
        # cv2.remap 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
        # 依据MATLAB测量数据重建无畸变图片

        imgL_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
        imgR_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)  
 
        imageL = cv2.cvtColor(imgL_rectified, cv2.COLOR_GRAY2BGR)  
        imageR = cv2.cvtColor(imgR_rectified, cv2.COLOR_GRAY2BGR)

        disparity = stereo.compute(imgL_rectified, imgR_rectified) # 计算视差

        threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True) * 22  #计算三维坐标数据值

        # 进行yolo推理(源使用矫正后的RGB图像;过滤置信0.25;显示推理过程窗口;使用GPU0加速推理;关闭日志显示)
        results = model.predict(source = frame1, 
							    save = False, 
							    conf = 0.25, 
							    show = False, 
							    device = 'cpu', 
							    stream = True,
                                verbose = False
							    )
        
        # 目标深度输出
        for r in results:
            for box in r.boxes:
                # 拿到目标↖与↘坐标 [x1, y1, x2, y2]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                # 计算目标中心点
                target_center_x, target_center_y = (x1 + x2) // 2, (y1 + y2) // 2
                
                # 获取该中心点的 3D 深度 (Z 轴)
                # 注意矩阵索引是 [行, 列] 对应 [y, x]
                target_point = threeD[target_center_y, target_center_x]
                z_depth = target_point[2] # 获取 Z 坐标

                # 从相机到点的真实直线距离
                distance = math.sqrt(target_point[0]**2 + target_point[1]**2 + target_point[2]**2)
                
                # 转换为米并显示
                if not np.isinf(distance):
                    # 若distance为有效数值
                    display_meter = distance / 1000.0
                else:
                    # 若distance为无穷大,计算失败,返回0
                    display_meter = 0.0

                # 日志记录
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                log_info = f"[{current_time}] 中心点坐标: ({target_center_x}, {target_center_y}) | 边界框: [{x1}, {y1}, {x2}, {y2}] | 深度距离: {display_meter:.2f}m\n"
                with open(log_filename, "a", encoding="utf-8") as file:
                    file.write(log_info)
                print(f"日志已记录 -> {log_info.strip()}")

        # circle=0
    # 结束标志
    print("进程结束")