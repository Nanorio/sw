#>>> 库导入
import os
import datetime
import yaml
import numpy
from pathlib import Path
import cv2
import math
from ultralytics import YOLO
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

def get_robust_depth(xyz_matrix, x1, y1, x2, y2):
    """区域中位数滤波，提取最稳的 3D 坐标"""
    # 稍微向内收缩一点框 (20%)，防止边缘背景噪点干扰
    margin_x = int((x2 - x1) * 0.2)
    margin_y = int((y2 - y1) * 0.2)
    roi_3d = xyz_matrix[y1+margin_y : y2-margin_y, x1+margin_x : x2-margin_x]
    
    if roi_3d.size == 0:
        return None
        
    # 提取 XYZ 并过滤噪点 (比如要求 Z 必须在 0 到 10000 毫米之间)
    valid_mask = (roi_3d[:, :, 2] > 0) & (roi_3d[:, :, 2] < 10000)
    valid_pts = roi_3d[valid_mask]
    
    # 如果框里有效的 3D 点太少，说明可能是误识别或者严重反光
    if len(valid_pts) < 5:
        return None
        
    # 分别求 X, Y, Z 的中位数，得到一个极其稳定的 3D 坐标
    median_x = numpy.median(valid_pts[:, 0])
    median_y = numpy.median(valid_pts[:, 1])
    median_z = numpy.median(valid_pts[:, 2])
    
    return numpy.array([median_x, median_y, median_z])

def outcome_action(results, xyz_matrix, rectify_bgr_left):
    # 1. 初始化保存目录（利用函数静态属性，只执行一次）
    if not hasattr(outcome_action, "seq_count"):
        outcome_action.seq_count = 1
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        
        run_idx = 1
        while True:
            dir_name = f"./outcome/{date_str}_{run_idx}"
            if not os.path.exists(dir_name):
                os.makedirs(dir_name)
                outcome_action.save_dir = dir_name
                break
            run_idx += 1

    save_img = rectify_bgr_left.copy()
    boxes = results[0].boxes
    
    # === [核心修改] 防反光机制：只要识别数 >= 3，就挑最靠谱的 3 个 ===
    if len(boxes) >= 3:
        # 1. 将所有的框按照 YOLO 的置信度 (conf) 从高到低进行排序
        sorted_boxes = sorted(boxes, key=lambda b: b.conf[0].item(), reverse=True)
        
        # 2. 永远只切取前 3 名“学霸”（最可能是真实的三个灯）
        best_3_boxes = sorted_boxes[:3]
        
        centers_2d_data = [] # 保存数据，不在此处画图
        
        # 遍历这最靠谱的 3 个框
        for target in best_3_boxes:
            x1, y1, x2, y2 = map(int, target.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            
            # 使用中位数滤波获取该灯的 3D 坐标
            pt_3d = get_robust_depth(xyz_matrix, x1, y1, x2, y2)
            
            if pt_3d is not None:
                # [重要修改：先保存数据以便后续识别底部灯，不在此处画圆]
                centers_2d_data.append({'cx': cx, 'cy': cy, 'pt_3d': pt_3d})
                
        # 只有这 3 个最靠谱的灯的三维坐标都提取成功了，才进行解算和保存
        if len(centers_2d_data) == 3:
            # 按 cy (画面Y坐标) 升序排列
            centers_2d_data.sort(key=lambda item: item['cy'])
            
            # === [重要修改] 在倒三角配置中认灯 ===
            # 在倒三角里，cy最大的（排在最后的）是底部的那 1 个灯
            bottom_light_data = centers_2d_data[2] # C点数据
            
            # [重要修改：使用红色标注底部灯的参考点，半径更大，实心]
            cv2.circle(save_img, (bottom_light_data['cx'], bottom_light_data['cy']), 8, (0, 0, 255), -1) # 红色，半径 8
            
            # 前两个是顶部的 2 个灯，按 cx 排序分出左右
            top_lights_data = centers_2d_data[:2]
            top_lights_data.sort(key=lambda item: item['cx'])
            left_light_data = top_lights_data[0]   # 左上角的灯 (A点数据)
            right_light_data = top_lights_data[1]  # 右上角的灯 (B点数据)
            
            # [重要修改：使用绿色标注顶部两个灯的参考点]
            cv2.circle(save_img, (left_light_data['cx'], left_light_data['cy']), 5, (0, 255, 0), -1) # 绿色，半径 5
            cv2.circle(save_img, (right_light_data['cx'], right_light_data['cy']), 5, (0, 255, 0), -1) # 绿色，半径 5
            
            # 提取 3D 点
            A = left_light_data['pt_3d']
            B = right_light_data['pt_3d']
            C = bottom_light_data['pt_3d']
            
            # 目标的绝对距离
            target_distance = (A[2] + B[2] + C[2]) / 3.0
            
            # === 三维向量叉乘求法向量 → 构建正交基 → 提取姿态角 ===
            # 相机坐标系: X-右, Y-下, Z-前 (OpenCV 惯例)
            vec_AB = B - A   # 目标"右"方向（顶部两灯连线）
            vec_AC = C - A   # 目标"下"方向（左上→底部）
            # 法向量 = AB × AC, 代表目标平面朝向 (指向相机)
            normal_vec = numpy.cross(vec_AB, vec_AC) 
            
            # 归一化法向量 → 目标坐标系的 Z 轴
            norm_length = numpy.linalg.norm(normal_vec)
            if norm_length > 0:
                normal_vec = normal_vec / norm_length
                nx, ny, nz = normal_vec
                
                # Yaw (偏航): 法向量在 XZ 平面上的方位角
                # nx/nz → 法向量绕 Y 轴的偏转
                yaw = math.degrees(math.atan2(nx, nz))
                
                # Pitch (俯仰): 法向量与水平面的夹角
                # 前倾时 ny < 0, 取负使 pitch > 0 表示低头
                pitch = math.degrees(math.asin(numpy.clip(-ny, -1.0, 1.0)))
                
                # Roll (横滚): 顶部两灯连线在相机 XY 平面内的转角
                # vec_AB 是立体深度恢复的三维向量, 包含真实 X/Y/Z 分量,
                # 而非 2D 像素坐标。使用 3D 向量的关键优势:
                # 在大俯仰/大偏航角下, 2D 像素投影会缩短, 导致角精度骤降;
                # 而 3D 向量保留完整分量, 始终给出正确的几何角度。
                roll = math.degrees(math.atan2(vec_AB[1], vec_AB[0]))
                
                # 终端输出
                print(f"🎯 锁定目标！距离: {target_distance:.2f} mm")
                print(f"✈️ 姿态 -> 偏航(Yaw): {yaw:.1f}° | 俯仰(Pitch): {pitch:.1f}° | 横滚(Roll): {roll:.1f}°")
                
                # === [新增视觉反馈] 在解算成功的情况下画三角形边，提供视觉参考 ===
                # A -> B 顶边 (绿色)
                cv2.line(save_img, (left_light_data['cx'], left_light_data['cy']), (right_light_data['cx'], right_light_data['cy']), (0, 255, 0), 2)
                # A -> C 左侧斜边 (红色)
                cv2.line(save_img, (left_light_data['cx'], left_light_data['cy']), (bottom_light_data['cx'], bottom_light_data['cy']), (0, 0, 255), 2)
                # B -> C 右侧斜边 (蓝色)
                cv2.line(save_img, (right_light_data['cx'], right_light_data['cy']), (bottom_light_data['cx'], bottom_light_data['cy']), (255, 0, 0), 2)

                # 画 Z 文本
                for item in centers_2d_data:
                     cv2.putText(save_img, f"Z:{item['pt_3d'][2]:.0f}", (item['cx']-20, item['cy']-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # 打水印
                cv2.putText(save_img, f"Dist: {target_distance:.0f}mm", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(save_img, f"Y:{yaw:.1f} P:{pitch:.1f} R:{roll:.1f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # === [核心修改：只有在此处保存图片和自增序号] ===
                # 丢帧时不执行此逻辑
                save_path = f"{outcome_action.save_dir}/{outcome_action.seq_count:04d}.jpg"
                cv2.imwrite(save_path, save_img)
                outcome_action.seq_count += 1
        
        else:
            print("选出的 3 个最佳目标中，有目标深度计算失败 (反光或噪点过大)，不保存图片。")
            
    else:
        print(f"当前只看清了 {len(boxes)} 个灯，无法进行姿态解算，不保存图片。")

if __name__ == '__main__':

    cam = StereoCamera()
    yolo = YOLOInit()
    print("\n\n————————————————————\n开始检测\n————————————————————")
    
    while True:
        loop_start = datetime.datetime.now()

        # [防爆破机制 1]: 没拿到图就直接重新拿，不要往下跑导致崩溃
        if not cam.get_split_img():
            print("相机无画面输入，尝试重新获取...")
            cv2.waitKey(500)
            continue

        try:
            # 严格按照顺序执行处理流水线
            cam.rectify_images() 
            yolo.predict(cam.rectify_bgr_left) 
            cam.cpt_disparity() 
            cam.cpt_xyz() 
            
            # 将数据喂给咱们刚写的全能分析函数
            outcome_action(yolo.results, cam.xyz, cam.rectify_bgr_left)
            
        except Exception as e:
            # [防爆破机制 2]: SGBM 或 remap 哪怕偶尔抽风报错，也能被捕获，不会导致整个程序闪退
            print(f"处理这一帧时发生异常: {e}")
            continue

        loop_end = datetime.datetime.now()
        duration = loop_end - loop_start
        total_seconds = duration.total_seconds()

        hz = 1.0 / total_seconds if total_seconds > 0 else 0
        print(f">>> 单次循环耗时: {total_seconds:.4f}s | 速率: {hz:.2f} Hz")
        print("-" * 50)