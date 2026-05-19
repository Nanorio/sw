import cv2
import numpy

# 1. 物理世界的已知条件 (3D Object Points)[X, Y, Z], 单位为mm
object_points = numpy.array([
    [-49.54,-28.6, 0.0],  # 0: 左上角顶点
    [ 49.54,-28.6, 0.0],  # 1: 右上角顶点
    [  0.00, 58.0, 0.0]   # 2: 最下方顶点 (蓝灯可能在这)
], dtype=numpy.float32)

# ==========================================
# 2. PnP 姿态解算函数
# ==========================================
def calculate_pose_p3p(image_points, camera_matrix, dist_coeffs):
    """
    输入:
        image_points: YOLO-Pose或传统视觉提取出的 3个顶点的像素坐标 (形状: 3x2)
        camera_matrix: 左相机内参矩阵 (代码里现成的 left_camera_matrix)
        dist_coeffs: 畸变系数 (代码里现成的 left_distortion)
    输出:
        distance: 距离 (米)
        angles: (Pitch, Yaw, Roll) 姿态角 (度)
    """
    # 强制将传入的像素点列表转为 float32 类型的 numpy 数组
    image_points = numpy.array(image_points, dtype=numpy.float32)

    # 使用 cv2.SOLVEPNP_P3P 标志位专门处理 3 个点的情况
    success, rvec, tvec = cv2.solvePnP(
        object_points, 
        image_points, 
        camera_matrix, 
        dist_coeffs, 
        flags=cv2.SOLVEPNP_SQPNP 
    )

    if not success:
        return None, None

    # 1. 获取距离 (tvec 的单位与 object_points 一致，即 mm)
    # 除以 1000 换算成米
    distance = numpy.linalg.norm(tvec) / 1000.0

    # 2. 获取姿态角 (将 rvec 旋转向量转换为直观的 Pitch/Yaw/Roll)
    rmat, _ = cv2.Rodrigues(rvec)

    # 拼合旋转矩阵和平移矩阵
    proj_matrix = numpy.hstack((rmat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)

    # 拍平数组
    pitch, yaw, roll = euler_angles.flatten()

    return distance, (pitch, yaw, roll)

# ==========================================
# 3. 模拟运行测试 
# ==========================================
if __name__ == "__main__":
    cam_matrix = numpy.array([[627.1991, 0.0, 941.6919],
                              [0.0, 624.4099, 538.9668],
                              [0.0, 0.0, 1.0]], dtype=numpy.float32)
    dist_coef = numpy.array([[0.0542, -0.0450, -0.0023, -0.0046, 0.0]], dtype=numpy.float32)

    # 模拟 3 个点在屏幕上的像素坐标 (u, v)
    simulated_2d_points = [
        [800, 400],  # 左上
        [1100, 420], # 右上
        [960, 700]   # 下方
    ]

    dist, angles = calculate_pose_p3p(simulated_2d_points, cam_matrix, dist_coef)
    
    if dist:
        print(f"解算成功！")
        print(f"直线距离: {dist:.2f} 米")
        print(f"姿态角度: Pitch(俯仰)={angles[0]:.1f}°, Yaw(偏航)={angles[1]:.1f}°, Roll(滚转)={angles[2]:.1f}°")