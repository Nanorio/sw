import cv2
import numpy
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# ==========================================
# 0. 绘制 3D 箭头的辅助类 (为了画出漂亮的 XYZ 轴)
# ==========================================
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        # 返回 z 轴的最小值，帮助 matplotlib 计算图层的前后遮挡关系
        return numpy.min(zs)

# ==========================================
# 1. 物理世界的已知条件 (3D Object Points)[X, Y, Z], 单位为mm
# ==========================================
# 注意：这里已经应用了 Y轴向下的修正 (顺应 OpenCV)
object_points = numpy.array([
    [-49.54, -28.6, 0.0],  # 0: 左上角顶点
    [ 49.54, -28.6, 0.0],  # 1: 右上角顶点
    [  0.00,  58.0, 0.0]   # 2: 最下方顶点 (蓝灯可能在这)
], dtype=numpy.float32)

# ==========================================
# 2. PnP 姿态解算核心函数
# ==========================================
def calculate_pose_sqpnp(image_points, camera_matrix, dist_coeffs):
    # 强转数据格式
    image_points = numpy.array(image_points, dtype=numpy.float32)

    # 调用 SQPNP 算法处理 3 个点
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs, 
        flags=cv2.SOLVEPNP_SQPNP 
    )

    if not success:
        return None, None, None, None

    # 获取直线距离 (米)
    distance_mm = numpy.linalg.norm(tvec)
    distance_m = distance_mm / 1000.0

    # 获取旋转矩阵 (3x3)
    rmat, _ = cv2.Rodrigues(rvec)

    # 获取欧拉角 (Pitch, Yaw, Roll) 用于显示
    proj_matrix = numpy.hstack((rmat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
    pitch, yaw, roll = euler_angles.flatten()

    # 返回可视化需要的所有数据：旋转矩阵，平移向量，和角度
    return distance_m, (pitch, yaw, roll), rmat, tvec

# ==========================================
# 3. 【核心改进】3D 监控器可视化函数
# ==========================================
def visualize_3d_setup(rmat, tvec, distance, angles):
    # 初始化 Matplotlib 3D 绘图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    plt.title(f"3D Monitor | Dist: {distance:.2f}m | P,Y,R: ({angles[0]:.1f}, {angles[1]:.1f}, {angles[2]:.1f})°")

    # ---------------------------------------------------
    # 步骤 A: 绘制绝对坐标系 (原点定在相机)
    # 定义：Y轴向下为正，Z轴向前为正 (OpenCV 标准)
    # ---------------------------------------------------
    axis_len = 100 # 轴的长度 (mm)
    # X轴 (红)
    ax.add_artist(Arrow3D([0, axis_len], [0, 0], [0, 0], mutation_scale=20, lw=3, arrowstyle="-|>", color="r"))
    ax.text(axis_len+5, 0, 0, 'W-X', color='r')
    # Y轴 (绿，向下)
    ax.add_artist(Arrow3D([0, 0], [0, axis_len], [0, 0], mutation_scale=20, lw=3, arrowstyle="-|>", color="g"))
    ax.text(0, axis_len+5, 0, 'W-Y', color='g')
    # Z轴 (蓝，向前)
    ax.add_artist(Arrow3D([0, 0], [0, 0], [0, axis_len], mutation_scale=20, lw=3, arrowstyle="-|>", color="b"))
    ax.text(0, 0, axis_len+5, 'W-Z', color='b')
    
    # 标注相机位置 (原点)
    ax.scatter(0, 0, 0, color='k', s=100, label='Camera (Origin)')
    ax.text(5, 5, 5, "Camera (0,0,0)", color='k')

    # ---------------------------------------------------
    # 步骤 B: 算出并绘制倒三角架子的 3D 顶点
    # 公式: P_cam = Rmat * P_obj + Tvec
    # ---------------------------------------------------
    points_in_world = []
    for p_obj in object_points:
        # 矩阵乘法 Tvec 的单位是 mm，所以算出的点也是 mm
        p_cam = numpy.dot(rmat, p_obj) + tvec.squeeze()
        points_in_world.append(p_cam)
    
    points_in_world = numpy.array(points_in_world)
    
    # 提取 XYZ 用于画图
    xw = points_in_world[:, 0]
    yw = points_in_world[:, 1]
    zw = points_in_world[:, 2]

    # 绘制顶点 (散点图)
    ax.scatter(xw, yw, zw, color='magenta', s=80, label='Object Points')
    # 标注 3 个角 (0, 1, 2)
    for i in range(3):
        ax.text(xw[i]+5, yw[i]+5, zw[i]+5, f"P{i}", color='magenta')

    # ---------------------------------------------------
    # 步骤 C: 连接 3 个顶点形成“倒三角形”
    # ---------------------------------------------------
    triangle_edges = [(0, 1), (1, 2), (2, 0)]
    for start, end in triangle_edges:
        ax.plot([xw[start], xw[end]], [yw[start], yw[end]], [zw[start], zw[end]], 'm-', lw=2)

    # ---------------------------------------------------
    # 步骤 D: 连接相机 (原点) 与 3 个顶点
    # ---------------------------------------------------
    for i in range(3):
        ax.plot([0, xw[i]], [0, yw[i]], [0, zw[i]], 'k--', lw=1, alpha=0.5)

    # ---------------------------------------------------
    # 步骤 E: 设定坐标系范围和标签 (关键：Matplotlib默认Y向上，我们需要手动反转视口)
    # ---------------------------------------------------
    max_range = numpy.array([xw.max()-xw.min(), yw.max()-yw.min(), zw.max()-zw.min()]).max() / 2.0
    mid_x = (xw.max()+xw.min()) * 0.5
    mid_y = (yw.max()+yw.min()) * 0.5
    mid_z = (zw.max()+zw.min()) * 0.5
    # 设置相等的轴比例 (为了看起来不失真)
    # 注意：Z轴必须留出空间包含原点(相机)
    z_min = min(0, zw.min()) - 20
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(z_min, mid_z + max_range)

    ax.set_xlabel('W-X')
    ax.set_ylabel('W-Y')
    ax.set_zlabel('W-Z')
    
    # 手动反转 Y 轴，在图形显示上强制“Y轴向下”
    ax.invert_yaxis()

    plt.legend()
    print("\n可视化窗口已弹出，请手动旋转视角查看 (关闭窗口程序结束)。")
    plt.show()

# ==========================================
# 4. 模拟运行与测试 (Ctrl+C 退出)
# ==========================================
if __name__ == "__main__":
    # 模拟相机内参
    cam_matrix = numpy.array([[627.1991, 0.0, 941.6919],
                              [0.0, 624.4099, 538.9668],
                              [0.0, 0.0, 1.0]], dtype=numpy.float32)
    dist_coef = numpy.array([[0.0542, -0.0450, -0.0023, -0.0046, 0.0]], dtype=numpy.float32)

    # 模拟 YOLO-Pose 识别到了符合特定姿态的 3 个像素点
    simulated_2d_points = [
        [800, 400],  # 左上
        [1100, 420], # 右上
        [960, 700]   # 最下 (蓝灯)
    ]

    print("正在解算姿态并在 3D 监控器中还原画面...")
    # 1. 解算
    dist, ang, rmat, tvec = calculate_pose_sqpnp(simulated_2d_points, cam_matrix, dist_coef)
    
    # 2. 只有解算成功才可视化
    if dist is not None:
        print(f"\n解算成功！直线距离: {dist:.2f} 米")
        print(f"姿态角度: Pitch={ang[0]:.1f}°, Yaw={ang[1]:.1f}°, Roll={ang[2]:.1f}°")
        
        # 调用改进的可视化功能
        visualize_3d_setup(rmat, tvec, dist, ang)