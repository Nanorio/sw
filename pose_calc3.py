import cv2
import numpy
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import math

# ==========================================
# 0. 绘制 3D 箭头的辅助类
# ==========================================
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return numpy.min(zs)

# ==========================================
# 1. 物理世界的已知条件 (世界坐标系中心定在架子)
# 严格遵守：X向右，Y向下，Z向前
# ==========================================
object_points = numpy.array([
    [-49.54, -28.6, 0.0],  # 0: 左上角
    [ 49.54, -28.6, 0.0],  # 1: 右上角
    [  0.00,  58.0, 0.0]   # 2: 最下方
], dtype=numpy.float32)

# ==========================================
# 2. PnP 姿态解算
# ==========================================
def calculate_pose_sqpnp(image_points, camera_matrix, dist_coeffs):
    image_points = numpy.array(image_points, dtype=numpy.float32)
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP 
    )
    if not success: return None, None, None, None

    distance_m = numpy.linalg.norm(tvec) / 1000.0
    rmat, _ = cv2.Rodrigues(rvec)

    proj_matrix = numpy.hstack((rmat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
    pitch, yaw, roll = euler_angles.flatten()

    return distance_m, (pitch, yaw, roll), rmat, tvec

# ==========================================
# 3. 【全新】工程图纸级圆弧标注生成器
# ==========================================
def draw_blueprint_arc(ax, center, vec, ref_vec, radius, color, label):
    """画出基准线、投影线，并用圆弧连接，完全符合图纸规范"""
    norm_vec = numpy.linalg.norm(vec)
    norm_ref = numpy.linalg.norm(ref_vec)
    if norm_vec < 1e-5 or norm_ref < 1e-5: return

    v1 = ref_vec / norm_ref
    v2 = vec / norm_vec

    # 计算夹角
    dot_p = numpy.clip(numpy.dot(v1, v2), -1.0, 1.0)
    angle_rad = math.acos(dot_p)
    if angle_rad < 1e-3: return # 角度太小不画

    # 找旋转轴
    normal = numpy.cross(v1, v2)
    if numpy.linalg.norm(normal) < 1e-5: return
    normal = normal / numpy.linalg.norm(normal)

    # 画基准虚线 (灰色) -> 注意：向绘图传递坐标时，全部互换 Y 和 Z！
    ref_end = center + v1 * radius * 1.5
    ax.plot([center[0], ref_end[0]], [center[2], ref_end[2]], [center[1], ref_end[1]], color='gray', linestyle=':', lw=1.5)

    # 画实际轴虚线 (彩色)
    vec_end = center + v2 * radius * 1.5
    ax.plot([center[0], vec_end[0]], [center[2], vec_end[2]], [center[1], vec_end[1]], color=color, linestyle='--', lw=1.5)

    # 生成平滑圆弧点
    t_vals = numpy.linspace(0, 1, 30)
    arc_x, arc_y, arc_z = [], [], []
    for t in t_vals:
        ang = t * angle_rad
        # 罗德里格斯公式旋转生成圆弧
        v_rot = v1 * math.cos(ang) + numpy.cross(normal, v1) * math.sin(ang) + normal * numpy.dot(normal, v1) * (1 - math.cos(ang))
        p = center + v_rot * radius
        arc_x.append(p[0])
        arc_y.append(p[1])
        arc_z.append(p[2])
    
    # 画圆弧 (互换 Y 和 Z)
    ax.plot(arc_x, arc_z, arc_y, color=color, lw=2)

    # 在圆弧外侧写上文字
    mid_ang = angle_rad / 2
    v_mid = v1 * math.cos(mid_ang) + numpy.cross(normal, v1) * math.sin(mid_ang) + normal * numpy.dot(normal, v1) * (1 - math.cos(mid_ang))
    label_pos = center + v_mid * (radius * 1.8)
    ax.text(label_pos[0], label_pos[2], label_pos[1], label, color=color, fontweight='bold', fontsize=11)

# ==========================================
# 4. 3D 可视化：绝对 Z 轴水平 + 投影圆弧 HUD
# ==========================================
def visualize_blueprint_hud(rmat, tvec, distance, angles):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    plt.title(f"Engineering Blueprint 3D | Dist: {distance:.2f}m", fontsize=14, fontweight='bold')

    # 【黑魔法核心】: 所有的绘图命令，原本是 (X, Y, Z)，现在全传 (X, Z, Y)
    # 这样能逼迫 Matplotlib 把真正的 Z 轴画成水平向前的！

    # ---------------------------------------------------
    # 步骤 A: 绘制世界坐标系 (原点定在架子中心)
    # ---------------------------------------------------
    w_len = 100
    # World X (红) -> 传 (X, Z, Y)
    ax.add_artist(Arrow3D([0, w_len], [0, 0], [0, 0], mutation_scale=15, lw=3, arrowstyle="-|>", color="red"))
    ax.text(w_len+5, 0, 0, 'World X (Right)', color='red')
    # World Y (绿) -> 传 (0, 0, w_len)
    ax.add_artist(Arrow3D([0, 0], [0, 0], [0, w_len], mutation_scale=15, lw=3, arrowstyle="-|>", color="green"))
    ax.text(0, 0, w_len+5, 'World Y (Down)', color='green')
    # World Z (蓝) -> 传 (0, w_len, 0)
    ax.add_artist(Arrow3D([0, 0], [0, w_len], [0, 0], mutation_scale=15, lw=3, arrowstyle="-|>", color="blue"))
    ax.text(0, w_len+5, 0, 'World Z (Forward)', color='blue')

    # ---------------------------------------------------
    # 步骤 B: 画出目标架子
    # ---------------------------------------------------
    xw, yw, zw = object_points[:, 0], object_points[:, 1], object_points[:, 2]
    # 传 (xw, zw, yw)
    ax.scatter(xw, zw, yw, color='magenta', s=80)
    for start, end in [(0, 1), (1, 2), (2, 0)]:
        ax.plot([xw[start], xw[end]], [zw[start], zw[end]], [yw[start], yw[end]], 'm-', lw=2)
    ax.scatter(0, 0, 0, color='black', marker='+', s=150)

    # ---------------------------------------------------
    # 步骤 C: 计算相机位置
    # ---------------------------------------------------
    R_inv = rmat.T  
    cam_pos = -numpy.dot(R_inv, tvec).squeeze()
    cx, cy, cz = cam_pos
    ax.scatter(cx, cz, cy, color='black', s=100) # 画相机

    # ---------------------------------------------------
    # 步骤 D: 相机局部坐标系 & 投影图纸级圆弧！
    # ---------------------------------------------------
    cam_axis_len = 50 
    cam_x_dir = numpy.dot(R_inv, numpy.array([1, 0, 0]))
    cam_y_dir = numpy.dot(R_inv, numpy.array([0, 1, 0]))
    cam_z_dir = numpy.dot(R_inv, numpy.array([0, 0, 1]))

    # 画相机 3 根实心短轴
    ax.add_artist(Arrow3D([cx, cx+cam_x_dir[0]*cam_axis_len], [cz, cz+cam_x_dir[2]*cam_axis_len], [cy, cy+cam_x_dir[1]*cam_axis_len], mutation_scale=10, lw=2, color="red", arrowstyle="-|>"))
    ax.add_artist(Arrow3D([cx, cx+cam_y_dir[0]*cam_axis_len], [cz, cz+cam_y_dir[2]*cam_axis_len], [cy, cy+cam_y_dir[1]*cam_axis_len], mutation_scale=10, lw=2, color="green", arrowstyle="-|>"))
    ax.add_artist(Arrow3D([cx, cx+cam_z_dir[0]*cam_axis_len], [cz, cz+cam_z_dir[2]*cam_axis_len], [cy, cy+cam_z_dir[1]*cam_axis_len], mutation_scale=10, lw=2, color="blue", arrowstyle="-|>"))

    pitch, yaw, roll = angles
    arc_rad = 35

    # 1. 偏航 (Yaw) - XZ平面投影 (Top View)
    # 比较: 世界 Z 轴与相机 Z 轴在 XZ 面上的投影
    yaw_ref = numpy.array([0, 0, 1])
    yaw_target = numpy.array([cam_z_dir[0], 0, cam_z_dir[2]])
    draw_blueprint_arc(ax, cam_pos, yaw_target, yaw_ref, arc_rad, 'green', f"Yaw: {yaw:.1f}°")

    # 2. 俯仰 (Pitch) - YZ平面投影 (Side View)
    # 比较: 世界 Z 轴与相机 Z 轴在 YZ 面上的投影
    pitch_ref = numpy.array([0, 0, 1])
    pitch_target = numpy.array([0, cam_z_dir[1], cam_z_dir[2]])
    draw_blueprint_arc(ax, cam_pos, pitch_target, pitch_ref, arc_rad, 'red', f"Pitch: {pitch:.1f}°")

    # 3. 滚转 (Roll) - XY平面投影 (Front View)
    # 比较: 世界 X 轴与相机 X 轴在 XY 面上的投影
    roll_ref = numpy.array([1, 0, 0])
    roll_target = numpy.array([cam_x_dir[0], cam_x_dir[1], 0])
    draw_blueprint_arc(ax, cam_pos, roll_target, roll_ref, arc_rad, 'blue', f"Roll: {roll:.1f}°")

    # ---------------------------------------------------
    # 步骤 E: 连接视线 (虚线)
    # ---------------------------------------------------
    for i in range(3):
        ax.plot([cx, xw[i]], [cz, zw[i]], [cy, yw[i]], 'k--', lw=1, alpha=0.3)

    # ---------------------------------------------------
    # 步骤 F: 调整画框，锁定真正水平的 Z 轴！
    # ---------------------------------------------------
    # 注意因为互换了，所以纵深范围其实是 all_z 也就是原来的 World Y
    all_x = numpy.append(xw, cx)
    all_y = numpy.append(zw, cz) # 绘图的 y 是真正的 z
    all_z = numpy.append(yw, cy) # 绘图的 z 是真正的 y
    
    max_range = numpy.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 2.0
    mid_x = (all_x.max()+all_x.min()) * 0.5
    mid_y = (all_y.max()+all_y.min()) * 0.5
    mid_z = (all_z.max()+all_z.min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # 设置标签名称
    ax.set_xlabel('World X (Right)')
    ax.set_ylabel('World Z (Forward)')
    ax.set_zlabel('World Y (Down)')
    
    # 【神来之笔】：反转绘图Z轴！
    # 因为绘图Z轴绑定的是 World Y，反转后，World Y 就是正儿八经的向下了！
    ax.invert_zaxis() 

    # 设定最佳观测视角：稍微斜侧一点看
    ax.view_init(elev=20, azim=-45)

    plt.show()

# ==========================================
# 5. 测试运行
# ==========================================
if __name__ == "__main__":
    cam_matrix = numpy.array([[627.1991, 0.0, 941.6919],
                              [0.0, 624.4099, 538.9668],
                              [0.0, 0.0, 1.0]], dtype=numpy.float32)
    dist_coef = numpy.array([[0.0542, -0.0450, -0.0023, -0.0046, 0.0]], dtype=numpy.float32)

    simulated_2d_points = [
        [800, 400],  
        [1100, 420], 
        [960, 700]    
    ]

    dist, ang, rmat, tvec = calculate_pose_sqpnp(simulated_2d_points, cam_matrix, dist_coef)
    
    if dist is not None:
        visualize_blueprint_hud(rmat, tvec, dist, ang)