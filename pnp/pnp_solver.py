# pnp_solver.py
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import yaml

OBJECT_POINTS_MM = None

def _load_object_points():
    """从配置文件读取目标物体三维坐标 (单位: mm)"""
    global OBJECT_POINTS_MM
    if OBJECT_POINTS_MM is not None:
        return OBJECT_POINTS_MM
    config_path = os.path.join(os.path.dirname(__file__), "..", "cap", "config", "object_points.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"找不到目标点配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    OBJECT_POINTS_MM = np.array(data["points"], dtype=np.float32)
    return OBJECT_POINTS_MM

OBJECT_POINTS_MM = _load_object_points()

def calculate_pose_sqpnp(image_points, camera_matrix, dist_coeffs):
    """
    使用 solvePnP 计算相机位姿
    :param image_points: 2D 点集，形状 (3,2) 或 (N,2)
    :param camera_matrix: 相机内参矩阵
    :param dist_coeffs: 畸变系数
    :return: (distance_m, (pitch, yaw, roll), rmat, tvec) 或 (None, None, None, None)
    """
    image_points = np.array(image_points, dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(
        OBJECT_POINTS_MM, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP
    )
    if not success:
        return None, None, None, None

    distance_m = np.linalg.norm(tvec) / 1000.0
    rmat, _ = cv2.Rodrigues(rvec)

    # 从旋转矩阵提取欧拉角（符合你的符号习惯）
    R_inv = rmat.T
    pitch_rad = math.asin(np.clip(-R_inv[1, 2], -1.0, 1.0))
    yaw_rad = math.atan2(R_inv[0, 2], R_inv[2, 2])
    roll_rad = math.atan2(R_inv[1, 0], R_inv[1, 1])

    pitch = math.degrees(pitch_rad)
    yaw = math.degrees(yaw_rad)
    roll = math.degrees(roll_rad)

    return distance_m, (pitch, yaw, roll), rmat, tvec

def draw_camera_frustum(ax, cam_pos, R_inv, scale=30):
    """在3D坐标系中绘制相机视锥体"""
    local_pts = np.array([
        [0, 0, 0],
        [scale, scale, scale*2],
        [-scale, scale, scale*2],
        [-scale, -scale, scale*2],
        [scale, -scale, scale*2]
    ])
    world_pts = []
    for p in local_pts:
        wp = np.dot(R_inv, p) + cam_pos
        world_pts.append(wp)
    world_pts = np.array(world_pts)
    px, py, pz = world_pts[:,0], world_pts[:,2], world_pts[:,1]  # 转换坐标轴
    color = 'c'
    for i in range(1, 5):
        ax.plot([px[0], px[i]], [py[0], py[i]], [pz[0], pz[i]], color=color, lw=1.5)
    corners = [1, 2, 3, 4, 1]
    for i in range(4):
        c1, c2 = corners[i], corners[i+1]
        ax.plot([px[c1], px[c2]], [py[c1], py[c2]], [pz[c1], pz[c2]], color=color, lw=1.5)

def visualize_minimalist_hud(rmat, tvec, distance, angles):
    """显示3D姿态可视化窗口"""
    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor('#f0f0f0')
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Camera Pose Estimation", fontsize=14, fontweight='bold')

    # 世界坐标系轴线
    ax.plot([-20, 20], [0, 0], [0, 0], 'r-', lw=1, alpha=0.5)
    ax.plot([0, 0], [-20, 20], [0, 0], 'b-', lw=1, alpha=0.5)
    ax.plot([0, 0], [0, 0], [-20, 20], 'g-', lw=1, alpha=0.5)

    # 目标物体三点
    xw, yw, zw = OBJECT_POINTS_MM[:,0], OBJECT_POINTS_MM[:,1], OBJECT_POINTS_MM[:,2]
    ax.scatter(xw, zw, yw, color='magenta', s=50)
    for start, end in [(0,1), (1,2), (2,0)]:
        ax.plot([xw[start], xw[end]], [zw[start], zw[end]], [yw[start], yw[end]], 'm-', lw=2)

    # 相机位置和视锥体
    R_inv = rmat.T
    cam_pos = -np.dot(R_inv, tvec).squeeze()
    draw_camera_frustum(ax, cam_pos, R_inv, scale=25)
    ax.scatter(cam_pos[0], cam_pos[2], cam_pos[1], color='black', s=40)

    # 中心视线
    target_center = np.mean(OBJECT_POINTS_MM, axis=0)
    ax.plot([cam_pos[0], target_center[0]],
            [cam_pos[2], target_center[2]],
            [cam_pos[1], target_center[1]],
            color='orange', linestyle='-.', lw=1.5)

    # 数据面板
    pitch, yaw, roll = angles
    hud_text = (
        f"TARGET DATA\n"
        f"------------------\n"
        f"Dist:  {distance:6.2f} m\n"
        f"Yaw:   {yaw:+6.1f}°\n"
        f"Pitch: {pitch:+6.1f}°\n"
        f"Roll:  {roll:+6.1f}°"
    )
    ax.text2D(0.05, 0.75, hud_text, transform=ax.transAxes,
              fontsize=12, family='monospace',
              bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8))

    # 调整坐标轴范围
    all_x = np.append(xw, cam_pos[0])
    all_y = np.append(zw, cam_pos[2])
    all_z = np.append(yw, cam_pos[1])
    max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 2.0
    mid_x, mid_y, mid_z = np.mean(all_x), np.mean(all_y), np.mean(all_z)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    ax.invert_zaxis()
    ax.view_init(elev=0, azim=-90)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    plt.show()

def solve_and_visualize(image_points, camera_matrix, dist_coeffs):
    """
    供外部调用的统一接口
    :param image_points: 2D 点集，例如 [(x1,y1), (x2,y2), (x3,y3)]
    :param camera_matrix: 内参矩阵
    :param dist_coeffs: 畸变系数
    """
    dist, angles, rmat, tvec = calculate_pose_sqpnp(image_points, camera_matrix, dist_coeffs)
    if dist is not None:
        print("\n=== PnP 解算成功 ===")
        print(f"距离: {dist:.2f} m")
        print(f"俯仰: {angles[0]:+.1f}°, 偏航: {angles[1]:+.1f}°, 滚转: {angles[2]:+.1f}°")
        visualize_minimalist_hud(rmat, tvec, dist, angles)
    else:
        print("PnP 解算失败，请检查输入点或内参")
