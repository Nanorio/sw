import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. 物理世界已知条件 (X向右, Y向下, Z向前)
# ==========================================
object_points = np.array([
    [-49.54, -28.6, 0.0],  
    [ 49.54, -28.6, 0.0],  
    [  0.00,  58.0, 0.0]   
], dtype=np.float32)

# ==========================================
# 2. 【核心改进】自定义矩阵欧拉角提取
# ==========================================
def calculate_pose_sqpnp(image_points, camera_matrix, dist_coeffs):
    image_points = np.array(image_points, dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP 
    )
    if not success: return None, None, None, None

    distance_m = np.linalg.norm(tvec) / 1000.0
    rmat, _ = cv2.Rodrigues(rvec)

    # ---------------------------------------------------------
    # 绝对精准的角度提取数学公式 (符合你指定的正负规范)
    # R_inv 就是相机坐标系到世界坐标系的转换矩阵
    # ---------------------------------------------------------
    R_inv = rmat.T  

    # Pitch (俯仰角): 绕 X 轴旋转。
    # R_inv[1, 2] 是相机视线(Z)向下的分量。低头时该分量为正。
    # 你要求低头为负，所以加上负号：-R_inv[1, 2]
    pitch_rad = math.asin(np.clip(-R_inv[1, 2], -1.0, 1.0))
    
    # Yaw (偏航角): 绕 Y 轴旋转。
    # R_inv[0, 2] 是视线偏右的分量。左偏为负。
    yaw_rad = math.atan2(R_inv[0, 2], R_inv[2, 2])
    
    # Roll (滚转角): 绕 Z 轴旋转。
    # R_inv[1, 0] 是相机右侧(X)向下的分量。左倾(右侧翘起)时该分量为负。
    roll_rad = math.atan2(R_inv[1, 0], R_inv[1, 1])

    # 弧度转角度
    pitch = math.degrees(pitch_rad)
    yaw = math.degrees(yaw_rad)
    roll = math.degrees(roll_rad)

    return distance_m, (pitch, yaw, roll), rmat, tvec

# ==========================================
# 3. 核心工具：画“相机视锥体”
# ==========================================
def draw_camera_frustum(ax, cam_pos, R_inv, scale=30):
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
    
    px, py, pz = world_pts[:,0], world_pts[:,2], world_pts[:,1]

    color = 'c'
    for i in range(1, 5):
        ax.plot([px[0], px[i]], [py[0], py[i]], [pz[0], pz[i]], color=color, lw=1.5)
    corners = [1, 2, 3, 4, 1]
    for i in range(4):
        c1, c2 = corners[i], corners[i+1]
        ax.plot([px[c1], px[c2]], [py[c1], py[c2]], [pz[c1], pz[c2]], color=color, lw=1.5)

# ==========================================
# 4. 极简风 3D 可视化
# ==========================================
def visualize_minimalist_hud(rmat, tvec, distance, angles):
    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor('#f0f0f0') 
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("SLAM Pose Monitor", fontsize=14, fontweight='bold')

    # A. 极简世界坐标系
    ax.plot([-20, 20], [0, 0], [0, 0], 'r-', lw=1, alpha=0.5) 
    ax.plot([0, 0], [-20, 20], [0, 0], 'b-', lw=1, alpha=0.5) 
    ax.plot([0, 0], [0, 0], [-20, 20], 'g-', lw=1, alpha=0.5) 
    
    # B. 画出目标架子
    xw, yw, zw = object_points[:, 0], object_points[:, 1], object_points[:, 2]
    ax.scatter(xw, zw, yw, color='magenta', s=50) 
    for start, end in [(0, 1), (1, 2), (2, 0)]:
        ax.plot([xw[start], xw[end]], [zw[start], zw[end]], [yw[start], yw[end]], 'm-', lw=2)

    # C. 解算相机位置并画出“视锥体”
    R_inv = rmat.T  
    cam_pos = -np.dot(R_inv, tvec).squeeze()
    
    draw_camera_frustum(ax, cam_pos, R_inv, scale=25)
    ax.scatter(cam_pos[0], cam_pos[2], cam_pos[1], color='black', s=40)

    # D. 单根核心视线
    target_center = np.mean(object_points, axis=0)
    ax.plot([cam_pos[0], target_center[0]], 
            [cam_pos[2], target_center[2]], 
            [cam_pos[1], target_center[1]], 
            color='orange', linestyle='-.', lw=1.5)

    # E. 【格式化对齐的大屏 2D 数据面板】
    pitch, yaw, roll = angles
    hud_text = (
        f"TARGET DATA\n"
        f"------------------\n"
        f"Dist:  {distance:6.2f} m\n"
        # 强制带上正负号，更符合工程习惯
        f"Yaw:   {yaw:+6.1f}°\n"
        f"Pitch: {pitch:+6.1f}°\n"
        f"Roll:  {roll:+6.1f}°"
    )
    ax.text2D(0.05, 0.75, hud_text, transform=ax.transAxes, 
              fontsize=12, family='monospace',
              bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8))

    # F. 视角调整
    all_x = np.append(xw, cam_pos[0])
    all_y = np.append(zw, cam_pos[2]) 
    all_z = np.append(yw, cam_pos[1]) 
    
    max_range = np.array([all_x.max()-all_x.min(), all_y.max()-all_y.min(), all_z.max()-all_z.min()]).max() / 2.0
    mid_x, mid_y, mid_z = np.mean(all_x), np.mean(all_y), np.mean(all_z)
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    ax.invert_zaxis() 
    
    # ---------------------------------------------------------
    # 【改动】强迫症狂喜：初始画面绝对正视 XOY 平面！
    # elev=0 : 视线与地平线平行，不俯瞰也不仰视
    # azim=-90 : 从 Z 轴前方直勾勾地盯着目标看
    # ---------------------------------------------------------
    ax.view_init(elev=0, azim=-90)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    plt.show()

# ==========================================
# 5. 测试运行
# ==========================================
if __name__ == "__main__":
    cam_matrix = np.array([[627.1991, 0.0, 941.6919],
                           [0.0, 624.4099, 538.9668],
                           [0.0, 0.0, 1.0]], dtype=np.float32)
    dist_coef = np.array([[0.0542, -0.0450, -0.0023, -0.0046, 0.0]], dtype=np.float32)

    # 模拟 YOLO 识别到了符合特定姿态的 3 个像素点
    simulated_2d_points = [
        [800, 400],  
        [1100, 420], 
        [960, 700]    
    ]

    dist, ang, rmat, tvec = calculate_pose_sqpnp(simulated_2d_points, cam_matrix, dist_coef)
    
    if dist is not None:
        # 添加的命令行显示内容
        print("\n==========================================")
        print("解算成功！")
        print(f"直线距离: {dist:.2f} 米")
        print(f"姿态角度: Pitch(俯仰)={ang[0]:+.1f}°, Yaw(偏航)={ang[1]:+.1f}°, Roll(滚转)={ang[2]:+.1f}°")
        print("==========================================\n")
        
        # 启动 3D 可视化面板
        visualize_minimalist_hud(rmat, tvec, dist, ang)
    else:
        print("解算失败，请检查输入的像素点或相机内参。")