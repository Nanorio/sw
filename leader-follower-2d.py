import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import random

class UnicycleAgent:
    """简化的 UUV (差速运动学模型)"""
    def __init__(self, id, x, y, theta):
        self.id = id
        self.x = x
        self.y = y
        self.theta = theta
        self.history_x = [x]
        self.history_y = [y]

    def update_state(self, v, w, dt=0.1):
        """更新位置与姿态，并记录轨迹"""
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += w * dt
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi
        
        self.history_x.append(self.x)
        self.history_y.append(self.y)
        # 限制轨迹长度，保持画面清爽
        if len(self.history_x) > 60:
            self.history_x.pop(0)
            self.history_y.pop(0)

def get_target_position(leader, offset_x, offset_y): # 返回期望位置x与y
    """局部坐标转换全局：通过领航者的绝对位置算出跟随者在全局水池中的期望坐标"""
    target_x = leader.x + offset_x * math.cos(leader.theta) - offset_y * math.sin(leader.theta)
    target_y = leader.y + offset_x * math.sin(leader.theta) + offset_y * math.cos(leader.theta)
    return target_x, target_y

def proportional_controller(agent, target_x, target_y): # agent即指定对象当前状态，targetx/y目标位置x与y，返回线速度与角速度
    """PID 控制器 (比例控制)，输出线速度和角速度"""
    error_x = target_x - agent.x
    error_y = target_y - agent.y
    distance_error = math.hypot(error_x, error_y) # 勾股定理，距离偏差
    
    target_theta = math.atan2(error_y, error_x)
    theta_error = target_theta - agent.theta
    theta_error = (theta_error + math.pi) % (2 * math.pi) - math.pi  # 角度偏差
    
    Kp_v = 1.5  # 稍微调高了跟随响应速度
    Kp_w = 5.0
    
    # 距离极小时刹车，防止由于惯性鬼畜抖动
    if distance_error < 1:
        v = 0
    else:
        v = Kp_v * distance_error
        
    w = Kp_w * theta_error
    
    # 模拟真实电机的物理极限 (最大前进速度与最大转向速度)
    v = np.clip(v, 0, 3.0)
    w = np.clip(w, -2.5, 2.5)
    return v, w

if __name__ == '__main__':
    POOL_SIZE = 30.0
    SPAWN_RANGE = 25.0

    # 初始化编队成员
    leader = UnicycleAgent(
        "Leader", 
        x = np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE), 
        y = np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE), 
        theta = np.random.uniform(-math.pi, math.pi)
    )

    follower_1 = UnicycleAgent(
        "F1", 
        x = np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE), 
        y = np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE), 
        theta = np.random.uniform(-math.pi, math.pi)
    )

    follower_2 = UnicycleAgent(
        "F2", 
        x = np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE), 
        y = np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE), 
        theta = np.random.uniform(-math.pi, math.pi)
    )

    # 初始位置固定
    # leader = UnicycleAgent("Leader",  0, 0, 0)
    # follower_1 = UnicycleAgent("F1", -2, 3, 0) 
    # follower_2 = UnicycleAgent("F2", -2, -3, 0)

    # 设定 V字阵 偏移量 (左后方和右后方)
    F1_offset = (-3.0, -2.5) 
    F2_offset = (-3.0, 2.5)

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 9))

    dt = 0.1
    time_steps = 4000 # 跑足够长的时间看触壁反弹
    
    for t in range(time_steps):
        # ==========================================
        # 1. 领航者 (Leader) 暴力边界限制逻辑
        # ==========================================
        margin = 3.0 # 距离墙壁 3 米开始报警
        
        if abs(leader.x) > (POOL_SIZE - margin) or abs(leader.y) > (POOL_SIZE - margin):
            # 撞墙报警！强制计算朝向水池中心点 (0,0) 的角度
            angle_to_center = math.atan2(-leader.y, -leader.x)
            diff = (angle_to_center - leader.theta + math.pi) % (2 * math.pi) - math.pi
            
            # 给定极大的转向角速度，强制掉头
            leader_w = 2.5 * np.sign(diff) if abs(diff) > 0.1 else 0
            leader_v = 0.8 # 减速转弯，防止冲出去
        else:
            # 正常区域：S型漫游
            leader_v = 2
            leader_w = 0.6 * math.sin(t / 20.0) 
            
        leader.update_state(leader_v, leader_w, dt)
        
        # 【核心补丁】：强行在物理层面限位，哪怕是由于步长计算溢出，也绝对切断越界的可能性
        leader.x = np.clip(leader.x, -POOL_SIZE, POOL_SIZE)
        leader.y = np.clip(leader.y, -POOL_SIZE, POOL_SIZE)

        # ==========================================
        # 2. 跟随者 (Followers) 编队控制闭环
        # ==========================================
        f1_target_x, f1_target_y = get_target_position(leader, F1_offset[0], F1_offset[1])
        f1_v, f1_w = proportional_controller(follower_1, f1_target_x, f1_target_y)
        follower_1.update_state(f1_v, f1_w, dt)
        # 跟随者同样加一道安全锁
        follower_1.x = np.clip(follower_1.x, -POOL_SIZE, POOL_SIZE)
        follower_1.y = np.clip(follower_1.y, -POOL_SIZE, POOL_SIZE)

        f2_target_x, f2_target_y = get_target_position(leader, F2_offset[0], F2_offset[1])
        f2_v, f2_w = proportional_controller(follower_2, f2_target_x, f2_target_y)
        follower_2.update_state(f2_v, f2_w, dt)
        follower_2.x = np.clip(follower_2.x, -POOL_SIZE, POOL_SIZE)
        follower_2.y = np.clip(follower_2.y, -POOL_SIZE, POOL_SIZE)

        # ==========================================
        # 3. 画板渲染 (彻底钉死上帝视角)
        # ==========================================
        if t % 2 == 0:
            ax.cla()
            
            # 【优化亮点】：视角刻度比活动范围大 5 米，彻底看清全局！
            VIEW_LIMIT = POOL_SIZE + 5.0 
            ax.set_xlim(-VIEW_LIMIT, VIEW_LIMIT)
            ax.set_ylim(-VIEW_LIMIT, VIEW_LIMIT)
            
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_title(f"Global View: Bounded Pool Formation (Time: {t*dt:.1f}s)")
            
            # 画出真正的水池物理边界 (一个 30x30 的黑框)
            rect = patches.Rectangle((-POOL_SIZE, -POOL_SIZE), POOL_SIZE*2, POOL_SIZE*2, 
                                     linewidth=2, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
            
            # 画尾迹
            ax.plot(leader.history_x, leader.history_y, 'r-', alpha=0.3, linewidth=2)
            ax.plot(follower_1.history_x, follower_1.history_y, 'b-', alpha=0.3, linewidth=2)
            ax.plot(follower_2.history_x, follower_2.history_y, 'g-', alpha=0.3, linewidth=2)
            
            # 画队形骨架
            ax.plot([leader.x, follower_1.x], [leader.y, follower_1.y], 'k--', alpha=0.5)
            ax.plot([leader.x, follower_2.x], [leader.y, follower_2.y], 'k--', alpha=0.5)
            
            # 画实体
            ax.plot(leader.x, leader.y, 'ro', markersize=10, label="Leader")
            ax.plot(follower_1.x, follower_1.y, 'bo', markersize=8, label="Follower 1")
            ax.plot(follower_2.x, follower_2.y, 'go', markersize=8, label="Follower 2")
            
            ax.legend(loc='upper right')
            plt.pause(0.01)

    plt.ioff()
    plt.show()
