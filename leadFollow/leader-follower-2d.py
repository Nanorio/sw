import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

# ==========================================
# 1. 独立的 PID 大脑类
# ==========================================
class PIDController:
    """工业标准的独立 PID 控制器"""
    def __init__(self, Kp, Ki, Kd, max_out):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_out = max_out
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error, dt):
        P_out = self.Kp * error
        self.integral += error * dt
        self.integral = np.clip(self.integral, -2.0, 2.0) # 积分限幅
        I_out = self.Ki * self.integral
        derivative = (error - self.prev_error) / dt
        D_out = self.Kd * derivative
        
        self.prev_error = error
        output = P_out + I_out + D_out
        return np.clip(output, -self.max_out, self.max_out)

# ==========================================
# 2. 机器人肉体类 (内置 PID 大脑)
# ==========================================
class UnicycleAgent:
    def __init__(self, id, x, y, theta):
        self.id = id
        self.x = x
        self.y = y
        self.theta = theta
        self.history_x = [x]
        self.history_y = [y]
        
        # 【核心改进】：每个机器人都有自己的专属 PID 控制器！不会互相干扰！
        self.pid_v = PIDController(Kp=1.2, Ki=0.05, Kd=0.2, max_out=3.5) # 控制油门
        self.pid_w = PIDController(Kp=4.0, Ki=0.0, Kd=0.5, max_out=3.0)  # 控制方向盘

    def update_state(self, v, w, dt=0.1):
        self.x += v * math.cos(self.theta) * dt
        self.y += v * math.sin(self.theta) * dt
        self.theta += w * dt
        self.theta = (self.theta + math.pi) % (2 * math.pi) - math.pi
        
        self.history_x.append(self.x)
        self.history_y.append(self.y)
        if len(self.history_x) > 40:
            self.history_x.pop(0)
            self.history_y.pop(0)

def get_target_position(leader, offset_x, offset_y):
    target_x = leader.x + offset_x * math.cos(leader.theta) - offset_y * math.sin(leader.theta)
    target_y = leader.y + offset_x * math.sin(leader.theta) + offset_y * math.cos(leader.theta)
    return target_x, target_y

# ==========================================
# 3. 终极控制器：APF防撞 + PID反馈
# ==========================================
def ultimate_controller(agent, target_x, target_y, obstacles, dt):
    """同时具备人工势场避障和 PID 平滑控制的终极大脑"""
    # [1] 计算引力 (想去目标)
    v_att_x = target_x - agent.x
    v_att_y = target_y - agent.y
    
    # [2] 计算斥力 (躲避队友和老大)
    v_rep_x = 0.0
    v_rep_y = 0.0
    SAFE_DIST = 3.0   
    K_rep = 8.0       
    
    for obs in obstacles:
        dist = math.hypot(obs.x - agent.x, obs.y - agent.y)
        if 0.1 < dist < SAFE_DIST:
            rep_strength = K_rep * (1.0 / dist - 1.0 / SAFE_DIST)
            angle_away = math.atan2(agent.y - obs.y, agent.x - obs.x)
            v_rep_x += rep_strength * math.cos(angle_away)
            v_rep_y += rep_strength * math.sin(angle_away)

    # [3] 向量合成 (实际想去的方向)
    final_x = v_att_x + v_rep_x
    final_y = v_att_y + v_rep_y
    
    # [4] 喂给自身的 PID 进行控制
    distance_error = math.hypot(final_x, final_y)
    target_theta = math.atan2(final_y, final_x)
    theta_error = target_theta - agent.theta
    theta_error = (theta_error + math.pi) % (2 * math.pi) - math.pi
    
    if distance_error < 0.1:
        agent.pid_v.integral = 0.0 # 极近距离清空积分，防止抖动
        v = 0.0
    else:
        v = agent.pid_v.update(distance_error, dt)
        
    w = agent.pid_w.update(theta_error, dt)
    
    # 不允许倒车
    v = max(0, v)
    return v, w

# ==========================================
# 4. 主程序仿真循环
# ==========================================
if __name__ == '__main__':
    POOL_SIZE = 30.0
    SPAWN_RANGE = 25.0

    leader = UnicycleAgent("Leader", 
        x=np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE), 
        y=np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE), 
        theta=np.random.uniform(-math.pi, math.pi))

    follower_1 = UnicycleAgent("F1", 
        x=np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE), 
        y=np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE), 
        theta=np.random.uniform(-math.pi, math.pi))

    follower_2 = UnicycleAgent("F2", 
        x=np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE), 
        y=np.random.uniform(-SPAWN_RANGE, SPAWN_RANGE), 
        theta=np.random.uniform(-math.pi, math.pi))

    # 倒 V 字阵 (左后方和右后方)
    F1_offset = (-4.0, 3.5) 
    F2_offset = (-4.0, -3.5)

    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 9))

    dt = 0.1
    time_steps = 2000 
    
    for t in range(time_steps):
        # --- 领航者逻辑 ---
        margin = 3.0 
        if abs(leader.x) > (POOL_SIZE - margin) or abs(leader.y) > (POOL_SIZE - margin):
            angle_to_center = math.atan2(-leader.y, -leader.x)
            diff = (angle_to_center - leader.theta + math.pi) % (2 * math.pi) - math.pi
            leader_w = 2.5 * np.sign(diff) if abs(diff) > 0.1 else 0
            leader_v = 1.0 # 减速转弯
        else:
            # 【修复物理矛盾】：把速度降回 2.0，保证小弟 (极速3.5) 能追得上
            leader_v = 2.0 
            leader_w = 0.8 * math.sin(t / 25.0) 
            
        leader.update_state(leader_v, leader_w, dt)
        leader.x = np.clip(leader.x, -POOL_SIZE, POOL_SIZE)
        leader.y = np.clip(leader.y, -POOL_SIZE, POOL_SIZE)

        # --- 跟随者 1 逻辑 ---
        f1_tx, f1_ty = get_target_position(leader, F1_offset[0], F1_offset[1])
        # 终极调用：把误差和障碍物都喂进去
        f1_v, f1_w = ultimate_controller(follower_1, f1_tx, f1_ty, obstacles=[follower_2, leader], dt=dt)
        follower_1.update_state(f1_v, f1_w, dt)
        follower_1.x = np.clip(follower_1.x, -POOL_SIZE, POOL_SIZE)
        follower_1.y = np.clip(follower_1.y, -POOL_SIZE, POOL_SIZE)

        # --- 跟随者 2 逻辑 ---
        f2_tx, f2_ty = get_target_position(leader, F2_offset[0], F2_offset[1])
        f2_v, f2_w = ultimate_controller(follower_2, f2_tx, f2_ty, obstacles=[follower_1, leader], dt=dt)
        follower_2.update_state(f2_v, f2_w, dt)
        follower_2.x = np.clip(follower_2.x, -POOL_SIZE, POOL_SIZE)
        follower_2.y = np.clip(follower_2.y, -POOL_SIZE, POOL_SIZE)

        # --- 画面渲染 ---
        if t % 2 == 0:
            ax.cla()
            VIEW_LIMIT = POOL_SIZE + 5.0 
            ax.set_xlim(-VIEW_LIMIT, VIEW_LIMIT)
            ax.set_ylim(-VIEW_LIMIT, VIEW_LIMIT)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.set_title(f"Ultimate: PID + APF Formation Control (Time: {t*dt:.1f}s)")
            
            rect = patches.Rectangle((-POOL_SIZE, -POOL_SIZE), POOL_SIZE*2, POOL_SIZE*2, 
                                     linewidth=2, edgecolor='black', facecolor='none')
            ax.add_patch(rect)
            
            ax.plot(leader.history_x, leader.history_y, 'r-', alpha=0.4, linewidth=2)
            ax.plot(follower_1.history_x, follower_1.history_y, 'b-', alpha=0.4, linewidth=2)
            ax.plot(follower_2.history_x, follower_2.history_y, 'g-', alpha=0.4, linewidth=2)
            
            ax.plot([leader.x, follower_1.x], [leader.y, follower_1.y], 'k--', alpha=0.4)
            ax.plot([leader.x, follower_2.x], [leader.y, follower_2.y], 'k--', alpha=0.4)
            
            # 画实体和安全预警圈 (展现 APF 半径)
            ax.plot(leader.x, leader.y, 'ro', markersize=12, label="Leader")
            ax.plot(follower_1.x, follower_1.y, 'bo', markersize=10, label="Follower 1")
            ax.plot(follower_2.x, follower_2.y, 'go', markersize=10, label="Follower 2")
            
            # 简单的防撞圈展示
            circle1 = plt.Circle((follower_1.x, follower_1.y), 3.0, color='b', fill=False, alpha=0.2)
            circle2 = plt.Circle((follower_2.x, follower_2.y), 3.0, color='g', fill=False, alpha=0.2)
            ax.add_patch(circle1)
            ax.add_patch(circle2)
            
            ax.legend(loc='upper right')
            plt.pause(0.01)

    plt.ioff()
    plt.show()