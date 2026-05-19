import numpy as np
import math

class PIDController:
    """工业标准的独立 PID 控制器类"""
    def __init__(self, Kp, Ki, Kd, max_out):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.max_out = max_out  # 输出限幅 (相当于 np.clip)
        
        self.integral = 0.0     # 误差的累积 (I)
        self.prev_error = 0.0   # 上一次的误差 (D)

    def update(self, error, dt):
        """核心计算公式：每帧调用一次"""
        # 1. 比例项 (P) - 看现在
        P_out = self.Kp * error
        
        # 2. 积分项 (I) - 看过去
        self.integral += error * dt
        # 【工程防坑】积分限幅 (Anti-windup)：防止卡住太久导致积分爆炸
        self.integral = np.clip(self.integral, -2.0, 2.0) 
        I_out = self.Ki * self.integral
        
        # 3. 微分项 (D) - 看未来
        derivative = (error - self.prev_error) / dt
        D_out = self.Kd * derivative
        
        # 保存这次误差，留给下一帧用
        self.prev_error = error
        
        # 4. 计算总输出并进行物理限幅
        output = P_out + I_out + D_out
        output = np.clip(output, -self.max_out, self.max_out)
        
        return output

# ==========================================
# 如何在你的主程序中使用这个改进版？
# ==========================================

# 1. 在循环外面，分别为线速度 (距离) 和角速度 (转向) 实例化两个独立的大脑
# (参数需要根据你真实的 UUV 慢慢调)
pid_v = PIDController(Kp=1.2, Ki=0.1, Kd=0.5, max_out=3.0)  # 控制油门
pid_w = PIDController(Kp=3.0, Ki=0.0, Kd=0.8, max_out=2.5)  # 控制方向盘

def advanced_controller(agent, target_x, target_y, dt):
    """新的闭环反馈执行函数"""
    error_x = target_x - agent.x
    error_y = target_y - agent.y
    distance_error = math.hypot(error_x, error_y)
    
    target_theta = math.atan2(error_y, error_x)
    theta_error = target_theta - agent.theta
    theta_error = (theta_error + math.pi) % (2 * math.pi) - math.pi
    
    # 【改动点】：把误差喂给刚才定义好的满血版 PID
    # 如果距离极小，清空积分并刹车，防止原地抖动
    if distance_error < 0.1:
        pid_v.integral = 0.0
        v = 0.0
    else:
        v = pid_v.update(distance_error, dt)
        
    w = pid_w.update(theta_error, dt)
    
    # 强制不允许倒车 (如果 UUV 物理上不支持的话)
    v = max(0, v)
    
    return v, w
