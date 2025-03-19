
from throwball_train import BallisticNet
import numpy as np
import torch
import math

# ========== 理论速度计算 ===========
# 假设这些常量已经定义
theta_deg = 65
START_H = 600
FRIC_K = 0.0163
BALL_M = 600
g = 9810
BALL_R = 121
GOAL_H = 2430 + BALL_R
PI = math.pi

def f(t, vx, vy):
    term1 = (START_H + vy / FRIC_K * BALL_M + g * BALL_M / FRIC_K * BALL_M / FRIC_K)
    term2 = (math.exp(-(FRIC_K * t) / BALL_M) * (g * BALL_M / FRIC_K * BALL_M / FRIC_K + vy / FRIC_K * BALL_M))
    term3 = (g * BALL_M * t) / FRIC_K
    return term1 - term2 - term3 - GOAL_H - 247

def df(t, vx, vy):
    return (math.exp((-(FRIC_K * t) / BALL_M)) * (g * BALL_M / FRIC_K + vy)) - ((g * BALL_M) / FRIC_K)

def hh(v0, t, dis):
    return (BALL_M * math.log((BALL_M + FRIC_K * t * v0) / BALL_M)) / FRIC_K - dis

def NewtonIter(vx, vy):
    # 飞行时间
    Flytime = 0
    # 迭代最大误差
    tol = 1e-6
    # 最大迭代次数
    max_iter = 20
    # 是否获得正确飞行时间
    flag = False
    for startx in range(0, 6, 1):
        startx = startx * 0.5
        x0 = startx
        for iter in range(1, max_iter + 1):
            x1 = x0 - f(x0, vx, vy) / df(x0, vx, vy)
            if abs(x1 - x0) < tol and df(x1, vx, vy) < 0:
                flag = True
                Flytime = x1
                break
            x0 = x1
        if flag:
            break
    return Flytime

def BulletModelCalc(angle, dis):
    # 二分法求解出射速度
    l_v0 = 0
    r_v0 = 20000
    flag = False
    while l_v0 <= r_v0:
        mid_v0 = (l_v0 + r_v0) / 2
        mid_vx = mid_v0 * math.cos(angle / 180 * PI)
        mid_vy = mid_v0 * math.sin(angle / 180 * PI)
        mid_t = NewtonIter(mid_vx, mid_vy)
        mid_now = hh(mid_vx, mid_t, dis)
        if abs(mid_now) < 1:
            flag = True
            return mid_v0
        if mid_now < 0:
            l_v0 = mid_v0 + 0.001
        else:
            r_v0 = mid_v0 - 0.001
    if not flag:
        return 0


# ========== 预测函数 ==========
def predict_speed(distance):
    """ 输入实际距离（mm），返回电机速度（rpm） """
    # 加载标准化参数
    X_mean = np.load("model/X_mean.npy")
    X_std = np.load("model/X_std.npy")
    y_mean = np.load("model/y_mean.npy")
    y_std = np.load("model/y_std.npy")

    # 标准化输入
    normalized_dist = (distance - X_mean) / X_std
    
    # 加载模型
    model = BallisticNet()
    model.load_state_dict(torch.load("model/best_model.pth"))
    model.eval()
    
    # 预测并反标准化
    with torch.no_grad():
        normalized_vel = model(torch.FloatTensor([[normalized_dist]]))
        real_vel = normalized_vel.item() * y_std + y_mean
    
    return real_vel  # 返回实际物理量纲的速度值

def main():
    # X_mean = np.load("model/X_mean.npy")
    # X_std = np.load("model/X_std.npy")
    # y_mean = np.load("model/y_mean.npy")
    # y_std = np.load("model/y_std.npy")

    for dis in range(3000,10000,100):
        predicted_speed = predict_speed(dis)
        # predicted_speed = predicted_speed * 2 * PI * 36.8 / 60 # rpm to mm/s
        
        # 理论计算验证
        theoretical_speed = BulletModelCalc(theta_deg, dis)
        
        print(f"距离 {dis} mm时：")
        print(f"预测速度：{predicted_speed:.2f} mm/s")
        # print(f"理论速度：{theoretical_speed:.2f} mm/s")
        # print(f"绝对误差：{abs(predicted_speed - theoretical_speed):.2f} mm/s")

# ========== 使用示例 ==========
if __name__ == "__main__":
    main()