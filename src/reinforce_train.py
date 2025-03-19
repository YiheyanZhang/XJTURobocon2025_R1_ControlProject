import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# 环境定义
class BasketballEnv:
    def __init__(self):
        self.g = 9.81  # 重力加速度
        self.target_distance = 5.0  # 目标水平距离（米）
        self.target_height = 3.05   # 篮筐高度（米）
        self.state_dim = 2          # 状态维度（目标距离，目标高度）
        self.action_dim = 2         # 动作维度（发射角度，初速度）

    def reset(self):
        # 返回目标位置作为状态（固定目标）
        return np.array([self.target_distance, self.target_height])

    def step(self, action):
        # 解析动作并限制范围
        theta = np.clip(action[0], 0, np.pi/2)  # 发射角度（0~90度）
        v0 = np.clip(action[1], 1, 20)          # 初速度（1~20 m/s）

        # 计算飞行时间（当篮球到达目标水平距离时）
        with np.errstate(divide='ignore'):
            t = self.target_distance / (v0 * np.cos(theta) + 1e-8)

        # 计算垂直位置
        y = v0 * np.sin(theta) * t - 0.5 * self.g * t**2

        # 计算水平误差
        x_error = abs(v0 * np.cos(theta) * t - self.target_distance)
        y_error = abs(y - self.target_height)
        distance = np.sqrt(x_error**2 + y_error**2)

        # 奖励函数设计
        reward = -distance  # 基础奖励为负距离
        if distance < 0.1:  # 成功条件
            reward += 100

        done = True  # 单步episode
        return self.reset(), reward, done, {}

# DDPG算法实现
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        # 动作缩放参数
        self.theta_scale = torch.tensor(np.pi/4)  # 输出范围[-1,1] -> [-π/4, π/4]
        self.v0_scale = torch.tensor(9.5)         # 输出范围[-1,1] -> [-9.5, 9.5]

    def forward(self, state):
        raw_action = self.net(state)
        # 统一处理成二维张量
        if raw_action.dim() == 1:
            raw_action = raw_action.unsqueeze(0)
        
        # 缩放计算（确保使用torch运算符）
        theta = raw_action[:, 0] * self.theta_scale + torch.tensor(np.pi/4).to(raw_action.device)
        v0 = raw_action[:, 1] * self.v0_scale + torch.tensor(10.5).to(raw_action.device)
        
        # 返回处理（保留必要维度）
        return torch.cat([theta.unsqueeze(1), v0.unsqueeze(1)], dim=1)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], 1))

class DDPG:
    def __init__(self, state_dim, action_dim):
        self.actor = Actor(state_dim, action_dim)
        self.actor_target = Actor(state_dim, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 64
        self.tau = 0.005
        self.gamma = 0.99

    def select_action(self, state, noise=True):
        # 确保输入转换正确
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0)  # numpy转tensor并添加batch维度
        else:
            state = state.unsqueeze(0) if state.dim() == 1 else state
        
        with torch.no_grad():
            action_tensor = self.actor(state)
            action = action_tensor.squeeze(0).cpu().numpy()  # 移除batch维度
        
        # 添加噪声和裁剪
        if noise:
            action += np.random.normal(0, 0.2, size=action.shape)
        action[0] = np.clip(action[0], 0, np.pi/2)
        action[1] = np.clip(action[1], 1, 20)
        return action

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # 从经验池采样
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

        # 更新Critic网络
        with torch.no_grad():
            target_actions = self.actor_target(next_states)
            target_Q = self.critic_target(next_states, target_actions)
            target_Q = rewards + (1 - dones) * self.gamma * target_Q

        current_Q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 更新Actor网络
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# 训练循环
env = BasketballEnv()
ddpg = DDPG(env.state_dim, env.action_dim)

episodes = 1000
for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    done = False
    
    # 单步episode
    action = ddpg.select_action(state)  # 直接传入numpy数组
    next_state, reward, done, _ = env.step(action)
    ddpg.replay_buffer.append((state, action, reward, next_state, done))
    ddpg.train()
    
    if (ep+1) % 50 == 0:
        # 测试当前策略（无探索噪声）
        test_action = ddpg.select_action(torch.FloatTensor(state), noise=False)
        _, test_reward, _, _ = env.step(test_action)
        print(f"Episode {ep+1}, Test Reward: {test_reward:.2f}")

# 最终测试
state = env.reset()
optimal_action = ddpg.select_action(torch.FloatTensor(state), noise=False)
print(f"\nOptimal Parameters: theta={optimal_action[0]:.2f} rad, v0={optimal_action[1]:.2f} m/s")