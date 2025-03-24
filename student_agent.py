import gym
import numpy as np
import importlib.util
import time
import random
import math
from IPython.display import clear_output

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple

# 使用namedtuple来存储经验，提高代码可读性
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class SimpleTaxiEnv():
    def __init__(self, grid_size=5, fuel_limit=200):
        self.grid_size = grid_size
        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit
        self.passenger_picked_up = False
        self.passenger_loc = None
        self.obstacles = set()  # 将存储障碍物位置
        self.destination = None
        self.obstacle_density = 0.1  # 障碍物密度 - 可调整

        # 统计数据
        self.successful_pickups = 0
        self.successful_dropoffs = 0
        self.action_history = []
        
        # 记录最近到达过的位置，用于检测循环行为
        self.recent_positions = []
        self.position_history_limit = 20
        
    def reset(self):
        """重置环境，生成随机障碍物并确保 Taxi、乘客与目的地互不重叠"""
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False
        self.action_history = []
        self.recent_positions = []
        
        # 重置障碍物
        self.obstacles = set()
        
        # 生成随机障碍物
        all_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        num_obstacles = int(self.obstacle_density * self.grid_size * self.grid_size)
        
        # 随机选择位置作为障碍物
        potential_obstacles = random.sample(all_positions, min(num_obstacles, len(all_positions)))
        for pos in potential_obstacles:
            self.obstacles.add(pos)
        
        # 随机生成可用位置（排除障碍物）
        available_positions = [pos for pos in all_positions if pos not in self.obstacles]
        
        # 确保有足够的空间放置taxi、passenger和destination
        if len(available_positions) < 3:
            # 如果空间不足，减少障碍物
            while len(available_positions) < 3:
                if not self.obstacles:
                    break
                self.obstacles.pop()
                available_positions = [pos for pos in all_positions if pos not in self.obstacles]
        
        # 随机选择不重叠的位置
        sampled_positions = random.sample(available_positions, 3)  # 我们需要taxi、passenger和destination三个位置
        
        self.taxi_pos = sampled_positions[0]
        self.passenger_loc = sampled_positions[1]
        self.destination = sampled_positions[2]
        
        return self.get_state(), {}

    def step(self, action):
        """更新环境状态并返回 (state, reward, done, info)"""
        old_state = self.get_state()
        old_pos = self.taxi_pos
        
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0
        done = False

        # 添加当前位置到历史记录
        self.recent_positions.append(self.taxi_pos)
        if len(self.recent_positions) > self.position_history_limit:
            self.recent_positions.pop(0)
            
        # 检测循环行为
        position_repeat_penalty = 0
        if len(self.recent_positions) > 5:
            position_counts = {}
            for pos in self.recent_positions:
                position_counts[pos] = position_counts.get(pos, 0) + 1
            # 如果某个位置重复超过3次，给予惩罚
            for pos, count in position_counts.items():
                if count > 3:
                    position_repeat_penalty = -1.0 * count  # 惩罚与重复次数成比例
            
        # 根据动作更新位置（移动动作）
        if action == 0:  # Move Down
            next_row += 1
        elif action == 1:  # Move Up
            next_row -= 1
        elif action == 2:  # Move Right
            next_col += 1
        elif action == 3:  # Move Left
            next_col -= 1

        if action in [0, 1, 2, 3]:
            # 计算距离
            if self.passenger_picked_up:
                old_distance = abs(taxi_row - self.destination[0]) + abs(taxi_col - self.destination[1])
            else:
                old_distance = abs(taxi_row - self.passenger_loc[0]) + abs(taxi_col - self.passenger_loc[1])
        
            # 处理移动
            if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward -= 5
            else:
                self.taxi_pos = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos
            
            # 基础移动惩罚，降低为-0.05，鼓励更多探索
            reward -= 0.05
            
            # 更合理的shaping rewards
            if self.passenger_picked_up:
                reward += 0.2  # 乘客在车上时的奖励
                new_distance = abs(self.taxi_pos[0] - self.destination[0]) + abs(self.taxi_pos[1] - self.destination[1])
                # 更平滑的奖励梯度
                if new_distance < old_distance:
                    reward += 2.0  # 减少朝目标移动的奖励，让agent更灵活
                elif new_distance > old_distance:
                    reward -= 0.5  # 减少远离目标的惩罚
                
                # 接近目的地时的额外奖励
                if new_distance == 1:  # 距离目的地仅1步
                    reward += 3.0
                elif new_distance == 0:  # 到达目的地
                    reward += 10.0  # 强烈鼓励在有乘客时到达目的地
            else:
                new_distance = abs(self.taxi_pos[0] - self.passenger_loc[0]) + abs(self.taxi_pos[1] - self.passenger_loc[1])
                if new_distance < old_distance:
                    reward += 0.5
                elif new_distance > old_distance:
                    reward -= 0.2
                
                # 接近乘客时的额外奖励
                if new_distance == 1:  # 距离乘客仅1步
                    reward += 1
                elif new_distance == 0:  # 到达乘客位置
                    reward += 3.0  # 强烈鼓励到达乘客位置
        else:
            # 非移动动作处理
            if action == 4:  # PICKUP
                if self.passenger_picked_up:
                    reward -= 5  # 减轻重复接客的惩罚
                elif self.taxi_pos == self.passenger_loc:
                    self.passenger_picked_up = True
                    self.passenger_loc = self.taxi_pos
                    reward += 30  # 增加接客奖励
                    self.successful_pickups += 1
                else:
                    reward -= 5  # 减轻错误接客的惩罚
            elif action == 5:  # DROPOFF
               # 在step函數中的DROPOFF動作部分
                if self.passenger_picked_up:
                    if self.taxi_pos == self.destination:
                        reward += 500  # 提高從100到500，使其遠高於其他獎勵
                        done = True
                        self.successful_dropoffs += 1
                    else:
                        reward -= 30  # 減輕對錯誤地點下客的懲罰
        # 基础操作惩罚
        reward -= 0.05

        # 扣除燃料
        self.current_fuel -= 1
        if self.current_fuel <= 0:
            reward -= 10
            done = True

        # 检查新旧状态是否相同
        new_state = self.get_state()
        if old_state == new_state and old_pos == self.taxi_pos:
            reward -= 1  # 减轻状态未变的惩罚

        # 更新并检查最近行动
        self.action_history.append(action)
        if len(self.action_history) > 8:
            self.action_history.pop(0)
            
        # 应用循环位置惩罚
        reward += position_repeat_penalty

         # 如果乘客在车上，每一步额外奖励 +0.1
        if self.passenger_picked_up:
            reward += 0.2
        return new_state, reward, done, {}

    def get_state(self):
        """返回当前环境状态，不依赖固定站点位置"""
        taxi_row, taxi_col = self.taxi_pos
        
        # 计算 taxi 到乘客的曼哈顿距离
        distance_to_passenger = abs(taxi_row - self.passenger_loc[0]) + abs(taxi_col - self.passenger_loc[1])
        
        # 计算 taxi 到目的地的曼哈顿距离
        distance_to_destination = abs(taxi_row - self.destination[0]) + abs(taxi_col - self.destination[1])
        
        # 障碍物信息：检测 taxi 四个方向是否存在障碍物或越界
        obstacle_north = int(taxi_row == 0 or (taxi_row - 1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row + 1, taxi_col) in self.obstacles)
        obstacle_east = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col + 1) in self.obstacles)
        obstacle_west = int(taxi_col == 0 or (taxi_row, taxi_col - 1) in self.obstacles)
        obstacles = [obstacle_north, obstacle_south, obstacle_east, obstacle_west]
        
        # 乘客是否在车上
        passenger_in_taxi = int(self.passenger_picked_up)
        
        # 乘客和目的地与Taxi的相对位置标志
        passenger_adjacent = int(distance_to_passenger <= 1) 
        destination_adjacent = int(distance_to_destination <= 1)
        
        # 组合所有特征成一个tuple
        state = tuple(obstacles + 
                    [passenger_in_taxi, distance_to_passenger, distance_to_destination,
                    passenger_adjacent, destination_adjacent])
        return state

    def render_env(self, taxi_pos, action=None, step=None, fuel=None):
        clear_output(wait=True)
        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]
        
        # 显示障碍物
        for obstacle_pos in self.obstacles:
            o_row, o_col = obstacle_pos
            if 0 <= o_row < self.grid_size and 0 <= o_col < self.grid_size:
                grid[o_row][o_col] = 'X'
        
        # 显示乘客和目的地
        p_row, p_col = self.passenger_loc
        if 0 <= p_row < self.grid_size and 0 <= p_col < self.grid_size and not self.passenger_picked_up:
            grid[p_row][p_col] = 'P'
            
        d_row, d_col = self.destination
        if 0 <= d_row < self.grid_size and 0 <= d_col < self.grid_size:
            grid[d_row][d_col] = 'D'
            
        # 显示 Taxi
        ty, tx = taxi_pos
        if 0 <= tx < self.grid_size and 0 <= ty < self.grid_size:
            grid[ty][tx] = '🚖'
            
        print(f"\nStep: {step}")
        print(f"Taxi Position: ({tx}, {ty})")
        print(f"Fuel Left: {fuel}")
        print(f"Last Action: {self.get_action_name(action)}")
        print(f"Passenger Picked Up: {self.passenger_picked_up}")
        print(f"Passenger Location: {self.passenger_loc}")
        print(f"Destination: {self.destination}\n")
        
        for row in grid:
            print(" ".join(row))
        print("\n")

    def get_action_name(self, action):
        actions = ["Move South", "Move North", "Move East", "Move West", "Pick Up", "Drop Off"]
        return actions[action] if action is not None else "None"

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        
        # 特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 价值流
        self.value_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, state):
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # 计算Q值: Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

class DQNAgent:
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = 0.99
        self.epsilon = 0.01  # 测试时使用低探索率
        
        # 使用Dueling DQN网络
        self.policy_net = DuelingDQN(state_dim, action_dim).to(device)
        
    def load_model(self, path):
        """加载模型"""
        try:
            # 使用map_location参数确保模型可以在CPU上加载，即使它是在GPU上保存的
            checkpoint = torch.load(path, map_location=torch.device('cpu'))
            self.policy_net.load_state_dict(checkpoint['policy_net'])
        except Exception as e:
            print(f"加载模型时出错: {e}")
            # 如果加载失败，使用初始化的模型
            pass

def process_state_for_network(state):
    """处理状态以适应网络输入"""
    # 假设最大曼哈顿距离为环境大小的2倍
    max_distance = 15.0  # 对于大一点的环境，给更大的最大距离
    
    if state is None:
        # 返回一个合理的默认状态
        return [0] * 9
    
    try:
        # 障碍物信息
        if len(state) >= 4:
            obstacles = list(state[:4])
        else:
            obstacles = [0, 0, 0, 0]
        
        # 乘客是否在车上
        passenger_in_taxi = [1] if len(state) > 4 and state[4] else [0]
        
        # 归一化到乘客和目的地的距离
        distance_to_passenger = [state[5] / max_distance] if len(state) > 5 else [0]
        distance_to_destination = [state[6] / max_distance] if len(state) > 6 else [0]
        
        # 乘客和目的地是否在相邻位置
        passenger_adjacent = [1] if len(state) > 7 and state[7] else [0]
        destination_adjacent = [1] if len(state) > 8 and state[8] else [0]
        
        # 合并所有特征
        processed_state = (obstacles + passenger_in_taxi + 
                        distance_to_passenger + distance_to_destination + 
                        passenger_adjacent + destination_adjacent)
        
        return processed_state
    except Exception as e:
        print(f"处理状态时出错: {e}")
        return [0] * 9  # 返回默认状态

def get_action(obs):
    """从环境观察返回行动"""
    # 始终使用CPU设备，避免CUDA设备不可用的问题
    device = torch.device("cpu")
    
    # 检查观察是否有效，如果无效则返回随机动作
    if obs is None:
        return random.randint(0, 5)
    
    try:
        # 处理观察
        processed_obs = process_state_for_network(obs)
        
        # 创建一个临时agent对象，仅用于加载模型
        state_dim = len(processed_obs)
        action_dim = 6
        agent = DQNAgent(state_dim, action_dim, device)
        
        # 尝试加载预训练模型，如果失败则使用启发式策略
        try:
            agent.load_model("best_taxi_model.pth")
            
            # 选择动作（测试模式，不需要探索）
            with torch.no_grad():
                state_tensor = torch.FloatTensor(processed_obs).unsqueeze(0).to(device)
                q_values = agent.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
        except Exception as e:
            # 如果模型加载失败，使用启发式策略
            if len(obs) > 4 and obs[4]:  # 乘客已上车
                # 距离目的地的曼哈顿距离
                dest_dist = obs[6] if len(obs) > 6 else 0
                if dest_dist == 0:  # 已到达目的地
                    action = 5  # DROPOFF
                else:
                    # 根据障碍物信息选择向目的地移动的方向
                    obstacles = obs[:4] if len(obs) >= 4 else [0, 0, 0, 0]
                    
                    # 有启发式地选择动作
                    if random.random() < 0.7:  # 70%的概率选择合理的动作
                        # 选择一个没有障碍物的方向
                        valid_actions = [i for i in range(4) if i < len(obstacles) and obstacles[i] == 0]
                        if valid_actions:
                            action = random.choice(valid_actions)
                        else:
                            action = random.randint(0, 3)  # 随机选择
                    else:
                        action = random.randint(0, 5)  # 完全随机
            else:  # 乘客未上车
                # 距离乘客的曼哈顿距离
                pass_dist = obs[5] if len(obs) > 5 else 0
                if pass_dist == 0:  # 已到达乘客位置
                    action = 4  # PICKUP
                else:
                    # 根据障碍物信息选择向乘客移动的方向
                    obstacles = obs[:4] if len(obs) >= 4 else [0, 0, 0, 0]
                    
                    # 有启发式地选择动作
                    if random.random() < 0.7:  # 70%的概率选择合理的动作
                        # 选择一个没有障碍物的方向
                        valid_actions = [i for i in range(4) if i < len(obstacles) and obstacles[i] == 0]
                        if valid_actions:
                            action = random.choice(valid_actions)
                        else:
                            action = random.randint(0, 3)  # 随机选择
                    else:
                        action = random.randint(0, 5)  # 完全随机
    except Exception as e:
        # 如果处理出错，返回随机动作
        action = random.randint(0, 5)
    
    return action
# 主函数
if __name__ == "__main__":
    # 环境配置
    env_config = {
        "grid_size": 5,
        "fuel_limit": 1000
    }
    env = SimpleTaxiEnv(**env_config)
    
    # 获取状态维度
    sample_state, _ = env.reset()
    processed_sample = process_state_for_network(sample_state)
    state_dim = len(processed_sample)
    action_dim = 6
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建DQN代理
    agent = DQNAgent(state_dim, action_dim, device, lr=5e-4, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.99999)  # 加快epsilon衰减
    
    # 训练代理
    train_results = train_dqn(env, agent, num_episodes=2000, save_interval=200)
    
    # 打印最终统计信息
    print("训练结束！")
    print(f"Total Successful Pickups: {env.successful_pickups}")
    print(f"Total Successful Dropoffs: {env.successful_dropoffs}")
    print(f"Final Pickup Success Rate: {env.successful_pickups/2000:.4f}")
    print(f"Final Dropoff Success Rate: {env.successful_dropoffs/2000:.4f}")
    
    # 显示模型参数数量
    num_params = sum(p.numel() for p in agent.policy_net.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in the DQN: {num_params}")
    
    # 保存最终模型
    agent.save_model("final_taxi_model.pth")