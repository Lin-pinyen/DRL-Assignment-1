import numpy as np
import random
import math
import os 
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
        self.stations = [(0, 0), (0, self.grid_size - 1),
                         (self.grid_size - 1, 0), (self.grid_size - 1, self.grid_size - 1)]
        self.passenger_loc = None
        self.obstacles = set()  # 简易版无障碍物
        self.destination = None

        # 统计数据
        self.successful_pickups = 0
        self.successful_dropoffs = 0
        self.action_history = []
        
        # 记录最近到达过的位置，用于检测循环行为
        self.recent_positions = []
        self.position_history_limit = 20
        
    def reset(self):
        """重置环境，确保 Taxi、乘客与目的地互不重叠"""
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False
        self.action_history = []
        self.recent_positions = []
        
        available_positions = [
            (x, y) for x in range(self.grid_size) for y in range(self.grid_size)
            if (x, y) not in self.stations and (x, y) not in self.obstacles
        ]
        self.taxi_pos = random.choice(available_positions)
        
        self.passenger_loc = random.choice(self.stations)
        possible_destinations = [s for s in self.stations if s != self.passenger_loc]
        self.destination = random.choice(possible_destinations)
        
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
                    reward += 1.5
                elif new_distance > old_distance:
                    reward -= 0.4
                
                # 接近乘客时的额外奖励
                if new_distance == 1:  # 距离乘客仅1步
                    reward += 2.0
                elif new_distance == 0:  # 到达乘客位置
                    reward += 5.0  # 强烈鼓励到达乘客位置
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

    # def get_state(self):
    #     """返回当前环境状态 (tuple)"""
    #     taxi_row, taxi_col = self.taxi_pos
    #     passenger_row, passenger_col = self.passenger_loc
    #     destination_row, destination_col = self.destination
        
    #     # 检查周围障碍物
    #     obstacle_north = int(taxi_row == 0 or (taxi_row-1, taxi_col) in self.obstacles)
    #     obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row+1, taxi_col) in self.obstacles)
    #     obstacle_east  = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col+1) in self.obstacles)
    #     obstacle_west  = int(taxi_col == 0 or (taxi_row, taxi_col-1) in self.obstacles)
        
    #     # 检查乘客位置相对于Taxi的位置
    #     passenger_loc_north = int((taxi_row - 1, taxi_col) == self.passenger_loc)
    #     passenger_loc_south = int((taxi_row + 1, taxi_col) == self.passenger_loc)
    #     passenger_loc_east  = int((taxi_row, taxi_col + 1) == self.passenger_loc)
    #     passenger_loc_west  = int((taxi_row, taxi_col - 1) == self.passenger_loc)
    #     passenger_loc_middle = int((taxi_row, taxi_col) == self.passenger_loc)
    #     passenger_look = (passenger_loc_north or passenger_loc_south or 
    #                       passenger_loc_east or passenger_loc_west or passenger_loc_middle)
        
    #     # 检查目的地位置相对于Taxi的位置
    #     destination_loc_north = int((taxi_row - 1, taxi_col) == self.destination)
    #     destination_loc_south = int((taxi_row + 1, taxi_col) == self.destination)
    #     destination_loc_east  = int((taxi_row, taxi_col + 1) == self.destination)
    #     destination_loc_west  = int((taxi_row, taxi_col - 1) == self.destination)
    #     destination_loc_middle = int((taxi_row, taxi_col) == self.destination)
    #     destination_look = (destination_loc_north or destination_loc_south or 
    #                         destination_loc_east or destination_loc_west or destination_loc_middle)
        
    #     # 状态信息：添加乘客是否在车上作为状态的一部分
    #     state = (taxi_row, taxi_col,
    #              self.stations[0][0], self.stations[0][1],
    #              self.stations[1][0], self.stations[1][1],
    #              self.stations[2][0], self.stations[2][1],
    #              self.stations[3][0], self.stations[3][1],
    #              obstacle_north, obstacle_south, obstacle_east, obstacle_west,
    #              passenger_look, destination_look, int(self.passenger_picked_up))  # 添加乘客是否在车上
    #     return state
    def get_state(self):
        """返回當前環境狀態 (tuple)，增加了taxi與passenger和destination的直接距離"""
        taxi_row, taxi_col = self.taxi_pos
        
        # 計算 taxi 到四個站點的曼哈頓距離
        distances_to_stations = [abs(taxi_row - station[0]) + abs(taxi_col - station[1]) for station in self.stations]
        
        # 計算 taxi 到乘客的曼哈頓距離
        distance_to_passenger = abs(taxi_row - self.passenger_loc[0]) + abs(taxi_col - self.passenger_loc[1])
        
        # 計算 taxi 到目的地的曼哈頓距離
        distance_to_destination = abs(taxi_row - self.destination[0]) + abs(taxi_col - self.destination[1])
        
        # 障礙物信息：檢測 taxi 四個方向是否存在障礙物或越界
        obstacle_north = int(taxi_row == 0 or (taxi_row - 1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row + 1, taxi_col) in self.obstacles)
        obstacle_east = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col + 1) in self.obstacles)
        obstacle_west = int(taxi_col == 0 or (taxi_row, taxi_col - 1) in self.obstacles)
        obstacles = [obstacle_north, obstacle_south, obstacle_east, obstacle_west]
        
        # 乘客是否在車上
        passenger_in_taxi = int(self.passenger_picked_up)
        
        # 乘客和目的地與Taxi的相對位置標誌（是否在Taxi的相鄰位置）
        passenger_adjacent = int(distance_to_passenger <= 1) 
        destination_adjacent = int(distance_to_destination <= 1)
        
        # 組合所有特徵成一個tuple
        state = tuple(distances_to_stations + obstacles + 
                    [passenger_in_taxi, distance_to_passenger, distance_to_destination,
                    passenger_adjacent, destination_adjacent])
        return state


    def render_env(self, taxi_pos, action=None, step=None, fuel=None):
        clear_output(wait=True)
        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]
        
        # 设置四个站点 (例如：R, G, Y, B)
        grid[0][0] = 'R'
        grid[0][self.grid_size - 1] = 'G'
        grid[self.grid_size - 1][0] = 'Y'
        grid[self.grid_size - 1][self.grid_size - 1] = 'B'
        
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
    def __init__(self, state_dim, action_dim, device, lr=3e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.998):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Epsilon策略参数
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 使用Dueling DQN网络
        self.policy_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 使用Adam优化器，学习率稍低一些
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        # self.optimizer = optim.SGD(self.policy_net.parameters(), lr=lr, momentum=0.9)
        
        # 更大的回放缓冲区
        self.memory = ReplayMemory(10000)
        self.batch_size = 64
        
        # 探索奖励相关
        self.state_counts = {}
        self.explore_coef = 0.5  # 减小探索奖励系数
        
        # 添加学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.5)
        
    def select_action(self, state):
        # Epsilon-贪婪策略选择动作
        if random.random() < self.epsilon:
            action = random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
        
        # Epsilon递减
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return action
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.FloatTensor(batch.state).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(batch.next_state).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        # 计算当前Q值
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        
        # Double DQN: 使用policy_net选择action，使用target_net评估
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
            expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        
        # 使用Huber Loss，对于异常值更鲁棒
        loss = F.smooth_l1_loss(q_values, expected_q_values)
        
        # 梯度优化
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """使用软更新策略更新目标网络"""
        tau = 0.01  # 软更新系数
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)
    
    def get_exploration_bonus(self, state):
        """计算探索奖励"""
        state_key = tuple(state)
        self.state_counts[state_key] = self.state_counts.get(state_key, 0) + 1
        bonus = self.explore_coef / math.sqrt(self.state_counts[state_key])
        return bonus
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']

def process_state_for_network(state):
    """
    處理 get_state() 返回的狀態，現在狀態包含：
      - 到4個站點的距離 (4個值)
      - 障礙物信息 (4個值)
      - 乘客是否在車上 (1個值)
      - 到乘客的距離 (1個值)
      - 到目的地的距離 (1個值)
      - 乘客是否在相鄰位置 (1個值)
      - 目的地是否在相鄰位置 (1個值)
    """
    # 假設最大曼哈頓距離為環境大小的2倍
    max_distance = 10.0  # 對於5x5環境，最大距離是8，稍微放寬一點
    
    # 歸一化到站點的距離
    distances_to_stations = [s / max_distance for s in state[:4]]
    
    # 障礙物信息，不需要歸一化
    obstacles = list(state[4:8])
    
    # 乘客是否在車上
    passenger_in_taxi = [state[8]]
    
    # 歸一化到乘客和目的地的距離
    distance_to_passenger = [state[9] / max_distance]
    distance_to_destination = [state[10] / max_distance]
    
    # 乘客和目的地是否在相鄰位置
    passenger_adjacent = [state[11]]
    destination_adjacent = [state[12]]
    
    # 合併所有特徵
    processed_state = (distances_to_stations + obstacles + passenger_in_taxi + 
                       distance_to_passenger + distance_to_destination + 
                       passenger_adjacent + destination_adjacent)
    
    return processed_state

def train_dqn(env, agent, num_episodes=1000, save_interval=200, render_interval=100):
    """训练DQN代理"""
    total_rewards = []
    avg_rewards = []  # 保存平均奖励
    best_avg_reward = -float('inf')
    episode_lengths = []
    
    # 每个episode的统计数据
    pickup_success_rate = []
    dropoff_success_rate = []
    
    # 每render_interval次评估一下模型
    evaluation_rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        processed_state = process_state_for_network(state)
        episode_reward = 0
        done = False
        step_count = 0
        
        pickup_attempted = False
        dropoff_attempted = False
        
        while not done:
            # 选择动作
            action = agent.select_action(processed_state)
            
            # 记录尝试的接客和送客
            if action == 4:  # PICKUP
                pickup_attempted = True
            elif action == 5:  # DROPOFF
                dropoff_attempted = True
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            processed_next_state = process_state_for_network(next_state)
            
            # 计算探索奖励（可选）
            if episode < num_episodes // 2:  # 只在前半部分训练中使用探索奖励
                bonus = agent.get_exploration_bonus(processed_next_state)
                total_reward = reward + bonus
            else:
                total_reward = reward
            
            # 存储经验
            agent.remember(processed_state, action, total_reward, processed_next_state, done)
            
            # 优化模型
            loss = agent.optimize_model()
            
            # 软更新目标网络
            agent.update_target_network()
            
            # 更新状态
            processed_state = processed_next_state
            episode_reward += reward
            step_count += 1
            
            # 防止过长的episode
            if step_count >= 100:
                done = True
        
        # 记录统计数据
        total_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        # 计算最近100个episode的平均奖励
        if len(total_rewards) >= 100:
            avg_reward = np.mean(total_rewards[-100:])
            avg_rewards.append(avg_reward)
            
            # 如果平均奖励提高，保存模型
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save_model("best_taxi_model.pth")
        else:
            avg_rewards.append(np.mean(total_rewards))
        
        # 计算接客和送客成功率
        pickup_success = env.successful_pickups / max(1, episode + 1)
        dropoff_success = env.successful_dropoffs / max(1, episode + 1)
        pickup_success_rate.append(pickup_success)
        dropoff_success_rate.append(dropoff_success)
        
        # 按指定间隔渲染环境和打印统计信息
        if episode % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Steps: {step_count}, "
                  f"Reward: {episode_reward:.2f}, Avg Reward: {avg_rewards[-1]:.2f}, "
                  f"Epsilon: {agent.epsilon:.4f}, Loss: {loss if loss else 'N/A'}")
            print(f"Successful Pickups: {env.successful_pickups}, "
                  f"Successful Dropoffs: {env.successful_dropoffs}, "
                  f"Pickup Success Rate: {pickup_success:.4f}, "
                  f"Dropoff Success Rate: {dropoff_success:.4f}")
        
        # 定期保存模型
        if episode % save_interval == 0 and episode > 0:
            agent.save_model(f"taxi_model_episode_{episode}.pth")
    
    # 返回训练数据
    return {
        'rewards': total_rewards,
        'avg_rewards': avg_rewards,
        'episode_lengths': episode_lengths,
        'pickup_success_rate': pickup_success_rate,
        'dropoff_success_rate': dropoff_success_rate
    }

# def student_agent_get_action(obs):
#     """
#     用于最终提交的函数，从环境观察返回行动
#     """
#     # 1. 加载模型
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # 创建一个临时agent对象，仅用于加载模型
#     state_dim = len(process_state_for_network(obs))
#     action_dim = 6
#     agent = DQNAgent(state_dim, action_dim, device)
    
#     # 加载预训练模型
#     agent.load_model("best_taxi_model.pth")
    
#     # 2. 处理观察
#     processed_obs = process_state_for_network(obs)
    
#     # 3. 选择动作（测试模式，不需要探索）
#     with torch.no_grad():
#         state_tensor = torch.FloatTensor(processed_obs).unsqueeze(0).to(device)
#         q_values = agent.policy_net(state_tensor)
#         action = q_values.max(1)[1].item()
    
#     return action
# 全局變量存儲模型
_model = None
_device = None
def get_action(obs):
    """
    評估函數：接收環境觀察並返回動作
    
    這是作業要求的主要函數，評估系統會調用此函數
    """
    global _model, _device
    
    # 首次調用時加載模型
    if _model is None:
        # 確定設備
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 計算狀態維度
        processed_obs = process_state_for_network(obs)
        state_dim = len(processed_obs)
        action_dim = 6
        
        # 創建模型
        _model = DuelingDQN(state_dim, action_dim).to(_device)
        
        # 嘗試加載模型，優先嘗試不同可能的路徑
        model_paths = [
            "best_taxi_model.pth",
            "final_taxi_model.pth",
            os.path.join(os.path.dirname(__file__), "best_taxi_model.pth"),
            os.path.join(os.path.dirname(__file__), "final_taxi_model.pth")
        ]
        
        for path in model_paths:
            try:
                checkpoint = torch.load(path, map_location=_device, weights_only=True)
                _model.load_state_dict(checkpoint['policy_net'])
                _model.eval()  # 設置為評估模式
                print(f"成功加載模型: {path}")
                break
            except Exception as e:
                continue
    
    # 處理觀察並選擇動作
    processed_obs = process_state_for_network(obs)
    
    with torch.no_grad():
        state_tensor = torch.FloatTensor(processed_obs).unsqueeze(0).to(_device)
        q_values = _model(state_tensor)
        action = q_values.max(1)[1].item()
    
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
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.99999)  # 加快epsilon衰減
    
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