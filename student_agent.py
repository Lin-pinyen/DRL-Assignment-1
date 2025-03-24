import gym
import numpy as np
import importlib.util
import time
import random
import math
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 使用 namedtuple 来存储经验
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

def get_optimal_distance(grid_size, start, end, obstacles):
    """使用 BFS 计算考虑障碍物的最短路径距离"""
    if start == end:
        return 0
    queue = deque([(start, 0)])  # (位置, 距离)
    visited = set([start])
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        (x, y), dist = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            new_pos = (nx, ny)
            if (0 <= nx < grid_size and 0 <= ny < grid_size and 
                new_pos not in obstacles and new_pos not in visited):
                if new_pos == end:
                    return dist + 1
                visited.add(new_pos)
                queue.append((new_pos, dist + 1))
    return float('inf')

class SimpleTaxiEnv():
    def __init__(self, grid_size=5, fuel_limit=200):
        self.grid_size = grid_size
        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit
        self.passenger_picked_up = False
        self.passenger_loc = None
        self.obstacles = set()  # 存储障碍物位置
        self.destination = None
        self.obstacle_density = 0.1  # 可调整的障碍物密度

        # 统计数据
        self.successful_pickups = 0
        self.successful_dropoffs = 0
        self.action_history = []
        
        # 记录最近到达过的位置，用于检测循环行为
        self.recent_positions = []
        self.position_history_limit = 20
        
    def is_connected(self, obstacles):
        """检查网格是否联通（使用 BFS 算法）"""
        if not obstacles:
            return True

        grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for obs in obstacles:
            grid[obs[0]][obs[1]] = 1

        start = None
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if grid[i][j] == 0:
                    start = (i, j)
                    break
            if start:
                break
        if not start:
            return False

        visited = set()
        queue = deque([start])
        visited.add(start)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x+dx, y+dy
                if (0 <= nx < self.grid_size and 0 <= ny < self.grid_size and 
                    grid[nx][ny]==0 and (nx, ny) not in visited):
                    queue.append((nx, ny))
                    visited.add((nx, ny))
        total_non_obstacles = self.grid_size * self.grid_size - len(obstacles)
        return len(visited) == total_non_obstacles

    def get_optimal_distance(self, start, end):
        """调用全局函数计算考虑障碍物的最短路径距离"""
        return get_optimal_distance(self.grid_size, start, end, self.obstacles)
    
    def reset(self):
        """重置环境，生成障碍物并确保图是联通的"""
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False
        self.action_history = []
        self.recent_positions = []
        self.obstacles = set()
        
        # 在 5x5 网格中放置一个障碍物（排除边界位置）
        if self.grid_size == 5:
            potential_obstacle_positions = [
                (i, j) for i in range(1, self.grid_size-1) for j in range(1, self.grid_size-1)
            ]
            if potential_obstacle_positions:
                obstacle_pos = random.choice(potential_obstacle_positions)
                self.obstacles.add(obstacle_pos)
                if not self.is_connected(self.obstacles):
                    self.obstacles.remove(obstacle_pos)
        
        # 随机选择 taxi、passenger 和 destination 的位置（排除障碍物）
        all_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)]
        available_positions = [pos for pos in all_positions if pos not in self.obstacles]
        sampled_positions = random.sample(available_positions, 3)
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

        self.recent_positions.append(self.taxi_pos)
        if len(self.recent_positions) > self.position_history_limit:
            self.recent_positions.pop(0)
            
        # 检测循环行为
        position_repeat_penalty = 0
        if len(self.recent_positions) > 5:
            position_counts = {}
            for pos in self.recent_positions:
                position_counts[pos] = position_counts.get(pos, 0) + 1
            for pos, count in position_counts.items():
                if count > 3:
                    position_repeat_penalty = -1.0 * count
        
        # 计算当前最优路径距离（考虑障碍物）
        if self.passenger_picked_up:
            old_optimal_distance = self.get_optimal_distance(self.taxi_pos, self.destination)
        else:
            old_optimal_distance = self.get_optimal_distance(self.taxi_pos, self.passenger_loc)
            
        if action in [0, 1, 2, 3]:
            if action == 0:  # Move Down
                next_row += 1
            elif action == 1:  # Move Up
                next_row -= 1
            elif action == 2:  # Move Right
                next_col += 1
            elif action == 3:  # Move Left
                next_col -= 1

            if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward -= 5
            else:
                self.taxi_pos = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos
            reward -= 0.05
            
            # 计算新位置的最优路径距离
            if self.passenger_picked_up:
                new_optimal_distance = self.get_optimal_distance(self.taxi_pos, self.destination)
                reward += 0.2
            else:
                new_optimal_distance = self.get_optimal_distance(self.taxi_pos, self.passenger_loc)
            
            if new_optimal_distance < old_optimal_distance:
                # 当距离最优路径缩短时给予正奖励
                reward += 2.0
            # 注意：远离目标时不再额外惩罚，避免绕路被误判
            
            if new_optimal_distance == 1:
                reward += 3.0
            elif new_optimal_distance == 0:
                if self.passenger_picked_up:
                    reward += 10.0
                else:
                    reward += 5.0
        else:
            if action == 4:  # PICKUP
                if self.passenger_picked_up:
                    reward -= 5
                elif self.taxi_pos == self.passenger_loc:
                    self.passenger_picked_up = True
                    self.passenger_loc = self.taxi_pos
                    reward += 30
                    self.successful_pickups += 1
                else:
                    reward -= 5
            elif action == 5:  # DROPOFF
                if self.passenger_picked_up:
                    if self.taxi_pos == self.destination:
                        reward += 500
                        done = True
                        self.successful_dropoffs += 1
                    else:
                        reward -= 30
        reward -= 0.05  # 基础操作惩罚
        self.current_fuel -= 1
        if self.current_fuel <= 0:
            reward -= 10
            done = True

        new_state = self.get_state()
        if old_state == new_state and old_pos == self.taxi_pos:
            reward -= 1
        
        self.action_history.append(action)
        if len(self.action_history) > 8:
            self.action_history.pop(0)
            
        reward += position_repeat_penalty
        if self.passenger_picked_up:
            reward += 0.2
        return new_state, reward, done, {}

    def get_state(self):
        """返回当前环境状态（不依赖于固定站点位置）"""
        taxi_row, taxi_col = self.taxi_pos
        distance_to_passenger = abs(taxi_row - self.passenger_loc[0]) + abs(taxi_col - self.passenger_loc[1])
        distance_to_destination = abs(taxi_row - self.destination[0]) + abs(taxi_col - self.destination[1])
        
        obstacle_north = int(taxi_row == 0 or (taxi_row - 1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row + 1, taxi_col) in self.obstacles)
        obstacle_east = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col + 1) in self.obstacles)
        obstacle_west = int(taxi_col == 0 or (taxi_row, taxi_col - 1) in self.obstacles)
        obstacles = [obstacle_north, obstacle_south, obstacle_east, obstacle_west]
        
        passenger_in_taxi = int(self.passenger_picked_up)
        passenger_adjacent = int(distance_to_passenger <= 1)
        destination_adjacent = int(distance_to_destination <= 1)
        
        state = tuple(obstacles + [passenger_in_taxi, distance_to_passenger,
                                     distance_to_destination, passenger_adjacent,
                                     destination_adjacent])
        return state

    def render_env(self, taxi_pos, action=None, step=None, fuel=None):
        clear_output(wait=True)
        grid = [['.'] * self.grid_size for _ in range(self.grid_size)]
        for obstacle_pos in self.obstacles:
            o_row, o_col = obstacle_pos
            if 0 <= o_row < self.grid_size and 0 <= o_col < self.grid_size:
                grid[o_row][o_col] = 'X'
        p_row, p_col = self.passenger_loc
        if 0 <= p_row < self.grid_size and 0 <= p_col < self.grid_size and not self.passenger_picked_up:
            grid[p_row][p_col] = 'P'
        d_row, d_col = self.destination
        if 0 <= d_row < self.grid_size and 0 <= d_col < self.grid_size:
            grid[d_row][d_col] = 'D'
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
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, state):
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

class DQNAgent:
    def __init__(self, state_dim, action_dim, device, lr=3e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9995):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Epsilon 策略参数
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 使用 Dueling DQN 网络
        self.policy_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 优化器及回放缓冲区
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(10000)
        self.batch_size = 64
        
        # 探索奖励相关
        self.state_counts = {}
        self.explore_coef = 0.5
        
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.5)
        
    def select_action(self, state):
        if random.random() < self.epsilon:
            action = random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
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
        
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
            expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        loss = F.smooth_l1_loss(q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        return loss.item()
    
    def update_target_network(self):
        tau = 0.01
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)
    
    def get_exploration_bonus(self, state):
        state_key = tuple(state)
        self.state_counts[state_key] = self.state_counts.get(state_key, 0) + 1
        bonus = self.explore_coef / math.sqrt(self.state_counts[state_key])
        return bonus
    
    def save_model(self, path):
        policy_net_cpu = {k: v.cpu() for k, v in self.policy_net.state_dict().items()}
        target_net_cpu = {k: v.cpu() for k, v in self.target_net.state_dict().items()}
        torch.save({
            'policy_net': policy_net_cpu,
            'target_net': target_net_cpu,
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path, _use_new_zipfile_serialization=True)
    
    def load_model(self, path):
        try:
            checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=True)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
        except Exception as e:
            print(f"加载模型时出错: {e}")

def process_state_for_network(state):
    """
    处理 get_state() 返回的状态:
      - 障碍物信息 (4个值)
      - 乘客是否在车上 (1个值)
      - 到乘客的距离 (1个值)
      - 到目的地的距离 (1个值)
      - 乘客是否在相邻位置 (1个值)
      - 目的地是否在相邻位置 (1个值)
    """
    max_distance = 10.0  # 对于5x5环境，最大距离约为8，稍微放宽一点
    obstacles = list(state[:4])
    passenger_in_taxi = [state[4]]
    distance_to_passenger = [state[5] / max_distance]
    distance_to_destination = [state[6] / max_distance]
    passenger_adjacent = [state[7]]
    destination_adjacent = [state[8]]
    processed_state = obstacles + passenger_in_taxi + distance_to_passenger + distance_to_destination + passenger_adjacent + destination_adjacent
    return processed_state

def train_dqn(env, agent, num_episodes=1000, save_interval=200, render_interval=100):
    total_rewards = []
    avg_rewards = []
    best_avg_reward = -float('inf')
    episode_lengths = []
    pickup_success_rate = []
    dropoff_success_rate = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        processed_state = process_state_for_network(state)
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done:
            action = agent.select_action(processed_state)
            next_state, reward, done, _ = env.step(action)
            processed_next_state = process_state_for_network(next_state)
            
            if episode < num_episodes // 2:
                bonus = agent.get_exploration_bonus(processed_next_state)
                total_reward = reward + bonus
            else:
                total_reward = reward
            
            agent.remember(processed_state, action, total_reward, processed_next_state, done)
            loss = agent.optimize_model()
            agent.update_target_network()
            
            processed_state = processed_next_state
            episode_reward += reward
            step_count += 1
            
            if step_count >= 100:
                done = True
        
        total_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        if len(total_rewards) >= 100:
            avg_reward = np.mean(total_rewards[-100:])
            avg_rewards.append(avg_reward)
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save_model("best_taxi_model.pth")
        else:
            avg_rewards.append(np.mean(total_rewards))
        
        pickup_success = env.successful_pickups / max(1, episode + 1)
        dropoff_success = env.successful_dropoffs / max(1, episode + 1)
        pickup_success_rate.append(pickup_success)
        dropoff_success_rate.append(dropoff_success)
        
        if episode % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}, Steps: {step_count}, Reward: {episode_reward:.2f}, Avg Reward: {avg_rewards[-1]:.2f}, Epsilon: {agent.epsilon:.4f}, Loss: {loss if loss else 'N/A'}")
            print(f"Pickups: {env.successful_pickups}, Dropoffs: {env.successful_dropoffs}, Pickup Rate: {pickup_success:.4f}, Dropoff Rate: {dropoff_success:.4f}")
        
        if episode % save_interval == 0 and episode > 0:
            agent.save_model(f"taxi_model_episode_{episode}.pth")
    
    return {
        'rewards': total_rewards,
        'avg_rewards': avg_rewards,
        'episode_lengths': episode_lengths,
        'pickup_success_rate': pickup_success_rate,
        'dropoff_success_rate': dropoff_success_rate
    }

def get_action(obs):
    """
    用于最终提交的函数，从环境观察返回行动
    """
    device = torch.device("cpu")
    if obs is None:
        return random.randint(0, 5)
    try:
        processed_obs = process_state_for_network(obs)
        state_dim = len(processed_obs)
        action_dim = 6
        agent = DQNAgent(state_dim, action_dim, device)
        try:
            agent.load_model("best_taxi_model.pth")
            with torch.no_grad():
                state_tensor = torch.FloatTensor(processed_obs).unsqueeze(0).to(device)
                q_values = agent.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
        except Exception as e:
            if len(obs) > 4 and obs[4]:
                dest_dist = obs[6] if len(obs) > 6 else 0
                if dest_dist == 0:
                    action = 5
                else:
                    obstacles = obs[:4] if len(obs) >= 4 else [0, 0, 0, 0]
                    if random.random() < 0.7:
                        valid_actions = [i for i in range(4) if obstacles[i] == 0]
                        action = random.choice(valid_actions) if valid_actions else random.randint(0, 3)
                    else:
                        action = random.randint(0, 5)
            else:
                pass_dist = obs[5] if len(obs) > 5 else 0
                if pass_dist == 0:
                    action = 4
                else:
                    obstacles = obs[:4] if len(obs) >= 4 else [0, 0, 0, 0]
                    if random.random() < 0.7:
                        valid_actions = [i for i in range(4) if obstacles[i] == 0]
                        action = random.choice(valid_actions) if valid_actions else random.randint(0, 3)
                    else:
                        action = random.randint(0, 5)
    except Exception as e:
        action = random.randint(0, 5)
    return action

if __name__ == "__main__":
    # 环境配置
    env_config = {
        "grid_size": 5,
        "fuel_limit": 1000
    }
    env = SimpleTaxiEnv(**env_config)
    
    sample_state, _ = env.reset()
    processed_sample = process_state_for_network(sample_state)
    state_dim = len(processed_sample)
    action_dim = 6
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    agent = DQNAgent(state_dim, action_dim, device, lr=5e-3, gamma=0.99,
                     epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.99999)
    
    train_results = train_dqn(env, agent, num_episodes=2000, save_interval=200)
    
    print("训练结束！")
    print(f"Total Successful Pickups: {env.successful_pickups}")
    print(f"Total Successful Dropoffs: {env.successful_dropoffs}")
    print(f"Final Pickup Success Rate: {env.successful_pickups/2000:.4f}")
    print(f"Final Dropoff Success Rate: {env.successful_dropoffs/2000:.4f}")
    
    num_params = sum(p.numel() for p in agent.policy_net.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in the DQN: {num_params}")
    
    agent.save_model("final_taxi_model.pth")
