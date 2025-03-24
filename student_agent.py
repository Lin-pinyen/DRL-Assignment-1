import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import math
from collections import deque, namedtuple

# 使用namedtuple來存儲經驗，提高代碼可讀性
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
        self.obstacles = self.generate_obstacles()  # 生成障礙物
        self.destination = None

        # 統計數據
        self.successful_pickups = 0
        self.successful_dropoffs = 0
        self.action_history = []
        
        # 記錄最近到達過的位置，用於檢測循環行為
        self.recent_positions = []
        self.position_history_limit = 20
    
    def generate_obstacles(self):
        """生成三個障礙物，確保它們不與站點重疊並且地圖保持連通"""
        obstacles = set()
        available_positions = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size)
                              if (x, y) not in self.stations]
        
        # 函數檢查當前障礙物配置是否保持地圖連通
        def is_connected(obstacles):
            # BFS檢查連通性
            grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
            for ox, oy in obstacles:
                grid[ox][oy] = 1  # 標記障礙物
                
            # 選擇起點（例如第一個站點）
            start = self.stations[0]
            queue = [start]
            visited = {start}
            
            while queue:
                x, y = queue.pop(0)
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # 四個方向
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size and grid[nx][ny] == 0 and (nx, ny) not in visited:
                        queue.append((nx, ny))
                        visited.add((nx, ny))
            
            # 檢查所有站點是否可達
            for station in self.stations:
                if station not in visited:
                    return False
            
            return True
        
        # 嘗試添加障礙物，確保地圖保持連通
        obstacle_positions = random.sample(available_positions, 10)  # 隨機選擇10個位置作為候選
        
        for pos in obstacle_positions:
            temp_obstacles = obstacles.copy()
            temp_obstacles.add(pos)
            if is_connected(temp_obstacles) and len(obstacles) < 3:  # 限制為3個障礙物
                obstacles.add(pos)
                if len(obstacles) >= 3:
                    break
        
        return obstacles
        
    def reset(self):
        """重置環境，確保 Taxi、乘客與目的地互不重疊"""
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False
        self.action_history = []
        self.recent_positions = []
        
        # 重新生成障礙物
        self.obstacles = self.generate_obstacles()
        
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
        """更新環境狀態並返回 (state, reward, done, info)"""
        old_state = self.get_state()
        old_pos = self.taxi_pos
        
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0
        done = False

        # 添加當前位置到歷史記錄
        self.recent_positions.append(self.taxi_pos)
        if len(self.recent_positions) > self.position_history_limit:
            self.recent_positions.pop(0)
            
        # 檢測循環行為
        position_repeat_penalty = 0
        if len(self.recent_positions) > 5:
            position_counts = {}
            for pos in self.recent_positions:
                position_counts[pos] = position_counts.get(pos, 0) + 1
            # 如果某個位置重複超過3次，給予懲罰
            for pos, count in position_counts.items():
                if count > 3:
                    position_repeat_penalty = -0.5 * count  # 減輕重複位置的懲罰
            
        # 根據動作更新位置（移動動作）
        if action == 0:  # Move Down
            next_row += 1
        elif action == 1:  # Move Up
            next_row -= 1
        elif action == 2:  # Move Right
            next_col += 1
        elif action == 3:  # Move Left
            next_col -= 1

        if action in [0, 1, 2, 3]:
            # 計算距離（考慮最短可能路徑，而不是直線距離）
            if self.passenger_picked_up:
                target = self.destination
            else:
                target = self.passenger_loc
                
            # 使用A*搜索計算最佳路徑長度
            old_path_length = self.astar_path_length(self.taxi_pos, target)
            new_path_length = float('inf')  # 初始設置為無窮大
            
            # 檢查移動是否有效（不是牆壁或障礙物）
            if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward -= 5  # 撞牆或障礙物的懲罰
            else:
                self.taxi_pos = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos
                new_path_length = self.astar_path_length(self.taxi_pos, target)
            
            # 基礎移動懲罰，降低為-0.05，鼓勵更多探索
            reward -= 0.05
            
            # 更合理的shaping rewards，考慮A*路徑長度
            if self.passenger_picked_up:
                reward += 0.2  # 乘客在車上時的獎勵
                
                # 使用路徑長度改進獎勵機制
                if new_path_length < old_path_length:
                    reward += 1.5  # 獎勵朝目標移動
                elif new_path_length > old_path_length and new_path_length != float('inf'):
                    reward -= 0.2  # 減輕遠離目標的懲罰
                
                # 接近目的地時的額外獎勵
                if new_path_length == 1:  # 距離目的地僅1步
                    reward += 5.0
                elif self.taxi_pos == self.destination:  # 到達目的地
                    reward += 10.0  # 強烈鼓勵在有乘客時到達目的地
            else:
                # 使用路徑長度改進獎勵機制
                if new_path_length < old_path_length:
                    reward += 1.0
                elif new_path_length > old_path_length and new_path_length != float('inf'):
                    reward -= 0.2  # 減輕遠離目標的懲罰
                
                # 接近乘客時的額外獎勵
                if new_path_length == 1:  # 距離乘客僅1步
                    reward += 3.0
                elif self.taxi_pos == self.passenger_loc:  # 到達乘客位置
                    reward += 5.0  # 強烈鼓勵到達乘客位置
        else:
            # 非移動動作處理
            if action == 4:  # PICKUP
                if self.passenger_picked_up:
                    reward -= 5  # 減輕重複接客的懲罰
                elif self.taxi_pos == self.passenger_loc:
                    self.passenger_picked_up = True
                    self.passenger_loc = self.taxi_pos
                    reward += 30  # 增加接客獎勵
                    self.successful_pickups += 1
                else:
                    reward -= 5  # 減輕錯誤接客的懲罰
            elif action == 5:  # DROPOFF
                if self.passenger_picked_up:
                    if self.taxi_pos == self.destination:
                        reward += 500  # 提高從100到500，使其遠高於其他獎勵
                        done = True
                        self.successful_dropoffs += 1
                    else:
                        reward -= 10  # 減輕對錯誤地點下客的懲罰
                else:
                    reward -= 10  # 在沒有乘客時嘗試下客
        
        # 基礎操作懲罰
        reward -= 0.05

        # 扣除燃料
        self.current_fuel -= 1
        if self.current_fuel <= 0:
            reward -= 10
            done = True

        # 檢查新舊狀態是否相同
        new_state = self.get_state()
        if old_state == new_state and old_pos == self.taxi_pos:
            reward -= 0.5  # 減輕狀態未變的懲罰

        # 更新並檢查最近行動
        self.action_history.append(action)
        if len(self.action_history) > 8:
            self.action_history.pop(0)
            
        # 應用循環位置懲罰
        reward += position_repeat_penalty

         # 如果乘客在車上，每一步額外獎勵 +0.1
        if self.passenger_picked_up:
            reward += 0.2
        
        return new_state, reward, done, {}
    
    def astar_path_length(self, start, goal):
        """使用A*算法計算從start到goal的最短路徑長度，考慮障礙物"""
        if start == goal:
            return 0
            
        # 啟發式函數：曼哈頓距離
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        # A*搜索
        open_set = {start}
        closed_set = set()
        
        g_score = {start: 0}  # 從起點到當前節點的實際距離
        f_score = {start: heuristic(start, goal)}  # g_score + 啟發式估計
        
        open_heap = [(f_score[start], start)]
        
        while open_set:
            current = min(open_set, key=lambda x: f_score.get(x, float('inf')))
            
            if current == goal:
                return g_score[current]
            
            open_set.remove(current)
            closed_set.add(current)
            
            # 檢查四個方向的相鄰節點
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not (0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size):
                    continue  # 超出地圖範圍
                
                if neighbor in self.obstacles:
                    continue  # 障礙物
                    
                if neighbor in closed_set:
                    continue  # 已經評估過
                
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in open_set:
                    open_set.add(neighbor)
                elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                    continue
                
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
        
        return float('inf')  # 如果沒有找到路徑

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

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        
        # 特徵提取層
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        # 價值流
        self.value_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 優勢流
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, state):
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # 計算Q值: Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

class DQNAgent:
    def __init__(self, state_dim, action_dim, device, lr=3e-4, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.998):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Epsilon策略參數
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # 使用Dueling DQN網絡
        self.policy_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # 使用Adam優化器，學習率稍低一些
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # 更大的回放緩衝區
        self.memory = ReplayMemory(10000)
        self.batch_size = 64
        
        # 探索獎勵相關
        self.state_counts = {}
        self.explore_coef = 0.3  # 減小探索獎勵係數
        
        # 添加學習率調度器
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.5)
        
    def select_action(self, state):
        # Epsilon-貪婪策略選擇動作
        if random.random() < self.epsilon:
            action = random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                action = q_values.max(1)[1].item()
        
        # Epsilon遞減
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
        
        # 計算當前Q值
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        
        # Double DQN: 使用policy_net選擇action，使用target_net評估
        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).max(1)[1].unsqueeze(1)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
            expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        
        # 使用Huber Loss，對於異常值更魯棒
        loss = F.smooth_l1_loss(q_values, expected_q_values)
        
        # 梯度優化
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """使用軟更新策略更新目標網絡"""
        tau = 0.01  # 軟更新係數
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)
    
    def get_exploration_bonus(self, state):
        """計算探索獎勵"""
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
        """加載模型"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            checkpoint = torch.load(path, map_location=device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
        except:
            print("無法載入模型，使用初始化的模型")

def get_action(obs):
    """用於提交的主函數，從環境觀察返回行動"""
    # 加載模型
    try:
        # 根據狀態維度創建代理
        processed_obs = process_state_for_network(obs)
        state_dim = len(processed_obs)
        action_dim = 6
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 創建DQN代理
        agent = DQNAgent(state_dim, action_dim, device)
        
        # 嘗試加載預訓練模型
        agent.load_model("best_taxi_model.pth")
        
        # 選擇動作（測試模式，較少隨機探索）
        with torch.no_grad():
            state_tensor = torch.FloatTensor(processed_obs).unsqueeze(0).to(device)
            q_values = agent.policy_net(state_tensor)
            action = q_values.max(1)[1].item()
        
        return action
    
    except Exception as e:
        # 如果加載模型失敗，使用啟發式策略
        print(f"Error: {e}")
        return heuristic_action(obs)

def heuristic_action(obs):
    """當無法加載模型時的備用啟發式策略"""
    # 解析觀察值
    distances_to_stations = obs[:4]
    obstacles = obs[4:8]
    passenger_in_taxi = obs[8]
    distance_to_passenger = obs[9]
    distance_to_destination = obs[10]
    passenger_adjacent = obs[11]
    destination_adjacent = obs[12]
    
    # 如果乘客已在車上且鄰近目的地，則嘗試下客
    if passenger_in_taxi and destination_adjacent:
        return 5  # DROPOFF
    
    # 如果乘客未上車且鄰近乘客，則嘗試接客
    if not passenger_in_taxi and passenger_adjacent:
        return 4  # PICKUP
    
    # 避開障礙物的移動策略
    if not passenger_in_taxi:
        # 優先朝乘客方向移動
        if distance_to_passenger > 0:
            # 嘗試向乘客移動
            if distances_to_stations[0] < distances_to_stations[1] and not obstacles[0]:  # 南方無障礙
                return 0  # Move South
            elif distances_to_stations[0] > distances_to_stations[1] and not obstacles[1]:  # 北方無障礙
                return 1  # Move North
            elif distances_to_stations[2] < distances_to_stations[3] and not obstacles[2]:  # 東方無障礙
                return 2  # Move East
            elif distances_to_stations[2] > distances_to_stations[3] and not obstacles[3]:  # 西方無障礙
                return 3  # Move West
    else:
        # 優先朝目的地方向移動
        if distance_to_destination > 0:
            # 嘗試向目的地移動
            if distances_to_stations[0] < distances_to_stations[1] and not obstacles[0]:  # 南方無障礙
                return 0  # Move South
            elif distances_to_stations[0] > distances_to_stations[1] and not obstacles[1]:  # 北方無障礙
                return 1  # Move North
            elif distances_to_stations[2] < distances_to_stations[3] and not obstacles[2]:  # 東方無障礙
                return 2  # Move East
            elif distances_to_stations[2] > distances_to_stations[3] and not obstacles[3]:  # 西方無障礙
                return 3  # Move West
    
    # 如果無法決定，選擇任意無障礙方向
    for i in range(4):
        if not obstacles[i]:
            return i
    
    # 如果所有方向都有障礙，隨機選擇
    return random.randint(0, 3)

def process_state_for_network(state):
    """處理 get_state() 返回的狀態"""
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
    """訓練DQN代理"""
    total_rewards = []
    avg_rewards = []  # 保存平均獎勵
    best_avg_reward = -float('inf')
    episode_lengths = []
    
    # 每個episode的統計數據
    pickup_success_rate = []
    dropoff_success_rate = []
    
    # 每render_interval次評估一下模型
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
            # 選擇動作
            action = agent.select_action(processed_state)
            
            # 記錄嘗試的接客和送客
            if action == 4:  # PICKUP
                pickup_attempted = True
            elif action == 5:  # DROPOFF
                dropoff_attempted = True
            
            # 執行動作
            next_state, reward, done, _ = env.step(action)
            processed_next_state = process_state_for_network(next_state)
            
            # 計算探索獎勵（可選）
            if episode < num_episodes // 2:  # 只在前半部分訓練中使用探索獎勵
                bonus = agent.get_exploration_bonus(processed_next_state)
                total_reward = reward + bonus
            else:
                total_reward = reward
            
            # 存儲經驗
            agent.remember(processed_state, action, total_reward, processed_next_state, done)
            
            # 優化模型
            loss = agent.optimize_model()
            
            # 軟更新目標網絡
            agent.update_target_network()
            
            # 更新狀態
            processed_state = processed_next_state
            episode_reward += reward
            step_count += 1
            
            # 防止過長的episode
            if step_count >= 100:
                done = True
        
        # 記錄統計數據
        total_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        
        # 計算最近100個episode的平均獎勵
        if len(total_rewards) >= 100:
            avg_reward = np.mean(total_rewards[-100:])
            avg_rewards.append(avg_reward)
            
            # 如果平均獎勵提高，保存模型
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save_model("best_taxi_model.pth")
        else:
            avg_rewards.append(np.mean(total_rewards))
        
        # 計算接客和送客成功率
        pickup_success = env.successful_pickups / max(1, episode + 1)
        dropoff_success = env.successful_dropoffs / max(1, episode + 1)
        pickup_success_rate.append(pickup_success)
        dropoff_success_rate.append(dropoff_success)
        
        # 按指定間隔渲染環境和打印統計信息
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
    
    # 返回訓練數據
    return {
        'rewards': total_rewards,
        'avg_rewards': avg_rewards,
        'episode_lengths': episode_lengths,
        'pickup_success_rate': pickup_success_rate,
        'dropoff_success_rate': dropoff_success_rate
    }
# 主函数
if __name__ == "__main__":
    # 环境配置
    env_config = {
        "grid_size": 5,
        "fuel_limit": 5000
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