import numpy as np
import pickle
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import deque
import time
class QLearningAgent:
    """
    Q-Learning Agent for Taxi Environment
    Uses tabular approach instead of neural networks
    """
    def __init__(self, action_dim=6, learning_rate=0.2, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9999):
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Initialize Q-table with optimistic values to encourage exploration
        self.q_table = defaultdict(lambda: np.zeros(action_dim) + 1.0)
        
        # Track state visits for adaptive learning rates
        self.visit_counts = defaultdict(int)
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        state_key = tuple(state)
        self.visit_counts[state_key] += 1
        
        # Exploration
        if random.random() < self.epsilon:
            # Smart exploration - prioritize pickup/dropoff when at target
            passenger_in_taxi = state[0]
            dist_to_passenger = state[1]
            dist_to_destination = state[2]
            
            if passenger_in_taxi and dist_to_destination == 0:
                # At destination with passenger, try DROPOFF
                return 5 if random.random() < 0.8 else random.randint(0, 5)
            elif not passenger_in_taxi and dist_to_passenger == 0:
                # At passenger location without passenger, try PICKUP
                return 4 if random.random() < 0.8 else random.randint(0, 5)
            else:
                return random.randint(0, 5)
        else:
            # Exploitation - choose best action
            return int(np.argmax(self.q_table[state_key]))
    
    def update(self, state, action, reward, next_state, done):
        """Update Q-value using Q-learning update rule"""
        state_key = tuple(state)
        next_state_key = tuple(next_state)
        
        # Adaptive learning rate - decreases with more visits
        effective_lr = self.lr / (1 + 0.05 * self.visit_counts[state_key])
        
        # Q-learning update formula
        best_next_action = np.argmax(self.q_table[next_state_key])
        td_target = reward + (1 - done) * self.gamma * self.q_table[next_state_key][best_next_action]
        td_error = td_target - self.q_table[state_key][action]
        self.q_table[state_key][action] += effective_lr * td_error
    
    def decay_epsilon(self):
        """Decay epsilon value for exploration"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath="q_table.pkl"):
        """Save Q-table to file"""
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
    
    def load(self, filepath="q_table.pkl"):
        """Load Q-table from file"""
        try:
            with open(filepath, 'rb') as f:
                table = pickle.load(f)
                self.q_table = defaultdict(lambda: np.zeros(self.action_dim), table)
            print(f"Q-table loaded from {filepath}")
            print(f"Q-table size: {len(self.q_table)}")
        except Exception as e:
            print(f"Error loading Q-table: {e}")

def process_state(state):
    """
    Process the environment state into a simplified representation
    for the Q-table to reduce state space and improve generalization
    """
    # Extract components from state
    distances_to_stations = state[:4]
    obstacles = state[4:8]
    passenger_in_taxi = state[8]
    distance_to_passenger = state[9]
    distance_to_destination = state[10]
    passenger_adjacent = state[11]
    destination_adjacent = state[12]
    
    # Discretize distances (0-5 only)
    disc_passenger_dist = min(5, int(distance_to_passenger))
    disc_destination_dist = min(5, int(distance_to_destination))
    
    # Create simplified state representation
    simplified_state = (
        int(passenger_in_taxi),                     # Is passenger in taxi? (0/1)
        disc_passenger_dist if not passenger_in_taxi else 0,  # Distance to passenger
        disc_destination_dist if passenger_in_taxi else 0,    # Distance to destination
        int(passenger_adjacent),                    # Is passenger adjacent? (0/1)
        int(destination_adjacent),                  # Is destination adjacent? (0/1)
        int(obstacles[0]),                          # North obstacle (0/1)
        int(obstacles[1]),                          # South obstacle (0/1)
        int(obstacles[2]),                          # East obstacle (0/1)
        int(obstacles[3])                           # West obstacle (0/1)
    )
    
    return simplified_state

def train(env, agent, num_episodes=5000, max_steps=100):
    """Train the Q-learning agent"""
    rewards_history = []
    best_avg_reward = -float('inf')
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = process_state(state)
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            next_state = process_state(next_state)
            
            # Update Q-table
            agent.update(state, action, reward, next_state, done)
            
            # Move to next state
            state = next_state
            total_reward += reward
            step += 1
        
        # Decay exploration rate
        agent.decay_epsilon()
        
        # Record results
        rewards_history.append(total_reward)
        
        # Save best model
        if len(rewards_history) >= 100:
            avg_reward = np.mean(rewards_history[-100:])
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save("best_q_table.pkl")
        
        # Print progress
        if episode % 10 == 0:
            avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
            pickup_rate = env.successful_pickups / (episode + 1)
            dropoff_rate = env.successful_dropoffs / (episode + 1)
            
            print(f"Episode {episode}: Reward = {total_reward:.2f}, Avg = {avg_reward:.2f}, ε = {agent.epsilon:.4f}")
            print(f"Pickups: {env.successful_pickups}, Dropoffs: {env.successful_dropoffs}")
            print(f"Pickup Rate: {pickup_rate:.4f}, Dropoff Rate: {dropoff_rate:.4f}")
            print(f"Q-table size: {len(agent.q_table)}")
        
        # Save intermediate models
        if episode % 500 == 0 and episode > 0:
            agent.save(f"q_table_episode_{episode}.pkl")
    
    # Save final model
    agent.save("final_q_table.pkl")
    
    return rewards_history

def get_action(obs):
    """
    Function for submission - Returns action based on observation
    
    Args:
        obs: Environment observation
        
    Returns:
        action: Integer action (0-5)
    """
    # Initialize agent if needed (only on first call)
    if not hasattr(get_action, "agent"):
        get_action.agent = QLearningAgent(action_dim=6)
        try:
            # Try to load pre-trained Q-table
            get_action.agent.load("best_q_table.pkl")
        except:
            print("Could not load Q-table, using fallback strategy")
        
    # Process observation
    processed_obs = process_state(obs)
    state_key = tuple(processed_obs)
    
    # Check if we have this state in our Q-table
    if state_key in get_action.agent.q_table:
        # Use learned policy
        return int(np.argmax(get_action.agent.q_table[state_key]))
    else:
        # Fallback strategy for unseen states
        passenger_in_taxi = processed_obs[0]
        dist_to_passenger = processed_obs[1]
        dist_to_destination = processed_obs[2]
        passenger_adjacent = processed_obs[3]
        destination_adjacent = processed_obs[4]
        obstacles = processed_obs[5:9]
        
        if passenger_in_taxi:
            # Passenger is in taxi
            if dist_to_destination == 0:
                return 5  # DROPOFF
            elif destination_adjacent:
                return 5  # Try DROPOFF when adjacent
            else:
                # Move in any non-obstacle direction
                for i, is_obstacle in enumerate(obstacles):
                    if not is_obstacle:
                        return i  # Return movement action
                # All directions blocked, try anything
                return random.randint(0, 3)
        else:
            # Passenger not in taxi
            if dist_to_passenger == 0:
                return 4  # PICKUP
            elif passenger_adjacent:
                return 4  # Try PICKUP when adjacent
            else:
                # Move in any non-obstacle direction
                for i, is_obstacle in enumerate(obstacles):
                    if not is_obstacle:
                        return i  # Return movement action
                return random.randint(0, 3)
class SimpleTaxiEnv():
    def __init__(self, grid_size=5, fuel_limit=200):
        self.grid_size = grid_size
        self.fuel_limit = fuel_limit
        self.current_fuel = fuel_limit
        self.passenger_picked_up = False
        self.stations = [(0, 0), (0, self.grid_size - 1),
                         (self.grid_size - 1, 0), (self.grid_size - 1, self.grid_size - 1)]
        self.passenger_loc = None
        self.obstacles = set()  # No obstacles in simplified version
        self.destination = None

        # Statistics
        self.successful_pickups = 0
        self.successful_dropoffs = 0
        self.recent_positions = []
        self.position_history_limit = 20
        
    def reset(self):
        """Reset the environment"""
        self.current_fuel = self.fuel_limit
        self.passenger_picked_up = False
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
        """Update environment state and return (state, reward, done, info)"""
        old_state = self.get_state()
        old_pos = self.taxi_pos
        
        taxi_row, taxi_col = self.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0
        done = False

        # Add position to history
        self.recent_positions.append(self.taxi_pos)
        if len(self.recent_positions) > self.position_history_limit:
            self.recent_positions.pop(0)
            
        # Check for repeating positions
        position_repeat_penalty = 0
        if len(self.recent_positions) > 5:
            position_counts = {}
            for pos in self.recent_positions:
                position_counts[pos] = position_counts.get(pos, 0) + 1
            # Penalize repeating positions
            for pos, count in position_counts.items():
                if count > 3:
                    position_repeat_penalty = -1.0 * count
            
        # Handle movement actions
        if action <= 3:  # Movement actions
            # Calculate distances
            if self.passenger_picked_up:
                old_distance = abs(taxi_row - self.destination[0]) + abs(taxi_col - self.destination[1])
            else:
                old_distance = abs(taxi_row - self.passenger_loc[0]) + abs(taxi_col - self.passenger_loc[1])
        
            # Update position based on action
            if action == 0:  # Move South
                next_row += 1
            elif action == 1:  # Move North
                next_row -= 1
            elif action == 2:  # Move East
                next_col += 1
            elif action == 3:  # Move West
                next_col -= 1

            # Handle movement
            if (next_row, next_col) in self.obstacles or not (0 <= next_row < self.grid_size and 0 <= next_col < self.grid_size):
                reward -= 5  # Penalty for hitting wall/obstacle
            else:
                self.taxi_pos = (next_row, next_col)
                if self.passenger_picked_up:
                    self.passenger_loc = self.taxi_pos
            
            # Movement penalty
            reward -= 0.1
            
            # Shaping rewards
            if self.passenger_picked_up:
                reward += 0.2  # Small reward for having passenger
                new_distance = abs(self.taxi_pos[0] - self.destination[0]) + abs(self.taxi_pos[1] - self.destination[1])
                
                if new_distance < old_distance:
                    reward += 1.0  # Reward for moving toward destination
                elif new_distance > old_distance:
                    reward -= 0.5  # Penalty for moving away
                
                # Extra reward for being close to destination
                if new_distance == 1:
                    reward += 2.0
                elif new_distance == 0:
                    reward += 5.0
            else:
                new_distance = abs(self.taxi_pos[0] - self.passenger_loc[0]) + abs(self.taxi_pos[1] - self.passenger_loc[1])
                
                if new_distance < old_distance:
                    reward += 1.0  # Reward for moving toward passenger
                elif new_distance > old_distance:
                    reward -= 0.5  # Penalty for moving away
                
                # Extra reward for being close to passenger
                if new_distance == 1:
                    reward += 2.0
                elif new_distance == 0:
                    reward += 3.0
        else:
            # Handle pickup/dropoff actions
            if action == 4:  # PICKUP
                if self.passenger_picked_up:
                    reward -= 5  # Already have passenger
                elif self.taxi_pos == self.passenger_loc:
                    self.passenger_picked_up = True
                    self.passenger_loc = self.taxi_pos
                    reward += 30  # Successful pickup
                    self.successful_pickups += 1
                else:
                    reward -= 5  # Invalid pickup
            elif action == 5:  # DROPOFF
                if self.passenger_picked_up:
                    if self.taxi_pos == self.destination:
                        reward += 100  # Successful dropoff
                        done = True
                        self.successful_dropoffs += 1
                    else:
                        reward -= 10  # Invalid dropoff location
                else:
                    reward -= 5  # No passenger to drop off

        # Reduce fuel
        self.current_fuel -= 1
        if self.current_fuel <= 0:
            reward -= 10
            done = True

        # Check if state unchanged
        new_state = self.get_state()
        if old_state == new_state and old_pos == self.taxi_pos:
            reward -= 1  # Penalty for no change

        # Apply position repeat penalty
        reward += position_repeat_penalty

        # Extra reward for having passenger
        if self.passenger_picked_up:
            reward += 0.2
            
        return new_state, reward, done, {}

    def get_state(self):
        """Return current environment state"""
        taxi_row, taxi_col = self.taxi_pos
        
        # Calculate Manhattan distances to stations
        distances_to_stations = [abs(taxi_row - station[0]) + abs(taxi_col - station[1]) for station in self.stations]
        
        # Calculate Manhattan distance to passenger
        distance_to_passenger = abs(taxi_row - self.passenger_loc[0]) + abs(taxi_col - self.passenger_loc[1])
        
        # Calculate Manhattan distance to destination
        distance_to_destination = abs(taxi_row - self.destination[0]) + abs(taxi_col - self.destination[1])
        
        # Check for obstacles in each direction
        obstacle_north = int(taxi_row == 0 or (taxi_row - 1, taxi_col) in self.obstacles)
        obstacle_south = int(taxi_row == self.grid_size - 1 or (taxi_row + 1, taxi_col) in self.obstacles)
        obstacle_east = int(taxi_col == self.grid_size - 1 or (taxi_row, taxi_col + 1) in self.obstacles)
        obstacle_west = int(taxi_col == 0 or (taxi_row, taxi_col - 1) in self.obstacles)
        obstacles = [obstacle_north, obstacle_south, obstacle_east, obstacle_west]
        
        # Passenger status
        passenger_in_taxi = int(self.passenger_picked_up)
        
        # Check if passenger or destination is adjacent
        passenger_adjacent = int(distance_to_passenger <= 1)
        destination_adjacent = int(distance_to_destination <= 1)
        
        # Combine all features
        state = tuple(distances_to_stations + obstacles + 
                    [passenger_in_taxi, distance_to_passenger, distance_to_destination,
                    passenger_adjacent, destination_adjacent])
        
        return state

def evaluate_agent(env, agent, num_episodes=50):
    """Evaluate agent performance without exploration"""
    total_rewards = []
    success_count = 0
    
    for i in range(num_episodes):
        state, _ = env.reset()
        state = process_state(state)
        done = False
        total_reward = 0
        step = 0
        
        while not done and step < 100:
            # Use greedy action selection
            state_key = tuple(state)
            if state_key in agent.q_table:
                action = np.argmax(agent.q_table[state_key])
            else:
                # Fallback for unseen states
                action = random.randint(0, 5)
            
            next_state, reward, done, _ = env.step(action)
            next_state = process_state(next_state)
            
            state = next_state
            total_reward += reward
            step += 1
            
            if done and reward > 50:  # Successfully completed task
                success_count += 1
        
        total_rewards.append(total_reward)
    
    avg_reward = np.mean(total_rewards)
    success_rate = success_count / num_episodes
    
    return avg_reward, success_rate

def plot_training_progress(rewards, window_size=100):
    """Plot training progress"""
    plt.figure(figsize=(12, 6))
    
    # Plot episode rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.6, color='blue', label='Episode Reward')
    
    # Plot moving average
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, color='red', label=f'{window_size}-Episode Moving Avg')
    
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    plt.show()

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Initialize environment
    env = SimpleTaxiEnv(grid_size=5, fuel_limit=200)
    
    # Create Q-learning agent
    agent = QLearningAgent(
        action_dim=6,
        learning_rate=0.2,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.9999
    )
    
    # Train agent
    print("Starting Q-learning training...")
    start_time = time.time()
    
    rewards = train(env, agent, num_episodes=10000, max_steps=100)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate agent
    print("\nEvaluating agent performance...")
    avg_reward, success_rate = evaluate_agent(env, agent, num_episodes=100)
    print(f"Evaluation - Avg Reward: {avg_reward:.2f}, Success Rate: {success_rate:.2f}")
    
    # Plot training progress
    plot_training_progress(rewards)
    
    # Print training statistics
    print("\nTraining Statistics:")
    print(f"Final Q-table size: {len(agent.q_table)}")
    print(f"Total successful pickups: {env.successful_pickups}")
    print(f"Total successful dropoffs: {env.successful_dropoffs}")
    print(f"Pickup success rate: {env.successful_pickups/10000:.4f}")
    print(f"Dropoff success rate: {env.successful_dropoffs/10000:.4f}")