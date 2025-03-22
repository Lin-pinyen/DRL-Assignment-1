import numpy as np
import random
import pickle
import time
# import matplotlib.pyplot as plt
from collections import defaultdict, deque
import os


class HierarchicalAgent:
    """
    Hierarchical Reinforcement Learning Agent for Taxi Environment
    Decomposes the task into 4 subtasks with specialized policies:
    1. Navigate to passenger
    2. Pick up passenger
    3. Navigate to destination
    4. Drop off passenger
    """
    def __init__(self):
        # Initialize sub-policies (options)
        self.navigate_to_passenger_policy = defaultdict(lambda: np.zeros(4) + 0.1)  # 4 movement actions
        self.pickup_policy = defaultdict(lambda: np.zeros(1) + 0.1)                 # 1 pickup action
        self.navigate_to_destination_policy = defaultdict(lambda: np.zeros(4) + 0.1) # 4 movement actions
        self.dropoff_policy = defaultdict(lambda: np.zeros(1) + 0.1)                # 1 dropoff action
        
        # Meta-policy to choose between options
        self.meta_policy = defaultdict(lambda: np.zeros(4) + 0.1)  # 4 options
        
        # Option names for debugging
        self.option_names = ["Navigate_to_Passenger", "Pickup", "Navigate_to_Destination", "Dropoff"]
        
        # Flag to indicate whether models are loaded
        self.loaded = False
        
    def load_policies(self, directory="./"):
        """Load all policies from files"""
        try:
            # Load meta-policy
            with open(os.path.join(directory, "meta_policy.pkl"), "rb") as f:
                self.meta_policy = defaultdict(lambda: np.zeros(4) + 0.1, pickle.load(f))
            
            # Load sub-policies
            with open(os.path.join(directory, "navigate_to_passenger_policy.pkl"), "rb") as f:
                self.navigate_to_passenger_policy = defaultdict(lambda: np.zeros(4) + 0.1, pickle.load(f))
                
            with open(os.path.join(directory, "pickup_policy.pkl"), "rb") as f:
                self.pickup_policy = defaultdict(lambda: np.zeros(1) + 0.1, pickle.load(f))
                
            with open(os.path.join(directory, "navigate_to_destination_policy.pkl"), "rb") as f:
                self.navigate_to_destination_policy = defaultdict(lambda: np.zeros(4) + 0.1, pickle.load(f))
                
            with open(os.path.join(directory, "dropoff_policy.pkl"), "rb") as f:
                self.dropoff_policy = defaultdict(lambda: np.zeros(1) + 0.1, pickle.load(f))
                
            self.loaded = True
            print("All policies loaded successfully")
        except Exception as e:
            print(f"Error loading policies: {e}")
            self.loaded = False
            
    def select_option(self, state):
        """Select which option to execute using meta-policy"""
        state_key = tuple(state)
        
        # Use hard-coded logic if policies not loaded or to override in certain situations
        passenger_in_taxi = state[0]
        distance_to_passenger = state[1]
        distance_to_destination = state[2]
        
        # If at passenger location without passenger, force pickup
        if not passenger_in_taxi and distance_to_passenger == 0:
            return 1  # Pickup option
        
        # If at destination with passenger, force dropoff
        if passenger_in_taxi and distance_to_destination == 0:
            return 3  # Dropoff option
        
        # Otherwise use meta policy
        if self.loaded:
            return np.argmax(self.meta_policy[state_key])
        else:
            # Fallback strategy if models aren't loaded
            if not passenger_in_taxi:
                return 0  # Navigate to passenger
            else:
                return 2  # Navigate to destination
    
    def select_action(self, state, option):
        """Select action based on the current option"""
        state_key = tuple(state)
        
        if option == 0:  # Navigate to passenger
            # Choose from the 4 movement actions (0-3)
            if self.loaded:
                action = np.argmax(self.navigate_to_passenger_policy[state_key])
            else:
                # Simple heuristic if policy not loaded
                passenger_dir_y = np.sign(state[5] - state[3])  # passenger y - taxi y
                passenger_dir_x = np.sign(state[6] - state[4])  # passenger x - taxi x
                
                # Prioritize y-axis movement
                if passenger_dir_y > 0:
                    action = 0  # Move south
                elif passenger_dir_y < 0:
                    action = 1  # Move north
                elif passenger_dir_x > 0:
                    action = 2  # Move east
                elif passenger_dir_x < 0:
                    action = 3  # Move west
                else:
                    action = 0  # Default
            return int(action)
            
        elif option == 1:  # Pickup
            return 4  # Pickup action
            
        elif option == 2:  # Navigate to destination
            # Choose from the 4 movement actions (0-3)
            if self.loaded:
                action = np.argmax(self.navigate_to_destination_policy[state_key])
            else:
                # Simple heuristic if policy not loaded
                dest_dir_y = np.sign(state[7] - state[3])  # destination y - taxi y
                dest_dir_x = np.sign(state[8] - state[4])  # destination x - taxi x
                
                # Prioritize y-axis movement
                if dest_dir_y > 0:
                    action = 0  # Move south
                elif dest_dir_y < 0:
                    action = 1  # Move north
                elif dest_dir_x > 0:
                    action = 2  # Move east
                elif dest_dir_x < 0:
                    action = 3  # Move west
                else:
                    action = 0  # Default
            return int(action)
            
        elif option == 3:  # Dropoff
            return 5  # Dropoff action
        
        # Default fallback
        return 0

def process_state(state):
    """
    Process and normalize the environment state for HRL
    Args:
        state: Raw state tuple from the environment
    Returns:
        processed_state: Tuple with normalized and processed features
    """
    # Extract state components
    distances_to_stations = list(state[:4])
    obstacles = list(state[4:8])
    passenger_in_taxi = state[8]
    distance_to_passenger = state[9]
    distance_to_destination = state[10]
    passenger_adjacent = state[11]
    destination_adjacent = state[12]
    
    # Max Manhattan distance in 5x5 grid
    max_distance = 8
    
    # Get taxi position relative to 0,0 (can derive this from distances to stations)
    # In a 5x5 grid, the corners are at (0,0), (0,4), (4,0), and (4,4)
    stations = [(0, 0), (0, 4), (4, 0), (4, 4)]
    
    # Estimate taxi position based on distances to stations
    # This is an approximation since we don't have the actual position
    taxi_row = 0
    taxi_col = 0
    
    # If we can match a pattern of distances that uniquely identifies a position, use it
    # Otherwise use a heuristic approach
    
    # Attempt to derive taxi position from distances to corners
    # In a square grid, we can use trilateration with Manhattan distances
    d1 = distances_to_stations[0]  # Distance to (0,0)
    d2 = distances_to_stations[1]  # Distance to (0,4)
    d3 = distances_to_stations[2]  # Distance to (4,0)
    
    # These equations work for Manhattan distance in a grid:
    # For a position (x,y), the Manhattan distances to corners satisfy:
    # d1 = x + y (distance to 0,0)
    # d2 = x + (4-y) (distance to 0,4) = x + 4 - y
    # d3 = (4-x) + y (distance to 4,0) = 4 - x + y
    
    # From d1 and d2:
    # d1 - d2 = x + y - (x + 4 - y) = 2y - 4
    # y = (d1 - d2 + 4)/2
    
    # From d1 and d3:
    # d1 - d3 = x + y - (4 - x + y) = 2x - 4
    # x = (d1 - d3 + 4)/2
    
    try:
        taxi_row = int((d1 - d2 + 4)/2)
        taxi_col = int((d1 - d3 + 4)/2)
        
        # Verify these values are in range
        if not (0 <= taxi_row < 5 and 0 <= taxi_col < 5):
            # If out of range, use fallback
            taxi_row = 2  # Middle of grid
            taxi_col = 2
    except:
        # Fallback if calculation fails
        taxi_row = 2
        taxi_col = 2
    
    # Passenger position can be inferred if it's at a station
    passenger_row, passenger_col = 0, 0
    if distance_to_passenger <= max_distance:
        for i, (r, c) in enumerate(stations):
            if distances_to_stations[i] == distance_to_passenger:
                passenger_row, passenger_col = r, c
                break
    
    # Destination position can be inferred if it's at a station
    destination_row, destination_col = 0, 0
    if distance_to_destination <= max_distance:
        for i, (r, c) in enumerate(stations):
            if distances_to_stations[i] == distance_to_destination:
                destination_row, destination_col = r, c
                break
    
    # Create processed state
    processed_state = (
        int(passenger_in_taxi),                      # 0: Is passenger in taxi?
        min(max_distance, int(distance_to_passenger)), # 1: Distance to passenger (capped)
        min(max_distance, int(distance_to_destination)), # 2: Distance to destination (capped)
        int(taxi_row),                                # 3: Taxi row
        int(taxi_col),                                # 4: Taxi column
        int(passenger_row),                          # 5: Passenger row
        int(passenger_col),                          # 6: Passenger column
        int(destination_row),                        # 7: Destination row
        int(destination_col),                        # 8: Destination column
        int(obstacles[0]),                           # 9: North obstacle
        int(obstacles[1]),                           # 10: South obstacle
        int(obstacles[2]),                           # 11: East obstacle
        int(obstacles[3])                            # 12: West obstacle
    )
    
    return processed_state

def get_action(obs):
    """
    Required function for submission - Gets action based on observation
    
    Args:
        obs: Raw observation from environment (state tuple)
        
    Returns:
        action: Integer action (0-5)
    """
    # Initialize agent if not already done
    if not hasattr(get_action, "agent"):
        get_action.agent = HierarchicalAgent()
        try:
            # Try to load policies
            get_action.agent.load_policies()
        except Exception as e:
            print(f"Error in agent initialization: {e}")
    
    # Process observation
    processed_obs = process_state(obs)
    
    # Select option using meta-policy
    option = get_action.agent.select_option(processed_obs)
    
    # Use fallback heuristic if no policy is loaded
    if not get_action.agent.loaded:
        # Direct hardcoded approach for better performance
        passenger_in_taxi = processed_obs[0]
        distance_to_passenger = processed_obs[1]
        distance_to_destination = processed_obs[2]
        
        taxi_row, taxi_col = processed_obs[3], processed_obs[4]
        passenger_row, passenger_col = processed_obs[5], processed_obs[6]
        destination_row, destination_col = processed_obs[7], processed_obs[8]
        obstacles = processed_obs[9:13]  # [North, South, East, West]
        
        # CASE 1: No passenger in taxi
        if not passenger_in_taxi:
            # At passenger location, pickup
            if distance_to_passenger == 0:
                return 4  # PICKUP
            
            # Navigate to passenger
            # Decide direction (try to minimize Manhattan distance)
            row_diff = passenger_row - taxi_row
            col_diff = passenger_col - taxi_col
            
            # Prioritize larger distance first
            if abs(row_diff) >= abs(col_diff):
                # Move vertically
                if row_diff > 0 and not obstacles[0]:  # Need to move South and not blocked
                    return 0  # Move South
                elif row_diff < 0 and not obstacles[1]:  # Need to move North and not blocked
                    return 1  # Move North
                # If vertical movement blocked, try horizontal
                elif col_diff > 0 and not obstacles[2]:  # Need to move East and not blocked
                    return 2  # Move East
                elif col_diff < 0 and not obstacles[3]:  # Need to move West and not blocked
                    return 3  # Move West
            else:
                # Move horizontally
                if col_diff > 0 and not obstacles[2]:  # Need to move East and not blocked
                    return 2  # Move East
                elif col_diff < 0 and not obstacles[3]:  # Need to move West and not blocked
                    return 3  # Move West
                # If horizontal movement blocked, try vertical
                elif row_diff > 0 and not obstacles[0]:  # Need to move South and not blocked
                    return 0  # Move South
                elif row_diff < 0 and not obstacles[1]:  # Need to move North and not blocked
                    return 1  # Move North
            
            # If all preferred directions are blocked, find any unblocked direction
            for i in range(4):
                if not obstacles[i]:
                    return i
            
            # All directions blocked (shouldn't happen in normal grid)
            return 0
            
        # CASE 2: Passenger in taxi
        else:
            # At destination, dropoff
            if distance_to_destination == 0:
                return 5  # DROPOFF
            
            # Navigate to destination
            row_diff = destination_row - taxi_row
            col_diff = destination_col - taxi_col
            
            # Prioritize larger distance first
            if abs(row_diff) >= abs(col_diff):
                # Move vertically
                if row_diff > 0 and not obstacles[0]:  # Need to move South and not blocked
                    return 0  # Move South
                elif row_diff < 0 and not obstacles[1]:  # Need to move North and not blocked
                    return 1  # Move North
                # If vertical movement blocked, try horizontal
                elif col_diff > 0 and not obstacles[2]:  # Need to move East and not blocked
                    return 2  # Move East
                elif col_diff < 0 and not obstacles[3]:  # Need to move West and not blocked
                    return 3  # Move West
            else:
                # Move horizontally
                if col_diff > 0 and not obstacles[2]:  # Need to move East and not blocked
                    return 2  # Move East
                elif col_diff < 0 and not obstacles[3]:  # Need to move West and not blocked
                    return 3  # Move West
                # If horizontal movement blocked, try vertical
                elif row_diff > 0 and not obstacles[0]:  # Need to move South and not blocked
                    return 0  # Move South
                elif row_diff < 0 and not obstacles[1]:  # Need to move North and not blocked
                    return 1  # Move North
            
            # If all preferred directions are blocked, find any unblocked direction
            for i in range(4):
                if not obstacles[i]:
                    return i
            
            # All directions blocked (shouldn't happen in normal grid)
            return 0
    
    # If policies are loaded, use them
    action = get_action.agent.select_action(processed_obs, option)
    return action
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
        self.successful_pickups = 0        # Total successful pickups across all episodes
        self.successful_dropoffs = 0       # Total successful dropoffs across all episodes
        self.episode_pickup_success = False  # Whether pickup was successful in current episode
        self.episode_dropoff_success = False # Whether dropoff was successful in current episode
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
                new_distance = abs(self.taxi_pos[0] - self.destination[0]) + abs(self.taxi_pos[1] - self.destination[1])
                
                if new_distance < old_distance:
                    reward += 2.0  # Reward for moving toward destination
                elif new_distance > old_distance:
                    reward -= 0.5  # Penalty for moving away
                
                # Extra reward for being close to destination
                if new_distance == 1:
                    reward += 3.0
                elif new_distance == 0:
                    reward += 10.0
            else:
                new_distance = abs(self.taxi_pos[0] - self.passenger_loc[0]) + abs(self.taxi_pos[1] - self.passenger_loc[1])
                
                if new_distance < old_distance:
                    reward += 1.5  # Reward for moving toward passenger
                elif new_distance > old_distance:
                    reward -= 0.4  # Penalty for moving away
                
                # Extra reward for being close to passenger
                if new_distance == 1:
                    reward += 2.0
                elif new_distance == 0:
                    reward += 5.0
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
                        reward += 500  # Successful dropoff - high reward
                        done = True
                        self.successful_dropoffs += 1
                    else:
                        reward -= 30  # Invalid dropoff location - big penalty
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

class HierarchicalTrainer:
    """
    Trainer for Hierarchical Reinforcement Learning Agent
    """
    def __init__(self, env, agent, save_dir="./"):
        self.env = env
        self.agent = agent
        self.save_dir = save_dir
        
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Learning parameters
        self.meta_alpha = 0.2   # Learning rate for meta-controller
        self.option_alpha = 0.3  # Learning rate for option policies
        self.gamma = 0.99       # Discount factor
        
        # Exploration parameters
        self.meta_epsilon = 0.3  # Exploration rate for meta-controller
        self.option_epsilon = 0.2  # Exploration rate for option policies
        self.epsilon_decay = 0.9999  # Epsilon decay rate
        self.min_epsilon = 0.01  # Minimum epsilon
        
        # Option termination criteria
        self.option_timeout = 10  # Maximum steps per option
        
        # Tracking variables
        self.meta_state_visits = defaultdict(int)
        self.option_state_visits = defaultdict(lambda: defaultdict(int))
        self.total_rewards = []
        self.episode_lengths = []
        self.successful_pickups = []
        self.successful_dropoffs = []
        
    def select_option(self, state, explore=True):
        """Select an option using meta-policy with exploration"""
        state_key = tuple(state)
        self.meta_state_visits[state_key] += 1
        
        # Hard-coded logic for certain states to speed up learning
        passenger_in_taxi = state[0]
        distance_to_passenger = state[1]
        distance_to_destination = state[2]
        
        # If at passenger location without passenger, force pickup
        if not passenger_in_taxi and distance_to_passenger == 0:
            return 1  # Pickup option
        
        # If at destination with passenger, force dropoff
        if passenger_in_taxi and distance_to_destination == 0:
            return 3  # Dropoff option
        
        # Exploration
        if explore and random.random() < self.meta_epsilon:
            if passenger_in_taxi:
                # If we have passenger, prioritize navigate-to-destination
                return 2 if random.random() < 0.8 else random.randint(0, 3)
            else:
                # If no passenger, prioritize navigate-to-passenger
                return 0 if random.random() < 0.8 else random.randint(0, 3)
        
        # Exploitation - choose best option
        return np.argmax(self.agent.meta_policy[state_key])
    
    def select_action(self, state, option, explore=True):
        """Select an action using the option's policy with exploration"""
        state_key = tuple(state)
        if option not in self.option_state_visits:
            self.option_state_visits[option] = defaultdict(int)
        self.option_state_visits[option][state_key] += 1
        
        # Determine action space based on option
        if option == 0:  # Navigate to passenger
            if explore and random.random() < self.option_epsilon:
                return random.randint(0, 3)  # 4 movement actions
            return np.argmax(self.agent.navigate_to_passenger_policy[state_key])
            
        elif option == 1:  # Pickup
            return 4  # Pickup action
            
        elif option == 2:  # Navigate to destination
            if explore and random.random() < self.option_epsilon:
                return random.randint(0, 3)  # 4 movement actions
            return np.argmax(self.agent.navigate_to_destination_policy[state_key])
            
        elif option == 3:  # Dropoff
            return 5  # Dropoff action
        
        # Default fallback
        return 0
    
    def should_terminate_option(self, state, option, step_count):
        """Determine if the current option should terminate"""
        # Timeout check
        if step_count >= self.option_timeout:
            return True
        
        passenger_in_taxi = state[0]
        distance_to_passenger = state[1]
        distance_to_destination = state[2]
        
        # Option-specific termination conditions
        if option == 0:  # Navigate to passenger
            # Terminate if at passenger or picked up passenger
            return distance_to_passenger == 0 or passenger_in_taxi
            
        elif option == 1:  # Pickup
            # Always terminate after one step
            return True
            
        elif option == 2:  # Navigate to destination
            # Terminate if at destination or lost passenger
            return distance_to_destination == 0 or not passenger_in_taxi
            
        elif option == 3:  # Dropoff
            # Always terminate after one step
            return True
        
        return False
    
    def update_meta_policy(self, state, option, reward, next_state, done):
        """Update the meta-controller policy"""
        state_key = tuple(state)
        next_state_key = tuple(next_state)
        
        # Adaptive learning rate based on visits
        effective_alpha = self.meta_alpha / (1 + 0.01 * self.meta_state_visits[state_key])
        
        # Current Q-value
        current_q = self.agent.meta_policy[state_key][option]
        
        # Next Q-value (max Q for next state)
        next_q = 0 if done else np.max(self.agent.meta_policy[next_state_key])
        
        # TD Update
        td_target = reward + self.gamma * next_q
        td_error = td_target - current_q
        
        # Update Q-value
        self.agent.meta_policy[state_key][option] += effective_alpha * td_error
    
    def update_option_policy(self, option, state, action, reward, next_state, done):
        """Update the option's policy"""
        state_key = tuple(state)
        next_state_key = tuple(next_state)
        
        # Adaptive learning rate based on visits
        effective_alpha = self.option_alpha / (1 + 0.01 * self.option_state_visits[option][state_key])
        
        # Update based on option
        if option == 0:  # Navigate to passenger
            # Adjust action index (0-3 for movement)
            action_idx = action
            
            # Current Q-value
            current_q = self.agent.navigate_to_passenger_policy[state_key][action_idx]
            
            # Next Q-value
            next_q = 0 if done else np.max(self.agent.navigate_to_passenger_policy[next_state_key])
            
            # TD Update
            td_target = reward + self.gamma * next_q
            td_error = td_target - current_q
            
            # Update Q-value
            self.agent.navigate_to_passenger_policy[state_key][action_idx] += effective_alpha * td_error
            
        elif option == 1:  # Pickup
            # Only one action (pickup = 4), so index is 0
            current_q = self.agent.pickup_policy[state_key][0]
            next_q = 0 if done else np.max(self.agent.pickup_policy[next_state_key])
            
            td_target = reward + self.gamma * next_q
            td_error = td_target - current_q
            
            self.agent.pickup_policy[state_key][0] += effective_alpha * td_error
            
        elif option == 2:  # Navigate to destination
            # Adjust action index (0-3 for movement)
            action_idx = action
            
            current_q = self.agent.navigate_to_destination_policy[state_key][action_idx]
            next_q = 0 if done else np.max(self.agent.navigate_to_destination_policy[next_state_key])
            
            td_target = reward + self.gamma * next_q
            td_error = td_target - current_q
            
            self.agent.navigate_to_destination_policy[state_key][action_idx] += effective_alpha * td_error
            
        elif option == 3:  # Dropoff
            # Only one action (dropoff = 5), so index is 0
            current_q = self.agent.dropoff_policy[state_key][0]
            next_q = 0 if done else np.max(self.agent.dropoff_policy[next_state_key])
            
            td_target = reward + self.gamma * next_q
            td_error = td_target - current_q
            
            self.agent.dropoff_policy[state_key][0] += effective_alpha * td_error
    
    def train(self, num_episodes=5000, max_steps=200):
        """Train the hierarchical agent"""
        print(f"Starting training for {num_episodes} episodes...")
        
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = process_state(state)
            
            episode_reward = 0
            step_count = 0
            done = False
            
            while not done and step_count < max_steps:
                # Select option using meta-policy
                option = self.select_option(state)
                
                # Execute option
                option_reward = 0
                option_step_count = 0
                option_terminal = False
                
                # Store initial state for the option
                option_initial_state = state
                
                while not done and not option_terminal and option_step_count < self.option_timeout:
                    # Select action using option policy
                    action = self.select_action(state, option)
                    
                    # Execute action in environment
                    next_state_raw, reward, done, _ = self.env.step(action)
                    next_state = process_state(next_state_raw)
                    
                    # Update option policy
                    self.update_option_policy(option, state, action, reward, next_state, done)
                    
                    # Accumulate option reward
                    option_reward += reward
                    episode_reward += reward
                    option_step_count += 1
                    step_count += 1
                    
                    # Update state
                    state = next_state
                    
                    # Check if option should terminate
                    option_terminal = self.should_terminate_option(state, option, option_step_count)
                
                # Update meta-policy with cumulative option reward
                self.update_meta_policy(option_initial_state, option, option_reward, state, done)
                
                # Decay exploration rates
                self.meta_epsilon = max(self.min_epsilon, self.meta_epsilon * self.epsilon_decay)
                self.option_epsilon = max(self.min_epsilon, self.option_epsilon * self.epsilon_decay)
            
            # Record episode results
            self.total_rewards.append(episode_reward)
            self.episode_lengths.append(step_count)
            self.successful_pickups.append(self.env.successful_pickups)
            self.successful_dropoffs.append(self.env.successful_dropoffs)
            
            # Print progress
            if (episode + 1) % 50 == 0:
                avg_reward = np.mean(self.total_rewards[-100:]) if len(self.total_rewards) >= 100 else np.mean(self.total_rewards)
                print(f"Episode {episode+1}/{num_episodes}: Reward = {episode_reward:.2f}, "
                      f"Avg Reward = {avg_reward:.2f}, Steps = {step_count}")
                print(f"Meta ε: {self.meta_epsilon:.4f}, Option ε: {self.option_epsilon:.4f}")
                print(f"Pickups: {self.env.successful_pickups}, Dropoffs: {self.env.successful_dropoffs}")
                print(f"Table sizes: Meta={len(self.agent.meta_policy)}, "
                      f"Nav.Pass={len(self.agent.navigate_to_passenger_policy)}, "
                      f"Nav.Dest={len(self.agent.navigate_to_destination_policy)}")
            
            # Save policies periodically
            if (episode + 1) % 500 == 0 or episode == num_episodes - 1:
                self.save_policies()
        
        # Save final policies
        self.save_policies()
        
        return {
            'rewards': self.total_rewards,
            'episode_lengths': self.episode_lengths,
            'pickups': self.successful_pickups,
            'dropoffs': self.successful_dropoffs
        }
    
    def save_policies(self):
        """Save all policies to files"""
        # Save meta-policy
        with open(os.path.join(self.save_dir, "meta_policy.pkl"), "wb") as f:
            pickle.dump(dict(self.agent.meta_policy), f)
        
        # Save option policies
        with open(os.path.join(self.save_dir, "navigate_to_passenger_policy.pkl"), "wb") as f:
            pickle.dump(dict(self.agent.navigate_to_passenger_policy), f)
        
        with open(os.path.join(self.save_dir, "pickup_policy.pkl"), "wb") as f:
            pickle.dump(dict(self.agent.pickup_policy), f)
        
        with open(os.path.join(self.save_dir, "navigate_to_destination_policy.pkl"), "wb") as f:
            pickle.dump(dict(self.agent.navigate_to_destination_policy), f)
        
        with open(os.path.join(self.save_dir, "dropoff_policy.pkl"), "wb") as f:
            pickle.dump(dict(self.agent.dropoff_policy), f)
        
        print("All policies saved successfully")
    
    def plot_results(self):
        """Plot training results"""
        plt.figure(figsize=(15, 10))
        
        # Plot rewards
        plt.subplot(2, 2, 1)
        plt.plot(self.total_rewards)
        plt.title('Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        
        # Plot moving average of rewards
        plt.subplot(2, 2, 2)
        window_size = 100
        if len(self.total_rewards) >= window_size:
            moving_avg = np.convolve(self.total_rewards, np.ones(window_size)/window_size, mode='valid')
            plt.plot(range(window_size-1, len(self.total_rewards)), moving_avg)
            plt.title(f'{window_size}-Episode Moving Average Reward')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
        
        # Plot successful pickups and dropoffs
        plt.subplot(2, 2, 3)
        plt.plot(self.successful_pickups, label='Pickups')
        plt.plot(self.successful_dropoffs, label='Dropoffs')
        plt.title('Successful Pickups and Dropoffs')
        plt.xlabel('Episode')
        plt.ylabel('Count')
        plt.legend()
        
        # Plot episode lengths
        plt.subplot(2, 2, 4)
        plt.plot(self.episode_lengths)
        plt.title('Episode Lengths')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_results.png'))
        plt.show()

def evaluate_agent(env, agent, num_episodes=100):
    """Evaluate agent performance"""
    total_rewards = []
    success_count = 0
    
    for i in range(num_episodes):
        state, _ = env.reset()
        state = process_state(state)
        done = False
        total_reward = 0
        
        while not done:
            # Select option
            option = agent.select_option(state)
            
            # Select action
            action = agent.select_action(state, option)
            
            # Take step in environment
            next_state, reward, done, _ = env.step(action)
            next_state = process_state(next_state)
            
            # Update state and rewards
            state = next_state
            total_reward += reward
            
            # Check for success
            if done and env.successful_dropoffs > 0:
                success_count += 1
        
        total_rewards.append(total_reward)
    
    avg_reward = np.mean(total_rewards)
    success_rate = success_count / num_episodes
    
    print(f"Evaluation results over {num_episodes} episodes:")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Success Rate: {success_rate:.2f}")
    
    return avg_reward, success_rate

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Create save directory
    save_dir = "./hrl_taxi_policies"
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize environment
    env = SimpleTaxiEnv(grid_size=5, fuel_limit=200)
    
    # Initialize hierarchical agent
    agent = HierarchicalAgent()
    
    # Initialize trainer
    trainer = HierarchicalTrainer(env, agent, save_dir=save_dir)
    
    # Train agent
    print("Starting training...")
    start_time = time.time()
    
    results = trainer.train(num_episodes=5000, max_steps=200)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Plot results
    trainer.plot_results()
    
    # Evaluate agent
    print("\nEvaluating agent...")
    avg_reward, success_rate = evaluate_agent(env, agent, num_episodes=100)
    
    # Print final statistics
    print("\nTraining Statistics:")
    print(f"Total Episodes: 5000")
    print(f"Final Meta-Policy Size: {len(agent.meta_policy)}")
    print(f"Final Navigate-to-Passenger Policy Size: {len(agent.navigate_to_passenger_policy)}")
    print(f"Final Navigate-to-Destination Policy Size: {len(agent.navigate_to_destination_policy)}")
    print(f"Total Successful Pickups: {env.successful_pickups}")
    print(f"Total Successful Dropoffs: {env.successful_dropoffs}")
    print(f"Pickup Success Rate: {env.successful_pickups/5000:.4f}")
    print(f"Dropoff Success Rate: {env.successful_dropoffs/5000:.4f}")