"""
OT-2 Digital Twin Environment Wrapper for Gymnasium

This module provides a Gymnasium environment wrapper for the OT-2 digital twin PyBullet environment.
The wrapper enables training reinforcement learning agents using Stable Baselines 3 to control
the OT-2 robot's movement to target positions within its working envelope.

Key Features:
    - Custom action space for controlling x, y, z velocities
    - Observation space including current position, goal position, and movement metrics
    - Reward function encouraging efficient movement to target positions
    - Boundary checking and episode termination conditions

Example:
    env = OT2Env(render=True)
    obs, info = env.reset()
    for _ in range(1000):
        action = env.action_space.sample()  # Replace with your agent's action
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
"""

# Import the libraries
import gymnasium as gym
import numpy as np
import random
from ot2_twin.sim_class import Simulation
from stable_baselines3.common.env_checker import check_env
from typing import Optional, Tuple

class OT2Env(gym.Env):
    """
    A Gymnasium environment wrapper for the OT-2 digital twin simulation.
    
    This environment trains an agent to move the OT-2's pipette to target positions
    within its working envelope. The agent controls the pipette's velocity in x, y, z
    directions and receives rewards based on its distance to the goal and movement efficiency.
    
    Attributes:
        action_space (gym.spaces.Box): Continuous action space for x, y, z velocities [-1, 1]
        observation_space (gym.spaces.Box): Space containing current position, goal position,
            distance to goal, and distance improvement
        x_bounds (tuple): Min and max x coordinates of working envelope
        y_bounds (tuple): Min and max y coordinates of working envelope
        z_bounds (tuple): Min and max z coordinates of working envelope
    """

    def __init__(self, render: bool = False, max_steps: int = 1000, random_initial_position: bool = False) -> None:
        """
        Initialize the OT-2 environment.

        Args:
            render (bool): Whether to render the simulation visually
            max_steps (int): Maximum number of steps per episode before truncation
        """
        # Inherit the parent class
        super(OT2Env, self).__init__()

        # Initialize visualizer (will be set externally if needed)
        self.visualizer = None

        # Target precision
        self.target_precision = 0.001

        #
        self.random_initial_position = random_initial_position

        ###########################
        #    Action Space        #
        ###########################

        # Action space: normalized velocities [-1, 1] for x, y, z axes
        # These values are scaled appropriately by the simulation
        self.action_space = gym.spaces.Box(
            low = -1, high = 1, shape = (3,), dtype = np.float32
        )

        # Action smoothing parameters
        self.max_velocity = 5     # m/s
        self.max_acceleration = 1.0  # Increased significantly to allow faster changes
        self.prev_action = np.zeros(3, dtype=np.float32)

        ###########################
        #    Observation Space    #
        ###########################

        # Working envelope - physical limits of the robot's movement
        self.x_bounds = (-0.1871, 0.253)    # meters
        self.y_bounds = (-0.1705, 0.2195)   # meters
        self.z_bounds = (0.1195, 0.2896)    # meters

        # Initialize state variables
        self.prev_distance = None
        self.prev_pos = None
        self.initial_pos = None  # Store the true initial position
        self.dt = 1/240  # PyBullet default timestep
        self.eps = 1e-6  # Small epsilon to handle numerical precision
        
        # Observation space contains:
        # 1. Relative position to goal (x, y, z)
        # 2. Current velocity (x, y, z)
        # 3. Normalized position in workspace [0,1] (x, y, z)
        # 4. Distance to goal
        # 5. Distance difference per coordinate (x, y, z)
        self.observation_space = gym.spaces.Box(
            low=np.array([
                -np.inf, -np.inf, -np.inf,  # Relative position
                -np.inf, -np.inf, -np.inf,  # Velocity
                0.0, 0.0, 0.0,              # Normalized position
                0.0,                        # Distance
                -np.inf, -np.inf, -np.inf,  # Distance difference per coordinate
            ], dtype=np.float32),
            high=np.array([
                np.inf, np.inf, np.inf,     # Relative position
                np.inf, np.inf, np.inf,     # Velocity
                1.0, 1.0, 1.0,              # Normalized position
                np.inf,                     # Distance
                np.inf, np.inf, np.inf,     # Distance difference per coordinate
            ], dtype=np.float32),
            dtype=np.float32
        )

        ###################
        #    Simulation   #
        ###################

        # Simulation parameters
        self.render = render
        self.steps = 0
        self.max_steps = max_steps
        self.episode_count = 0
        self.episode_reward = 0.0
        
        # Create simulation instance
        self.sim = Simulation(num_agents=1, render=render, rgb_array=False)
        self.reset()

    def reset(self, seed: Optional[int] = None, goal_pos: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset the environment to an initial state.
        """
        # Set seed for reproducibility
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        # Reset counters
        self.steps = 0
        self.episode_reward = 0.0
        self.episode_count += 1

        # Reset the simulation
        self.sim.reset(num_agents=1)

        # Get the robot ID
        self.robot_id = self.sim.robotIds[0]

        # Generate random goal if not provided
        if goal_pos is None:
            self.goal_pos = np.array([
                random.uniform(*self.x_bounds),
                random.uniform(*self.y_bounds),
                random.uniform(*self.z_bounds)
            ], dtype=np.float32)
        else:
            self.goal_pos = np.array(goal_pos, dtype=np.float32)

        # Get initial position
        if self.random_initial_position==True:
            # Generate random initial position
            initial_pos = np.array([
                random.uniform(*self.x_bounds),
                random.uniform(*self.y_bounds),
                random.uniform(*self.z_bounds)
            ], dtype=np.float32)
        else:
            # Use the actual default pipette position
            initial_pos = np.array([0.073, 0.090, 0.119], dtype=np.float32)
            
        # Set initial position using existing method
        self.sim.set_start_position(initial_pos[0], initial_pos[1], initial_pos[2])
        self.initial_pos = initial_pos.copy()  # Store the true initial position
        
        # Now get the observation after setting the position
        observation = self.sim.get_states()

        # Calculate relative position and normalized position
        relative_pos = initial_pos - self.goal_pos
        
        # Normalize positions with epsilon and clipping
        normalized_pos = np.clip(np.array([
            (initial_pos[0] - self.x_bounds[0]) / (self.x_bounds[1] - self.x_bounds[0] + self.eps),
            (initial_pos[1] - self.y_bounds[0]) / (self.y_bounds[1] - self.y_bounds[0] + self.eps),
            (initial_pos[2] - self.z_bounds[0]) / (self.z_bounds[1] - self.z_bounds[0] + self.eps)
        ], dtype=np.float32), 0.0, 1.0)
        
        # Debug prints only in test mode
        if hasattr(self, '_testing') and self._testing:
            print("\nDebug Normalization:")
            print(f"Initial position: {initial_pos}")
            print(f"X bounds: {self.x_bounds}")
            print(f"Y bounds: {self.y_bounds}")
            print(f"Z bounds: {self.z_bounds}")
            print(f"Normalized position: {normalized_pos}")

        # Initial velocity is zero
        initial_vel = np.zeros(3, dtype=np.float32)
        self.prev_pos = initial_pos.copy()

        # Calculate initial distance
        initial_distance = np.linalg.norm(relative_pos)
        self.initial_distance = initial_distance  # Store the true initial distance
        self.prev_distance = initial_distance
        self.prev_relative_pos = relative_pos.copy()  # Store previous relative position

        # Construct observation
        observation = np.concatenate([
            relative_pos,       # Relative position to goal
            initial_vel,        # Initial velocity (zero)
            normalized_pos,     # Normalized position in workspace
            [initial_distance], # Distance to goal
            np.zeros(3, dtype=np.float32)  # Initial distance difference per coordinate (zero)
        ]).astype(np.float32)

        return observation, {}

    def step(self, action: np.ndarray, injection_action: float = 0) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one timestep in the environment.
        
        This method:
        1. Executes the provided action
        2. Updates the environment state
        3. Calculates rewards
        4. Determines if the episode should end
        
        The reward function considers:
        - Distance to goal (negative reward)
        - Distance improvement (positive reward)
        - Time penalty (small negative reward)
        - Success bonus (large positive reward)
        - Boundary penalty (negative reward if outside bounds)
        
        Args:
            action (np.ndarray): Array of normalized velocities for x, y, z axes [-1, 1]
            
        Returns:
            Tuple containing:
            - observation (np.ndarray): Current state observation
            - reward (float): Reward for the current step
            - terminated (bool): Whether the episode has ended
            - truncated (bool): Whether the episode was truncated
            - info (dict): Additional information
        """
        #####################
        #    TAKE ACTION    #
        #####################

        # If we have a hybrid controller, blend PID and RL actions
        if hasattr(self, 'hybrid_controller'):
            # Get current position and calculate PID action
            current_pos = np.array(self.sim.get_states()[f'robotId_{self.robot_id}']['pipette_position'])
            pid_action = self.hybrid_controller.pid_controller.compute(current_pos, self.goal_pos, self.hybrid_controller.dt)
            
            # Blend actions
            action = self.hybrid_controller.blend_actions(pid_action, action)
            
            # Update alpha based on training progress
            if hasattr(self, 'steps'):
                progress = min(1.0, self.steps / (self.max_steps * 500))  # Use 100 episodes worth of steps
                self.hybrid_controller.update_alpha(progress)

        # Clip action magnitude
        action = np.clip(action, -1, 1)

        # Apply acceleration limits
        action_diff = action - self.prev_action
        action_diff = np.clip(
            action_diff, 
            -self.max_acceleration * self.dt,
            self.max_acceleration * self.dt
        )
        smoothed_action = self.prev_action + action_diff
        self.prev_action = smoothed_action

        # Scale to actual velocity
        scaled_action = smoothed_action * self.max_velocity

        # Execute action (append 0 for drop action)
        action_with_drop = [*scaled_action, injection_action]
        self.sim.run([action_with_drop], num_steps=1)

        #####################
        #    OBSERVATION    #
        #####################

        # Get current position
        current_pos = np.array(self.sim.get_states()[f'robotId_{self.robot_id}']['pipette_position'], dtype=np.float32)

        # Calculate velocity if we have previous position
        current_vel = np.zeros(3, dtype=np.float32) if self.prev_pos is None else (current_pos - self.prev_pos) / self.dt
        
        # Clip velocity to max_velocity
        current_vel_magnitude = np.linalg.norm(current_vel)
        if current_vel_magnitude > self.max_velocity:
            current_vel = current_vel * (self.max_velocity / current_vel_magnitude)
        
        self.prev_pos = current_pos.copy()

        # Calculate relative position and normalized position
        relative_pos = current_pos - self.goal_pos
        normalized_pos = np.clip(np.array([
            (current_pos[0] - self.x_bounds[0]) / (self.x_bounds[1] - self.x_bounds[0] + self.eps),
            (current_pos[1] - self.y_bounds[0]) / (self.y_bounds[1] - self.y_bounds[0] + self.eps),
            (current_pos[2] - self.z_bounds[0]) / (self.z_bounds[1] - self.z_bounds[0] + self.eps)
        ], dtype=np.float32), 0.0, 1.0)

        # Calculate distance
        current_distance = np.linalg.norm(relative_pos)
        distance_improvement = self.prev_distance - current_distance if self.prev_distance is not None else 0
        
        # Calculate distance difference per coordinate
        distance_diff_per_coord = self.prev_relative_pos - relative_pos if self.prev_relative_pos is not None else np.zeros(3)
        
        self.prev_distance = current_distance
        self.prev_relative_pos = relative_pos.copy()

        # Construct observation
        observation = np.concatenate([
            relative_pos,      # Relative position to goal
            current_vel,       # Current velocity
            normalized_pos,    # Normalized position in workspace
            [current_distance], # Distance to goal
            distance_diff_per_coord  # Distance difference per coordinate
        ]).astype(np.float32)

        #####################
        #    REWARD         #
        #####################

        # 1. Radius-based reward (exponential decay based on distance)
        # Reward decreases exponentially as distance increases
        reward_distance = np.exp(-10 * current_distance)  # Exponential decay factor of 10 for high precision

        # 2. Small time penalty to encourage efficiency
        reward_time = -0.01

        # 3. Success bonus (reaching within 0.1mm of goal)
        reward_success = 1000.0 if current_distance < self.target_precision else 0.0

        # Total reward combines distance-based reward, time penalty and success bonus
        reward = (
            reward_distance +  # Scale up the distance reward
            reward_time +      # Small time penalty
            reward_success     # Success bonus
        )

        
    
        #####################
        #    TERMINATED     #
        #####################

        # Early termination conditions
        terminated = False
        
        # Success condition
        if current_distance < self.target_precision:
            terminated = True        

        #####################
        #    TRUNCATED     #
        #####################

        # Episode truncates if we exceed max steps
        truncated = bool(self.steps > self.max_steps)

        # Print episode summary if episode is ending
        if terminated or truncated:
            print("\n" + "╔" + "═" * 50 + "╗")
            print("║" + f"{'Episode ' + str(self.episode_count) + ' Summary':^50}" + "║")
            print("╠" + "═" * 50 + "╣")
            print("║" + f"{'Initial Position:':<25}{self.initial_pos[0]:>7.3f}, {self.initial_pos[1]:>7.3f}, {self.initial_pos[2]:>7.3f}" + " ║")
            print("║" + f"{'Current Position:':<25}{current_pos[0]:>7.3f}, {current_pos[1]:>7.3f}, {current_pos[2]:>7.3f}" + " ║")
            print("║" + f"{'Goal Position:':<25}{self.goal_pos[0]:>7.3f}, {self.goal_pos[1]:>7.3f}, {self.goal_pos[2]:>7.3f}" + " ║")
            print("║" + f"{'Initial Distance (m):':<25}{self.initial_distance:>25.4f}" + " ║")
            print("║" + f"{'Final Distance (m):':<25}{current_distance:>25.4f}" + " ║")
            print("║" + f"{'Distance Improvement (m):':<25}{(self.initial_distance - current_distance):>25.4f}" + " ║")
            print("║" + f"{'Steps Taken:':<25}{self.steps:>25d}" + " ║")
            print("║" + f"{'Total Reward:':<25}{self.episode_reward:>25.2f}" + " ║")
            if self.steps > 0:
                print("║" + f"{'Average Reward/Step:':<25}{(self.episode_reward/self.steps):>25.2f}" + " ║")
            else:
                print("║" + f"{'Average Reward/Step:':<25}{'N/A':>25}" + " ║")
            status = 'Goal Reached! 🎯' if terminated else 'Max Steps Reached ⌛'
            print("║" + f"{'Status:':<25}{status:>25}" + " ║")
            print("╚" + "═" * 50 + "╝\n")

        #####################
        #    INFO           #
        #####################

        # Return distance metrics for monitoring
        info = {
            'distance': current_distance,
            'distance_improvement': distance_improvement
        }

        #####################
        #    INCREMENT      #
        #####################

        # Increment the number of steps
        self.steps += 1

        # Accumulate episode reward
        self.episode_reward += reward

        return observation, reward, terminated, truncated, info

    def is_within_bounds(self, position: np.ndarray) -> bool:
        """
        Check if a position is within the robot's working envelope.
        
        Args:
            position (np.ndarray): Position to check (x, y, z)
            
        Returns:
            bool: True if position is within bounds, False otherwise
        """
        x, y, z = position
        return (self.x_bounds[0] <= x <= self.x_bounds[1] and
                self.y_bounds[0] <= y <= self.y_bounds[1] and
                self.z_bounds[0] <= z <= self.z_bounds[1])
    
    def render(self, mode: str = 'human') -> None:
        """
        Render the environment.
        
        Currently not implemented as rendering is handled by the simulation.
        
        Args:
            mode (str): Rendering mode
        """
        pass
    
    def close(self) -> None:
        """
        Clean up resources used by the environment.
        """
        self.sim.close()
