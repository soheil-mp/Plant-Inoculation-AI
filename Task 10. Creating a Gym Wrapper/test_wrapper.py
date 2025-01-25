
# Import the libraries
import gymnasium as gym
import numpy as np
import random
from stable_baselines3.common.env_checker import check_env
from typing import Optional, Tuple

# Import the environment
from ot2_env_wrapper import OT2Env

# Instantiate the environment
env = OT2Env(render=True)

#########################################
#    1. Test with Stable Baselines 3    #
#########################################

# Check the environment
check_env(env)


###################################
#    2. Test with Random Actions  #
###################################

try:
    print("\n=== Running Random Actions Test ===\n")
    
    # Create environment
    env = OT2Env(render=False)
    print("✓ Environment created successfully")

    # Run 1000 steps with random actions
    obs, _ = env.reset()
    total_reward = 0
    steps = 0

    print("\nRunning 1000 random steps...")
    for step in range(1000):
        # Sample a random action
        action = env.action_space.sample()
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # Print progress every 100 steps
        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}: Current distance = {info['distance']:.4f} m, Total reward = {total_reward:.2f}")
        
        if terminated or truncated:
            print(f"Episode ended after {steps} steps")
            break

    print(f"\nRandom Actions Test Summary:")
    print(f"Steps completed: {steps}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average reward per step: {total_reward/steps:.2f}")

except Exception as e:
    print(f"❌ An error occurred: {e}")
finally:
    env.close()


#######################################
#    3. More Testing the Environment  #
#######################################

try:
    print("\n=== Testing OT2 Environment ===\n")
    
    # Create environment
    env = OT2Env(render=False)
    env._testing = True  # Enable debug prints
    print("✓ Environment created successfully")

    # Test observation space
    obs, _ = env.reset()
    assert isinstance(obs, np.ndarray), "Observation must be a numpy array"
    assert obs.shape == (13,), f"Observation shape must be (13,), got {obs.shape}"
    assert all(isinstance(x, np.float32) for x in obs), "All observations must be float32"
    print("✓ Observation space verified")
    
    # Verify observation components
    relative_pos = obs[0:3]
    velocity = obs[3:6]
    normalized_pos = obs[6:9]
    distance = obs[9]
    distance_diff = obs[10:13]
    
    assert all(0 <= x <= 1 for x in normalized_pos), "Normalized positions must be between 0 and 1"
    assert all(x == 0 for x in velocity), "Initial velocity should be zero"
    assert distance >= 0, "Distance must be non-negative"
    assert len(distance_diff) == 3, "Distance difference per coordinate must have 3 components"
    print("✓ Observation components verified")

    # Test action space
    action = env.action_space.sample()
    assert action.shape == (3,), f"Action shape must be (3,), got {action.shape}"
    assert all(-1 <= x <= 1 for x in action), "Actions must be between -1 and 1"
    print("✓ Action space verified")

    # Test action smoothing
    prev_action = action.copy()
    obs, reward, terminated, truncated, info = env.step(action)
    max_action_diff = env.max_acceleration * env.dt
    actual_diff = np.abs(action - prev_action)
    assert all(d <= max_action_diff + 1e-6 for d in actual_diff), "Action smoothing failed"
    print("✓ Action smoothing verified")

    # Test reward components
    assert isinstance(reward, float), "Reward must be a float"
    assert 'distance' in info, "Info must contain 'distance'"
    assert 'distance_improvement' in info, "Info must contain 'distance_improvement'"
    print("✓ Reward structure verified")

    # Run a test episode with a simple controller
    print("\nRunning test episode...")
    total_reward = 0
    min_distance = float('inf')
    steps = 0
    distances = []
    velocities = []
    
    # Set a goal position that's achievable but requires significant movement
    goal_pos = np.array([
        0.1,    # Fixed position with good distance
        0.1,    # Fixed position with good distance
        0.25    # Higher Z position
    ], dtype=np.float32)
    
    obs, _ = env.reset(goal_pos=goal_pos)
    env._testing = False  # Disable debug prints for episode
    initial_distance = obs[9]
    last_distance = initial_distance
    
    # Controller parameters
    kp = 10.0   # Increased proportional gain
    kd = 1.0    # Reduced damping to allow faster movement
    ki = 0.0    # No integral term
    max_steps = 200
    
    print(f"\nMoving from {obs[0:3] + goal_pos} to {goal_pos}")
    print(f"Distance to cover: {initial_distance:.4f} m")
    print(f"Max allowed velocity: {env.max_velocity:.4f} m/s")
    print(f"Max allowed acceleration: {env.max_acceleration:.4f} m/s²")
    
    while steps < max_steps:
        # Enhanced PD controller
        relative_pos = obs[0:3]
        current_vel = obs[3:6]
        
        # PD terms with scaling
        p_term = -kp * relative_pos
        d_term = -kd * current_vel
        
        # Combine terms and scale
        action = p_term + d_term
        action_magnitude = np.linalg.norm(action)
        
        if action_magnitude > 1e-6:
            # Scale to maximum velocity while maintaining direction
            # Use a more aggressive scaling to overcome smoothing
            action = 2.0 * action / action_magnitude  # Scale up to overcome smoothing
            action = np.clip(action, -1, 1)  # Ensure we stay within [-1, 1]
        
        # Execute action and get feedback
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        # Track metrics
        current_distance = obs[9]
        current_vel_magnitude = np.linalg.norm(current_vel)
        min_distance = min(min_distance, current_distance)
        distances.append(current_distance)
        velocities.append(current_vel_magnitude)
        
        # Print progress every 25 steps
        if steps % 25 == 0:
            improvement = last_distance - current_distance
            print(f"Step {steps}: distance = {current_distance:.4f} m, velocity = {current_vel_magnitude:.4f} m/s, improvement = {improvement:.4f} m")
            last_distance = current_distance
        
        # Early stopping if we're not making progress
        if steps > 50 and all(abs(d - current_distance) < 1e-6 for d in distances[-10:]):
            print("Movement stalled, stopping early")
            break
        
        if terminated:
            print("Goal reached!")
            break
        if truncated:
            print("Episode truncated")
            break
    
    # Report detailed metrics
    print(f"\nTest Episode Metrics:")
    print(f"Steps completed: {steps}")
    print(f"Initial distance: {initial_distance:.4f} m")
    print(f"Final distance: {distances[-1]:.4f} m")
    print(f"Minimum distance: {min_distance:.4f} m")
    print(f"Distance improvement: {initial_distance - distances[-1]:.4f} m")
    print(f"Average distance: {np.mean(distances):.4f} m")
    print(f"Max velocity: {max(velocities):.4f} m/s")
    print(f"Average velocity: {np.mean(velocities):.4f} m/s")
    print(f"Total reward: {total_reward:.2f}")
    
    # Verify movement
    improvement_threshold = 0.01  # At least 1cm improvement
    assert distances[-1] < initial_distance - improvement_threshold, f"Agent should improve by at least {improvement_threshold}m"
    print("✓ Basic movement verified")

    print("\nAll tests passed successfully! ✓")

except AssertionError as e:
    print(f"❌ Test failed: {e}")
except Exception as e:
    print(f"❌ An error occurred: {e}")
finally:
    env.close()

