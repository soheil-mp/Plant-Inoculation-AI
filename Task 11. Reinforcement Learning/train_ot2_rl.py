"""
OT2 Robot Arm Reinforcement Learning Training Script

This script implements a training pipeline for teaching an OT2 robot arm to perform precise movements
using Soft Actor-Critic (SAC) reinforcement learning, combined with a PID controller. The hybrid
approach allows for smooth transition from traditional control to learned behavior.

Key components:
- SAC algorithm for deep reinforcement learning
- Hybrid controller blending PID and RL outputs
- WandB integration for experiment tracking
- Evaluation and checkpoint callbacks for monitoring progress
- Replay buffer management for experience replay
- Environment normalization for stable training
"""

# Core Python imports for basic functionality
import os
import sys
import random
import time
import logging
import numpy as np
import argparse
from collections import deque

# Add paths for custom modules from other tasks
sys.path.append("./../task10")
sys.path.append("./../task12")
sys.path.append("./..")

# External library imports for simulation, RL, and monitoring
import pybullet as p                # Physics simulation
import gymnasium as gym            # RL environment framework
import wandb                       # Experiment tracking
import torch                       # Deep learning framework
from stable_baselines3 import SAC  # SAC algorithm implementation

# Stable-baselines3 utilities for environment handling and callbacks
from stable_baselines3.common.callbacks import (
    EvalCallback, 
    CheckpointCallback, 
    BaseCallback
)
from stable_baselines3.common.vec_env import (
    VecNormalize,    # Observation/reward normalization
    SubprocVecEnv,   # Parallel environment execution
    DummyVecEnv      # Single environment wrapper
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from wandb.integration.sb3 import WandbCallback

# Local module imports
from callbacks import *            # Custom training callbacks
from pid_controller import *       # PID control implementation
from ot2_env_wrapper import *

# Custom callback for saving replay buffer
class SaveReplayBufferCallback(BaseCallback):
    """
    Callback for periodically saving the replay buffer during training.
    
    This allows for training to be resumed with accumulated experience,
    and enables offline analysis of the agent's learning process.
    
    Args:
        check_freq (int): How often to save the buffer (in timesteps)
        save_path (str): Directory path for saving the buffer
        verbose (int): Verbosity level (0: no output, 1: info)
    """
    def __init__(self, check_freq: int, save_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        """Called after every step. Saves buffer if check_freq steps have passed."""
        if self.n_calls % self.check_freq == 0:
            path = os.path.join(self.save_path, f"replay_buffer_{self.n_calls}")
            self.model.save_replay_buffer(path)
        return True

class HybridController:
    """
    Combines PID and RL controllers with weighted blending for smooth transition.
    
    The controller gradually shifts control from PID to RL as training progresses,
    allowing for safe exploration while maintaining performance. The blending is
    controlled by an exponentially decaying alpha parameter.
    
    Args:
        pid_gains (dict): PID controller gains for each axis {'kp', 'ki', 'kd'}
        alpha (float): Initial blending factor (1.0 = pure PID, 0.0 = pure RL)
    """
    def __init__(self, pid_gains, alpha=1.0):
        self.pid_controller = PIDController(
            kp=pid_gains['kp'],
            ki=pid_gains['ki'],
            kd=pid_gains['kd']
        )
        self.alpha = alpha
        self.dt = 1/240  # PyBullet simulation timestep

    def blend_actions(self, pid_action, rl_action):
        """Combines PID and RL actions using the current blending factor."""
        return self.alpha * pid_action + (1 - self.alpha) * rl_action

    def update_alpha(self, progress):
        """Updates blending factor using exponential decay for smoother transition."""
        self.alpha = np.exp(-2 * progress)

# Training hyperparameters with detailed explanations
hyperparams = {

    # Core learning parameters
    "learning_rate": 1e-3,          # Step size for policy updates
    "buffer_size": 100000,          # Number of transitions to store
    "batch_size": 256,              # Transitions per gradient update
    "tau": 0.005,                   # Target network update rate
    "gamma": 0.995,                 # Future reward discount factor
    "train_freq": (1, "step"),      # Policy update frequency
    "n_envs": 1,                    # Number of parallel environments
    "gradient_steps": 1,            # Gradient updates per environment step
    
    # Neural network architecture
    "actor_layers": [512, 512],     # Actor network layer sizes
    "critic_layers": [512, 512],    # Critic network layer sizes
    "activation_fn": "relu",        # Network activation function
    "net_arch": "default",          # Network architecture type
    
    # Training process
    "total_timesteps": 1_000_000,   # Total training steps
    "learning_starts": 1000,        # Steps before learning starts
    "ent_coef": "auto_1.0",        # Automatic entropy adjustment
    
    # Environment settings
    "max_episode_steps": 2_000,     # Maximum steps per episode
    "normalize_observations": True,  # Normalize observation space
    "normalize_rewards": True,      # Normalize rewards
    "reward_scale": 1.0,            # Reward scaling factor
    "clip_obs": 5.0,               # Observation clipping range
    "clip_rewards": 5.0,           # Reward clipping range
    
    # Evaluation settings
    "eval_freq": 50_000,           # Steps between evaluations
    "n_eval_episodes": 10,         # Episodes per evaluation
    "eval_deterministic": True,    # Use deterministic policy for eval
}

def make_env(rank: int = 0):
    """
    Creates a wrapped and monitored OT2 environment instance.
    
    This factory function creates an environment with:
    1. The OT2 robot simulation environment
    2. A hybrid PID-RL controller with default gains
    3. Monitoring wrapper for logging metrics
    
    Args:
        rank (int): Environment index for parallel environments
        
    Returns:
        callable: A function that creates and returns the configured environment
    """
    def _init():
        env = OT2Env(
            render=False,
            max_steps=hyperparams["max_episode_steps"]
        )
        # Initialize hybrid controller with empirically tuned PID gains
        pid_gains = {
            'kp': [8.0, 8.0, 7.0],  # Proportional gains for x, y, z
            'ki': [1.0, 1.0, 0.8],  # Integral gains for x, y, z
            'kd': [0.6, 0.6, 1.2]   # Derivative gains for x, y, z
        }
        env.hybrid_controller = HybridController(pid_gains)
        
        # Add monitoring wrapper for metrics collection
        env = Monitor(
            env,
            filename=f'./logs/env_{rank}',
            info_keywords=('distance', 'distance_improvement')
        )
        return env
    return _init

def setup_environment(load_previous: bool = True):
    """
    Creates and configures the training environment with normalization.
    
    Sets up a vectorized environment with observation/reward normalization
    and optionally loads previous normalization statistics for continuity
    in training across sessions.
    
    Args:
        load_previous (bool): Whether to load previous normalization stats
        
    Returns:
        VecNormalize: The normalized vectorized environment
    """
    # Create parallel environments
    env = SubprocVecEnv([make_env(i) for i in range(hyperparams["n_envs"])])
    
    # Add observation and reward normalization
    env = VecNormalize(
        env,
        norm_obs=hyperparams["normalize_observations"],
        norm_reward=hyperparams["normalize_rewards"],
        clip_obs=hyperparams["clip_obs"],
        clip_reward=hyperparams["clip_rewards"],
        gamma=hyperparams["gamma"],
        training=True
    )

    # Attempt to load previous normalization statistics
    if load_previous and os.path.exists("./best_model/vec_normalize.pkl"):
        try:
            env = VecNormalize.load("./best_model/vec_normalize.pkl", env)
            print("Normalization stats loaded successfully")
        except AssertionError as e:
            print("Warning: Could not load previous normalization stats due to observation space mismatch.")
            print("This is expected if you modified the observation space. Starting with fresh normalization.")
            os.remove("./best_model/vec_normalize.pkl")
        
    return env

def setup_eval_environment(render: bool = False):
    """
    Creates and configures a separate environment for evaluation.
    
    The evaluation environment uses the same normalization as training
    but does not update the normalization statistics during evaluation.
    It also supports rendering for visualization if needed.
    
    Args:
        render (bool): Whether to enable rendering during evaluation
        
    Returns:
        VecNormalize: The normalized evaluation environment
    """
    eval_env = OT2Env(render=render, max_steps=hyperparams["max_episode_steps"], random_initial_position=True)
    eval_env = Monitor(
        eval_env,
        filename='./logs/eval_env',
        info_keywords=('distance', 'distance_improvement')
    )
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=hyperparams["normalize_observations"],
        norm_reward=hyperparams["normalize_rewards"],
        clip_obs=hyperparams["clip_obs"],
        clip_reward=hyperparams["clip_rewards"],
        gamma=hyperparams["gamma"],
        training=False  # Don't update normalization stats during evaluation
    )
    return eval_env

def main(custom_hyperparams=None, trial_dir=None):
    """
    Main training loop for the OT2 RL agent.
    
    This function:
    1. Sets up training and evaluation environments
    2. Initializes or loads a pre-trained SAC model
    3. Configures training callbacks for monitoring and checkpoints
    4. Runs the training loop with specified hyperparameters
    5. Saves the final model and normalization statistics
    
    Args:
        custom_hyperparams (dict, optional): Override default hyperparameters
        trial_dir (str, optional): Directory for saving trial results
        
    Returns:
        float: Final evaluation reward (for hyperparameter optimization)
    """
    # Parse arguments only when running directly
    if custom_hyperparams is None and trial_dir is None:
        try:
            parser = argparse.ArgumentParser(description='Train OT2 RL agent')
            args = parser.parse_args([])
        except:
            pass

    # Update hyperparameters if custom ones provided
    if custom_hyperparams is not None:
        hyperparams.update(custom_hyperparams)

    # Setup directory structure for logging and model saving
    base_dir = trial_dir if trial_dir else "."
    log_dir = os.path.join(base_dir, "logs")
    best_model_dir = os.path.join(base_dir, "best_model")
    checkpoint_dir = os.path.join(base_dir, "checkpoints")
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize environments
    env = setup_environment()
    eval_env = setup_eval_environment()

    # Initialize WandB for experiment tracking
    run = wandb.init(
        project="ot2-twin-monitoring",
        name="training_sac",
        config=hyperparams,
        sync_tensorboard=True,
        save_code=True,
    )

    # Configure neural network architecture
    policy_kwargs = dict(
        net_arch=dict(
            pi=hyperparams["actor_layers"],
            qf=hyperparams["critic_layers"]
        ),
        activation_fn=torch.nn.ReLU,
        normalize_images=False
    )

    model = None

    # Attempt to load previous model and replay buffer
    if os.path.exists("./best_model/best_model.zip"):
        try:
            model = SAC.load(
                "./best_model/best_model.zip",
                env=env,
                device='cpu',
                custom_objects={
                    "learning_rate": hyperparams["learning_rate"],
                    "buffer_size": hyperparams["buffer_size"],
                    "learning_starts": hyperparams["learning_starts"],
                    "batch_size": hyperparams["batch_size"],
                    "tau": hyperparams["tau"],
                    "gamma": hyperparams["gamma"],
                    "train_freq": hyperparams["train_freq"],
                    "gradient_steps": hyperparams["gradient_steps"],
                    "ent_coef": hyperparams["ent_coef"],
                }
            )
            print("Previous best model loaded successfully")
            
            # Load replay buffer if available
            if os.path.exists("./best_model/replay_buffer_final"):
                model.load_replay_buffer("./best_model/replay_buffer_final")
                print("Previous replay buffer loaded successfully")
        except (ValueError, AssertionError) as e:
            print(f"Warning: Could not load previous model due to: {str(e)}")
            print("This is expected if you modified the observation/action space. Creating new model...")
            if os.path.exists("./best_model/final_model.zip"):
                os.remove("./best_model/final_model.zip")
            if os.path.exists("./best_model/replay_buffer_final"):
                os.remove("./best_model/replay_buffer_final")
            model = None

    # Create new model if loading failed or no previous model exists
    if model is None:
        print("Creating new model...")
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=hyperparams["learning_rate"],
            buffer_size=hyperparams["buffer_size"],
            learning_starts=hyperparams["learning_starts"],
            batch_size=hyperparams["batch_size"],
            tau=hyperparams["tau"],
            gamma=hyperparams["gamma"],
            train_freq=hyperparams["train_freq"],
            gradient_steps=hyperparams["gradient_steps"],
            policy_kwargs=policy_kwargs,
            ent_coef=hyperparams["ent_coef"],
            target_entropy="auto",
            tensorboard_log=f"runs/{run.id}",
            device='cpu',
            verbose=2
        )

    # Setup training callbacks
    eval_callback = CustomEvalCallback(
        eval_env,
        best_model_save_path='./best_model/',
        log_path='./logs/eval/',
        eval_freq=hyperparams["eval_freq"],
        n_eval_episodes=hyperparams["n_eval_episodes"],
        deterministic=hyperparams["eval_deterministic"],
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=hyperparams["eval_freq"],
        save_path='./checkpoints/',
        name_prefix='sac_model',
        save_replay_buffer=True,
        save_vecnormalize=True
    )

    save_buffer_callback = SaveReplayBufferCallback(
        check_freq=hyperparams["eval_freq"],
        save_path='./best_model/'
    )

    # Run training loop with error handling
    try:
        model.learn(
            total_timesteps=hyperparams["total_timesteps"],
            callback=[
                CustomWandbCallback(),
                checkpoint_callback,
                eval_callback,
                save_buffer_callback
            ],
            progress_bar=True,
            log_interval=1
        )
        
        # Save final model state
        model.save("./best_model/final_model")
        model.save_replay_buffer("./best_model/replay_buffer_final")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving current model state...")
        model.save("./best_model/interrupted_model")
        model.save_replay_buffer("./best_model/replay_buffer_interrupted")

    # Save final normalization statistics
    env.save(os.path.join(best_model_dir, "vec_normalize.pkl"))
    eval_env.save(os.path.join(best_model_dir, "vec_normalize_eval.pkl"))

    return eval_callback.best_mean_reward

if __name__ == "__main__":
    main()