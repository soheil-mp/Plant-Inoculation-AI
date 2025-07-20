# Import the libraries
import os
import time
import logging
import numpy as np
import gymnasium as gym
import wandb
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


########################
#    Eval Callback    #
########################

class CustomEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate the current model
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
            )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print(f"Eval num_timesteps={self.num_timesteps}, "
                      f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

            # Save the model if it's better than the previous best
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                    # Also save the replay buffer
                    self.model.save_replay_buffer(os.path.join(self.best_model_save_path, "replay_buffer"))
        return True


########################
#    Wandb Callback    #
########################

class CustomWandbCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.training_start = time.time()
        # Episode tracking
        self.episode_start_distance = None
        self.episode_min_distance = float('inf')
        self.episode_start_time = None
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.episode_steps = 0
        
        # Success tracking
        self.successful_episodes = 0
        self.total_episodes = 0
        self.termination_reasons = {}
        
        # Curriculum tracking
        self.precision_targets = []
        self.success_rates = []
        
        # Performance tracking
        self.distance_improvements = []
        self.episode_durations = []
        self.velocity_metrics = []
        
    def _on_step(self):
        try:
            # Get the current info and reward
            infos = self.locals.get('infos', [{}])
            rewards = self.locals.get('rewards', [0])
            
            # Handle vectorized environment - take first environment's info
            info = infos[0] if len(infos) > 0 else {}
            reward = rewards[0] if len(rewards) > 0 else 0
            
            # Update episode metrics
            self.current_episode_reward += reward
            self.episode_steps += 1
            
            # Track episode start metrics
            if self.episode_start_distance is None:
                self.episode_start_distance = info.get('distance', 0)
                self.episode_start_time = time.time()
            
            # Update minimum distance
            current_distance = info.get('distance', float('inf'))
            self.episode_min_distance = min(self.episode_min_distance, current_distance)
            
            # Extract all available metrics from info
            distance_improvement = info.get('distance_improvement', 0)
            current_precision_target = info.get('current_precision_target', 0)
            success_rate = info.get('success_rate', 0)
            termination_reason = info.get('termination_reason', None)
            
            # Check for episode termination
            done = self.locals.get('dones', [False])[0]
            if done:
                # Update episode counters
                self.episode_rewards.append(self.current_episode_reward)
                self.total_episodes += 1
                
                # Track termination reasons
                if termination_reason:
                    self.termination_reasons[termination_reason] = self.termination_reasons.get(termination_reason, 0) + 1
                    if termination_reason == "Goal Reached! 🎯":
                        self.successful_episodes += 1
                
                # Track curriculum metrics
                self.precision_targets.append(current_precision_target)
                self.success_rates.append(success_rate)
                
                # Track performance metrics
                self.distance_improvements.append(self.episode_start_distance - current_distance)
                if self.episode_start_time:
                    self.episode_durations.append(time.time() - self.episode_start_time)
                
                # Calculate termination statistics
                termination_stats = {
                    reason: (count / self.total_episodes) * 100 
                    for reason, count in self.termination_reasons.items()
                }
                
                # Log episode-specific metrics
                wandb.log({
                    # Episode completion metrics
                    "episode/termination_reason": termination_reason,
                    "episode/duration": time.time() - self.episode_start_time if self.episode_start_time else 0,
                    "episode/total_reward": self.current_episode_reward,
                    "episode/steps": self.episode_steps,
                    "episode/final_distance": current_distance,
                    "episode/total_improvement": self.episode_start_distance - current_distance,
                    "episode/improvement_percentage": ((self.episode_start_distance - current_distance) / self.episode_start_distance * 100) if self.episode_start_distance and self.episode_start_distance != 0 else 0,
                    
                    # Curriculum metrics
                    "curriculum/precision_target": current_precision_target,
                    "curriculum/success_rate": success_rate,
                    "curriculum/avg_precision_target": np.mean(self.precision_targets[-100:]) if self.precision_targets else 0,
                    
                    # Overall statistics
                    "stats/success_rate": (self.successful_episodes / self.total_episodes) * 100,
                    "stats/avg_episode_length": np.mean(self.episode_durations[-100:]) if self.episode_durations else 0,
                    "stats/avg_distance_improvement": np.mean(self.distance_improvements[-100:]) if self.distance_improvements else 0,
                    "stats/avg_reward": np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
                    "stats/best_reward": max(self.episode_rewards) if self.episode_rewards else 0,
                    
                    # Termination statistics
                    **{f"termination/{reason}": percentage for reason, percentage in termination_stats.items()},
                })
                
                # Reset episode-specific variables
                self.current_episode_reward = 0
                self.episode_steps = 0
                self.episode_start_distance = None
                self.episode_min_distance = float('inf')
                self.episode_start_time = None
            
            # Log step-specific metrics
            wandb.log({
                # Distance metrics
                "distance/current": current_distance,
                "distance/improvement": distance_improvement,
                
                # Training progress
                "training/timesteps": self.num_timesteps,
                "training/episodes": self.total_episodes,
                "training/time_elapsed": time.time() - self.training_start,
                "training/steps_per_second": self.num_timesteps / (time.time() - self.training_start),
                
                # Current episode progress
                "current/episode_step": self.episode_steps,
                "current/episode_reward": self.current_episode_reward,
                "current/min_distance": self.episode_min_distance,
            })
            
            return True
            
        except Exception as e:
            print(f"Error in CustomWandbCallback: {str(e)}")
            return True


#############################
#    Checkpoint Callback    #
#############################

checkpoint_callback = CheckpointCallback(
    save_freq=10000,  # Save every 1000 steps
    save_path="./checkpoints/",
    name_prefix="model_checkpoint"
)

