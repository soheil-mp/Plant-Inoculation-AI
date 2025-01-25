# OT2 Robot Arm Reinforcement Learning

A sophisticated reinforcement learning implementation for training an OT2 robot arm to perform precise movements using a hybrid approach combining Soft Actor-Critic (SAC) and PID control.

<br>

## 🎯 Project Overview

This project implements an advanced training pipeline that teaches an OT2 robot arm to execute precise movements by combining traditional PID control with modern reinforcement learning techniques. The hybrid approach enables smooth transition from conventional control methods to learned behaviors, ensuring both safety and performance.

### Key Features

- 🤖 Soft Actor-Critic (SAC) implementation for deep reinforcement learning
- 🎮 Hybrid controller blending PID and RL outputs
- 📊 Comprehensive monitoring with Weights & Biases integration
- 💾 Checkpoint system for experiment tracking and model persistence
- 🔄 Experience replay buffer management
- 📈 Environment normalization for stable training

<br>

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- PyBullet
- PyTorch
- Stable-Baselines3
- Weights & Biases
- Gymnasium

<br>

## 📖 Usage

### Training

To start training the model:

```bash
python train_ot2_rl.py
```

The training script includes:
- Automatic environment setup with normalization
- Loading/saving of model checkpoints
- Integration with Weights & Biases for experiment tracking
- Hybrid controller implementation
- Custom callbacks for monitoring and evaluation

### Testing

Use the Jupyter notebook `test_ot2_rl.ipynb` to:
- Load and evaluate trained models
- Visualize robot arm movements
- Test different goal positions
- Analyze model performance

<br>

## 🔧 Hyperparameter Search

Based on extensive hyperparameter search, we analyzed three key parameters and their effects on model performance:

1. **Learning Rate**: Different rates (0.0001, 0.001, 3e-05) showed minimal impact on both total reward and final distance metrics, suggesting the model is relatively robust to learning rate variations.

2. **Max Episode Steps**: Higher values performed better, with 2000 steps showing improved total reward compared to 1000 and 500 steps. This indicates longer episodes allow for better learning and task completion.

3. **Buffer Size**: Various sizes (50000, 200000, 100000) demonstrated no significant difference in performance, suggesting the replay buffer size is not a critical factor within this range.

![alt text](assets/hyperparameter_charts.png)

The final configuration uses:

```python
hyperparams = {
    "learning_rate": 0.001,     
    "max_episode_steps": 2000,  
    "buffer_size": 100000,      
    ...
}
```

<br>

## 📈 Performance Analysis

The results demonstrate stable and effective learning across multiple metrics:

### Distance to Target
- Maintains consistently low final distance (around 0.01 units) throughout training
- Shows minimal variance in performance, indicating reliable positioning accuracy
- Demonstrates stable convergence after initial training phase

![alt text](assets/final_distance_to_target.png)

### Episode Rewards
- Mean rewards stabilize around 850-880 range
- Shows healthy exploration with regular fluctuations
- Peak performance observed around 600k-700k steps with rewards reaching 880
- Maintains consistent performance in later training stages

![alt text](assets/rollout-ep_rew_mean.png)

### Episode Length
- Average episode length fluctuates between 970-1000 steps
- Shows periodic variations but maintains overall stability
- Slight downward trend between steps 400k-800k before recovering, suggesting exploration phases

![alt text](assets/rollout-ep_len_mean.png)

These metrics indicate that the model successfully learns to:
- Achieve precise positioning with minimal error
- Complete tasks efficiently within reasonable time frames
- Maintain stable performance over extended training periods

<br>

## 📊 Monitoring

The project includes comprehensive monitoring capabilities:

- **Training Metrics**
  - Episode rewards and lengths
  - Success rates
  - Distance improvements
  - Training time statistics

- **Curriculum Progress**
  - Precision targets
  - Success rates
  - Learning progression

- **Performance Analysis**
  - Distance improvements
  - Episode durations
  - Velocity metrics

<br>

## 🔍 Project Structure

```
task11/
├── train_ot2_rl.py      # Main training script
├── callbacks.py         # Custom callback implementations
├── test_ot2_rl.ipynb   # Testing and visualization notebook
└── README.md           # Project documentation
```
