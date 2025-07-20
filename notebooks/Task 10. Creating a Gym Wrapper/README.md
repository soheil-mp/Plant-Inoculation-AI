# OT-2 Digital Twin Environment for Reinforcement Learning

A Gymnasium-compatible environment wrapper for training reinforcement learning agents to control an OT-2 liquid handling robot in a digital twin simulation using PyBullet.

## Overview

This environment enables training of reinforcement learning agents to control the OT-2 robot's movement to target positions within its working envelope. It provides a standardized interface compatible with popular RL frameworks like Stable Baselines 3.

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

1. Create and activate a virtual environment (recommended):
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On Linux/Mac
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Verifying Installation

Run the test suite to verify everything is working correctly:
```bash
python test_wrapper.py
```

If all tests pass, you're ready to use the environment!

## Features

- 🎮 **Custom Action Space**: Continuous control of x, y, z velocities in range [-1, 1]
- 📊 **Rich Observation Space**: Includes:
  - Relative position to goal
  - Current velocity
  - Normalized position in workspace
  - Distance to goal
  - Distance improvement metrics
- 🎯 **Sophisticated Reward Function**: Combines:
  - Distance-based rewards
  - Movement efficiency incentives
  - Success bonuses
- 🛡️ **Safety Features**:
  - Action smoothing
  - Velocity and acceleration limits
  - Working envelope boundary checking
- 📈 **Detailed Performance Metrics**:
  - Episode summaries
  - Distance tracking
  - Movement efficiency stats

## Technical Specifications

### Working Envelope
- X bounds: [-0.1871m, 0.253m]
- Y bounds: [-0.1705m, 0.2195m]
- Z bounds: [0.1195m, 0.2896m]

### Control Parameters
- Maximum velocity: 5 m/s
- Maximum acceleration: 1.0 m/s²
- Control frequency: 240 Hz
- Target precision: 0.001m (1mm)

## Usage

```python
from ot2_env_wrapper import OT2Env

# Create environment
env = OT2Env(render=True)

# Reset environment
obs, info = env.reset()

# Run episode
for _ in range(1000):
    action = env.action_space.sample()  # Replace with your agent's action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

# Clean up
env.close()
```

## Testing

The environment comes with comprehensive tests covering:
1. Stable Baselines 3 compatibility
2. Random action evaluation
3. Observation/action space validation
4. Action smoothing verification
5. Reward structure testing
6. Basic movement capabilities using a PD controller

Run tests using:
```python
python test_wrapper.py
```

## Environment Details

### Observation Space (13 dimensions)
1. Relative position to goal (3D)
2. Current velocity (3D)
3. Normalized position in workspace (3D)
4. Distance to goal (1D)
5. Distance improvement per coordinate (3D)

### Action Space (3 dimensions)
- Continuous velocities for x, y, z axes
- Range: [-1, 1] (scaled internally)

### Reward Structure
- Distance-based reward (exponential decay)
- Time penalty for efficiency
- Large success bonus for reaching target
- Boundary violation penalties

## Dependencies

- gymnasium
- numpy
- pybullet
- stable-baselines3 (for testing)

## Notes

- The environment includes action smoothing to prevent abrupt movements
- Early termination occurs upon reaching the target (within 1mm)
- Episodes truncate after 1000 steps
- Detailed episode summaries are provided for debugging and analysis

