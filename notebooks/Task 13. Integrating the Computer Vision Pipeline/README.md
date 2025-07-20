# Task 13: Integrating the Computer Vision Pipeline with Robot Control

## Overview
This task focuses on integrating a computer vision pipeline with robotic control systems to automate the inoculation of root tips in multiple specimen plates. The project combines computer vision for root tip detection with precise robotic control (both PID and RL-based) to achieve accurate specimen handling.

![alt text](PIDcontrollerdemo-ezgif.com-speed.gif)

## Features
- Integration of computer vision pipeline with robotic control
- Support for both PID and Reinforcement Learning controllers
- Real-time root tip detection and coordinate mapping
- Automated specimen plate handling
- Simulated droplet placement system
- Multi-robot support

## Prerequisites
- Python 3.x
- TensorFlow
- OpenCV (cv2)
- PyBullet
- NumPy
- Matplotlib

## Project Structure
```
task13/
├── ot2_twin/               # Robot simulation components
│   ├── sim_class.py       # Simulation implementation
│   └── ot2_simulation_v6.urdf  # Robot URDF model
├── task13.ipynb           # Main implementation notebook
└── README.md             # This file
```

## Setup and Installation
1. Ensure all prerequisites are installed
2. Clone the repository
3. Navigate to the task13 directory
4. Run the Jupyter notebook to start the implementation

## Usage
1. Initialize the environment and controllers:
```python
from ot2_env_wrapper import OT2Env
env = OT2Env(render=True)
```

2. Load and process specimen images using the CV pipeline
3. Map detected root tip coordinates to robot coordinates
4. Execute automated inoculation using either PID or RL controller

## Key Components
1. **Computer Vision Pipeline**
   - Root tip detection
   - Coordinate extraction
   - Image preprocessing

2. **Robot Control**
   - PID controller implementation
   - Reinforcement Learning controller
   - Position tracking and adjustment

3. **Simulation Environment**
   - PyBullet-based 3D simulation
   - Droplet placement system
   - Multi-robot coordination

## Notes
- Ensure proper calibration between image coordinates and robot coordinates
- The system supports multiple specimen plates
- Real-time visualization is available through the simulation environment
