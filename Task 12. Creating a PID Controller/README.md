# OT-2 Robot PID Controller

A robust and efficient PID (Proportional-Integral-Derivative) controller implementation for precise position control of the OT-2 robot. This controller provides accurate 3D position control with features like anti-windup protection and output saturation.

![alt text](PIDcontrollerdemo-ezgif.com-speed.gif)

## Features

- 🎯 Independent PID control for each axis (X, Y, Z)
- 🔄 Anti-windup protection to prevent integral term saturation
- 📊 Output saturation for safe control signals
- 📈 Performance tracking and error monitoring
- 🔧 Easily configurable gains for each axis
- 🧮 Support for variable time steps

## Usage

Here's a simple example of how to use the PID controller:

```python
import numpy as np
from pid_controller import PIDController

# Initialize the controller with gains for each axis [x, y, z]
controller = PIDController(
    kp=[1.0, 1.0, 1.0],  # Proportional gains
    ki=[0.1, 0.1, 0.1],  # Integral gains
    kd=[0.05, 0.05, 0.05]  # Derivative gains
)

# Define target and current positions
target_pos = np.array([0.1, 0.1, 0.2])
current_pos = np.array([0.0, 0.0, 0.15])

# Time step (PyBullet default)
dt = 1/240

# Compute control action
action = controller.compute(current_pos, target_pos, dt)
```

## Architecture

The implementation consists of two main classes:

1. `PIDController`: Main class that handles 3D position control
   - Manages separate controllers for each axis
   - Provides unified interface for 3D control
   - Tracks overall performance

2. `AxisPIDController`: Single-axis controller
   - Implements core PID algorithm
   - Handles anti-windup protection
   - Manages output saturation

## Configuration

The controller can be fine-tuned using several parameters:

```python
# PID gains for each axis
kp = [1.0, 1.0, 1.0]      # Higher values = faster response
ki = [0.1, 0.1, 0.1]     # Higher values = better steady-state error
kd = [0.05, 0.05, 0.05]  # Higher values = more damping

# Output limits
output_limits = (-1, 1)  # Constrains control signal

# Anti-windup limit
windup_limit = 2.0  # Prevents integral term saturation
```

## Performance Monitoring

The controller provides methods to monitor its performance:

```python
# Get current errors for each axis
errors = controller.get_errors()

# Get configured gains
gains = controller.get_gains()
```

