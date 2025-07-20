"""
PID Controller for OT-2 Robot Position Control

This module implements a PID controller for precise position control of the OT-2 robot.
The controller maintains separate PID loops for each axis (X, Y, Z) and includes
features like anti-windup and output saturation.

Example:
    controller = PIDController(kp=[1.0, 1.0, 1.0], 
                             ki=[0.1, 0.1, 0.1], 
                             kd=[0.05, 0.05, 0.05])
    target_pos = np.array([0.1, 0.1, 0.2])
    current_pos = np.array([0.0, 0.0, 0.15])
    dt = 1/240  # PyBullet default timestep
    
    action = controller.compute(current_pos, target_pos, dt)
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import time

@dataclass
class PIDGains:
    """Container for PID gains for a single axis"""
    kp: float  # Proportional gain
    ki: float  # Integral gain
    kd: float  # Derivative gain

class AxisPIDController:
    """PID controller for a single axis"""
    
    def __init__(self, kp: float, ki: float, kd: float, output_limits: Tuple[float, float] = (-1, 1)):
        """
        Initialize a single-axis PID controller.
        
        Args:
            kp: Proportional gain
            ki: Integral gain
            kd: Derivative gain
            output_limits: Tuple of (min, max) output values
        """
        self.gains = PIDGains(kp=kp, ki=ki, kd=kd)
        self.output_limits = output_limits
        
        # Initialize error terms
        self.prev_error = 0.0
        self.integral = 0.0
        
        # Anti-windup parameters
        self.windup_limit = 2.0  # Changed from 1.0 to 2.0
        
        # Performance tracking
        self.last_output = 0.0
        self.last_time = time.time()
        
    def reset(self):
        """Reset the controller's internal state"""
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_output = 0.0
        
    def compute(self, current: float, target: float, dt: float) -> float:
        """
        Compute control output for the axis.
        
        Args:
            current: Current position
            target: Target position
            dt: Time step in seconds
            
        Returns:
            float: Control output in range [-1, 1]
        """
        # Calculate error
        error = target - current
        
        # Proportional term
        p_term = self.gains.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.windup_limit, self.windup_limit)
        i_term = self.gains.ki * self.integral
        
        # Derivative term (on measurement to avoid derivative kick)
        d_term = self.gains.kd * (error - self.prev_error) / dt if dt > 0 else 0
        
        # Calculate total output
        output = p_term + i_term + d_term
        
        # Apply output limits
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        # Update state
        self.prev_error = error
        self.last_output = output
        
        return output

class PIDController:
    """PID controller for 3D position control"""
    
    def __init__(self, 
                 kp: List[float], 
                 ki: List[float], 
                 kd: List[float],
                 output_limits: Tuple[float, float] = (-1, 1)):
        """
        Initialize PID controllers for each axis.
        
        Args:
            kp: List of proportional gains [x, y, z]
            ki: List of integral gains [x, y, z]
            kd: List of derivative gains [x, y, z]
            output_limits: Tuple of (min, max) output values
        """
        # Create separate controllers for each axis
        self.controllers = [
            AxisPIDController(kp[i], ki[i], kd[i], output_limits)
            for i in range(3)
        ]
        
        # Store axis names for debugging
        self.axis_names = ['x', 'y', 'z']
        
        # Performance tracking
        self.last_time = time.time()
        self.last_errors = np.zeros(3)
        
    def reset(self):
        """Reset all axis controllers"""
        for controller in self.controllers:
            controller.reset()
            
    def compute(self, current_pos: np.ndarray, target_pos: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute control outputs for all axes.
        
        Args:
            current_pos: Current position [x, y, z]
            target_pos: Target position [x, y, z]
            dt: Time step in seconds
            
        Returns:
            np.ndarray: Control outputs for each axis [-1, 1]
        """
        # Compute control output for each axis
        outputs = np.array([
            controller.compute(current, target, dt)
            for controller, current, target in zip(self.controllers, current_pos, target_pos)
        ])
        
        # Track performance
        self.last_errors = target_pos - current_pos
        self.last_time = time.time()
        
        return outputs
    
    def get_errors(self) -> np.ndarray:
        """Get the last error for each axis"""
        return self.last_errors
    
    def get_gains(self) -> List[PIDGains]:
        """Get the gains for each axis"""
        return [controller.gains for controller in self.controllers] 