"""
PID Controller for OT-2 Robot Position Control.

This module implements a PID controller for precise position control of the 
OT-2 robot. The controller maintains separate PID loops for each axis (X, Y, Z) 
and includes features like anti-windup and output saturation.
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class PIDGains:
    """Container for PID gains for a single axis."""
    kp: float  # Proportional gain
    ki: float  # Integral gain
    kd: float  # Derivative gain


class AxisPIDController:
    """PID controller for a single axis."""
    
    def __init__(
        self, 
        kp: float, 
        ki: float, 
        kd: float, 
        output_limits: Tuple[float, float] = (-1, 1)
    ):
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
        
        # Performance tracking
        self.last_time: Optional[float] = None
        self.error_history: List[float] = []
        
    def compute(
        self, 
        current_value: float, 
        setpoint: float, 
        dt: float
    ) -> float:
        """
        Compute PID output for a single axis.
        
        Args:
            current_value: Current position
            setpoint: Desired position
            dt: Time step
            
        Returns:
            Control output
        """
        # Calculate error
        error = setpoint - current_value
        
        # Proportional term
        proportional = self.gains.kp * error
        
        # Integral term with anti-windup
        self.integral += error * dt
        
        # Anti-windup: Clamp integral if output would saturate
        integral_term = self.gains.ki * self.integral
        temp_output = proportional + integral_term
        
        if temp_output > self.output_limits[1]:
            self.integral = (
                (self.output_limits[1] - proportional) / self.gains.ki
            )
        elif temp_output < self.output_limits[0]:
            self.integral = (
                (self.output_limits[0] - proportional) / self.gains.ki
            )
        
        integral_term = self.gains.ki * self.integral
        
        # Derivative term
        if dt > 0:
            derivative = self.gains.kd * (error - self.prev_error) / dt
        else:
            derivative = 0.0
        
        # Combine terms
        output = proportional + integral_term + derivative
        
        # Apply output limits
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        # Store for next iteration
        self.prev_error = error
        self.error_history.append(error)
        
        # Keep only recent history
        if len(self.error_history) > 100:
            self.error_history.pop(0)
        
        return output
    
    def reset(self) -> None:
        """Reset the controller state."""
        self.prev_error = 0.0
        self.integral = 0.0
        self.error_history = []
    
    def get_performance_metrics(self) -> dict:
        """Get performance metrics for this axis."""
        if not self.error_history:
            return {'rmse': 0.0, 'max_error': 0.0, 'steady_state_error': 0.0}
        
        errors = np.array(self.error_history)
        return {
            'rmse': float(np.sqrt(np.mean(errors**2))),
            'max_error': float(np.max(np.abs(errors))),
            'steady_state_error': float(np.mean(errors[-10:]) if len(errors) >= 10 else np.mean(errors))
        }


class PIDController:
    """Multi-axis PID controller for 3D position control."""
    
    def __init__(
        self,
        kp: List[float],
        ki: List[float], 
        kd: List[float],
        output_limits: Optional[List[Tuple[float, float]]] = None
    ):
        """
        Initialize 3D PID controller.
        
        Args:
            kp: Proportional gains for [x, y, z] axes
            ki: Integral gains for [x, y, z] axes
            kd: Derivative gains for [x, y, z] axes
            output_limits: Optional output limits for each axis
        """
        if len(kp) != 3 or len(ki) != 3 or len(kd) != 3:
            raise ValueError("PID gains must be provided for all 3 axes")
        
        if output_limits is None:
            output_limits = [(-1, 1), (-1, 1), (-1, 1)]
        
        self.controllers = []
        for i in range(3):
            controller = AxisPIDController(
                kp=kp[i], 
                ki=ki[i], 
                kd=kd[i],
                output_limits=output_limits[i]
            )
            self.controllers.append(controller)
        
        self.axis_names = ['x', 'y', 'z']
        
    def compute(
        self, 
        current_pos: np.ndarray, 
        target_pos: np.ndarray, 
        dt: float
    ) -> np.ndarray:
        """
        Compute PID output for all axes.
        
        Args:
            current_pos: Current position [x, y, z]
            target_pos: Target position [x, y, z]
            dt: Time step
            
        Returns:
            Control actions [x, y, z]
        """
        if len(current_pos) != 3 or len(target_pos) != 3:
            raise ValueError("Position arrays must have 3 elements")
        
        outputs = []
        for i, controller in enumerate(self.controllers):
            output = controller.compute(current_pos[i], target_pos[i], dt)
            outputs.append(output)
        
        return np.array(outputs, dtype=np.float32)
    
    def reset(self) -> None:
        """Reset all axis controllers."""
        for controller in self.controllers:
            controller.reset()
    
    def get_performance_metrics(self) -> dict:
        """Get performance metrics for all axes."""
        metrics = {}
        for i, (controller, axis) in enumerate(zip(self.controllers, self.axis_names)):
            axis_metrics = controller.get_performance_metrics()
            for metric, value in axis_metrics.items():
                metrics[f'{axis}_{metric}'] = value
        
        # Overall metrics
        rmse_values = [metrics[f'{axis}_rmse'] for axis in self.axis_names]
        metrics['overall_rmse'] = float(np.sqrt(np.mean(np.array(rmse_values)**2)))
        
        return metrics
    
    def tune_gains(
        self, 
        axis: int, 
        kp: Optional[float] = None,
        ki: Optional[float] = None,
        kd: Optional[float] = None
    ) -> None:
        """
        Tune gains for a specific axis.
        
        Args:
            axis: Axis index (0=x, 1=y, 2=z)
            kp: New proportional gain (optional)
            ki: New integral gain (optional)
            kd: New derivative gain (optional)
        """
        if axis < 0 or axis >= 3:
            raise ValueError("Axis must be 0, 1, or 2")
        
        controller = self.controllers[axis]
        
        if kp is not None:
            controller.gains.kp = kp
        if ki is not None:
            controller.gains.ki = ki
        if kd is not None:
            controller.gains.kd = kd
    
    def get_gains(self) -> dict:
        """Get current PID gains for all axes."""
        gains = {}
        for i, (controller, axis) in enumerate(zip(self.controllers, self.axis_names)):
            gains[axis] = {
                'kp': controller.gains.kp,
                'ki': controller.gains.ki,
                'kd': controller.gains.kd
            }
        return gains
    
    def set_output_limits(
        self, 
        axis: int, 
        min_output: float, 
        max_output: float
    ) -> None:
        """
        Set output limits for a specific axis.
        
        Args:
            axis: Axis index (0=x, 1=y, 2=z)
            min_output: Minimum output value
            max_output: Maximum output value
        """
        if axis < 0 or axis >= 3:
            raise ValueError("Axis must be 0, 1, or 2")
        
        self.controllers[axis].output_limits = (min_output, max_output)


class HybridController:
    """Hybrid controller combining PID and RL outputs."""
    
    def __init__(
        self,
        pid_controller: PIDController,
        alpha_start: float = 1.0,
        alpha_end: float = 0.1,
        dt: float = 1/240
    ):
        """
        Initialize hybrid controller.
        
        Args:
            pid_controller: PID controller instance
            alpha_start: Initial PID weight (1.0 = pure PID)
            alpha_end: Final PID weight (0.0 = pure RL)
            dt: Control time step
        """
        self.pid_controller = pid_controller
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.current_alpha = alpha_start
        self.dt = dt
    
    def blend_actions(
        self, 
        pid_action: np.ndarray, 
        rl_action: np.ndarray
    ) -> np.ndarray:
        """
        Blend PID and RL actions based on current alpha.
        
        Args:
            pid_action: PID controller output
            rl_action: RL controller output
            
        Returns:
            Blended action
        """
        blended = (self.current_alpha * pid_action + 
                  (1 - self.current_alpha) * rl_action)
        return blended.astype(np.float32)
    
    def update_alpha(self, progress: float) -> None:
        """
        Update blending parameter based on training progress.
        
        Args:
            progress: Training progress (0.0 to 1.0)
        """
        progress = np.clip(progress, 0.0, 1.0)
        self.current_alpha = (
            self.alpha_start + 
            (self.alpha_end - self.alpha_start) * progress
        )
    
    def get_status(self) -> dict:
        """Get current controller status."""
        return {
            'alpha': self.current_alpha,
            'pid_weight': self.current_alpha,
            'rl_weight': 1 - self.current_alpha
        } 