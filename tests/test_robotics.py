"""Tests for robotics modules."""

import pytest
import numpy as np
from unittest.mock import Mock

from plant_inoculation_ai.robotics.controllers.pid import (
    PIDGains, AxisPIDController, PIDController, HybridController
)


class TestPIDGains:
    """Test PIDGains dataclass."""
    
    def test_init(self):
        """Test PID gains initialization."""
        gains = PIDGains(kp=1.0, ki=0.1, kd=0.05)
        assert gains.kp == 1.0
        assert gains.ki == 0.1
        assert gains.kd == 0.05


class TestAxisPIDController:
    """Test AxisPIDController class."""
    
    def test_init(self):
        """Test controller initialization."""
        controller = AxisPIDController(kp=1.0, ki=0.1, kd=0.05)
        assert controller.gains.kp == 1.0
        assert controller.gains.ki == 0.1
        assert controller.gains.kd == 0.05
        assert controller.output_limits == (-1, 1)
        assert controller.prev_error == 0.0
        assert controller.integral == 0.0
    
    def test_init_with_limits(self):
        """Test controller initialization with custom limits."""
        controller = AxisPIDController(
            kp=1.0, ki=0.1, kd=0.05, output_limits=(-5, 5)
        )
        assert controller.output_limits == (-5, 5)
    
    def test_compute_proportional_only(self):
        """Test PID computation with only proportional term."""
        controller = AxisPIDController(kp=1.0, ki=0.0, kd=0.0)
        
        # Simple proportional control
        output = controller.compute(current_value=0.0, setpoint=1.0, dt=0.1)
        assert output == 1.0  # P term: 1.0 * (1.0 - 0.0) = 1.0
    
    def test_compute_with_integral(self):
        """Test PID computation with integral term."""
        controller = AxisPIDController(kp=1.0, ki=0.5, kd=0.0)
        
        # First call
        output1 = controller.compute(current_value=0.0, setpoint=1.0, dt=0.1)
        expected1 = 1.0 + 0.5 * 1.0 * 0.1  # P + I terms
        assert abs(output1 - expected1) < 1e-6
        
        # Second call (integral accumulates)
        output2 = controller.compute(current_value=0.0, setpoint=1.0, dt=0.1)
        expected2 = 1.0 + 0.5 * 1.0 * 0.2  # P + accumulated I
        assert abs(output2 - expected2) < 1e-6
    
    def test_compute_with_derivative(self):
        """Test PID computation with derivative term."""
        controller = AxisPIDController(kp=1.0, ki=0.0, kd=0.2)
        
        # First call (no derivative yet)
        output1 = controller.compute(current_value=0.0, setpoint=1.0, dt=0.1)
        assert output1 == 1.0  # Only P term
        
        # Second call (derivative based on error change)
        output2 = controller.compute(current_value=0.5, setpoint=1.0, dt=0.1)
        # Error changed from 1.0 to 0.5, so derivative = 0.2 * (0.5 - 1.0) / 0.1
        expected_d = 0.2 * (0.5 - 1.0) / 0.1
        expected2 = 0.5 + expected_d  # P + D terms
        assert abs(output2 - expected2) < 1e-6
    
    def test_output_limits(self):
        """Test output limiting."""
        controller = AxisPIDController(
            kp=10.0, ki=0.0, kd=0.0, output_limits=(-1, 1)
        )
        
        # Large error should be limited
        output = controller.compute(current_value=0.0, setpoint=10.0, dt=0.1)
        assert output == 1.0  # Should be clipped to upper limit
        
        output = controller.compute(current_value=10.0, setpoint=0.0, dt=0.1)
        assert output == -1.0  # Should be clipped to lower limit
    
    def test_anti_windup(self):
        """Test anti-windup functionality."""
        controller = AxisPIDController(
            kp=1.0, ki=1.0, kd=0.0, output_limits=(-1, 1)
        )
        
        # Apply large error repeatedly to trigger saturation
        for _ in range(10):
            controller.compute(current_value=0.0, setpoint=10.0, dt=0.1)
        
        # Integral should be limited to prevent windup
        assert controller.integral <= 1.0  # Should be clamped
    
    def test_reset(self):
        """Test controller reset."""
        controller = AxisPIDController(kp=1.0, ki=0.1, kd=0.05)
        
        # Run controller to accumulate history
        controller.compute(current_value=0.0, setpoint=1.0, dt=0.1)
        controller.compute(current_value=0.5, setpoint=1.0, dt=0.1)
        
        # Check that state has changed
        assert controller.prev_error != 0.0
        assert controller.integral != 0.0
        assert len(controller.error_history) > 0
        
        # Reset
        controller.reset()
        
        # Check that state is cleared
        assert controller.prev_error == 0.0
        assert controller.integral == 0.0
        assert len(controller.error_history) == 0
    
    def test_performance_metrics_empty(self):
        """Test performance metrics with no history."""
        controller = AxisPIDController(kp=1.0, ki=0.1, kd=0.05)
        
        metrics = controller.get_performance_metrics()
        assert metrics['rmse'] == 0.0
        assert metrics['max_error'] == 0.0
        assert metrics['steady_state_error'] == 0.0
    
    def test_performance_metrics_with_data(self):
        """Test performance metrics with error history."""
        controller = AxisPIDController(kp=1.0, ki=0.1, kd=0.05)
        
        # Generate some errors
        errors = [1.0, 0.8, 0.6, 0.4, 0.2]
        for error in errors:
            controller.compute(current_value=1.0-error, setpoint=1.0, dt=0.1)
        
        metrics = controller.get_performance_metrics()
        assert metrics['rmse'] > 0.0
        assert metrics['max_error'] == 1.0
        assert abs(metrics['steady_state_error'] - 0.2) < 1e-6


class TestPIDController:
    """Test PIDController class."""
    
    def test_init(self):
        """Test multi-axis controller initialization."""
        controller = PIDController(
            kp=[1.0, 1.5, 2.0],
            ki=[0.1, 0.15, 0.2],
            kd=[0.05, 0.075, 0.1]
        )
        assert len(controller.controllers) == 3
        assert controller.controllers[0].gains.kp == 1.0
        assert controller.controllers[1].gains.ki == 0.15
        assert controller.controllers[2].gains.kd == 0.1
    
    def test_init_invalid_gains(self):
        """Test initialization with invalid gain arrays."""
        with pytest.raises(ValueError, match="PID gains must be provided for all 3 axes"):
            PIDController(kp=[1.0, 1.5], ki=[0.1, 0.15], kd=[0.05, 0.075])
    
    def test_compute_3d(self):
        """Test 3D position control."""
        controller = PIDController(
            kp=[1.0, 1.0, 1.0],
            ki=[0.0, 0.0, 0.0],
            kd=[0.0, 0.0, 0.0]
        )
        
        current_pos = np.array([0.0, 0.0, 0.0])
        target_pos = np.array([1.0, 2.0, 3.0])
        
        output = controller.compute(current_pos, target_pos, dt=0.1)
        
        assert output.shape == (3,)
        assert output.dtype == np.float32
        np.testing.assert_array_almost_equal(output, [1.0, 2.0, 3.0])
    
    def test_compute_invalid_positions(self):
        """Test computation with invalid position arrays."""
        controller = PIDController(
            kp=[1.0, 1.0, 1.0],
            ki=[0.0, 0.0, 0.0],
            kd=[0.0, 0.0, 0.0]
        )
        
        with pytest.raises(ValueError, match="Position arrays must have 3 elements"):
            controller.compute(
                np.array([1.0, 2.0]),  # Invalid length
                np.array([1.0, 2.0, 3.0]),
                dt=0.1
            )
    
    def test_reset_all_controllers(self):
        """Test resetting all axis controllers."""
        controller = PIDController(
            kp=[1.0, 1.0, 1.0],
            ki=[0.1, 0.1, 0.1],
            kd=[0.0, 0.0, 0.0]
        )
        
        # Run to accumulate state
        controller.compute(
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            dt=0.1
        )
        
        # Reset all
        controller.reset()
        
        # Check all controllers are reset
        for axis_controller in controller.controllers:
            assert axis_controller.prev_error == 0.0
            assert axis_controller.integral == 0.0
    
    def test_tune_gains(self):
        """Test gain tuning for specific axis."""
        controller = PIDController(
            kp=[1.0, 1.0, 1.0],
            ki=[0.1, 0.1, 0.1],
            kd=[0.05, 0.05, 0.05]
        )
        
        # Tune x-axis gains
        controller.tune_gains(axis=0, kp=2.0, ki=0.2)
        
        assert controller.controllers[0].gains.kp == 2.0
        assert controller.controllers[0].gains.ki == 0.2
        assert controller.controllers[0].gains.kd == 0.05  # Unchanged
        
        # Other axes unchanged
        assert controller.controllers[1].gains.kp == 1.0
        assert controller.controllers[2].gains.kp == 1.0
    
    def test_tune_gains_invalid_axis(self):
        """Test tuning with invalid axis."""
        controller = PIDController(
            kp=[1.0, 1.0, 1.0],
            ki=[0.1, 0.1, 0.1],
            kd=[0.05, 0.05, 0.05]
        )
        
        with pytest.raises(ValueError, match="Axis must be 0, 1, or 2"):
            controller.tune_gains(axis=3, kp=2.0)


class TestHybridController:
    """Test HybridController class."""
    
    def test_init(self):
        """Test hybrid controller initialization."""
        pid_controller = Mock()
        
        hybrid = HybridController(
            pid_controller=pid_controller,
            alpha_start=1.0,
            alpha_end=0.1
        )
        
        assert hybrid.pid_controller == pid_controller
        assert hybrid.alpha_start == 1.0
        assert hybrid.alpha_end == 0.1
        assert hybrid.current_alpha == 1.0
    
    def test_blend_actions_pure_pid(self):
        """Test action blending with pure PID (alpha=1.0)."""
        pid_controller = Mock()
        hybrid = HybridController(pid_controller, alpha_start=1.0, alpha_end=0.0)
        
        pid_action = np.array([1.0, 2.0, 3.0])
        rl_action = np.array([4.0, 5.0, 6.0])
        
        blended = hybrid.blend_actions(pid_action, rl_action)
        
        # Should be pure PID
        np.testing.assert_array_almost_equal(blended, [1.0, 2.0, 3.0])
    
    def test_blend_actions_pure_rl(self):
        """Test action blending with pure RL (alpha=0.0)."""
        pid_controller = Mock()
        hybrid = HybridController(pid_controller, alpha_start=0.0, alpha_end=0.0)
        hybrid.current_alpha = 0.0
        
        pid_action = np.array([1.0, 2.0, 3.0])
        rl_action = np.array([4.0, 5.0, 6.0])
        
        blended = hybrid.blend_actions(pid_action, rl_action)
        
        # Should be pure RL
        np.testing.assert_array_almost_equal(blended, [4.0, 5.0, 6.0])
    
    def test_blend_actions_mixed(self):
        """Test action blending with mixed weights."""
        pid_controller = Mock()
        hybrid = HybridController(pid_controller, alpha_start=0.5, alpha_end=0.5)
        
        pid_action = np.array([2.0, 4.0, 6.0])
        rl_action = np.array([0.0, 0.0, 0.0])
        
        blended = hybrid.blend_actions(pid_action, rl_action)
        
        # Should be 50% PID + 50% RL
        expected = 0.5 * pid_action + 0.5 * rl_action
        np.testing.assert_array_almost_equal(blended, expected)
    
    def test_update_alpha(self):
        """Test alpha update based on progress."""
        pid_controller = Mock()
        hybrid = HybridController(
            pid_controller, alpha_start=1.0, alpha_end=0.0
        )
        
        # Start of training (progress=0.0)
        hybrid.update_alpha(0.0)
        assert hybrid.current_alpha == 1.0
        
        # Middle of training (progress=0.5)
        hybrid.update_alpha(0.5)
        assert hybrid.current_alpha == 0.5
        
        # End of training (progress=1.0)
        hybrid.update_alpha(1.0)
        assert hybrid.current_alpha == 0.0
    
    def test_update_alpha_clipped(self):
        """Test alpha update with out-of-bounds progress."""
        pid_controller = Mock()
        hybrid = HybridController(
            pid_controller, alpha_start=1.0, alpha_end=0.0
        )
        
        # Progress < 0
        hybrid.update_alpha(-0.5)
        assert hybrid.current_alpha == 1.0
        
        # Progress > 1
        hybrid.update_alpha(1.5)
        assert hybrid.current_alpha == 0.0
    
    def test_get_status(self):
        """Test status reporting."""
        pid_controller = Mock()
        hybrid = HybridController(
            pid_controller, alpha_start=1.0, alpha_end=0.0
        )
        hybrid.current_alpha = 0.7
        
        status = hybrid.get_status()
        
        assert status['alpha'] == 0.7
        assert status['pid_weight'] == 0.7
        assert status['rl_weight'] == 0.3


if __name__ == "__main__":
    pytest.main([__file__]) 