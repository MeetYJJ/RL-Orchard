#!/usr/bin/env python3
"""Pure mock tests for the future delta-joint SAC control boundary."""

import unittest

import numpy as np

from orchard_picker.rl_control.action_space import (
    ActionValidationError,
    JointActionConfig,
    compute_joint_target,
    normalized_action_to_delta_joint,
    validate_normalized_action,
)
from orchard_picker.rl_control.safety_filter import JointActionSafetyFilter


class RlControlTest(unittest.TestCase):
    def test_actor_action_shape_is_six(self):
        result = validate_normalized_action(np.zeros((6,)))
        self.assertEqual(result.shape, (6,))

    def test_normalized_bounds(self):
        np.testing.assert_array_equal(
            validate_normalized_action([-1, -0.5, 0, 0.5, 1, 0]),
            [-1, -0.5, 0, 0.5, 1, 0],
        )
        with self.assertRaises(ActionValidationError):
            validate_normalized_action([1.1, 0, 0, 0, 0, 0])

    def test_action_scaling(self):
        result = normalized_action_to_delta_joint(
            [-1, -0.5, 0, 0.5, 1, 0],
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        )
        np.testing.assert_allclose(result, [-0.1, -0.1, 0, 0.2, 0.5, 0])

    def test_per_joint_scaling_is_not_scalar_scaling(self):
        config = JointActionConfig((0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
        result = normalized_action_to_delta_joint([1] * 6, config.max_joint_step)
        np.testing.assert_allclose(result, config.max_joint_step)

    def test_delta_joint_and_target(self):
        delta = normalized_action_to_delta_joint([0.5] * 6, [0.2] * 6)
        target = compute_joint_target([1, 2, 3, 4, 5, 6], delta)
        np.testing.assert_allclose(delta, [0.1] * 6)
        np.testing.assert_allclose(target, [1.1, 2.1, 3.1, 4.1, 5.1, 6.1])

    def test_clip_behavior(self):
        result = validate_normalized_action([2, -2, 0, 0, 0, 0], clip=True)
        np.testing.assert_array_equal(result, [1, -1, 0, 0, 0, 0])
        safety = JointActionSafetyFilter(max_joint_step=[0.1] * 6, clip=True)
        filtered = safety.filter([0] * 6, [0.2] * 6)
        self.assertTrue(filtered.valid)
        self.assertTrue(filtered.clipped)
        np.testing.assert_allclose(filtered.delta_joint_safe, [0.1] * 6)

    def test_joint_limit_behavior_rejects_by_default(self):
        safety = JointActionSafetyFilter(
            joint_limits=[[-1, 1]] * 6, max_joint_step=[0.5] * 6
        )
        filtered = safety.filter([0.9] * 6, [0.2] * 6)
        self.assertFalse(filtered.valid)
        self.assertEqual(filtered.reason, "joint_limit_exceeded")

    def test_joint_limit_behavior_can_clip_explicitly(self):
        safety = JointActionSafetyFilter(
            joint_limits=[[-1, 1]] * 6, max_joint_step=[0.5] * 6, clip=True
        )
        filtered = safety.filter([0.9] * 6, [0.2] * 6)
        self.assertTrue(filtered.valid)
        self.assertTrue(filtered.clipped)
        np.testing.assert_allclose(filtered.q_target_safe, [1.0] * 6)

    def test_nan_action_rejected(self):
        with self.assertRaises(ActionValidationError):
            normalized_action_to_delta_joint([np.nan] * 6, [0.1] * 6)

    def test_inf_action_rejected(self):
        with self.assertRaises(ActionValidationError):
            normalized_action_to_delta_joint([np.inf] * 6, [0.1] * 6)

    def test_wrong_action_dimension_rejected(self):
        with self.assertRaises(ActionValidationError):
            validate_normalized_action([0, 0, 0, 0, 0])

    def test_safety_reject_reason(self):
        safety = JointActionSafetyFilter(max_joint_step=[0.1] * 6)
        filtered = safety.filter([0] * 6, [0.2] * 6, emergency_stop=True)
        self.assertFalse(filtered.valid)
        self.assertEqual(filtered.reason, "emergency_stop")
        np.testing.assert_array_equal(filtered.delta_joint_safe, [0] * 6)
        np.testing.assert_array_equal(filtered.q_target_safe, [0] * 6)

    def test_robot_state_and_error_checks(self):
        safety = JointActionSafetyFilter()
        self.assertEqual(
            safety.filter([0] * 6, [0] * 6, robot_state_valid=False).reason,
            "robot_state_invalid",
        )
        self.assertEqual(
            safety.filter([0] * 6, [0] * 6, robot_error_code=7).reason,
            "robot_error_code_nonzero",
        )

    def test_velocity_limit_requires_period_and_is_deterministic(self):
        safety = JointActionSafetyFilter(
            max_joint_velocity=[1.0] * 6, control_period_sec=0.1
        )
        first = safety.filter([0] * 6, [0.2] * 6)
        second = safety.filter([0] * 6, [0.2] * 6)
        self.assertFalse(first.valid)
        self.assertEqual(first.reason, "velocity_limit_exceeded")
        np.testing.assert_array_equal(first.q_target_safe, second.q_target_safe)

    def test_workspace_and_collision_are_optional_rejectors(self):
        safety = JointActionSafetyFilter(
            workspace_checker=lambda target, context: False,
            collision_checker=None,
        )
        self.assertEqual(
            safety.filter([0] * 6, [0] * 6).reason, "workspace_rejected"
        )
        safety = JointActionSafetyFilter(
            collision_checker=lambda current, target, context: False,
        )
        self.assertEqual(
            safety.filter([0] * 6, [0] * 6).reason, "collision_rejected"
        )


if __name__ == "__main__":
    unittest.main()
