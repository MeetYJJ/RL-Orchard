"""Pure software interfaces for the future delta-joint SAC controller.

This package deliberately contains no ROS node, socket, service client, or
robot motion.  It is the boundary between a normalized policy action and a
future, separately implemented safety-aware joint executor.
"""

from orchard_picker.rl_control.action_space import (
    ACTION_DIM,
    ActionValidationError,
    JointActionConfig,
    normalized_action_to_delta_joint,
    compute_joint_target,
)
from orchard_picker.rl_control.safety_filter import (
    JointActionSafetyFilter,
    SafetyFilterResult,
)

__all__ = [
    "ACTION_DIM",
    "ActionValidationError",
    "JointActionConfig",
    "normalized_action_to_delta_joint",
    "compute_joint_target",
    "JointActionSafetyFilter",
    "SafetyFilterResult",
]

