"""Normalized six-joint incremental action primitives.

The functions in this module are intentionally independent of ROS and any
RealMan implementation.  They do not choose robot limits or motion speeds.
The caller must provide an explicitly reviewed per-joint ``max_joint_step``.
"""

from dataclasses import dataclass

import numpy as np


ACTION_DIM = 6
NORMALIZED_LOW = -1.0
NORMALIZED_HIGH = 1.0


class ActionValidationError(ValueError):
    """Raised when a policy action or joint vector is malformed."""


def _vector(value, name, dimension=ACTION_DIM):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ActionValidationError("{} is not numeric: {}".format(name, exc))
    if result.shape != (dimension,):
        raise ActionValidationError(
            "{} must have shape ({},), got {}".format(name, dimension, result.shape)
        )
    if not np.all(np.isfinite(result)):
        raise ActionValidationError("{} contains NaN or Inf".format(name))
    return result


def _max_joint_step(value):
    result = _vector(value, "max_joint_step")
    if np.any(result <= 0.0):
        raise ActionValidationError("max_joint_step must be strictly positive")
    return result


@dataclass(frozen=True)
class JointActionConfig:
    """Explicit action scaling configuration; no hardware defaults are filled."""

    max_joint_step: tuple
    normalized_low: float = NORMALIZED_LOW
    normalized_high: float = NORMALIZED_HIGH

    def __post_init__(self):
        steps = _max_joint_step(self.max_joint_step)
        low = float(self.normalized_low)
        high = float(self.normalized_high)
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise ActionValidationError("normalized action bounds are invalid")
        if low != NORMALIZED_LOW or high != NORMALIZED_HIGH:
            raise ActionValidationError(
                "the first SAC interface is fixed to normalized range [-1, 1]"
            )
        object.__setattr__(self, "max_joint_step", tuple(float(x) for x in steps))


def validate_normalized_action(action, clip=False):
    """Validate an Actor output in ``[-1, 1]^6``.

    NaN/Inf and wrong shape are always errors.  With ``clip=True`` only finite
    out-of-range values are clipped, which is useful at an explicit policy
    boundary and is reported by the caller's safety layer.
    """
    result = _vector(action, "normalized_action")
    if clip:
        return np.clip(result, NORMALIZED_LOW, NORMALIZED_HIGH)
    if np.any(result < NORMALIZED_LOW) or np.any(result > NORMALIZED_HIGH):
        raise ActionValidationError("normalized_action must lie in [-1, 1]^6")
    return result


def normalized_action_to_delta_joint(action, max_joint_step, clip=False):
    """Map normalized Actor output to per-joint position increments.

    ``delta_joint[i] = action[i] * max_joint_step[i]``.  Units are the units
    selected by the caller (normally radians); no degree/radian conversion is
    guessed here.
    """
    normalized = validate_normalized_action(action, clip=clip)
    steps = _max_joint_step(max_joint_step)
    return (normalized * steps).astype(np.float32)


def compute_joint_target(current_joint, delta_joint):
    """Compute ``q_target = q_current + delta_joint`` without I/O."""
    current = _vector(current_joint, "current_joint")
    delta = _vector(delta_joint, "delta_joint")
    return (current + delta).astype(np.float32)

