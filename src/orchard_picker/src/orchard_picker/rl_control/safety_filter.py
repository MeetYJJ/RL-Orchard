"""Independent, hardware-free safety boundary for future joint actions."""

from dataclasses import dataclass

import numpy as np

from orchard_picker.rl_control.action_space import (
    ACTION_DIM,
    ActionValidationError,
    _vector,
)


@dataclass(frozen=True)
class SafetyFilterResult:
    """Result of validating/clipping one requested joint increment."""

    delta_joint_safe: np.ndarray
    q_target_safe: np.ndarray
    valid: bool
    reason: str
    clipped: bool = False


def _limits(value, name):
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ActionValidationError("{} is not numeric: {}".format(name, exc))
    if result.shape != (ACTION_DIM, 2):
        raise ActionValidationError(
            "{} must have shape (6,2), got {}".format(name, result.shape)
        )
    if not np.all(np.isfinite(result)):
        raise ActionValidationError("{} contains NaN or Inf".format(name))
    if np.any(result[:, 0] > result[:, 1]):
        raise ActionValidationError("{} lower bound exceeds upper bound".format(name))
    return result


class JointActionSafetyFilter:
    """Check a policy delta before a future executor receives a target.

    This class does not know ROS, RealMan, torque, collision geometry, or
    project-specific limits.  Optional limits/checkers must be supplied by a
    separately reviewed integration.  With ``clip=False`` violations are
    rejected.  With ``clip=True`` configured step and joint limits are clipped
    and reported in ``reason``; emergency/error/workspace failures remain
    rejected.
    """

    def __init__(
        self,
        max_joint_step=None,
        joint_limits=None,
        max_joint_velocity=None,
        control_period_sec=None,
        clip=False,
        workspace_checker=None,
        collision_checker=None,
    ):
        self.max_joint_step = (
            None if max_joint_step is None else _vector(max_joint_step, "max_joint_step")
        )
        if self.max_joint_step is not None and np.any(self.max_joint_step <= 0.0):
            raise ActionValidationError("max_joint_step must be strictly positive")
        self.joint_limits = _limits(joint_limits, "joint_limits")
        self.max_joint_velocity = (
            None if max_joint_velocity is None
            else _vector(max_joint_velocity, "max_joint_velocity")
        )
        if self.max_joint_velocity is not None and np.any(self.max_joint_velocity <= 0.0):
            raise ActionValidationError("max_joint_velocity must be strictly positive")
        if control_period_sec is not None:
            if float(control_period_sec) <= 0.0 or not np.isfinite(float(control_period_sec)):
                raise ActionValidationError("control_period_sec must be positive")
        self.control_period_sec = control_period_sec
        self.clip = bool(clip)
        self.workspace_checker = workspace_checker
        self.collision_checker = collision_checker

    @staticmethod
    def _invalid(current, requested, reason):
        current = _vector(current, "current_joint")
        _vector(requested, "delta_joint_requested")
        # A rejected result must be safe to inspect and impossible to mistake
        # for an executable request: hold the current target with zero delta.
        return SafetyFilterResult(
            np.zeros((ACTION_DIM,), dtype=np.float32),
            current.astype(np.float32),
            False,
            reason,
            False,
        )

    def filter(
        self,
        current_joint,
        delta_joint_requested,
        robot_state_valid=True,
        robot_error_code=0,
        emergency_stop=False,
        workspace_context=None,
        collision_context=None,
    ):
        """Return a safe candidate; this method never sends it anywhere."""
        current = _vector(current_joint, "current_joint")
        requested = _vector(delta_joint_requested, "delta_joint_requested")
        if emergency_stop:
            return self._invalid(current, requested, "emergency_stop")
        if not bool(robot_state_valid):
            return self._invalid(current, requested, "robot_state_invalid")
        try:
            error_code = int(robot_error_code)
        except (TypeError, ValueError, OverflowError):
            return self._invalid(current, requested, "robot_error_code_invalid")
        if error_code != 0:
            return self._invalid(current, requested, "robot_error_code_nonzero")

        delta = requested.copy()
        reasons = []
        clipped = False
        if self.max_joint_step is not None:
            if np.any(np.abs(delta) > self.max_joint_step):
                if not self.clip:
                    return self._invalid(current, requested, "delta_exceeds_max_joint_step")
                delta = np.clip(delta, -self.max_joint_step, self.max_joint_step)
                reasons.append("delta_clipped_to_max_joint_step")
                clipped = True

        if self.max_joint_velocity is not None:
            if self.control_period_sec is None:
                return self._invalid(current, requested, "velocity_limit_missing_control_period")
            velocity = np.abs(delta) / float(self.control_period_sec)
            if np.any(velocity > self.max_joint_velocity):
                if not self.clip:
                    return self._invalid(current, requested, "velocity_limit_exceeded")
                delta = np.clip(
                    delta,
                    -self.max_joint_velocity * float(self.control_period_sec),
                    self.max_joint_velocity * float(self.control_period_sec),
                )
                reasons.append("delta_clipped_to_velocity_limit")
                clipped = True

        target = current + delta
        if self.joint_limits is not None:
            below = target < self.joint_limits[:, 0]
            above = target > self.joint_limits[:, 1]
            if np.any(below | above):
                if not self.clip:
                    return self._invalid(current, requested, "joint_limit_exceeded")
                target = np.clip(target, self.joint_limits[:, 0], self.joint_limits[:, 1])
                delta = target - current
                reasons.append("target_clipped_to_joint_limits")
                clipped = True

        if self.workspace_checker is not None:
            try:
                workspace_ok = bool(self.workspace_checker(target, workspace_context))
            except Exception as exc:
                return self._invalid(current, requested, "workspace_checker_error:{}".format(exc))
            if not workspace_ok:
                return self._invalid(current, requested, "workspace_rejected")
        if self.collision_checker is not None:
            try:
                collision_ok = bool(self.collision_checker(current, target, collision_context))
            except Exception as exc:
                return self._invalid(current, requested, "collision_checker_error:{}".format(exc))
            if not collision_ok:
                return self._invalid(current, requested, "collision_rejected")

        reason = ";".join(reasons) if reasons else "accepted"
        return SafetyFilterResult(
            delta.astype(np.float32), target.astype(np.float32), True, reason, clipped
        )
