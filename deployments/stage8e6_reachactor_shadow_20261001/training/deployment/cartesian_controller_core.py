"""Hardware-independent deterministic Cartesian controller.

This is the part of the validated MuJoCo controller that can be moved to a
future Origin process.  It computes a 15D state, turns a Cartesian residual
into a bounded target pose, and performs generic safety checks.  DLS IK,
physics stepping, ROS, RealMan protocol encoding, and network access remain in
their respective adapters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pose_math import (
    compose_pose,
    finite_array,
    matrix_to_rotvec,
    validate_rotation_matrix,
)


@dataclass
class Pose:
    """Rigid pose with SI units."""

    position_m: np.ndarray
    rotation: np.ndarray

    def __post_init__(self):
        self.position_m = finite_array(self.position_m, (3,), "position_m")
        self.rotation = validate_rotation_matrix(self.rotation)


@dataclass
class RobotState:
    """One feedback/goal snapshot consumed by the controller."""

    joint_position_rad: np.ndarray
    tcp_pose: Pose
    goal_pose: Pose
    timestamp_s: float

    def __post_init__(self):
        self.joint_position_rad = finite_array(
            self.joint_position_rad, (6,), "joint_position_rad"
        )
        self.timestamp_s = float(self.timestamp_s)
        if not np.isfinite(self.timestamp_s):
            raise ValueError("timestamp_s must be finite")
        if not isinstance(self.tcp_pose, Pose) or not isinstance(self.goal_pose, Pose):
            raise TypeError("tcp_pose and goal_pose must be Pose instances")


@dataclass
class CartesianControllerConfig:
    """Controller constants and optional hardware-neutral safety bounds."""

    position_gain: float = 0.8
    rotation_gain: float = 0.8
    translation_bound_m: float = 0.0015
    rotation_bound_rad: float = 0.0075
    workspace_min_m: np.ndarray | None = None
    workspace_max_m: np.ndarray | None = None
    joint_limits_rad: np.ndarray | None = None
    joint_margin_rad: float = 0.0
    reject_near_joint_limits: bool = False

    def __post_init__(self):
        values = (
            self.position_gain,
            self.rotation_gain,
            self.translation_bound_m,
            self.rotation_bound_rad,
            self.joint_margin_rad,
        )
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("controller configuration must be finite")
        if self.position_gain < 0.0 or self.rotation_gain < 0.0:
            raise ValueError("controller gains must be non-negative")
        if self.translation_bound_m <= 0.0 or self.rotation_bound_rad <= 0.0:
            raise ValueError("action bounds must be positive")
        if self.joint_margin_rad < 0.0:
            raise ValueError("joint_margin_rad must be non-negative")
        if (self.workspace_min_m is None) != (self.workspace_max_m is None):
            raise ValueError("workspace_min_m and workspace_max_m must be paired")
        if self.workspace_min_m is not None:
            self.workspace_min_m = finite_array(self.workspace_min_m, (3,), "workspace_min_m")
            self.workspace_max_m = finite_array(self.workspace_max_m, (3,), "workspace_max_m")
            if np.any(self.workspace_min_m > self.workspace_max_m):
                raise ValueError("workspace minimum exceeds maximum")
        if self.joint_limits_rad is not None:
            self.joint_limits_rad = finite_array(self.joint_limits_rad, (6, 2), "joint_limits_rad")
            if np.any(self.joint_limits_rad[:, 0] > self.joint_limits_rad[:, 1]):
                raise ValueError("joint lower limit exceeds upper limit")


@dataclass
class ControllerResult:
    target_pose: Pose | None
    state_vector: np.ndarray | None
    requested_residual: np.ndarray | None
    applied_residual: np.ndarray | None
    safety_ok: bool
    translation_clamped: bool
    rotation_clamped: bool
    near_joint_limit: bool
    error_reason: str | None


class CartesianControllerCore:
    """Compute the validated deterministic Cartesian residual controller."""

    def __init__(self, config: CartesianControllerConfig | None = None):
        self.config = config or CartesianControllerConfig()

    def build_state(self, robot_state: RobotState) -> np.ndarray:
        """Return the frozen 15D state used by the Cartesian controller."""
        if not isinstance(robot_state, RobotState):
            raise TypeError("robot_state must be a RobotState")
        relative_position = robot_state.goal_pose.position_m - robot_state.tcp_pose.position_m
        relative_rotation = matrix_to_rotvec(
            robot_state.goal_pose.rotation @ robot_state.tcp_pose.rotation.T
        )
        state = np.concatenate(
            (robot_state.joint_position_rad, robot_state.tcp_pose.position_m,
             relative_position, relative_rotation)
        )
        if state.shape != (15,) or not np.all(np.isfinite(state)):
            raise ValueError("constructed state must be finite with shape (15,)")
        return state

    def nominal_residual(self, robot_state: RobotState) -> np.ndarray:
        state = self.build_state(robot_state)
        return np.concatenate(
            (
                self.config.position_gain * state[9:12],
                self.config.rotation_gain * state[12:15],
            )
        )

    def apply_residual(self, tcp_pose: Pose, residual) -> tuple[Pose, np.ndarray, bool, bool]:
        """Clamp a residual and compose its target pose."""
        if not isinstance(tcp_pose, Pose):
            raise TypeError("tcp_pose must be a Pose")
        requested = finite_array(residual, (6,), "residual")
        translation = np.clip(
            requested[:3],
            -self.config.translation_bound_m,
            self.config.translation_bound_m,
        )
        rotation = requested[3:].copy()
        rotation_norm = float(np.linalg.norm(rotation))
        rotation_clamped = rotation_norm > self.config.rotation_bound_rad
        if rotation_clamped:
            rotation *= self.config.rotation_bound_rad / rotation_norm
        translation_clamped = bool(
            np.any(np.abs(requested[:3]) > self.config.translation_bound_m)
        )
        target_position, target_rotation = compose_pose(
            tcp_pose.position_m, tcp_pose.rotation, translation, rotation
        )
        return Pose(target_position, target_rotation), np.concatenate((translation, rotation)), translation_clamped, rotation_clamped

    def _joint_safety(self, joint_position):
        if self.config.joint_limits_rad is None:
            return True, False, None
        q = finite_array(joint_position, (6,), "joint_position_rad")
        low = self.config.joint_limits_rad[:, 0]
        high = self.config.joint_limits_rad[:, 1]
        if np.any(q < low) or np.any(q > high):
            return False, False, "joint_limit_violation"
        near = bool(np.any((q - low) <= self.config.joint_margin_rad) or np.any((high - q) <= self.config.joint_margin_rad))
        if near and self.config.reject_near_joint_limits:
            return False, True, "joint_limit_margin"
        return True, near, None

    def compute(self, robot_state: RobotState, residual=None) -> ControllerResult:
        """Compute a target pose and return a safe rejection instead of sending anything."""
        try:
            state = self.build_state(robot_state)
            requested = self.nominal_residual(robot_state) if residual is None else finite_array(residual, (6,), "residual")
            target_pose, applied, translation_clamped, rotation_clamped = self.apply_residual(
                robot_state.tcp_pose, requested
            )
            joint_ok, near_joint_limit, joint_reason = self._joint_safety(robot_state.joint_position_rad)
            if not joint_ok:
                return ControllerResult(None, state, requested, applied, False, translation_clamped, rotation_clamped, near_joint_limit, joint_reason)
            if self.config.workspace_min_m is not None:
                if np.any(target_pose.position_m < self.config.workspace_min_m) or np.any(target_pose.position_m > self.config.workspace_max_m):
                    return ControllerResult(None, state, requested, applied, False, translation_clamped, rotation_clamped, near_joint_limit, "workspace_rejected")
            return ControllerResult(target_pose, state, requested, applied, True, translation_clamped, rotation_clamped, near_joint_limit, None)
        except (TypeError, ValueError, FloatingPointError) as exc:
            return ControllerResult(None, None, None, None, False, False, False, False, str(exc))
