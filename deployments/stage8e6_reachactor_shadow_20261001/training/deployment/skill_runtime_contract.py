"""Hardware-free runtime contracts for the two frozen skill Actors.

This module deliberately has no ROS, socket, RealMan SDK or MuJoCo import. It
can be unit-tested on the training machine and later wrapped by an Origin
feedback/command adapter. ``MockCommandWriter`` records commands only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .cartesian_controller_core import Pose
from .pose_math import matrix_to_rotvec, validate_rotation_matrix, finite_array


SKILL_PHASE_ORDER = ('GRASP_READY', 'GRIPPER_CLOSE', 'CONTACT_CONFIRM', 'LIFT', 'HOLD', 'SUCCESS')
REACH_ACTION_BOUNDS = np.asarray([.0015, .0015, .0015, .0075, .0075, .0075], dtype=np.float64)


@dataclass(frozen=True)
class HardwareFeedback:
    """Only measurements required by the frozen observation contracts."""
    joint_position_rad: np.ndarray
    tcp_pose: Pose
    timestamp_s: float
    arm_err: int | None
    sys_err: int | None
    apple_position_m: np.ndarray | None = None
    apple_rotation: np.ndarray | None = None
    gripper_aperture_m: float | None = None

    def __post_init__(self):
        object.__setattr__(self, 'joint_position_rad', finite_array(self.joint_position_rad, (6,), 'joint_position_rad'))
        if not np.isfinite(self.timestamp_s): raise ValueError('feedback timestamp must be finite')
        if self.apple_position_m is not None:
            object.__setattr__(self, 'apple_position_m', finite_array(self.apple_position_m, (3,), 'apple_position_m'))
        if self.apple_rotation is not None:
            object.__setattr__(self, 'apple_rotation', validate_rotation_matrix(self.apple_rotation, 'apple_rotation'))
        if self.gripper_aperture_m is not None and (not np.isfinite(self.gripper_aperture_m) or self.gripper_aperture_m < 0):
            raise ValueError('gripper aperture must be finite and nonnegative')


class RuntimePhase(str, Enum):
    IDLE = 'IDLE'; REACH = 'REACH'; GRASP_LIFT = 'GRASP_LIFT'
    RETURN_HOME = 'RETURN_HOME'; RELEASE = 'RELEASE'; DONE = 'DONE'; FAILED = 'FAILED'


class ObservationBuilder:
    """Build exactly the frozen 15D and 31D Actor inputs."""
    def __init__(self, reach_mean, reach_std, skill_mean, skill_std):
        self.reach_mean = finite_array(reach_mean, (15,), 'reach_mean')
        self.reach_std = np.maximum(finite_array(reach_std, (15,), 'reach_std'), 1e-6)
        self.skill_mean = finite_array(skill_mean, (31,), 'skill_mean')
        self.skill_std = np.maximum(finite_array(skill_std, (31,), 'skill_std'), 1e-6)

    @staticmethod
    def _raw_reach(feedback: HardwareFeedback, goal: Pose):
        rel_p = goal.position_m - feedback.tcp_pose.position_m
        rel_r = matrix_to_rotvec(goal.rotation @ feedback.tcp_pose.rotation.T)
        return np.r_[feedback.joint_position_rad, feedback.tcp_pose.position_m, rel_p, rel_r]

    def reach(self, feedback, goal):
        raw = self._raw_reach(feedback, goal)
        return ((raw-self.reach_mean)/self.reach_std).astype(np.float32)

    def grasp_lift(self, feedback, goal, *, gripper_command, phase,
                   initial_apple_height_m, hold_fraction):
        if feedback.apple_position_m is None or feedback.apple_rotation is None:
            raise ValueError('GraspLift observation requires measured apple pose')
        if feedback.gripper_aperture_m is None:
            raise ValueError('GraspLift observation requires measured gripper aperture')
        if phase not in SKILL_PHASE_ORDER: raise ValueError('unknown skill phase')
        if not np.isfinite(initial_apple_height_m) or not 0 <= hold_fraction <= 1:
            raise ValueError('invalid skill progress fields')
        raw_reach = self._raw_reach(feedback, goal)
        apple_rel_p = feedback.apple_position_m-feedback.tcp_pose.position_m
        apple_rel_r = matrix_to_rotvec(feedback.apple_rotation @ feedback.tcp_pose.rotation.T)
        phase_onehot = np.zeros(6); phase_onehot[SKILL_PHASE_ORDER.index(phase)] = 1
        raw = np.r_[raw_reach, apple_rel_p, apple_rel_r,
                    feedback.gripper_aperture_m,
                    gripper_command, phase_onehot, initial_apple_height_m, hold_fraction]
        if raw.shape != (31,) or not np.all(np.isfinite(raw)): raise ValueError('skill state is not finite')
        return ((raw-self.skill_mean)/self.skill_std).astype(np.float32)


def normalized_reach_action(action_normalized):
    value = finite_array(action_normalized, (6,), 'reach_action_normalized')
    if np.any(np.abs(value) > 1.0 + 1e-7): raise ValueError('reach action outside [-1,1]')
    return value * REACH_ACTION_BOUNDS


def normalized_skill_action(action_normalized):
    value = finite_array(action_normalized, (7,), 'skill_action_normalized')
    if np.any(np.abs(value) > 1.0 + 1e-7): raise ValueError('skill action outside [-1,1]')
    return np.r_[value[:6] * REACH_ACTION_BOUNDS, value[6]]


@dataclass(frozen=True)
class SafetyDecision:
    accepted: bool
    reason: str | None
    applied_action_m_rad: np.ndarray | None


class ActionSafetyGate:
    """Offline equivalent of the Origin pre-send safety gate."""
    def __init__(self, max_feedback_age_s=.1): self.max_feedback_age_s = float(max_feedback_age_s)
    def check_feedback(self, feedback: HardwareFeedback, now_s):
        if feedback.arm_err != 0: return False, 'arm_err'
        if feedback.sys_err != 0: return False, 'sys_err'
        age = float(now_s)-float(feedback.timestamp_s)
        if not np.isfinite(age) or age < 0 or age > self.max_feedback_age_s: return False, 'stale_feedback'
        return True, None
    def cartesian(self, action_m_rad, feedback: HardwareFeedback, now_s):
        ok, reason = self.check_feedback(feedback, now_s)
        if not ok: return SafetyDecision(False, reason, None)
        action = finite_array(action_m_rad, (6,), 'cartesian_action')
        applied = np.clip(action, -REACH_ACTION_BOUNDS, REACH_ACTION_BOUNDS)
        return SafetyDecision(True, None, applied)


@dataclass
class MockCommandWriter:
    """Offline command sink: records only, never opens a connection."""
    cartesian: list = field(default_factory=list)
    gripper: list = field(default_factory=list)

    def write_cartesian(self, action_m_rad, timestamp_s, *, source):
        action = finite_array(action_m_rad, (6,), 'cartesian_action')
        self.cartesian.append(dict(action_m_rad=action.tolist(), timestamp_s=float(timestamp_s), source=str(source)))
        return dict(accepted=True, command_type='cartesian_target_candidate', hardware_sent=False)

    def write_gripper(self, command, timestamp_s, *, source):
        if not np.isfinite(command) or command < -1 or command > 1: raise ValueError('invalid gripper command')
        self.gripper.append(dict(command=float(command), timestamp_s=float(timestamp_s), source=str(source)))
        return dict(accepted=True, command_type='gripper_candidate', hardware_sent=False)


class RuntimeStateMachine:
    """MuJoCo-independent phase gate; hardware events are supplied by adapter."""
    def __init__(self): self.phase = RuntimePhase.IDLE
    def start(self): self.phase = RuntimePhase.REACH
    def event(self, *, success=False, failure=None):
        if failure:
            self.phase = RuntimePhase.FAILED; return self.phase
        if self.phase == RuntimePhase.REACH and success: self.phase = RuntimePhase.GRASP_LIFT
        elif self.phase == RuntimePhase.GRASP_LIFT and success: self.phase = RuntimePhase.RETURN_HOME
        elif self.phase == RuntimePhase.RETURN_HOME and success: self.phase = RuntimePhase.RELEASE
        elif self.phase == RuntimePhase.RELEASE and success: self.phase = RuntimePhase.DONE
        return self.phase
