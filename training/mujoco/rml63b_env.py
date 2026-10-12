"""Minimal MuJoCo position-control environment for the RML63-B arm."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


JOINT_NAMES = tuple("joint{}".format(index) for index in range(1, 7))
LINK_NAMES = tuple("Link{}".format(index) for index in range(1, 7))
TCP_SITE_NAME = "robot_end"


class RML63BPregraspEnv:
    """Fixed-base RML63-B model, controlled with normalized joint deltas."""

    def __init__(self, xml_path, action_scale, distance_threshold_m=0.005,
                 frame_skip=125):
        self.xml_path = Path(xml_path).expanduser().resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.action_scale = np.asarray(action_scale, dtype=np.float64)
        if self.action_scale.shape != (6,) or not np.all(np.isfinite(self.action_scale)) or np.any(self.action_scale <= 0):
            raise ValueError("action_scale must be finite, positive, and shape [6]")
        self.distance_threshold_m = float(distance_threshold_m)
        if not np.isfinite(self.distance_threshold_m) or self.distance_threshold_m <= 0:
            raise ValueError("distance_threshold_m must be positive and finite")
        self.frame_skip = int(frame_skip)
        if self.frame_skip < 1:
            raise ValueError("frame_skip must be positive")

        self.joint_ids = np.asarray([self._id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES], dtype=np.int32)
        self.qpos_addresses = self.model.jnt_qposadr[self.joint_ids].astype(np.int32)
        self.dof_addresses = self.model.jnt_dofadr[self.joint_ids].astype(np.int32)
        self.link_body_ids = np.asarray([
            self._id(mujoco.mjtObj.mjOBJ_BODY, name) for name in LINK_NAMES
        ], dtype=np.int32)
        self.base_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self.actuator_ids = np.asarray([
            self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, "position_" + name)
            for name in JOINT_NAMES
        ], dtype=np.int32)
        self.site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, TCP_SITE_NAME)
        if not np.array_equal(self.model.actuator_trnid[self.actuator_ids, 0], self.joint_ids):
            raise ValueError("position actuators do not follow joint1..joint6 order")
        if not np.all(self.model.jnt_limited[self.joint_ids]):
            raise ValueError("all six RML63-B joints must have limits")
        if not np.all(self.model.jnt_type[self.joint_ids] == mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError("expected six hinge joints")
        self.joint_limits = self.model.jnt_range[self.joint_ids].copy()
        self.goal_xyz = None

    def _id(self, object_type, name):
        value = mujoco.mj_name2id(self.model, object_type, name)
        if value < 0:
            raise ValueError("MuJoCo model is missing {} {!r}".format(object_type, name))
        return int(value)

    def _state(self):
        joints = self.data.qpos[self.qpos_addresses].copy()
        tcp_xyz = self.data.site_xpos[self.site_id].copy()
        state = np.concatenate((joints, tcp_xyz, self.goal_xyz - tcp_xyz)).astype(np.float32)
        if state.shape != (12,):
            raise RuntimeError("RML63-B policy state must have shape [12]")
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("MuJoCo state contains NaN/Inf")
        return state

    def _distance(self):
        tcp_xyz = self.data.site_xpos[self.site_id]
        distance = float(np.linalg.norm(tcp_xyz - self.goal_xyz))
        if not np.isfinite(distance):
            raise FloatingPointError("TCP distance contains NaN/Inf")
        return distance

    def joint_positions(self):
        """Return the six model joints in the fixed joint1..joint6 order."""
        return self.data.qpos[self.qpos_addresses].copy()

    def tcp_xyz(self):
        """Return the robot_end site position in the MuJoCo world/base frame."""
        return self.data.site_xpos[self.site_id].copy()

    def get_link_pose(self, link_name):
        """Return a named kinematic link pose, never a mesh/inertial pose."""
        body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, str(link_name))
        return {
            "position": self.data.xpos[body_id].copy(),
            "rotation": self.data.xmat[body_id].reshape(3, 3).copy(),
        }

    def get_flange_pose(self):
        """Return the Link6/robot_end flange pose in the MuJoCo base frame."""
        pose = self.get_link_pose("Link6")
        site_position = self.data.site_xpos[self.site_id]
        if not np.allclose(pose["position"], site_position, atol=1e-10, rtol=0.0):
            raise RuntimeError("robot_end site is not coincident with the Link6 flange")
        return pose

    def get_tool_pose(self, tool_offset_xyz, tool_offset_rotation=None):
        """Apply an explicitly supplied flange->tool transform.

        No tool transform is guessed or enabled by default.  Callers must
        provide the separately documented, physical tool transform.
        """
        offset_xyz = np.asarray(tool_offset_xyz, dtype=np.float64)
        if offset_xyz.shape != (3,) or not np.all(np.isfinite(offset_xyz)):
            raise ValueError("tool_offset_xyz must be finite with shape [3]")
        if tool_offset_rotation is None:
            offset_rotation = np.eye(3, dtype=np.float64)
        else:
            offset_rotation = np.asarray(tool_offset_rotation, dtype=np.float64)
            if offset_rotation.shape != (3, 3) or not np.all(np.isfinite(offset_rotation)):
                raise ValueError("tool_offset_rotation must be finite with shape [3,3]")
        flange = self.get_flange_pose()
        return {
            "position": flange["position"] + flange["rotation"] @ offset_xyz,
            "rotation": flange["rotation"] @ offset_rotation,
        }

    def reset(self, initial_joint_pos, pregrasp_xyz):
        q0 = np.asarray(initial_joint_pos, dtype=np.float64)
        goal = np.asarray(pregrasp_xyz, dtype=np.float64)
        if q0.shape != (6,) or not np.all(np.isfinite(q0)):
            raise ValueError("initial_joint_pos must be finite with shape [6]")
        if goal.shape != (3,) or not np.all(np.isfinite(goal)):
            raise ValueError("pregrasp_xyz must be finite with shape [3]")
        tolerance = 1e-8
        if np.any(q0 < self.joint_limits[:, 0] - tolerance) or np.any(q0 > self.joint_limits[:, 1] + tolerance):
            raise ValueError("episode initial_joint_pos is outside RML63-B limits")

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_addresses] = q0
        self.data.qvel[self.dof_addresses] = 0.0
        self.goal_xyz = goal.copy()
        self.data.ctrl[self.actuator_ids] = q0
        mujoco.mj_forward(self.model, self.data)
        return self._state()

    def step(self, action_norm):
        action = np.asarray(action_norm, dtype=np.float64)
        if action.shape != (6,):
            raise ValueError("action_norm must have shape [6]")
        if not np.all(np.isfinite(action)):
            raise FloatingPointError("policy action contains NaN/Inf")
        if np.any(np.abs(action) > 1.0 + 1e-6):
            raise ValueError("action_norm must be in [-1,1]")

        return self.step_delta_q(action * self.action_scale)

    def step_delta_q(self, delta_q):
        """Apply one raw executed_delta_joint action in radians."""
        delta_q = np.asarray(delta_q, dtype=np.float64)
        if delta_q.shape != (6,):
            raise ValueError("delta_q must have shape [6]")
        if not np.all(np.isfinite(delta_q)):
            raise FloatingPointError("delta_q contains NaN/Inf")

        q_current = self.data.qpos[self.qpos_addresses].copy()
        q_target_requested = q_current + delta_q
        below = q_target_requested < self.joint_limits[:, 0]
        above = q_target_requested > self.joint_limits[:, 1]
        joint_limit_violation = bool(np.any(below | above))
        q_target = np.clip(q_target_requested, self.joint_limits[:, 0], self.joint_limits[:, 1])

        self.data.ctrl[self.actuator_ids] = q_target
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        q_after = self.data.qpos[self.qpos_addresses]
        actual_violation = bool(np.any(q_after < self.joint_limits[:, 0] - 1e-5) or
                                np.any(q_after > self.joint_limits[:, 1] + 1e-5))
        joint_limit_violation = joint_limit_violation or actual_violation
        state = self._state()
        distance = self._distance()
        info = {
            "distance_m": distance,
            "success": distance < self.distance_threshold_m,
            "joint_limit_violation": joint_limit_violation,
            "nan_inf": False,
            "q_current": q_current,
            "delta_q": delta_q,
            "q_target": q_target,
            "q_target_requested": q_target_requested,
        }
        return state, bool(info["success"]), info
