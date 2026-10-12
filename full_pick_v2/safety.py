from __future__ import annotations
from collections import deque
from pathlib import Path
import time
import xml.etree.ElementTree as ET
import numpy as np
from .transforms import actual_tcp, so3_log, so3_exp

URDF = Path("/home/nvidia/orchard_experiment_rl_collect/rm_63_b_description/urdf/rm_63_b_description.urdf")

class PickFault(RuntimeError):
    """A precise reason that blocks all downstream grasp phases."""


SYSTEM_ERROR_CODES = {
    0x0000: "normal system",
    0x1001: "abnormal joint communication",
    0x1002: "target angle out of limit",
    0x1003: "point unreachable as a singularity",
    0x1004: "real-time kernel communication error",
    0x1005: "joint communication bus error",
    0x1006: "planning layer kernel error",
    0x1007: "joint overspeed",
    0x1008: "end interface board disconnected",
    0x1009: "speed out of limit",
    0x100A: "acceleration out of limit",
    0x100B: "joint brake release failed",
    0x100C: "overspeed during drag teaching",
    0x100D: "robot collision",
    0x100E: "null work frame",
    0x100F: "null tool frame",
    0x1010: "joint disabling error",
    0x1011: "circular planning error",
    0x1012: "self-collision error",
}

JOINT_ERROR_CODES = {
    0x0000: "normal joint",
    0x0001: "FOC error",
    0x0002: "overvoltage",
    0x0004: "undervoltage",
    0x0008: "over-temperature",
    0x0010: "startup failure",
    0x0020: "encoder error",
    0x0040: "overcurrent",
    0x0080: "software error",
    0x0100: "temperature sensor error",
    0x0200: "position overrun error",
}


def _as_error_values(value):
    if value is None:
        return []
    values = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
    try:
        return [int(item) for item in values]
    except (TypeError, ValueError):
        return []


def describe_controller_errors(err, joint_err):
    system = _as_error_values(err)
    joints = _as_error_values(joint_err)
    system_details = [
        "%d(0x%04X):%s" % (code, code, SYSTEM_ERROR_CODES.get(code, "UNKNOWN"))
        for code in system
    ]
    joint_details = [
        "%d(0x%04X):%s" % (code, code, JOINT_ERROR_CODES.get(code, "UNKNOWN"))
        for code in joints
    ]
    unknown = any(code not in SYSTEM_ERROR_CODES for code in system)
    unknown = unknown or any(code not in JOINT_ERROR_CODES and code != 0 for code in joints)
    return system_details, joint_details, unknown

class Safety:
    def __init__(self, config):
        self.config = config
        self.floor = float(config.raw["pick_task_manager"]["ros__parameters"].get("min_axis_endpoint_z_m", .010))
        # Command diagnostics are deliberately one-shot on a valid command.
        # Printing from the 200 Hz transport path would itself perturb timing;
        # invalid commands are always printed when they are rejected.
        self._command_target_logged = False
        self.chain = []
        root = ET.parse(str(URDF)).getroot()
        # URDF order is not guaranteed; follow the base -> robot_end chain.
        joints = {j.find("child").get("link"): j for j in root.findall("joint")}
        child = "Link6"
        while child in joints:
            j = joints[child]; self.chain.insert(0, j); child = j.find("parent").get("link")
        if sum(j.get("type") in ("revolute", "continuous") for j in self.chain) != 6:
            raise PickFault("URDF_JOINT_CHAIN_INVALID")

    def _validate_position_basics(self, p, label):
        p = np.asarray(p, dtype=np.float64)
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            raise PickFault(f"WORKSPACE_INVALID:{label}")
        if p[2] < self.floor:
            raise PickFault(f"WORKSPACE_FLOOR:{label}")
        return p

    def current_position(self, p, label="current_robot_state", emit=True):
        """Validate feedback pose without treating it as a command target.

        The configured Cartesian box constrains poses that we *command*.  A
        robot can legitimately be outside that box at startup (for example
        after a manual recovery), so current feedback is checked for finite
        values and the floor only.  Joint/error/enable checks remain in
        ``feedback`` and are still mandatory.
        """
        try:
            p = self._validate_position_basics(p, label)
        except PickFault:
            if emit:
                print("CHECK_CURRENT_ROBOT_STATE", flush=True)
                print("current_xyz=" + np.array2string(np.asarray(p), precision=9), flush=True)
                print("target_xyz=none", flush=True)
                print("workspace_result=invalid_current_state", flush=True)
            raise
        if emit:
            print("CHECK_CURRENT_ROBOT_STATE", flush=True)
            print("current_xyz=" + np.array2string(p, precision=9), flush=True)
            print("target_xyz=none", flush=True)
            print("workspace_result=current_state_valid_command_box_not_applied", flush=True)
        return p

    def position(self, p, label="pose"):
        """Validate a pose using the legacy configured policy when requested."""
        p = self._validate_position_basics(p, label)
        if self.config.enforce_workspace and (np.any(p < self.config.workspace_min) or np.any(p > self.config.workspace_max)):
            raise PickFault(f"WORKSPACE_LIMIT:{label}")
        return p

    def command_position(self, p, label="commanded_pose"):
        """Validate a final command without the legacy Cartesian box.

        The configured ``workspace_min``/``workspace_max`` values are kept in
        the formal picker configuration for compatibility and diagnostics, but
        their axis-aligned base-frame interpretation is not valid for this
        RealMan installation.  Command validation therefore retains finite
        value and floor checks here; joint limits, controller feedback, UDP
        freshness, serialized-pose validation, and stop-on-fault remain active
        in their existing layers.
        """
        p = np.asarray(p, dtype=np.float64)
        diagnostic = not self._command_target_logged
        if diagnostic or p.shape != (3,) or not np.all(np.isfinite(p)):
            print("TARGET_POSITION_CHECK:", flush=True)
            print("xyz=" + np.array2string(p, precision=9), flush=True)
            print("frame=base_link->robot_end", flush=True)
            print("unit=m", flush=True)
        try:
            p = self._validate_position_basics(p, label)
        except PickFault:
            print("WORKSPACE_CARTESIAN_LIMIT_ENABLED=false", flush=True)
            print("WORKSPACE_CHECK_SKIPPED_FOR_CARTESIAN_LIMIT=true", flush=True)
            raise
        if diagnostic:
            print("WORKSPACE_CARTESIAN_LIMIT_ENABLED=false", flush=True)
            print("WORKSPACE_CHECK_SKIPPED_FOR_CARTESIAN_LIMIT=true", flush=True)
            print("Other safety checks active", flush=True)
            self._command_target_logged = True
        return p

    def joints(self, q):
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise PickFault("JOINT_FEEDBACK_INVALID")
        p, R, i = np.zeros(3), np.eye(3), 0
        for joint in self.chain:
            origin = joint.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
            p = p + R @ xyz
            R = R @ so3_exp([0,0,rpy[2]]) @ so3_exp([0,rpy[1],0]) @ so3_exp([rpy[0],0,0])
            if joint.get("type") in ("revolute", "continuous"):
                if p[2] < self.floor:
                    raise PickFault(f"AXIS_ENDPOINT_FLOOR:J{i+1}")
                limit = joint.find("limit")
                if limit is not None and joint.get("type") != "continuous":
                    if q[i] < float(limit.get("lower")) - 1e-5 or q[i] > float(limit.get("upper")) + 1e-5:
                        raise PickFault(f"JOINT_LIMIT:J{i+1}")
                axis = joint.find("axis")
                v = np.fromstring(axis.get("xyz", "0 0 1"), sep=" ") if axis is not None else np.asarray([0,0,1.])
                R = R @ so3_exp(v*q[i]); i += 1

    def feedback(self, state):
        if state is None:
            raise PickFault("UDP_UNAVAILABLE")
        if not state.fresh:
            raise PickFault("UDP_STALE")
        if not state.healthy:
            system_details, joint_details, unknown = describe_controller_errors(
                state.err, state.joint_err_code
            )
            if system_details:
                print("CONTROLLER_ERROR_CODE = " + repr(system_details), flush=True)
            if joint_details:
                print("JOINT_ERROR_CODE = " + repr(joint_details), flush=True)
            if unknown:
                print("CONTROLLER_ERROR_CODE_UNKNOWN = true", flush=True)
            raise PickFault(f"CONTROLLER_ERROR:err={state.err},joint_err={state.joint_err_code},enabled={state.joint_en_flag}")
        # Feedback is current robot state, not a command.  Do not apply the
        # Cartesian command box here; retain finite/floor and joint safety.
        self.current_position(state.end_xyz_m, "actual_robot_end", emit=False)
        self.joints(state.joint_rad)

class SettleWindow:
    def __init__(self):
        self.samples = deque()
        self.stable_since = None

    def add(self, state, R, apple):
        now = state.receipt_mono
        if self.samples and now <= self.samples[-1][0]:
            return False
        tcp = actual_tcp(state.end_xyz_m, R)
        self.samples.append((now, tcp.copy(), R.copy()))
        while len(self.samples) > 2 and now - self.samples[1][0] >= .2:
            self.samples.popleft()
        stable = False
        if len(self.samples) >= 3 and now-self.samples[0][0] >= .2:
            duration = now-self.samples[0][0]
            span = float(np.max(np.linalg.norm(np.asarray([s[1] for s in self.samples])-tcp, axis=1)))
            speed = np.linalg.norm(tcp-self.samples[0][1])/duration
            angular_speed = np.linalg.norm(so3_log(R @ self.samples[0][2].T))/duration
            stable = (span <= .0005 and speed <= .002 and angular_speed <= .020 and
                      all(np.linalg.norm(s[1]-apple) <= .005 for s in self.samples))
        if not stable:
            self.stable_since = None
        elif self.stable_since is None:
            self.stable_since = now
        return stable and now-self.stable_since >= .2

class MotionWindow:
    """Windowed progress guards; never classify a single 20 ms UDP delta."""
    def __init__(self):
        self.samples = deque()
        self.stationary_since = None
        self.reverse_since = None

    def check(self, now, tcp, apple, distance):
        self.samples.append((now, np.asarray(tcp).copy(), np.asarray(apple).copy(), distance))
        while len(self.samples) > 2 and now-self.samples[1][0] >= 1.0:
            self.samples.popleft()
        if now-self.samples[0][0] < 1.0 or distance < .010:
            return
        first = self.samples[0]
        # Moving fruit invalidates comparisons against an old world target.
        if np.linalg.norm(apple-first[2]) > .010:
            self.stationary_since = self.reverse_since = None
            return
        travel = float(np.linalg.norm(tcp-first[1]))
        stagnant = travel < .0003 or first[3]-distance < .0002
        reverse = distance > first[3]+.005
        self.stationary_since = (self.stationary_since or now) if stagnant else None
        self.reverse_since = (self.reverse_since or now) if reverse else None
        if self.reverse_since is not None and now-self.reverse_since >= 1.0:
            raise PickFault("MOTION_REVERSED:rolling_window")
        if self.stationary_since is not None and now-self.stationary_since >= 4.0:
            raise PickFault("POLICY_NOT_CONVERGED:rolling_window_stagnation")
