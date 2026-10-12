"""Independent serial-chain FK for the original RML63-B URDF.

This module deliberately does not read MuJoCo body poses.  It is used as an
independent check of the URDF-to-MJCF conversion and therefore only consumes
the original URDF joint origins, axes, and joint values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


LINK_NAMES = tuple("Link{}".format(index) for index in range(1, 7))
JOINT_NAMES = tuple("joint{}".format(index) for index in range(1, 7))


def _vector(value, default):
    return np.fromstring(value if value is not None else default, sep=" ", dtype=np.float64)


def _rpy_matrix(rpy):
    """Return URDF fixed-axis roll-pitch-yaw rotation, Rz(yaw)Ry(pitch)Rx(roll)."""
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.asarray([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def _axis_angle_matrix(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm <= 0.0 or not np.isfinite(norm):
        raise ValueError("URDF joint axis must be finite and non-zero")
    x, y, z = axis / norm
    c, s = np.cos(angle), np.sin(angle)
    one_minus_c = 1.0 - c
    return np.asarray([
        [c + x * x * one_minus_c, x * y * one_minus_c - z * s,
         x * z * one_minus_c + y * s],
        [y * x * one_minus_c + z * s, c + y * y * one_minus_c,
         y * z * one_minus_c - x * s],
        [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s,
         c + z * z * one_minus_c],
    ], dtype=np.float64)


def _transform(rotation, translation):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


@dataclass(frozen=True)
class URDFJoint:
    name: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray


class RML63BURDFChain:
    """The six-joint base_link -> Link6 chain from an original RML63-B URDF."""

    def __init__(self, urdf_path):
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        root = ET.parse(str(self.urdf_path)).getroot()
        joints = {}
        for element in root.findall("joint"):
            name = element.attrib.get("name")
            if name not in JOINT_NAMES:
                continue
            origin = element.find("origin")
            axis = element.find("axis")
            parent = element.find("parent")
            child = element.find("child")
            if origin is None or axis is None or parent is None or child is None:
                raise ValueError("URDF joint {!r} is missing kinematic fields".format(name))
            joints[name] = URDFJoint(
                name=name,
                parent=parent.attrib["link"],
                child=child.attrib["link"],
                origin_xyz=_vector(origin.attrib.get("xyz"), "0 0 0"),
                origin_rpy=_vector(origin.attrib.get("rpy"), "0 0 0"),
                axis=_vector(axis.attrib.get("xyz"), "0 0 1"),
            )
        missing = [name for name in JOINT_NAMES if name not in joints]
        if missing:
            raise ValueError("URDF is missing RML63-B joints: {}".format(missing))
        self.joints = tuple(joints[name] for name in JOINT_NAMES)
        if self.joints[0].parent != "base_link":
            raise ValueError("expected joint1 parent to be base_link")
        for previous, current in zip(self.joints, self.joints[1:]):
            if previous.child != current.parent:
                raise ValueError("URDF joints are not a serial base_link -> Link6 chain")

    def forward(self, joint_positions):
        q = np.asarray(joint_positions, dtype=np.float64)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            raise ValueError("joint_positions must be finite with shape [6]")
        poses = {"base_link": np.eye(4, dtype=np.float64)}
        parent_pose = poses["base_link"]
        for joint, angle, expected_link in zip(self.joints, q, LINK_NAMES):
            if joint.child != expected_link:
                raise ValueError("unexpected child link order in URDF: {}".format(joint.child))
            origin = _transform(_rpy_matrix(joint.origin_rpy), joint.origin_xyz)
            motion = _transform(_axis_angle_matrix(joint.axis, angle), np.zeros(3))
            child_pose = parent_pose @ origin @ motion
            poses[joint.child] = child_pose
            parent_pose = child_pose
        return poses


def rotation_error_rad(first, second):
    """Return the angle of first^-1 * second for two 3x3 rotations."""
    relative = np.asarray(first, dtype=np.float64).T @ np.asarray(second, dtype=np.float64)
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))
