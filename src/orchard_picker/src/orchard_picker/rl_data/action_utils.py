"""Quaternion utilities used to derive executed Cartesian actions."""

import math

import numpy as np


def normalize_quaternion(quaternion):
    """Return a normalized xyzw quaternion or identity for a zero input."""
    value = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm < 1e-12:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return value / norm


def quaternion_multiply(left, right):
    """Multiply xyzw quaternions and return a normalized xyzw quaternion."""
    lx, ly, lz, lw = normalize_quaternion(left)
    rx, ry, rz, rw = normalize_quaternion(right)
    return normalize_quaternion([
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    ])


def relative_rotation_vector(current_xyzw, next_xyzw):
    """Return rotvec(inverse(current) * next), choosing the shortest arc."""
    current = normalize_quaternion(current_xyzw)
    inverse = np.asarray(
        [-current[0], -current[1], -current[2], current[3]],
        dtype=np.float64,
    )
    relative = quaternion_multiply(inverse, next_xyzw)
    if relative[3] < 0.0:
        relative = -relative
    vector = relative[:3]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, float(relative[3]))
    return vector * (angle / vector_norm)


def executed_delta_tcp(current_pose, next_pose):
    """Compute [dx,dy,dz,dRx,dRy,dRz] from two xyz+xyzw poses."""
    current = np.asarray(current_pose, dtype=np.float64)
    following = np.asarray(next_pose, dtype=np.float64)
    if current.shape != (7,) or following.shape != (7,):
        raise ValueError("TCP poses must have shape (7,)")
    translation = following[:3] - current[:3]
    rotation = relative_rotation_vector(current[3:], following[3:])
    return np.concatenate((translation, rotation)).astype(np.float32)
