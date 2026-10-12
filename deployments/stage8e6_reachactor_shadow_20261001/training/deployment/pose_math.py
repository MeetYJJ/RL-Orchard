"""Small, hardware-independent rigid-pose math helpers.

All translations are metres and all rotations are radians.  The functions in
this module have no simulator, ROS, PyTorch, or RealMan dependency.
"""

from __future__ import annotations

import math

import numpy as np


def finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return array.copy()


def validate_rotation_matrix(rotation, name="rotation"):
    value = finite_array(rotation, (3, 3), name)
    if not np.allclose(value.T @ value, np.eye(3), atol=1e-7):
        raise ValueError(f"{name} must be orthonormal")
    if not np.isclose(np.linalg.det(value), 1.0, atol=1e-7):
        raise ValueError(f"{name} must have determinant +1")
    return value


def rotvec_to_matrix(rotvec):
    """SO(3) exponential map using the fixed/base-frame convention."""
    value = finite_array(rotvec, (3,), "rotvec")
    angle = float(np.linalg.norm(value))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = value / angle
    x, y, z = axis
    skew = np.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def matrix_to_rotvec(rotation):
    """Shortest SO(3) logarithm, including the near-pi branch."""
    value = validate_rotation_matrix(rotation)
    cosine = float(np.clip((np.trace(value) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew = np.asarray(
        [
            value[2, 1] - value[1, 2],
            value[0, 2] - value[2, 0],
            value[1, 0] - value[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1e-8:
        return 0.5 * skew
    if math.pi - angle < 1e-5:
        eigenvalues, eigenvectors = np.linalg.eig(value)
        index = int(np.argmin(np.abs(eigenvalues - 1.0)))
        axis = np.real(eigenvectors[:, index])
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        if float(np.dot(axis, skew)) < 0.0:
            axis = -axis
        return axis * angle
    return skew * (angle / (2.0 * math.sin(angle)))


def quaternion_xyzw_to_matrix(quaternion):
    q = finite_array(quaternion, (4,), "quaternion")
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        raise ValueError("quaternion norm is zero")
    x, y, z, w = q / norm
    return validate_rotation_matrix(
        np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        ),
        "quaternion rotation",
    )


def make_transform(rotation, translation):
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = validate_rotation_matrix(rotation)
    result[:3, 3] = finite_array(translation, (3,), "translation")
    return result


def invert_transform(transform):
    value = finite_array(transform, (4, 4), "transform")
    if not np.allclose(value[3], [0.0, 0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("transform must have homogeneous last row")
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = value[:3, :3].T
    result[:3, 3] = -value[:3, :3].T @ value[:3, 3]
    return result


def compose_pose(current_position, current_rotation, translation_delta, rotation_delta):
    """Apply a base-frame residual: ``p+d``, ``Exp(dr) @ R``."""
    position = finite_array(current_position, (3,), "current_position")
    rotation = validate_rotation_matrix(current_rotation, "current_rotation")
    delta_position = finite_array(translation_delta, (3,), "translation_delta")
    delta_rotation = finite_array(rotation_delta, (3,), "rotation_delta")
    return position + delta_position, rotvec_to_matrix(delta_rotation) @ rotation
