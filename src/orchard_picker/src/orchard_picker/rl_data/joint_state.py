"""Pure helpers for normalising and checking six-axis joint feedback."""

import numpy as np


class JointStateMappingError(ValueError):
    """Raised when a JointState message cannot be mapped unambiguously."""


def normalize_joint_positions(names, positions, expected_names):
    """Return positions ordered by ``expected_names`` without guessing.

    ``sensor_msgs/JointState`` does not guarantee order.  Missing, duplicate,
    non-finite or dimension-mismatched data is rejected instead of being
    silently assigned to the wrong joint.
    """
    expected = tuple(str(name) for name in expected_names)
    if len(expected) != 6 or len(set(expected)) != 6:
        raise JointStateMappingError("expected_names must contain six unique names")
    actual = [str(name) for name in (names or [])]
    values = np.asarray(positions, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] != len(actual):
        raise JointStateMappingError("joint name/position dimensions do not match")
    if len(actual) != len(set(actual)):
        raise JointStateMappingError("duplicate joint names")
    indices = []
    for name in expected:
        if name not in actual:
            raise JointStateMappingError("missing joint name: {}".format(name))
        indices.append(actual.index(name))
    ordered = values[np.asarray(indices, dtype=np.int64)]
    if not np.all(np.isfinite(ordered)):
        raise JointStateMappingError("joint positions contain NaN or Inf")
    return ordered.astype(np.float32)


def finite_joint_vector(value):
    """Return a finite float32 six-vector, or ``None``."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.shape != (6,) or not np.all(np.isfinite(array)):
        return None
    return array.astype(np.float32)


def estimate_rate_hz(timestamps_ns):
    """Estimate source feedback rate from callback timestamp span.

    Duplicate timestamps are repeated deliveries of the same feedback frame;
    they are removed.  The result uses the number of observed intervals over
    the full source-time span, rather than recorder sample timing or a median
    interval that can hide dropped frames.
    """
    values = np.asarray(list(timestamps_ns), dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return np.nan
    values = np.sort(values)
    values = np.unique(values)
    if values.size < 2:
        return np.nan
    span_ns = float(values[-1] - values[0])
    if span_ns <= 0.0:
        return np.nan
    return float((values.size - 1) * 1e9 / span_ns)


def transition_delta(current, next_value):
    """Compute q[t+1]-q[t] only for two finite six-vectors."""
    first = finite_joint_vector(current)
    second = finite_joint_vector(next_value)
    if first is None or second is None:
        return None
    return (second - first).astype(np.float32)
