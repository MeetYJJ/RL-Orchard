"""Read-only HDF5 pipeline for low-dimensional BC/SAC-compatible data."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


STATE_DIM = 12
ACTION_DIM = 6
SUPPORTED_SCHEMA = "apple_rl_v1.2_sac_goal"


class DatasetValidationError(RuntimeError):
    """Raised when a transition claims to be valid but is misaligned."""


@dataclass
class EpisodeStatus:
    path: str
    loaded: bool
    reason: str
    policy_transitions: int = 0
    skipped_transitions: int = 0


@dataclass
class LoadedTransitions:
    states: np.ndarray
    actions: np.ndarray
    episode_paths: List[str]
    transition_indices: np.ndarray
    goals_xyz: np.ndarray
    initial_joints: np.ndarray
    statuses: List[EpisodeStatus]
    max_delta_joint_alignment_error: float
    max_next_joint_alignment_error: float
    max_next_tcp_alignment_error: float

    @property
    def episodes_loaded(self):
        return sum(1 for item in self.statuses if item.loaded)

    @property
    def episodes_skipped(self):
        return sum(1 for item in self.statuses if not item.loaded)


def _bool_attr(handle, name, default=False):
    return bool(handle.attrs.get(name, default))


def _text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _frozen_goal(episode, tolerance=1e-6):
    """Return the recorded pre-grasp goal; never recompute it from apple XYZ."""
    path = "episode_meta/pregrasp_pose_base"
    if path in episode:
        value = np.asarray(episode[path][:], dtype=np.float64)
        if value.shape == (7,) and np.all(np.isfinite(value)):
            return value, "episode_meta/pregrasp_pose_base"

    fallback = "teacher/pregrasp_pose_base"
    if fallback not in episode:
        return None, "missing pregrasp goal"
    values = np.asarray(episode[fallback][:], dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (7,):
        return None, "teacher pregrasp goal shape is not [T,7]"
    finite = np.all(np.isfinite(values), axis=1)
    if not np.any(finite):
        return None, "teacher pregrasp goal has no finite row"
    first = values[np.flatnonzero(finite)[0]]
    if not np.allclose(values[finite], first, atol=tolerance, rtol=0.0):
        return None, "teacher pregrasp goal is not frozen"
    return first, "teacher/pregrasp_pose_base fallback"


def _require_array(episode, path, shape_tail):
    if path not in episode:
        raise DatasetValidationError("missing dataset {}".format(path))
    value = np.asarray(episode[path][:])
    if value.ndim != 1 + len(shape_tail) or value.shape[1:] != tuple(shape_tail):
        raise DatasetValidationError(
            "{} has shape {}, expected [T,{}]".format(
                path, value.shape, ",".join(str(item) for item in shape_tail)
            )
        )
    return value


def load_bc_transitions(
    dataset_dir,
    max_transitions=None,
    alignment_tolerance=1e-5,
    strict_alignment=True,
):
    """Load valid Initial->Pre-Grasp expert transitions from HDF5 episodes."""
    root = Path(os.path.abspath(os.path.expanduser(str(dataset_dir))))
    paths = sorted(root.glob("episode_*.hdf5"))
    states = []
    actions = []
    episode_paths = []
    transition_indices = []
    goals = []
    initial_joints = []
    statuses = []
    max_delta_error = 0.0
    max_next_joint_error = 0.0
    max_next_tcp_error = 0.0

    for path in paths:
        try:
            with h5py.File(str(path), "r") as episode:
                schema = _text(episode.attrs.get("dataset_schema_version", "legacy"))
                if schema != SUPPORTED_SCHEMA:
                    statuses.append(EpisodeStatus(
                        str(path), False,
                        "legacy episode skipped (schema={})".format(schema),
                    ))
                    continue
                if not _bool_attr(episode, "episode_valid", False):
                    statuses.append(EpisodeStatus(str(path), False, "episode_valid=false"))
                    continue
                if _bool_attr(episode, "discarded", False):
                    statuses.append(EpisodeStatus(str(path), False, "discarded=true"))
                    continue

                goal, goal_source = _frozen_goal(episode)
                if goal is None:
                    statuses.append(EpisodeStatus(str(path), False, goal_source))
                    continue

                joint = _require_array(episode, "obs/joint_pos", (6,)).astype(np.float64)
                tcp = _require_array(episode, "obs/tcp_pose", (7,)).astype(np.float64)
                delta = _require_array(
                    episode, "actions/executed_delta_joint", (6,)
                ).astype(np.float64)
                next_joint = _require_array(
                    episode, "next_obs/joint_pos", (6,)
                ).astype(np.float64)
                next_tcp = _require_array(
                    episode, "next_obs/tcp_pose", (7,)
                ).astype(np.float64)
                policy = np.asarray(episode["valid/policy_action"][:], dtype=bool)
                delta_valid = np.asarray(
                    episode["valid/executed_delta_joint"][:], dtype=bool
                )
                count = len(joint)
                if any(len(item) != count for item in (
                    tcp, delta, next_joint, next_tcp, policy, delta_valid
                )):
                    raise DatasetValidationError("T dimension mismatch in {}".format(path))

                candidate = np.logical_and(policy, delta_valid)
                indices = np.flatnonzero(candidate)
                # A transition requires obs[t+1], regardless of an erroneous mask.
                indices = indices[indices + 1 < count]
                if not len(indices):
                    statuses.append(EpisodeStatus(
                        str(path), False, "no valid policy joint transition"
                    ))
                    continue

                physical_expected = joint[indices + 1] - joint[indices]
                delta_error = float(np.max(np.abs(delta[indices] - physical_expected)))
                next_joint_error = float(np.max(
                    np.abs(next_joint[indices] - joint[indices + 1])
                ))
                next_tcp_error = float(np.max(
                    np.abs(next_tcp[indices] - tcp[indices + 1])
                ))
                max_delta_error = max(max_delta_error, delta_error)
                max_next_joint_error = max(max_next_joint_error, next_joint_error)
                max_next_tcp_error = max(max_next_tcp_error, next_tcp_error)
                worst = max(delta_error, next_joint_error, next_tcp_error)
                if strict_alignment and worst > float(alignment_tolerance):
                    raise DatasetValidationError(
                        "alignment error in {}: delta={:.3e}, next_joint={:.3e}, "
                        "next_tcp={:.3e}, tolerance={:.3e}".format(
                            path, delta_error, next_joint_error, next_tcp_error,
                            float(alignment_tolerance),
                        )
                    )

                relative = goal[:3][None, :] - tcp[indices, :3]
                episode_states = np.concatenate(
                    (joint[indices], tcp[indices, :3], relative), axis=1
                )
                episode_actions = delta[indices]
                finite = np.all(np.isfinite(episode_states), axis=1)
                finite &= np.all(np.isfinite(episode_actions), axis=1)
                valid_indices = indices[finite]
                skipped = int(len(indices) - len(valid_indices))
                if not len(valid_indices):
                    statuses.append(EpisodeStatus(
                        str(path), False, "all candidate transitions contain NaN/Inf",
                        skipped_transitions=skipped,
                    ))
                    continue

                episode_states = episode_states[finite].astype(np.float32)
                episode_actions = episode_actions[finite].astype(np.float32)
                states.append(episode_states)
                actions.append(episode_actions)
                episode_paths.extend([str(path)] * len(valid_indices))
                transition_indices.extend(valid_indices.tolist())
                goals.append(goal[:3].astype(np.float32))
                if "episode_meta/initial_joint_pos" in episode:
                    initial = np.asarray(
                        episode["episode_meta/initial_joint_pos"][:], dtype=np.float32
                    )
                    if initial.shape == (6,) and np.all(np.isfinite(initial)):
                        initial_joints.append(initial)
                statuses.append(EpisodeStatus(
                    str(path), True,
                    "loaded via {}".format(goal_source),
                    policy_transitions=len(valid_indices),
                    skipped_transitions=skipped,
                ))
        except (OSError, KeyError, ValueError, DatasetValidationError) as exc:
            if isinstance(exc, DatasetValidationError) and strict_alignment and "alignment error" in str(exc):
                raise
            statuses.append(EpisodeStatus(str(path), False, str(exc)))

    if not states:
        raise DatasetValidationError(
            "no usable {} policy transitions found in {}".format(
                SUPPORTED_SCHEMA, root
            )
        )
    state_array = np.concatenate(states, axis=0)
    action_array = np.concatenate(actions, axis=0)
    index_array = np.asarray(transition_indices, dtype=np.int64)
    if max_transitions is not None and int(max_transitions) > 0:
        limit = min(int(max_transitions), len(state_array))
        state_array = state_array[:limit]
        action_array = action_array[:limit]
        episode_paths = episode_paths[:limit]
        index_array = index_array[:limit]

    return LoadedTransitions(
        states=state_array,
        actions=action_array,
        episode_paths=episode_paths,
        transition_indices=index_array,
        goals_xyz=np.asarray(goals, dtype=np.float32).reshape((-1, 3)),
        initial_joints=np.asarray(initial_joints, dtype=np.float32).reshape((-1, 6)),
        statuses=statuses,
        max_delta_joint_alignment_error=max_delta_error,
        max_next_joint_alignment_error=max_next_joint_error,
        max_next_tcp_alignment_error=max_next_tcp_error,
    )


@dataclass
class NormalizationStats:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_scale: np.ndarray
    eps: float = 1e-8
    action_scale_method: str = "per_joint_max_abs"

    @classmethod
    def fit(cls, states, actions, eps=1e-8):
        states = np.asarray(states, dtype=np.float64)
        actions = np.asarray(actions, dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != STATE_DIM:
            raise ValueError("states must have shape [N,12]")
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError("actions must have shape [N,6]")
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(actions)):
            raise ValueError("normalization input contains NaN/Inf")
        return cls(
            state_mean=np.mean(states, axis=0).astype(np.float32),
            state_std=np.std(states, axis=0).astype(np.float32),
            action_scale=np.maximum(
                np.max(np.abs(actions), axis=0), float(eps)
            ).astype(np.float32),
            eps=float(eps),
        )

    def normalize_states(self, states):
        value = np.asarray(states, dtype=np.float32)
        return (value - self.state_mean) / (self.state_std + self.eps)

    def normalize_actions(self, actions):
        return np.asarray(actions, dtype=np.float32) / self.action_scale

    def denormalize_actions(self, normalized_actions):
        return np.asarray(normalized_actions, dtype=np.float32) * self.action_scale

    def action_overflow(self, actions, tolerance=1e-6):
        normalized = self.normalize_actions(actions)
        magnitude = np.abs(normalized)
        return {
            "count_per_joint": np.sum(magnitude > 1.0 + tolerance, axis=0),
            "max_abs_per_joint": np.max(magnitude, axis=0),
        }

    def save(self, path, metadata=None):
        payload = {
            "state_mean": self.state_mean,
            "state_std": self.state_std,
            "action_scale": self.action_scale,
            "eps": np.asarray(self.eps, dtype=np.float64),
            "action_scale_method": np.asarray(self.action_scale_method),
        }
        if metadata:
            for key, value in metadata.items():
                if key in payload:
                    raise ValueError("normalization metadata collides with {}".format(key))
                payload[str(key)] = np.asarray(value)
        np.savez(str(path), **payload)

    @classmethod
    def load(cls, path):
        with np.load(str(path), allow_pickle=False) as values:
            return cls(
                state_mean=np.asarray(values["state_mean"], dtype=np.float32),
                state_std=np.asarray(values["state_std"], dtype=np.float32),
                action_scale=np.asarray(values["action_scale"], dtype=np.float32),
                eps=float(values["eps"]),
                action_scale_method=str(values["action_scale_method"]),
            )


class HDF5BCDataset(Dataset):
    """In-memory normalized tensors produced from validated HDF5 rows."""

    def __init__(self, transitions, normalization, clip_actions=True):
        normalized_states = normalization.normalize_states(transitions.states)
        normalized_actions = normalization.normalize_actions(transitions.actions)
        self.unclipped_normalized_actions = normalized_actions.copy()
        if clip_actions:
            normalized_actions = np.clip(normalized_actions, -1.0, 1.0)
        self.states = torch.from_numpy(normalized_states.astype(np.float32))
        self.actions = torch.from_numpy(normalized_actions.astype(np.float32))
        self.physical_actions = torch.from_numpy(
            transitions.actions.astype(np.float32)
        )

    def __len__(self):
        return len(self.states)

    def __getitem__(self, index):
        return self.states[index], self.actions[index]
