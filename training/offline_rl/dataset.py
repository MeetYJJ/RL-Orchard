"""Read-only Offline RL transition dataset for Stage 3A.

This module deliberately stops at ``(s, a, r, s', done)``.  It does not
contain a replay buffer or a learning loop.  HDF5 files are never modified.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from ..dataset import ACTION_DIM, STATE_DIM, SUPPORTED_SCHEMA
    from ..models import SACActor
except ImportError:  # direct execution with training/ on sys.path
    from dataset import ACTION_DIM, STATE_DIM, SUPPORTED_SCHEMA
    from models import SACActor

from .reward import OfflineRLRewardComposer, RewardConfig


def _text(value):
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no", ""):
            return False
    if isinstance(value, np.ndarray):
        return bool(value.reshape(-1)[0]) if value.size else bool(default)
    return bool(value)


def episode_id(path):
    return Path(path).stem


def manifest_hash(manifest):
    """Match the hash used by ``train_actor_bc_full.py``."""
    payload = {
        key: manifest[key]
        for key in (
            "dataset_schema_version", "seed", "train_ratio", "val_ratio",
            "test_ratio", "train_episode_ids", "validation_episode_ids",
            "test_episode_ids",
        ) if key in manifest
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def load_split_manifest(path, dataset_dir=None):
    path = Path(path).expanduser().resolve()
    with path.open("r") as stream:
        manifest = json.load(stream)
    if manifest.get("dataset_schema_version") != SUPPORTED_SCHEMA:
        raise ValueError("split manifest schema is not {}".format(SUPPORTED_SCHEMA))
    expected = manifest_hash(manifest)
    if manifest.get("manifest_hash", expected) != expected:
        raise ValueError("split manifest hash does not match its contents")
    groups = {
        "train": set(manifest.get("train_episode_ids", [])),
        "validation": set(manifest.get("validation_episode_ids", [])),
        "test": set(manifest.get("test_episode_ids", [])),
    }
    names = tuple(groups)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            overlap = groups[first] & groups[second]
            if overlap:
                raise ValueError("episode split overlap: {}".format(sorted(overlap)))
    if any(not groups[name] for name in names):
        raise ValueError("train, validation and test must all be non-empty")
    if dataset_dir is not None:
        root = Path(dataset_dir).expanduser().resolve()
        available = {p.stem for p in root.glob("episode_*.hdf5")}
        unknown = (groups["train"] | groups["validation"] | groups["test"]) - available
        if unknown:
            raise ValueError("manifest references missing episodes: {}".format(sorted(unknown)))
    return manifest


def _load_normalization(path, manifest=None):
    path = Path(path).expanduser().resolve()
    with np.load(str(path), allow_pickle=False) as data:
        required = ("state_mean", "state_std", "action_scale")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError("normalization missing {}".format(missing))
        values = {key: np.asarray(data[key], dtype=np.float32) for key in required}
        for key, shape in (("state_mean", (12,)), ("state_std", (12,)), ("action_scale", (6,))):
            if values[key].shape != shape or not np.all(np.isfinite(values[key])):
                raise ValueError("{} must be finite with shape {}".format(key, shape))
        if np.any(values["action_scale"] <= 0):
            raise ValueError("action_scale must be positive")
        metadata = {}
        for key in data.files:
            if key not in required:
                raw = data[key]
                metadata[key] = raw.item() if raw.shape == () else raw.tolist()
    if manifest is not None:
        expected = manifest_hash(manifest)
        actual = metadata.get("split_manifest_hash")
        if actual not in (None, expected, str(expected)):
            raise ValueError("normalization split hash does not match manifest")
    if metadata.get("statistics_source") not in (None, "train_only"):
        raise ValueError("normalization statistics_source must be train_only")
    if metadata.get("dataset_schema_version") not in (None, SUPPORTED_SCHEMA):
        raise ValueError("normalization dataset schema is incompatible")
    return values, metadata


def audit_actor_checkpoint(path, manifest=None):
    """Load metadata and weights, without training or changing the checkpoint."""
    checkpoint = torch.load(str(Path(path).expanduser().resolve()), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("actor checkpoint must be a mapping")
    required = {"actor_class": "SACActor", "state_dim": 12, "action_dim": 6, "hidden_dim": 256}
    for key, expected in required.items():
        if checkpoint.get(key) != expected:
            raise ValueError("checkpoint {}={} expected {}".format(key, checkpoint.get(key), expected))
    if checkpoint.get("architecture_version") not in (None, SACActor.architecture_version):
        raise ValueError("checkpoint architecture_version is incompatible")
    if manifest is not None:
        if checkpoint.get("dataset_schema_version") not in (None, SUPPORTED_SCHEMA):
            raise ValueError("checkpoint dataset schema is incompatible")
        expected_hash = manifest_hash(manifest)
        if checkpoint.get("split_manifest_hash") not in (None, expected_hash, str(expected_hash)):
            raise ValueError("checkpoint split hash is incompatible")
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
    if state_dict is None:
        raise ValueError("checkpoint has no model_state_dict")
    actor = SACActor()
    actor.load_state_dict(state_dict)
    frozen_before = all(not p.requires_grad for p in actor.log_std_head.parameters())
    actor.unfreeze_log_std()
    unfrozen_after = all(p.requires_grad for p in actor.log_std_head.parameters())
    actor.freeze_log_std()
    return {
        "checkpoint": checkpoint,
        "actor": actor,
        "log_std_frozen_metadata": checkpoint.get("log_std_frozen"),
        "log_std_frozen_on_load": frozen_before,
        "unfreeze_log_std_verified": unfrozen_after,
    }


@dataclass
class EpisodeAudit:
    episode_id: str
    loaded: bool
    reason: str
    policy_transitions: int = 0
    reached_pregrasp: bool = False
    whole_task_success: Optional[bool] = None
    terminal_state: str = ""
    termination_type: str = ""
    final_distance: Optional[float] = None
    min_distance: Optional[float] = None
    return_value: Optional[float] = None
    initial_joint: Optional[List[float]] = None

    def as_dict(self):
        return dict(self.__dict__)


class OfflineRLTransitionDataset(Dataset):
    """Materialized, validated transitions for one BC split."""

    def __init__(self, dataset_dir, episode_ids, normalization_path,
                 reward_config, alignment_tolerance=1e-5,
                 primitive_tolerance=1e-4, strict_alignment=False):
        self.dataset_dir = Path(dataset_dir).expanduser().resolve()
        self.episode_ids = [str(item) for item in episode_ids]
        self.alignment_tolerance = float(alignment_tolerance)
        self.primitive_tolerance = float(primitive_tolerance)
        self.strict_alignment = bool(strict_alignment)
        self.normalization, self.normalization_metadata = _load_normalization(normalization_path)
        self.reward_config = (reward_config if isinstance(reward_config, RewardConfig)
                              else RewardConfig.load(reward_config))
        self.composer = OfflineRLRewardComposer(self.reward_config)
        self.records: List[Dict] = []
        self.episode_audits: List[EpisodeAudit] = []
        self.problems: List[Dict] = []
        self._load()
        self._finalize_arrays()

    def _problem(self, eid, kind, **details):
        item = {"episode_id": eid, "kind": kind}
        item.update(details)
        self.problems.append(item)

    def _goal(self, handle):
        path = "episode_meta/pregrasp_pose_base"
        if path in handle:
            value = np.asarray(handle[path][:], dtype=np.float64)
            if value.shape == (7,) and np.all(np.isfinite(value)):
                return value
        path = "teacher/pregrasp_pose_base"
        if path in handle:
            value = np.asarray(handle[path][:], dtype=np.float64)
            if value.ndim == 2 and value.shape[1:] == (7,):
                finite = np.all(np.isfinite(value), axis=1)
                if np.any(finite):
                    first = value[np.flatnonzero(finite)[0]]
                    if np.allclose(value[finite], first, atol=1e-6, rtol=0):
                        return first
        return None

    def _load(self):
        for eid in self.episode_ids:
            path = self.dataset_dir / (eid + ".hdf5")
            if not path.exists():
                raise ValueError("manifest episode does not exist: {}".format(eid))
            audit = EpisodeAudit(eid, False, "unprocessed")
            try:
                with h5py.File(str(path), "r") as handle:
                    schema = _text(handle.attrs.get("dataset_schema_version", "legacy"))
                    if schema != SUPPORTED_SCHEMA:
                        audit.reason = "legacy episode skipped (schema={})".format(schema)
                        self.episode_audits.append(audit); continue
                    if not _bool(handle.attrs.get("episode_valid"), False):
                        audit.reason = "episode_valid=false"; self.episode_audits.append(audit); continue
                    if _bool(handle.attrs.get("discarded"), False):
                        audit.reason = "discarded=true"; self.episode_audits.append(audit); continue
                    goal = self._goal(handle)
                    if goal is None:
                        audit.reason = "finite frozen pregrasp goal unavailable"; self.episode_audits.append(audit); continue
                    q = np.asarray(handle["obs/joint_pos"][:], dtype=np.float64)
                    tcp = np.asarray(handle["obs/tcp_pose"][:], dtype=np.float64)
                    nq = np.asarray(handle["next_obs/joint_pos"][:], dtype=np.float64)
                    ntcp = np.asarray(handle["next_obs/tcp_pose"][:], dtype=np.float64)
                    action = np.asarray(handle["actions/executed_delta_joint"][:], dtype=np.float64)
                    policy = np.asarray(handle["valid/policy_action"][:], dtype=bool)
                    action_valid = np.asarray(handle["valid/executed_delta_joint"][:], dtype=bool)
                    count = len(q)
                    if q.ndim != 2 or q.shape[1] != 6 or tcp.ndim != 2 or tcp.shape[1] < 3:
                        raise ValueError("invalid obs dimensions")
                    if any(len(value) != count for value in (nq, ntcp, action, policy, action_valid)):
                        raise ValueError("episode arrays have inconsistent lengths")
                    reached = _bool(handle.attrs.get("reached_pregrasp"), False)
                    if "task/reached_pregrasp" in handle:
                        task_reached = np.asarray(handle["task/reached_pregrasp"][:], dtype=bool)
                        if task_reached.size and bool(np.any(task_reached)) != reached:
                            self._problem(eid, "reached_pregrasp_mismatch", root=reached, task_any=bool(np.any(task_reached)))
                    audit.reached_pregrasp = reached
                    if "episode_meta/initial_joint_pos" in handle:
                        initial = np.asarray(handle["episode_meta/initial_joint_pos"][:], dtype=np.float64)
                        if initial.shape == (6,) and np.all(np.isfinite(initial)):
                            audit.initial_joint = initial.tolist()
                    whole = handle.attrs.get("task_success", handle.attrs.get("success"))
                    audit.whole_task_success = None if whole is None else _bool(whole)
                    audit.terminal_state = _text(handle.attrs.get("terminal_state", ""))
                    audit.termination_type = _text(handle.attrs.get("termination_type", ""))
                    candidate = np.flatnonzero(policy & action_valid)
                    candidate = candidate[candidate + 1 < count]
                    if candidate.size == 0:
                        audit.reason = "zero policy transitions"
                        self._problem(eid, "zero_policy_transitions")
                        self.episode_audits.append(audit); continue
                    distances_all = np.linalg.norm(goal[:3][None, :] - tcp[:, :3], axis=1)
                    audit.final_distance = float(distances_all[-1]) if len(distances_all) else None
                    audit.min_distance = float(np.min(distances_all)) if len(distances_all) else None
                    stored_distance = np.asarray(handle["reward_terms/distance_to_pregrasp_m"][:], dtype=np.float64) if "reward_terms/distance_to_pregrasp_m" in handle else None
                    stored_progress = np.asarray(handle["reward_terms/progress_m"][:], dtype=np.float64) if "reward_terms/progress_m" in handle else None
                    stored_action_norm = np.asarray(handle["reward_terms/action_norm"][:], dtype=np.float64) if "reward_terms/action_norm" in handle else None
                    # A policy transition only survives all finite/alignment checks.
                    valid_records = []
                    for index, t in enumerate(candidate):
                        expected_delta = nq[t] - q[t]
                        delta_error = float(np.max(np.abs(action[t] - expected_delta)))
                        next_joint_error = float(np.max(np.abs(nq[t] - q[t + 1])))
                        next_tcp_error = float(np.max(np.abs(ntcp[t] - tcp[t + 1])))
                        if max(delta_error, next_joint_error, next_tcp_error) > self.alignment_tolerance:
                            self._problem(eid, "alignment_error", transition_index=int(t), delta_error=delta_error, next_joint_error=next_joint_error, next_tcp_error=next_tcp_error)
                            continue
                        state = np.concatenate((q[t], tcp[t, :3], goal[:3] - tcp[t, :3]))
                        next_state = np.concatenate((nq[t], ntcp[t, :3], goal[:3] - ntcp[t, :3]))
                        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(next_state)) or not np.all(np.isfinite(action[t])):
                            self._problem(eid, "nan_or_inf", transition_index=int(t)); continue
                        distance = float(distances_all[t])
                        next_distance = float(distances_all[t + 1])
                        progress = distance - next_distance
                        if stored_distance is not None and t < len(stored_distance) and np.isfinite(stored_distance[t]) and abs(stored_distance[t] - distance) > self.primitive_tolerance:
                            self._problem(eid, "distance_inconsistent", transition_index=int(t), stored=float(stored_distance[t]), recomputed=distance, error=abs(float(stored_distance[t] - distance)))
                        if stored_progress is not None and t < len(stored_progress) and np.isfinite(stored_progress[t]) and abs(stored_progress[t] - progress) > self.primitive_tolerance:
                            self._problem(eid, "progress_inconsistent", transition_index=int(t), stored=float(stored_progress[t]), recomputed=progress, error=abs(float(stored_progress[t] - progress)))
                        if stored_action_norm is not None and t < len(stored_action_norm) and np.isfinite(stored_action_norm[t]):
                            raw_norm = float(np.linalg.norm(action[t]))
                            if abs(float(stored_action_norm[t]) - raw_norm) > self.primitive_tolerance:
                                self._problem(eid, "action_norm_inconsistent", transition_index=int(t), stored=float(stored_action_norm[t]), recomputed=raw_norm)
                        valid_records.append((int(t), state, action[t], next_state, distance, progress))
                    if not valid_records:
                        audit.reason = "all policy transitions failed validation"; self.episode_audits.append(audit); continue
                    done_index = len(valid_records) - 1
                    raw_actions = np.asarray([item[2] for item in valid_records], dtype=np.float32)
                    action_norm = raw_actions / self.normalization["action_scale"]
                    done = np.zeros(len(valid_records), dtype=bool); done[done_index] = True
                    success = done & bool(reached)
                    composed = self.composer.compose(np.asarray([item[5] for item in valid_records]), action_norm, done, success)
                    for row, item in enumerate(valid_records):
                        t, state, action_row, next_state, distance, progress = item
                        self.records.append({
                            "state_raw": state.astype(np.float32), "state_norm": ((state - self.normalization["state_mean"]) / (self.normalization["state_std"] + 1e-8)).astype(np.float32),
                            "action_rad": action_row.astype(np.float32), "action_norm": action_norm[row].astype(np.float32),
                            "next_state_raw": next_state.astype(np.float32), "next_state_norm": ((next_state - self.normalization["state_mean"]) / (self.normalization["state_std"] + 1e-8)).astype(np.float32),
                            "reward": float(composed["reward"][row]), "progress": float(progress), "distance_to_goal": float(distance),
                            "action_cost": float(composed["action_cost"][row]), "done": bool(done[row]), "terminal_success": bool(success[row]),
                            "episode_id": eid, "transition_index": t,
                        })
                    audit.loaded = True; audit.reason = "loaded"; audit.policy_transitions = len(valid_records)
                    audit.return_value = float(np.sum(composed["reward"]))
                    if reached and not audit.whole_task_success:
                        self._problem(eid, "policy_subtask_success_whole_task_failure", terminal_state=audit.terminal_state)
                    if not reached:
                        self._problem(eid, "reached_pregrasp_false", final_distance=audit.final_distance, min_distance=audit.min_distance, policy_transitions=len(valid_records), terminal_state=audit.terminal_state)
            except (OSError, KeyError, ValueError, TypeError) as exc:
                audit.reason = str(exc)
            self.episode_audits.append(audit)

    def _finalize_arrays(self):
        keys = ("state_raw", "state_norm", "action_rad", "action_norm", "next_state_raw", "next_state_norm")
        for key in keys:
            setattr(self, key, np.asarray([r[key] for r in self.records], dtype=np.float32).reshape((-1, STATE_DIM if "state" in key else ACTION_DIM)))
        for key in ("reward", "progress", "distance_to_goal", "action_cost"):
            setattr(self, key, np.asarray([r[key] for r in self.records], dtype=np.float32).reshape((-1, 1)))
        for key in ("done", "terminal_success"):
            setattr(self, key, np.asarray([r[key] for r in self.records], dtype=np.float32).reshape((-1, 1)))
        self.episode_id_array = np.asarray([r["episode_id"] for r in self.records], dtype=object)
        self.transition_index = np.asarray([r["transition_index"] for r in self.records], dtype=np.int64)
        if self.records:
            if not all(np.all(np.isfinite(getattr(self, key))) for key in keys + ("reward",)):
                raise ValueError("Offline RL dataset contains NaN/Inf after validation")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return {
            "state_raw": torch.from_numpy(self.state_raw[index]), "state_norm": torch.from_numpy(self.state_norm[index]),
            "action_rad": torch.from_numpy(self.action_rad[index]), "action_norm": torch.from_numpy(self.action_norm[index]),
            "reward": torch.from_numpy(self.reward[index]), "next_state_raw": torch.from_numpy(self.next_state_raw[index]),
            "next_state_norm": torch.from_numpy(self.next_state_norm[index]), "done": torch.from_numpy(self.done[index]),
            "terminal_success": torch.from_numpy(self.terminal_success[index]), "episode_id": self.episode_id_array[index],
            "transition_index": int(self.transition_index[index]),
        }

    def summaries(self):
        return [item.as_dict() for item in self.episode_audits]

    def data_loader(self, batch_size=64, shuffle=False):
        return torch.utils.data.DataLoader(self, batch_size=int(batch_size), shuffle=bool(shuffle))
