"""Configurable candidate reward built only from recorded primitives."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml


@dataclass(frozen=True)
class RewardConfig:
    reward_version: str
    status: str
    progress_weight: float
    action_weight: float
    success_bonus: float
    failure_penalty: float

    @classmethod
    def load(cls, path):
        with open(str(Path(path).expanduser().resolve()), "r") as stream:
            value = yaml.safe_load(stream)
        if not isinstance(value, dict):
            raise ValueError("reward config must be a mapping")
        config = cls(
            reward_version=str(value["reward_version"]),
            status=str(value.get("status", "candidate/provisional")),
            progress_weight=float(value["progress_weight"]),
            action_weight=float(value["action_weight"]),
            success_bonus=float(value["success_bonus"]),
            failure_penalty=float(value["failure_penalty"]),
        )
        numbers = (
            config.progress_weight, config.action_weight,
            config.success_bonus, config.failure_penalty,
        )
        if not np.all(np.isfinite(numbers)):
            raise ValueError("reward weights must be finite")
        if config.action_weight < 0.0:
            raise ValueError("action_weight must be non-negative")
        if config.success_bonus < 0.0 or config.failure_penalty < 0.0:
            raise ValueError("terminal bonus/penalty magnitudes must be non-negative")
        return config


class OfflineRLRewardComposer:
    """Compose reward_v0 without changing the source HDF5 files."""

    def __init__(self, config):
        self.config = config

    def compose(self, progress_m, action_normalized, done, terminal_success):
        progress = np.asarray(progress_m, dtype=np.float64)
        action = np.asarray(action_normalized, dtype=np.float64)
        done = np.asarray(done, dtype=bool)
        success = np.asarray(terminal_success, dtype=bool)
        if action.ndim != 2 or action.shape[1] != 6:
            raise ValueError("action_normalized must have shape [N,6]")
        if progress.shape != (len(action),):
            raise ValueError("progress_m must have shape [N]")
        if done.shape != progress.shape or success.shape != progress.shape:
            raise ValueError("done and terminal_success must have shape [N]")
        if not np.all(np.isfinite(progress)) or not np.all(np.isfinite(action)):
            raise ValueError("reward inputs contain NaN/Inf")
        action_cost = np.sum(np.square(action), axis=1)
        progress_component = self.config.progress_weight * progress
        action_component = -self.config.action_weight * action_cost
        terminal_component = np.zeros_like(progress, dtype=np.float64)
        terminal_component[done & success] = self.config.success_bonus
        terminal_component[done & ~success] = -self.config.failure_penalty
        reward = progress_component + action_component + terminal_component
        if not np.all(np.isfinite(reward)):
            raise ValueError("composed reward contains NaN/Inf")
        return {
            "reward": reward.astype(np.float32),
            "progress_component": progress_component.astype(np.float32),
            "action_cost": action_cost.astype(np.float32),
            "action_component": action_component.astype(np.float32),
            "terminal_component": terminal_component.astype(np.float32),
        }
