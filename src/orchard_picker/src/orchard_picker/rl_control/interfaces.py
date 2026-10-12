"""Future Gym-style contracts; deliberately no robot or ROS implementation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from orchard_picker.rl_control.action_space import ACTION_DIM


@dataclass(frozen=True)
class Observation:
    """Extensible observation envelope, without fixing a visual encoder/vector."""

    visual_state: Any
    joint_position: np.ndarray
    tcp_pose: np.ndarray
    target_state: Any
    robot_state: Optional[Any] = None


@dataclass(frozen=True)
class JointTransition:
    """Conceptual transition returned by a future environment step."""

    observation: Observation
    action: np.ndarray
    reward: float
    next_observation: Observation
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]


class JointControlExecutor(ABC):
    """Integration boundary for a future non-blocking/suitable joint executor."""

    action_dimension = ACTION_DIM

    @abstractmethod
    def get_current_joint(self) -> np.ndarray:
        """Read the current six-joint position in one documented unit."""
        raise NotImplementedError

    @abstractmethod
    def execute_joint_target(self, joint_target: np.ndarray) -> None:
        """Execute one approved target; this is intentionally not implemented here."""
        raise NotImplementedError

    @abstractmethod
    def wait_for_next_state(self) -> None:
        """Wait for the next control observation without defining timing here."""
        raise NotImplementedError


class FutureJointSACEnv(ABC):
    """Documentation-only Gym-like shape for a later real environment."""

    action_shape: Tuple[int, ...] = (ACTION_DIM,)

    @abstractmethod
    def reset(self, *, seed=None, options=None):
        """Return an observation and reset info; no implementation in this overlay."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action):
        """Return obs_next, reward, terminated, truncated, info."""
        raise NotImplementedError

