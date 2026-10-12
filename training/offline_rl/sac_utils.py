"""Small, testable SAC Bellman utilities for Stage 3B.

There is intentionally no optimizer, replay buffer, actor loss, or training
loop in this module.
"""

import torch
from torch import nn

from .networks import SACQNetwork


def _batch_shape(name, value, batch, width=1):
    if not torch.is_tensor(value) or value.ndim != 2 or value.shape != (batch, width):
        raise ValueError("{} must have shape [{},{}], got {}".format(name, batch, width, tuple(value.shape) if torch.is_tensor(value) else type(value)))
    if not torch.isfinite(value).all():
        raise FloatingPointError("{} contains NaN/Inf".format(name))


@torch.no_grad()
def sac_bellman_components(actor, target_q1, target_q2, next_state, reward, done,
                           gamma=0.99, alpha=0.2):
    """Return target and detached stochastic-policy/Q components."""
    if not 0.0 <= float(gamma) <= 1.0:
        raise ValueError("gamma must be in [0,1]")
    if float(alpha) < 0.0:
        raise ValueError("alpha must be non-negative")
    if next_state.ndim != 2 or next_state.shape[1] != actor.state_dim:
        raise ValueError("next_state must have shape [B,{}]".format(actor.state_dim))
    batch = next_state.shape[0]
    _batch_shape("reward", reward, batch, 1)
    _batch_shape("done", done, batch, 1)
    if not torch.isfinite(next_state).all():
        raise FloatingPointError("next_state contains NaN/Inf")
    next_action, next_log_prob, _ = actor.sample(next_state)
    _batch_shape("next_action", next_action, batch, actor.action_dim)
    _batch_shape("next_log_prob", next_log_prob, batch, 1)
    q1_value = target_q1(next_state, next_action)
    q2_value = target_q2(next_state, next_action)
    min_target_q = torch.minimum(q1_value, q2_value)
    target = reward + float(gamma) * (1.0 - done) * (min_target_q - float(alpha) * next_log_prob)
    _batch_shape("target", target, batch, 1)
    if not torch.isfinite(target).all():
        raise FloatingPointError("Bellman target contains NaN/Inf")
    return {
        "target": target,
        "next_action": next_action,
        "next_log_prob": next_log_prob,
        "target_q1": q1_value,
        "target_q2": q2_value,
        "min_target_q": min_target_q,
    }


@torch.no_grad()
def sac_bellman_target(actor, target_q1, target_q2, next_state, reward, done,
                       gamma=0.99, alpha=0.2):
    """Compute r + gamma*(1-done)*(min(Q1',Q2') - alpha*log pi)."""
    return sac_bellman_components(
        actor, target_q1, target_q2, next_state, reward, done, gamma, alpha
    )["target"]


def critic_predictions(q1, q2, state, action, target):
    """Return shape-checked Q predictions and separate MSE losses."""
    q1_value = q1(state, action)
    q2_value = q2(state, action)
    if target.ndim != 2 or target.shape != q1_value.shape:
        raise ValueError("target must have shape [B,1]")
    if not torch.isfinite(target).all():
        raise FloatingPointError("target contains NaN/Inf")
    q1_loss = nn.functional.mse_loss(q1_value, target)
    q2_loss = nn.functional.mse_loss(q2_value, target)
    return {"q1": q1_value, "q2": q2_value,
            "q1_loss": q1_loss, "q2_loss": q2_loss,
            "critic_loss": q1_loss + q2_loss}
