"""Single-step Offline SAC+BC update helpers.

The functions here are intentionally small and testable.  They do not own a
replay buffer and never interact with ROS or an environment.
"""

import torch
from torch import nn

from .networks import soft_update
from .sac_utils import critic_predictions, sac_bellman_components


def _finite(name, value):
    if not torch.isfinite(value).all():
        raise FloatingPointError("{} contains NaN/Inf".format(name))


def _grad_norm(module):
    grads = [p.grad.detach().norm() for p in module.parameters() if p.grad is not None]
    value = torch.stack(grads).norm() if grads else torch.zeros((), device=next(module.parameters()).device)
    _finite("gradient norm", value)
    return value


def set_requires_grad(module, enabled):
    for parameter in module.parameters():
        parameter.requires_grad_(bool(enabled))


def critic_update_step(actor, q1, q2, target_q1, target_q2, batch,
                       optimizer, gamma=0.99, alpha=0.2):
    """Update Q1/Q2 only; target and actor remain gradient-free."""
    state = batch["state_norm"]
    action = batch["action_norm"]
    reward = batch["reward"]
    next_state = batch["next_state_norm"]
    done = batch["done"]
    components = sac_bellman_components(
        actor, target_q1, target_q2, next_state, reward, done, gamma, alpha
    )
    target = components["target"]
    result = critic_predictions(q1, q2, state, action, target)
    for name in ("q1", "q2", "q1_loss", "q2_loss", "critic_loss"):
        _finite(name, result[name])
    optimizer.zero_grad(set_to_none=True)
    result["critic_loss"].backward()
    q1_grad = _grad_norm(q1)
    q2_grad = _grad_norm(q2)
    optimizer.step()
    for name, network in (("q1 parameters", q1), ("q2 parameters", q2)):
        for parameter in network.parameters():
            _finite(name, parameter)
    return {
        **result,
        "target": target,
        "target_components": components,
        "q1_grad_norm": q1_grad.detach(),
        "q2_grad_norm": q2_grad.detach(),
    }


def actor_update_step(actor, q1, q2, batch, optimizer, alpha=0.2,
                      bc_weight=1.0, gradient_clip_norm=0.0):
    """Update Actor with SAC objective plus supervised expert BC penalty.

    Q parameters are temporarily frozen, but their forward graph remains
    active so dQ/da reaches the Actor.
    """
    expert_action = batch["action_norm"]
    state = batch["state_norm"]
    q1_state = [p.requires_grad for p in q1.parameters()]
    q2_state = [p.requires_grad for p in q2.parameters()]
    set_requires_grad(q1, False)
    set_requires_grad(q2, False)
    try:
        policy_action, log_prob, mean_action = actor.sample(state)
        q1_policy = q1(state, policy_action)
        q2_policy = q2(state, policy_action)
        policy_q = torch.minimum(q1_policy, q2_policy)
        sac_loss = torch.mean(float(alpha) * log_prob - policy_q)
        bc_loss = nn.functional.mse_loss(mean_action, expert_action)
        total_loss = sac_loss + float(bc_weight) * bc_loss
        _finite("sac_actor_loss", sac_loss)
        _finite("bc_loss", bc_loss)
        _finite("actor_loss", total_loss)
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        if float(gradient_clip_norm) > 0.0:
            torch.nn.utils.clip_grad_norm_(actor.parameters(), float(gradient_clip_norm))
        actor_grad = _grad_norm(actor)
        optimizer.step()
        for parameter in actor.parameters():
            _finite("actor parameters", parameter)
        return {
            "actor_loss": total_loss.detach(), "sac_actor_loss": sac_loss.detach(),
            "bc_loss": bc_loss.detach(), "policy_action": policy_action.detach(),
            "mean_action": mean_action.detach(), "log_prob": log_prob.detach(),
            "policy_q": policy_q.detach(), "actor_grad_norm": actor_grad.detach(),
        }
    finally:
        for parameter, enabled in zip(q1.parameters(), q1_state):
            parameter.requires_grad_(enabled)
        for parameter, enabled in zip(q2.parameters(), q2_state):
            parameter.requires_grad_(enabled)


def update_targets(q1, q2, target_q1, target_q2, tau):
    soft_update(target_q1, q1, tau)
    soft_update(target_q2, q2, tau)
