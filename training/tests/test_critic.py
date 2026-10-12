"""Synthetic Stage 3B tests; no ROS, HDF5, or robot is used."""

import pytest

torch = pytest.importorskip("torch")

from training.models import SACActor
from training.offline_rl.networks import SACQNetwork, freeze_target_network, hard_update, soft_update
from training.offline_rl.sac_utils import critic_predictions, sac_bellman_components, sac_bellman_target


class FixedActor:
    state_dim = 12
    action_dim = 6

    def sample(self, state):
        action = torch.zeros((state.shape[0], 6), dtype=state.dtype, device=state.device)
        log_prob = torch.full((state.shape[0], 1), 0.5, dtype=state.dtype, device=state.device)
        return action, log_prob, action


def _constant_q(value):
    q = SACQNetwork()
    for parameter in q.parameters():
        parameter.data.zero_()
    q.net[-1].bias.data.fill_(value)
    return q


def test_q_shapes_and_independent_parameters():
    q1, q2 = SACQNetwork(), SACQNetwork()
    assert q1(torch.zeros(4, 12), torch.zeros(4, 6)).shape == (4, 1)
    assert all(a is not b for a, b in zip(q1.parameters(), q2.parameters()))
    assert any(a.data_ptr() != b.data_ptr() for a, b in zip(q1.parameters(), q2.parameters()))


def test_target_initialization_and_freeze():
    q, target = SACQNetwork(), SACQNetwork()
    hard_update(target, q)
    assert all(torch.equal(a, b) for a, b in zip(q.parameters(), target.parameters()))
    freeze_target_network(target)
    assert all(not p.requires_grad for p in target.parameters())


def test_actor_sample_shapes_range_and_log_prob():
    actor = SACActor()
    actor.unfreeze_log_std()
    action, log_prob, mean_action = actor.sample(torch.zeros(8, 12))
    assert action.shape == (8, 6) and log_prob.shape == (8, 1) and mean_action.shape == (8, 6)
    assert torch.isfinite(log_prob).all()
    assert torch.all(action <= 1) and torch.all(action >= -1)


def test_bellman_min_entropy_sign_and_done_mask():
    actor = FixedActor(); q1 = _constant_q(2.0); q2 = _constant_q(1.0)
    state = torch.zeros(2, 12); reward = torch.tensor([[3.0], [3.0]])
    done = torch.tensor([[0.0], [1.0]])
    target = sac_bellman_target(actor, q1, q2, state, reward, done, gamma=.99, alpha=.2)
    assert torch.allclose(target[0], torch.tensor([3.0 + .99 * (1.0 - .2 * .5)]))
    assert torch.allclose(target[1], reward[1])
    assert not target.requires_grad


def test_target_q_has_no_gradient_and_critic_updates_only_online():
    q1, q2 = SACQNetwork(), SACQNetwork(); t1, t2 = SACQNetwork(), SACQNetwork()
    hard_update(t1, q1); hard_update(t2, q2); freeze_target_network(t1); freeze_target_network(t2)
    actor = SACActor(); actor.unfreeze_log_std()
    state = torch.randn(8, 12); action = torch.tanh(torch.randn(8, 6)); reward = torch.randn(8, 1); done = torch.zeros(8, 1)
    target = sac_bellman_target(actor, t1, t2, state, reward, done)
    assert not target.requires_grad
    result = critic_predictions(q1, q2, state, action, target)
    optimizer = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=1e-3)
    before_actor = [p.detach().clone() for p in actor.parameters()]
    optimizer.zero_grad(); result["critic_loss"].backward(); optimizer.step()
    assert all(p.grad is None for p in t1.parameters())
    assert all(torch.equal(a, b) for a, b in zip(before_actor, actor.parameters()))


def test_soft_update_formula():
    source, target = SACQNetwork(), SACQNetwork()
    for p in source.parameters(): p.data.fill_(2.0)
    for p in target.parameters(): p.data.fill_(0.0)
    soft_update(target, source, .25)
    assert all(torch.allclose(p, torch.full_like(p, .5)) for p in target.parameters())


def test_frozen_target_q_fit_decreases_loss():
    q1, q2 = SACQNetwork(), SACQNetwork()
    state = torch.randn(32, 12); action = torch.tanh(torch.randn(32, 6)); target = torch.ones(32, 1)
    optimizer = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=1e-3)
    first = None; last = None
    for _ in range(100):
        result = critic_predictions(q1, q2, state, action, target)
        if first is None: first = float(result["critic_loss"])
        optimizer.zero_grad(); result["critic_loss"].backward(); optimizer.step(); last = float(result["critic_loss"])
    assert last < first
