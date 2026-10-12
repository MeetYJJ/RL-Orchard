"""Stage 3C update invariants; all tests are synthetic and hardware-free."""

import torch
import numpy as np

from training.models import SACActor
from training.offline_rl.networks import (SACQNetwork, freeze_target_network,
                                           hard_update, soft_update)
from training.offline_rl.sac_training import actor_update_step, critic_update_step
from training.offline_rl.dataset import audit_actor_checkpoint
from training.train_offline_sac_bc import _evaluate


def _batch(batch_size=8):
    return {
        "state_norm": torch.randn(batch_size, 12),
        "action_norm": torch.tanh(torch.randn(batch_size, 6)),
        "reward": torch.randn(batch_size, 1),
        "next_state_norm": torch.randn(batch_size, 12),
        "done": torch.randint(0, 2, (batch_size, 1)).float(),
    }


def _nets():
    actor = SACActor(); actor.unfreeze_log_std()
    q1, q2 = SACQNetwork(), SACQNetwork()
    target_q1, target_q2 = SACQNetwork(), SACQNetwork()
    hard_update(target_q1, q1); hard_update(target_q2, q2)
    freeze_target_network(target_q1); freeze_target_network(target_q2)
    return actor, q1, q2, target_q1, target_q2


def test_bc_checkpoint_loads_and_log_std_unfreezes(tmp_path):
    actor = SACActor()
    path = tmp_path / "actor.pt"
    torch.save({"actor_class": "SACActor", "architecture_version": actor.architecture_version,
                "state_dim": 12, "action_dim": 6, "hidden_dim": 256,
                "model_state_dict": actor.state_dict()}, path)
    loaded = audit_actor_checkpoint(path)["actor"]
    assert all(not p.requires_grad for p in loaded.log_std_head.parameters())
    loaded.unfreeze_log_std()
    assert all(p.requires_grad for p in loaded.log_std_head.parameters())
    assert all(torch.equal(a, b) for a, b in zip(actor.parameters(), loaded.parameters()))


def test_critic_step_updates_only_online_q():
    actor, q1, q2, t1, t2 = _nets()
    before_actor = [p.detach().clone() for p in actor.parameters()]
    before_targets = [[p.detach().clone() for p in net.parameters()] for net in (t1, t2)]
    before_q = [[p.detach().clone() for p in net.parameters()] for net in (q1, q2)]
    optimizer = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=1e-3)
    result = critic_update_step(actor, q1, q2, t1, t2, _batch(), optimizer)
    assert torch.isfinite(result["critic_loss"])
    assert any(not torch.equal(a, b) for old, net in zip(before_q, (q1, q2)) for a, b in zip(old, net.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(before_actor, actor.parameters()))
    assert all(torch.equal(a, b) for old, net in zip(before_targets, (t1, t2)) for a, b in zip(old, net.parameters()))


def test_actor_step_updates_actor_without_changing_critics_and_backpropagates_through_q():
    actor, q1, q2, _, _ = _nets()
    before_q = [[p.detach().clone() for p in net.parameters()] for net in (q1, q2)]
    before_actor = [p.detach().clone() for p in actor.parameters()]
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-3)
    result = actor_update_step(actor, q1, q2, _batch(), optimizer, alpha=.2, bc_weight=1.)
    assert result["actor_grad_norm"] > 0
    assert any(not torch.equal(a, b) for a, b in zip(before_actor, actor.parameters()))
    assert all(torch.equal(a, b) for old, net in zip(before_q, (q1, q2)) for a, b in zip(old, net.parameters()))
    assert all(p.requires_grad for p in list(q1.parameters()) + list(q2.parameters()))


def test_actor_loss_is_sac_plus_weighted_bc_and_zero_weight_is_sac():
    actor, q1, q2, _, _ = _nets()
    batch = _batch()
    initial = [p.detach().clone() for p in actor.parameters()]
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.0)
    result = actor_update_step(actor, q1, q2, batch, optimizer, alpha=.2, bc_weight=2.5)
    assert torch.allclose(result["actor_loss"], result["sac_actor_loss"] + 2.5 * result["bc_loss"])
    assert all(torch.equal(a, b) for a, b in zip(initial, actor.parameters()))
    zero = actor_update_step(actor, q1, q2, batch, optimizer, alpha=.2, bc_weight=0.)
    assert torch.allclose(zero["actor_loss"], zero["sac_actor_loss"])
    expected = torch.nn.functional.mse_loss(result["mean_action"], batch["action_norm"])
    assert torch.allclose(result["bc_loss"], expected)


def test_shape_and_finite_guards_prevent_broadcasting():
    actor, q1, q2, t1, t2 = _nets()
    batch = _batch()
    batch["reward"] = torch.zeros(8)
    optimizer = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=1e-3)
    try:
        critic_update_step(actor, q1, q2, t1, t2, batch, optimizer)
    except ValueError:
        pass
    else:
        raise AssertionError("malformed [B] reward must not silently broadcast")
    batch = _batch(); batch["reward"][0, 0] = float("nan")
    try:
        critic_update_step(actor, q1, q2, t1, t2, batch, optimizer)
    except (ValueError, FloatingPointError):
        pass
    else:
        raise AssertionError("NaN reward must fail fast")


def test_soft_update_and_actor_output_shapes_and_bounds():
    actor, q1, _, target_q1, _ = _nets()
    for source in q1.parameters(): source.data.fill_(2.)
    for target in target_q1.parameters(): target.data.zero_()
    soft_update(target_q1, q1, .005)
    assert all(torch.allclose(p, torch.full_like(p, .01)) for p in target_q1.parameters())
    action, log_prob, mean = actor.sample(torch.randn(5, 12))
    assert action.shape == mean.shape == (5, 6) and log_prob.shape == (5, 1)
    assert torch.isfinite(action).all() and torch.isfinite(log_prob).all()
    assert torch.all(action >= -1) and torch.all(action <= 1)


def test_checkpoint_round_trip_and_physical_denormalization(tmp_path):
    actor = SACActor()
    path = tmp_path / "actor.pt"
    torch.save({"model_state_dict": actor.state_dict()}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    restored = SACActor(); restored.load_state_dict(loaded["model_state_dict"])
    state = torch.randn(3, 12)
    assert torch.equal(actor.deterministic_action(state), restored.deterministic_action(state))
    scale = torch.tensor([.1, .2, .3, .4, .5, .6])
    normalized = torch.ones(2, 6) * .5
    assert torch.allclose(normalized * scale, scale.expand(2, -1) * .5)


def test_validation_evaluation_does_not_update_parameters():
    class ValidationRows:
        def __init__(self):
            self.rows = []
            for row in _batch(4)["state_norm"]:
                self.rows.append({
                    "state_norm": row,
                    "action_norm": torch.zeros(6),
                    "reward": torch.zeros(1),
                    "next_state_norm": torch.randn(12),
                    "done": torch.ones(1),
                })

        def __len__(self): return len(self.rows)
        def __getitem__(self, index): return self.rows[index]

    actor, q1, q2, target_q1, target_q2 = _nets()
    networks = (actor, q1, q2, target_q1, target_q2)
    before = [[p.detach().clone() for p in network.parameters()] for network in networks]
    config = {"batch_size": 2, "gamma": .99, "alpha": .2}
    result = _evaluate(actor, q1, q2, target_q1, target_q2, ValidationRows(),
                       list(range(4)), config, torch.device("cpu"), np.ones(6))
    assert result["actor_bc_mse"] >= 0
    assert result["critic_bellman_mse"] >= 0
    assert all(torch.equal(old, new) for group, network in zip(before, networks)
               for old, new in zip(group, network.parameters()))
