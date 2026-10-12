"""Offline RL Stage 3A tests; all fixtures are synthetic and hardware-free."""

import json
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")
torch = pytest.importorskip("torch")

from training.models import SACActor
from training.offline_rl.dataset import (OfflineRLTransitionDataset,
                                          audit_actor_checkpoint, load_split_manifest,
                                          manifest_hash)
from training.offline_rl.reward import RewardConfig
from training.offline_rl.reward import OfflineRLRewardComposer


def _episode(path, episode_id, reached=True, whole_success=False):
    with h5py.File(str(path / (episode_id + ".hdf5")), "w") as f:
        f.attrs["dataset_schema_version"] = "apple_rl_v1.2_sac_goal"
        f.attrs["episode_valid"] = True
        f.attrs["discarded"] = False
        f.attrs["reached_pregrasp"] = reached
        f.attrs["task_success"] = whole_success
        f.attrs["terminal_state"] = "DONE" if whole_success else "FAILED"
        f.create_dataset("episode_meta/pregrasp_pose_base", data=np.array([1, 0, 0, 0, 0, 0, 1], dtype=np.float64))
        q = np.array([[0, 0, 0, 0, 0, 0], [0.1, 0, 0, 0, 0, 0], [0.2, 0, 0, 0, 0, 0]], dtype=np.float64)
        tcp = np.array([[0, 0, 0, 0, 0, 0, 1], [0.2, 0, 0, 0, 0, 0, 1], [0.4, 0, 0, 0, 0, 0, 1]], dtype=np.float64)
        f.create_dataset("obs/joint_pos", data=q)
        f.create_dataset("obs/tcp_pose", data=tcp)
        f.create_dataset("next_obs/joint_pos", data=np.vstack((q[1:], q[-1:])))
        f.create_dataset("next_obs/tcp_pose", data=np.vstack((tcp[1:], tcp[-1:])))
        action = np.zeros((3, 6), dtype=np.float64); action[:2, 0] = 0.1
        f.create_dataset("actions/executed_delta_joint", data=action)
        f.create_dataset("valid/policy_action", data=np.array([True, True, False]))
        f.create_dataset("valid/executed_delta_joint", data=np.array([True, True, False]))
        goal = np.array([1, 0, 0])
        dist = np.linalg.norm(goal[None, :] - tcp[:, :3], axis=1)
        f.create_dataset("reward_terms/distance_to_pregrasp_m", data=dist)
        f.create_dataset("reward_terms/progress_m", data=np.array([dist[0] - dist[1], dist[1] - dist[2], np.nan]))
        f.create_dataset("reward_terms/action_norm", data=np.linalg.norm(action, axis=1))
        f.create_dataset("task/reached_pregrasp", data=np.array([False, reached, reached]))


def _artifacts(tmp_path):
    dataset = tmp_path / "dataset"; dataset.mkdir()
    for eid, reached, success in (("episode_000001", True, True), ("episode_000002", False, False), ("episode_000003", True, False)):
        _episode(dataset, eid, reached, success)
    manifest = {"dataset_schema_version": "apple_rl_v1.2_sac_goal", "seed": 42, "train_ratio": .7, "val_ratio": .15, "test_ratio": .15,
                "train_episode_ids": ["episode_000001"], "validation_episode_ids": ["episode_000002"], "test_episode_ids": ["episode_000003"]}
    manifest["manifest_hash"] = manifest_hash(manifest)
    manifest_path = tmp_path / "split_manifest.json"; manifest_path.write_text(json.dumps(manifest))
    norm_path = tmp_path / "normalization.npz"
    np.savez(norm_path, state_mean=np.zeros(12, np.float32), state_std=np.ones(12, np.float32), action_scale=np.ones(6, np.float32), split_manifest_hash=manifest["manifest_hash"], statistics_source="train_only")
    config_path = tmp_path / "reward.yaml"
    config_path.write_text("reward_version: offline_rl_reward_v0\nstatus: candidate/provisional\nprogress_weight: 1.0\naction_weight: 0.001\nsuccess_bonus: 1.0\nfailure_penalty: 1.0\n")
    return dataset, manifest, manifest_path, norm_path, config_path


def test_manifest_reuse_and_disjoint(tmp_path):
    dataset, manifest, manifest_path, norm_path, config_path = _artifacts(tmp_path)
    loaded = load_split_manifest(manifest_path, dataset)
    assert set(loaded["train_episode_ids"]).isdisjoint(loaded["test_episode_ids"])
    train = OfflineRLTransitionDataset(dataset, loaded["train_episode_ids"], norm_path, config_path)
    assert len(train) == 2 and np.all(train.done[:-1] == 0) and train.done[-1] == 1


def test_episode_boundary_and_terminal_semantics(tmp_path):
    dataset, manifest, _, norm_path, config_path = _artifacts(tmp_path)
    combined = OfflineRLTransitionDataset(dataset, manifest["train_episode_ids"] + manifest["validation_episode_ids"], norm_path, config_path)
    assert len(combined) == 4
    assert np.sum(combined.done) == 2
    assert not bool(combined.terminal_success[2])
    assert any(item["kind"] == "reached_pregrasp_false" for item in combined.problems)


def test_reward_and_normalization_finite(tmp_path):
    dataset, manifest, _, norm_path, config_path = _artifacts(tmp_path)
    value = OfflineRLTransitionDataset(dataset, manifest["train_episode_ids"], norm_path, config_path)
    assert value.state_norm.shape[1] == 12
    assert value.action_norm.shape[1] == 6
    assert np.all(np.isfinite(value.reward))
    assert np.allclose(value.action_norm * value.normalization["action_scale"], value.action_rad)
    assert np.allclose(value.next_state_raw[:, :6], value.state_raw[:, :6] + value.action_rad)


def test_episode_a_does_not_connect_episode_b(tmp_path):
    dataset, manifest, _, norm_path, config_path = _artifacts(tmp_path)
    value = OfflineRLTransitionDataset(dataset, manifest["train_episode_ids"] + manifest["test_episode_ids"], norm_path, config_path)
    assert value.episode_id_array[1] != value.episode_id_array[2]
    assert value.done[1] == 1 and value.done[2] == 0


def test_whole_task_failure_does_not_discard_policy_success(tmp_path):
    dataset, manifest, _, norm_path, config_path = _artifacts(tmp_path)
    value = OfflineRLTransitionDataset(dataset, manifest["test_episode_ids"], norm_path, config_path)
    assert len(value) == 2
    assert bool(value.terminal_success[-1])
    assert any(item["kind"] == "policy_subtask_success_whole_task_failure" for item in value.problems)


def test_reward_component_sum_is_finite():
    config = RewardConfig("offline_rl_reward_v0", "candidate/provisional", 1.0, 0.001, 1.0, 1.0)
    result = OfflineRLRewardComposer(config).compose(np.array([.2, -.1]), np.ones((2, 6)) * .1, np.array([False, True]), np.array([False, True]))
    expected = result["progress_component"] + result["action_component"] + result["terminal_component"]
    assert np.allclose(result["reward"], expected)


def test_actor_checkpoint_load_and_unfreeze(tmp_path):
    actor = SACActor()
    checkpoint = {"actor_class": "SACActor", "architecture_version": actor.architecture_version, "state_dim": 12, "action_dim": 6, "hidden_dim": 256,
                  "model_state_dict": actor.state_dict(), "dataset_schema_version": "apple_rl_v1.2_sac_goal", "log_std_frozen": True}
    path = tmp_path / "actor.pt"; torch.save(checkpoint, path)
    result = audit_actor_checkpoint(path)
    assert result["unfreeze_log_std_verified"]
    assert all(not p.requires_grad for p in result["actor"].log_std_head.parameters())
