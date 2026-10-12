import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import NormalizationStats
from train_actor_bc_full import (
    _baseline_metrics,
    create_episode_split,
    manifest_hash,
    resolve_device,
)


class FullBCTests(unittest.TestCase):
    def test_episode_split_is_disjoint_and_seed_reproducible(self):
        episodes = ["episode_{:06d}".format(index) for index in range(12)]
        first = create_episode_split(episodes, seed=42)
        second = create_episode_split(episodes, seed=42)
        self.assertEqual(first, second)
        train = set(first["train_episode_ids"])
        validation = set(first["validation_episode_ids"])
        test = set(first["test_episode_ids"])
        self.assertFalse(train & validation)
        self.assertFalse(train & test)
        self.assertFalse(validation & test)
        self.assertEqual(train | validation | test, set(episodes))
        self.assertEqual(first["manifest_hash"], manifest_hash(first))

    def test_normalization_is_train_only(self):
        train_states = np.zeros((2, 12), dtype=np.float32)
        train_actions = np.ones((2, 6), dtype=np.float32)
        held_out_states = np.full((2, 12), 100.0, dtype=np.float32)
        held_out_actions = np.full((2, 6), 100.0, dtype=np.float32)
        stats = NormalizationStats.fit(train_states, train_actions)
        self.assertTrue(np.allclose(stats.state_mean, 0.0))
        self.assertTrue(np.allclose(stats.action_scale, 1.0))
        self.assertFalse(np.allclose(
            NormalizationStats.fit(
                np.vstack((train_states, held_out_states)),
                np.vstack((train_actions, held_out_actions)),
            ).state_mean,
            stats.state_mean,
        ))

    def test_cpu_device_and_baselines(self):
        device = resolve_device("cpu")
        self.assertEqual(device.type, "cpu")
        auto_device = resolve_device("auto")
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(auto_device.type, expected)
        target = np.asarray([[1.0, -2.0, 0.5, 0.0, 1.0, -1.0]])
        scale = np.ones(6)
        metrics = _baseline_metrics(target, scale, target)
        self.assertAlmostEqual(metrics["zero_action"]["physical_mae_rad"], 0.9166666666666666)
        self.assertTrue(np.isfinite(metrics["train_mean_action"]["normalized_mse"]))

    def test_cuda_forward_when_available(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available on this host")
        from models import SACActor
        actor = SACActor().cuda()
        mean, log_std = actor(torch.zeros(2, 12, device="cuda"))
        self.assertEqual(tuple(mean.shape), (2, 6))
        self.assertEqual(tuple(log_std.shape), (2, 6))


if __name__ == "__main__":
    unittest.main()
