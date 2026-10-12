import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import HDF5BCDataset, NormalizationStats, load_bc_transitions


def _write_episode(path, valid=True, policy=None, nan_goal=False):
    t = 4
    joint = np.asarray([
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        [0.01, 0.09, 0.20, 0.31, 0.39, 0.50],
        [0.02, 0.08, 0.20, 0.32, 0.38, 0.50],
        [0.03, 0.07, 0.20, 0.33, 0.37, 0.50],
    ], dtype=np.float64)
    tcp = np.asarray([
        [0.1, 0.2, 0.3, 0, 0, 0, 1],
        [0.11, 0.2, 0.3, 0, 0, 0, 1],
        [0.12, 0.2, 0.3, 0, 0, 0, 1],
        [0.13, 0.2, 0.3, 0, 0, 0, 1],
    ], dtype=np.float64)
    delta = np.zeros((t, 6), dtype=np.float64)
    delta[:-1] = joint[1:] - joint[:-1]
    if policy is None:
        policy = np.asarray([True, False, False, False])
    with h5py.File(str(path), "w") as handle:
        handle.attrs["dataset_schema_version"] = "apple_rl_v1.2_sac_goal"
        handle.attrs["episode_valid"] = bool(valid)
        handle.attrs["discarded"] = False
        episode_meta = handle.create_group("episode_meta")
        obs = handle.create_group("obs")
        next_obs = handle.create_group("next_obs")
        actions = handle.create_group("actions")
        valid_group = handle.create_group("valid")
        episode_meta.create_dataset("pregrasp_pose_base", data=(
            np.full(7, np.nan) if nan_goal else
            np.asarray([0.5, 0.6, 0.7, 0, 0, 0, 1], dtype=np.float64)
        ))
        episode_meta.create_dataset("initial_joint_pos", data=joint[0])
        obs.create_dataset("joint_pos", data=joint)
        obs.create_dataset("tcp_pose", data=tcp)
        next_obs.create_dataset("joint_pos", data=np.vstack((joint[1:], joint[-1:])))
        next_obs.create_dataset("tcp_pose", data=np.vstack((tcp[1:], tcp[-1:])))
        actions.create_dataset("executed_delta_joint", data=delta)
        valid_group.create_dataset("policy_action", data=np.asarray(policy))
        valid_group.create_dataset(
            "executed_delta_joint",
            data=np.asarray([True, True, True, False]),
        )


class DatasetTests(unittest.TestCase):
    def test_state_action_filter_alignment_and_relative_goal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_episode(root / "episode_000001.hdf5")
            transitions = load_bc_transitions(root)
            # Only t=0 is both policy and joint-action valid.
            self.assertEqual(transitions.states.shape, (1, 12))
            self.assertEqual(transitions.actions.shape, (1, 6))
            self.assertTrue(np.allclose(
                transitions.states[0, 9:], [0.4, 0.4, 0.4]
            ))
            self.assertTrue(np.allclose(
                transitions.actions[0], [0.01, -0.01, 0.0, 0.01, -0.01, 0.0]
            ))
            self.assertLessEqual(transitions.max_delta_joint_alignment_error, 1e-12)
            self.assertLessEqual(transitions.max_next_joint_alignment_error, 1e-12)
            self.assertLessEqual(transitions.max_next_tcp_alignment_error, 1e-12)

    def test_invalid_episode_and_nan_goal_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_episode(root / "episode_000001.hdf5", valid=False)
            _write_episode(root / "episode_000002.hdf5", nan_goal=True)
            _write_episode(root / "episode_000003.hdf5", valid=True)
            with h5py.File(str(root / "episode_000004.hdf5"), "w") as handle:
                handle.attrs["dataset_schema_version"] = "apple_rl_v1.1"
                handle.attrs["episode_valid"] = True
            transitions = load_bc_transitions(root)
            self.assertEqual(transitions.episodes_loaded, 1)
            reasons = [item.reason for item in transitions.statuses if not item.loaded]
            self.assertTrue(any("episode_valid=false" in item for item in reasons))
            self.assertTrue(any("pregrasp" in item for item in reasons))
            self.assertTrue(any("legacy episode skipped" in item for item in reasons))

    def test_normalization_round_trip(self):
        states = np.arange(36, dtype=np.float32).reshape(3, 12)
        actions = np.asarray([
            [0.1, -0.2, 0.3, -0.4, 0.5, -0.6],
            [0.2, -0.1, 0.1, -0.2, 0.3, -0.4],
            [-0.1, 0.1, -0.2, 0.2, -0.3, 0.2],
        ], dtype=np.float32)
        stats = NormalizationStats.fit(states, actions)
        self.assertTrue(np.allclose(
            stats.denormalize_actions(stats.normalize_actions(actions)), actions
        ))
        dataset = HDF5BCDataset(
            type("Transitions", (), {"states": states, "actions": actions})(),
            stats,
        )
        self.assertEqual(tuple(dataset.states.shape), (3, 12))
        self.assertEqual(tuple(dataset.actions.shape), (3, 6))


if __name__ == "__main__":
    unittest.main()
