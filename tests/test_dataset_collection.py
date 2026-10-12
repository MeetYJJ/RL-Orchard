#!/usr/bin/env python3

import importlib.util
import json
import os
import tempfile
import unittest

import h5py
import numpy as np

from mock_data import create_episode
from orchard_picker.rl_data.dataset_tools import (
    ManifestStore,
    apply_physical_label,
    collection_coverage,
    inspect_episode_quality,
)
from orchard_picker.rl_data.episode_writer import EpisodeWriter
from orchard_picker.rl_data.schema import (
    CONTROL_LEARNABLE_POLICY,
    CONTROL_SCRIPTED_FINAL_APPROACH,
    TASK_STATE_MAPPING,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_script(name):
    path = os.path.join(ROOT, "src", "orchard_picker", "scripts", name)
    spec = importlib.util.spec_from_file_location("test_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DatasetCollectionTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = create_episode(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_physical_success_label(self):
        row = apply_physical_label(self.path, 1, "", "verified by operator")
        self.assertEqual(int(row["physical_grasp_success"]), 1)
        with h5py.File(self.path, "r") as episode:
            self.assertEqual(int(episode["labels"].attrs["physical_grasp_success"]), 1)

    def test_unknown_can_be_revised_to_failure_and_success(self):
        apply_physical_label(self.path, 0, "apple_slip", "first review")
        apply_physical_label(self.path, 1, "", "second review")
        with h5py.File(self.path, "r") as episode:
            self.assertEqual(int(episode.attrs["physical_grasp_success"]), 1)
            self.assertEqual(str(episode.attrs["operator_note"]), "second review")

    def test_manifest_append_and_reload(self):
        manifest = os.path.join(self.temporary.name, "collection_manifest.csv")
        rows = ManifestStore(manifest).read()
        self.assertEqual(len(rows), 1)
        apply_physical_label(self.path, 0, "grasp_miss", "", manifest)
        rows = ManifestStore(manifest).read()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["failure_reason"], "grasp_miss")

    def test_policy_action_mask(self):
        with h5py.File(self.path, "r") as episode:
            modes = np.asarray(episode["task/control_mode"][:])
            mask = np.asarray(episode["valid/policy_action"][:], dtype=bool)
            valid = np.asarray(episode["valid/action"][:], dtype=bool)
            np.testing.assert_array_equal(
                mask, np.logical_and(valid, modes == CONTROL_LEARNABLE_POLICY)
            )
            self.assertFalse(mask[-1])

    def test_scripted_final_approach_phase_and_distance(self):
        with h5py.File(self.path, "r") as episode:
            modes = np.asarray(episode["task/control_mode"][:])
            self.assertTrue(np.any(modes == CONTROL_SCRIPTED_FINAL_APPROACH))
            self.assertAlmostEqual(
                float(episode.attrs["scripted_final_approach_distance_m"]), 0.08, places=5
            )
            self.assertEqual(len(episode["events/expert_timestamp_ns"]), 4)

    def test_sac_goal_is_frozen_and_reward_primitives_are_recoverable(self):
        with h5py.File(self.path, "r") as episode:
            pre = episode["episode_meta/pregrasp_pose_base"][:]
            grasp = episode["episode_meta/grasp_pose_base"][:]
            np.testing.assert_allclose(pre, [0.40, -0.10, 0.55, 0, 0, 0, 1])
            np.testing.assert_allclose(grasp, [0.48, -0.10, 0.55, 0, 0, 0, 1])
            np.testing.assert_allclose(
                episode["episode_meta/approach_vector_base"][:], [1.0, 0.0, 0.0]
            )
            self.assertAlmostEqual(float(episode["episode_meta/pregrasp_offset_m"][()]), 0.08, places=5)
            rows = episode["teacher/pregrasp_pose_base"][:]
            self.assertEqual(len(np.unique(rows, axis=0)), 1)
            self.assertTrue(bool(episode.attrs["reached_pregrasp"]))
            distance = episode["reward_terms/distance_to_pregrasp_m"][:]
            progress = episode["reward_terms/progress_m"][:]
            action_norm = episode["reward_terms/action_norm"][:]
            self.assertTrue(np.all(np.isfinite(distance)))
            self.assertTrue(np.all(np.isfinite(progress[:-1])))
            self.assertTrue(np.isnan(progress[-1]))
            self.assertTrue(np.all(np.isfinite(action_norm[:-1])))
            self.assertTrue(np.isnan(action_norm[-1]))

    def test_pregrasp_only_episode_freezes_goal_and_marks_reached(self):
        writer = EpisodeWriter(
            self.temporary.name,
            "episode_000020",
            2,
            2,
            {
                "min_episode_samples": 5,
                "min_episode_duration_sec": 0.3,
                "max_sensor_age_ms": 150.0,
                "max_robot_state_age_ms": 300.0,
            },
        )
        pregrasp = [0.40, -0.10, 0.55, 0.0, 0.0, 0.0, 1.0]
        for index in range(5):
            events = []
            if index == 0:
                events = [{
                    "phase": "MOVE_PRE_GRASP",
                    "command_type": "movej_p",
                    "target_tcp": pregrasp,
                    "target_joint": [np.nan] * 6,
                    "gripper_command": -1,
                    "speed": 20,
                    "force": 0,
                }]
            x = 0.30 + (0.103 * index / 4.0)
            writer.append({
                "timestamp_ns": 1_000_000_000 + index * 100_000_000,
                "rgb_timestamp_ns": 1_000_000_000 + index * 100_000_000,
                "depth_timestamp_ns": 1_000_000_000 + index * 100_000_000,
                "robot_state_timestamp_ns": 1_000_000_000 + index * 100_000_000,
                "joint_state_timestamp_ns": 1_000_000_000 + index * 100_000_000,
                "rgb_age_ms": 10.0,
                "depth_age_ms": 10.0,
                "robot_state_age_ms": 20.0,
                "joint_state_age_ms": 20.0,
                "rgb_depth_delta_ms": 0.0,
                "rgb": np.full((2, 2, 3), 50, dtype=np.uint8),
                "depth": np.full((2, 2), 600, dtype=np.uint16),
                "joint_pos": np.asarray([0.01 * index] * 6, dtype=np.float32),
                "tcp_pose": [x, -0.10, 0.55, 0.0, 0.0, 0.0, 1.0],
                "task_state": "MOVE_PRE_GRASP",
                "task_active": True,
                "teacher": {
                    "apple_camera_xyz": [0.0, 0.0, 0.5],
                    "apple_base_xyz": [0.48, -0.10, 0.55],
                    "apple_id": 1,
                    "confidence": 0.9,
                    "bbox": [0, 0, 1, 1],
                    "target_valid": True,
                },
                "expert_events": events,
                "rgb_valid": True,
                "depth_valid": True,
                "robot_valid": True,
                "joint_valid": True,
            })
            if index == 0:
                np.testing.assert_allclose(
                    writer.file["episode_meta/pregrasp_pose_base"][:], pregrasp
                )

        path = writer.finalize({
            "success": True,
            "terminal_state": "MOVE_PRE_GRASP",
            "joint_state_observed_rate_hz": 5.0,
        })
        with h5py.File(path, "r") as episode:
            np.testing.assert_allclose(
                episode["episode_meta/pregrasp_pose_base"][:], pregrasp
            )
            self.assertTrue(np.all(np.isnan(
                episode["episode_meta/grasp_pose_base"][:]
            )))
            self.assertTrue(np.all(np.isnan(
                episode["episode_meta/approach_vector_base"][:]
            )))
            self.assertTrue(np.isnan(
                episode["episode_meta/pregrasp_offset_m"][()]
            ))
            self.assertTrue(bool(episode.attrs["reached_pregrasp"]))
            self.assertTrue(bool(episode["task/reached_pregrasp"][-1]))
        quality = inspect_episode_quality(path)
        self.assertTrue(quality["quality_pass"], quality["quality_reason"])
        self.assertTrue(quality["sac_goal_valid"])

    def test_joint_transition_and_command_source_semantics(self):
        with h5py.File(self.path, "r") as episode:
            delta = np.asarray(episode["actions/executed_delta_joint"][:])
            q = np.asarray(episode["obs/joint_pos"][:])
            q_next = np.asarray(episode["next_obs/joint_pos"][:])
            valid = np.asarray(episode["valid/executed_delta_joint"][:], dtype=bool)
            np.testing.assert_allclose(delta[valid], (q_next[valid] - q[valid]))
            self.assertFalse(valid[-1])
            self.assertTrue(np.all(np.isnan(delta[-1])))
            source = [item.decode() if isinstance(item, bytes) else str(item)
                      for item in episode["actions/command_source"][:]]
            self.assertEqual(source[6], "MOVEJ")
            self.assertTrue(bool(episode["valid/commanded_joint_target"][6]))
            self.assertFalse(bool(episode["valid/commanded_joint_target"][1]))
            self.assertTrue(np.all(np.isnan(
                episode["actions/commanded_joint_target"][1]
            )))

    def test_repeated_joint_timestamp_is_reported_without_time_regression(self):
        with h5py.File(self.path, "r+") as episode:
            timestamps = episode["time/joint_state_timestamp_ns"][:]
            timestamps[2] = timestamps[1]
            episode["time/joint_state_timestamp_ns"][:] = timestamps
        result = inspect_episode_quality(self.path)
        self.assertTrue(result["quality_pass"], result["quality_reason"])
        self.assertTrue(result["joint_timestamp_monotonic"])
        self.assertFalse(result["joint_timestamp_strictly_increasing"])
        self.assertGreater(result["joint_repeated_frame_ratio"], 0.0)

    def test_stale_joint_state_is_not_marked_valid(self):
        stale_path = os.path.join(self.temporary.name, "episode_000010.hdf5")
        writer = EpisodeWriter(self.temporary.name, "episode_000010", 2, 2, {})
        base = {
            "timestamp_ns": 1, "rgb": np.ones((2, 2, 3), dtype=np.uint8),
            "depth": np.ones((2, 2), dtype=np.uint16), "joint_pos": [0] * 6,
            "tcp_pose": [0, 0, 0, 0, 0, 0, 1], "task_state": "MOVE_PRE_GRASP",
            "rgb_valid": True, "depth_valid": True, "robot_valid": True,
            "joint_valid": False, "joint_state_timestamp_ns": -1,
        }
        writer.append(base)
        base["timestamp_ns"] = 2
        writer.append(base)
        writer.finalize({"success": False, "terminal_state": "FAILED"})
        with h5py.File(stale_path, "r") as episode:
            self.assertFalse(bool(episode["valid/joint_state"][0]))
            self.assertFalse(bool(episode["valid/executed_delta_joint"][0]))

    def test_episode_qc(self):
        result = inspect_episode_quality(self.path)
        self.assertTrue(result["quality_pass"], result["quality_reason"])

    def test_policy_segment_validity_metrics(self):
        result = inspect_episode_quality(self.path)
        self.assertAlmostEqual(result["policy_segment_rgb_valid_ratio"], 1.0)
        self.assertAlmostEqual(result["policy_segment_depth_valid_ratio"], 1.0)
        self.assertAlmostEqual(result["policy_segment_all_valid_ratio"], 1.0)

    def test_task_state_mapping_is_global_and_idle_episode_is_not_success(self):
        short = EpisodeWriter(
            self.temporary.name, "episode_000010", 2, 2,
            {
                "min_episode_samples": 5,
                "min_episode_duration_sec": 0.5,
                "initial_joint_pos": [0.0] * 6,
                "initial_tcp_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                "initial_apple_camera_xyz": [1.0, 2.0, 3.0],
                "initial_apple_base_xyz": [4.0, 5.0, 6.0],
                "initial_apple_id": 7,
                "initial_apple_confidence": 0.9,
                "initial_apple_bbox": [1, 2, 3, 4],
            },
        )
        short.append({
            "timestamp_ns": 0,
            "rgb": np.ones((2, 2, 3), dtype=np.uint8),
            "depth": np.ones((2, 2), dtype=np.uint16),
            "joint_pos": [0.0] * 6,
            "tcp_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "task_state": "IDLE", "rgb_valid": True, "depth_valid": True,
            "robot_valid": True, "joint_valid": True,
            "teacher": {"target_valid": True},
        })
        path = short.finalize({"success": True, "terminal_state": "IDLE"})
        with h5py.File(path, "r") as episode:
            self.assertFalse(bool(episode.attrs["success"]))
            self.assertFalse(bool(episode.attrs["task_success"]))
            self.assertEqual(
                episode.attrs["state_mapping"],
                json.dumps(TASK_STATE_MAPPING, sort_keys=True),
            )
            np.testing.assert_allclose(
                episode["episode_meta/initial_joint_pos"][:], [0.0] * 6
            )
            np.testing.assert_allclose(
                episode["episode_meta/initial_apple_base_xyz"][:], [4.0, 5.0, 6.0]
            )
        quality = inspect_episode_quality(path)
        self.assertFalse(quality["quality_pass"])
        self.assertIn("invalid_terminal_state:IDLE", quality["quality_reason"])

    def test_interrupted_tmp_episode_is_preserved(self):
        writer = EpisodeWriter(
            self.temporary.name, "episode_000099", 4, 4,
            {"record_hz": 10.0}, compression="lzf",
        )
        writer.close_incomplete()
        self.assertTrue(os.path.isfile(writer.temp_path))
        result = inspect_episode_quality(writer.temp_path)
        self.assertFalse(result["quality_pass"])

    def test_backward_compatibility_with_existing_hdf5(self):
        self._make_legacy()
        result = inspect_episode_quality(self.path)
        self.assertTrue(result["quality_pass"], result["quality_reason"])

    def test_inspector_can_open_old_episode(self):
        self._make_legacy()
        inspector = _load_script("inspect_rl_dataset.py")
        self.assertEqual(inspector.inspect(self.path), 0)

    def test_replay_can_open_old_episode_headless(self):
        self._make_legacy()
        replay = _load_script("replay_episode.py")
        self.assertEqual(replay.replay(self.path, headless=True), 0)

    def test_three_mock_outcomes_coverage(self):
        create_episode(
            self.temporary.name, "episode_000002", physical=0,
            task_success=False, terminal="FAILED", failure_reason="grasp_miss",
        )
        create_episode(
            self.temporary.name, "episode_000003", physical=-1,
            task_success=False, terminal="ABORT_REQUESTED", failure_reason="operator_abort",
        )
        apply_physical_label(self.path, 1)
        report = collection_coverage(self.temporary.name)
        self.assertEqual(report["episode_count"], 3)
        self.assertEqual(report["physical_success"], 1)
        self.assertEqual(report["physical_failure"], 1)
        self.assertEqual(report["physical_unknown"], 1)

    def _make_legacy(self):
        with h5py.File(self.path, "r+") as episode:
            for name in (
                "task/control_mode", "valid/policy_action", "time/rgb_age_ms",
                "time/depth_age_ms", "time/robot_state_age_ms",
                "time/rgb_depth_delta_ms", "episode_meta",
                "labels", "events", "actions/executed_delta_joint",
                "actions/commanded_joint_target", "actions/command_source",
                "valid/joint_state", "valid/executed_delta_joint",
                "valid/commanded_joint_target", "valid/expert_joint_action",
                "time/joint_state_timestamp_ns", "time/joint_state_age_ms",
            ):
                if name in episode:
                    del episode[name]
            for name in (
                "dataset_schema_version", "control_mode_mapping",
                "policy_action_semantics", "initial_pose_id",
            ):
                if name in episode.attrs:
                    del episode.attrs[name]


if __name__ == "__main__":
    unittest.main()
