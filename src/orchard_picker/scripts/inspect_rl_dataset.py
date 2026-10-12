#!/usr/bin/env python3
"""Validate and summarize one apple_rl_v1 or apple_rl_v1.1 HDF5 episode."""

import argparse
import json
import math
import os
import sys

import h5py
import numpy as np

from orchard_picker.rl_data.dataset_tools import (
    TIME_SERIES_PREFIXES,
    TIME_SERIES_ROOTS,
    inspect_episode_quality,
)


def _attr(file_handle, name, default="unknown"):
    value = file_handle.attrs.get(name, default)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _ratio(dataset):
    if len(dataset) == 0:
        return 0.0
    return float(np.count_nonzero(dataset[:])) / float(len(dataset))


def _finite_minmax(dataset, valid=None, batch=32):
    minimum = math.inf
    maximum = -math.inf
    nan_count = 0
    inf_count = 0
    for start in range(0, len(dataset), batch):
        values = np.asarray(dataset[start : start + batch])
        if valid is not None:
            mask = np.asarray(valid[start : start + batch], dtype=bool)
            values = values[mask]
        if not np.issubdtype(values.dtype, np.number):
            continue
        if np.issubdtype(values.dtype, np.floating):
            nan_count += int(np.isnan(values).sum())
            inf_count += int(np.isinf(values).sum())
            values = values[np.isfinite(values)]
        if values.size:
            minimum = min(minimum, float(values.min()))
            maximum = max(maximum, float(values.max()))
    if not math.isfinite(minimum):
        return None, None, nan_count, inf_count
    return minimum, maximum, nan_count, inf_count


def inspect(path):
    errors = []
    warnings = []
    with h5py.File(path, "r") as episode:
        required = [
            "obs/rgb",
            "obs/depth",
            "obs/joint_pos",
            "obs/tcp_pose",
            "actions/executed_delta_tcp",
            "task/state",
            "time/timestamp_ns",
            "valid/rgb",
            "valid/depth",
            "valid/robot_state",
            "valid/apple_target",
            "rewards",
            "dones",
        ]
        for name in required:
            if name not in episode:
                errors.append("missing dataset {}".format(name))
        if errors:
            for error in errors:
                print("ERROR:", error)
            return 1

        timestamps = np.asarray(episode["time/timestamp_ns"][:], dtype=np.int64)
        sample_count = len(timestamps)
        lengths = {}

        def visitor(name, item):
            if (
                isinstance(item, h5py.Dataset) and item.shape
                and (name in TIME_SERIES_ROOTS or name.startswith(TIME_SERIES_PREFIXES))
            ):
                lengths[name] = int(item.shape[0])

        episode.visititems(visitor)
        inconsistent = {
            name: length for name, length in lengths.items()
            if length != sample_count
        }
        if inconsistent:
            errors.append("dataset T mismatch: {}".format(inconsistent))

        monotonic = bool(
            sample_count < 2 or np.all(np.diff(timestamps) > 0)
        )
        if not monotonic:
            errors.append("timestamps are not strictly increasing")

        rgb = episode["obs/rgb"]
        depth = episode["obs/depth"]
        joint = episode["obs/joint_pos"]
        tcp = episode["obs/tcp_pose"]
        action = episode["actions/executed_delta_tcp"]
        joint_delta = episode.get("actions/executed_delta_joint")
        if joint.ndim != 2 or joint.shape[1] != 6:
            errors.append("joint_pos must have shape [T,6]")
        if tcp.ndim != 2 or tcp.shape[1] != 7:
            errors.append("tcp_pose must have shape [T,7]")

        rgb_min, rgb_max, _, _ = _finite_minmax(rgb)
        depth_min, depth_max, _, _ = _finite_minmax(depth)
        if sample_count and rgb_max == 0:
            errors.append("RGB is entirely black")
        if sample_count and depth_max == 0:
            errors.append("depth is entirely zero")

        valid_robot = episode["valid/robot_state"]
        joint_min, joint_max, joint_nan, joint_inf = _finite_minmax(
            joint, valid_robot
        )
        tcp_min, tcp_max, tcp_nan, tcp_inf = _finite_minmax(
            tcp, valid_robot
        )
        action_min, action_max, action_nan, action_inf = _finite_minmax(
            action, episode.get("valid/action")
        )
        nan_count = joint_nan + tcp_nan + action_nan
        inf_count = joint_inf + tcp_inf + action_inf
        if inf_count:
            errors.append("numeric robot/action data contains Inf")

        valid_mask = np.asarray(valid_robot[:], dtype=bool)
        if sample_count and np.any(valid_mask):
            quaternions = np.asarray(tcp[:, 3:7])[valid_mask]
            norms = np.linalg.norm(quaternions, axis=1)
            bad = np.logical_or(~np.isfinite(norms), np.abs(norms - 1.0) > 0.02)
            if np.any(bad):
                errors.append(
                    "{} valid TCP quaternions are not unit length".format(
                        int(np.count_nonzero(bad))
                    )
                )

        dones = np.asarray(episode["dones"][:], dtype=bool)
        terminal_state = str(_attr(episode, "terminal_state", ""))
        complete = bool(_attr(episode, "complete", False))
        if not complete or not terminal_state:
            errors.append("episode has no finalized terminal metadata")
        if sample_count == 0 or not dones[-1] or np.any(dones[:-1]):
            errors.append("dones must be true only at the final timestep")

        duration = float(_attr(episode, "duration_sec", 0.0))
        measured_rate = (
            float(sample_count - 1) / duration
            if sample_count > 1 and duration > 0.0
            else 0.0
        )
        mapping_text = str(_attr(episode, "state_mapping", "{}"))
        try:
            state_mapping = json.loads(mapping_text)
        except ValueError:
            state_mapping = {}
            warnings.append("state_mapping attribute is invalid JSON")
        state_names = {int(code): name for name, code in state_mapping.items()}
        phase_sequence = []
        for code in np.asarray(episode["task/state"][:], dtype=np.int64):
            phase = state_names.get(int(code), "UNKNOWN")
            if not phase_sequence or phase_sequence[-1] != phase:
                phase_sequence.append(phase)

        print("Episode:", _attr(episode, "episode_id", os.path.basename(path)))
        print("Samples:", sample_count)
        print("Duration: {:.3f} s".format(duration))
        print("Sample rate: {:.3f} Hz".format(measured_rate))
        print()
        print("RGB shape:", rgb.shape)
        print("Depth shape:", depth.shape)
        print("Joint shape:", joint.shape)
        print("TCP shape:", tcp.shape)
        print("Action shape:", action.shape)
        if joint_delta is not None:
            print("Executed joint delta shape:", joint_delta.shape)
        print()
        print("Terminal state:", terminal_state or "MISSING")
        min_samples = max(1, int(_attr(episode, "min_episode_samples", 5)))
        min_duration = max(0.0, float(_attr(episode, "min_episode_duration_sec", 0.5)))
        terminal_valid = terminal_state.strip().upper() not in ("", "IDLE", "UNKNOWN")
        task_success = bool(_attr(
            episode, "task_success", _attr(episode, "success", False)
        )) and sample_count >= min_samples and duration >= min_duration and terminal_valid
        print("Task success:", task_success)
        print("Episode valid:", bool(_attr(episode, "episode_valid", True)))
        print("Discarded:", bool(_attr(episode, "discarded", False)))
        print("Termination type:", _attr(episode, "termination_type", "normal"))
        print("Discard reason:", _attr(episode, "discard_reason", ""))
        print("Emergency stop timestamp:", _attr(
            episode, "emergency_stop_timestamp_ns", -1
        ))
        print("Physical grasp success:", _attr(episode, "physical_grasp_success", -1))
        print("Phase sequence:", " -> ".join(phase_sequence))
        print("Expert event count:", int(_attr(
            episode, "expert_event_count",
            len(episode["events/expert_timestamp_ns"])
            if "events/expert_timestamp_ns" in episode else 0,
        )))
        print("Failure reason:", _attr(episode, "failure_reason", ""))
        print("Operator note:", _attr(episode, "operator_note", ""))
        print("Initial pose:", _attr(episode, "initial_pose_id", "unknown"))
        print("Initialization mode:", _attr(episode, "initialization_mode", "unknown"))
        print("Apple position bin:", _attr(episode, "apple_position_bin", "DISABLED"))
        print("Scripted final approach distance (m):", _attr(
            episode, "scripted_final_approach_distance_m", "unknown"
        ))
        print("Reached pre-grasp:", bool(_attr(episode, "reached_pregrasp", False)))
        if "episode_meta/pregrasp_pose_base" in episode:
            print("Pre-grasp pose (base):", episode["episode_meta/pregrasp_pose_base"][:].tolist())
        if "episode_meta/grasp_pose_base" in episode:
            print("Grasp pose (base):", episode["episode_meta/grasp_pose_base"][:].tolist())
        print("SAC goal valid:", bool(
            "episode_meta/pregrasp_pose_base" in episode
            and np.all(np.isfinite(episode["episode_meta/pregrasp_pose_base"][:]))
            and "episode_meta/grasp_pose_base" in episode
            and np.all(np.isfinite(episode["episode_meta/grasp_pose_base"][:]))
        ))
        print("Reward version:", _attr(episode, "reward_version", "unassigned"))
        print()
        print("RGB valid ratio: {:.3f}".format(_ratio(episode["valid/rgb"])))
        print("Depth valid ratio: {:.3f}".format(_ratio(episode["valid/depth"])))
        print("Robot state valid ratio: {:.3f}".format(_ratio(valid_robot)))
        if "valid/joint_state" in episode:
            print("Joint state valid ratio: {:.3f}".format(_ratio(episode["valid/joint_state"])))
            print("Joint state source:", _attr(episode, "joint_state_source", "unknown"))
            print("Joint state effective rate (Hz):", _attr(episode, "joint_state_observed_rate_hz", "unknown"))
            if "time/joint_state_age_ms" in episode:
                ages = np.asarray(episode["time/joint_state_age_ms"][:], dtype=np.float64)
                ages = ages[np.isfinite(ages)]
                print("Joint state mean/max age (ms):", float(np.mean(ages)) if ages.size else None, float(np.max(ages)) if ages.size else None)
        if "valid/executed_delta_joint" in episode:
            print("Executed joint delta valid ratio: {:.3f}".format(_ratio(episode["valid/executed_delta_joint"])))
        if "valid/commanded_joint_target" in episode:
            print("Commanded joint target valid ratio: {:.3f}".format(_ratio(episode["valid/commanded_joint_target"])))
        print("Apple target valid ratio: {:.3f}".format(_ratio(episode["valid/apple_target"])))
        if "valid/policy_action" in episode:
            print("Policy action valid ratio: {:.3f}".format(
                _ratio(episode["valid/policy_action"])
            ))
        else:
            print("Policy action valid ratio: legacy episode (not recorded)")
        print()
        print("RGB min/max:", rgb_min, rgb_max)
        print("Depth min/max:", depth_min, depth_max)
        print("TCP min/max:", tcp_min, tcp_max)
        print("Joint min/max:", joint_min, joint_max)
        joint_valid_dataset = episode.get("valid/joint_state", valid_robot)
        joint_valid_mask = np.asarray(joint_valid_dataset[:], dtype=bool)
        joint_rows = np.asarray(joint[:])[joint_valid_mask]
        if joint_rows.size:
            print("Joint per-axis min:", np.min(joint_rows, axis=0).tolist())
            print("Joint per-axis max:", np.max(joint_rows, axis=0).tolist())
            print("Joint per-axis mean:", np.mean(joint_rows, axis=0).tolist())
            print("Joint per-axis std:", np.std(joint_rows, axis=0).tolist())
        print("Action min/max:", action_min, action_max)
        if joint_delta is not None:
            delta_valid = episode.get("valid/executed_delta_joint")
            delta_min, delta_max, delta_nan, delta_inf = _finite_minmax(joint_delta, delta_valid)
            print("Executed joint delta min/max:", delta_min, delta_max)
            print("Executed joint delta NaN/Inf:", delta_nan, delta_inf)
            delta_mask = (
                np.asarray(delta_valid[:], dtype=bool)
                if delta_valid is not None
                else np.zeros(len(joint_delta), dtype=bool)
            )
            delta_rows = np.asarray(joint_delta[:])[delta_mask]
            if delta_rows.size:
                print("Joint delta per-axis min:", np.min(delta_rows, axis=0).tolist())
                print("Joint delta per-axis max:", np.max(delta_rows, axis=0).tolist())
                print("Joint delta per-axis mean:", np.mean(delta_rows, axis=0).tolist())
                print("Joint delta per-axis std:", np.std(delta_rows, axis=0).tolist())
                print("Joint delta per-axis max abs:", np.max(np.abs(delta_rows), axis=0).tolist())
            if "time/joint_state_timestamp_ns" in episode:
                jt = np.asarray(episode["time/joint_state_timestamp_ns"][:], dtype=np.int64)
                jt = jt[jt >= 0]
                print("Joint timestamp nondecreasing:", bool(len(jt) < 2 or np.all(np.diff(jt) >= 0)))
        print("NaN count (valid robot/action rows):", nan_count)
        print("Inf count (valid robot/action rows):", inf_count)
        print("Timestamp monotonic:", monotonic)

    quality = inspect_episode_quality(path)
    print("Quality pass:", quality["quality_pass"])
    print("Quality reason:", quality["quality_reason"] or "none")
    if "joint_repeated_frame_ratio" in quality:
        print("Joint repeated frame ratio:", quality["joint_repeated_frame_ratio"])
    if "joint_state_timestamp_rate_hz" in quality:
        print("Joint timestamp update rate (Hz):", quality["joint_state_timestamp_rate_hz"])
    if "joint_delta_transition_max_error" in quality:
        print("Joint transition max consistency error:", quality["joint_delta_transition_max_error"])
    if "next_obs_joint_alignment_max_error" in quality:
        print("Next joint alignment max error:", quality["next_obs_joint_alignment_max_error"])
    if "next_obs_tcp_alignment_max_error" in quality:
        print("Next TCP alignment max error:", quality["next_obs_tcp_alignment_max_error"])
    if "sac_goal_valid" in quality:
        print("SAC goal valid:", quality["sac_goal_valid"])
    if "reached_pregrasp" in quality:
        print("Reached pre-grasp:", quality["reached_pregrasp"])
    print("Policy-segment RGB valid ratio:", quality.get(
        "policy_segment_rgb_valid_ratio", 0.0
    ))
    print("Policy-segment depth valid ratio:", quality.get(
        "policy_segment_depth_valid_ratio", 0.0
    ))
    print("Policy-segment all-valid ratio:", quality.get(
        "policy_segment_all_valid_ratio", 0.0
    ))

    for warning in warnings:
        print("WARNING:", warning)
    for error in errors:
        print("ERROR:", error)
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", help="episode_XXXXXX.hdf5 path")
    arguments = parser.parse_args()
    if not os.path.isfile(arguments.episode):
        parser.error("file does not exist: {}".format(arguments.episode))
    return inspect(arguments.episode)


if __name__ == "__main__":
    sys.exit(main())
