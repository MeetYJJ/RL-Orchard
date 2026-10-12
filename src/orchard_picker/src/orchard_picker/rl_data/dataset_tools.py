"""Episode labels, quality checks, summaries, and collection manifest helpers."""

import csv
import json
import math
import os
import tempfile
from datetime import datetime

import fcntl
import h5py
import numpy as np


MANIFEST_FIELDS = [
    "episode_id", "file", "start_time", "duration", "sample_count",
    "initial_pose_id", "initial_joint_summary", "initial_tcp_summary",
    "initialization_mode", "initialization_seed", "apple_position_bin",
    "apple_base_x", "apple_base_y", "apple_base_z", "target_id",
    "target_confidence", "task_success", "physical_grasp_success",
    "terminal_state", "failure_reason", "operator_note", "reached_pregrasp",
    "pregrasp_offset_m", "sac_goal_valid", "reward_version", "rgb_valid_ratio",
    "depth_valid_ratio", "robot_state_valid_ratio", "target_valid_ratio",
    "action_valid_ratio", "policy_action_ratio",
    "policy_segment_rgb_valid_ratio", "policy_segment_depth_valid_ratio",
    "policy_segment_all_valid_ratio",
    "joint_state_source", "joint_state_effective_rate_hz",
    "joint_valid_ratio", "joint_state_mean_age_ms",
    "executed_delta_joint_valid_ratio", "commanded_joint_target_valid_ratio",
    "joint_delta_max_abs",
    "scripted_final_approach_distance", "episode_valid", "discarded",
    "termination_type", "discard_reason", "quality_pass", "quality_reason",
]

TIME_SERIES_PREFIXES = (
    "obs/", "next_obs/", "actions/", "teacher/", "task/", "time/", "valid/",
    "reward_terms/", "debug/",
)
TIME_SERIES_ROOTS = ("rewards", "dones")


def _text(value, default=""):
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _scalar_attr(handle, name, default=None):
    value = handle.attrs.get(name, default)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _ratio(handle, path):
    if path not in handle or len(handle[path]) == 0:
        return 0.0
    values = np.asarray(handle[path][:], dtype=bool)
    return float(np.count_nonzero(values)) / float(len(values))


def _masked_ratio(values, mask):
    """Return the fraction of ``values`` that is true inside ``mask``."""
    values = np.asarray(values, dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if values.shape != mask.shape or not np.any(mask):
        return 0.0
    return float(np.count_nonzero(values[mask])) / float(np.count_nonzero(mask))


def _has_nonzero(dataset, batch_size=16):
    """Check large image datasets without loading a full episode into RAM."""
    for start in range(0, len(dataset), batch_size):
        if np.any(dataset[start:start + batch_size]):
            return True
    return False


def _json_mapping(handle, name):
    try:
        return json.loads(_text(handle.attrs.get(name, "{}")))
    except (TypeError, ValueError):
        return {}


def _state_names(handle):
    mapping = _json_mapping(handle, "state_mapping")
    return {int(code): str(name) for name, code in mapping.items()}


def _phase_sequence(handle):
    if "task/state" not in handle:
        return []
    names = _state_names(handle)
    return [names.get(int(code), "UNKNOWN") for code in handle["task/state"][:]]


def _time_series_lengths(handle):
    result = {}

    def visitor(name, item):
        if not isinstance(item, h5py.Dataset) or not item.shape:
            return
        if name in TIME_SERIES_ROOTS or name.startswith(TIME_SERIES_PREFIXES):
            result[name] = int(item.shape[0])

    handle.visititems(visitor)
    return result


def inspect_episode_quality(path):
    """Run lightweight, non-destructive QC and return JSON-serializable metrics."""
    reasons = []
    metrics = {
        "quality_pass": False,
        "quality_reason": "",
        "sample_count": 0,
        "phase_counts": {},
    }
    required = (
        "obs/rgb", "obs/depth", "obs/joint_pos", "obs/tcp_pose",
        "actions/executed_delta_tcp", "task/state", "time/timestamp_ns",
        "valid/rgb", "valid/depth", "valid/robot_state",
        "valid/apple_target", "valid/action", "dones",
    )
    try:
        with h5py.File(path, "r") as episode:
            missing = [name for name in required if name not in episode]
            if missing:
                reasons.append("missing datasets: {}".format(",".join(missing)))
                metrics["quality_reason"] = "; ".join(reasons)
                return metrics

            timestamps = np.asarray(episode["time/timestamp_ns"][:], dtype=np.int64)
            count = len(timestamps)
            metrics["sample_count"] = count
            # These thresholds are stored per episode so older datasets remain
            # readable while newly recorded short/idle files cannot be treated
            # as successful demonstrations.
            timestamp_duration_sec = (
                float(timestamps[-1] - timestamps[0]) / 1e9
                if count > 1 else 0.0
            )
            duration_sec = float(_scalar_attr(
                episode, "duration_sec", timestamp_duration_sec
            ))
            if not math.isfinite(duration_sec):
                duration_sec = timestamp_duration_sec
            min_samples = int(_scalar_attr(episode, "min_episode_samples", 5))
            min_duration = float(_scalar_attr(
                episode, "min_episode_duration_sec", 0.5
            ))
            min_samples = max(1, min_samples)
            min_duration = max(0.0, min_duration)
            metrics["duration_sec"] = duration_sec
            metrics["min_episode_samples"] = min_samples
            metrics["min_episode_duration_sec"] = min_duration
            if count < min_samples:
                reasons.append(
                    "sample_count_below_minimum:{}<{}".format(
                        count, min_samples
                    )
                )
            if duration_sec < min_duration:
                reasons.append(
                    "duration_below_minimum:{:.6f}<{:.6f}".format(
                        duration_sec, min_duration
                    )
                )
            terminal_state = _text(
                episode.attrs.get("terminal_state", "UNKNOWN")
            ).strip()
            if terminal_state.upper() in ("", "IDLE", "UNKNOWN"):
                reasons.append(
                    "invalid_terminal_state:{}".format(terminal_state or "EMPTY")
                )
            lengths = _time_series_lengths(episode)
            mismatch = {name: value for name, value in lengths.items() if value != count}
            if mismatch:
                reasons.append("T mismatch: {}".format(json.dumps(mismatch, sort_keys=True)))
            if count < 1:
                reasons.append("empty episode")
            elif count > 1 and not bool(np.all(np.diff(timestamps) > 0)):
                reasons.append("timestamps not strictly increasing")

            if count and not _has_nonzero(episode["obs/rgb"]):
                reasons.append("RGB entirely black")
            if count and not _has_nonzero(episode["obs/depth"]):
                reasons.append("depth entirely zero")

            robot_valid = np.asarray(episode["valid/robot_state"][:], dtype=bool)
            if count and not np.any(robot_valid):
                reasons.append("no valid robot state")
            joint = np.asarray(episode["obs/joint_pos"][:])
            tcp = np.asarray(episode["obs/tcp_pose"][:])
            if joint.ndim != 2 or joint.shape[1:] != (6,):
                reasons.append("joint_pos shape is not [T,6]")
            if tcp.ndim != 2 or tcp.shape[1:] != (7,):
                reasons.append("tcp_pose shape is not [T,7]")
            joint_valid = (
                np.asarray(episode["valid/joint_state"][:], dtype=bool)
                if "valid/joint_state" in episode
                else robot_valid
            )
            if joint.ndim == 2 and len(joint) == len(joint_valid):
                values = joint[joint_valid]
                if np.any(~np.isfinite(values)):
                    reasons.append("valid joint state contains NaN/Inf")
            if tcp.ndim == 2 and tcp.shape[1:] == (7,) and len(tcp) == len(robot_valid):
                values = tcp[robot_valid]
                if np.any(~np.isfinite(values)):
                    reasons.append("valid TCP state contains NaN/Inf")
                elif len(values):
                    norms = np.linalg.norm(values[:, 3:7], axis=1)
                    if np.any(np.abs(norms - 1.0) > 0.02):
                        reasons.append("valid TCP quaternion is not unit length")

            # Joint-space expert transition fields are optional for backward
            # compatibility with apple_rl_v1 episodes.  When present, they
            # must obey q[t+1] - q[t] exactly and carry explicit validity.
            if "actions/executed_delta_joint" in episode:
                joint_delta = np.asarray(episode["actions/executed_delta_joint"][:])
                joint_delta_valid = np.asarray(
                    episode.get("valid/executed_delta_joint", np.zeros(count, dtype=bool))[:],
                    dtype=bool,
                )
                if joint_delta.ndim != 2 or joint_delta.shape[1:] != (6,):
                    reasons.append("executed_delta_joint shape is not [T,6]")
                if len(joint_delta_valid) != count:
                    reasons.append("executed_delta_joint valid mask length mismatch")
                elif joint_delta.shape == (count, 6):
                    if np.any(~np.isfinite(joint_delta[joint_delta_valid])):
                        reasons.append("valid executed_delta_joint contains NaN/Inf")
                    if count and joint_delta_valid[-1]:
                        reasons.append("last valid/executed_delta_joint must be false")
                    if count > 1 and np.any(joint_delta_valid[:-1]) and "next_obs/joint_pos" in episode:
                        q0 = joint[:-1]
                        q1 = np.asarray(episode["next_obs/joint_pos"][:-1])
                        check = joint_delta_valid[:-1]
                        finite = np.all(np.isfinite(q0), axis=1) & np.all(
                            np.isfinite(q1), axis=1
                        )
                        check = check & finite
                        if np.any(check):
                            error = np.max(
                                np.abs(q0[check] + joint_delta[:-1][check] - q1[check])
                            )
                            metrics["joint_delta_transition_max_error"] = float(error)
                            if error > 1e-5:
                                reasons.append(
                                    "executed_delta_joint does not equal q[t+1]-q[t]"
                                )
                metrics["executed_delta_joint_valid_ratio"] = _ratio(
                    episode, "valid/executed_delta_joint"
                )
                if joint_delta.ndim == 2 and joint_delta.shape[1:] == (6,):
                    finite_delta = joint_delta[np.isfinite(joint_delta)]
                    if finite_delta.size:
                        metrics["joint_delta_max_abs"] = float(np.max(np.abs(finite_delta)))
                    valid_delta = joint_delta[joint_delta_valid]
                    if valid_delta.size:
                        metrics["joint_delta_per_joint"] = {
                            "min": np.min(valid_delta, axis=0).astype(float).tolist(),
                            "max": np.max(valid_delta, axis=0).astype(float).tolist(),
                            "mean": np.mean(valid_delta, axis=0).astype(float).tolist(),
                            "std": np.std(valid_delta, axis=0).astype(float).tolist(),
                            "max_abs": np.max(np.abs(valid_delta), axis=0).astype(float).tolist(),
                        }
            if "actions/commanded_joint_target" in episode:
                commanded = np.asarray(episode["actions/commanded_joint_target"][:])
                commanded_valid = np.asarray(
                    episode.get("valid/commanded_joint_target", np.zeros(count, dtype=bool))[:],
                    dtype=bool,
                )
                if commanded.ndim != 2 or commanded.shape[1:] != (6,):
                    reasons.append("commanded_joint_target shape is not [T,6]")
                elif len(commanded_valid) != count:
                    reasons.append("commanded_joint_target valid mask length mismatch")
                else:
                    if np.any(~np.isfinite(commanded[commanded_valid])):
                        reasons.append("valid commanded_joint_target contains NaN/Inf")
                    invalid_rows = commanded[np.logical_not(commanded_valid)]
                    if invalid_rows.size and np.any(np.isfinite(invalid_rows)):
                        reasons.append("invalid commanded_joint_target must be NaN")
                if "actions/command_source" in episode:
                    sources = np.asarray([
                        _text(value) for value in episode["actions/command_source"][:]
                    ])
                    allowed_sources = {
                        "NONE", "MOVEJ", "MOVEJ_P", "MOVEL", "MOVEJ_FOLLOW",
                        "MOVEJ_CANFD", "GRIPPER", "OTHER",
                    }
                    unknown_sources = sorted(set(sources) - allowed_sources)
                    if unknown_sources:
                        reasons.append("unknown command_source: {}".format(
                            ",".join(unknown_sources)
                        ))
                    joint_target_sources = np.isin(
                        sources, ["MOVEJ", "MOVEJ_FOLLOW", "MOVEJ_CANFD"]
                    )
                    if len(sources) != count:
                        reasons.append("command_source length mismatch")
                    elif np.any(np.logical_and(commanded_valid, ~joint_target_sources)):
                        reasons.append(
                            "Cartesian/gripper command has a fabricated joint target"
                        )
                metrics["commanded_joint_target_valid_ratio"] = _ratio(
                    episode, "valid/commanded_joint_target"
                )
            if "valid/expert_joint_action" in episode:
                metrics["expert_joint_action_valid_ratio"] = _ratio(
                    episode, "valid/expert_joint_action"
                )
                if (
                    "task/control_mode" in episode
                    and "valid/executed_delta_joint" in episode
                ):
                    expert_joint = np.asarray(
                        episode["valid/expert_joint_action"][:], dtype=bool
                    )
                    executed_valid = np.asarray(
                        episode["valid/executed_delta_joint"][:], dtype=bool
                    )
                    control_modes = np.asarray(episode["task/control_mode"][:])
                    mapping = _json_mapping(episode, "control_mode_mapping")
                    learnable_code = mapping.get("LEARNABLE_POLICY")
                    if learnable_code is not None:
                        expected = np.logical_and(
                            executed_valid, control_modes == int(learnable_code)
                        )
                        if not np.array_equal(expert_joint, expected):
                            reasons.append("expert_joint_action mask is inconsistent")
            if "valid/joint_state" in episode:
                metrics["joint_valid_ratio"] = _ratio(episode, "valid/joint_state")
            if "time/joint_state_age_ms" in episode:
                ages = np.asarray(episode["time/joint_state_age_ms"][:], dtype=np.float64)
                ages = ages[np.isfinite(ages)]
                metrics["joint_state_mean_age_ms"] = float(np.mean(ages)) if ages.size else None
                metrics["joint_state_age_ms_max"] = float(np.max(ages)) if ages.size else None
            if "time/joint_state_timestamp_ns" in episode:
                all_joint_times = np.asarray(
                    episode["time/joint_state_timestamp_ns"][:], dtype=np.int64
                )
                if len(all_joint_times) == len(joint_valid):
                    all_joint_times = all_joint_times[joint_valid]
                joint_times = all_joint_times[all_joint_times >= 0]
                diffs = np.diff(joint_times) if len(joint_times) >= 2 else np.asarray([])
                nondecreasing = bool(len(diffs) == 0 or np.all(diffs >= 0))
                metrics["joint_timestamp_monotonic"] = nondecreasing
                metrics["joint_timestamp_strictly_increasing"] = bool(
                    len(diffs) == 0 or np.all(diffs > 0)
                )
                metrics["joint_repeated_frame_ratio"] = (
                    float(np.count_nonzero(diffs == 0)) / float(len(diffs))
                    if len(diffs) else 0.0
                )
                if not nondecreasing:
                    reasons.append("joint timestamps decreased")
                if len(joint_times) >= 2:
                    positive_diffs = diffs[diffs > 0]
                    metrics["joint_state_timestamp_rate_hz"] = (
                        float(1e9 / np.median(positive_diffs))
                        if len(positive_diffs) else None
                    )
            observed_rate = float(_scalar_attr(
                episode, "joint_state_observed_rate_hz", np.nan
            ))
            metrics["joint_state_effective_rate_hz"] = (
                observed_rate if math.isfinite(observed_rate)
                else metrics.get("joint_state_timestamp_rate_hz")
            )
            metrics["joint_state_source"] = _text(
                episode.attrs.get("joint_state_source", "unknown")
            )

            # SAC goal fields are optional so apple_rl_v1/v1 episodes remain
            # readable.  The learnable first stage only requires a finite,
            # frozen pre-grasp goal.  The later MOVE_GRASP pose and derived
            # approach metadata are optional because an episode can end
            # immediately after the pre-grasp segment.
            goal_paths = (
                "teacher/pregrasp_pose_base", "teacher/grasp_pose_base",
                "teacher/approach_vector_base", "teacher/pregrasp_offset_m",
            )
            if any(path in episode for path in goal_paths):
                goal_valid = False
                pre = np.asarray(episode.get("teacher/pregrasp_pose_base", np.empty((0, 7)))[:])
                grasp = np.asarray(episode.get("teacher/grasp_pose_base", np.empty((0, 7)))[:])
                approach = np.asarray(episode.get("teacher/approach_vector_base", np.empty((0, 3)))[:])
                offset = np.asarray(episode.get("teacher/pregrasp_offset_m", np.empty((0,)))[:])
                meta_pre = np.asarray(
                    episode.get(
                        "episode_meta/pregrasp_pose_base", np.empty((0,))
                    )[:]
                )
                meta_pre_valid = bool(
                    meta_pre.shape == (7,) and np.all(np.isfinite(meta_pre))
                )
                schema_version = _text(
                    episode.attrs.get("dataset_schema_version", "")
                )
                meta_pre_required = schema_version.startswith("apple_rl_v1.2")
                if meta_pre_required and not meta_pre_valid:
                    reasons.append("pregrasp goal unavailable")
                if pre.shape == (count, 7):
                    pre_rows = np.all(np.isfinite(pre), axis=1)
                    if np.any(pre_rows):
                        goal_valid = bool(meta_pre_valid or not meta_pre_required)
                        first = int(np.flatnonzero(pre_rows)[0])
                        if not np.allclose(pre[pre_rows], pre[first], atol=1e-6):
                            reasons.append("pregrasp goal is not frozen")
                        if meta_pre_valid and not np.allclose(
                            pre[first], meta_pre, atol=1e-6
                        ):
                            reasons.append(
                                "episode_meta pregrasp goal differs from teacher goal"
                            )

                        # Grasp/approach fields are validated only when the
                        # scripted MOVE_GRASP event was actually observed.
                        if grasp.shape == (count, 7):
                            grasp_rows = np.all(np.isfinite(grasp), axis=1)
                            if np.any(grasp_rows):
                                grasp_first = int(np.flatnonzero(grasp_rows)[0])
                                if not np.allclose(
                                    grasp[grasp_rows], grasp[grasp_first], atol=1e-6
                                ):
                                    reasons.append("grasp goal is not frozen")
                                if not np.all(np.isfinite(pre[grasp_first])):
                                    reasons.append("grasp goal has no finite pregrasp pair")
                                else:
                                    delta = grasp[grasp_first, :3] - pre[grasp_first, :3]
                                    distance = float(np.linalg.norm(delta))
                                    if distance <= 1e-12:
                                        reasons.append("grasp/pregrasp goal distance is zero")
                                    elif (
                                        approach.shape == (count, 3)
                                        and np.any(np.all(np.isfinite(approach), axis=1))
                                    ):
                                        expected = delta / distance
                                        approach_rows = np.all(np.isfinite(approach), axis=1)
                                        if not np.allclose(
                                            approach[approach_rows], expected, atol=1e-5
                                        ):
                                            reasons.append(
                                                "approach_vector_base is inconsistent with goals"
                                            )
                                    if offset.shape == (count,):
                                        offset_rows = np.isfinite(offset)
                                        if np.any(offset_rows) and not np.allclose(
                                            offset[offset_rows], distance, atol=1e-5
                                        ):
                                            reasons.append(
                                                "pregrasp_offset_m is inconsistent with goals"
                                            )
                    elif count:
                        reasons.append("teacher pregrasp goal unavailable")
                elif count:
                    reasons.append("teacher pregrasp goal shape mismatch")
                metrics["sac_goal_valid"] = bool(goal_valid)
                metrics["reached_pregrasp"] = bool(_scalar_attr(
                    episode, "reached_pregrasp", False
                ))

            # next_obs[t] is deliberately back-filled from obs[t+1].  Check
            # this independently of the joint-delta check for BC/SAC loaders.
            if count > 1 and "next_obs/joint_pos" in episode and "next_obs/tcp_pose" in episode:
                transition_mask = np.asarray(episode.get(
                    "valid/action", np.zeros(count, dtype=bool)
                )[:], dtype=bool)[:-1]
                next_joint = np.asarray(episode["next_obs/joint_pos"][:-1])
                next_tcp = np.asarray(episode["next_obs/tcp_pose"][:-1])
                expected_joint = joint[1:]
                expected_tcp = tcp[1:]
                finite_joint = np.all(np.isfinite(next_joint), axis=1) & np.all(
                    np.isfinite(expected_joint), axis=1
                )
                finite_tcp = np.all(np.isfinite(next_tcp), axis=1) & np.all(
                    np.isfinite(expected_tcp), axis=1
                )
                joint_check = transition_mask & finite_joint
                tcp_check = transition_mask & finite_tcp
                metrics["next_obs_joint_alignment_max_error"] = float(
                    np.max(np.abs(next_joint[joint_check] - expected_joint[joint_check]))
                ) if np.any(joint_check) else 0.0
                metrics["next_obs_tcp_alignment_max_error"] = float(
                    np.max(np.abs(next_tcp[tcp_check] - expected_tcp[tcp_check]))
                ) if np.any(tcp_check) else 0.0
                if metrics["next_obs_joint_alignment_max_error"] > 1e-5:
                    reasons.append("next_obs/joint_pos is not obs[t+1]")
                if metrics["next_obs_tcp_alignment_max_error"] > 1e-5:
                    reasons.append("next_obs/tcp_pose is not obs[t+1]")

            if "reward_terms/distance_to_pregrasp_m" in episode:
                for name in (
                    "reward_terms/distance_to_pregrasp_m",
                    "reward_terms/progress_m", "reward_terms/action_norm",
                ):
                    if name in episode and len(episode[name]) != count:
                        reasons.append("{} length mismatch".format(name))

            action_valid = np.asarray(episode["valid/action"][:], dtype=bool)
            actions = np.asarray(episode["actions/executed_delta_tcp"][:])
            if len(actions) == len(action_valid) and np.any(
                ~np.isfinite(actions[action_valid])
            ):
                reasons.append("valid action contains NaN/Inf")
            dones = np.asarray(episode["dones"][:], dtype=bool)
            if count == 0 or len(dones) != count or not dones[-1] or np.any(dones[:-1]):
                reasons.append("dones must be true only on final row")
            if count and action_valid[-1]:
                reasons.append("last valid/action must be false")
            if not bool(_scalar_attr(episode, "complete", False)):
                reasons.append("episode not finalized")
            if not _text(episode.attrs.get("terminal_state", "")):
                reasons.append("terminal state missing")

            target_ratio = _ratio(episode, "valid/apple_target")
            expert_ratio = _ratio(episode, "valid/expert_action")
            if count and target_ratio <= 0.0:
                reasons.append("no valid apple target")
            if count and expert_ratio <= 0.0:
                reasons.append("no valid expert action")

            phases = _phase_sequence(episode)
            phase_counts = {}
            for phase in phases:
                phase_counts[phase] = phase_counts.get(phase, 0) + 1
            metrics["phase_counts"] = phase_counts
            if phases and phases[0] in (
                "MOVE_GRASP", "CLOSE_GRIPPER", "RETREAT", "HOME",
                "RELEASE_GRIPPER", "DONE",
            ):
                reasons.append("episode starts after learnable approach phase")
            ordered = {
                "OPEN_GRIPPER": 0, "MOVE_PRE_GRASP": 1, "MOVE_GRASP": 2,
                "CLOSE_GRIPPER": 3, "GRASPED_HOLDING": 4, "TWIST_J6": 5,
                "RETREAT": 6, "HOME": 7, "RELEASE_GRIPPER": 8, "DONE": 9,
            }
            ranks = [ordered[phase] for phase in phases if phase in ordered]
            if any(following < current for current, following in zip(ranks, ranks[1:])):
                reasons.append("task phase order regressed")

            rgb_stamp = episode.get("time/rgb_timestamp_ns")
            depth_stamp = episode.get("time/depth_timestamp_ns")
            if rgb_stamp is not None and depth_stamp is not None and count:
                rgb_ns = np.asarray(rgb_stamp[:], dtype=np.int64)
                depth_ns = np.asarray(depth_stamp[:], dtype=np.int64)
                valid = np.logical_and(rgb_ns >= 0, depth_ns >= 0)
                deltas = np.abs(rgb_ns[valid] - depth_ns[valid]) / 1e6
                metrics["rgb_depth_delta_ms_max"] = (
                    float(deltas.max()) if deltas.size else None
                )
            if "time/robot_state_age_ms" in episode:
                ages = np.asarray(episode["time/robot_state_age_ms"][:], dtype=np.float64)
                ages = ages[np.isfinite(ages)]
                metrics["robot_state_age_ms_max"] = (
                    float(ages.max()) if ages.size else None
                )
            age_checks = (
                ("time/rgb_age_ms", "valid/rgb", "max_sensor_age_ms", "RGB"),
                ("time/depth_age_ms", "valid/depth", "max_sensor_age_ms", "depth"),
                (
                    "time/robot_state_age_ms", "valid/robot_state",
                    "max_robot_state_age_ms", "robot state",
                ),
                (
                    "time/joint_state_age_ms", "valid/joint_state",
                    "max_robot_state_age_ms", "joint state",
                ),
            )
            for age_path, valid_path, attr_name, label in age_checks:
                limit = float(_scalar_attr(episode, attr_name, np.nan))
                if age_path not in episode or not math.isfinite(limit):
                    continue
                age_values = np.asarray(episode[age_path][:], dtype=np.float64)
                age_valid = np.asarray(episode[valid_path][:], dtype=bool)
                checked = age_values[np.logical_and(age_valid, np.isfinite(age_values))]
                if checked.size and np.any(checked > limit + 1e-3):
                    reasons.append("{} age exceeds recorded limit".format(label))

            metrics.update({
                "rgb_valid_ratio": _ratio(episode, "valid/rgb"),
                "depth_valid_ratio": _ratio(episode, "valid/depth"),
                "robot_state_valid_ratio": _ratio(episode, "valid/robot_state"),
                "target_valid_ratio": target_ratio,
                "action_valid_ratio": _ratio(episode, "valid/action"),
                "expert_action_valid_ratio": expert_ratio,
                "policy_action_ratio": _ratio(episode, "valid/policy_action"),
            })
            policy_mask = np.asarray(
                episode.get("valid/policy_action", np.zeros(count, dtype=bool))[:],
                dtype=bool,
            )
            rgb_valid = np.asarray(episode["valid/rgb"][:], dtype=bool)
            depth_valid = np.asarray(episode["valid/depth"][:], dtype=bool)
            all_valid = (
                rgb_valid
                & np.asarray(episode["valid/depth"][:], dtype=bool)
                & np.asarray(episode["valid/robot_state"][:], dtype=bool)
                & np.asarray(episode["valid/apple_target"][:], dtype=bool)
            )
            metrics["policy_segment_rgb_valid_ratio"] = _masked_ratio(
                rgb_valid, policy_mask
            )
            metrics["policy_segment_depth_valid_ratio"] = _masked_ratio(
                depth_valid, policy_mask
            )
            metrics["policy_segment_all_valid_ratio"] = _masked_ratio(
                all_valid, policy_mask
            )
    except (OSError, ValueError) as exc:
        reasons.append("cannot read episode: {}".format(exc))

    metrics["quality_pass"] = not reasons
    metrics["quality_reason"] = "; ".join(reasons)
    return metrics


def write_quality_result(path, result):
    """Persist QC status without changing any recorded trajectory data."""
    with h5py.File(path, "r+") as episode:
        episode.attrs["quality_pass"] = bool(result.get("quality_pass", False))
        episode.attrs["quality_reason"] = str(result.get("quality_reason", ""))
        episode.attrs["quality_checked_at"] = datetime.now().astimezone().isoformat()
        episode.attrs["quality_metrics_json"] = json.dumps(result, sort_keys=True)
        episode.flush()


def _vector_summary(value):
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return "[{}]".format(",".join(
        "nan" if not math.isfinite(item) else "{:.6g}".format(item)
        for item in array
    ))


def summarize_episode(path, quality=None):
    """Return one manifest row from either v1 or extended episodes."""
    if quality is None:
        quality = inspect_episode_quality(path)
    with h5py.File(path, "r") as episode:
        count = len(episode["time/timestamp_ns"]) if "time/timestamp_ns" in episode else 0
        base = np.asarray([np.nan, np.nan, np.nan], dtype=np.float64)
        if "teacher/apple_base_xyz" in episode and count:
            valid = (
                np.asarray(episode["valid/apple_target"][:], dtype=bool)
                if "valid/apple_target" in episode else np.ones(count, dtype=bool)
            )
            if np.any(valid):
                base = np.asarray(episode["teacher/apple_base_xyz"][:])[valid][0]
        target_id = -1
        confidence = np.nan
        if count and "teacher/apple_id" in episode:
            target_id = int(episode["teacher/apple_id"][0])
        if count and "teacher/confidence" in episode:
            confidence = float(episode["teacher/confidence"][0])

        initial_joint = episode.attrs.get("initial_joint_pos", [np.nan] * 6)
        initial_tcp = episode.attrs.get("initial_tcp_pose", [np.nan] * 7)
        if "episode_meta/initial_joint_pos" in episode:
            initial_joint = episode["episode_meta/initial_joint_pos"][:]
        if "episode_meta/initial_tcp_pose" in episode:
            initial_tcp = episode["episode_meta/initial_tcp_pose"][:]
        physical = int(_scalar_attr(episode, "physical_grasp_success", -1))
        terminal_state = _text(episode.attrs.get("terminal_state", ""))
        stored_task_success = bool(_scalar_attr(episode, "task_success", False))
        stored_episode_valid = bool(_scalar_attr(episode, "episode_valid", True))
        discarded = bool(_scalar_attr(episode, "discarded", False))
        termination_type = _text(episode.attrs.get("termination_type", "normal"))
        discard_reason = _text(episode.attrs.get("discard_reason", ""))
        # The explicit short-duration/short-episode/IDLE rules are hard
        # invalidations.  Other QC failures remain separate from the task
        # state-machine outcome.
        quality_reason = str(quality.get("quality_reason", ""))
        invalid_episode = any(
            marker in quality_reason
            for marker in (
                "sample_count_below_minimum:",
                "duration_below_minimum:",
                "invalid_terminal_state:",
            )
        )
        effective_task_success = stored_task_success and not invalid_episode and (
            terminal_state.strip().upper() not in ("", "IDLE", "UNKNOWN")
        )
        effective_episode_valid = bool(
            stored_episode_valid and not discarded and not invalid_episode
        )
        row = {
            "episode_id": _text(episode.attrs.get("episode_id", os.path.basename(path))),
            "file": os.path.abspath(path),
            "start_time": _text(episode.attrs.get("start_time", "")),
            "duration": float(_scalar_attr(episode, "duration_sec", 0.0)),
            "sample_count": count,
            "initial_pose_id": _text(episode.attrs.get("initial_pose_id", "unknown")),
            "initial_joint_summary": _vector_summary(initial_joint),
            "initial_tcp_summary": _vector_summary(initial_tcp),
            "initialization_mode": _text(episode.attrs.get("initialization_mode", "unknown")),
            "initialization_seed": int(_scalar_attr(episode, "initialization_seed", -1)),
            "apple_position_bin": _text(episode.attrs.get("apple_position_bin", "DISABLED")),
            "apple_base_x": float(base[0]),
            "apple_base_y": float(base[1]),
            "apple_base_z": float(base[2]),
            "target_id": target_id,
            "target_confidence": confidence,
            "task_success": int(effective_task_success),
            "physical_grasp_success": physical,
            "terminal_state": terminal_state,
            "failure_reason": _text(episode.attrs.get("failure_reason", "")),
            "operator_note": _text(episode.attrs.get("operator_note", "")),
            "reached_pregrasp": int(bool(_scalar_attr(
                episode, "reached_pregrasp", False
            ))),
            "pregrasp_offset_m": float(_scalar_attr(
                episode, "pregrasp_offset_m", np.nan
            )),
            "sac_goal_valid": int(bool(quality.get("sac_goal_valid", False))),
            "reward_version": _text(episode.attrs.get("reward_version", "unassigned")),
            "rgb_valid_ratio": quality.get("rgb_valid_ratio", 0.0),
            "depth_valid_ratio": quality.get("depth_valid_ratio", 0.0),
            "robot_state_valid_ratio": quality.get("robot_state_valid_ratio", 0.0),
            "target_valid_ratio": quality.get("target_valid_ratio", 0.0),
            "action_valid_ratio": quality.get("action_valid_ratio", 0.0),
            "policy_action_ratio": quality.get("policy_action_ratio", 0.0),
            "policy_segment_rgb_valid_ratio": quality.get(
                "policy_segment_rgb_valid_ratio", 0.0
            ),
            "policy_segment_depth_valid_ratio": quality.get(
                "policy_segment_depth_valid_ratio", 0.0
            ),
            "policy_segment_all_valid_ratio": quality.get(
                "policy_segment_all_valid_ratio", 0.0
            ),
            "joint_state_source": _text(episode.attrs.get("joint_state_source", "unknown")),
            "joint_state_effective_rate_hz": quality.get(
                "joint_state_effective_rate_hz", _scalar_attr(
                    episode, "joint_state_observed_rate_hz", np.nan
                )
            ),
            "joint_valid_ratio": quality.get("joint_valid_ratio", ""),
            "joint_state_mean_age_ms": quality.get("joint_state_mean_age_ms", ""),
            "executed_delta_joint_valid_ratio": quality.get(
                "executed_delta_joint_valid_ratio", ""
            ),
            "commanded_joint_target_valid_ratio": quality.get(
                "commanded_joint_target_valid_ratio", ""
            ),
            "joint_delta_max_abs": quality.get("joint_delta_max_abs", ""),
            "scripted_final_approach_distance": float(_scalar_attr(
                episode, "scripted_final_approach_distance_m", np.nan
            )),
            "episode_valid": int(effective_episode_valid),
            "discarded": int(discarded),
            "termination_type": termination_type,
            "discard_reason": discard_reason,
            "quality_pass": int(bool(quality.get("quality_pass", False))),
            "quality_reason": str(quality.get("quality_reason", "")),
        }
    return row


class ManifestStore:
    """Locked, atomic collection_manifest.csv append/update helper."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.lock_path = self.path + ".lock"

    def read(self):
        if not os.path.isfile(self.path):
            return []
        with open(self.path, "r", encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def append_or_update(self, row):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.lock_path, "a+", encoding="utf-8") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            rows = self.read()
            key = str(row.get("episode_id", ""))
            normalized = {name: row.get(name, "") for name in MANIFEST_FIELDS}
            replaced = False
            for index, existing in enumerate(rows):
                if str(existing.get("episode_id", "")) == key:
                    rows[index] = normalized
                    replaced = True
                    break
            if not replaced:
                rows.append(normalized)
            descriptor, temporary = tempfile.mkstemp(
                prefix=".collection_manifest.", suffix=".tmp",
                dir=os.path.dirname(self.path), text=True,
            )
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return normalized


def refresh_manifest_for_episode(path, manifest_path=None):
    """QC one finalized episode and append/update its manifest row."""
    result = inspect_episode_quality(path)
    write_quality_result(path, result)
    row = summarize_episode(path, result)
    if manifest_path is None:
        manifest_path = os.path.join(os.path.dirname(path), "collection_manifest.csv")
    ManifestStore(manifest_path).append_or_update(row)
    return row


def apply_physical_label(path, label, failure_reason="", operator_note="", manifest_path=None):
    """Set a human physical outcome (-1/0/1), preserving task_success."""
    value = int(label)
    if value not in (-1, 0, 1):
        raise ValueError("physical label must be -1, 0, or 1")
    with h5py.File(path, "r+") as episode:
        episode.attrs["physical_grasp_success"] = value
        episode.attrs["failure_reason"] = str(failure_reason)
        episode.attrs["operator_note"] = str(operator_note)
        episode.attrs["physical_label_time"] = datetime.now().astimezone().isoformat()
        labels = episode.require_group("labels")
        labels.attrs["physical_grasp_success"] = value
        labels.attrs["failure_reason"] = str(failure_reason)
        labels.attrs["operator_note"] = str(operator_note)
        episode.flush()
    return refresh_manifest_for_episode(path, manifest_path)


def collection_coverage(dataset_root):
    """Aggregate volume, quality, phase, and control-mode coverage."""
    paths = sorted(
        os.path.join(dataset_root, name)
        for name in os.listdir(dataset_root)
        if name.startswith("episode_") and name.endswith(".hdf5")
        and not name.endswith(".tmp.hdf5")
    ) if os.path.isdir(dataset_root) else []
    report = {
        "episode_count": len(paths),
        "physical_success": 0,
        "physical_failure": 0,
        "physical_unknown": 0,
        "task_success": 0,
        "episode_valid": 0,
        "episode_invalid": 0,
        "termination_type_counts": {},
        "quality_pass": 0,
        "quality_fail": 0,
        "sac_goal_valid": 0,
        "reached_pregrasp": 0,
        "mean_duration_sec": 0.0,
        "init_pose_counts": {},
        "apple_position_bin_counts": {},
        "failure_reason_counts": {},
        "phase_sample_counts": {},
        "control_mode_sample_counts": {},
        "valid_ratio_means": {},
        "interrupted_tmp_count": 0,
    }
    if os.path.isdir(dataset_root):
        report["interrupted_tmp_count"] = len([
            name for name in os.listdir(dataset_root)
            if name.startswith("episode_") and name.endswith(".tmp.hdf5")
        ])
    durations = []
    ratio_names = (
        "rgb_valid_ratio", "depth_valid_ratio", "robot_state_valid_ratio",
        "target_valid_ratio", "action_valid_ratio", "policy_action_ratio",
        "policy_segment_rgb_valid_ratio", "policy_segment_depth_valid_ratio",
        "policy_segment_all_valid_ratio",
        "joint_valid_ratio", "executed_delta_joint_valid_ratio",
        "commanded_joint_target_valid_ratio",
    )
    ratios = {name: [] for name in ratio_names}
    mode_names = {}
    for path in paths:
        episode_quality = inspect_episode_quality(path)
        row = summarize_episode(path, episode_quality)
        physical = int(row["physical_grasp_success"])
        if physical == 1:
            report["physical_success"] += 1
        elif physical == 0:
            report["physical_failure"] += 1
        else:
            report["physical_unknown"] += 1
        report["task_success"] += int(row["task_success"])
        report["sac_goal_valid"] += int(row.get("sac_goal_valid", 0))
        report["reached_pregrasp"] += int(row.get("reached_pregrasp", 0))
        if int(row.get("episode_valid", 1)):
            report["episode_valid"] += 1
        else:
            report["episode_invalid"] += 1
        termination = str(row.get("termination_type", "normal") or "normal")
        report["termination_type_counts"][termination] = (
            report["termination_type_counts"].get(termination, 0) + 1
        )
        quality = bool(int(row["quality_pass"]))
        report["quality_pass" if quality else "quality_fail"] += 1
        durations.append(float(row["duration"]))
        for field, destination in (
            ("initial_pose_id", "init_pose_counts"),
            ("apple_position_bin", "apple_position_bin_counts"),
            ("failure_reason", "failure_reason_counts"),
        ):
            value = str(row[field] or "(empty)")
            report[destination][value] = report[destination].get(value, 0) + 1
        for name in ratio_names:
            ratios[name].append(float(row[name]))
        with h5py.File(path, "r") as episode:
            for phase, count in episode_quality.get("phase_counts", {}).items():
                report["phase_sample_counts"][phase] = (
                    report["phase_sample_counts"].get(phase, 0) + int(count)
                )
            if "task/control_mode" in episode:
                mapping = _json_mapping(episode, "control_mode_mapping")
                mode_names = {int(code): name for name, code in mapping.items()}
                values, counts = np.unique(episode["task/control_mode"][:], return_counts=True)
                for value, count in zip(values, counts):
                    name = mode_names.get(int(value), "UNKNOWN_{}".format(int(value)))
                    report["control_mode_sample_counts"][name] = (
                        report["control_mode_sample_counts"].get(name, 0) + int(count)
                    )
            else:
                count = int(row["sample_count"])
                report["control_mode_sample_counts"]["LEGACY_UNSPECIFIED"] = (
                    report["control_mode_sample_counts"].get("LEGACY_UNSPECIFIED", 0)
                    + count
                )
    report["mean_duration_sec"] = float(np.mean(durations)) if durations else 0.0
    report["physical_success_rate_labeled"] = (
        float(report["physical_success"])
        / float(report["physical_success"] + report["physical_failure"])
        if report["physical_success"] + report["physical_failure"] else None
    )
    report["valid_ratio_means"] = {
        name: float(np.mean(values)) if values else 0.0
        for name, values in ratios.items()
    }
    return report


def training_episode_paths(dataset_root):
    """Return only complete, QC-passing episodes eligible for training.

    Invalid/discarded episodes are intentionally retained on disk for safety
    review and failure analysis, but this helper makes the default training
    selection exclude them.  Older files without the new attributes remain
    readable and are evaluated through the existing QC rules.
    """
    if not os.path.isdir(dataset_root):
        return []
    paths = sorted(
        os.path.join(dataset_root, name)
        for name in os.listdir(dataset_root)
        if name.startswith("episode_")
        and name.endswith(".hdf5")
        and not name.endswith(".tmp.hdf5")
    )
    eligible = []
    for path in paths:
        try:
            quality = inspect_episode_quality(path)
            row = summarize_episode(path, quality)
            if int(row.get("episode_valid", 0)) and int(
                row.get("quality_pass", 0)
            ):
                eligible.append(path)
        except (OSError, ValueError, KeyError):
            # A damaged file is retained, but never enters a training list.
            continue
    return eligible
