#!/usr/bin/env python3
"""Check recorded Expert joint deltas against the RML63-B MuJoCo model.

This script intentionally does not load an Actor, use normalization, start
ROS, or connect to a robot. It checks the recorded HDF5 trajectory itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.mujoco.rml63b_env import RML63BPregraspEnv


DEFAULT_DATASET = ROOT / "datasets/apple_rl_dataset_v1_20260922"
DEFAULT_MANIFEST = ROOT / "training/runs/bc_pretrain_full_20260922_211200/split_manifest.json"
DEFAULT_MJCF = Path(__file__).resolve().parent / "rml63b.xml"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--episode-id", default="episode_000043")
    parser.add_argument("--split-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--mujoco-xml", default=str(DEFAULT_MJCF))
    parser.add_argument("--max-reset-tcp-error-m", type=float, default=0.01,
                        help="Stop before replay if reset TCP error exceeds this value.")
    parser.add_argument("--frame-skip", type=int, default=125,
                        help="Physics steps per recorded Expert joint delta.")
    return parser


def _json_array(value):
    return np.asarray(value, dtype=np.float64).tolist()


def _max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _read_manifest(path, episode_id):
    if not path:
        return None
    with open(Path(path).expanduser().resolve(), "r") as stream:
        manifest = json.load(stream)
    validation = set(manifest.get("validation_episode_ids", []))
    if validation and episode_id not in validation:
        raise ValueError("{} is not in the manifest Validation split".format(episode_id))
    return manifest


def _read_episode(path, episode_id):
    episode_path = Path(path).expanduser().resolve() / (str(episode_id) + ".hdf5")
    with h5py.File(str(episode_path), "r") as episode:
        required = (
            "obs/joint_pos", "obs/tcp_pose",
            "episode_meta/pregrasp_pose_base",
            "actions/executed_delta_joint",
            "valid/policy_action", "valid/executed_delta_joint",
        )
        missing = [name for name in required if name not in episode]
        if missing:
            raise ValueError("{} missing {}".format(episode_path, missing))
        joints = np.asarray(episode["obs/joint_pos"][:], dtype=np.float64)
        tcp = np.asarray(episode["obs/tcp_pose"][:], dtype=np.float64)
        goal_pose = np.asarray(episode["episode_meta/pregrasp_pose_base"][:], dtype=np.float64)
        deltas = np.asarray(episode["actions/executed_delta_joint"][:], dtype=np.float64)
        policy = np.asarray(episode["valid/policy_action"][:], dtype=bool)
        delta_valid = np.asarray(episode["valid/executed_delta_joint"][:], dtype=bool)
    if joints.ndim != 2 or joints.shape[1:] != (6,) or len(joints) < 2:
        raise ValueError("obs/joint_pos must have shape [T,6] with T >= 2")
    if tcp.ndim != 2 or tcp.shape[0] != len(joints) or tcp.shape[1] < 3:
        raise ValueError("obs/tcp_pose must align with obs/joint_pos")
    if goal_pose.shape != (7,) or not np.all(np.isfinite(goal_pose)):
        raise ValueError("episode frozen pregrasp pose must be finite with shape [7]")
    if any(len(value) != len(joints) for value in (deltas, policy, delta_valid)):
        raise ValueError("HDF5 transition arrays have inconsistent lengths")
    transition_mask = policy & delta_valid
    candidate_indices = np.flatnonzero(transition_mask & (np.arange(len(joints)) + 1 < len(joints))).astype(np.int64)
    if len(candidate_indices) == 0:
        raise ValueError("episode has no valid policy transitions")

    first_gap = np.flatnonzero(np.diff(candidate_indices) != 1)
    segment_end = int(first_gap[0]) + 1 if len(first_gap) else len(candidate_indices)
    transition_indices = candidate_indices[:segment_end]

    finite = (
        np.all(np.isfinite(joints[transition_indices]), axis=1)
        & np.all(np.isfinite(tcp[transition_indices, :3]), axis=1)
        & np.all(np.isfinite(deltas[transition_indices]), axis=1)
        & np.all(np.isfinite(joints[transition_indices + 1]), axis=1)
        & np.all(np.isfinite(tcp[transition_indices + 1, :3]), axis=1)
    )
    if not np.all(finite):
        bad = transition_indices[~finite].tolist()
        raise FloatingPointError(
            "valid policy transitions contain NaN/Inf at indices {}".format(bad)
        )
    return {
        "episode_id": str(episode_id),
        "joints": joints,
        "tcp": tcp[:, :3],
        "goal": goal_pose[:3],
        "deltas": deltas,
        "transition_indices": transition_indices,
    }


def _reset_report(env, data):
    index = int(data["transition_indices"][0])
    env.reset(data["joints"][index], data["goal"])
    sim_q0 = env.joint_positions()
    sim_tcp0 = env.tcp_xyz()
    dataset_q0 = data["joints"][index]
    dataset_tcp0 = data["tcp"][index]
    report = {
        "dataset_joint_q0": _json_array(dataset_q0),
        "sim_joint_q0": _json_array(sim_q0),
        "max_joint_error": _max_abs(sim_q0, dataset_q0),
        "dataset_tcp_xyz0": _json_array(dataset_tcp0),
        "sim_tcp_xyz0": _json_array(sim_tcp0),
        "tcp_position_error_m": float(np.linalg.norm(sim_tcp0 - dataset_tcp0)),
    }
    return report


def _one_step_report(env, data):
    index = int(data["transition_indices"][0])
    q0 = data["joints"][index]
    expert_delta_q = data["deltas"][index]
    q_target = q0 + expert_delta_q
    sim_state, _success, info = env.step_delta_q(expert_delta_q)
    sim_q_after = env.joint_positions()
    dataset_q_next = data["joints"][index + 1]
    sim_tcp_after = env.tcp_xyz()
    dataset_tcp_next = data["tcp"][index + 1]
    return {
        "expert_delta_q": _json_array(expert_delta_q),
        "q_target": _json_array(q_target),
        "sim_q_after": _json_array(sim_q_after),
        "dataset_q_next": _json_array(dataset_q_next),
        "max_sim_q_vs_target_error": _max_abs(sim_q_after, q_target),
        "max_sim_q_vs_dataset_next_error": _max_abs(sim_q_after, dataset_q_next),
        "sim_tcp_after": _json_array(sim_tcp_after),
        "dataset_tcp_next": _json_array(dataset_tcp_next),
        "tcp_next_position_error_m": float(np.linalg.norm(sim_tcp_after - dataset_tcp_next)),
        "joint_limit_violation": bool(info["joint_limit_violation"]),
        "physics_substeps": int(env.frame_skip),
    }


def _continuous_replay(env, data):
    first_index = int(data["transition_indices"][0])
    env.reset(data["joints"][first_index], data["goal"])
    initial_distance = env._distance()
    min_distance = initial_distance
    joint_limit_violation = False
    for index in data["transition_indices"]:
        _state, _success, info = env.step_delta_q(data["deltas"][int(index)])
        min_distance = min(min_distance, info["distance_m"])
        joint_limit_violation = joint_limit_violation or bool(info["joint_limit_violation"])
    sim_final_distance = env._distance()
    final_sample = int(data["transition_indices"][-1]) + 1
    dataset_final_distance = float(np.linalg.norm(data["tcp"][final_sample] - data["goal"]))
    return {
        "episode_id": data["episode_id"],
        "first_policy_index": first_index,
        "last_policy_index": int(data["transition_indices"][-1]),
        "policy_transition_count": int(len(data["transition_indices"])),
        "number_of_expert_actions": int(len(data["transition_indices"])),
        "initial_distance": float(initial_distance),
        "dataset_final_distance": dataset_final_distance,
        "sim_final_distance": float(sim_final_distance),
        "min_sim_distance": float(min_distance),
        "joint_limit_violation": bool(joint_limit_violation),
        "physics_substeps_per_action": int(env.frame_skip),
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.frame_skip <= 0 or args.max_reset_tcp_error_m <= 0:
        raise SystemExit("frame-skip and max-reset-tcp-error-m must be positive")
    episode_id = str(args.episode_id)
    _read_manifest(args.split_manifest, episode_id)
    data = _read_episode(args.dataset_dir, episode_id)
    env = RML63BPregraspEnv(args.mujoco_xml, np.ones(6), frame_skip=args.frame_skip)

    reset = _reset_report(env, data)
    print("policy_segment=" + json.dumps({
        "first_policy_index": int(data["transition_indices"][0]),
        "last_policy_index": int(data["transition_indices"][-1]),
        "policy_transition_count": int(len(data["transition_indices"])),
    }, sort_keys=True))
    print("reset_consistency=" + json.dumps(reset, sort_keys=True))
    if reset["tcp_position_error_m"] > args.max_reset_tcp_error_m:
        print("BLOCKED: reset TCP position error exceeds threshold; Expert replay stopped")
        return 2

    one_step = _one_step_report(env, data)
    print("one_step_expert_replay=" + json.dumps(one_step, sort_keys=True))

    replay = _continuous_replay(env, data)
    print("continuous_expert_replay=" + json.dumps(replay, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
