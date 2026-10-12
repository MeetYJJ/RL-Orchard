#!/usr/bin/env python3
"""Diagnose HDF5 TCP versus independent URDF and MuJoCo FK.

This is a pre-replay check.  It does not load an Actor, modify the HDF5,
start ROS, or run any policy/expert action.  A reset TCP mismatch stops here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.mujoco.check_expert_replay_consistency import _read_episode, _read_manifest
from training.mujoco.rml63b_env import RML63BPregraspEnv
from training.mujoco.urdf_fk import RML63BURDFChain, rotation_error_rad


DEFAULT_DATASET = ROOT / "datasets/apple_rl_dataset_v1_20260922"
DEFAULT_MANIFEST = ROOT / "training/runs/bc_pretrain_full_20260922_211200/split_manifest.json"
DEFAULT_MJCF = Path(__file__).resolve().parent / "rml63b.xml"
DEFAULT_URDF = Path(__file__).resolve().parent / "assets/rml63b/rm_63_b_description.urdf"
PROJECT_TOOL_OFFSET_XYZ = np.asarray([0.0, 0.0, 0.168], dtype=np.float64)


def _vector(text, expected):
    value = np.fromstring(str(text), sep=" ", dtype=np.float64)
    if value.shape != (expected,) or not np.all(np.isfinite(value)):
        raise argparse.ArgumentTypeError("expected {} finite space-separated values".format(expected))
    return value


def _json_array(value):
    return np.asarray(value, dtype=np.float64).tolist()


def _episode_attrs(dataset_dir, episode_id):
    path = Path(dataset_dir).expanduser().resolve() / (str(episode_id) + ".hdf5")
    with h5py.File(str(path), "r") as episode:
        attrs = {}
        for name in ("tcp_frame", "tcp_child_frame", "teacher_target_tcp_child_frame"):
            value = episode.attrs.get(name, "unknown")
            attrs[name] = value.decode() if isinstance(value, bytes) else str(value)
    return attrs


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--episode-id", default="episode_000043")
    parser.add_argument("--split-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--mujoco-xml", default=str(DEFAULT_MJCF))
    parser.add_argument("--urdf", default=str(DEFAULT_URDF))
    parser.add_argument("--max-reset-error-m", type=float, default=0.01)
    parser.add_argument(
        "--tool-offset-xyz", type=lambda value: _vector(value, 3),
        default=PROJECT_TOOL_OFFSET_XYZ.copy(),
        help="explicit configured robot_end->gripper_tip_center translation; default is project config 0 0 0.168",
    )
    parser.add_argument("--candidate-tool-z-m", type=float, default=0.158,
                        help="diagnostic-only flange-local Z candidate; never applied to the model")
    return parser


def _report_fk(env, urdf_chain, q0):
    urdf_poses = urdf_chain.forward(q0)
    levels = []
    for name in ("Link1", "Link2", "Link3", "Link4", "Link5", "Link6"):
        mjcf_pose = env.get_link_pose(name)
        urdf_pose = urdf_poses[name]
        delta = mjcf_pose["position"] - urdf_pose[:3, 3]
        levels.append({
            "link": name,
            "urdf_position": _json_array(urdf_pose[:3, 3]),
            "mjcf_position": _json_array(mjcf_pose["position"]),
            "position_error_xyz": _json_array(delta),
            "position_error_norm_m": float(np.linalg.norm(delta)),
            "rotation_error_rad": rotation_error_rad(urdf_pose[:3, :3], mjcf_pose["rotation"]),
        })
    base_pose = env.get_link_pose("base_link")
    base_delta = base_pose["position"]
    return {
        "base_frame": {
            "urdf_base_link_position": [0.0, 0.0, 0.0],
            "mjcf_world_base_link_position": _json_array(base_pose["position"]),
            "position_error_xyz": _json_array(base_delta),
            "position_error_norm_m": float(np.linalg.norm(base_delta)),
            "rotation_error_rad": rotation_error_rad(np.eye(3), base_pose["rotation"]),
        },
        "links": levels,
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.max_reset_error_m <= 0 or not np.isfinite(args.max_reset_error_m):
        raise SystemExit("--max-reset-error-m must be positive and finite")
    if args.candidate_tool_z_m <= 0 or not np.isfinite(args.candidate_tool_z_m):
        raise SystemExit("--candidate-tool-z-m must be positive and finite")

    episode_id = str(args.episode_id)
    _read_manifest(args.split_manifest, episode_id)
    data = _read_episode(args.dataset_dir, episode_id)
    attrs = _episode_attrs(args.dataset_dir, episode_id)
    transition_index = int(data["transition_indices"][0])
    q0 = data["joints"][transition_index]
    dataset_tcp = data["tcp"][transition_index]
    env = RML63BPregraspEnv(args.mujoco_xml, np.ones(6), frame_skip=125)
    env.reset(q0, data["goal"])
    urdf_chain = RML63BURDFChain(args.urdf)

    flange = env.get_flange_pose()
    tool = env.get_tool_pose(args.tool_offset_xyz)
    candidate = env.get_tool_pose([0.0, 0.0, args.candidate_tool_z_m])
    reset_report = {
        "first_policy_index": transition_index,
        "dataset_joint_q0": _json_array(q0),
        "sim_joint_q0": _json_array(env.joint_positions()),
        "max_joint_error": float(np.max(np.abs(env.joint_positions() - q0))),
        "dataset_tcp_xyz0": _json_array(dataset_tcp),
        "sim_robot_end_xyz0": _json_array(env.tcp_xyz()),
        "flange_xyz0": _json_array(flange["position"]),
        "tcp_position_error_m": float(np.linalg.norm(env.tcp_xyz() - dataset_tcp)),
        "flange_error_m": float(np.linalg.norm(flange["position"] - dataset_tcp)),
        "configured_tool_offset_xyz_m": _json_array(args.tool_offset_xyz),
        "tool_error_m": float(np.linalg.norm(tool["position"] - dataset_tcp)),
        "candidate_158mm_error_m": float(np.linalg.norm(candidate["position"] - dataset_tcp)),
        "candidate_158mm_position_xyz": _json_array(candidate["position"]),
        "diagnostic_only_candidate": True,
    }

    print("hdf5_tcp_semantics=" + json.dumps({
        "parent_frame": attrs["tcp_frame"],
        "child_frame": attrs["tcp_child_frame"],
        "teacher_child_frame": attrs["teacher_target_tcp_child_frame"],
        "project_interpretation": "base_link -> robot_end flange; tool TCP is separate",
    }, sort_keys=True))
    print("reset_consistency=" + json.dumps(reset_report, sort_keys=True))
    print("fk_levels=" + json.dumps(_report_fk(env, urdf_chain, q0), sort_keys=True))

    if reset_report["tcp_position_error_m"] > args.max_reset_error_m:
        print("BLOCKED: reset TCP/FK mismatch; do not continue to Expert replay")
        return 2
    print("PASS: reset TCP/FK error is within threshold; one-step replay may continue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
