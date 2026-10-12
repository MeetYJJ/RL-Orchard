#!/usr/bin/env python3
"""Run 1-3 held-out HDF5 initial-to-pregrasp rollouts in MuJoCo only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.models import SACActor
from training.mujoco.rml63b_env import RML63BPregraspEnv


DEFAULT_RUN = ROOT / "training/runs/offline_sac_bc_20260923_112947"
DEFAULT_BC_RUN = ROOT / "training/runs/bc_pretrain_full_20260922_211200"
DEFAULT_DATASET = ROOT / "datasets/apple_rl_dataset_v1_20260922"
DEFAULT_MJCF = Path(__file__).resolve().parent / "rml63b.xml"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET))
    parser.add_argument("--split-manifest", default=str(DEFAULT_BC_RUN / "split_manifest.json"))
    parser.add_argument("--normalization", default=str(DEFAULT_BC_RUN / "normalization.npz"))
    parser.add_argument("--actor-checkpoint", default=str(DEFAULT_RUN / "actor_best.pt"))
    parser.add_argument("--mujoco-xml", default=str(DEFAULT_MJCF))
    parser.add_argument("--episode-ids", nargs="*", default=None,
                        help="Defaults to the first held-out Validation episodes in the BC manifest.")
    parser.add_argument("--max-episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=0,
                        help="0 uses this episode's recorded policy transition count.")
    parser.add_argument("--pregrasp-distance-threshold-m", type=float, default=0.005,
                        help="Provisional default matches the recorder's 5 mm pregrasp tolerance.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def _resolve_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else name)


def _load_policy(actor_path, normalization_path, device):
    with np.load(str(Path(normalization_path).expanduser().resolve()), allow_pickle=False) as norm:
        required = ("state_mean", "state_std", "action_scale")
        missing = [key for key in required if key not in norm]
        if missing:
            raise ValueError("formal BC normalization is missing {}".format(missing))
        state_mean = np.asarray(norm["state_mean"], dtype=np.float32)
        state_std = np.asarray(norm["state_std"], dtype=np.float32)
        action_scale = np.asarray(norm["action_scale"], dtype=np.float32)
    if state_mean.shape != (12,) or state_std.shape != (12,) or action_scale.shape != (6,):
        raise ValueError("normalization must have state [12] and action scale [6]")
    if not all(np.all(np.isfinite(x)) for x in (state_mean, state_std, action_scale)) or np.any(action_scale <= 0):
        raise ValueError("normalization contains invalid values")

    checkpoint = torch.load(str(Path(actor_path).expanduser().resolve()), map_location=device)
    if checkpoint.get("actor_class") != "SACActor" or (checkpoint.get("state_dim"), checkpoint.get("action_dim"), checkpoint.get("hidden_dim")) != (12, 6, 256):
        raise ValueError("checkpoint is not the expected 12/6/256 SACActor")
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("actor_state_dict", checkpoint.get("state_dict")))
    if state_dict is None:
        raise ValueError("SAC+BC checkpoint has no actor state dict")
    actor = SACActor(state_dim=12, action_dim=6, hidden_dim=256)
    actor.load_state_dict(state_dict, strict=True)
    actor.to(device).eval()
    return actor, state_mean, state_std, action_scale


def _read_episode(path, episode_id):
    episode_path = path / (str(episode_id) + ".hdf5")
    with h5py.File(str(episode_path), "r") as episode:
        required = (
            "episode_meta/initial_joint_pos", "episode_meta/pregrasp_pose_base",
            "obs/joint_pos", "valid/policy_action",
            "valid/executed_delta_joint",
        )
        missing = [key for key in required if key not in episode]
        if missing:
            raise ValueError("{} missing {}".format(episode_path, missing))
        initial_joint = np.asarray(episode[required[0]][:], dtype=np.float64)
        pregrasp_pose = np.asarray(episode[required[1]][:], dtype=np.float64)
        joint_samples = len(episode["obs/joint_pos"])
        policy = np.asarray(episode["valid/policy_action"][:], dtype=bool)
        delta_valid = np.asarray(episode["valid/executed_delta_joint"][:], dtype=bool)
    if initial_joint.shape != (6,) or pregrasp_pose.shape != (7,):
        raise ValueError("{} has invalid initial joint/goal shape".format(episode_id))
    if not np.all(np.isfinite(initial_joint)) or not np.all(np.isfinite(pregrasp_pose)):
        raise ValueError("{} initial pose or pregrasp goal contains NaN/Inf".format(episode_id))
    if len(policy) != joint_samples or len(delta_valid) != joint_samples:
        raise ValueError("{} policy validity arrays do not align with obs/joint_pos".format(episode_id))
    policy_transition_count = int(np.count_nonzero(policy & delta_valid))
    if policy_transition_count <= 0:
        raise ValueError("{} has no recorded policy transitions".format(episode_id))
    return initial_joint, pregrasp_pose[:3], policy_transition_count


def _episode_ids(manifest_path, explicit_ids, maximum):
    with open(Path(manifest_path).expanduser().resolve(), "r") as stream:
        manifest = json.load(stream)
    candidates = list(explicit_ids) if explicit_ids else list(manifest.get("validation_episode_ids", []))
    if not candidates:
        raise ValueError("no episode IDs were supplied and Validation split is empty")
    if len(candidates) > maximum:
        candidates = candidates[:maximum]
    if len(set(candidates)) != len(candidates):
        raise ValueError("episode IDs must be unique")
    return candidates


def _rollout(env, actor, state_mean, state_std, episode_id, initial_joint, goal,
             device, max_steps):
    state = env.reset(initial_joint, goal)
    initial_distance = env._distance()
    min_distance = initial_distance
    success = initial_distance < env.distance_threshold_m
    joint_limit_violation = False
    nan_inf = False
    steps = 0
    try:
        while not success and not joint_limit_violation and steps < max_steps:
            state_norm = (state - state_mean) / (state_std + 1e-8)
            state_tensor = torch.as_tensor(state_norm, dtype=torch.float32, device=device).unsqueeze(0)
            if state_tensor.shape != (1, 12) or not torch.isfinite(state_tensor).all():
                raise FloatingPointError("normalized policy state is invalid")
            with torch.no_grad():
                action_tensor = actor.deterministic_action(state_tensor)
            if action_tensor.shape != (1, 6) or not torch.isfinite(action_tensor).all():
                raise FloatingPointError("actor output must be finite with shape [1,6]")
            action_norm = action_tensor[0].cpu().numpy()
            state, success, info = env.step(action_norm)
            steps += 1
            joint_limit_violation = info["joint_limit_violation"]
            nan_inf = info["nan_inf"]
            min_distance = min(min_distance, info["distance_m"])
    except FloatingPointError:
        nan_inf = True

    final_distance = env._distance() if not nan_inf else float("nan")
    return {
        "episode_id": str(episode_id),
        "initial_distance_m": float(initial_distance),
        "final_distance_m": float(final_distance),
        "min_distance_m": float(min_distance),
        "steps": int(steps),
        "success": bool(success and not nan_inf),
        "joint_limit_violation": bool(joint_limit_violation),
        "nan_inf": bool(nan_inf),
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 1 <= args.max_episodes <= 3:
        raise SystemExit("--max-episodes must be in [1,3]")
    if args.max_steps < 0:
        raise SystemExit("--max-steps must be non-negative")
    device = _resolve_device(args.device)
    actor, state_mean, state_std, action_scale = _load_policy(
        args.actor_checkpoint, args.normalization, device)
    episode_ids = _episode_ids(args.split_manifest, args.episode_ids, args.max_episodes)
    env = RML63BPregraspEnv(
        args.mujoco_xml, action_scale,
        distance_threshold_m=args.pregrasp_distance_threshold_m,
        frame_skip=125,
    )
    print("model_loaded=true joints=joint1,joint2,joint3,joint4,joint5,joint6 tcp=robot_end")
    print("physics_timestep_s={:.6f} policy_timestep_s={:.6f} normalization={}".format(
        env.model.opt.timestep, env.model.opt.timestep * env.frame_skip,
        Path(args.normalization).expanduser().resolve()))

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    for episode_id in episode_ids:
        initial_joint, goal, recorded_policy_steps = _read_episode(dataset_dir, episode_id)
        horizon = recorded_policy_steps if args.max_steps == 0 else min(args.max_steps, recorded_policy_steps)
        result = _rollout(env, actor, state_mean, state_std, episode_id,
                          initial_joint, goal, device, horizon)
        print(json.dumps(result, sort_keys=True, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
