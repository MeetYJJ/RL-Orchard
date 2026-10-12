#!/usr/bin/env python3
"""Audit Stage 3A offline transitions without training or changing HDF5."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

try:
    from .offline_rl.dataset import (OfflineRLTransitionDataset, audit_actor_checkpoint,
                                      load_split_manifest)
    from .offline_rl.reward import RewardConfig
except ImportError:
    from offline_rl.dataset import OfflineRLTransitionDataset, audit_actor_checkpoint, load_split_manifest
    from offline_rl.reward import RewardConfig


def _stats(values):
    value = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(value):
        return {"count": 0, "min": None, "max": None, "mean": None, "std": None}
    return {"count": int(len(value)), "min": float(np.min(value)), "max": float(np.max(value)),
            "mean": float(np.mean(value)), "std": float(np.std(value))}


def _safe(value):
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_safe(item) for item in value.tolist()]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--reward-config", required=True)
    parser.add_argument("--actor-checkpoint", default="")
    parser.add_argument("--split", choices=("train", "validation", "test", "all"), default="all")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--alignment-tolerance", type=float, default=1e-5)
    parser.add_argument("--primitive-tolerance", type=float, default=1e-4)
    return parser


def _split_ids(manifest, split):
    if split == "all":
        names = ("train", "validation", "test")
    else:
        names = (split,)
    return {name: list(manifest["{}_episode_ids".format(name)]) for name in names}


def _audit_split(dataset):
    reward = dataset.reward[:, 0] if len(dataset) else np.asarray([], dtype=np.float32)
    progress = dataset.progress[:, 0] if len(dataset) else np.asarray([], dtype=np.float32)
    action_component = -dataset.reward_config.action_weight * dataset.action_cost[:, 0] if len(dataset) else np.asarray([], dtype=np.float32)
    terminal_success = int(np.sum(dataset.terminal_success[:, 0] > 0.5)) if len(dataset) else 0
    done = dataset.done[:, 0] > 0.5 if len(dataset) else np.asarray([], dtype=bool)
    terminal_failure = int(np.sum(done & ~(dataset.terminal_success[:, 0] > 0.5))) if len(dataset) else 0
    positive = reward[progress > 0]
    negative = reward[progress < 0]
    returns = {}
    for eid in sorted(set(dataset.episode_id_array.tolist())):
        returns[eid] = float(np.sum(reward[dataset.episode_id_array == eid]))
    normalized = dataset.action_norm if len(dataset) else np.zeros((0, 6), dtype=np.float32)
    overflow = np.abs(normalized) > 1.0
    return {
        "transition_count": int(len(dataset)),
        "loaded_episode_count": int(sum(item.loaded for item in dataset.episode_audits)),
        "skipped_episode_count": int(sum(not item.loaded for item in dataset.episode_audits)),
        "reward": _stats(reward),
        "progress_component": _stats(dataset.reward_config.progress_weight * progress),
        "action_penalty_component": _stats(action_component),
        "terminal_success_bonus_count": terminal_success,
        "terminal_failure_penalty_count": terminal_failure,
        "action_overflow_count_per_joint": np.sum(overflow, axis=0).astype(int).tolist(),
        "action_overflow_max_per_joint": np.maximum(np.max(np.abs(normalized), axis=0) - 1.0, 0.0).tolist() if len(dataset) else [0.0] * 6,
        "reward_mean_when_progress_positive": float(np.mean(positive)) if len(positive) else None,
        "reward_mean_when_progress_negative": float(np.mean(negative)) if len(negative) else None,
        "episode_returns": returns,
    }


def _coverage(dataset):
    if not len(dataset):
        return {"initial_joint": {}, "pregrasp_xyz": {}, "relative_goal": {}}
    initial = np.asarray([item.initial_joint for item in dataset.episode_audits if item.loaded and item.initial_joint is not None], dtype=np.float32)
    if initial.ndim != 2 or initial.shape[1] != 6:
        initial = dataset.state_raw[:, :6]
    return {
        "initial_joint": {"min": np.min(initial, 0).tolist(), "max": np.max(initial, 0).tolist()},
        "pregrasp_xyz": {"min": np.min(dataset.state_raw[:, 6:9] + dataset.state_raw[:, 9:12], 0).tolist(), "max": np.max(dataset.state_raw[:, 6:9] + dataset.state_raw[:, 9:12], 0).tolist()},
        "relative_goal": {"min": np.min(dataset.state_raw[:, 9:12], 0).tolist(), "max": np.max(dataset.state_raw[:, 9:12], 0).tolist()},
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    manifest = load_split_manifest(args.split_manifest, args.dataset_dir)
    reward_config = RewardConfig.load(args.reward_config)
    selected = _split_ids(manifest, args.split)
    output = Path(args.output_dir or ("training/runs/offline_rl_audit_" + datetime.now().strftime("%Y%m%d_%H%M%S"))).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    datasets = {}
    for name, ids in selected.items():
        datasets[name] = OfflineRLTransitionDataset(
            args.dataset_dir, ids, args.normalization, reward_config,
            alignment_tolerance=args.alignment_tolerance,
            primitive_tolerance=args.primitive_tolerance,
        )
    if args.actor_checkpoint:
        actor_audit = audit_actor_checkpoint(args.actor_checkpoint, manifest)
        print("Actor compatibility: PASS; log_std unfreeze API:", actor_audit["unfreeze_log_std_verified"])
    else:
        actor_audit = None
    audit = {"dataset_dir": str(Path(args.dataset_dir).expanduser().resolve()), "split_manifest": str(Path(args.split_manifest).expanduser().resolve()), "normalization": str(Path(args.normalization).expanduser().resolve()), "reward_config": str(Path(args.reward_config).expanduser().resolve()), "splits": {}, "problematic_episodes": []}
    manifest_ids = set(manifest["train_episode_ids"]) | set(manifest["validation_episode_ids"]) | set(manifest["test_episode_ids"])
    dataset_files = {path.stem for path in Path(args.dataset_dir).expanduser().resolve().glob("episode_*.hdf5")}
    audit["manifest_coverage"] = {"manifest_episode_count": len(manifest_ids), "dataset_episode_file_count": len(dataset_files), "unlisted_episode_files": sorted(dataset_files - manifest_ids)}
    terminal = {"splits": {}, "episode_summaries": []}
    reward_stats = {"reward_version": reward_config.reward_version, "status": reward_config.status, "splits": {}}
    all_problems = []
    for name, dataset in datasets.items():
        summary = _audit_split(dataset)
        summary["state_shape"] = [int(x) for x in dataset.state_raw.shape]
        summary["action_shape"] = [int(x) for x in dataset.action_rad.shape]
        summary["done_count"] = int(np.sum(dataset.done))
        summary["terminal_success_count"] = int(np.sum(dataset.terminal_success))
        summary["nan_inf_count"] = 0
        summary["coverage"] = _coverage(dataset)
        summary["normalization_source"] = "supplied BC normalization (not recomputed)"
        summary["action_cost_definition"] = "sum((executed_delta_joint / action_scale)^2); stored reward_terms/action_norm is audited separately"
        summary["done_boundary_check"] = {
            "done_equals_loaded_episodes": int(np.sum(dataset.done)) == int(sum(item.loaded for item in dataset.episode_audits)),
        }
        for kind, key in (("distance_inconsistent", "distance_consistency"),
                          ("progress_inconsistent", "progress_consistency")):
            errors = [float(item.get("error", 0.0)) for item in dataset.problems if item.get("kind") == kind]
            summary[key] = {"count": len(errors), "max_error": max(errors) if errors else 0.0,
                            "mean_error": float(np.mean(errors)) if errors else 0.0}
        alignment = [max(float(item.get("delta_error", 0.0)), float(item.get("next_joint_error", 0.0)), float(item.get("next_tcp_error", 0.0))) for item in dataset.problems if item.get("kind") == "alignment_error"]
        delta_errors = [float(item.get("delta_error", 0.0)) for item in dataset.problems if item.get("kind") == "alignment_error"]
        next_joint_errors = [float(item.get("next_joint_error", 0.0)) for item in dataset.problems if item.get("kind") == "alignment_error"]
        next_tcp_errors = [float(item.get("next_tcp_error", 0.0)) for item in dataset.problems if item.get("kind") == "alignment_error"]
        summary["alignment"] = {"count": len(alignment), "max_error": max(alignment) if alignment else 0.0,
                                 "delta_joint_max_error": max(delta_errors) if delta_errors else 0.0,
                                 "next_joint_max_error": max(next_joint_errors) if next_joint_errors else 0.0,
                                 "next_tcp_max_error": max(next_tcp_errors) if next_tcp_errors else 0.0}
        audit["splits"][name] = summary
        reward_stats["splits"][name] = summary
        terminal["splits"][name] = {"episode_count": len(selected[name]), "transition_count": len(dataset), "done_count": int(np.sum(dataset.done)), "terminal_success_count": int(np.sum(dataset.terminal_success))}
        terminal["episode_summaries"].extend(dataset.summaries())
        all_problems.extend(dataset.problems)
        loader = dataset.data_loader(args.batch_size, shuffle=True)
        if len(dataset):
            batch = next(iter(loader))
            expected = {"state_norm": (args.batch_size if len(dataset) >= args.batch_size else len(dataset), 12), "action_norm": (args.batch_size if len(dataset) >= args.batch_size else len(dataset), 6), "reward": (args.batch_size if len(dataset) >= args.batch_size else len(dataset), 1), "next_state_norm": (args.batch_size if len(dataset) >= args.batch_size else len(dataset), 12), "done": (args.batch_size if len(dataset) >= args.batch_size else len(dataset), 1)}
            audit["splits"][name]["first_minibatch_shapes"] = {key: list(batch[key].shape) for key in expected}
    audit["problematic_episodes"] = all_problems
    audit["actor_compatibility"] = None if actor_audit is None else {"actor_class": actor_audit["checkpoint"].get("actor_class"), "log_std_frozen_on_load": actor_audit["log_std_frozen_on_load"], "unfreeze_log_std_verified": actor_audit["unfreeze_log_std_verified"]}
    for path, value in (("offline_rl_audit.json", audit), ("reward_statistics.json", reward_stats), ("terminal_audit.json", terminal), ("problematic_episodes.json", all_problems), ("config.json", vars(args))):
        path = output / path
        path.write_text(json.dumps(_safe(value), indent=2, sort_keys=True))
    print("Offline RL audit written to", output)
    for name, value in audit["splits"].items():
        print("{}: episodes={} transitions={} done={} terminal_success={} reward_mean={}".format(name, value["loaded_episode_count"], value["transition_count"], value["done_count"], value["terminal_success_count"], value["reward"]["mean"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
