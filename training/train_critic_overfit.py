#!/usr/bin/env python3
"""Stage 3B Twin-Q/Bellman target small-dataset smoke test.

This is deliberately not a SAC trainer.  It only verifies Q networks,
Bellman targets, target updates and critic optimization on Train episodes.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

try:
    from .models import SACActor
    from .offline_rl.dataset import (OfflineRLTransitionDataset,
                                      audit_actor_checkpoint,
                                      load_split_manifest)
    from .offline_rl.networks import SACQNetwork, freeze_target_network, hard_update, soft_update
    from .offline_rl.reward import RewardConfig
    from .offline_rl.sac_utils import critic_predictions, sac_bellman_components, sac_bellman_target
except ImportError:
    from models import SACActor
    from offline_rl.dataset import OfflineRLTransitionDataset, audit_actor_checkpoint, load_split_manifest
    from offline_rl.networks import SACQNetwork, freeze_target_network, hard_update, soft_update
    from offline_rl.reward import RewardConfig
    from offline_rl.sac_utils import critic_predictions, sac_bellman_components, sac_bellman_target


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--actor-checkpoint", required=True)
    parser.add_argument("--reward-config", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--mode", choices=("frozen_target", "dynamic_bellman", "both"), default="both")
    parser.add_argument("--max-transitions", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _set_seed(seed):
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    value = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    device = torch.device(value)
    print("Requested device:", requested)
    print("Resolved device:", device)
    if device.type == "cuda":
        print("CUDA device:", torch.cuda.get_device_name(device))
    return device


def _load_config(path, args):
    value = {}
    if path:
        with open(str(Path(path).expanduser().resolve()), "r") as stream:
            value = yaml.safe_load(stream) or {}
    for key in ("gamma", "alpha", "learning_rate", "batch_size", "epochs", "tau", "seed"):
        if key in value and getattr(args, key) == build_parser().get_default(key):
            setattr(args, key, type(getattr(args, key))(value[key]))
    return value


def _batch_from_dataset(dataset, indices, device):
    rows = [dataset[int(index)] for index in indices]
    keys = ("state_norm", "action_norm", "reward", "next_state_norm", "done")
    result = {key: torch.stack([row[key] for row in rows], dim=0).to(device) for key in keys}
    for key, width in (("state_norm", 12), ("action_norm", 6), ("reward", 1), ("next_state_norm", 12), ("done", 1)):
        if result[key].shape != (len(rows), width):
            raise ValueError("{} shape is {} expected [{},{}]".format(key, tuple(result[key].shape), len(rows), width))
    return result


def _stats(value):
    value = value.detach().cpu().numpy().reshape(-1)
    return {"mean": float(np.mean(value)), "std": float(np.std(value)), "min": float(np.min(value)), "max": float(np.max(value))}


def _gradient_norm(network):
    values = [parameter.grad.detach().norm() for parameter in network.parameters() if parameter.grad is not None]
    return float(torch.stack(values).norm().detach().cpu()) if values else 0.0


def _make_networks(device):
    q1 = SACQNetwork().to(device)
    q2 = SACQNetwork().to(device)
    target_q1 = SACQNetwork().to(device)
    target_q2 = SACQNetwork().to(device)
    hard_update(target_q1, q1); hard_update(target_q2, q2)
    freeze_target_network(target_q1); freeze_target_network(target_q2)
    return q1, q2, target_q1, target_q2


def _train_mode(name, dataset, indices, actor, args, device, output):
    q1, q2, target_q1, target_q2 = _make_networks(device)
    optimizer = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.learning_rate)
    all_indices = list(indices)
    # Frozen targets are indexed in the fixed selected order; dynamic mode may
    # shuffle because each target is recomputed for the current batch.
    loader = DataLoader(Subset(dataset, all_indices), batch_size=args.batch_size, shuffle=(name == "dynamic_bellman"))
    first_batch = _batch_from_dataset(dataset, all_indices[:min(len(all_indices), args.batch_size)], device)
    if name == "frozen_target":
        target = sac_bellman_target(actor, target_q1, target_q2, first_batch["next_state_norm"], first_batch["reward"], first_batch["done"], args.gamma, args.alpha)
        frozen_target = target.detach().clone()
    else:
        frozen_target = None
    history = []
    initial = None
    last_diag = None
    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        for batch in loader:
            state = batch["state_norm"].to(device); action = batch["action_norm"].to(device)
            reward = batch["reward"].to(device); next_state = batch["next_state_norm"].to(device); done = batch["done"].to(device)
            if name == "frozen_target":
                # A fixed target is constructed for the fixed overfit batch.
                target = frozen_target
                if len(batch["state_norm"]) != len(target):
                    raise ValueError("frozen target mode requires batch_size >= selected transitions")
            else:
                components = sac_bellman_components(actor, target_q1, target_q2, next_state, reward, done, args.gamma, args.alpha)
                target = components["target"]
            result = critic_predictions(q1, q2, state, action, target)
            if initial is None:
                initial = {"q1": float(result["q1_loss"].detach()), "q2": float(result["q2_loss"].detach()), "combined": float(result["critic_loss"].detach())}
            optimizer.zero_grad(set_to_none=True); result["critic_loss"].backward()
            grad_norm = _gradient_norm(q1) + _gradient_norm(q2)
            optimizer.step()
            if name == "dynamic_bellman":
                soft_update(target_q1, q1, args.tau); soft_update(target_q2, q2, args.tau)
            with torch.no_grad():
                target_error = result["q1"].detach() - target.detach()
                target_rmse = torch.sqrt(torch.mean(torch.square(target_error)))
            epoch_losses.append((float(result["q1_loss"].detach()), float(result["q2_loss"].detach()), float(result["critic_loss"].detach())))
            last_diag = {"q1": _stats(result["q1"]), "q2": _stats(result["q2"]), "target": _stats(target), "reward": _stats(reward), "next_log_prob": _stats(components["next_log_prob"]) if name == "dynamic_bellman" else {}}
            last_diag.update({"gradient_norm": grad_norm, "target_mae": float(torch.mean(torch.abs(target_error)).cpu()), "target_rmse": float(target_rmse.cpu())})
        values = np.asarray(epoch_losses, dtype=np.float64)
        history.append({"epoch": epoch, "q1_loss": float(values[:, 0].mean()), "q2_loss": float(values[:, 1].mean()), "critic_loss": float(values[:, 2].mean()), "gradient_norm": last_diag["gradient_norm"]})
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0:
            print(name, "epoch", epoch, "critic_loss", history[-1]["critic_loss"])
    final = history[-1]
    metrics = {"mode": name, "initial": initial, "final": final, "loss_reduction_ratio": initial["combined"] / max(final["critic_loss"], 1e-12), "target_mae": last_diag["target_mae"], "target_rmse": last_diag["target_rmse"], "diagnostics": last_diag, "shape_audit": {"state": [len(all_indices), 12], "action": [len(all_indices), 6], "reward": [len(all_indices), 1], "next_state": [len(all_indices), 12], "done": [len(all_indices), 1]}}
    mode_dir = output / name; mode_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": q1.state_dict(), "network": "SACQNetwork", "mode": name}, mode_dir / "q1.pt")
    torch.save({"model_state_dict": q2.state_dict(), "network": "SACQNetwork", "mode": name}, mode_dir / "q2.pt")
    torch.save({"model_state_dict": target_q1.state_dict(), "network": "SACQNetwork", "mode": name}, mode_dir / "target_q1.pt")
    torch.save({"model_state_dict": target_q2.state_dict(), "network": "SACQNetwork", "mode": name}, mode_dir / "target_q2.pt")
    (mode_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    with (mode_dir / "loss.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=history[0].keys()); writer.writeheader(); writer.writerows(history)
    return metrics


def main(argv=None):
    args = build_parser().parse_args(argv); _load_config(args.config, args); _set_seed(args.seed)
    device = resolve_device(args.device)
    manifest = load_split_manifest(args.split_manifest, args.dataset_dir)
    reward_config = RewardConfig.load(args.reward_config)
    dataset = OfflineRLTransitionDataset(args.dataset_dir, manifest["train_episode_ids"], args.normalization, reward_config)
    preferred_ids = {item.episode_id for item in dataset.episode_audits if item.loaded and item.reached_pregrasp}
    preferred = [index for index, eid in enumerate(dataset.episode_id_array.tolist()) if eid in preferred_ids]
    source_indices = preferred if preferred else list(range(len(dataset)))
    if args.max_transitions > 0:
        selected = source_indices[:min(args.max_transitions, len(source_indices))]
    else:
        selected = source_indices
    if not selected:
        raise RuntimeError("no Train transitions available")
    if len(selected) > args.batch_size and args.mode == "frozen_target":
        raise ValueError("frozen_target mode requires batch_size >= max-transitions so one fixed target covers the overfit set")
    actor_info = audit_actor_checkpoint(args.actor_checkpoint, manifest)
    actor = actor_info["actor"].to(device); actor.unfreeze_log_std(); actor.eval()
    output = Path(args.output_dir or (Path(__file__).resolve().parent / "runs" / ("critic_overfit_" + datetime.now().strftime("%Y%m%d_%H%M%S")))).expanduser().resolve(); output.mkdir(parents=True, exist_ok=False)
    config = vars(args).copy(); config.update({"resolved_device": str(device), "dataset_schema_version": "apple_rl_v1.2_sac_goal", "actor_optimizer": False, "stage": "3B_critic_only"}); (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    results = {}
    for mode in (("frozen_target", "dynamic_bellman") if args.mode == "both" else (args.mode,)):
        results[mode] = _train_mode(mode, dataset, selected, actor, args, device, output)
    (output / "metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    print("Output:", output); print("Actor optimizer: not created"); print("Stage 3B critic-only smoke complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
