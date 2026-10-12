#!/usr/bin/env python3
"""Train Offline SAC with BC regularization on fixed expert transitions.

This entry point is hardware-independent. It does not collect experience,
import ROS, or interact with a robot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import warnings
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
    from .offline_rl.networks import (SACQNetwork, freeze_target_network,
                                      hard_update, soft_update)
    from .offline_rl.reward import RewardConfig
    from .offline_rl.sac_training import (actor_update_step,
                                          critic_update_step)
except ImportError:
    from models import SACActor
    from offline_rl.dataset import OfflineRLTransitionDataset, audit_actor_checkpoint, load_split_manifest
    from offline_rl.networks import SACQNetwork, freeze_target_network, hard_update, soft_update
    from offline_rl.reward import RewardConfig
    from offline_rl.sac_training import actor_update_step, critic_update_step


DEFAULTS = {
    "gamma": 0.99, "alpha": 0.2, "tau": 0.005,
    "critic_learning_rate": 0.001, "actor_learning_rate": 0.0003,
    "batch_size": 64, "epochs": 200, "bc_weight": 1.0, "seed": 42,
    "device": "auto", "patience": 30, "validation_interval": 1,
    "max_q_abs": 1000.0, "max_policy_saturation_fraction": 0.50,
    "max_validation_bc_ratio": 1.50, "max_validation_bc_increase": 0.02,
    "ood_q_gap_increase": 0.05, "ood_action_deviation_increase": 0.01,
    "gradient_clip_norm": 0.0,
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--normalization", required=True)
    parser.add_argument("--actor-checkpoint", required=True)
    parser.add_argument("--reward-config", default=str(Path(__file__).parent / "configs/offline_rl_reward_v0.yaml"))
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs/offline_sac_bc_v0.yaml"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--dataset-max-transitions", type=int, default=0,
                        help="Restrict Train split to its first N transitions for pipeline smoke only.")
    parser.add_argument("--max-transitions", type=int, default=0,
                        help="Alias for --dataset-max-transitions.")
    for key, default in DEFAULTS.items():
        arg = "--" + key.replace("_", "-")
        typ = type(default)
        parser.add_argument(arg, dest=key, type=typ, default=None)
    return parser


def _load_config(args):
    with open(Path(args.config).expanduser(), "r") as stream:
        config = yaml.safe_load(stream) or {}
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    merged = dict(DEFAULTS)
    merged.update(config)
    for key in DEFAULTS:
        override = getattr(args, key)
        if override is not None:
            merged[key] = override
    for key in ("gamma", "tau"):
        if not 0 <= float(merged[key]) <= 1:
            raise ValueError("{} must be in [0,1]".format(key))
    for key in ("alpha", "bc_weight", "max_q_abs", "max_policy_saturation_fraction",
                "max_validation_bc_ratio", "max_validation_bc_increase", "ood_q_gap_increase",
                "ood_action_deviation_increase", "gradient_clip_norm"):
        if float(merged[key]) < 0:
            raise ValueError("{} must be non-negative".format(key))
    if float(merged["max_policy_saturation_fraction"]) > 1.0:
        raise ValueError("max_policy_saturation_fraction must be at most 1")
    if min(int(merged["batch_size"]), int(merged["epochs"]), int(merged["validation_interval"])) <= 0:
        raise ValueError("batch_size, epochs, validation_interval must be positive")
    if int(merged["patience"]) < 0:
        raise ValueError("patience must be non-negative")
    return merged


def _set_seed(seed):
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)


def _device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    resolved = "cuda" if requested == "auto" and torch.cuda.is_available() else requested
    return torch.device(resolved)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _batch(dataset, indices, device):
    rows = [dataset[int(i)] for i in indices]
    names = ("state_norm", "action_norm", "reward", "next_state_norm", "done")
    widths = (12, 6, 1, 12, 1)
    values = {name: torch.stack([row[name] for row in rows]).to(device) for name in names}
    for name, width in zip(names, widths):
        if values[name].shape != (len(rows), width):
            raise ValueError("{} must have shape [B,{}]".format(name, width))
        if not torch.isfinite(values[name]).all():
            raise FloatingPointError("{} contains NaN/Inf".format(name))
    return values


def _array_stats(value):
    value = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(value)):
        raise FloatingPointError("evaluation produced NaN/Inf")
    return {"mean": float(np.mean(value)), "std": float(np.std(value)),
            "min": float(np.min(value)), "max": float(np.max(value))}


def _evaluate(actor, q1, q2, target_q1, target_q2, dataset, indices,
              config, device, action_scale):
    """Full split evaluation without optimizer calls or parameter mutation."""
    actor.eval(); q1.eval(); q2.eval(); target_q1.eval(); target_q2.eval()
    action_scale_tensor = torch.as_tensor(action_scale, dtype=torch.float32, device=device)
    if action_scale_tensor.shape != (6,) or not torch.isfinite(action_scale_tensor).all():
        raise ValueError("action_scale must be finite with shape [6]")
    action_scale_np = action_scale_tensor.cpu().numpy()
    parts = {key: [] for key in ("bc", "physical_error", "expert", "mean_action",
                                "policy_action", "q_data", "q_policy", "bellman_mse",
                                "q1", "q2", "target")}
    with torch.no_grad():
        for start in range(0, len(indices), int(config["batch_size"])):
            batch = _batch(dataset, indices[start:start + int(config["batch_size"])], device)
            mean_action = actor.deterministic_action(batch["state_norm"])
            policy_action, log_prob, _ = actor.sample(batch["state_norm"])
            qd = torch.minimum(q1(batch["state_norm"], batch["action_norm"]),
                               q2(batch["state_norm"], batch["action_norm"]))
            qp1 = q1(batch["state_norm"], policy_action)
            qp2 = q2(batch["state_norm"], policy_action)
            qp = torch.minimum(qp1, qp2)
            next_action, next_log_prob, _ = actor.sample(batch["next_state_norm"])
            tq = torch.minimum(target_q1(batch["next_state_norm"], next_action),
                               target_q2(batch["next_state_norm"], next_action))
            target = batch["reward"] + float(config["gamma"]) * (1 - batch["done"]) * (
                tq - float(config["alpha"]) * next_log_prob)
            pred1 = q1(batch["state_norm"], batch["action_norm"])
            pred2 = q2(batch["state_norm"], batch["action_norm"])
            if not all(torch.isfinite(x).all() for x in (mean_action, policy_action, log_prob, qd, qp, target, pred1, pred2)):
                raise FloatingPointError("validation produced NaN/Inf")
            expert = batch["action_norm"]
            parts["bc"].append((mean_action - expert).pow(2).cpu().numpy())
            scale = action_scale_tensor.to(dtype=mean_action.dtype)
            parts["physical_error"].append(((mean_action - expert) * scale).cpu().numpy())
            parts["expert"].append(expert.cpu().numpy())
            parts["mean_action"].append(mean_action.cpu().numpy())
            parts["policy_action"].append(policy_action.cpu().numpy())
            parts["q_data"].append(qd.cpu().numpy())
            parts["q_policy"].append(qp.cpu().numpy())
            parts["bellman_mse"].append(((pred1 - target).pow(2) + (pred2 - target).pow(2)).cpu().numpy() * 0.5)
            parts["q1"].append(pred1.cpu().numpy()); parts["q2"].append(pred2.cpu().numpy())
            parts["target"].append(target.cpu().numpy())
    joined = {key: np.concatenate(value, axis=0) for key, value in parts.items()}
    error_rad = joined["physical_error"]
    expert_rad = joined["expert"] * action_scale_np
    policy = joined["policy_action"]
    q_data = joined["q_data"].reshape(-1); q_policy = joined["q_policy"].reshape(-1)
    per_joint = np.mean(np.abs(error_rad), axis=0)
    denom = np.abs(expert_rad) > 1e-8
    sign_match = (np.sign(joined["mean_action"] * action_scale_np) == np.sign(expert_rad)) | ~denom
    return {
        "actor_bc_mse": float(np.mean(joined["bc"])),
        "actor_physical_mae_rad": float(np.mean(np.abs(error_rad))),
        "actor_physical_rmse_rad": float(np.sqrt(np.mean(error_rad ** 2))),
        "per_joint_mae_rad": per_joint.tolist(),
        "sign_agreement": float(np.sum(sign_match & denom) / max(1, np.sum(denom))),
        "mean_abs_policy_expert_norm": float(np.mean(np.abs(policy - joined["expert"]))),
        "policy_action_saturation": {
            "gt_0.90": float(np.mean(np.abs(policy) > .90)),
            "gt_0.95": float(np.mean(np.abs(policy) > .95)),
            "gt_0.99": float(np.mean(np.abs(policy) > .99)),
        },
        "critic_bellman_mse": float(np.mean(joined["bellman_mse"])),
        "q_data": _array_stats(q_data), "q_policy": _array_stats(q_policy),
        "q_policy_minus_q_data_mean": float(np.mean(q_policy - q_data)),
        "q1": _array_stats(joined["q1"]), "q2": _array_stats(joined["q2"]),
        "target_q": _array_stats(joined["target"]),
        "q_abs_max": float(max(np.max(np.abs(joined[k])) for k in ("q1", "q2", "target", "q_data", "q_policy"))),
    }


def _snapshot(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _save(path, module, metadata):
    torch.save({"model_state_dict": _snapshot(module), **metadata}, str(path))


def _write_json(path, value):
    with open(path, "w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _grad_norm(module):
    grads = [p.grad.detach().norm() for p in module.parameters() if p.grad is not None]
    value = torch.stack(grads).norm() if grads else torch.zeros(())
    if not torch.isfinite(value):
        raise FloatingPointError("gradient norm is NaN/Inf")
    return float(value.detach().cpu())


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = _load_config(args)
    _set_seed(config["seed"])
    device = _device(config["device"])
    limit = int(args.dataset_max_transitions or args.max_transitions)
    if limit < 0:
        raise ValueError("max transitions must be non-negative")

    manifest_path = Path(args.split_manifest).expanduser().resolve()
    norm_path = Path(args.normalization).expanduser().resolve()
    actor_path = Path(args.actor_checkpoint).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    manifest = load_split_manifest(manifest_path, dataset_dir)
    audit = audit_actor_checkpoint(actor_path, manifest)
    actor = audit["actor"].to(device)
    actor.unfreeze_log_std()
    if not all(p.requires_grad for p in actor.log_std_head.parameters()):
        raise RuntimeError("Actor log_std parameters did not unfreeze")

    reward_config = RewardConfig.load(args.reward_config)
    train_data = OfflineRLTransitionDataset(dataset_dir, manifest["train_episode_ids"], norm_path, reward_config)
    val_data = OfflineRLTransitionDataset(dataset_dir, manifest["validation_episode_ids"], norm_path, reward_config)
    if not len(train_data) or not len(val_data):
        raise ValueError("Train and Validation must both contain transitions")
    train_indices = list(range(min(len(train_data), limit) if limit else len(train_data)))
    val_indices = list(range(len(val_data)))
    if limit:
        print("SMOKE MODE: using {} of {} Train transitions; full Validation split retained".format(len(train_indices), len(train_data)))

    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(__file__).parent / "runs" / ("offline_sac_bc_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    q1, q2 = SACQNetwork().to(device), SACQNetwork().to(device)
    target_q1, target_q2 = SACQNetwork().to(device), SACQNetwork().to(device)
    hard_update(target_q1, q1); hard_update(target_q2, q2)
    freeze_target_network(target_q1); freeze_target_network(target_q2)
    actor_optimizer = torch.optim.Adam((p for p in actor.parameters() if p.requires_grad), lr=float(config["actor_learning_rate"]))
    critic_optimizer = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=float(config["critic_learning_rate"]))
    action_scale = train_data.normalization["action_scale"].copy()

    # Persist references/copies to the exact upstream artifacts; no Test data is instantiated.
    (output / "normalization.npz").write_bytes(norm_path.read_bytes())
    (output / "split_manifest.json").write_bytes(manifest_path.read_bytes())
    metadata = {
        "actor_class": "SACActor", "architecture_version": SACActor.architecture_version,
        "state_dim": 12, "action_dim": 6, "hidden_dim": 256,
        "dataset_schema_version": manifest["dataset_schema_version"],
        "split_manifest_hash": manifest["manifest_hash"],
        "source_actor_checkpoint": str(actor_path),
    }
    baseline = _evaluate(actor, q1, q2, target_q1, target_q2, val_data, val_indices, config, device, action_scale)
    _write_json(output / "bc_initial_validation.json", baseline)
    if baseline["q_abs_max"] > float(config["max_q_abs"]):
        raise FloatingPointError("initial Q magnitude exceeds max_q_abs")

    history = []
    best_score = float("inf")
    best_epoch = 0
    stale = 0
    warning_active = False
    previous_q_gap = None
    previous_action_deviation = None
    train_loader = DataLoader(Subset(train_data, train_indices), batch_size=int(config["batch_size"]), shuffle=True)
    for epoch in range(1, int(config["epochs"]) + 1):
        actor.train(); q1.train(); q2.train()
        epoch_values = {key: [] for key in ("critic_loss", "q1_grad_norm", "q2_grad_norm", "actor_loss", "sac_actor_loss", "bc_loss", "actor_grad_norm")}
        for batch in train_loader:
            batch = {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
            c = critic_update_step(actor, q1, q2, target_q1, target_q2, batch,
                                  critic_optimizer, config["gamma"], config["alpha"])
            a = actor_update_step(actor, q1, q2, batch, actor_optimizer, config["alpha"], config["bc_weight"], config["gradient_clip_norm"])
            soft_update(target_q1, q1, config["tau"]); soft_update(target_q2, q2, config["tau"])
            for key, value in (("critic_loss", c["critic_loss"]), ("q1_grad_norm", c["q1_grad_norm"]),
                               ("q2_grad_norm", c["q2_grad_norm"]), ("actor_loss", a["actor_loss"]),
                               ("sac_actor_loss", a["sac_actor_loss"]), ("bc_loss", a["bc_loss"]),
                               ("actor_grad_norm", a["actor_grad_norm"])):
                scalar = float(value.detach().cpu())
                if not np.isfinite(scalar):
                    raise FloatingPointError("{} is NaN/Inf at epoch {}".format(key, epoch))
                epoch_values[key].append(scalar)

        record = {"epoch": epoch}
        record.update({"train_" + key: float(np.mean(values)) for key, values in epoch_values.items()})
        if epoch % int(config["validation_interval"]) == 0 or epoch == int(config["epochs"]):
            validation = _evaluate(actor, q1, q2, target_q1, target_q2, val_data, val_indices, config, device, action_scale)
            record.update({"validation_" + key: value for key, value in validation.items() if not isinstance(value, dict)})
            record["validation_per_joint_mae_rad"] = validation["per_joint_mae_rad"]
            record["validation_saturation"] = validation["policy_action_saturation"]
            record["validation_q_data"] = validation["q_data"]
            record["validation_q_policy"] = validation["q_policy"]
            record["validation_q1"] = validation["q1"]
            record["validation_q2"] = validation["q2"]
            record["validation_target_q"] = validation["target_q"]
            if validation["q_abs_max"] > float(config["max_q_abs"]):
                raise FloatingPointError("Q explosion: abs max {:.6g} > configured {:.6g}".format(validation["q_abs_max"], config["max_q_abs"]))
            if validation["policy_action_saturation"]["gt_0.99"] > float(config["max_policy_saturation_fraction"]):
                raise FloatingPointError("policy saturation guard exceeded")
            bc_safe = validation["actor_bc_mse"] <= max(
                baseline["actor_bc_mse"] * float(config["max_validation_bc_ratio"]),
                baseline["actor_bc_mse"] + float(config["max_validation_bc_increase"]))
            safe = bc_safe and validation["q_abs_max"] <= float(config["max_q_abs"])
            score = validation["actor_bc_mse"] + 1e-6 * validation["critic_bellman_mse"]
            if safe and score < best_score:
                best_score, best_epoch, stale = score, epoch, 0
                for name, network in (("actor", actor), ("q1", q1), ("q2", q2), ("target_q1", target_q1), ("target_q2", target_q2)):
                    _save(output / (name + "_best.pt"), network, {**metadata, "epoch": epoch, "validation_score": score})
            else:
                stale += 1
            delta_q = validation["q_policy_minus_q_data_mean"]
            deviation = validation["mean_abs_policy_expert_norm"]
            if (previous_q_gap is not None and previous_action_deviation is not None and
                    delta_q > previous_q_gap + float(config["ood_q_gap_increase"]) and
                    deviation > previous_action_deviation + float(config["ood_action_deviation_increase"])):
                warning_active = True
                warnings.warn("possible offline Q extrapolation / OOD exploitation: Q_policy exceeds Q_data while policy-expert deviation has grown", RuntimeWarning)
            previous_q_gap = delta_q
            previous_action_deviation = deviation
            record["validation_safe"] = bool(safe)
            record["possible_ood_exploitation"] = bool(warning_active)
            if epoch == 1 or epoch % 10 == 0:
                print("epoch {} critic={:.6g} actor={:.6g} BC={:.6g} val_BC={:.6g} Qdata={:.4g} Qpolicy={:.4g}".format(epoch, record["train_critic_loss"], record["train_actor_loss"], record["train_bc_loss"], validation["actor_bc_mse"], validation["q_data"]["mean"], validation["q_policy"]["mean"]))
            if config["patience"] and stale >= int(config["patience"]):
                print("Early stopping at epoch {} (patience={})".format(epoch, config["patience"]))
        history.append(record)
        for name, network in (("actor", actor), ("q1", q1), ("q2", q2), ("target_q1", target_q1), ("target_q2", target_q2)):
            _save(output / (name + "_last.pt"), network, {**metadata, "epoch": epoch})
        if config["patience"] and stale >= int(config["patience"]):
            break

    if not best_epoch:
        raise RuntimeError("no validation-safe checkpoint was produced")
    fieldnames = sorted({key for row in history for key in row if not isinstance(row[key], (dict, list))})
    with open(output / "history.csv", "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames); writer.writeheader()
        for row in history:
            writer.writerow({key: value for key, value in row.items() if key in fieldnames})
    final_val = _evaluate(actor, q1, q2, target_q1, target_q2, val_data, val_indices, config, device, action_scale)
    _write_json(output / "metrics.json", {
        "status": "completed", "best_epoch": best_epoch, "best_validation_score": best_score,
        "bc_initial_validation": baseline, "last_validation": final_val,
        "train_transition_count": len(train_indices), "train_split_transition_count": len(train_data),
        "validation_transition_count": len(val_data), "test_split_loaded": False,
        "test_split_used_for_tuning": False, "actor_drift_warning_seen": warning_active,
    })
    config_record = dict(config, resolved_device=str(device), mode="offline_sac_bc",
                         smoke_mode=bool(limit), max_transitions=limit,
                         source_paths={"dataset": str(dataset_dir), "split_manifest": str(manifest_path), "normalization": str(norm_path), "actor_checkpoint": str(actor_path)},
                         source_hashes={"split_manifest_sha256": _sha256(manifest_path), "normalization_sha256": _sha256(norm_path), "actor_checkpoint_sha256": _sha256(actor_path)},
                         split_manifest_hash=manifest["manifest_hash"], best_epoch=best_epoch)
    _write_json(output / "config.json", config_record)
    print("Offline SAC+BC complete: {} (best epoch {})".format(output, best_epoch))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
