#!/usr/bin/env python3
"""Episode-level full BC training for the shared future SACActor.

This script is deliberately offline and hardware-independent.  It never
imports ROS and never executes a robot action.  The split is made before
normalization so validation and test episodes cannot influence statistics.
"""

import argparse
import csv
import hashlib
import json
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import NormalizationStats, load_bc_transitions
from models import SACActor


SCHEMA = "apple_rl_v1.2_sac_goal"
DEG_PER_RAD = 180.0 / np.pi


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--split-manifest", default="")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--alignment-tolerance", type=float, default=1e-5)
    return parser


def _set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(requested):
    requested = str(requested)
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested, but torch.cuda.is_available() is false"
        )
    if requested == "auto":
        resolved = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved = requested
    device = torch.device(resolved)
    print("Requested device:", requested)
    print("Resolved device:", device)
    if device.type == "cuda":
        print("CUDA device:", torch.cuda.get_device_name(device))
    return device


def episode_id_from_path(path):
    """Use the stable filename stem as the manifest episode identifier."""
    return Path(path).stem


def _validate_ratios(train_ratio, val_ratio, test_ratio):
    values = (float(train_ratio), float(val_ratio), float(test_ratio))
    if any(value <= 0.0 for value in values):
        raise ValueError("train/val/test ratios must be positive")
    if not np.isclose(sum(values), 1.0, atol=1e-8):
        raise ValueError("train/val/test ratios must sum to 1.0")
    return values


def create_episode_split(episode_ids, seed=42, train_ratio=0.70,
                         val_ratio=0.15, test_ratio=0.15):
    """Create a deterministic, disjoint episode-level split."""
    train_ratio, val_ratio, test_ratio = _validate_ratios(
        train_ratio, val_ratio, test_ratio
    )
    unique = sorted(set(str(item) for item in episode_ids))
    if len(unique) < 3:
        raise ValueError("at least three eligible episodes are required")
    shuffled = list(unique)
    random.Random(int(seed)).shuffle(shuffled)
    train_count = max(1, int(np.floor(len(shuffled) * train_ratio)))
    val_count = max(1, int(np.floor(len(shuffled) * val_ratio)))
    if train_count + val_count >= len(shuffled):
        val_count = max(1, len(shuffled) - train_count - 1)
    test_count = len(shuffled) - train_count - val_count
    if test_count < 1:
        train_count -= 1
        test_count = 1
    manifest = {
        "manifest_version": 1,
        "dataset_schema_version": SCHEMA,
        "seed": int(seed),
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "train_episode_ids": sorted(shuffled[:train_count]),
        "validation_episode_ids": sorted(
            shuffled[train_count:train_count + val_count]
        ),
        "test_episode_ids": sorted(shuffled[train_count + val_count:]),
    }
    _validate_manifest_sets(manifest, set(unique))
    manifest["manifest_hash"] = manifest_hash(manifest)
    return manifest


def _manifest_sets(manifest):
    return {
        "train": set(str(item) for item in manifest.get("train_episode_ids", [])),
        "validation": set(
            str(item) for item in manifest.get("validation_episode_ids", [])
        ),
        "test": set(str(item) for item in manifest.get("test_episode_ids", [])),
    }


def _validate_manifest_sets(manifest, eligible_ids):
    groups = _manifest_sets(manifest)
    names = tuple(groups)
    for index, first in enumerate(names):
        for second in names[index + 1:]:
            overlap = groups[first] & groups[second]
            if overlap:
                raise ValueError(
                    "episode split overlap between {} and {}: {}".format(
                        first, second, sorted(overlap)
                    )
                )
    listed = groups["train"] | groups["validation"] | groups["test"]
    missing = set(eligible_ids) - listed
    unknown = listed - set(eligible_ids)
    if missing:
        raise ValueError("manifest omits eligible episodes: {}".format(sorted(missing)))
    if unknown:
        raise ValueError(
            "manifest references episodes not present in dataset: {}".format(
                sorted(unknown)
            )
        )
    if not groups["train"] or not groups["validation"] or not groups["test"]:
        raise ValueError("train, validation, and test splits must all be non-empty")


def manifest_hash(manifest):
    payload = {
        key: manifest[key]
        for key in (
            "dataset_schema_version", "seed", "train_ratio", "val_ratio",
            "test_ratio", "train_episode_ids", "validation_episode_ids",
            "test_episode_ids",
        )
        if key in manifest
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def load_or_create_manifest(path, eligible_ids, seed, train_ratio, val_ratio,
                            test_ratio):
    if path:
        manifest_path = Path(path).expanduser().resolve()
        with open(str(manifest_path), "r") as stream:
            manifest = json.load(stream)
        _validate_manifest_sets(manifest, set(eligible_ids))
        expected_hash = manifest_hash(manifest)
        if manifest.get("manifest_hash", expected_hash) != expected_hash:
            raise ValueError("split manifest hash does not match its contents")
        manifest["manifest_hash"] = expected_hash
        return manifest
    return create_episode_split(
        eligible_ids, seed=seed, train_ratio=train_ratio,
        val_ratio=val_ratio, test_ratio=test_ratio,
    )


def _transition_split_indices(transitions, manifest):
    groups = _manifest_sets(manifest)
    result = {}
    episode_ids = np.asarray(
        [episode_id_from_path(path) for path in transitions.episode_paths],
        dtype=object,
    )
    for name, ids in groups.items():
        result[name] = np.flatnonzero(np.isin(episode_ids, list(ids)))
    if set(np.concatenate(list(result.values())).tolist()) != set(range(len(episode_ids))):
        raise ValueError("transition split does not cover every loaded transition")
    return result


def _split_values(values, indices):
    return np.asarray(values)[indices]


def _finite_metrics(metrics):
    def walk(value):
        if isinstance(value, dict):
            return all(walk(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return all(walk(item) for item in value)
        if isinstance(value, (float, int, np.floating, np.integer)):
            return bool(np.isfinite(value))
        return True
    return walk(metrics)


def _error_metrics(prediction, target, action_scale):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    error = prediction - target
    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(np.square(error), axis=0))
    max_abs = np.max(np.abs(error), axis=0)
    norm_target = target / np.asarray(action_scale, dtype=np.float64)
    norm_prediction = prediction / np.asarray(action_scale, dtype=np.float64)
    denominator = np.abs(norm_target) > 0.05
    signs = np.sign(norm_prediction) == np.sign(norm_target)
    sign_rates = []
    sign_counts = []
    for joint in range(target.shape[1]):
        count = int(np.sum(denominator[:, joint]))
        sign_counts.append(count)
        sign_rates.append(
            float(np.mean(signs[denominator[:, joint], joint]))
            if count else 0.0
        )
    return {
        "normalized_mse": float(np.mean(np.square(
            (prediction - target) / np.asarray(action_scale)
        ))),
        "physical_mae_rad": float(np.mean(mae)),
        "physical_rmse_rad": float(np.sqrt(np.mean(np.square(error)))),
        "mae_rad_per_joint": mae.tolist(),
        "rmse_rad_per_joint": rmse.tolist(),
        "max_abs_rad_per_joint": max_abs.tolist(),
        "mae_degree_per_joint": (mae * DEG_PER_RAD).tolist(),
        "rmse_degree_per_joint": (rmse * DEG_PER_RAD).tolist(),
        "max_abs_degree_per_joint": (max_abs * DEG_PER_RAD).tolist(),
        "sign_agreement_per_joint": sign_rates,
        "sign_count_per_joint": sign_counts,
    }


def _baseline_metrics(target, action_scale, train_actions):
    zeros = np.zeros_like(target)
    mean_action = np.mean(train_actions, axis=0)[None, :]
    mean_prediction = np.repeat(mean_action, len(target), axis=0)
    return {
        "zero_action": _error_metrics(zeros, target, action_scale),
        "train_mean_action": _error_metrics(
            mean_prediction, target, action_scale
        ),
    }


def _saturation_metrics(actions, action_scale):
    normalized = np.abs(np.asarray(actions) / action_scale)
    return {
        "over_0_90_count": np.sum(normalized > 0.90, axis=0).astype(int).tolist(),
        "over_0_90_ratio": np.mean(normalized > 0.90, axis=0).tolist(),
        "over_0_95_count": np.sum(normalized > 0.95, axis=0).astype(int).tolist(),
        "over_0_95_ratio": np.mean(normalized > 0.95, axis=0).tolist(),
    }


def _action_audit(actions, action_scale):
    actions = np.asarray(actions, dtype=np.float64)
    abs_actions = np.abs(actions)
    normalized = abs_actions / action_scale
    overflow = normalized > 1.0 + 1e-6
    return {
        "min_rad_per_joint": np.min(actions, axis=0).tolist(),
        "max_rad_per_joint": np.max(actions, axis=0).tolist(),
        "mean_rad_per_joint": np.mean(actions, axis=0).tolist(),
        "std_rad_per_joint": np.std(actions, axis=0).tolist(),
        "max_abs_rad_per_joint": np.max(abs_actions, axis=0).tolist(),
        "p95_abs_rad_per_joint": np.percentile(abs_actions, 95, axis=0).tolist(),
        "p99_abs_rad_per_joint": np.percentile(abs_actions, 99, axis=0).tolist(),
        "overflow_count_per_joint": np.sum(overflow, axis=0).astype(int).tolist(),
        "overflow_max_normalized_per_joint": np.max(normalized, axis=0).tolist(),
    }


def _coverage_metrics(transitions, indices):
    indices = np.asarray(indices, dtype=np.int64)
    states = transitions.states[indices]
    # State layout is joint[0:6], tcp_xyz[6:9], relative_goal[9:12].
    goals = states[:, 6:9] + states[:, 9:12]
    initial = []
    episode_ids = set(episode_id_from_path(transitions.episode_paths[i])
                      for i in indices)
    for status in transitions.statuses:
        if not status.loaded or episode_id_from_path(status.path) not in episode_ids:
            continue
        try:
            with h5py.File(status.path, "r") as episode:
                value = np.asarray(
                    episode["episode_meta/initial_joint_pos"][:], dtype=np.float64
                )
                if value.shape == (6,) and np.all(np.isfinite(value)):
                    initial.append(value)
        except (OSError, KeyError, ValueError):
            continue

    def range_dict(value):
        value = np.asarray(value, dtype=np.float64)
        return {
            "min": np.min(value, axis=0).tolist(),
            "max": np.max(value, axis=0).tolist(),
        }

    return {
        "initial_joint_pos": range_dict(np.asarray(initial)) if initial else None,
        "pregrasp_xyz": range_dict(goals),
        "relative_goal_xyz": range_dict(states[:, 9:12]),
    }


def _tensor_loader(states, actions, batch_size, shuffle):
    tensors = TensorDataset(
        torch.from_numpy(np.asarray(states, dtype=np.float32)),
        torch.from_numpy(np.asarray(actions, dtype=np.float32)),
    )
    return DataLoader(
        tensors, batch_size=min(int(batch_size), len(tensors)), shuffle=shuffle
    )


def _evaluate_actor(actor, states, targets, normalization, device):
    actor.eval()
    with torch.no_grad():
        state_tensor = torch.from_numpy(states.astype(np.float32)).to(device)
        prediction_norm = actor.deterministic_action(state_tensor).cpu().numpy()
    prediction_physical = normalization.denormalize_actions(prediction_norm)
    target_physical = np.asarray(targets, dtype=np.float32)
    metrics = _error_metrics(
        prediction_physical, target_physical, normalization.action_scale
    )
    metrics["saturation"] = _saturation_metrics(
        target_physical, normalization.action_scale
    )
    metrics["action_audit"] = _action_audit(
        target_physical, normalization.action_scale
    )
    return metrics, prediction_physical


def _save_checkpoint(path, actor, epoch, validation_loss, normalization, manifest):
    state_dict = {
        key: value.detach().cpu().clone()
        for key, value in actor.state_dict().items()
    }
    torch.save({
        "model_state_dict": state_dict,
        "actor_state_dict": state_dict,
        "actor_class": "SACActor",
        "architecture_version": SACActor.architecture_version,
        "state_dim": 12,
        "action_dim": 6,
        "hidden_dim": 256,
        "dataset_schema_version": SCHEMA,
        "best_epoch": int(epoch),
        "validation_loss": float(validation_loss),
        "log_std_frozen": True,
        "normalization_source": "train_only",
        "split_manifest_hash": manifest["manifest_hash"],
    }, str(path))


def _output_directory(value):
    if value:
        output = Path(value).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = Path(__file__).resolve().parent / "runs" / (
            "bc_pretrain_full_" + stamp
        )
    output.mkdir(parents=True, exist_ok=False)
    return output


def main(argv=None):
    args = build_parser().parse_args(argv)
    _set_seed(args.seed)
    try:
        device = resolve_device(args.device)
    except RuntimeError as exc:
        raise SystemExit("ERROR: {}".format(exc))
    if args.batch_size <= 0 or args.epochs <= 0 or args.learning_rate <= 0:
        raise SystemExit("ERROR: epochs, batch-size and learning-rate must be positive")
    if args.patience < 0:
        raise SystemExit("ERROR: patience must be non-negative")

    transitions = load_bc_transitions(
        args.dataset_dir,
        alignment_tolerance=args.alignment_tolerance,
        strict_alignment=True,
    )
    eligible_ids = sorted(set(
        episode_id_from_path(status.path)
        for status in transitions.statuses if status.loaded
    ))
    try:
        manifest = load_or_create_manifest(
            args.split_manifest, eligible_ids, args.seed, args.train_ratio,
            args.val_ratio, args.test_ratio,
        )
        split_indices = _transition_split_indices(transitions, manifest)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit("ERROR: split manifest: {}".format(exc))

    train_idx = split_indices["train"]
    val_idx = split_indices["validation"]
    test_idx = split_indices["test"]
    normalization = NormalizationStats.fit(
        transitions.states[train_idx], transitions.actions[train_idx]
    )
    action_scale = normalization.action_scale.astype(np.float64)
    normalized_actions = normalization.normalize_actions(transitions.actions)
    for name, indices in split_indices.items():
        values = normalized_actions[indices]
        overflow = np.abs(values) > 1.0 + 1e-6
        print("{} action overflow count:".format(name),
              np.sum(overflow, axis=0).astype(int).tolist())
    train_action_audit = _action_audit(
        transitions.actions[train_idx], action_scale
    )
    train_saturation = _saturation_metrics(
        transitions.actions[train_idx], action_scale
    )
    print("Train action audit:", json.dumps(train_action_audit, sort_keys=True))
    print("Action scale (train only):", action_scale.tolist())
    train_coverage = _coverage_metrics(transitions, train_idx)
    validation_coverage = _coverage_metrics(transitions, val_idx)
    test_coverage = _coverage_metrics(transitions, test_idx)
    print("Train coverage:", json.dumps(train_coverage, sort_keys=True))
    print("Validation coverage:", json.dumps(validation_coverage, sort_keys=True))
    print("Test coverage:", json.dumps(test_coverage, sort_keys=True))

    normalized_states = normalization.normalize_states(transitions.states)
    train_states, val_states, test_states = (
        normalized_states[train_idx], normalized_states[val_idx], normalized_states[test_idx]
    )
    train_actions, val_actions, test_actions = (
        normalized_actions[train_idx], normalized_actions[val_idx], normalized_actions[test_idx]
    )
    train_loader = _tensor_loader(
        train_states, train_actions, args.batch_size, shuffle=True
    )

    actor = SACActor().to(device)
    actor.freeze_log_std()
    trainable = [parameter for parameter in actor.parameters()
                 if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []
    output = _output_directory(args.output_dir)
    manifest_to_save = deepcopy(manifest)
    manifest_to_save["source_dataset_dir"] = str(
        Path(args.dataset_dir).expanduser().resolve()
    )
    with open(str(output / "split_manifest.json"), "w") as stream:
        json.dump(manifest_to_save, stream, indent=2, sort_keys=True)
        stream.write("\n")
    normalization.save(
        output / "normalization.npz",
        metadata={
            "statistics_source": "train_only",
            "dataset_schema_version": SCHEMA,
            "split_manifest_hash": manifest["manifest_hash"],
        },
    )

    for epoch in range(1, args.epochs + 1):
        actor.train()
        train_losses = []
        for state_batch, action_batch in train_loader:
            state_batch = state_batch.to(device)
            action_batch = action_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = actor.deterministic_action(state_batch)
            loss = loss_fn(prediction, action_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))
        actor.eval()
        with torch.no_grad():
            val_state_tensor = torch.from_numpy(val_states).to(device)
            val_action_tensor = torch.from_numpy(val_actions).to(device)
            val_loss = float(loss_fn(
                actor.deterministic_action(val_state_tensor), val_action_tensor
            ).cpu().item())
        train_loss = float(np.mean(train_losses))
        history.append((epoch, train_loss, val_loss))
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(
                output / "actor_best.pt", actor, epoch, val_loss,
                normalization, manifest,
            )
        else:
            epochs_without_improvement += 1
        _save_checkpoint(
            output / "actor_last.pt", actor, epoch, val_loss,
            normalization, manifest,
        )
        if epoch == 1 or epoch % 10 == 0:
            print("epoch {:4d} train_mse={:.9g} val_mse={:.9g}".format(
                epoch, train_loss, val_loss
            ))
        if epochs_without_improvement >= args.patience:
            print("Early stop at epoch {} (patience={})".format(epoch, args.patience))
            break

    best_checkpoint = torch.load(
        str(output / "actor_best.pt"), map_location=device
    )
    actor.load_state_dict(best_checkpoint["model_state_dict"])
    actor.to(device)
    val_metrics, _ = _evaluate_actor(
        actor, normalized_states[val_idx], transitions.actions[val_idx],
        normalization, device,
    )
    test_metrics, _ = _evaluate_actor(
        actor, normalized_states[test_idx], transitions.actions[test_idx],
        normalization, device,
    )
    val_metrics["baselines"] = _baseline_metrics(
        transitions.actions[val_idx], action_scale,
        transitions.actions[train_idx],
    )
    test_metrics["baselines"] = _baseline_metrics(
        transitions.actions[test_idx], action_scale,
        transitions.actions[train_idx],
    )
    if not _finite_metrics({"validation": val_metrics, "test": test_metrics}):
        raise RuntimeError("non-finite validation/test metrics")

    with open(str(output / "history.csv"), "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "train_normalized_mse", "validation_normalized_mse"))
        writer.writerows(history)
    with open(str(output / "validation_metrics.json"), "w") as stream:
        json.dump(val_metrics, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    with open(str(output / "test_metrics.json"), "w") as stream:
        json.dump(test_metrics, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    config = vars(args).copy()
    config.update({
        "resolved_device": str(device),
        "dataset_schema_version": SCHEMA,
        "state_definition": "joint_pos[6],tcp_xyz[3],pregrasp_xyz-tcp_xyz[3]",
        "action_definition": "executed_delta_joint[6] rad",
        "normalization_source": "train_only",
        "split_manifest_hash": manifest["manifest_hash"],
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
    })
    with open(str(output / "config.json"), "w") as stream:
        json.dump(config, stream, indent=2, sort_keys=True)
        stream.write("\n")
    all_metrics = {
        "episodes_loaded": transitions.episodes_loaded,
        "episodes_skipped": transitions.episodes_skipped,
        "transitions_total": len(transitions.states),
        "split_episode_counts": {
            key: len(value) for key, value in _manifest_sets(manifest).items()
        },
        "split_transition_counts": {
            key: int(len(value)) for key, value in split_indices.items()
        },
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "train": {
            "last_normalized_mse": history[-1][1],
            "action_audit": train_action_audit,
            "saturation": train_saturation,
        },
        "max_delta_joint_alignment_error": transitions.max_delta_joint_alignment_error,
        "max_next_joint_alignment_error": transitions.max_next_joint_alignment_error,
        "max_next_tcp_alignment_error": transitions.max_next_tcp_alignment_error,
        "normalization_source": "train_only",
        "action_scale": action_scale.tolist(),
        "train_action_audit": train_action_audit,
        "train_saturation": train_saturation,
        "coverage": {
            "train": train_coverage,
            "validation": validation_coverage,
            "test": test_coverage,
        },
        "validation": val_metrics,
        "test": test_metrics,
    }
    with open(str(output / "training_metrics.json"), "w") as stream:
        json.dump(all_metrics, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

    print("Best epoch:", best_epoch)
    print("Best validation normalized MSE:", best_loss)
    print("Validation physical MAE (rad):", val_metrics["physical_mae_rad"])
    print("Validation physical RMSE (rad):", val_metrics["physical_rmse_rad"])
    print("Test normalized MSE:", test_metrics["normalized_mse"])
    print("Test physical MAE (rad):", test_metrics["physical_mae_rad"])
    print("Test physical RMSE (rad):", test_metrics["physical_rmse_rad"])
    print("Output:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
