#!/usr/bin/env python3
"""Overfit a tiny BC actor to verify the complete HDF5 training pipeline."""

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import HDF5BCDataset, NormalizationStats, load_bc_transitions
from models import SACActor


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-transitions", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=0,
                        help="0 uses full batch when N <= 64")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stop-mse", type=float, default=1e-5)
    parser.add_argument("--pass-mse", type=float, default=1e-4)
    parser.add_argument("--pass-reduction", type=float, default=100.0)
    parser.add_argument("--alignment-tolerance", type=float, default=1e-5)
    return parser.parse_args()


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _output_directory(value):
    if value:
        root = Path(value).expanduser().resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        root = Path(__file__).resolve().parent / "runs" / ("bc_pretrain_overfit_" + stamp)
    root.mkdir(parents=True, exist_ok=False)
    return root


def _physical_errors(prediction, target):
    error = np.asarray(prediction) - np.asarray(target)
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "max_abs_error": float(np.max(np.abs(error))),
        "mae_per_joint": np.mean(np.abs(error), axis=0).tolist(),
        "rmse_per_joint": np.sqrt(np.mean(np.square(error), axis=0)).tolist(),
        "max_abs_error_per_joint": np.max(np.abs(error), axis=0).tolist(),
    }


def main():
    args = _args()
    _seed_everything(args.seed)
    torch.set_num_threads(1)
    transitions = load_bc_transitions(
        args.dataset_dir,
        max_transitions=args.max_transitions,
        alignment_tolerance=args.alignment_tolerance,
        strict_alignment=True,
    )
    normalization = NormalizationStats.fit(
        transitions.states, transitions.actions
    )
    overflow = normalization.action_overflow(transitions.actions)
    overflow_counts = overflow["count_per_joint"].astype(int)
    overflow_max = overflow["max_abs_per_joint"]
    if np.any(overflow_counts):
        print("WARNING: normalized action exceeds [-1,1]")
        for joint, (count, maximum) in enumerate(
            zip(overflow_counts, overflow_max), start=1
        ):
            if count:
                print("  J{}: count={} max_abs={:.8f}".format(joint, count, maximum))

    dataset = HDF5BCDataset(transitions, normalization, clip_actions=True)
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        batch_size = len(dataset) if len(dataset) <= 64 else 64
    loader = DataLoader(
        dataset, batch_size=min(batch_size, len(dataset)), shuffle=True
    )

    device = torch.device("cpu")
    actor = SACActor().to(device)
    trainable = [parameter for parameter in actor.parameters()
                 if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    full_states = dataset.states.to(device)
    full_targets = dataset.actions.to(device)
    with torch.no_grad():
        initial_loss = float(
            loss_fn(torch.tanh(actor(full_states)[0]), full_targets).item()
        )

    loss_rows = []
    final_loss = initial_loss
    epochs_run = 0
    for epoch in range(1, int(args.epochs) + 1):
        actor.train()
        for state_batch, action_batch in loader:
            state_batch = state_batch.to(device)
            action_batch = action_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(torch.tanh(actor(state_batch)[0]), action_batch)
            loss.backward()
            optimizer.step()
        actor.eval()
        with torch.no_grad():
            final_loss = float(
                loss_fn(torch.tanh(actor(full_states)[0]), full_targets).item()
            )
        loss_rows.append((epoch, final_loss))
        epochs_run = epoch
        if epoch == 1 or epoch % 100 == 0:
            print("epoch {:4d} normalized_mse={:.9g}".format(epoch, final_loss))
        if final_loss <= float(args.early_stop_mse):
            break

    actor.eval()
    with torch.no_grad():
        normalized_prediction = actor.deterministic_action(full_states).cpu().numpy()
    physical_prediction = normalization.denormalize_actions(normalized_prediction)
    physical_metrics = _physical_errors(
        physical_prediction, transitions.actions
    )
    reduction = initial_loss / max(final_loss, 1e-30)
    # This is a pipeline diagnostic, not a generalization score.  Either an
    # exceptionally small final error or a >=100x reduction demonstrates that
    # gradients, normalization, and targets are connected.  Physical errors
    # are always reported alongside this boolean so a weak fit is visible.
    passed = bool(
        final_loss < float(args.pass_mse)
        or reduction >= float(args.pass_reduction)
    )

    output = _output_directory(args.output_dir)
    normalization.save(output / "normalization.npz")
    torch.save({
        "actor_state_dict": actor.state_dict(),
        "actor_class": "SACActor",
        "architecture_version": SACActor.architecture_version,
        "state_dim": 12,
        "action_dim": 6,
        "hidden_dim": 256,
        "normalized_action": True,
        "log_std_frozen_during_bc": True,
        "dataset_schema": "apple_rl_v1.2_sac_goal",
    }, str(output / "actor.pt"))
    with open(str(output / "loss.csv"), "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("epoch", "normalized_action_mse"))
        writer.writerows(loss_rows)

    config = {
        "dataset_dir": str(Path(args.dataset_dir).expanduser().resolve()),
        "max_transitions": int(args.max_transitions),
        "epochs": int(args.epochs),
        "epochs_run": epochs_run,
        "batch_size": min(batch_size, len(dataset)),
        "learning_rate": float(args.learning_rate),
        "seed": int(args.seed),
        "alignment_tolerance": float(args.alignment_tolerance),
        "state_definition": "joint_pos[6],tcp_xyz[3],pregrasp_xyz-tcp_xyz[3]",
        "action_definition": "executed_delta_joint[6] rad",
        "action_scale_method": normalization.action_scale_method,
    }
    with open(str(output / "config.yaml"), "w") as stream:
        json.dump(config, stream, indent=2, sort_keys=True)
        stream.write("\n")

    examples = []
    for index in range(min(5, len(dataset))):
        examples.append({
            "episode": transitions.episode_paths[index],
            "transition_index": int(transitions.transition_indices[index]),
            "target_delta_q_rad": transitions.actions[index].tolist(),
            "predicted_delta_q_rad": physical_prediction[index].tolist(),
        })
    metrics = {
        "pass": passed,
        "episodes_loaded": transitions.episodes_loaded,
        "episodes_skipped": transitions.episodes_skipped,
        "transitions_used": len(dataset),
        "episodes_used_for_transitions": len(set(transitions.episode_paths)),
        "initial_bc_loss": initial_loss,
        "final_bc_loss": final_loss,
        "initial_normalized_mse": initial_loss,
        "final_normalized_mse": final_loss,
        "loss_reduction_ratio": reduction,
        "physical_action_error": physical_metrics,
        "state_mean": normalization.state_mean.tolist(),
        "state_std": normalization.state_std.tolist(),
        "action_scale": normalization.action_scale.tolist(),
        "action_overflow_count_per_joint": overflow_counts.tolist(),
        "action_normalized_max_abs_per_joint": overflow_max.tolist(),
        "max_delta_joint_alignment_error": transitions.max_delta_joint_alignment_error,
        "max_next_joint_alignment_error": transitions.max_next_joint_alignment_error,
        "max_next_tcp_alignment_error": transitions.max_next_tcp_alignment_error,
        "prediction_examples": examples,
    }
    with open(str(output / "training_metrics.json"), "w") as stream:
        json.dump(metrics, stream, indent=2, sort_keys=True)
        stream.write("\n")

    print("\nEpisodes loaded:", transitions.episodes_loaded)
    print("Episodes used for selected transitions:",
          len(set(transitions.episode_paths)))
    print("Episodes skipped:", transitions.episodes_skipped)
    print("Policy transitions:", len(dataset))
    print("State shape:", tuple(transitions.states.shape))
    print("Action shape:", tuple(transitions.actions.shape))
    print("State mean:", normalization.state_mean.tolist())
    print("State std:", normalization.state_std.tolist())
    print("Action scale (per-joint max_abs, training normalization only):",
          normalization.action_scale.tolist())
    print("Initial BC loss:", initial_loss)
    print("Final BC loss:", final_loss)
    print("Loss reduction ratio:", reduction)
    print("Physical delta-q MAE:", physical_metrics["mae"])
    print("Per-joint MAE:", physical_metrics["mae_per_joint"])
    print("Delta-joint alignment max error:",
          transitions.max_delta_joint_alignment_error)
    print("next_obs joint alignment max error:",
          transitions.max_next_joint_alignment_error)
    print("next_obs TCP alignment max error:",
          transitions.max_next_tcp_alignment_error)
    for index, example in enumerate(examples):
        print("Example {} target={} prediction={}".format(
            index, example["target_delta_q_rad"],
            example["predicted_delta_q_rad"],
        ))
    print("Output:", output)
    print("OVERFIT {}".format("PASS" if passed else "FAIL"))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
