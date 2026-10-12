#!/usr/bin/env python3
"""Inspect the exact transitions consumed by the BC/SAC-compatible loader."""

import argparse
from pathlib import Path
import h5py
import numpy as np

from dataset import NormalizationStats, load_bc_transitions


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--alignment-tolerance", type=float, default=1e-5)
    return parser.parse_args()


def _stats(value):
    value = np.asarray(value, dtype=np.float64)
    return {
        "min": np.min(value, axis=0).tolist(),
        "max": np.max(value, axis=0).tolist(),
        "mean": np.mean(value, axis=0).tolist(),
        "std": np.std(value, axis=0).tolist(),
    }


def main():
    args = _args()
    transitions = load_bc_transitions(
        args.dataset_dir,
        alignment_tolerance=args.alignment_tolerance,
        strict_alignment=True,
    )
    state = transitions.states
    action = transitions.actions
    print("Dataset:", args.dataset_dir)
    print("Episodes found:", len(list(Path(args.dataset_dir).expanduser().glob("episode_*.hdf5"))))
    print("Supported schema:", "apple_rl_v1.2_sac_goal")
    print("Valid episodes:", transitions.episodes_loaded)
    print("Skipped episodes:", transitions.episodes_skipped)
    for status in transitions.statuses:
        print("  {}: {} ({})".format(
            "LOADED" if status.loaded else "SKIPPED",
            status.path,
            status.reason,
        ))
    print("Policy transitions:", len(state))
    print("State shape:", tuple(state.shape))
    print("Action shape:", tuple(action.shape))
    print("State NaN count:", int(np.isnan(state).sum()))
    print("State Inf count:", int(np.isinf(state).sum()))
    print("Action NaN count:", int(np.isnan(action).sum()))
    print("Action Inf count:", int(np.isinf(action).sum()))
    print("State dimensions [joint6,tcp_xyz3,relative_goal3]:")
    for index, values in enumerate(zip(
        state.min(axis=0), state.max(axis=0),
        state.mean(axis=0), state.std(axis=0),
    )):
        print("  s{:02d}: min={:.9g} max={:.9g} mean={:.9g} std={:.9g}".format(
            index, values[0], values[1], values[2], values[3]
        ))
    print("Joint action dimensions [rad]:")
    for index, values in enumerate(zip(
        action.min(axis=0), action.max(axis=0),
        action.mean(axis=0), action.std(axis=0),
    )):
        print("  dq{}: min={:.9g} max={:.9g} mean={:.9g} std={:.9g}".format(
            index + 1, values[0], values[1], values[2], values[3]
        ))
    if len(transitions.goals_xyz):
        print("Pregrasp goal XYZ range:", _stats(transitions.goals_xyz))
    if len(transitions.initial_joints):
        print("Initial joint range:", _stats(transitions.initial_joints))
    normalization = NormalizationStats.fit(state, action)
    print("Action scale (per-joint max_abs, normalization only):",
          normalization.action_scale.tolist())
    reached = []
    for status in transitions.statuses:
        if not status.loaded:
            continue
        try:
            with h5py.File(status.path, "r") as episode:
                if "reached_pregrasp" in episode.attrs:
                    reached.append(bool(episode.attrs["reached_pregrasp"]))
        except (OSError, KeyError, ValueError):
            continue
    if reached:
        print("Reached-pregrasp statistics: count={} true={} false={}".format(
            len(reached), sum(reached), len(reached) - sum(reached)
        ))
    else:
        print("Reached-pregrasp statistics: unavailable")
    print("max delta_joint alignment error:",
          transitions.max_delta_joint_alignment_error)
    print("max next_obs joint alignment error:",
          transitions.max_next_joint_alignment_error)
    print("max next_obs TCP alignment error:",
          transitions.max_next_tcp_alignment_error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
