#!/usr/bin/env python3
"""Read-only episode outcome audit, including reached_pregrasp=False cases."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


SCHEMA = "apple_rl_v1.2_sac_goal"


def _bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no", ""):
            return False
    if isinstance(value, np.ndarray):
        return bool(value.reshape(-1)[0]) if value.size else default
    return bool(value)


def _text(value):
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def inspect(path):
    eid = path.stem
    with h5py.File(str(path), "r") as f:
        goal = np.asarray(f["episode_meta/pregrasp_pose_base"][:], dtype=np.float64) if "episode_meta/pregrasp_pose_base" in f else None
        tcp = np.asarray(f["obs/tcp_pose"][:], dtype=np.float64) if "obs/tcp_pose" in f else None
        policy = np.asarray(f["valid/policy_action"][:], dtype=bool) if "valid/policy_action" in f else np.zeros(0, dtype=bool)
        action_valid = np.asarray(f["valid/executed_delta_joint"][:], dtype=bool) if "valid/executed_delta_joint" in f else np.zeros(0, dtype=bool)
        count = min(len(policy), len(action_valid))
        transitions = int(np.sum(policy[:count] & action_valid[:count] & (np.arange(count) + 1 < len(tcp)))) if tcp is not None else 0
        distances = None
        if goal is not None and goal.shape == (7,) and np.all(np.isfinite(goal)) and tcp is not None and len(tcp):
            distances = np.linalg.norm(goal[:3][None, :] - tcp[:, :3], axis=1)
        return {
            "episode_id": eid,
            "schema": _text(f.attrs.get("dataset_schema_version", "legacy")),
            "episode_valid": _bool(f.attrs.get("episode_valid")),
            "discarded": _bool(f.attrs.get("discarded")),
            "reached_pregrasp": _bool(f.attrs.get("reached_pregrasp")),
            "success": None if f.attrs.get("success") is None else _bool(f.attrs.get("success")),
            "task_success": None if f.attrs.get("task_success") is None else _bool(f.attrs.get("task_success")),
            "terminal_state": _text(f.attrs.get("terminal_state", "")),
            "termination_type": _text(f.attrs.get("termination_type", "")),
            "failure_reason": _text(f.attrs.get("failure_reason", "")),
            "policy_transition_count": transitions,
            "final_distance_m": None if distances is None else float(distances[-1]),
            "min_distance_m": None if distances is None else float(np.min(distances)),
        }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    root = Path(args.dataset_dir).expanduser().resolve()
    rows = []
    for path in sorted(root.glob("episode_*.hdf5")):
        try:
            row = inspect(path)
        except (OSError, KeyError, ValueError) as exc:
            row = {"episode_id": path.stem, "error": str(exc)}
        rows.append(row)
    false_rows = [row for row in rows if row.get("reached_pregrasp") is False]
    result = {"dataset_dir": str(root), "episodes": rows, "episode_count": len(rows), "reached_pregrasp_false_count": len(false_rows), "reached_pregrasp_false": false_rows}
    output = Path(args.output).expanduser().resolve() if args.output else Path("training/runs/offline_rl_episode_outcome_audit.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print("Episodes:", len(rows), "reached_pregrasp=False:", len(false_rows))
    for row in false_rows:
        print("{} final_distance_m={} min_distance_m={} policy_transitions={} terminal_state={}".format(row["episode_id"], row["final_distance_m"], row["min_distance_m"], row["policy_transition_count"], row["terminal_state"]))
    print("Audit:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
