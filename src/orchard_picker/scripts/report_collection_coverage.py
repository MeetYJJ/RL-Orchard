#!/usr/bin/env python3
"""Report collection volume, coverage, labels, and quality without thresholds."""

import argparse
import json
import sys

from orchard_picker.rl_data.dataset_tools import collection_coverage


def _section(name, values):
    print(name + ":")
    if not values:
        print("  (none)")
    for key in sorted(values):
        print("  {}: {}".format(key, values[key]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()
    report = collection_coverage(arguments.dataset_root)
    if arguments.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    print("Episodes:", report["episode_count"])
    print("Physical success/failure/unknown: {}/{}/{}".format(
        report["physical_success"], report["physical_failure"],
        report["physical_unknown"],
    ))
    print("Task success:", report["task_success"])
    print("SAC goal valid:", report.get("sac_goal_valid", 0))
    print("Reached pre-grasp:", report.get("reached_pregrasp", 0))
    print("Episode valid/invalid: {}/{}".format(
        report["episode_valid"], report["episode_invalid"]
    ))
    print("Quality pass/fail: {}/{}".format(
        report["quality_pass"], report["quality_fail"]
    ))
    print("Interrupted .tmp:", report["interrupted_tmp_count"])
    print("Mean duration: {:.3f} s".format(report["mean_duration_sec"]))
    rate = report["physical_success_rate_labeled"]
    print("Physical success rate (labeled only):", "N/A" if rate is None else "{:.3f}".format(rate))
    _section("Init poses", report["init_pose_counts"])
    _section("Apple position bins", report["apple_position_bin_counts"])
    _section("Failure reasons", report["failure_reason_counts"])
    _section("Termination types", report["termination_type_counts"])
    _section("Phase samples", report["phase_sample_counts"])
    _section("Control-mode samples", report["control_mode_sample_counts"])
    _section("Mean valid ratios", report["valid_ratio_means"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
