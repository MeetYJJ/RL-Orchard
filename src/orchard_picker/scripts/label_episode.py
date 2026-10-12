#!/usr/bin/env python3
"""Add or revise a human physical-grasp label without altering trajectory data."""

import argparse
import os
import sys

from orchard_picker.rl_data.dataset_tools import apply_physical_label


LABELS = {"1": 1, "0": 0, "u": -1, "U": -1}


def _prompt(prompt, default=""):
    value = input(prompt).strip()
    return value if value else default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode")
    parser.add_argument("--label", choices=("1", "0", "u"))
    parser.add_argument("--failure-reason", default=None)
    parser.add_argument("--operator-note", default=None)
    arguments = parser.parse_args()
    if not os.path.isfile(arguments.episode):
        parser.error("episode does not exist: {}".format(arguments.episode))
    label_text = arguments.label
    while label_text not in LABELS:
        label_text = _prompt("Physical grasp success [1/0/u]: ")
    failure_reason = arguments.failure_reason
    if failure_reason is None:
        failure_reason = _prompt("Failure reason (free text, Enter for empty): ")
    operator_note = arguments.operator_note
    if operator_note is None:
        operator_note = _prompt("Operator note (Enter for empty): ")
    row = apply_physical_label(
        arguments.episode, LABELS[label_text], failure_reason, operator_note
    )
    print("updated:", os.path.abspath(arguments.episode))
    print("physical_grasp_success:", row["physical_grasp_success"])
    print("failure_reason:", row["failure_reason"])
    print("operator_note:", row["operator_note"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

