#!/usr/bin/env python3
"""Create success, failure, and unknown/aborted mock episodes without ROS."""

import argparse
import os
import sys

from mock_data import create_episode


def create(output_root):
    os.makedirs(output_root, exist_ok=True)
    for name in os.listdir(output_root):
        if name.startswith("episode_") or name.startswith("collection_manifest"):
            path = os.path.join(output_root, name)
            if os.path.isfile(path):
                os.unlink(path)
    return [
        create_episode(output_root, "episode_000001", physical=1),
        create_episode(
            output_root, "episode_000002", physical=0,
            task_success=False, terminal="FAILED", failure_reason="grasp_miss",
        ),
        create_episode(
            output_root, "episode_000003", physical=-1,
            task_success=False, terminal="ABORT_REQUESTED",
            failure_reason="operator_abort",
        ),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="/tmp/apple_rl_mock")
    arguments = parser.parse_args()
    for path in create(arguments.output_root):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
