#!/usr/bin/env python3
"""Visually replay an HDF5 episode; this program never connects to ROS."""

import argparse
import json
import os
import sys

import cv2
import h5py
import numpy as np


def _text(value):
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _depth_view(depth):
    nonzero = depth[depth > 0]
    if nonzero.size == 0:
        scaled = np.zeros(depth.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(nonzero, [2.0, 98.0])
        if high <= low:
            high = low + 1.0
        scaled = np.clip((depth.astype(np.float32) - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)
        scaled[depth == 0] = 0
    return cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)


def _put_lines(image, lines):
    overlay = image.copy()
    height = 28 + 23 * len(lines)
    cv2.rectangle(overlay, (0, 0), (image.shape[1], height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)
    for index, line in enumerate(lines):
        cv2.putText(image, line, (12, 24 + index * 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def replay(path, headless=False):
    with h5py.File(path, "r") as episode:
        count = len(episode["time/timestamp_ns"])
        if count == 0:
            print("Episode has no samples", file=sys.stderr)
            return 1
        mapping = json.loads(_text(episode.attrs.get("state_mapping", "{}")))
        state_names = {int(code): name for name, code in mapping.items()}
        rate = float(episode.attrs.get("record_hz", 10.0))
        delay_ms = max(1, int(round(1000.0 / max(rate, 0.1))))
        index = 0
        paused = False
        window = "apple_rl replay"
        if headless:
            required = ("obs/rgb", "obs/depth", "obs/joint_pos", "obs/tcp_pose")
            missing = [name for name in required if name not in episode]
            if missing:
                print("Missing replay datasets: {}".format(", ".join(missing)), file=sys.stderr)
                return 1
            print("Replay compatible: {} samples ({})".format(
                count, "extended" if "task/control_mode" in episode else "legacy"
            ))
            return 0
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        while True:
            rgb = np.asarray(episode["obs/rgb"][index])
            color = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            depth = _depth_view(np.asarray(episode["obs/depth"][index]))
            if depth.shape[:2] != color.shape[:2]:
                depth = cv2.resize(depth, (color.shape[1], color.shape[0]), interpolation=cv2.INTER_NEAREST)
            frame = np.hstack((color, depth))

            joint = np.asarray(episode["obs/joint_pos"][index])
            tcp = np.asarray(episode["obs/tcp_pose"][index])
            apple = np.asarray(episode["teacher/apple_base_xyz"][index])
            action = np.asarray(episode["actions/executed_delta_tcp"][index])
            joint_delta = (
                np.asarray(episode["actions/executed_delta_joint"][index])
                if "actions/executed_delta_joint" in episode
                else None
            )
            state_code = int(episode["task/state"][index])
            control_text = "legacy"
            if "task/control_mode" in episode:
                try:
                    control_mapping = json.loads(_text(
                        episode.attrs.get("control_mode_mapping", "{}")
                    ))
                except ValueError:
                    control_mapping = {}
                inverse_control = {
                    int(code): name for name, code in control_mapping.items()
                }
                control_code = int(episode["task/control_mode"][index])
                control_text = inverse_control.get(control_code, str(control_code))
            lines = [
                "{}/{}  state={}  {}".format(index, count - 1, state_names.get(state_code, "UNKNOWN({})".format(state_code)), "PAUSED" if paused else "PLAY"),
                "control={} policy_action={}".format(
                    control_text,
                    bool(episode["valid/policy_action"][index])
                    if "valid/policy_action" in episode else "legacy",
                ),
                "J(rad): " + " ".join("{:+.3f}".format(value) for value in joint),
                "TCP xyz(m): " + " ".join("{:+.4f}".format(value) for value in tcp[:3]),
                "apple base xyz(m): " + " ".join("{:+.4f}".format(value) for value in apple),
                "executed delta: " + " ".join("{:+.4f}".format(value) for value in action),
            ]
            if joint_delta is not None:
                lines.append("delta joint(rad): " + " ".join(
                    "{:+.4f}".format(value) for value in joint_delta
                ))
                if "time/joint_state_timestamp_ns" in episode:
                    lines.append("joint source={} ts_ns={}".format(
                        _text(episode.attrs.get("joint_state_source", "unknown")),
                        int(episode["time/joint_state_timestamp_ns"][index]),
                    ))
            _put_lines(frame, lines)
            cv2.imshow(window, frame)
            key = cv2.waitKeyEx(0 if paused else delay_ms)
            if key in (ord("q"), ord("Q"), 27):
                break
            if key == ord(" "):
                paused = not paused
                continue
            if key in (81, 2424832, 65361):
                paused = True
                index = max(0, index - 1)
                continue
            if key in (83, 2555904, 65363):
                paused = True
                index = min(count - 1, index + 1)
                continue
            if not paused:
                index += 1
                if index >= count:
                    index = count - 1
                    paused = True
        cv2.destroyAllWindows()
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", help="episode_XXXXXX.hdf5 path")
    parser.add_argument(
        "--headless", action="store_true",
        help="validate replay compatibility without opening a window",
    )
    arguments = parser.parse_args()
    if not os.path.isfile(arguments.episode):
        parser.error("file does not exist: {}".format(arguments.episode))
    return replay(arguments.episode, headless=arguments.headless)


if __name__ == "__main__":
    sys.exit(main())
