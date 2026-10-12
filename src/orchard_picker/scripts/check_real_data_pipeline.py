#!/usr/bin/env python3
"""Read-only preflight for the first real expert-data capture.

Only subscribes to existing topics.  It never creates a motion, gripper,
home, start, or run_once client.
"""

import argparse
import json
import os
import shutil
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from orchard_picker.msg import RobotState
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, String

from orchard_picker.rl_data.joint_state import (
    JointStateMappingError,
    estimate_rate_hz,
    normalize_joint_positions,
)


def stamp_ns(message):
    return int(message.header.stamp.sec) * 1000000000 + int(message.header.stamp.nanosec)


def latched_qos():
    qos = QoSProfile(depth=1)
    try:
        from rclpy.qos import DurabilityPolicy
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    except ImportError:
        from rclpy.qos import QoSDurabilityPolicy as DurabilityPolicy
        qos.durability = DurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL
    return qos


class Preflight(Node):
    def __init__(self, args):
        super().__init__("rl_data_preflight")
        self.args = args
        self.started_ns = time.monotonic_ns()
        self.latest = {}
        self.joint_times = []
        self.joint_history = []
        self.joint_values = None
        self.connected = None
        self.robot_error_code = None
        self.camera_metadata_valid = False
        self.target_info_valid = False
        self.target_base_valid = False
        self.create_subscription(Image, args.rgb_topic, self._cb("rgb"), qos_profile_sensor_data)
        self.create_subscription(Image, args.depth_topic, self._cb("depth"), qos_profile_sensor_data)
        latched = latched_qos()
        self.create_subscription(String, args.camera_metadata_topic, self._camera_metadata_cb, latched)
        self.create_subscription(RobotState, args.robot_topic, self._robot_cb, 10)
        self.create_subscription(Bool, args.connected_topic, self._connected_cb, latched)
        self.create_subscription(String, args.target_info_topic, self._target_info_cb, latched)
        self.create_subscription(PointStamped, args.target_base_topic, self._target_base_cb, latched)
        if args.joint_source == "sensor_msgs_joint_state":
            self.create_subscription(JointState, args.joint_topic, self._joint_cb, qos_profile_sensor_data)

    def _cb(self, name):
        def callback(message):
            self.latest[name] = time.monotonic_ns()
        return callback

    def _robot_cb(self, message):
        received_ns = time.monotonic_ns()
        self.latest["robot"] = received_ns
        self.robot_error_code = int(message.error_code)
        if self.args.joint_source == "rm_json_bridge":
            values = np.asarray(message.joint_pos_rad, dtype=np.float64)
            if values.shape == (6,) and np.all(np.isfinite(values)):
                self.joint_values = values
                source_ns = stamp_ns(message)
                if source_ns <= 0:
                    source_ns = received_ns
                self.joint_times.append(source_ns)
                self.joint_history.append(values.copy())
                self.latest["joint"] = received_ns

    def _connected_cb(self, message):
        self.connected = bool(message.data)
        self.latest["connected"] = time.monotonic_ns()

    def _camera_metadata_cb(self, message):
        try:
            payload = json.loads(message.data)
            self.camera_metadata_valid = (
                isinstance(payload, dict)
                and float(payload.get("depth_scale_m", 0.0)) > 0.0
                and bool(str(payload.get("camera_serial", "")).strip())
            )
        except (TypeError, ValueError):
            self.camera_metadata_valid = False
        self.latest["camera_metadata"] = time.monotonic_ns()

    def _target_info_cb(self, message):
        try:
            payload = json.loads(message.data)
            xyz = np.asarray(payload.get("camera_xyz_m", []), dtype=np.float64)
            self.target_info_valid = bool(
                payload.get("target_valid", False)
                and xyz.shape == (3,)
                and np.all(np.isfinite(xyz))
            )
        except (TypeError, ValueError):
            self.target_info_valid = False
        self.latest["target_info"] = time.monotonic_ns()

    def _target_base_cb(self, message):
        values = np.asarray(
            [message.point.x, message.point.y, message.point.z], dtype=np.float64
        )
        self.target_base_valid = bool(np.all(np.isfinite(values)))
        self.latest["target_base"] = time.monotonic_ns()

    def _joint_cb(self, message):
        try:
            values = normalize_joint_positions(
                message.name, message.position, self.args.joint_names
            )
        except (JointStateMappingError, TypeError, ValueError):
            return
        now = time.monotonic_ns()
        self.joint_values = values
        source_ns = stamp_ns(message)
        if source_ns <= 0:
            source_ns = now
        self.joint_times.append(source_ns)
        self.joint_history.append(np.asarray(values, dtype=np.float64))
        self.latest["joint"] = now

    def report(self):
        now = time.monotonic_ns()
        checks = []
        def fresh(name, label, max_age=None):
            age = (now - self.latest[name]) / 1e6 if name in self.latest else np.inf
            ok = np.isfinite(age) and (max_age is None or age <= max_age)
            checks.append((label, ok, "age_ms={:.1f}".format(age)))
        fresh("connected", "Robot connection telemetry")
        checks.append(("Robot connected", self.connected is True, "value={}".format(self.connected)))
        # Camera and apple-teacher topics use the visual-data freshness
        # budget.  Robot feedback is sampled independently and has its own
        # budget below; do not apply one threshold to both streams.
        fresh("rgb", "RGB topic", self.args.max_sensor_age_ms)
        fresh("depth", "Depth topic", self.args.max_sensor_age_ms)
        fresh("camera_metadata", "Camera metadata topic")
        checks.append(("Camera metadata valid", self.camera_metadata_valid, "serial/depth_scale"))
        fresh("robot", "TCP/robot state", self.args.max_robot_age_ms)
        checks.append(("Robot error code", self.robot_error_code == 0, "error_code={}".format(self.robot_error_code)))
        fresh("joint", "Joint state", self.args.max_robot_age_ms)
        fresh("target_info", "Apple camera target topic", self.args.max_sensor_age_ms)
        checks.append(("Apple camera target valid", self.target_info_valid, "target_valid={}".format(self.target_info_valid)))
        fresh("target_base", "Apple base target topic", self.args.max_sensor_age_ms)
        checks.append(("Apple base target valid", self.target_base_valid, "finite_xyz"))
        rate = estimate_rate_hz(self.joint_times)
        checks.append(("Joint shape == 6", self.joint_values is not None and self.joint_values.shape == (6,), str(None if self.joint_values is None else self.joint_values.shape)))
        checks.append(("Joint rate covers recorder", np.isfinite(rate) and rate >= self.args.record_hz, "rate_hz={:.3f}, required_hz={:.3f}".format(rate, self.args.record_hz) if np.isfinite(rate) else "unavailable"))
        node_names = set(self.get_node_names())
        checks.append(("Recorder ready", "rl_data_recorder" in node_names, "nodes={}".format(",".join(sorted(node_names)))))
        disk_path = os.path.abspath(self.args.dataset_root)
        while not os.path.exists(disk_path):
            parent = os.path.dirname(disk_path)
            if parent == disk_path:
                break
            disk_path = parent
        free = shutil.disk_usage(disk_path).free
        checks.append(("Disk space", free > self.args.min_free_bytes, "free_bytes={}".format(free)))
        print("RL data preflight (read-only)")
        for label, ok, detail in checks:
            print("{}: {} ({})".format("PASS" if ok else "FAIL", label, detail))
        all_ok = all(item[1] for item in checks)
        print("joint_state_source: {}".format(self.args.joint_source))
        print("joint_rate_hz: {}".format(rate))
        if len(self.joint_history) > 1:
            deltas = np.diff(np.asarray(self.joint_history), axis=0)
            print("stationary_delta_mean_rad: {}".format(np.mean(deltas, axis=0).tolist()))
            print("stationary_delta_std_rad: {}".format(np.std(deltas, axis=0).tolist()))
            print("stationary_delta_max_abs_rad: {}".format(np.max(np.abs(deltas), axis=0).tolist()))
        else:
            print("stationary_delta_stats: unavailable")
        return 0 if all_ok else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--record-hz", type=float, default=10.0)
    # New streams have separate budgets: visual/teacher data defaults to
    # 150 ms, while the 5 Hz robot feedback needs 300 ms.  Keep the legacy
    # option for existing callers; when both forms are supplied, the new
    # stream-specific options take precedence.
    parser.add_argument(
        "--max-sensor-age-ms",
        type=float,
        default=None,
        help="maximum age for RGB, depth, and apple target data (default: 150 ms)",
    )
    parser.add_argument(
        "--max-robot-age-ms",
        type=float,
        default=None,
        help="maximum age for TCP/RobotState and joint data (default: 300 ms)",
    )
    parser.add_argument(
        "--max-age-ms",
        type=float,
        default=None,
        help=(
            "legacy common age limit; fills any omitted stream-specific limit "
            "(explicit --max-sensor-age-ms/--max-robot-age-ms take precedence)"
        ),
    )
    parser.add_argument("--min-free-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--dataset-root", default="/home/nvidia/RL-Orchard/rl_dataset")
    parser.add_argument("--joint-source", choices=("rm_json_bridge", "sensor_msgs_joint_state"), default="rm_json_bridge")
    parser.add_argument("--joint-topic", default="/joint_states")
    parser.add_argument("--joint-names", default="")
    parser.add_argument("--rgb-topic", default="/apple_vision/rgb")
    parser.add_argument("--depth-topic", default="/apple_vision/aligned_depth")
    parser.add_argument("--camera-metadata-topic", default="/apple_vision/camera_metadata")
    parser.add_argument("--robot-topic", default="/rm_json_bridge/arm_state")
    parser.add_argument("--connected-topic", default="/rm_json_bridge/connected")
    parser.add_argument("--target-info-topic", default="/apple/target_info")
    parser.add_argument("--target-base-topic", default="/pick_task_manager/target_base")
    args = parser.parse_args(argv)
    legacy_age = args.max_age_ms
    args.max_sensor_age_ms = (
        args.max_sensor_age_ms
        if args.max_sensor_age_ms is not None
        else (legacy_age if legacy_age is not None else 150.0)
    )
    args.max_robot_age_ms = (
        args.max_robot_age_ms
        if args.max_robot_age_ms is not None
        else (legacy_age if legacy_age is not None else 300.0)
    )
    if args.max_sensor_age_ms < 0.0 or args.max_robot_age_ms < 0.0:
        parser.error("freshness limits must be non-negative")
    args.joint_names = tuple(item.strip() for item in args.joint_names.split(",") if item.strip())
    if args.joint_source == "sensor_msgs_joint_state" and len(args.joint_names) != 6:
        parser.error("sensor_msgs_joint_state requires six verified --joint-names")
    rclpy.init()
    node = Preflight(args)
    try:
        deadline = time.monotonic() + max(0.1, args.duration)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        return node.report()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
