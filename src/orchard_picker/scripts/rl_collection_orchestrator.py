#!/usr/bin/env python3
"""Operator-in-the-loop collection coordinator using existing control services.

Dry-run is the default unless both ORCHARD_ARM_ENABLE and
ORCHARD_COLLECTION_ENABLE are exactly YES.  This node never implements robot
kinematics or safety limits; it only invokes explicitly configured, previously
verified services.
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger

try:
    from rclpy.qos import DurabilityPolicy
except ImportError:
    from rclpy.qos import QoSDurabilityPolicy as DurabilityPolicy

from orchard_picker.msg import RobotState
from orchard_picker.rl_data.collection_planning import (
    CollectionConfigError,
    compute_position_bin,
    generate_joint_jitter,
    load_collection_config,
    load_manifest_counts,
    plan_to_json,
    select_least_sampled_pose,
    suggest_least_sampled_region,
)
from orchard_picker.rl_data.dataset_tools import apply_physical_label
from orchard_picker.srv import MoveJ


def _latched_qos():
    profile = QoSProfile(depth=1)
    profile.durability = DurabilityPolicy.TRANSIENT_LOCAL
    return profile


def _authorized():
    return (
        os.environ.get("ORCHARD_ARM_ENABLE") == "YES"
        and os.environ.get("ORCHARD_COLLECTION_ENABLE") == "YES"
    )


def _finite_sequence(values, length):
    try:
        return len(values) == length and all(
            math.isfinite(float(value)) for value in values
        )
    except (TypeError, ValueError, OverflowError):
        return False


class CollectionOrchestrator(Node):
    """Coordinate operator, recorder, and existing expert grasp services."""

    def __init__(self, config, dry_run):
        super().__init__("rl_collection_orchestrator")
        self.config = config
        self.dry_run = bool(dry_run)
        self.topic_config = config.get("topics", {})
        self.service_config = config.get("services", {})
        self.latest = {}
        self.finalized = None
        self.selection_index = 0
        self.collection_state = "WAIT_OPERATOR"
        self.recorder_disabled_by_orchestrator = False

        self.create_subscription(
            Image, self.topic_config["rgb"], lambda msg: self._mark("rgb", msg),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Image, self.topic_config["depth"], lambda msg: self._mark("depth", msg),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            RobotState, self.topic_config["robot_state"], self._robot_cb, 10
        )
        self.create_subscription(
            String, self.topic_config["target_info"], self._target_info_cb,
            _latched_qos(),
        )
        self.create_subscription(
            PointStamped, self.topic_config["target_base"], self._target_base_cb,
            _latched_qos(),
        )
        self.create_subscription(
            Bool, self.topic_config["task_active"], self._active_cb, _latched_qos()
        )
        self.create_subscription(
            String, self.topic_config["task_state"], self._state_cb, _latched_qos()
        )
        self.create_subscription(
            String, self.topic_config["recorder_finalized"], self._finalized_cb,
            _latched_qos(),
        )
        self.metadata_pub = self.create_publisher(
            String, self.topic_config["episode_metadata"], _latched_qos()
        )
        self.start_client = self.create_client(
            Trigger, self.service_config["expert_start"]
        )
        self.recorder_client = self.create_client(
            SetBool, self.service_config["recorder_enable"]
        )

    def set_collection_state(self, state):
        self.collection_state = str(state)
        self.get_logger().info("[RL Collection] state={}".format(self.collection_state))

    def _mark(self, name, message):
        self.latest[name] = {"message": message, "received_ns": time.monotonic_ns()}

    def _robot_cb(self, message):
        pose = message.tcp_pose
        self.latest["robot"] = {
            "received_ns": time.monotonic_ns(),
            "joint_pos": [float(value) for value in message.joint_pos_rad],
            "tcp_pose": [
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y,
                pose.orientation.z, pose.orientation.w,
            ],
            "error_code": int(message.error_code),
        }

    def _target_info_cb(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError) as exc:
            self.get_logger().warning("Invalid target_info JSON: {}".format(exc))
            return
        payload["received_ns"] = time.monotonic_ns()
        self.latest["target_info"] = payload

    def _target_base_cb(self, message):
        self.latest["target_base"] = {
            "received_ns": time.monotonic_ns(),
            "xyz": [message.point.x, message.point.y, message.point.z],
        }

    def _active_cb(self, message):
        self.latest["active"] = {
            "received_ns": time.monotonic_ns(), "value": bool(message.data)
        }

    def _state_cb(self, message):
        self.latest["state"] = {
            "received_ns": time.monotonic_ns(), "value": str(message.data)
        }

    def _finalized_cb(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError) as exc:
            self.get_logger().warning("Invalid recorder finalized JSON: {}".format(exc))
            return
        payload["received_ns"] = time.monotonic_ns()
        self.finalized = payload

    def _pump_until(self, predicate, timeout_sec):
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if predicate():
                return True
        return bool(predicate())

    def _call(self, client, request, timeout_sec=30.0):
        if not client.wait_for_service(timeout_sec=2.0):
            raise RuntimeError("service unavailable: {}".format(client.srv_name))
        future = client.call_async(request)
        if not self._pump_until(future.done, timeout_sec):
            raise RuntimeError("service timeout: {}".format(client.srv_name))
        response = future.result()
        if response is None:
            raise RuntimeError("service returned no response: {}".format(client.srv_name))
        return response

    def readiness(self):
        now = time.monotonic_ns()
        sensor_age = float(self.config.get("max_sensor_age_ms", 250.0)) * 1e6
        robot_age = float(self.config.get("max_robot_state_age_ms", 250.0)) * 1e6
        status = {}
        for name in ("rgb", "depth"):
            item = self.latest.get(name)
            expected_encodings = (
                ("bgr8", "rgb8") if name == "rgb" else ("16UC1", "mono16")
            )
            status[name] = bool(
                item and now - item["received_ns"] <= sensor_age
                and int(item["message"].height) > 0
                and int(item["message"].width) > 0
                and str(item["message"].encoding) in expected_encodings
            )
        robot = self.latest.get("robot")
        status["robot"] = bool(
            robot and now - robot["received_ns"] <= robot_age
            and _finite_sequence(robot["joint_pos"], 6)
            and _finite_sequence(robot["tcp_pose"], 7)
            and robot["error_code"] == 0
        )
        info = self.latest.get("target_info")
        base = self.latest.get("target_base")
        camera_xyz = info.get("camera_xyz_m", []) if info else []
        base_xyz = base.get("xyz", []) if base else []
        status["target"] = bool(
            info and base and bool(info.get("target_valid", info.get("camera_xyz_m")))
            and now - info["received_ns"] <= sensor_age
            and now - base["received_ns"] <= sensor_age
            and _finite_sequence(camera_xyz, 3) and _finite_sequence(base_xyz, 3)
        )
        return status

    def wait_ready(self):
        timeout = float(self.config.get("readiness_timeout_sec", 15.0))
        okay = self._pump_until(
            lambda: all(self.readiness().values()), timeout
        )
        return okay, self.readiness()

    def set_recorder_enabled(self, enabled):
        request = SetBool.Request()
        request.data = bool(enabled)
        response = self._call(self.recorder_client, request, timeout_sec=5.0)
        if not response.success:
            raise RuntimeError("recorder rejected request: {}".format(response.message))
        self.recorder_disabled_by_orchestrator = not bool(enabled)

    def execute_initial_pose(self, pose, offset):
        """Call only the explicitly configured verified initialization service."""
        motion = str(pose.get("motion", "home_service"))
        service = str(pose.get("service", ""))
        if motion in ("home_service", "trigger_service"):
            client = self.create_client(Trigger, service)
            response = self._call(client, Trigger.Request(), timeout_sec=120.0)
        elif motion == "movej":
            if service.rstrip("/") == "/rm_json_bridge/movej":
                raise RuntimeError("direct rm_json_bridge/movej is forbidden")
            if not bool(pose.get("verified", False)):
                raise RuntimeError("initial pose is not marked verified")
            client = self.create_client(MoveJ, service)
            request = MoveJ.Request()
            request.joint_deg = [
                float(value) + float(delta)
                for value, delta in zip(pose["joint_deg"], offset)
            ]
            request.speed = int(pose["speed"])
            request.blend_radius = int(pose.get("blend_radius", 0))
            request.wait = True
            response = self._call(client, request, timeout_sec=120.0)
        else:
            raise RuntimeError("unsupported initial motion {}".format(motion))
        if not response.success:
            raise RuntimeError("initial pose failed: {}".format(response.message))

    def publish_metadata(self, pose, offset):
        robot = self.latest["robot"]
        target_info = self.latest["target_info"]
        target_base = self.latest["target_base"]
        bins = self.config.get("apple_position_bins", {})
        apple_bin = compute_position_bin(target_base["xyz"], bins)
        metadata = {
            "initial_pose_id": str(pose["id"]),
            "initial_joint_pos": list(robot["joint_pos"]),
            "initial_tcp_pose": list(robot["tcp_pose"]),
            "initialization_mode": str(self.config["initialization"].get("mode", "pose_library")),
            "initialization_seed": int(self.config.get("selection_seed", 0)) + self.selection_index,
            "initialization_offset": [float(value) for value in offset],
            "initialization_offset_unit": "degree_joint",
            "initialization_timestamp": datetime.now().astimezone().isoformat(),
            "initial_pose_bin": str(pose["id"]),
            "apple_position_bin": apple_bin,
            "initial_apple_camera_xyz": list(target_info.get("camera_xyz_m", [])),
            "initial_apple_base_xyz": list(target_base["xyz"]),
            "initial_apple_id": int(target_info.get("id", -1)),
            "initial_apple_confidence": float(target_info.get("confidence", np.nan)),
            "initial_apple_bbox": list(target_info.get("bbox_xyxy", [-1] * 4)),
        }
        message = String()
        message.data = json.dumps(metadata, sort_keys=True)
        self.metadata_pub.publish(message)
        # Transient-local delivery is immediate in normal operation; spinning
        # briefly also lets the recorder subscription process the metadata.
        end = time.monotonic() + 0.25
        while time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
        return metadata

    def run_expert_episode(self):
        self.finalized = None
        response = self._call(self.start_client, Trigger.Request(), timeout_sec=10.0)
        if not response.success:
            raise RuntimeError("expert start rejected: {}".format(response.message))
        self.set_collection_state("WAIT_EPISODE_FINISH")
        timeout = float(self.config.get("episode_timeout_sec", 180.0))
        if not self._pump_until(
            lambda: bool(self.latest.get("active", {}).get("value", False)), 5.0
        ):
            raise RuntimeError("task active did not become true")
        if not self._pump_until(
            lambda: not bool(self.latest.get("active", {}).get("value", True)), timeout
        ):
            raise RuntimeError("episode did not finish within timeout")
        finalize_timeout = float(
            self.config.get("recorder_finalize_timeout_sec", 15.0)
        )
        if not self._pump_until(lambda: self.finalized is not None, finalize_timeout):
            raise RuntimeError("recorder did not publish finalized result")
        return dict(self.finalized)


def _choose_pose(config, selection_index):
    initialization = config["initialization"]
    manifest = config.get(
        "manifest_file", os.path.join(config["dataset_root"], "collection_manifest.csv")
    )
    counts = load_manifest_counts(manifest, "initial_pose_id")
    seed = int(config.get("selection_seed", 0))
    pose = select_least_sampled_pose(
        initialization["initial_pose_library"], counts, seed, selection_index
    )
    offset = generate_joint_jitter(config, seed, selection_index)
    return pose, offset, counts


def _print_plan(config, selection_index=0):
    pose, offset, counts = _choose_pose(config, selection_index)
    print("Collection plan (NO MOTION):")
    print("  pose counts:", json.dumps(counts, sort_keys=True))
    print("  selected:", plan_to_json(
        pose, config["initialization"].get("mode", "pose_library"),
        int(config.get("selection_seed", 0)) + selection_index, offset,
    ))
    print("  motion service:", pose.get("service", "UNSET"))
    print("  bounded jitter enabled:", bool(
        config["initialization"].get("bounded_random_jitter", {}).get("enabled", False)
    ))
    region_counts = load_manifest_counts(
        config.get("manifest_file", ""), "apple_position_bin"
    )
    suggestion = suggest_least_sampled_region(
        config.get("apple_position_bins", {}).get("suggested_regions", []),
        region_counts,
    )
    if suggestion:
        print("  suggested next apple region:", suggestion)
    return pose, offset


def _label_episode(path, label_later):
    if label_later:
        return
    while True:
        answer = input("Physical grasp success? [1] success [0] failure [u] unknown [l] later: ").strip().lower()
        if answer in ("1", "0", "u", "l"):
            break
    if answer == "l":
        return
    value = {"1": 1, "0": 0, "u": -1}[answer]
    reason = ""
    if value == 0:
        reason = input("Failure reason (free text): ").strip()
    note = input("Operator note (Enter for empty): ").strip()
    apply_physical_label(path, value, reason, note)


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="/home/nvidia/RL-Orchard/src/orchard_picker/config/rl_collection.yaml"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--label-later", action="store_true")
    parser.add_argument("--episodes", type=int, default=0, help="0 means repeat until Ctrl+C")
    arguments = parser.parse_args(args)
    try:
        config = load_collection_config(arguments.config)
    except (OSError, CollectionConfigError, yaml.YAMLError) as exc:
        print("Invalid collection config: {}".format(exc), file=sys.stderr)
        return 2
    dry_run = bool(arguments.dry_run or not _authorized())
    if arguments.plan_only:
        _print_plan(config)
        return 0
    if dry_run:
        print("DRY RUN: no motion/start/home/gripper service will be called.")
        _print_plan(config)
    else:
        print("ARMED COLLECTION: both environment gates are YES.")

    try:
        rclpy.init(args=None)
    except Exception as exc:
        print("ROS initialization failed: {}".format(exc), file=sys.stderr)
        return 1
    node = None
    try:
        node = CollectionOrchestrator(config, dry_run)
        if dry_run:
            node._pump_until(lambda: False, 2.0)
            print("Read-only ROS readiness:", json.dumps(node.readiness(), sort_keys=True))
            return 0
        completed = 0
        while arguments.episodes <= 0 or completed < arguments.episodes:
            print("-" * 48)
            node.set_collection_state("WAIT_OPERATOR")
            region_counts = load_manifest_counts(
                config.get("manifest_file", ""), "apple_position_bin"
            )
            suggestion = suggest_least_sampled_region(
                config.get("apple_position_bins", {}).get("suggested_regions", []),
                region_counts,
            )
            if suggestion:
                print("Suggested next apple region:", suggestion)
            input("Please place apple. Press ENTER when ready (Ctrl+C to stop). ")
            node.set_collection_state("CHECK_SYSTEM_READY")
            preliminary_ready, preliminary = node.wait_ready()
            print("Pre-initialization readiness:", json.dumps(preliminary, sort_keys=True))
            if not preliminary_ready:
                raise RuntimeError("system not ready; initialization motion not requested")
            node.set_collection_state("SELECT_INITIAL_POSE")
            pose, offset, counts = _choose_pose(config, node.selection_index)
            print("Suggested init pose:", pose["id"])
            node.set_collection_state("MOVE_INITIAL_POSE")
            node.set_recorder_enabled(False)
            node.execute_initial_pose(pose, offset)
            node.set_collection_state("WAIT_STABLE")
            # The configured service must return only after its own verified
            # completion. We do not invent a second stability tolerance here.
            node.set_collection_state("CHECK_CAMERA_AND_TARGET")
            ready, status = node.wait_ready()
            print("Camera: {}\nDepth: {}\nRobot: {}\nTarget: {}".format(
                "OK" if status["rgb"] else "NOT READY",
                "OK" if status["depth"] else "NOT READY",
                "OK" if status["robot"] else "NOT READY",
                "OK" if status["target"] else "NOT READY",
            ))
            if not ready:
                raise RuntimeError("system readiness timeout; expert grasp not started")
            metadata = node.publish_metadata(pose, offset)
            print("Apple bin:", metadata["apple_position_bin"])
            node.set_recorder_enabled(True)
            node.set_collection_state("READY_TO_START")
            node.set_collection_state("RUN_EXISTING_EXPERT_GRASP")
            finalized = node.run_expert_episode()
            path = str(finalized["file"])
            print("saved:", path)
            quality = finalized.get("quality", {})
            print("QC:", "PASS" if quality.get("quality_pass") else "FAIL")
            if quality.get("quality_reason"):
                print("QC reason:", quality["quality_reason"])
            node.set_collection_state("REQUEST_PHYSICAL_LABEL")
            _label_episode(path, arguments.label_later)
            node.set_collection_state("SAVE_MANIFEST")
            # Finalize/label helpers have already atomically updated manifest.
            completed += 1
            node.selection_index += 1
            node.set_collection_state("RETURN_NEXT_EPISODE")
    except KeyboardInterrupt:
        print("Collection stopped by operator.")
        return 0
    except Exception as exc:
        if node is not None:
            node.get_logger().error("Collection stopped safely: {}".format(exc))
        else:
            print("Collection node creation failed: {}".format(exc), file=sys.stderr)
        return 1
    finally:
        if (
            node is not None and not dry_run
            and node.recorder_disabled_by_orchestrator
            and rclpy.ok()
        ):
            try:
                node.set_recorder_enabled(True)
            except Exception as exc:
                node.get_logger().error(
                    "Could not restore recorder enable state: {}".format(exc)
                )
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
