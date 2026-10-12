#!/usr/bin/env python3
"""Read-only ROS 2 episode recorder for orchard imitation/RL datasets."""

import json
import math
import os
import re
import threading
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool, Trigger

try:
    from rclpy.qos import DurabilityPolicy
except ImportError:  # ROS 2 Foxy compatibility
    from rclpy.qos import QoSDurabilityPolicy as DurabilityPolicy

from orchard_picker.msg import ExpertAction, RobotState, TaskOutcome
from orchard_picker.rl_data.episode_writer import EpisodeWriter
from orchard_picker.rl_data.joint_state import (
    JointStateMappingError,
    estimate_rate_hz,
    normalize_joint_positions,
)
from orchard_picker.srv import GetArmState


def _latched_qos():
    qos = QoSProfile(depth=1)
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    return qos


def _stamp_ns(header):
    return int(header.stamp.sec) * 1000000000 + int(header.stamp.nanosec)


class RlDataRecorder(Node):
    """Observe existing ROS interfaces and record one HDF5 file per task."""

    def __init__(self):
        super().__init__("rl_data_recorder")
        self.enabled = bool(self._param("enabled", True))
        self.dataset_root = str(
            self._param(
                "dataset_root", "/home/nvidia/RL-Orchard/rl_dataset"
            )
        )
        self.record_hz = max(0.1, float(self._param("record_hz", 10.0)))
        self.max_sensor_age_ns = int(
            max(0.0, float(self._param("max_sensor_age_ms", 150.0))) * 1e6
        )
        self.max_robot_age_ns = int(
            max(0.0, float(self._param("max_robot_state_age_ms", 300.0)))
            * 1e6
        )
        self.robot_state_poll_hz = max(
            0.0, float(self._param("robot_state_poll_hz", 5.0))
        )
        self.save_rgb = bool(self._param("save_rgb", True))
        self.save_depth = bool(self._param("save_depth", True))
        self.save_teacher_info = bool(self._param("save_teacher_info", True))
        self.save_raw_command_debug = bool(
            self._param("save_raw_command_debug", False)
        )
        self.auto_start_on_active = bool(
            self._param("auto_start_on_active", True)
        )
        self.outcome_wait_ns = int(
            max(0.0, float(self._param("outcome_wait_ms", 250.0))) * 1e6
        )
        self.min_episode_samples = max(
            1, int(self._param("min_episode_samples", 5))
        )
        self.min_episode_duration_sec = max(
            0.0, float(self._param("min_episode_duration_sec", 0.5))
        )
        self.dataset_version = str(
            self._param("dataset_version", "apple_rl_v1.2")
        )
        self.joint_state_source = str(
            self._param("joint_state_source", "rm_json_bridge")
        ).strip().lower()
        self.joint_state_topic = str(
            self._param("joint_state_topic", "/rm_json_bridge/arm_state")
        )
        configured_joint_names = self._param(
            "joint_state_joint_names", []
        )
        self.joint_state_joint_names = tuple(
            str(name) for name in (configured_joint_names or [])
        )
        self.joint_state_nominal_rate_hz = float(
            self._param("joint_state_nominal_rate_hz", 0.0)
        )
        if self.joint_state_source not in ("rm_json_bridge", "sensor_msgs_joint_state"):
            raise ValueError(
                "joint_state_source must be rm_json_bridge or sensor_msgs_joint_state"
            )
        if self.joint_state_source == "sensor_msgs_joint_state" and len(
            self.joint_state_joint_names
        ) != 6:
            raise ValueError("sensor_msgs_joint_state requires six joint names")
        self.image_width = int(self._param("record_image_width", 640))
        self.image_height = int(self._param("record_image_height", 480))
        self.queue_size = max(4, int(self._param("sample_queue_size", 64)))
        self.compression = str(self._param("hdf5_compression", "lzf"))
        self.flush_every_n_samples = max(
            1, int(self._param("flush_every_n_samples", 20))
        )
        self.collection_metadata_max_age_ns = int(
            max(0.0, float(self._param("collection_metadata_max_age_sec", 30.0)))
            * 1e9
        )

        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("record_image_width/height must be positive")

        self.rgb_topic = str(self._param("rgb_topic", "/apple_vision/rgb"))
        self.depth_topic = str(
            self._param("depth_topic", "/apple_vision/aligned_depth")
        )
        self.camera_metadata_topic = str(
            self._param(
                "camera_metadata_topic", "/apple_vision/camera_metadata"
            )
        )
        self.robot_state_topic = str(
            self._param("robot_state_topic", "/rm_json_bridge/arm_state")
        )
        self.robot_state_service = str(
            self._param(
                "robot_state_service", "/rm_json_bridge/get_arm_state"
            )
        )
        self.task_state_topic = str(
            self._param("task_state_topic", "/pick_task_manager/state")
        )
        self.task_active_topic = str(
            self._param("task_active_topic", "/pick_task_manager/active")
        )
        self.task_outcome_topic = str(
            self._param("task_outcome_topic", "/pick_task_manager/outcome")
        )
        self.expert_action_topic = str(
            self._param(
                "expert_action_topic", "/pick_task_manager/expert_action"
            )
        )
        self.apple_camera_topic = str(
            self._param("apple_camera_topic", "/apple/target_info")
        )
        self.apple_base_topic = str(
            self._param(
                "apple_base_topic", "/pick_task_manager/target_base"
            )
        )
        self.raw_tx_topic = str(
            self._param("raw_tx_topic", "/rm_json_bridge/raw_tx")
        )
        self.collection_metadata_topic = str(self._param(
            "collection_metadata_topic", "/rl_collection/episode_metadata"
        ))
        self.recording_control_service = str(self._param(
            "recording_control_service", "/rl_data_recorder/set_enabled"
        ))
        self.finalized_topic = str(self._param(
            "finalized_topic", "/rl_data_recorder/finalized"
        ))
        self.prepare_episode_service = str(self._param(
            "prepare_episode_service", "/rl_data_recorder/prepare_episode"
        ))
        self.abort_episode_service = str(self._param(
            "abort_episode_service", "/rl_data_recorder/abort_episode"
        ))
        self.recording_state_topic = str(self._param(
            "recording_state_topic", "/rl_data_recorder/state"
        ))
        self.emergency_stop_topic = str(self._param(
            "emergency_stop_topic", "/rm_json_bridge/emergency_stop"
        ))

        self._lock = threading.RLock()
        self._image_parts = {}
        self._latest_image_pair = None
        self._latest_robot = None
        self._latest_joint = None
        self._joint_source_times = deque(maxlen=1000)
        self._latest_target_info = None
        self._latest_target_base = None
        self._camera_metadata = {}
        self._task_state = "UNKNOWN"
        self._task_active = False
        self._latest_outcome = None
        self._expert_events = []
        self._latest_raw_tx = ""
        self._recording = False
        self._episode_teacher = {}
        self._episode_id = None
        self._finish_deadline_ns = None
        self._last_robot_poll_ns = 0
        self._pending_robot_future = None
        self._warn_times = {}
        self._received_once = set()
        self._latest_collection_metadata = {}
        self._latest_collection_metadata_received_ns = 0
        self._episode_start_pending = False
        self._episode_prepared = False
        self._emergency_stop_info = None

        self._condition = threading.Condition()
        self._sample_queue = deque()
        self._control_queue = deque()
        self._writer_stop = False
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="rl_hdf5_writer",
            daemon=True,
        )

        os.makedirs(self.dataset_root, exist_ok=True)
        self._next_episode_number = self._find_next_episode_number()

        self.create_subscription(
            Image, self.rgb_topic, self._rgb_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            Image, self.depth_topic, self._depth_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            RobotState, self.robot_state_topic, self._robot_state_cb, 10
        )
        if self.joint_state_source == "sensor_msgs_joint_state":
            self.create_subscription(
                JointState, self.joint_state_topic, self._joint_state_cb,
                qos_profile_sensor_data,
            )
        self.create_subscription(
            String,
            self.camera_metadata_topic,
            self._camera_metadata_cb,
            _latched_qos(),
        )
        self.create_subscription(
            String, self.task_state_topic, self._task_state_cb, _latched_qos()
        )
        self.create_subscription(
            Bool, self.task_active_topic, self._task_active_cb, _latched_qos()
        )
        self.create_subscription(
            TaskOutcome,
            self.task_outcome_topic,
            self._task_outcome_cb,
            _latched_qos(),
        )
        self.create_subscription(
            ExpertAction, self.expert_action_topic, self._expert_action_cb, 50
        )
        self.create_subscription(
            String,
            self.apple_camera_topic,
            self._target_info_cb,
            _latched_qos(),
        )
        self.create_subscription(
            PointStamped,
            self.apple_base_topic,
            self._target_base_cb,
            _latched_qos(),
        )
        if self.save_raw_command_debug:
            self.create_subscription(
                String, self.raw_tx_topic, self._raw_tx_cb, 50
            )
        self.create_subscription(
            String, self.collection_metadata_topic,
            self._collection_metadata_cb, _latched_qos(),
        )
        self.create_subscription(
            Bool, self.emergency_stop_topic, self._emergency_stop_cb,
            _latched_qos(),
        )
        self._finalized_pub = self.create_publisher(
            String, self.finalized_topic, _latched_qos()
        )
        self._recording_state_pub = self.create_publisher(
            String, self.recording_state_topic, _latched_qos()
        )
        self._recording_control = self.create_service(
            SetBool, self.recording_control_service,
            self._set_recording_enabled_cb,
        )
        self._prepare_episode = self.create_service(
            Trigger, self.prepare_episode_service,
            self._prepare_episode_cb,
        )
        self._abort_episode = self.create_service(
            Trigger, self.abort_episode_service,
            self._abort_episode_cb,
        )

        self._state_client = self.create_client(
            GetArmState, self.robot_state_service
        )
        self._writer_thread.start()
        self._timer = self.create_timer(1.0 / self.record_hz, self._sample_tick)
        self.get_logger().info(
            "[RL Recorder] ready (enabled={}, record_hz={}, dataset_root={})".format(
                self.enabled, self.record_hz, self.dataset_root
            )
        )
        self._publish_recording_state("IDLE")

    def _param(self, name, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _find_next_episode_number(self):
        maximum = 0
        pattern = re.compile(r"^episode_(\d+)(?:\.tmp)?\.hdf5$")
        for name in os.listdir(self.dataset_root):
            match = pattern.match(name)
            if match:
                maximum = max(maximum, int(match.group(1)))
        return maximum + 1

    def _log_first(self, key, description):
        if key not in self._received_once:
            self._received_once.add(key)
            self.get_logger().info("[RL Recorder] {} received".format(description))

    def _rgb_cb(self, message):
        self._image_cb("rgb", message)
        self._log_first("rgb", "RGB")

    def _depth_cb(self, message):
        self._image_cb("depth", message)
        self._log_first("depth", "aligned depth")

    def _image_cb(self, kind, message):
        now_ns = time.monotonic_ns()
        stamp = _stamp_ns(message.header)
        with self._lock:
            entry = self._image_parts.setdefault(stamp, {})
            entry[kind] = (message, now_ns)
            if "rgb" in entry and "depth" in entry:
                self._latest_image_pair = (
                    entry["rgb"][0],
                    entry["depth"][0],
                    entry["rgb"][1],
                    entry["depth"][1],
                )
                self._image_parts.clear()
            elif len(self._image_parts) > 4:
                oldest = sorted(self._image_parts.keys())[:-4]
                for key in oldest:
                    self._image_parts.pop(key, None)

    def _robot_state_cb(self, message):
        self._cache_robot_message(message, time.monotonic_ns())
        self._log_first("robot", "robot state")

    def _cache_robot_message(self, message, received_ns):
        pose = message.tcp_pose
        value = {
            "joint_pos": np.asarray(message.joint_pos_rad, dtype=np.float32),
            "tcp_pose": np.asarray([
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ], dtype=np.float32),
            "error_code": int(message.error_code),
            "gripper_state": int(message.gripper_measured_state),
            "gripper_position": float(message.gripper_position),
            "gripper_force": float(message.gripper_force),
            "gripper_current": float(message.gripper_current),
            "gripper_moving": bool(message.gripper_moving),
            "gripper_measurement_valid": bool(
                message.gripper_measurement_valid
            ),
            "stamp_ns": _stamp_ns(message.header),
            "received_ns": int(received_ns),
        }
        with self._lock:
            self._latest_robot = value
            if self.joint_state_source == "rm_json_bridge":
                self._latest_joint = {
                    "joint_pos": value["joint_pos"].copy(),
                    "stamp_ns": value["stamp_ns"],
                    "received_ns": int(received_ns),
                    "source": "rm_json_bridge",
                }
                # Measure the Recorder's actual source callback cadence.  The
                # RobotState header is generated by rm_json_bridge after a
                # service-backed read and is not a physical servo-rate clock.
                self._joint_source_times.append(int(received_ns))

    def _joint_state_cb(self, message):
        received_ns = time.monotonic_ns()
        try:
            ordered = normalize_joint_positions(
                message.name, message.position, self.joint_state_joint_names
            )
        except (JointStateMappingError, TypeError, ValueError) as exc:
            self._warn_throttled(
                "joint_mapping", "invalid /joint_states mapping: {}".format(exc)
            )
            return
        stamp_ns = _stamp_ns(message.header)
        with self._lock:
            self._latest_joint = {
                "joint_pos": ordered,
                "stamp_ns": stamp_ns,
                "received_ns": int(received_ns),
                "source": "sensor_msgs_joint_state",
            }
            self._joint_source_times.append(int(received_ns))
            # RobotState remains the source of TCP/error/gripper.  Only replace
            # its joint vector when the configured high-rate source is active.
            if self._latest_robot is not None:
                self._latest_robot["joint_pos"] = ordered.copy()
                self._latest_robot["joint_state_timestamp_ns"] = stamp_ns
        self._log_first("joint_state", "joint state")

    def _camera_metadata_cb(self, message):
        try:
            value = json.loads(message.data)
        except (TypeError, ValueError) as exc:
            self.get_logger().warning(
                "Invalid camera metadata JSON: {}".format(exc)
            )
            return
        with self._lock:
            self._camera_metadata = value
        self._log_first("camera_metadata", "camera metadata")

    def _task_state_cb(self, message):
        should_start = False
        with self._lock:
            self._task_state = str(message.data)
            should_start = (
                self._episode_start_pending
                and self._task_active
                and self._episode_start_state_is_valid(self._task_state)
            )
            if should_start:
                self._episode_start_pending = False
        self._log_first("task_state", "task state")
        if should_start and self.enabled and self.auto_start_on_active:
            self._start_episode()

    def _emergency_stop_cb(self, message):
        if not bool(message.data):
            return
        with self._lock:
            if not self._recording:
                return
            self._emergency_stop_info = {
                "timestamp_ns": time.monotonic_ns(),
                "termination_type": "emergency_stop",
                "discard_reason": "robot_emergency_stop",
            }
        self.get_logger().warning(
            "[RL Recorder] emergency stop received; invalidating current episode"
        )
        self._finish_episode()

    def _task_outcome_cb(self, message):
        with self._lock:
            self._latest_outcome = {
                "success": bool(message.success),
                "terminal_state": str(message.terminal_state),
                "failure_reason": str(message.failure_reason),
                "physical_grasp_success": int(
                    message.physical_grasp_success
                ),
            }

    def _expert_action_cb(self, message):
        event = {
            "timestamp_ns": _stamp_ns(message.header),
            "phase": str(message.phase),
            "command_type": str(message.command_type),
            "target_tcp": list(message.target_tcp),
            "target_joint": list(message.target_joint_rad),
            "gripper_command": int(message.gripper_command),
            "speed": int(message.speed),
            "force": int(message.force),
        }
        with self._lock:
            if self._recording:
                self._expert_events.append(event)
                if len(self._expert_events) > 100:
                    self._expert_events = self._expert_events[-100:]

    def _target_info_cb(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError):
            return
        payload["received_ns"] = time.monotonic_ns()
        with self._lock:
            self._latest_target_info = payload
        self._log_first("teacher", "apple teacher info")

    def _target_base_cb(self, message):
        value = {
            "xyz": [message.point.x, message.point.y, message.point.z],
            "frame": message.header.frame_id or "base_link",
            "received_ns": time.monotonic_ns(),
        }
        with self._lock:
            self._latest_target_base = value

    def _raw_tx_cb(self, message):
        with self._lock:
            self._latest_raw_tx = str(message.data)

    def _collection_metadata_cb(self, message):
        try:
            payload = json.loads(message.data)
        except (TypeError, ValueError) as exc:
            self.get_logger().warning(
                "[RL Recorder] invalid collection metadata: {}".format(exc)
            )
            return
        if not isinstance(payload, dict):
            self.get_logger().warning(
                "[RL Recorder] collection metadata must be a JSON object"
            )
            return
        with self._lock:
            self._latest_collection_metadata = payload
            self._latest_collection_metadata_received_ns = time.monotonic_ns()
        self._log_first("collection_metadata", "collection metadata")

    def _set_recording_enabled_cb(self, request, response):
        """Gate recording only; this service has no robot-control side effect."""
        with self._lock:
            if self._recording and not bool(request.data):
                response.success = False
                response.message = "cannot disable during an active episode"
                return response
            self.enabled = bool(request.data)
        response.success = True
        response.message = "recording enabled={}".format(self.enabled)
        self.get_logger().info("[RL Recorder] {}".format(response.message))
        return response

    def _task_active_cb(self, message):
        active = bool(message.data)
        with self._lock:
            previous = self._task_active
            self._task_active = active
        self._log_first("active", "task active")
        if not self.enabled or not self.auto_start_on_active:
            return
        if active and self._recording:
            self._publish_recording_state("RECORDING")
            return
        if active and not previous:
            with self._lock:
                pending_finish = self._finish_deadline_ns is not None
            if pending_finish:
                self._finish_episode()
            with self._lock:
                state = self._task_state
                start_now = self._episode_start_state_is_valid(state)
                self._episode_start_pending = not start_now
            if start_now:
                self._start_episode()
            else:
                self.get_logger().info(
                    "[RL Recorder] active=true while task state is {}; "
                    "waiting for a real task phase before starting episode".format(
                        state or "UNKNOWN"
                    )
                )
        elif previous and not active:
            with self._lock:
                self._episode_start_pending = False
                self._finish_deadline_ns = (
                    time.monotonic_ns() + self.outcome_wait_ns
                )

    @staticmethod
    def _episode_start_state_is_valid(state):
        """Allow starts only in audited execution phases, never IDLE/manual."""
        return str(state or "").strip().upper() in {
            "OPEN_GRIPPER", "MOVE_PRE_GRASP"
        }

    def _snapshot_teacher(self):
        info = self._latest_target_info or {}
        base = self._latest_target_base or {}
        camera_xyz = info.get("camera_xyz_m", [])
        base_xyz = base.get("xyz", [])
        valid = (
            bool(info.get("target_valid", bool(camera_xyz)))
            and len(camera_xyz) == 3
            and len(base_xyz) == 3
        )
        return {
            "apple_camera_xyz": list(camera_xyz),
            "apple_base_xyz": list(base_xyz),
            "apple_id": int(info.get("id", -1)),
            "confidence": float(info.get("confidence", np.nan)),
            "bbox": list(info.get("bbox_xyxy", [-1, -1, -1, -1])),
            "target_valid": valid,
            "camera_frame": str(info.get("camera_frame", "camera_link")),
            "base_frame": str(base.get("frame", "base_link")),
        }

    def _prepare_episode_cb(self, _request, response):
        with self._lock:
            if self._recording:
                response.success = False
                response.message = "an episode is already recording"
                return response
            if self._task_active:
                response.success = False
                response.message = "cannot prepare while task is active"
                return response
        if not self.enabled:
            response.success = False
            response.message = "recorder is disabled"
            return response
        self._start_episode(force=True)
        with self._lock:
            prepared = bool(self._recording and self._episode_prepared)
        response.success = prepared
        response.message = "episode armed; call run_once next" if prepared else "episode preparation failed"
        return response

    def _abort_episode_cb(self, _request, response):
        """Finalize the current episode as INVALID without touching the robot."""
        with self._lock:
            if not self._recording:
                response.success = False
                response.message = "no active episode"
                return response
            self._latest_outcome = {
                "success": False,
                "physical_grasp_success": -1,
                "failure_reason": "operator_stopped_recording",
                "terminal_state": self._task_state,
                "episode_valid": False,
                "discarded": True,
                "termination_type": "operator_abort",
                "discard_reason": "recording_stopped_by_operator",
            }
            self._finish_deadline_ns = None
        self._finish_episode()
        response.success = True
        response.message = "active episode finalized as INVALID"
        return response

    def _publish_recording_state(self, state):
        message = String(data=str(state))
        self._recording_state_pub.publish(message)
        self.get_logger().info("[RL Recorder] state={}".format(state))

    def _start_episode(self, force=False):
        with self._lock:
            if self._recording:
                return
            if force:
                if self._task_active:
                    self._warn_throttled(
                        "prepare_active", "episode preparation rejected while task is active"
                    )
                    return
            elif not self._task_active or not self._episode_start_state_is_valid(
                self._task_state
            ):
                self._warn_throttled(
                    "episode_gate",
                    "episode start rejected outside a real task phase: {}".format(
                        self._task_state or "UNKNOWN"
                    ),
                )
                return
            number = self._next_episode_number
            self._next_episode_number += 1
            self._episode_id = "episode_{:06d}".format(number)
            self._episode_teacher = self._snapshot_teacher()
            self._latest_outcome = None
            self._expert_events = []
            self._finish_deadline_ns = None
            self._emergency_stop_info = None
            self._episode_prepared = bool(force and not self._task_active)
            self._recording = True
            self._joint_source_times.clear()
            metadata = dict(self._camera_metadata)
            metadata.update({
                "start_time": datetime.now().astimezone().isoformat(),
                "record_hz": self.record_hz,
                "dataset_version": self.dataset_version,
                "robot_model": "RML63-B",
                "tcp_frame": "base_link",
                "tcp_child_frame": "robot_end",
                "apple_camera_frame": self._episode_teacher.get(
                    "camera_frame", "camera_link"
                ),
                "apple_base_frame": self._episode_teacher.get(
                    "base_frame", "base_link"
                ),
                "max_sensor_age_ms": self.max_sensor_age_ns / 1e6,
                "max_robot_state_age_ms": self.max_robot_age_ns / 1e6,
                "joint_state_source": self.joint_state_source,
                "joint_state_topic": self.joint_state_topic,
                "joint_state_nominal_rate_hz": self.joint_state_nominal_rate_hz,
                "source_joint_unit": "radian",
                "min_episode_samples": self.min_episode_samples,
                "min_episode_duration_sec": self.min_episode_duration_sec,
            })
            initial = self._snapshot_initial_state()
            metadata.update(initial)
            metadata_age = (
                time.monotonic_ns() - self._latest_collection_metadata_received_ns
            )
            if (
                self._latest_collection_metadata
                and metadata_age <= self.collection_metadata_max_age_ns
            ):
                metadata.update(dict(self._latest_collection_metadata))
            self._latest_collection_metadata = {}
            self._latest_collection_metadata_received_ns = 0
            episode_id = self._episode_id
        self._enqueue_control(("start", episode_id, metadata))
        self._publish_recording_state(
            "ARMED" if self._episode_prepared else "RECORDING"
        )
        self.get_logger().info(
            "[RL Recorder] {} started".format(episode_id)
        )

    def _snapshot_initial_state(self):
        """Capture fresh robot and teacher values at the actual episode start."""
        now_ns = time.monotonic_ns()
        robot = self._latest_robot
        joint = self._latest_joint
        result = {
            "initial_joint_pos": [],
            "initial_tcp_pose": [],
            "initial_apple_camera_xyz": list(
                self._episode_teacher.get("apple_camera_xyz", [])
            ),
            "initial_apple_base_xyz": list(
                self._episode_teacher.get("apple_base_xyz", [])
            ),
            "initial_apple_id": int(self._episode_teacher.get("apple_id", -1)),
            "initial_apple_confidence": float(
                self._episode_teacher.get("confidence", np.nan)
            ),
            "initial_apple_bbox": list(
                self._episode_teacher.get("bbox", [-1, -1, -1, -1])
            ),
        }
        if (
            joint is not None
            and now_ns - int(joint["received_ns"]) <= self.max_robot_age_ns
            and np.asarray(joint["joint_pos"]).shape == (6,)
            and np.all(np.isfinite(joint["joint_pos"]))
        ):
            result["initial_joint_pos"] = np.asarray(
                joint["joint_pos"], dtype=np.float32
            ).tolist()
        if (
            robot is not None
            and now_ns - int(robot["received_ns"]) <= self.max_robot_age_ns
            and np.asarray(robot["tcp_pose"]).shape == (7,)
            and np.all(np.isfinite(robot["tcp_pose"]))
        ):
            result["initial_tcp_pose"] = np.asarray(
                robot["tcp_pose"], dtype=np.float32
            ).tolist()
        return result

    def _finish_episode(self):
        with self._lock:
            if not self._recording:
                return
        self._capture_sample(force=True)
        with self._lock:
            outcome = dict(self._latest_outcome or {})
            outcome.setdefault("success", False)
            outcome.setdefault("terminal_state", self._task_state)
            if not self._latest_outcome:
                outcome.setdefault(
                    "failure_reason", "task outcome topic was unavailable"
                )
                outcome.setdefault("physical_grasp_success", -1)
            emergency = dict(self._emergency_stop_info or {})
            if emergency:
                outcome.update({
                    "success": False,
                    "episode_valid": False,
                    "discarded": True,
                    "termination_type": emergency.get(
                        "termination_type", "emergency_stop"
                    ),
                    "discard_reason": emergency.get(
                        "discard_reason", "robot_emergency_stop"
                    ),
                    "emergency_stop_timestamp_ns": int(
                        emergency.get("timestamp_ns", time.monotonic_ns())
                    ),
                    "failure_reason": "robot_emergency_stop",
                })
            outcome["end_time"] = datetime.now().astimezone().isoformat()
            outcome["joint_state_observed_rate_hz"] = estimate_rate_hz(
                self._joint_source_times
            )
            self._recording = False
            self._episode_prepared = False
            self._emergency_stop_info = None
            self._episode_id = None
            self._finish_deadline_ns = None
        self._enqueue_control(("finalize", outcome))
        self._publish_recording_state("FINALIZING")

    def _sample_tick(self):
        with self._lock:
            recording = self._recording
            finish_deadline_ns = self._finish_deadline_ns
        if not recording:
            return
        if finish_deadline_ns is not None:
            if time.monotonic_ns() >= finish_deadline_ns:
                self._finish_episode()
            return
        self._maybe_poll_robot_state()
        self._capture_sample(force=False)

    def _maybe_poll_robot_state(self):
        if self.robot_state_poll_hz <= 0.0:
            return
        now_ns = time.monotonic_ns()
        with self._lock:
            robot = self._latest_robot
        robot_age = now_ns - robot["received_ns"] if robot else math.inf
        min_period_ns = int(1e9 / self.robot_state_poll_hz)
        if robot_age <= self.max_robot_age_ns:
            return
        if now_ns - self._last_robot_poll_ns < min_period_ns:
            return
        if self._pending_robot_future is not None:
            if not self._pending_robot_future.done():
                return
            self._pending_robot_future = None
        if not self._state_client.service_is_ready():
            return
        self._last_robot_poll_ns = now_ns
        future = self._state_client.call_async(GetArmState.Request())
        self._pending_robot_future = future
        future.add_done_callback(self._fallback_robot_state_done)

    def _fallback_robot_state_done(self, future):
        try:
            response = future.result()
        except Exception as exc:
            self._warn_throttled(
                "robot_poll", "robot state fallback query failed: {}".format(exc)
            )
            return
        if response is None or not response.success or len(response.joint_deg) != 6:
            return
        pose = response.pose
        value = {
            "joint_pos": np.asarray(
                [math.radians(v) for v in response.joint_deg], dtype=np.float32
            ),
            "tcp_pose": np.asarray([
                pose.position.x,
                pose.position.y,
                pose.position.z,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ], dtype=np.float32),
            "error_code": int(response.err),
            "gripper_state": -1,
            "gripper_position": np.nan,
            "gripper_force": np.nan,
            "gripper_current": np.nan,
            "gripper_moving": False,
            "gripper_measurement_valid": False,
            "stamp_ns": self.get_clock().now().nanoseconds,
            "received_ns": time.monotonic_ns(),
        }
        with self._lock:
            self._latest_robot = value
            if self.joint_state_source == "rm_json_bridge":
                self._latest_joint = {
                    "joint_pos": value["joint_pos"].copy(),
                    "stamp_ns": value["stamp_ns"],
                    "received_ns": int(value["received_ns"]),
                    "source": "rm_json_bridge",
                }
                self._joint_source_times.append(int(value["received_ns"]))

    def _capture_sample(self, force):
        now_ns = time.monotonic_ns()
        with self._lock:
            if not self._recording:
                return
            pair = self._latest_image_pair
            robot = None if self._latest_robot is None else dict(
                self._latest_robot
            )
            joint_state = None if self._latest_joint is None else dict(
                self._latest_joint
            )
            teacher = dict(self._episode_teacher)
            state = self._task_state
            active = self._task_active
            events = list(self._expert_events)
            self._expert_events = []
            raw_tx = self._latest_raw_tx

        # A prepared episode owns its writer before run_once, but samples
        # begin only after a real task phase.  This prevents one-sample IDLE or
        # manual-movement episodes from entering the dataset.
        if not active or not self._episode_start_state_is_valid(state):
            return

        rgb_valid = False
        depth_valid = False
        rgb_stamp = -1
        depth_stamp = -1
        rgb = np.zeros(
            (self.image_height, self.image_width, 3), dtype=np.uint8
        )
        depth = np.zeros(
            (self.image_height, self.image_width), dtype=np.uint16
        )
        rgb_age_ms = np.nan
        depth_age_ms = np.nan
        rgb_depth_delta_ms = np.nan
        if pair is not None:
            rgb_message, depth_message, rgb_received, depth_received = pair
            rgb_valid = now_ns - rgb_received <= self.max_sensor_age_ns
            depth_valid = now_ns - depth_received <= self.max_sensor_age_ns
            rgb_age_ms = float(now_ns - rgb_received) / 1e6
            depth_age_ms = float(now_ns - depth_received) / 1e6
            rgb_stamp = _stamp_ns(rgb_message.header)
            depth_stamp = _stamp_ns(depth_message.header)
            if rgb_stamp >= 0 and depth_stamp >= 0:
                rgb_depth_delta_ms = abs(rgb_stamp - depth_stamp) / 1e6
            if self.save_rgb and rgb_valid:
                try:
                    rgb = self._decode_rgb(rgb_message)
                except (TypeError, ValueError) as exc:
                    rgb_valid = False
                    self._warn_throttled("rgb_decode", "RGB decode failed: {}".format(exc))
            if self.save_depth and depth_valid:
                try:
                    depth = self._decode_depth(depth_message)
                except (TypeError, ValueError) as exc:
                    depth_valid = False
                    self._warn_throttled(
                        "depth_decode", "depth decode failed: {}".format(exc)
                    )
        if not self.save_rgb:
            rgb_valid = False
        if not self.save_depth:
            depth_valid = False

        joint_valid = bool(
            joint_state is not None
            and now_ns - joint_state["received_ns"] <= self.max_robot_age_ns
            and np.asarray(joint_state["joint_pos"]).shape == (6,)
            and np.all(np.isfinite(joint_state["joint_pos"]))
        )
        joint_age_ms = (
            float(now_ns - joint_state["received_ns"]) / 1e6
            if joint_state is not None else np.nan
        )
        robot_valid = bool(
            robot is not None
            and now_ns - robot["received_ns"] <= self.max_robot_age_ns
            and robot["tcp_pose"].shape == (7,)
            and np.all(np.isfinite(robot["tcp_pose"]))
            and joint_valid
        )
        robot_age_ms = (
            float(now_ns - robot["received_ns"]) / 1e6
            if robot is not None else np.nan
        )
        if robot_valid:
            tcp = robot["tcp_pose"]
            robot_stamp = robot["stamp_ns"]
            gripper_state = robot["gripper_state"]
            gripper_measurement_valid = robot["gripper_measurement_valid"]
            gripper_position = robot["gripper_position"]
            gripper_force = robot["gripper_force"]
            gripper_current = robot["gripper_current"]
            gripper_moving = robot["gripper_moving"]
            robot_error_code = robot["error_code"]
        else:
            tcp = np.full((7,), np.nan, dtype=np.float32)
            robot_stamp = -1
            gripper_state = -1
            gripper_measurement_valid = False
            gripper_position = np.nan
            gripper_force = np.nan
            gripper_current = np.nan
            gripper_moving = False
            robot_error_code = 0
        if joint_valid:
            joint = np.asarray(joint_state["joint_pos"], dtype=np.float32)
            joint_stamp = int(joint_state.get("stamp_ns", -1))
        else:
            joint = np.full((6,), np.nan, dtype=np.float32)
            joint_stamp = -1

        if not rgb_valid:
            self._warn_throttled("rgb_stale", "RGB stale or unavailable")
        if not depth_valid:
            self._warn_throttled(
                "depth_stale", "aligned depth stale or unavailable"
            )
        if not robot_valid:
            self._warn_throttled(
                "robot_stale", "robot state stale or unavailable"
            )
        if not joint_valid:
            self._warn_throttled(
                "joint_stale", "joint state stale, invalid, or unavailable"
            )

        sample = {
            "timestamp_ns": now_ns,
            "rgb_timestamp_ns": rgb_stamp,
            "depth_timestamp_ns": depth_stamp,
            "robot_state_timestamp_ns": robot_stamp,
            "joint_state_timestamp_ns": joint_stamp,
            "rgb_age_ms": rgb_age_ms,
            "depth_age_ms": depth_age_ms,
            "robot_state_age_ms": robot_age_ms,
            "joint_state_age_ms": joint_age_ms,
            "rgb_depth_delta_ms": rgb_depth_delta_ms,
            "rgb": rgb,
            "depth": depth,
            "joint_pos": joint,
            "tcp_pose": tcp,
            "gripper_state": gripper_state,
            "gripper_position": gripper_position,
            "gripper_force": gripper_force,
            "gripper_current": gripper_current,
            "gripper_moving": gripper_moving,
            "gripper_measurement_valid": gripper_measurement_valid,
            "robot_error_code": robot_error_code,
            "task_state": state,
            "task_active": active,
            "teacher": teacher if self.save_teacher_info else {},
            "expert_events": events,
            "raw_tx": raw_tx,
            "rgb_valid": rgb_valid,
            "depth_valid": depth_valid,
            "robot_valid": robot_valid,
            "joint_valid": joint_valid,
            "joint_state_source": (
                joint_state.get("source", self.joint_state_source)
                if joint_state is not None else self.joint_state_source
            ),
        }
        if not self._enqueue_sample(sample):
            self._warn_throttled(
                "queue_full", "sample queue full; dropping newest sample"
            )

    def _decode_rgb(self, message):
        if message.encoding not in ("bgr8", "rgb8"):
            raise ValueError("unsupported RGB encoding {}".format(message.encoding))
        row = np.frombuffer(message.data, dtype=np.uint8).reshape(
            int(message.height), int(message.step)
        )
        image = row[:, : int(message.width) * 3].reshape(
            int(message.height), int(message.width), 3
        )
        if message.encoding == "bgr8":
            image = image[:, :, ::-1]
        if image.shape[:2] != (self.image_height, self.image_width):
            image = cv2.resize(
                image,
                (self.image_width, self.image_height),
                interpolation=cv2.INTER_AREA,
            )
        return np.ascontiguousarray(image, dtype=np.uint8)

    def _decode_depth(self, message):
        if message.encoding not in ("16UC1", "mono16"):
            raise ValueError(
                "unsupported depth encoding {}".format(message.encoding)
            )
        row_words = int(message.step) // 2
        depth = np.frombuffer(message.data, dtype=np.uint16).reshape(
            int(message.height), row_words
        )[:, : int(message.width)]
        if bool(message.is_bigendian) != (np.dtype(np.uint16).byteorder == ">"):
            depth = depth.byteswap()
        if depth.shape != (self.image_height, self.image_width):
            depth = cv2.resize(
                depth,
                (self.image_width, self.image_height),
                interpolation=cv2.INTER_NEAREST,
            )
        return np.ascontiguousarray(depth, dtype=np.uint16)

    def _warn_throttled(self, key, message, period_sec=5.0):
        now = time.monotonic()
        previous = self._warn_times.get(key, 0.0)
        if now - previous >= period_sec:
            self._warn_times[key] = now
            self.get_logger().warning("[RL Recorder] {}".format(message))

    def _enqueue_sample(self, sample):
        with self._condition:
            if len(self._sample_queue) >= self.queue_size:
                return False
            self._sample_queue.append(sample)
            self._condition.notify()
            return True

    def _enqueue_control(self, command):
        with self._condition:
            self._control_queue.append(command)
            self._condition.notify()

    def _drain_samples(self, writer):
        while True:
            with self._condition:
                if not self._sample_queue:
                    return
                sample = self._sample_queue.popleft()
            writer.append(sample)

    def _writer_loop(self):
        writer = None
        while True:
            with self._condition:
                while not self._control_queue and not self._sample_queue:
                    self._condition.wait()
                command = (
                    self._control_queue.popleft()
                    if self._control_queue
                    else None
                )
                sample = (
                    None
                    if command is not None or not self._sample_queue
                    else self._sample_queue.popleft()
                )
            try:
                if command is not None:
                    kind = command[0]
                    if kind == "start":
                        if writer is not None:
                            writer.close_incomplete()
                        writer = EpisodeWriter(
                            dataset_root=self.dataset_root,
                            episode_id=command[1],
                            image_height=self.image_height,
                            image_width=self.image_width,
                            metadata=command[2],
                            compression=self.compression,
                            save_raw_command_debug=self.save_raw_command_debug,
                            flush_every_n_samples=self.flush_every_n_samples,
                        )
                    elif kind == "finalize":
                        if writer is not None:
                            self._drain_samples(writer)
                            path = writer.finalize(command[1])
                            self.get_logger().info(
                                "[RL Recorder] {} finalized\n"
                                "samples: {}\n"
                                "duration: {:.3f} s\n"
                                "success: {}\n"
                                "terminal_state: {}\n"
                                "file: {}".format(
                                    writer.episode_id,
                                    writer.sample_count,
                                    writer.duration_sec,
                                    bool(command[1].get("success", False)),
                                    command[1].get("terminal_state", "UNKNOWN"),
                                    path,
                                )
                            )
                            state = (
                                "VALID"
                                if bool((writer.manifest_row or {}).get(
                                    "episode_valid", 0
                                ))
                                else "INVALID"
                            )
                            self._publish_recording_state(state)
                            finalized = String()
                            finalized.data = json.dumps({
                                "episode_id": writer.episode_id,
                                "file": path,
                                "sample_count": writer.sample_count,
                                "duration_sec": writer.duration_sec,
                                "quality": writer.quality_result or {},
                            }, sort_keys=True)
                            self._finalized_pub.publish(finalized)
                            writer = None
                    elif kind == "shutdown":
                        if writer is not None:
                            self._drain_samples(writer)
                            writer.close_incomplete()
                        return
                elif sample is not None and writer is not None:
                    writer.append(sample)
            except Exception as exc:
                self.get_logger().error(
                    "[RL Recorder] writer error; temporary file retained: {}".format(
                        exc
                    )
                )
                if writer is not None:
                    try:
                        writer.close_incomplete()
                    except Exception as close_exc:
                        self.get_logger().error(
                            "[RL Recorder] failed to close temporary file: {}".format(
                                close_exc
                            )
                        )
                writer = None

    def destroy_node(self):
        with self._condition:
            self._control_queue.append(("shutdown",))
            self._condition.notify()
        if self._writer_thread.is_alive():
            self._writer_thread.join(timeout=10.0)
        if self._writer_thread.is_alive():
            self.get_logger().error(
                "[RL Recorder] writer thread did not stop within 10 seconds"
            )
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RlDataRecorder()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
