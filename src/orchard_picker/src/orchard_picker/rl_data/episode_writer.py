"""Streaming, crash-tolerant HDF5 episode writer."""

import os
from datetime import datetime

import h5py
import numpy as np

from orchard_picker.rl_data.action_utils import executed_delta_tcp
from orchard_picker.rl_data.schema import (
    CONTROL_LEARNABLE_POLICY,
    command_source_for_type,
    command_source_mapping_json,
    control_mode_for_phase,
    control_mode_mapping_json,
    TASK_STATE_MAPPING,
    task_state_mapping_json,
)


class EpisodeWriter:
    """Append fixed-rate samples to one temporary HDF5 episode."""

    # This matches the audited PickTaskManager motion_position_tolerance_m
    # (0.005 m).  It is used only to label completion of the learnable
    # pre-grasp segment; it does not affect robot motion or safety checks.
    PREGRASP_REACHED_TOLERANCE_M = 0.005

    def __init__(
        self,
        dataset_root,
        episode_id,
        image_height,
        image_width,
        metadata,
        compression="lzf",
        save_raw_command_debug=False,
        flush_every_n_samples=20,
    ):
        self.dataset_root = os.path.abspath(os.path.expanduser(dataset_root))
        self.episode_id = str(episode_id)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.compression = compression if compression in ("lzf", "gzip") else None
        self.save_raw_command_debug = bool(save_raw_command_debug)
        self.flush_every_n_samples = max(1, int(flush_every_n_samples))
        self.sample_count = 0
        self.state_mapping = dict(TASK_STATE_MAPPING)
        self.datasets = {}
        self._closed = False
        self.duration_sec = 0.0
        self.quality_result = None
        self.manifest_row = None
        self.expert_event_count = 0
        self.event_datasets = {}
        self._pre_grasp_target = None
        self._grasp_target = None
        self._pre_grasp_goal = None
        self._grasp_goal = None
        self._approach_vector = None
        self._pregrasp_offset_m = np.nan
        self._goal_frozen = False
        self._reached_pregrasp = False
        self.min_episode_samples = max(
            1, self._safe_int(metadata.get("min_episode_samples", 5), 5)
        )
        self.min_episode_duration_sec = max(
            0.0,
            self._safe_float(
                metadata.get("min_episode_duration_sec", 0.5), 0.5
            ),
        )

        os.makedirs(self.dataset_root, exist_ok=True)
        basename = "{}.hdf5".format(self.episode_id)
        self.final_path = os.path.join(self.dataset_root, basename)
        self.temp_path = os.path.join(
            self.dataset_root, "{}.tmp.hdf5".format(self.episode_id)
        )
        if os.path.exists(self.final_path) or os.path.exists(self.temp_path):
            raise FileExistsError("episode path already exists: {}".format(basename))

        self.file = h5py.File(self.temp_path, "x")
        self._write_initial_metadata(metadata)
        self._create_datasets()
        self.file.flush()

    def _write_initial_metadata(self, metadata):
        attrs = self.file.attrs
        attrs["episode_id"] = self.episode_id
        attrs["complete"] = False
        attrs["episode_valid"] = False
        attrs["discarded"] = False
        attrs["termination_type"] = "incomplete"
        attrs["discard_reason"] = ""
        attrs["emergency_stop_timestamp_ns"] = -1
        attrs["start_time"] = str(
            metadata.get("start_time", datetime.now().astimezone().isoformat())
        )
        attrs["record_hz"] = float(metadata.get("record_hz", 10.0))
        attrs["robot_model"] = str(metadata.get("robot_model", "RML63-B"))
        attrs["camera_model"] = str(metadata.get("camera_model", "unknown"))
        attrs["camera_serial"] = str(metadata.get("camera_serial", "unknown"))
        attrs["rgb_resolution"] = np.asarray(
            [self.image_width, self.image_height], dtype=np.int32
        )
        attrs["depth_resolution"] = np.asarray(
            [self.image_width, self.image_height], dtype=np.int32
        )
        attrs["depth_scale_m"] = float(metadata.get("depth_scale_m", np.nan))
        attrs["depth_unit"] = str(metadata.get("depth_unit", "device_unit"))
        attrs["source_joint_unit"] = str(
            metadata.get("source_joint_unit", "degree")
        )
        attrs["stored_joint_unit"] = "radian"
        attrs["joint_unit"] = "radian"
        attrs["tcp_position_unit"] = "meter"
        attrs["tcp_orientation"] = "quaternion_xyzw"
        attrs["tcp_frame"] = str(metadata.get("tcp_frame", "base_link"))
        attrs["tcp_child_frame"] = str(
            metadata.get("tcp_child_frame", "robot_end")
        )
        attrs["apple_camera_frame"] = str(
            metadata.get("apple_camera_frame", "camera_link")
        )
        attrs["apple_base_frame"] = str(
            metadata.get("apple_base_frame", "base_link")
        )
        attrs["dataset_version"] = str(
            metadata.get("dataset_version", "apple_rl_v1.2")
        )
        attrs["dataset_schema_version"] = str(
            metadata.get("dataset_schema_version", "apple_rl_v1.2_sac_goal")
        )
        attrs["software_version"] = str(
            metadata.get("software_version", "orchard_picker-0.2.0")
        )
        attrs["physical_grasp_success"] = -1
        attrs["operator_note"] = ""
        attrs["success_semantics"] = "pick state machine completed"
        attrs["physical_grasp_success_encoding"] = "-1=unknown,0=false,1=true"
        attrs["gripper_state_encoding"] = "-1=unknown,0=open,1=closed"
        attrs["teacher_target_tcp_child_frame"] = "robot_end"
        attrs["control_mode_mapping"] = control_mode_mapping_json()
        attrs["command_source_mapping"] = command_source_mapping_json()
        attrs["joint_action_unit"] = "rad"
        attrs["joint_state_source"] = str(
            metadata.get("joint_state_source", "unknown")
        )
        attrs["joint_state_topic"] = str(
            metadata.get("joint_state_topic", "")
        )
        attrs["joint_state_nominal_rate_hz"] = self._safe_float(
            metadata.get("joint_state_nominal_rate_hz", np.nan), np.nan
        )
        attrs["joint_state_observed_rate_hz"] = np.nan
        attrs["joint_state_observed_rate_semantics"] = (
            "rate of source callback timestamps, not recorder sample timestamps"
        )
        attrs["executed_delta_joint_semantics"] = "q[t+1] - q[t]"
        attrs["policy_action_semantics"] = (
            "valid executed joint transition during MOVE_PRE_GRASP"
        )
        attrs["sac_state_semantics"] = (
            "joint_pos(6), tcp_position_xyz(3), target_relative_xyz(3)"
        )
        attrs["sac_action_semantics"] = "executed_delta_joint[t] = q[t+1] - q[t]"
        attrs["goal_frame"] = "base_link"
        attrs["goal_semantics"] = (
            "frozen Expert MOVE_PRE_GRASP pregrasp pose; optional MOVE_GRASP grasp pose"
        )
        attrs["approach_vector_semantics"] = (
            "unit vector from pregrasp position to grasp position"
        )
        attrs["reward_version"] = "unassigned"
        attrs["reached_pregrasp"] = False
        attrs["initial_pose_id"] = str(metadata.get("initial_pose_id", "unknown"))
        attrs["initialization_mode"] = str(
            metadata.get("initialization_mode", "unknown")
        )
        attrs["initialization_seed"] = self._safe_int(
            metadata.get("initialization_seed", -1), -1
        )
        attrs["initialization_timestamp"] = str(
            metadata.get("initialization_timestamp", "")
        )
        attrs["initialization_offset_unit"] = str(
            metadata.get("initialization_offset_unit", "unknown")
        )
        attrs["apple_position_bin"] = str(
            metadata.get("apple_position_bin", "DISABLED")
        )
        attrs["initial_pose_bin"] = str(
            metadata.get("initial_pose_bin", "DISABLED")
        )
        attrs["max_sensor_age_ms"] = float(
            metadata.get("max_sensor_age_ms", np.nan)
        )
        attrs["max_robot_state_age_ms"] = float(
            metadata.get("max_robot_state_age_ms", np.nan)
        )
        attrs["min_episode_samples"] = int(self.min_episode_samples)
        attrs["min_episode_duration_sec"] = float(
            self.min_episode_duration_sec
        )
        attrs["state_mapping"] = task_state_mapping_json()

        initial_joint = self._metadata_vector(
            metadata.get("initial_joint_pos", []), 6
        )
        initial_tcp = self._metadata_vector(
            metadata.get("initial_tcp_pose", []), 7
        )
        initial_offset = self._metadata_vector(
            metadata.get("initialization_offset", []), 6
        )
        attrs["initial_joint_pos"] = initial_joint
        attrs["initial_tcp_pose"] = initial_tcp
        attrs["initialization_offset"] = initial_offset
        attrs["initial_apple_camera_xyz"] = self._metadata_vector(
            metadata.get("initial_apple_camera_xyz", []), 3
        )
        attrs["initial_apple_base_xyz"] = self._metadata_vector(
            metadata.get("initial_apple_base_xyz", []), 3
        )
        attrs["initial_apple_id"] = self._safe_int(
            metadata.get("initial_apple_id", -1), -1
        )
        attrs["initial_apple_confidence"] = self._safe_float(
            metadata.get("initial_apple_confidence", np.nan), np.nan
        )
        attrs["initial_apple_bbox"] = self._metadata_vector(
            metadata.get("initial_apple_bbox", []), 4,
            fill=-1, dtype=np.int32,
        )
        episode_meta = self.file.require_group("episode_meta")
        episode_meta.create_dataset("initial_joint_pos", data=initial_joint)
        episode_meta.create_dataset("initial_tcp_pose", data=initial_tcp)
        episode_meta.create_dataset("initialization_offset", data=initial_offset)
        episode_meta.create_dataset(
            "initial_apple_camera_xyz",
            data=self._metadata_vector(
                metadata.get("initial_apple_camera_xyz", []), 3
            ),
        )
        episode_meta.create_dataset(
            "initial_apple_base_xyz",
            data=self._metadata_vector(
                metadata.get("initial_apple_base_xyz", []), 3
            ),
        )
        episode_meta.create_dataset(
            "initial_apple_id",
            data=np.int32(metadata.get("initial_apple_id", -1)),
        )
        episode_meta.create_dataset(
            "initial_apple_confidence",
            data=np.float32(metadata.get("initial_apple_confidence", np.nan)),
        )
        episode_meta.create_dataset(
            "initial_apple_bbox",
            data=self._metadata_vector(
                metadata.get("initial_apple_bbox", []), 4,
                fill=-1, dtype=np.int32,
            ),
        )
        episode_meta.create_dataset("grasp_pose_base", data=np.full((7,), np.nan, dtype=np.float32))
        episode_meta.create_dataset("pregrasp_pose_base", data=np.full((7,), np.nan, dtype=np.float32))
        episode_meta.create_dataset("approach_vector_base", data=np.full((3,), np.nan, dtype=np.float32))
        episode_meta.create_dataset("pregrasp_offset_m", data=np.float32(np.nan))
        labels = self.file.require_group("labels")
        labels.attrs["physical_grasp_success"] = -1
        labels.attrs["failure_reason"] = ""
        labels.attrs["operator_note"] = ""

    @staticmethod
    def _metadata_vector(value, length, fill=np.nan, dtype=np.float32):
        try:
            array = np.asarray(value, dtype=dtype)
        except (TypeError, ValueError):
            array = np.asarray([], dtype=dtype)
        if array.shape != (length,):
            return np.full((length,), fill, dtype=dtype)
        return array

    @staticmethod
    def _safe_int(value, default):
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return int(default)

    @staticmethod
    def _safe_float(value, default):
        try:
            result = float(value)
        except (TypeError, ValueError, OverflowError):
            return float(default)
        return result

    def _dataset(self, path, tail_shape, dtype, compress=True):
        shape = (0,) + tuple(tail_shape)
        maxshape = (None,) + tuple(tail_shape)
        if tail_shape:
            chunks = (1,) + tuple(tail_shape)
        else:
            chunks = (256,)
        kwargs = {
            "shape": shape,
            "maxshape": maxshape,
            "chunks": chunks,
            "dtype": dtype,
        }
        if compress and self.compression is not None:
            kwargs["compression"] = self.compression
        dataset = self.file.create_dataset(path, **kwargs)
        self.datasets[path] = dataset
        return dataset

    def _event_dataset(self, path, tail_shape, dtype):
        shape = (0,) + tuple(tail_shape)
        dataset = self.file.create_dataset(
            path, shape=shape, maxshape=(None,) + tuple(tail_shape),
            chunks=(64,) + tuple(tail_shape), dtype=dtype,
        )
        self.event_datasets[path] = dataset
        return dataset

    def _create_datasets(self):
        height = self.image_height
        width = self.image_width
        string_dtype = h5py.string_dtype(encoding="utf-8")

        self._dataset("obs/rgb", (height, width, 3), np.uint8)
        self._dataset("obs/depth", (height, width), np.uint16)
        self._dataset("obs/joint_pos", (6,), np.float32)
        self._dataset("obs/tcp_pose", (7,), np.float32)
        self._dataset("obs/gripper_state", (), np.int8, compress=False)
        self._dataset("obs/gripper_position", (), np.float32, compress=False)
        self._dataset("obs/gripper_force", (), np.float32, compress=False)
        self._dataset("obs/gripper_current", (), np.float32, compress=False)
        self._dataset("obs/gripper_moving", (), np.bool_, compress=False)
        self._dataset("obs/robot_error_code", (), np.int32, compress=False)
        self._dataset("next_obs/joint_pos", (6,), np.float32)
        self._dataset("next_obs/tcp_pose", (7,), np.float32)
        self._dataset("actions/executed_delta_tcp", (6,), np.float32)
        self._dataset("actions/executed_delta_joint", (6,), np.float32)
        self._dataset("actions/commanded_joint_target", (6,), np.float32)
        self._dataset("actions/command_source", (), string_dtype, compress=False)
        self._dataset("actions/gripper", (), np.int8, compress=False)

        self._dataset("teacher/apple_camera_xyz", (3,), np.float32)
        self._dataset("teacher/apple_base_xyz", (3,), np.float32)
        self._dataset("teacher/grasp_pose_base", (7,), np.float32)
        self._dataset("teacher/pregrasp_pose_base", (7,), np.float32)
        self._dataset("teacher/approach_vector_base", (3,), np.float32)
        self._dataset("teacher/pregrasp_offset_m", (), np.float32, compress=False)
        self._dataset("teacher/apple_id", (), np.int32, compress=False)
        self._dataset("teacher/confidence", (), np.float32, compress=False)
        self._dataset("teacher/bbox", (4,), np.int32)
        self._dataset("teacher/target_valid", (), np.bool_, compress=False)
        self._dataset("teacher/action_type", (), string_dtype, compress=False)
        self._dataset("teacher/action_phase", (), string_dtype, compress=False)
        self._dataset("teacher/target_tcp", (7,), np.float32)
        self._dataset("teacher/target_joint", (6,), np.float32)
        self._dataset("teacher/gripper_command", (), np.int8, compress=False)
        self._dataset("teacher/command_speed", (), np.int32, compress=False)
        self._dataset("teacher/command_force", (), np.int32, compress=False)
        self._dataset("teacher/action_event_count", (), np.uint16, compress=False)

        self._dataset("task/state", (), np.int16, compress=False)
        self._dataset("task/active", (), np.bool_, compress=False)
        self._dataset("task/control_mode", (), np.int8, compress=False)
        self._dataset("task/reached_pregrasp", (), np.bool_, compress=False)
        self._dataset("time/timestamp_ns", (), np.int64, compress=False)
        self._dataset("time/rgb_timestamp_ns", (), np.int64, compress=False)
        self._dataset("time/depth_timestamp_ns", (), np.int64, compress=False)
        self._dataset("time/robot_state_timestamp_ns", (), np.int64, compress=False)
        self._dataset("time/joint_state_timestamp_ns", (), np.int64, compress=False)
        self._dataset("time/rgb_age_ms", (), np.float32, compress=False)
        self._dataset("time/depth_age_ms", (), np.float32, compress=False)
        self._dataset("time/robot_state_age_ms", (), np.float32, compress=False)
        self._dataset("time/joint_state_age_ms", (), np.float32, compress=False)
        self._dataset("time/rgb_depth_delta_ms", (), np.float32, compress=False)
        self._dataset("valid/rgb", (), np.bool_, compress=False)
        self._dataset("valid/depth", (), np.bool_, compress=False)
        self._dataset("valid/robot_state", (), np.bool_, compress=False)
        self._dataset("valid/joint_state", (), np.bool_, compress=False)
        self._dataset("valid/apple_target", (), np.bool_, compress=False)
        self._dataset("valid/action", (), np.bool_, compress=False)
        self._dataset("valid/policy_action", (), np.bool_, compress=False)
        self._dataset("valid/executed_delta_joint", (), np.bool_, compress=False)
        self._dataset("valid/commanded_joint_target", (), np.bool_, compress=False)
        self._dataset("valid/expert_joint_action", (), np.bool_, compress=False)
        self._dataset("valid/expert_action", (), np.bool_, compress=False)
        self._dataset("valid/gripper_measured", (), np.bool_, compress=False)
        self._dataset("rewards", (), np.float32, compress=False)
        self._dataset("dones", (), np.bool_, compress=False)
        self._dataset("reward_terms/distance_to_pregrasp_m", (), np.float32, compress=False)
        self._dataset("reward_terms/progress_m", (), np.float32, compress=False)
        self._dataset("reward_terms/action_norm", (), np.float32, compress=False)
        if self.save_raw_command_debug:
            self._dataset("debug/raw_tx", (), string_dtype, compress=False)
        self._event_dataset("events/expert_timestamp_ns", (), np.int64)
        self._event_dataset("events/expert_phase", (), string_dtype)
        self._event_dataset("events/expert_action_type", (), string_dtype)
        self._event_dataset("events/expert_target_tcp", (7,), np.float32)
        self._event_dataset("events/expert_target_joint", (6,), np.float32)
        self._event_dataset("events/expert_gripper_command", (), np.int8)
        self._event_dataset("events/expert_command_speed", (), np.int32)
        self._event_dataset("events/expert_command_force", (), np.int32)

    def _state_code(self, state):
        state = str(state or "UNKNOWN")
        return int(self.state_mapping.get(state, TASK_STATE_MAPPING["UNKNOWN"]))

    @staticmethod
    def _array(value, shape, dtype, fill):
        array = np.asarray(value, dtype=dtype)
        if array.shape != shape:
            return np.full(shape, fill, dtype=dtype)
        return array

    def append(self, sample):
        """Append one observation and back-fill the preceding transition."""
        if self._closed:
            raise RuntimeError("episode writer is closed")
        index = self.sample_count
        nan32 = np.float32(np.nan)
        robot_valid = bool(sample.get("robot_valid", False))
        joint_vector = self._array(
            sample.get("joint_pos", []), (6,), np.float32, nan32
        )
        joint_valid = bool(sample.get("joint_valid", robot_valid))
        joint_valid = joint_valid and bool(np.all(np.isfinite(joint_vector)))
        teacher = sample.get("teacher", {})
        target_valid = bool(teacher.get("target_valid", False))
        joint = joint_vector
        tcp = self._array(sample.get("tcp_pose", []), (7,), np.float32, nan32)

        task_state = str(sample.get("task_state") or "UNKNOWN")
        goal_pre = self._pre_grasp_goal if self._pre_grasp_goal is not None else np.full((7,), nan32)
        goal_grasp = self._grasp_goal if self._grasp_goal is not None else np.full((7,), nan32)
        approach_vector = self._approach_vector if self._approach_vector is not None else np.full((3,), nan32)
        offset_m = np.float32(self._pregrasp_offset_m)
        values = {
            "obs/rgb": self._array(
                sample.get("rgb", []),
                (self.image_height, self.image_width, 3),
                np.uint8,
                0,
            ),
            "obs/depth": self._array(
                sample.get("depth", []),
                (self.image_height, self.image_width),
                np.uint16,
                0,
            ),
            "obs/joint_pos": joint,
            "obs/tcp_pose": tcp,
            "obs/gripper_state": np.int8(sample.get("gripper_state", -1)),
            "obs/gripper_position": np.float32(
                sample.get("gripper_position", np.nan)
            ),
            "obs/gripper_force": np.float32(
                sample.get("gripper_force", np.nan)
            ),
            "obs/gripper_current": np.float32(
                sample.get("gripper_current", np.nan)
            ),
            "obs/gripper_moving": bool(sample.get("gripper_moving", False)),
            "obs/robot_error_code": np.int32(
                sample.get("robot_error_code", 0)
            ),
            "next_obs/joint_pos": np.full((6,), nan32, dtype=np.float32),
            "next_obs/tcp_pose": np.full((7,), nan32, dtype=np.float32),
            "actions/executed_delta_tcp": np.zeros((6,), dtype=np.float32),
            "actions/executed_delta_joint": np.full((6,), nan32, dtype=np.float32),
            "actions/commanded_joint_target": np.full(
                (6,), nan32, dtype=np.float32
            ),
            "actions/command_source": "NONE",
            "actions/gripper": np.int8(-1),
            "teacher/apple_camera_xyz": self._array(
                teacher.get("apple_camera_xyz", []), (3,), np.float32, nan32
            ),
            "teacher/apple_base_xyz": self._array(
                teacher.get("apple_base_xyz", []), (3,), np.float32, nan32
            ),
            "teacher/grasp_pose_base": np.asarray(goal_grasp, dtype=np.float32),
            "teacher/pregrasp_pose_base": np.asarray(goal_pre, dtype=np.float32),
            "teacher/approach_vector_base": np.asarray(approach_vector, dtype=np.float32),
            "teacher/pregrasp_offset_m": offset_m,
            "teacher/apple_id": np.int32(teacher.get("apple_id", -1)),
            "teacher/confidence": np.float32(
                teacher.get("confidence", np.nan)
            ),
            "teacher/bbox": self._array(
                teacher.get("bbox", []), (4,), np.int32, -1
            ),
            "teacher/target_valid": target_valid,
            "teacher/action_type": "none",
            "teacher/action_phase": "",
            "teacher/target_tcp": np.full((7,), nan32, dtype=np.float32),
            "teacher/target_joint": np.full((6,), nan32, dtype=np.float32),
            "teacher/gripper_command": np.int8(-1),
            "teacher/command_speed": np.int32(0),
            "teacher/command_force": np.int32(0),
            "teacher/action_event_count": np.uint16(0),
            "task/state": np.int16(self._state_code(task_state)),
            "task/active": bool(sample.get("task_active", True)),
            "task/control_mode": np.int8(control_mode_for_phase(task_state)),
            "task/reached_pregrasp": bool(self._reached_pregrasp),
            "time/timestamp_ns": np.int64(sample["timestamp_ns"]),
            "time/rgb_timestamp_ns": np.int64(
                sample.get("rgb_timestamp_ns", -1)
            ),
            "time/depth_timestamp_ns": np.int64(
                sample.get("depth_timestamp_ns", -1)
            ),
            "time/robot_state_timestamp_ns": np.int64(
                sample.get("robot_state_timestamp_ns", -1)
            ),
            "time/joint_state_timestamp_ns": np.int64(
                sample.get("joint_state_timestamp_ns", -1)
            ),
            "time/rgb_age_ms": np.float32(sample.get("rgb_age_ms", np.nan)),
            "time/depth_age_ms": np.float32(sample.get("depth_age_ms", np.nan)),
            "time/robot_state_age_ms": np.float32(
                sample.get("robot_state_age_ms", np.nan)
            ),
            "time/joint_state_age_ms": np.float32(
                sample.get("joint_state_age_ms", np.nan)
            ),
            "time/rgb_depth_delta_ms": np.float32(
                sample.get("rgb_depth_delta_ms", np.nan)
            ),
            "valid/rgb": bool(sample.get("rgb_valid", False)),
            "valid/depth": bool(sample.get("depth_valid", False)),
            "valid/robot_state": robot_valid,
            "valid/joint_state": joint_valid,
            "valid/apple_target": target_valid,
            "valid/action": False,
            "valid/policy_action": False,
            "valid/executed_delta_joint": False,
            "valid/commanded_joint_target": False,
            "valid/expert_joint_action": False,
            "valid/expert_action": False,
            "valid/gripper_measured": bool(
                sample.get("gripper_measurement_valid", False)
            ),
            "rewards": np.float32(0.0),
            "dones": False,
            "reward_terms/distance_to_pregrasp_m": np.float32(np.nan),
            "reward_terms/progress_m": np.float32(np.nan),
            "reward_terms/action_norm": np.float32(np.nan),
        }
        if self.save_raw_command_debug:
            values["debug/raw_tx"] = str(sample.get("raw_tx", ""))

        for path, dataset in self.datasets.items():
            dataset.resize((index + 1,) + dataset.shape[1:])
            dataset[index] = values[path]

        if index > 0:
            previous = index - 1
            previous_valid = bool(self.datasets["valid/robot_state"][previous])
            previous_joint_valid = bool(
                self.datasets["valid/joint_state"][previous]
            )
            if previous_valid and robot_valid:
                previous_tcp = self.datasets["obs/tcp_pose"][previous]
                self.datasets["next_obs/joint_pos"][previous] = joint
                self.datasets["next_obs/tcp_pose"][previous] = tcp
                self.datasets["actions/executed_delta_tcp"][previous] = (
                    executed_delta_tcp(previous_tcp, tcp)
                )
                self.datasets["valid/action"][previous] = True
                previous_mode = int(self.datasets["task/control_mode"][previous])
                self.datasets["valid/policy_action"][previous] = (
                    previous_mode == CONTROL_LEARNABLE_POLICY
                )
            if previous_joint_valid and joint_valid:
                previous_joint = self.datasets["obs/joint_pos"][previous]
                self.datasets["actions/executed_delta_joint"][previous] = (
                    joint - previous_joint
                ).astype(np.float32)
                self.datasets["valid/executed_delta_joint"][previous] = True
                previous_mode = int(self.datasets["task/control_mode"][previous])
                self.datasets["valid/expert_joint_action"][previous] = (
                    previous_mode == CONTROL_LEARNABLE_POLICY
                )

        events = sample.get("expert_events", [])
        if events:
            for expert_event in events:
                self._append_expert_event(expert_event)
            target_index = max(0, index - 1)
            event = events[-1]
            self.datasets["teacher/action_type"][target_index] = str(
                event.get("command_type", "unknown")
            )
            self.datasets["teacher/action_phase"][target_index] = str(
                event.get("phase", "")
            )
            self.datasets["teacher/target_tcp"][target_index] = self._array(
                event.get("target_tcp", []), (7,), np.float32, nan32
            )
            self.datasets["teacher/target_joint"][target_index] = self._array(
                event.get("target_joint", []), (6,), np.float32, nan32
            )
            gripper_command = int(event.get("gripper_command", -1))
            self.datasets["teacher/gripper_command"][target_index] = (
                gripper_command
            )
            self.datasets["actions/gripper"][target_index] = gripper_command
            self.datasets["teacher/command_speed"][target_index] = int(
                event.get("speed", 0)
            )
            self.datasets["teacher/command_force"][target_index] = int(
                event.get("force", 0)
            )
            self.datasets["teacher/action_event_count"][target_index] = min(
                len(events), np.iinfo(np.uint16).max
            )
            self.datasets["valid/expert_action"][target_index] = True
            source = command_source_for_type(event.get("command_type", ""))
            self.datasets["actions/command_source"][target_index] = source
            target_joint = self._array(
                event.get("target_joint", []), (6,), np.float32, nan32
            )
            if source in ("MOVEJ", "MOVEJ_FOLLOW", "MOVEJ_CANFD") and np.all(
                np.isfinite(target_joint)
            ):
                self.datasets["actions/commanded_joint_target"][target_index] = (
                    target_joint
                )
                self.datasets["valid/commanded_joint_target"][target_index] = True

        self.sample_count += 1
        if self.sample_count % self.flush_every_n_samples == 0:
            self.file.flush()

    def _append_expert_event(self, event):
        """Preserve every structured command event independently of sample rate."""
        index = self.expert_event_count
        nan32 = np.float32(np.nan)
        target_tcp = self._array(
            event.get("target_tcp", []), (7,), np.float32, nan32
        )
        target_joint = self._array(
            event.get("target_joint", []), (6,), np.float32, nan32
        )
        values = {
            "events/expert_timestamp_ns": np.int64(event.get("timestamp_ns", -1)),
            "events/expert_phase": str(event.get("phase", "")),
            "events/expert_action_type": str(event.get("command_type", "unknown")),
            "events/expert_target_tcp": target_tcp,
            "events/expert_target_joint": target_joint,
            "events/expert_gripper_command": np.int8(event.get("gripper_command", -1)),
            "events/expert_command_speed": np.int32(event.get("speed", 0)),
            "events/expert_command_force": np.int32(event.get("force", 0)),
        }
        for path, dataset in self.event_datasets.items():
            dataset.resize((index + 1,) + dataset.shape[1:])
            dataset[index] = values[path]
        phase = str(event.get("phase", ""))
        if np.all(np.isfinite(target_tcp[:3])):
            if phase == "MOVE_PRE_GRASP":
                self._pre_grasp_target = np.asarray(target_tcp[:3], dtype=np.float64)
                if self._pre_grasp_goal is None:
                    self._pre_grasp_goal = target_tcp.astype(np.float32).copy()
                    # The pre-grasp goal is useful even when an episode ends
                    # before the scripted MOVE_GRASP event is observed.
                    self._freeze_pregrasp_goal()
            elif phase == "MOVE_GRASP":
                self._grasp_target = np.asarray(target_tcp[:3], dtype=np.float64)
                if self._grasp_goal is None:
                    self._grasp_goal = target_tcp.astype(np.float32).copy()
                self._reached_pregrasp = True
                if self.sample_count:
                    self.datasets["task/reached_pregrasp"][self.sample_count - 1] = True
                if self.sample_count < len(self.datasets["task/reached_pregrasp"]):
                    self.datasets["task/reached_pregrasp"][self.sample_count] = True
        self._freeze_goal_if_complete()
        self._backfill_goal_fields()
        self.expert_event_count += 1

    def _freeze_pregrasp_goal(self):
        """Persist the first real MOVE_PRE_GRASP target immediately."""
        if self._pre_grasp_goal is None:
            return
        pre = np.asarray(self._pre_grasp_goal, dtype=np.float32)
        self.file["episode_meta/pregrasp_pose_base"][:] = pre
        self.file.attrs["pregrasp_pose_base"] = pre
        if self.sample_count:
            self.datasets["teacher/pregrasp_pose_base"][:] = pre

    def _backfill_goal_fields(self):
        """Keep event-derived goal fields constant over all recorded rows."""
        if not self.sample_count:
            return
        if self._pre_grasp_goal is not None:
            self.datasets["teacher/pregrasp_pose_base"][:] = np.asarray(
                self._pre_grasp_goal, dtype=np.float32
            )
        if self._grasp_goal is not None:
            self.datasets["teacher/grasp_pose_base"][:] = np.asarray(
                self._grasp_goal, dtype=np.float32
            )
        if self._approach_vector is not None:
            self.datasets["teacher/approach_vector_base"][:] = self._approach_vector
            self.datasets["teacher/pregrasp_offset_m"][:] = np.float32(
                self._pregrasp_offset_m
            )

    def _freeze_goal_if_complete(self):
        """Freeze the actual Expert pre-grasp/grasp targets once both exist."""
        if self._goal_frozen or self._pre_grasp_goal is None or self._grasp_goal is None:
            return
        pre = np.asarray(self._pre_grasp_goal, dtype=np.float64)
        grasp = np.asarray(self._grasp_goal, dtype=np.float64)
        delta = grasp[:3] - pre[:3]
        distance = float(np.linalg.norm(delta))
        if distance <= 1e-12 or not np.isfinite(distance):
            return
        self._approach_vector = (delta / distance).astype(np.float32)
        self._pregrasp_offset_m = distance
        self._goal_frozen = True
        if self.sample_count:
            self.datasets["teacher/grasp_pose_base"][:] = grasp.astype(np.float32)
            self.datasets["teacher/pregrasp_pose_base"][:] = pre.astype(np.float32)
            self.datasets["teacher/approach_vector_base"][:] = self._approach_vector
            self.datasets["teacher/pregrasp_offset_m"][:] = np.float32(distance)
        self.file["episode_meta/grasp_pose_base"][:] = grasp.astype(np.float32)
        self.file["episode_meta/pregrasp_pose_base"][:] = pre.astype(np.float32)
        self.file["episode_meta/approach_vector_base"][:] = self._approach_vector
        self.file["episode_meta/pregrasp_offset_m"][()] = np.float32(distance)
        self.file.attrs["grasp_pose_base"] = grasp.astype(np.float32)
        self.file.attrs["pregrasp_pose_base"] = pre.astype(np.float32)
        self.file.attrs["approach_vector_base"] = self._approach_vector
        self.file.attrs["pregrasp_offset_m"] = np.float32(distance)

    def finalize(self, outcome):
        """Mark terminal data, close, and atomically rename the episode."""
        if self._closed:
            return self.final_path
        if self.sample_count:
            self.datasets["dones"][self.sample_count - 1] = True
            timestamps = self.datasets["time/timestamp_ns"]
            duration_sec = float(timestamps[-1] - timestamps[0]) / 1e9
        else:
            duration_sec = 0.0
        self.duration_sec = duration_sec

        # A normal Expert episode may end immediately after MOVE_PRE_GRASP;
        # do not require the later scripted MOVE_GRASP event to label the
        # learnable segment as reached.
        self._infer_reached_pregrasp()

        terminal_state = str(outcome.get("terminal_state", "UNKNOWN") or "UNKNOWN")
        requested_success = bool(outcome.get("success", False))
        invalid_reasons = []
        if self.sample_count < self.min_episode_samples:
            invalid_reasons.append(
                "sample_count_below_minimum:{}<{}".format(
                    self.sample_count, self.min_episode_samples
                )
            )
        if duration_sec < self.min_episode_duration_sec:
            invalid_reasons.append(
                "duration_below_minimum:{:.6f}<{:.6f}".format(
                    duration_sec, self.min_episode_duration_sec
                )
            )
        if terminal_state.strip().upper() in ("", "IDLE", "UNKNOWN"):
            invalid_reasons.append(
                "invalid_terminal_state:{}".format(terminal_state or "EMPTY")
            )
        effective_success = requested_success and not invalid_reasons
        failure_reason = str(outcome.get("failure_reason", "") or "")
        if invalid_reasons:
            suffix = "invalid_episode:" + ",".join(invalid_reasons)
            failure_reason = (
                suffix if not failure_reason
                else failure_reason + "; " + suffix
            )

        attrs = self.file.attrs
        attrs["end_time"] = str(
            outcome.get("end_time", datetime.now().astimezone().isoformat())
        )
        attrs["duration_sec"] = duration_sec
        attrs["sample_count"] = int(self.sample_count)
        attrs["expert_event_count"] = int(self.expert_event_count)
        attrs["success"] = effective_success
        attrs["task_success"] = effective_success
        requested_valid = bool(outcome.get("episode_valid", True))
        discarded = bool(outcome.get("discarded", False))
        termination_type = str(
            outcome.get("termination_type", "normal") or "normal"
        )
        discard_reason = str(outcome.get("discard_reason", "") or "")
        episode_valid = bool(
            requested_valid and not invalid_reasons and not discarded
        )
        if termination_type == "emergency_stop":
            episode_valid = False
            discarded = True
            discard_reason = discard_reason or "robot_emergency_stop"
        attrs["episode_valid"] = episode_valid
        attrs["discarded"] = discarded
        attrs["termination_type"] = termination_type
        attrs["discard_reason"] = discard_reason
        attrs["emergency_stop_timestamp_ns"] = int(
            outcome.get("emergency_stop_timestamp_ns", -1)
        )
        attrs["terminal_state"] = terminal_state
        attrs["failure_reason"] = failure_reason
        attrs["task_failure_reason"] = failure_reason
        attrs["operator_note"] = str(outcome.get("operator_note", ""))
        attrs["physical_grasp_success"] = int(
            outcome.get("physical_grasp_success", -1)
        )
        observed_rate = outcome.get("joint_state_observed_rate_hz", np.nan)
        observed_rate = self._safe_float(observed_rate, np.nan)
        # Do not infer source feedback rate from recorder samples: repeated
        # joint frames would measure record_hz instead of the arm feedback.
        # The recorder supplies this value from its callback timestamp cache.
        attrs["joint_state_observed_rate_hz"] = observed_rate
        attrs["state_mapping"] = task_state_mapping_json()
        attrs["reached_pregrasp"] = bool(
            self._reached_pregrasp or outcome.get("reached_pregrasp", False)
        )
        self._backfill_goal_fields()
        self._write_reward_primitives()
        attrs["scripted_final_approach_distance_m"] = (
            self._scripted_final_approach_distance()
        )
        attrs["scripted_final_approach_distance_source"] = (
            "structured expert MOVE_PRE_GRASP and MOVE_GRASP target_tcp"
        )
        attrs["complete"] = True
        labels = self.file["labels"]
        labels.attrs["task_success"] = effective_success
        labels.attrs["physical_grasp_success"] = int(
            outcome.get("physical_grasp_success", -1)
        )
        labels.attrs["failure_reason"] = failure_reason
        labels.attrs["operator_note"] = str(outcome.get("operator_note", ""))
        self.file.flush()
        self.file.close()
        self._closed = True
        os.replace(self.temp_path, self.final_path)
        try:
            from orchard_picker.rl_data.dataset_tools import (
                refresh_manifest_for_episode,
            )
            self.manifest_row = refresh_manifest_for_episode(self.final_path)
            self.quality_result = {
                "quality_pass": bool(self.manifest_row.get("quality_pass", False)),
                "quality_reason": str(self.manifest_row.get("quality_reason", "")),
            }
        except Exception as exc:
            self.quality_result = {
                "quality_pass": False,
                "quality_reason": "post-finalize QC/manifest failed: {}".format(exc),
            }
            try:
                with h5py.File(self.final_path, "r+") as episode:
                    episode.attrs["quality_pass"] = False
                    episode.attrs["quality_reason"] = self.quality_result[
                        "quality_reason"
                    ]
                    episode.flush()
            except (OSError, ValueError) as attr_exc:
                self.quality_result["quality_reason"] += (
                    "; could not persist QC failure: {}".format(attr_exc)
                )
        if not bool(self.quality_result.get("quality_pass", False)):
            try:
                with h5py.File(self.final_path, "r+") as episode:
                    episode.attrs["episode_valid"] = False
                    episode.attrs["discarded"] = True
                    episode.attrs["discard_reason"] = str(
                        episode.attrs.get("discard_reason", "")
                    ) or "quality_check_failed"
                    episode.flush()
            except (OSError, ValueError):
                pass
            try:
                from orchard_picker.rl_data.dataset_tools import (
                    refresh_manifest_for_episode,
                )
                self.manifest_row = refresh_manifest_for_episode(self.final_path)
            except Exception:
                pass
        return self.final_path

    def _infer_reached_pregrasp(self):
        """Infer pre-grasp completion from the final real TCP sample."""
        if self._reached_pregrasp or self._pre_grasp_goal is None or not self.sample_count:
            return
        states = np.asarray(self.datasets["task/state"][:], dtype=np.int16)
        if not np.any(states == self._state_code("MOVE_PRE_GRASP")):
            return
        valid = np.asarray(self.datasets["valid/robot_state"][:], dtype=bool)
        tcp = np.asarray(self.datasets["obs/tcp_pose"][:, :3], dtype=np.float64)
        finite = valid & np.all(np.isfinite(tcp), axis=1)
        indices = np.flatnonzero(finite)
        if not len(indices):
            return
        last = int(indices[-1])
        distance = float(
            np.linalg.norm(tcp[last] - np.asarray(self._pre_grasp_goal[:3], dtype=np.float64))
        )
        if np.isfinite(distance) and distance <= self.PREGRASP_REACHED_TOLERANCE_M:
            self._reached_pregrasp = True
            self.datasets["task/reached_pregrasp"][last] = True

    def _write_reward_primitives(self):
        """Compute raw offline-reward quantities without selecting weights."""
        count = self.sample_count
        if not count:
            return
        distances = np.full((count,), np.nan, dtype=np.float32)
        if self._pre_grasp_goal is not None:
            goal = np.asarray(self._pre_grasp_goal[:3], dtype=np.float64)
            tcp = np.asarray(self.datasets["obs/tcp_pose"][:, :3], dtype=np.float64)
            valid = np.asarray(self.datasets["valid/robot_state"][:], dtype=bool)
            finite = np.all(np.isfinite(tcp), axis=1) & np.all(np.isfinite(goal))
            mask = valid & finite
            distances[mask] = np.linalg.norm(tcp[mask] - goal, axis=1).astype(np.float32)
        self.datasets["reward_terms/distance_to_pregrasp_m"][:] = distances
        progress = np.full((count,), np.nan, dtype=np.float32)
        if count > 1:
            valid_transition = np.asarray(self.datasets["valid/action"][:-1], dtype=bool)
            both = np.isfinite(distances[:-1]) & np.isfinite(distances[1:])
            mask = valid_transition & both
            progress_values = distances[:-1] - distances[1:]
            progress[:-1] = np.where(mask, progress_values, np.nan)
        self.datasets["reward_terms/progress_m"][:] = progress
        action_norm = np.full((count,), np.nan, dtype=np.float32)
        joint_delta = np.asarray(self.datasets["actions/executed_delta_joint"][:], dtype=np.float64)
        valid_delta = np.asarray(self.datasets["valid/executed_delta_joint"][:], dtype=bool)
        if joint_delta.shape == (count, 6):
            action_norm[valid_delta] = np.linalg.norm(joint_delta[valid_delta], axis=1).astype(np.float32)
        self.datasets["reward_terms/action_norm"][:] = action_norm

    def _scripted_final_approach_distance(self):
        pre_target = self._pre_grasp_target
        grasp_target = self._grasp_target
        if pre_target is not None and grasp_target is not None:
            return float(np.linalg.norm(grasp_target - pre_target))
        phases = self.datasets["teacher/action_phase"][:]
        targets = self.datasets["teacher/target_tcp"][:]
        for phase_value, target in zip(phases, targets):
            if isinstance(phase_value, bytes):
                phase = phase_value.decode("utf-8", errors="replace")
            else:
                phase = str(phase_value)
            if np.all(np.isfinite(target[:3])):
                if phase == "MOVE_PRE_GRASP":
                    pre_target = np.asarray(target[:3], dtype=np.float64)
                elif phase == "MOVE_GRASP":
                    grasp_target = np.asarray(target[:3], dtype=np.float64)
        if pre_target is None or grasp_target is None:
            return np.nan
        return float(np.linalg.norm(grasp_target - pre_target))

    def close_incomplete(self):
        """Flush and close while deliberately preserving the .tmp.hdf5 file."""
        if self._closed:
            return
        self.file.attrs["sample_count"] = int(self.sample_count)
        self.file.attrs["state_mapping"] = task_state_mapping_json()
        self.file.flush()
        self.file.close()
        self._closed = True
