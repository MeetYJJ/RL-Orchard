"""Deterministic mock episode fixtures; no ROS or hardware imports."""

from datetime import datetime

import numpy as np

from orchard_picker.rl_data.episode_writer import EpisodeWriter


def create_episode(
    root, episode_id="episode_000001", physical=-1,
    task_success=True, terminal="DONE", failure_reason="",
):
    writer = EpisodeWriter(
        dataset_root=root,
        episode_id=episode_id,
        image_height=12,
        image_width=16,
        metadata={
            "start_time": datetime.now().astimezone().isoformat(),
            "record_hz": 10.0,
            "dataset_version": "apple_rl_v1.1",
            "joint_state_source": "rm_json_bridge",
            "joint_state_topic": "/rm_json_bridge/arm_state",
            "joint_state_nominal_rate_hz": 10.0,
            "source_joint_unit": "radian",
            "camera_model": "Intel RealSense D435i (mock)",
            "camera_serial": "MOCK",
            "depth_scale_m": 0.001,
            "depth_unit": "device_unit",
            "initial_pose_id": "init_01",
            "initial_joint_pos": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "initial_tcp_pose": [0.3, -0.1, 0.55, 0.0, 0.0, 0.0, 1.0],
            "initialization_mode": "pose_library",
            "initialization_seed": 17,
            "initialization_offset": [0.0] * 6,
            "initialization_timestamp": "2026-09-17T12:00:00+08:00",
            "apple_position_bin": "CENTER_MIDDLE_MIDDLE",
        },
        compression="lzf",
    )
    phases = [
        "OPEN_GRIPPER", "MOVE_PRE_GRASP", "MOVE_PRE_GRASP",
        "MOVE_PRE_GRASP", "MOVE_GRASP", "MOVE_GRASP",
        "CLOSE_GRIPPER", "RETREAT", terminal,
    ]
    start_ns = 1000000000
    for index, phase in enumerate(phases):
        rgb = np.full((12, 16, 3), 25 + index, dtype=np.uint8)
        rgb[3:9, 5:11, 1] = 180
        depth = np.full((12, 16), 600 + index, dtype=np.uint16)
        tcp = np.asarray(
            [0.30 + 0.01 * index, -0.10, 0.55,
             0.0, 0.0, 0.0, 1.0], dtype=np.float32,
        )
        events = []
        if index == 2:
            events = [{
                "phase": "MOVE_PRE_GRASP", "command_type": "movej_p",
                "target_tcp": [0.40, -0.10, 0.55, 0.0, 0.0, 0.0, 1.0],
                "target_joint": [np.nan] * 6, "gripper_command": -1,
                "speed": 20, "force": 0,
            }]
        elif index == 4:
            events = [{
                "phase": "MOVE_GRASP", "command_type": "movel",
                "target_tcp": [0.48, -0.10, 0.55, 0.0, 0.0, 0.0, 1.0],
                "target_joint": [np.nan] * 6, "gripper_command": -1,
                "speed": 10, "force": 0,
            }]
        elif index == 6:
            events = [{
                "phase": "CLOSE_GRIPPER", "command_type": "gripper_pick",
                "target_tcp": [np.nan] * 7,
                "target_joint": [np.nan] * 6, "gripper_command": 1,
                "speed": 150, "force": 150,
            }]
        elif index == 7:
            events = [{
                "phase": "RETREAT", "command_type": "movej",
                "target_tcp": [np.nan] * 7,
                "target_joint": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
                "gripper_command": -1, "speed": 20, "force": 0,
            }]
        writer.append({
            "timestamp_ns": start_ns + index * 100000000,
            "rgb_timestamp_ns": start_ns + index * 100000000,
            "depth_timestamp_ns": start_ns + index * 100000000,
            "robot_state_timestamp_ns": start_ns + index * 100000000,
            "rgb_age_ms": 10.0,
            "depth_age_ms": 11.0,
            "robot_state_age_ms": 20.0,
            "joint_state_timestamp_ns": start_ns + index * 100000000,
            "joint_state_age_ms": 18.0,
            "rgb_depth_delta_ms": 0.0,
            "rgb": rgb,
            "depth": depth,
            "joint_pos": np.linspace(0.0, 0.5, 6, dtype=np.float32) + index * 0.001,
            "tcp_pose": tcp,
            "gripper_state": -1,
            "gripper_measurement_valid": False,
            "task_state": phase,
            "task_active": index != len(phases) - 1,
            "teacher": {
                "apple_camera_xyz": [0.0, 0.0, 0.5],
                "apple_base_xyz": [0.42, -0.1, 0.55],
                "apple_id": 7, "confidence": 0.95,
                "bbox": [5, 3, 11, 9], "target_valid": True,
            },
            "expert_events": events,
            "rgb_valid": True, "depth_valid": True, "robot_valid": True,
            "joint_valid": True, "joint_state_source": "rm_json_bridge",
        })
    return writer.finalize({
        "end_time": datetime.now().astimezone().isoformat(),
        "success": task_success,
        "terminal_state": terminal,
        "failure_reason": failure_reason,
        "physical_grasp_success": physical,
    })
