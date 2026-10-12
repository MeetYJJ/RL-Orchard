"""Backward-compatible schema helpers for apple RL episodes."""

import json


CONTROL_NON_POLICY = 0
CONTROL_LEARNABLE_POLICY = 1
CONTROL_SCRIPTED_FINAL_APPROACH = 2
CONTROL_SCRIPTED_GRIPPER = 3
CONTROL_POST_GRASP_SCRIPTED = 4

CONTROL_MODE_MAPPING = {
    "NON_POLICY": CONTROL_NON_POLICY,
    "LEARNABLE_POLICY": CONTROL_LEARNABLE_POLICY,
    "SCRIPTED_FINAL_APPROACH": CONTROL_SCRIPTED_FINAL_APPROACH,
    "SCRIPTED_GRIPPER": CONTROL_SCRIPTED_GRIPPER,
    "POST_GRASP_SCRIPTED": CONTROL_POST_GRASP_SCRIPTED,
}

# Structured expert-command source.  Keep this as a string enum in HDF5 so
# episodes remain readable without importing this package.
COMMAND_SOURCE_MAPPING = {
    "NONE": "NONE",
    "MOVEJ": "MOVEJ",
    "MOVEJ_P": "MOVEJ_P",
    "MOVEL": "MOVEL",
    "MOVEJ_FOLLOW": "MOVEJ_FOLLOW",
    "MOVEJ_CANFD": "MOVEJ_CANFD",
    "GRIPPER": "GRIPPER",
    "OTHER": "OTHER",
}

# Task-state codes are global schema constants.  Do not rebuild this mapping
# from the states observed in one episode: that would make the same integer
# mean different phases across HDF5 files.  UNKNOWN is the safe fallback for
# a future/unrecognized state.
TASK_STATE_MAPPING = {
    "UNKNOWN": 0,
    "IDLE": 1,
    "INITIALIZING": 2,
    "OPEN_GRIPPER": 3,
    "MOVE_PRE_GRASP": 4,
    "MOVE_GRASP": 5,
    "CLOSE_GRIPPER": 6,
    "GRASPED_HOLDING": 7,
    "TWIST_J6": 8,
    "RETREAT": 9,
    "HOME": 10,
    "RELEASE_GRIPPER": 11,
    "DONE": 12,
    "FAILED": 13,
    "ABORT_REQUESTED": 14,
    "SAFETY_STOP": 15,
    "DRY_RUN_DONE": 16,
}


def task_state_mapping_json():
    """Return the immutable task-state mapping stored in every new episode."""
    return json.dumps(TASK_STATE_MAPPING, sort_keys=True)


def command_source_for_type(command_type):
    """Map an observed expert event type to the conservative source enum."""
    value = str(command_type or "").strip().lower()
    if not value:
        return "NONE"
    if value in ("movej", "move_j"):
        return "MOVEJ"
    if value in ("movej_p", "movejp", "move_pose", "movep"):
        return "MOVEJ_P"
    if value in ("movel", "move_l"):
        return "MOVEL"
    if value in ("movej_follow", "move_j_follow"):
        return "MOVEJ_FOLLOW"
    if value in ("movej_canfd", "move_j_canfd"):
        return "MOVEJ_CANFD"
    if "gripper" in value or value in ("open", "close"):
        return "GRIPPER"
    return "OTHER"


def command_source_mapping_json():
    """Return a stable source mapping for HDF5 root attributes."""
    return json.dumps(COMMAND_SOURCE_MAPPING, sort_keys=True)

# These names come from the audited pick_task_manager.py state machine.  Keep
# unknown phases non-policy: an unrecognized state must never silently become a
# training action.
PHASE_CONTROL_MODE = {
    "MOVE_PRE_GRASP": CONTROL_LEARNABLE_POLICY,
    "MOVE_GRASP": CONTROL_SCRIPTED_FINAL_APPROACH,
    "OPEN_GRIPPER": CONTROL_SCRIPTED_GRIPPER,
    "CLOSE_GRIPPER": CONTROL_SCRIPTED_GRIPPER,
    "RELEASE_GRIPPER": CONTROL_SCRIPTED_GRIPPER,
    "GRASPED_HOLDING": CONTROL_POST_GRASP_SCRIPTED,
    "TWIST_J6": CONTROL_POST_GRASP_SCRIPTED,
    "RETREAT": CONTROL_POST_GRASP_SCRIPTED,
    "HOME": CONTROL_POST_GRASP_SCRIPTED,
}


def control_mode_for_phase(phase):
    """Return a conservative fixed enum for an audited task phase."""
    return PHASE_CONTROL_MODE.get(str(phase or ""), CONTROL_NON_POLICY)


def is_policy_action(phase, action_valid=True):
    """Return whether one executed motion transition is policy-learnable."""
    return bool(action_valid) and (
        control_mode_for_phase(phase) == CONTROL_LEARNABLE_POLICY
    )


def control_mode_mapping_json():
    """Return the stable enum mapping for HDF5 attributes."""
    return json.dumps(CONTROL_MODE_MAPPING, sort_keys=True)
