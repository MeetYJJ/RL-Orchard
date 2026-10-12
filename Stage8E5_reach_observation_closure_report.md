# Stage8E5 ReachActor Observation Closure Report

## Safety and scope

This audit used only fixed read-only TCP queries, passive UDP receive, source/config inspection, and an independent vision detector. No Actor checkpoint was loaded, no inference was run, and no command writer was started. No MoveJ, MoveL, Movep_CANFD, servo, gripper, Home, stop, tool/work-frame change, network change, or controller configuration command was sent. Existing `rm_json_bridge`/`pick_task_manager` processes were not launched or stopped by this audit and were not called.

## 1. UDP waypoint pose validation

Capture files:

- UDP: `641` packets from `['192.168.1.18:42620']`, `201.659 Hz`.
- TCP: `7` `get_current_arm_state` samples.
- Matched TCP/UDP samples: `7`.
- Maximum host receive time gap: `0.000125 s`.
- Maximum position residual: `0.000034307 m`.
- Project RPY rotation residual: `0.000000000 rad`.
- UDP joint raw ranges during capture: `[6, 9, 6, 7, 6, 7]` counts of `0.001 degree`; this supports a static-arm capture.

Result: **UDP waypoint == robot_end numerically for the observed current configuration**. This is supported by seven near-synchronous comparisons, the mature Origin contract that `get_current_arm_state.pose` is `base_link -> robot_end` flange pose, active tool `Arm_Tip` with zero transform, and work frame `Base` with zero transform. The protocol payload itself still has no frame/origin field. Therefore this conclusion is conditional on the unchanged zero-offset configuration and must be invalidated if tool/work configuration changes.

The raw paired samples and residuals are in `tcp_state_samples.jsonl` and `sync_summary.json`. Position protocol scale is `1e-6 m`; Euler scale is `1e-3 rad`. Euler order remains an official-protocol UNKNOWN; the zero rotation residual is a project-convention numeric check.

## 2. TCP source

For high-frequency feedback, the usable current source is the UDP `waypoint.position` field after the static capture validation above. The mature low-rate source is the fixed allow-list TCP `get_current_arm_state`, whose project semantics are `base_link/robot_end` flange. The active tool/work queries in this capture returned `Arm_Tip` and `Base`, both zero transforms.

## 3. Vision goal and transform

The independent detector was started with camera serial `908212070350`. The code/configuration exposes:

- `/apple_positions`: `ApplePositionArray`, position in metres, `camera_link` frame, ROS header timestamp.
- `/apple/target_point`: `PointStamped`, position in metres, `camera_link` frame, ROS header timestamp; published after target stability checks.
- `/apple/target_info`: JSON with `camera_xyz_m`, `camera_frame`, `confidence`, `id`, `target_valid`; no orientation.

No stable apple target was emitted during this run: **APPLE_GOAL_UNAVAILABLE**. The independent TF check also found no live `base_link` frame because the mature arm/TF stack was not started. The static code path is present: `base_link -> robot_end` dynamic TF plus the calibrated `robot_end -> camera_link` matrix. No goal position was filled with zero.

## 4. ReachActor 15D source matrix

| Observation | Source | Status |
|---|---|---|
| joint position (6) | UDP `joint_status.joint_position` | **AVAILABLE**; J1–J6, `0.001°` to rad |
| tcp xyz (3) | UDP waypoint validated against TCP query | **AVAILABLE_CONDITIONAL_CURRENT_CONFIGURATION**; `base_link/robot_end` |
| goal relative xyz (3) | camera target plus TF to `base_link` | **APPLE_GOAL_UNAVAILABLE** |
| orientation error (3) | current pose plus policy goal orientation | **UNKNOWN**; goal has no orientation and ReachActor convention is absent |

Overall result: **REACH_OBSERVATION_BLOCKED**.

## 5. Remaining blockers

1. A visible/stable apple target must be present and published.
2. The mature arm/TF publishers must be running in the same ROS domain if a live `camera_link -> base_link` transformed target is required.
3. The frozen ReachActor observation metadata must state its goal orientation and rotation error convention. The repository's `relative_rotation_vector` is a pose-action utility, not proof of the Actor's input convention.
4. The selected tool center version must be fixed. Current launch YAML has `robot_end -> gripper_tip_center = [0.0005,-0.005,0.158] m`; older documents contain `[0,0,0.167]` and `[0,0,0.168]` candidates. No value was changed.

## 6. Shadow inference gate

Shadow inference is **not ready** for this run. It can proceed only after one live apple target is captured in `camera_link`, transformed to `base_link`, and the ReachActor orientation field definition is supplied. No Actor was loaded or executed.

Evidence directory: `/tmp/stage8e5_reach_audit`.
