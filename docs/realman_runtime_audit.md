# RealMan runtime audit (read-only)

Audit date: 2026-09-18.  No motion, gripper, home, start, `run_once`,
`movej_follow`, or `movej_canfd` command was sent.

## What is known locally

- Project configuration and the existing stack identify the arm as **RML63-B**.
- The overlay contains the custom `/rm_json_bridge` bridge.  Its inspected
  motion services are blocking `MoveJ`, `MovePose` (`movej_p`/`movel`) and
  gripper services.  There is no local `movej_follow` or `movej_canfd`
  service/topic implementation.
- `/rm_json_bridge/get_arm_state` is a read-only query returning six joint
  values in degree in the service response; `RobotState.joint_pos_rad` is the
  bridge's converted radian publisher field.
- The inspected `tcp_tf_publisher` default polling period is 5 Hz, and the
  Recorder's stale-state fallback is also configured at 5 Hz.  These are query
  consumers, not a guaranteed robot feedback stream; a live preflight must
  measure the actual rate before a 10 Hz episode is accepted.
- The workspace does not contain an official `rm_driver`, `Jointpos`, UDP
  realtime receiver, or driver launch package.  A package-level installation
  was not changed or assumed.

## Runtime graph observed in this environment

The read-only ROS graph check found only `/rosout` and
`/parameter_events`.  `/joint_states` and `/rm_json_bridge/arm_state` were
not online, so an effective joint feedback rate, timestamp age, controller
generation, controller software version, and joint firmware version could not
be measured here.  This is an environment observation, not a claim that the
hardware cannot publish those interfaces when the normal stack is running.

## Official capabilities versus local validation

RealMan's [JSON motion documentation](https://develop.realman-robotics.com/en/robot/json/motionConfig/)
describes `movej_follow` (successive absolute joint targets) and
`movej_canfd` (joint-angle CANFD pass-through).  The official
[ROS2 driver documentation](https://develop.realman-robotics.com/en/robot/ros2/ros2Description/)
describes `/rm_driver/movej_canfd_cmd` with
`rm_ros_interfaces/msg/Jointpos` (`float32[6]`, radians), while its
[driver state documentation](https://develop.realman-robotics.com/en/robot4th/ros2/driver/)
describes realtime UDP feedback and `/joint_states`.  None of those interfaces
was found or exercised in this overlay, and no package was installed to obtain
them.

The current final joint-state source therefore remains:

```text
joint_state_source = rm_json_bridge
source topic       = /rm_json_bridge/arm_state (RobotState.joint_pos_rad)
measured rate      = unavailable until the real stack is running
```

The `sensor_msgs/JointState` path is implemented as an opt-in recorder source
and requires explicit six-name mapping plus a live read-only preflight.

## Version query status

`scripts/query_realman_version.py` contains an allowlist for the vendor's
read-only version/info command names documented in the
[system configuration reference](https://develop.realman-robotics.com/en/robot/json/systemConfig/).
It refuses arbitrary JSON and was not run during this audit because the
controller was not online.  Thus
the following remain **unknown**: controller generation, controller software
version, joint firmware version, installed RealMan SDK/API version, and an
official ROS2 driver version.

Run the query only after the operator confirms the controller's documented
read-only command name and the normal stack is already online:

```bash
source /home/nvidia/RL-Orchard/scripts/setup_rl_env.sh
python /home/nvidia/RL-Orchard/src/orchard_picker/scripts/query_realman_version.py \
  get_arm_software_info
```

This command is not part of recorder startup and is not required for static
tests.  Never replace it with an arbitrary `raw_json` request.

## Consequence for future SAC execution

The custom bridge currently has no continuous joint executor.  Official
`movej_follow`/`movej_canfd` support must still be version- and controller-
validated.  In particular, CANFD pass-through needs a stable continuous
target stream (the vendor high-follow guidance is a period of at most 10 ms),
feedback, interpolation, network-latency measurement, stop/timeout/error
semantics, and a separate hardware safety review.  This round deliberately
does not publish or test either interface.
