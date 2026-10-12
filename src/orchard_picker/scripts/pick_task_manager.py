#!/usr/bin/env python3
import math
import threading
import time

import rclpy
import tf2_geometry_msgs  # noqa: F401 - registers geometry message transforms
import tf2_ros
from geometry_msgs.msg import PointStamped, Pose, PoseStamped, Quaternion
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.time import Time
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger

from orchard_picker.msg import ExpertAction, TaskOutcome

try:
    from rclpy.qos import DurabilityPolicy
except ImportError:  # ROS2 Foxy compatibility
    from rclpy.qos import QoSDurabilityPolicy as DurabilityPolicy

from orchard_picker.protocol import quaternion_from_euler
from orchard_picker.srv import (
    GetArmState,
    GripperPick,
    GripperRelease,
    MoveJ,
    MovePose,
)


def _transient_local_durability():
    if hasattr(DurabilityPolicy, "TRANSIENT_LOCAL"):
        return DurabilityPolicy.TRANSIENT_LOCAL
    return DurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL


class PickTaskManager(Node):
    # Joint origins and limits are taken directly from the supplied
    # rm_63_b_description.urdf.  Keeping the small FK chain here avoids adding
    # a second robot-control dependency merely for the Z safety guard.
    _JOINT_CHAIN = (
        ((0.0, 0.0, 0.172), (0.0, 0.0, 0.0), (-3.107, 3.107)),
        ((-0.086, 0.0, 0.0), (-1.5708, -1.5708, 0.0), (-3.107, 3.107)),
        ((0.38, 0.0, -0.00030011), (0.0, 0.0, 1.5708), (-3.107, 2.531)),
        ((0.067482, -0.405, -0.00051458), (1.5708, 0.0, 0.0), (-3.107, 3.107)),
        ((0.0, 0.0, 0.0), (1.5708, 0.0, 3.1416), (-3.107, 3.107)),
        ((-0.0015175, 0.115, 0.00081415), (1.5708, 0.0, 3.1416), (-6.283, 6.283)),
    )

    def __init__(self):
        super().__init__("pick_task_manager")
        self.callback_group = ReentrantCallbackGroup()

        self.base_frame = self._param("base_frame", "base_link")
        self.use_tf = bool(self._param("use_tf", True))
        self.target_timeout = float(self._param("target_timeout", 2.0))
        self.target_use_latest_tf = bool(self._param("target_use_latest_tf", True))
        self.clear_target_on_success = bool(self._param("clear_target_on_success", True))
        self.clear_target_on_failure = bool(self._param("clear_target_on_failure", False))
        self.transform_timeout = float(self._param("transform_timeout", 0.5))
        self.service_call_timeout = float(self._param("service_call_timeout", 90.0))
        # A controller trajectory acknowledgement can mean that a command was
        # accepted, not that the TCP has physically settled at its target.
        # Verify the measured arm state before allowing the next action.
        self.motion_verify_timeout = float(
            self._param("motion_verify_timeout", 20.0)
        )
        self.motion_position_tolerance = float(
            self._param("motion_position_tolerance_m", 0.005)
        )
        self.home_joint_tolerance = float(
            self._param("home_joint_tolerance_deg", 1.0)
        )
        self.dry_run = bool(self._param("dry_run", True))
        self.robot_end_frame = self._param("robot_end_frame", "robot_end")
        self.max_robot_end_tf_age = float(
            self._param("max_robot_end_tf_age", 1.0)
        )

        self.approach_vector = self._normalized(
            self._param("approach_vector_base", [1.0, 0.0, 0.0])
        )
        self.approach_along_gripper_axis = bool(
            self._param("approach_along_gripper_axis", True)
        )
        self.use_current_gripper_orientation = bool(
            self._param("use_current_gripper_orientation", True)
        )
        self.approach_distance = float(self._param("approach_distance", 0.05))
        self.retreat_distance = float(self._param("retreat_distance", 0.05))
        self.grasp_offset = [float(v) for v in self._param("grasp_offset_base", [0.0, 0.0, 0.0])]
        self.gripper_tip_rpy = [
            float(v)
            for v in self._param(
                "gripper_tip_rpy_rad", [2.555, -0.397, 3.048]
            )
        ]
        self.gripper_length_to_tip = float(
            self._param("gripper_length_to_tip_m", 0.158)
        )
        self.flange_adapter_length = float(
            self._param("flange_adapter_length_m", 0.010)
        )
        if self.gripper_length_to_tip < 0.0 or self.flange_adapter_length < 0.0:
            raise ValueError("gripper and flange adapter lengths must be non-negative")
        self.gripper_tcp_xyz = self._three_finite_values(
            self._param(
                "gripper_tcp_xyz_m",
                [
                    0.0,
                    0.0,
                    self.gripper_length_to_tip + self.flange_adapter_length,
                ],
            ),
            "gripper_tcp_xyz_m",
        )
        self.gripper_tcp_rpy_deg = self._three_finite_values(
            self._param("gripper_tcp_rpy_deg", [0.0, 0.0, 0.0]),
            "gripper_tcp_rpy_deg",
        )
        self.gripper_tcp_quaternion = quaternion_from_euler(
            *[math.radians(value) for value in self.gripper_tcp_rpy_deg]
        )

        self.min_axis_endpoint_z = float(
            self._param("min_axis_endpoint_z_m", 0.010)
        )
        if not math.isfinite(self.min_axis_endpoint_z) or self.min_axis_endpoint_z < 0.0:
            raise ValueError("min_axis_endpoint_z_m must be a non-negative finite value")

        self.workspace_min = [float(v) for v in self._param("workspace_min", [0.05, -0.50, 0.02])]
        self.workspace_max = [float(v) for v in self._param("workspace_max", [0.80, 0.50, 0.80])]
        self.enforce_workspace_limits = bool(
            self._param("enforce_workspace_limits", False)
        )
        self.home_joint_deg = [
            float(v)
            for v in self._param(
                "home_joint_deg",
                [-0.267, -26.516, 101.196, -159.288, -98.282, 210.182],
            )
        ]
        self._validate_home_joint_values(self.home_joint_deg)
        self._check_axis_endpoint_z(self.home_joint_deg, "configured Home")

        self.movej_p_speed = int(self._param("movej_p_speed", 8))
        self.movel_speed = int(self._param("movel_speed", 5))
        self.home_speed = int(self._param("home_speed", 8))
        self.twist_speed = int(self._param("twist_speed", 5))
        self.blend_radius = int(self._param("blend_radius", 0))

        self.gripper_release_speed = int(self._param("gripper_release_speed", 300))
        self.gripper_pick_speed = int(self._param("gripper_pick_speed", 150))
        self.gripper_pick_force = int(self._param("gripper_pick_force", 150))

        self.twist_joint_index = int(self._param("twist_joint_index", 5))
        self.twist_delta_deg = float(self._param("twist_delta_deg", 45.0))
        self.twist_min_deg = float(self._param("twist_min_deg", -360.0))
        self.twist_max_deg = float(self._param("twist_max_deg", 360.0))
        self.enable_twist = bool(self._param("enable_twist", False))
        # Complete the configured sequence by default: retreat, Home, release.
        self.stop_after_grasp = bool(self._param("stop_after_grasp", False))
        self.sleep_after_release = float(self._param("sleep_after_release", 0.5))
        self.sleep_after_grip = float(self._param("sleep_after_grip", 0.3))
        self.sleep_after_twist = float(self._param("sleep_after_twist", 0.2))

        self.return_home_on_failure = bool(self._param("return_home_on_failure", False))
        self.release_on_failure = bool(self._param("release_on_failure", False))

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.latest_target_base = None
        self.latest_target_stamp = None
        self.latest_target_received_monotonic = None
        self.latest_target_source_frame = ""
        self.latest_target_source_point = None
        self.latest_target_kind = None
        self.lock = threading.RLock()
        self.active = False
        self.current_state = "INITIALIZING"
        self.abort_requested = False
        self._safety_state_future = None
        self._safety_stop_sent = False
        self.add_on_set_parameters_callback(self._on_parameters_set)

        latched_qos = QoSProfile(depth=1)
        latched_qos.durability = _transient_local_durability()
        self.state_pub = self.create_publisher(String, "~/state", latched_qos)
        self.active_pub = self.create_publisher(Bool, "~/active", latched_qos)
        # Expose the already validated Camera -> Base result to operator UIs.
        # The GUI must not duplicate hand-eye or TF calculations.
        self.target_base_pub = self.create_publisher(
            PointStamped, "~/target_base", latched_qos
        )
        self.expert_action_pub = self.create_publisher(
            ExpertAction, "~/expert_action", 10
        )
        self.outcome_pub = self.create_publisher(
            TaskOutcome, "~/outcome", latched_qos
        )

        pose_topic = self._param("target_pose_topic", "/apple/target_pose")
        point_topic = self._param("target_point_topic", "/apple/target_point")
        self.create_subscription(
            PoseStamped,
            pose_topic,
            self._pose_target_cb,
            1,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            PointStamped,
            point_topic,
            self._point_target_cb,
            1,
            callback_group=self.callback_group,
        )

        self._connect_services()
        self.create_timer(
            0.20, self._safety_monitor_tick, callback_group=self.callback_group
        )

        self.create_service(Trigger, "~/run_once", self.handle_run_once, callback_group=self.callback_group)
        self.create_service(Trigger, "~/start", self.handle_start, callback_group=self.callback_group)
        self.create_service(Trigger, "~/abort", self.handle_abort, callback_group=self.callback_group)
        self.create_service(Trigger, "~/home", self.handle_home, callback_group=self.callback_group)

        self.start_on_target = bool(self._param("start_on_target", False))
        self.get_logger().info(
            "tool center: xyz_m={} rpy_deg={}; min_axis_z={:.3f} m; dry_run={}".format(
                self._format_point(self.gripper_tcp_xyz),
                self._format_point(self.gripper_tcp_rpy_deg),
                self.min_axis_endpoint_z,
                self.dry_run,
            )
        )
        self._set_state("IDLE")

    def _param(self, name, default):
        self.declare_parameter(name, default)
        return self.get_parameter(name).value

    def _connect_services(self):
        prefix = self._param("driver_ns", "/rm_json_bridge")
        wait_for_driver = bool(self._param("wait_for_driver", True))
        service_names = {
            "get_state": prefix + "/get_arm_state",
            "movej": prefix + "/movej",
            "movej_p": prefix + "/movej_p",
            "movel": prefix + "/movel",
            "gripper_release": prefix + "/gripper_release",
            "gripper_pick": prefix + "/gripper_pick",
            "stop": prefix + "/stop",
        }

        self.get_state_srv = self.create_client(
            GetArmState, service_names["get_state"], callback_group=self.callback_group
        )
        self.movej_srv = self.create_client(MoveJ, service_names["movej"], callback_group=self.callback_group)
        self.movej_p_srv = self.create_client(
            MovePose, service_names["movej_p"], callback_group=self.callback_group
        )
        self.movel_srv = self.create_client(MovePose, service_names["movel"], callback_group=self.callback_group)
        self.gripper_release_srv = self.create_client(
            GripperRelease, service_names["gripper_release"], callback_group=self.callback_group
        )
        self.gripper_pick_srv = self.create_client(
            GripperPick, service_names["gripper_pick"], callback_group=self.callback_group
        )
        self.stop_srv = self.create_client(
            Trigger, service_names["stop"], callback_group=self.callback_group
        )

        if wait_for_driver:
            for name, client in (
                (service_names["get_state"], self.get_state_srv),
                (service_names["movej"], self.movej_srv),
                (service_names["movej_p"], self.movej_p_srv),
                (service_names["movel"], self.movel_srv),
                (service_names["gripper_release"], self.gripper_release_srv),
                (service_names["gripper_pick"], self.gripper_pick_srv),
                (service_names["stop"], self.stop_srv),
            ):
                while rclpy.ok() and not client.wait_for_service(timeout_sec=1.0):
                    self.get_logger().info("Waiting for service {}".format(name))

    def _pose_target_cb(self, msg):
        if self._accept_target(msg, "pose") and self.start_on_target:
            self._start_async()

    def _point_target_cb(self, msg):
        if self._accept_target(msg, "point") and self.start_on_target:
            self._start_async()

    def _accept_target(self, msg, kind):
        # The current task has already locked its target.  Ignore subsequent
        # camera frames while the arm is moving; transforming them is both
        # unnecessary and can produce stale-TF warning spam.
        with self.lock:
            if self.active:
                return False
        try:
            source_point = self._target_message_point(msg, kind)
            target_base = self._target_message_to_base_point(msg, kind)
        except Exception as exc:
            self.get_logger().warning("Rejected apple target: {}".format(exc))
            return False

        with self.lock:
            self.latest_target_base = target_base
            self.latest_target_stamp = msg.header.stamp
            self.latest_target_received_monotonic = time.monotonic()
            self.latest_target_source_frame = msg.header.frame_id
            self.latest_target_source_point = source_point
            self.latest_target_kind = kind

        target_message = PointStamped()
        target_message.header.stamp = self.get_clock().now().to_msg()
        target_message.header.frame_id = self.base_frame
        target_message.point.x = float(target_base[0])
        target_message.point.y = float(target_base[1])
        target_message.point.z = float(target_base[2])
        self.target_base_pub.publish(target_message)

        self.get_logger().info(
            "Locked apple target in {} from {}: x={:.4f}, y={:.4f}, z={:.4f}".format(
                self.base_frame,
                msg.header.frame_id or self.base_frame,
                target_base[0],
                target_base[1],
                target_base[2],
            )
        )
        return True

    def handle_run_once(self, _req, resp):
        if not self._claim_active():
            resp.success = False
            resp.message = "picker is already active"
            return resp
        try:
            resp.success, resp.message = self._execute_pick()
            self._publish_outcome(resp.success, resp.message)
            return resp
        finally:
            self._release_active()

    def handle_start(self, _req, resp):
        if not self._start_async():
            resp.success = False
            resp.message = "picker is already active"
            return resp
        resp.success = True
        resp.message = "started"
        return resp

    def handle_abort(self, _req, resp):
        with self.lock:
            self.abort_requested = True
            was_active = self.active
        self._set_state("ABORT_REQUESTED")
        try:
            stop_resp = self._call_service(
                self.stop_srv, Trigger.Request(), timeout=min(5.0, self.service_call_timeout)
            )
            resp.success = bool(stop_resp.success)
            if was_active:
                resp.message = "已发送软件急停并终止抓取；{}".format(stop_resp.message)
            else:
                resp.message = "已向控制器发送软件急停；当前无抓取任务"
        except Exception as exc:
            resp.success = False
            resp.message = "抓取终止标志已设置，但控制器停止指令失败：{}".format(exc)
        return resp

    def handle_home(self, _req, resp):
        if self.dry_run:
            resp.success = False
            resp.message = "home motion is disabled while dry_run=true"
            return resp
        if not self._claim_active():
            resp.success = False
            resp.message = "picker is already active"
            return resp
        try:
            self._set_state("HOME")
            home_ok, home_message = self._home()
            if not home_ok:
                self._set_state("FAILED")
                resp.success = False
                resp.message = home_message
                self._publish_outcome(False, resp.message)
                return resp
            self._set_state("RELEASE_GRIPPER")
            release_ok, release_message = self._gripper_release()
            if not release_ok:
                self._set_state("FAILED")
                resp.success = False
                resp.message = "Home reached, but gripper release failed: {}".format(
                    release_message
                )
                self._publish_outcome(False, resp.message)
                return resp
            self._set_state("IDLE")
            resp.success = True
            resp.message = "Home reached and gripper opened"
            self._publish_outcome(True, resp.message)
            return resp
        finally:
            self._release_active()

    def _start_async(self):
        if not self._claim_active():
            return False
        thread = threading.Thread(target=self._execute_pick_async, daemon=True)
        thread.start()
        return True

    def _execute_pick_async(self):
        try:
            success, message = self._execute_pick()
            self._publish_outcome(success, message)
            if success:
                self.get_logger().info("Pick task finished: {}".format(message))
            else:
                self.get_logger().error("Pick task failed: {}".format(message))
        finally:
            self._release_active()

    def _claim_active(self):
        with self.lock:
            if self.active:
                return False
            self.active = True
            self.abort_requested = False
            self._safety_stop_sent = False
            self.active_pub.publish(Bool(data=True))
            return True

    def _release_active(self):
        with self.lock:
            self.active = False
            self.active_pub.publish(Bool(data=False))

    def _execute_pick(self):
        try:
            with self.lock:
                source_frame = self.latest_target_source_frame
            if self.use_tf and source_frame not in ("", self.base_frame):
                self._check_robot_end_tf_freshness()
            target = self._get_target_in_base()
            arm_state = self._get_checked_arm_state()
            self._check_axis_endpoint_z(arm_state.joint_deg, "current arm")
            if self.use_current_gripper_orientation:
                orientation = self._quaternion_message(
                    self._quaternion_multiply(
                        self._pose_quaternion(arm_state.pose),
                        self.gripper_tcp_quaternion,
                    )
                )
            else:
                q = quaternion_from_euler(
                    self.gripper_tip_rpy[0],
                    self.gripper_tip_rpy[1],
                    self.gripper_tip_rpy[2],
                )
                orientation = Pose().orientation
                orientation.x, orientation.y, orientation.z, orientation.w = q
            approach_vector = self.approach_vector
            if self.approach_along_gripper_axis:
                approach_vector = self._normalized(
                    self._rotate_vector_by_quaternion(
                        [0.0, 0.0, 1.0],
                        (
                            orientation.x,
                            orientation.y,
                            orientation.z,
                            orientation.w,
                        ),
                    )
                )
            tip_poses, end_poses = self._build_pick_poses(
                target, orientation, approach_vector
            )
            pre_grasp, grasp, retreat = end_poses
            self._log_pick_plan(target, tip_poses, end_poses)

            self._check_workspace(target, "apple center")
            active_pose_names = ("pre-grasp robot_end", "grasp robot_end")
            active_end_poses = end_poses[:2]
            if not self.stop_after_grasp:
                active_pose_names += ("retreat robot_end",)
                active_end_poses = end_poses
            for name, pose in zip(active_pose_names, active_end_poses):
                self._check_workspace(self._pose_point(pose), name)
                self._check_robot_end_z(pose, name)

            if self.dry_run:
                self._set_state("DRY_RUN_DONE")
                return True, "dry run completed; no gripper or motion commands sent"

            self._step("OPEN_GRIPPER", lambda: self._gripper_release())
            if self.sleep_after_release > 0:
                time.sleep(self.sleep_after_release)
            self._step("MOVE_PRE_GRASP", lambda: self._move_pose(self.movej_p_srv, pre_grasp, self.movej_p_speed))
            self._step("MOVE_GRASP", lambda: self._move_pose(self.movel_srv, grasp, self.movel_speed))
            self._step("CLOSE_GRIPPER", lambda: self._gripper_pick())
            if self.sleep_after_grip > 0:
                time.sleep(self.sleep_after_grip)
            if self.stop_after_grasp:
                self._set_state("GRASPED_HOLDING")
                if self.clear_target_on_success:
                    self._clear_locked_target()
                return True, "apple grasped; holding position with gripper closed"
            if self.enable_twist:
                self._step("TWIST_J6", self._twist_joint)
                if self.sleep_after_twist > 0:
                    time.sleep(self.sleep_after_twist)
            #self._step("RETREAT", lambda: self._move_pose(self.movel_srv, retreat, self.movel_speed))
            self._step(
                "RETREAT",
                lambda: self._retreat_with_current_orientation(tip_poses[2]),
            )
            self._step("HOME", self._home)
            self._step("RELEASE_GRIPPER", lambda: self._gripper_release())
            self._set_state("DONE")
            if self.clear_target_on_success:
                self._clear_locked_target()
            return True, "pick sequence completed"
        except Exception as exc:
            self.get_logger().error("Pick sequence error: {}".format(exc))
            self._set_state("FAILED")
            if self.clear_target_on_failure:
                self._clear_locked_target()
            if not self.dry_run:
                self._failure_recovery()
            else:
                self.get_logger().warning(
                    "dry_run=true: skipped all failure-recovery motion"
                )
            return False, str(exc)

    def _failure_recovery(self):
        if self.return_home_on_failure:
            ok, message = self._home()
            self.get_logger().warning("Failure recovery home: {} {}".format(ok, message))
        if self.release_on_failure:
            ok, message = self._gripper_release()
            self.get_logger().warning("Failure recovery release: {} {}".format(ok, message))

    def _step(self, state, action):
        self._check_abort()
        self._set_state(state)
        ok, message = action()
        if not ok:
            raise RuntimeError("{} failed: {}".format(state, message))

    def _check_abort(self):
        with self.lock:
            if self.abort_requested:
                raise RuntimeError("aborted by user")

    def _get_target_in_base(self):
        with self.lock:
            target = None if self.latest_target_base is None else list(self.latest_target_base)
            received = self.latest_target_received_monotonic

        if target is None:
            raise RuntimeError("no locked apple target has been received")
        if (
            self.target_timeout > 0.0
            and received is not None
            and time.monotonic() - received > self.target_timeout
        ):
            raise RuntimeError(
                "locked apple target is stale: {:.2f}s".format(
                    time.monotonic() - received
                )
            )

        return target

    def _target_message_to_base_point(self, target, kind):
        stamp = target.header.stamp
        if self.target_timeout > 0.0 and (stamp.sec != 0 or stamp.nanosec != 0):
            age = (self.get_clock().now() - Time.from_msg(stamp)).nanoseconds / 1e9
            if age > self.target_timeout:
                raise RuntimeError("apple target is stale: {:.2f}s".format(age))

        if not self.use_tf or target.header.frame_id in ("", self.base_frame):
            return self._target_message_point(target, kind)

        self._check_robot_end_tf_freshness()
        target_for_tf = self._copy_target_with_stamp(target, kind, use_latest=self.target_use_latest_tf)
        try:
            transformed = self.tf_buffer.transform(
                target_for_tf,
                self.base_frame,
                timeout=Duration(seconds=self.transform_timeout),
            )
        except Exception as exc:
            raise RuntimeError("failed to transform target to {}: {}".format(self.base_frame, exc))

        if kind == "pose":
            return [
                transformed.pose.position.x,
                transformed.pose.position.y,
                transformed.pose.position.z,
            ]
        return [transformed.point.x, transformed.point.y, transformed.point.z]

    def _check_robot_end_tf_freshness(self):
        try:
            transform = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.robot_end_frame,
                Time(),
                timeout=Duration(seconds=self.transform_timeout),
            )
        except Exception as exc:
            raise RuntimeError(
                "failed to read dynamic TF {} -> {}: {}".format(
                    self.base_frame, self.robot_end_frame, exc
                )
            )
        stamp = transform.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            raise RuntimeError(
                "dynamic TF {} -> {} has no timestamp".format(
                    self.base_frame, self.robot_end_frame
                )
            )
        age = (self.get_clock().now() - Time.from_msg(stamp)).nanoseconds / 1e9
        if self.max_robot_end_tf_age > 0.0 and age > self.max_robot_end_tf_age:
            raise RuntimeError(
                "dynamic TF {} -> {} is stale: {:.3f}s > {:.3f}s".format(
                    self.base_frame,
                    self.robot_end_frame,
                    age,
                    self.max_robot_end_tf_age,
                )
            )
        if age < -0.1:
            raise RuntimeError(
                "dynamic TF {} -> {} is {:.3f}s in the future".format(
                    self.base_frame, self.robot_end_frame, -age
                )
            )

    @staticmethod
    def _target_message_point(target, kind):
        point = target.pose.position if kind == "pose" else target.point
        return [point.x, point.y, point.z]

    @staticmethod
    def _copy_target_with_stamp(target, kind, use_latest):
        if kind == "pose":
            copied = PoseStamped()
            copied.pose = target.pose
        else:
            copied = PointStamped()
            copied.point = target.point
        copied.header.frame_id = target.header.frame_id
        copied.header.stamp = Time().to_msg() if use_latest else target.header.stamp
        return copied

    def _clear_locked_target(self):
        with self.lock:
            self.latest_target_base = None
            self.latest_target_stamp = None
            self.latest_target_received_monotonic = None
            self.latest_target_source_frame = ""
            self.latest_target_source_point = None
            self.latest_target_kind = None

    def _check_workspace(self, point, label="target"):
        if not self.enforce_workspace_limits:
            return
        for i, axis in enumerate(("x", "y", "z")):
            if point[i] < self.workspace_min[i] or point[i] > self.workspace_max[i]:
                raise RuntimeError(
                    "{} {}={} outside workspace [{}, {}]".format(
                        label, axis, point[i], self.workspace_min[i], self.workspace_max[i]
                    )
                )

    @staticmethod
    def _three_finite_values(values, label):
        result = [float(value) for value in values]
        if len(result) != 3 or not all(math.isfinite(value) for value in result):
            raise ValueError("{} must contain three finite values".format(label))
        return result

    def _validate_home_joint_values(self, values):
        joints = [float(value) for value in values]
        if len(joints) != len(self._JOINT_CHAIN):
            raise ValueError("home_joint_deg must contain six values")
        if not all(math.isfinite(value) for value in joints):
            raise ValueError("home_joint_deg must contain only finite values")
        for index, (joint_deg, (_xyz, _rpy, limits_rad)) in enumerate(
            zip(joints, self._JOINT_CHAIN)
        ):
            lower_deg = math.degrees(limits_rad[0])
            upper_deg = math.degrees(limits_rad[1])
            if joint_deg < lower_deg or joint_deg > upper_deg:
                raise ValueError(
                    "J{}={:.4f} deg is outside URDF limit [{:.4f}, {:.4f}] deg".format(
                        index + 1, joint_deg, lower_deg, upper_deg
                    )
                )

    @staticmethod
    def _matrix_multiply(first, second):
        return [
            [
                sum(first[row][index] * second[index][column] for index in range(3))
                for column in range(3)
            ]
            for row in range(3)
        ]

    @staticmethod
    def _matrix_vector(rotation, vector):
        return [
            sum(rotation[row][column] * vector[column] for column in range(3))
            for row in range(3)
        ]

    @staticmethod
    def _rpy_rotation(rpy):
        roll, pitch, yaw = [float(value) for value in rpy]
        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        return [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]

    @staticmethod
    def _z_rotation(angle):
        cosine, sine = math.cos(angle), math.sin(angle)
        return [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]

    def _axis_endpoints(self, joint_deg):
        joints = [float(value) for value in joint_deg]
        if len(joints) != len(self._JOINT_CHAIN):
            raise RuntimeError("arm state must contain six joint values")
        if not all(math.isfinite(value) for value in joints):
            raise RuntimeError("arm state contains a non-finite joint value")

        position = [0.0, 0.0, 0.0]
        rotation = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        endpoints = []
        for joint_value, (origin_xyz, origin_rpy, _limits) in zip(
            joints, self._JOINT_CHAIN
        ):
            translated = self._matrix_vector(rotation, origin_xyz)
            position = [position[index] + translated[index] for index in range(3)]
            rotation = self._matrix_multiply(
                rotation, self._rpy_rotation(origin_rpy)
            )
            endpoints.append(tuple(position))
            rotation = self._matrix_multiply(
                rotation, self._z_rotation(math.radians(joint_value))
            )
        return endpoints

    def _check_axis_endpoint_z(self, joint_deg, label="arm"):
        endpoints = self._axis_endpoints(joint_deg)
        violations = [
            (index + 1, point[2])
            for index, point in enumerate(endpoints)
            if point[2] < self.min_axis_endpoint_z
        ]
        if violations:
            details = ", ".join(
                "J{} axis z={:.4f} m".format(index, height)
                for index, height in violations
            )
            raise RuntimeError(
                "{} violates minimum axis height {:.4f} m: {}".format(
                    label, self.min_axis_endpoint_z, details
                )
            )
        return endpoints

    def _check_robot_end_z(self, pose, label="robot_end"):
        height = float(pose.position.z)
        if not math.isfinite(height):
            raise RuntimeError("{} z is not finite".format(label))
        if height < self.min_axis_endpoint_z:
            raise RuntimeError(
                "{} z={:.4f} m is below minimum {:.4f} m".format(
                    label, height, self.min_axis_endpoint_z
                )
            )

    def _safety_monitor_tick(self):
        with self.lock:
            active = self.active
        if not active:
            self._safety_state_future = None
            self._safety_stop_sent = False
            return
        if self._safety_stop_sent or not self.get_state_srv.service_is_ready():
            return

        future = self._safety_state_future
        if future is None:
            self._safety_state_future = self.get_state_srv.call_async(
                GetArmState.Request()
            )
            return
        if not future.done():
            return
        self._safety_state_future = None
        try:
            state = future.result()
            if state is None or not state.success or int(state.err) != 0:
                return
            self._check_axis_endpoint_z(state.joint_deg, "live arm state")
        except Exception as exc:
            # A missing state sample is handled by the normal connection
            # monitor.  Only an explicit height violation triggers a stop.
            if "violates minimum axis height" not in str(exc):
                return
            with self.lock:
                self.abort_requested = True
                self._safety_stop_sent = True
            self.get_logger().error(
                "Axis-height safety stop: {}".format(exc)
            )
            self._set_state("SAFETY_STOP")
            if self.stop_srv.service_is_ready():
                self.stop_srv.call_async(Trigger.Request())

    def _on_parameters_set(self, parameters):
        speed_attributes = {
            "movej_p_speed": "movej_p_speed",
            "movel_speed": "movel_speed",
            "home_speed": "home_speed",
        }
        updates = {}
        try:
            for parameter in parameters:
                attribute = speed_attributes.get(parameter.name)
                if attribute is not None:
                    value = int(parameter.value)
                    if value < 1 or value > 100:
                        raise ValueError(
                            "{} must be in [1, 100]".format(parameter.name)
                        )
                    updates[attribute] = value
                elif parameter.name == "home_joint_deg":
                    home = [float(value) for value in parameter.value]
                    self._validate_home_joint_values(home)
                    self._check_axis_endpoint_z(home, "new Home")
                    updates["home_joint_deg"] = home
                elif parameter.name == "gripper_tcp_xyz_m":
                    xyz = self._three_finite_values(
                        parameter.value, parameter.name
                    )
                    if any(abs(value) > 1.0 for value in xyz):
                        raise ValueError(
                            "gripper_tcp_xyz_m must be within +/-1 m"
                        )
                    updates["gripper_tcp_xyz"] = xyz
                elif parameter.name == "gripper_tcp_rpy_deg":
                    rpy = self._three_finite_values(
                        parameter.value, parameter.name
                    )
                    if any(abs(value) > 360.0 for value in rpy):
                        raise ValueError(
                            "gripper_tcp_rpy_deg must be within +/-360 deg"
                        )
                    updates["gripper_tcp_rpy_deg"] = rpy
                    updates["gripper_tcp_quaternion"] = quaternion_from_euler(
                        *[math.radians(value) for value in rpy]
                    )
        except ValueError as exc:
            return SetParametersResult(successful=False, reason=str(exc))

        if updates:
            with self.lock:
                if self.active:
                    return SetParametersResult(
                        successful=False,
                        reason="cannot change motion parameters during an active task",
                    )
                for attribute, value in updates.items():
                    setattr(self, attribute, value)
            self.get_logger().info(
                "Motion parameters updated: speeds={}/{}/{}%, Home={}, tool center xyz_m={} rpy_deg={}".format(
                    self.movej_p_speed,
                    self.movel_speed,
                    self.home_speed,
                    self._format_point(self.home_joint_deg),
                    self._format_point(self.gripper_tcp_xyz),
                    self._format_point(self.gripper_tcp_rpy_deg),
                )
            )
        return SetParametersResult(successful=True)

    def _build_pick_poses(self, target, orientation, approach_vector):
        grasp_point = [target[i] + self.grasp_offset[i] for i in range(3)]
        pre_point = [
            grasp_point[i] - approach_vector[i] * self.approach_distance
            for i in range(3)
        ]
        retreat_point = [
            grasp_point[i] - approach_vector[i] * self.retreat_distance
            for i in range(3)
        ]
        tip_poses = (
            self._make_tip_pose(pre_point, orientation),
            self._make_tip_pose(grasp_point, orientation),
            self._make_tip_pose(retreat_point, orientation),
        )
        end_poses = tuple(
            self._robot_end_pose_for_tip_pose(pose) for pose in tip_poses
        )
        return tip_poses, end_poses

    def _make_tip_pose(self, point, orientation):
        pose = Pose()
        pose.position.x = point[0]
        pose.position.y = point[1]
        pose.position.z = point[2]
        pose.orientation.x = orientation.x
        pose.orientation.y = orientation.y
        pose.orientation.z = orientation.z
        pose.orientation.w = orientation.w
        return pose

    def _robot_end_pose_for_tip_pose(self, tip_pose):
        tip_quaternion = self._pose_quaternion(tip_pose)
        end_quaternion = self._quaternion_multiply(
            tip_quaternion,
            self._quaternion_conjugate(self.gripper_tcp_quaternion),
        )
        offset_base = self._rotate_vector_by_quaternion(
            self.gripper_tcp_xyz, end_quaternion
        )
        pose = Pose()
        pose.position.x = tip_pose.position.x - offset_base[0]
        pose.position.y = tip_pose.position.y - offset_base[1]
        pose.position.z = tip_pose.position.z - offset_base[2]
        pose.orientation = self._quaternion_message(end_quaternion)
        return pose

    def _log_pick_plan(self, apple_center_base, tip_poses, end_poses):
        prefix = "DRY_RUN" if self.dry_run else "PLAN"
        with self.lock:
            source_frame = self.latest_target_source_frame or self.base_frame
            source_point = (
                None
                if self.latest_target_source_point is None
                else list(self.latest_target_source_point)
            )
        if source_point is not None:
            self.get_logger().info(
                "[{}] apple_center_{} xyz_m={}".format(
                    prefix, source_frame, self._format_point(source_point)
                )
            )
        self.get_logger().info(
            "[{}] apple_center_{} xyz_m={}".format(
                prefix, self.base_frame, self._format_point(apple_center_base)
            )
        )
        self.get_logger().info(
            "[{}] T_robot_end_gripper_tip_center xyz_m={} quat_xyzw={}".format(
                prefix,
                self._format_point(self.gripper_tcp_xyz),
                self._format_values(self.gripper_tcp_quaternion),
            )
        )
        pose_names = ("pre_grasp", "grasp")
        if not self.stop_after_grasp:
            pose_names += ("retreat",)
        for name, tip_pose, end_pose in zip(pose_names, tip_poses, end_poses):
            self.get_logger().info(
                "[{}] {} desired_gripper_tip_center_{} xyz_m={} quat_xyzw={}".format(
                    prefix,
                    name,
                    self.base_frame,
                    self._format_point(self._pose_point(tip_pose)),
                    self._format_quaternion(tip_pose),
                )
            )
            self.get_logger().info(
                "[{}] {} desired_robot_end_{} xyz_m={} quat_xyzw={}".format(
                    prefix,
                    name,
                    self.base_frame,
                    self._format_point(self._pose_point(end_pose)),
                    self._format_quaternion(end_pose),
                )
            )

    def _move_pose(self, client, pose, speed):
        try:
            self._check_robot_end_z(pose, "motion target robot_end")
        except Exception as exc:
            return False, str(exc)
        req = MovePose.Request()
        req.pose = pose
        req.speed = int(speed)
        req.blend_radius = int(self.blend_radius)
        req.wait = True
        try:
            command_type = "movej_p" if client is self.movej_p_srv else "movel"
            self._publish_expert_action(
                command_type=command_type,
                target_pose=pose,
                speed=req.speed,
            )
            resp = self._call_service(client, req)
            if not resp.success:
                return False, resp.message
            return self._wait_for_pose_reached(pose)
        except Exception as exc:
            return False, str(exc)

    def _wait_for_pose_reached(self, pose):
        target = self._pose_point(pose)
        deadline = time.monotonic() + self.motion_verify_timeout
        last_error = None
        while time.monotonic() < deadline:
            self._check_abort()
            try:
                state = self._get_checked_arm_state()
                self._check_axis_endpoint_z(state.joint_deg, "live arm state")
                current = self._pose_point(state.pose)
                error = math.sqrt(
                    sum((current[i] - target[i]) ** 2 for i in range(3))
                )
                if error <= self.motion_position_tolerance:
                    return True, "motion reached target (position error {:.4f} m)".format(error)
                last_error = error
            except Exception as exc:
                last_error = str(exc)
            time.sleep(0.1)
        return False, "motion target not reached within {:.1f}s (last error {})".format(
            self.motion_verify_timeout, last_error
        )

    def _retreat_with_current_orientation(self, retreat_tip_pose):
        try:
            state = self._get_checked_arm_state()
        except Exception as exc:
            return False, str(exc)

        retreat_tip_pose.orientation = self._quaternion_message(
            self._quaternion_multiply(
                self._pose_quaternion(state.pose),
                self.gripper_tcp_quaternion,
            )
        )
        retreat_end_pose = self._robot_end_pose_for_tip_pose(retreat_tip_pose)
        return self._move_pose(self.movel_srv, retreat_end_pose, self.movel_speed)

    def _home(self):
        try:
            self._validate_home_joint_values(self.home_joint_deg)
            self._check_axis_endpoint_z(self.home_joint_deg, "Home target")
        except Exception as exc:
            return False, str(exc)
        req = MoveJ.Request()
        req.joint_deg = list(self.home_joint_deg)
        req.speed = int(self.home_speed)
        req.blend_radius = int(self.blend_radius)
        req.wait = True
        try:
            self._publish_expert_action(
                command_type="movej",
                target_joint_deg=req.joint_deg,
                speed=req.speed,
            )
            resp = self._call_service(self.movej_srv, req)
            if not resp.success:
                return False, resp.message
            return self._wait_for_home_reached()
        except Exception as exc:
            return False, str(exc)

    def _wait_for_home_reached(self):
        deadline = time.monotonic() + self.motion_verify_timeout
        last_error = None
        while time.monotonic() < deadline:
            self._check_abort()
            try:
                state = self._get_checked_arm_state()
                self._check_axis_endpoint_z(state.joint_deg, "live arm state")
                if len(state.joint_deg) != len(self.home_joint_deg):
                    last_error = "arm state did not return six joints"
                else:
                    error = max(
                        abs(state.joint_deg[i] - self.home_joint_deg[i])
                        for i in range(len(self.home_joint_deg))
                    )
                    if error <= self.home_joint_tolerance:
                        return True, "home reached (max joint error {:.3f} deg)".format(error)
                    last_error = "max joint error {:.3f} deg".format(error)
            except Exception as exc:
                last_error = str(exc)
            time.sleep(0.1)
        return False, "home target not reached within {:.1f}s ({})".format(
            self.motion_verify_timeout, last_error
        )

    def _gripper_release(self):
        req = GripperRelease.Request()
        req.speed = int(self.gripper_release_speed)
        req.wait = True
        try:
            self._publish_expert_action(
                command_type="gripper_release",
                gripper_command=0,
                speed=req.speed,
            )
            resp = self._call_service(self.gripper_release_srv, req)
            return resp.success, resp.message
        except Exception as exc:
            return False, str(exc)

    def _gripper_pick(self):
        req = GripperPick.Request()
        req.speed = int(self.gripper_pick_speed)
        req.force = int(self.gripper_pick_force)
        req.wait = True
        try:
            self._publish_expert_action(
                command_type="gripper_pick",
                gripper_command=1,
                speed=req.speed,
                force=req.force,
            )
            resp = self._call_service(self.gripper_pick_srv, req)
            return resp.success, resp.message
        except Exception as exc:
            return False, str(exc)

    def _twist_joint(self):
        try:
            state = self._get_checked_arm_state()
        except Exception as exc:
            return False, str(exc)
        joints = list(state.joint_deg)
        if len(joints) <= self.twist_joint_index:
            return False, "arm state did not return joint {}".format(self.twist_joint_index + 1)
        joints[self.twist_joint_index] += self.twist_delta_deg
        if joints[self.twist_joint_index] < self.twist_min_deg or joints[self.twist_joint_index] > self.twist_max_deg:
            return False, "twist target joint limit exceeded: {:.3f} deg".format(joints[self.twist_joint_index])
        try:
            self._check_axis_endpoint_z(joints, "twist target")
        except Exception as exc:
            return False, str(exc)

        req = MoveJ.Request()
        req.joint_deg = joints
        req.speed = int(self.twist_speed)
        req.blend_radius = int(self.blend_radius)
        req.wait = True
        try:
            self._publish_expert_action(
                command_type="movej",
                target_joint_deg=req.joint_deg,
                speed=req.speed,
            )
            resp = self._call_service(self.movej_srv, req)
            return resp.success, resp.message
        except Exception as exc:
            return False, str(exc)

    def _get_checked_arm_state(self):
        state = self._call_service(self.get_state_srv, GetArmState.Request())
        if not state.success:
            raise RuntimeError(state.message)
        if int(state.err) != 0:
            raise RuntimeError("mechanical arm reports error code {}".format(state.err))
        return state

    def _call_service(self, client, request, timeout=None):
        done = threading.Event()
        future = client.call_async(request)
        future.add_done_callback(lambda _future: done.set())
        wait_timeout = self.service_call_timeout if timeout is None else float(timeout)
        if not done.wait(wait_timeout):
            future.cancel()
            raise RuntimeError("timeout waiting for service response")
        if future.exception() is not None:
            raise future.exception()
        result = future.result()
        if result is None:
            raise RuntimeError("service returned no response")
        return result

    def _set_state(self, state):
        self.current_state = str(state)
        self.get_logger().info("picker state: {}".format(state))
        self.state_pub.publish(String(data=state))

    def _publish_expert_action(
        self,
        command_type,
        target_pose=None,
        target_joint_deg=None,
        gripper_command=-1,
        speed=0,
        force=0,
    ):
        """Describe the next teacher command without changing its execution."""
        try:
            nan = float("nan")
            message = ExpertAction()
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = self.base_frame
            message.phase = self.current_state
            message.command_type = str(command_type)
            message.target_tcp = [nan] * 7
            if target_pose is not None:
                message.target_tcp = [
                    float(target_pose.position.x),
                    float(target_pose.position.y),
                    float(target_pose.position.z),
                    float(target_pose.orientation.x),
                    float(target_pose.orientation.y),
                    float(target_pose.orientation.z),
                    float(target_pose.orientation.w),
                ]
            message.target_joint_rad = [nan] * 6
            if target_joint_deg is not None and len(target_joint_deg) == 6:
                message.target_joint_rad = [
                    math.radians(float(value)) for value in target_joint_deg
                ]
            message.gripper_command = int(gripper_command)
            message.speed = int(speed)
            message.force = int(force)
            self.expert_action_pub.publish(message)
        except Exception as exc:
            # Observation must never gate or alter the command path.
            self.get_logger().warning(
                "Expert-action telemetry publish failed: {}".format(exc)
            )

    def _publish_outcome(self, success, message):
        """Publish the state-machine result before active becomes false."""
        try:
            outcome = TaskOutcome()
            outcome.header.stamp = self.get_clock().now().to_msg()
            outcome.success = bool(success)
            outcome.terminal_state = self.current_state
            outcome.failure_reason = "" if success else str(message)
            # No verified sensor proves that an apple is physically held.
            outcome.physical_grasp_success = -1
            self.outcome_pub.publish(outcome)
        except Exception as exc:
            self.get_logger().warning(
                "Task-outcome telemetry publish failed: {}".format(exc)
            )

    @staticmethod
    def _normalized(values):
        vector = [float(v) for v in values]
        norm = math.sqrt(sum(v * v for v in vector))
        if norm < 1e-9:
            raise ValueError("approach_vector_base must not be zero")
        return [v / norm for v in vector]

    @staticmethod
    def _normalized_quaternion(quaternion):
        values = [float(value) for value in quaternion]
        if len(values) != 4 or not all(math.isfinite(value) for value in values):
            raise ValueError("quaternion must contain four finite values")
        norm = math.sqrt(sum(value * value for value in values))
        if norm < 1e-9:
            raise ValueError("quaternion must not be zero")
        return tuple(value / norm for value in values)

    @classmethod
    def _quaternion_multiply(cls, first, second):
        ax, ay, az, aw = cls._normalized_quaternion(first)
        bx, by, bz, bw = cls._normalized_quaternion(second)
        return cls._normalized_quaternion(
            (
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            )
        )

    @classmethod
    def _quaternion_conjugate(cls, quaternion):
        x, y, z, w = cls._normalized_quaternion(quaternion)
        return (-x, -y, -z, w)

    @staticmethod
    def _pose_quaternion(pose):
        return (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )

    @classmethod
    def _quaternion_message(cls, quaternion):
        x, y, z, w = cls._normalized_quaternion(quaternion)
        return Quaternion(x=x, y=y, z=z, w=w)

    @staticmethod
    def _rotate_vector_by_quaternion(vector, quaternion):
        qx, qy, qz, qw = [float(v) for v in quaternion]
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm < 1e-9:
            raise ValueError("gripper tip orientation quaternion must not be zero")
        qx, qy, qz, qw = [v / norm for v in (qx, qy, qz, qw)]
        vx, vy, vz = [float(v) for v in vector]
        return [
            (1.0 - 2.0 * (qy * qy + qz * qz)) * vx
            + 2.0 * (qx * qy - qz * qw) * vy
            + 2.0 * (qx * qz + qy * qw) * vz,
            2.0 * (qx * qy + qz * qw) * vx
            + (1.0 - 2.0 * (qx * qx + qz * qz)) * vy
            + 2.0 * (qy * qz - qx * qw) * vz,
            2.0 * (qx * qz - qy * qw) * vx
            + 2.0 * (qy * qz + qx * qw) * vy
            + (1.0 - 2.0 * (qx * qx + qy * qy)) * vz,
        ]

    @staticmethod
    def _pose_point(pose):
        return [pose.position.x, pose.position.y, pose.position.z]

    @staticmethod
    def _format_point(point):
        return "[{:.6f}, {:.6f}, {:.6f}]".format(*point)

    @staticmethod
    def _format_values(values):
        return "[{}]".format(
            ", ".join("{:.6f}".format(float(value)) for value in values)
        )

    @staticmethod
    def _format_quaternion(pose):
        return "[{:.6f}, {:.6f}, {:.6f}, {:.6f}]".format(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )


def main(args=None):
    rclpy.init(args=args)
    node = PickTaskManager()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
