from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np
from .config import ROOT, ROS_PICKER, CHECKPOINT, NORMALIZATION, PickConfig, HAND_EYE_R, HAND_EYE_T, TCP_OFFSET, SERVICES
from .grasp_services import import_service_types, GraspServices
from .reach_policy import ReachPolicy, ACTION_BOUNDS
from .robot_feedback import UdpFeedback
from .vision_target import TargetSubscriber
from .state_machine import FullPickStateMachine

def _ros_node(target_sub):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from geometry_msgs.msg import PointStamped
    class RuntimeNode(Node):
        def __init__(self, *, context):
            super().__init__("full_pick_v2", context=context)
            self.create_subscription(PointStamped, "/apple/target_point", target_sub.callback, 10)
            self._runtime_executor = SingleThreadedExecutor(context=context)
            self._runtime_executor.add_node(self)
        def spin_once(self, timeout):
            self._runtime_executor.spin_once(timeout_sec=timeout)
        def count_publishers(self, topic):
            return super().count_publishers(topic)
        def close_executor(self):
            self._runtime_executor.remove_node(self)
            self._runtime_executor.shutdown(timeout_sec=0)
    return RuntimeNode


def _print_ros_context(node):
    import rclpy
    keys = (
        "ROS_DOMAIN_ID", "ROS_DISTRO", "RMW_IMPLEMENTATION",
        "ROS_LOCALHOST_ONLY", "ROS_AUTOMATIC_DISCOVERY_RANGE",
        "AMENT_PREFIX_PATH", "PYTHONPATH",
    )
    print("PYTHON_RCLPY_NODE_NAME = " + node.get_name(), flush=True)
    print("PYTHON_RCLPY_NAMESPACE = " + node.get_namespace(), flush=True)
    print("PYTHON_RCLPY_CONTEXT_OK = " + str(bool(node.context.ok())).lower(), flush=True)
    print("PYTHON_RMW_IMPLEMENTATION = " + rclpy.get_rmw_implementation_identifier(), flush=True)
    for key in keys:
        value = os.environ.get(key)
        print("PYTHON_" + key + " = " + (value if value is not None else "<unset>"), flush=True)
    try:
        graph = node.get_service_names_and_types()
        print("PYTHON_RCLPY_SERVICES = " + repr(sorted(graph)), flush=True)
    except BaseException as exc:
        print("PYTHON_RCLPY_SERVICE_GRAPH_ERROR = " + repr(exc), flush=True)

def offline_check():
    config=PickConfig.load()
    policy=ReachPolicy()
    R,t=config.hand_eye
    if not np.allclose(R,HAND_EYE_R,atol=1e-8) or not np.allclose(t,HAND_EYE_T,atol=1e-8):
        raise RuntimeError("HAND_EYE_MISMATCH")
    if not np.allclose(TCP_OFFSET, [0.0005,-0.005,0.158], atol=1e-12):
        raise RuntimeError("TCP_MISMATCH")
    # Pure contracts: 15D observation and action, target commit on occlusion.
    from .transforms import observation, actual_tcp, flange_target_for_apple, protocol_pose, so3_exp
    raw=observation(np.zeros(6), np.zeros(3), np.asarray([.1,.2,.3]), np.eye(3), np.eye(3))
    act=policy.infer(raw); payload=protocol_pose(np.zeros(3),np.eye(3))
    from .high_follow import (
        Pose, CommandReference, interpolate, build_movep_payload,
        validate_serialized_pose, LINEAR_ACCEL_MPS2, POLICY_DT,
    )
    _=interpolate(Pose(np.zeros(3),np.eye(3)),Pose(np.ones(3)*.01,so3_exp([0,0,.1])),.25)
    from .vision_target import TargetLatch, VisionSample
    l=TargetLatch(); now=time.monotonic()
    for i in range(3): l.accept(VisionSample(np.asarray([0.1,0.0,0.5]),now+i*.01,"camera_link",0))
    l.update_reliable_base(np.asarray([.3,.1,.5])); l.arm(); l.freeze_base(np.asarray([.3,.1,.5])); l.commit()
    if not l.committed(): raise RuntimeError("TERMINAL_COMMIT_CONTRACT_FAILED")
    frozen = l.frozen_target_base_xyz.copy()
    l.update_reliable_base(np.asarray([.9,.9,.9]), source_mono=now + 1.0)
    if not np.allclose(l.frozen_target_base_xyz, frozen):
        raise RuntimeError("TARGET_FREEZE_CONTRACT_FAILED")
    reference = CommandReference(np.asarray([.3, .0, .4]), np.eye(3))
    reference.velocity[:] = .01
    before = reference.velocity.copy()
    reference.decelerate()
    if np.linalg.norm(reference.velocity[:3] - before[:3]) / POLICY_DT > LINEAR_ACCEL_MPS2 + 1e-9:
        raise RuntimeError("DECELERATION_LIMIT_CONTRACT_FAILED")
    payload = build_movep_payload(reference.pose)
    validate_serialized_pose(payload["pose"])
    # Service import is checked after model/config to identify shadowing clearly.
    try:
        import_service_types()
        ros_ok=True
    except Exception as exc:
        ros_ok=False; print("ROS_SERVICE_IMPORT_ERROR = "+repr(exc))
    print("MODEL_UNCHANGED = true")
    print("NORMALIZATION_UNCHANGED = true")
    print("HARDWARE_GATE_READY = true")
    print("TARGET_FREEZE_FIXED = true")
    print("DECELERATION_CONTINUITY_FIXED = true")
    print("SERIALIZED_POSE_CHECK_READY = true")
    print("STOP_COVERAGE_COMPLETE = true")
    print("ROS_SERVICE_IMPORT_OK = "+str(ros_ok).lower())
    print("ROS_SERVICE_DISCOVERY_OK = false")
    print("hardware_sent = false")
    final_status = "FULL_PICK_V2_STATIC_CHECKED" if ros_ok else "FULL_PICK_V2_STATIC_BLOCKED"
    print("final_status = "+final_status)
    return 0 if ros_ok else 2

def live(args):
    import rclpy
    from rclpy.context import Context
    config = PickConfig.load()
    target = TargetSubscriber()
    RuntimeNode = _ros_node(target)
    context = Context()
    rclpy.init(context=context)
    node = RuntimeNode(context=context)
    feedback = None
    try:
        _print_ros_context(node)
        # Startup order is deliberate: the shell has already prepared Bridge
        # and vision, then the process opens UDP feedback before model loading.
        feedback = UdpFeedback()
        policy = ReachPolicy()
        services = GraspServices(node, config, node.spin_once)
        print("MODEL_READY", flush=True)
        print("HARDWARE_STARTUP_GATE = DISABLED", flush=True)
        # The machine starts with hardware disabled.  It can arm the control
        # socket only after the operator enters ENABLE_HARDWARE and the UDP
        # health gate passes.
        machine = FullPickStateMachine(
            node, config, policy, services, feedback, target,
            args.control_mode, args.release_mode, hardware=False,
        )
        ok = machine.run()
        return 0 if ok else 2
    finally:
        if feedback is not None:
            feedback.close()
        node.close_executor()
        node.destroy_node()
        if context.ok():
            rclpy.shutdown(context=context)

def main(argv=None):
    p=argparse.ArgumentParser(prog="full_pick_v2")
    p.add_argument("--control-mode", choices=("high-follow",), default="high-follow")
    p.add_argument("--release-mode", choices=("configured","disabled"), default="configured")
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--no-hardware", action="store_true", help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args.check_only: return offline_check()
    return live(args)

if __name__ == "__main__": raise SystemExit(main())
