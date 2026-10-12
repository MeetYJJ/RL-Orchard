from __future__ import annotations
from collections import deque
from enum import Enum
import logging
import time
import numpy as np
from .config import TCP_OFFSET
from .transforms import actual_tcp, apple_camera_to_base, flange_target_for_apple, observation, protocol_pose, q_from_R, realman_udp_euler_to_matrix, so3_log
from .high_follow import HighFollow, CommandReference, Pose, POLICY_DT
from .reach_policy import ACTION_BOUNDS
from .safety import PickFault, Safety, SettleWindow, MotionWindow
from .vision_target import TargetLatch

logger = logging.getLogger(__name__)

# Transport keeps the existing 150 ms heartbeat contract.  The 100/80 ms
# levels below are diagnostics only: they never change control behavior.
HEARTBEAT_DIAGNOSTIC_WARN_MS = 100.0
HEARTBEAT_DIAGNOSTIC_CLEAR_MS = 80.0
STOP_FEEDBACK_TIMEOUT_S = 3.0
STOP_FEEDBACK_WINDOW_S = 0.20
STOP_TCP_SPAN_M = 0.0005
STOP_TCP_SPEED_MPS = 0.002
STOP_ANGULAR_SPEED_RPS = 0.020
TERMINAL_FREEZE_DISTANCE_M = 0.100
TERMINAL_COMMIT_DISTANCE_M = 0.080

class Phase(str, Enum):
    INIT="INIT"; PRECHECK="PRECHECK"; TARGET_ACQUISITION="TARGET_ACQUISITION"; TRACKING_REACH="TRACKING_REACH"; TERMINAL_REACH="TERMINAL_REACH"; REACH_DECELERATING="REACH_DECELERATING"; TCP_SETTLED="TCP_SETTLED"; HIGH_FOLLOW_STOPPED="HIGH_FOLLOW_STOPPED"; CLOSE="CLOSE"; HOME="HOME"; RELEASE="RELEASE"; DONE="DONE"; FAILED="FAILED"

class FullPickStateMachine:
    def __init__(self, node, config, policy, services, feedback, target_sub, control_mode="high-follow", release_mode="configured", hardware=False):
        self.node, self.config, self.policy, self.services = node, config, policy, services
        self.feedback, self.target_sub = feedback, target_sub
        self.control_mode, self.release_mode = control_mode, release_mode
        # A caller cannot silently arm a robot socket.  Hardware is enabled
        # only after the explicit startup gate in wait_for_start().
        self.hardware = False
        self._hardware_requested = bool(hardware)
        self.phase = Phase.INIT
        self.safety = Safety(config)
        self.transport = None
        self.held_R = None
        self.apple_base = None
        self.goal_end = None
        self.reference = None
        self.reason = ""
        self.rotation_diagnostics_printed = False
        self.close_done = self.home_done = self.release_done = False
        self._fault_stop_attempted = False
        self._stop_confirmed = False
        self._hardware_motion_locked = False
        self._service_motion_active = False
        self._stop_reason = None
        self._udp_state_samples = 0
        self._last_udp_age_ms = None
        self._max_udp_age_ms = 0.0
        self._last_inference_ms = None
        self._max_inference_ms = 0.0
        self._inference_count = 0
        self._last_publish_udp_age_ms = None
        self._first_publish_recorded = False
        self._timing_warning_active = {
            "state": False,
            "inference": False,
            "publish": False,
        }

    def set_phase(self, phase):
        self.phase = phase
        print(f"PHASE -> {phase.value}", flush=True)

    def _spin(self, timeout=.005):
        self.node.spin_once(timeout)
        self.feedback.poll()

    def _timing_warning(self, key, label, value_ms):
        """Print a timing warning on threshold entry, not every loop."""
        value_ms = float(value_ms)
        active = bool(self._timing_warning_active[key])
        if value_ms >= HEARTBEAT_DIAGNOSTIC_WARN_MS:
            if not active:
                print(f"{label} = {value_ms:.3f}", flush=True)
            self._timing_warning_active[key] = True
        elif value_ms < HEARTBEAT_DIAGNOSTIC_CLEAR_MS:
            self._timing_warning_active[key] = False

    def _record_udp_age(self, receipt_mono, label="UDP_AGE_AT_STATE_MS", warning_key="state", count_state=True):
        # receipt_mono is the original UDP receive timestamp.  It is never
        # replaced with the current time for either diagnostics or transport.
        age_ms = max(0.0, (time.monotonic() - float(receipt_mono)) * 1000.0)
        if count_state:
            self._udp_state_samples += 1
            self._last_udp_age_ms = age_ms
            self._max_udp_age_ms = max(self._max_udp_age_ms, age_ms)
        self._timing_warning(warning_key, label, age_ms)
        return age_ms

    def _record_inference(self, elapsed_s):
        elapsed_ms = max(0.0, float(elapsed_s) * 1000.0)
        self._inference_count += 1
        self._last_inference_ms = elapsed_ms
        self._max_inference_ms = max(self._max_inference_ms, elapsed_ms)
        self._timing_warning("inference", "ACTOR_INFERENCE_MS", elapsed_ms)
        return elapsed_ms

    def _infer_policy(self, raw):
        started = time.monotonic()
        try:
            return self.policy.infer(raw)
        finally:
            self._record_inference(time.monotonic() - started)

    def _print_timing_summary(self):
        print("UDP_STATE_SAMPLES = %d" % self._udp_state_samples, flush=True)
        if self._last_udp_age_ms is not None:
            print("UDP_AGE_AT_STATE_LAST_MS = %.3f" % self._last_udp_age_ms, flush=True)
            print("UDP_AGE_AT_STATE_MAX_MS = %.3f" % self._max_udp_age_ms, flush=True)
        print("ACTOR_INFERENCE_COUNT = %d" % self._inference_count, flush=True)
        if self._last_inference_ms is not None:
            print("ACTOR_INFERENCE_LAST_MS = %.3f" % self._last_inference_ms, flush=True)
            print("ACTOR_INFERENCE_MAX_MS = %.3f" % self._max_inference_ms, flush=True)
        if self._last_publish_udp_age_ms is not None:
            print("UDP_AGE_BEFORE_PUBLISH_LAST_MS = %.3f" % self._last_publish_udp_age_ms, flush=True)

    def _state(self):
        self._spin()
        state = self.feedback.latest
        if state is not None:
            try:
                self._rotation(state)
            except (TypeError, ValueError) as exc:
                raise PickFault("UDP_ROTATION_INVALID:" + str(exc)) from exc
            self._record_udp_age(state.receipt_mono)
        self.safety.feedback(state)
        return state

    def _rotation(self, state):
        return realman_udp_euler_to_matrix(state.euler_raw)

    def _print_rotation_diagnostics(self, state, rotation):
        if self.rotation_diagnostics_printed:
            return
        raw = np.asarray(state.euler_raw, dtype=np.float64)
        rotation = np.asarray(rotation, dtype=np.float64)
        print("ROTATION_SOURCE = RealMan UDP waypoint.euler", flush=True)
        print("ROTATION_RAW = " + repr(raw.tolist()), flush=True)
        print("ROTATION_FORMAT = roll,pitch,yaw", flush=True)
        print("ROTATION_UNIT = 0.001 rad", flush=True)
        print("ROTATION_CONVERSION_FUNCTION = orchard_picker.protocol.protocol_to_pose -> training.deployment.pose_math.quaternion_xyzw_to_matrix", flush=True)
        print("ROTATION_SHAPE = " + repr(rotation.shape), flush=True)
        print("ROTATION_DETERMINANT = %.12f" % float(np.linalg.det(rotation)), flush=True)
        print("ROTATION_ORTHOGONALITY_ERROR = %.12e" % float(np.linalg.norm(rotation @ rotation.T - np.eye(3))), flush=True)
        self.rotation_diagnostics_printed = True

    def _publisher_ready(self):
        return int(self.node.count_publishers("/apple/target_point")) > 0

    def precheck(self, wait_s=20.):
        self.set_phase(Phase.PRECHECK)
        deadline = time.monotonic()+wait_s
        last_feedback_error = None
        while time.monotonic() < deadline:
            # Wait through the same rclpy Node/Context used by the clients.
            # ``ros2 service list`` may be backed by a stale CLI daemon, so it
            # is never used as evidence that these clients can reach a server.
            remaining = max(0.01, min(.25, deadline - time.monotonic()))
            if not self.services.wait_for_required(wait_s=remaining):
                self._spin(.05); continue
            if not self._publisher_ready():
                self._spin(.1); continue
            try:
                state = self._state()
            except PickFault as exc:
                last_feedback_error = str(exc)
                self._spin(.1); continue
            if state is not None:
                self.services.diagnose(wait_s=0.0, emit=True)
                print("BRIDGE_READY", flush=True); print("VISION_READY", flush=True); print("UDP_READY", flush=True)
                return True
        report = self.services.diagnose(wait_s=0.0, emit=True)
        missing = self.services.missing()
        print("PRECHECK_MISSING_SERVICES = "+repr(missing), flush=True)
        if not self._publisher_ready(): print("PRECHECK_MISSING_VISION_PUBLISHER = /apple/target_point", flush=True)
        if last_feedback_error: print("PRECHECK_UDP_ERROR = "+last_feedback_error, flush=True)
        endpoint_missing = [
            item["name"] for item in report.values()
            if item["visible"] and item["type_match"] and not item["ready"]
        ]
        if endpoint_missing:
            print("PRECHECK_VISIBLE_BUT_NOT_READY = " + repr(endpoint_missing), flush=True)
            raise PickFault("PRECHECK_SERVICE_ENDPOINT_UNAVAILABLE")
        raise PickFault("PRECHECK_SERVICES_NOT_VISIBLE")

    @staticmethod
    def _all_zero(value):
        if value is None:
            return False
        values = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
        try:
            return bool(values) and all(int(item) == 0 for item in values)
        except (TypeError, ValueError):
            return False

    def _hardware_startup_preflight(self, samples=3):
        """Require consecutive healthy UDP frames before arming hardware."""
        if not self.services.wait_for_required(wait_s=.5):
            raise PickFault("HARDWARE_GATE_SERVICES_NOT_READY")
        for index in range(int(samples)):
            state = self._state()
            enabled = getattr(state, "joint_en_flag", None)
            if (state is None or not state.fresh or not state.healthy or
                    not isinstance(enabled, (list, tuple, np.ndarray)) or
                    len(enabled) != 6 or not all(bool(int(item)) for item in enabled) or
                    not self._all_zero(getattr(state, "err", None)) or
                    not self._all_zero(getattr(state, "joint_err_code", None))):
                raise PickFault("HARDWARE_GATE_FEEDBACK_NOT_SAFE")
            print("HARDWARE_GATE_FEEDBACK_SAMPLE = %d" % (index + 1), flush=True)
            self._spin(.01)
        print("HARDWARE_GATE_FEEDBACK_OK = true", flush=True)

    def wait_for_start(self):
        print("\\nREADY_FULL_PICK\\n", flush=True)
        print("HARDWARE_GATE_READY = true", flush=True)
        enable = input("Enter ENABLE_HARDWARE: ").strip()
        if enable != "ENABLE_HARDWARE":
            raise PickFault("HARDWARE_ENABLE_CANCELLED")
        self._hardware_startup_preflight()
        self.hardware = True
        print("HARDWARE_STARTUP_GATE = ENABLED", flush=True)
        value = input("Enter START_FULL_PICK: ").strip()
        if value != "START_FULL_PICK":
            raise PickFault("START_CANCELLED")

    def _target_base_now(self, state, rotation=None):
        if self.target_sub.latch.target is None:
            return None
        R = self._rotation(state) if rotation is None else rotation
        return apple_camera_to_base(state.end_xyz_m, R, self.target_sub.latch.target.camera_xyz)

    def _update_latch(self, state, rotation=None):
        latch = self.target_sub.latch
        if latch.frozen():
            return latch.frozen_target_base_xyz.copy()
        base = self._target_base_now(state, rotation)
        if (base is not None and latch.target is not None and
                latch.stable_count >= latch.required_samples):
            # Only a new camera callback may update the base target.  Reusing
            # an old eye-in-hand point must not refresh its timestamp.
            latch.update_reliable_base(
                base, source_mono=latch.target.received_mono
            )
        return latch.reliable_base_xyz

    def _wait_for_target(self):
        self.set_phase(Phase.TARGET_ACQUISITION)
        deadline = time.monotonic()+30.
        last_udp_error = None
        while time.monotonic() < deadline:
            try:
                state = self._state()
            except PickFault as exc:
                reason = str(exc)
                if reason.startswith(("UDP_UNAVAILABLE", "UDP_STALE", "UDP_ROTATION_INVALID")):
                    last_udp_error = reason
                    continue
                raise
            if state is None: continue
            R = self._rotation(state)
            if self.held_R is None:
                self.held_R = R.copy()
                self._print_rotation_diagnostics(state, R)
                print("HELD_ORIENTATION_CAPTURED = true", flush=True)
            base = self._update_latch(state, R)
            if self.target_sub.latch.reliable_base_xyz is not None:
                self.apple_base = self.target_sub.latch.reliable_base_xyz.copy()
                print("TARGET_RELIABLE = true; stable_samples=3", flush=True)
                return state
        malformed = getattr(self.feedback, "last_malformed_error", None)
        if malformed:
            print("TARGET_ACQUISITION_LAST_REJECTED_UDP = " + malformed, flush=True)
        if last_udp_error:
            print("TARGET_ACQUISITION_UDP_ERROR = " + last_udp_error, flush=True)
        raise PickFault("TARGET_ACQUISITION_TIMEOUT:" + (last_udp_error or malformed or "NO_VALID_CAMERA_TARGET"))

    def _start_transport(self, initial_pose):
        if self.transport is not None:
            raise PickFault("TRANSPORT_ALREADY_STARTED")
        start_mono = time.monotonic()
        try:
            self.transport = HighFollow(hardware=self.hardware)
            # The worker keeps sequence zero until the first publish, so this
            # may open its TCP connection but cannot send a motion payload.
            self.transport.start(initial_pose)
        except BaseException:
            if self.transport is not None:
                try:
                    self.transport.close()
                except Exception as cleanup_exc:
                    print("TRANSPORT_CLEANUP_ERROR = " + str(cleanup_exc), flush=True)
                else:
                    self.transport = None
            raise
        finally:
            print("TRANSPORT_START_LATENCY_MS = %.3f" %
                  ((time.monotonic() - start_mono) * 1000.0), flush=True)
        return time.monotonic()

    def _udp_after_transport_start(self, start_complete_mono, wait_s=1.0):
        deadline = time.monotonic() + wait_s
        last_error = "NO_NEW_UDP_AFTER_TRANSPORT_START"
        while time.monotonic() < deadline:
            try:
                state = self._state()
            except PickFault as exc:
                reason = str(exc)
                if reason.startswith(("UDP_UNAVAILABLE", "UDP_STALE", "UDP_ROTATION_INVALID")):
                    last_error = reason
                    continue
                raise
            if state.receipt_mono > start_complete_mono:
                return state
        raise PickFault("POST_TRANSPORT_UDP_TIMEOUT:" + last_error)

    def _wait_for_stop_feedback(self, timeout=STOP_FEEDBACK_TIMEOUT_S):
        """Require a fresh, continuous UDP window before declaring stopped."""
        samples = deque()
        stable_since = None
        last_receipt = None
        deadline = time.monotonic() + float(timeout)
        while time.monotonic() < deadline:
            self._spin(.01)
            state = self.feedback.latest
            if state is None or not state.fresh or not state.healthy:
                stable_since = None
                continue
            receipt = float(state.receipt_mono)
            if last_receipt is not None and receipt <= last_receipt:
                continue
            try:
                rotation = self._rotation(state)
                tcp = actual_tcp(state.end_xyz_m, rotation)
            except (TypeError, ValueError, FloatingPointError):
                stable_since = None
                continue
            last_receipt = receipt
            samples.append((receipt, tcp.copy(), rotation.copy()))
            while len(samples) > 2 and receipt - samples[1][0] >= STOP_FEEDBACK_WINDOW_S:
                samples.popleft()
            if len(samples) < 3 or receipt - samples[0][0] < STOP_FEEDBACK_WINDOW_S:
                continue
            first = samples[0]
            span = float(np.max(np.linalg.norm(
                np.asarray([item[1] for item in samples]) - tcp, axis=1)))
            duration = receipt - first[0]
            speed = float(np.linalg.norm(tcp - first[1]) / max(duration, 1e-9))
            # Compute the angular rate through the same validated SO(3) log
            # path used by the normal settle check.
            angular_speed = float(np.linalg.norm(so3_log(rotation @ first[2].T)) / max(duration, 1e-9))
            stable = (
                span <= STOP_TCP_SPAN_M and
                speed <= STOP_TCP_SPEED_MPS and
                angular_speed <= STOP_ANGULAR_SPEED_RPS
            )
            if stable:
                if stable_since is None:
                    stable_since = receipt
                if receipt - stable_since >= STOP_FEEDBACK_WINDOW_S:
                    print("STOP_FEEDBACK_STATIC = true", flush=True)
                    print("STOP_FEEDBACK_WINDOW_MS = %.3f" %
                          ((receipt - stable_since) * 1000.0), flush=True)
                    return True
            else:
                stable_since = None
        return False

    def stop_on_fault(self, reason):
        """Fence High-Follow, request controller stop once, then verify UDP."""
        if self._fault_stop_attempted:
            return self._stop_confirmed
        self._fault_stop_attempted = True
        self._hardware_motion_locked = True
        self._stop_reason = str(reason)
        print("STOP_ON_FAULT = " + self._stop_reason, flush=True)

        transport_barrier_ok = True
        active_motion = bool(getattr(self, "_service_motion_active", False))
        if self.transport is not None:
            active_motion = active_motion or bool(getattr(self.transport, "sequence", 0))
            try:
                transport_barrier_ok = bool(self.transport.request_stop(reason))
                print("TRANSPORT_STOP_BARRIER = " +
                      str(transport_barrier_ok).lower(), flush=True)
            except BaseException as exc:
                transport_barrier_ok = False
                print("TRANSPORT_STOP_ERROR = " + repr(exc), flush=True)
            try:
                self.transport.close()
                print("TRANSPORT_CLEANUP_OK = true", flush=True)
            except BaseException as exc:
                transport_barrier_ok = False
                print("TRANSPORT_CLEANUP_ERROR = " + repr(exc), flush=True)

        controller_stop_ok = not active_motion
        if active_motion:
            if not self.hardware:
                controller_stop_ok = False
                print("CONTROLLER_STOP_SKIPPED = hardware_disabled", flush=True)
            else:
                try:
                    result = self.services.stop()
                    controller_stop_ok = True
                    message = getattr(result, "message", "accepted")
                    print("CONTROLLER_STOP_SERVICE_RESULT = " + str(message), flush=True)
                except BaseException as exc:
                    controller_stop_ok = False
                    print("CONTROLLER_STOP_SERVICE_ERROR = " + repr(exc), flush=True)

        feedback_ok = True if not active_motion else self._wait_for_stop_feedback()
        self._stop_confirmed = bool(transport_barrier_ok and controller_stop_ok and feedback_ok)
        if self._stop_confirmed:
            print("STOP_CONFIRMED", flush=True)
        else:
            print("STOP_UNCONFIRMED", flush=True)
        return self._stop_confirmed

    def _publish_reference(self, pose, state, action=None):
        if self.transport is None:
            raise PickFault("TRANSPORT_NOT_STARTED")
        self.safety.command_position(pose.position, "commanded_robot_end")
        self.safety.joints(state.joint_rad)
        # Keep the feedback timestamp from the UDP packet.  Measuring here
        # identifies time spent between _state() and transport.publish().
        self._last_publish_udp_age_ms = self._record_udp_age(
            state.receipt_mono, "UDP_AGE_BEFORE_PUBLISH_MS", "publish", False
        )
        if not self._first_publish_recorded:
            print("FIRST_UDP_AGE_BEFORE_PUBLISH_MS = %.3f" %
                  self._last_publish_udp_age_ms, flush=True)
            self._first_publish_recorded = True
        self._print_debug_command(action, state, self._rotation(state), pose)
        self.transport.publish(pose, state.receipt_mono)

    def _print_debug_command(self, action, state, current_rotation, target_pose):
        """Print the complete command/safety snapshot immediately before publish."""
        action = np.zeros(6, dtype=np.float64) if action is None else np.asarray(action, dtype=np.float64)
        delta_xyz = np.asarray(target_pose.position, dtype=np.float64) - np.asarray(state.end_xyz_m, dtype=np.float64)
        delta_rotation = so3_log(np.asarray(target_pose.rotation) @ np.asarray(current_rotation).T)
        serialized = protocol_pose(target_pose.position, target_pose.rotation)
        quaternion = q_from_R(target_pose.rotation)
        print("DEBUG_COMMAND:", flush=True)
        print("action_xyz=" + np.array2string(action[:3], precision=8), flush=True)
        print("action_rot=" + np.array2string(action[3:], precision=8), flush=True)
        print("action_norm=%.9f" % float(np.linalg.norm(action)), flush=True)
        print("current_position=" + np.array2string(state.end_xyz_m, precision=8), flush=True)
        print("target_position=" + np.array2string(target_pose.position, precision=8), flush=True)
        print("delta_xyz=" + np.array2string(delta_xyz, precision=8), flush=True)
        print("delta_angle=%.9f" % float(np.linalg.norm(delta_rotation)), flush=True)
        print("quaternion_xyzw=" + np.array2string(np.asarray(quaternion), precision=8), flush=True)
        print("euler_mrad=" + repr(serialized[3:6]), flush=True)
        print("joint_state=" + np.array2string(np.asarray(state.joint_rad), precision=8), flush=True)
        print("joint_error=" + repr(state.joint_err_code), flush=True)

    def _print_initial_robot_end_diagnostics(self, state, rotation):
        """Print the complete pre-safety pose provenance once per episode."""
        xyz = np.asarray(state.end_xyz_m, dtype=np.float64)
        tcp = actual_tcp(xyz, rotation)
        minimum = np.asarray(self.config.workspace_min, dtype=np.float64)
        maximum = np.asarray(self.config.workspace_max, dtype=np.float64)
        below = xyz < minimum
        above = xyz > maximum
        age_ms = max(0.0, (time.monotonic() - float(state.receipt_mono)) * 1000.0)
        print("DEBUG_INITIAL_ROBOT_END:", flush=True)
        print("xyz=" + np.array2string(xyz, precision=9), flush=True)
        print("frame=base_link->robot_end", flush=True)
        print("unit=m", flush=True)
        print("source=RealMan UDP waypoint.position / 1e6", flush=True)
        print("timestamp_receipt_mono=%.9f" % float(state.receipt_mono), flush=True)
        print("timestamp_age_ms=%.3f" % age_ms, flush=True)
        print("current_flange_robot_end=" + np.array2string(xyz, precision=9), flush=True)
        print("current_tcp=" + np.array2string(tcp, precision=9), flush=True)
        print("target_apple_base=" + repr(None if self.apple_base is None else np.asarray(self.apple_base).tolist()), flush=True)
        target_end = self.goal_end
        if target_end is None and self.apple_base is not None and self.held_R is not None:
            target_end = flange_target_for_apple(self.apple_base, self.held_R)
        print("target_robot_end=" + repr(None if target_end is None else np.asarray(target_end).tolist()), flush=True)
        print("target_frame=camera_link->base_link (already transformed)", flush=True)
        print("workspace_min=" + np.array2string(minimum, precision=9), flush=True)
        print("workspace_max=" + np.array2string(maximum, precision=9), flush=True)
        print("workspace_enforce_config=" + str(bool(self.config.enforce_workspace)).lower(), flush=True)
        print("workspace_below_min=" + repr(below.tolist()), flush=True)
        print("workspace_above_max=" + repr(above.tolist()), flush=True)

    def _reach(self):
        self._wait_for_target()
        if self.held_R is None:
            raise PickFault("HELD_ORIENTATION_UNAVAILABLE")
        # Acquisition may have printed diagnostics after its last UDP poll.
        # Take one current, safety-checked pose before starting the worker.
        state = self._state()
        rotation = self._rotation(state)
        self._print_initial_robot_end_diagnostics(state, rotation)
        # This is observed feedback, not a pose we are about to command.
        # Keep current-state validation (finite/floor/joints) separate from
        # the Cartesian box applied to each generated target pose.
        self.safety.current_position(state.end_xyz_m, "initial_robot_end", emit=True)
        initial_pose = Pose(state.end_xyz_m.copy(), self.held_R.copy())
        start_complete_mono = self._start_transport(initial_pose)
        state = self._udp_after_transport_start(start_complete_mono)
        # The first command and its interpolation origin both use the new
        # post-start feedback; the pre-start packet is never republished.
        self.reference = CommandReference(state.end_xyz_m, self.held_R)
        self.transport.set_initial_pose_before_first_publish(self.reference.pose)
        self.set_phase(Phase.TRACKING_REACH)
        print("REACH_HIGH_FOLLOW_ENTRY_OK = true", flush=True)
        latch = self.target_sub.latch
        motion = MotionWindow()
        terminal = False
        started = time.monotonic()
        max_time = 180.
        first_cycle = True
        next_policy_deadline = time.monotonic()
        while time.monotonic()-started < max_time:
            if first_cycle:
                first_cycle = False
            else:
                while time.monotonic() < next_policy_deadline:
                    self._spin(min(.005, max(0.0, next_policy_deadline - time.monotonic())))
                state = self._state()
            if state is None: continue
            R = self._rotation(state)
            self.safety.feedback(state)
            current_apple = self._update_latch(state)
            if current_apple is not None and not latch.committed() and not latch.frozen():
                self.apple_base = current_apple.copy()
            if self.apple_base is None:
                raise PickFault("TARGET_INVALID")
            tcp = actual_tcp(state.end_xyz_m, R)
            dist = float(np.linalg.norm(self.apple_base-tcp))
            if dist <= TERMINAL_FREEZE_DISTANCE_M and not terminal:
                if self.apple_base is None or not latch.freeze_base(self.apple_base):
                    raise PickFault("TARGET_FREEZE_FAILED")
                latch.arm()
                terminal = True
                self.apple_base = latch.frozen_target_base_xyz.copy()
                print("TARGET_FREEZE_COMMITTED = true", flush=True)
                self.set_phase(Phase.TERMINAL_REACH)
            if terminal and not latch.committed() and latch.reliable_base_xyz is not None:
                if latch.reliable_age() > 1.0:
                    raise PickFault("TERMINAL_TARGET_STALE_BEFORE_COMMIT")
                # A stale/absent eye-in-hand frame in the 80--100 mm band is
                # itself the handoff signal; do not wait for a two-second loss
                # confirmation and then reject the one-second-old sample.
                if self.target_sub.latch.age() > .30:
                    latch.commit(latch.reliable_base_xyz, "EARLY_OCCLUSION")
                    self.apple_base = latch.committed_base_xyz.copy()
                    print("TERMINAL_COMMITTED = true; reason=EARLY_OCCLUSION", flush=True)
                elif dist <= TERMINAL_COMMIT_DISTANCE_M:
                    latch.commit(latch.reliable_base_xyz, "PREGRASP_DISTANCE")
                    self.apple_base = latch.committed_base_xyz.copy()
                    print("TERMINAL_COMMITTED = true; reason=PREGRASP_DISTANCE", flush=True)
            if latch.committed():
                self.apple_base = latch.committed_base_xyz.copy()
            if dist <= .005:
                self.set_phase(Phase.REACH_DECELERATING)
                # Policy targets are no longer extended. Send a bounded taper.
                for _ in range(12):
                    state = self._state(); R = self._rotation(state)
                    self.reference.decelerate()
                    self._publish_reference(self.reference.pose, state, np.zeros(6, dtype=np.float64))
                    next_policy_deadline = time.monotonic() + POLICY_DT
                    while time.monotonic() < next_policy_deadline:
                        self._spin(min(.005, max(0.0, next_policy_deadline - time.monotonic())))
                self._settle_and_close_gripper()
                return
            goal_end = flange_target_for_apple(self.apple_base, self.held_R)
            self.goal_end = goal_end
            raw = observation(state.joint_rad, state.end_xyz_m, goal_end, R, self.held_R)
            action = self._infer_policy(raw)
            # The frozen actor output is applied through CommandReference's
            # smoothing and velocity limits below.  A one-frame dot-product
            # test against the instantaneous goal is deliberately diagnostic
            # only: a valid policy can briefly correct sideways or reverse a
            # component while the TCP orientation and the latest UDP pose
            # change.  Turning that sample into an exception used to stop the
            # producer immediately after a command had already been published,
            # leaving the transport to hold its last target and making a short
            # vibration appear at startup.  Actual protection remains in the
            # finite-action checks, CommandReference caps, workspace checks,
            # rolling MotionWindow, UDP heartbeat, controller feedback, and
            # bounded cleanup below.
            action = np.asarray(action, dtype=np.float64)
            if action.shape != (6,) or not np.all(np.isfinite(action)):
                raise PickFault("ACTOR_ACTION_INVALID")
            if np.any(np.abs(action) > ACTION_BOUNDS + 1e-9):
                raise PickFault("ACTOR_ACTION_RANGE")
            next_pose = self.reference.step(action, goal_end, dist)
            self.safety.command_position(next_pose.position, "commanded_robot_end")
            self._publish_reference(next_pose, state, action)
            next_policy_deadline = time.monotonic() + POLICY_DT
            motion.check(time.monotonic(), tcp, self.apple_base, dist)
        raise PickFault("REACH_TIMEOUT")

    def _settle_and_close_gripper(self):
        """Stop the zero-velocity follower, then confirm fresh UDP TCP stability."""
        if self.transport is None:
            raise PickFault("TRANSPORT_NOT_STARTED")
        # The 200 Hz worker expects policy publications every 150 ms.  Once
        # tapering is complete there are no new motion targets, so fence its
        # sends before waiting for the longer physical settle window.  Retain
        # the transport object until settling succeeds: on a later fault,
        # stop_on_fault() must still request the controller stop and verify it.
        self.set_phase(Phase.TCP_SETTLED)
        self.transport.close()
        settle = SettleWindow()
        deadline = time.monotonic() + 3.0
        last_receipt = None
        stale_warned = False
        while time.monotonic() < deadline:
            self._spin(.01)
            state = self.feedback.latest
            if state is None or not state.fresh:
                if not stale_warned:
                    print("TCP_SETTLE_UDP_STALE_WARNING = true", flush=True)
                    stale_warned = True
                settle = SettleWindow()
                continue
            receipt = float(state.receipt_mono)
            if last_receipt is not None and receipt <= last_receipt:
                continue
            last_receipt = receipt
            self._record_udp_age(receipt)
            # A fresh controller error is never treated as a transient UDP
            # gap.  The normal fault path still stops and locks arm motion.
            self.safety.feedback(state)
            rotation = self._rotation(state)
            tcp = actual_tcp(state.end_xyz_m, rotation)
            if settle.add(state, rotation, self.apple_base):
                if np.linalg.norm(self.apple_base - tcp) > .005:
                    raise PickFault("POST_STOP_TCP_OUTSIDE_5MM")
                self.transport = None
                self.set_phase(Phase.HIGH_FOLLOW_STOPPED)
                self._close_gripper_once()
                return
        if stale_warned:
            raise PickFault("TCP_SETTLE_FEEDBACK_TIMEOUT")
        raise PickFault("TCP_SETTLE_FAILED")

    def _home_monitor(self):
        state = self._state()
        self.safety.feedback(state)
        qdeg = np.asarray(state.joint_rad)*180/np.pi
        err = np.max(np.abs(qdeg-self.config.home_joint_deg))
        if err <= self.config.home_tolerance_deg:
            self.home_done = True

    def finish(self):
        if self._hardware_motion_locked:
            raise PickFault("HARDWARE_MOTION_LOCKED_AFTER_FAULT")
        self._close_gripper_once()
        self.set_phase(Phase.HOME)
        # Validate the configured Home against the same URDF endpoint floor
        # before submitting the one-shot MoveJ request.
        self.safety.joints(np.deg2rad(self.config.home_joint_deg))
        print("HOME_REQUESTED", flush=True)
        self._service_motion_active = True
        self.services.home(self._home_monitor)
        self._service_motion_active = False
        deadline = time.monotonic()+self.config.service_timeout
        while time.monotonic() < deadline and not self.home_done:
            self._home_monitor()
        if not self.home_done:
            raise PickFault("HOME_NOT_REACHED")
        print("HOME_REACHED = true", flush=True)
        if self.release_mode == "disabled":
            raise PickFault("RELEASE_DISABLED")
        if self.release_mode != "configured":
            raise PickFault("UNSUPPORTED_RELEASE_MODE")
        self.set_phase(Phase.RELEASE)
        print("RELEASE_REQUESTED", flush=True)
        self.services.release(); self.release_done=True
        print("RELEASE_SENT = true", flush=True)
        self.set_phase(Phase.DONE)
        print("FULL_PICK_COMPLETED", flush=True)

    def _close_gripper_once(self):
        """Close exactly once after High-Follow has reached and stopped."""
        if self.close_done:
            return
        self.set_phase(Phase.CLOSE)
        self.services.close_gripper()
        self.close_done = True

    def run(self):
        try:
            self.set_phase(Phase.INIT)
            self.precheck()
            print("PRECHECK_PASSED", flush=True)
            self.wait_for_start()
            self._reach()
            self.finish()
            return True
        except BaseException as exc:
            try:
                self.stop_on_fault(str(exc))
            except BaseException:
                logger.exception("FULL_PICK_STOP_ON_FAULT_EXCEPTION")
            logger.exception("FULL_PICK_RUNTIME_EXCEPTION")
            self.reason = str(exc)
            self.phase = Phase.FAILED
            self._print_timing_summary()
            print(f"FULL_PICK_FAILED = {self.reason}", flush=True)
            return False
        finally:
            if self.transport is not None:
                try: self.transport.close()
                except BaseException as exc: print(f"TRANSPORT_CLEANUP_ERROR = {exc}", flush=True)
