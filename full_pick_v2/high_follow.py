"""Independent high-follow transport: one persistent TCP socket, no recv/retry."""
from __future__ import annotations
from dataclasses import dataclass
import json
import multiprocessing as mp
import socket
import time
import numpy as np
from .transforms import so3_exp, so3_log, protocol_pose
from .safety import PickFault

POLICY_HZ = 20
TRANSPORT_HZ = 200
SPEED_SCALE = 0.2
HIGH_FOLLOW_SPEED_SCALE = SPEED_SCALE
SMOOTHING_ALPHA = 0.35
HIGH_FOLLOW_TARGET_SMOOTHING_ALPHA = SMOOTHING_ALPHA
HEARTBEAT_TIMEOUT_S = 0.150
# These are the existing bounded deceleration limits.  Applying the same
# limits to normal reference integration prevents a one-step actor change
# from reversing the commanded velocity, without changing actor outputs or
# the configured speed scale.
LINEAR_ACCEL_LIMIT_M_S2 = 0.030
ANGULAR_ACCEL_LIMIT_RAD_S2 = 0.100
LINEAR_ACCEL_MPS2 = LINEAR_ACCEL_LIMIT_M_S2
ANGULAR_ACCEL_RPS2 = ANGULAR_ACCEL_LIMIT_RAD_S2
POLICY_DT = 1.0 / POLICY_HZ
TRANSPORT_DT = 1.0 / TRANSPORT_HZ
# RealMan movep_canfd high-follow requires a pass-through period below 10 ms.
# This is a send-period guard, separate from the scheduler deadline guard.
MAX_TRANSPORT_INTERVAL_S = 0.010
# A single scheduler hiccup is recorded but is not enough to abort an
# otherwise healthy high-follow stream.  The existing 10 ms interval limit is
# unchanged; this is only the consecutive-failure count.
TRANSPORT_OVERRUN_CONSECUTIVE_LIMIT = 3
# The worker's absolute monotonic scheduling deadline is a separate guard.
# Linux/Python can occasionally be descheduled for more than one 5 ms tick
# while the actual send stream remains healthy.  Keep the guard bounded, but
# require repeated lateness before declaring a transport fault.
TRANSPORT_DEADLINE_LATE_LIMIT_S = 0.030
TRANSPORT_DEADLINE_CONSECUTIVE_LIMIT = 5
MAX_COMMAND_TRANSLATION_STEP_M = 0.0005
MAX_COMMAND_ROTATION_STEP_RAD = float(np.deg2rad(0.5))
# Keep the old speed ceilings as an upper bound, while the strict per-policy
# step gate below makes the effective translation ceiling 10 mm/s at 20 Hz.
LINEAR_SPEED_MPS = min(0.030, MAX_COMMAND_TRANSLATION_STEP_M / POLICY_DT)
ANGULAR_SPEED_RPS = min(0.150, MAX_COMMAND_ROTATION_STEP_RAD / POLICY_DT)
INTERPOLATION_STEPS = max(1, int(round(float(TRANSPORT_HZ) / float(POLICY_HZ))))
PROTOCOL_POSITION_UNIT_M = 1e-6
PROTOCOL_ROTATION_UNIT_RAD = 1e-3


def _transport_overrun_state(period_ms, consecutive_count):
    """Return (updated consecutive count, fault flag) for one send interval."""
    if float(period_ms) > MAX_TRANSPORT_INTERVAL_S * 1000.0:
        count = int(consecutive_count) + 1
        return count, count >= TRANSPORT_OVERRUN_CONSECUTIVE_LIMIT
    return 0, False

@dataclass
class Pose:
    position: np.ndarray
    rotation: np.ndarray

def interpolate(start, target, fraction):
    return Pose(start.position + fraction*(target.position-start.position),
                so3_exp(fraction*so3_log(target.rotation @ start.rotation.T)) @ start.rotation)

# Public name used by the verified transport tests.
def interpolate_pose(start, target, fraction):
    return interpolate(start, target, fraction)

def _continuous_protocol_pose(pose, previous_euler=None):
    """Serialize a pose without introducing an Euler wrap discontinuity.

    RealMan's JSON pose schema used by the local bridge is position in 0.001 mm
    (micrometres) followed by roll/pitch/yaw in 0.001 rad.  The controller
    accepts equivalent Euler branches, so choose the branch closest to the
    previously sent one when a previous sample is available.  The default
    path remains byte-for-byte compatible with ``protocol_pose``.
    """
    values = protocol_pose(pose.position, pose.rotation)
    if previous_euler is None:
        return values
    previous = np.asarray(previous_euler, dtype=np.float64)
    if previous.shape != (3,) or not np.all(np.isfinite(previous)):
        raise ValueError("invalid previous Euler reference")
    current = np.asarray(values[3:6], dtype=np.float64) / 1000.0
    current += 2.0 * np.pi * np.round((previous - current) / (2.0 * np.pi))
    return values[:3] + [int(round(float(x) * 1000.0)) for x in current]


def build_movep_payload(pose, previous_euler=None):
    return {"command": "movep_canfd", "pose": _continuous_protocol_pose(pose, previous_euler), "follow": True, "trajectory_mode": 0, "radio": 0}


def _unwrap_angle_delta(delta):
    return (np.asarray(delta, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def validate_serialized_pose(values, previous=None, previous_velocity=None,
                             timestamp_now=None, timestamp_last=None):
    """Validate the final integer pose that will reach RealMan.

    This validates only final-wire pose, step, and velocity constraints.
    Acceleration is deliberately checked on the continuous float trajectory
    before this serialization stage; quantization can create a false discrete
    acceleration spike.
    """
    values = np.asarray(values)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise PickFault("SERIALIZED_POSE_INVALID")
    values = values.astype(np.int64, copy=False)
    velocity = None
    if previous is not None:
        previous = np.asarray(previous, dtype=np.int64)
        if previous.shape != (6,):
            raise PickFault("SERIALIZED_POSE_HISTORY_INVALID")
        delta = np.concatenate((
            (values[:3] - previous[:3]).astype(np.float64) * PROTOCOL_POSITION_UNIT_M,
            _unwrap_angle_delta((values[3:] - previous[3:]).astype(np.float64) * PROTOCOL_ROTATION_UNIT_RAD),
        ))
        if (np.linalg.norm(delta[:3]) > MAX_COMMAND_TRANSLATION_STEP_M + 1e-12 or
                np.linalg.norm(delta[3:]) > MAX_COMMAND_ROTATION_STEP_RAD + 1e-12):
            raise PickFault("SERIALIZED_COMMAND_STEP_TOO_LARGE")
        velocity = delta / TRANSPORT_DT
        if (np.linalg.norm(velocity[:3]) > LINEAR_SPEED_MPS + 1e-6 / TRANSPORT_DT or
                np.linalg.norm(velocity[3:]) > ANGULAR_SPEED_RPS + 1e-3 / TRANSPORT_DT):
            raise PickFault("SERIALIZED_VELOCITY_LIMIT")
    return velocity, None


def validate_float_motion(previous_velocity, current_velocity,
                          timestamp_now=None, timestamp_last=None):
    """Check velocity and acceleration before protocol quantization."""
    previous_velocity = np.asarray(previous_velocity, dtype=np.float64)
    current_velocity = np.asarray(current_velocity, dtype=np.float64)
    if previous_velocity.shape != (6,) or current_velocity.shape != (6,):
        raise PickFault("FLOAT_VELOCITY_HISTORY_INVALID")
    if not np.all(np.isfinite(previous_velocity)) or not np.all(np.isfinite(current_velocity)):
        raise PickFault("FLOAT_VELOCITY_INVALID")
    if (np.linalg.norm(current_velocity[:3]) > LINEAR_SPEED_MPS + 1e-12 or
            np.linalg.norm(current_velocity[3:]) > ANGULAR_SPEED_RPS + 1e-12):
        raise PickFault("FLOAT_VELOCITY_LIMIT")
    acceleration = (current_velocity - previous_velocity) / TRANSPORT_DT
    linear_limit = LINEAR_ACCEL_MPS2
    angular_limit = ANGULAR_ACCEL_RPS2
    if (np.linalg.norm(acceleration[:3]) > linear_limit + 1e-12 or
            np.linalg.norm(acceleration[3:]) > angular_limit + 1e-12):
        now = time.monotonic() if timestamp_now is None else float(timestamp_now)
        print("ACCELERATION_CHECK_SOURCE=FLOAT_REFERENCE", flush=True)
        print("FLOAT_ACCEL_DEBUG:", flush=True)
        print("timestamp_now=%.9f" % now, flush=True)
        if timestamp_last is None:
            print("timestamp_last=none", flush=True)
            print("dt_actual_ms=none", flush=True)
        else:
            print("timestamp_last=%.9f" % float(timestamp_last), flush=True)
            print("dt_actual_ms=%.6f" % ((now - float(timestamp_last)) * 1000.0), flush=True)
        print("dt=%.9f" % TRANSPORT_DT, flush=True)
        print("last_velocity=" + np.array2string(previous_velocity, precision=9), flush=True)
        print("current_velocity=" + np.array2string(current_velocity, precision=9), flush=True)
        print("raw_velocity_delta=" + np.array2string(current_velocity - previous_velocity, precision=9), flush=True)
        print("velocity_unit=linear_m_per_s; angular_rad_per_s", flush=True)
        print("acceleration=" + np.array2string(acceleration, precision=9), flush=True)
        print("acceleration_unit=linear_m_per_s2; angular_rad_per_s2", flush=True)
        print("limit=" + repr({
            "linear_m_per_s2": float(linear_limit),
            "angular_rad_per_s2": float(angular_limit),
        }), flush=True)
        raise PickFault("FLOAT_ACCELERATION_LIMIT")
    return acceleration


def _limit_velocity_change(current, desired, acceleration, dt):
    current = np.asarray(current, dtype=np.float64)
    desired = np.asarray(desired, dtype=np.float64)
    change = desired - current
    norm = float(np.linalg.norm(change))
    max_change = float(acceleration) * float(dt)
    if norm > max_change and norm > 1e-12:
        change *= max_change / norm
    return current + change


def _stopping_speed(distance, acceleration, dt):
    distance = max(0.0, float(distance))
    acceleration = float(acceleration)
    dt = float(dt)
    return max(0.0, np.sqrt((acceleration * dt) ** 2 + 2.0 * acceleration * distance) - acceleration * dt)

class CommandReference:
    """Integrate each scaled Actor action once into a continuous reference."""
    def __init__(self, position, rotation):
        self.pose = Pose(np.asarray(position).copy(), np.asarray(rotation).copy())
        self.filtered = np.zeros(6)
        self.velocity = np.zeros(6)

    def step(self, actor_action, goal, distance):
        factor = float(np.clip(distance/.050, .15, 1.0))
        action = np.asarray(actor_action, dtype=np.float64)*SPEED_SCALE*factor
        self.filtered = SMOOTHING_ALPHA*action + (1-SMOOTHING_ALPHA)*self.filtered
        desired_delta = self.filtered.copy()
        # Software velocity caps, applied before integration.
        for part, cap in ((slice(0,3), LINEAR_SPEED_MPS/POLICY_HZ),
                          (slice(3,6), ANGULAR_SPEED_RPS/POLICY_HZ)):
            norm = np.linalg.norm(desired_delta[part])
            if norm > cap:
                desired_delta[part] *= cap/norm
        # Bound velocity changes at the policy boundary.  The transport keeps
        # the same velocity state while producing its interpolated points.
        desired_velocity = desired_delta / POLICY_DT
        remaining = np.asarray(goal, dtype=np.float64) - self.pose.position
        desired_velocity[:3] = _limit_norm(
            desired_velocity[:3],
            _stopping_speed(np.linalg.norm(remaining), LINEAR_ACCEL_MPS2, POLICY_DT),
        )
        for part, acceleration in ((slice(0,3), LINEAR_ACCEL_MPS2),
                                   (slice(3,6), ANGULAR_ACCEL_RPS2)):
            self.velocity[part] = _limit_velocity_change(
                self.velocity[part], desired_velocity[part], acceleration, POLICY_DT
            )
        delta = self.velocity * POLICY_DT
        delta[:3] = _limit_norm(delta[:3], MAX_COMMAND_TRANSLATION_STEP_M)
        delta[3:] = _limit_norm(delta[3:], MAX_COMMAND_ROTATION_STEP_RAD)
        self.pose = Pose(self.pose.position+delta[:3], so3_exp(delta[3:]) @ self.pose.rotation)
        return self.pose

    def decelerate(self):
        for part, acceleration in ((slice(0,3), LINEAR_ACCEL_MPS2),
                                   (slice(3,6), ANGULAR_ACCEL_RPS2)):
            self.velocity[part] = _limit_velocity_change(
                self.velocity[part], np.zeros(3), acceleration, POLICY_DT
            )
        delta = self.velocity/POLICY_HZ
        delta[:3] = _limit_norm(delta[:3], MAX_COMMAND_TRANSLATION_STEP_M)
        delta[3:] = _limit_norm(delta[3:], MAX_COMMAND_ROTATION_STEP_RAD)
        self.pose = Pose(self.pose.position+delta[:3], so3_exp(delta[3:]) @ self.pose.rotation)
        return self.pose

    @property
    def stopped(self):
        return np.linalg.norm(self.velocity) < 1e-10


def _limit_norm(value, limit):
    value = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm > float(limit) and norm > 1e-12:
        return value * (float(limit) / norm)
    return value


def _transport_step(reference, candidate, velocity):
    """Advance one transport point with continuous velocity and bounded accel.

    ``candidate`` is still the nominal SO(3) interpolation target.
    The sent point is integrated from the last sent point using the worker's
    velocity state.  This prevents a new policy target from replacing the
    previous segment velocity instantaneously at a transport boundary.
    """
    delta = np.concatenate((
        np.asarray(candidate.position, dtype=np.float64) - np.asarray(reference.position, dtype=np.float64),
        so3_log(np.asarray(candidate.rotation, dtype=np.float64) @ np.asarray(reference.rotation, dtype=np.float64).T),
    ))
    desired = delta / TRANSPORT_DT
    velocity = np.asarray(velocity, dtype=np.float64).copy()
    for part, speed, acceleration in (
        (slice(0, 3), LINEAR_SPEED_MPS, LINEAR_ACCEL_MPS2),
        (slice(3, 6), ANGULAR_SPEED_RPS, ANGULAR_ACCEL_RPS2),
    ):
        desired[part] = _limit_norm(desired[part], speed)
        # Do not build speed that cannot be removed before the current
        # candidate.  This is only a local braking bound; it does not change
        # the policy action or the 5 mm reach criterion.
        distance = float(np.linalg.norm(delta[part]))
        # Discrete-time stopping bound: after one more transport interval the
        # remaining speed can still be reduced by ``acceleration * dt``.
        stopping_speed = max(
            0.0,
            np.sqrt((acceleration * TRANSPORT_DT) ** 2 + 2.0 * acceleration * distance)
            - acceleration * TRANSPORT_DT,
        )
        desired[part] = _limit_norm(desired[part], stopping_speed)
        change = desired[part] - velocity[part]
        change = _limit_norm(change, acceleration * TRANSPORT_DT)
        velocity[part] += change
    step = velocity * TRANSPORT_DT
    next_pose = Pose(
        np.asarray(reference.position, dtype=np.float64) + step[:3],
        so3_exp(step[3:]) @ np.asarray(reference.rotation, dtype=np.float64),
    )
    return next_pose, velocity

def _worker(shared, lock, stopped, started, status, faults, host, port, hardware):
    sock = None
    # The origin for a new target is always the last pose whose payload was
    # successfully sent, never an unsent interpolation candidate.
    reference = None
    sequence = 0
    # Keep one continuously retargeted endpoint.  Restarting a 10-sample
    # interpolation segment at every policy update makes the nominal candidate
    # jump back toward the last sent point and repeatedly brakes the transport.
    # The endpoint is updated in place while _transport_step() retains its
    # velocity state and applies the existing acceleration limits.
    active_target = None
    transport_velocity = np.zeros(6, dtype=np.float64)
    last_euler = None
    last_protocol_pose = None
    last_protocol_velocity = np.zeros(6, dtype=np.float64)
    last_send_mono = None
    send_interval_sum_ms = 0.0
    send_interval_count = 0
    send_interval_max_ms = 0.0
    float_acceleration_logged = False
    continuous_overrun_count = 0
    consecutive_deadline_misses = 0

    def heartbeat_fault(reason, now, row):
        """Emit one fault snapshot using the shared monotonic timestamps."""
        policy_age_ms = (now - float(row[1])) * 1000.0
        udp_age_ms = (now - float(row[2])) * 1000.0
        last_sequence = int(row[0])
        print("HEARTBEAT_FAULT = " + reason, flush=True)
        print("POLICY_AGE_MS = %.3f" % policy_age_ms, flush=True)
        print("UDP_AGE_MS = %.3f" % udp_age_ms, flush=True)
        print("LAST_SEQUENCE = %d" % last_sequence, flush=True)

    try:
        if hardware:
            sock = socket.create_connection((host, port), timeout=1.0)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            # Bound a blocked send; recv is never called.
            sock.settimeout(.05)
        started.set()
        deadline = time.monotonic()
        while not stopped.is_set():
            now = time.monotonic()
            if now < deadline:
                time.sleep(min(deadline-now, .001))
                continue
            # Hold the same lock used by request_stop() through the send.  A
            # stop request can therefore linearize before this block or after
            # this one complete payload, but never between its final check and
            # a later payload.
            with lock:
                if stopped.is_set():
                    break
                row = list(shared[:])
                if row[0] > 0:
                    if now-row[1] > HEARTBEAT_TIMEOUT_S:
                        heartbeat_fault("POLICY_HEARTBEAT_STALE", now, row)
                        raise PickFault("POLICY_HEARTBEAT_STALE")
                    if now-row[2] > HEARTBEAT_TIMEOUT_S:
                        heartbeat_fault("UDP_HEARTBEAT_STALE", now, row)
                        raise PickFault("UDP_HEARTBEAT_STALE")
                    target = Pose(np.asarray(row[3:6]), np.asarray(row[6:15]).reshape(3,3))
                    if reference is None:
                        reference = Pose(np.asarray(row[15:18]), np.asarray(row[18:27]).reshape(3,3))
                        last_euler = np.asarray(
                            build_movep_payload(reference)["pose"][3:6],
                            dtype=np.float64,
                        ) / 1000.0
                    if int(row[0]) != sequence:
                        sequence = int(row[0])
                        active_target = target
                    if active_target is not None:
                        # _transport_step() interpolates translation and SO(3)
                        # from the last sent reference under the existing
                        # speed and acceleration limits.  Pointing it at the
                        # latest endpoint avoids a per-target segment restart.
                        candidate = active_target
                    else:
                        candidate = reference
                    previous_transport_velocity = transport_velocity.copy()
                    next_reference, transport_velocity = _transport_step(
                        reference, candidate, transport_velocity
                    )
                    if not float_acceleration_logged:
                        print("ACCELERATION_CHECK_SOURCE=FLOAT_REFERENCE", flush=True)
                        float_acceleration_logged = True
                    validate_float_motion(
                        previous_transport_velocity, transport_velocity,
                        timestamp_now=time.monotonic(), timestamp_last=last_send_mono,
                    )
                    sent_delta = np.concatenate((
                        next_reference.position - reference.position,
                        so3_log(next_reference.rotation @ reference.rotation.T),
                    ))
                    if (np.linalg.norm(sent_delta[:3]) > MAX_COMMAND_TRANSLATION_STEP_M + 1e-9 or
                            np.linalg.norm(sent_delta[3:]) > MAX_COMMAND_ROTATION_STEP_RAD + 1e-9):
                        raise PickFault("COMMAND_STEP_TOO_LARGE")
                    payload = build_movep_payload(next_reference, last_euler)
                    # The first payload has no previous wire pose.  Establish
                    # its wire velocity history for diagnostics; acceleration
                    # safety has already run on the continuous float velocity.
                    first_protocol_sample = last_protocol_pose is None
                    first_protocol_velocity = None
                    if first_protocol_sample:
                        initial_protocol_pose = np.asarray(
                            build_movep_payload(reference)["pose"],
                            dtype=np.int64,
                        )
                        first_protocol_velocity, _ = validate_serialized_pose(
                            payload["pose"], initial_protocol_pose, None
                        )
                        protocol_velocity = None
                    else:
                        validation_now = time.monotonic()
                        protocol_velocity, _ = validate_serialized_pose(
                            payload["pose"], last_protocol_pose, last_protocol_velocity,
                            timestamp_now=validation_now, timestamp_last=last_send_mono,
                        )
                    if stopped.is_set():
                        break
                    # Measure the interval at the actual send boundary.  The
                    # command is rejected before transmission if scheduling or
                    # a blocked socket made the previous send more than 10 ms
                    # old; no abnormal trajectory is allowed to continue.
                    send_mono = time.monotonic()
                    if last_send_mono is not None:
                        send_interval_ms = (send_mono - last_send_mono) * 1000.0
                        send_interval_count += 1
                        send_interval_sum_ms += send_interval_ms
                        send_interval_max_ms = max(send_interval_max_ms, send_interval_ms)
                        limit_ms = MAX_TRANSPORT_INTERVAL_S * 1000.0
                        continuous_overrun_count, overrun_fault = _transport_overrun_state(
                            send_interval_ms, continuous_overrun_count
                        )
                        if send_interval_ms > limit_ms:
                            print("TRANSPORT_PERIOD_OVERRUN_DEBUG:", flush=True)
                            print("period_ms=%.3f" % send_interval_ms, flush=True)
                            print("limit_ms=%.3f" % limit_ms, flush=True)
                            print("continuous_overrun_count=%d" %
                                  continuous_overrun_count, flush=True)
                            if overrun_fault:
                                mean_ms = send_interval_sum_ms / float(send_interval_count)
                                print(
                                    "TRANSPORT_PERIOD_OVERRUN = %.3f ms" % send_interval_ms,
                                    flush=True,
                                )
                                print("SEND_INTERVAL_MEAN_MS = %.3f" % mean_ms, flush=True)
                                print("SEND_INTERVAL_MAX_MS = %.3f" % send_interval_max_ms, flush=True)
                                raise PickFault("TRANSPORT_PERIOD_OVERRUN")
                        else:
                            continuous_overrun_count = 0
                    if sock is not None:
                        sock.sendall((json.dumps(payload, separators=(",",":"))+"\r\n").encode("utf-8"))
                    last_send_mono = send_mono
                    # Only advance the interpolation origin after the send has
                    # succeeded. A failed send faults the worker and prevents
                    # any later command from using an unsent pose.
                    reference = next_reference
                    last_euler = np.asarray(payload["pose"][3:6], dtype=np.float64) / 1000.0
                    last_protocol_pose = np.asarray(payload["pose"], dtype=np.int64)
                    if first_protocol_sample and first_protocol_velocity is not None:
                        last_protocol_velocity = first_protocol_velocity
                        print("SERIALIZED_PROTOCOL_VELOCITY_INITIALIZED=true", flush=True)
                        print("SERIALIZED_PROTOCOL_VELOCITY_INIT=" +
                              np.array2string(last_protocol_velocity, precision=9), flush=True)
                    elif protocol_velocity is not None:
                        last_protocol_velocity = protocol_velocity
                    with status.get_lock():
                        status[0] = send_mono
                        status[1] += 1
                        status[2] = sequence
                        status[3] = (
                            send_interval_sum_ms / float(send_interval_count)
                            if send_interval_count else 0.0
                        )
                        status[4] = send_interval_max_ms
            deadline += TRANSPORT_DT
            deadline_lateness_s = time.monotonic() - deadline
            if deadline_lateness_s > TRANSPORT_DEADLINE_LATE_LIMIT_S:
                consecutive_deadline_misses += 1
                print("TRANSPORT_DEADLINE_DEBUG:", flush=True)
                print("deadline_lateness_ms=%.3f" %
                      (deadline_lateness_s * 1000.0), flush=True)
                print("deadline_limit_ms=%.3f" %
                      (TRANSPORT_DEADLINE_LATE_LIMIT_S * 1000.0), flush=True)
                print("continuous_deadline_miss_count=%d" %
                      consecutive_deadline_misses, flush=True)
                if consecutive_deadline_misses >= TRANSPORT_DEADLINE_CONSECUTIVE_LIMIT:
                    raise PickFault("TRANSPORT_DEADLINE_MISSED")
            else:
                consecutive_deadline_misses = 0
    except BaseException as exc:
        faults.put(f"{type(exc).__name__}:{exc}")
        started.set()
    finally:
        if sock is not None:
            sock.close()

class HighFollow:
    def __init__(self, host="192.168.1.18", port=8080, hardware=False):
        self.ctx = mp.get_context("spawn")
        self.shared = self.ctx.Array("d", 27, lock=False)
        self.lock = self.ctx.Lock()
        self.stop_event = self.ctx.Event()
        self.start_event = self.ctx.Event()
        # [last_send_mono, send_count, last_sequence,
        #  interval_mean_ms, interval_max_ms]
        self.status = self.ctx.Array("d", 5)
        self.faults = self.ctx.Queue()
        self.process = self.ctx.Process(target=_worker, args=(self.shared, self.lock, self.stop_event, self.start_event, self.status, self.faults, host, port, hardware), name="full_pick_v2_transport")
        self.sequence = 0
        self.started = False
        self.initial = None
        self._stop_requested = False
        self._stop_barrier_ok = False
        self._stop_reason = None
        self._last_published_pose = None

    def start(self, pose):
        if self._stop_requested:
            raise PickFault("TRANSPORT_STOP_REQUESTED")
        self.initial = pose
        self._last_published_pose = Pose(
            np.asarray(pose.position, dtype=np.float64).copy(),
            np.asarray(pose.rotation, dtype=np.float64).copy(),
        )
        self.process.start()
        self.started = True
        if not self.start_event.wait(3.0):
            self.close()
            raise PickFault("TRANSPORT_START_TIMEOUT")
        self.check()

    def set_initial_pose_before_first_publish(self, pose):
        """Align the first interpolation origin with post-start UDP feedback.

        The worker sends nothing until publish() sets a positive sequence.
        This only updates the first command reference in the parent process.
        """
        if not self.started or self.sequence != 0:
            raise PickFault("TRANSPORT_INITIAL_POSE_UPDATE_INVALID")
        self.check()
        self.initial = Pose(
            np.asarray(pose.position, dtype=np.float64).copy(),
            np.asarray(pose.rotation, dtype=np.float64).copy(),
        )
        self._last_published_pose = Pose(
            self.initial.position.copy(), self.initial.rotation.copy()
        )

    def check(self):
        if self._stop_requested:
            raise PickFault("TRANSPORT_STOP_REQUESTED")
        if not self.faults.empty():
            raise PickFault("TRANSPORT_FAULT:"+self.faults.get())
        if self.started and not self.process.is_alive():
            raise PickFault("TRANSPORT_PROCESS_EXITED")
        if self.sequence and time.monotonic()-self.status[0] > HEARTBEAT_TIMEOUT_S:
            raise PickFault("TRANSPORT_SEND_STALE")

    def publish(self, pose, feedback_mono):
        self.check()
        pose = Pose(np.asarray(pose.position, dtype=np.float64),
                    np.asarray(pose.rotation, dtype=np.float64))
        if pose.position.shape != (3,) or pose.rotation.shape != (3, 3):
            raise PickFault("COMMAND_POSE_INVALID")
        if not np.all(np.isfinite(pose.position)) or not np.all(np.isfinite(pose.rotation)):
            raise PickFault("COMMAND_POSE_INVALID")
        if self._last_published_pose is not None:
            step_xyz = np.linalg.norm(pose.position - self._last_published_pose.position)
            step_rot = np.linalg.norm(so3_log(pose.rotation @ self._last_published_pose.rotation.T))
            if step_xyz > MAX_COMMAND_TRANSLATION_STEP_M + 1e-9 or step_rot > MAX_COMMAND_ROTATION_STEP_RAD + 1e-9:
                raise PickFault("COMMAND_STEP_TOO_LARGE")
        with self.lock:
            # Serialize the stop barrier with both shared-memory publication
            # and the worker's socket send.  Once the barrier is acquired no
            # new motion target can enter the worker.
            if self._stop_requested or self.stop_event.is_set():
                raise PickFault("TRANSPORT_STOP_REQUESTED")
            self.sequence += 1
            self.shared[:] = [self.sequence, time.monotonic(), feedback_mono, *pose.position, *pose.rotation.reshape(-1), *self.initial.position, *self.initial.rotation.reshape(-1)]
            self._last_published_pose = Pose(pose.position.copy(), pose.rotation.copy())

    def request_stop(self, reason):
        """Atomically fence new motion targets and ask the worker to exit.

        The shared lock is also held around the worker's send.  Therefore the
        first successful acquisition here is the linearization point for a
        stop request: a send already holding the lock may finish, but no later
        ``movep_canfd`` payload can be emitted.
        """
        if self._stop_requested:
            return self._stop_barrier_ok
        acquired = False
        try:
            acquired = bool(self.lock.acquire(timeout=.250))
        except TypeError:
            # Python implementations without the keyword form still provide a
            # bounded positional timeout for multiprocessing locks.
            acquired = bool(self.lock.acquire(True, .250))
        try:
            self._stop_requested = True
            self._stop_reason = str(reason)
            self.stop_event.set()
            self._stop_barrier_ok = acquired
        finally:
            if acquired:
                self.lock.release()
        return acquired

    def close(self):
        # Normal close and fault close use the same send barrier.  This keeps
        # cleanup from racing a final worker payload.
        self.request_stop(self._stop_reason or "transport_close")
        if self.started:
            self.stop_event.set()
            self.process.join(2.0)
            forced = self.process.is_alive()
            if forced:
                self.process.terminate()
                self.process.join(2.0)
            self.started = False
            if self.process.is_alive():
                raise PickFault("TRANSPORT_CLEANUP_FAILED")
            if forced:
                raise PickFault("TRANSPORT_FORCED_STOP")
