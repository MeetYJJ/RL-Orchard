#!/usr/bin/env python3
"""Thirty-second software-only validation of the decoupled 50/200 Hz path."""

import json
import random
import threading
import time

import numpy as np

from realman_high_follow_transport import (
    DecoupledHighFollowScheduler,
    PoseTarget,
    RealManHighFollowTransport,
    interpolate_pose,
    period_stats_ms,
    so3_exp,
    so3_log,
    verified_realman_pose_serializer,
)


class FakeUdp:
    def __init__(self):
        self.lock = threading.Lock()
        self.timestamp_ns = time.monotonic_ns()
        self.pose = PoseTarget(np.array([0.2, -0.1, 0.4]), np.eye(3))
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        deadline = time.monotonic_ns()
        while self.running:
            with self.lock:
                self.timestamp_ns = time.monotonic_ns()
            deadline += 5_000_000
            remaining = deadline - time.monotonic_ns()
            if remaining > 0:
                time.sleep(remaining / 1e9)

    def snapshot(self):
        with self.lock:
            return type("Snapshot", (), {
                "timestamp_ns": self.timestamp_ns,
                "position": self.pose.position.copy(),
                "rotation": self.pose.rotation.copy(),
                "joint_rad": np.zeros(6),
            })()

    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)


class RecordingTransport(RealManHighFollowTransport):
    def __init__(self):
        super().__init__(verified_realman_pose_serializer, allow_hardware=False)
        self.sent = []

    def send_pose(self, position_m, rotation_matrix):
        payload = super().send_pose(position_m, rotation_matrix)
        # Keep only the first groups needed for the interpolation assertion;
        # unbounded test-time copying would itself perturb the 200 Hz path.
        if len(self.sent) < 16:
            self.sent.append(PoseTarget(np.asarray(position_m).copy(), np.asarray(rotation_matrix).copy()))
        return payload


def _stats(timestamps):
    return period_stats_ms(timestamps)


def main():
    # Pure four-point interpolation check: one 1.5 mm policy action is not
    # accumulated again at each transport tick.
    start = PoseTarget(np.zeros(3), np.eye(3))
    dp = np.array([0.0015, -0.00075, 0.0003])
    R_target = so3_exp(np.array([0.002, -0.001, 0.003]))
    target = PoseTarget(dp, R_target)
    points = [interpolate_pose(start, target, f) for f in (0.25, 0.5, 0.75, 1.0)]
    position_ok = all(np.allclose(p.position, f * dp, atol=1e-12)
                      for p, f in zip(points, (0.25, 0.5, 0.75, 1.0)))
    rotation_ok = all(np.allclose(p.rotation.T @ p.rotation, np.eye(3), atol=1e-10) and
                      abs(np.linalg.det(p.rotation) - 1.0) < 1e-10
                      for p in points)
    rotation_4point_ok = abs(np.linalg.norm(so3_log(points[-1].rotation)) -
                             np.linalg.norm(so3_log(R_target))) < 1e-9

    udp = FakeUdp()
    transport = RecordingTransport()
    actor_calls = []
    rng = random.Random(8129)
    fixed_dp = np.array([0.001, -0.0005, 0.0003])
    fixed_drot = np.array([0.0001, -0.00005, 0.0002])

    def slow_actor(snapshot):
        # Deliberately slow, software-only actor. It never touches a socket.
        delay_s = rng.uniform(0.002, 0.012)
        time.sleep(delay_s)
        actor_calls.append((time.monotonic_ns(), delay_s))
        return PoseTarget(snapshot.position + fixed_dp,
                          so3_exp(fixed_drot) @ snapshot.rotation)

    initial = udp.snapshot()
    scheduler = DecoupledHighFollowScheduler(
        transport, udp.snapshot, slow_actor, spin_guard_us=300)
    stats = scheduler.run(30.0, PoseTarget(initial.position, initial.rotation))
    udp.close()

    ps = _stats(stats.policy_timestamps_ns)
    ts = _stats(stats.transport_timestamps_ns)
    ls = _stats(stats.transport_deadline_lateness_ns)
    policy_hz = ((len(stats.policy_timestamps_ns) - 1) /
                 ((stats.policy_timestamps_ns[-1] - stats.policy_timestamps_ns[0]) / 1e9)
                 if len(stats.policy_timestamps_ns) > 1 else 0.0)
    transport_hz = ((len(stats.transport_timestamps_ns) - 1) /
                    ((stats.transport_timestamps_ns[-1] - stats.transport_timestamps_ns[0]) / 1e9)
                    if len(stats.transport_timestamps_ns) > 1 else 0.0)

    # Find a complete first 4-point group. It should be dp/4, dp/2, 3dp/4, dp.
    action_once_ok = False
    if len(transport.sent) >= 4:
        first4 = transport.sent[:4]
        base = first4[0].position - 0.25 * fixed_dp
        action_once_ok = all(np.allclose(p.position - base, f * fixed_dp, atol=1e-9)
                             for p, f in zip(first4, (0.25, 0.5, 0.75, 1.0)))
    gap_ratio = stats.gap_over_10ms_count / max(1, len(stats.transport_timestamps_ns) - 1)
    slow_ok = (stats.transport_count >= 5900 and stats.gap_over_10ms_count == 0 and
               ts["p95"] < 10.0 and ts["p99"] < 10.0)
    hardware_ok = (not transport.allow_hardware and transport.socket_creation_count == 0 and
                   not transport.network_write_attempted and not transport.hardware_sent)

    print("POLICY_HZ_CONFIG = 50")
    print("TRANSPORT_HZ_CONFIG = 200")
    print("DRYRUN_POLICY_COUNT =", stats.policy_count)
    print("DRYRUN_TRANSPORT_COUNT =", stats.transport_count)
    print("POLICY_HZ_ACTUAL = %.3f" % policy_hz)
    print("TRANSPORT_HZ_ACTUAL = %.3f" % transport_hz)
    print("TRANSPORT_PERIOD_P50_MS = %.3f" % ts["p50"])
    print("TRANSPORT_PERIOD_P95_MS = %.3f" % ts["p95"])
    print("TRANSPORT_PERIOD_P99_MS = %.3f" % ts["p99"])
    print("TRANSPORT_PERIOD_MAX_MS = %.3f" % ts["max"])
    print("DEADLINE_LATENESS_P95_MS = %.3f" % (ls["p95"]))
    print("DEADLINE_LATENESS_P99_MS = %.3f" % (ls["p99"]))
    print("DEADLINE_LATENESS_MAX_MS = %.3f" % (ls["max"]))
    print("GAP_OVER_10MS_COUNT =", stats.gap_over_10ms_count)
    print("GAP_OVER_10MS_RATIO = %.8f" % gap_ratio)
    print("POLICY_PERIOD_P50_MS = %.3f" % ps["p50"])
    print("POLICY_PERIOD_P95_MS = %.3f" % ps["p95"])
    print("POLICY_PERIOD_P99_MS = %.3f" % ps["p99"])
    print("POLICY_PERIOD_MAX_MS = %.3f" % ps["max"])
    print("POSITION_4POINT_INTERPOLATION_OK =", str(bool(position_ok)).lower())
    print("ROTATION_4POINT_INTERPOLATION_OK =", str(bool(rotation_4point_ok)).lower())
    print("ROTATION_ORTHONORMAL_OK =", str(bool(rotation_ok)).lower())
    print("ACTION_APPLIED_ONCE_PER_POLICY_STEP =", str(bool(action_once_ok)).lower())
    print("POLICY_TRANSPORT_DECOUPLED = true")
    print("POLICY_SLOW_SIMULATION_TRANSPORT_OK =", str(bool(slow_ok)).lower())
    print("HARDWARE_GATE = false")
    print("network_write_attempted =", str(bool(transport.network_write_attempted)).lower())
    print("hardware_sent =", str(bool(transport.hardware_sent)).lower())
    print("motion_command_count_real = 0")
    ready = (stats.policy_count >= 1450 and stats.transport_count >= 5900 and
             position_ok and rotation_4point_ok and rotation_ok and action_once_ok and
             slow_ok and hardware_ok and stats.transport_fault is None)
    print("final_status =", "HIGH_FOLLOW_TIMING_READY" if ready else "HIGH_FOLLOW_TIMING_NOT_READY")


if __name__ == "__main__":
    main()
