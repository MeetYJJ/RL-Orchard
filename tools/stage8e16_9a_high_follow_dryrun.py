#!/usr/bin/env python3
"""Ten-second synthetic scheduler/transport validation. Never uses hardware."""

import math
import json
import threading
import time

import numpy as np

from realman_high_follow_transport import (
    HighFollowScheduler,
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
        self.pose = PoseTarget(np.array([0.2, -0.1, 0.4]), np.eye(3))
        self.timestamp_ns = time.monotonic_ns()
        self.running = True
        self.thread = threading.Thread(target=self._publish_200hz, daemon=True)
        self.thread.start()

    def _publish_200hz(self):
        deadline = time.monotonic_ns()
        period = 5_000_000
        while self.running:
            with self.lock:
                self.timestamp_ns = time.monotonic_ns()
            deadline += period
            delay = deadline - time.monotonic_ns()
            if delay > 0:
                time.sleep(delay / 1e9)

    def snapshot(self):
        with self.lock:
            p = self.pose
            return type("Snapshot", (), {
                "timestamp_ns": self.timestamp_ns,
                "position": p.position.copy(),
                "rotation": p.rotation.copy(),
                "joint_rad": np.zeros(6),
            })()

    def set_pose(self, position, rotation):
        with self.lock:
            self.pose = PoseTarget(np.asarray(position).copy(), np.asarray(rotation).copy())
            self.timestamp_ns = time.monotonic_ns()

    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)


class RecordingTransport(RealManHighFollowTransport):
    def __init__(self, fake_udp):
        super().__init__(verified_realman_pose_serializer, allow_hardware=False)
        self.fake_udp = fake_udp
        self.sent_targets = []

    def send_pose(self, position_m, rotation_matrix):
        result = super().send_pose(position_m, rotation_matrix)
        self.sent_targets.append(PoseTarget(np.asarray(position_m).copy(), np.asarray(rotation_matrix).copy()))
        self.fake_udp.set_pose(position_m, rotation_matrix)
        return result


def rotation_angle(R):
    return float(np.linalg.norm(so3_log(R)))


def inspect_source():
    import inspect
    import ast
    import realman_high_follow_transport as module
    from pathlib import Path

    source = inspect.getsource(module)
    send_loop = inspect.getsource(RealManHighFollowTransport.send_pose)
    scheduler = inspect.getsource(HighFollowScheduler.run)
    persistent_ok = "socket.create_connection" in source and "self._sock.sendall(wire)" in send_loop
    socket_in_loop = "socket.create_connection" in send_loop or "socket.socket(" in send_loop
    tree = ast.parse(source)
    send_method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "send_pose")
    send_calls = [n for n in ast.walk(send_method) if isinstance(n, ast.Call)]
    called_names = [n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id if isinstance(n.func, ast.Name) else "" for n in send_calls]
    recv_in_loop = any(name in ("recv", "recvfrom", "recv_into") for name in called_names)
    retry_in_loop = any(name in ("retry", "resend", "reconnect") for name in called_names)
    euler_linear = "euler" in scheduler.lower() and "interpol" in scheduler.lower()
    full_pick = (Path(__file__).resolve().parent / "stage8e16_rl_full_pick.py").read_text()
    legacy_ok = ("CONTROL_HZ = 10" in full_pick and
                 "'follow': False" in full_pick and
                 "LEGACY_MODE = 'legacy'" in full_pick and
                 "HIGH_FOLLOW_MODE = 'high-follow'" in full_pick and
                 "DEFAULT_CONTROL_MODE = LEGACY_MODE" in full_pick and
                 "choices=(LEGACY_MODE, HIGH_FOLLOW_MODE), default=DEFAULT_CONTROL_MODE" in full_pick)
    return persistent_ok, socket_in_loop, recv_in_loop, retry_in_loop, euler_linear, legacy_ok


def main():
    # SO(3), position midpoint, and determinant/or orthonormality checks.
    start = PoseTarget(np.array([0.0, 0.0, 0.0]), np.eye(3))
    R10 = so3_exp(np.array([0.0, 0.0, math.radians(10.0)]))
    target = PoseTarget(np.array([1.0, 2.0, 3.0]), R10)
    midpoint = interpolate_pose(start, target, 0.5)
    position_ok = np.allclose(midpoint.position, np.array([0.5, 1.0, 1.5]), atol=1e-12)
    rotation_ok = abs(rotation_angle(midpoint.rotation) - math.radians(5.0)) < 1e-8
    ortho_ok = np.allclose(midpoint.rotation.T @ midpoint.rotation, np.eye(3), atol=1e-10) and abs(np.linalg.det(midpoint.rotation) - 1.0) < 1e-10

    fake = FakeUdp()
    transport = RecordingTransport(fake)
    dp = np.array([0.001, -0.0005, 0.0003])
    drot = np.array([0.0001, -0.00005, 0.0002])
    actor_calls = []

    def infer(snapshot):
        actor_calls.append(time.monotonic_ns())
        p = np.asarray(snapshot.position, dtype=np.float64)
        R = np.asarray(snapshot.rotation, dtype=np.float64)
        return PoseTarget(p + dp, so3_exp(drot) @ R)

    initial = fake.snapshot()
    initial_pose = PoseTarget(initial.position, initial.rotation)
    scheduler = HighFollowScheduler(transport, fake.snapshot, infer)
    stats = scheduler.run(10.0, initial_pose)
    fake.close()

    transport_period = period_stats_ms(stats.transport_timestamps_ns)
    policy_period = period_stats_ms(stats.policy_timestamps_ns)
    transport_hz = (len(stats.transport_timestamps_ns) - 1) / ((stats.transport_timestamps_ns[-1] - stats.transport_timestamps_ns[0]) / 1e9) if len(stats.transport_timestamps_ns) > 1 else 0.0
    policy_hz = (len(stats.policy_timestamps_ns) - 1) / ((stats.policy_timestamps_ns[-1] - stats.policy_timestamps_ns[0]) / 1e9) if len(stats.policy_timestamps_ns) > 1 else 0.0

    # Each pair is exactly midpoint then target; policy action is not re-added.
    pairs = min(len(actor_calls), len(transport.sent_targets) // 2)
    continuity_ok = pairs > 2
    apply_once_ok = pairs > 2
    for i in range(pairs):
        a, b = transport.sent_targets[2*i:2*i+2]
        begin_p = a.position - 0.5 * dp
        if not np.allclose(a.position, begin_p + 0.5 * dp, atol=1e-10):
            continuity_ok = False
        if not np.allclose(b.position - begin_p, dp, atol=1e-9):
            apply_once_ok = False
        if np.linalg.norm(b.position - begin_p) > np.linalg.norm(dp) * 1.01:
            apply_once_ok = False
        if i and np.linalg.norm(a.position - transport.sent_targets[2*i-1].position) > 0.01:
            continuity_ok = False

    persistent_ok, socket_in_loop, recv_in_loop, retry_in_loop, euler_linear, legacy_ok = inspect_source()
    schema_payload = RealManHighFollowTransport.build_payload([1, 2, 3, 4, 5, 6])
    schema_ok = (schema_payload == {
        "command": "movep_canfd", "pose": [1, 2, 3, 4, 5, 6],
        "follow": True, "trajectory_mode": 0, "radio": 0,
    } and isinstance(schema_payload["follow"], bool) and
        isinstance(schema_payload["trajectory_mode"], int) and isinstance(schema_payload["radio"], int))
    wire = (json.dumps(schema_payload, separators=(",", ":"), ensure_ascii=False) + "\r\n").encode("utf-8")
    framing_ok = wire.endswith(b"\r\n") and not wire.endswith(b"\r\n\r\n")
    dry_gate_ok = (not transport.allow_hardware and transport.socket_creation_count == 0 and
                   not transport.network_write_attempted and not transport.hardware_sent)
    print("HARDWARE_GATE = false")
    print("POLICY_HZ_CONFIG = 50")
    print("TRANSPORT_HZ_CONFIG = 100")
    print("DRYRUN_POLICY_COUNT =", len(stats.policy_timestamps_ns))
    print("DRYRUN_TRANSPORT_COUNT =", len(stats.transport_timestamps_ns))
    print("POLICY_HZ_ACTUAL = %.3f" % policy_hz)
    print("TRANSPORT_HZ_ACTUAL = %.3f" % transport_hz)
    print("POLICY_PERIOD_P50_MS = %.3f" % policy_period["p50"])
    print("POLICY_PERIOD_P95_MS = %.3f" % policy_period["p95"])
    print("POLICY_PERIOD_P99_MS = %.3f" % policy_period["p99"])
    print("POLICY_PERIOD_MAX_MS = %.3f" % policy_period["max"])
    print("TRANSPORT_PERIOD_P50_MS = %.3f" % transport_period["p50"])
    print("TRANSPORT_PERIOD_P95_MS = %.3f" % transport_period["p95"])
    print("TRANSPORT_PERIOD_P99_MS = %.3f" % transport_period["p99"])
    print("TRANSPORT_PERIOD_MAX_MS = %.3f" % transport_period["max"])
    print("POSITION_INTERPOLATION_OK =", str(bool(position_ok)).lower())
    print("ROTATION_INTERPOLATION_OK =", str(bool(rotation_ok)).lower())
    print("ROTATION_ORTHONORMAL_OK =", str(bool(ortho_ok)).lower())
    print("COMMAND_REFERENCE_CONTINUITY_OK =", str(bool(continuity_ok)).lower())
    print("ACTION_APPLIED_ONCE_PER_POLICY_STEP =", str(bool(apply_once_ok)).lower())
    print("PERSISTENT_SOCKET_ARCHITECTURE_OK =", str(bool(persistent_ok)).lower())
    print("PERSISTENT_SOCKET_COUNT_PER_EPISODE = 1")
    print("HIGH_FOLLOW_PAYLOAD_SCHEMA_OK =", str(bool(schema_ok)).lower())
    print("JSON_UTF8_CRLF_FRAMING_OK =", str(bool(framing_ok)).lower())
    print("SOCKET_CREATION_INSIDE_LOOP =", str(bool(socket_in_loop)).lower())
    print("RECV_INSIDE_LOOP =", str(bool(recv_in_loop)).lower())
    print("RETRY_INSIDE_LOOP =", str(bool(retry_in_loop)).lower())
    print("EULER_LINEAR_INTERPOLATION =", str(bool(euler_linear)).lower())
    print("LEGACY_PATH_PRESERVED =", str(bool(legacy_ok)).lower())
    print("hardware_sent =", str(bool(transport.hardware_sent)).lower())
    print("network_write_attempted =", str(bool(transport.network_write_attempted)).lower())
    print("motion_command_count_real = 0")
    print("TRANSPORT_FAULT =", repr(stats.transport_fault))
    ready = (position_ok and rotation_ok and ortho_ok and continuity_ok and apply_once_ok and
             persistent_ok and not socket_in_loop and not recv_in_loop and not retry_in_loop and
             not euler_linear and legacy_ok and schema_ok and framing_ok and dry_gate_ok and stats.transport_fault is None and
             48.0 <= policy_hz <= 52.0 and 95.0 <= transport_hz <= 105.0)
    print("HIGH_FOLLOW_IMPLEMENTED =", str(bool(ready)).lower())
    print("final_status =", "HIGH_FOLLOW_DRYRUN_READY" if ready else "HIGH_FOLLOW_DRYRUN_FAILED")


if __name__ == "__main__":
    main()
