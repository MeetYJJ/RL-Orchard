"""Read-only RealMan UDP active-reporting receiver.

This module deliberately has no transmit path.  It accepts the JSON datagrams
emitted by the controller's UDP active-reporting interface, records host
monotonic receive time, and maps only fields whose meaning is documented by the
packet itself.  It does not infer a TCP-frame pose origin and therefore never
populates ``tcp_pose`` until that origin is independently resolved.
"""

from __future__ import annotations

import json
import math
import socket
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def _finite_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite_vector(value: Any, length: int) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    result = [_finite_float(item) for item in value]
    return result if all(item is not None for item in result) else None


def _scalar_error(value: Any) -> Optional[int]:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        value = value[0]
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _p95(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    position = 0.95 * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


@dataclass(frozen=True)
class UdpFeedbackFrame:
    """One decoded datagram and its host-side receive metadata."""

    receive_timestamp_s: float
    source_ip: str
    source_port: int
    packet_size_bytes: int
    raw_text: str
    payload: Dict[str, Any]

    def _joint_status(self) -> Dict[str, Any]:
        value = self.payload.get("joint_status")
        return value if isinstance(value, dict) else {}

    def joint_position_rad(self) -> Optional[List[float]]:
        """Convert documented 0.001-degree positions to radians."""
        values = _finite_vector(self._joint_status().get("joint_position"), 6)
        if values is None:
            return None
        return [value * 0.001 * math.pi / 180.0 for value in values]

    def controller_reported_pose(self) -> Dict[str, Any]:
        waypoint = self.payload.get("waypoint")
        if not isinstance(waypoint, dict):
            return {
                "raw": None,
                "frame": None,
                "origin_status": "UNAVAILABLE",
            }
        position = waypoint.get("position")
        euler = waypoint.get("euler")
        quaternion = waypoint.get("quat", waypoint.get("quaternion"))
        raw = {
            "position": position,
            "euler": euler,
            "quaternion": quaternion,
        }
        return {
            "raw": raw,
            "frame": None,
            "origin_status": "POSE_ORIGIN_UNRESOLVED",
            "raw_units": {
                "position": "1e-6 m per protocol",
                "euler": "1e-3 rad per protocol",
                "quaternion": "protocol scale; convention unresolved",
            },
        }

    def to_robot_state_mapping(self) -> Dict[str, Any]:
        """Return a ROS-independent RobotState mapping.

        The mapping keeps unresolved semantics explicit.  ``tcp_pose`` remains
        None because the UDP waypoint origin has not been proven to be the
        policy TCP.  ``sys_err`` is mapped from the documented packet ``err``;
        no separate ``arm_err`` field is invented.
        """
        joint_status = self._joint_status()
        sys_err = _scalar_error(self.payload.get("err"))
        joint_err_code = joint_status.get("joint_err_code")
        if not isinstance(joint_err_code, list) or len(joint_err_code) != 6:
            joint_err_code = None
        enable_flags = joint_status.get("joint_en_flag")
        enable = None
        if isinstance(enable_flags, list) and len(enable_flags) == 6:
            enable = all(bool(int(value)) for value in enable_flags)
        return {
            "joint_position_rad": self.joint_position_rad(),
            "controller_reported_pose": self.controller_reported_pose(),
            "tcp_pose": None,
            "error_code": sys_err,
            "sys_err": sys_err,
            "arm_err": None,
            "joint_err_code": joint_err_code,
            "enable": enable,
            "frame": None,
            "frame_status": "UNKNOWN",
            "controller_timestamp_s": None,
            "host_receive_timestamp_s": self.receive_timestamp_s,
            "source": [self.source_ip, self.source_port],
        }


def decode_udp_feedback(
    datagram: bytes,
    source: Tuple[str, int],
    receive_timestamp_s: Optional[float] = None,
) -> UdpFeedbackFrame:
    """Decode one RealMan JSON datagram without any network transmission."""
    if receive_timestamp_s is None:
        receive_timestamp_s = time.monotonic()
    raw_text = datagram.decode("utf-8").strip()
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise ValueError("UDP payload must be a JSON object")
    return UdpFeedbackFrame(
        receive_timestamp_s=float(receive_timestamp_s),
        source_ip=str(source[0]),
        source_port=int(source[1]),
        packet_size_bytes=len(datagram),
        raw_text=raw_text,
        payload=payload,
    )


class UdpFeedbackReader:
    """Short-lived, read-only UDP sampler for active reporting validation."""

    def __init__(
        self,
        bind_ip: str = "192.168.1.100",
        port: int = 8090,
        expected_period_s: Optional[float] = None,
        socket_timeout_s: float = 0.25,
        max_datagram_bytes: int = 65535,
    ):
        self.bind_ip = str(bind_ip)
        self.port = int(port)
        self.expected_period_s = expected_period_s
        self.socket_timeout_s = float(socket_timeout_s)
        self.max_datagram_bytes = int(max_datagram_bytes)

    def collect(self, duration_s: float, max_packets: Optional[int] = None) -> Dict[str, Any]:
        """Collect packets for a bounded interval and return decoded evidence.

        The socket is bound only for receiving.  No ``send`` or ``sendto`` path
        exists in this class.  Exact UDP loss is unavailable because the
        documented datagram has no sequence number; an optional rate-based loss
        estimate is reported separately.
        """
        duration_s = float(duration_s)
        if duration_s <= 0.0:
            raise ValueError("duration_s must be positive")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        frames: List[UdpFeedbackFrame] = []
        malformed = 0
        timeout_count = 0
        errors: List[str] = []
        start = time.monotonic()
        deadline = start + duration_s
        try:
            sock.bind((self.bind_ip, self.port))
            sock.settimeout(self.socket_timeout_s)
            while time.monotonic() < deadline:
                if max_packets is not None and len(frames) >= int(max_packets):
                    break
                try:
                    datagram, source = sock.recvfrom(self.max_datagram_bytes)
                except socket.timeout:
                    timeout_count += 1
                    continue
                except OSError as exc:
                    errors.append("recv: {}".format(exc))
                    break
                received = time.monotonic()
                try:
                    frames.append(decode_udp_feedback(datagram, source, received))
                except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
                    malformed += 1
                    errors.append("malformed: {}".format(exc))
        finally:
            sock.close()
        elapsed = (frames[-1].receive_timestamp_s - frames[0].receive_timestamp_s
                   if len(frames) >= 2 else 0.0)
        intervals = [
            frames[index].receive_timestamp_s - frames[index - 1].receive_timestamp_s
            for index in range(1, len(frames))
        ]
        sizes = [frame.packet_size_bytes for frame in frames]
        expected = None
        estimated_lost = None
        if self.expected_period_s and self.expected_period_s > 0.0 and elapsed > 0.0:
            expected = int(round(elapsed / self.expected_period_s)) + 1
            estimated_lost = max(expected - len(frames), 0)
        return {
            "bind_ip": self.bind_ip,
            "port": self.port,
            "duration_s": duration_s,
            "actual_elapsed_s": elapsed,
            "packet_count": len(frames),
            "malformed_count": malformed,
            "socket_timeout_count": timeout_count,
            "errors": errors,
            "source_endpoints": sorted({
                "{}:{}".format(frame.source_ip, frame.source_port)
                for frame in frames
            }),
            "frequency_hz": (len(frames) / elapsed if elapsed > 0.0 else None),
            "interval_s": {
                "mean": statistics.mean(intervals) if intervals else None,
                "median": statistics.median(intervals) if intervals else None,
                "p95": _p95(intervals),
                "max": max(intervals) if intervals else None,
            },
            "packet_size_bytes": {
                "min": min(sizes) if sizes else None,
                "median": statistics.median(sizes) if sizes else None,
                "max": max(sizes) if sizes else None,
            },
            "packet_loss": {
                "exact": None,
                "expected_period_s": self.expected_period_s,
                "expected_count_rate_estimate": expected,
                "estimated_lost_count": estimated_lost,
                "status": "ESTIMATE_ONLY_NO_SEQUENCE_NUMBER",
            },
            "frames": frames,
        }
