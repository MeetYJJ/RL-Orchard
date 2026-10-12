"""Dry-run-only Movep_CANFD payload builder.

This module deliberately has no controller socket or ROS service dependency.
It can serialize the reviewed schema and enforces a one-send budget.  The
endpoint identity is recorded below, but transport remains intentionally
unavailable in this dry-run stage.
"""

import json
import threading


class MovepCanfdWriter:
    """Build one reviewed payload without opening or writing a socket.

    The controller identity and framing below were verified by a separate
    read-only query in Stage8E12.2.  This class remains dry-run-only until a
    separately authorized stage implements and reviews the real transport.
    """

    VERIFIED_CONTROLLER_IP = "192.168.1.18"
    VERIFIED_CONTROLLER_PORT = 8080
    VERIFIED_TRANSPORT = "TCP"
    VERIFIED_FRAMING = "JSON_UTF8_CRLF"
    IDENTITY_EVIDENCE = "/tmp/stage8e12_2_transport_verification/controller_identity.json"

    def __init__(self, controller_ip, controller_port, *, dry_run=True):
        self.controller_ip = str(controller_ip)
        self.controller_port = int(controller_port)
        self.dry_run = bool(dry_run)
        self.send_count = 0
        self._lock = threading.Lock()

    @staticmethod
    def build_payload(protocol_pose, follow=False, trajectory_mode=0, radio=0):
        pose = [int(value) for value in protocol_pose]
        if len(pose) != 6:
            raise ValueError("Movep_CANFD pose must contain six integers")
        if not isinstance(follow, bool):
            raise TypeError("follow must be bool")
        if not isinstance(trajectory_mode, int) or not isinstance(radio, int):
            raise TypeError("trajectory_mode and radio must be integers")
        return {
            "command": "movep_canfd",
            "pose": pose,
            "follow": follow,
            "trajectory_mode": trajectory_mode,
            "radio": radio,
        }

    @classmethod
    def serialize_payload(cls, protocol_pose, follow=False, trajectory_mode=0, radio=0):
        return json.dumps(
            cls.build_payload(protocol_pose, follow, trajectory_mode, radio),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def verified_transport_metadata(cls):
        """Return read-only identity metadata; does not open a connection."""
        return {
            "controller_ip": cls.VERIFIED_CONTROLLER_IP,
            "controller_port": cls.VERIFIED_CONTROLLER_PORT,
            "transport": cls.VERIFIED_TRANSPORT,
            "framing": cls.VERIFIED_FRAMING,
            "identity_evidence": cls.IDENTITY_EVIDENCE,
        }

    def send_once(self, payload):
        """Refuse transport in this stage; enforce the one-send budget."""
        with self._lock:
            if self.send_count >= 1:
                raise RuntimeError("Stage8E12 single-step writer already consumed")
            if not self.dry_run:
                raise RuntimeError("controller transport is disabled until endpoint review")
            self.send_count += 1
            return {
                "payload": payload,
                "hardware_sent": False,
                "network_write_attempted": False,
            }
