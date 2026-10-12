from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
from .config import TARGET_FRAME

@dataclass(frozen=True)
class VisionSample:
    camera_xyz: np.ndarray
    received_mono: float
    frame_id: str
    ros_stamp_s: float

@dataclass
class TargetLatch:
    required_samples: int = 3
    stable_radius_m: float = 0.025
    target: VisionSample | None = None
    stable_count: int = 0
    reliable_camera_xyz: np.ndarray | None = None
    reliable_base_xyz: np.ndarray | None = None
    reliable_received_mono: float | None = None
    reliable_base_source_mono: float | None = None
    frozen_target_base_xyz: np.ndarray | None = None
    committed_base_xyz: np.ndarray | None = None
    committed_mono: float | None = None
    phase: str = "TARGET_ACQUISITION"

    def accept(self, sample: VisionSample) -> bool:
        if sample.frame_id != TARGET_FRAME or sample.camera_xyz.shape != (3,) or not np.all(np.isfinite(sample.camera_xyz)) or np.linalg.norm(sample.camera_xyz) < 1e-9:
            return False
        if self.target is not None and np.linalg.norm(sample.camera_xyz - self.target.camera_xyz) <= self.stable_radius_m:
            self.stable_count += 1
        else:
            self.stable_count = 1
        self.target = sample
        if self.stable_count >= self.required_samples:
            self.reliable_camera_xyz = sample.camera_xyz.copy()
            self.reliable_received_mono = sample.received_mono
            # The base target is recomputed once for this new camera sample.
            # Repeated state polls must not make an old camera frame fresh.
            if (self.reliable_base_source_mono is not None and
                    sample.received_mono != self.reliable_base_source_mono):
                self.reliable_base_xyz = None
        return True

    def base_target(self, p_end, R_base_end) -> np.ndarray | None:
        if self.target is None:
            return None
        from .transforms import apple_camera_to_base
        base = apple_camera_to_base(p_end, R_base_end, self.target.camera_xyz)
        if np.all(np.isfinite(base)):
            return base
        return None

    def update_reliable_base(self, base_xyz, now=None, source_mono=None):
        if (self.frozen_target_base_xyz is not None or
                self.stable_count < self.required_samples or base_xyz is None):
            return False
        source = (self.reliable_received_mono if source_mono is None
                  else float(source_mono))
        if (source is None or
                source == self.reliable_base_source_mono):
            return False
        if np.asarray(base_xyz).shape == (3,) and np.all(np.isfinite(base_xyz)):
            self.reliable_base_xyz = np.asarray(base_xyz, dtype=np.float64).copy()
            # ``reliable_received_mono`` is the camera sample's monotonic
            # receipt time, never the time of a later robot poll.
            self.reliable_received_mono = source
            self.reliable_base_source_mono = source
            return True
        return False

    def freeze_base(self, base_xyz):
        """Freeze one valid base-frame target for the terminal approach."""
        candidate = np.asarray(base_xyz, dtype=np.float64)
        if candidate.shape != (3,) or not np.all(np.isfinite(candidate)):
            return False
        if self.frozen_target_base_xyz is None:
            self.frozen_target_base_xyz = candidate.copy()
        self.reliable_base_xyz = self.frozen_target_base_xyz.copy()
        self.phase = "TERMINAL_ARMED"
        return True

    def frozen(self):
        return self.frozen_target_base_xyz is not None

    def age(self, now=None):
        now = time.monotonic() if now is None else float(now)
        return float("inf") if self.target is None else max(0.0, now - self.target.received_mono)

    def reliable_age(self, now=None):
        now = time.monotonic() if now is None else float(now)
        return float("inf") if self.reliable_received_mono is None else max(0.0, now - self.reliable_received_mono)

    def arm(self):
        if self.reliable_base_xyz is not None:
            self.phase = "TERMINAL_ARMED"
            return True
        return False

    def commit(self, base_xyz=None, reason="PREGRASP_DISTANCE", now=None):
        candidate = (self.frozen_target_base_xyz if base_xyz is None and self.frozen()
                     else self.reliable_base_xyz if base_xyz is None
                     else np.asarray(base_xyz, dtype=np.float64))
        if candidate is None or np.asarray(candidate).shape != (3,) or not np.all(np.isfinite(candidate)):
            return False
        self.committed_base_xyz = np.asarray(candidate, dtype=np.float64).copy()
        self.committed_mono = time.monotonic() if now is None else float(now)
        self.phase = "TERMINAL_COMMITTED"
        return True

    def committed(self):
        return self.committed_base_xyz is not None

    def update_phase(self, distance_m, vision_fresh=True):
        if self.committed():
            self.phase = "TERMINAL_COMMITTED"
        elif ((self.frozen_target_base_xyz is not None or
               self.reliable_base_xyz is not None) and float(distance_m) <= 0.100):
            self.phase = "TERMINAL_ARMED"
        elif self.reliable_base_xyz is not None:
            self.phase = "TRACKING"
        return self.phase

class TargetSubscriber:
    """ROS callback adapter; all time comparisons use monotonic receipt time."""
    def __init__(self, latch: TargetLatch | None = None):
        self.latch = latch or TargetLatch()
        self.last_callback_mono = None
        self.callback_count = 0
        self.rejected_count = 0
        self.last_frame = None

    def callback(self, msg):
        self.last_callback_mono = time.monotonic()
        self.callback_count += 1
        frame = str(getattr(getattr(msg, "header", None), "frame_id", ""))
        point = getattr(msg, "point", None)
        if point is None:
            self.rejected_count += 1
            return
        xyz = np.asarray([point.x, point.y, point.z], dtype=np.float64)
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        ros_s = float(getattr(stamp, "sec", 0)) + float(getattr(stamp, "nanosec", 0))*1e-9
        self.last_frame = frame
        if not self.latch.accept(VisionSample(xyz, self.last_callback_mono, frame, ros_s)):
            self.rejected_count += 1

    @property
    def ready(self):
        return self.callback_count > 0

    @property
    def target_fresh(self):
        return self.latch.age() <= 0.30
