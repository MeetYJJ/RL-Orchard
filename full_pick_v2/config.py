from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROS_PICKER = Path(os.environ.get("RL_ORCHARD_PICKER_CONFIG", "/home/nvidia/ros2_ws/src/orchard_picker/config/picker_params.yaml"))
BUNDLE = ROOT / "deployments/stage8e6_reachactor_shadow_20261001"
RUN = BUNDLE / "training/runs/stage8b2_1_cartesian_bc_dagger_20260929"
CHECKPOINT = RUN / "actor_best.pt"
NORMALIZATION = RUN / "normalization.npz"
VISION_SCRIPT = ROOT / "scripts/run_rl_vision.sh"
CAMERA_SERIAL = "908212070350"
TARGET_TOPIC = "/apple/target_point"
TARGET_FRAME = "camera_link"
HAND_EYE_R = np.asarray([[-0.85684041, -0.51514442, 0.02123071], [0.51506258, -0.85709906, -0.00957856], [0.02313116, 0.00272785, 0.99972872]], dtype=np.float64)
HAND_EYE_T = np.asarray([0.0757663, 0.0625736, 0.01734084], dtype=np.float64)
TCP_OFFSET = np.asarray([0.0005, -0.005, 0.158], dtype=np.float64)
SERVICES = ("/rm_json_bridge/get_arm_state", "/rm_json_bridge/gripper_pick", "/rm_json_bridge/gripper_release", "/rm_json_bridge/movej")

@dataclass(frozen=True)
class PickConfig:
    raw: dict
    home_joint_deg: np.ndarray
    home_speed: int
    home_tolerance_deg: float
    pick_speed: int
    pick_force: int
    release_speed: int
    workspace_min: np.ndarray
    workspace_max: np.ndarray
    enforce_workspace: bool
    target_timeout: float
    service_timeout: float

    @classmethod
    def load(cls, path: Path = ROS_PICKER) -> "PickConfig":
        import yaml
        if not path.is_file():
            raise FileNotFoundError(f"picker_params.yaml not found: {path}")
        data = yaml.safe_load(path.read_text()) or {}
        p = data.get("pick_task_manager", {}).get("ros__parameters", {})
        home = np.asarray(p.get("home_joint_deg", []), dtype=np.float64)
        if home.shape != (6,) or not np.all(np.isfinite(home)):
            raise ValueError("picker_params home_joint_deg must contain six finite values")
        tcp = np.asarray(p.get("gripper_tcp_xyz_m", []), dtype=np.float64)
        if tcp.shape != (3,) or not np.allclose(tcp, TCP_OFFSET, atol=1e-9, rtol=0):
            raise ValueError(f"configured TCP differs from frozen Reach TCP: {tcp.tolist()}")
        if (int(p.get("gripper_pick_speed", 150)), int(p.get("gripper_pick_force", 150)), int(p.get("gripper_release_speed", 300))) != (150, 150, 300):
            raise ValueError("picker gripper speeds/force do not match Full Pick V2 contract")
        return cls(data, home, int(p.get("home_speed", 10)), float(p.get("home_joint_tolerance_deg", 1.0)), int(p.get("gripper_pick_speed", 150)), int(p.get("gripper_pick_force", 150)), int(p.get("gripper_release_speed", 300)), np.asarray(p.get("workspace_min", [-np.inf]*3), dtype=np.float64), np.asarray(p.get("workspace_max", [np.inf]*3), dtype=np.float64), bool(p.get("enforce_workspace_limits", False)), float(p.get("target_timeout", 2.0)), float(p.get("service_call_timeout", 90.0)))

    @property
    def hand_eye(self):
        p = self.raw.get("hand_eye_static_tf_publisher", {}).get("ros__parameters", {})
        R = np.asarray(p.get("rotation_matrix", HAND_EYE_R.reshape(-1)), dtype=np.float64).reshape(3, 3)
        t = np.asarray(p.get("translation_m", HAND_EYE_T), dtype=np.float64)
        if not np.allclose(R, HAND_EYE_R, atol=1e-8, rtol=0) or not np.allclose(t, HAND_EYE_T, atol=1e-8, rtol=0):
            raise ValueError("picker hand-eye calibration differs from verified calibration")
        if p.get("parent_frame") != "robot_end" or p.get("child_frame") != "camera_link" or bool(p.get("invert_matrix", False)):
            raise ValueError("hand-eye frame contract is not robot_end -> camera_link")
        return R, t
