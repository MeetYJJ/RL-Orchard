from __future__ import annotations
import importlib.util
import math
import numpy as np
from .config import BUNDLE, HAND_EYE_R, HAND_EYE_T, TCP_OFFSET

_POSE_MATH = None

def validate_rotation(R):
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3,3) or not np.all(np.isfinite(R)) or not np.allclose(R @ R.T, np.eye(3), atol=1e-5) or np.linalg.det(R) < 0.99:
        raise ValueError("invalid rotation")
    return R

def _verified_pose_math():
    """Load the same hardware-independent pose math used by legacy Reach."""
    global _POSE_MATH
    if _POSE_MATH is None:
        path = BUNDLE / "training/deployment/pose_math.py"
        spec = importlib.util.spec_from_file_location("_full_pick_verified_pose_math", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"verified pose math unavailable: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _POSE_MATH = module
    return _POSE_MATH

def realman_udp_euler_to_matrix(euler_raw):
    """RealMan waypoint Euler (roll/pitch/yaw, 0.001 rad) to R_base_end.

    This is the verified Stage8E15 conversion path: RealMan protocol values
    become an XYZW quaternion through orchard_picker.protocol, then the frozen
    deployment pose math converts that quaternion to a rotation matrix.
    """
    raw = np.asarray(euler_raw, dtype=np.float64)
    if raw.shape != (3,) or not np.all(np.isfinite(raw)):
        raise ValueError("invalid UDP robot_end euler")
    from orchard_picker.protocol import protocol_to_pose
    pose = protocol_to_pose([0.0, 0.0, 0.0, *raw.tolist()])
    quaternion_xyzw = np.asarray([
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ], dtype=np.float64)
    if not np.all(np.isfinite(quaternion_xyzw)) or float(np.linalg.norm(quaternion_xyzw)) < 1e-12:
        raise ValueError("invalid UDP robot_end quaternion")
    return validate_rotation(
        _verified_pose_math().quaternion_xyzw_to_matrix(quaternion_xyzw)
    )

def so3_exp(v):
    v = np.asarray(v, dtype=np.float64); n = float(np.linalg.norm(v)); K = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]], dtype=np.float64)
    if n < 1e-9: return np.eye(3) + K + .5*K@K
    return np.eye(3) + math.sin(n)/n*K + (1-math.cos(n))/(n*n)*(K@K)

def so3_log(R):
    R = validate_rotation(R); c = float(np.clip((np.trace(R)-1)/2, -1, 1)); n = math.acos(c)
    v = np.asarray([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]], dtype=np.float64)
    if n < 1e-8: return .5*v
    return v * (n/(2*math.sin(n)))

def rpy_to_matrix(rpy):
    rpy = np.asarray(rpy, dtype=np.float64)
    if rpy.shape != (3,) or not np.all(np.isfinite(rpy)):
        raise ValueError("invalid roll/pitch/yaw")
    # Keep this conversion on the audited RealMan protocol path.  The former
    # hand-written ZYX matrix had ``-sy*sr`` in R[1,2] (it must be
    # ``-cy*sr``), which made valid UDP Euler samples non-orthogonal.
    return realman_udp_euler_to_matrix(rpy * 1000.0)

def camera_pose_from_end(p_end, R_base_end):
    p_end = np.asarray(p_end, dtype=np.float64); R_base_end = validate_rotation(R_base_end)
    return p_end + R_base_end @ HAND_EYE_T, R_base_end @ HAND_EYE_R

def apple_camera_to_base(p_end, R_base_end, apple_camera):
    p_cam, R_base_cam = camera_pose_from_end(p_end, R_base_end)
    return p_cam + R_base_cam @ np.asarray(apple_camera, dtype=np.float64)

def flange_target_for_apple(apple_base, held_R, tcp_offset=TCP_OFFSET):
    return np.asarray(apple_base, dtype=np.float64) - validate_rotation(held_R) @ np.asarray(tcp_offset, dtype=np.float64)

def actual_tcp(p_end, R_base_end, tcp_offset=TCP_OFFSET):
    return np.asarray(p_end, dtype=np.float64) + validate_rotation(R_base_end) @ np.asarray(tcp_offset, dtype=np.float64)

def observation(joint_rad, end_xyz, apple_base, current_R, held_R):
    j = np.asarray(joint_rad, dtype=np.float64); p = np.asarray(end_xyz, dtype=np.float64); target = np.asarray(apple_base, dtype=np.float64)
    if j.shape != (6,) or p.shape != (3,) or target.shape != (3,): raise ValueError("invalid observation dimensions")
    return np.concatenate((j, p, target-p, so3_log(validate_rotation(held_R) @ validate_rotation(current_R).T)))

def q_from_R(R):
    R = validate_rotation(R); t = float(np.trace(R))
    if t > 0:
        s=math.sqrt(t+1)*2; q=[(R[2,1]-R[1,2])/s,(R[0,2]-R[2,0])/s,(R[1,0]-R[0,1])/s,.25*s]
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s=math.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2; q=[.25*s,(R[0,1]+R[1,0])/s,(R[0,2]+R[2,0])/s,(R[2,1]-R[1,2])/s]
    elif R[1,1] > R[2,2]:
        s=math.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2; q=[(R[0,1]+R[1,0])/s,.25*s,(R[1,2]+R[2,1])/s,(R[0,2]-R[2,0])/s]
    else:
        s=math.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2; q=[(R[0,2]+R[2,0])/s,(R[1,2]+R[2,1])/s,.25*s,(R[1,0]-R[0,1])/s]
    q=np.asarray(q); return (q/np.linalg.norm(q)).tolist()

def protocol_pose(position_m, R):
    qx,qy,qz,qw=q_from_R(R); roll=math.atan2(2*(qw*qx+qy*qz),1-2*(qx*qx+qy*qy)); sp=2*(qw*qy-qz*qx); pitch=math.copysign(math.pi/2,sp) if abs(sp)>=1 else math.asin(sp); yaw=math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
    return [int(round(float(v)*1e6)) for v in np.asarray(position_m)] + [int(round(roll*1000)),int(round(pitch*1000)),int(round(yaw*1000))]

# Explicit names used by the audited calibration contract.
camera_pose_from_robot_end = camera_pose_from_end
matrix_to_rotvec = so3_log
rotvec_to_matrix = so3_exp

def tcp_geometry(position, rotation, apple_base, tcp_offset=TCP_OFFSET):
    tcp = actual_tcp(position, rotation, tcp_offset)
    return tcp, np.asarray(apple_base, dtype=np.float64)-np.asarray(position, dtype=np.float64), np.asarray(apple_base, dtype=np.float64)-tcp
