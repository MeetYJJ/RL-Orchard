"""Pure base-link geometry shared by the real Reach entry points."""
import numpy as np


def camera_pose_from_robot_end(position, rotation, hand_eye_rotation, hand_eye_translation):
    # T_base_camera = T_base_end @ T_end_camera.  The translation is
    # expressed in robot_end, so R_base_end (not R_end_camera) rotates it.
    return (np.asarray(position) + np.asarray(rotation) @ np.asarray(hand_eye_translation),
            np.asarray(rotation) @ np.asarray(hand_eye_rotation))


def tcp_geometry(position, rotation, apple_base, tcp_offset):
    tcp = np.asarray(position) + np.asarray(rotation) @ np.asarray(tcp_offset)
    return tcp, np.asarray(apple_base) - np.asarray(position), np.asarray(apple_base) - tcp
