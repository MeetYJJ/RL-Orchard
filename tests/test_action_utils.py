#!/usr/bin/env python3

import math
import unittest

import numpy as np

from orchard_picker.rl_data.action_utils import executed_delta_tcp


class ActionUtilsTest(unittest.TestCase):
    def test_translation_and_rotation_vector(self):
        current = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        angle = math.pi / 2.0
        following = [
            0.1, -0.2, 0.3,
            0.0, 0.0, math.sin(angle / 2.0), math.cos(angle / 2.0),
        ]
        delta = executed_delta_tcp(current, following)
        np.testing.assert_allclose(delta[:3], [0.1, -0.2, 0.3], atol=1e-6)
        np.testing.assert_allclose(delta[3:], [0.0, 0.0, angle], atol=1e-6)

    def test_quaternion_sign_does_not_change_action(self):
        pose = [0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.9]
        same_pose_negative_q = [0.0, 0.0, 0.0, -0.1, -0.2, -0.3, -0.9]
        delta = executed_delta_tcp(pose, same_pose_negative_q)
        np.testing.assert_allclose(delta, np.zeros(6), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
