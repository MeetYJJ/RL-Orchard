import unittest

import numpy as np

from orchard_picker.rl_data.joint_state import (
    JointStateMappingError,
    estimate_rate_hz,
    normalize_joint_positions,
    transition_delta,
)


class JointStateTest(unittest.TestCase):
    names = ("J1", "J2", "J3", "J4", "J5", "J6")

    def test_name_reorder(self):
        actual = ("J3", "J1", "J6", "J2", "J5", "J4")
        values = normalize_joint_positions(actual, [3, 1, 6, 2, 5, 4], self.names)
        np.testing.assert_array_equal(values, [1, 2, 3, 4, 5, 6])

    def test_wrong_dimension_and_missing_name_rejected(self):
        with self.assertRaises(JointStateMappingError):
            normalize_joint_positions(self.names, [1, 2], self.names)
        with self.assertRaises(JointStateMappingError):
            normalize_joint_positions(("J1", "J2", "J3", "J4", "J5", "X"), [1] * 6, self.names)

    def test_nan_rejected(self):
        with self.assertRaises(JointStateMappingError):
            normalize_joint_positions(self.names, [1, 2, np.nan, 4, 5, 6], self.names)

    def test_transition_and_rate(self):
        np.testing.assert_allclose(transition_delta([0] * 6, [1, 2, 3, 4, 5, 6]), [1, 2, 3, 4, 5, 6])
        self.assertAlmostEqual(estimate_rate_hz([0, 100000000, 200000000]), 10.0)
        self.assertAlmostEqual(
            estimate_rate_hz([0, 0, 200000000, 200000000, 400000000]), 5.0
        )


if __name__ == "__main__":
    unittest.main()
