import importlib.util
import json
import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(
    ROOT, "src", "orchard_picker", "scripts", "query_realman_version.py"
)
SPEC = importlib.util.spec_from_file_location("query_realman_version", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class VersionQueryValidationTest(unittest.TestCase):
    def test_accepts_gen3_and_joint_version_payloads(self):
        arm = MODULE.validate_version_response(
            "get_arm_software_info",
            json.dumps({"ack": {
                "command": "arm_software_info", "Product_version": "RML63-B",
                "ctrl_info": {"version": "V1"},
            }}),
        )
        self.assertEqual(arm["Product_version"], "RML63-B")
        joint = MODULE.validate_version_response(
            "get_joint_software_version",
            json.dumps({"ack": {
                "state": "joint_software_version", "version": [1] * 6,
            }}),
        )
        self.assertEqual(len(joint["version"]), 6)

    def test_rejects_unrelated_response(self):
        with self.assertRaises(RuntimeError):
            MODULE.validate_version_response(
                "get_arm_software_info",
                json.dumps({"ack": {"state": "current_arm_state"}}),
            )


if __name__ == "__main__":
    unittest.main()
