#!/usr/bin/env python3

import copy
import unittest

from orchard_picker.rl_data.collection_planning import (
    CollectionConfigError,
    generate_joint_jitter,
    select_least_sampled_pose,
    validate_collection_config,
)


def safe_config():
    return {
        "initialization": {
            "mode": "pose_library",
            "allow_verified_movej_service": False,
            "initial_pose_library": [
                {"id": "home", "verified": True, "motion": "home_service", "service": "/pick_task_manager/home"},
                {"id": "init_01", "verified": True, "motion": "trigger_service", "service": "/verified/init_01"},
                {"id": "init_02", "verified": True, "motion": "trigger_service", "service": "/verified/init_02"},
            ],
            "bounded_random_jitter": {
                "enabled": False,
                "joint_offset_min_deg": [0.0] * 6,
                "joint_offset_max_deg": [0.0] * 6,
            },
        }
    }


class CollectionPlanningTest(unittest.TestCase):
    def test_init_pose_selection_returns_enabled_pose(self):
        library = safe_config()["initialization"]["initial_pose_library"]
        library[0]["enabled"] = False
        selected = select_least_sampled_pose(library, {}, seed=1)
        self.assertIn(selected["id"], ("init_01", "init_02"))

    def test_least_sampled_coverage_balancing(self):
        library = safe_config()["initialization"]["initial_pose_library"]
        selected = select_least_sampled_pose(
            library, {"home": 20, "init_01": 6, "init_02": 7}, seed=3
        )
        self.assertEqual(selected["id"], "init_01")

    def test_deterministic_random_seed(self):
        library = safe_config()["initialization"]["initial_pose_library"]
        first = select_least_sampled_pose(library, {}, seed=42, selection_index=9)
        second = select_least_sampled_pose(library, {}, seed=42, selection_index=9)
        self.assertEqual(first["id"], second["id"])

    def test_disabled_jitter_is_exactly_zero(self):
        config = safe_config()
        validate_collection_config(config)
        self.assertEqual(generate_joint_jitter(config, 1), [0.0] * 6)

    def test_safe_config_rejects_direct_bridge_movej(self):
        config = safe_config()
        config["initialization"]["initial_pose_library"] = [{
            "id": "unsafe", "motion": "movej", "verified": True,
            "joint_deg": [0.0] * 6, "speed": 5,
            "service": "/rm_json_bridge/movej",
        }]
        with self.assertRaises(CollectionConfigError):
            validate_collection_config(config)

    def test_enabled_jitter_requires_explicit_gate(self):
        config = safe_config()
        config["initialization"]["mode"] = "bounded_random_jitter"
        config["initialization"]["bounded_random_jitter"]["enabled"] = True
        config["initialization"]["bounded_random_jitter"]["joint_offset_max_deg"][0] = 1.0
        with self.assertRaises(CollectionConfigError):
            validate_collection_config(config)

    def test_jitter_is_deterministic_after_explicit_configuration(self):
        config = safe_config()
        config["initialization"]["mode"] = "bounded_random_jitter"
        config["initialization"]["allow_verified_movej_service"] = True
        config["initialization"]["initial_pose_library"] = [{
            "id": "safe_jitter_base", "motion": "movej", "verified": True,
            "joint_deg": [0.0] * 6, "speed": 5,
            "service": "/verified/move_initial_joint",
        }]
        config["initialization"]["bounded_random_jitter"] = {
            "enabled": True,
            "joint_offset_min_deg": [-1.0] * 6,
            "joint_offset_max_deg": [1.0] * 6,
        }
        validate_collection_config(config)
        self.assertEqual(
            generate_joint_jitter(config, 123, 4),
            generate_joint_jitter(copy.deepcopy(config), 123, 4),
        )


if __name__ == "__main__":
    unittest.main()
