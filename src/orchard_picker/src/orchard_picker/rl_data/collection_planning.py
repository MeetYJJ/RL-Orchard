"""Pure, hardware-free planning helpers for semi-automatic collection."""

import json
import math
import os
import random

import yaml


class CollectionConfigError(ValueError):
    """Raised when collection configuration is unsafe or incomplete."""


def _finite_vector(value, length, name):
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise CollectionConfigError("{} must contain {} values".format(name, length))
    result = [float(item) for item in value]
    if not all(math.isfinite(item) for item in result):
        raise CollectionConfigError("{} must contain only finite values".format(name))
    return result


def load_collection_config(path):
    """Load and validate a collection YAML without contacting ROS or hardware."""
    with open(path, "r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    config = payload.get("rl_collection", payload)
    validate_collection_config(config)
    return config


def validate_collection_config(config):
    """Reject implicit/random motion and incomplete direct-motion settings."""
    if not isinstance(config, dict):
        raise CollectionConfigError("rl_collection must be a mapping")
    initialization = config.get("initialization", {})
    if not isinstance(initialization, dict):
        raise CollectionConfigError("initialization must be a mapping")
    mode = str(initialization.get("mode", "pose_library"))
    if mode not in ("pose_library", "bounded_random_jitter"):
        raise CollectionConfigError("unsupported initialization mode: {}".format(mode))

    library = initialization.get("initial_pose_library", [])
    if not library:
        raise CollectionConfigError("initial_pose_library must not be empty")
    identifiers = set()
    for index, pose in enumerate(library):
        if not isinstance(pose, dict):
            raise CollectionConfigError("initial pose {} must be a mapping".format(index))
        pose_id = str(pose.get("id", "")).strip()
        if not pose_id or pose_id in identifiers:
            raise CollectionConfigError("initial pose ids must be non-empty and unique")
        identifiers.add(pose_id)
        motion = str(pose.get("motion", "home_service"))
        if motion not in ("home_service", "trigger_service", "movej"):
            raise CollectionConfigError("{} has unsupported motion {}".format(pose_id, motion))
        if not bool(pose.get("verified", False)):
            raise CollectionConfigError("{} is not marked verified".format(pose_id))
        service = str(pose.get("service", "")).strip()
        if motion in ("home_service", "trigger_service") and not service:
            raise CollectionConfigError("{} requires an explicit service".format(pose_id))
        if motion == "movej":
            _finite_vector(pose.get("joint_deg"), 6, "{}.joint_deg".format(pose_id))
            speed = pose.get("speed")
            if (
                speed is None or not math.isfinite(float(speed))
                or int(speed) <= 0 or float(speed) != float(int(speed))
            ):
                raise CollectionConfigError("{} requires an explicit positive speed".format(pose_id))
            blend = pose.get("blend_radius", 0)
            if int(blend) < 0 or float(blend) != float(int(blend)):
                raise CollectionConfigError("{} blend_radius must be a nonnegative integer".format(pose_id))
            if not service:
                raise CollectionConfigError("{} requires a verified motion service".format(pose_id))
            if service.rstrip("/") == "/rm_json_bridge/movej":
                raise CollectionConfigError(
                    "direct rm_json_bridge/movej is forbidden; use a safety-validating service"
                )

    jitter = initialization.get("bounded_random_jitter", {})
    if not isinstance(jitter, dict):
        raise CollectionConfigError("bounded_random_jitter must be a mapping")
    enabled = bool(jitter.get("enabled", False))
    lower = _finite_vector(
        jitter.get("joint_offset_min_deg", [0.0] * 6), 6,
        "bounded_random_jitter.joint_offset_min_deg",
    )
    upper = _finite_vector(
        jitter.get("joint_offset_max_deg", [0.0] * 6), 6,
        "bounded_random_jitter.joint_offset_max_deg",
    )
    if any(low > high for low, high in zip(lower, upper)):
        raise CollectionConfigError("jitter minimum must not exceed maximum")
    if enabled:
        if mode != "bounded_random_jitter":
            raise CollectionConfigError("enabled jitter requires bounded_random_jitter mode")
        if not bool(initialization.get("allow_verified_movej_service", False)):
            raise CollectionConfigError(
                "enabled jitter requires allow_verified_movej_service=true"
            )
        incompatible = [
            str(pose.get("id", "")) for pose in library
            if bool(pose.get("enabled", True)) and pose.get("motion") != "movej"
        ]
        if incompatible:
            raise CollectionConfigError(
                "enabled jitter requires MoveJ-capable poses: {}".format(
                    ",".join(incompatible)
                )
            )
        if not any(abs(value) > 0.0 for value in lower + upper):
            raise CollectionConfigError("enabled jitter has only zero bounds")
    elif mode == "bounded_random_jitter":
        raise CollectionConfigError("bounded_random_jitter mode is disabled")

    bin_config = config.get("apple_position_bins", {})
    if bool(bin_config.get("enabled", False)):
        for axis in ("x", "y", "z"):
            axis_cfg = bin_config.get(axis, {})
            boundaries = _finite_vector(
                axis_cfg.get("boundaries", []),
                len(axis_cfg.get("boundaries", [])),
                "apple_position_bins.{}.boundaries".format(axis),
            )
            if boundaries != sorted(boundaries) or len(set(boundaries)) != len(boundaries):
                raise CollectionConfigError(
                    "apple_position_bins.{} boundaries must be strictly increasing".format(axis)
                )
            labels = axis_cfg.get("labels", [])
            if len(labels) != len(boundaries) + 1:
                raise CollectionConfigError(
                    "apple_position_bins.{}.labels must have N+1 entries".format(axis)
                )

    return True


def load_manifest_counts(manifest_path, field):
    """Count non-empty values in one manifest column."""
    import csv

    counts = {}
    if not os.path.isfile(manifest_path):
        return counts
    with open(manifest_path, "r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            value = str(row.get(field, "")).strip()
            if value:
                counts[value] = counts.get(value, 0) + 1
    return counts


def select_least_sampled_pose(library, counts, seed, selection_index=0):
    """Choose among least-sampled enabled poses with deterministic tie breaking."""
    candidates = [pose for pose in library if bool(pose.get("enabled", True))]
    if not candidates:
        raise CollectionConfigError("no enabled initial pose")
    minimum = min(int(counts.get(str(pose["id"]), 0)) for pose in candidates)
    tied = [
        pose for pose in candidates
        if int(counts.get(str(pose["id"]), 0)) == minimum
    ]
    tied.sort(key=lambda pose: str(pose["id"]))
    rng = random.Random(int(seed) + int(selection_index))
    return dict(tied[rng.randrange(len(tied))])


def suggest_least_sampled_region(regions, counts):
    """Suggest an operator-configured apple region without moving anything."""
    names = sorted(str(region) for region in regions if str(region).strip())
    if not names:
        return None
    minimum = min(int(counts.get(name, 0)) for name in names)
    return next(name for name in names if int(counts.get(name, 0)) == minimum)


def generate_joint_jitter(config, seed, selection_index=0):
    """Generate deterministic configured offsets; never invent bounds."""
    initialization = config["initialization"]
    jitter = initialization.get("bounded_random_jitter", {})
    if not bool(jitter.get("enabled", False)):
        return [0.0] * 6
    lower = _finite_vector(jitter["joint_offset_min_deg"], 6, "jitter min")
    upper = _finite_vector(jitter["joint_offset_max_deg"], 6, "jitter max")
    rng = random.Random(int(seed) + int(selection_index))
    return [rng.uniform(low, high) for low, high in zip(lower, upper)]


def compute_position_bin(xyz, bin_config):
    """Bin xyz using operator-provided boundaries, or return DISABLED/UNKNOWN."""
    if not bool(bin_config.get("enabled", False)):
        return "DISABLED"
    try:
        point = _finite_vector(xyz, 3, "apple xyz")
    except CollectionConfigError:
        return "UNKNOWN"
    names = []
    for axis, value in zip(("x", "y", "z"), point):
        axis_cfg = bin_config.get(axis, {})
        boundaries = [float(item) for item in axis_cfg.get("boundaries", [])]
        labels = [str(item) for item in axis_cfg.get("labels", [])]
        if len(labels) != len(boundaries) + 1:
            raise CollectionConfigError(
                "apple_position_bins.{}.labels must have N+1 entries".format(axis)
            )
        position = 0
        while position < len(boundaries) and value >= boundaries[position]:
            position += 1
        names.append(labels[position])
    return "_".join(names)


def plan_to_json(pose, mode, seed, offset):
    """Return stable plan text for logs/tests."""
    return json.dumps({
        "initial_pose_id": str(pose["id"]),
        "initialization_mode": str(mode),
        "initialization_seed": int(seed),
        "initialization_offset": [float(value) for value in offset],
    }, sort_keys=True)
