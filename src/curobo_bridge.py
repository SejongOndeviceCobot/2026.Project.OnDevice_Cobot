"""CPU-only serialization checks for the isolated cuRobo V2 planner.

Poses use meters and wxyz quaternions relative to the robot base. Joint
positions use radians. This module never imports GPU libraries.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from motion_profiles import LIMIT_FIELDS, resolve_motion_profile

REQUEST_SCHEMA = "depallet.curobo.v2.request.v1"
RESULT_SCHEMA = "depallet.curobo.v2.result.v1"
DEFAULT_PLANNER_RANDOM_SEED = 42
MAX_PLANNER_RANDOM_SEED = 2**31 - 1


def planner_random_seed(request):
    """Resolve a bounded deterministic planner seed without mutating the request."""
    if not isinstance(request, dict):
        raise ValueError("Planner request must be an object")
    seed = request.get("planner_random_seed", DEFAULT_PLANNER_RANDOM_SEED)
    if type(seed) is not int or not 0 <= seed <= MAX_PLANNER_RANDOM_SEED:
        raise ValueError(
            f"planner_random_seed must be an integer in [0,{MAX_PLANNER_RANDOM_SEED}]")
    return seed


def request_motion_limits(request):
    """Resolve named limits, retaining the original caps for legacy requests."""
    if not isinstance(request, dict):
        raise ValueError("Planner request must be an object")
    if "motion_profile" not in request:
        defaults = resolve_motion_profile("baseline")
        result = {field: request.get(field, defaults[field]) for field in LIMIT_FIELDS}
        result.update(motion_profile=None, legacy_request=True)
        return result
    profile = resolve_motion_profile(request["motion_profile"])
    for field in LIMIT_FIELDS:
        value = request.get(field)
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or float(value) != profile[field]):
            raise ValueError("Named motion profile requires exact matching limits")
    return {**profile, "legacy_request": False}


def validate_result_motion_limits(request, result):
    """Bind a named request to the worker's runtime-limit and phase receipts."""
    limits = request_motion_limits(request)
    if limits["legacy_request"]:
        return {**limits, "named_result_receipt_required": False}
    runtime = result.get("runtime_joint_limits") if isinstance(result, dict) else None
    if not isinstance(runtime, dict) or runtime.get("passed") is not True:
        raise ValueError("Named motion profile requires a passing runtime-limit receipt")
    expected, actual = runtime.get("expected"), runtime.get("actual")
    if not isinstance(expected, dict) or not isinstance(actual, dict):
        raise ValueError("Named motion profile runtime-limit receipt is incomplete")
    for key, field in (("velocity", "maximum_velocity_rad_s"),
                       ("acceleration", "maximum_acceleration_rad_s2"),
                       ("jerk", "maximum_jerk_rad_s3")):
        expected_values = vector(expected.get(key), 6, "expected runtime "+key)
        actual_values = vector(actual.get(key), 6, "actual runtime "+key)
        if any(abs(a-b) > 1e-6 for a, b in zip(actual_values, expected_values)):
            raise ValueError("Worker runtime limits differ from its expected limits")
        cap = limits[field]
        if key == "velocity":
            if any(not 0 < value <= cap+1e-9 for value in expected_values):
                raise ValueError("Runtime velocity limits exceed the named motion profile")
        elif any(abs(value-cap) > 1e-6 for value in expected_values):
            raise ValueError("Runtime acceleration/jerk limits differ from the named motion profile")
    phases = result.get("phases")
    if not isinstance(phases, list) or not phases:
        raise ValueError("Named motion profile requires per-phase trajectory limit receipts")
    for phase in phases:
        if not isinstance(phase, dict):
            raise ValueError("Invalid trajectory phase receipt")
        for key, field in (("maximum_velocity_rad_s", "maximum_velocity_rad_s"),
                           ("maximum_finite_difference_velocity_rad_s", "maximum_velocity_rad_s"),
                           ("maximum_acceleration_rad_s2", "maximum_acceleration_rad_s2")):
            value = phase.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value > limits[field]+1e-5):
                raise ValueError("Trajectory phase exceeds the named motion profile")
    return {**limits, "named_result_receipt_required": True}


def vector(value, length, label):
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{label}: expected {length} numbers")
    if any(isinstance(x, bool) or not isinstance(x, (int, float))
           or not math.isfinite(x) for x in value):
        raise ValueError(f"{label}: non-finite or non-numeric value")
    return [float(x) for x in value]


def pose(value, label):
    result = vector(value, 7, label)
    if abs(math.sqrt(sum(x*x for x in result[3:])) - 1) > 1e-4:
        raise ValueError(f"{label}: quaternion must be normalized")
    if max(abs(x) for x in result[:3]) > 10:
        raise ValueError(f"{label}: outside 10 m workcell bound")
    return result


def validate_request(request):
    if not isinstance(request, dict) or request.get("schema") != REQUEST_SCHEMA:
        raise ValueError("Unknown request schema")
    if request.get("length_unit", "m") != "m":
        raise ValueError("Planner positions require meters")
    if request.get("quaternion_order", "wxyz") != "wxyz":
        raise ValueError("Planner quaternion order must be wxyz")
    config = request.get("robot_config")
    if not isinstance(config, str) or not Path(config).is_absolute():
        raise ValueError("robot_config must be an absolute YAML path")
    names = request.get("joint_names")
    if (not isinstance(names, list) or len(names) != 6
            or any(not isinstance(x, str) or not x for x in names)
            or len(set(names)) != 6):
        raise ValueError("Six unique H2017 joint names are required")
    vector(request.get("start_position_rad"), 6, "start_position_rad")
    vector(request.get("start_velocity_rad_s", [0.] * 6), 6, "start_velocity_rad_s")
    dt = request.get("interpolation_dt_s", 1/60)
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not 1/240 <= dt <= .05:
        raise ValueError("interpolation_dt_s outside [1/240,0.05]")
    motion_limits = request_motion_limits(request)
    velocity_upper = (.25 if motion_limits["legacy_request"]
                      else motion_limits["maximum_velocity_rad_s"])
    for name, lower, upper in (
            ("position_tolerance_m", .0001, .005),
            ("endpoint_position_tolerance_m", .0001, .006),
            ("maximum_velocity_rad_s", .01, velocity_upper),
            ("maximum_acceleration_rad_s2", .01, .5),
            ("maximum_jerk_rad_s3", .1, 5.),
            ("maximum_trajectory_dt_s", .002, .4)):
        if name in request:
            value = request[name]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not lower <= value <= upper):
                raise ValueError(f"{name} outside [{lower},{upper}]")
    if "upright_seed_search" in request and type(request["upright_seed_search"]) is not bool:
        raise ValueError("upright_seed_search must be boolean")
    if request.get("upright_seed_search") and (not request.get("payload") or not request.get("payload_orientation_policy")):
        raise ValueError("upright_seed_search requires measured payload and upright policy")
    planner_random_seed(request)
    seed_count = request.get("num_trajopt_seeds", 4)
    if type(seed_count) is not int or seed_count not in (4, 8, 16, 32):
        raise ValueError("num_trajopt_seeds must be one of 4,8,16,32")
    goals = request.get("goals")
    if not isinstance(goals, list) or not 1 <= len(goals) <= 16:
        raise ValueError("One to sixteen goals required")
    ids = []
    for goal in goals:
        if not isinstance(goal, dict) or not isinstance(goal.get("id"), str):
            raise ValueError("Each goal requires a string id")
        ids.append(goal["id"])
        if "joint_target_rad" in goal:
            vector(goal["joint_target_rad"],6,"joint target")
            if goal.get("linear_axis") is not None:
                raise ValueError("Joint target cannot replace a Cartesian linear constraint")
        if not isinstance(goal.get("tcp_frame"), str) or not goal["tcp_frame"]:
            raise ValueError("Each goal requires tcp_frame")
        position = vector(goal.get("position_m"), 3, "goal position")
        quat = vector(goal.get("quaternion_wxyz"), 4, "goal quaternion")
        pose(position + quat, "goal pose")
        if goal.get("linear_axis") not in (None, "x", "y", "z"):
            raise ValueError("linear_axis must be x, y, z or null")
        if not isinstance(goal.get("linear_in_tool_frame", True), bool):
            raise ValueError("linear_in_tool_frame must be boolean")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate goal id")
    scene = request.get("scene")
    if not isinstance(scene, dict) or set(scene) != {"cuboid"}:
        raise ValueError("This worker accepts cuboid scenes only")
    boxes = scene["cuboid"]
    if not isinstance(boxes, dict) or not 1 <= len(boxes) <= 128:
        raise ValueError("One to 128 scene cuboids required")
    for name, box in boxes.items():
        if not isinstance(name, str) or not isinstance(box, dict):
            raise ValueError("Invalid scene cuboid")
        dims = vector(box.get("dims"), 3, name + " dimensions")
        if not all(.001 <= x <= 10 for x in dims):
            raise ValueError("Cuboid dimensions outside [.001,10] meters")
        pose(box.get("pose"), name + " pose")
    payload = request.get("payload")
    if payload is not None:
        if not isinstance(payload, dict) or payload.get("box_id") not in boxes:
            raise ValueError("Payload must name an existing scene box")
        hypothetical = payload.get("grasp_confirmed") is False
        hypothesis = payload.get("attachment_hypothesis")
        diagnostic_hypothesis = bool(
            hypothetical
            and request.get("diagnostic_only") is True
            and request.get("robot_execution_authorized") is False
            and request.get("physical_execution_validated") is False
            and isinstance(hypothesis, dict)
            and hypothesis.get("status") == "hypothetical_unverified"
            and hypothesis.get("grasp_evidence_available") is False
            and set(hypothesis) == {
                "status", "grasp_evidence_available", "source_receipt_sha256"})
        if payload.get("grasp_confirmed") is not True and not diagnostic_hypothesis:
            raise ValueError("Payload requires a confirmed physical grasp or an exact diagnostic hypothesis")
        if request.get("diagnostic_only") is True and not diagnostic_hypothesis:
            raise ValueError("Diagnostic payload planning may not claim a confirmed grasp")
        dims = vector(payload.get("dimensions_m"), 3, "payload dimensions")
        if dims != boxes[payload["box_id"]]["dims"]:
            raise ValueError("Payload/scene dimensions differ")
        payload_pose = pose(payload.get("pose_base_wxyz"), "payload pose")
        scene_pose = boxes[payload["box_id"]]["pose"]
        position_difference, angle_difference = pose_error(
            payload_pose[:3], payload_pose[3:], scene_pose[:3], scene_pose[3:])
        if position_difference > 1e-6 or angle_difference > 1e-6:
            raise ValueError("Payload pose differs from the current scene box pose")
        mass = payload.get("mass_kg")
        if isinstance(mass, bool) or not isinstance(mass, (int, float)) or not 0 < mass <= 20:
            raise ValueError("Payload mass outside (0,20] kg")
        count = payload.get("num_spheres", 64)
        if isinstance(count, bool) or not isinstance(count, int) or not 4 <= count <= 512:
            raise ValueError("Payload sphere budget outside [4,512]")
        from payload_cover_profiles import payload_cover_cells
        payload_cover_cells(payload)
    if request.get("disable_collision_links"):
        raise ValueError("Whole-link collision disabling is forbidden")
    return request


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pose_error(actual_position, actual_quaternion, target_position, target_quaternion):
    a = vector(list(actual_position), 3, "FK position")
    q = vector(list(actual_quaternion), 4, "FK quaternion")
    b = vector(list(target_position), 3, "target position")
    r = vector(list(target_quaternion), 4, "target quaternion")
    norm = math.sqrt(sum(x*x for x in q) * sum(x*x for x in r))
    if norm < 1e-9:
        raise ValueError("Zero quaternion")
    dot = min(1., abs(sum(x*y for x,y in zip(q,r))) / norm)
    return math.dist(a,b), 2 * math.acos(dot)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
