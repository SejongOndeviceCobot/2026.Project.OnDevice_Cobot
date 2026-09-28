"""CPU-only assumed extension of an open aggregate suction attachment anchor.

This computes a candidate USD localPos0; it neither updates a joint nor moves
an actor. Inputs are measured simulation-oracle poses, not visual estimates.
The native-error prediction assumes a zero clearanceOffset, identity localRot0,
and an unobstructed +flange-Z ray hitting the target's cuboid top face.

Native formula reference: NVIDIA IsaacSim v6.1.0, commit
7c206f75bdadd9e05fc457f19863ca4c3f0cb693, SurfaceGripperComponent.cpp:719-727.
It is a source-based prediction, not native joint telemetry or force prediction.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np


MAX_EXTENSION_M = .004
MAX_NORMAL_ANGLE_DEG = 3.


def _array(value, shape, label):
    try:
        raw = np.asarray(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label}: finite numeric array required") from exc
    if raw.shape != shape or raw.dtype.kind not in "fiu" or not np.isfinite(raw).all():
        raise ValueError(f"{label}: finite numeric shape {shape} required")
    return raw.astype(float, copy=True)


def _transform(value, label):
    result = _array(value, (4, 4), label)
    rotation = result[:3, :3]
    if (not np.allclose(result[3], [0., 0., 0., 1.], atol=1e-10, rtol=0.)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0.)
            or abs(float(np.linalg.det(rotation))-1.) > 1e-8):
        raise ValueError(f"{label}: proper SE3 without scaling or reflection required")
    return result


def _positive(value, label, maximum):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(float(value)) or not 0 < float(value) <= maximum):
        raise ValueError(f"{label}: finite value within (0, {maximum}] required")
    return float(value)


def project_contact_anchor(T_world_flange, T_world_box, nominal_anchor_flange_m,
                           box_dimensions_m, *, ray_clearance_m=.000025,
                           max_extension_m=.004):
    """Return an assumed +flange-Z anchor extension, or raise ValueError.

    The nominal origin must be 0..4mm above the box top. The +flange-Z ray
    must point into that face within 3deg and intersect within its footprint.
    Extension = measured ray distance - positive requested ray clearance,
    bounded by max_extension_m (itself at most 4mm). A negative extension,
    including an origin already closer than the requested clearance, fails.

    No robot/TCP/body/joint state is changed. The requested joint anchor may
    lie beyond the nominal pad surface; this is unmeasured pad compliance.
    Downstream code must validate the complete nominal pad patch, collision
    world, supported native joint update and actual attachment independently.
    """
    flange = _transform(T_world_flange, "T_world_flange")
    box = _transform(T_world_box, "T_world_box")
    nominal = _array(nominal_anchor_flange_m, (3,), "nominal_anchor_flange_m")
    dimensions = _array(box_dimensions_m, (3,), "box_dimensions_m")
    if np.any(dimensions <= 0):
        raise ValueError("box_dimensions_m: all dimensions must be positive")
    maximum = _positive(max_extension_m, "max_extension_m", MAX_EXTENSION_M)
    clearance = _positive(ray_clearance_m, "ray_clearance_m", maximum)

    direction = flange[:3, 2]
    normal = box[:3, 2]
    approach_cosine = -float(normal @ direction)
    if approach_cosine < math.cos(math.radians(MAX_NORMAL_ANGLE_DEG)):
        raise ValueError("Flange +Z ray must enter the box top within 3 degrees")
    angle = math.acos(min(1., max(-1., approach_cosine)))
    origin = flange[:3, :3] @ nominal + flange[:3, 3]
    origin_box = box[:3, :3].T @ (origin-box[:3, 3])
    normal_gap = float(origin_box[2]-dimensions[2]/2)
    if not 0. <= normal_gap <= MAX_EXTENSION_M:
        raise ValueError("Nominal anchor must be 0..4mm above the top face")
    distance = normal_gap / approach_cosine
    hit = origin + direction * distance
    hit_box = box[:3, :3].T @ (hit-box[:3, 3])
    if np.any(np.abs(hit_box[:2]) > dimensions[:2]/2):
        raise ValueError("Ray intersection lies outside the box top footprint")
    extension = distance-clearance
    if not 0. <= extension <= maximum:
        raise ValueError("Required anchor extension must be nonnegative and within max_extension_m")
    new_anchor = nominal + [0., 0., extension]
    new_origin = flange[:3, :3] @ new_anchor + flange[:3, 3]

    def mismatch(ray_distance):
        # Native actor1 anchor is translated from actor0 by -direction*distance
        # when isaac:clearanceOffset=0. No native data is queried here.
        return {
            "actor1_minus_actor0_world_m": (-direction*ray_distance).tolist(),
            "actor1_minus_actor0_joint0_m": [0., 0., -float(ray_distance)],
            "magnitude_m": float(ray_distance),
            "source": "official_v6.1_formula_prediction_not_native_telemetry",
        }

    return {
        "schema": "depallet.projected_contact_anchor.v1",
        "passed": True,
        "nominal_anchor_flange_m": nominal.tolist(),
        "new_anchor_flange_m": new_anchor.tolist(),
        "nominal_anchor_world_m": origin.tolist(),
        "new_anchor_world_m": new_origin.tolist(),
        "target_top_intersection_world_m": hit.tolist(),
        "target_top_intersection_box_m": hit_box.tolist(),
        "box_dimensions_m": dimensions.tolist(),
        "suction_direction_world": direction.tolist(),
        "suction_normal_angle_deg": math.degrees(angle),
        "measured_normal_gap_m": normal_gap,
        "measured_ray_distance_m": distance,
        "ray_clearance_m": clearance,
        "predicted_remaining_normal_gap_m": clearance*approach_cosine,
        "required_extension_m": extension,
        "maximum_extension_m": maximum,
        "before_native_actor1_mismatch": mismatch(distance),
        "after_native_actor1_mismatch": mismatch(clearance),
        "native_prediction_assumptions": {
            "isaac_clearanceOffset_m": 0.,
            "joint_localRot0_identity": True,
            "unobstructed_ray_hits_target_top": True,
            "native_localPos0_update_took_effect": "required_but_not_verified",
            "official_isaacsim_version": "6.1.0",
            "official_commit": "7c206f75bdadd9e05fc457f19863ca4c3f0cb693",
        },
        "measurement_source": "simulation_oracle",
        "measurement_oracle": True,
        "nominal_tcp_unchanged": True,
        "nominal_pad_geometry_unchanged": True,
        "actor_pose_changes": False,
        "joint_state_changed": False,
        "assumed_pad_extension_not_measured": True,
        "physical_parameters_measured": False,
        "native_anchors_measured": False,
        "full_pad_patch_verified": False,
        "world_collision_verified": False,
        "execution_authorized": False,
        "physical_execution_validated": False,
        "gpu_used": False,
        "scope": "candidate aggregate D6 anchor only; no pose correction or command",
    }
