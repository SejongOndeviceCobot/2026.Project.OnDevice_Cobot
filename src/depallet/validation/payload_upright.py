"""CPU certificates for nominal loaded-trajectory upright orientation.

The cuRobo non-terminal rotation criterion is an optimization cost. This module
separately certifies the exported, piecewise-linear joint-position curve using
URDF FK and a serial-revolute angular Lipschitz bound. It does not certify the
executed physical curve or remove the runtime attachment/tilt measurements.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from depallet.motion.curobo_bridge import pose, sha256, vector
from depallet.manipulation.payload_self_contact import attachment_uncertainty_policy

SCHEMA = "depallet.payload_upright.v1"


def upright_policy(value, *, world_up_base=None):
    if value is None:
        return None
    required = {"schema", "maximum_nominal_tilt_rad", "maximum_actual_tilt_rad"}
    if (not isinstance(value, dict) or not required <= set(value)
            or set(value)-required-{"world_up_base", "allow_transit_yaw"} or value.get("schema") != SCHEMA):
        raise ValueError("Invalid payload upright policy schema or fields")
    result = {"schema": SCHEMA}
    if 'allow_transit_yaw' in value:
        if type(value['allow_transit_yaw']) is not bool:raise ValueError('allow_transit_yaw must be boolean')
        result['allow_transit_yaw']=value['allow_transit_yaw']
    for key, upper in (("maximum_nominal_tilt_rad", .02), ("maximum_actual_tilt_rad", .03)):
        val = value[key]
        if (isinstance(val, bool) or not isinstance(val, (int, float))
                or not math.isfinite(val) or not .001 <= val <= upper):
            raise ValueError("Invalid bounded upright policy: "+key)
        result[key] = float(val)
    if result["maximum_actual_tilt_rad"] < result["maximum_nominal_tilt_rad"]:
        raise ValueError("Actual tilt allowance is below nominal allowance")
    up = world_up_base if world_up_base is not None else value.get("world_up_base")
    if up is not None:
        up = vector(list(up), 3, "world up in planning base")
        if abs(np.linalg.norm(up)-1.) > 1e-8:
            raise ValueError("World up must be a unit vector")
        if world_up_base is not None and "world_up_base" in value:
            if not np.allclose(up, value["world_up_base"], atol=1e-10, rtol=0.):
                raise ValueError("Policy world up differs from measured planning base")
        result["world_up_base"] = up
    return result


def validate_upright_request(request):
    policy = upright_policy(request.get("payload_orientation_policy"))
    if policy is None:
        return None
    if not request.get("payload") or "world_up_base" not in policy:
        raise ValueError("Upright transport requires a payload and measured world-up vector")
    uncertainty = attachment_uncertainty_policy(request.get("attachment_uncertainty"))
    if uncertainty is None:
        raise ValueError("Upright transport requires explicit attachment rotation uncertainty")
    if (policy["maximum_nominal_tilt_rad"]+uncertainty["rotation_rad"]
            > policy["maximum_actual_tilt_rad"]+1e-12):
        raise ValueError("Nominal plus elastic rotation exceeds actual tilt allowance")
    if any("joint_target_rad" in goal for goal in request["goals"]):
        raise ValueError("Upright transport requires Cartesian pose optimization")
    return policy


def orientation_criteria_kwargs(goal, policy):
    """Arguments for the official ToolPoseCriteria; no GPU imports here."""
    if policy is None or goal.get("linear_axis"):
        return None  # Existing linear_motion already tracks all three rotations.
    return dict(terminal_pose_axes_weight_factor=[1., 1., 1., 1., 1., 1.],
                non_terminal_pose_axes_weight_factor=[0., 0., 0., 1., 1., 0. if policy.get("allow_transit_yaw",False) else 1.])


class URDFOrientationModel:
    """Parsed once, independent of cuRobo and its kinematics tensors."""
    def __init__(self, urdf, joint_names, tip, base):
        self.names = list(joint_names)
        if len(self.names) != 6 or len(set(self.names)) != 6:
            raise ValueError("Expected six unique H2017 joints")
        tree = ET.parse(urdf).getroot()
        by_child = {j.find("child").get("link"): j for j in tree.findall("joint")}
        chain = []; link = tip
        while link != base:
            if link not in by_child or len(chain) > len(by_child):
                raise ValueError("No acyclic URDF chain to upright TCP")
            j = by_child[link]; chain.append(j); link = j.find("parent").get("link")
        self.chain = []; active = []
        for j in reversed(chain):
            origin = j.find("origin")
            rpy = [0., 0., 0.] if origin is None else [float(x) for x in origin.get("rpy", "0 0 0").split()]
            fixed = Rotation.from_euler("xyz", rpy).as_matrix()
            if j.get("type") == "fixed":
                self.chain.append((fixed, None, None)); continue
            if j.get("type") not in ("revolute", "continuous") or j.get("name") not in self.names:
                raise ValueError("Upright bound requires fully specified revolute URDF chain")
            index = self.names.index(j.get("name")); active.append(index)
            tag = j.find("axis")
            axis = np.array([float(x) for x in (tag.get("xyz", "1 0 0") if tag is not None else "1 0 0").split()])
            if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-12:
                raise ValueError("Invalid URDF rotation axis")
            self.chain.append((fixed, index, axis/np.linalg.norm(axis)))
        if sorted(active) != list(range(6)):
            raise ValueError("Upright bound requires each of the six joints exactly once")

    def rotation(self, q):
        r = np.eye(3)
        for fixed, index, axis in self.chain:
            r = r@fixed
            if index is not None:
                r = r@Rotation.from_rotvec(axis*q[index]).as_matrix()
        return r


def certify_linear_curve(positions, rotation_at, tcp_to_box_rotation, world_up_base,
                         maximum_tilt_rad, *, maximum_depth=12):
    """Fail closed unless every q-linear interval has a proved tilt upper bound.

    Along q(t)=(1-t)qa+t qb the SO(3) speed is at most sum(abs(qb-qa)).
    Tilt of any fixed box axis is 1-Lipschitz in that angular distance. The
    smaller of the two endpoint cones is at most (tilt_a+tilt_b+L)/2;
    adaptive bisection tightens it without assuming sampled FK is sufficient.
    """
    q = np.asarray(positions, dtype=float)
    if q.ndim != 2 or q.shape[1] != 6 or not 2 <= len(q) <= 32000 or not np.isfinite(q).all():
        raise ValueError("Invalid upright joint-position array")
    relative = np.asarray(tcp_to_box_rotation, float); up = np.asarray(world_up_base, float)
    if (relative.shape != (3, 3) or not np.isfinite(relative).all()
            or not np.allclose(relative.T@relative, np.eye(3), atol=1e-8, rtol=0.)
            or abs(np.linalg.det(relative)-1.) > 1e-8):
        raise ValueError("Invalid nominal TCP-to-payload rotation")
    if up.shape != (3,) or not np.isfinite(up).all() or abs(np.linalg.norm(up)-1.) > 1e-8:
        raise ValueError("Invalid world-up axis")
    if not math.isfinite(maximum_tilt_rad) or not 0 < maximum_tilt_rad <= .02:
        raise ValueError("Invalid maximum nominal tilt")
    report = dict(passed=False, original_samples=len(q), original_intervals=len(q)-1,
                  evaluated_fk_samples=0, adaptive_midpoints=0, certified_leaf_intervals=0,
                  maximum_observed_tilt_rad=0., maximum_certified_tilt_bound_rad=0.,
                  maximum_nominal_tilt_rad=float(maximum_tilt_rad),
                  interpolation="piecewise linear in the six URDF revolute joint positions",
                  bound_formula="max(tilt_a,tilt_b,(tilt_a+tilt_b+sum(abs(q_b-q_a)))/2)",
                  physical_execution_validated=False, physical_continuous_tilt_guarantee=False)

    def sample(row):
        r = np.asarray(rotation_at(row), float)
        if (r.shape != (3, 3) or not np.isfinite(r).all()
                or not np.allclose(r.T@r, np.eye(3), atol=1e-7, rtol=0.) or np.linalg.det(r) < .999999):
            raise ValueError("Independent FK returned an invalid rotation")
        axis = r@relative[:, 2]
        angle = math.atan2(float(np.linalg.norm(np.cross(up, axis))), float(np.dot(up, axis)))
        report["evaluated_fk_samples"] += 1
        report["maximum_observed_tilt_rad"] = max(report["maximum_observed_tilt_rad"], angle)
        return angle

    def interval(a, b, ta, tb, depth):
        if max(ta, tb) > maximum_tilt_rad:
            report["failure"] = "An independent FK sample exceeds the nominal upright limit"
            return False
        length = float(np.sum(np.abs(b-a)))
        bound = max(ta, tb, (ta+tb+length)/2.) + 1e-12
        if bound <= maximum_tilt_rad:
            report["certified_leaf_intervals"] += 1
            report["maximum_certified_tilt_bound_rad"] = max(report["maximum_certified_tilt_bound_rad"], bound)
            return True
        if depth >= maximum_depth:
            report["failure"] = "Upright interval could not be certified within subdivision bound"
            return False
        m = (a+b)/2.; tm = sample(m); report["adaptive_midpoints"] += 1
        return interval(a, m, ta, tm, depth+1) and interval(m, b, tm, tb, depth+1)

    tilts = [sample(row) for row in q]
    for i in range(len(q)-1):
        if not interval(q[i], q[i+1], tilts[i], tilts[i+1], 0):
            report["failure_interval_index"] = i
            return report
    report["passed"] = True
    return report


def certify_upright_trajectory(request, positions, *, robot_config=None):
    policy = validate_upright_request(request)
    if policy is None:
        raise ValueError("Upright certification requires explicit request policy")
    import yaml
    config_path = Path(request["robot_config"])
    cfg = yaml.safe_load(config_path.read_text()) if robot_config is None else robot_config
    kin = cfg.get("robot_cfg", cfg)["kinematics"]
    frames = {g["tcp_frame"] for g in request["goals"]}
    if len(frames) != 1:
        raise ValueError("Exactly one TCP frame is required")
    model = URDFOrientationModel(kin["urdf_path"], request["joint_names"], next(iter(frames)), kin["base_link"])
    initial_q = np.array(vector(request["start_position_rad"], 6, "upright initial q"))
    payload_pose = pose(request["payload"]["pose_base_wxyz"], "upright initial box pose")
    w, x, y, z = payload_pose[3:]
    box_rotation = Rotation.from_quat([x, y, z, w]).as_matrix()
    relative = model.rotation(initial_q).T@box_rotation
    array = np.asarray(positions, float)
    # Include the supplied initial state; a planner's first row is not trusted
    # to exactly equal the measured start. This boundary is also certified.
    checked = np.vstack((initial_q, array))
    report = certify_linear_curve(checked, model.rotation, relative, policy["world_up_base"],
                                  policy["maximum_nominal_tilt_rad"])
    report.update(schema="depallet.payload_upright.certificate.v1", policy=policy,
                  trajectory_samples=len(array), initial_state_boundary_included=True,
                  nominal_tcp_to_payload_rotation=relative.tolist(),
                  robot_config_sha256=sha256(config_path), urdf_sha256=sha256(kin["urdf_path"]),
                  checker_sha256=sha256(__file__),
                  request_content_sha256=hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
                  joint_positions_float64_sha256=hashlib.sha256(np.ascontiguousarray(array, dtype="<f8").tobytes()).hexdigest(),
                  actual_tilt_bound_requires_runtime_relative_rotation_and_world_tilt_checks=True,
                  collision_certificate=False)
    return report
