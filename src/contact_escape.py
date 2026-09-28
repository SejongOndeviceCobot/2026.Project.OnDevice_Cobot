"""CPU-only, bounded vertical departure from one measured support contact.

This is an explicit contact-aware simulation diagnostic, not a cuRobo plan.
The full enclosing payload is retained. Only its one support/world pair uses
exact box SAT during departure; every other sphere pair remains enabled.
Runtime must verify the same rigid attachment, stopped start and current scene.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import yaml

from curobo_bridge import sha256, validate_request, write_json
from payload_self_contact import derive_payload_config, obb_separation, payload_cover_for_request

SCHEMA = "depallet.contact_escape.result.v1"


def pose_matrix(pose):
    pose = np.asarray(pose, float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Invalid pose")
    if abs(np.linalg.norm(pose[3:])-1) > 1e-4:
        raise ValueError("Non-unit quaternion")
    result = np.eye(4)
    result[:3, 3] = pose[:3]
    result[:3, :3] = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    return result


def matrix_pose(t):
    xyzw = Rotation.from_matrix(t[:3, :3]).as_quat()
    return [*t[:3, 3].tolist(), float(xyzw[3]), *xyzw[:3].tolist()]


class RobotModel:
    """Parse reviewed URDF once, then perform independent NumPy/SciPy FK."""
    def __init__(self, kin, names):
        self.kin, self.names = kin, names
        self.base = kin["base_link"]
        tree = ET.parse(kin["urdf_path"]).getroot()
        self.joints = []
        self.lower, self.upper, self.velocity = [], [], []
        by_name = {j.get("name"): j for j in tree.findall("joint")}
        for name in names:
            joint = by_name[name]
            if joint.get("type") != "revolute":
                raise ValueError("Bounded revolute H2017 joints required")
            limit = joint.find("limit")
            self.lower.append(float(limit.get("lower")))
            self.upper.append(float(limit.get("upper")))
            self.velocity.append(float(limit.get("velocity")))
        pending = list(tree.findall("joint"))
        known = {self.base}
        while pending:
            ready = [j for j in pending if j.find("parent").get("link") in known]
            if not ready:
                break  # Ancestors above the chosen robot base are intentionally absent.
            for j in ready:
                parent, child = j.find("parent").get("link"), j.find("child").get("link")
                origin = j.find("origin")
                xyz = np.array([float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()])
                rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
                t = np.eye(4); t[:3, 3] = xyz
                t[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
                kind = j.get("type")
                idx = None if kind == "fixed" else names.index(j.get("name"))
                axis = j.find("axis")
                axis = np.array([float(v) for v in (axis.get("xyz") if axis is not None else "1 0 0").split()])
                axis /= np.linalg.norm(axis)
                self.joints.append((parent, child, t, idx, axis))
                known.add(child); pending.remove(j)
        if "suction_tcp" not in known:
            raise ValueError("Missing suction_tcp chain")
        self.lower, self.upper = np.asarray(self.lower), np.asarray(self.upper)
        self.velocity = np.asarray(self.velocity)
        # For any link point, this radius bounds displacement under rotation
        # about every ancestor joint. Triangle inequality makes it conservative.
        self.origin_reach = sum(np.linalg.norm(j[2][:3, 3]) for j in self.joints)
        self.ancestors = {self.base: []}
        for parent, child, _, idx, _ in self.joints:
            self.ancestors[child] = self.ancestors[parent] + ([] if idx is None else [idx])

    def transforms(self, q):
        q = np.asarray(q, float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("Invalid joint position")
        transforms = {self.base: np.eye(4)}
        for parent, child, origin, idx, axis in self.joints:
            t = transforms[parent] @ origin
            if idx is not None:
                t = t.copy()
                t[:3, :3] = t[:3, :3] @ Rotation.from_rotvec(axis*q[idx]).as_matrix()
            transforms[child] = t
        return transforms


def sphere_box_clearance(centers, radii, center, axes, half):
    local = (centers-center) @ axes
    delta = np.abs(local)-half
    signed = np.linalg.norm(np.maximum(delta, 0), axis=1) + np.minimum(np.max(delta, axis=1), 0)
    return signed-radii


class CollisionValidator:
    def __init__(self, request, robot, support_box_id, *, numerical_support_overlap_m=0.):
        if isinstance(numerical_support_overlap_m, bool) or not math.isfinite(numerical_support_overlap_m) or not 0 <= numerical_support_overlap_m <= .0001:
            raise ValueError("Numerical support overlap allowance must be within 0–0.0001m")
        self.numerical_support_overlap_m = float(numerical_support_overlap_m)
        policy = request.get("contact_escape_policy", {})
        if not isinstance(policy, dict):
            raise ValueError("Contact escape policy must be an object")
        declared = policy.get("numerical_support_overlap_m", 0.)
        if isinstance(declared, bool) or declared != self.numerical_support_overlap_m:
            raise ValueError("Numerical support overlap allowance must match explicit request policy")
        validate_request(request)
        self.request = request
        self.payload = request.get("payload")
        if not self.payload or self.payload.get("grasp_confirmed") is not True:
            raise ValueError("Confirmed measured attachment required")
        self.box_id = self.payload["box_id"]
        self.support = support_box_id
        if support_box_id == self.box_id or support_box_id not in request["scene"]["cuboid"]:
            raise ValueError("One distinct existing support is required")
        effective, self.self_proof = derive_payload_config(robot, request)
        if not self.self_proof["safe_to_plan"]:
            raise ValueError("Actual gripper/payload OBB overlap")
        self.kin = effective.get("robot_cfg", effective)["kinematics"]
        self.model = RobotModel(self.kin, request["joint_names"])
        self.q0 = np.asarray(request["start_position_rad"], float)
        self.tcp0 = self.model.transforms(self.q0)["suction_tcp"]
        self.box0 = pose_matrix(self.payload["pose_base_wxyz"])
        self.relative = np.linalg.inv(self.tcp0) @ self.box0
        self.half = np.asarray(self.payload["dimensions_m"])/2
        spheres, self.coverage, self.attachment_uncertainty = payload_cover_for_request(request)
        from payload_cover_profiles import resolve_payload_cover_profile
        expected_count=resolve_payload_cover_profile(self.payload.get("cover_profile","legacy64"))["sphere_count"]
        if len(spheres) != expected_count:
            raise ValueError("Reviewed contact departure requires every declared profile sphere")
        self.pc = np.array([s["center"] for s in spheres]); self.pr = np.array([s["radius"] for s in spheres])
        self.world = {}
        for name, box in request["scene"]["cuboid"].items():
            if name == self.box_id:
                continue  # This physical object is present exactly once, as payload.
            t = pose_matrix(box["pose"])
            self.world[name] = (t[:3, 3], t[:3, :3], np.array(box["dims"])/2)
        self.links = {}
        for link in self.kin["collision_link_names"]:
            if link == "attached_object":
                continue
            spheres = self.kin["collision_spheres"].get(link)
            if not spheres:
                raise ValueError(f"Missing collision spheres on {link}")
            self.links[link] = (np.array([s["center"] for s in spheres]), np.array([s["radius"] for s in spheres]))
        self.ignore = self.kin.get("self_collision_ignore", {})
        if float(self.kin.get("collision_sphere_buffer", 0)) != 0:
            raise ValueError("Nonzero collision sphere buffer requires explicit handling")
        if any(float(v) != 0 for v in self.kin.get("self_collision_buffer", {}).values()):
            raise ValueError("Nonzero self buffers require explicit handling")
        self.up = pose_matrix([0, 0, 0, *request.get("base_world_quaternion_wxyz", [1, 0, 0, 0])])[:3, :3].T @ [0, 0, 1]
        sc, sr, sh = self.world[self.support]
        initial = obb_separation(self.box0[:3, 3], self.box0[:3, :3], self.half, sc, sr, sh)
        initial_gap = initial["separating_gap_lower_bound_m"]
        if (initial_gap > .005 or initial_gap < -self.numerical_support_overlap_m
                or (self.numerical_support_overlap_m == 0 and not initial["nonoverlap"])):
            raise ValueError("Support must start nonpenetrating within 5mm unless an explicit bounded numerical overlap allowance covers its initial penetration")
        axis = np.asarray(initial["separating_axis_base"])
        if np.dot(self.box0[:3, 3]-sc, axis) < 0:
            axis = -axis
        if np.dot(axis, self.up) < .98 or np.dot(self.box0[:3, 3]-sc, self.up) <= 0:
            raise ValueError("Support must be below payload with nearly vertical separating normal")
        self.support_axis = axis
        self.initial_support_gap = self.support_gap(self.box0)
        self.support_gap_floor = min(0., self.initial_support_gap)
        self.radius_bound = self.model.origin_reach + max(np.linalg.norm(c, axis=1).max() for c, _ in self.links.values()) + np.linalg.norm(self.relative[:3, 3]) + np.linalg.norm(self.half)

    def ignored(self, a, b):
        return b in self.ignore.get(a, []) or a in self.ignore.get(b, [])

    def support_gap(self, box):
        sc, sr, sh = self.world[self.support]; axis = self.support_axis
        return float(np.dot(box[:3, 3]-sc, axis)-np.dot(self.half, np.abs(box[:3, :3].T@axis))-np.dot(sh, np.abs(sr.T@axis)))

    def state(self, q, endpoint=False):
        q = np.asarray(q)
        if np.any(q < self.model.lower) or np.any(q > self.model.upper):
            raise ValueError("Joint position limit violation")
        ts = self.model.transforms(q); tcp = ts["suction_tcp"]; box = tcp @ self.relative
        pc = self.pc @ box[:3, :3].T + box[:3, 3]
        world_min, self_min, payload_min = {}, {}, {}
        exact_min = math.inf
        centers = {name: c @ ts[name][:3, :3].T + ts[name][:3, 3] for name, (c, _) in self.links.items()}
        for name, (c, rot, half) in self.world.items():
            exact = obb_separation(box[:3, 3], box[:3, :3], self.half, c, rot, half)
            gap = exact["separating_gap_lower_bound_m"]
            if name == self.support:
                support_gap = self.support_gap(box)
                if support_gap < self.support_gap_floor-1e-10:
                    raise ValueError("Support penetration increased below its initial signed gap")
                if endpoint and support_gap <= 0.:
                    raise ValueError("Endpoint requires strictly positive support separation")
            if gap < -1e-10 and (name != self.support or self.support_gap_floor == 0.):
                raise ValueError(f"Actual payload OBB penetrates {name}: {gap:.9g}m")
            exact_min = min(exact_min, gap)
            clear = float(sphere_box_clearance(pc, self.pr, c, rot, half).min())
            if clear < -1e-10 and (name != self.support or endpoint):
                raise ValueError(f"Payload sphere/world collision with {name}: {clear:.9g}m")
            payload_min[name] = clear
            for link, (_, radii) in self.links.items():
                clear = float(sphere_box_clearance(centers[link], radii, c, rot, half).min())
                world_min[link] = min(world_min.get(link, math.inf), clear)
                if clear < -1e-10:
                    raise ValueError(f"Robot/world collision {link}/{name}: {clear:.9g}m")
        names = list(self.links)
        for i, a in enumerate(names):
            ac, ar = centers[a], self.links[a][1]
            for b in names[i+1:] + ["attached_object"]:
                if self.ignored(a, b):
                    continue
                bc, br = (pc, self.pr) if b == "attached_object" else (centers[b], self.links[b][1])
                clear = float((np.linalg.norm(ac[:, None]-bc[None, :], axis=2)-ar[:, None]-br[None, :]).min())
                self_min[(a, b)] = clear
                if clear < -1e-10:
                    raise ValueError(f"Robot self collision {a}/{b}: {clear:.9g}m")
        return {"tcp": tcp, "box": box, "world": world_min, "self": self_min,
                "payload": payload_min, "support_gap": self.support_gap(box), "exact_world_min": exact_min}

    def support_monotonic_interval(self, qa, qb):
        """Prove every payload vertex moves away from the support plane.

        At interval midpoint, project each vertex's geometric Jacobian on the
        fixed separating normal. A 2*radius*sum(abs(dq))**2 second-derivative
        bound encloses its derivative throughout the normalized interval.
        The minimum of eight nondecreasing projections also cannot decrease.
        """
        dq=np.asarray(qb)-qa; qm=(np.asarray(qa)+qb)/2
        transforms={self.model.base:np.eye(4)}; axes={}; origins={}
        for parent,child,origin,idx,axis in self.model.joints:
            t=transforms[parent]@origin
            if idx is not None:
                axes[idx]=t[:3,:3]@axis; origins[idx]=t[:3,3].copy()
                t=t.copy();t[:3,:3]=t[:3,:3]@Rotation.from_rotvec(axis*qm[idx]).as_matrix()
            transforms[child]=t
        box=transforms["suction_tcp"]@self.relative
        signs=np.array([[a,b,c] for a in (-1,1) for b in (-1,1) for c in (-1,1)])
        points=(signs*self.half)@box[:3,:3].T+box[:3,3]
        derivative=np.zeros(8)
        for idx in self.model.ancestors["suction_tcp"]:
            derivative+=np.cross(axes[idx],points-origins[idx])@self.support_axis*dq[idx]
        bound=self.radius_bound*float(np.abs(dq).sum())**2
        return float(derivative.min())-bound>=-1e-12

    def interval_safe(self, state, q_abs_variation, *, support_monotonic=False):
        """Lipschitz enclosure of every point on a joint-linear interval.

        Sum of joint variations times a conservative chain radius bounds sphere
        center motion and every payload point. Negative results demand bisection.
        """
        move = {link: self.radius_bound*sum(q_abs_variation[i] for i in self.model.ancestors[link]) for link in self.links}
        move["attached_object"] = self.radius_bound*sum(q_abs_variation)
        margins = [clear-move[link] for link, clear in state["world"].items()]
        margins += [clear-move[a]-move[b] for (a, b), clear in state["self"].items()]
        margins += [clear-move["attached_object"] for name, clear in state["payload"].items() if name != self.support]
        # This fixed support separating plane remains valid for the entire OBB.
        if support_monotonic and self.support_gap_floor < 0.:
            # The separately proved 8-vertex monotonic interval cannot deepen
            # the one measured numerical contact. All other swept bounds stay.
            margins.append(state["support_gap"]-self.support_gap_floor)
        else:
            margins.append(state["support_gap"]-move["attached_object"])
        return min(margins) >= -1e-12


def _smooth_trajectory(knots, positions, duration, dt):
    spline = CubicSpline(knots, positions, axis=0)
    n = int(math.ceil(duration/dt)); duration = n*dt
    t = np.arange(n+1)*dt; u = t/duration
    s = 10*u**3-15*u**4+6*u**5
    ds = (30*u**2-60*u**3+30*u**4)/duration
    dds = (60*u-180*u**2+120*u**3)/duration**2
    ddds = (60-360*u+360*u**2)/duration**3
    q = spline(s); qs = spline(s, 1); qss = spline(s, 2); qsss = spline(s, 3)
    qd = qs*ds[:, None]
    qdd = qss*ds[:, None]**2+qs*dds[:, None]
    qddd = qsss*ds[:, None]**3+3*qss*ds[:, None]*dds[:, None]+qs*ddds[:, None]
    return {"position_rad": q, "velocity_rad_s": qd, "acceleration_rad_s2": qdd,
            "jerk_rad_s3": qddd, "time_s": t}, s


def verify_limits(arrays, lower, upper, velocity=.10, acceleration=.20, jerk=2.):
    q, qd, qdd, qddd, t = [arrays[k] for k in ("position_rad", "velocity_rad_s", "acceleration_rad_s2", "jerk_rad_s3", "time_s")]
    if any(not np.isfinite(x).all() for x in (q, qd, qdd, qddd, t)):
        raise ValueError("Nonfinite trajectory")
    if q.shape != qd.shape or q.shape != qdd.shape or q.shape != qddd.shape or q.shape[1:] != (6,) or len(t) != len(q):
        raise ValueError("Invalid trajectory array shape")
    if np.any(np.diff(t) <= 0) or t[0] != 0:
        raise ValueError("Invalid trajectory times")
    if np.any(q < lower) or np.any(q > upper):
        raise ValueError("Joint position limit violation")
    fd = np.diff(q, axis=0)/np.diff(t)[:, None]
    peaks = [float(np.abs(a).max()) for a in (qd, qdd, qddd, fd)]
    if any(p > cap+1e-9 for p, cap in zip(peaks, [velocity, acceleration, jerk, velocity])):
        raise ValueError(f"Trajectory velocity/acceleration/jerk limits exceeded: {peaks}")
    if max(np.abs(qd[[0, -1]]).max(), np.abs(qdd[[0, -1]]).max()) > 1e-8:
        raise ValueError("Trajectory must start and end at rest")
    return dict(zip(["velocity_rad_s", "acceleration_rad_s2", "jerk_rad_s3", "finite_difference_velocity_rad_s"], peaks))


def generate_escape(request, support_box_id, lift_m=.08, *, historical_diagnostic=False, numerical_support_overlap_m=0.):
    """Return {trajectory, certificate, endpoint_request}; never start execution.

    Arrays use worker-compatible position_rad, velocity_rad_s,
    acceleration_rad_s2, time_s, plus jerk_rad_s3. Runtime must follow the
    validated joint-linear position samples, monitor attachment and separation,
    then remeasure/replan the full cuRobo transport at the attained endpoint.
    """
    validate_request(request)
    if not math.isfinite(lift_m) or not .06 <= lift_m <= .10:
        raise ValueError("Contact departure lift must be within 6–10cm")
    stopped = max(abs(v) for v in request.get("start_velocity_rad_s", [0.]*6)) <= .01
    if not stopped and not historical_diagnostic:
        raise ValueError("Remeasure after a stable hold: start joint speed exceeds .01rad/s")
    robot = yaml.safe_load(Path(request["robot_config"]).read_text())
    check = CollisionValidator(request, robot, support_box_id, numerical_support_overlap_m=numerical_support_overlap_m)
    start = check.state(check.q0)
    knots = np.linspace(0, 1, int(math.ceil(lift_m/.002))+1)
    positions = [check.q0]
    max_ik_position, max_ik_rotation = 0., 0.
    for s in knots[1:]:
        target = check.tcp0.copy(); target[:3, 3] += check.up*lift_m*s
        def residual(q):
            tcp = check.model.transforms(q)["suction_tcp"]
            return np.r_[tcp[:3, 3]-target[:3, 3], Rotation.from_matrix(target[:3, :3].T@tcp[:3, :3]).as_rotvec()]
        result = least_squares(residual, positions[-1], bounds=(check.model.lower, check.model.upper),
                               max_nfev=100, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        error = residual(result.x); pe = np.linalg.norm(error[:3]); re = np.linalg.norm(error[3:])
        if pe > 1e-6 or re > 1e-6 or max(abs(result.x-positions[-1])) > .04:
            raise ValueError(f"Continuous seeded IK failed at lift {s*lift_m:.4f}m: {pe:.6g}m/{re:.6g}rad")
        positions.append(result.x)
        max_ik_position=max(max_ik_position,float(pe)); max_ik_rotation=max(max_ik_rotation,float(re))
    dt = request.get("interpolation_dt_s", 1/60)
    duration = 6.
    for _ in range(6):
        arrays, path_fraction = _smooth_trajectory(knots, np.array(positions), duration, dt)
        try:
            peaks = verify_limits(arrays, check.model.lower, check.model.upper)
            break
        except ValueError:
            duration *= 1.3
    else:
        raise ValueError("No bounded time scaling under 23s")
    min_support, max_line_error, max_angle = math.inf, 0., 0.
    min_arm, min_self, min_other_payload, min_exact = math.inf, math.inf, math.inf, math.inf
    states=[]
    previous_gap = None
    for i, q in enumerate(arrays["position_rad"]):
        state = check.state(q, endpoint=(i==len(arrays["position_rad"])-1)); states.append(state)
        desired = check.tcp0[:3, 3]+check.up*lift_m*path_fraction[i]
        line_error = float(np.linalg.norm(state["tcp"][:3, 3]-desired))
        angle = float(Rotation.from_matrix(check.tcp0[:3, :3].T@state["tcp"][:3, :3]).magnitude())
        if line_error > .00002 or angle > .0001:
            raise ValueError("Joint interpolation departs from the vertical fixed-orientation path")
        gap=state["support_gap"]
        if previous_gap is not None and gap < previous_gap-1e-9:
            raise ValueError("Support separation must increase monotonically")
        previous_gap=gap
        min_support=min(min_support,gap); min_arm=min(min_arm,min(state["world"].values()))
        min_self=min(min_self,min(state["self"].values())); min_exact=min(min_exact,state["exact_world_min"])
        min_other_payload=min(min_other_payload,min(v for k,v in state["payload"].items() if k!=support_box_id))
        max_line_error=max(max_line_error,line_error); max_angle=max(max_angle,angle)
    # Certify all joint-linear segments with conservative swept bounds.
    # Adaptive bisection preserves both endpoint poses and the original samples.
    subdivisions=0; checked_intervals=0
    def certify(qa, qb, sa, sb, depth=0):
        nonlocal subdivisions, checked_intervals
        delta=np.abs(qb-qa)
        monotonic = check.support_monotonic_interval(qa, qb)
        if monotonic and (check.interval_safe(sa, delta, support_monotonic=True) or check.interval_safe(sb, delta, support_monotonic=True)):
            checked_intervals+=1; return
        if depth>=12:
            raise ValueError("Continuous collision enclosure could not prove a joint-linear segment safe")
        qm=(qa+qb)/2; sm=check.state(qm); subdivisions+=1
        if not sa["support_gap"]-1e-9 <= sm["support_gap"] <= sb["support_gap"]+1e-9:
            raise ValueError("Support gap reverses during interpolated departure")
        certify(qa,qm,sa,sm,depth+1); certify(qm,qb,sm,sb,depth+1)
    for i in range(len(states)-1):
        certify(arrays["position_rad"][i],arrays["position_rad"][i+1],states[i],states[i+1])
    final=states[-1]
    endpoint=copy.deepcopy(request)
    endpoint["start_position_rad"]=arrays["position_rad"][-1].tolist()
    endpoint["start_velocity_rad_s"]=[0.]*6
    endpoint["payload"]["pose_base_wxyz"]=matrix_pose(final["box"])
    endpoint["scene"]["cuboid"][check.box_id]["pose"]=endpoint["payload"]["pose_base_wxyz"]
    endpoint["pose_source"]="predicted_contact_escape_endpoint_diagnostic"
    endpoint["requires_runtime_remeasurement_before_execution"]=True
    canonical=json.dumps(request,sort_keys=True,separators=(",",":"),allow_nan=False).encode()
    certificate={"schema":SCHEMA,"planner":"CPUContactEscape","success":True,
        "contact_escape_validated":True,"payload_collision_enabled":True,"payload_box_id":check.box_id,
        "waypoints":len(states),"duration_seconds":float(arrays["time_s"][-1]),
        "physical_execution_validated":False,"historical_diagnostic":bool(historical_diagnostic),
        "measured_start_within_velocity_gate":stopped,
        "requires_live_scene_attachment_and_stopped_start_validation":True,
        "requires_post_escape_remeasurement_and_curobo_replan":True,
        "request_canonical_sha256":hashlib.sha256(canonical).hexdigest(),
        "robot_config_sha256":sha256(request["robot_config"]),"urdf_sha256":sha256(check.kin["urdf_path"]),
        "box_id":check.box_id,"support_box_id":support_box_id,"lift_m":lift_m,
        "numerical_support_overlap_m":check.numerical_support_overlap_m,
        "joint_names":request["joint_names"],"start_position_rad":check.q0.tolist(),
        "end_position_rad":arrays["position_rad"][-1].tolist(),"interpolation_dt_s":dt,
        "duration_s":float(arrays["time_s"][-1]),"trajectory_samples":len(states),
        "limits":{"velocity_rad_s":.10,"acceleration_rad_s2":.20,"jerk_rad_s3":2.},"observed_peaks":peaks,
        "maximum_ik_position_error_m":max_ik_position,"maximum_ik_rotation_error_rad":max_ik_rotation,
        "maximum_sampled_vertical_position_error_m":max_line_error,"maximum_sampled_orientation_error_rad":max_angle,
        "minimum_support_separation_m":min_support,"end_support_separation_m":final["support_gap"],
        "support_separation_monotonic_at_validated_samples":True,
        "support_separation_continuously_monotonic":True,
        "monotonic_certificate":"all 8 payload vertex plane derivatives bounded using joint Jacobians and conservative Hessian radius",
        "minimum_arm_world_clearance_m":min_arm,"minimum_self_clearance_m":min_self,
        "minimum_other_payload_sphere_world_clearance_m":min_other_payload,"minimum_exact_payload_world_gap_m":min_exact,
        "endpoint_all_payload_sphere_world_clearance_m":min(final["payload"].values()),
        "continuous_collision_check":"conservative Lipschitz enclosure of joint-linear sample intervals",
        "certified_intervals":checked_intervals,"adaptive_midpoints":subdivisions,
        "support_exception":{"only_pair":[check.box_id,support_box_id],"method":"actual full OBB SAT plus fixed separating-plane monotonic departure",
            "numerical_overlap_allowance_m":check.numerical_support_overlap_m,
            "initial_signed_support_gap_m":check.initial_support_gap,
            "initial_support_penetration_m":max(0.,-check.initial_support_gap),
            "initial_numerical_overlap_accepted":check.initial_support_gap < 0.,
            "support_gap_lower_bound_m":check.support_gap_floor,
            "support_penetration_does_not_increase":True,
            "endpoint_strictly_positive_separation":final["support_gap"] > 0.,
            "strict_nonpenetration_certified":check.initial_support_gap >= 0.,
            "requires_runtime_stable_hold":{"duration_s":.25,"joint_speed_rad_s":.01,"box_linear_speed_m_s":.02,"box_angular_speed_rad_s":.05},
            "support_collider_removed":False,"payload_dimensions_changed":False},
        "payload_sphere_count":len(check.pc),"active_robot_sphere_count":sum(len(c) for c,_ in check.links.values()),
        "world_obstacles_checked":list(check.world),"fixed_payload_self_contact_certificate":check.self_proof,
        "fixed_tcp_to_payload_transform":check.relative.tolist(),
        "scope":"rigid simulated box; no real vacuum seal/load certification"}
    if check.attachment_uncertainty is not None:
        certificate.update(attachment_uncertainty=check.attachment_uncertainty,
            payload_sphere_padding_m=check.coverage["margin_m"],
            attachment_model="nominal transform with bounded elastic pose uncertainty",
            physical_continuous_collision_guarantee=False,
            actual_support_departure_certified=False,
            requires_runtime_actual_support_departure_checks=True,
            scope="nominal source departure; padded non-support collision checks conditional on monitored attachment bounds")
        certificate["support_exception"].update(
            certification_scope="nominal joint-linear path only; actual elastic support gap must be checked during physics",
            actual_elastic_support_separation_proven=False)
    return {"trajectory":arrays,"certificate":certificate,"endpoint_request":endpoint}


def export_escape(request, output, *, support_box_id, lift_m=.08, historical_diagnostic=False, numerical_support_overlap_m=0.):
    """Write a fresh artifact into a run folder, preserving an identical request."""
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    if any((output/name).exists() for name in ("trajectory.npz","result.json","endpoint-request.json")):
        raise ValueError("Refusing to overwrite an existing contact escape result")
    request_path=output/"request.json"
    if request_path.exists():
        if json.loads(request_path.read_text()) != request:
            raise ValueError("Existing measured request differs; refusing to overwrite")
    else:
        write_json(request_path,request)
    result=generate_escape(request,support_box_id,lift_m,historical_diagnostic=historical_diagnostic,numerical_support_overlap_m=numerical_support_overlap_m)
    np.savez_compressed(output/"trajectory.npz",**result["trajectory"],joint_names=np.asarray(request["joint_names"],dtype="U64"),
                        numerical_support_overlap_m=np.asarray(result["certificate"]["numerical_support_overlap_m"]))
    certificate=result["certificate"]
    certificate.update(request_path=str(request_path.resolve()),request_sha256=sha256(request_path),
                       trajectory_sha256=sha256(output/"trajectory.npz"),
                       source_code_sha256=sha256(Path(__file__)))
    write_json(output/"endpoint-request.json",result["endpoint_request"])
    write_json(output/"result.json",certificate)
    return result


def validate_saved_escape(request_path, npz_path, result_path):
    """Regenerate IK, limits and continuous collision checks; never trust JSON success."""
    request_path,npz_path,result_path=map(Path,(request_path,npz_path,result_path))
    request=json.loads(request_path.read_text()); result=json.loads(result_path.read_text())
    robot=yaml.safe_load(Path(request["robot_config"]).read_text())
    kin=robot.get("robot_cfg",robot)["kinematics"]
    required={"schema":SCHEMA,"planner":"CPUContactEscape","success":True,
              "contact_escape_validated":True,"payload_collision_enabled":True,
              "payload_box_id":request["payload"]["box_id"],"historical_diagnostic":False,
              "measured_start_within_velocity_gate":True,
              "request_sha256":sha256(request_path),"trajectory_sha256":sha256(npz_path),
              "robot_config_sha256":sha256(request["robot_config"]),"urdf_sha256":sha256(kin["urdf_path"])}
    # This reviewed predecessor used identical zero-padding rigid checks.
    # Legacy artifacts still undergo full current geometry/IK revalidation.
    legacy_rigid_source="0eeca907c44282d1e25217b242da1d75b5dd90bd9c33a1bd5f4c57416cbc47af"
    source=result.get("source_code_sha256")
    for key,value in required.items():
        if result.get(key) != value:
            raise ValueError(f"Saved contact escape provenance rejected: {key}")
    if source != sha256(Path(__file__)) and not (source==legacy_rigid_source and "attachment_uncertainty" not in request):
        raise ValueError("Saved contact escape provenance rejected: source_code_sha256")
    with np.load(npz_path,allow_pickle=False) as saved:
        if saved["joint_names"].tolist() != request["joint_names"]:
            raise ValueError("Saved joint order differs from measured request")
        arrays={key:saved[key].copy() for key in ("position_rad","velocity_rad_s","acceleration_rad_s2","jerk_rad_s3","time_s")}
        saved_allowance=saved["numerical_support_overlap_m"]
        if saved_allowance.shape != () or float(saved_allowance) != result.get("numerical_support_overlap_m"):
            raise ValueError("Saved numerical support allowance differs from hashed trajectory")
    model=RobotModel(kin,request["joint_names"])
    verify_limits(arrays,model.lower,model.upper)
    if not np.allclose(arrays["position_rad"][0],request["start_position_rad"],atol=1e-12,rtol=0):
        raise ValueError("Saved path does not start at measured joint state")
    fresh=generate_escape(request,result["support_box_id"],result["lift_m"],numerical_support_overlap_m=result["numerical_support_overlap_m"])
    for key,value in fresh["trajectory"].items():
        if arrays[key].shape != value.shape or not np.allclose(arrays[key],value,atol=1e-10,rtol=1e-10):
            raise ValueError(f"Saved {key} differs from independently regenerated checked path")
    for key in ("waypoints","joint_names","interpolation_dt_s","support_box_id","lift_m",
                "payload_sphere_count","active_robot_sphere_count","world_obstacles_checked","support_exception","numerical_support_overlap_m"):
        if result.get(key) != fresh["certificate"][key]:
            raise ValueError(f"Saved validation summary differs: {key}")
    if request.get("attachment_uncertainty") is not None:
        for key in ("attachment_uncertainty","payload_sphere_padding_m","attachment_model",
                    "physical_continuous_collision_guarantee","actual_support_departure_certified",
                    "requires_runtime_actual_support_departure_checks"):
            if result.get(key)!=fresh["certificate"][key]:
                raise ValueError("Saved elastic validation summary differs: "+key)
    return fresh["certificate"]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--support-box-id",required=True); p.add_argument("--lift-m",type=float,default=.08)
    p.add_argument("--historical-diagnostic",action="store_true")
    p.add_argument("--numerical-support-overlap-m",type=float,default=0.)
    args=p.parse_args()
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2])); os.nice(10)
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        result=export_escape(json.loads(args.request.read_text()),args.output,support_box_id=args.support_box_id,
                             lift_m=args.lift_m,historical_diagnostic=args.historical_diagnostic,numerical_support_overlap_m=args.numerical_support_overlap_m)
        print(json.dumps(result["certificate"],indent=2))
    except Exception as exc:
        write_json(args.output/"result.json",{"schema":SCHEMA,"planner":"CPUContactEscape","success":False,
            "physical_execution_validated":False,"error":type(exc).__name__,"message":str(exc),"request_sha256":sha256(args.request)})
        raise


if __name__ == "__main__":
    main()
