"""CPU checks and measured-state payload requests for one simulated carton.

This module does not start Isaac, invoke a planner, or mark a physical grasp valid.
The caller supplies rigid-body measurements after Surface Gripper confirmation.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from curobo_bridge import validate_request, pose_error
from payload_self_contact import attachment_uncertainty_policy, attachment_uncertainty_geometry, payload_cover


def transform(position, quaternion):
    p, q = np.asarray(position, float), np.asarray(quaternion, float)
    if p.shape != (3,) or q.shape != (4,) or not np.isfinite(np.r_[p, q]).all() or abs(np.linalg.norm(q)-1) > 1e-4:
        raise ValueError("Expected finite metric position and normalized wxyz quaternion")
    w, x, y, z = q
    matrix = np.eye(4)
    matrix[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    matrix[:3, 3] = p
    return matrix


def pose_from_transform(matrix):
    m = np.asarray(matrix, float)
    if m.shape != (4, 4) or not np.isfinite(m).all() or not np.allclose(m[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError("Invalid homogeneous transform")
    r = m[:3, :3]
    if not np.allclose(r.T@r, np.eye(3), atol=1e-5) or abs(np.linalg.det(r)-1) > 1e-5:
        raise ValueError("Transform rotation is not proper orthonormal")
    if np.trace(r) > 0:
        s = np.sqrt(np.trace(r)+1)*2
        q = np.array([s/4, (r[2, 1]-r[1, 2])/s, (r[0, 2]-r[2, 0])/s, (r[1, 0]-r[0, 1])/s])
    else:
        i = int(np.argmax(np.diag(r))); j, k = (i+1) % 3, (i+2) % 3
        s = np.sqrt(1+r[i, i]-r[j, j]-r[k, k])*2
        q = np.zeros(4); q[i+1] = s/4
        q[0] = (r[k, j]-r[j, k])/s
        q[j+1] = (r[j, i]+r[i, j])/s; q[k+1] = (r[k, i]+r[i, k])/s
    q /= np.linalg.norm(q)
    if q[0] < 0: q = -q
    return m[:3, 3].tolist()+q.tolist()


def load_execution_plan(path, data_root, run_root):
    plan = json.loads(Path(path).read_text())
    if plan.get("schema") != "depallet.single_box_execution.v1":
        raise ValueError("Unknown single-box execution schema")
    if plan.get("perception_source") not in ("simulation_oracle", "point2pose"):
        raise ValueError("Explicit perception_source is required")
    if not isinstance(plan.get("box_id"), str):
        raise ValueError("A single target box_id is required")
    pose_from_transform(transform(plan["goal_position_m"], plan["goal_quaternion_wxyz"]))
    if "request_path" not in plan: plan["request_path"] = plan.get("approach_request")
    if "assembly_manifest" not in plan:
        plan["assembly_manifest"] = str(Path(data_root)/"h2017-vgp20-v2/manifest.json")
    roots = [Path(data_root).resolve(), Path(run_root).resolve()]
    for key in ("source_scene_run", "request_path", "approach_run", "robot_config", "assembly_manifest"):
        value = plan.get(key)
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"Completed absolute {key} path required")
        resolved = Path(value).resolve()
        if not any(resolved.is_relative_to(root) and resolved != root for root in roots) or not resolved.exists():
            raise ValueError(f"{key} must exist inside this project's data/runs")
        plan[key] = str(resolved)
    for key, expected in [("source_pose", [0., -.95, 0., 0.]), ("goal_pose", [0., .95, 0., 0.])]:
        if not np.allclose(plan.get(key, expected), expected, atol=1e-9):
            raise ValueError("Execution currently supports only the reviewed source/goal workcell anchors")
        plan[key] = expected
    request_bytes = Path(plan["request_path"]).read_bytes()
    if plan.get("request_sha256") and hashlib.sha256(request_bytes).hexdigest() != plan["request_sha256"]:
        raise ValueError("Execution request changed since the reviewed manifest")
    request = validate_request(json.loads(request_bytes))
    if Path(request["robot_config"]).resolve() != Path(plan["robot_config"]).resolve() or request.get("payload"):
        raise ValueError("Approach config mismatch or unexpected attached payload")
    if plan["perception_source"] == "simulation_oracle" and request.get("pose_source") != "oracle_diagnostic":
        raise ValueError("Oracle execution must preserve oracle_diagnostic request provenance")
    if plan["perception_source"] == "point2pose" and request.get("pose_source") != "point2pose":
        raise ValueError("Point2Pose execution must carry matching request provenance")
    plan["approach_request_data"] = request
    return plan


def measured_scene_check(request, box_ids, state, boxes, base_pose, *, position_tolerance_m=.002, angle_tolerance_rad=.005):
    """Reject an approach planned for a different or subsequently moved scene."""
    inverse = np.linalg.inv(transform(base_pose[:3], base_pose[3:]))
    cuboids = request["scene"]["cuboid"]
    checks = []
    for index, key in enumerate(box_ids):
        if key not in cuboids or not np.allclose(cuboids[key]["dims"], boxes[key]["dimensions_m"], atol=1e-6):
            raise ValueError("Plan omits or resizes scene box: "+key)
        actual = pose_from_transform(inverse@transform(state["positions_m"][index], state["quaternions_wxyz"][index]))
        expected = cuboids[key]["pose"]
        distance, angle = pose_error(actual[:3], actual[3:], expected[:3], expected[3:])
        checks.append(dict(box_id=key, position_error_m=distance, rotation_error_rad=angle))
        if distance > position_tolerance_m or angle > angle_tolerance_rad:
            raise ValueError("Scene moved since approach planning: "+key)
    return dict(passed=True, checks=checks, source="actual PhysX rigid-body state", calibration_claim=False)


def static_scene_check(request, base_pose, pallet_dimensions):
    """Validate the reviewed workcell, including its declared fixed-mount pocket."""
    inv = np.linalg.inv(transform(base_pose[:3], base_pose[3:]))
    fixed = {"ground": ([0., 0., -.025], [7., 7., .05]),
             "back_wall": ([0., 2.4, 1.5], [7., .08, 3.]),
             "source_pallet": ([0., -.95, pallet_dimensions[2]/2], pallet_dimensions),
             "goal_pallet": ([0., .95, pallet_dimensions[2]/2], pallet_dimensions)}
    mount = request.get("mount_contact_exception", {})
    if (mount.get("type") != "fixed_base_mount_contact_only" or mount.get("all_base_spheres_retained") is not True
            or mount.get("robot_collision_links_disabled") != []):
        raise ValueError("Reviewed fixed base mount description is missing")
    lo, hi = np.asarray(mount.get("pocket_min_nominal_base_m"), float), np.asarray(mount.get("pocket_max_nominal_base_m"), float)
    if (lo.shape != (3,) or hi.shape != (3,) or not np.isfinite(np.r_[lo, hi]).all()
            or not np.all(lo < hi) or np.any(abs(lo[:2]) > .18) or np.any(abs(hi[:2]) > .18)
            or not -.05 <= lo[2] <= -.02 or abs(hi[2]) > 1e-8):
        raise ValueError("Fixed-mount pocket exceeds reviewed dimensions")
    a, b = [-.275, -.275, -.25], [.275, .275, 0.]
    regions = {"pedestal_bottom": (a, [b[0], b[1], lo[2]]),
               "pedestal_rim_left": ([a[0], a[1], lo[2]], [lo[0], b[1], 0.]),
               "pedestal_rim_right": ([hi[0], a[1], lo[2]], b),
               "pedestal_rim_front": ([lo[0], a[1], lo[2]], [hi[0], lo[1], 0.]),
               "pedestal_rim_back": ([lo[0], hi[1], lo[2]], [hi[0], b[1], 0.])}
    for name, (low, high) in regions.items():
        low, high = np.asarray(low), np.asarray(high)
        fixed[name] = ((low+high)/2+[-.5, 0., .25], high-low)
    for name, (position, dims) in fixed.items():
        item = request["scene"]["cuboid"].get(name)
        expected = pose_from_transform(inv@transform(position, [1, 0, 0, 0]))
        if item is None or not np.allclose(item["dims"], dims, atol=1e-6):
            raise ValueError("Missing or resized static obstacle: "+name)
        distance, angle = pose_error(item["pose"][:3], item["pose"][3:], expected[:3], expected[3:])
        if distance > .001 or angle > .001:
            raise ValueError("Static obstacle transform differs from physical stage: "+name)
    return dict(passed=True, checked_obstacles=list(fixed), fixed_mount_pocket_declared=True)


def goal_on_empty_pallet(plan, target, pallet_dimensions):
    """First-box execution requires full support on the initially empty goal pallet."""
    if plan.get("goal_available_now", True) is not True or plan.get("goal_support_id", "goal_pallet") != "goal_pallet":
        raise ValueError("Single-box goal requires a currently present empty pallet support")
    goal = transform(plan["goal_position_m"], plan["goal_quaternion_wxyz"])
    if not np.allclose(goal[:3, 2], [0, 0, 1], atol=1e-5):
        raise ValueError("Single-box goal must be upright")
    dims = np.asarray(target["dimensions_m"])
    corners = np.array([[x, y, -dims[2]/2, 1] for x in [-dims[0]/2, dims[0]/2] for y in [-dims[1]/2, dims[1]/2]])
    world = (goal@corners.T).T
    anchor = np.asarray(plan["goal_pose"][:3])
    if (np.any(np.abs(world[:, :2]-anchor[:2]) > np.asarray(pallet_dimensions[:2])/2-.01+1e-8)
            or not np.allclose(world[:, 2], anchor[2]+pallet_dimensions[2], atol=.001)):
        raise ValueError("Goal carton lacks full pallet support, 10 mm edge margin, or correct height")
    return True


def payload_request(plan, state, box_ids, boxes, measured_q, measured_v, base_pose, tcp_world_pose, *, attachment_confirmed, departure_completed=False):
    if attachment_confirmed is not True:
        raise ValueError("Measured target attachment is required before payload planning")
    key = plan["box_id"]; index = box_ids.index(key)
    request = copy.deepcopy(plan["approach_request_data"])
    if "attachment_uncertainty" in plan:
        request["attachment_uncertainty"]=attachment_uncertainty_policy(plan["attachment_uncertainty"])
    T_base_world = np.linalg.inv(transform(base_pose[:3], base_pose[3:]))
    if "payload_orientation_policy" in plan:
        from payload_upright import upright_policy
        request["payload_orientation_policy"] = upright_policy(
            plan["payload_orientation_policy"], world_up_base=T_base_world[:3, :3]@np.array([0., 0., 1.]))
    overhead_m = plan.get("transport_overhead_m", .20)
    if (isinstance(overhead_m, bool) or not isinstance(overhead_m, (int, float))
            or not math.isfinite(overhead_m) or not .03 <= overhead_m <= .50):
        raise ValueError("transport_overhead_m must be finite within 0.03..0.50 m")
    for i, box_id in enumerate(box_ids):
        request["scene"]["cuboid"][box_id]["pose"] = pose_from_transform(T_base_world@transform(state["positions_m"][i], state["quaternions_wxyz"][i]))
    T_world_box = transform(state["positions_m"][index], state["quaternions_wxyz"][index])
    T_world_tcp = transform(tcp_world_pose[:3], tcp_world_pose[3:])
    T_tcp_box = np.linalg.inv(T_world_tcp)@T_world_box
    T_goal_tcp = transform(plan["goal_position_m"], plan["goal_quaternion_wxyz"])@np.linalg.inv(T_tcp_box)
    dims = np.asarray(boxes[key]["dimensions_m"])
    # Every declared cell is fully enclosed, including unchanged attachment padding.
    from payload_cover_profiles import resolve_payload_cover_profile
    cover_profile=resolve_payload_cover_profile(plan.get("payload_cover_profile","legacy64"))
    count=cover_profile["sphere_count"]
    cover_cells=None if cover_profile["profile"]=="legacy64" else cover_profile["cells"]
    # Plan above the actual covering-sphere lower envelope.
    # Actual detached gravity settling must still reach the unchanged physical goal.
    uncertainty=attachment_uncertainty_geometry(dims.tolist(),request.get("attachment_uncertainty"))
    padding=0. if uncertainty is None else uncertainty["payload_padding_m"]
    _,coverage=payload_cover(dims.tolist(),count,padding_m=padding,cells=cover_cells)
    sphere_radius = coverage["radius_m"]
    bottom_overhang=sphere_radius-float(coverage["cell_widths_m"][2]/2.)
    release_gap = bottom_overhang+.005
    if not .005 <= release_gap <= .08:
        raise ValueError("Payload sphere release clearance exceeds 80 mm bound")
    T_goal_tcp[2, 3] += release_gap
    corners = np.array([[x, y, z, 1] for x in [-dims[0]/2, dims[0]/2] for y in [-dims[1]/2, dims[1]/2] for z in [-dims[2]/2, dims[2]/2]])
    offset = (T_world_box@corners.T).T[:, 2].min()-T_world_tcp[2, 3]
    other_tops = [state["positions_m"][i][2]+boxes[k]["dimensions_m"][2]/2 for i, k in enumerate(box_ids) if k != key]
    clear_z = max(T_world_tcp[2, 3]+.15, T_goal_tcp[2, 3]+.15, max(other_tops, default=.15)+.10-offset)
    if clear_z > 2.5: raise ValueError("Transport clearance exceeds bounded workcell height")
    lift, over = T_world_tcp.copy(), T_goal_tcp.copy()
    lift[2, 3] = clear_z
    # Source clearance and far-goal overhead are independent; the complete-world
    # collision planner determines a transit arc inside the robot workspace.
    over[2, 3] = T_goal_tcp[2, 3]+overhead_m
    goals = []
    stages = [("clear_transport_over_goal", over), ("release_above_goal_pallet", T_goal_tcp)]
    if not departure_completed:
        stages.insert(0, ("confirmed_payload_lift", lift))
    for name, T in stages:
        pose = pose_from_transform(T_base_world@T)
        item = dict(id=name, tcp_frame="suction_tcp", position_m=pose[:3], quaternion_wxyz=pose[3:])
        if name != "clear_transport_over_goal": item.update(linear_axis="z", linear_in_tool_frame=False)
        goals.append(item)
    request.update(goals=goals, start_position_rad=list(measured_q), start_velocity_rad_s=list(measured_v),
                   pose_source="measured_simulation_after_confirmed_grasp", perception_source=plan["perception_source"],
                   payload=dict(box_id=key, grasp_confirmed=True, dimensions_m=dims.tolist(),
                                pose_base_wxyz=request["scene"]["cuboid"][key]["pose"],
                                mass_kg=boxes[key]["physical"]["mass_kg"], num_spheres=count))
    if cover_profile["profile"]!="legacy64":
        request["payload"]["cover_profile"]=cover_profile["profile"]
    validate_request(request)
    if "payload_orientation_policy" in request:
        from payload_upright import validate_upright_request
        validate_upright_request(request)
    evidence = dict(T_tcp_box_measured=T_tcp_box.tolist(), tcp_world_pose=tcp_world_pose,
                    transport_clearance_z_world_m=None if departure_completed else clear_z, goal_overhead_z_world_m=float(over[2, 3]),
                    transport_overhead_m=float(overhead_m),
                    payload_orientation_policy=request.get("payload_orientation_policy"),
                    measured_contact_departure_completed=bool(departure_completed),
                    redundant_vertical_lift_removed=bool(departure_completed),
                    release_above_physical_goal_m=release_gap, payload_sphere_radius_m=sphere_radius,
                    payload_sphere_bottom_overhang_m=bottom_overhang,
                    requires_actual_gravity_settling_to_goal=True, box_id=key, actual_physics_state=state,
                    attachment_confirmed=True, payload_mass_measured=False, physical_execution_validated=False)
    if uncertainty is not None:
        evidence.update(attachment_uncertainty=uncertainty,payload_sphere_padding_m=padding,
                        attachment_model="nominal transform with bounded elastic pose uncertainty",
                        physical_continuous_collision_guarantee=False)
    return request, evidence


def contact_classification(pair, target_path, source_support_path, goal_pallet_path, *, phase="APPROACH", lift_confirmed=False):
    """External-contact screen; self-contact certification is deliberately separate."""
    def inside(path, root): return path == root or path.startswith(root+"/")
    a, b = pair
    robot = "/World/H2017"
    if inside(a, robot) and inside(b, robot):
        def index(path):
            name = path[len(robot)+1:].split("/")[0]
            if name == "base_link": return 0
            if name.startswith("link_") and name[5:].isdigit(): return int(name[5:])
            return None
        ia, ib = index(a), index(b)
        if ia is not None and ib is not None and abs(ia-ib) <= 1: return "adjacent_or_same_link_contact"
        return "unexpected_robot_self_nonadjacent"
    if inside(b, robot): a, b = b, a
    if inside(a, robot):
        if inside(a, robot+"/base_link") and inside(b, "/World/RobotPedestal"): return "fixed_robot_mount"
        if inside(a, robot+"/link_6/VGP20BodyCollision") and inside(b, target_path):
            if phase in ("APPROACH", "PREGRASP_SETTLE", "CONTACT_READY", "CLOSING", "GRASP_CONFIRMED", "ESCAPE", "ESCAPED", "TRANSPORT", "RELEASING"):
                return "intended_tool_target"
            return "unexpected_tool_target_after_release_or_before_approach"
        return "unexpected_robot_external"
    if inside(b, target_path): a, b = b, a
    if inside(a, target_path):
        source_allowed = phase in ("INSPECTION_MOVE", "INSPECTION_SETTLE", "INSPECTION_HOLD", "INITIALIZING", "IDLE", "APPROACH", "PREGRASP_SETTLE", "CONTACT_READY", "CLOSING", "GRASP_CONFIRMED") or (phase in ("ESCAPE", "TRANSPORT") and not lift_confirmed)
        if inside(b, source_support_path) and source_allowed: return "intended_target_support"
        if inside(b, goal_pallet_path) and phase in ("TRANSPORT", "RELEASING", "RELEASED", "RETREAT", "SETTLING", "DONE"):
            return "intended_target_support"
        return "unexpected_target_external"
    return "other_scene_contact"


def frozen_state_check(before, after, before_q, after_q, before_v, after_v, *, before_base, after_base, before_tcp, after_tcp):
    """Check complete physical state with no relative-tolerance loophole."""
    failed = []
    for name in ("physics_step", "sim_time"):
        if before.get(name) != after.get(name): failed.append(name)
    values = [(name, before[name], after[name]) for name in
              ("positions_m", "quaternions_wxyz", "linear_velocities_m_s", "angular_velocities_rad_s")]
    values.extend([("joint_positions_rad", before_q, after_q), ("joint_velocities_rad_s", before_v, after_v),
                   ("base_world_pose", before_base, after_base), ("tcp_world_pose", before_tcp, after_tcp)])
    for name, a, b in values:
        a, b = np.asarray(a, float), np.asarray(b, float)
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a, b, rtol=0., atol=1e-9):
            failed.append(name)
    return dict(passed=not failed, failed_components=failed, absolute_tolerance=1e-9,
                relative_tolerance=0., physics_clock_frozen=not any(k in failed for k in ("physics_step", "sim_time")))


def attachment_rigidity_check(expected_tcp_to_box, tcp_world_pose, box_position, box_quaternion, *, attachment_uncertainty=None):
    """Compare measured rigid attachment; never substitute planned box poses."""
    actual = np.linalg.inv(transform(tcp_world_pose[:3], tcp_world_pose[3:])) @ transform(box_position, box_quaternion)
    expected_pose = pose_from_transform(np.asarray(expected_tcp_to_box, float))
    actual_pose = pose_from_transform(actual)
    pe, re = pose_error(actual_pose[:3],actual_pose[3:],expected_pose[:3],expected_pose[3:])
    policy=attachment_uncertainty_policy(attachment_uncertainty)
    ptol=.005 if policy is None else policy["translation_m"]
    rtol=.03 if policy is None else policy["rotation_rad"]
    return dict(passed=pe <= ptol and re <= rtol, position_error_m=pe, orientation_error_rad=re,
                position_tolerance_m=ptol, orientation_tolerance_rad=rtol,
                source="actual TCP and box measurements", calibration_claim=False)
