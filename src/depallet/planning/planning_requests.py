"""CPU preparation of auditable cuRobo requests from a settled Isaac snapshot."""
from __future__ import annotations

import copy
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from depallet.motion.curobo_bridge import REQUEST_SCHEMA, pose, sha256, validate_request, write_json
from depallet.planning.motion_profiles import apply_motion_profile, resolve_motion_profile
from depallet.motion.urdf_fk import matrix as fk_matrix


def rotation(wxyz):
    pose([0., 0., 0.] + list(wxyz), "orientation")
    return Rotation.from_quat(list(wxyz[1:]) + [wxyz[0]])


def quaternion(rot):
    x, y, z, w = rot.as_quat()
    return [float(w), float(x), float(y), float(z)]


def relative_pose(position, orientation, base_position, base_orientation):
    inverse = rotation(base_orientation).inv()
    return inverse.apply(np.asarray(position) - base_position).tolist() + quaternion(
        inverse * rotation(orientation))


def grasp_poses(position, orientation, dimensions, clearance=.001, hover=.15):
    """Top-center poses: tool +Z approaches along the box's negative local Z."""
    r = rotation(orientation)
    top = np.asarray(position) + r.apply([0., 0., dimensions[2]/2 + clearance])
    above = top + r.apply([0., 0., hover])
    q = quaternion(r * Rotation.from_euler("x", math.pi))
    return above.tolist() + q, top.tolist() + q


def slow_robot_config(config, output, motion_profile="baseline", *, native_urdf=None):
    """Stage profile limits from the native URDF; never stretch timestamps."""
    profile = resolve_motion_profile(motion_profile)
    config, output = Path(config).resolve(), Path(output).resolve()
    value = yaml.safe_load(config.read_text())
    kin = value.get("robot_cfg", value)["kinematics"]
    if kin["base_link"] != "base_link":
        raise ValueError("H2017 planning base must be base_link")
    for link in ("vgp20_body", "vgp20_adapter"):
        if link not in kin["collision_link_names"] or not kin["collision_spheres"].get(link):
            raise ValueError(f"Missing complete gripper collision: {link}")
    configured_urdf=Path(kin["urdf_path"]).resolve()
    source_urdf=Path(native_urdf).resolve() if native_urdf is not None else configured_urdf
    tree=ET.parse(source_urdf)
    joints = {j.get("name"): j for j in tree.getroot().findall("joint")}
    names = kin["cspace"]["joint_names"]
    limits = [float(joints[n].find("limit").get("velocity")) for n in names]
    if any(v <= 0 or not math.isfinite(v) for v in limits):
        raise ValueError("Invalid URDF joint velocity limits")
    # Pinned V2 applies velocity_scale in both KinematicsLoader and
    # KinematicsParams. Explicit private URDF limits + unit scale are invariant
    # under repeated scaling, without modifying upstream or original assets.
    limited_urdf=output.with_name("h2017_vgp20_velocity_limited.urdf")
    for name,limit in zip(names,limits):
        joints[name].find("limit").set("velocity",str(min(profile["maximum_velocity_rad_s"],limit)))
    tree.write(limited_urdf,encoding="utf-8",xml_declaration=True)
    kin["urdf_path"]=str(limited_urdf)
    kin["cspace"]["velocity_scale"] = [1.] * len(names)
    kin["cspace"]["max_acceleration"] = profile["maximum_acceleration_rad_s2"]
    kin["cspace"]["max_jerk"] = profile["maximum_jerk_rad_s3"]
    output.write_text(yaml.safe_dump(value, sort_keys=False))
    return value, {"schema":"depallet.motion_profile_config.v1",
                   "profile":profile["profile"],"robot_config":str(output),
                   "source_config": str(config), "source_config_sha256": sha256(config),
                   "robot_config_sha256": sha256(output), "joint_names": names,
                   "urdf_velocity_limits_rad_s": limits,
                   "configured_urdf":str(configured_urdf),"configured_urdf_sha256":sha256(configured_urdf),
                   "source_urdf":str(source_urdf),"source_urdf_sha256":sha256(source_urdf),
                   "native_urdf_override_used":native_urdf is not None,
                   "limited_urdf":str(limited_urdf),"limited_urdf_sha256":sha256(limited_urdf),
                   "velocity_scale_application_invariant":True,
                   "motion_profile": profile["profile"],
                   "effective_velocity_limits_rad_s": [min(profile["maximum_velocity_rad_s"],v) for v in limits],
                   "max_acceleration_rad_s2": profile["maximum_acceleration_rad_s2"],
                   "max_jerk_rad_s3": profile["maximum_jerk_rad_s3"],
                   "source_config_collision_geometry_preserved":True,
                   "native_joint_velocity_limits_are_upper_bounds":True,
                   "timestamps_rescaled":False,"physical_execution_validated":False}


def pedestal_cuboids(kin, base_position, base_orientation):
    """Resolve fixed mounting contact using a documented, local fixture recess.

    The fitted *fixed base* spheres extend below the physical mounting plane.
    Keep all robot collision spheres and self collision. Only the overlapping
    part of the static pedestal's mounting pocket is omitted from the world
    approximation. The remainder of the pedestal has five cuboids; this is not
    a certificate for motion into the bolted mounting interface.
    """
    if np.linalg.norm(rotation(base_orientation).as_rotvec()) > 1e-5:
        raise ValueError("Pedestal mount exception requires upright base")
    if np.linalg.norm(np.asarray(base_position) - [-.5, 0., .25]) > 1e-4:
        raise ValueError("Pedestal mount exception belongs to this scene only")
    penetrating = [s for s in kin["collision_spheres"]["base_link"]
                   if s["center"][2] - s["radius"] < 0]
    # Full XY projection is deliberately broader than each below-plane cap.
    x0 = min(s["center"][0]-s["radius"] for s in penetrating) - .003
    x1 = max(s["center"][0]+s["radius"] for s in penetrating) + .003
    y0 = min(s["center"][1]-s["radius"] for s in penetrating) - .003
    y1 = max(s["center"][1]+s["radius"] for s in penetrating) + .003
    z0 = min(s["center"][2]-s["radius"] for s in penetrating) - .003
    lo, hi, bottom = -.275, .275, -.25
    if not (lo < x0 < x1 < hi and lo < y0 < y1 < hi and bottom < z0 < 0):
        raise ValueError("Unexpected fixed-base mounting penetration")
    boxes = {}
    def add(name, mins, maxs):
        a,b = np.asarray(mins), np.asarray(maxs)
        boxes[name] = {"dims": (b-a).tolist(), "pose": ((a+b)/2).tolist()+[1.,0.,0.,0.]}
    add("pedestal_bottom", [lo,lo,bottom], [hi,hi,z0])
    add("pedestal_rim_left", [lo,lo,z0], [x0,hi,0.])
    add("pedestal_rim_right", [x1,lo,z0], [hi,hi,0.])
    add("pedestal_rim_front", [x0,lo,z0], [x1,y0,0.])
    add("pedestal_rim_back", [x0,y1,z0], [x1,hi,0.])
    # Coordinates above are authored relative to the nominal mount. Translate
    # the few nanometers between nominal and measured base into the real frame.
    delta = np.asarray([-.5,0.,.25]) - base_position
    for value in boxes.values():
        value["pose"][:3] = (np.asarray(value["pose"][:3]) + delta).tolist()
    return boxes, {"type": "fixed_base_mount_contact_only",
                   "physical_pedestal_dimensions_m": [.55,.55,.25],
                   "pocket_min_nominal_base_m": [x0,y0,z0],
                   "pocket_max_nominal_base_m": [x1,y1,0.],
                   "fitted_base_maximum_under_plane_m": -z0-.003,
                   "robot_collision_links_disabled": [],
                   "all_base_spheres_retained": True,
                   "scope": "fixed bolted base only; pocket is not traversable workspace",
                   "static_fixture_approximation_requires_execution_contact_monitor": True}


def scene_cuboids(spec, build, settle, robot, kin, source_pose, goal_pose):
    if not settle.get("passed") or not all(settle["per_object_settled"]):
        raise ValueError("Scene must be physically settled")
    ids = settle["box_ids"]
    boxes = {x["id"]: x for x in build["transformed_boxes"]}
    box_count = len(spec.get("boxes", ()))
    if (box_count < 1 or len(ids) != box_count or len(set(ids)) != box_count
            or len(boxes) != box_count or set(ids) != set(boxes)):
        raise ValueError(
            f"Exactly all {box_count} unique scene boxes must be preserved")
    state = settle["final_physics_state"]
    if any(len(state[key]) != box_count for key in ("positions_m", "quaternions_wxyz")):
        raise ValueError("Incomplete settled box pose array")
    bp, bq = robot["measured_base_position_m"], robot["measured_base_quaternion_wxyz"]
    result = {}
    objects = {}
    for i, name in enumerate(ids):
        p,q = state["positions_m"][i], state["quaternions_wxyz"][i]
        q = quaternion(rotation(q))
        result[name] = {"dims": boxes[name]["dimensions_m"], "pose": relative_pose(p,q,bp,bq)}
        objects[name] = {"position_m": p, "quaternion_wxyz": q,
                         "dimensions_m": boxes[name]["dimensions_m"]}
    def add(name, dims, p, q=(1.,0.,0.,0.)):
        if name in result:
            raise ValueError("Duplicate obstacle name")
        result[name] = {"dims": list(dims), "pose": relative_pose(p,q,bp,bq)}
    for name, pallet_pose in (("source_pallet",source_pose), ("goal_pallet",goal_pose)):
        dims = spec["pallet"]["dimensions_m"]
        r = Rotation.from_euler("z", pallet_pose[3])
        center = np.asarray(pallet_pose[:3])+r.apply([0.,0.,dims[2]/2])
        add(name, dims, center.tolist(), quaternion(r))
    add("ground", [7.,7.,.05], [0.,0.,-.025])
    add("back_wall", [7.,.08,3.], [0.,2.4,1.5])
    pedestal, mount = pedestal_cuboids(kin,bp,bq)
    result.update(pedestal)
    return result, objects, mount


def initial_world_clearance(kin, joint_names, joints, cuboids):
    """Independent CPU sphere/OBB initial-state check (proxy geometry only)."""
    collisions = []
    minimum = float("inf")
    for link, spheres in kin["collision_spheres"].items():
        active = [s for s in spheres if s["radius"] > 0]
        if not active:
            continue
        t = fk_matrix(kin["urdf_path"], joint_names, joints, tip=link, base=kin["base_link"])
        for index,s in enumerate(active):
            center = t[:3,:3] @ np.asarray(s["center"]) + t[:3,3]
            for name, box in cuboids.items():
                local = rotation(box["pose"][3:]).inv().apply(center-box["pose"][:3])
                # Signed sphere-to-solid-cuboid clearance.
                delta = np.abs(local)-np.asarray(box["dims"])/2
                sdf = np.linalg.norm(np.maximum(delta,0.)) + min(float(np.max(delta)),0.)
                clearance = float(sdf-s["radius"])
                minimum = min(minimum,clearance)
                if clearance < -1e-6:
                    collisions.append({"link":link,"sphere":index,"obstacle":name,
                                       "clearance_m":clearance})
    return {"passed":not collisions,"minimum_clearance_m":minimum,"collisions":collisions,
            "scope":"independent CPU URDF sphere/OBB initial-state check; no path validated"}


def make_request(box_id, robot_config, robot, cuboids, objects, mount, pose_override=None,
                 motion_profile="baseline"):
    obj = copy.deepcopy(objects[box_id])
    source = "oracle_diagnostic"
    if pose_override is not None:
        if (pose_override.get("schema") != "depallet.validated_pose.v1"
                or pose_override.get("box_id") != box_id
                or pose_override.get("frame") != "world"
                or pose_override.get("method") != "Point2Pose"
                or pose_override.get("validation",{}).get("passed") is not True):
            raise ValueError("Point2Pose override requires a validated world pose for this box")
        validation = pose_override["validation"]
        if (validation.get("position_error_bound_m",float("inf")) > .002
                or validation.get("orientation_error_bound_rad",float("inf")) > .05):
            raise ValueError("Point2Pose uncertainty exceeds the contact planning budget")
        pose(list(pose_override["position_m"])+list(pose_override["quaternion_wxyz"]), "validated pose")
        obj.update({k:pose_override[k] for k in ("position_m","quaternion_wxyz")})
        source = "point2pose_validated_target_with_oracle_scene_obstacles"
    bp,bq = robot["measured_base_position_m"], robot["measured_base_quaternion_wxyz"]
    above, touch = grasp_poses(obj["position_m"],obj["quaternion_wxyz"],obj["dimensions_m"])
    goals = []
    for name,value in (("pregrasp",above),("contact",touch)):
        p = relative_pose(value[:3],value[3:],bp,bq)
        g = {"id":name,"tcp_frame":"suction_tcp","position_m":p[:3],"quaternion_wxyz":p[3:]}
        if name == "contact":
            g.update({"linear_axis":"z","linear_in_tool_frame":True})
        goals.append(g)
    request = {"schema":REQUEST_SCHEMA,"robot_config":str(Path(robot_config).resolve()),
               "joint_names":[f"joint_{i}" for i in range(1,7)],
               "start_position_rad":robot["final_joints_rad"],"start_velocity_rad_s":[0.]*6,
               "interpolation_dt_s":1/60,"length_unit":"m","quaternion_order":"wxyz",
               "base_frame":"base_link","base_world_position_m":bp,"base_world_quaternion_wxyz":bq,
               "scene":{"cuboid":copy.deepcopy(cuboids)},"goals":goals,
               "box_id":box_id,"pose_source":source,"obstacle_pose_source":"settled_simulation_oracle",
               "target_box_world_pose":obj["position_m"]+obj["quaternion_wxyz"],
               "grasp_tcp_world_pose":touch,"pregrasp_tcp_world_pose":above,
               "position_tolerance_m":.001,"endpoint_position_tolerance_m":.002,
               "mount_contact_exception":mount,"payload":None,
               "physical_execution_validated":False,"point2pose_pipeline_validated":False}
    apply_motion_profile(request, motion_profile)
    if pose_override is not None:
        request["target_pose_validation"] = pose_override["validation"]
    return validate_request(request)
