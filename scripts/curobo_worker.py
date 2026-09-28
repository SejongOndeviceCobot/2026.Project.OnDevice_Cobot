#!/usr/bin/env python3
"""Isolated cuRobo V2 worker, requiring the project's GPU guard.

This is a planner, not an Isaac process or real-robot controller. Outputs are
joint motor targets. Reference: NVlabs/curobo commit
78fd485fa82d9b9a063fb4985e371814587e666a (Apache-2.0 API).
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback
import xml.etree.ElementTree as ET

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from depallet.motion.curobo_bridge import (RESULT_SCHEMA, planner_random_seed, pose_error, sha256,
                           validate_request, write_json)
from depallet.motion.curobo_geometry import augment_robot_config, box_cover

SOURCE_COMMIT = "78fd485fa82d9b9a063fb4985e371814587e666a"


def guarded_output(selected):
    if os.environ.get("ISAAC_P0_GUARDED") != "1":
        raise RuntimeError("cuRobo GPU work requires guarded_run.py")
    output = Path(os.environ["ISAAC_P0_OUTPUT"]).resolve()
    if selected and Path(selected).resolve() != output:
        raise ValueError("Output must equal the guard-owned ISAAC_P0_OUTPUT")
    if any((output / x).exists() for x in ("result.json", "trajectory.npz")):
        raise FileExistsError("Refusing to overwrite a previous planning result")
    return output


def configure_runtime():
    import torch
    import warp as wp
    import curobo.runtime as runtime
    cache = Path(os.environ.get("ISAAC_P0_RUNTIME_CACHE", os.environ["ISAAC_P0_CACHE"])) / "curobo-runtime"
    cache.mkdir(parents=True, exist_ok=True)
    runtime.cache_dir = str(cache)
    wp.config.kernel_cache_dir = str(cache / "warp")
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    # Torch's allocator cap is supplementary. The guard monitors all CUDA users,
    # including Warp allocations, and enforces total VRAM/RSS/time limits.
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(.9, 10 * 1024**3 / total), 0)
    return torch, wp


def prepare_robot(args, output):
    """Build collision spheres through the real V2 RobotBuilder under the guard."""
    torch, wp = configure_runtime()
    import numpy as np
    from curobo.robot_builder import RobotBuilder
    if not args.urdf or not args.asset_path or not args.tcp_frame:
        raise ValueError("--prepare-robot requires --urdf, --asset-path, --tcp-frame")
    np.random.seed(42)
    torch.manual_seed(42)
    tree_before_fit = ET.parse(args.urdf).getroot()
    joint_one = next(j for j in tree_before_fit.findall("joint") if j.get("name") == "joint_1")
    if joint_one.find("parent").get("link") != args.base_link:
        raise ValueError("Planning base must equal the actual joint_1 parent")
    builder = RobotBuilder(urdf_path=str(args.urdf.resolve()),
                           asset_path=str(args.asset_path.resolve()),
                           tool_frames=[args.tcp_frame])
    builder.fit_collision_spheres(sphere_density=.75, compute_metrics=True,
                                   use_collision_mesh=True, iterations=100)
    # Skip the upstream heuristic that ignores default-pose collisions.
    config = builder.build()
    config.base_link = args.base_link
    tree = ET.parse(args.urdf).getroot()
    adjacency = {}
    for joint in tree.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        adjacency.setdefault(parent, []).append(child)
    config.self_collision_ignore = adjacency
    builder.save(config, str(output / "robot.yml"))
    import yaml
    path = output / "robot.yml"
    saved = yaml.safe_load(path.read_text())
    kin = saved.get("robot_cfg", saved)["kinematics"]
    if any(link.attrib["name"] == "attached_object" for link in tree.findall("link")):
        kin.setdefault("collision_link_names", []).append("attached_object")
        kin["collision_link_names"] = list(dict.fromkeys(kin["collision_link_names"]))
        kin["extra_collision_spheres"] = dict(kin.get("extra_collision_spheres") or {})
        kin["extra_collision_spheres"]["attached_object"] = 64
    if not isinstance(kin.get("cspace"), dict):
        raise ValueError("RobotBuilder did not produce a valid cspace configuration")
    kin["cspace"]["max_acceleration"] = 1.0
    kin["cspace"]["max_jerk"] = 10.0
    kin["grasp_contact_link_names"] = []
    saved, coverage = augment_robot_config(saved, args.urdf)
    path.write_text(yaml.safe_dump(saved, sort_keys=False))
    write_json(output / "tool-coverage.json", coverage)
    write_json(output / "result.json", {
        "schema": RESULT_SCHEMA, "success": True, "mode": "prepare_robot",
        "source_reference_commit": SOURCE_COMMIT,
        "robot_config": str(output / "robot.yml"),
        "urdf_sha256": sha256(args.urdf),
        "collision_spheres": sum(len(s) for s in kin["collision_spheres"].values()),
        "collision_link_names": list(kin["collision_link_names"]),
        "self_collision_exclusions_require_review": True,
        "physical_execution_validated": False,
    })


def verify_runtime_limits(planner, request, output, robot):
    """Fail before solving if V2 changed the intended numerical joint limits."""
    kin=robot.get("robot_cfg",robot)["kinematics"]
    tree=ET.parse(kin["urdf_path"]).getroot()
    joints={j.get("name"):j for j in tree.findall("joint")}
    expected={
        "velocity":[min(float(joints[n].find("limit").get("velocity")),
                        float(request.get("maximum_velocity_rad_s",.25))) for n in request["joint_names"]],
        "acceleration":[float(request.get("maximum_acceleration_rad_s2",.5))]*6,
        "jerk":[float(request.get("maximum_jerk_rad_s3",5.))]*6}
    transition=planner.trajopt_solver.transition_model
    actual={name:getattr(transition,"max_"+name).detach().cpu().reshape(-1).tolist()
            for name in expected}
    passed=all(len(actual[name])==6 and all(abs(a-b)<1e-6 for a,b in zip(actual[name],expected[name]))
               for name in expected)
    record={"passed":passed,"expected":expected,"actual":actual,
            "robot_config":request["robot_config"],"urdf_path":kin["urdf_path"]}
    write_json(output/"runtime-limits.json",record)
    if not passed:
        raise ValueError("Runtime joint limits differ from intended limits; see runtime-limits.json")
    return record


def result_execution_contract(request, *, diagnostic_plan=False):
    diagnostic = diagnostic_plan or request.get("diagnostic_only") is True
    return {
        "mode": "diagnostic_only" if diagnostic else "planning",
        "robot_execution_authorized": False,
        "executable_trajectory": False if diagnostic else None,
        "physical_execution_validated": False,
    }


def validate_diagnostic_plan_request(request):
    if (request.get("robot_execution_authorized") is not False
            or request.get("physical_execution_validated") is not False):
        raise ValueError(
            "--diagnostic-plan requires explicit false execution and physical-validation fields")
    return request


def plan(request, output, *, diagnostic_plan=False):
    from depallet.validation.payload_upright import validate_upright_request, orientation_criteria_kwargs, certify_upright_trajectory
    upright_policy = validate_upright_request(request)
    torch, wp = configure_runtime()
    import numpy as np
    import yaml
    from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
    from curobo.types import GoalToolPose, JointState, Pose
    from depallet.motion.urdf_fk import pose as cpu_urdf_pose
    from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

    tensor = lambda data: torch.tensor(data, dtype=torch.float32, device="cuda:0")
    robot_path = Path(request["robot_config"])
    robot = yaml.safe_load(robot_path.read_text())
    payload_contact_certificate=None
    effective_robot_path=robot_path
    if request.get("payload"):
        from depallet.manipulation.payload_self_contact import derive_payload_config
        robot,payload_contact_certificate=derive_payload_config(robot,request)
        write_json(output/"payload-self-contact-validation.json",payload_contact_certificate)
        if not payload_contact_certificate["safe_to_plan"]:
            raise ValueError("Expanded gripper body and payload OBB are not safely separated")
        effective_robot_path=output/"effective-robot-payload.yml"
        effective_robot_path.write_text(yaml.safe_dump(robot,sort_keys=False))
    dt = float(request.get("interpolation_dt_s", 1/60))
    random_seed = planner_random_seed(request)
    cfg = MotionPlannerCfg.create(
        robot=robot, scene_model=request["scene"], self_collision_check=True,
        collision_cache={"cuboid": max(32, len(request["scene"]["cuboid"]))},
        max_batch_size=1, max_goalset=1, num_ik_seeds=32, num_trajopt_seeds=request.get("num_trajopt_seeds", 4),
        position_tolerance=float(request.get("position_tolerance_m", .005)),
        orientation_tolerance=.005 if upright_policy else .05,
        interpolation_dt=dt, interpolation_buffer_size=2000, random_seed=random_seed,
    )
    cfg.trajopt_solver_config.maximum_trajectory_dt=float(request.get("maximum_trajectory_dt_s",.2))
    with MotionPlanner(cfg) as planner:
        runtime_limits=verify_runtime_limits(planner,request,output,robot)
        if list(planner.joint_names) != request["joint_names"]:
            raise ValueError(f"Joint order mismatch: {planner.joint_names}")
        frames = {g["tcp_frame"] for g in request["goals"]}
        if frames != set(planner.tool_frames) or len(frames) != 1:
            raise ValueError("Exactly one matching TCP frame required")
        q = JointState.from_position(tensor([request["start_position_rad"]]),
                                     joint_names=planner.joint_names)
        q.velocity = tensor([request.get("start_velocity_rad_s", [0.] * 6)])
        payload = request.get("payload")
        if payload:
            robot_cfg = robot.get("robot_cfg", robot)
            kin = robot_cfg["kinematics"]
            extra = (kin.get("extra_links") or {}).get("attached_object")
            if extra:
                parent = extra["parent_link_name"]
                identity = extra["fixed_transform"] == [0,0,0,1,0,0,0]
            else:
                tree = ET.parse(kin["urdf_path"]).getroot()
                joint = next(j for j in tree.findall("joint")
                             if j.find("child").attrib["link"] == "attached_object")
                parent = joint.find("parent").attrib["link"]
                origin = joint.find("origin")
                xyz = origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"
                rpy = origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
                identity = joint.get("type") == "fixed" and all(
                    abs(float(v)) < 1e-10 for v in (xyz + " " + rpy).split())
            if parent != planner.tool_frames[0] or not identity:
                raise ValueError("Payload frame must be identity-relative to first TCP")
            if kin["extra_collision_spheres"]["attached_object"] < payload.get("num_spheres", 64):
                raise ValueError("Insufficient reserved payload sphere slots")
            if robot_cfg.get("load_dynamics", False):
                raise ValueError("Payload dynamics require calibrated TCP-frame inertia")
            # Exact enclosing cell spheres, not a sampled MorphIt approximation.
            # The named world copy is disabled only after geometry registration.
            from depallet.manipulation.payload_self_contact import payload_cover_for_request
            spheres, payload_cover, payload_uncertainty = payload_cover_for_request(request)
            sphere_tensor = tensor([s["center"]+[s["radius"]] for s in spheres])
            planner.attachment_manager.update(
                sphere_tensor, q, link_name="attached_object",
                world_objects_pose_offset=Pose.from_list(payload["pose_base_wxyz"]))
            planner.scene_collision_checker.enable_obstacle(payload["box_id"], enable=False)
            write_json(output / "payload-coverage.json", payload_cover)
            from depallet.validation.payload_gpu_audit import audit as audit_gpu_payload
            start_audit=audit_gpu_payload(planner,q,request,robot,output)
            if payload.get("cover_profile") and start_audit.get("passed") is not True:
                raise ValueError("Declared payload cover failed actual GPU start collision audit")
        pieces = {key: [] for key in ("position", "velocity", "acceleration")}
        phases = []
        for goal in request["goals"]:
            started = time.monotonic()
            criteria = (ToolPoseCriteria.linear_motion(
                axis=goal["linear_axis"], non_terminal_scale=1.,
                project_distance_to_goal=goal.get("linear_in_tool_frame", True))
                if goal.get("linear_axis") else ToolPoseCriteria())
            upright_kwargs = orientation_criteria_kwargs(goal, upright_policy)
            if upright_kwargs is not None:
                criteria = ToolPoseCriteria(**upright_kwargs)
            planner.update_tool_pose_criteria({goal["tcp_frame"]: criteria})
            target = Pose(position=tensor([goal["position_m"]]),
                          quaternion=tensor([goal["quaternion_wxyz"]]))
            goal_pose = GoalToolPose.from_poses({goal["tcp_frame"]: target}, num_goalset=1)
            if "joint_target_rad" in goal:
                kinematic_cfg=robot.get("robot_cfg",robot)["kinematics"]
                target_position,target_quaternion=cpu_urdf_pose(
                    kinematic_cfg["urdf_path"],request["joint_names"],goal["joint_target_rad"],
                    goal["tcp_frame"],kinematic_cfg["base_link"])
                target_pe,target_re=pose_error(target_position,target_quaternion,
                                               goal["position_m"],goal["quaternion_wxyz"])
                if target_pe>float(request.get("endpoint_position_tolerance_m",.002)) or target_re>.05:
                    raise ValueError("Joint target does not satisfy requested Cartesian TCP pose")
                joint_goal=JointState.from_position(tensor([goal["joint_target_rad"]]),joint_names=planner.joint_names)
                result=planner.plan_cspace(joint_goal,q,max_attempts=3)
            elif payload:
                from depallet.motion.curobo_seed_repair import plan_payload_pose, make_fk_orientation_ranker
                periodic_bounds=None
                orientation_ranker=None
                if upright_policy is not None:
                    kin_for_bounds=robot.get("robot_cfg",robot)["kinematics"]
                    by_name={j.get("name"):j for j in ET.parse(kin_for_bounds["urdf_path"]).getroot().findall("joint")}
                    selected_joints=[by_name[n] for n in request["joint_names"]]
                    if any(j.get("type")!="revolute" for j in selected_joints):
                        raise ValueError("Periodic IK normalization requires six revolute URDF joints")
                    periodic_bounds=[[float(j.find("limit").get(side)) for j in selected_joints]
                                     for side in ("lower","upper")]
                    orientation_ranker=make_fk_orientation_ranker(
                        kin_for_bounds,request['joint_names'],goal['tcp_frame'],goal['quaternion_wxyz'])
                candidate_validator=None
                if upright_policy is not None:
                    from depallet.validation.payload_candidate_validation import make_upright_candidate_validator
                    candidate_validator=make_upright_candidate_validator(
                        request,robot,pieces["position"],output,len(phases)+1)
                trajectory_seed_search=None
                if request.get("upright_seed_search"):
                    from depallet.planning.upright_seed_search import make_seed_search
                    trajectory_seed_search=make_seed_search(request,robot)
                result=plan_payload_pose(planner,goal_pose,q,output,goal["id"],
                                         periodic_joint_bounds=periodic_bounds,
                                         orientation_ranker=orientation_ranker,
                                         candidate_validator=candidate_validator,trajectory_seed_search=trajectory_seed_search)
            else:
                result = planner.plan_pose(goal_pose, q, max_attempts=3)
            if result is None or result.success is None or not bool(result.success.all().item()):
                from depallet.motion.curobo_diagnostics import solver_summary, compact
                details={"goal":goal,"motion":solver_summary(result),"current_q":q.position.detach().cpu().tolist()}
                if request.get("payload"):
                    ik=planner.ik_solver.solve_pose(goal_pose,current_state=q,return_seeds=32)
                    details["ik"]=solver_summary(ik)
                    details["ik_positions"]=ik.solution.detach().cpu().reshape(-1,6).tolist()
                    details["ik_success"]=ik.success.detach().cpu().reshape(-1).tolist()
                    if planner.graph_planner is not None:
                        details["graph_start_feasible"]=compact(planner.graph_planner.check_samples_feasibility(q.position))
                        details["graph_ik_feasible"]=compact(planner.graph_planner.check_samples_feasibility(ik.solution.reshape(-1,6)))
                write_json(output/(goal["id"]+"-failure-diagnostic.json"),details)
                raise RuntimeError(f"Planning failed for {goal['id']}")
            trajectory = result.get_interpolated_plan()
            if trajectory is None:
                raise RuntimeError("Planner success without interpolated trajectory")
            arrays = {}
            for key in pieces:
                value = getattr(trajectory, key, None)
                if value is None:
                    raise RuntimeError(f"Trajectory missing {key}")
                array = value.detach().cpu().numpy()
                if array.shape[-1] != 6 or any(n != 1 for n in array.shape[:-2]):
                    raise ValueError(f"Unexpected trajectory shape {array.shape}")
                array = array.reshape(-1, 6).astype(np.float64)
                if not np.isfinite(array).all():
                    raise ValueError(f"Non-finite {key}")
                arrays[key] = array
            positions = arrays["position"]
            if not 2 <= len(positions) <= 2000:
                raise ValueError("Trajectory length outside configured bounds")
            if np.max(np.abs(positions[0] - q.position.detach().cpu().numpy()[0])) > .02:
                raise ValueError("Planned path does not start at supplied state")
            if np.max(np.abs(np.diff(positions, axis=0))) > .1:
                raise ValueError("Path exceeds 0.1 rad per waypoint continuity bound")
            velocity_limit = float(request.get("maximum_velocity_rad_s", .25))
            acceleration_limit = float(request.get("maximum_acceleration_rad_s2", .5))
            velocity_peak = float(np.max(np.abs(arrays["velocity"])))
            finite_difference_velocity_peak = float(np.max(np.abs(np.diff(positions, axis=0)))/dt)
            acceleration_peak = float(np.max(np.abs(arrays["acceleration"])))
            if max(velocity_peak, finite_difference_velocity_peak) > velocity_limit + 1e-5:
                raise ValueError(f"Interpolated path exceeds {velocity_limit} rad/s: "
                                 f"{velocity_peak}, finite difference {finite_difference_velocity_peak}")
            if acceleration_peak > acceleration_limit + 1e-4:
                raise ValueError(f"Interpolated path exceeds {acceleration_limit} rad/s2")
            end = JointState.from_position(tensor([positions[-1].tolist()]),
                                           joint_names=planner.joint_names)
            actual = planner.compute_kinematics(end).tool_poses.get_link_pose(goal["tcp_frame"])
            pe, re = pose_error(actual.position.detach().cpu().numpy().reshape(3).tolist(),
                                actual.quaternion.detach().cpu().numpy().reshape(4).tolist(),
                                goal["position_m"], goal["quaternion_wxyz"])
            endpoint_tolerance = float(request.get("endpoint_position_tolerance_m", .006))
            if pe > endpoint_tolerance or re > .06:
                raise ValueError(f"Post-solve FK check failed: {pe} m, {re} rad")
            kinematic_cfg = robot.get("robot_cfg", robot)["kinematics"]
            cpu_position, cpu_quaternion = cpu_urdf_pose(
                kinematic_cfg["urdf_path"], planner.joint_names, positions[-1].tolist(),
                goal["tcp_frame"], kinematic_cfg["base_link"])
            cpu_pe, cpu_re = pose_error(cpu_position, cpu_quaternion,
                                       goal["position_m"], goal["quaternion_wxyz"])
            if cpu_pe > endpoint_tolerance or cpu_re > .06:
                raise ValueError(f"Independent CPU URDF FK failed: {cpu_pe} m, {cpu_re} rad")
            drop = 1 if phases else 0
            first = sum(len(p) for p in pieces["position"])
            for key in pieces:
                pieces[key].append(arrays[key][drop:])
            phases.append({"id": goal["id"], "first_index": first,
                           "planning_method":("MotionPlanner.plan_cspace" if "joint_target_rad" in goal else
                              "cuRobo IKSolver+TrajOptSolver with valid IK seed repair" if payload else "MotionPlanner.plan_pose"),
                           "last_index": first + len(positions) - drop - 1,
                           "endpoint_position_error_m": pe,
                           "endpoint_orientation_error_rad": re,
                           "cpu_urdf_position_error_m": cpu_pe,
                           "cpu_urdf_orientation_error_rad": cpu_re,
                           "maximum_velocity_rad_s": velocity_peak,
                           "maximum_finite_difference_velocity_rad_s": finite_difference_velocity_peak,
                           "maximum_acceleration_rad_s2": acceleration_peak,
                           "solve_wall_seconds": time.monotonic() - started})
            q = end
            q.velocity = tensor([arrays["velocity"][-1].tolist()])
        arrays = {key: np.concatenate(parts) for key,parts in pieces.items()}
        count = len(arrays["position"])
        upright_certificate = None
        if upright_policy is not None:
            upright_certificate = certify_upright_trajectory(request, arrays["position"], robot_config=robot)
            write_json(output / "upright-certificate.json", upright_certificate)
            if not upright_certificate["passed"]:
                np.savez_compressed(output / "upright-rejected-trajectory.npz",
                                    position_rad=arrays["position"], velocity_rad_s=arrays["velocity"],
                                    acceleration_rad_s2=arrays["acceleration"],
                                    time_s=np.arange(count, dtype=np.float64)*dt)
                raise ValueError("Nominal loaded trajectory failed independent upright certification")
        np.savez_compressed(output / "trajectory.npz", position_rad=arrays["position"],
                            velocity_rad_s=arrays["velocity"],
                            acceleration_rad_s2=arrays["acceleration"],
                            time_s=np.arange(count, dtype=np.float64)*dt,
                            joint_names=np.asarray(planner.joint_names, dtype="U64"))
        write_json(output / "result.json", {
            "schema": RESULT_SCHEMA, "success": True, "planner": "cuRoboV2",
            **result_execution_contract(request, diagnostic_plan=diagnostic_plan),
            "source_reference_commit": SOURCE_COMMIT,
            "curobo_version": importlib.metadata.version("nvidia-curobo"),
            "torch_version": torch.__version__, "torch_cuda": torch.version.cuda,
            "warp_version": wp.__version__, "robot_config_sha256": sha256(robot_path),
            "effective_robot_config_sha256":sha256(effective_robot_path),
            "payload_self_contact_certificate":payload_contact_certificate,
            "attachment_uncertainty":None if payload_contact_certificate is None else payload_contact_certificate.get("attachment_uncertainty"),
            "payload_sphere_padding_m":0. if not payload else payload_cover["margin_m"],
            "payload_upright_certificate":upright_certificate,
            "physical_continuous_collision_guarantee":False,
            "interpolation_dt_s": dt, "planner_random_seed": random_seed,
            "waypoints": count,
            "runtime_joint_limits":runtime_limits,"maximum_trajectory_dt_s":cfg.trajopt_solver_config.maximum_trajectory_dt,
            "duration_seconds": (count-1)*dt, "phases": phases,
            "joint_names": list(planner.joint_names),
            "payload_collision_enabled": bool(payload),
            "payload_box_id": payload["box_id"] if payload else None,
            "grasp_confirmed":payload.get("grasp_confirmed") is True if payload else False,
            "request_sha256": sha256(output / "planner-request.json"),
            "payload_dynamics_validated": False,
            "trajectory_sha256": sha256(output / "trajectory.npz"),
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-request", action="store_true")
    parser.add_argument("--diagnose-request", action="store_true")
    parser.add_argument("--diagnostic-plan", action="store_true")
    parser.add_argument("--diagnostic-seed", type=Path)
    parser.add_argument("--diagnostic-max-dt", type=float)
    parser.add_argument("--prepare-robot", action="store_true")
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--asset-path", type=Path)
    parser.add_argument("--tcp-frame")
    parser.add_argument("--base-link", default="base_link")
    args = parser.parse_args()
    if args.diagnostic_plan and (args.prepare_robot or args.check_request or args.diagnose_request):
        parser.error("--diagnostic-plan is a separate planner-only mode")
    if args.prepare_robot and (args.request or args.check_request):
        parser.error("--prepare-robot is separate from request modes")
    request = None
    if not args.prepare_robot:
        if not args.request:
            parser.error("--request is required")
        request = validate_request(json.loads(args.request.read_text()))
        if args.diagnostic_plan:
            validate_diagnostic_plan_request(request)
        from depallet.validation.payload_upright import validate_upright_request
        validate_upright_request(request)
    if args.check_request:
        print(json.dumps({"request_valid": True, "schema": request["schema"]}))
        return 0
    output = guarded_output(args.output)
    if request:
        write_json(output / "planner-request.json", request)
    try:
        if args.prepare_robot:
            prepare_robot(args, output)
        elif args.diagnose_request:
            from depallet.motion.curobo_diagnostics import diagnose
            torch, wp = configure_runtime()
            diagnose(request, output, torch, wp, args.diagnostic_seed, args.diagnostic_max_dt)
        elif args.diagnostic_plan:
            plan(request, output, diagnostic_plan=True)
        else:
            plan(request, output)
        print((output / "result.json").read_text())
        return 0
    except Exception as exc:
        write_json(output / "result.json", {"schema": RESULT_SCHEMA, "success": False,
                    "planner": "cuRoboV2",
                    **result_execution_contract(request or {}, diagnostic_plan=args.diagnostic_plan),
                    "planner_random_seed": None if request is None else planner_random_seed(request),
                    "error": str(exc),
                    "exception_type": type(exc).__name__,
                    })
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
