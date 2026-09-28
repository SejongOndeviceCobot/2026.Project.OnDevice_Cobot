"""Guarded diagnostics. Reduced-collision IK never exports a motion plan."""
from __future__ import annotations

import dataclasses
import gc
import json
import math
from pathlib import Path
import time

from depallet.motion.curobo_bridge import RESULT_SCHEMA, pose_error, sha256, vector, write_json


def compact(value, depth=0):
    if value is None or isinstance(value,(str,bool,int)):
        return value
    if isinstance(value,float):
        return value if math.isfinite(value) else str(value)
    if hasattr(value,"detach"):
        import numpy as np
        a=value.detach().cpu().numpy()
        finite=np.isfinite(a)
        out={"shape":list(a.shape),"dtype":str(a.dtype),"nonfinite":int(np.count_nonzero(~finite))}
        if a.size and finite.any():
            out.update(min=float(a[finite].min()),max=float(a[finite].max()),
                       nonzero=int(np.count_nonzero(a[finite])))
        if a.size<=64 and finite.all():
            out["values"]=a.tolist()
        return out
    if depth>=4:
        return {"type":type(value).__name__}
    if isinstance(value,dict):
        return {str(k):compact(v,depth+1) for k,v in list(value.items())[:48]}
    if isinstance(value,(list,tuple)):
        return [compact(v,depth+1) for v in value[:48]]
    if dataclasses.is_dataclass(value):
        return {f.name:compact(getattr(value,f.name),depth+1)
                for f in dataclasses.fields(value) if not f.name.startswith("_")}
    return {"type":type(value).__name__}


def solver_summary(result):
    if result is None:
        return {"returned_none":True}
    names=("success","feasible","position_error","rotation_error","cspace_error",
           "position_tolerance","orientation_tolerance","seed_cost","solve_time","total_time",
           "maximum_trajectory_dt","minimum_trajectory_dt","interpolated_last_tstep")
    out={n:compact(getattr(result,n,None)) for n in names}
    for name in ("metrics","interpolated_metrics"):
        metric=getattr(result,name,None)
        if metric is not None:
            out[name]={n:compact(getattr(metric,n,None)) for n in ("costs_and_constraints","convergence")}
    return out


def cpu_self_collisions(kin,names,joints):
    import numpy as np
    from depallet.motion.urdf_fk import matrix
    world={}
    ignore=kin.get("self_collision_ignore",{})
    for link,spheres in kin["collision_spheres"].items():
        active=[s for s in spheres if s["radius"]>0]
        if not active:
            continue
        t=matrix(kin["urdf_path"],names,joints,tip=link,base=kin["base_link"])
        centers=np.asarray([s["center"] for s in active])@t[:3,:3].T+t[:3,3]
        world[link]=(centers,np.asarray([s["radius"] for s in active]))
    pairs=[]
    for i,a in enumerate(world):
        for b in list(world)[i+1:]:
            if b in ignore.get(a,[]) or a in ignore.get(b,[]):
                continue
            ca,ra=world[a];cb,rb=world[b]
            d=np.linalg.norm(ca[:,None,:]-cb[None,:,:],axis=-1)-ra[:,None]-rb[None,:]
            if float(d.min())< -1e-6:
                pairs.append({"links":[a,b],"minimum_clearance_m":float(d.min())})
    return {"passed":not pairs,"overlapping_link_pairs":pairs}


def candidate_checks(result,request,kin,goal,horizon,max_dt):
    import numpy as np
    from depallet.planning.planning_requests import initial_world_clearance
    from depallet.motion.urdf_fk import pose
    solution=getattr(result,"solution",None)
    if solution is None:
        return []
    solutions=solution.detach().cpu().numpy().reshape(-1,6)
    successes=result.success.detach().cpu().numpy().reshape(-1)
    records=[]
    for i,joints in enumerate(solutions[:4]):
        if not np.isfinite(joints).all():
            records.append({"index":i,"finite":False});continue
        position,orientation=pose(kin["urdf_path"],request["joint_names"],joints.tolist(),
                                  goal["tcp_frame"],kin["base_link"])
        pe,re=pose_error(position,orientation,goal["position_m"],goal["quaternion_wxyz"])
        distance=np.abs(joints-np.asarray(request["start_position_rad"]))
        v=request.get("maximum_velocity_rad_s",.25)
        a=request.get("maximum_acceleration_rad_s2",.5)
        times=np.where(distance<=v*v/a,2*np.sqrt(distance/a),distance/v+v/a)
        world=initial_world_clearance(kin,request["joint_names"],joints.tolist(),request["scene"]["cuboid"])
        world["collisions"]=world["collisions"][:32]
        records.append({"index":i,"solver_success":bool(successes[i]),"joint_position_rad":joints.tolist(),
             "cpu_urdf_position_error_m":pe,"cpu_urdf_orientation_error_rad":re,
             "cpu_fk_within_2mm":pe<=.002 and re<=.05,"full_world_proxy_check":world,
             "self_collision_proxy_check":cpu_self_collisions(kin,request["joint_names"],joints.tolist()),
             "rest_to_rest_time_lower_bound_s":float(times.max()),
             "configured_maximum_duration_s":(horizon-1)*max_dt,
             "time_lower_bound_exceeds_configured_maximum":bool(times.max()>(horizon-1)*max_dt)})
    return records


def diagnose(request,output,torch,wp,cpu_seed_path=None,comparison_max_dt=None):
    import yaml
    from curobo.motion_planner import MotionPlanner,MotionPlannerCfg
    from curobo.types import GoalToolPose,JointState,Pose
    from depallet.motion.urdf_fk import pose as cpu_pose
    if request.get("payload"):
        raise ValueError("Diagnosis currently accepts unloaded requests only")
    robot=yaml.safe_load(Path(request["robot_config"]).read_text())
    kin=robot.get("robot_cfg",robot)["kinematics"]
    report={"schema":"depallet.curobo.v2.diagnostic.v1","request_sha256":sha256(output/"planner-request.json"),
            "mode":"diagnostic_only","motion_exported":False,"physical_execution_validated":False,
            "cases":[],"maximum_candidate_checks_per_goal":4}
    seed=None
    if cpu_seed_path is not None:
        data=json.loads(Path(cpu_seed_path).read_text())
        if data.get("box_id")!=request["box_id"]:
            raise ValueError("CPU seed must refer to the requested box")
        seed=vector(data["least_joint_time_found_solution"]["q_rad"],6,"CPU seed")
        report["cpu_seed"]={"path":str(cpu_seed_path),"sha256":sha256(cpu_seed_path),"q_rad":seed,
                            "collision_validated_before_diagnosis":False}
    tensor=lambda x:torch.tensor(x,dtype=torch.float32,device="cuda:0")
    modes=[("full_world_and_self",True,request["scene"],.2),
           ("self_only_ik_diagnostic",True,None,.2),
           ("kinematics_only_ik_diagnostic",False,None,.2)]
    if comparison_max_dt is not None:
        if not math.isfinite(comparison_max_dt) or not .2 < comparison_max_dt <= .4:
            raise ValueError("Diagnostic maximum dt comparison must be in (.2,.4]")
        modes=[modes[0],("full_world_longer_time",True,request["scene"],comparison_max_dt)]
        report["comparison_maximum_dt_s"]=comparison_max_dt
    for mode,self_check,scene,configured_max_dt in modes:
        started=time.monotonic()
        cfg=MotionPlannerCfg.create(robot=robot,scene_model=scene,self_collision_check=self_check,
              collision_cache={"cuboid":max(32,len(request["scene"]["cuboid"]))},
              max_batch_size=1,max_goalset=1,num_ik_seeds=32,num_trajopt_seeds=4,
              position_tolerance=float(request.get("position_tolerance_m",.001)),orientation_tolerance=.05,
              interpolation_dt=float(request.get("interpolation_dt_s",1/60)),
              interpolation_buffer_size=2000,random_seed=42)
        cfg.trajopt_solver_config.maximum_trajectory_dt=configured_max_dt
        case={"mode":mode,"world_collision":scene is not None,"self_collision":self_check,"goals":[]}
        report["cases"].append(case)
        with MotionPlanner(cfg) as planner:
            horizon=int(planner.trajopt_solver.horizon)
            max_dt=float(planner.trajopt_solver.config.maximum_trajectory_dt)
            case["time_configuration"]={"state_horizon":horizon,
                 "action_horizon":int(planner.trajopt_solver.action_horizon),
                 "minimum_trajectory_dt_s":float(planner.trajopt_solver.config.minimum_trajectory_dt),
                 "maximum_trajectory_dt_s":max_dt,"maximum_duration_s":(horizon-1)*max_dt}
            q=JointState.from_position(tensor([request["start_position_rad"]]),joint_names=request["joint_names"])
            q.velocity=tensor([request.get("start_velocity_rad_s",[0.]*6)])
            home=planner.compute_kinematics(q).tool_poses.get_link_pose(request["goals"][0]["tcp_frame"])
            hp=home.position.detach().cpu().numpy().reshape(3).tolist()
            hq=home.quaternion.detach().cpu().numpy().reshape(4).tolist()
            cp,cq=cpu_pose(kin["urdf_path"],request["joint_names"],request["start_position_rad"],
                          request["goals"][0]["tcp_frame"],kin["base_link"])
            pe,re=pose_error(hp,hq,cp,cq)
            case["home_fk_comparison"]={"gpu_position_m":hp,"gpu_quaternion_wxyz":hq,
                   "cpu_position_m":cp,"cpu_quaternion_wxyz":cq,"position_difference_m":pe,
                   "orientation_difference_rad":re,"passed":pe<1e-4 and re<1e-4}
            for index,goal in enumerate(request["goals"][:2]):
                target=Pose(position=tensor([goal["position_m"]]),quaternion=tensor([goal["quaternion_wxyz"]]))
                target=GoalToolPose.from_poses({goal["tcp_frame"]:target},num_goalset=1)
                result=planner.ik_solver.solve_pose(target,current_state=q,return_seeds=32)
                entry={"goal_id":goal["id"],"ik":solver_summary(result),
                       "independent_candidates":candidate_checks(result,request,kin,goal,horizon,max_dt)}
                case["goals"].append(entry)
                write_json(output/"diagnostics.json",report)
                print(f"DIAG {mode} {goal['id']} IK success {compact(result.success)}",flush=True)
                if index==0 and seed is not None and scene is not None and self_check:
                    seeded=planner.ik_solver.solve_pose(target,current_state=q,return_seeds=32,
                                                       seed_config=tensor([[seed]]))
                    entry["cpu_seeded_full_collision_ik"]=solver_summary(seeded)
                    entry["cpu_seeded_independent_candidates"]=candidate_checks(seeded,request,kin,goal,horizon,max_dt)
                if scene is not None and self_check and index==0 and pe<1e-4 and re<1e-4:
                    from depallet.motion.curobo_motion_diagnostics import capture_motion
                    entry["full_collision_motion_one_attempt"]=capture_motion(planner,target,q,request,output,mode)
                    write_json(output/"diagnostics.json",report)
            case["wall_seconds"]=time.monotonic()-started
        del planner,cfg
        gc.collect()
        torch.cuda.empty_cache()
    write_json(output/"diagnostics.json",report)
    write_json(output/"result.json",{"schema":RESULT_SCHEMA,"success":True,
               "mode":"diagnostic_only","planner":"cuRoboV2","motion_exported":False,
               "diagnostic_report":str(output/"diagnostics.json"),"physical_execution_validated":False})
