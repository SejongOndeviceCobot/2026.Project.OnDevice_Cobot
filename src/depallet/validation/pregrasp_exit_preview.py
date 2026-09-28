"""Nominal pre-pick vertical-clearance reach screen, not execution permission."""
import math
import numpy as np
import yaml
from pathlib import Path
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from depallet.motion.contact_escape import RobotModel, pose_matrix


def required_lift(request, margin_m=.02):
    if not math.isfinite(margin_m) or not 0<=margin_m<=.05:
        raise ValueError('Bounded finite clearance margin required')
    world=request['scene']['cuboid'];selected=request['box_id']
    up=pose_matrix([0,0,0,*request['base_world_quaternion_wxyz']])[:3,:3].T@np.array([0.,0.,1.])
    def extent(box):
        t=pose_matrix(box['pose']);dims=np.asarray(box['dims'],float)
        if dims.shape!=(3,) or not np.isfinite(dims).all() or np.any(dims<=0):raise ValueError('Positive finite box dimensions required')
        c=float(t[:3,3]@up);half=float((dims/2)@np.abs(t[:3,:3].T@up))
        return c-half,c+half
    bottom,_=extent(world[selected])
    excluded=set(request.get('completed_box_ids',[]))|{selected}
    neighbors={name:extent(box)[1] for name,box in world.items() if name.startswith('box_') and name not in excluded}
    highest=max(neighbors.values(),default=bottom-margin_m)
    return max(0.,highest+margin_m-bottom),up,neighbors


def preview(request, contact_q):
    lift,up,neighbors=required_lift(request)
    report={'schema':'depallet.pregrasp_exit_preview.v1','mode':'nominal_vertical_reach_advisory',
        'box_id':request['box_id'],'required_vertical_lift_m':lift,
        'source_box_top_projections_m':neighbors,'margin_m':.02,
        'collision_checked':False,'attachment_measured':False,'transport_planned':False,
        'execution_authorized':False,'selection_changed':False,
        'alternative_exit_routes_ruled_out':False}
    if lift>.6:
        return dict(report,status='outside_bounded_preview')
    robot=yaml.safe_load(Path(request['robot_config']).read_text())
    model=RobotModel(robot.get('robot_cfg',robot)['kinematics'],request['joint_names'])
    q=np.asarray(contact_q,float)
    initial=model.transforms(q)['suction_tcp'];steps=max(1,int(math.ceil(lift/.01)))
    checked=0
    for offset in np.linspace(0,lift,steps+1):
        target=initial.copy();target[:3,3]+=up*offset
        def residual(v):
            t=model.transforms(v)['suction_tcp']
            return np.r_[t[:3,3]-target[:3,3],Rotation.from_matrix(target[:3,:3].T@t[:3,:3]).as_rotvec()]
        fit=least_squares(residual,q,bounds=(model.lower+1e-7,model.upper-1e-7),max_nfev=80)
        err=residual(fit.x);pe=float(np.linalg.norm(err[:3]));re=float(np.linalg.norm(err[3:]))
        if pe>.001 or re>.005:
            return dict(report,status='not_found_on_current_local_ik_branch',checked_samples=checked,
                        first_failed_offset_m=float(offset),position_error_m=pe,rotation_error_rad=re)
        q=fit.x;checked+=1
    return dict(report,status='nominal_vertical_reach_found_collision_unchecked',checked_samples=checked)
