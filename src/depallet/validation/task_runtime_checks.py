"""Measured whole-scene gates for sequential simulated transfers; no actuator calls."""
from __future__ import annotations
import copy
import math
import numpy as np
from depallet.validation.source_stack_integrity import validate_remaining_source_stack
from depallet.planning.depallet_execution_plan import transform, pose_from_transform
from depallet.motion.curobo_bridge import pose_error


def validate_task_state(spec, packing, box_ids, state, *, source_pose, goal_pose,
                        completed_ids, active_id=None):
    expected = {b['id'] for b in spec['boxes']}
    completed = list(completed_ids)
    if (set(box_ids) != expected or len(box_ids) != len(expected)
            or len(set(completed)) != len(completed) or not set(completed) <= expected
            or active_id is not None and (active_id not in expected or active_id in completed)):
        raise ValueError('Measured task IDs, completed IDs or active target invalid')
    placements = {p['box_id']:p for p in packing['placements']}
    if set(placements) != expected: raise ValueError('Whole task packing required')
    source = validate_remaining_source_stack(spec, box_ids, state,
        source_pose=source_pose, excluded_ids=completed+([] if active_id is None else [active_id]))
    goal_spec=copy.deepcopy(spec)
    for box in goal_spec['boxes']:
        item=placements[box['id']]
        box['support_id']='source_pallet' if item['support_id']=='goal_pallet' else item['support_id']
        box['position_source_m']=item['position_goal_m']
        box['yaw_source_rad']=item['yaw_goal_rad']
    goal=validate_remaining_source_stack(goal_spec,box_ids,state,source_pose=goal_pose,
        excluded_ids=[name for name in box_ids if name not in completed])
    placement_checks=[]
    c,s=math.cos(goal_pose[3]/2),math.sin(goal_pose[3]/2)
    T_world_goal=transform(goal_pose[:3],[c,0,0,s])
    for name in completed:
        index=box_ids.index(name);item=placements[name];a=item['yaw_goal_rad']/2
        pose=pose_from_transform(T_world_goal@transform(item['position_goal_m'],[math.cos(a),0,0,math.sin(a)]))
        pe,re=pose_error(state['positions_m'][index],state['quaternions_wxyz'][index],pose[:3],pose[3:])
        placement_checks.append({'box_id':name,'position_error_m':pe,'orientation_error_rad':re,
                                 'passed':pe<=.008 and re<=math.radians(3)})
    from depallet.scene.scenario_suite import validate_measured_goal_com
    goal_com=validate_measured_goal_com(spec,packing,completed,box_ids,state,goal_pose=goal_pose)
    return {'schema':'depallet.measured_task_state.v1',
        'passed':source['passed'] and goal['passed'] and goal_com['passed'] and all(x['passed'] for x in placement_checks),
        'placed_goal_com':goal_com,
        'measurement_source':'isaac_runtime_simulation_oracle',
        'state_sim_time_s':state['sim_time'],'physics_step':state['physics_step'],
        'remaining_source':source,'placed_goal':goal,'placement_checks':placement_checks,
        'completed_ids':completed,'active_target_excluded':active_id,
        'all_boxes_at_goal':len(completed)==len(expected),'poses_restored_or_fabricated':False,
        'scope':'measured support, tilt, velocity and committed-goal poses at this snapshot',
        'whole_history_no_contact_claimed':False}


def commit_transfer(previous_ids, expected_order, result, whole_state_gate, *, box_id):
    completed=list(previous_ids)
    if completed!=list(expected_order[:len(completed)]) or len(completed)>=len(expected_order) or expected_order[len(completed)]!=box_id:
        raise ValueError('Duplicate, skipped or out-of-order commit')
    required=('passed','attachment_confirmed','measured_lift_confirmed','release_confirmed')
    if any(result.get(k) is not True for k in required) or result.get('state')!='DONE':
        raise ValueError('Actual full physical cycle evidence required')
    if result.get('settled_seconds',0)<.5:raise ValueError('Actual continuous goal settle required')
    if (whole_state_gate.get('passed') is not True
            or whole_state_gate.get('completed_ids')!=completed+[box_id]
            or whole_state_gate.get('active_target_excluded') is not None):
        raise ValueError('Fresh complete source/goal support evidence required')
    events=result.get('events',[])
    if not events or whole_state_gate['state_sim_time_s']<=events[-1]['sim_time_s']:
        raise ValueError('Commit requires a fresh post-DONE measured state')
    return completed+[box_id]


def module_progress(state, completed, total):
    scores={'IDLE':0,'APPROACH':1,'PREGRASP_SETTLE':1,'CONTACT_READY':1,'CLOSING':1,
            'GRASP_CONFIRMED':2,'ESCAPE':2,'ESCAPED':3,'TRANSPORT':3,'PRE_RELEASE_SETTLE':3,'RELEASING':3,
            'RELEASED':4,'DONE':5}
    sparse=scores.get(state,0)
    return {'binary_task_success':int(completed==total), 'module_sparse_stage_0_to_5':sparse,
            'task_progress_0_to_1':min(1.,(completed+sparse/5)/total),
            'physical_commits':completed,'total_boxes':total,
            'reward_used_for_training':False,'dense_score_kind':'observable milestone progress, not learned reward'}


def measured_upright_check(quaternion_wxyz, limit_rad=.03):
    q=np.asarray(quaternion_wxyz,dtype=float)
    if q.shape!=(4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>1e-4:
        raise ValueError('Actual normalized box quaternion required')
    q=q/np.linalg.norm(q)
    angle=math.acos(float(np.clip(1-2*(q[1]**2+q[2]**2),-1,1)))
    return {'passed':angle<=limit_rad,'world_tilt_rad':angle,'maximum_world_tilt_rad':limit_rad,
            'measurement_source':'actual_rigid_body_quaternion','continuous_between_samples_guarantee':False}


def measured_path_progress(joints_rad, path_positions_rad):
    """Actual joint projection onto a checked path; elapsed time is not progress."""
    q=np.asarray(joints_rad,float);path=np.asarray(path_positions_rad,float)
    if q.shape!=(6,) or path.ndim!=2 or path.shape[1]!=6 or len(path)<2 or not np.isfinite(q).all() or not np.isfinite(path).all():
        raise ValueError('Finite actual joints and checked six-axis path required')
    starts=path[:-1];delta=np.diff(path,axis=0);length=np.linalg.norm(delta,axis=1)
    denominator=np.sum(delta*delta,axis=1)
    fractions=np.clip(np.divide(np.sum((q-starts)*delta,axis=1),denominator,
        out=np.zeros_like(denominator),where=denominator>1e-16),0,1)
    distances=np.linalg.norm(q-(starts+fractions[:,None]*delta),axis=1)
    index=int(np.argmin(distances));total=float(length.sum());prefix=np.r_[0.,np.cumsum(length)]
    progress=0. if total<1e-12 else float((prefix[index]+fractions[index]*length[index])/total)
    return {'progress_0_to_1':progress,'distance_to_path_rad':float(distances[index]),
        'path_segment_index':index,'actual_joint_measurement_used':True,'wall_clock_used':False,
        'completion_certified':False}


def measured_release_pose_check(box_state, goal_position_m, goal_quaternion_wxyz, release_gap_m,
                                *, measurement_source):
    """Actual pose above the checked physical goal; no landing guarantee."""
    if measurement_source not in ('isaac_runtime','controlled_test_stub'):
        raise ValueError('Explicit actual runtime or controlled test provenance required')
    def vector(value, shape):
        a=np.asarray(value,float)
        if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Finite release pose required')
        return a
    p=vector(box_state['position_m'],(3,));q=vector(box_state['quaternion_wxyz'],(4,))
    target=vector(goal_position_m,(3,)).copy();g=vector(goal_quaternion_wxyz,(4,))
    if any(abs(np.linalg.norm(a)-1)>1e-4 for a in (q,g)):
        raise ValueError('Normalized release quaternions required')
    q=q/np.linalg.norm(q);g=g/np.linalg.norm(g)
    if np.linalg.norm(g[1:3])>1e-8:raise ValueError('Upright nominal release goal required')
    if isinstance(release_gap_m,bool) or not isinstance(release_gap_m,(int,float)) or not math.isfinite(release_gap_m) or not .005<=release_gap_m<=.08:
        raise ValueError('Checked payload-envelope release gap must be 5..80 mm')
    target[2]+=release_gap_m
    measured_rotation=transform([0.,0.,0.],q)[:3,:3]
    goal_rotation=transform([0.,0.,0.],g)[:3,:3]
    relative=goal_rotation.T@measured_rotation
    thresholds={'xy_m':.00075,'z_m':.004,'world_tilt_rad':.001,'yaw_rad':.003}
    errors={'xy_m':float(np.linalg.norm(p[:2]-target[:2])), 'z_m':float(abs(p[2]-target[2])),
        'world_tilt_rad':float(math.atan2(np.linalg.norm(measured_rotation[:2,2]),measured_rotation[2,2])),
        'yaw_rad':float(abs(math.atan2(relative[1,0],relative[0,0])))}
    gates={'xy':errors['xy_m']<=thresholds['xy_m'],'z':errors['z_m']<=thresholds['z_m'],
           'world_tilt':errors['world_tilt_rad']<=thresholds['world_tilt_rad'],
           'yaw':errors['yaw_rad']<=thresholds['yaw_rad']}
    return {'schema':'depallet.prerelease_pose_check.v1','passed':all(gates.values()),
        'measurement_source':measurement_source,'measured_box_position_m':p.tolist(),
        'measured_box_quaternion_wxyz':q.tolist(),'expected_box_position_m':target.tolist(),
        'expected_box_quaternion_wxyz':g.tolist(),'errors':errors,'thresholds':thresholds,'gates':gates,
        'release_above_physical_goal_m':release_gap_m,'physical_goal_position_m':list(goal_position_m),
        'landing_pose_certified':False,'continuous_between_samples_guarantee':False}
