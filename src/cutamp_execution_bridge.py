"""Convert a cuTAMP top-layer horizon into resting targets for fresh motion planning.

No trajectory or physical-success certificate is produced. An explicitly supplied committed set permits static box supports in a later
horizon. This converter does not resume simulator state or certify load bearing.
"""
import copy
import math
import re
import numpy as np
from cutamp_grasp_bindings import pose_matrix
from photo_scene import _overlap_xy,_rectangle


def resting_horizon(plan,meta,environment,scenario,templates,completed_ids=()):
    if plan.get('provider')!='cutamp_gpu_optimization' or plan.get('satisfying_particles',0)<1:
        raise ValueError('Feasible cuTAMP candidate required')
    if plan.get('graspgen_constraints_connected') is not True:
        raise ValueError('Actual GraspGen-connected plan required')
    selected=plan['selected'];poses=selected['final_object_poses_base']
    order=[]
    for action in selected['plan_skeleton']:
        match=re.fullmatch(r'Pick\(([^,]+), [^,]+, [^)]+\)',action)
        if match:order.append(match.group(1))
    if len(set(order))!=len(order) or set(order)!=set(poses) or set(order)!=set(meta['movable_ids']):
        raise ValueError('Complete unique horizon order and poses required')
    cell=scenario['cell'];boxes={b['id']:b for b in scenario['spec']['boxes']}
    if not 1<=len(order)<=4 or not set(order)<=set(boxes):raise ValueError('Bounded top-layer horizon required')
    completed=set(completed_ids)
    if len(completed)!=len(completed_ids) or not completed<=set(boxes) or completed&set(order):
        raise ValueError('Unique disjoint completed box IDs required')
    if any(b['support_id'] in order for b in boxes.values() if b['id'] not in completed):raise ValueError('Horizon contains a covered source box')
    base=pose_matrix([*meta['base_world_position_m'],*meta['base_world_quaternion_wxyz']])
    if not np.allclose(base,pose_matrix([*cell['robot_base_position_m'],*cell['robot_base_quaternion_wxyz']]),atol=1e-5):
        raise ValueError('Robot base changed since task planning')
    anchor=cell['goal_pose'];goal=pose_matrix([*anchor[:3],math.cos(anchor[3]/2),0,0,math.sin(anchor[3]/2)])
    pallet=cell['pallet_dimensions_m'];result=[];rectangles=[];corrections={}
    env=environment['geometries']['cuboid']
    actual_goal=base@pose_matrix(env['goal_pallet']['pose'])
    expected_center=goal@np.array([0,0,pallet[2]/2,1.])
    if not np.allclose(actual_goal[:3,3],expected_center[:3],atol=1e-5) or not np.allclose(actual_goal[:3,:3],goal[:3,:3],atol=1e-5):
        raise ValueError('Goal pallet changed since task planning')
    assigned={item['On'][0]:item['On'][1] for item in environment.get('goal',[]) if 'On' in item}
    if not assigned:assigned={name:'goal_pallet' for name in order}
    if set(assigned)!=set(order) or not set(assigned.values())<={'goal_pallet',*completed}:
        raise ValueError('Every target must use the goal pallet or a committed static support')
    committed_rectangles={}
    for name in completed:
        body=env[name];local=np.linalg.inv(goal)@base@pose_matrix(body['pose'])
        if not np.allclose(local[2,:3],[0,0,1],atol=1e-4):raise ValueError('Committed support is tilted')
        if not np.allclose(body['dims'],boxes[name]['dimensions_m'],atol=1e-9):raise ValueError('Committed dimensions changed')
        rectangle={'id':name,'position_source_m':local[:3,3].tolist(),
            'yaw_source_rad':math.atan2(local[1,0],local[0,0]),'dimensions_m':body['dims']}
        if (local[2,3]-body['dims'][2]/2 < pallet[2]-.002 or
                any(abs(x)>pallet[0]/2+1e-7 or abs(y)>pallet[1]/2+1e-7 for x,y in _rectangle(rectangle))):
            raise ValueError('Committed support is not on the goal pallet')
        committed_rectangles[name]=(rectangle,local)
    for name in order:
        T=np.asarray(poses[name],float)
        if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]):raise ValueError('Invalid planned pose')
        if not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-5) or np.linalg.det(T[:3,:3])<.99999:raise ValueError('Invalid rotation')
        local=np.linalg.inv(goal)@base@T
        if not np.allclose(local[2,:3],[0,0,1],atol=1e-4):raise ValueError('Only upright pallet placements supported')
        dims=boxes[name]['dimensions_m']
        if not np.allclose(env[name]['dims'],dims,atol=1e-9):raise ValueError('Box dimensions changed')
        support=assigned[name]
        support_top=pallet[2] if support=='goal_pallet' else (
            committed_rectangles[support][1][2,3]+env[support]['dims'][2]/2)
        position=local[:3,3].tolist();resting_z=support_top+dims[2]/2
        correction=resting_z-position[2]
        if abs(correction)>.0121:raise ValueError('Plan outside declared placement-height envelope')
        position[2]=resting_z;yaw=math.atan2(local[1,0],local[0,0])
        rectangle={'id':name,'position_source_m':position,'yaw_source_rad':yaw,'dimensions_m':dims}
        if any(abs(x)>pallet[0]/2-.01+1e-7 or abs(y)>pallet[1]/2-.01+1e-7 for x,y in _rectangle(rectangle)):
            raise ValueError('Full box footprint lacks 10mm goal support reserve')
        if support!='goal_pallet':
            support_local=committed_rectangles[support][1]
            for x,y in _rectangle(rectangle):
                point=np.linalg.inv(support_local)@np.array([x,y,position[2],1.])
                if any(abs(point[i])>env[support]['dims'][i]/2+1e-7 for i in (0,1)):
                    raise ValueError('Target footprint overhangs its committed support')
        for other in [*rectangles,*(item[0] for item in committed_rectangles.values())]:
            height_gap=abs(position[2]-other['position_source_m'][2])
            vertical_overlap=height_gap < (dims[2]+other['dimensions_m'][2])/2-1e-7
            if vertical_overlap and _overlap_xy(rectangle,other):raise ValueError('Goal box volumes overlap')
        rectangles.append(rectangle)
        item=copy.deepcopy(templates[name]);item.update(position_goal_m=position,yaw_goal_rad=yaw,
            footprint_goal_m=(np.abs(local[:2,:2])@np.asarray(dims[:2])).tolist(),
            top_face_center_goal_m=[*position[:2],position[2]+dims[2]/2],
            stack_id='cutamp_'+name,support_id=support)
        result.append(item);corrections[name]=correction
    return {'schema':'depallet.cutamp_execution_horizon.v1','order':order,'placements':result,
        'resting_height_corrections_m':corrections,'goal_initially_empty_required':not bool(completed),
        'completed_ids':sorted(completed),'goal_support_assignments':assigned,
        'measured_support_stability_revalidation_required':True,
        'full_scene_box_count':len(boxes),'horizon_box_count':len(order),
        'fresh_motion_planning_required':True,'physical_execution_validated':False}


def check_followup_world(meta,environment,current_request):
    """Compare planner inputs with a caller's freshly measured frozen world."""
    if current_request is None:raise ValueError('Follow-up requires current measured scene and robot state')
    if current_request.get('length_unit')!='m' or current_request.get('quaternion_order')!='wxyz':
        raise ValueError('Current metric world required')
    if set(current_request['scene'])!={'cuboid'}:raise ValueError('Unsupported current collision geometry')
    planned=environment['geometries']['cuboid'];current=current_request['scene']['cuboid']
    if set(planned)!=set(current):raise ValueError('Collision inventory changed since task planning')
    old_base=pose_matrix([*meta['base_world_position_m'],*meta['base_world_quaternion_wxyz']])
    new_base=pose_matrix([*current_request['base_world_position_m'],*current_request['base_world_quaternion_wxyz']])
    if not np.allclose(old_base,new_base,atol=1e-6,rtol=0):raise ValueError('Measured robot base changed')
    q=np.asarray(current_request['start_position_rad'],float);prior=np.asarray(meta['q_init'],float)
    if q.shape!=prior.shape or not np.isfinite(q).all() or not np.allclose(q,prior,atol=1e-4,rtol=0):
        raise ValueError('Robot moved since task planning')
    for name in planned:
        a,b=planned[name],current[name]
        if not np.allclose(a['dims'],b['dims'],atol=1e-9,rtol=0):raise ValueError('Collision dimensions changed: '+name)
        if not np.allclose(pose_matrix(a['pose']),pose_matrix(b['pose']),atol=1e-5,rtol=0):
            raise ValueError('Collision pose changed: '+name)
    return dict(passed=True,collision_bodies_checked=len(planned),robot_joint_tolerance_rad=1e-4,
        rigid_transform_element_tolerance=1e-5,caller_must_supply_fresh_measurement=True,
        wall_clock_freshness_certified=False)


def load_execution(plan_path, input_root, scenario, planning, runs, max_transfers, *, completed_ids=(), current_request=None):
    """Rebuild a bounded execution prefix from verified planner/model artifacts."""
    import json
    import hashlib
    from pathlib import Path
    import yaml
    from cutamp_grasp_bindings import load_verified
    from modular_pick_bridge import load_surfaces
    runs=Path(runs).resolve();path=Path(plan_path).resolve();root=Path(input_root).resolve()
    if not path.is_relative_to(runs) or not root.is_relative_to(runs):
        raise ValueError('Project planner artifacts required')
    receipt=json.loads(path.with_name('exit.json').read_text())
    if receipt.get('status')!='success' or receipt.get('child_returncode')!=0:
        raise ValueError('Successful guarded cuTAMP execution required')
    learned=load_verified(path.with_name('grasp-bindings.json'),root,runs)
    meta=json.loads((root/'manifest.json').read_text())
    env=yaml.safe_load((root/'environment.yml').read_text())
    completed=list(completed_ids)
    if completed!=planning['order'][:len(completed)]:raise ValueError('Completed history differs from execution order')
    world_check=check_followup_world(meta,env,current_request) if completed or current_request is not None else None
    templates={p['box_id']:p for p in planning['packing']['placements']}
    horizon=resting_horizon(json.loads(path.read_text()),meta,env,scenario,templates,completed_ids=completed)
    if world_check is not None:horizon['current_world_check']=world_check
    if max_transfers!=horizon['horizon_box_count']:
        raise ValueError('Execution must stop at the validated cuTAMP horizon')
    packets=[load_surfaces(learned['geometry_source'],runs,s) for s in learned['sources']]
    observed=copy.deepcopy(packets[0]);observed['candidates']=[c for p in packets for c in p['candidates']]
    observed.pop('grasp_candidates_source',None);observed.pop('grasp_candidates_sha256',None)
    observed['grasp_sources']=learned['sources']
    observed['modules'].update(task_planning='cutamp_gpu_horizon',cutamp='gpu_plan_connected')
    observed['cutamp_plan_source']=str(path)
    observed['cutamp_plan_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    updated=copy.deepcopy(planning)
    updates={p['box_id']:p for p in horizon['placements']}
    updated['packing']['placements']=[updates.get(p['box_id'],p) for p in updated['packing']['placements']]
    updated['order']=completed+horizon['order']+[b for b in planning['order'] if b not in updates and b not in completed]
    updated['packing']['order']=list(updated['order'])
    updated['execution_horizon']=horizon
    updated['full_goal_layout_validated']=False
    updated['remaining_placements_executable']=False
    return updated,observed,horizon
