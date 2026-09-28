"""Pinned V2 API composition with explicit valid IK seed replacement.

Upstream chained boolean indexing writes a temporary tensor. This local adapter
keeps all collision costs and uses only actual successful IK states as seeds.
"""
import math
import time
from depallet.motion.curobo_bridge import write_json


def valid_seed_batch(solution, success, current_position, count, *, periodic_joint_bounds=None,
                     orientation_ranker=None, selection_diagnostics=None):
    import torch
    if solution.ndim != 3 or solution.shape[0] != 1 or solution.shape[-1] != 6:
        raise ValueError('Expected one H2017 IK batch')
    if success.shape != solution.shape[:2] or success.dtype != torch.bool:
        raise ValueError('IK success mask shape/type mismatch')
    if current_position.shape!=(1,6) or not torch.isfinite(current_position).all():
        raise ValueError('Expected one finite H2017 current joint state')
    good=solution[success].clone()
    if not len(good) or not torch.isfinite(good).all():
        raise ValueError('No finite successful IK seed')
    if not isinstance(count,int) or not 1<=count<=32:
        raise ValueError('Bounded seed count required')
    if orientation_ranker is not None and periodic_joint_bounds is None:
        raise ValueError("Orientation seed ranking requires explicit URDF periodic bounds")
    if periodic_joint_bounds is not None:
        bounds=torch.as_tensor(periodic_joint_bounds,device=good.device,dtype=torch.float64)
        if bounds.shape!=(2,6) or not torch.isfinite(bounds).all() or not bool((bounds[0]<bounds[1]).all()):
            raise ValueError('Finite ordered lower/upper bounds for all six revolute joints required')
        # Every candidate is exactly the same six physical rotations, with only
        # integer 2*pi turns changed. Never clamp a joint angle to a limit.
        raw=good.to(torch.float64);near=current_position.reshape(1,6).to(torch.float64)
        turns_min=torch.ceil((bounds[0]-raw)/(2*math.pi))
        turns_max=torch.floor((bounds[1]-raw)/(2*math.pi))
        if not bool((turns_min<=turns_max).all()):
            raise ValueError('Successful IK seed has no equivalent inside URDF limits')
        turns=torch.round((near-raw)/(2*math.pi))
        turns=torch.maximum(turns_min,torch.minimum(turns_max,turns))
        normalized=raw+2*math.pi*turns
        if not bool(((normalized>=bounds[0]) & (normalized<=bounds[1])).all()):
            raise ValueError('Periodic normalization escaped URDF joint limits')
        if orientation_ranker is not None:
            return _orientation_seed_batch(good, current_position, count, bounds,
                                           orientation_ranker, selection_diagnostics)
        good=normalized.to(good.dtype)
    costs=(good-current_position.reshape(1,6)).abs().amax(dim=-1)
    good=good[costs.argsort()]
    # Repeated numerical IK solutions must not displace a distinct branch.
    # Keep an original successful row (no averaging or angle clamping).
    unique=[]
    for row in good:
        if not unique or not bool(((torch.stack(unique)-row).abs().amax(dim=-1)<1e-4).any()):
            unique.append(row)
    if selection_diagnostics is not None:
        selection_diagnostics.update(successful_rows_before_deduplication=len(good),
            distinct_nearest_periodic_seeds=len(unique),deduplication_tolerance_rad=1e-4,
            collision_validated_path=False)
    good=torch.stack(unique)
    selected=good.repeat((math.ceil(count/len(good)),1))[:count]
    return selected.unsqueeze(0).contiguous()


def _orientation_seed_batch(good, current, count, bounds, ranker, diagnostics):
    """Rank bounded exact-angle equivalents; this does not certify a trajectory."""
    import itertools
    import numpy as np
    import torch
    raw=good.detach().cpu().double().numpy()
    limits=bounds.detach().cpu().numpy()
    start=current.detach().cpu().double().numpy().reshape(6)
    if raw.shape[0]>32 or np.any(limits[1]-limits[0]>4*math.pi+1e-4):
        raise ValueError('Periodic orientation ranking exceeds bounded H2017 search domain')
    candidates=[];origins=[];enumerated=0
    for row_index,row in enumerate(raw):
        options=[]
        for j in range(6):
            low=math.ceil((limits[0,j]-row[j])/(2*math.pi))
            high=math.floor((limits[1,j]-row[j])/(2*math.pi))
            options.append([row[j]+k*2*math.pi for k in range(low,high+1)])
        if any(not x or len(x)>3 for x in options):
            raise ValueError('Invalid or excessive equivalent-angle count')
        enumerated+=math.prod(map(len,options))
        if enumerated>2048:
            raise ValueError('Periodic orientation ranking exceeds 2048 equivalent candidates')
        for values in itertools.product(*options):
            candidate=np.asarray(values)
            # Retain a real successful IK state, not an average of duplicate rows.
            if candidates and np.any(np.max(np.abs(np.asarray(candidates)-candidate),axis=1)<1e-4):
                continue
            candidates.append(candidate);origins.append(row_index)
    values=np.asarray(candidates)
    # Rank exactly the float precision which will be passed back to cuRobo.
    actual=torch.as_tensor(values,device=good.device,dtype=good.dtype)
    if not bool(((actual.double()>=bounds[0]) & (actual.double()<=bounds[1])).all()):
        raise ValueError('Equivalent seed rounding escaped URDF limits')
    values=actual.detach().cpu().double().numpy()
    scores=np.asarray(ranker(start,values),float)
    if scores.shape!=(len(values),) or not np.isfinite(scores).all() or np.any(scores<0):
        raise ValueError('Orientation ranker must return finite nonnegative candidate scores')
    displacement=np.max(np.abs(values-start),axis=1)
    # 5 mrad buckets prevent tiny IK numerical noise from choosing extra revolutions.
    order=sorted(range(len(values)),key=lambda i:(math.ceil(scores[i]/.005),displacement[i],scores[i],i))
    chosen=(order*math.ceil(count/len(order)))[:count]
    if diagnostics is not None:
        diagnostics.update(schema='depallet.periodic_orientation_seed_ranking.v1',
            candidate_count=len(values),enumerated_before_deduplication=enumerated,
            maximum_enumerated_candidates=2048,orientation_bucket_rad=.005,
            selected_original_successful_row=[origins[i] for i in chosen],
            selected_sampled_maximum_orientation_error_rad=[float(scores[i]) for i in chosen],
            selected_maximum_joint_displacement_rad=[float(displacement[i]) for i in chosen],
            sampled_orientation_is_ranking_only=True,collision_validated_path=False,
            continuous_upright_certificate=False,
            ranker_provenance=getattr(ranker,'provenance',{}))
    return actual[chosen].unsqueeze(0).contiguous()


def make_fk_orientation_ranker(kinematics,joint_names,tool_frame,target_quaternion_wxyz):
    """Independent URDF FK; 17 q-linear samples rank seeds, never authorize motion."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    from depallet.motion.contact_escape import RobotModel
    from depallet.motion.curobo_bridge import sha256
    from pathlib import Path
    model=RobotModel(kinematics,joint_names)
    quat=np.asarray(target_quaternion_wxyz,float)
    if quat.shape!=(4,) or not np.isfinite(quat).all() or abs(np.linalg.norm(quat)-1)>1e-5:
        raise ValueError('Unit finite target quaternion required for orientation ranking')
    target=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
    def rank(start,candidates):
        scores=[]
        for goal in candidates:
            peak=0.
            for u in np.linspace(0.,1.,17):
                rotation=model.transforms(start*(1-u)+goal*u)[tool_frame][:3,:3]
                peak=max(peak,float(Rotation.from_matrix(target.T@rotation).magnitude()))
            scores.append(peak)
        return np.asarray(scores)
    rank.provenance={'kind':'independent_URDF_FK_orientation_seed_ranking',
        'sample_count_per_candidate':17,'tool_frame':tool_frame,
        'target_quaternion_wxyz':quat.tolist(),'urdf_path':kinematics['urdf_path'],
        'urdf_sha256':sha256(Path(kinematics['urdf_path'])),
        'robot_model_source_sha256':sha256(Path(__file__).with_name('contact_escape.py')),
        'full_SO3_orientation_ranked':True,'world_collision_not_evaluated_by_ranker':True}
    return rank


def plan_payload_pose(planner, goal, q, output, goal_id, max_attempts=3, *, periodic_joint_bounds=None,
                      orientation_ranker=None, candidate_validator=None, trajectory_seed_search=None):
    """Preserve legacy solver calls; optional validation is required before accept.

    This is planner-only retry. A successful legacy trajectory returns immediately;
    physical execution/contact failures never enter this fallback mechanism.

    An optional read-only callback receives (result, context) only for optimizer
    successes and must return a JSON dict with an exact bool ``passed`` field.
    A rejected candidate continues the same bounded legacy/ranked schedule.
    Validator errors stop planning; no uncertified final result is returned.
    """
    import json
    import torch
    from depallet.motion.curobo_diagnostics import solver_summary
    if isinstance(max_attempts,bool) or not isinstance(max_attempts,int) or not 1<=max_attempts<=3:
        raise ValueError('Planner attempts per mode must be bounded within 1..3')
    if candidate_validator is not None and not callable(candidate_validator):
        raise ValueError('candidate_validator must be callable or None')
    count=planner.trajopt_solver.config.num_seeds
    records=[];result=None
    modes=[('legacy_nearest_periodic',None)]
    if orientation_ranker is not None:
        modes.append(('orientation_ranked_fallback',orientation_ranker))
    for mode,active_ranker in modes:
        for attempt in range(max_attempts):
            ik=planner.ik_solver.solve_pose(goal,current_state=q.clone(),return_seeds=32 if periodic_joint_bounds is not None else count)
            row={'attempt':len(records)+1,'attempt_within_mode':attempt+1,
                 'seed_selection_mode':mode,'maximum_attempts_per_mode':max_attempts,
                 'fallback_requires_all_legacy_attempts_failed':mode=='orientation_ranked_fallback',
                 'ik_success_count':int(torch.count_nonzero(ik.success)),
                 'all_collision_costs_preserved':True,'seed_repair':'explicit gather and repeat of successful IK rows'}
            records.append(row)
            if not row['ik_success_count']:
                write_json(output/(goal_id+'-seed-repair.json'),records);continue
            ranking={}
            seeds=valid_seed_batch(ik.solution,ik.success,q.position,count,
                                  periodic_joint_bounds=periodic_joint_bounds,
                                  orientation_ranker=active_ranker,selection_diagnostics=ranking)
            if ranking:row['orientation_seed_ranking' if active_ranker is not None else 'seed_deduplication']=ranking
            row['periodic_URDF_bound_normalization']=periodic_joint_bounds is not None
            if periodic_joint_bounds is not None:
                row['periodic_joint_bounds_rad']=periodic_joint_bounds
                row['normalization_changes_only_integer_2pi_turns']=True
                row['original_successful_joint_seeds']=ik.solution[ik.success].detach().cpu().tolist()
                row['selected_maximum_joint_displacement_rad']=(seeds-q.position.reshape(1,1,6)).abs().amax(dim=-1).detach().cpu().tolist()
            row['selected_joint_seeds']=seeds.detach().cpu().tolist()
            graph=None
            if attempt==0 and trajectory_seed_search is not None:
                graph,search_receipt=trajectory_seed_search(planner,q,seeds)
                row['upright_seed_search']=search_receipt
                if graph is not None:
                    seeds=graph[:,:,-1,:].contiguous()
            if attempt>=1 and planner.graph_planner is not None:
                graph=planner._get_graph_seed_trajectories(q.clone(),seeds)
                row['graph_seed_returned']=graph is not None
                if graph is None:
                    write_json(output/(goal_id+'-seed-repair.json'),records);continue
            started=time.monotonic()
            from depallet.motion.curobo_motion_diagnostics import capture_optimizer_constraints
            row['optimization_iterations_before_topk']=[]
            with capture_optimizer_constraints(planner.trajopt_solver,row['optimization_iterations_before_topk']):
                result=planner.trajopt_solver.solve_pose(goal,q.clone(),seed_config=seeds,seed_traj=graph,
                    use_implicit_goal=True,finetune_attempts=3 if graph is not None else 1,
                    finetune_dt_scale=.75 if graph is not None else .55)
            row.update(solve_wall_seconds=time.monotonic()-started,result=solver_summary(result))
            write_json(output/(goal_id+'-seed-repair.json'),records)
            if result is not None and result.success is not None and bool(result.success.all()):
                if candidate_validator is None:
                    return result
                context={'goal_id':goal_id,'attempt':row['attempt'],
                         'attempt_within_mode':row['attempt_within_mode'],
                         'seed_selection_mode':mode,'maximum_attempts_per_mode':max_attempts}
                try:
                    validation=candidate_validator(result,dict(context))
                    if not isinstance(validation,dict) or type(validation.get('passed')) is not bool:
                        raise ValueError('Candidate validator must return a dict with a bool passed field')
                    json.dumps(validation,allow_nan=False)
                except Exception as exc:
                    row['candidate_validation']={'passed':False,'validator_error':True,
                        'exception_type':type(exc).__name__,'error':str(exc),'context':context}
                    write_json(output/(goal_id+'-seed-repair.json'),records)
                    raise ValueError('Candidate validator failed closed: '+str(exc)) from exc
                row['candidate_validation']=validation
                row['candidate_validation_context']=context
                write_json(output/(goal_id+'-seed-repair.json'),records)
                if validation['passed']:
                    return result
    # In opt-in mode a solver success alone must never escape the validator.
    return None if candidate_validator is not None else result
