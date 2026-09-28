"""Integration baseline: saved SAM3.1 surface targets + measured simulation world.

This explicit hybrid uses known dimensions/orientation and oracle association.
Optional GraspGen supplies contact positions. This is not live perception,
cuTAMP, or an independent pose validation.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from depallet_execution_plan import transform
from curobo_bridge import validate_request


def load_surfaces(path, runs, grasp_candidates=None):
    path = Path(path).resolve()
    if not path.is_relative_to(Path(runs).resolve()):
        raise ValueError('Surface input must be under project runs')
    raw = path.read_bytes()
    packet = json.loads(raw)
    candidates = []
    for region in packet['regions']:
        g = region['geometry']
        if not g['top_pose_usable']:
            continue
        if (g['provenance']['mask_provider'] != 'sam3.1_multiplex_video_predictor' or
                g['provenance']['ground_truth_read'] is not False):
            raise ValueError('Expected actual SAM3.1 RGB-D geometry')
        top = np.asarray(g['observed_footprint']['centre_world_m'], float)
        ext = np.asarray(g['observed_footprint']['observed_extent_m'], float)
        normal = np.asarray(g['top_plane']['normal_world'], float)
        if top.shape != (3,) or ext.shape != (2,) or normal.shape != (3,) or not np.isfinite(np.r_[top,ext,normal]).all():
            raise ValueError('Invalid observed surface')
        # Nominal VGP20 rectangular pad envelope, axes fixed by this V1 recipe.
        if np.min(ext) < .1841+.02 or np.max(ext) < .2684+.02 or normal[2] < .999:
            continue
        candidates.append({'sam31_object_id': region['sam31_object_id'],
                           'top_world_m': top.tolist(), 'extent_m': ext.tolist(),
                           'capture': g['capture']})
    if not candidates:
        raise ValueError('No observed surface supports nominal VGP20 envelope')
    result = {'schema':'depallet.hybrid_surface_bridge.v1', 'source':str(path),
            'sha256':hashlib.sha256(raw).hexdigest(), 'candidates':candidates,
            'modules': {'segmentation':'saved_sam3.1_gpu', 'geometry':'observed_rgbd_top',
                        'grasp':'geometric_top_center_v1', 'task_planning':'existing_rule_packing',
                        'motion_planning':'curobo_v2', 'execution':'isaac_surface_gripper',
                        'graspgen':'not_connected', 'cutamp':'not_connected'},
            'obstacle_and_identity_source':'measured_simulation_oracle',
            'orientation_and_dimensions_source':'known_scenario',
            'live_perception':False}
    if grasp_candidates is not None:
        from graspgen_vgp20_adapter import select_vgp20_candidate
        gp=Path(grasp_candidates).resolve()
        if not gp.is_relative_to(Path(runs).resolve()):raise ValueError('Project grasp result required')
        learned=json.loads(gp.read_text())
        receipt=json.loads(gp.with_name('exit.json').read_text())
        if receipt.get('status')!='success' or receipt.get('child_returncode')!=0:
            raise ValueError('Successful guarded GraspGen execution required')
        if learned.get('provider')!='graspgen_suction' or learned.get('frame')!='world':
            raise ValueError('World-frame GraspGen candidates required')
        sequence=path.parent/'input/sequence.json'
        if hashlib.sha256(sequence.read_bytes()).hexdigest()!=learned['source_sequence_sha256']:
            raise ValueError('GraspGen must use the same model-mask RGB-D packet')
        sid=learned['initialization_provenance']['sam31_object_id']
        region=next(r for r in packet['regions'] if r['sam31_object_id']==sid)
        if region['geometry']['capture']['frame_id']!=learned['source_frame_id']:
            raise ValueError('Grasp/geometry frame mismatch')
        review=select_vgp20_candidate(learned['candidates'],region['geometry'],yaw_policy=learned.get('vgp20_yaw_policy','fixed'),position_policy=learned.get('vgp20_position_policy','score'))
        if review['selected'] is None:raise ValueError('No learned suction candidate fits the VGP20 envelope')
        candidate=next(c for c in candidates if c['sam31_object_id']==sid)
        candidate['grasp_world_m']=review['selected']['projected_contact_world_m']
        if 'grasp_world_yaw_rad' in review['selected']:
            candidate['grasp_world_yaw_rad']=review['selected']['grasp_world_yaw_rad']
        candidate['graspgen_review']=review
        result['candidates']=[candidate]
        result['modules']['grasp']=('graspgen_suction_position_observed_footprint_yaw' if 'grasp_world_yaw_rad' in candidate else 'graspgen_suction_position_fixed_v1_orientation')
        result['modules']['graspgen']='gpu_inference_connected'
        result['grasp_candidates_source']=str(gp)
        result['grasp_candidates_sha256']=hashlib.sha256(gp.read_bytes()).hexdigest()
    return result


def apply_surface(request, plan, packet):
    """Change actual approach endpoints; preserve full-world collision checks."""
    request, plan = copy.deepcopy(request), copy.deepcopy(plan)
    nominal = np.asarray(request['grasp_tcp_world_pose'][:3], float)
    nominal_top = nominal - np.array([0.,0.,.001])
    ranked = sorted((float(np.linalg.norm(np.asarray(c['top_world_m'])-nominal_top)), i)
                    for i,c in enumerate(packet['candidates']))
    if ranked[0][0] > .02 or (len(ranked)>1 and ranked[1][0]-ranked[0][0]<.05):
        raise ValueError('Saved surface does not unambiguously match current simulated target')
    candidate = packet['candidates'][ranked[0][1]]
    delta = np.asarray(candidate.get('grasp_world_m',candidate['top_world_m']))-nominal_top
    if abs(delta[2]) > .002:
        raise ValueError('Observed top height inconsistent with current V1 scene')
    base = transform(request['base_world_position_m'], request['base_world_quaternion_wxyz'])
    local_delta = base[:3,:3].T@delta
    for goal in request['goals']:
        if goal['id'] in ('pregrasp','contact'):
            if 'joint_target_rad' in goal:
                raise ValueError('Observed target requires fresh Cartesian planning')
            goal['position_m'] = (np.asarray(goal['position_m'])+local_delta).tolist()
    for key in ('grasp_tcp_world_pose','pregrasp_tcp_world_pose'):
        request[key][:3] = (np.asarray(request[key][:3])+delta).tolist()
    if 'grasp_world_yaw_rad' in candidate:
        from scipy.spatial.transform import Rotation
        yaw=float(candidate['grasp_world_yaw_rad'])
        world_rotation=Rotation.from_euler('z',yaw)*Rotation.from_euler('x',np.pi)
        q=world_rotation.as_quat()[[3,0,1,2]].tolist()
        local_q=(Rotation.from_matrix(base[:3,:3]).inv()*world_rotation).as_quat()[[3,0,1,2]].tolist()
        for goal in request['goals']:
            if goal['id'] in ('pregrasp','contact'):goal['quaternion_wxyz']=local_q
        for key in ('grasp_tcp_world_pose','pregrasp_tcp_world_pose'):request[key][3:]=q
    receipt = {'backend':'sam31_rgbd_geometric_hybrid', 'box_id':request['box_id'],
               'selected_surface':candidate, 'source':packet['source'], 'source_sha256':packet['sha256'],
               'association_distance_m':ranked[0][0], 'target_delta_world_m':delta.tolist(),
               'simulator_identity_association_used':True, 'live_perception':False,
               'full_pad_contact_certified':False, 'modules':packet['modules']}
    request['pose_source'] = 'sam31_rgbd_geometric_hybrid'
    request['observed_surface_bridge'] = receipt
    plan['perception_source'] = 'sam31_rgbd_geometric_hybrid'
    plan['observed_surface_bridge'] = receipt
    plan['approach_request_data'] = copy.deepcopy(request)
    validate_request(request)
    return plan, request, receipt
