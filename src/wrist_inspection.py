"""Opt-in oracle-planned wrist inspection, with recorded-frame admission only.

Geometry here chooses a simulated camera viewpoint. It is not a perception
estimate, collision certificate, hardware mount model, or grasp authorization.
"""
from __future__ import annotations
import copy
import numpy as np
from camera_rig import camera_specs, look_at_cv
from depallet_execution_plan import transform, pose_from_transform
from point2pose_adapter import checked_transform

POLICY = 'wrist_oblique_v1'
POLICIES = ('wrist_top_v1', 'wrist_oblique_v1', 'wrist_pallet_v1')
STANDOFF_M = .65


def inspection_request(request, T_flange_tcp, *, policy=POLICY, survey_candidate_id="oblique"):
    if policy not in POLICIES:raise ValueError("Unknown inspection viewpoint policy")
    if policy == "wrist_pallet_v1":
        from pallet_survey import survey_request
        return survey_request(request,T_flange_tcp,candidate_id=survey_candidate_id)
    if request.get('payload') is not None:
        raise ValueError('Inspection requires an unloaded gripper')
    result = copy.deepcopy(request)
    ids = [g['id'] for g in result['goals']]
    if ids not in (['pregrasp', 'contact'], ['empty_tool_retreat', 'pregrasp', 'contact']):
        raise ValueError('Inspection expects the canonical unloaded approach goals')
    tcp_local = checked_transform(T_flange_tcp, 'T_flange_tcp')
    wrist = next(x for x in camera_specs() if x['id'] == 'wrist')
    target = np.asarray(request['grasp_tcp_world_pose'][:3], dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError('Finite oracle target required for diagnostic viewpoint')
    # Both policies centre the optical frame on the target; the oblique
    # policy preserves the established suction tool orientation.
    if policy=='wrist_top_v1':
        camera_world = look_at_cv(target + [0., 0., STANDOFF_M], target, (0., 1., 0.))
    else:
        # Preserve the canonical suction orientation to avoid folding the wrist
        # merely to point its off-axis camera straight down. Centre the target
        # by translating back along the existing calibrated optical axis.
        grasp=request['grasp_tcp_world_pose']
        grasp_rotation=transform(grasp[:3],grasp[3:])[:3,:3]
        flange_rotation=grasp_rotation @ tcp_local[:3,:3].T
        camera_world=np.eye(4)
        camera_world[:3,:3]=flange_rotation @ np.asarray(wrist['T_parent_camera_cv'])[:3,:3]
        camera_world[:3,3]=target-STANDOFF_M*camera_world[:3,2]
    flange_world = camera_world @ np.linalg.inv(np.asarray(wrist['T_parent_camera_cv']))
    tcp_world = flange_world @ tcp_local
    if not .2 < tcp_world[2, 3] < 2.5:
        raise ValueError('Inspection TCP exceeds bounded workcell height')
    base_world = transform(request['base_world_position_m'], request['base_world_quaternion_wxyz'])
    p = pose_from_transform(np.linalg.inv(base_world) @ tcp_world)
    goal = dict(id='inspection', tcp_frame='suction_tcp', position_m=p[:3], quaternion_wxyz=p[3:])
    result['goals'].insert(ids.index('pregrasp'), goal)
    result['inspection_view'] = dict(policy=policy, camera_id='wrist', standoff_m=STANDOFF_M,
        planned_T_world_camera_cv=camera_world.tolist(), planned_tcp_world_pose=pose_from_transform(tcp_world),
        target_world_m=target.tolist(), viewpoint_source='simulation_oracle_diagnostic',
        camera_mount_collision_modelled=False, visibility_validated=False,
        perception_controls_robot=False)
    return result


def inspection_boundary(trajectory, request):
    if request.get('inspection_view', {}).get('policy') not in POLICIES:
        raise ValueError('Explicit inspection policy required')
    phases = trajectory.result.get('phases', [])
    if [p.get('id') for p in phases] != [g['id'] for g in request['goals']]:
        raise ValueError('Planner phase IDs differ from bound request goals')
    previous = -1
    for phase in phases:
        first, last = phase.get('first_index'), phase.get('last_index')
        if (type(first) is not int or type(last) is not int or first != previous+1
                or last < first or last >= len(trajectory.times)):
            raise ValueError('Planner phase boundaries are not a complete partition')
        previous = last
    if previous != len(trajectory.times)-1:
        raise ValueError('Planner phase suffix is missing')
    indices = [p['last_index'] for p in phases if p['id'] == 'inspection']
    if len(indices) != 1 or not 0 < indices[0] < len(trajectory.times)-1:
        raise ValueError('One interior inspection endpoint required')
    if np.max(np.abs(trajectory.velocities[indices[0]])) > 1e-5:
        raise ValueError('Inspection endpoint must have zero planned velocity')
    return indices[0]


class InspectionFrames:
    """Admit only accepted, capture-time frames during the measured stable hold."""
    def __init__(self, hold_start):
        if not np.isfinite(hold_start) or hold_start < 0:
            raise ValueError('Finite nonnegative hold time required')
        self.hold_start = float(hold_start)
        self.rows = {'overhead': [], 'wrist': []}

    def append(self, camera_id, frame_index, image_time):
        if camera_id not in self.rows or type(frame_index) is not int or frame_index < 0:
            raise ValueError('Policy camera and nonnegative integer frame required')
        if not np.isfinite(image_time):
            raise ValueError('Finite image time required')
        if image_time < self.hold_start:
            return False
        rows = self.rows[camera_id]
        if rows and (frame_index != rows[-1]['frame_index']+1
                     or abs(image_time-rows[-1]['image_sim_time']-1/30) > 1e-6):
            raise ValueError('Inspection capture gap or duplicate')
        rows.append(dict(frame_index=frame_index, image_sim_time=float(image_time)))
        return True

    def result(self):
        a, b = self.rows['overhead'], self.rows['wrist']
        pairs = len(a) == len(b) and all(x['image_sim_time'] == y['image_sim_time'] for x, y in zip(a, b))
        span = a[-1]['image_sim_time']-a[0]['image_sim_time'] if a else 0.
        return dict(ready=bool(pairs and len(a) >= 16 and span >= .5-1e-9),
            hold_started_sim_time=self.hold_start, paired_frame_count=len(a) if pairs else 0,
            exact_timestamp_pairs=bool(pairs), recorded_span_s=span,
            cameras=copy.deepcopy(self.rows), semantic_or_fov_success_claimed=False)
