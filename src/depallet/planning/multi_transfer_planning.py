"""CPU request builders for repeated, measured-state simulated transfers.

Every call includes every dynamic box, including committed goal boxes. This
module does not move actors, invoke GPU planners, infer perception, or authorize
execution. Source support and goal support are distinct, explicit IDs.
"""
from __future__ import annotations
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from depallet.motion.actual_contact_escape import obb_separation
from depallet.motion.curobo_bridge import REQUEST_SCHEMA, pose, validate_request, vector
from depallet.planning.depallet_execution_plan import transform, pose_from_transform, payload_request
from depallet.manipulation.payload_self_contact import attachment_uncertainty_policy
from depallet.planning.planning_requests import relative_pose, quaternion, rotation
from depallet.planning.motion_profiles import apply_motion_profile, resolve_motion_profile

JOINT_NAMES = [f'joint_{i}' for i in range(1, 7)]
SIGNS = np.array(list(itertools.product((-1., 1.), repeat=3)))
DEFAULT_UNCERTAINTY = dict(translation_m=.004, rotation_rad=.01, payload_padding_m=.0075)


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def anchor_pose(value):
    """Pallet bottom-origin pose: explicit xyz+yaw or xyz+wxyz."""
    value = list(value)
    if len(value) == 4:
        value = vector(value, 4, 'pallet xyz+yaw')
        value = value[:3]+[math.cos(value[3]/2), 0., 0., math.sin(value[3]/2)]
    return pose(value, 'world anchor')


def _matrix_pose(value):
    value = pose(list(value), 'world pose')
    return transform(value[:3], value[3:])


def _obb(matrix, dimensions):
    half = np.asarray(dimensions)/2
    return dict(center=matrix[:3, 3], rotation=matrix[:3, :3], half=half,
                vertices=(SIGNS*half)@matrix[:3, :3].T+matrix[:3, 3])


def _catalog(spec, box_ids, state):
    ids = list(box_ids)
    boxes = {b['id']: b for b in spec['boxes']}
    box_count = len(spec['boxes'])
    if (box_count < 1 or len(ids) != box_count or len(set(ids)) != box_count
            or len(boxes) != box_count or set(ids) != set(boxes)):
        raise ValueError(
            f'Exactly all {box_count} unique declared boxes must remain in each measured scene')
    if 'box_ids' in state and state['box_ids'] != ids:
        raise ValueError('Measured box ID order changed')
    if (type(state.get('physics_step')) is not int or state['physics_step'] < 0
            or isinstance(state.get('sim_time'), bool) or not isinstance(state.get('sim_time'), (int, float))
            or not math.isfinite(state['sim_time']) or state['sim_time'] < 0):
        raise ValueError('Actual snapshot requires finite sim_time and integer physics_step')
    arrays = {}
    for key, width in [('positions_m', 3), ('quaternions_wxyz', 4),
                       ('linear_velocities_m_s', 3), ('angular_velocities_rad_s', 3)]:
        a = np.asarray(state.get(key))
        if a.shape != (box_count, width) or a.dtype.kind not in 'iuf' or not np.isfinite(a).all():
            raise ValueError('Incomplete or nonfinite measured '+key)
        arrays[key] = a.astype(float)
    world = {}
    for i, name in enumerate(ids):
        dims = vector(list(boxes[name]['dimensions_m']), 3, name+' dimensions')
        if min(dims) < .01 or max(dims) > 2:
            raise ValueError('Box dimensions outside bounded mockup range')
        world[name] = _obb(transform(arrays['positions_m'][i], arrays['quaternions_wxyz'][i]), dims)
    return ids, boxes, world, arrays


def _mount_cuboids(kin, measured_base_pose, nominal_mount_pose):
    """Same five-solid reviewed recess, transformed with the explicit fixture."""
    mount = _matrix_pose(nominal_mount_pose)
    if not np.allclose(mount[:3, :3], np.eye(3), atol=1e-7, rtol=0):
        raise ValueError('Reviewed pedestal fixture must remain upright and unrotated')
    if abs(mount[2, 3]-.25) > 1e-6:
        raise ValueError('Reviewed pedestal is 0.25 m high')
    current = _matrix_pose(measured_base_pose)
    if np.max(np.abs(current-mount)) > .001:
        raise ValueError('Measured base differs from the fixed pedestal mount')
    spheres = kin['collision_spheres']['base_link']
    penetrating = [s for s in spheres if s['radius'] > 0 and s['center'][2]-s['radius'] < 0]
    if not penetrating:
        raise ValueError('Reviewed fixed-base recess requires the actual base collision spheres')
    lo = np.min([np.asarray(s['center'])-s['radius'] for s in penetrating], axis=0)-.003
    hi = np.max([np.asarray(s['center'])+s['radius'] for s in penetrating], axis=0)+.003
    hi[2] = 0.
    if not (-.275 < lo[0] < hi[0] < .275 and -.275 < lo[1] < hi[1] < .275 and -.05 <= lo[2] <= -.02):
        raise ValueError('Unexpected fixed-base recess dimensions')
    a, b = np.array([-.275, -.275, -.25]), np.array([.275, .275, 0.])
    regions = {'pedestal_bottom': (a, [b[0], b[1], lo[2]]),
        'pedestal_rim_left': ([a[0], a[1], lo[2]], [lo[0], b[1], 0.]),
        'pedestal_rim_right': ([hi[0], a[1], lo[2]], b),
        'pedestal_rim_front': ([lo[0], a[1], lo[2]], [hi[0], lo[1], 0.]),
        'pedestal_rim_back': ([lo[0], hi[1], lo[2]], [hi[0], b[1], 0.])}
    cuboids = {}
    for name, (low, high) in regions.items():
        low, high = np.asarray(low), np.asarray(high)
        local = np.eye(4); local[:3, 3] = (low+high)/2
        cuboids[name] = {'dims': (high-low).tolist(), 'pose_world': pose_from_transform(mount@local)}
    evidence = dict(type='fixed_base_mount_contact_only', physical_pedestal_dimensions_m=[.55, .55, .25],
        nominal_mount_pose_world=list(nominal_mount_pose), pocket_min_nominal_base_m=lo.tolist(),
        pocket_max_nominal_base_m=hi.tolist(), fitted_base_maximum_under_plane_m=float(-lo[2]-.003),
        robot_collision_links_disabled=[], all_base_spheres_retained=True,
        scope='fixed bolted base only; recess is not traversable workspace',
        static_fixture_approximation_requires_execution_contact_monitor=True)
    return cuboids, evidence


def measured_world(*, spec, box_ids, state, base_pose, source_pose, goal_pose, robot_config,
                   nominal_mount_pose=None, extra_static_world=None):
    """Build all dynamic+static cuboids in measured base coordinates."""
    ids, boxes, world, arrays = _catalog(spec, box_ids, state)
    base_pose = pose(list(base_pose), 'measured base pose')
    source_pose, goal_pose = anchor_pose(source_pose), anchor_pose(goal_pose)
    config = Path(robot_config).resolve()
    value = yaml.safe_load(config.read_text()); kin = value.get('robot_cfg', value)['kinematics']
    if kin['base_link'] != 'base_link' or kin['cspace']['joint_names'] != JOINT_NAMES:
        raise ValueError('Reviewed H2017 base_link and joint order required')
    for link in ('vgp20_body', 'vgp20_adapter'):
        if link not in kin['collision_link_names'] or not kin['collision_spheres'].get(link):
            raise ValueError('Full gripper collision geometry required')
    pallet_dims = vector(list(spec['pallet']['dimensions_m']), 3, 'pallet dimensions')
    if min(pallet_dims) <= 0:
        raise ValueError('Positive pallet dimensions required')
    static = {'ground': {'dims': [7., 7., .05], 'pose_world': [0., 0., -.025, 1., 0., 0., 0.]},
              'back_wall': {'dims': [7., .08, 3.], 'pose_world': [0., 2.4, 1.5, 1., 0., 0., 0.]}}
    for name, anchor in [('source_pallet', source_pose), ('goal_pallet', goal_pose)]:
        t = _matrix_pose(anchor)
        if not np.allclose(t[:3, 2], [0., 0., 1.], atol=1e-7, rtol=0):
            raise ValueError('Pallets must be upright')
        t[:3, 3] += t[:3, 2]*pallet_dims[2]/2
        static[name] = {'dims': pallet_dims, 'pose_world': pose_from_transform(t)}
    mount_pose = base_pose if nominal_mount_pose is None else pose(list(nominal_mount_pose), 'nominal mount pose')
    pedestal, mount_evidence = _mount_cuboids(kin, base_pose, mount_pose)
    static.update(pedestal)
    if extra_static_world is not None:
        if not isinstance(extra_static_world, dict): raise ValueError('Additional static obstacles must be an explicit mapping')
        for name, item in extra_static_world.items():
            if name in static or name in boxes: raise ValueError('Static obstacle overwrite is forbidden: '+name)
            static[name] = copy.deepcopy(item)
    inverse = np.linalg.inv(_matrix_pose(base_pose)); cuboids = {}
    for name in ids:
        i = ids.index(name)
        t = transform(arrays['positions_m'][i], arrays['quaternions_wxyz'][i])
        cuboids[name] = {'dims': list(boxes[name]['dimensions_m']), 'pose': pose_from_transform(inverse@t)}
    for name, item in static.items():
        t = _matrix_pose(item['pose_world']); d = vector(list(item['dims']), 3, name+' dimensions')
        if min(d) <= 0: raise ValueError('Positive static dimensions required')
        cuboids[name] = {'dims': d, 'pose': pose_from_transform(inverse@t)}
        world[name] = _obb(t, d)
    return dict(cuboids=cuboids, objects_world=world, boxes=boxes, box_ids=ids,
        base_pose=base_pose, source_pose=source_pose, goal_pose=goal_pose, mount=mount_evidence,
        static_world=static, robot_config=str(config), snapshot_sha256=canonical_sha256(dict(box_ids=ids,state=state)),
        robot_config_sha256=hashlib.sha256(config.read_bytes()).hexdigest())


def _support_check(target, support, *, pallet=False):
    """Full measured bottom footprint/plane gate, not a force certificate."""
    lower = target['vertices'][SIGNS[:, 2] < 0]
    local = (lower-support['center'])@support['rotation']
    gap = local[:, 2]-support['half'][2]
    allowance = -.01 if pallet else .002
    if (np.max(np.abs(local[:, :2])-support['half'][:2]) > allowance+1e-9
            or np.max(np.abs(gap)) > .002+1e-9):
        raise ValueError('Target lacks current full support footprint or correct support-plane height')
    if target['rotation'][2, 2] < math.cos(math.radians(3)):
        raise ValueError('Target tilt exceeds 3 degrees')
    return dict(passed=True, minimum_bottom_plane_gap_m=float(gap.min()), maximum_bottom_plane_gap_m=float(gap.max()),
                full_footprint_checked=True, force_support_certified=False)



def _resolve_goal_on_measured_support(nominal, world, state, box_id, support_id):
    """Center an upright candidate on a measured support; never move an actor."""
    resolved = nominal.copy()
    support = world['objects_world'][support_id]
    dims = np.asarray(world['boxes'][box_id]['dimensions_m'], float)
    support_matrix = np.eye(4)
    support_matrix[:3, :3] = support['rotation']
    support_matrix[:3, 3] = support['center']
    evidence = dict(schema='depallet.measured_support_goal_resolution.v1', applied=False,
        target_box_id=box_id, support_id=support_id, declared_parent_identity_preserved=True,
        nominal_goal_pose_world_wxyz=pose_from_transform(nominal),
        actual_support_pose_world_wxyz=pose_from_transform(support_matrix),
        maximum_nominal_goal_correction_m=.008, nominal_goal_rotation_preserved=True,
        maximum_support_overhang_m=.002, maximum_support_plane_gap_m=.002,
        maximum_support_linear_speed_m_s=.02, maximum_support_angular_speed_rad_s=.05,
        minimum_target_com_support_margin_m=.005, actors_moved=False,
        physical_execution_validated=False,
        requires_existing_whole_scene_and_actual_com_gates=True)
    if support_id == 'goal_pallet':
        evidence.update(method='authored goal on the unchanged static pallet',
            resolved_goal_pose_world_wxyz=pose_from_transform(resolved),
            correction_world_m=[0., 0., 0.], correction_norm_m=0.)
        return resolved, evidence
    if not np.allclose(nominal[:3, 2], [0., 0., 1.], atol=1e-8, rtol=0):
        raise ValueError('Measured-support resolution requires an upright nominal goal')
    index = world['box_ids'].index(support_id)
    linear = float(np.linalg.norm(np.asarray(state['linear_velocities_m_s'][index], float)))
    angular = float(np.linalg.norm(np.asarray(state['angular_velocities_rad_s'][index], float)))
    normal = support['rotation'][:, 2]
    tilt = math.atan2(float(np.linalg.norm(normal[:2])), float(normal[2]))
    if linear > .02 or angular > .05 or tilt > math.radians(3):
        raise ValueError('Measured goal support violates existing velocity or tilt bounds')
    # Preserve nominal yaw; account for support roll/pitch in the measured
    # top-face center and the height of every bottom corner.
    top_center = support['center']+normal*support['half'][2]
    resolved[:2, 3] = top_center[:2]
    bottom_offsets = (SIGNS[SIGNS[:, 2] < 0]*dims/2)@nominal[:3, :3].T
    bottom_at_zero = bottom_offsets+np.array([resolved[0, 3], resolved[1, 3], 0.])
    gap_at_zero = (bottom_at_zero-support['center'])@normal-support['half'][2]
    resolved[2, 3] = -float(gap_at_zero.min())/float(normal[2])+1e-8
    delta = resolved[:3, 3]-nominal[:3, 3]
    if float(np.linalg.norm(delta)) > .008:
        raise ValueError('Measured-support goal correction exceeds unchanged nominal 8 mm bound')
    desired = _obb(resolved, dims)
    support_check = _support_check(desired, support)
    bottom_local = (desired['vertices'][SIGNS[:, 2] < 0]-support['center'])@support['rotation']
    gaps = bottom_local[:, 2]-support['half'][2]
    if float(gaps.min()) < -1e-9:
        raise ValueError('Resolved goal penetrates the actual support top plane')
    com = np.array(vector(list(world['boxes'][box_id]['physical']['center_of_mass_local_m']),
                          3, 'declared target center of mass'), float)
    if np.any(np.abs(com) >= dims/2):
        raise ValueError('Target COM lies outside the declared carton')
    world_com = resolved[:3, :3]@com+resolved[:3, 3]
    fall = float((np.dot(world_com-support['center'], normal)-support['half'][2])/normal[2])
    if fall < 0:
        raise ValueError('Resolved target COM lies below support plane')
    projected = world_com-np.array([0., 0., fall])
    projected_local = (projected-support['center'])@support['rotation']
    com_margin = float(np.min(support['half'][:2]-np.abs(projected_local[:2])))
    if com_margin < .005:
        raise ValueError('Resolved target COM violates existing 5 mm support margin')
    evidence.update(applied=True, method='measured top-face XY center and minimum nonpenetrating upright height',
        resolved_goal_pose_world_wxyz=pose_from_transform(resolved), correction_world_m=delta.tolist(),
        correction_norm_m=float(np.linalg.norm(delta)), actual_support_top_center_world_m=top_center.tolist(),
        actual_support_linear_speed_m_s=linear, actual_support_angular_speed_rad_s=angular,
        actual_support_tilt_rad=tilt, support_check=support_check,
        resolved_bottom_plane_gaps_m=gaps.tolist(),
        resolved_maximum_xy_overhang_m=max(0., float(np.max(np.abs(bottom_local[:, :2])-support['half'][:2]))),
        target_com_projection_support_local_m=projected_local.tolist(),
        target_com_support_margin_m=com_margin)
    return resolved, evidence


def _request(world, state, measured_q, measured_v, goals, box_id, *, observation_source,
             motion_profile='baseline'):
    q = vector(list(measured_q), 6, 'measured joint positions')
    v = vector(list(measured_v), 6, 'measured joint velocities')
    if max(abs(x) for x in v) > .01:
        raise ValueError('Planning requires measured stopped joints <=0.01 rad/s')
    if observation_source not in ('isaac_runtime', 'cpu_authored_scenario'):
        raise ValueError('Explicit simulation observation provenance required')
    request = dict(schema=REQUEST_SCHEMA, robot_config=world['robot_config'], joint_names=JOINT_NAMES[:],
        start_position_rad=q, start_velocity_rad_s=v, interpolation_dt_s=1/60, length_unit='m', quaternion_order='wxyz',
        base_frame='base_link', base_world_position_m=world['base_pose'][:3], base_world_quaternion_wxyz=world['base_pose'][3:],
        scene={'cuboid':copy.deepcopy(world['cuboids'])}, goals=goals, box_id=box_id, pose_source='oracle_diagnostic',
        obstacle_pose_source=observation_source, observation_sim_time_s=state['sim_time'], observation_physics_step=state['physics_step'],
        source_snapshot_sha256=world['snapshot_sha256'], maximum_trajectory_dt_s=.4,
        position_tolerance_m=.001, endpoint_position_tolerance_m=.002,
        mount_contact_exception=world['mount'], payload=None, physical_execution_validated=False, point2pose_pipeline_validated=False,
        observation_is_live=observation_source=='isaac_runtime')
    apply_motion_profile(request, motion_profile)
    return validate_request(request)


def build_cycle_plan(*, spec, box_ids, state, measured_q, measured_v, base_pose, source_pose, goal_pose,
                     robot_config, assembly_manifest, source_scene_run, box_id, placement, gripper_detached,
                     completed_ids=(), hover_m=.08, clearance_m=.001, tool_yaw_box_rad=0.,
                     attachment_uncertainty=DEFAULT_UNCERTAINTY, nominal_mount_pose=None,
                     extra_static_world=None, observation_source='isaac_runtime',
                     retreat_tcp_world_pose=None, retreat_m=.10,
                     resolve_goal_on_measured_support=False, goal_packing=None,
                     motion_profile='baseline', observation_only=False):
    """Return (single-box-compatible plan, fresh unloaded approach request).

    Caller writes request/manifest and binds actual planner run paths afterward.
    placement uses goal-pallet bottom-frame position_goal_m/yaw_goal_rad and an
    explicit already-present support_id. No historical trajectory is reused.
    """
    if gripper_detached is not True: raise ValueError('Approach requires an actually detached empty gripper')
    if type(observation_only) is not bool or (observation_only and placement.get('observation_only_placeholder') is not True):
        raise ValueError('Observation-only mode requires an explicit placeholder')
    if not isinstance(resolve_goal_on_measured_support, bool):
        raise ValueError('Measured-support goal resolution must be an explicit boolean')
    if resolve_goal_on_measured_support:
        if not isinstance(goal_packing, dict) or not isinstance(goal_packing.get('placements'), list):
            raise ValueError('Measured-support resolution requires complete canonical goal_packing')
        declared = {p['box_id']: p for p in goal_packing['placements']}
        box_count = len(box_ids)
        if (box_count < 1 or len(declared) != box_count
                or len(goal_packing['placements']) != box_count
                or set(declared) != set(box_ids) or declared.get(box_id) != placement):
            raise ValueError('Resolved placement must match its canonical goal_packing parent and pose')
    world = measured_world(spec=spec, box_ids=box_ids, state=state, base_pose=base_pose, source_pose=source_pose,
        goal_pose=goal_pose, robot_config=robot_config, nominal_mount_pose=nominal_mount_pose, extra_static_world=extra_static_world)
    complete = set(completed_ids)
    if len(complete) != len(completed_ids) or not complete <= set(box_ids) or box_id in complete or box_id not in box_ids:
        raise ValueError('Invalid completed/selected box IDs')
    if placement.get('box_id') != box_id or not np.allclose(placement['dimensions_m'], world['boxes'][box_id]['dimensions_m'], atol=1e-9, rtol=0):
        raise ValueError('Placement must preserve selected box ID and dimensions')
    if not (.03 <= hover_m <= .20 and 0. <= clearance_m <= .002 and math.isfinite(tool_yaw_box_rad)):
        raise ValueError('Bounded hover, clearance and finite tool yaw required')
    width,depth = .1841,.2684
    footprint = np.abs(Rotation.from_euler('z',tool_yaw_box_rad).as_matrix()[:2,:2])@np.array([width,depth])
    if np.any(footprint+.02>np.asarray(world['boxes'][box_id]['dimensions_m'][:2])+1e-9):
        raise ValueError('Chosen tool yaw lacks full nominal gripper footprint plus 10 mm edge margin')
    source_support = world['boxes'][box_id]['support_id']
    if source_support not in world['objects_world'] or source_support in complete:
        raise ValueError('Declared current source support is unavailable')
    target = world['objects_world'][box_id]
    source_check = _support_check(target, world['objects_world'][source_support], pallet=source_support=='source_pallet')
    # Conservative measured projected footprints screen exposed top faces.
    mins, maxs = target['vertices'].min(axis=0), target['vertices'].max(axis=0)
    for name in box_ids:
        if name == box_id: continue
        other = world['objects_world'][name]; lo, hi = other['vertices'].min(axis=0), other['vertices'].max(axis=0)
        if hi[2] > maxs[2]+.001 and np.all(np.minimum(maxs[:2], hi[:2])-np.maximum(mins[:2], lo[:2]) > 1e-6):
            raise ValueError('Selected top face is blocked by '+name)
    goal_support = placement.get('support_id')
    if goal_support != 'goal_pallet' and goal_support not in complete:
        raise ValueError('Goal support box has not been committed and observed')
    if goal_support not in world['objects_world'] or goal_support == box_id:
        raise ValueError('Distinct current goal support is required')
    local_goal = transform(placement['position_goal_m'], quaternion(Rotation.from_euler('z', placement['yaw_goal_rad'])))
    goal_matrix = _matrix_pose(world['goal_pose'])@local_goal
    nominal_goal_matrix = goal_matrix.copy()
    goal_resolution = None
    if resolve_goal_on_measured_support and not observation_only:
        goal_matrix, goal_resolution = _resolve_goal_on_measured_support(
            nominal_goal_matrix, world, state, box_id, goal_support)
        # This is a separate, explicitly hypothetical future state. Only the
        # candidate body is represented at its proposed goal; no actor or caller
        # snapshot is changed. Keep the canonical packing and parent identities.
        from depallet.scene.scenario_suite import validate_measured_goal_com
        future = copy.deepcopy(state)
        future['positions_m'] = np.asarray(state['positions_m'], float).tolist()
        future['quaternions_wxyz'] = np.asarray(state['quaternions_wxyz'], float).tolist()
        index = list(box_ids).index(box_id)
        proposed_pose = pose_from_transform(goal_matrix)
        future['positions_m'][index] = proposed_pose[:3]
        future['quaternions_wxyz'][index] = proposed_pose[3:]
        goal_anchor = _matrix_pose(world['goal_pose'])
        com_goal_pose = goal_anchor[:3, 3].tolist()+[math.atan2(goal_anchor[1, 0], goal_anchor[0, 0])]
        com_check = validate_measured_goal_com(spec, goal_packing, list(completed_ids)+[box_id],
                                             list(box_ids), future, goal_pose=com_goal_pose)
        goal_resolution['proposed_future_state_com'] = dict(
            state_kind='hypothetical future placement', changed_box_ids=[box_id],
            source_snapshot_sha256=world['snapshot_sha256'], proposed_state_sha256=canonical_sha256(future),
            physical_execution_validated=False, actual_actor_or_input_state_modified=False,
            ancestor_resultant_com_check=com_check)
        if not com_check['passed']:
            raise ValueError('Hypothetical resolved placement violates existing ancestor/resultant COM bounds')
    desired = _obb(goal_matrix, world['boxes'][box_id]['dimensions_m'])
    goal_check = _support_check(desired, world['objects_world'][goal_support], pallet=goal_support=='goal_pallet')
    if not observation_only:
        for name, other in world['objects_world'].items():
            if name in (box_id, goal_support): continue
            gap = obb_separation(desired, other)[0]
            if gap <= 1e-9: raise ValueError('Desired placement intersects existing obstacle '+name)
    top = target['center']+target['rotation'][:, 2]*(target['half'][2]+clearance_m)
    tcp_rotation = target['rotation']@Rotation.from_euler('z', tool_yaw_box_rad).as_matrix()@Rotation.from_euler('x', math.pi).as_matrix()
    tcp = np.eye(4); tcp[:3,:3] = tcp_rotation; tcp[:3,3] = top
    above = tcp.copy(); above[:3,3] += target['rotation'][:,2]*hover_m
    inverse = np.linalg.inv(_matrix_pose(world['base_pose'])); goals=[]
    for name, matrix in [('pregrasp',above),('contact',tcp)]:
        p = pose_from_transform(inverse@matrix)
        goal = dict(id=name,tcp_frame='suction_tcp',position_m=p[:3],quaternion_wxyz=p[3:])
        if name=='contact':goal.update(linear_axis='z',linear_in_tool_frame=True)
        goals.append(goal)
    if retreat_tcp_world_pose is not None:
        if isinstance(retreat_m,bool) or not isinstance(retreat_m,(int,float)) or not .03<=retreat_m<=.15:
            raise ValueError('Combined approach retreat distance must be within 3..15 cm')
        retreat=_matrix_pose(retreat_tcp_world_pose);retreat[2,3]+=retreat_m
        p=pose_from_transform(inverse@retreat)
        goals.insert(0,dict(id='empty_tool_retreat',tcp_frame='suction_tcp',position_m=p[:3],quaternion_wxyz=p[3:],
                           linear_axis='z',linear_in_tool_frame=False))
    profile = resolve_motion_profile(motion_profile)
    request = _request(world,state,measured_q,measured_v,goals,box_id,
        observation_source=observation_source,motion_profile=profile['profile'])
    i = list(box_ids).index(box_id)
    request.update(target_box_world_pose=list(state['positions_m'][i])+list(state['quaternions_wxyz'][i]),
        grasp_tcp_world_pose=pose_from_transform(tcp),pregrasp_tcp_world_pose=pose_from_transform(above))
    if goal_resolution is not None:
        request['goal_resolution'] = copy.deepcopy(goal_resolution)
    goal_world = pose_from_transform(goal_matrix)
    plan = dict(schema='depallet.single_box_execution.v1', box_id=box_id,
        source_scene_run=str(Path(source_scene_run).resolve()),robot_config=world['robot_config'],
        assembly_manifest=str(Path(assembly_manifest).resolve()),source_pose=list(source_pose),goal_pose=list(goal_pose),
        goal_position_m=goal_world[:3],goal_quaternion_wxyz=goal_world[3:],goal_support_id=goal_support,
        source_support_id=source_support,goal_available_now=not observation_only,
        observation_only_placeholder=observation_only,perception_source='simulation_oracle',
        approach_request_data=copy.deepcopy(request), completed_ids=list(completed_ids),
        source_snapshot_sha256=world['snapshot_sha256'], source_support_check=source_check,goal_support_check=goal_check,
        cycle_planning_schema='depallet.multi_transfer_cycle.v1',placement=copy.deepcopy(placement),
        world_parameters=dict(spec=copy.deepcopy(spec),source_pose=list(source_pose),goal_pose=list(goal_pose),
            nominal_mount_pose=list(nominal_mount_pose) if nominal_mount_pose is not None else list(base_pose),
            extra_static_world=copy.deepcopy(extra_static_world)),
        observation_source=observation_source,physical_execution_validated=False,full_task_validated=False,
        motion_profile=profile['profile'])
    if attachment_uncertainty is not None:plan['attachment_uncertainty']=attachment_uncertainty_policy(attachment_uncertainty)
    if goal_resolution is not None:
        nominal_pose = pose_from_transform(nominal_goal_matrix)
        plan.update(goal_resolution=goal_resolution, nominal_goal_position_m=nominal_pose[:3],
                    nominal_goal_quaternion_wxyz=nominal_pose[3:], resolve_goal_on_measured_support=True)
    return plan,request


def _fresh_world_for_plan(plan,state,box_ids,base_pose):
    parameters=plan['world_parameters']
    return measured_world(spec=parameters['spec'],box_ids=box_ids,state=state,base_pose=base_pose,
        source_pose=parameters['source_pose'],goal_pose=parameters['goal_pose'],robot_config=plan['robot_config'],
        nominal_mount_pose=parameters['nominal_mount_pose'],extra_static_world=parameters['extra_static_world'])


def make_payload_request(plan,state,box_ids,measured_q,measured_v,base_pose,tcp_world_pose,*,attachment_confirmed,departure_completed=False):
    """Preserve actual T_tcp_box and all newly measured goal/source obstacles."""
    if plan.get('observation_only_placeholder'):
        raise ValueError('Observation-only placeholder cannot authorize payload motion')
    if attachment_confirmed is not True:raise ValueError('Confirmed actual attachment required for payload planning')
    velocity=vector(list(measured_v),6,'measured payload start velocity')
    if max(abs(v) for v in velocity)>.01:raise ValueError('Payload planning requires measured stopped joints')
    world=_fresh_world_for_plan(plan,state,box_ids,base_pose)
    desired=_obb(transform(plan['goal_position_m'],plan['goal_quaternion_wxyz']),world['boxes'][plan['box_id']]['dimensions_m'])
    goal_support=plan['goal_support_id']
    _support_check(desired,world['objects_world'][goal_support],pallet=goal_support=='goal_pallet')
    for name,other in world['objects_world'].items():
        if name in (plan['box_id'],goal_support):continue
        if obb_separation(desired,other)[0]<=1e-9:raise ValueError('Current placement obstructed by '+name)
    fresh=copy.deepcopy(plan);request=fresh['approach_request_data']
    request.update(scene={'cuboid':world['cuboids']},base_world_position_m=world['base_pose'][:3],
        base_world_quaternion_wxyz=world['base_pose'][3:],source_snapshot_sha256=world['snapshot_sha256'],
        observation_sim_time_s=state['sim_time'],observation_physics_step=state['physics_step'])
    generated,evidence=payload_request(fresh,state,list(box_ids),world['boxes'],measured_q,measured_v,base_pose,tcp_world_pose,
        attachment_confirmed=attachment_confirmed,departure_completed=departure_completed)
    generated['source_support_id']=plan['source_support_id'];generated['goal_support_id']=plan['goal_support_id']
    evidence.update(source_snapshot_sha256=world['snapshot_sha256'],all_dynamic_box_ids=list(box_ids),
        current_goal_boxes_retained=list(plan['completed_ids']),physical_execution_validated=False)
    if plan.get('goal_resolution') is not None:
        evidence.update(goal_resolution=copy.deepcopy(plan['goal_resolution']),
                        resolved_goal_reused_without_further_recentering=True,
                        resolved_goal_rechecked_against_current_support=True)
    return generated,evidence


def make_retreat_request(plan,state,box_ids,measured_q,measured_v,base_pose,tcp_world_pose,*,gripper_detached,retreat_m=.08):
    """Unloaded vertical retreat from actual TCP; released target stays obstacle."""
    if gripper_detached is not True:raise ValueError('Retreat requires actual Open/detached evidence')
    if not isinstance(retreat_m,(int,float)) or isinstance(retreat_m,bool) or not .03<=retreat_m<=.15:
        raise ValueError('Retreat distance must be within 3..15 cm')
    world=_fresh_world_for_plan(plan,state,box_ids,base_pose)
    t=_matrix_pose(tcp_world_pose);t[2,3]+=retreat_m
    p=pose_from_transform(np.linalg.inv(_matrix_pose(base_pose))@t)
    goals=[dict(id='empty_tool_retreat',tcp_frame='suction_tcp',position_m=p[:3],quaternion_wxyz=p[3:],
                linear_axis='z',linear_in_tool_frame=False)]
    request=_request(world,state,measured_q,measured_v,goals,plan['box_id'],
        observation_source=plan['observation_source'],motion_profile=plan.get('motion_profile','baseline'))
    request.update(request_role='unloaded_post_release_retreat',released_box_retained=True,source_support_id=plan['source_support_id'],goal_support_id=plan['goal_support_id'])
    return request


def validate_current_request(request, *, spec,box_ids,state,base_pose,source_pose,goal_pose,nominal_mount_pose=None,extra_static_world=None):
    """Full measured scene/base/config binding before a caller may execute."""
    validate_request(request)
    world=measured_world(spec=spec,box_ids=box_ids,state=state,base_pose=base_pose,source_pose=source_pose,
        goal_pose=goal_pose,robot_config=request['robot_config'],nominal_mount_pose=nominal_mount_pose,extra_static_world=extra_static_world)
    if set(request['scene']['cuboid'])!=set(world['cuboids']):raise ValueError('Current world obstacle set differs')
    for name,current in world['cuboids'].items():
        old=request['scene']['cuboid'][name]
        if not np.allclose(old['dims'],current['dims'],atol=1e-9,rtol=0):raise ValueError('Obstacle dimensions changed: '+name)
        a,b=_matrix_pose(old['pose']),_matrix_pose(current['pose'])
        if np.linalg.norm(a[:3,3]-b[:3,3])>.002 or Rotation.from_matrix(a[:3,:3].T@b[:3,:3]).magnitude()>.005:
            raise ValueError('Measured obstacle pose changed: '+name)
    if np.max(np.abs(_matrix_pose(request['base_world_position_m']+request['base_world_quaternion_wxyz'])-_matrix_pose(base_pose)))>.0001:
        raise ValueError('Measured base changed')
    return dict(passed=True,checked_box_ids=list(box_ids),checked_obstacles=list(world['cuboids']),
        source='current measured world',snapshot_sha256=world['snapshot_sha256'],physical_execution_validated=False)
