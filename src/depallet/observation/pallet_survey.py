"""Whole-source survey geometry and oracle depth diagnostics, never perception.

A frustum fit is not visibility. Recorded capture-time poses and depth must
also support the exposed surface samples. Hidden cartons are never claimed seen.
"""
from __future__ import annotations
import copy
from itertools import product
import numpy as np
from scipy.spatial.transform import Rotation
from depallet.observation.camera_rig import camera_specs, intrinsics, look_at_cv
from depallet.planning.depallet_execution_plan import transform, pose_from_transform
from depallet.observation.point2pose_adapter import checked_transform

POLICY = 'wrist_pallet_v1'
CANDIDATE_IDS = ('oblique', 'top_0', 'top_90', 'top_180', 'top_270', 'top_180_clear')
MARGIN_PX = 24.


def _box_corners(dims):
    dims = np.asarray(dims, float)
    if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0):
        raise ValueError('Positive finite cuboid dimensions required')
    return np.asarray(list(product((-.5, .5), repeat=3))) * dims


def _apply(T, points):
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


def survey_geometry(request):
    """Oracle envelope and exposed surfaces; deliberately independent of box_id."""
    if request.get('payload') is not None:
        raise ValueError('Whole-pallet survey requires an unloaded gripper')
    base = transform(request['base_world_position_m'], request['base_world_quaternion_wxyz'])
    cuboids = request['scene']['cuboid']
    pallet = cuboids['source_pallet']
    pallet_T = base @ transform(pallet['pose'][:3], pallet['pose'][3:])
    if np.dot(pallet_T[:3, 2], [0, 0, 1]) < .999:
        raise ValueError('Survey requires an upright source pallet')
    dims = np.asarray(pallet['dims'], float); _box_corners(dims)
    inv = np.linalg.inv(pallet_T)
    deck = dims[2] / 2
    local_envelope = [np.array([x*dims[0]/2, y*dims[1]/2, deck]) for x, y in product((-1, 1), repeat=2)]
    boxes = []
    for name, box in cuboids.items():
        if not name.startswith('box_'):
            continue
        T = base @ transform(box['pose'][:3], box['pose'][3:])
        corners = _apply(inv, _apply(T, _box_corners(box['dims'])))
        if (corners[:, 2].max() < deck-.01 or np.any(corners[:, :2].min(0) > dims[:2]/2)
                or np.any(corners[:, :2].max(0) < -dims[:2]/2)):
            continue
        # Geometry supports the current rigid upright scenario family only.
        if np.dot(T[:3, 2], [0, 0, 1]) < .999:
            raise ValueError('Tilted cartons require a general surface visibility model')
        local_envelope.extend(corners)
        boxes.append((name, T, np.asarray(box['dims'], float)))
    points = np.asarray(local_envelope)
    lo, hi = points.min(0), points.max(0)
    envelope = _apply(pallet_T, np.array(list(product(*zip(lo, hi)))))
    target = _apply(pallet_T, np.array([[(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, hi[2]]]))[0]
    surfaces = []
    for name, T, d in boxes:
        samples = _apply(T, [[x*d[0], y*d[1], d[2]/2] for x, y in product(np.linspace(-.42,.42,9), repeat=2)])
        exposed = np.ones(len(samples), dtype=bool)
        for other, U, e in boxes:
            if other == name:
                continue
            local = _apply(np.linalg.inv(U), samples)
            above = (np.abs(local[:,0]) < e[0]/2+.002) & (np.abs(local[:,1]) < e[1]/2+.002) & (local[:,2] < -e[2]/2+.002)
            exposed &= ~above
        if exposed.any():
            surfaces.append(dict(box_id=name, world_points_m=samples[exposed].tolist(),
                                 scope='oracle vertically exposed top samples, not segmentation'))
    return dict(coverage_envelope_world_m=envelope.tolist(), target_world_m=target.tolist(),
                source_box_ids=[b[0] for b in boxes], exposed_surfaces=surfaces,
                source='simulation_oracle_source_geometry', hidden_surfaces_observed=False)


def project_coverage(points, camera_world, *, margin_px=MARGIN_PX):
    if isinstance(margin_px,bool) or not np.isfinite(margin_px) or not 0 <= margin_px < 240:
        raise ValueError('Finite bounded pixel margin required')
    wrist = next(s for s in camera_specs() if s['id']=='wrist')
    T = checked_transform(camera_world, 'T_world_camera_cv')
    pts = np.asarray(points, float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts)==0 or not np.isfinite(pts).all():
        raise ValueError('Nonempty finite world points required')
    xyz = _apply(np.linalg.inv(T), pts)
    # Avoid nonfinite JSON on a rejected zero-depth projection.
    safe = np.where(np.abs(xyz[:,2])>1e-9, xyz[:,2], 1e-9)
    uv = (xyz @ intrinsics(wrist).T)[:, :2] / safe[:,None]
    h,w = wrist['resolution_hw']
    margins = np.minimum.reduce((uv[:,0], uv[:,1], w-1-uv[:,0], h-1-uv[:,1]))
    ok = (xyz[:,2] > .1) & (xyz[:,2] < 5.) & (margins >= margin_px)
    return dict(passed=bool(np.all(ok)), point_count=len(pts), points_inside=int(ok.sum()),
                minimum_border_margin_px=float(margins.min()), required_margin_px=margin_px,
                uv=uv.tolist(), optical_depth_m=xyz[:,2].tolist(),
                occlusion_validated=False)


def survey_candidates(request, T_flange_tcp):
    geometry = survey_geometry(request)
    tcp_local = checked_transform(T_flange_tcp, 'T_flange_tcp')
    wrist = next(s for s in camera_specs() if s['id']=='wrist')
    mount = np.asarray(wrist['T_parent_camera_cv'])
    # Fixed world suction orientation; never derived from the selected target.
    canonical = Rotation.from_euler('x', np.pi).as_matrix() @ tcp_local[:3,:3].T @ mount[:3,:3]
    target = np.asarray(geometry['target_world_m']); envelope = geometry['coverage_envelope_world_m']
    result = []
    for name in CANDIDATE_IDS:
        rot = canonical if name=='oblique' else look_at_cv(target+[0,0,1],target,(0,1,0))[:3,:3] @ Rotation.from_euler('z',int(name.split('_')[1]),degrees=True).as_matrix()
        T = np.eye(4); T[:3,:3] = rot
        # Development capture top_180 showed the tool on image-right. Fit the
        # SAME full envelope into a conservative left ROI, without remounting
        # the camera or hiding robot geometry. Recorded depth still decides.
        usable_right = 450. if name.endswith('_clear') else 639.-MARGIN_PX
        optical_shift = (320.-(MARGIN_PX+usable_right)/2)/intrinsics(wrist)[0,0] if name.endswith('_clear') else 0.
        # Fixed bounded deterministic search, fits footprint AND height envelope.
        for distance in np.arange(.45, 2.501, .025):
            T[:3,3] = target-distance*T[:3,2]+distance*optical_shift*T[:3,0]
            projection = project_coverage(envelope,T)
            projection['usable_right_px']=usable_right
            projection['passed']=bool(projection['passed'] and np.max(np.asarray(projection['uv'])[:,0])<=usable_right)
            if projection['passed'] and T[2,3]>=np.max(np.asarray(envelope)[:,2])+.25:
                break
        else:
            continue
        tcp_world = T @ np.linalg.inv(mount) @ tcp_local
        if not .2 < tcp_world[2,3] < 3.5:
            continue
        result.append(dict(id=name,policy=POLICY,camera_id='wrist',standoff_m=float(distance),
            planned_T_world_camera_cv=T.tolist(),planned_tcp_world_pose=pose_from_transform(tcp_world),
            **copy.deepcopy(geometry),projection=projection,viewpoint_source='simulation_oracle_whole_source_envelope',
            camera_mount_collision_modelled=False,visibility_validated=False,perception_controls_robot=False))
    if not result:
        raise ValueError('No bounded whole-pallet frustum-fit candidate')
    return result


def survey_request(request,T_flange_tcp,*,candidate_id='oblique'):
    if candidate_id not in CANDIDATE_IDS:
        raise ValueError('Unknown survey candidate')
    ids=[g['id'] for g in request['goals']]
    if ids not in (['pregrasp','contact'],['empty_tool_retreat','pregrasp','contact']):
        raise ValueError('Survey expects canonical unloaded approach goals')
    choices={c['id']:c for c in survey_candidates(request,T_flange_tcp)}
    if candidate_id not in choices:
        raise ValueError('Requested survey candidate has no bounded frustum fit')
    selected=choices[candidate_id]
    T=transform(selected['planned_tcp_world_pose'][:3],selected['planned_tcp_world_pose'][3:])
    B=transform(request['base_world_position_m'],request['base_world_quaternion_wxyz'])
    pose=pose_from_transform(np.linalg.inv(B)@T)
    result=copy.deepcopy(request)
    result['goals'].insert(ids.index('pregrasp'),dict(id='inspection',tcp_frame='suction_tcp',position_m=pose[:3],quaternion_wxyz=pose[3:]))
    result['inspection_view']=selected
    return result


def measured_survey_coverage(planning,camera_world,depth,*,scope="all_exposed",required_box_id=None):
    """Recorded oracle diagnostic: reject missing/occluded top samples.

    This does not infer hidden boxes or satisfy a semantic survey contract.
    Pose and depth must be from the same accepted capture; caller binds hashes.
    """
    if planning.get('policy') != POLICY:
        raise ValueError('Whole-pallet survey policy required')
    if scope not in ('all_exposed','highest_layer'):raise ValueError('Unknown survey scope')
    depth=np.asarray(depth)
    if depth.shape != (480,640) or not np.issubdtype(depth.dtype,np.number) or np.iscomplexobj(depth):
        raise ValueError('Native 640x480 depth required')
    envelope=project_coverage(planning['coverage_envelope_world_m'],camera_world)
    rows=[]
    for surface in planning['exposed_surfaces']:
        p=project_coverage(surface['world_points_m'],camera_world)
        uv=np.rint(np.asarray(p['uv'])).astype(int);z=np.asarray(p['optical_depth_m'])
        inside=(z>.1)&(uv[:,0]>=0)&(uv[:,0]<640)&(uv[:,1]>=0)&(uv[:,1]<480)
        agree=np.zeros(len(z),bool);ids=np.where(inside)[0]
        d=depth[uv[ids,1],uv[ids,0]]
        agree[ids]=np.isfinite(d)&(d>0)&(np.abs(d-z[ids])<=.015)
        ratio=float(agree.mean())
        rows.append(dict(box_id=surface['box_id'],samples=len(z),depth_consistent=int(agree.sum()),
                         agreement_fraction=ratio,passed=p['passed'] and ratio>=.9))
    heights={surface['box_id']:float(np.mean(np.asarray(surface['world_points_m'])[:,2]))
             for surface in planning['exposed_surfaces']}
    highest=max(heights.values(),default=-float('inf'))
    required=[r for r in rows if scope=='all_exposed' or heights[r['box_id']]>=highest-.02]
    required_ids=[r['box_id'] for r in required]
    target_in_scope=required_box_id is None or required_box_id in required_ids
    passed=bool(envelope['passed'] and required and target_in_scope and all(r['passed'] for r in required))
    return dict(schema='depallet.measured_pallet_survey.v1',passed=passed,envelope=envelope,
                exposed_surfaces=rows,depth_tolerance_m=.015,minimum_surface_agreement=.9,
                observation_scope=scope,required_box_ids=required_ids,
                deferred_box_ids=[r['box_id'] for r in rows if r not in required],
                required_target_in_scope=target_in_scope,highest_layer_band_m=.02,
                all_exposed_surfaces_passed=bool(rows and all(r['passed'] for r in rows)),
                source='oracle source geometry compared with recorded wrist depth',
                semantic_coverage_validated=False,hidden_surfaces_observed=False,
                perception_controls_robot=False)
