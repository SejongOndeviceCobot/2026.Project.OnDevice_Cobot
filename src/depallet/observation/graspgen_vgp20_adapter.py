"""Retarget learned suction positions to the fixed-yaw V1 VGP20 recipe.

This is a geometric proposal filter, not a vacuum seal or collision certificate.
All original learned poses and rejection reasons remain available.
"""
import numpy as np
from depallet.observation.point2pose_adapter import checked_transform


def select_vgp20_candidate(candidates, geometry, *, edge_margin_m=.01, yaw_policy="fixed", position_policy="score"):
    if position_policy not in ("score","center"):raise ValueError("Unknown position ranking policy")
    if yaw_policy not in ("fixed","observed_footprint"):raise ValueError("Unknown pad yaw policy")
    if geometry.get('top_pose_usable') is not True:
        raise ValueError('Usable observed top required')
    frame=checked_transform(geometry['observed_footprint']['T_world_observed_top'])
    extents=np.asarray(geometry['observed_footprint']['observed_extent_m'],float)
    if extents.shape!=(2,) or not np.isfinite(extents).all() or min(extents)<=0:
        raise ValueError('Finite positive observed footprint required')
    if edge_margin_m<.01:raise ValueError('At least 10mm pad-envelope margin required')
    normal=frame[:3,2]
    if normal[2]<.999:raise ValueError('V1 retarget requires a near-horizontal top')
    # Existing V1 recipe: VGP20 envelope X=.1841m, Y=.2684m, fixed world yaw=0.
    corners=np.array([[x*.1841/2,y*.2684/2,0.] for x in (-1,1) for y in (-1,1)])
    reviewed=[]
    for index,candidate in enumerate(candidates):
        matrix=checked_transform(candidate['T_world_grasp'])
        score=float(candidate['score'])
        if not np.isfinite(score):raise ValueError('Finite model score required')
        local=frame[:3,:3].T@(matrix[:3,3]-frame[:3,3])
        reasons=[]
        # Suction asset +Z points away from the contact surface (official data generator).
        cosine=float(matrix[:3,2]@normal)
        if cosine<np.cos(np.deg2rad(15)):reasons.append('non_top_down_suction')
        if abs(local[2])>.015:reasons.append('candidate_far_from_observed_plane')
        projected=matrix[:3,3]-normal*local[2]
        theta=float(np.arctan2(frame[1,0],frame[0,0]))
        yaws=[0.] if yaw_policy=='fixed' else [0.,theta,theta+np.pi/2]
        fit=[]
        for yaw in yaws:
            c,s=np.cos(yaw),np.sin(yaw)
            rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
            local_corners=(projected+corners@rotation.T-frame[:3,3])@frame[:3,:3]
            margin=float(np.min(extents/2-np.abs(local_corners[:,:2])-edge_margin_m))
            fit.append((margin,yaw))
        margin,yaw=max(fit,key=lambda x:x[0])
        if margin<0:
            reasons.append('vgp20_envelope_outside_observed_top')
        reviewed.append({'index':index,'score':score,'accepted':not reasons,'rejection_reasons':reasons,
                         'projected_contact_world_m':projected.tolist(),
                         'plane_projection_distance_m':abs(float(local[2]))})
        if yaw_policy!='fixed':reviewed[-1]['grasp_world_yaw_rad']=float(yaw)
        if position_policy=='center':reviewed[-1]['observed_center_offset_m']=float(np.linalg.norm(projected-frame[:3,3]))
    eligible=[r for r in reviewed if r['accepted']]
    selected=(min(eligible,key=lambda r:(r['observed_center_offset_m'],-r['score'])) if position_policy=='center'
              else max(eligible,key=lambda r:r['score'])) if eligible else None
    return {'schema':'depallet.graspgen_vgp20_retarget.v1','selected':selected,'reviewed':reviewed,
            'orientation_policy':('fixed_yaw_v1_recipe_not_learned_rotation' if yaw_policy=='fixed' else 'observed_footprint_yaw_not_learned_rotation'),
            'position_policy':('learned_suction_position_projected_to_measured_plane' if position_policy=='score' else 'nearest_observed_center_among_valid_learned_positions'),
            'seal_validated':False,'collision_validated':False,'planner_eligible':False}
