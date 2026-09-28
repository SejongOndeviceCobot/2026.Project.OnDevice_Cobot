"""Non-executable goal placeholder for motion to an online observation hold.

The actual placement must be supplied by the subsequent checked online plan.
"""
from copy import deepcopy


def observation_planning_inputs(spec, placement, packing):
    placeholder=deepcopy(placement)
    dims=placeholder['dimensions_m'];pallet=spec['pallet']['dimensions_m']
    if any(dims[i]+.02>pallet[i] for i in (0,1)):
        raise ValueError('Observed target exceeds pallet footprint')
    z=pallet[2]+dims[2]/2
    placeholder.update(support_id='goal_pallet',position_goal_m=[0.,0.,z],yaw_goal_rad=0.,
        footprint_goal_m=list(dims[:2]),top_face_center_goal_m=[0.,0.,z+dims[2]/2],
        observation_only_placeholder=True)
    updated=deepcopy(packing)
    updated['placements']=[placeholder if p['box_id']==placement['box_id'] else p for p in updated['placements']]
    return placeholder,updated
