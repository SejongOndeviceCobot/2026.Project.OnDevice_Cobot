"""Bind verified model suction candidates to measured cuTAMP objects (explicit hybrid)."""
import math
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
from scipy.spatial.transform import Rotation
from modular_pick_bridge import load_surfaces


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pose_matrix(pose):
    pose=np.asarray(pose,float)
    if pose.shape!=(7,) or not np.isfinite(pose).all():raise ValueError('Invalid metric pose')
    result=np.eye(4);result[:3,3]=pose[:3]
    result[:3,:3]=Rotation.from_quat(pose[[4,5,6,3]]).as_matrix()
    return result


def associate(packet,env,meta):
    if len(packet['candidates'])!=1 or packet['modules']['graspgen']!='gpu_inference_connected':
        raise ValueError('One validated learned candidate required')
    candidate=packet['candidates'][0]
    world_base=pose_matrix([*meta['base_world_position_m'],*meta['base_world_quaternion_wxyz']])
    bodies=env['geometries']['cuboid'];ranked=[];transforms={}
    for name in meta['movable_ids']:
        body=bodies[name];world_obj=world_base@pose_matrix(body['pose']);transforms[name]=world_obj
        top=world_obj@np.array([0,0,body['dims'][2]/2,1.])
        ranked.append((float(np.linalg.norm(top[:3]-candidate['top_world_m'])),name))
    ranked.sort()
    if ranked[0][0]>.02 or (len(ranked)>1 and ranked[1][0]-ranked[0][0]<.05):
        raise ValueError('Ambiguous or stale learned-candidate association')
    distance,name=ranked[0];obj_world=np.linalg.inv(transforms[name])
    contact=np.array(candidate['grasp_world_m'])+np.array([0.,0.,.001])
    xyz=(obj_world@np.r_[contact,1.])[:3]
    yaw=float(np.arctan2(obj_world[1,0],obj_world[0,0]))
    if 'grasp_world_yaw_rad' in candidate:yaw+=float(candidate['grasp_world_yaw_rad'])
    return name,{'provider':'graspgen_suction','xyz_yaw_object':[*xyz.tolist(),yaw],
        'association_distance_m':distance,'association_source':'measured_simulation_oracle',
        'orientation_policy':('observed_footprint_world_yaw' if 'grasp_world_yaw_rad' in candidate else 'world_yaw_zero_v1'),'sam31_object_id':candidate['sam31_object_id'],
        'candidate_source':packet['grasp_candidates_source'],'candidate_sha256':packet['grasp_candidates_sha256'],
        'review':candidate['graspgen_review']}


def build(input_root,geometry,sources,runs):
    runs=Path(runs).resolve();root=Path(input_root).resolve();geometry=Path(geometry).resolve()
    if not root.is_relative_to(runs) or not geometry.is_relative_to(runs):raise ValueError('Project inputs required')
    meta=json.loads((root/'manifest.json').read_text());env=yaml.safe_load((root/'environment.yml').read_text())
    bindings={}
    for source in sources:
        packet=load_surfaces(geometry,runs,source)
        name,binding=associate(packet,env,meta)
        if name in bindings:raise ValueError('Duplicate learned box binding: '+name)
        bindings[name]=binding
    if set(bindings)!=set(meta['movable_ids']):
        raise ValueError('Missing learned grasps: '+repr(sorted(set(meta['movable_ids'])-set(bindings))))
    return {'schema':'depallet.cutamp_grasp_bindings.v1','input_manifest_sha256':digest(root/'manifest.json'),
        'environment_sha256':digest(root/'environment.yml'),'geometry_source':str(geometry),
        'geometry_sha256':digest(geometry),'sources':[str(Path(s).resolve()) for s in sources],
        'bindings':bindings,'live_perception':False,'fallback_used':False,'physical_execution_validated':False}


def same_binding(left,right,field=None):
    """Exact provenance; permit eight ULPs only in re-derived orientation fields."""
    if type(left) is not type(right):return False
    if isinstance(left,dict):
        return left.keys()==right.keys() and all(same_binding(left[k],right[k],k) for k in left)
    if isinstance(left,list):
        if len(left)!=len(right):return False
        return all(same_binding(a,b,'grasp_world_yaw_rad' if field=='xyz_yaw_object' and i==3 else None)
                   for i,(a,b) in enumerate(zip(left,right)))
    if isinstance(left,float) and field=='grasp_world_yaw_rad':
        return math.isfinite(left) and math.isfinite(right) and abs(left-right)<=8*max(math.ulp(left),math.ulp(right))
    return left==right


def load_verified(path,input_root,runs):
    path=Path(path).resolve()
    if not path.is_relative_to(Path(runs).resolve()):raise ValueError('Project bindings required')
    packet=json.loads(path.read_text())
    rebuilt=build(input_root,packet['geometry_source'],packet['sources'],runs)
    if not same_binding(packet,rebuilt):raise ValueError('Learned binding provenance or input changed')
    return packet
