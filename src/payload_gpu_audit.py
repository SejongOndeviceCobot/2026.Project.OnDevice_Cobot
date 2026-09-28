"""Read actual GPU FK spheres after attachment; compare world/self on CPU."""
from pathlib import Path
import numpy as np
from contact_escape import pose_matrix, sphere_box_clearance
from curobo_bridge import write_json
from payload_self_contact import validate_payload_sphere_radii


def audit(planner, q, request, robot, output):
    kin=robot.get('robot_cfg',robot)['kinematics']
    params=planner.attachment_manager.kinematics_params
    spheres=planner.compute_kinematics(q).robot_spheres.detach().cpu().numpy().reshape(-1,4)
    groups={}
    for link in kin['collision_link_names']:
        idx=params.get_sphere_index_from_link_name(link).detach().cpu().numpy().astype(int).reshape(-1)
        groups[link]=spheres[idx].tolist()
    padding_check=validate_payload_sphere_radii(groups.get("attached_object",[]),request)
    center_check=None
    if request["payload"].get("cover_profile"):
        from payload_self_contact import payload_cover_for_request
        expected,cover,_=payload_cover_for_request(request)
        actual=np.asarray(groups["attached_object"],float);actual=actual[actual[:,3]>0]
        t=pose_matrix(request["payload"]["pose_base_wxyz"])
        centers=np.asarray([s["center"] for s in expected])@t[:3,:3].T+t[:3,3]
        if actual.shape!=(len(expected),4) or not np.allclose(actual[:,:3],centers,rtol=0.,atol=2e-6):
            raise ValueError("Actual GPU payload sphere centers differ from complete covering grid")
        center_check={"passed":True,"profile":request["payload"]["cover_profile"],"cells":cover["cells"],
            "sphere_count":len(expected),"maximum_center_error_m":float(np.max(np.linalg.norm(actual[:,:3]-centers,axis=1))),
            "absolute_per_coordinate_tolerance_m":2e-6,"matching_order_required":True}
    world=[];self_pairs=[];minimum=1e6
    for link, values in groups.items():
        s=np.array(values);s=s[s[:,3]>0]
        if not len(s):continue
        for name,box in request['scene']['cuboid'].items():
            if name==request['payload']['box_id']:continue
            t=pose_matrix(box['pose'])
            d=sphere_box_clearance(s[:,:3],s[:,3],t[:3,3],t[:3,:3],np.array(box['dims'])/2)
            minimum=min(minimum,float(d.min()))
            if d.min() < -1e-6:world.append({'link':link,'obstacle':name,'clearance_m':float(d.min())})
    ignore=kin.get('self_collision_ignore',{})
    for i,a in enumerate(groups):
        for b in list(groups)[i+1:]:
            if b in ignore.get(a,[]) or a in ignore.get(b,[]):continue
            sa,sb=np.array(groups[a]),np.array(groups[b]);sa=sa[sa[:,3]>0];sb=sb[sb[:,3]>0]
            if not len(sa) or not len(sb):continue
            d=np.linalg.norm(sa[:,None,:3]-sb[None,:,:3],axis=-1)-sa[:,None,3]-sb[None,:,3]
            if d.min() < -1e-6:self_pairs.append({'links':[a,b],'clearance_m':float(d.min())})
    record={'source':'actual cuRobo GPU FK spheres after attachment','world_collisions':world,'self_collisions':self_pairs,'minimum_world_clearance_m':minimum,'sphere_count':len(spheres),'spheres_by_link':groups,'payload_sphere_padding_check':padding_check,'passed':not world and not self_pairs}
    if center_check is not None:record['payload_sphere_center_check']=center_check
    write_json(Path(output)/'payload-gpu-start-audit.json',record)
    return record
