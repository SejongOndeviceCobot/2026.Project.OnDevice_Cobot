"""Certify a fixed tool/payload pair before excluding sphere-only self overlap.

This never removes world collision or any moving-arm collision pair. Actual
OBB intersection (including a 3 mm expanded tool body) fails closed.
"""
from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from depallet.motion.curobo_bridge import pose, vector
from depallet.motion.curobo_geometry import box_cover
from depallet.motion.urdf_fk import matrix


def obb_separation(center_a, axes_a, half_a, center_b, axes_b, half_b):
    """15-axis SAT and a conservative separating-distance lower bound."""
    ca,cb=np.asarray(center_a,float),np.asarray(center_b,float)
    ra,rb=np.asarray(axes_a,float),np.asarray(axes_b,float)
    ha,hb=np.asarray(half_a,float),np.asarray(half_b,float)
    if ca.shape!=(3,) or cb.shape!=(3,) or ra.shape!=(3,3) or rb.shape!=(3,3):
        raise ValueError("OBB dimensions must be 3D")
    if any(not np.isfinite(x).all() for x in (ca,cb,ra,rb,ha,hb)) or np.any(ha<=0) or np.any(hb<=0):
        raise ValueError("Invalid OBB values")
    for r in (ra,rb):
        if not np.allclose(r.T@r,np.eye(3),atol=1e-7) or np.linalg.det(r)<.999999:
            raise ValueError("OBB axes must be a proper orthonormal rotation")
    axes=[ra[:,i] for i in range(3)]+[rb[:,i] for i in range(3)]
    axes += [np.cross(ra[:,i],rb[:,j]) for i in range(3) for j in range(3)]
    best=-float("inf");best_axis=None;count=0
    for axis in axes:
        norm=np.linalg.norm(axis)
        if norm<1e-9:
            continue
        axis=axis/norm
        gap=abs(float(np.dot(cb-ca,axis)))-float(np.dot(ha,np.abs(ra.T@axis)))-float(np.dot(hb,np.abs(rb.T@axis)))
        count+=1
        if gap>best:
            best=gap;best_axis=axis.tolist()
    return {"nonoverlap":best>0.,"separating_gap_lower_bound_m":best,
            "separating_axis_base":best_axis,"axes_checked":count}


def attachment_uncertainty_policy(spec):
    """Validate explicit COM translation/box rotation bounds and sphere padding."""
    if spec is None:
        return None
    if not isinstance(spec,dict) or set(spec)!={"translation_m","rotation_rad","payload_padding_m"}:
        raise ValueError("attachment_uncertainty requires translation_m, rotation_rad and payload_padding_m")
    limits={"translation_m":.004,"rotation_rad":.01,"payload_padding_m":.02}
    result={}
    for key,upper in limits.items():
        value=spec[key]
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0 <= value <= upper:
            raise ValueError("Invalid bounded attachment uncertainty: "+key)
        result[key]=float(value)
    return result


def attachment_uncertainty_geometry(dimensions,spec):
    policy=attachment_uncertainty_policy(spec)
    if policy is None:
        return None
    dims=vector(dimensions,3,"uncertain payload dimensions")
    if min(dims)<=0:
        raise ValueError("Payload dimensions must be positive")
    radius=math.sqrt(sum((d/2)**2 for d in dims))
    bound=policy["translation_m"]+2*radius*math.sin(policy["rotation_rad"]/2)
    if policy["payload_padding_m"]+1e-12 < bound:
        raise ValueError("Payload padding does not enclose attachment uncertainty")
    return {**policy,"nominal_com_to_corner_radius_m":radius,
            "maximum_payload_point_displacement_m":bound,
            "formula":"translation_m + 2*COM_corner_radius_m*sin(rotation_rad/2)",
            "conditional_on_runtime_relative_pose_bounds":True,
            "physical_continuous_collision_guarantee":False}


def payload_cover(dimensions,budget,*,padding_m=0.,cells=None):
    if isinstance(padding_m,bool) or not math.isfinite(padding_m) or not 0 <= padding_m <= .02:
        raise ValueError("Payload sphere padding must be finite within 0..20mm")
    if cells is not None:
        if (not isinstance(cells,(list,tuple)) or len(cells)!=3
                or any(type(n) is not int or not 1<=n<=16 for n in cells)
                or type(budget) is not int or math.prod(cells)!=budget or budget>512):
            raise ValueError("Invalid explicit payload covering grid")
        return box_cover(dimensions,max_cell=[d/n*(1+1e-12) for d,n in zip(dimensions,cells)],margin=padding_m)
    side=max(1,int(round(budget**(1/3))))
    while side**3>budget:
        side-=1
    return box_cover(dimensions,max_cell=[d/side*(1+1e-12) for d in dimensions],margin=padding_m)


def payload_cover_for_request(request):
    payload=request["payload"]
    uncertainty=attachment_uncertainty_geometry(payload["dimensions_m"],request.get("attachment_uncertainty"))
    padding=0. if uncertainty is None else uncertainty["payload_padding_m"]
    from depallet.manipulation.payload_cover_profiles import payload_cover_cells
    cells=payload_cover_cells(payload)
    spheres,cover=payload_cover(payload["dimensions_m"],payload.get("num_spheres",64),padding_m=padding,cells=cells)
    return spheres,cover,uncertainty


def validate_payload_sphere_radii(spheres,request):
    """Check actual registered GPU radii; fail if uncertainty padding was omitted."""
    expected,cover,uncertainty=payload_cover_for_request(request)
    actual=np.asarray(spheres,float)
    if actual.ndim!=2 or actual.shape[1]!=4 or not np.isfinite(actual).all():
        raise ValueError("Invalid actual payload GPU spheres")
    actual=actual[actual[:,3]>0]
    if len(actual)!=len(expected) or not np.allclose(actual[:,3],cover["radius_m"],atol=1e-7,rtol=1e-6):
        raise ValueError("Actual payload GPU radii/count omit expected padding")
    return {"passed":True,"expected_sphere_count":len(expected),"expected_radius_m":cover["radius_m"],
            "actual_minimum_radius_m":float(actual[:,3].min()),"actual_maximum_radius_m":float(actual[:,3].max()),
            "payload_padding_m":cover["margin_m"],"attachment_uncertainty":uncertainty}


def derive_payload_config(robot,request):
    """Return a copied config and an explicit certificate, never mutate input.

    Exclusion is valid only while this same confirmed rigid attachment and
    relative transform remain in place. Recompute after every new attachment.
    """
    payload=request.get("payload")
    hypothesis = payload.get("attachment_hypothesis") if isinstance(payload,dict) else None
    diagnostic_hypothesis = bool(
        payload and payload.get("grasp_confirmed") is False
        and request.get("diagnostic_only") is True
        and request.get("robot_execution_authorized") is False
        and request.get("physical_execution_validated") is False
        and isinstance(hypothesis,dict)
        and hypothesis.get("status") == "hypothetical_unverified"
        and hypothesis.get("grasp_evidence_available") is False)
    if not payload or (payload.get("grasp_confirmed") is not True and not diagnostic_hypothesis):
        raise ValueError("A physically confirmed payload or exact diagnostic hypothesis is required")
    config=copy.deepcopy(robot)
    kin=config.get("robot_cfg",config)["kinematics"]
    from depallet.manipulation.payload_cover_profiles import payload_cover_cells
    explicit_cells=payload_cover_cells(payload)
    if explicit_cells is not None:
        # Allocate only request-local attachment slots; preserve the pinned robot asset.
        kin.setdefault("extra_collision_spheres",{})["attached_object"]=math.prod(explicit_cells)
    tree=ET.parse(kin["urdf_path"]).getroot()
    body=next(link for link in tree.findall("link") if link.get("name")=="vgp20_body")
    colliders=body.findall("collision")
    if len(colliders)!=1 or colliders[0].find("geometry/box") is None:
        raise ValueError("Certificate requires exactly one VGP20 body collision box")
    collision=colliders[0]
    dimensions=np.asarray([float(v) for v in collision.find("geometry/box").get("size").split()])
    origin=collision.find("origin")
    xyz=[0.,0.,0.] if origin is None else [float(v) for v in origin.get("xyz","0 0 0").split()]
    rpy=[0.,0.,0.] if origin is None else [float(v) for v in origin.get("rpy","0 0 0").split()]
    body_pose=matrix(kin["urdf_path"],request["joint_names"],request["start_position_rad"],
                     tip="vgp20_body",base=kin["base_link"])
    body_center=body_pose[:3,3]+body_pose[:3,:3]@xyz
    body_rotation=body_pose[:3,:3]@Rotation.from_euler("xyz",rpy).as_matrix()
    bp=pose(payload["pose_base_wxyz"],"actual payload pose")
    box_center=np.asarray(bp[:3])
    box_rotation=Rotation.from_quat(bp[4:]+[bp[3]]).as_matrix()
    box_dimensions=np.asarray(vector(payload["dimensions_m"],3,"payload dimensions"))
    margin=.003
    sat=obb_separation(body_center,body_rotation,dimensions/2+margin,
                       box_center,box_rotation,box_dimensions/2)
    # Compare the same enclosing spheres used by the actual planner.
    body_spheres=kin["collision_spheres"]["vgp20_body"]
    pspheres,cover,uncertainty=payload_cover_for_request(request)
    bc=np.asarray([s["center"] for s in body_spheres])@body_pose[:3,:3].T+body_pose[:3,3]
    pc=np.asarray([s["center"] for s in pspheres])@box_rotation.T+box_center
    br=np.asarray([s["radius"] for s in body_spheres])
    pr=np.asarray([s["radius"] for s in pspheres])
    distances=np.linalg.norm(bc[:,None,:]-pc[None,:,:],axis=-1)-br[:,None]-pr[None,:]
    minimum=float(distances.min())
    location=np.unravel_index(int(distances.argmin()),distances.shape)
    relative=np.eye(4)
    relative[:3,:3]=body_pose[:3,:3].T@box_rotation
    relative[:3,3]=body_pose[:3,:3].T@(box_center-body_pose[:3,3])
    required_gap=.001+(0. if uncertainty is None else uncertainty["maximum_payload_point_displacement_m"])
    safe=bool(sat["nonoverlap"] and (sat["separating_gap_lower_bound_m"]>=required_gap if uncertainty is None
                                   else sat["separating_gap_lower_bound_m"]>required_gap))
    existing=kin.get("self_collision_ignore",{})
    preexisting=("vgp20_body" in existing.get("attached_object",[]) or
                 "attached_object" in existing.get("vgp20_body",[]))
    if preexisting:
        raise ValueError("Input config already excludes body/payload; a fresh certificate is required")
    applied=safe and minimum< -1e-6
    if applied:
        kin.setdefault("self_collision_ignore",{}).setdefault("attached_object",[]).append("vgp20_body")
    certificate={"schema":"depallet.fixed_payload_self_contact.v1",
        "safe_to_plan":safe,"box_id":payload["box_id"],
        "grasp_confirmed":payload.get("grasp_confirmed") is True,
        "diagnostic_hypothesis":diagnostic_hypothesis,
        "body_obb_inflation_per_face_m":margin,"required_separating_gap_m":required_gap,
        "body_collision_dimensions_m":dimensions.tolist(),"payload_dimensions_m":box_dimensions.tolist(),
        "actual_joint_position_rad":request["start_position_rad"],
        "payload_pose_base_wxyz":bp,"fixed_body_to_payload_transform":relative.tolist(),
        "obb_sat":sat,"sphere_pair_minimum_clearance_m":minimum,
        "sphere_pair_indices":[int(i) for i in location],
        "payload_cover":cover,
        "payload_axis_overhang_m":[cover["radius_m"]-w/2 for w in cover["cell_widths_m"]],
        "exception_applied":applied,
        "excluded_self_pairs":[["attached_object","vgp20_body"]] if applied else [],
        "all_world_collision_preserved":True,"all_other_self_pairs_preserved":True,
        "valid_only_while_same_confirmed_rigid_attachment":not diagnostic_hypothesis,
        "robot_execution_authorized":False if diagnostic_hypothesis else None,
        "physical_execution_validated":False,
        "scope":"simulation body collider + 3mm CAD allowance; not real suction or CAD mesh certification"}

    if uncertainty is not None:
        certificate.update(schema="depallet.uncertain_payload_self_contact.v1",
            attachment_uncertainty=uncertainty,
            nominal_body_to_payload_transform=relative.tolist(),
            nominal_gap_after_uncertainty_m=sat["separating_gap_lower_bound_m"]-uncertainty["maximum_payload_point_displacement_m"],
            valid_only_while_same_confirmed_rigid_attachment=False,
            valid_only_while_same_confirmed_attachment_and_relative_pose_bounds=True,
            physical_continuous_collision_guarantee=False,
            scope="nominal payload with bounded attachment uncertainty; runtime pose bounds and actual contacts required")
    return config,certificate
