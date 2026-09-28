"""CPU runtime gate for the remaining source stack using actual body states.

Box dimensions/support IDs are priors; box poses are never read from authored
layout fields. The pallet is an explicitly assumed static rectangular envelope.
This gate covers support/tilt/rest integrity, not all-pairs collision or grasp.
"""
from __future__ import annotations
import math
import numpy as np

LIMITS={"maximum_xy_overhang_m":.002,"maximum_absolute_bottom_plane_gap_m":.002,
        "maximum_world_tilt_deg":3.,"maximum_linear_speed_m_s":.02,
        "maximum_angular_speed_rad_s":.05}


def _array(value, shape, label):
    array=np.asarray(value)
    if array.dtype.kind not in "iuf" or array.shape!=shape:
        raise ValueError(label+": invalid shape or non-numeric dtype")
    array=array.astype(float)
    if not np.isfinite(array).all():raise ValueError(label+": nonfinite values")
    return array


def _ids(values,label):
    if not isinstance(values,(list,tuple)):
        raise ValueError(label+": expected list/tuple of unique IDs")
    if any(not isinstance(key,str) or not key for key in values):
        raise ValueError(label+": invalid ID")
    if len(set(values))!=len(values):raise ValueError(label+": duplicate IDs")
    return list(values)


def _rotation(q):
    norm=float(np.linalg.norm(q))
    if abs(norm-1.)>1e-4:raise ValueError("quaternions_wxyz: non-unit or zero quaternion")
    w,x,y,z=q/norm
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def _validate(spec, box_ids, state, source_pose, excluded_ids, report):
    if not isinstance(spec,dict) or not isinstance(state,dict):
        raise ValueError("spec and state must be mappings")
    boxes=spec["boxes"]
    if not isinstance(boxes,list) or not 1<=len(boxes)<=128:
        raise ValueError("spec.boxes must contain 1..128 declared boxes")
    spec_ids=_ids([box["id"] for box in boxes],"spec IDs")
    if "source_pallet" in spec_ids:raise ValueError("source_pallet is reserved")
    indexed={box["id"]:box for box in boxes}
    sizes={}
    for box in boxes:
        sizes[box["id"]]=_array(box["dimensions_m"],(3,),"box dimensions")
        if np.any(sizes[box["id"]]<=0):raise ValueError("box dimensions must be positive")
        support=box["support_id"]
        if not isinstance(support,str) or support not in set(spec_ids)|{"source_pallet"}:
            raise ValueError("unknown declared support: "+str(support))
    for key in spec_ids:
        seen=set();cursor=key
        while cursor!="source_pallet":
            if cursor in seen:raise ValueError("cyclic declared support graph")
            seen.add(cursor);cursor=indexed[cursor]["support_id"]
    pallet=_array(spec["pallet"]["dimensions_m"],(3,),"pallet dimensions")
    if np.any(pallet<=0):raise ValueError("pallet dimensions must be positive")
    if spec["pallet"].get("body_type")!="static_fixture":
        raise ValueError("source pallet must be an explicitly static fixture")
    source=_array(source_pose,(4,),"source_pose")
    supplied=_ids(box_ids,"measured box IDs");excluded=_ids(excluded_ids,"excluded IDs")
    unknown=set(supplied)-set(spec_ids)
    if unknown:raise ValueError("unknown measured box IDs: "+",".join(sorted(unknown)))
    if set(excluded)-set(spec_ids):raise ValueError("unknown excluded box IDs")
    remaining=[key for key in spec_ids if key not in excluded]
    missing=set(remaining)-set(supplied)
    if missing:raise ValueError("missing remaining measured box IDs: "+",".join(sorted(missing)))
    count=len(supplied)
    arrays={key:_array(state[key],(count,width),key) for key,width in
            [("positions_m",3),("quaternions_wxyz",4),("linear_velocities_m_s",3),("angular_velocities_rad_s",3)]}
    rotations=[_rotation(q) for q in arrays["quaternions_wxyz"]]
    if "box_ids" in state and _ids(state["box_ids"],"state.box_ids")!=supplied:
        raise ValueError("state.box_ids ordering differs from measured box_ids")
    if "sim_time" in state:
        time=_array(state["sim_time"],(),"sim_time").item()
        if time<0:raise ValueError("negative state sim_time")
        report["state_sim_time_s"]=time
    c,s=math.cos(source[3]),math.sin(source[3])
    pallet_rotation=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    pallet_position=source[:3]+pallet_rotation@np.array([0.,0.,pallet[2]/2])
    states={key:{"position":arrays["positions_m"][index],"rotation":rotations[index],
                 "linear_velocity":arrays["linear_velocities_m_s"][index],
                 "angular_velocity":arrays["angular_velocities_rad_s"][index]}
            for index,key in enumerate(supplied)}
    report.update(input_valid=True,remaining_ids=remaining,excluded_ids=excluded,
                  measured_ids=supplied,source_pose=list(source.astype(float)),
                  pallet_dimensions_m=pallet.tolist())
    signs=np.array([[-1.,-1.,-1.],[-1.,1.,-1.],[1.,1.,-1.],[1.,-1.,-1.]])
    for key in remaining:
        box=indexed[key];measured=states[key];R=measured["rotation"];p=measured["position"]
        support_id=box["support_id"];reasons=[]
        tilt=math.degrees(math.acos(float(np.clip(R[2,2],-1,1))))
        speed=float(np.linalg.norm(measured["linear_velocity"]))
        angular=float(np.linalg.norm(measured["angular_velocity"]))
        if tilt>LIMITS["maximum_world_tilt_deg"]+1e-10:reasons.append("world_tilt_exceeds_3deg")
        if speed>LIMITS["maximum_linear_speed_m_s"]+1e-12:reasons.append("linear_speed_exceeds_0p02m_s")
        if angular>LIMITS["maximum_angular_speed_rad_s"]+1e-12:reasons.append("angular_speed_exceeds_0p05rad_s")
        item={"box_id":key,"support_id":support_id,"passed":False,"failure_reasons":reasons,
              "position_world_m":p.tolist(),"world_tilt_deg":tilt,
              "linear_speed_m_s":speed,"angular_speed_rad_s":angular,
              "support_pose_source":"assumed_static_pallet" if support_id=="source_pallet" else "actual_measured_box"}
        if support_id in excluded:
            reasons.append("remaining_box_declares_excluded_support")
            report["per_box"].append(item);continue
        if support_id=="source_pallet":
            ps,rs,ds=pallet_position,pallet_rotation,pallet
        else:
            support=states[support_id]
            ps,rs,ds=support["position"],support["rotation"],sizes[support_id]
        corners=(signs*sizes[key]/2)@R.T+p
        local=(corners-ps)@rs
        xy_clearance=ds[:2]/2-np.abs(local[:,:2])
        overhang=max(0.,-float(xy_clearance.min()))
        gaps=local[:,2]-ds[2]/2
        if overhang>LIMITS["maximum_xy_overhang_m"]+1e-10:
            reasons.append("bottom_footprint_overhang_exceeds_2mm")
        if np.max(np.abs(gaps))>LIMITS["maximum_absolute_bottom_plane_gap_m"]+1e-10:
            reasons.append("bottom_plane_gap_or_penetration_exceeds_2mm")
        item.update(bottom_corners_world_m=corners.tolist(),
            bottom_corners_support_local_m=local.tolist(),
            bottom_corner_support_plane_gaps_m=gaps.tolist(),
            minimum_bottom_plane_gap_m=float(gaps.min()),maximum_bottom_plane_gap_m=float(gaps.max()),
            minimum_xy_edge_clearance_m=float(xy_clearance.min()),maximum_xy_overhang_m=overhang,
            passed=not reasons)
        report["per_box"].append(item)
    report["failed_box_ids"]=[item["box_id"] for item in report["per_box"] if not item["passed"]]
    report["failure_reasons"]=["remaining_source_stack_integrity_failed"] if report["failed_box_ids"] else []
    report["passed"]=not report["failed_box_ids"]
    return report


def validate_remaining_source_stack(spec, box_ids, state, source_pose=(0.,-.95,0.,0.), excluded_ids=()):
    """Fail closed on missing/unknown IDs or invalid measured poses/velocities.

    Excluded known boxes may be present in the measurement array or already
    absent; every remaining declared box is mandatory. A remaining box cannot
    claim an excluded support. No completed/future box is silently invented.
    """
    report={"schema":"depallet.remaining_source_stack_integrity.v1","passed":False,
        "input_valid":False,"failure_reasons":[],"per_box":[],"failed_box_ids":[],
        "limits":dict(LIMITS),"authored_box_poses_used":False,
        "box_state_source":"caller_supplied_actual_measured_rigid_body_states",
        "pallet_pose_source":"assumed_static_spec_envelope_plus_source_pose",
        "scope":"instantaneous remaining-source support footprint, plane gap, tilt and velocity",
        "whole_history_undisturbed_claimed":False,"full_collision_certificate":False,
        "real_robot_ready":False}
    try:
        return _validate(spec,box_ids,state,source_pose,excluded_ids,report)
    except (KeyError,ValueError,TypeError,IndexError,OverflowError) as error:
        report.update(passed=False,input_valid=False,
                      failure_reasons=["invalid_or_incomplete_measured_input: "+str(error)],
                      per_box=[],failed_box_ids=[])
        return report
