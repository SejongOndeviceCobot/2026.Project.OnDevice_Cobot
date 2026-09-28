"""Isaac Surface Gripper attachment points for the assumed VGP20 fixture.

Rigid simulated bodies use a D6 attachment, optionally with spring damping.
Force limits and pad grid are simulation
assumptions; cardboard leakage, crushing and vacuum circuit dynamics are absent.
"""
def create_vgp20_surface(stage, flange_path, assembly, *, attachment_policy="distributed", payload_mass_kg=None, payload_dimensions_m=None, physics_dt_s=1/240, damping_ratio=2., lateral_compliance_m=0.):
    from isaacsim.robot.surface_gripper import create_surface_gripper
    from pxr import Gf, Sdf, UsdPhysics, UsdGeom
    from usd.schema.isaac import robot_schema
    prim=create_surface_gripper(stage,flange_path)
    values={robot_schema.Attributes.MAX_GRIP_DISTANCE.name:.004,
            robot_schema.Attributes.COAXIAL_FORCE_LIMIT.name:450.,
            robot_schema.Attributes.SHEAR_FORCE_LIMIT.name:250.}
    for name,value in values.items():
        attribute=prim.GetAttribute(name)
        if not attribute:
            raise RuntimeError("Missing installed gripper schema attribute: "+name)
        attribute.Set(value)
    if attachment_policy not in ("distributed", "aggregate_center", "aggregate_compliant", "aggregate_compliant_si"):
        raise ValueError("Unknown rigid vacuum attachment policy")
    compliance=None
    if attachment_policy == "aggregate_compliant_si":
        from depallet.manipulation.vacuum_compliance import build_vacuum_compliance
        compliance=build_vacuum_compliance(payload_mass_kg,payload_dimensions_m,
            float(payload_dimensions_m[2])/2+.004,physics_dt_s=physics_dt_s,damping_ratio=damping_ratio,lateral_compliance_m=lateral_compliance_m)
    points=assembly["pads_flange_m"]
    if attachment_policy.startswith("aggregate_"):
        points=[[sum(p[axis] for p in points)/len(points) for axis in range(3)]]
    paths=[];applied_drives=[]
    flange_world=UsdGeom.Xformable(stage.GetPrimAtPath(flange_path)).ComputeLocalToWorldTransform(0.)
    world_rotation=flange_world.ExtractRotationQuat()
    for index,point in enumerate(points):
        path=Sdf.Path(flange_path+f"/VacuumContact_{index:02d}")
        joint=UsdPhysics.Joint.Define(stage,path)
        robot_schema.ApplyAttachmentPointAPI(joint.GetPrim())
        joint.GetPrim().GetAttribute(robot_schema.Attributes.FORWARD_AXIS.name).Set(UsdPhysics.Tokens.z)
        for axis in ("rotX","rotY","rotZ","transX","transY","transZ"):
            limit=UsdPhysics.LimitAPI.Apply(joint.GetPrim(),axis)
            if attachment_policy == "aggregate_compliant_si" and axis in compliance['axes']:
                params=compliance['axes'][axis]
                limit.CreateLowAttr(params['low']);limit.CreateHighAttr(params['high'])
                drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),axis)
                drive.CreateTypeAttr('force')
                drive.CreateStiffnessAttr(params['stiffness_usd']);drive.CreateDampingAttr(params['damping_usd'])
                drive.CreateTargetPositionAttr(0.);drive.CreateTargetVelocityAttr(0.)
                applied_drives.append({'joint_path':str(path),'axis':axis,
                    'low':limit.GetLowAttr().Get(),'high':limit.GetHighAttr().Get(),
                    'stiffness_usd':drive.GetStiffnessAttr().Get(),'damping_usd':drive.GetDampingAttr().Get(),
                    'drive_type':drive.GetTypeAttr().Get()})
            elif attachment_policy == "aggregate_compliant" and axis.startswith("rot"):
                limit.CreateLowAttr(-3.);limit.CreateHighAttr(3.)
                drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),axis)
                drive.CreateStiffnessAttr(10000. if axis == "rotZ" else 100.)
            elif attachment_policy == "aggregate_compliant" and axis == "transZ":
                limit.CreateLowAttr(0.);limit.CreateHighAttr(.01)
                drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),axis)
                drive.CreateDampingAttr(100.);drive.CreateStiffnessAttr(5000.)
            else:
                limit.CreateLowAttr(1.);limit.CreateHighAttr(-1.)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(flange_path)])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*point))
        joint.CreateLocalRot0Attr(Gf.Quatf(1.,0.,0.,0.))
        # PhysX must instantiate each D6 before Surface Gripper can own it.
        # Official SurfaceGripper_gantry.usda authors jointEnabled=true.
        # An initially empty body1 means world: align its anchor to the current
        # pad so initialization cannot pull the wrist toward world origin.
        world_point=flange_world.Transform(Gf.Vec3d(*point))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*world_point))
        joint.CreateLocalRot1Attr(Gf.Quatf(world_rotation))
        joint.CreateJointEnabledAttr(True)
        joint.CreateExcludeFromArticulationAttr(True)
        paths.append(path)
    prim.GetRelationship(robot_schema.Relations.ATTACHMENT_POINTS.name).SetTargets(paths)
    return {"gripper_path":str(prim.GetPath()),"joint_paths":[str(p) for p in paths],
        "contact_axis":"Z","max_grip_distance_m":.004,
        "assumed_coaxial_force_limit_N":450.,"assumed_shear_force_limit_N":250.,
        "attachment_policy":attachment_policy,"constraint_count":len(points),
        "nominal_pad_count":len(assembly["pads_flange_m"]),
        "aggregate_requires_full_pad_patch_geometry_check":attachment_policy.startswith("aggregate_"),
        "joint_compliance_source":("SI stiffness and payload-inertia-based damping, converted to USD degree units" if compliance is not None else "installed NVIDIA SurfaceGripper_gantry.usda D6 pattern" if attachment_policy=="aggregate_compliant" else None),
        "compliance_parameters_measured":False,"compliance_parameters":compliance,"applied_drives_readback":applied_drives,
        "pad_grid_source":"nominal_unmeasured","physics_attachment_validated":False,
        "joint_registration":"enabled D6 with initially aligned world anchor; Surface Gripper manages open/closed state",
        "initial_joint_body1":"unbound world; never a box preattachment"}


def sample_attachment_joint_schema(stage, gripper_info):
    """USD attributes only; native plugin attachment transforms may differ."""
    result=[]
    for path in gripper_info['joint_paths']:
        prim=stage.GetPrimAtPath(path)
        row={'joint_path':path,'source':'USD schema, not native constraint telemetry'}
        for name in ['isaac:clearanceOffset','physics:localPos0','physics:localPos1','physics:localRot0','physics:localRot1','physics:jointEnabled']:
            attr=prim.GetAttribute(name);value=attr.Get() if attr else None
            row[name]=value if value is None or isinstance(value,(int,float,bool,str)) else str(value)
        row['body0']=[str(p) for p in prim.GetRelationship('physics:body0').GetTargets()]
        row['body1']=[str(p) for p in prim.GetRelationship('physics:body1').GetTargets()]
        result.append(row)
    return result
