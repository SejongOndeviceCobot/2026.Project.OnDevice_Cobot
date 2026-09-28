"""Photo-informed pallet assets; CPU USD authoring without simulator startup.

Dimensions and physics are explicit mockup assumptions, never image measurements.
SI units, Z-up, column-vector transforms; pallet origin is bottom centre.
"""
from __future__ import annotations

import copy
import math


def default_spec():
    def physical(mass, interval):
        return dict(mass_kg=mass, mass_range_kg=interval, static_friction=.5, dynamic_friction=.4,
                    restitution=0., contact_offset_m=.002, rest_offset_m=0.,
                    center_of_mass_local_m=[0., 0., 0.],
                    inertia_model="uniform_solid_cuboid_proxy_about_center", measured=False,
                    source="deliberate simulation assumptions; unknown box contents",
                    excluded_physics=["cardboard deformation", "crush strength", "vacuum leakage", "moving payload"])
    boxes = []
    for column, x in enumerate([-.32, 0., .32]):
        support, bottom = "source_pallet", .15
        for layer, height in enumerate([.18, .18, .18, .32]):
            name = f"front_{column+1}_layer_{layer+1}"
            boxes.append(dict(id=name, dimensions_m=[.30, .57, height],
                position_source_m=[x, -.20, bottom+height/2], yaw_source_rad=0., support_id=support,
                physical=physical(1.5 if layer == 3 else 1., [.25, 3.]),
                visual_role="tall_front_carton" if layer == 3 else "flat_front_carton"))
            bottom += height
            support = name
    boxes.append(dict(id="rear_large", dimensions_m=[.66, .36, .65],
        position_source_m=[0., .315, .15+.65/2], yaw_source_rad=0., support_id="source_pallet",
        physical=physical(2., [.5, 5.]), visual_role="large_rear_carton"))
    support, bottom = "rear_large", .80
    for layer in range(3):
        name = f"rear_upper_{layer+1}"
        boxes.append(dict(id=name, dimensions_m=[.42, .32, .23],
            position_source_m=[0., .315, bottom+.115], yaw_source_rad=0., support_id=support,
            physical=physical(.7, [.2, 2.]), visual_role="rear_tower_carton"))
        bottom += .23
        support = name
    return dict(schema_version=1, scene_id="photo_green_pallet_v1",
        description="Photo-informed green lattice pallet and 16 supported cartons; not metric reconstruction",
        photo_observation=dict(visible_cuboid_hypothesis=16,
            layout="three front columns of four; one rear large carton and three above it",
            hidden_objects="unknown; no hidden objects invented", scale_calibrated=False,
            camera_calibrated=False, occluded_rear_geometry="assumed large carton extends down to pallet"),
        units=dict(length="m", mass="kg", angle="rad"),
        frame_convention="right-handed Z-up; column-vector T_world_object; quaternion wxyz",
        pallet=dict(dimensions_m=[1.10, 1.10, .15], dimensions_measured=False,
            material_identity="unconfirmed plastic", body_type="static_fixture", mass_kg=None,
            mass_reason="static fixture; no fabricated dynamic mass", static_friction=.4,
            dynamic_friction=.3, restitution=0., contact_offset_m=.002, rest_offset_m=0.,
            measured=False, source="visual approximation and assumed 1100 mm square pallet",
            geometry="open compound lattice and three runners; not manufacturer CAD"),
        boxes=boxes, physical_calibration_complete=False, suction_calibrated=False, valid_sim_hours=0.)


def _check(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(values):
    return all(isinstance(v, (int, float)) and math.isfinite(v) for v in values)


def _rectangle(box):
    x, y, _ = box["position_source_m"]
    dx, dy, _ = box["dimensions_m"]
    a = box["yaw_source_rad"]
    c, s = math.cos(a), math.sin(a)
    return [(x+c*u-s*v, y+s*u+c*v) for u, v in
            [(-dx/2, -dy/2), (dx/2, -dy/2), (dx/2, dy/2), (-dx/2, dy/2)]]


def _overlap_xy(a, b, tolerance=1e-8):
    ra, rb = _rectangle(a), _rectangle(b)
    for angle in [a["yaw_source_rad"], b["yaw_source_rad"]]:
        for axis in [(math.cos(angle), math.sin(angle)), (-math.sin(angle), math.cos(angle))]:
            pa = [x*axis[0]+y*axis[1] for x, y in ra]
            pb = [x*axis[0]+y*axis[1] for x, y in rb]
            if min(max(pa), max(pb))-max(min(pa), min(pb)) <= tolerance:
                return False
    return True


def validate_spec(spec):
    """Check finite SI geometry, support DAG, full footprints and volume intersections."""
    dims = spec["pallet"]["dimensions_m"]
    _check(len(dims) == 3 and _finite(dims) and dims[0] >= .3 and dims[1] >= .3 and dims[2] >= .08,
           "Pallet dimensions below supported lattice construction limits")
    pallet_physics = spec["pallet"]
    _check(pallet_physics["measured"] is False and pallet_physics["body_type"] == "static_fixture",
           "Pallet must be an explicitly unmeasured static fixture")
    _check(_finite([pallet_physics[k] for k in ["static_friction", "dynamic_friction", "restitution", "contact_offset_m", "rest_offset_m"]])
           and 0 <= pallet_physics["dynamic_friction"] <= pallet_physics["static_friction"] <= 1.5
           and 0 <= pallet_physics["restitution"] <= 1
           and 0 <= pallet_physics["rest_offset_m"] < pallet_physics["contact_offset_m"] < .01,
           "Invalid pallet contact parameters")
    boxes = spec["boxes"]
    ids = [b["id"] for b in boxes]
    _check(len(ids) == len(set(ids)) and "source_pallet" not in ids, "Duplicate/reserved box ID")
    _check(all(i and i.replace("_", "").isalnum() and not i[0].isdigit() for i in ids), "Invalid prim ID")
    indexed = {b["id"]: b for b in boxes}
    pallet = dict(position_source_m=[0., 0., dims[2]/2], dimensions_m=dims, yaw_source_rad=0.)
    support_checks = []
    for box in boxes:
        bd, pos, yaw = box["dimensions_m"], box["position_source_m"], box["yaw_source_rad"]
        _check(len(bd) == 3 and len(pos) == 3 and _finite([*bd, *pos, yaw]) and min(bd) > 0,
               f"Invalid box geometry: {box['id']}")
        p = box["physical"]
        vals = [p[k] for k in ["mass_kg", "static_friction", "dynamic_friction", "restitution", "contact_offset_m", "rest_offset_m"]]
        _check(_finite(vals) and vals[0] > 0 and 0 <= vals[2] <= vals[1] <= 1.5 and 0 <= vals[3] <= 1,
               f"Invalid physical assumptions: {box['id']}")
        _check(0 <= vals[5] < vals[4] < min(bd)/4, f"Invalid contact offsets: {box['id']}")
        _check(p["measured"] is False, "This generator cannot label assumptions as measured")
        com = p["center_of_mass_local_m"]
        _check(len(com) == 3 and _finite(com) and all(abs(com[i]) < bd[i]/2 for i in range(3)),
               f"Declared rigid COM must lie inside its cuboid: {box['id']}")
        _check(p["inertia_model"] in ("uniform_solid_cuboid_proxy_about_center",
                                      "cuboid_diagonal_proxy_about_declared_com"),
               "Unsupported inertia model")
        _check(p["inertia_model"] != "uniform_solid_cuboid_proxy_about_center" or com == [0., 0., 0.],
               "Centered inertia model cannot declare offset COM")
        _check(box.get("shape", {}).get("kind", "rigid_cuboid") == "rigid_cuboid",
               "Non-cuboid, deformable or moving-content shapes are not supported by this author")
        visual = box.get("visual", {})
        if visual:
            _check(visual.get("measured") is False, "Visual parameters must be explicit assumptions")
            color = visual.get("color_rgb")
            roughness = visual.get("roughness")
            _check(isinstance(color, (list, tuple)) and len(color) == 3 and _finite(color)
                   and all(0 <= c <= 1 for c in color) and _finite([roughness]) and 0 <= roughness <= 1,
                   "Invalid assumed color or roughness")
            _check(isinstance(visual.get("material_category"), str) and visual["material_category"],
                   "Material category must be named")
        interval = p["mass_range_kg"]
        _check(len(interval) == 2 and _finite(interval) and 0 < interval[0] <= vals[0] <= interval[1],
               "Mass must lie within its positive assumption range")
        ancestors = set()
        current = box["id"]
        while current != "source_pallet":
            _check(current in indexed and current not in ancestors, f"Missing support or cyclic support graph: {box['id']}")
            ancestors.add(current)
            current = indexed[current]["support_id"]
        support = pallet if box["support_id"] == "source_pallet" else indexed[box["support_id"]]
        gap = pos[2]-bd[2]/2-(support["position_source_m"][2]+support["dimensions_m"][2]/2)
        _check(abs(gap) < 1e-7, f"Unsupported vertical gap: {box['id']}: {gap}")
        sx, sy, _ = support["position_source_m"]
        c, s = math.cos(support["yaw_source_rad"]), math.sin(support["yaw_source_rad"])
        for x, y in _rectangle(box):
            local = (c*(x-sx)+s*(y-sy), -s*(x-sx)+c*(y-sy))
            _check(abs(local[0]) <= support["dimensions_m"][0]/2+1e-8 and
                   abs(local[1]) <= support["dimensions_m"][1]/2+1e-8,
                   f"Box footprint overhangs support: {box['id']}")
        support_checks.append(dict(box_id=box["id"], support_id=box["support_id"], footprint_supported=True, gap_m=gap))
    pair_count = 0
    for i, a in enumerate(boxes):
        for b in boxes[i+1:]:
            pair_count += 1
            az, bz = a["position_source_m"][2], b["position_source_m"][2]
            ah, bh = a["dimensions_m"][2], b["dimensions_m"][2]
            z_overlap = min(az+ah/2, bz+bh/2)-max(az-ah/2, bz-bh/2)
            _check(z_overlap <= 1e-8 or not _overlap_xy(a, b), f"Box volume intersection: {a['id']}, {b['id']}")
    parents = {b["support_id"] for b in boxes}
    return dict(passed=True, box_count=len(boxes), pairwise_checks=pair_count,
                support_checks=support_checks, top_box_candidates=[b["id"] for b in boxes if b["id"] not in parents],
                validation_scope="analytic cuboid layout only; pallet lattice and dynamics require Isaac validation",
                physics_executed=False, photo_reconstruction_verified=False)


def world_boxes(spec, source_pose=(0., 0., 0., 0.)):
    _check(len(source_pose) == 4 and _finite(source_pose), "Invalid source pose")
    tx, ty, tz, yaw = source_pose
    c, s = math.cos(yaw), math.sin(yaw)
    result = []
    for original in spec["boxes"]:
        box = copy.deepcopy(original)
        x, y, z = box["position_source_m"]
        pos = [tx+c*x-s*y, ty+s*x+c*y, tz+z]
        a = yaw+box["yaw_source_rad"]
        box.update(position_world_m=pos, quaternion_world_wxyz=[math.cos(a/2), 0., 0., math.sin(a/2)],
            T_world_object=[[math.cos(a), -math.sin(a), 0., pos[0]], [math.sin(a), math.cos(a), 0., pos[1]],
                            [0., 0., 1., pos[2]], [0., 0., 0., 1.]],
            top_face_center_world_m=[pos[0], pos[1], pos[2]+box["dimensions_m"][2]/2],
            top_face_normal_world=[0., 0., 1.], pose_source="authored_sim_ground_truth_not_pose_estimation")
        result.append(box)
    return result


def build_scene(stage, spec, root_path="/World/PhotoScene", source_pose=(0., 0., 0., 0.),
                goal_pose=(1.5, 0., 0., 0.), include_boxes=True):
    """Author static compound pallets and rigid cartons; no lights/robot/physics scene.

    Caller owns stage root identity, robot placement, timeline and runtime checks.
    Each box is one rigid root with a cuboid collider and visual tape children.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade
    report = validate_spec(spec)
    _check(abs(UsdGeom.GetStageMetersPerUnit(stage)-1.) < 1e-9 and UsdGeom.GetStageUpAxis(stage) == "Z",
           "build_scene requires a stage configured for metres and Z-up")
    _check(len(goal_pose) == 4 and _finite(goal_pose), "Invalid goal pose")
    _check(not stage.GetPrimAtPath(root_path).IsValid(), f"Output prim already exists: {root_path}")
    authored = world_boxes(spec, source_pose)
    pallet_dims = spec["pallet"]["dimensions_m"]
    source_rect = dict(position_source_m=source_pose[:3], yaw_source_rad=source_pose[3], dimensions_m=pallet_dims)
    goal_rect = dict(position_source_m=goal_pose[:3], yaw_source_rad=goal_pose[3], dimensions_m=pallet_dims)
    _check(abs(source_pose[2]-goal_pose[2]) >= pallet_dims[2] or not _overlap_xy(source_rect, goal_rect),
           "Source and goal pallets intersect")
    UsdGeom.Xform.Define(stage, root_path)
    UsdGeom.Scope.Define(stage, root_path+"/Looks")

    def material(name, color, roughness, physical=None):
        mat = UsdShade.Material.Define(stage, root_path+"/Looks/"+name)
        shader = UsdShade.Shader.Define(stage, str(mat.GetPath())+"/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.)
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.))
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        if physical:
            api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            api.CreateStaticFrictionAttr(physical["static_friction"])
            api.CreateDynamicFrictionAttr(physical["dynamic_friction"])
            api.CreateRestitutionAttr(physical["restitution"])
        return mat

    pallet_mat = material("GreenPlastic", (.025, .19, .14), .72, spec["pallet"])
    tape = material("PackingTape", (.40, .265, .13), .58)
    edge = material("FoldSeam", (.22, .135, .06), .94)

    def cube(path, position, dimensions, mat, physical=None):
        geom = UsdGeom.Cube.Define(stage, path)
        geom.CreateSizeAttr(1.)
        xf = UsdGeom.Xformable(geom)
        xf.AddTranslateOp().Set(Gf.Vec3d(*position))
        xf.AddScaleOp().Set(Gf.Vec3f(*dimensions))
        prim = geom.GetPrim()
        bind = UsdShade.MaterialBindingAPI.Apply(prim)
        bind.Bind(mat)
        if physical:
            bind.Bind(mat, materialPurpose="physics")
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            # Author PhysX tokens without loading Isaac or requiring its plugin.
            schemas = list(prim.GetMetadata("apiSchemas").GetAddedOrExplicitItems())
            prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(schemas+["PhysxCollisionAPI"]))
            for attr, val in [("physxCollision:contactOffset", physical["contact_offset_m"]),
                              ("physxCollision:restOffset", physical["rest_offset_m"])]:
                prim.CreateAttribute(attr, Sdf.ValueTypeNames.Float, custom=False).Set(val)
        return geom

    pallet_colliders = []
    def pallet(name, pose):
        path = root_path+"/"+name
        xf = UsdGeom.Xform.Define(stage, path)
        xf.AddTranslateOp().Set(Gf.Vec3d(*pose[:3]))
        xf.AddRotateZOp().Set(math.degrees(pose[3]))
        dx, dy, height = spec["pallet"]["dimensions_m"]
        deck_t, rib_w = min(.022, height/5), .026
        for axis in [0, 1]:
            span, long_span = (dx, dy) if axis == 0 else (dy, dx)
            for index in range(12):
                offset = -span/2+rib_w/2+index*(span-rib_w)/11
                pos = [offset, 0., height-deck_t/2] if axis == 0 else [0., offset, height-deck_t/2]
                size = [rib_w, long_span, deck_t] if axis == 0 else [long_span, rib_w, deck_t]
                geom = cube(path+f"/Deck_{axis}_{index}", pos, size, pallet_mat, spec["pallet"])
                pallet_colliders.append(str(geom.GetPath()))
        runner_h = .025
        for index, x in enumerate([-dx/2+.075, 0., dx/2-.075]):
            geom = cube(path+f"/Runner_{index}", (x, 0., runner_h/2), (.14, dy, runner_h), pallet_mat, spec["pallet"])
            pallet_colliders.append(str(geom.GetPath()))
            for post, y in enumerate([-dy/2+.08, 0., dy/2-.08]):
                geom = cube(path+f"/Post_{index}_{post}", (x, y, (runner_h+height-deck_t)/2),
                            (.14, .14, height-deck_t-runner_h), pallet_mat, spec["pallet"])
                pallet_colliders.append(str(geom.GetPath()))
        xf.GetPrim().CreateAttribute("photoScene:measured", Sdf.ValueTypeNames.Bool).Set(False)
        return path

    source_path, goal_path = pallet("SourcePallet", source_pose), pallet("GoalPallet", goal_pose)
    paths = {}
    if include_boxes:
        for box in authored:
            path = root_path+"/Boxes/"+box["id"]
            xf = UsdGeom.Xform.Define(stage, path)
            xf.AddTranslateOp().Set(Gf.Vec3d(*box["position_world_m"]))
            q = box["quaternion_world_wxyz"]
            xf.AddOrientOp().Set(Gf.Quatf(q[0], Gf.Vec3f(*q[1:])))
            prim = xf.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
            mass = UsdPhysics.MassAPI.Apply(prim)
            m = box["physical"]["mass_kg"]
            dx, dy, dz = box["dimensions_m"]
            mass.CreateMassAttr(m)
            mass.CreateCenterOfMassAttr(Gf.Vec3f(*box["physical"]["center_of_mass_local_m"]))
            # For offset COM this is an explicitly assumed diagonal inertia ABOUT
            # that COM; it is not a uniform-density claim or a calibrated tensor.
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(m*(dy*dy+dz*dz)/12, m*(dx*dx+dz*dz)/12, m*(dx*dx+dy*dy)/12))
            mass.CreatePrincipalAxesAttr(Gf.Quatf(1., Gf.Vec3f(0.)))
            # Modest carton colour variation and visible folded edges separate
            # touching boxes without introducing a gap in the collision model.
            tone = (0.91, 1.04, 0.97, 1.10)[len(paths) % 4]
            visual = box.get("visual", {})
            mat = material("Physics_"+box["id"],
                           visual.get("color_rgb", (.43*tone, .28*tone, .15*tone)),
                           visual.get("roughness", .94), box["physical"])
            prim.CreateAttribute("photoScene:materialCategory", Sdf.ValueTypeNames.String).Set(
                visual.get("material_category", "brown_cardboard_proxy"))
            prim.CreateAttribute("photoScene:inertiaModel", Sdf.ValueTypeNames.String).Set(
                box["physical"]["inertia_model"])
            cube(path+"/Body", (0., 0., 0.), box["dimensions_m"], mat, box["physical"])
            cube(path+"/TapeTop", (0., 0., dz/2+.00015), (.035, dy, .0003), tape)
            cube(path+"/TapeFront", (0., -dy/2-.00015, 0.), (.035, .0003, dz), tape)
            cube(path+"/TopFold", (0., 0., dz/2+.00035), (.0018, dy, .00015), edge)
            for sign, suffix in [(-1., "Bottom"), (1., "Top")]:
                z = sign*(dz/2-.001)
                cube(path+"/FrontFold"+suffix, (0., -dy/2-.00025, z), (dx, .0004, .002), edge)
                cube(path+"/RightFold"+suffix, (dx/2+.00025, 0., z), (.0004, dy, .002), edge)
                cube(path+"/LeftFold"+suffix, (-dx/2-.00025, 0., z), (.0004, dy, .002), edge)
            for attr, value in [("boxId", box["id"]), ("supportId", box["support_id"])]:
                prim.CreateAttribute("photoScene:"+attr, Sdf.ValueTypeNames.String).Set(value)
            prim.CreateAttribute("photoScene:measured", Sdf.ValueTypeNames.Bool).Set(False)
            paths[box["id"]] = path
    return dict(source_pallet_path=source_path, goal_pallet_path=goal_path, box_paths=paths,
                pallet_collider_paths=pallet_colliders, transformed_boxes=authored,
                layout_validation=report, root_path=root_path, physics_executed=False)
