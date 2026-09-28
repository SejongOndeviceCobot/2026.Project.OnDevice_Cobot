"""Deterministic CPU scenario definitions and geometry-only whole-pallet plans.

No simulator, model inference, robot command, or measured-material claim lives here.
The authored specification is simulation ground truth, never perception output.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import random

from photo_scene import default_spec, validate_spec, _overlap_xy
from task_planning import deterministic_box_order, plan_goal_packing, validate_box_order

CELL = dict(source_pose=[0., -.78, 0., 0.], goal_pose=[0., .78, 0., 0.],
            robot_base_position_m=[-.5, 0., .25], robot_base_quaternion_wxyz=[1., 0., 0., 0.],
            pallet_dimensions_m=[1., .8, .15], robot_reach_verified=False)
SCENARIO_IDS = ("v1_uniform", "v2_three_sizes_single_material",
                "v2_three_sizes_mixed_material", "v3_unseen_cuboids",
                "v3_unseen_offset_com", "v3_non_cuboid_probe", "v3_labile_probe")
MATERIALS = {
    "brown_cardboard": dict(color_rgb=[.43, .28, .15], roughness=.94, static_friction=.5, dynamic_friction=.4),
    "white_coated_cardboard": dict(color_rgb=[.78, .78, .73], roughness=.55, static_friction=.4, dynamic_friction=.3),
    "blue_polymer_proxy": dict(color_rgb=[.06, .17, .39], roughness=.35, static_friction=.35, dynamic_friction=.25),
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def _finite(values):
    return all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in values)


def make_scenario(scenario_id, seed=0):
    """Return {spec, cell, capabilities, provenance}; no implicit physics execution."""
    if scenario_id not in SCENARIO_IDS:
        raise ValueError("Unknown scenario ID")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an unsigned 32-bit integer")
    rng = random.Random(seed)
    com_rng = random.Random(seed ^ 0xC0FFEE)
    level = scenario_id[:2].upper()
    spec = default_spec()
    spec["scene_id"] = scenario_id + "_seed_" + str(seed)
    spec["description"] = "Synthetic deterministic scenario, not photo reconstruction or model inference"
    spec["photo_observation"] = dict(applicable=False, scale_calibrated=False, camera_calibrated=False)
    spec["pallet"]["dimensions_m"] = list(CELL["pallet_dimensions_m"])
    spec["pallet"]["source"] = "assumed compact benchmark pallet, not manufacturer measurement"
    spec["boxes"] = []
    tower_xy = [(-.235, -.185), (.235, -.185), (-.235, .185), (.235, .185)]
    template = default_spec()["boxes"][0]["physical"]
    for tower, (cx, cy) in enumerate(tower_xy):
        if level == "V1":
            dx, dy, height = .42, .32, .23
            mass = .7
            yaw = 0.
        elif level == "V2":
            dx, dy, height, mass = [(.42, .32, .18, .6), (.40, .30, .22, .8),
                                    (.38, .30, .26, 1.), (.42, .32, .18, .6)][tower]
            yaw = 0.
        else:
            dx, dy = round(rng.uniform(.35, .42), 6), round(rng.uniform(.295, .32), 6)
            height, mass = None, None
            yaw = rng.uniform(-.02, .02)
            cx, cy = cx+rng.uniform(-.002, .002), cy+rng.uniform(-.002, .002)
        bottom, support = .15, "source_pallet"
        for layer in range(4):
            h = height if height is not None else round(rng.uniform(.17, .23), 6)
            m = mass if mass is not None else round(rng.uniform(.4, .9), 6)
            name = f"box_{tower*4+layer+1:02d}"
            mixed = scenario_id == "v2_three_sizes_mixed_material" or level == "V3"
            category = list(MATERIALS)[(tower+layer) % 3] if mixed else "brown_cardboard"
            material = copy.deepcopy(MATERIALS[category])
            if level == "V3":
                material["roughness"] = round(rng.uniform(.3, .96), 6)
                material["static_friction"] = round(rng.uniform(.38, .62), 6)
                material["dynamic_friction"] = round(material["static_friction"] * rng.uniform(.65, .85), 6)
            p = copy.deepcopy(template)
            p.update(mass_kg=m, mass_range_kg=[.4, .9] if level == "V3" else [m, m],
                     static_friction=material["static_friction"],
                     dynamic_friction=material["dynamic_friction"],
                     source="seeded synthetic assumption; not measured or estimated")
            if scenario_id == "v3_unseen_offset_com":
                p["center_of_mass_local_m"] = [round(com_rng.uniform(-.005, .005), 6) for _ in range(3)]
                p["inertia_model"] = "cuboid_diagonal_proxy_about_declared_com"
            box = dict(id=name, dimensions_m=[dx, dy, h],
                       position_source_m=[cx, cy, bottom+h/2], yaw_source_rad=yaw, support_id=support,
                       physical=p, visual_role="scenario_carton",
                       visual=dict(color_rgb=material["color_rgb"], roughness=material["roughness"],
                                   material_category=category, measured=False),
                       shape=dict(kind="rigid_cuboid", dimensions_source="authored_simulation_ground_truth",
                                  known_to_perception=level != "V3"),
                       surface=dict(condition="planar_dry_proxy", roughness=material["roughness"],
                                    measured=False, suction_seal_model_validated=False))
            spec["boxes"].append(box)
            bottom, support = bottom+h, name
    unsupported = []
    if scenario_id == "v3_non_cuboid_probe":
        spec["boxes"][-1]["shape"]["kind"] = "non_cuboid_crumpled_bag"
        unsupported = ["non-cuboid collision, grasp contact, and support models are not implemented"]
    if scenario_id == "v3_labile_probe":
        spec["boxes"][-1]["shape"]["kind"] = "labile_contents"
        unsupported = ["moving contents and deformation are not implemented"]
    return dict(schema="depallet.scenario.v1", scenario_id=scenario_id, level=level, seed=seed,
                representative=scenario_id in ("v1_uniform", "v2_three_sizes_single_material", "v3_unseen_cuboids"),
                spec=spec, cell=copy.deepcopy(CELL),
                capabilities=dict(geometry_kind="rigid_cuboid", unsupported_reasons=unsupported,
                                  noncuboid_supported=False, moving_contents_supported=False,
                                  measured_physics=False, physical_execution_validated=False),
                provenance=dict(generator="scenario_suite.make_scenario", provider="deterministic_synthetic_definition",
                    dimensions_source="authored_simulation_ground_truth",
                    material_source="unmeasured bounded synthetic assumptions",
                    com_source="unmeasured declared rigid-body assumption",
                    inference_executed=False, autonomous_model_executed=False, fallback_used=False,
                    unseen_definition="V3 dimensions sampled outside the finite V1/V2 size catalog; no model training distribution was audited",
                    split="held_out_size_catalog" if level == "V3" else "development_catalog",
                    irregular_stack_definition="V3 unequal heights and tower footprints, small source yaw/XY variation; upright rigid supported towers",
                    per_object_unique_xy_dimensions=False))


def _center_packing(plan):
    rects = plan["floor_rectangles"]
    lo = [min(r[k] for r in rects) for k in ("x", "y")]
    hi = [max(r[k]+r[s] for r in rects) for k, s in (("x", "w"), ("y", "d"))]
    shift = [(plan["pallet_dimensions_m"][i]-lo[i]-hi[i])/2 for i in range(2)]
    for rect in rects:
        rect["x"] += shift[0]
        rect["y"] += shift[1]
    for p in plan["placements"]:
        for key in ("position_goal_m", "top_face_center_goal_m"):
            p[key][0] += shift[0]
            p[key][1] += shift[1]
    plan["centering_translation_m"] = shift


def validate_goal_layout(plan, spec):
    """Check every declared ID, full support, overlap, and subtree COM stability.

    Offset COM is an assumed fixed rigid-body parameter. Load/crush capacity,
    frictional dynamics, disturbance stability, and robot reach remain unverified.
    """
    errors = []
    try:
        source = {b["id"]: b for b in spec["boxes"]}
        box_count = len(spec["boxes"])
        if box_count < 1 or len(source) != box_count:
            raise ValueError("At least one uniquely named source box is required")
        if plan["pallet_dimensions_m"] != spec["pallet"]["dimensions_m"]:
            raise ValueError("Pallet dimensions mismatch")
        order, placements = plan["order"], plan["placements"]
        if len(order) != box_count or set(order) != set(source) or len(set(order)) != box_count:
            raise ValueError("Order must contain all source IDs exactly once")
        if [p["box_id"] for p in placements] != order:
            raise ValueError("Placement sequence does not match order")
        pd = plan["pallet_dimensions_m"]
        margin, gap, limit = plan["edge_margin_m"], plan["inter_stack_gap_m"], plan["max_cargo_height_m"]
        if not _finite([*pd, margin, gap, limit]) or min(pd) <= 0 or min(margin, gap) < 0 or limit <= 0:
            raise ValueError("Invalid packing constraints")
        seen, boxes, descendants = {}, [], {}
        for p in placements:
            name = p["box_id"]
            bd, pos, yaw, com = p["dimensions_m"], p["position_goal_m"], p["yaw_goal_rad"], p["center_of_mass_local_m"]
            if len(bd) != 3 or len(pos) != 3 or len(com) != 3 or not _finite([*bd, *pos, yaw, *com, p["mass_kg"]]) or min(bd) <= 0 or p["mass_kg"] <= 0:
                raise ValueError("Invalid placement values: " + name)
            actual = source[name]
            if bd != actual["dimensions_m"] or com != actual["physical"]["center_of_mass_local_m"] or p["mass_kg"] != actual["physical"]["mass_kg"]:
                raise ValueError("Placement changes declared dimensions/mass/COM: " + name)
            if any(abs(com[i]) >= bd[i]/2 for i in range(3)):
                raise ValueError("COM outside box: " + name)
            if abs(yaw/(math.pi/2)-round(yaw/(math.pi/2))) > 1e-7:
                raise ValueError("Goal yaw must be orthogonal: " + name)
            c, s = math.cos(yaw), math.sin(yaw)
            w, d = abs(c)*bd[0]+abs(s)*bd[1], abs(s)*bd[0]+abs(c)*bd[1]
            if len(p["footprint_goal_m"]) != 2 or max(abs(p["footprint_goal_m"][i]-v) for i,v in enumerate((w,d))) > 1e-7:
                raise ValueError("Footprint inconsistent with yaw: " + name)
            if abs(pos[0])+w/2 > pd[0]/2-margin+1e-8 or abs(pos[1])+d/2 > pd[1]/2-margin+1e-8:
                errors.append("Pallet bounds: " + name)
            if pos[2]+bd[2]/2 > pd[2]+limit+1e-8:
                errors.append("Cargo height: " + name)
            support = p["support_id"]
            if support == "goal_pallet":
                sx, sy, sw, sd, top = 0., 0., pd[0], pd[1], pd[2]
            elif support in seen:
                below = seen[support]
                sx, sy = below["position_goal_m"][:2]
                sw, sd = below["footprint_goal_m"]
                top = below["position_goal_m"][2]+below["dimensions_m"][2]/2
                if below["stack_id"] != p["stack_id"]:
                    errors.append("Declared stack changes at support: " + name)
            else:
                raise ValueError("Support must be placed first: " + name)
            if abs(pos[2]-bd[2]/2-top) > 1e-7:
                errors.append("Support gap or penetration: " + name)
            if abs(pos[0]-sx)+w/2 > sw/2+1e-8 or abs(pos[1]-sy)+d/2 > sd/2+1e-8:
                errors.append("Incomplete support footprint: " + name)
            box = dict(id=name, position_source_m=pos, dimensions_m=bd, yaw_source_rad=yaw)
            for other in boxes:
                oz, oh = other["position_source_m"][2], other["dimensions_m"][2]
                if min(pos[2]+bd[2]/2, oz+oh/2)-max(pos[2]-bd[2]/2, oz-oh/2) > 1e-8:
                    if _overlap_xy(box, other):
                        errors.append("Volume overlap: "+name+"/"+other["id"])
                    op = seen[other["id"]]
                    if op["stack_id"] != p["stack_id"]:
                        ox, oy = op["position_goal_m"][:2]
                        ow, od = op["footprint_goal_m"]
                        if abs(pos[0]-ox) < (w+ow)/2+gap-1e-8 and abs(pos[1]-oy) < (d+od)/2+gap-1e-8:
                            errors.append("Inter-stack clearance: "+name+"/"+other["id"])
            seen[name] = p
            boxes.append(box)
            descendants[name] = [name]
        # Mass-weighted resultant of each full supported subtree must lie within
        # its own bottom footprint. Check each interface, not only each box COM.
        for name in reversed(order):
            support = seen[name]["support_id"]
            if support in descendants:
                descendants[support].extend(descendants[name])
        com_checks = []
        for name in order:
            p, weighted, mass = seen[name], [0., 0.], 0.
            for child in descendants[name]:
                item = seen[child]
                a = item["yaw_goal_rad"]
                cx, cy, _ = item["center_of_mass_local_m"]
                world = [item["position_goal_m"][0]+math.cos(a)*cx-math.sin(a)*cy,
                         item["position_goal_m"][1]+math.sin(a)*cx+math.cos(a)*cy]
                m = item["mass_kg"]
                mass += m
                weighted = [weighted[i]+m*world[i] for i in range(2)]
            center = [v/mass for v in weighted]
            stability_margin = min(p["footprint_goal_m"][i]/2-abs(center[i]-p["position_goal_m"][i]) for i in range(2))
            ok = stability_margin >= .005
            if not ok:
                errors.append("Subtree COM outside 5mm support margin: "+name)
            com_checks.append(dict(box_id=name, supported_mass_kg=mass, resultant_xy_m=center,
                                   minimum_support_margin_m=stability_margin, passed=ok))
        return dict(passed=not errors, errors=errors, placement_count=len(placements),
                    full_rectangular_support_checked=True, subtree_com_checks=com_checks,
                    assumed_static_stability_checked=True, dynamics_validated=False,
                    crush_capacity_validated=False, robot_reach_verified=False)
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
        return dict(passed=False, errors=errors+[str(exc)], fail_closed=True,
                    dynamics_validated=False, robot_reach_verified=False)


def packing_metrics(plan):
    """Reported objectives, not a claim of globally optimal packing."""
    ps = plan["placements"]
    tops = [p["position_goal_m"][2]+p["dimensions_m"][2]/2 for p in ps]
    stack_tops = {}
    levels = {}
    for p in ps:
        stack_tops[p["stack_id"]] = max(stack_tops.get(p["stack_id"], 0.), p["position_goal_m"][2]+p["dimensions_m"][2]/2)
        z = p["position_goal_m"][2]-p["dimensions_m"][2]/2
        levels.setdefault(round(z, 6), []).append(p["box_id"])
    heights = list(stack_tops.values())
    mean = sum(heights)/len(heights)
    max_height = max(tops)-plan["pallet_dimensions_m"][2]
    volume = sum(math.prod(p["dimensions_m"]) for p in ps)
    area = math.prod(plan["pallet_dimensions_m"][:2])
    return dict(floor_area_utilization=plan["floor_area_utilization"],
                cargo_bounding_volume_utilization=volume/(area*max_height),
                maximum_cargo_height_m=max_height, stack_top_height_std_m=(sum((h-mean)**2 for h in heights)/len(heights))**.5,
                bottom_plane_level_count=len(levels), bottom_plane_level_groups=levels,
                largest_shared_layer_fraction=max(map(len, levels.values()))/len(ps),
                goal_yaw_orthogonal_fraction=sum(abs(p["yaw_goal_rad"]/(math.pi/2)-round(p["yaw_goal_rad"]/(math.pi/2)))<1e-7 for p in ps)/len(ps),
                neatness_definition="orthogonal goals, full support, shared bottom planes and stack-top height spread; report tradeoffs without a combined success score",
                global_optimum_claimed=False)


def evaluate_scenario(scenario):
    spec = scenario["spec"]
    reasons = list(scenario["capabilities"]["unsupported_reasons"])
    reasons += [b["id"]+": unsupported shape "+str(b.get("shape", {}).get("kind"))
                for b in spec["boxes"] if b.get("shape", {}).get("kind") != "rigid_cuboid"]
    common = dict(schema="depallet.scenario_evaluation.v1", scenario_sha256=_digest(scenario),
                  physics_executed=False, inference_executed=False, full_task_validated=False,
                  valid_sim_hours=0., required_reobserve_after_each_transfer=True)
    if reasons:
        return dict(common, status="UNSUPPORTED", passed=False, errors=reasons, rule_plan=None)
    try:
        source_validation = validate_spec(spec)
        # Existing heuristic rejects offset COM. It sees a centered geometry-only
        # view; actual mass/COM is restored before independent acceptance.
        geometry = copy.deepcopy(spec)
        for box in geometry["boxes"]:
            box["physical"]["center_of_mass_local_m"] = [0., 0., 0.]
            box["physical"]["inertia_model"] = "uniform_solid_cuboid_proxy_about_center"
        order = deterministic_box_order(geometry)
        packing = plan_goal_packing(geometry, order, inter_stack_gap_m=.05, max_cargo_height_m=1.2)
        _center_packing(packing)
        actual = {b["id"]: b for b in spec["boxes"]}
        for p in packing["placements"]:
            p["center_of_mass_local_m"] = list(actual[p["box_id"]]["physical"]["center_of_mass_local_m"])
        packing.pop("validation", None)
        packing["validation"] = validate_goal_layout(packing, spec)
        rules = dict(schema="depallet.scenario_rule_plan.v1", order=order, packing=packing,
                     order_validation=validate_box_order(geometry, order),
                     provider="deterministic_rule_baseline", model_proposal=None,
                     inferred_by_vlm=False, autonomous_model_executed=False, fallback_used=False,
                     geometry_generator_used_centered_com_view=True,
                     actual_com_preserved_in_output=True, simulation_oracle_world=True,
                     execution_authorized=False, need_reobserve_after_each_transfer=True)
        ok = source_validation["passed"] and packing["validation"]["passed"]
        return dict(common, status="CPU_READY" if ok else "BLOCKED", passed=ok,
                    source_validation=source_validation, rule_plan=rules,
                    metrics=packing_metrics(packing), errors=packing["validation"]["errors"])
    except (KeyError, ValueError, TypeError, IndexError) as exc:
        return dict(common, status="BLOCKED", passed=False, errors=[str(exc)], rule_plan=None)


def evaluation_contract():
    return dict(schema="depallet.scenario_success_contract.v1",
                trial_unit="one fresh complete 16-box source-to-goal episode",
                success_requires=["16 unique boxes committed by runtime evidence", "source emptied",
                                  "goal bounds and non-overlap", "actual full support and COM assumptions checked",
                                  "actual attachment, release and settle", "remaining-source gate after every transfer",
                                  "fresh observation and validated task/motion plan before next box"],
                observed_module_denominators="separate attempted episodes, boxes, model frames and planning requests",
                model_failure_policy="abstain/fail closed; named rule fallback is a separate baseline and not model success",
                neatness_metrics=["floor_area_utilization", "cargo_bounding_volume_utilization",
                                  "stack_top_height_std_m", "bottom_plane_level_count",
                                  "goal_yaw_orthogonal_fraction"],
                unsupported_shape_policy="no cuboid replacement counted as non-cuboid or labile success",
                safety_and_physics_validated=False)


def validate_measured_goal_com(spec, packing, committed_ids, box_ids, state, *, goal_pose=None):
    """Runtime COM check using only supplied actual poses and declared fixed masses.

    Call alongside the existing actual OBB support/velocity gate. Authored object
    positions never fill a missing measurement. Gravity is world -Z; tilt <=3deg.
    """
    try:
        goal_pose = list(CELL["goal_pose"] if goal_pose is None else goal_pose)
        if len(goal_pose) != 4 or not _finite(goal_pose):
            raise ValueError("Invalid goal pose")
        source = {b["id"]: b for b in spec["boxes"]}
        declared = {p["box_id"]: p for p in packing["placements"]}
        box_count = len(spec["boxes"])
        if (box_count < 1 or len(source) != box_count
                or set(source) != set(declared)
                or len(packing["placements"]) != box_count):
            raise ValueError("Complete unique scene/packing ID catalog required")
        committed, ids = list(committed_ids), list(box_ids)
        if len(set(committed)) != len(committed) or committed != packing["order"][:len(committed)]:
            raise ValueError("Committed IDs must be the exact executed order prefix")
        if len(set(ids)) != len(ids) or not set(ids) <= set(source):
            raise ValueError("Invalid measured ID catalog")
        if not set(committed) <= set(ids):
            raise ValueError("Missing committed object measurement")
        ps, qs = state["positions_m"], state["quaternions_wxyz"]
        if len(ps) != len(ids) or len(qs) != len(ids):
            raise ValueError("Measured state length mismatch")
        actual, children = {}, {name:[name] for name in committed}
        for i, name in enumerate(ids):
            pos, q = ps[i], qs[i]
            if len(pos) != 3 or len(q) != 4 or not _finite([*pos, *q]):
                raise ValueError("Invalid measured pose: "+name)
            norm = math.sqrt(sum(x*x for x in q))
            if abs(norm-1.) > .001:
                raise ValueError("Measured quaternion is not near unit: "+name)
            w,x,y,z = [v/norm for v in q]
            r = [[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                 [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                 [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]]
            if name not in children:
                continue
            dims, physical = source[name]["dimensions_m"], source[name]["physical"]
            com, mass = physical["center_of_mass_local_m"], physical["mass_kg"]
            if len(dims) != 3 or len(com) != 3 or not _finite([*dims, *com, mass]) or min(dims) <= 0 or mass <= 0 or any(abs(com[k])>=dims[k]/2 for k in range(3)):
                raise ValueError("Invalid declared COM/mass/dimensions: "+name)
            if r[2][2] < math.cos(math.radians(3.)):
                raise ValueError("Measured tilt exceeds 3 degrees: "+name)
            actual[name] = dict(position=list(pos), rotation=r, mass=mass,
                                com=[pos[k]+sum(r[k][j]*com[j] for j in range(3)) for k in range(3)])
        for name in reversed(committed):
            support = declared[name]["support_id"]
            if support != "goal_pallet":
                if support not in children or committed.index(support) >= committed.index(name):
                    raise ValueError("Missing/uncommitted support: "+name)
                children[support].extend(children[name])
        checks = []
        for name in committed:
            item = actual[name]
            mass = sum(actual[k]["mass"] for k in children[name])
            center = [sum(actual[k]["mass"]*actual[k]["com"][i] for k in children[name])/mass for i in range(3)]
            relative = [center[i]-item["position"][i] for i in range(3)]
            r = item["rotation"]
            local = [sum(r[j][i]*relative[j] for j in range(3)) for i in range(3)]
            gravity = [-r[2][i] for i in range(3)]
            dims = source[name]["dimensions_m"]
            distance = (-dims[2]/2-local[2])/gravity[2]
            if distance < 0:
                raise ValueError("Resultant COM lies below support plane: "+name)
            hit = [local[i]+distance*gravity[i] for i in range(2)]
            margin = min(dims[i]/2-abs(hit[i]) for i in range(2))
            checks.append(dict(box_id=name, supported_mass_kg=mass, bottom_plane_resultant_local_xy_m=hit,
                               minimum_support_margin_m=margin, passed=margin >= .005))
        pallet = None
        if committed:
            mass = sum(actual[k]["mass"] for k in committed)
            center = [sum(actual[k]["mass"]*actual[k]["com"][i] for k in committed)/mass for i in range(2)]
            dx,dy = center[0]-goal_pose[0], center[1]-goal_pose[1]
            c,s = math.cos(goal_pose[3]),math.sin(goal_pose[3])
            local = [c*dx+s*dy,-s*dx+c*dy]
            margin = min(spec["pallet"]["dimensions_m"][i]/2-abs(local[i]) for i in range(2))
            pallet = dict(resultant_xy_goal_m=local, minimum_support_margin_m=margin, passed=margin >= .005)
        return dict(passed=all(c["passed"] for c in checks) and (pallet is None or pallet["passed"]),
                    per_box=checks, pallet=pallet, measured_pose_count=len(committed),
                    pose_source="caller_supplied_actual_physics_state",
                    authored_object_positions_used=False, assumed_fixed_com_and_mass=True,
                    companion_actual_support_velocity_gate_required=True,
                    dynamics_or_crush_capacity_verified=False)
    except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError) as exc:
        return dict(passed=False, fail_closed=True, errors=[str(exc)],
                    authored_object_positions_used=False, companion_actual_support_velocity_gate_required=True)


def perception_catalog(scenario):
    """Declared IDs/classes and broad bounds; V3 exact dimensions/physics withheld."""
    unknown = scenario["level"] == "V3"
    objects = []
    for b in scenario["spec"]["boxes"]:
        row = dict(id=b["id"], shape_family=b["shape"]["kind"],
                   dimension_source="unknown_must_estimate" if unknown else "declared_catalog_assumption")
        if not unknown:
            row["dimensions_m"] = list(b["dimensions_m"])
        objects.append(row)
    return dict(schema="depallet.scenario_perception_catalog.v1", objects=objects,
                exact_v3_dimensions_withheld=unknown, masses_and_com_withheld=True,
                v3_dimension_bounds_m=[[.35,.42],[.295,.32],[.17,.23]] if unknown else None,
                source="generator_metadata; not a detector or pose estimate", inference_executed=False)
