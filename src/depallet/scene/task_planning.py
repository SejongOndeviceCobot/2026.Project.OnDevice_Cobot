"""Deterministic proposal validation and supported pallet packing.

This module is NOT a VLM, 6D pose estimator, motion planner or suction simulator.
It validates explicitly authored scene geometry and produces heuristic plans.
"""
from __future__ import annotations

import math
from collections import defaultdict

from depallet.scene.photo_scene import _overlap_xy, validate_spec

DEFAULT_TOOL = dict(nominal_footprint_m=[.184, .268], edge_clearance_m=.01,
                    assumed_tool_mass_kg=2.85, robot_payload_limit_kg=20.,
                    measured=False, suction_patch_calibrated=False,
                    source="nominal VGP20 envelope; assumed tool+adapter mass; not a calibrated suction footprint")


def grip_screen(box, tool=None):
    """Conservative geometric/payload screen; passing does not prove suction."""
    tool = dict(DEFAULT_TOOL if tool is None else tool)
    w, d = tool["nominal_footprint_m"]
    margin = tool["edge_clearance_m"]
    dims = box["dimensions_m"]
    values = [w, d, margin, tool["assumed_tool_mass_kg"], tool["robot_payload_limit_kg"]]
    if not all(math.isfinite(v) for v in values) or min(w, d) <= 0 or margin < 0 or min(values[3:]) <= 0:
        raise ValueError("Invalid tool screening parameters")
    fits = []
    for yaw, a, b in [(0., w, d), (math.pi/2, d, w)]:
        if a+2*margin <= dims[0]+1e-9 and b+2*margin <= dims[1]+1e-9:
            fits.append(yaw)
    mass_bound = max(box["physical"]["mass_range_kg"])
    available = tool["robot_payload_limit_kg"]-tool["assumed_tool_mass_kg"]
    return dict(box_id=box["id"], passed=bool(fits) and mass_bound <= available,
                footprint_fits=bool(fits), allowed_tool_yaws_box_rad=fits,
                mass_upper_bound_kg=mass_bound, remaining_payload_kg=available,
                payload_passed=mass_bound <= available, tested_footprint_m=[w, d], edge_clearance_m=margin,
                actual_suction_verified=False, structural_or_robot_reach_verified=False)


def exposed_candidates(boxes, remaining_ids):
    """Veto any higher remaining object whose projected footprint overlaps the top."""
    indexed = {box["id"]: box for box in boxes}
    remaining = set(remaining_ids)
    if not remaining <= indexed.keys():
        raise ValueError("Unknown remaining box ID")
    clear, blockers = [], {}
    for name in sorted(remaining):
        box = indexed[name]
        top = box["position_source_m"][2]+box["dimensions_m"][2]/2
        blocked = []
        for other_name in sorted(remaining-{name}):
            other = indexed[other_name]
            other_top = other["position_source_m"][2]+other["dimensions_m"][2]/2
            if other_top > top+1e-6 and _overlap_xy(box, other):
                blocked.append(other_name)
        if blocked:
            blockers[name] = blocked
        else:
            clear.append(name)
    return dict(candidate_ids=clear, blocked_by=blockers)


def validate_box_order(spec, proposed_ids, *, completed_ids=(), require_complete=True, tool=None):
    """Validate external VLM/human order; never silently repair rejected proposals."""
    validate_spec(spec)
    indexed = {box["id"]: box for box in spec["boxes"]}
    errors, accepted = [], []
    if not isinstance(proposed_ids, (list, tuple)) or not all(isinstance(i, str) for i in proposed_ids):
        return dict(passed=False, errors=[dict(reason="order must be a list of box IDs")], accepted_prefix=[])
    completed = set(completed_ids)
    if len(completed) != len(completed_ids) or not completed <= indexed.keys():
        return dict(passed=False, errors=[dict(reason="invalid completed IDs")], accepted_prefix=[])
    remaining = set(indexed)-completed
    seen = set()
    for step, name in enumerate(proposed_ids):
        if name not in indexed:
            errors.append(dict(step=step, box_id=name, reason="unknown ID"))
            break
        if name in seen or name in completed:
            errors.append(dict(step=step, box_id=name, reason="duplicate or already completed ID"))
            break
        seen.add(name)
        exposure = exposed_candidates(spec["boxes"], remaining)
        if name not in exposure["candidate_ids"]:
            errors.append(dict(step=step, box_id=name, reason="blocked top face", blockers=exposure["blocked_by"][name]))
            break
        screen = grip_screen(indexed[name], tool)
        if not screen["passed"]:
            errors.append(dict(step=step, box_id=name, reason="gripper footprint or payload veto", grip_screen=screen))
            break
        accepted.append(name)
        remaining.remove(name)
    if require_complete and remaining and not errors:
        errors.append(dict(reason="incomplete order", remaining_ids=sorted(remaining)))
    return dict(passed=not errors, errors=errors, accepted_prefix=accepted, remaining_ids=sorted(remaining),
                validation_scope="authored geometry, top exposure, nominal tool envelope and assumed payload only",
                motion_validated=False, actual_suction_verified=False, inferred_by_vlm=False)


def deterministic_box_order(spec, tool=None):
    """Highest exposed face first, lexical tie-break; explicit non-VLM fallback."""
    validate_spec(spec)
    indexed = {box["id"]: box for box in spec["boxes"]}
    remaining, order = set(indexed), []
    while remaining:
        candidates = exposed_candidates(spec["boxes"], remaining)["candidate_ids"]
        candidates = [name for name in candidates if grip_screen(indexed[name], tool)["passed"]]
        if not candidates:
            raise ValueError("No exposed, footprint-compatible, payload-compatible box")
        chosen = min(candidates, key=lambda name: (-(indexed[name]["position_source_m"][2]+indexed[name]["dimensions_m"][2]/2), name))
        order.append(chosen)
        remaining.remove(chosen)
    return order


def _rect_intersects(a, b, gap):
    return (a["x"] < b["x"]+b["w"]+gap-1e-9 and b["x"] < a["x"]+a["w"]+gap-1e-9 and
            a["y"] < b["y"]+b["d"]+gap-1e-9 and b["y"] < a["y"]+a["d"]+gap-1e-9)


def _floor_layout(stacks, pallet_dims, margin, gap, max_search_nodes):
    """Bounded extreme-point backtracking with 90-degree rotations."""
    ordered = sorted(stacks, key=lambda s: (-s["footprint_m"][0]*s["footprint_m"][1], s["stack_id"]))
    nodes = 0

    def search(index, placed):
        nonlocal nodes
        nodes += 1
        if nodes > max_search_nodes:
            return None
        if index == len(ordered):
            return list(placed)
        item = ordered[index]
        a, b = item["footprint_m"]
        xs = sorted({margin, *[round(r["x"]+r["w"]+gap, 10) for r in placed]})
        ys = sorted({margin, *[round(r["y"]+r["d"]+gap, 10) for r in placed]})
        rotations = sorted({(a, b), (b, a)}, key=lambda wh: -wh[0])
        for y in ys:
            for x in xs:
                for w, d in rotations:
                    rect = dict(stack_id=item["stack_id"], x=x, y=y, w=w, d=d)
                    if x+w > pallet_dims[0]-margin+1e-9 or y+d > pallet_dims[1]-margin+1e-9:
                        continue
                    if any(_rect_intersects(rect, other, gap) for other in placed):
                        continue
                    result = search(index+1, [*placed, rect])
                    if result is not None:
                        return result
                    if nodes >= max_search_nodes:
                        return None
        return None

    result = search(0, [])
    if result is None:
        raise ValueError(f"No packing found within {max_search_nodes} search nodes; this is not a proof of infeasibility")
    return result, nodes


def plan_goal_packing(spec, order, *, edge_margin_m=.01, inter_stack_gap_m=.015,
                      max_cargo_height_m=1.2, max_search_nodes=20000):
    """Reserve floor footprints, then stack same-footprint cartons in arrival order.

    This heuristic intentionally requires complete rectangular support and never
    bridges two cartons or places larger cartons on smaller ones. Reserved slots
    need not be filled at the beginning; that lets later large cartons keep a floor
    position without forcing source/goal support-order cycles.
    """
    validated = validate_box_order(spec, order)
    if not validated["passed"]:
        raise ValueError(f"Cannot pack invalid source order: {validated['errors']}")
    if (not all(math.isfinite(v) and v >= 0 for v in [edge_margin_m, inter_stack_gap_m])
            or not math.isfinite(max_cargo_height_m) or max_cargo_height_m <= 0
            or not isinstance(max_search_nodes, int) or not 1 <= max_search_nodes <= 100000):
        raise ValueError("Invalid packing margins/height/search limit")
    indexed = {box["id"]: box for box in spec["boxes"]}
    groups = defaultdict(list)
    for name in order:
        box = indexed[name]
        groups[tuple(sorted(round(v, 8) for v in box["dimensions_m"][:2]))].append(name)
    stacks = []
    for footprint, names in sorted(groups.items()):
        total_height = sum(indexed[name]["dimensions_m"][2] for name in names)
        count = max(1, math.ceil((total_height-1e-9)/max_cargo_height_m))
        group_stacks = [dict(stack_id=f"stack_{len(stacks)+i+1:02d}", footprint_m=list(footprint), box_ids=[], height_m=0.) for i in range(count)]
        for name in names:
            height = indexed[name]["dimensions_m"][2]
            eligible = [s for s in group_stacks if s["height_m"]+height <= max_cargo_height_m+1e-9]
            if not eligible:
                raise ValueError("Arrival-order stacking did not fit height bound; heuristic requires a revised layout")
            stack = min(eligible, key=lambda s: (s["height_m"], s["stack_id"]))
            stack["box_ids"].append(name)
            stack["height_m"] += height
        stacks.extend(group_stacks)
    pallet_dims = spec["pallet"]["dimensions_m"]
    layout, nodes = _floor_layout(stacks, pallet_dims, edge_margin_m, inter_stack_gap_m, max_search_nodes)
    rectangles = {r["stack_id"]: r for r in layout}
    placements = {}
    for stack in stacks:
        rect = rectangles[stack["stack_id"]]
        bottom, support = pallet_dims[2], "goal_pallet"
        for name in stack["box_ids"]:
            box = indexed[name]
            height = box["dimensions_m"][2]
            yaw = 0. if abs(box["dimensions_m"][0]-rect["w"]) < 1e-7 else math.pi/2
            placements[name] = dict(box_id=name, stack_id=stack["stack_id"], support_id=support,
                position_goal_m=[rect["x"]+rect["w"]/2-pallet_dims[0]/2,
                                 rect["y"]+rect["d"]/2-pallet_dims[1]/2, bottom+height/2],
                yaw_goal_rad=yaw, dimensions_m=list(box["dimensions_m"]), footprint_goal_m=[rect["w"], rect["d"]],
                top_face_center_goal_m=[rect["x"]+rect["w"]/2-pallet_dims[0]/2,
                                       rect["y"]+rect["d"]/2-pallet_dims[1]/2, bottom+height],
                mass_kg=box["physical"]["mass_kg"], mass_upper_bound_kg=max(box["physical"]["mass_range_kg"]),
                center_of_mass_local_m=list(box["physical"]["center_of_mass_local_m"]))
            bottom += height
            support = name
    plan = dict(method="deterministic reserved-floor and equal-footprint arrival-order stacking heuristic",
                global_optimum_claimed=False, inferred_by_vlm=False, physics_executed=False,
                packing_collision_geometry="full cuboids; static pallet support represented by nominal top rectangle",
                order=list(order), placements=[placements[name] for name in order], stacks=stacks,
                floor_rectangles=layout, pallet_dimensions_m=list(pallet_dims), edge_margin_m=edge_margin_m,
                inter_stack_gap_m=inter_stack_gap_m, max_cargo_height_m=max_cargo_height_m, search_nodes=nodes,
                floor_area_utilization=sum(r["w"]*r["d"] for r in layout)/(pallet_dims[0]*pallet_dims[1]),
                total_cargo_volume_m3=sum(math.prod(box["dimensions_m"]) for box in spec["boxes"]),
                actual_suction_verified=False, crush_load_capacity_verified=False, robot_reach_verified=False,
                need_reobserve_after_each_transfer=True)
    plan["validation"] = validate_packing(plan, spec)
    if not plan["validation"]["passed"]:
        raise ValueError(f"Generated packing failed independent validation: {plan['validation']['errors']}")
    return plan


def validate_packing(plan, spec=None):
    """Recheck placement order, full support, centered COM, edge margins and overlap."""
    errors, seen, boxes = [], {}, plan["placements"]
    dims, margin = plan["pallet_dimensions_m"], plan["edge_margin_m"]
    source = {box["id"]: box for box in spec["boxes"]} if spec is not None else None
    if source is not None:
        if sorted(plan["order"]) != sorted(source):
            errors.append("declared order does not contain every source box exactly once")
        if dims != spec["pallet"]["dimensions_m"]:
            errors.append("goal pallet dimensions differ from source specification")
    for step, p in enumerate(boxes):
        name = p["box_id"]
        if name in seen:
            errors.append(f"duplicate placement: {name}")
        x, y, z = p["position_goal_m"]
        w, d = p["footprint_goal_m"]
        height = p["dimensions_m"][2]
        if source is not None and (name not in source or p["dimensions_m"] != source[name]["dimensions_m"]):
            errors.append(f"carton dimensions or ID differ from source specification: {name}")
        yaw = p["yaw_goal_rad"]
        if not math.isfinite(yaw) or abs(yaw/(math.pi/2)-round(yaw/(math.pi/2))) > 1e-7:
            errors.append(f"non-orthogonal goal yaw: {name}")
        else:
            expected_w = abs(math.cos(yaw))*p["dimensions_m"][0]+abs(math.sin(yaw))*p["dimensions_m"][1]
            expected_d = abs(math.sin(yaw))*p["dimensions_m"][0]+abs(math.cos(yaw))*p["dimensions_m"][1]
            if abs(w-expected_w) > 1e-7 or abs(d-expected_d) > 1e-7:
                errors.append(f"footprint inconsistent with dimensions and yaw: {name}")
        if not all(math.isfinite(v) for v in [x, y, z, w, d, height]) or min(w, d, height) <= 0:
            errors.append(f"invalid numeric geometry: {name}")
            continue
        if abs(x)+w/2 > dims[0]/2-margin+1e-8 or abs(y)+d/2 > dims[1]/2-margin+1e-8:
            errors.append(f"pallet edge margin violation: {name}")
        if z+height/2 > dims[2]+plan["max_cargo_height_m"]+1e-8:
            errors.append(f"height bound violation: {name}")
        support = p["support_id"]
        if support == "goal_pallet":
            support_xy, support_dims, support_top = [0., 0.], dims[:2], dims[2]
        elif support in seen:
            below = seen[support]
            support_xy, support_dims = below["position_goal_m"][:2], below["footprint_goal_m"]
            support_top = below["position_goal_m"][2]+below["dimensions_m"][2]/2
        else:
            errors.append(f"support not placed before target: {name}")
            seen[name] = p
            continue
        if abs(z-height/2-support_top) > 1e-7:
            errors.append(f"vertical support gap: {name}")
        if abs(x-support_xy[0])+w/2 > support_dims[0]/2+1e-8 or abs(y-support_xy[1])+d/2 > support_dims[1]/2+1e-8:
            errors.append(f"incomplete rectangular support: {name}")
        if p["center_of_mass_local_m"] != [0., 0., 0.]:
            errors.append(f"COM assumption unsupported: {name}")
        for other in seen.values():
            ox, oy, oz = other["position_goal_m"]
            ow, od = other["footprint_goal_m"]
            oh = other["dimensions_m"][2]
            if min(z+height/2, oz+oh/2)-max(z-height/2, oz-oh/2) > 1e-8:
                a, b = dict(x=x-w/2, y=y-d/2, w=w, d=d), dict(x=ox-ow/2, y=oy-od/2, w=ow, d=od)
                required_gap = 0. if p["stack_id"] == other["stack_id"] else plan["inter_stack_gap_m"]
                if _rect_intersects(a, b, required_gap):
                    errors.append(f"box overlap or stack clearance violation: {name}/{other['box_id']}")
        seen[name] = p
    if [p["box_id"] for p in boxes] != plan["order"]:
        errors.append("placement sequence differs from declared source order")
    return dict(passed=not errors, errors=errors, placement_count=len(boxes),
                full_rectangular_support_required=True, centered_COM_proxy_checked=True,
                source_specification_crosschecked=source is not None,
                dynamics_or_crush_strength_validated=False)
