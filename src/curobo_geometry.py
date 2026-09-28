"""CPU geometry additions omitted by cuRobo's mesh-only RobotBuilder.

Every rectangular cell is enclosed by its sphere. Their union therefore
contains the entire box, unlike a fit checked only at sampled points.
"""
from __future__ import annotations

import itertools
import math
import xml.etree.ElementTree as ET


def box_cover(dimensions, center=(0.,0.,0.), max_cell=(.02,.02,.04), margin=0.):
    if len(dimensions) != 3 or len(center) != 3 or len(max_cell) != 3:
        raise ValueError("Box geometry must be 3D")
    if any(not math.isfinite(x) or x <= 0 for x in (*dimensions,*max_cell)):
        raise ValueError("Box dimensions and cell bounds must be positive")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("Margin must be finite and nonnegative")
    cells = [math.ceil(d/h) for d,h in zip(dimensions,max_cell)]
    widths = [d/n for d,n in zip(dimensions,cells)]
    radius = math.sqrt(sum((w/2)**2 for w in widths)) + margin
    result = []
    for index in itertools.product(*(range(n) for n in cells)):
        c = [center[i]-dimensions[i]/2+(index[i]+.5)*widths[i] for i in range(3)]
        result.append({"center": c, "radius": radius})
    return result, {"cells": cells, "cell_widths_m": widths,
                    "radius_m": radius, "margin_m": margin,
                    "box_enclosure_by_cell_diagonal": True,
                    "sphere_count": len(result)}


def tool_box_spheres(urdf):
    tree = ET.parse(urdf).getroot()
    result, records = {}, {}
    for name in ("vgp20_body", "vgp20_adapter"):
        link = next(link for link in tree.findall("link") if link.get("name") == name)
        geometries = link.findall("collision")
        if len(geometries) != 1 or geometries[0].find("geometry/box") is None:
            raise ValueError(f"Expected exactly one box collision on {name}")
        geom = geometries[0]
        dims = [float(v) for v in geom.find("geometry/box").get("size").split()]
        origin = geom.find("origin")
        center = [0.,0.,0.] if origin is None else [float(v) for v in origin.get("xyz","0 0 0").split()]
        rpy = [0.,0.,0.] if origin is None else [float(v) for v in origin.get("rpy","0 0 0").split()]
        if max(abs(x) for x in rpy) > 1e-10:
            raise ValueError("Rotated collision boxes need explicit sphere transforms")
        margin = .003 if name == "vgp20_body" else 0.
        result[name], record = box_cover(dims, center, margin=margin)
        records[name] = dict(record, dimensions_m=dims, center_m=center)
    return result, records


def augment_robot_config(config, urdf):
    """Add complete tool covers and inert payload slots to a saved robot config."""
    spheres, records = tool_box_spheres(urdf)
    kin = config.get("robot_cfg", config)["kinematics"]
    if kin["base_link"] != "base_link":
        raise ValueError("H2017 must use the actual joint_1 parent base_link")
    kin["collision_spheres"].update(spheres)
    kin["collision_link_names"] = list(dict.fromkeys(
        kin["collision_link_names"] + list(spheres) + ["attached_object"]))
    kin["extra_collision_spheres"] = dict(kin.get("extra_collision_spheres") or {})
    kin["extra_collision_spheres"]["attached_object"] = 64
    # These two fixed siblings meet at their manufactured mounting interface;
    # their sphere covers also overlap. No moving arm pairs are excluded here.
    ignores = kin.setdefault("self_collision_ignore", {})
    ignores.setdefault("vgp20_adapter", [])
    if "vgp20_body" not in ignores["vgp20_adapter"]:
        ignores["vgp20_adapter"].append("vgp20_body")
    kin["grasp_contact_link_names"] = []
    report = {"tool_covers": records, "payload_sphere_slots": 64,
              "fixed_sibling_exclusions": [["vgp20_adapter","vgp20_body"]],
              "whole_link_collision_disabled": False,
              "total_active_spheres_before_grasp": sum(len(v) for v in kin["collision_spheres"].values()),
              "physical_collision_validation_pending": True}
    return config, report
