"""Measured-pose geometric gate for the nominal 16-pad VGP20 cuboid proxy.

Column-vector T_world_* matrices, SI metres, and box local +Z as its top face.
This verifies whole circular pad projections and plane distances; it models no
vacuum pressure, sealing, leakage, cardboard deformation, or holding force.
"""
from __future__ import annotations

import math
import numpy as np


def _rigid_transform(value, name):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f'{name} must be a finite 4x4 column-vector transform')
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0., 0., 0., 1.], atol=1e-8, rtol=0.) or
            not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-5, rtol=0.) or
            abs(np.linalg.det(rotation)-1.) > 1e-5):
        raise ValueError(f'{name} must contain a proper rigid rotation, no scale/reflection')
    return matrix


def validate_vacuum_patch(T_world_flange, assembly, T_world_box, box_dimensions_m):
    """Return a JSON-ready pre-close gate using actual current body transforms.

    Required assembly keys: pads_flange_m (16x3), pad_radius_m, and
    suction_direction_flange (+Z). Every circular disk, including its rim, must
    lie fully inside the projected box top, with signed top-plane distances in
    [-1 mm, +4 mm]. Suction direction must oppose box top normal within 3 deg.
    Positive gap means the pad is above the box. Geometric failures return
    passed=False; malformed/unsupported input raises ValueError (fail closed).
    """
    flange = _rigid_transform(T_world_flange, 'T_world_flange')
    box = _rigid_transform(T_world_box, 'T_world_box')
    dims = np.asarray(box_dimensions_m, dtype=float)
    pads = np.asarray(assembly.get('pads_flange_m'), dtype=float)
    radius = assembly.get('pad_radius_m')
    direction = np.asarray(assembly.get('suction_direction_flange'), dtype=float)
    if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0):
        raise ValueError('Box dimensions must be three finite positive metre values')
    if pads.shape != (16, 3) or not np.isfinite(pads).all() or len(np.unique(pads, axis=0)) != 16:
        raise ValueError('Exactly 16 distinct finite pad centers are required')
    if not np.allclose(pads[:, 2], pads[0, 2], atol=1e-8, rtol=0.):
        raise ValueError('This gate requires one coplanar nominal pad array')
    if (not isinstance(radius, (float, int)) or isinstance(radius, bool) or
            not math.isfinite(radius) or radius <= 0):
        raise ValueError('Finite positive nominal pad radius is required')
    if direction.shape != (3,) or not np.allclose(direction, [0., 0., 1.], atol=1e-8, rtol=0.):
        raise ValueError('Reviewed pad disks require the flange +Z suction direction')
    relative = np.linalg.inv(box)@flange
    centers = pads@relative[:3, :3].T+relative[:3, 3]
    # Exact extrema for projections of each full circular disk. This checks the
    # full footprint and full rim height without sampling or center-only tests.
    disk_extent = radius*np.linalg.norm(relative[:3, :2], axis=1)
    suction_world = flange[:3, :3]@direction
    top_normal_world = box[:3, 2]
    cosine = -float(suction_world@top_normal_world)/(np.linalg.norm(suction_world)*np.linalg.norm(top_normal_world))
    angle = math.acos(float(np.clip(cosine, -1., 1.)))
    angle_ok = angle <= math.radians(3.)+1e-10
    per_pad, reasons = [], []
    if not angle_ok:
        reasons.append('suction_normal_angle_exceeds_3_degrees')
    for index, center in enumerate(centers):
        xy_clearance = dims[:2]/2-np.abs(center[:2])-disk_extent[:2]
        center_gap = float(center[2]-dims[2]/2)
        low_gap, high_gap = center_gap-disk_extent[2], center_gap+disk_extent[2]
        inside = bool(np.min(xy_clearance) >= -1e-9)
        gap_ok = bool(low_gap >= -.001-1e-9 and high_gap <= .004+1e-9)
        if not inside:
            reasons.append(f'pad_{index:02d}_footprint_outside_box_top')
        if not gap_ok:
            reasons.append(f'pad_{index:02d}_surface_gap_outside_minus1_to_plus4_mm')
        per_pad.append(dict(pad_index=index, center_box_m=center.tolist(),
            projected_disk_half_extents_box_m=disk_extent.tolist(),
            xy_edge_clearance_m=xy_clearance.tolist(), center_surface_gap_m=center_gap,
            minimum_surface_gap_m=float(low_gap), maximum_surface_gap_m=float(high_gap),
            full_footprint_inside=inside, full_disk_surface_gap_passed=gap_ok,
            passed=inside and gap_ok and angle_ok))
    return dict(schema='depallet.vacuum_patch_geometry.v1', passed=not reasons,
        failure_reasons=reasons, pad_count=16, pad_radius_m=float(radius),
        suction_normal_angle_rad=angle, suction_normal_angle_deg=math.degrees(angle),
        maximum_normal_angle_deg=3., accepted_surface_gap_m=[-.001, .004],
        minimum_xy_edge_clearance_m=min(min(p['xy_edge_clearance_m']) for p in per_pad),
        minimum_surface_gap_m=min(p['minimum_surface_gap_m'] for p in per_pad),
        maximum_surface_gap_m=max(p['maximum_surface_gap_m'] for p in per_pad),
        per_pad=per_pad, scope='whole nominal circular pads against measured-pose box local +Z cuboid face',
        real_vacuum_verified=False, holding_force_verified=False, actual_attachment_verified=False,
        requires_actual_surface_attachment_confirmation=True,
        assumptions=['nominal unmeasured pad geometry', 'USD rigid cuboid top face',
                     'no cardboard deformation, sealing, leakage, or vacuum-pressure model'])
