"""Measured-snapshot watchdog for an elastically attached box departing support.

This module runs no physics, moves no objects and assumes no fixed TCP-to-box
transform. A 4 mm / 0.01 rad attachment-compliance limit is NOT a contact
penetration allowance or the nominal fixed-attachment continuous-path proof.
All geometry below uses actual supplied box poses at each observed timestamp.
"""
from __future__ import annotations

import copy
import itertools
import math
import numpy as np


GEOMETRY_EPS_M = 1e-9
MAX_CONTACT_ALLOWANCE_M = .0001
SIGNS = np.array(list(itertools.product((-1., 1.), repeat=3)))


def _array(value, shape, label):
    raw = np.asarray(value)
    if raw.shape != shape or raw.dtype.kind not in 'iuf' or not np.isfinite(raw).all():
        raise ValueError(f'{label}: finite numeric array of shape {shape} required')
    return raw.astype(float, copy=True)


def _ids(value, label):
    if not isinstance(value, (list, tuple)) or not value or len(value) > 128:
        raise ValueError(f'{label}: 1..128 IDs required')
    if any(not isinstance(v, str) or not v.strip() for v in value) or len(set(value)) != len(value):
        raise ValueError(f'{label}: unique nonempty string IDs required')
    return list(value)


def _rotation(q):
    q = _array(q, (4,), 'quaternion_wxyz')
    norm = float(np.linalg.norm(q))
    if abs(norm-1.) > 1e-4:
        raise ValueError('Quaternion must be unit length; order is wxyz')
    w, x, y, z = q/norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _transform(value):
    t = _array(value, (4, 4), 'T_world_object')
    r = t[:3, :3]
    if (not np.allclose(t[3], [0., 0., 0., 1.], atol=1e-8, rtol=0.) or
            not np.allclose(r.T@r, np.eye(3), atol=1e-7, rtol=0.) or abs(np.linalg.det(r)-1.) > 1e-7):
        raise ValueError('T_world_object must be a column-vector rigid transform, without scale/reflection')
    return t


def _obb(position, rotation, dimensions):
    half = dimensions/2
    return dict(center=position, rotation=rotation, half=half,
                vertices=(SIGNS*half)@rotation.T+position)


def obb_separation(a, b):
    """15-axis SAT with normalized axes; positive gap proves OBB disjointness.

    A negative best gap is a signed separating-plane overlap bound, not an
    exact Euclidean penetration-depth or flexible-mesh measurement.
    """
    ra, rb = a['rotation'], b['rotation']
    axes = [ra[:, i] for i in range(3)]+[rb[:, i] for i in range(3)]
    axes += [np.cross(ra[:, i], rb[:, j]) for i in range(3) for j in range(3)]
    best, normal, count = -math.inf, None, 0
    for axis in axes:
        length = float(np.linalg.norm(axis))
        if length < 1e-9:
            continue
        axis = axis/length
        gap = (abs(float((a['center']-b['center'])@axis))-
               float(a['half']@np.abs(ra.T@axis))-float(b['half']@np.abs(rb.T@axis)))
        if gap > best:
            best, normal = gap, axis
        count += 1
    return float(best), normal.copy(), count


class ActualContactEscapeWatchdog:
    """Fail-latched actual-OBB check at measured snapshots, never continuous proof.

    Constructor:
      ``(spec, box_ids, initial_state, target_id=..., support_id=...,
         static_obstacles={id: {dimensions_m, T_world_object}},
         measurement_source='isaac_runtime', support_geometry_policy='fixed_plane')``

    ``spec['boxes']`` supplies IDs, fixed nominal dimensions and support IDs.
    ``state`` supplies ``sim_time``, ``physics_step``, positions_m (N,3) and
    quaternions_wxyz (N,4), in the constructor's exact box_ids order. Optional
    state.box_ids must match. Every declared dynamic box remains mandatory.
    Static obstacles are explicit world-frame rigid cuboid envelopes; include
    the complete relevant fixed world. An empty catalog is reported as empty,
    never as world completeness. No robot links are inferred or ignored here.

    Call observe(state) once per NEW measured state during departure. After any
    failure it remains failed. Optional endpoint=True additionally requires
    strictly positive separation from support. Default fixed_plane preserves the
    original conservative projection-plane gates. Opt-in obb_sat gates actual
    15-axis OBB gap instead, retaining plane values as diagnostics; its endpoint
    also requires measured target center world-Z rise of at least 0.05 m.
    SAT gaps are signed separating-axis bounds, not flexible penetration depths.
    result() returns the last report.
    finish() validates the last measured endpoint without inventing another
    timestamp; call it after the controller confirms ESCAPED.
    Sampling can miss inter-sample collisions. Keep actual attachment, robot
    contact, speed, source-integrity and endpoint replanning gates separately.
    """
    def __init__(self, spec, box_ids, initial_state, *, target_id, support_id,
                 static_obstacles, measurement_source='caller_supplied_rigid_body_states',
                 initial_overlap_allowance_m=.0001, jitter_allowance_m=.0001,
                 maximum_sample_interval_s=.05, support_geometry_policy='fixed_plane'):
        if not isinstance(support_geometry_policy, str) or support_geometry_policy not in ('fixed_plane', 'obb_sat'):
            raise ValueError('support_geometry_policy must be fixed_plane or obb_sat')
        self.support_geometry_policy = support_geometry_policy
        self.ids = _ids(box_ids, 'measured box IDs')
        declared = _ids([b['id'] for b in spec['boxes']], 'spec box IDs')
        if set(self.ids) != set(declared):
            raise ValueError('Every declared dynamic box must have exactly one measured pose')
        self.dimensions = {}
        for box in spec['boxes']:
            dims = _array(box['dimensions_m'], (3,), 'box dimensions')
            if np.any(dims <= 0):
                raise ValueError('Box dimensions must be positive')
            self.dimensions[box['id']] = dims
        if target_id not in self.ids or target_id == support_id:
            raise ValueError('Distinct declared target and support required')
        target_spec = next(b for b in spec['boxes'] if b['id'] == target_id)
        if target_spec.get('support_id') != support_id:
            raise ValueError('Support must match the target declared support ID')
        if not isinstance(static_obstacles, dict) or len(static_obstacles) > 128:
            raise ValueError('Explicit static obstacle catalog required (maximum 128 cuboids)')
        self.static = {}
        for name, data in static_obstacles.items():
            if not isinstance(name, str) or not name.strip() or name in self.ids:
                raise ValueError('Static IDs must be unique and distinct from dynamic box IDs')
            dims = _array(data['dimensions_m'], (3,), 'static dimensions')
            if np.any(dims <= 0):
                raise ValueError('Static dimensions must be positive')
            t = _transform(data['T_world_object'])
            self.static[name] = _obb(t[:3, 3], t[:3, :3], dims)
        if support_id not in self.ids and support_id not in self.static:
            raise ValueError('Declared support is absent from measured/static world')
        for name, value in [('initial_overlap_allowance_m', initial_overlap_allowance_m),
                            ('jitter_allowance_m', jitter_allowance_m)]:
            if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                    not math.isfinite(value) or not 0 <= value <= MAX_CONTACT_ALLOWANCE_M):
                raise ValueError(f'{name} must be within 0..0.0001 m')
        if (not isinstance(maximum_sample_interval_s, (int, float)) or isinstance(maximum_sample_interval_s, bool) or
                not math.isfinite(maximum_sample_interval_s) or not 0 < maximum_sample_interval_s <= .05):
            raise ValueError('Maximum sample interval must be within (0, 0.05] seconds')
        if not isinstance(measurement_source, str) or not measurement_source:
            raise ValueError('Explicit nonempty measurement provenance required')
        self.target_id, self.support_id = target_id, support_id
        self.source = measurement_source
        self.allowance, self.jitter = float(initial_overlap_allowance_m), float(jitter_allowance_m)
        self.max_dt = float(maximum_sample_interval_s)
        self.failed, self.sample_count, self.finalized = False, 0, False
        self.maximum_fixed_gap = self.maximum_relative_gap = self.maximum_sat_gap = -math.inf
        self.minimum_observed_fixed_gap = self.minimum_observed_relative_gap = self.minimum_observed_sat_gap = math.inf
        now, step, world = self._state(initial_state)
        target, support = world[target_id], world[support_id]
        initial_sat, normal, _ = obb_separation(target, support)
        self.initial_sat_gap = initial_sat
        if float((target['center']-support['center'])@normal) < 0:
            normal = -normal
        if normal[2] < .98 or float((target['center']-support['center'])@normal) <= 0:
            raise ValueError('Initial support must be below target with a nearly vertical separating plane')
        self.normal = normal
        self.plane_offset = float((support['vertices']@normal).max())
        self.initial_gap = float((target['vertices']@normal).min()-self.plane_offset)
        initial_gate_gap = initial_sat if self.support_geometry_policy == 'obb_sat' else self.initial_gap
        if initial_sat > .005 or initial_gate_gap > .005 or initial_gate_gap < -self.allowance-GEOMETRY_EPS_M:
            raise ValueError('Initial support contact must be within 5 mm and the explicit <=0.1 mm overlap cap')
        self.last_time, self.last_step = now, step
        self.initial_target = target['center'].copy()
        self.initial_support = support['center'].copy()
        report = self._evaluate(now, step, world, endpoint=False, initial=True)
        if not report['passed']:
            raise ValueError('Initial actual contact escape validation failed: '+', '.join(report['failure_reasons']))

    def _state(self, state):
        if not isinstance(state, dict):
            raise ValueError('Measured state must be a mapping')
        if 'box_ids' in state and list(state['box_ids']) != self.ids:
            raise ValueError('Measured box ordering changed')
        now, step = state.get('sim_time'), state.get('physics_step')
        if (not isinstance(now, (float, int)) or isinstance(now, bool) or not math.isfinite(now) or now < 0 or
                not isinstance(step, int) or isinstance(step, bool) or step < 0):
            raise ValueError('Valid measured simulation time and integer physics step required')
        positions = _array(state.get('positions_m'), (len(self.ids), 3), 'positions_m')
        quaternions = _array(state.get('quaternions_wxyz'), (len(self.ids), 4), 'quaternions_wxyz')
        world = dict(self.static)
        for i, name in enumerate(self.ids):
            world[name] = _obb(positions[i], _rotation(quaternions[i]), self.dimensions[name])
        return float(now), step, world

    def _evaluate(self, now, step, world, *, endpoint, initial=False):
        target, support = world[self.target_id], world[self.support_id]
        projections = target['vertices']@self.normal
        fixed_gaps = projections-self.plane_offset
        relative_gaps = projections-float((support['vertices']@self.normal).max())
        fixed, relative = float(fixed_gaps.min()), float(relative_gaps.min())
        reasons = []
        if self.support_geometry_policy == 'fixed_plane':
            for label, gap, peak in [('initial_fixed_plane', fixed, self.maximum_fixed_gap),
                                      ('actual_support_plane', relative, self.maximum_relative_gap)]:
                if gap < -self.allowance-GEOMETRY_EPS_M:
                    reasons.append(label+'_penetration_exceeds_absolute_cap')
                if gap < self.initial_gap-self.jitter-GEOMETRY_EPS_M:
                    reasons.append(label+'_deeper_than_initial_gap_plus_jitter')
                if not initial and gap < peak-self.jitter-GEOMETRY_EPS_M:
                    reasons.append(label+'_reverses_more_than_jitter_from_observed_maximum')
        collisions, closest = [], None
        support_sat, support_axes = None, None
        for name, obstacle in world.items():
            if name == self.target_id:
                continue
            gap, _, axes = obb_separation(target, obstacle)
            item = dict(obstacle_id=name, signed_sat_gap_m=gap, axes_checked=axes,
                        pose_source='explicit_static_world_envelope' if name in self.static else self.source)
            if name == self.support_id:
                support_sat, support_axes = gap, axes
                if self.support_geometry_policy == 'obb_sat':
                    if gap < self.initial_sat_gap-self.jitter-GEOMETRY_EPS_M:
                        reasons.append('actual_support_OBB_deeper_than_initial_SAT_gap_plus_jitter')
                    if not initial and gap < self.maximum_sat_gap-self.jitter-GEOMETRY_EPS_M:
                        reasons.append('actual_support_OBB_reverses_more_than_jitter_from_observed_SAT_maximum')
                if gap < -self.allowance-GEOMETRY_EPS_M:
                    reasons.append('actual_support_OBB_overlap_exceeds_absolute_cap')
            else:
                if closest is None or gap < closest['signed_sat_gap_m']:
                    closest = item
                if gap <= GEOMETRY_EPS_M:
                    collisions.append(item)
                    reasons.append('non_support_OBBs_not_strictly_disjoint:'+name)
        vertical_displacement = float(target['center'][2]-self.initial_target[2])
        if endpoint:
            reasons.extend(self._endpoint_reasons(fixed, relative, support_sat, vertical_displacement))
        self.sample_count += 1
        self.minimum_observed_fixed_gap = min(self.minimum_observed_fixed_gap, fixed)
        self.minimum_observed_relative_gap = min(self.minimum_observed_relative_gap, relative)
        self.minimum_observed_sat_gap = min(self.minimum_observed_sat_gap, support_sat)
        self.failed = self.failed or bool(reasons)
        self.last_report = dict(schema=('depallet.actual_contact_escape.v2' if self.support_geometry_policy == 'obb_sat'
                                        else 'depallet.actual_contact_escape.v1'), passed=not self.failed,
            support_geometry_policy=self.support_geometry_policy,
            projection_plane_gates_enabled=self.support_geometry_policy == 'fixed_plane',
            projection_plane_diagnostics_only=self.support_geometry_policy == 'obb_sat',
            support_gap_gate_metric=('15_axis_OBB_SAT_best_signed_gap' if self.support_geometry_policy == 'obb_sat'
                                     else 'fixed_and_moving_support_projection_planes_plus_OBB_absolute_cap'),
            signed_SAT_gap_is_exact_flexible_penetration_depth=False,
            failure_latched=self.failed, failure_reasons=reasons, measurement_source=self.source,
            monitoring_finalized=False,
            sample_count=self.sample_count, state_sim_time_s=now, physics_step=step,
            sample_dt_s=0. if initial else now-self.last_time,
            target_id=self.target_id, support_id=self.support_id, endpoint_check=bool(endpoint),
            initial_fixed_plane_normal_world=self.normal.tolist(), initial_fixed_plane_offset_m=self.plane_offset,
            initial_signed_support_gap_m=self.initial_gap,
            initial_penetration_m=max(0., -self.initial_gap),
            fixed_plane_signed_gap_m=fixed, actual_support_plane_signed_gap_m=relative,
            actual_support_sat_gap_m=support_sat, initial_support_sat_gap_m=self.initial_sat_gap,
            actual_support_sat_axes_checked=support_axes,
            previous_maximum_sat_gap_m=None if initial else self.maximum_sat_gap,
            minimum_observed_sat_gap_m=self.minimum_observed_sat_gap,
            measured_target_vertical_displacement_m=vertical_displacement,
            all_eight_target_corner_fixed_plane_gaps_m=fixed_gaps.tolist(),
            bottom_four_target_corner_fixed_plane_gaps_m=fixed_gaps[SIGNS[:, 2] < 0].tolist(),
            all_eight_target_corner_actual_support_plane_gaps_m=relative_gaps.tolist(),
            previous_maximum_fixed_gap_m=None if initial else self.maximum_fixed_gap,
            previous_maximum_relative_gap_m=None if initial else self.maximum_relative_gap,
            minimum_observed_fixed_gap_m=self.minimum_observed_fixed_gap,
            minimum_observed_relative_gap_m=self.minimum_observed_relative_gap,
            target_displacement_world_m=(target['center']-self.initial_target).tolist(),
            support_displacement_world_m=(support['center']-self.initial_support).tolist(),
            non_support_collisions=collisions, closest_non_support_pair=closest,
            checked_dynamic_box_ids=list(self.ids), checked_static_obstacle_ids=list(self.static),
            checked_target_obstacle_pair_count=len(world)-1,
            checked_non_support_target_obstacle_pair_count=len(world)-2,
            world_catalog_completeness_verified=False,
            limits=dict(absolute_support_overlap_m=self.allowance,
                maximum_gap_reversal_from_observed_peak_m=self.jitter, maximum_sample_interval_s=self.max_dt,
                numerical_geometry_epsilon_m=GEOMETRY_EPS_M,
                endpoint_minimum_target_vertical_displacement_m=.05 if self.support_geometry_policy == 'obb_sat' else None),
            strict_support_nonpenetration_observed=(self.minimum_observed_sat_gap >= 0. if self.support_geometry_policy == 'obb_sat'
                else min(self.minimum_observed_fixed_gap, self.minimum_observed_relative_gap) >= 0.),
            continuous_collision_guarantee=False, fixed_attachment_assumed=False,
            scope='actual measured-snapshot target/world OBB watchdog; no continuous, robot-link, contact-force or vacuum-seal proof',
            compliance_is_not_contact_allowance=True)
        self.maximum_fixed_gap = max(self.maximum_fixed_gap, fixed)
        self.maximum_relative_gap = max(self.maximum_relative_gap, relative)
        self.maximum_sat_gap = max(self.maximum_sat_gap, support_sat)
        self.last_time, self.last_step = now, step
        return self.result()

    def _endpoint_reasons(self, fixed, relative, sat, vertical_displacement):
        reasons = []
        gap = sat if self.support_geometry_policy == 'obb_sat' else min(fixed, relative, sat)
        if gap <= GEOMETRY_EPS_M:
            reasons.append('endpoint_support_not_strictly_separated')
        if self.support_geometry_policy == 'obb_sat' and vertical_displacement < .05-GEOMETRY_EPS_M:
            reasons.append('endpoint_target_vertical_displacement_below_0p05m')
        return reasons

    def observe(self, state, *, endpoint=False):
        if self.failed:
            return self.result()
        try:
            if self.finalized:
                raise ValueError('Watchdog already finalized; no further observations are accepted')
            if type(endpoint) is not bool:
                raise ValueError('endpoint must be boolean')
            now, step, world = self._state(state)
            if now <= self.last_time or now-self.last_time > self.max_dt+1e-12 or step <= self.last_step:
                raise ValueError('Measured physics time/step must increase within the sample interval limit')
            return self._evaluate(now, step, world, endpoint=endpoint)
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            self.failed = True
            self.last_report = {**self.last_report, 'passed': False, 'failure_latched': True,
                                'failure_reasons': ['invalid_measured_snapshot: '+str(exc)]}
            return self.result()

    def finish(self):
        """Check the most recent actual snapshot; do not advance its clock/count.

        The caller must first observe the physics snapshot where its controller
        reaches ESCAPED. This is a measured support-separation check, not a
        declaration that the planned trajectory or a transfer has succeeded.
        """
        if self.failed or self.finalized:
            return self.result()
        report = self.last_report
        reasons = []
        if self.sample_count < 2:
            reasons.append('endpoint_requires_a_new_measured_snapshot_after_initialization')
        reasons.extend(self._endpoint_reasons(report['fixed_plane_signed_gap_m'],
            report['actual_support_plane_signed_gap_m'], report['actual_support_sat_gap_m'],
            report['measured_target_vertical_displacement_m']))
        self.failed, self.finalized = bool(reasons), True
        self.last_report = {**report, 'passed': not self.failed, 'failure_latched': self.failed,
            'failure_reasons': reasons, 'endpoint_check': True, 'monitoring_finalized': True,
            'endpoint_used_existing_measured_snapshot': True}
        return self.result()

    def result(self):
        return copy.deepcopy(self.last_report)
