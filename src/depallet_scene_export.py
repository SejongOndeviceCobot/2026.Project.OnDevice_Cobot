"""CPU-only aggregation for real scene telemetry and oracle mask provenance."""
from __future__ import annotations
import numpy as np


def object_masks(segmentation, id_to_labels, object_paths):
    """Merge real annotator mesh IDs under exact object-root path boundaries."""
    values = np.asarray(segmentation)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2 or values.dtype.kind not in 'iu':
        raise ValueError('instance ID image must be a 2D integer array')
    if not isinstance(id_to_labels, dict):
        raise ValueError('missing instance idToLabels mapping')
    labels = {int(key): str(value) for key, value in id_to_labels.items()}
    masks = {}
    for object_id, prefix in object_paths.items():
        ids = [key for key, value in labels.items() if value == prefix or value.startswith(prefix+'/')]
        masks[object_id] = np.isin(values, ids)
    return masks


def summarize_settle(samples, authored_positions, authored_quaternions):
    """Summarize a measured two-second window; never manufacture stable samples."""
    if len(samples) < 2:
        return {'passed': False, 'reason': 'insufficient physical samples', 'sample_count': len(samples)}
    times = np.asarray([item['sim_time'] for item in samples], float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('physical sample times must strictly increase')
    positions = np.asarray([item['positions_m'] for item in samples], float)
    quaternions = np.asarray([item['quaternions_wxyz'] for item in samples], float)
    linear = np.asarray([item['linear_velocities_m_s'] for item in samples], float)
    angular = np.asarray([item['angular_velocities_rad_s'] for item in samples], float)
    for array in [positions, quaternions, linear, angular]:
        if not np.isfinite(array).all():
            raise ValueError('nonfinite PhysX state')
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms < .5):
        raise ValueError('invalid measured quaternion')
    quaternions = quaternions / norms
    qref = np.asarray(authored_quaternions, float)
    qref /= np.linalg.norm(qref, axis=-1, keepdims=True)
    position_span = np.linalg.norm(positions-positions[0], axis=-1).max(axis=0)
    angle_span = (2*np.arccos(np.clip(np.abs(np.sum(quaternions*quaternions[0], axis=-1)), 0, 1))).max(axis=0)
    from_authored = np.linalg.norm(positions[-1]-np.asarray(authored_positions), axis=-1)
    angle_authored = 2*np.arccos(np.clip(np.abs(np.sum(quaternions[-1]*qref, axis=-1)), 0, 1))
    linear_peak = np.linalg.norm(linear, axis=-1).max(axis=0)
    angular_peak = np.linalg.norm(angular, axis=-1).max(axis=0)
    enough_time = times[-1]-times[0] >= 1.95
    per_object = (position_span <= .005) & (angle_span <= np.deg2rad(1)) & (linear_peak <= .02) & (angular_peak <= .05)
    layout_retained = (from_authored <= .05) & (angle_authored <= np.deg2rad(5))
    return {
        'passed': bool(enough_time and np.all(per_object) and np.all(layout_retained)),
        'window_start_sim_time': float(times[0]), 'window_end_sim_time': float(times[-1]),
        'window_seconds': float(times[-1]-times[0]), 'sample_count': len(samples),
        'maximum_translation_in_window_m': position_span.tolist(),
        'maximum_rotation_in_window_rad': angle_span.tolist(),
        'maximum_linear_velocity_m_s': linear_peak.tolist(),
        'maximum_angular_velocity_rad_s': angular_peak.tolist(),
        'final_translation_from_authored_m': from_authored.tolist(),
        'final_rotation_from_authored_rad': angle_authored.tolist(),
        'per_object_settled': per_object.tolist(), 'per_object_layout_retained': layout_retained.tolist(),
        'limits': {'window_seconds_min': 1.95, 'translation_m': .005,
                   'rotation_rad': float(np.deg2rad(1)), 'linear_velocity_m_s': .02,
                   'angular_velocity_rad_s': .05, 'authored_translation_m': .05,
                   'authored_rotation_rad': float(np.deg2rad(5))},
        'scope': 'measured rigid-body rest and layout drift; not cardboard or suction calibration',
    }


def initial_settle_decision(report, *, elapsed_s, home_error_rad, robot_scene_headers=0, unexpected_contacts=None, timeout_s=8.):
    """Wait for a complete resting window; real contact/layout/home faults abort.

    Initial transient speed peaks may remain in the rolling two-second window
    after boxes are already resting. They require further measured samples, not
    relaxed thresholds or a fabricated stable history.
    """
    if not np.isfinite([elapsed_s, home_error_rad, timeout_s]).all() or elapsed_s < 0 or not 2 <= timeout_s <= 8:
        raise ValueError('Invalid bounded initialization clock or measured home error')
    failures = []
    layout = report.get('per_object_layout_retained')
    layout_ok = None if layout is None else bool(all(layout))
    if layout_ok is False: failures.append('layout_drift')
    if robot_scene_headers: failures.append('robot_scene_contact')
    if unexpected_contacts: failures.append('unexpected_contact')
    if home_error_rad > .01: failures.append('home_joint_error')
    settled = report.get('passed') is True
    if not settled and elapsed_s >= timeout_s: failures.append('settle_timeout')
    decision = 'ABORT' if failures else ('READY' if settled else 'WAIT')
    return dict(decision=decision, passed=decision == 'READY', failed_conditions=failures,
                waiting_for=None if settled or failures else 'complete measured resting window',
                elapsed_sim_seconds=float(elapsed_s), maximum_wait_sim_seconds=float(timeout_s),
                measured_home_error_rad=float(home_error_rad), maximum_home_error_rad=.01,
                robot_scene_contact_headers=int(robot_scene_headers), unexpected_contact_count=len(unexpected_contacts or {}),
                layout_retained=layout_ok, measured_rest_window_passed=settled,
                rest_thresholds_unchanged=True)
