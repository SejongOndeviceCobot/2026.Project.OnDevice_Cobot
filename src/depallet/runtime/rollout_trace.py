"""Append-only native observation/command trace. No Isaac imports or accepted-hour claims.

Actuator proxies preserve backend calls and their order. Camera frames point to
exact post-physics snapshots; completed backend calls are not called measured
physical actions. Independent audit/export is required before dataset acceptance.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
import math
from pathlib import Path
import time

from depallet.observation.sensor_contract import (LEGACY_PROFILE, CAMERA_PROFILE_V2, camera_profile as get_camera_profile,
                             camera_pose as validate_camera_pose, pose_matrix)

ARM_METHODS = (
    'set_dof_position_targets', 'set_dof_velocity_targets',
    'set_dof_efforts', 'set_dof_max_efforts',
)
VACUUM_METHODS = ('open_gripper', 'close_gripper')


def _json(value):
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(',', ':'))


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _vector(value, count):
    # Backend arrays passed to setters are list-of-one-vector in this runtime.
    if hasattr(value, 'numpy'):
        value = value.numpy()
    if hasattr(value, 'tolist'):
        value = value.tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f'Expected {count} actuator values')
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError('Nonfinite or nonnumeric actuator/state value')
    return [float(x) for x in value]


class ActuatorProxy:
    """Pass through attributes; trace only the explicit actuation methods."""
    def __init__(self, backend, trace, methods):
        self._backend, self._trace, self._methods = backend, trace, frozenset(methods)

    def __getattr__(self, name):
        original = getattr(self._backend, name)
        if name not in self._methods:
            return original

        def invoke(*args, **kwargs):
            if not args:
                raise ValueError('Trace requires the positional actuator value used by this runtime')
            value = (0 if name == 'open_gripper' else 1) if name in VACUUM_METHODS else _vector(args[0], 6)
            attempt = self._trace.command_attempt(name, value)
            try:
                result = original(*args, **kwargs)
            except BaseException as error:
                self._trace.command_completed(name, value, attempt, status='raised', error=repr(error))
                raise
            self._trace.command_completed(name, value, attempt, status='returned')
            return result
        return invoke


class RolloutTrace:
    def __init__(self, run, *, clock, camera_ids=None, physics_hz=240,
                 joint_names=tuple(f'joint_{i}' for i in range(1, 7)), context=None,
                 camera_profile=LEGACY_PROFILE):
        profile = get_camera_profile(camera_profile)
        if camera_ids is None:
            camera_ids = profile['camera_ids']
        if tuple(camera_ids) != tuple(profile['camera_ids']):
            raise ValueError('Policy cameras must exactly match the camera profile; observer is excluded')
        self.camera_profile = camera_profile
        self.run = Path(run).resolve()
        self.folder = self.run / 'rollout'
        self.folder.mkdir(exist_ok=False)
        self.clock = clock
        self.context = context or (lambda: {})
        self.camera_ids = tuple(camera_ids)
        if len(set(self.camera_ids)) != len(self.camera_ids) or physics_hz != 240:
            raise ValueError('Expected unique cameras and reviewed 240Hz physics')
        self.physics_hz = physics_hz
        self.contract = {
            'schema': 'depallet.native_rollout_trace.v1', 'physics_hz': physics_hz,
            'camera_ids': list(camera_ids), 'camera_hz': profile['camera_hz'],
            'joint_names': list(joint_names), 'position_unit': 'radian',
            'velocity_unit': 'radian/second', 'effort_unit': 'newton_metre',
            'tcp_quaternion_order': 'wxyz', 'tcp_position_unit': 'metre',
            'alignment_tolerance_s': 1e-6, 'max_camera_age_s': profile['max_camera_age_s'],
            'teacher_input_provider': 'simulation_oracle',
            'task_instruction': 'Move every carton from the source pallet onto the target pallet in a stable, orderly arrangement.',
            'teacher_uses_rgb': False, 'physics_feedback_source': 'simulation_oracle',
            'sensor_timestamp_source': 'shared_render_product_simulation_time_and_reference_time',
            'sensor_buffer_consistency_check': 'clock_unchanged_before_after_rgb_depth_read',
            'alignment_claim': 'timestamp_correspondence; visual/physics sensor validation pending',
            'command_semantics': 'backend_target_calls_in_issue_order; physical response separately measured',
            'application_semantics': 'post-step active command references; superseded same-time calls retained',
            'recording_starts_after_initialization': True,
            'reset_after_recording_allowed': False, 'training_dataset': False,
            'valid_sim_hours': 0,
        }
        if camera_profile == CAMERA_PROFILE_V2:
            self.contract.update(profile, camera_profile=camera_profile,
                                 observer_excluded_from_policy=True,
                                 camera_pose_semantics='capture_time_world_from_opencv; wrist_extrinsics_relative_to_flange',
                                 flange_quaternion_order='wxyz', camera_pose_composition_tolerance=1e-4)
        (self.folder / 'contract.json').write_text(_json(self.contract) + '\n')
        self.stream = (self.folder / 'events.jsonl').open('x', buffering=1024 * 1024)
        self.seq = 0
        self.latest_physics = None
        self.physics_history = deque()
        self.command_refs = {}
        self.pending_attempts = {}
        self.available_observations = {}
        self.vacuum_command = None
        self.callback_error = None
        self.phase, self.transfer_id = 'INITIALIZING', None
        self.last_flush = time.monotonic()
        self.closed = False

    def _write(self, kind, **fields):
        if self.closed:
            raise RuntimeError('Trace is finalized')
        row = {'seq': self.seq, 'kind': kind, **fields}
        encoded = _json(row)
        self.stream.write(encoded + '\n')
        self.seq += 1
        if time.monotonic() - self.last_flush > 1.:
            self.stream.flush()
            self.last_flush = time.monotonic()
        return row

    def _context(self):
        result = {'phase': self.phase, 'transfer_id': self.transfer_id}
        result.update(self.context())
        return result

    def command_attempt(self, method, value):
        t, step = self.clock()
        row = self._write('command_attempt', sim_time=t, physics_step=step,
                          method=method, value=value,
                          available_observations=dict(self.available_observations),
                          **self._context())
        self.pending_attempts[row['seq']] = row
        return row['seq']

    def command_completed(self, method, value, attempt, *, status, error=None):
        t, step = self.clock()
        issued = self.pending_attempts.get(attempt)
        if issued is None or issued['method'] != method or issued['value'] != value:
            raise ValueError('Completion does not match an outstanding command attempt')
        if status not in ('returned', 'raised'):
            raise ValueError('Invalid command completion status')
        row = self._write('command', sim_time=issued['sim_time'], physics_step=issued['physics_step'],
                          returned_sim_time=t, returned_physics_step=step,
                          method=method, value=value, attempt_seq=attempt,
                          status=status, error=error,
                          available_observations=dict(issued['available_observations']),
                          phase=issued['phase'], transfer_id=issued['transfer_id'])
        del self.pending_attempts[attempt]
        if status == 'returned':
            key = 'vacuum' if method in VACUUM_METHODS else method
            self.command_refs[key] = row['seq']
            if key == 'vacuum':
                self.vacuum_command = value
        return row

    def wrap_robot(self, robot):
        return ActuatorProxy(robot, self, ARM_METHODS)

    def wrap_surface(self, surface):
        return ActuatorProxy(surface, self, VACUUM_METHODS)

    def physics(self, *, sim_time, physics_step, q_rad, dq_rad_s,
                tcp_pose_wxyz, gripper_status, flange_pose_wxyz=None):
        if not math.isfinite(sim_time) or sim_time < 0 or type(physics_step) is not int or physics_step < 0:
            raise ValueError('Invalid physics clock')
        if self.latest_physics is not None:
            previous = self.latest_physics
            if physics_step != previous['physics_step'] + 1:
                raise ValueError(f"Missing/duplicate physics snapshot {previous['physics_step']} -> {physics_step}")
            if abs(sim_time - previous['sim_time'] - 1 / self.physics_hz) > 1e-6:
                raise ValueError('Physics time reset/gap')
        tcp = _vector(tcp_pose_wxyz, 7)
        if abs(sum(x*x for x in tcp[3:]) - 1.) > 1e-4:
            raise ValueError('TCP quaternion is not normalized')
        extra_state = {}
        if flange_pose_wxyz is not None:
            flange = _vector(flange_pose_wxyz, 7)
            pose_matrix(flange)
            extra_state['flange_pose_wxyz'] = flange
        row = self._write('physics', sim_time=float(sim_time), physics_step=physics_step,
                          state={'q_rad': _vector(q_rad, 6), 'dq_rad_s': _vector(dq_rad_s, 6),
                                 'tcp_pose_wxyz': tcp, 'vacuum_command': self.vacuum_command,
                                 'gripper_status': str(gripper_status), **extra_state},
                          active_command_refs=dict(self.command_refs), **self._context())
        self.latest_physics = row
        self.physics_history.append(row)
        while self.physics_history and self.physics_history[0]['sim_time'] < sim_time - 2.:
            self.physics_history.popleft()
        return row

    def check_current_physics(self, sim_time, physics_step):
        if self.callback_error:
            raise RuntimeError('Rollout physics callback: ' + self.callback_error)
        row = self.latest_physics
        if row is None or row['physics_step'] != physics_step or abs(row['sim_time'] - sim_time) > 1e-6:
            raise RuntimeError('Post-step callback clock/state differs from main-loop physics')

    def observation(self, camera_id, frame_index, stamp, *, rgb_path, depth_path, camera_pose=None):
        if camera_id not in self.camera_ids:
            raise ValueError('Unknown camera')
        t, step = self.clock()
        image_time = stamp['image_sim_time']
        if isinstance(image_time, bool) or not isinstance(image_time, (int, float)) or not math.isfinite(image_time) or image_time < 0:
            raise ValueError('Invalid image timestamp')
        if image_time > t + 1e-6:
            raise ValueError('Image timestamp is in the future of receipt')
        candidates = [r for r in self.physics_history if abs(r['sim_time'] - image_time) <= 1e-6]
        captured = candidates[0] if len(candidates) == 1 else None
        if captured is not None and (captured['physics_step'] > step or captured['sim_time'] > t + 1e-6):
            raise ValueError('Capture state is in the future of receipt')
        if captured is None and self.transfer_id is not None:
            raise RuntimeError(f'Active image {image_time} has no exact post-step state')
        pose_fields = {}
        if camera_pose is not None or (self.camera_profile == CAMERA_PROFILE_V2 and camera_id == 'wrist'):
            pose_fields['camera_pose'] = validate_camera_pose(camera_pose, image_time)
        row = self._write('observation', camera_id=camera_id, frame_index=frame_index,
                          rgb_path=rgb_path, depth_path=depth_path, image_sim_time=image_time,
                          received_sim_time=t, received_physics_step=step,
                          capture_state_seq=None if captured is None else captured['seq'],
                          capture_time_error_s=None if captured is None else captured['sim_time']-image_time,
                          render_reference=stamp, **pose_fields)
        self.available_observations[camera_id] = row['seq']
        return row

    def skill(self, phase, transfer_id, instruction):
        self.phase, self.transfer_id = phase, transfer_id
        t, step = self.clock()
        return self._write('skill', sim_time=t, physics_step=step,
                           phase=phase, transfer_id=transfer_id, instruction=instruction)

    def control(self):
        t, step = self.clock()
        self.check_current_physics(t, step)
        return self._write('control', sim_time=t, physics_step=step,
                           state_seq=self.latest_physics['seq'],
                           command_refs=dict(self.command_refs),
                           available_observations=dict(self.available_observations), **self._context())

    def finish(self, *, task_complete=False, requested_prefix_passed=False, error=None):
        if self.closed:
            return
        self._write('end', task_complete=bool(task_complete),
                    requested_prefix_passed=bool(requested_prefix_passed), error=error,
                    callback_error=self.callback_error, pending_attempts=sorted(self.pending_attempts))
        self.stream.flush()
        import os
        os.fsync(self.stream.fileno())
        self.stream.close()
        self.closed = True
        names = ['rollout/contract.json', 'rollout/events.jsonl', 'camera.json',
                 'manifest.json', 'source_camera/camera.json', 'source_camera/manifest.json',
                 'task-result.json', 'code-snapshot/manifest.json', 'runtime-contract.json',
                 'robot-scene.json', 'scenario.json', 'task-plan.json', 'assembly-input.json']
        if self.camera_profile == CAMERA_PROFILE_V2:
            names = [name for name in names if not name.startswith('source_camera/')]
            names += ['wrist_camera/camera.json', 'wrist_camera/manifest.json', 'camera-rig.json']
        hashes = {name: _sha(self.run / name) for name in names if (self.run / name).is_file()}
        manifest = {'schema': 'depallet.native_rollout_manifest.v1', 'finalized': True,
                    'task_complete': bool(task_complete),
                    'requested_prefix_passed': bool(requested_prefix_passed),
                    'error': error, 'callback_error': self.callback_error, 'event_count': self.seq,
                    'files': hashes, 'training_dataset': False, 'valid_sim_hours': 0}
        (self.folder / 'manifest.json').write_text(_json(manifest) + '\n')
