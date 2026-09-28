"""Measured-state single-box execution using joint drive targets and Surface Gripper.

No simulation startup, reset, joint-state write or object-pose write is performed.
Call step(sim_time) once AFTER each physics step. At GRASP_CONFIRMED, the caller
must obtain a payload-aware transport plan from the current measured state.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import io
import json
import math
import zipfile
from pathlib import Path
import numpy as np

JOINT_NAMES = tuple(f"joint_{i}" for i in range(1, 7))
STILLNESS_THRESHOLDS = dict(required_stable_seconds=.25, timeout_s=4.,
    joint_api_speed_rad_s=.01, joint_fd_speed_rad_s=.001,
    tool_translation_fd_speed_m_s=.0005, tool_orientation_fd_speed_rad_s=.001,
    box_speed_m_s=.001, box_angular_speed_rad_s=.005)


class ExecutionError(RuntimeError):
    pass


def vector(value, length, name):
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size != length or not np.isfinite(array).all():
        raise ExecutionError(f"Invalid measured {name}")
    return array


def matches(path, target):
    return path == target or path.startswith(target+"/")


def _verify_upright_load(request, positions, result, result_path):
    """Bind planner receipts, then independently certify the actual loaded curve.

    Only the explicit cuRobo upright branch uses this helper. Dedicated CPU
    contact-escape validation and legacy policy-free trajectories remain separate.
    """
    try:
        import yaml
        from depallet.validation import payload_upright
        from depallet.motion.curobo_bridge import validate_request
        from depallet.validation.payload_upright import validate_upright_request, certify_upright_trajectory
        validate_request(request)
        policy = validate_upright_request(request)
        if policy is None:
            raise ValueError("Missing explicit upright request policy")
        if tuple(request["joint_names"]) != JOINT_NAMES:
            raise ValueError("Upright request joint order differs from H2017")
        if (result.get("payload_collision_enabled") is not True
                or result.get("payload_box_id") != request["payload"]["box_id"]):
            raise ValueError("Upright result does not name the collision-enabled requested payload")
        certificate = result.get("payload_upright_certificate")
        certificate_path = Path(result_path).parent/"upright-certificate.json"
        if not isinstance(certificate, dict) or not certificate_path.is_file():
            raise ValueError("Missing embedded or separate upright certificate")
        separate = json.loads(certificate_path.read_text())
        canonical = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if canonical(certificate) != canonical(separate):
            raise ValueError("Embedded and separate upright certificates differ")
        if (certificate.get("schema") != "depallet.payload_upright.certificate.v1"
                or certificate.get("passed") is not True
                or certificate.get("initial_state_boundary_included") is not True
                or certificate.get("trajectory_samples") != len(positions)
                or certificate.get("policy") != policy):
            raise ValueError("Upright certificate success/policy/sample metadata is invalid")
        config_path = Path(request["robot_config"])
        config_raw = config_path.read_bytes()
        config_digest = hashlib.sha256(config_raw).hexdigest()
        config = yaml.safe_load(config_raw)
        kinematics = config.get("robot_cfg", config)["kinematics"]
        urdf_path = Path(kinematics["urdf_path"])
        urdf_digest = hashlib.sha256(urdf_path.read_bytes()).hexdigest()
        expected = dict(
            request_content_sha256=hashlib.sha256(canonical(request).encode()).hexdigest(),
            joint_positions_float64_sha256=hashlib.sha256(np.ascontiguousarray(positions, dtype="<f8").tobytes()).hexdigest(),
            robot_config_sha256=config_digest, urdf_sha256=urdf_digest,
            checker_sha256=hashlib.sha256(Path(payload_upright.__file__).read_bytes()).hexdigest())
        for key, value in expected.items():
            if certificate.get(key) != value:
                raise ValueError("Upright certificate linkage mismatch: " + key)
        if result.get("robot_config_sha256") != config_digest:
            raise ValueError("Upright planner result robot config hash differs")
        effective_path = Path(result_path).parent/"effective-robot-payload.yml"
        effective_raw = effective_path.read_bytes()
        effective_digest = hashlib.sha256(effective_raw).hexdigest()
        if result.get("effective_robot_config_sha256") != effective_digest:
            raise ValueError("Upright effective robot config hash differs")
        effective = yaml.safe_load(effective_raw)
        effective_kinematics = effective.get("robot_cfg", effective)["kinematics"]
        if (effective_kinematics["base_link"] != kinematics["base_link"]
                or Path(effective_kinematics["urdf_path"]).resolve() != urdf_path.resolve()):
            raise ValueError("Upright effective configuration changed the reviewed FK chain")
        # Use the original hashed request config, not the worker's kinematic object.
        recomputed = certify_upright_trajectory(request, positions, robot_config=config)
        if recomputed.get("passed") is not True:
            raise ValueError("Independent upright recertification failed: " + str(recomputed.get("failure")))

        def equivalent(a, b):
            if isinstance(a, dict) and isinstance(b, dict):
                return a.keys() == b.keys() and all(equivalent(a[k], b[k]) for k in a)
            if isinstance(a, list) and isinstance(b, list):
                return len(a) == len(b) and all(equivalent(x, y) for x, y in zip(a, b))
            if isinstance(a, bool) or isinstance(b, bool):
                return type(a) is type(b) and a == b
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
            return type(a) is type(b) and a == b

        if not equivalent(certificate, recomputed):
            raise ValueError("Upright certificate differs from independent FK recertification")
        if (hashlib.sha256(config_path.read_bytes()).hexdigest() != config_digest
                or hashlib.sha256(urdf_path.read_bytes()).hexdigest() != urdf_digest
                or hashlib.sha256(effective_path.read_bytes()).hexdigest() != effective_digest):
            raise ValueError("Upright robot inputs changed during recertification")
        return dict(passed=True, independently_recomputed_from_loaded_positions=True,
                    certificate_path=str(certificate_path), certificate_sha256=hashlib.sha256(certificate_path.read_bytes()).hexdigest(),
                    effective_robot_config_sha256=effective_digest, **expected,
                    maximum_certified_tilt_bound_rad=recomputed["maximum_certified_tilt_bound_rad"],
                    physical_execution_validated=False, runtime_measured_tilt_and_attachment_checks_required=True)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError) as error:
        raise ExecutionError("Upright trajectory verification failed: " + str(error)) from error


@dataclass(frozen=True)
class Trajectory:
    positions: np.ndarray
    velocities: np.ndarray
    times: np.ndarray
    result: dict
    sha256: str
    source_verified: bool = False

    @classmethod
    def load(cls, npz_path, result_path, limits, max_velocities, request_path=None):
        path = Path(npz_path)
        if not path.is_file() or path.stat().st_size > 32*1024*1024:
            raise ExecutionError("Trajectory file missing or above 32 MiB")
        result = json.loads(Path(result_path).read_text())
        npz_bytes = path.read_bytes()
        digest = hashlib.sha256(npz_bytes).hexdigest()
        curobo = result.get("schema") == "depallet.curobo.v2.result.v1" and result.get("planner") == "cuRoboV2"
        escape = (result.get("schema") == "depallet.contact_escape.result.v1"
                  and result.get("planner") == "CPUContactEscape"
                  and result.get("contact_escape_validated") is True
                  and result.get("payload_collision_enabled") is True)
        if (not (curobo or escape) or result.get("success") is not True
                or result.get("trajectory_sha256") != digest):
            raise ExecutionError("Trajectory result provenance/hash is invalid")
        if escape and request_path is None:
            raise ExecutionError("Contact escape requires its measured-state request")
        with zipfile.ZipFile(io.BytesIO(npz_bytes)) as archive:
            if sum(item.file_size for item in archive.infolist()) > 64*1024*1024:
                raise ExecutionError("Trajectory expands above 64 MiB")
        request = None
        motion_limits = None
        if request_path is not None:
            request_bytes = Path(request_path).read_bytes()
            request_digest = hashlib.sha256(request_bytes).hexdigest()
            if result.get("request_sha256") != request_digest:
                raise ExecutionError("Planner result does not match the supplied request")
            try:
                request = json.loads(request_bytes)
                if not isinstance(request, dict):
                    raise ValueError("Expected a request object")
            except (ValueError, TypeError) as error:
                raise ExecutionError("Invalid trajectory request JSON") from error
            if curobo:
                try:
                    from depallet.motion.curobo_bridge import validate_request, validate_result_motion_limits
                    validate_request(request)
                    motion_limits = validate_result_motion_limits(request, result)
                except (ValueError, TypeError) as error:
                    raise ExecutionError("Trajectory motion-profile evidence is invalid") from error
        with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as raw:
            names = tuple(raw["joint_names"].tolist())
            q, v, t = [np.asarray(raw[key], dtype=float).copy() for key in ["position_rad", "velocity_rad_s", "time_s"]]
            named_motion_profile = (motion_limits is not None
                                    and not motion_limits["legacy_request"])
            if named_motion_profile and "acceleration_rad_s2" not in raw.files:
                raise ExecutionError("Named-profile trajectory lacks acceleration samples")
            acceleration = (np.asarray(raw["acceleration_rad_s2"], dtype=float).copy()
                            if named_motion_profile else None)
        if names != JOINT_NAMES or tuple(result.get("joint_names", [])) != JOINT_NAMES:
            raise ExecutionError("Trajectory joint order differs from H2017")
        if q.ndim != 2 or q.shape[1] != 6 or not 2 <= len(q) <= 32000 or v.shape != q.shape or t.shape != (len(q),):
            raise ExecutionError("Invalid trajectory dimensions")
        if acceleration is not None and acceleration.shape != q.shape:
            raise ExecutionError("Invalid trajectory acceleration dimensions")
        if (not np.isfinite(q).all() or not np.isfinite(v).all() or not np.isfinite(t).all()
                or (acceleration is not None and not np.isfinite(acceleration).all())):
            raise ExecutionError("Non-finite trajectory data")
        dt = np.diff(t)
        if abs(t[0]) > 1e-9 or dt.min() < 1/240-1e-8 or dt.max() > .05+1e-8 or not np.allclose(dt, dt[0], atol=1e-8) or t[-1] > 120:
            raise ExecutionError("Trajectory clock is irregular or out of bounds")
        if result.get("waypoints") != len(q) or abs(float(result.get("interpolation_dt_s", 0.))-dt[0]) > 1e-8:
            raise ExecutionError("Trajectory timing differs from planner result")
        bounds, velocity_limits = np.asarray(limits, float), np.asarray(max_velocities, float)
        if bounds.shape != (6, 2) or velocity_limits.shape != (6,) or not np.isfinite(bounds).all() or not np.isfinite(velocity_limits).all():
            raise ExecutionError("Invalid reviewed joint limits")
        if np.any(bounds[:, 0] >= bounds[:, 1]) or np.any(velocity_limits <= 0):
            raise ExecutionError("Invalid reviewed joint limits")
        if named_motion_profile:
            velocity_limits = np.minimum(velocity_limits, motion_limits["maximum_velocity_rad_s"])
            if np.any(np.abs(acceleration) > motion_limits["maximum_acceleration_rad_s2"]+1e-4):
                raise ExecutionError("Trajectory exceeds request acceleration limit")
        if np.any(q < bounds[:, 0]-1e-6) or np.any(q > bounds[:, 1]+1e-6) or np.any(np.abs(v) > velocity_limits+1e-5):
            raise ExecutionError("Trajectory exceeds joint position/velocity limits")
        if np.max(np.abs(np.diff(q, axis=0))) > .1+1e-8:
            raise ExecutionError("Trajectory waypoint discontinuity")
        if np.any(np.abs(np.diff(q, axis=0)/dt[:, None]) > velocity_limits+1e-4):
            raise ExecutionError("Trajectory finite-difference velocity exceeds limits")
        if escape:
            from depallet.motion.contact_escape import validate_saved_escape
            validate_saved_escape(Path(request_path), path, Path(result_path))
        elif curobo:
            policy_requested = request is not None and request.get("payload_orientation_policy") is not None
            certificate_present = (result.get("payload_upright_certificate") is not None
                                   or (Path(result_path).parent/"upright-certificate.json").exists())
            if policy_requested:
                if path.resolve().parent != Path(result_path).resolve().parent:
                    raise ExecutionError("Upright trajectory/result artifacts must share their run directory")
                result = dict(result, execution_upright_revalidation=_verify_upright_load(request, q, result, result_path))
            elif certificate_present:
                raise ExecutionError("Upright certificate requires its explicit policy-bearing request")
        for a in (q, v, t):
            a.setflags(write=False)
        return cls(q, v, t, result, digest, True)

    def sample(self, elapsed):
        if elapsed >= self.times[-1]:
            return self.positions[-1].copy(), np.zeros(6), True
        index = max(0, int(np.searchsorted(self.times, elapsed, side="right"))-1)
        alpha = max(0., (elapsed-self.times[index])/(self.times[index+1]-self.times[index]))
        return ((1-alpha)*self.positions[index]+alpha*self.positions[index+1],
                (1-alpha)*self.velocities[index]+alpha*self.velocities[index+1], False)


class SingleBoxStepper:
    """Drive one approach, confirmed grip, payload transport and measured release.

    measure_box() returns position_m, quaternion_wxyz, linear_velocity_m_s and
    angular_velocity_rad_s from the current simulated rigid body. The optional
    surface/status parameters enable CPU controlled stubs without importing Isaac.
    Optional preclose_stability/prerelease_stability require measure_tool_pose() returning current
    position_m and normalized quaternion_wxyz; it holds the endpoint until joint,
    tool and target speeds remain within fixed limits for 0.25 measured seconds.
    """
    def __init__(self, robot, stage, gripper_info, measure_box, *, box_path,
                 goal_position_m, limits, max_velocities, box_id=None, goal_quaternion_wxyz=(1., 0., 0., 0.),
                 surface=None, closed_status=None, open_status=None, measurement_source="unspecified",
                 tracking_limit_rad=.15, grasp_timeout_s=3., release_timeout_s=3., settle_timeout_s=8.,
                 gravity_compensation=False, max_efforts_nm=None, contact_escape=False, pre_grasp_check=None,
                 preclose_stability=False, measure_tool_pose=None, prerelease_stability=False,
                 prerelease_payload_compensation=None, prerelease_pose_check=None):
        if surface is None:
            from isaacsim.robot.surface_gripper import _surface_gripper
            surface = _surface_gripper.acquire_surface_gripper_interface()
            closed_status, open_status = _surface_gripper.GripperStatus.Closed, _surface_gripper.GripperStatus.Open
        if closed_status is None or open_status is None:
            raise ExecutionError("Explicit gripper status values are required")
        if tuple(robot.dof_names) != JOINT_NAMES:
            raise ExecutionError("Runtime robot joint order differs from H2017")
        self.robot, self.stage, self.measure_box = robot, stage, measure_box
        self.surface, self.closed_status, self.open_status = surface, closed_status, open_status
        self.gripper = gripper_info["gripper_path"]
        self.joint_paths = list(gripper_info["joint_paths"])
        self.box_path = str(box_path)
        self.box_id = box_id or self.box_path.rsplit("/", 1)[-1]
        if not self.box_path.startswith("/") or not self.gripper.startswith("/") or not self.joint_paths:
            raise ExecutionError("Expected absolute prim paths and attachment joints")
        self.goal = vector(goal_position_m, 3, "goal")
        self.goal_quat = vector(goal_quaternion_wxyz, 4, "goal quaternion")
        if abs(np.linalg.norm(self.goal_quat)-1) > 1e-4:
            raise ExecutionError("Goal quaternion must be normalized")
        self.limits, self.max_velocities = np.asarray(limits, float), vector(max_velocities, 6, "velocity limits")
        if self.limits.shape != (6, 2) or not np.isfinite(self.limits).all() or np.any(self.limits[:, 0] >= self.limits[:, 1]) or np.any(self.max_velocities <= 0):
            raise ExecutionError("Invalid joint limits")
        self.tracking_limit, self.grasp_timeout = float(tracking_limit_rad), float(grasp_timeout_s)
        self.release_timeout, self.settle_timeout = float(release_timeout_s), float(settle_timeout_s)
        if not all(math.isfinite(v) and v > 0 for v in [self.tracking_limit, self.grasp_timeout, self.release_timeout, self.settle_timeout]):
            raise ExecutionError("Invalid execution thresholds")
        if type(preclose_stability) is not bool:
            raise ExecutionError("preclose_stability must be boolean")
        if type(prerelease_stability) is not bool:
            raise ExecutionError("prerelease_stability must be boolean")
        if measure_tool_pose is not None and not callable(measure_tool_pose):
            raise ExecutionError("measure_tool_pose must be callable")
        if preclose_stability and not callable(measure_tool_pose):
            raise ExecutionError("Pre-close stabilization requires measured tool pose callback")
        if prerelease_stability and not callable(measure_tool_pose):
            raise ExecutionError("Pre-release stabilization requires measured tool pose callback")
        self.preclose_stability = preclose_stability
        self.measure_tool_pose = measure_tool_pose
        self.preclose_stable_seconds = 0.
        self.preclose_stability_samples = []
        self._preclose_previous = None
        self._preclose_stable_since = None
        self._preclose_passed_at = None
        self.inspection_stable_seconds = 0.
        self.inspection_stability_samples = []
        self._inspection_previous = None
        self._inspection_stable_since = None
        self._inspection_passed_at = None
        self._inspection_resumed_at = None
        self._inspection_index = None
        self._inspection_boundary_index = None
        self._inspection_cut_time = None
        self._inspection_hold_q = None
        self.prerelease_stability = prerelease_stability
        self.prerelease_stable_seconds = 0.
        self.prerelease_stability_samples = []
        self._prerelease_previous = None
        self._prerelease_stable_since = None
        self._prerelease_started_at = None
        self._prerelease_passed_at = None
        self._prerelease_hold_q = None
        for name, callback in (("prerelease_payload_compensation", prerelease_payload_compensation),
                               ("prerelease_pose_check", prerelease_pose_check)):
            if callback is not None and not callable(callback):
                raise ExecutionError(name+" must be callable or None")
            if callback is not None and not prerelease_stability:
                raise ExecutionError(name+" requires prerelease_stability")
        if prerelease_payload_compensation is not None and not gravity_compensation:
            raise ExecutionError("Payload compensation requires native gravity compensation")
        self.prerelease_payload_compensation = prerelease_payload_compensation
        self.prerelease_pose_check = prerelease_pose_check
        self._payload_efforts = np.zeros(6)
        self._payload_native_cache = None
        self._payload_binding = None
        self._payload_ramp_fraction = 0.
        self._payload_full_ramp_at = None
        self._payload_current_receipt = None
        self._payload_samples = []
        self._payload_cleared_at_open = None
        self._prerelease_pose_last = None
        self._release_pose_binding = None
        self.last_combined_feedforward = np.zeros(6)
        self.peak_combined_feedforward = np.zeros(6)
        self.pre_grasp_check = pre_grasp_check
        self.pre_grasp_evidence = None
        self.contact_escape = bool(contact_escape)
        self.grasp_stable_seconds = 0.
        self.last_joint_velocities = np.zeros(6)
        self.grasp_stability_samples = []
        self.escape_confirmed = False
        self.gravity_compensation = bool(gravity_compensation)
        self.max_efforts = None
        self.gravity_samples, self.last_gravity_efforts, self.peak_gravity_efforts = 0, np.zeros(6), np.zeros(6)
        if self.gravity_compensation:
            if any(not callable(getattr(robot, name, None)) for name in ("get_dof_gravity_compensation_forces", "set_dof_efforts", "set_dof_max_efforts")):
                raise ExecutionError("Installed articulation gravity compensation/effort API unavailable")
            self.max_efforts = vector(max_efforts_nm, 6, "effort limits")
            if np.any(self.max_efforts <= 0): raise ExecutionError("Positive reviewed effort limits required")
        self.measurement_source = measurement_source
        self.state, self.failure = "IDLE", None
        self.last_time, self.state_since = None, None
        self.trajectory, self.trajectory_since = None, None
        self.commanded_q, self.commanded_v = None, None
        self.initial_box, self.max_box_z = None, -math.inf
        self.grasp_confirmed, self.lift_confirmed, self.release_confirmed = False, False, False
        self.attachment_samples, self.endpoint_samples, self.settled_seconds = 0, 0, 0.
        self.peak_tracking_error, self.samples = 0., 0
        self.events, self.trajectory_hashes = [], []
        self.last_evidence, self.last_box = {}, None

    def _transition(self, state, sim_time):
        self.state, self.state_since = state, float(sim_time)
        self.events.append(dict(state=state, sim_time_s=float(sim_time)))

    def _joint_state(self):
        return vector(self.robot.get_dof_positions(), 6, "joint positions"), vector(self.robot.get_dof_velocities(), 6, "joint velocities")

    def _box_state(self):
        raw = self.measure_box()
        value = {key: vector(raw[key], size, key) for key, size in
                 [("position_m", 3), ("quaternion_wxyz", 4), ("linear_velocity_m_s", 3), ("angular_velocity_rad_s", 3)]}
        if abs(np.linalg.norm(value["quaternion_wxyz"])-1) > .001:
            raise ExecutionError("Measured box quaternion is not normalized")
        return value

    def _command(self, q, v):
        # Drive targets only. Never call set_dof_positions or reset_to_default_state.
        self.robot.set_dof_position_targets([q.tolist()])
        self.robot.set_dof_velocity_targets([v.tolist()])
        if self.gravity_compensation:
            efforts = (vector(self.robot.get_dof_gravity_compensation_forces(), 6, "gravity compensation efforts")
                       if self._payload_native_cache is None else self._payload_native_cache.copy())
            combined = efforts+self._payload_efforts
            if not np.isfinite(combined).all() or np.any(np.abs(combined) > self.max_efforts):
                raise ExecutionError("Gravity compensation exceeds reviewed URDF torque limits")
            # Reserve torque headroom for the position drive. Even opposing drive
            # and feedforward signs cannot exceed the URDF bound in magnitude.
            self.robot.set_dof_max_efforts([(self.max_efforts-np.abs(combined)).tolist()])
            self.robot.set_dof_efforts([combined.tolist()])
            self.last_combined_feedforward = combined.copy()
            self.peak_combined_feedforward = np.maximum(self.peak_combined_feedforward, np.abs(combined))
            self.last_gravity_efforts = efforts.copy()
            self.peak_gravity_efforts = np.maximum(self.peak_gravity_efforts, np.abs(efforts))
            self.gravity_samples += 1
        self.commanded_q, self.commanded_v = q.copy(), v.copy()

    def _hold(self, q=None):
        if q is None:
            q = self.commanded_q if self.commanded_q is not None else self._joint_state()[0]
        self._command(q, np.zeros(6))

    def _abort(self, message):
        self.failure, self.state = str(message), "FAILED"
        self._payload_efforts = np.zeros(6)
        self._payload_native_cache = None
        self._payload_ramp_fraction = 0.
        try:
            q = self._joint_state()[0]
            self.robot.set_dof_position_targets([q.tolist()])
            self.robot.set_dof_velocity_targets([[0.]*6])
            if self.gravity_compensation:
                self.robot.set_dof_efforts([[0.]*6])
                self.robot.set_dof_max_efforts([self.max_efforts.tolist()])
            self.commanded_q, self.commanded_v = q, np.zeros(6)
        except Exception:
            pass
        raise ExecutionError(str(message))

    def _start_path(self, trajectory, sim_time, state):
        if not isinstance(trajectory, Trajectory) or not math.isfinite(sim_time) or sim_time < 0:
            raise ExecutionError("Expected checked trajectory and finite simulation time")
        if not trajectory.source_verified and self.measurement_source != "controlled_test_stub":
            raise ExecutionError("Runtime execution requires a hash-verified Trajectory.load result")
        if self.last_time is not None and abs(sim_time-self.last_time) > 1e-6:
            raise ExecutionError("New trajectory must start at the last measured simulation step")
        q, v = self._joint_state()
        if np.max(np.abs(q-trajectory.positions[0])) > .025:
            raise ExecutionError("Plan start differs from measured joints; replan required")
        if np.max(np.abs(v-trajectory.velocities[0])) > .05:
            raise ExecutionError("Plan start velocity differs from measured joints; replan required")
        if np.any(trajectory.positions < self.limits[:, 0]-1e-6) or np.any(trajectory.positions > self.limits[:, 1]+1e-6) or np.any(np.abs(trajectory.velocities) > self.max_velocities+1e-5):
            raise ExecutionError("Trajectory not within this controller's reviewed limits")
        self.trajectory, self.trajectory_since = trajectory, float(sim_time)
        self.trajectory_hashes.append(trajectory.sha256)
        self.endpoint_samples = 0
        self.last_time = float(sim_time)
        self._transition(state, sim_time)
        self._command(trajectory.positions[0], trajectory.velocities[0])

    def start_approach(self, trajectory, sim_time, *, inspection_index=None):
        if self.state != "IDLE":
            raise ExecutionError("Single-box approach can start only once")
        if inspection_index is not None:
            if (not isinstance(inspection_index, int) or isinstance(inspection_index, bool)
                    or not 0 < inspection_index < len(trajectory.positions)-1):
                raise ExecutionError("Inspection index must be an interior integer trajectory sample")
            if not self.preclose_stability or not callable(self.measure_tool_pose):
                raise ExecutionError("Inspection pause requires measured pre-close stability")
            if np.max(np.abs(trajectory.velocities[inspection_index])) > 1e-5:
                raise ExecutionError("Inspection boundary velocity must be at most 1e-5 rad/s")

        self.initial_box = self._box_state()
        self.max_box_z = float(self.initial_box["position_m"][2])
        if self.surface.get_gripped_objects(self.gripper):
            raise ExecutionError("Cannot start while the gripper already holds an object")
        self.surface.open_gripper(self.gripper)
        if inspection_index is None:
            self._start_path(trajectory, sim_time, "APPROACH")
        else:
            self._inspection_index = inspection_index
            self._inspection_cut_time = float(trajectory.times[inspection_index])
            self._inspection_boundary_index = inspection_index
            self._inspection_hold_q = trajectory.positions[inspection_index].copy()
            self._start_path(trajectory, sim_time, "INSPECTION_MOVE")

    def resume_after_inspection(self, sim_time):
        """Acknowledge a completed camera hold and resume the checked path suffix."""
        if self.state != "INSPECTION_HOLD" or self._inspection_index is None:
            raise ExecutionError("Inspection resume requires INSPECTION_HOLD")
        if (not isinstance(sim_time, (int, float)) or isinstance(sim_time, bool)
                or not math.isfinite(sim_time) or self.last_time is None
                or abs(float(sim_time)-self.last_time) > 1e-6):
            raise ExecutionError("Inspection resume must use the current measured simulation time")
        if sim_time-self.state_since < .5-1e-9:
            raise ExecutionError("Inspection hold requires at least 0.5 measured seconds")
        latest = self.inspection_stability_samples[-1] if self.inspection_stability_samples else None
        if (latest is None or latest.get("stable") is not True
                or abs(float(latest["sim_time_s"])-self.last_time) > 1e-6
                or self.inspection_stable_seconds < STILLNESS_THRESHOLDS["required_stable_seconds"]-1e-9):
            raise ExecutionError("Inspection resume requires current measured stability")
        if self._attachment_evidence().get("detached") is not True:
            raise ExecutionError("Inspection resume requires an open unloaded gripper")
        self.trajectory_since = float(sim_time)-self._inspection_cut_time
        self._inspection_resumed_at = float(sim_time)
        self._inspection_index = None
        self._transition("APPROACH", sim_time)

    def start_escape(self, trajectory, sim_time):
        if not self.contact_escape or self.state != "GRASP_CONFIRMED" or self.grasp_stable_seconds < .25-1e-9:
            raise ExecutionError("Contact escape requires stable confirmed attachment")
        if (trajectory.result.get("planner") != "CPUContactEscape"
                or trajectory.result.get("payload_box_id") != self.box_id
                or trajectory.result.get("contact_escape_validated") is not True):
            raise ExecutionError("Expected a checked contact escape for this payload")
        if not self._attachment_evidence()["target_attached"]:
            raise ExecutionError("Attachment disappeared before contact escape")
        _, velocity = self._joint_state()
        current_box = self._box_state()
        if (np.max(np.abs(velocity)) > .01 or np.linalg.norm(current_box["linear_velocity_m_s"]) > .02
                or np.linalg.norm(current_box["angular_velocity_rad_s"]) > .05):
            raise ExecutionError("Contact escape requires currently stable stopped measurements")
        self._start_path(trajectory, sim_time, "ESCAPE")

    def start_transport(self, trajectory, sim_time):
        expected_state = "ESCAPED" if self.contact_escape else "GRASP_CONFIRMED"
        if self.state != expected_state or not self.grasp_confirmed:
            raise ExecutionError("Transport requires a physically confirmed target attachment")
        if trajectory.result.get("payload_collision_enabled") is not True:
            raise ExecutionError("Transport requires payload collision geometry in the planner")
        if trajectory.result.get("payload_box_id") != self.box_id:
            raise ExecutionError("Payload plan belongs to a different target box")
        if not self._attachment_evidence()["target_attached"]:
            raise ExecutionError("Attachment disappeared before transport")
        self._start_path(trajectory, sim_time, "TRANSPORT")

    def start_retreat(self, trajectory, sim_time):
        if self.state != "RELEASED" or not self.release_confirmed:
            raise ExecutionError("Retreat requires confirmed release")
        if trajectory.result.get("payload_collision_enabled"):
            raise ExecutionError("Retreat plan incorrectly treats the released box as attached")
        self.settled_seconds = 0.
        self._start_path(trajectory, sim_time, "RETREAT")

    def _prepare_contact_or_close(self, sim_time):
        # This preserves the existing contact-preparation transaction. A caller
        # may require CONTACT_READY and a new physics snapshot before closure.
        if self.pre_grasp_check is not None:
            self.pre_grasp_evidence = self.pre_grasp_check()
            if not isinstance(self.pre_grasp_evidence, dict) or self.pre_grasp_evidence.get("passed") is not True:
                raise ExecutionError("Actual vacuum pad patch does not meet contact geometry gate")
        if self.pre_grasp_evidence and self.pre_grasp_evidence.get("requires_physics_update_before_close"):
            self._transition("CONTACT_READY", sim_time)
        else:
            self.surface.close_gripper(self.gripper)
            self._transition("CLOSING", sim_time)

    @staticmethod
    def _callback_record(raw, schema):
        if not isinstance(raw, dict) or raw.get("schema") != schema:
            raise ExecutionError("Invalid callback receipt schema: "+schema)
        try:
            # Copy callback-owned data and reject NaN/Infinity/non-JSON values.
            return json.loads(json.dumps(raw, allow_nan=False))
        except (ValueError, TypeError) as exc:
            raise ExecutionError("Non-finite or non-serializable callback receipt") from exc

    @staticmethod
    def _normalized_quaternion(raw, name):
        value = vector(raw, 4, name)
        length = float(np.linalg.norm(value))
        if abs(length-1.) > .001:
            raise ExecutionError("Invalid normalized callback quaternion: "+name)
        return value/length

    @classmethod
    def _same_quaternion(cls, first, second):
        a, b = cls._normalized_quaternion(first, "binding"), cls._normalized_quaternion(second, "measured")
        return min(float(np.linalg.norm(a-b)), float(np.linalg.norm(a+b))) <= 1e-7

    @staticmethod
    def _rotation_wxyz(quaternion):
        w, x, y, z = quaternion
        return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

    def _prepare_payload_compensation(self, sim_time, q, box):
        if self.prerelease_payload_compensation is None:
            return
        raw = self.prerelease_payload_compensation(q.copy(), {k: v.copy() for k, v in box.items()})
        receipt = self._callback_record(raw, "depallet.payload_gravity.evaluation.v1")
        candidate = vector(receipt["payload_compensation_nm"], 6, "payload compensation")
        own = vector(receipt["optional_articulation_gravity_nm"], 6, "analytic arm/tool gravity compensation")
        if np.any(np.abs(candidate) > .15*self.max_efforts+1e-12):
            raise ExecutionError("Payload compensation candidate exceeds 15 percent motor bound")
        sample, provenance = receipt["sample"], receipt["artifact_provenance"]
        if not isinstance(sample, dict) or not isinstance(provenance, dict):
            raise ExecutionError("Payload callback sample/provenance must be mappings")
        if not np.allclose(vector(sample["actual_joint_positions_rad"], 6, "callback actual q"), q, rtol=0, atol=1e-10):
            raise ExecutionError("Payload callback joint-state binding differs")
        box_pose = vector(sample["box_pose_world_wxyz"], 7, "callback box pose")
        if (not np.allclose(box_pose[:3], box["position_m"], rtol=0, atol=1e-8)
                or not self._same_quaternion(box_pose[3:], box["quaternion_wxyz"])):
            raise ExecutionError("Payload callback box-state binding differs")
        tool = self.measure_tool_pose()
        tcp = vector(sample["tcp_pose_world_wxyz"], 7, "callback TCP pose")
        if (not isinstance(tool, dict) or not np.allclose(tcp[:3], vector(tool["position_m"], 3, "actual TCP"), rtol=0, atol=1e-7)
                or not self._same_quaternion(tcp[3:], tool["quaternion_wxyz"])):
            raise ExecutionError("Payload callback TCP-state binding differs")
        if (tuple(provenance["joint_names"]) != JOINT_NAMES or provenance["base_frame"] != "base_link"
                or provenance["tcp_frame"] != "suction_tcp"):
            raise ExecutionError("Payload callback frame/joint provenance differs")
        base_pose = vector(provenance["base_pose_world_wxyz"], 7, "payload base frame")
        self._normalized_quaternion(base_pose[3:], "base frame")
        if not np.allclose(vector(provenance["max_efforts_nm"], 6, "payload effort limits"), self.max_efforts, rtol=0, atol=1e-8):
            raise ExecutionError("Payload callback motor limits differ")
        mass = provenance["mass_kg"]
        if isinstance(mass, bool) or not isinstance(mass, (int, float)) or not math.isfinite(mass) or mass <= 0:
            raise ExecutionError("Payload callback mass is invalid")
        if np.any(vector(provenance["dimensions_m"], 3, "payload dimensions") <= 0):
            raise ExecutionError("Payload callback dimensions are invalid")
        com_local = vector(provenance["center_of_mass_local_m"], 3, "payload local COM")
        gravity = vector(provenance["gravity_world_m_s2"], 3, "payload gravity frame")
        if not np.allclose(gravity, [0., 0., -9.81], rtol=0, atol=1e-8):
            raise ExecutionError("Payload callback world gravity differs")
        for key in ("robot_config_sha256", "urdf_sha256", "evaluator_sha256", "independent_fk_sha256"):
            value = provenance[key]
            if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ExecutionError("Payload callback artifact hash is invalid: "+key)
        binding = json.dumps(provenance, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if self._payload_binding is not None and binding != self._payload_binding:
            raise ExecutionError("Payload callback artifact binding changed during hold")
        self._payload_binding = binding
        rotation = self._rotation_wxyz(self._normalized_quaternion(box["quaternion_wxyz"], "box COM"))
        expected_com = box["position_m"]+rotation@com_local
        if not np.allclose(vector(sample["measured_com_world_m"], 3, "actual COM"), expected_com, rtol=0, atol=1e-7):
            raise ExecutionError("Payload callback measured COM binding differs")
        jacobian = np.asarray(sample["jacobian_world_m_per_rad"], dtype=float)
        if jacobian.shape != (3, 6) or not np.isfinite(jacobian).all():
            raise ExecutionError("Payload callback Jacobian must be finite 3x6")
        fractions = vector(sample["candidate_motor_effort_fraction"], 6, "payload motor fractions")
        if not np.allclose(fractions, np.abs(candidate)/self.max_efforts, rtol=1e-8, atol=1e-8):
            raise ExecutionError("Payload callback motor fraction linkage differs")
        recomputed = -(jacobian.T@(mass*gravity))
        if (not np.allclose(candidate, recomputed, rtol=1e-8, atol=1e-8)
                or not np.allclose(vector(sample["external_gravity_generalized_force_nm"], 6, "payload gravity load"), -candidate, rtol=1e-8, atol=1e-8)):
            raise ExecutionError("Payload compensation sign/Jacobian linkage differs")
        native = vector(self.robot.get_dof_gravity_compensation_forces(), 6, "native gravity compensation")
        tolerance = .1+.001*np.abs(native)
        norm = float(np.linalg.norm(candidate))
        discrimination_floor = 2*float(np.linalg.norm(tolerance))
        own_error, combined_error = native-own, native-(own+candidate)
        own_matches = bool(np.all(np.abs(own_error) <= tolerance))
        already_included = bool(np.all(np.abs(combined_error) <= tolerance))
        check = dict(passed=own_matches and not already_included and norm > discrimination_floor,
            analytic_arm_tool_gravity_nm=own.tolist(), native_minus_own_nm=own_error.tolist(),
            native_minus_own_plus_payload_nm=combined_error.tolist(), tolerance_nm=tolerance.tolist(),
            absolute_tolerance_nm=.1, relative_tolerance=.001, native_matches_own=own_matches,
            native_matches_own_plus_payload=already_included, payload_norm_nm=norm,
            minimum_distinguishable_payload_norm_nm=discrimination_floor)
        previous = self._payload_efforts.copy()
        previous_fraction = self._payload_ramp_fraction
        elapsed = float(sim_time-self._prerelease_started_at)
        fraction = 1. if elapsed >= .5-1e-9 else max(0., elapsed/.5)
        applied = fraction*candidate
        total = native+applied
        combined_within_limits = bool(np.all(np.abs(total) <= self.max_efforts))
        record = dict(schema="depallet.prerelease_payload_command.v1", sim_time_s=float(sim_time),
            callback_response=receipt, ramp_fraction=fraction, ramp_seconds=.5,
            previous_applied_payload_nm=previous.tolist(), previous_ramp_fraction=previous_fraction,
            native_gravity_compensation_nm=native.tolist(), payload_candidate_nm=candidate.tolist(),
            applied_payload_nm=applied.tolist(), combined_feedforward_nm=total.tolist(),
            drive_headroom_nm=(self.max_efforts-np.abs(total)).tolist(),
            native_own_gravity_check=check, combined_within_motor_limits=combined_within_limits,
            command_issued=False, ready=False)
        self._payload_current_receipt = record
        self._payload_samples.append(record)
        if norm <= discrimination_floor:
            raise ExecutionError("Payload gravity too small to distinguish native inclusion safely")
        if already_included:
            raise ExecutionError("Native gravity already includes payload; duplicate compensation rejected")
        if not own_matches:
            raise ExecutionError("Native and independent arm/tool gravity disagree")
        if not combined_within_limits:
            raise ExecutionError("Combined gravity and payload compensation exceeds motor limit")
        if fraction == 1. and self._payload_full_ramp_at is None:
            self._payload_full_ramp_at = float(sim_time)
        full_precedes = (self._payload_full_ramp_at is not None and sim_time > self._payload_full_ramp_at+1e-9
                         and previous_fraction == 1.)
        record.update(first_full_ramp_command_at_sim_time_s=self._payload_full_ramp_at,
                      full_ramp_command_precedes_measurement=full_precedes, ready=bool(fraction == 1. and full_precedes))
        self._payload_efforts, self._payload_native_cache = applied.copy(), native.copy()
        self._payload_ramp_fraction = fraction

    def _checked_release_pose(self, q, box):
        raw = self.prerelease_pose_check(q.copy(), {k: v.copy() for k, v in box.items()})
        result = self._callback_record(raw, "depallet.prerelease_pose_check.v1")
        if type(result.get("passed")) is not bool or result.get("measurement_source") != self.measurement_source:
            raise ExecutionError("Release pose callback source/success is invalid")
        measured = vector(result["measured_box_position_m"], 3, "release measured box")
        if (not np.allclose(measured, box["position_m"], rtol=0, atol=1e-8)
                or not self._same_quaternion(result["measured_box_quaternion_wxyz"], box["quaternion_wxyz"])):
            raise ExecutionError("Release pose callback measured-state binding differs")
        expected = vector(result["expected_box_position_m"], 3, "expected release position")
        if not .005-1e-12 <= float(expected[2]-self.goal[2]) <= .08+1e-12:
            raise ExecutionError("Expected release height must preserve checked 5..80 mm gap")
        expected_q = self._normalized_quaternion(result["expected_box_quaternion_wxyz"], "expected release orientation")
        measured_q = self._normalized_quaternion(box["quaternion_wxyz"], "measured release orientation")
        if (not np.allclose(expected[:2], self.goal[:2], rtol=0, atol=1e-9)
                or not self._same_quaternion(expected_q, self.goal_quat)):
            raise ExecutionError("Release pose callback changed the reviewed goal XY/orientation")
        binding = np.r_[expected, expected_q]
        if self._release_pose_binding is not None:
            if (not np.allclose(binding[:3], self._release_pose_binding[:3], rtol=0, atol=1e-9)
                    or not self._same_quaternion(binding[3:], self._release_pose_binding[3:])):
                raise ExecutionError("Expected release pose changed during hold")
        self._release_pose_binding = binding.copy()
        rotation = self._rotation_wxyz(measured_q)
        relative = self._rotation_wxyz(expected_q).T@rotation
        errors = dict(xy_m=float(np.linalg.norm(measured[:2]-expected[:2])), z_m=abs(float(measured[2]-expected[2])),
            world_tilt_rad=math.atan2(float(np.linalg.norm(rotation[:2, 2])), float(rotation[2, 2])),
            yaw_rad=abs(math.atan2(float(relative[1, 0]), float(relative[0, 0]))))
        thresholds = dict(xy_m=.00075, z_m=.004, world_tilt_rad=.001, yaw_rad=.003)
        if result["thresholds"] != thresholds:
            raise ExecutionError("Release pose callback thresholds differ")
        gates = {name: errors[key] <= thresholds[key] for name, key in
                 (("xy", "xy_m"), ("z", "z_m"), ("world_tilt", "world_tilt_rad"), ("yaw", "yaw_rad"))}
        if (set(result["errors"]) != set(errors) or any(not math.isclose(float(result["errors"][k]), v, rel_tol=1e-8, abs_tol=1e-9) for k,v in errors.items())
                or set(result["gates"]) != set(gates) or any(type(result["gates"][k]) is not bool or result["gates"][k] != v for k,v in gates.items())
                or result["passed"] != all(gates.values())):
            raise ExecutionError("Release pose callback geometry/gates disagree with actual pose")
        self._prerelease_pose_last = result
        return result

    def _hold_prerelease(self, sim_time, q, box):
        self._prepare_payload_compensation(sim_time, q, box)
        self._hold(self._prerelease_hold_q)
        if self._payload_current_receipt is not None:
            self._payload_current_receipt["command_issued"] = True

    def _open_after_prerelease(self, sim_time):
        self.surface.open_gripper(self.gripper)
        if self.prerelease_payload_compensation is not None:
            previous = self._payload_efforts.copy()
            self._payload_efforts = np.zeros(6)
            self._payload_native_cache = None
            self._payload_ramp_fraction = 0.
            self._hold(self._prerelease_hold_q)
            self._payload_cleared_at_open = dict(sim_time_s=float(sim_time),
                payload_effort_before_open_nm=previous.tolist(), applied_payload_nm=[0.]*6,
                native_gravity_compensation_nm=self.last_gravity_efforts.tolist(),
                combined_feedforward_nm=self.last_combined_feedforward.tolist(),
                drive_headroom_nm=(self.max_efforts-np.abs(self.last_combined_feedforward)).tolist(),
                command_issued=True, cleared_before_next_physics_step=True)
        self._transition("RELEASING", sim_time)

    def _payload_compensation_evidence(self):
        return dict(schema="depallet.prerelease_payload_compensation.v1",
            enabled=self.prerelease_payload_compensation is not None, ramp_seconds=.5,
            measurement_source=self.measurement_source, samples=list(self._payload_samples),
            all_observed_samples_retained=True, first_full_ramp_command_at_sim_time_s=self._payload_full_ramp_at,
            cleared_at_open=self._payload_cleared_at_open, currently_applied_payload_nm=self._payload_efforts.tolist(),
            native_gravity_api_external_D6_payload_included=False,
            analytic_load_uses_assumed_mass=True, hardware_compensation_calibrated=False)

    def _observe_measured_stability(self, sim_time, q, v, box, kind):
        # Separate state/receipts, shared measured thresholds. API velocities do
        # not substitute for finite differences of actual articulation/TCP poses.
        if kind not in ("inspection", "preclose", "prerelease"):
            raise ExecutionError("Unknown measured stabilization phase")
        raw = self.measure_tool_pose()
        if not isinstance(raw, dict):
            raise ExecutionError("Measured tool pose callback must return a mapping")
        position = vector(raw["position_m"], 3, "tool position")
        quaternion = vector(raw["quaternion_wxyz"], 4, "tool quaternion")
        norm = float(np.linalg.norm(quaternion))
        if abs(norm-1.) > 1e-4:
            raise ExecutionError("Measured tool quaternion must be normalized")
        quaternion = quaternion/norm
        previous = getattr(self, "_"+kind+"_previous")
        sample_dt = None if previous is None else float(sim_time-previous["sim_time_s"])
        joint_fd = tool_linear_fd = tool_angular_fd = None
        if previous is not None:
            if sample_dt <= 0 or sample_dt > .05+1e-8:
                raise ExecutionError("Measured stabilization clock must advance within 50 ms")
            joint_fd = float(np.max(np.abs(q-previous["q"]))/sample_dt)
            tool_linear_fd = float(np.linalg.norm(position-previous["position"])/sample_dt)
            dot = min(1., max(0., abs(float(quaternion@previous["quaternion"]))))
            tool_angular_fd = 2*math.acos(dot)/sample_dt
        joint_api = float(np.max(np.abs(v)))
        box_linear = float(np.linalg.norm(box["linear_velocity_m_s"]))
        box_angular = float(np.linalg.norm(box["angular_velocity_rad_s"]))
        limits = STILLNESS_THRESHOLDS
        gates = dict(rate_baseline_available=previous is not None,
            joint_api=joint_api <= limits["joint_api_speed_rad_s"],
            joint_fd=joint_fd is not None and joint_fd <= limits["joint_fd_speed_rad_s"],
            tool_translation_fd=tool_linear_fd is not None and tool_linear_fd <= limits["tool_translation_fd_speed_m_s"],
            tool_orientation_fd=tool_angular_fd is not None and tool_angular_fd <= limits["tool_orientation_fd_speed_rad_s"],
            box_linear=box_linear <= limits["box_speed_m_s"],
            box_angular=box_angular <= limits["box_angular_speed_rad_s"])
        endpoint_error = None
        if kind == "prerelease":
            endpoint_error = float(np.max(np.abs(q-self._prerelease_hold_q)))
            gates["trajectory_endpoint_position"] = endpoint_error <= .02
            # step() checks attachment immediately before this observation.
            gates["target_attached"] = self.last_evidence.get("target_attached") is True
            if self.prerelease_payload_compensation is not None:
                gates["payload_compensation_ready"] = bool(self._payload_current_receipt and self._payload_current_receipt["ready"])
            if self.prerelease_pose_check is not None:
                gates["release_pose_check"] = self._checked_release_pose(q, box)["passed"]
        stable = all(gates.values())
        stable_since = getattr(self, "_"+kind+"_stable_since")
        if stable:
            if stable_since is None:
                stable_since = float(sim_time)
            stable_seconds = float(sim_time-stable_since)
        else:
            stable_since, stable_seconds = None, 0.
        setattr(self, "_"+kind+"_stable_since", stable_since)
        setattr(self, kind+"_stable_seconds", stable_seconds)
        setattr(self, "_"+kind+"_previous", dict(sim_time_s=float(sim_time), q=q.copy(),
            position=position.copy(), quaternion=quaternion.copy()))
        sample = dict(sim_time_s=float(sim_time), sample_dt_s=sample_dt,
            joint_api_speed_rad_s=joint_api, joint_fd_speed_rad_s=joint_fd,
            tool_translation_fd_speed_m_s=tool_linear_fd, tool_orientation_fd_speed_rad_s=tool_angular_fd,
            box_speed_m_s=box_linear, box_angular_speed_rad_s=box_angular,
            joints_rad=q.tolist(), joint_velocities_rad_s=v.tolist(),
            tool_position_m=position.tolist(), tool_quaternion_wxyz=quaternion.tolist(),
            box_position_m=box["position_m"].tolist(), box_quaternion_wxyz=box["quaternion_wxyz"].tolist(),
            box_linear_velocity_m_s=box["linear_velocity_m_s"].tolist(),
            box_angular_velocity_rad_s=box["angular_velocity_rad_s"].tolist(),
            gates={k: bool(value) for k, value in gates.items()}, stable=bool(stable),
            consecutive_stable_seconds=stable_seconds)
        if kind == "prerelease":
            sample.update(target_attached=bool(self.last_evidence.get("target_attached")),
                          trajectory_endpoint_error_rad=endpoint_error)
            if self.prerelease_payload_compensation is not None:
                sample["payload_compensation"] = self._payload_current_receipt
            if self.prerelease_pose_check is not None:
                sample["release_pose_check"] = self._prerelease_pose_last
        samples = getattr(self, kind+"_stability_samples")
        samples.append(sample)
        if kind in ("inspection", "preclose"):
            samples[:] = samples[-60:]
        return stable and stable_seconds >= limits["required_stable_seconds"]-1e-9

    def _observe_preclose_stability(self, sim_time, q, v, box):
        return self._observe_measured_stability(sim_time, q, v, box, "preclose")

    def _inspection_evidence(self):
        return dict(schema="depallet.inspection_stability.v1",
            enabled=self._inspection_boundary_index is not None,
            trajectory_boundary_index=self._inspection_boundary_index,
            trajectory_cut_time_s=self._inspection_cut_time,
            passed=self._inspection_passed_at is not None,
            passed_at_sim_time_s=self._inspection_passed_at,
            resumed_at_sim_time_s=self._inspection_resumed_at,
            consecutive_stable_seconds=self.inspection_stable_seconds,
            measurement_source=self.measurement_source, tool_pose_source="measure_tool_pose_callback",
            samples=list(self.inspection_stability_samples), samples_retained=len(self.inspection_stability_samples),
            thresholds=dict(STILLNESS_THRESHOLDS, minimum_camera_hold_seconds=.5, acknowledgement_timeout_seconds=3.),

            joint_or_object_teleport_used=False, hardware_execution_certified=False)
    def _preclose_evidence(self):
        return dict(schema="depallet.preclose_stability.v1", enabled=self.preclose_stability,
            passed=self._preclose_passed_at is not None, passed_at_sim_time_s=self._preclose_passed_at,
            consecutive_stable_seconds=self.preclose_stable_seconds,
            measurement_source=self.measurement_source, tool_pose_source="measure_tool_pose_callback",
            samples=list(self.preclose_stability_samples), samples_retained=len(self.preclose_stability_samples),
            first_sample_can_pass=False, timestamp_source="actual_step_sim_time",
            thresholds=dict(STILLNESS_THRESHOLDS),
            continuous_motion_guarantee=False, hardware_execution_certified=False)

    def _prerelease_evidence(self):
        schema = ("depallet.prerelease_stability.v2" if self.prerelease_payload_compensation is not None or self.prerelease_pose_check is not None
                  else "depallet.prerelease_stability.v1")
        return dict(schema=schema, enabled=self.prerelease_stability,
            passed=self._prerelease_passed_at is not None, passed_at_sim_time_s=self._prerelease_passed_at,
            started_at_sim_time_s=self._prerelease_started_at,
            consecutive_stable_seconds=self.prerelease_stable_seconds,
            measurement_source=self.measurement_source, tool_pose_source="measure_tool_pose_callback",
            target_box_id=self.box_id, target_box_path=self.box_path,
            hold_joint_positions_rad=None if self._prerelease_hold_q is None else self._prerelease_hold_q.tolist(),
            samples=list(self.prerelease_stability_samples), samples_retained=len(self.prerelease_stability_samples),
            all_observed_samples_retained=True, first_sample_can_pass=False, timestamp_source="actual_step_sim_time",
            thresholds=dict(STILLNESS_THRESHOLDS, trajectory_endpoint_error_rad=.02),
            continuous_motion_guarantee=False, hardware_execution_certified=False)

    def _attachment_evidence(self):
        status = self.surface.get_gripper_status(self.gripper)
        gripped = [str(p) for p in self.surface.get_gripped_objects(self.gripper)]
        active = []
        for path in self.joint_paths:
            prim = self.stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                raise ExecutionError("Configured attachment joint disappeared")
            enabled = prim.GetAttribute("physics:jointEnabled").Get()
            if enabled:
                for target in prim.GetRelationship("physics:body1").GetTargets():
                    active.append(dict(joint=path, body1=str(target)))
        wrong = [p for p in gripped if not matches(p, self.box_path)]
        wrong.extend(j["body1"] for j in active if not matches(j["body1"], self.box_path))
        if wrong:
            raise ExecutionError("Gripper attached an unexpected object: "+repr(wrong))
        target_reported = any(matches(p, self.box_path) for p in gripped)
        return dict(status=str(status), status_closed=status == self.closed_status, status_open=status == self.open_status,
                    gripped_objects=gripped, active_D6_body1_targets=active,
                    target_attached=bool(status == self.closed_status and target_reported),
                    detached=bool(status == self.open_status and not gripped and not active))

    def step(self, sim_time):
        if self.state in ("IDLE", "FAILED"):
            raise ExecutionError("step requires an active single-box execution")
        try:
            if not math.isfinite(sim_time) or sim_time <= self.last_time or sim_time-self.last_time > .05+1e-8:
                raise ExecutionError("Physics clock must advance once per bounded physics step")
            dt, self.last_time = sim_time-self.last_time, float(sim_time)
            q, v = self._joint_state()
            previous_q = getattr(self, '_diagnostic_previous_q', None)
            self._diagnostic_joint_fd = None if previous_q is None else (q-previous_q)/dt
            self._diagnostic_previous_q = q.copy()
            self.last_joint_velocities = v.copy()
            if np.any(q < self.limits[:, 0]-.01) or np.any(q > self.limits[:, 1]+.01):
                raise ExecutionError("Measured joints exceed limits")
            if np.any(np.abs(v) > 1.5*self.max_velocities+.05):
                raise ExecutionError("Measured joint velocity exceeds execution limit")
            error = float(np.max(np.abs(q-self.commanded_q)))
            self.peak_tracking_error = max(self.peak_tracking_error, error)
            if error > self.tracking_limit:
                raise ExecutionError("Measured drive tracking error exceeds bound")
            box = self._box_state()
            self.last_box = {k: value.tolist() for k, value in box.items()}
            self.max_box_z = max(self.max_box_z, float(box["position_m"][2]))
            evidence = self._attachment_evidence()
            self.last_evidence, self.samples = evidence, self.samples+1
            if self.state in ("GRASP_CONFIRMED", "ESCAPE", "ESCAPED", "TRANSPORT", "PRE_RELEASE_SETTLE") and not evidence["target_attached"]:
                raise ExecutionError("Target attachment lost before release")
            if self.state in ("INSPECTION_MOVE", "INSPECTION_SETTLE", "INSPECTION_HOLD") and not evidence["detached"]:
                raise ExecutionError("Inspection requires an open unloaded gripper")
            if self.release_confirmed and not evidence["detached"]:
                raise ExecutionError("Unexpected attachment after release")
            if self.state in ("ESCAPE", "TRANSPORT") and box["position_m"][2] >= self.initial_box["position_m"][2]+.05:
                self.lift_confirmed = True
            if self.state in ("INSPECTION_MOVE", "APPROACH", "ESCAPE", "TRANSPORT", "RETREAT"):
                elapsed = sim_time-self.trajectory_since
                inspection_endpoint = self.state == "INSPECTION_MOVE" and elapsed >= self._inspection_cut_time
                sample_time = min(elapsed, self._inspection_cut_time) if self.state == "INSPECTION_MOVE" else elapsed
                target_q, target_v, ended = self.trajectory.sample(sample_time)
                self._command(target_q, target_v)
                at_end = (inspection_endpoint or ended) and np.max(np.abs(q-target_q)) <= .02 and np.max(np.abs(v)) <= .05
                self.endpoint_samples = self.endpoint_samples+1 if at_end else 0
                deadline = self._inspection_cut_time if self.state == "INSPECTION_MOVE" else self.trajectory.times[-1]
                if (inspection_endpoint or ended) and elapsed > deadline+3.:
                    if self.state == "INSPECTION_MOVE":
                        raise ExecutionError("Robot did not settle at inspection endpoint")
                    raise ExecutionError("Robot did not settle at trajectory endpoint")
                if self.endpoint_samples >= 3:
                    if self.state == "INSPECTION_MOVE":
                        self._hold(self._inspection_hold_q)
                        self._transition("INSPECTION_SETTLE", sim_time)
                        self._inspection_previous = None
                        self._inspection_stable_since = None
                        self.inspection_stable_seconds = 0.
                        self.inspection_stability_samples = []
                        self._observe_measured_stability(sim_time, q, v, box, "inspection")
                    elif self.state == "APPROACH":
                        if self.preclose_stability:
                            self._transition("PREGRASP_SETTLE", sim_time)
                            self._preclose_previous = None
                            self._preclose_stable_since = None
                            self.preclose_stable_seconds = 0.
                            self._observe_preclose_stability(sim_time, q, v, box)
                        else:
                            self._prepare_contact_or_close(sim_time)
                    elif self.state == "ESCAPE":
                        if not self.lift_confirmed:
                            raise ExecutionError("Contact escape failed measured 5 cm target lift")
                        self.escape_confirmed = True
                        self._transition("ESCAPED", sim_time)
                    elif self.state == "TRANSPORT":
                        if not self.lift_confirmed:
                            raise ExecutionError("No measured 5 cm target lift during transport")
                        if self.prerelease_stability:
                            self._transition("PRE_RELEASE_SETTLE", sim_time)
                            self._prerelease_started_at = float(sim_time)
                            self._prerelease_hold_q = target_q.copy()
                            self._hold_prerelease(sim_time, q, box)
                            self._observe_measured_stability(sim_time, q, v, box, "prerelease")
                        else:
                            self.surface.open_gripper(self.gripper)
                            self._transition("RELEASING", sim_time)
                    else:
                        self._transition("SETTLING", sim_time)
            elif self.state == "INSPECTION_SETTLE":
                self._hold(self._inspection_hold_q)
                ready = self._observe_measured_stability(sim_time, q, v, box, "inspection")
                if sim_time-self.state_since > STILLNESS_THRESHOLDS["timeout_s"]+1e-9:
                    raise ExecutionError("Target/tool failed inspection stabilization within 4 seconds")
                if ready:
                    self._inspection_passed_at = float(sim_time)
                    self._transition("INSPECTION_HOLD", sim_time)
            elif self.state == "INSPECTION_HOLD":
                self._hold(self._inspection_hold_q)
                ready = self._observe_measured_stability(sim_time, q, v, box, "inspection")
                if not ready:
                    raise ExecutionError("Target/tool lost stability during inspection hold")
                if sim_time-self.state_since > 3.+1e-9:
                    raise ExecutionError("Inspection hold acknowledgement timed out after 3 seconds")
            elif self.state == "PREGRASP_SETTLE":
                self._hold()
                ready = self._observe_preclose_stability(sim_time, q, v, box)
                if sim_time-self.state_since > 4.+1e-9:
                    raise ExecutionError("Target/tool failed pre-close stabilization within 4 seconds")
                if ready:
                    self._preclose_passed_at = float(sim_time)
                    self._prepare_contact_or_close(sim_time)
            elif self.state == "PRE_RELEASE_SETTLE":
                self._hold_prerelease(sim_time, q, box)
                ready = self._observe_measured_stability(sim_time, q, v, box, "prerelease")
                if sim_time-self.state_since > STILLNESS_THRESHOLDS["timeout_s"]+1e-9:
                    raise ExecutionError("Target/tool failed pre-release stabilization within 4 seconds")
                if ready:
                    self._prerelease_passed_at = float(sim_time)
                    self._open_after_prerelease(sim_time)
            elif self.state == "CONTACT_READY":
                self._hold()
                if self.pre_grasp_check is None:
                    raise ExecutionError("Contact preparation requires a fresh patch check")
                self.pre_grasp_evidence=self.pre_grasp_check()
                if (not isinstance(self.pre_grasp_evidence,dict) or self.pre_grasp_evidence.get("passed") is not True
                        or self.pre_grasp_evidence.get("requires_physics_update_before_close")):
                    raise ExecutionError("Prepared contact pad patch was not verified after physics update")
                self.surface.close_gripper(self.gripper)
                self._transition("CLOSING",sim_time)
            elif self.state == "CLOSING":
                self._hold()
                self.attachment_samples = self.attachment_samples+1 if evidence["target_attached"] else 0
                if self.attachment_samples >= 3:
                    self.grasp_confirmed = True
                    self._transition("GRASP_CONFIRMED", sim_time)
                elif sim_time-self.state_since > self.grasp_timeout:
                    raise ExecutionError("Actual target attachment not confirmed before grasp timeout")
            elif self.state in ("GRASP_CONFIRMED", "ESCAPED"):
                self._hold()
                if self.state == "GRASP_CONFIRMED" and self.contact_escape:
                    stable = (np.max(np.abs(v)) <= .01
                              and np.linalg.norm(box["linear_velocity_m_s"]) <= .02
                              and np.linalg.norm(box["angular_velocity_rad_s"]) <= .05)
                    self.grasp_stability_samples.append(dict(sim_time_s=sim_time,joint_speed_rad_s=float(np.max(np.abs(v))),
                        joint_fd_speed_rad_s=None if self._diagnostic_joint_fd is None else float(np.max(np.abs(self._diagnostic_joint_fd))),
                        sample_dt_s=dt,joints_rad=q.tolist(),joint_velocities_rad_s=v.tolist(),
                        box_speed_m_s=float(np.linalg.norm(box["linear_velocity_m_s"])),
                        box_angular_speed_rad_s=float(np.linalg.norm(box["angular_velocity_rad_s"])),stable=bool(stable)))
                    self.grasp_stability_samples = self.grasp_stability_samples[-60:]
                    self.grasp_stable_seconds = self.grasp_stable_seconds+dt if stable else 0.
                    if sim_time-self.state_since > 4.:
                        raise ExecutionError("Attached target failed pre-escape stabilization")
            elif self.state == "RELEASING":
                self._hold()
                if evidence["detached"]:
                    self.release_confirmed = True
                    self._transition("RELEASED", sim_time)
                elif sim_time-self.state_since > self.release_timeout:
                    raise ExecutionError("Gripper did not physically release before timeout")
            elif self.state in ("RELEASED", "SETTLING", "DONE"):
                self._hold()
                if not evidence["detached"]:
                    raise ExecutionError("Unexpected attachment after release")
                qdot = min(1., abs(float(np.dot(box["quaternion_wxyz"], self.goal_quat))))
                orientation_error = 2*math.acos(qdot)
                stable = (np.linalg.norm(box["position_m"][:2]-self.goal[:2]) <= .03
                          and abs(box["position_m"][2]-self.goal[2]) <= .02 and orientation_error <= .1
                          and np.linalg.norm(box["linear_velocity_m_s"]) <= .03
                          and np.linalg.norm(box["angular_velocity_rad_s"]) <= .3)
                if self.state == "DONE" and not stable:
                    raise ExecutionError("Box left its settled goal during completion recording")
                self.settled_seconds = self.settled_seconds+dt if stable else 0.
                if self.state != "DONE" and self.settled_seconds >= .5-1e-9:
                    self._transition("DONE", sim_time)
                elif self.state != "DONE" and sim_time-self.state_since > self.settle_timeout:
                    raise ExecutionError("Released box failed goal pose/velocity settling checks")
            return self.result()
        except Exception as exc:
            self._abort(exc)

    def abort(self, reason):
        """Caller can stop on external contact/scene checks; hold measured joints."""
        self._abort(reason)

    def result(self):
        passed = (self.state == "DONE" and self.grasp_confirmed and self.lift_confirmed and self.release_confirmed
                  and (self._inspection_boundary_index is None or self._inspection_resumed_at is not None)
                  and (not self.prerelease_stability or self._prerelease_passed_at is not None)
                  and (self.prerelease_payload_compensation is None or self._payload_cleared_at_open is not None)
                  and (self.prerelease_pose_check is None or bool(self._prerelease_pose_last and self._prerelease_pose_last["passed"])))
        return dict(state=self.state, passed=passed, failure=self.failure, sample_count=self.samples,
                    requires_transport_plan=(self.state == "ESCAPED" if self.contact_escape else self.state == "GRASP_CONFIRMED"),
                    requires_contact_escape=(self.contact_escape and self.state == "GRASP_CONFIRMED"
                                             and self.grasp_stable_seconds >= .25-1e-9),
                    contact_escape_enabled=self.contact_escape, contact_escape_confirmed=self.escape_confirmed,
                    pre_grasp_patch_evidence=self.pre_grasp_evidence,
                    inspection_stability_evidence=self._inspection_evidence(),
                    preclose_stability_enabled=self.preclose_stability,
                    preclose_stability_evidence=self._preclose_evidence(),
                    prerelease_stability_enabled=self.prerelease_stability,
                    prerelease_stability_evidence=self._prerelease_evidence(),
                    prerelease_payload_compensation_enabled=self.prerelease_payload_compensation is not None,
                    prerelease_payload_compensation_evidence=self._payload_compensation_evidence(),
                    prerelease_pose_check_enabled=self.prerelease_pose_check is not None,
                    last_combined_feedforward_nm=self.last_combined_feedforward.tolist(),
                    peak_combined_feedforward_nm=self.peak_combined_feedforward.tolist(),
                    grasp_stable_seconds=self.grasp_stable_seconds, release_confirmed=self.release_confirmed,
                    last_joint_velocities_rad_s=self.last_joint_velocities.tolist(),grasp_stability_samples=self.grasp_stability_samples,
                    attachment_confirmed=self.grasp_confirmed, measured_lift_confirmed=self.lift_confirmed,
                    max_box_height_m=self.max_box_z if math.isfinite(self.max_box_z) else None,
                    settled_seconds=self.settled_seconds, peak_tracking_error_rad=self.peak_tracking_error,
                    measurement_source=self.measurement_source, physics_grasp_validated=passed and self.measurement_source == "isaac_runtime",
                    target_box_path=self.box_path, goal_position_m=self.goal.tolist(),
                    last_gripper_evidence=self.last_evidence, last_box_state=self.last_box,
                    trajectories_sha256=list(self.trajectory_hashes), events=list(self.events),
                    gravity_compensation_enabled=self.gravity_compensation, gravity_compensation_samples=self.gravity_samples,
                    last_gravity_compensation_nm=self.last_gravity_efforts.tolist(), peak_gravity_compensation_nm=self.peak_gravity_efforts.tolist(),
                    gravity_compensation_source="PhysX get_dof_gravity_compensation_forces" if self.gravity_compensation else None,
                    gravity_compensation_external_D6_payload_included=False, feedforward_torque_limits_checked=self.gravity_compensation,
                    combined_command_effort_bound_enabled=self.gravity_compensation, physical_joint_force_limit_certified=False,
                    joint_or_object_teleport_used=False, scope="single-box only", valid_sim_hours=0.)
