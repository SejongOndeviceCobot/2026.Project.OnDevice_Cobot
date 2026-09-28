"""CPU-only instantaneous gravity candidate for a separately attached payload.

This module never commands a robot. The caller must prove that the native
articulation gravity term excludes the separately attached body, retain actual
attachment/contact checks, and bound native + payload feedforward + drive effort.
Mass/COM are explicit simulation assumptions, not measured real vacuum loads.
"""
from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from depallet.motion.contact_escape import RobotModel


JOINT_NAMES = [f"joint_{i}" for i in range(1, 7)]
MAX_FK_POSITION_ERROR_M = .0001
MAX_FK_ORIENTATION_ERROR_RAD = .0001
MAX_CANDIDATE_MOTOR_FRACTION = .15


def _vector(value, size, label):
    raw = np.asarray(value)
    if raw.shape != (size,) or raw.dtype.kind not in "iuf" or not np.isfinite(raw).all():
        raise ValueError(f"{label} requires {size} finite real numbers")
    return raw.astype(float)


def _pose(value, label, *, quaternion_tolerance=1e-4):
    values = _vector(value, 7, label)
    norm = float(np.linalg.norm(values[3:]))
    if abs(norm-1.) > quaternion_tolerance:
        raise ValueError(label+" quaternion must be near unit")
    values[3:] /= norm
    matrix = np.eye(4)
    matrix[:3, 3] = values[:3]
    matrix[:3, :3] = Rotation.from_quat(values[[4, 5, 6, 3]]).as_matrix()
    return matrix, values


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PayloadGravityEvaluator:
    """Parse one reviewed URDF and evaluate -Jv.T m g from measured poses.

    ``box_state`` requires ``position_m`` and ``quaternion_wxyz``. TCP and box
    must be simultaneous actual world measurements belonging to ``q_rad``;
    clock, body identity and attachment status remain caller-enforced contracts.
    The world base is fixed at construction. Pose quaternions are normalized
    only within the explicit tolerance (box 1e-3; base/TCP 1e-4).
    """

    def __init__(self, robot_config, base_pose_world_wxyz, mass_kg, dimensions_m,
                 center_of_mass_local_m, max_efforts_nm, *,
                 include_articulation_gravity=False,
                 gravity_world_m_s2=(0., 0., -9.81)):
        if isinstance(mass_kg, bool) or not isinstance(mass_kg, (int, float)) or not math.isfinite(mass_kg) or not 0 < mass_kg <= 5.:
            raise ValueError("Assumed payload mass must be positive and at most 5 kg")
        if type(include_articulation_gravity) is not bool:
            raise ValueError("include_articulation_gravity must be boolean")
        self.mass = float(mass_kg)
        self.dimensions = _vector(dimensions_m, 3, "payload dimensions")
        if np.any(self.dimensions < .01) or np.any(self.dimensions > 2.):
            raise ValueError("Payload dimensions must lie within 0.01..2 m")
        self.com = _vector(center_of_mass_local_m, 3, "payload COM")
        if np.any(np.abs(self.com) >= self.dimensions/2):
            raise ValueError("Payload COM must be strictly inside the declared cuboid")
        self.base, self.base_pose = _pose(base_pose_world_wxyz, "fixed world base")
        self.gravity = _vector(gravity_world_m_s2, 3, "world gravity")
        if not 1. <= np.linalg.norm(self.gravity) <= 20.:
            raise ValueError("World gravity magnitude must lie within 1..20 m/s^2")
        self.max_efforts = _vector(max_efforts_nm, 6, "motor effort limits")
        if np.any(self.max_efforts <= 0):
            raise ValueError("Motor effort limits must be positive")
        self.include_articulation_gravity = include_articulation_gravity
        config_path = Path(robot_config).resolve(strict=True)
        config_sha = _sha(config_path)
        raw = yaml.safe_load(config_path.read_text())
        kin = raw.get("robot_cfg", raw)["kinematics"]
        if kin["base_link"] != "base_link" or kin["cspace"]["joint_names"] != JOINT_NAMES:
            raise ValueError("Exact H2017 base_link and joint_1..joint_6 order required")
        urdf_path = Path(kin["urdf_path"]).resolve(strict=True)
        urdf_sha = _sha(urdf_path)
        self.model = RobotModel(kin, JOINT_NAMES)
        tree = ET.parse(urdf_path).getroot()
        by_joint = {joint.get("name"): joint for joint in tree.findall("joint")}
        urdf_efforts = np.array([float(by_joint[name].find("limit").get("effort")) for name in JOINT_NAMES])
        if not np.isfinite(urdf_efforts).all() or np.any(urdf_efforts <= 0) or np.any(self.max_efforts > urdf_efforts+1e-9):
            raise ValueError("Motor effort caps must not exceed reviewed URDF limits")
        self.link_masses = []
        for link in tree.findall("link"):
            mass_node = link.find("inertial/mass")
            if mass_node is None:
                continue
            mass = float(mass_node.get("value"))
            if not math.isfinite(mass) or mass < 0:
                raise ValueError("Invalid articulation link mass")
            if mass == 0:
                continue
            name = link.get("name")
            if name == "attached_object":
                raise ValueError("URDF already contains a massive attached_object; duplicate payload compensation risk")
            if name not in self.model.ancestors:
                raise ValueError("Positive link mass outside reviewed FK tree")
            origin = link.find("inertial/origin")
            local = _vector([float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()], 3, "link COM")
            self.link_masses.append((name, mass, local))
        if _sha(config_path) != config_sha or _sha(urdf_path) != urdf_sha:
            raise ValueError("Robot geometry changed during CPU model initialization")
        self.provenance = dict(schema="depallet.payload_gravity_model.v1",
            robot_config=str(config_path), robot_config_sha256=config_sha,
            urdf=str(urdf_path), urdf_sha256=urdf_sha,
            evaluator_sha256=_sha(__file__), independent_fk_sha256=_sha(Path(__file__).with_name("contact_escape.py")),
            joint_names=JOINT_NAMES[:], base_frame="base_link", tcp_frame="suction_tcp",
            base_pose_world_wxyz=self.base_pose.tolist(), mass_kg=self.mass,
            dimensions_m=self.dimensions.tolist(), center_of_mass_local_m=self.com.tolist(),
            gravity_world_m_s2=self.gravity.tolist(), max_efforts_nm=self.max_efforts.tolist(),
            maximum_candidate_motor_fraction=MAX_CANDIDATE_MOTOR_FRACTION,
            maximum_fk_position_error_m=MAX_FK_POSITION_ERROR_M,
            maximum_fk_orientation_error_rad=MAX_FK_ORIENTATION_ERROR_RAD,
            mass_and_com_source="explicit fixed simulation assumptions", physical_parameters_measured=False,
            real_payload_known=False, native_gravity_payload_exclusion_verified=False,
            include_articulation_gravity=include_articulation_gravity,
            articulation_mass_catalog=[dict(link=name, mass_kg=mass, center_of_mass_local_m=local.tolist()) for name, mass, local in self.link_masses])

    def _axes(self, transforms):
        axes, origins = {}, {}
        for parent, _, origin, index, axis in self.model.joints:
            if index is not None:
                world = self.base @ transforms[parent] @ origin
                axes[index] = world[:3, :3] @ axis
                origins[index] = world[:3, 3]
        return axes, origins

    def _jacobian(self, point, ancestors, axes, origins):
        jacobian = np.zeros((3, 6))
        for index in ancestors:
            jacobian[:, index] = np.cross(axes[index], point-origins[index])
        return jacobian

    def evaluate(self, q_rad, box_state, *, tcp_world_pose_wxyz):
        q = _vector(q_rad, 6, "actual joints")
        if np.any(q < self.model.lower) or np.any(q > self.model.upper):
            raise ValueError("Actual joints exceed reviewed URDF limits")
        if not isinstance(box_state, dict):
            raise ValueError("Measured box_state must be a mapping")
        box_values = list(box_state["position_m"])+list(box_state["quaternion_wxyz"])
        box, box_pose = _pose(box_values, "actual box", quaternion_tolerance=1e-3)
        tcp, tcp_pose = _pose(tcp_world_pose_wxyz, "actual TCP")
        transforms = self.model.transforms(q)
        fk_tcp = self.base @ transforms["suction_tcp"]
        pe = float(np.linalg.norm(fk_tcp[:3, 3]-tcp[:3, 3]))
        re = float(Rotation.from_matrix(fk_tcp[:3, :3].T @ tcp[:3, :3]).magnitude())
        if pe > MAX_FK_POSITION_ERROR_M or re > MAX_FK_ORIENTATION_ERROR_RAD:
            raise ValueError("Independent FK does not agree with actual TCP within 0.1 mm/0.1 mrad")
        measured_com = (box @ np.r_[self.com, 1.])[:3]
        tcp_to_com = np.linalg.inv(tcp) @ np.r_[measured_com, 1.]
        # Rebase the instantaneous measured COM onto the verified FK frame. This
        # local derivative does not presume a permanently rigid elastic joint.
        nominal_com = (fk_tcp @ tcp_to_com)[:3]
        axes, origins = self._axes(transforms)
        jacobian = self._jacobian(nominal_com, self.model.ancestors["suction_tcp"], axes, origins)
        external = jacobian.T @ (self.mass*self.gravity)
        candidate = -external
        fractions = np.abs(candidate)/self.max_efforts
        if not np.isfinite(candidate).all() or np.any(fractions > MAX_CANDIDATE_MOTOR_FRACTION):
            raise ValueError("Payload compensation exceeds bounded 15% motor effort domain")
        articulation = None
        if self.include_articulation_gravity:
            articulation = np.zeros(6)
            for name, mass, local in self.link_masses:
                point = (self.base @ transforms[name] @ np.r_[local, 1.])[:3]
                link_jacobian = self._jacobian(point, self.model.ancestors[name], axes, origins)
                articulation -= link_jacobian.T @ (mass*self.gravity)
            if not np.isfinite(articulation).all():
                raise ValueError("Nonfinite independent articulation gravity estimate")
        return dict(schema="depallet.payload_gravity.evaluation.v1",
            payload_compensation_nm=candidate.tolist(),
            optional_articulation_gravity_nm=None if articulation is None else articulation.tolist(),
            artifact_provenance=copy.deepcopy(self.provenance),
            sample=dict(actual_joint_positions_rad=q.tolist(), box_pose_world_wxyz=box_pose.tolist(),
                tcp_pose_world_wxyz=tcp_pose.tolist(), measured_com_world_m=measured_com.tolist(),
                instantaneous_com_in_tcp_m=tcp_to_com[:3].tolist(), nominal_fk_com_world_m=nominal_com.tolist(),
                jacobian_world_m_per_rad=jacobian.tolist(), fk_position_error_m=pe,
                fk_orientation_error_rad=re, external_gravity_generalized_force_nm=external.tolist(),
                candidate_motor_effort_fraction=fractions.tolist(),
                compensation_sign="-Jv_world.T @ (mass_kg * gravity_world_m_s2)",
                simultaneous_measurements_required=True, actual_attachment_required=True,
                native_gravity_exclusion_check_required=True,
                caller_must_bound_total_native_plus_payload_feedforward_and_drive=True,
                instantaneous_static_model_only=True, joint_effort_applied=False,
                inertia_contact_friction_or_elastic_dynamics_certified=False))
