"""Independent CPU forward kinematics from URDF joint transforms.

This implementation does not import cuRobo or USD. It is used to check the
planner's terminal FK against the same reviewed URDF, not to drive a robot.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation


def matrix(urdf, joint_names, joint_values, tip="suction_tcp", base="base_link"):
    if len(joint_names) != len(joint_values) or len(set(joint_names)) != len(joint_names):
        raise ValueError("Invalid joint configuration")
    positions = dict(zip(joint_names,joint_values))
    if not all(math.isfinite(v) for v in positions.values()):
        raise ValueError("Joint values must be finite")
    tree = ET.parse(urdf).getroot()
    by_child = {j.find("child").get("link"):j for j in tree.findall("joint")}
    chain = []
    link = tip
    while link != base:
        if link not in by_child or len(chain) > len(by_child):
            raise ValueError("No acyclic chain from base to tip")
        joint = by_child[link]
        chain.append(joint)
        link = joint.find("parent").get("link")
    transform = np.eye(4)
    for joint in reversed(chain):
        origin = joint.find("origin")
        xyz = [0.,0.,0.] if origin is None else [float(v) for v in origin.get("xyz","0 0 0").split()]
        rpy = [0.,0.,0.] if origin is None else [float(v) for v in origin.get("rpy","0 0 0").split()]
        step = np.eye(4)
        step[:3,:3] = Rotation.from_euler("xyz",rpy).as_matrix()
        step[:3,3] = xyz
        transform = transform @ step
        kind = joint.get("type")
        if kind == "fixed":
            continue
        if kind not in ("revolute","continuous"):
            raise ValueError(f"Unsupported joint type {kind}")
        axis_tag = joint.find("axis")
        axis = np.asarray([float(v) for v in (axis_tag.get("xyz") if axis_tag is not None else "1 0 0").split()])
        length = np.linalg.norm(axis)
        if length < 1e-12:
            raise ValueError("Zero joint axis")
        step = np.eye(4)
        step[:3,:3] = Rotation.from_rotvec(axis/length*positions[joint.get("name")]).as_matrix()
        transform = transform @ step
    return transform


def pose(urdf, joint_names, joint_values, tip="suction_tcp", base="base_link"):
    transform = matrix(urdf,joint_names,joint_values,tip,base)
    xyzw = Rotation.from_matrix(transform[:3,:3]).as_quat()
    return transform[:3,3].tolist(), [float(xyzw[3]), *xyzw[:3].tolist()]
