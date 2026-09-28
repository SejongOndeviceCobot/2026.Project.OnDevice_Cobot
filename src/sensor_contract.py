"""Versioned policy camera profiles and CPU-only rigid pose validation.

World camera poses are capture-time evidence, never policy state. A wrist
calibration stores only its fixed flange-relative transform.
"""
from __future__ import annotations

import math

LEGACY_PROFILE = 'overview_source_v1'
CAMERA_PROFILE_V2 = 'overhead_wrist_v2'


def camera_profile(name=LEGACY_PROFILE):
    if name == LEGACY_PROFILE:
        return {'camera_ids': ['overview', 'source'], 'camera_hz': 10,
                'resolution_hw': [240, 320], 'max_camera_age_s': .15,
                'camera_roots': {'overview': '.', 'source': 'source_camera'}}
    if name == CAMERA_PROFILE_V2:
        return {'camera_ids': ['overhead', 'wrist'], 'camera_hz': 30,
                'resolution_hw': [480, 640], 'max_camera_age_s': .10,
                'camera_roots': {'overhead': '.', 'wrist': 'wrist_camera'}}
    raise ValueError('Unknown camera profile')


def contract_profile(contract):
    """Preserve unnamed legacy readers; a named new profile must be exact."""
    name = contract.get('camera_profile', LEGACY_PROFILE)
    profile = camera_profile(name)
    if name == CAMERA_PROFILE_V2:
        if any(contract.get(key) != value for key, value in profile.items()):
            raise ValueError('Camera profile fields do not match overhead_wrist_v2')
        if contract.get('observer_excluded_from_policy') is not True:
            raise ValueError('Observer must be excluded from policy cameras')
    return name, profile


def rigid_transform(value, name='transform'):
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f'{name} must be a rigid 4x4 matrix')
    result = []
    for row in value:
        if not isinstance(row, (list, tuple)) or len(row) != 4:
            raise ValueError(f'{name} must be a rigid 4x4 matrix')
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in row):
            raise ValueError(f'{name} must contain finite numbers')
        result.append([float(x) for x in row])
    if any(abs(result[3][j] - (1. if j == 3 else 0.)) > 1e-6 for j in range(4)):
        raise ValueError(f'{name} must have homogeneous last row')
    r = [row[:3] for row in result[:3]]
    if any(abs(sum(r[k][i] * r[k][j] for k in range(3)) - float(i == j)) > 1e-5
           for i in range(3) for j in range(3)):
        raise ValueError(f'{name} rotation is not orthonormal')
    determinant = (r[0][0] * (r[1][1]*r[2][2] - r[1][2]*r[2][1])
                   - r[0][1] * (r[1][0]*r[2][2] - r[1][2]*r[2][0])
                   + r[0][2] * (r[1][0]*r[2][1] - r[1][1]*r[2][0]))
    if abs(determinant - 1.) > 1e-5:
        raise ValueError(f'{name} rotation must be proper (determinant +1)')
    return result


def camera_pose(value, image_sim_time):
    if not isinstance(value, dict):
        raise ValueError('camera_pose is required for a wrist capture')
    matrix = rigid_transform(value.get('T_world_camera_cv'), 'camera_pose.T_world_camera_cv')
    timestamp = value.get('pose_sim_time')
    if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp) or timestamp < 0
            or abs(timestamp - image_sim_time) > 1e-6):
        raise ValueError('camera_pose pose_sim_time must match image capture time')
    if not isinstance(value.get('pose_source'), str) or not value['pose_source'].strip():
        raise ValueError('camera_pose pose_source must be nonempty')
    return {**value, 'T_world_camera_cv': matrix, 'pose_sim_time': float(timestamp)}


def pose_matrix(pose):
    if (not isinstance(pose, (list, tuple)) or len(pose) != 7
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in pose)):
        raise ValueError('flange_pose_wxyz must contain seven finite numbers')
    px, py, pz, w, x, y, z = pose
    norm = w*w + x*x + y*y + z*z
    if abs(norm - 1.) > 1e-4:
        raise ValueError('flange_pose_wxyz quaternion is not normalized')
    scale = 2. / norm
    return [[1-scale*(y*y+z*z), scale*(x*y-z*w), scale*(x*z+y*w), px],
            [scale*(x*y+z*w), 1-scale*(x*x+z*z), scale*(y*z-x*w), py],
            [scale*(x*z-y*w), scale*(y*z+x*w), 1-scale*(x*x+y*y), pz],
            [0., 0., 0., 1.]]


def compose(a, b):
    return [[sum(a[i][k]*b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def transforms_match(a, b, tolerance=1e-4):
    return all(abs(a[i][j]-b[i][j]) <= tolerance for i in range(4) for j in range(4))
