"""CPU-only geometry/provenance boundary for the optional Point2Pose worker.

Visible-depth PCA defines a repeatable local tracking frame, not a measured
box centre or an unambiguous box orientation. No simulator object pose is read.
"""
from __future__ import annotations

import numpy as np


def checked_transform(value, name="transform"):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(result[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} is not a rigid transform")
    if not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
        raise ValueError(f"{name} has a reflection")
    return result


def visible_depth_anchor(depth_m, mask, intrinsics):
    depth = np.asarray(depth_m)
    mask = np.asarray(mask, dtype=bool)
    K = np.asarray(intrinsics, dtype=np.float64)
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ValueError("depth and mask must be matching HxW arrays")
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise ValueError("intrinsics must be finite 3x3")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
        raise ValueError("invalid pinhole intrinsics")
    yy, xx = np.nonzero(mask & np.isfinite(depth) & (depth > 0.1) & (depth < 10))
    if len(xx) < 64:
        raise ValueError("fewer than 64 valid masked depth samples")
    # A deterministic stride bounds CPU geometry work without clipping the asset.
    stride = max(1, len(xx) // 30000)
    yy, xx = yy[::stride], xx[::stride]
    z = depth[yy, xx].astype(np.float64)
    points = np.column_stack(((xx - K[0, 2]) * z / K[0, 0],
                              (yy - K[1, 2]) * z / K[1, 1], z))
    centre = np.median(points, axis=0)
    _, singular, vh = np.linalg.svd(points - centre, full_matrices=False)
    rotation = vh.T
    # Deterministic signs, with proper handedness. Symmetric axes remain ambiguous.
    for axis in range(2):
        major = np.argmax(np.abs(rotation[:, axis]))
        if rotation[major, axis] < 0:
            rotation[:, axis] *= -1
    rotation[:, 2] = np.cross(rotation[:, 0], rotation[:, 1])
    local = (points - centre) @ rotation
    lo, hi = np.quantile(local, [0.01, 0.99], axis=0)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = centre + rotation @ ((lo + hi) / 2)
    return {
        "T_camera_anchor_initial": checked_transform(transform).tolist(),
        "visible_extent_m": (hi - lo).tolist(),
        "valid_depth_points": len(points),
        "pca_singular_values": singular.tolist(),
        "pose_frame_definition": "initial_visible_depth_pca",
        "object_geometric_centre_verified": False,
        "full_dimensions_available": False,
        "box_axis_symmetry_resolved": False,
        "planner_eligible": False,
    }


def estimate_row(*, object_id, frame_id, sim_time, relative_pose, anchor,
                 T_world_camera_cv, lost, residuals, inliers, initialization=False,
                 mask_source="unknown", init_provenance=None,
                 source_sequence_sha256=None,
                 camera_extrinsics_source="unspecified"):
    relative = checked_transform(relative_pose, "Point2Pose relative pose")
    initial = checked_transform(anchor["T_camera_anchor_initial"], "initial anchor")
    world_camera = checked_transform(T_world_camera_cv, "camera extrinsics")
    if not isinstance(frame_id, int) or frame_id < 0 or not np.isfinite(sim_time):
        raise ValueError("invalid observation identity")
    errors = np.asarray([] if residuals is None else residuals, dtype=float).reshape(-1)
    supports = np.asarray([] if inliers is None else inliers, dtype=bool).reshape(-1)
    if supports.size != errors.size:
        raise ValueError("residual and inlier arrays must have equal length")
    finite_errors = errors[np.isfinite(errors) & (errors >= 0)]
    count = int(np.count_nonzero(supports))
    mean_error = float(finite_errors.mean()) if finite_errors.size else None
    inlier_errors = errors[supports]
    valid_inlier_errors = (inlier_errors.size == count and
                           np.isfinite(inlier_errors).all() and
                           (inlier_errors >= 0).all())
    mean_inlier_error = float(inlier_errors.mean()) if valid_inlier_errors and count else None
    inlier_ratio = float(count / errors.size) if errors.size else 0.0
    valid = (not initialization and not lost and count >= 6 and inlier_ratio >= 0.5
             and mean_inlier_error is not None and mean_inlier_error <= 0.006)
    T_camera_anchor = relative @ initial
    row = {
        "instance_id": object_id, "frame_id": frame_id, "sensor_time_s": float(sim_time),
        "source": "point2pose_initialization" if initialization else "point2pose_inference",
        "mask_initialization_source": mask_source,
        "camera_extrinsics_source": camera_extrinsics_source,
        "T_world_camera_cv": world_camera.tolist(),
        "tracking_valid": bool(valid), "lost": bool(lost),
        "inlier_count": count, "mean_residual_m": mean_error,
        "inlier_ratio": inlier_ratio,
        "mean_inlier_residual_m": mean_inlier_error,
        "T_camera_object": T_camera_anchor.tolist(),
        "T_world_object": (world_camera @ T_camera_anchor).tolist(),
        "relative_pose": relative.tolist(),
        **{key: value for key, value in anchor.items() if key != "T_camera_anchor_initial"},
        # Even a well-tracked visible patch is not yet a verified suction target.
        "planner_eligible": False,
    }
    if source_sequence_sha256 is not None:
        row["source_sequence_sha256"] = str(source_sequence_sha256)
    if init_provenance is not None:
        row["initialization_provenance"] = dict(init_provenance)
    return row


def upright_box_anchor(depth_m, mask, intrinsics, T_world_camera_cv,
                       dimensions_m, dimensions_source, gravity_world=(0, 0, -1)):
    """Visible top-plane fit plus explicitly sourced box-dimension prior.

    Hidden dimensions and 180-degree yaw are not inferred or certified.
    """
    import cv2
    fallback = visible_depth_anchor(depth_m, mask, intrinsics)
    sizes = np.asarray(dimensions_m, dtype=float)
    if sizes.shape != (3,) or not np.isfinite(sizes).all() or np.any(sizes <= 0):
        raise ValueError("dimensions_m must contain three positive lengths")
    if dimensions_source not in {"assumed_scene_dimensions", "measured_dimensions", "manufacturer_dimensions"}:
        raise ValueError("unknown dimensions provenance")
    world_camera = checked_transform(T_world_camera_cv)
    gravity = np.asarray(gravity_world, dtype=float)
    if gravity.shape != (3,) or not np.isfinite(gravity).all() or np.linalg.norm(gravity) < .1:
        raise ValueError("invalid gravity")
    up = -gravity / np.linalg.norm(gravity)
    depth, K = np.asarray(depth_m), np.asarray(intrinsics)
    yy, xx = np.nonzero(np.asarray(mask, bool) & np.isfinite(depth) & (depth > .1) & (depth < 10))
    z = depth[yy, xx]
    points = np.column_stack(((xx - K[0, 2]) * z / K[0, 0], (yy - K[1, 2]) * z / K[1, 1], z))
    world = points @ world_camera[:3, :3].T + world_camera[:3, 3]
    heights = world @ up
    top_height = float(np.quantile(heights, .95))
    top = world[np.abs(heights - top_height) <= .012]
    fallback.update(dimensions_prior_m=sizes.tolist(), dimensions_source=dimensions_source,
                    top_surface_points=len(top))
    if len(top) < 64:
        fallback["top_fit_rejected_reason"] = "insufficient_visible_top_points"
        return fallback
    basis_x = np.array([1., 0., 0.])
    if abs(basis_x @ up) > .9:
        basis_x = np.array([0., 1., 0.])
    basis_x -= up * (basis_x @ up)
    basis_x /= np.linalg.norm(basis_x)
    basis_y = np.cross(up, basis_x)
    plane = np.column_stack((top @ basis_x, top @ basis_y)).astype(np.float32)
    rectangle = cv2.minAreaRect(plane)
    corners = cv2.boxPoints(rectangle).astype(float)
    edges = np.roll(corners, -1, axis=0) - corners
    lengths = np.linalg.norm(edges, axis=1)
    if max(lengths) <= 1e-6:
        fallback["top_fit_rejected_reason"] = "degenerate_top_rectangle"
        return fallback
    edge = edges[int(np.argmax(lengths))]
    edge /= np.linalg.norm(edge)
    long_axis = edge[0] * basis_x + edge[1] * basis_y
    if long_axis[np.argmax(np.abs(long_axis))] < 0:
        long_axis *= -1
    x_axis = long_axis if sizes[0] >= sizes[1] else np.cross(long_axis, up)
    y_axis = np.cross(up, x_axis)
    observed = np.sort(np.asarray(rectangle[1], float))
    coverage = observed / np.sort(sizes[:2])
    if np.any(coverage < .7) or np.any(coverage > 1.2):
        fallback.update(top_fit_rejected_reason="visible_rectangle_does_not_cover_dimension_prior",
                        top_extent_to_prior_ratio=coverage.tolist())
        return fallback
    centre_xy = np.asarray(rectangle[0])
    top_centre = basis_x * centre_xy[0] + basis_y * centre_xy[1] + up * np.median(top @ up)
    world_anchor = np.eye(4)
    world_anchor[:3, :3] = np.column_stack((x_axis, y_axis, up))
    world_anchor[:3, 3] = top_centre - up * (sizes[2] / 2)
    return {**fallback,
            "T_camera_anchor_initial": (np.linalg.inv(world_camera) @ world_anchor).tolist(),
            "pose_frame_definition": "upright_visible_top_rectangle_plus_dimension_prior",
            "top_extent_to_prior_ratio": coverage.tolist(), "upright_cuboid_assumed": True,
            "initial_top_centre_world_estimate": top_centre.tolist(),
            "full_dimensions_available": True, "full_dimensions_estimated": False,
            "box_axis_symmetry_resolved": False, "object_geometric_centre_verified": False,
            "planner_eligible": False}
