"""Independent upright acceptance for planner candidates, before robot execution."""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
from depallet.motion.curobo_bridge import write_json, sha256
from depallet.validation.payload_upright import certify_upright_trajectory


def make_upright_candidate_validator(request, robot_config, accepted_pieces, output, goal_number):
    """Certify the original measured attachment over the whole accepted prefix.

    No physics, trajectory repair, changed tolerance, or new attachment reference.
    The caller retains its final whole-trajectory certificate and other checks.
    """
    if isinstance(goal_number, bool) or not isinstance(goal_number, int) or not 1 <= goal_number <= 32:
        raise ValueError("Bounded goal number required")
    original_request = copy.deepcopy(request)
    original_robot = copy.deepcopy(robot_config)
    prefix = (np.concatenate(accepted_pieces, axis=0).astype(np.float64, copy=True)
              if accepted_pieces else np.empty((0, 6), dtype=np.float64))
    if prefix.ndim != 2 or prefix.shape[1] != 6 or not np.isfinite(prefix).all():
        raise ValueError("Invalid previously accepted prefix")
    destination = Path(output) / "candidate-upright" / f"goal-{goal_number:02d}"

    def validate(result, context):
        attempt = context.get("attempt")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or not 1 <= attempt <= 6:
            raise ValueError("Bounded candidate attempt required")
        trajectory = result.get_interpolated_plan()
        if trajectory is None or getattr(trajectory, "position", None) is None:
            raise ValueError("Candidate success without positions")
        array = trajectory.position.detach().cpu().numpy()
        if array.ndim < 2 or array.shape[-1] != 6 or any(n != 1 for n in array.shape[:-2]):
            raise ValueError("Unexpected candidate position shape")
        positions = array.reshape(-1, 6).astype(np.float64, copy=True)
        if not 2 <= len(positions) <= 2000 or not np.isfinite(positions).all():
            raise ValueError("Invalid candidate positions")
        combined = np.concatenate((prefix, positions[1:] if len(prefix) else positions), axis=0)
        certificate = certify_upright_trajectory(original_request, combined, robot_config=original_robot)
        if type(certificate.get("passed")) is not bool:
            raise ValueError("Upright certificate lacks a boolean verdict")
        path = destination / f"attempt-{attempt:02d}.json"
        rejected = destination / f"attempt-{attempt:02d}-rejected.npz"
        if path.exists() or rejected.exists():
            raise FileExistsError("Refusing to overwrite candidate evidence")
        destination.mkdir(parents=True, exist_ok=True)
        write_json(path, certificate)
        if not certificate["passed"]:
            np.savez_compressed(rejected, position_rad=combined)
        return dict(schema="depallet.upright_candidate_acceptance.v1", passed=certificate["passed"],
                    certificate_path=str(path.relative_to(output)), certificate_sha256=sha256(path),
                    accepted_prefix_rows=len(prefix), candidate_rows=len(positions),
                    combined_rows=len(combined), original_measured_attachment_reference_preserved=True,
                    maximum_observed_tilt_rad=certificate["maximum_observed_tilt_rad"],
                    maximum_certified_tilt_bound_rad=certificate["maximum_certified_tilt_bound_rad"],
                    maximum_nominal_tilt_rad=certificate["maximum_nominal_tilt_rad"],
                    failure=certificate.get("failure"), physical_execution_performed=False)
    return validate
