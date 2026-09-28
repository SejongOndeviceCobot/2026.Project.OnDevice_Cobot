#!/usr/bin/env python3
"""Read-only, CPU-only asset preflight for the verified V1 link24 run.

This checks the exact example inputs and external robot assets used by the
16/16 Isaac Sim baseline. It does not import Isaac Sim, cuRobo, or CUDA and
does not claim that a new physical run will pass.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
DATA_REL = Path("data/depallet_isaac_p0")
SCENARIO_REL = Path(
    "scenario-suite-v1/v1_uniform_box16_first_box13_early_final_layer_v1"
)
EXAMPLES = Path("examples/v1_uniform")
PROFILE = "link24_hull_margin10mm"
UPSTREAM_REL = Path("repos/external/github.com/DoosanRobotics/doosan-robot2")

# These are the bytes saved with the independently audited 16/16 run.
EXAMPLE_SHA256 = {
    "scenario.json": "6de8ca5ea0832b77d577943ab534445c80c7696bff483edfb7057e85590e8e0e",
    "rule-plan.json": "7b10b2caa7359608f1ce101a4a816a2851559c4cc112936f79e012c543d0da0c",
}

# Runtime assets stay in workspace/data. These pins describe the successful
# link2 -> link24 collision profile and its physical robot inputs.
ASSET_SHA256 = {
    "h2017-vgp20-v2/manifest.json": "fdb8a4b3549dedbe83a18f36dc5c09e05d46e20ab541ba63ebd9349c01d29563",
    "h2017-vgp20-v2/h2017_vgp20.usda": "dab19df825fb1cc68c3aadc233ae367d4c5ff54f2e55ca1eb8357633503ef27c",
    "h2017-vgp20-v2/h2017_vgp20.urdf": "d605728ab139dd167728e4c710cd20b7db870edf5dc74998e90ea940083001f8",
    "h2017-dynamics-v1/independent-validation.json": "35a82e193bbf271210c46bb964217200e383a92b54a7e66efd93a1c2fc8b54fe",
    "grasp-requests-v2/rear_upper_3.request.json": "d508af1a7f205258f8ec0bc64d1675e912687951f276062e98a3f1c455bbe223",
    "grasp-requests-v2/robot-slow.yml": "c11db2bbbf2868a2f290aa671ce26b4022594e254d673a275848226806785e05",
    "grasp-requests-v2/h2017_vgp20_velocity_limited.urdf": "1510e86713f806b0a493c9c45db8f7730571b113a77cef6845461b617b1d445f",
    "curobo-assets-v1/manifest.json": "e201aff3278d9b5c4e15a9363fd9aa08a7be169f2456ffd9b1a122f035bf3ede",
    "curobo-assets-v1/h2017_vgp20_planner.urdf": "bf93e2278a5e9e6bb177cd85a716b3d3be52ac4b5e2746453cb5100c23b2875c",
    "robot-collision-candidates-v1/h2017_link2_hull12_margin10mm/robot.yml": "06b51958bc7e400ebb7ab3f3fc364b0862ca7d3e3e75b263a5fa4d4e94794fa6",
    "robot-collision-candidates-v1/h2017_link2_hull12_margin10mm/manifest.json": "3c10031ac5915455275e7713f938c195445a8792e57c7e1a9d95714b77fb3034",
    "robot-collision-candidates-v1/h2017_link2_hull12_margin10mm/validation.json": "f73d16b929142b1b9622d619709aac624a69efaf1767ee6bbd252c276e6f68bf",
    "robot-collision-candidates-v2/h2017_link24_hull_margin10mm/robot.yml": "145003cb16ae494c6587067a0d31e237f2e01dae191e9deb1e3ecb717b7e7be6",
    "robot-collision-candidates-v2/h2017_link24_hull_margin10mm/manifest.json": "8f71f18e71e6b16f1a33da884ec329b9940498a29e771e88b5cf3f43752d24d0",
    "robot-collision-candidates-v2/h2017_link24_hull_margin10mm/validation.json": "79fb42c6f947cfb32223671ac8a4afcfa91a656c5d576698125520743f0fea4b",
}

LINK2 = Path("robot-collision-candidates-v1/h2017_link2_hull12_margin10mm")
LINK24 = Path("robot-collision-candidates-v2/h2017_link24_hull_margin10mm")


class PreflightError(ValueError):
    """A required file, digest, or path binding is absent or invalid."""


def _directory(path: Path, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PreflightError(f"{label} does not exist: {path}") from exc
    if not resolved.is_dir():
        raise PreflightError(f"{label} is not a directory: {resolved}")
    return resolved


def _file(root: Path, relative: Path, label: str) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise PreflightError(f"{label} has an invalid relative path: {relative}")
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise PreflightError(f"{label} must not use a symlink below {root}: {current}")
    try:
        resolved = current.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PreflightError(f"Missing {label}: {current}") from exc
    if not resolved.is_file() or not resolved.is_relative_to(root):
        raise PreflightError(f"{label} must be a regular file under {root}: {current}")
    return resolved


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pinned(path: Path, expected: str, label: str) -> None:
    actual = _digest(path)
    if actual != expected:
        raise PreflightError(
            f"{label} SHA-256 mismatch: {path} (expected {expected}, got {actual})"
        )


def _json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"Invalid JSON in {label}: {path}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"{label} must be a JSON object: {path}")
    return value


def _bound_path(recorded: Any, expected: Path, label: str) -> None:
    if not isinstance(recorded, str) or not Path(recorded).is_absolute():
        raise PreflightError(f"{label} must contain an absolute asset path")
    try:
        actual = Path(recorded).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PreflightError(f"{label} points to a missing file: {recorded}") from exc
    if actual != expected:
        raise PreflightError(f"{label} points to {actual}, expected {expected}")


def _data_reference(recorded: Any, data: Path, label: str) -> Path:
    if not isinstance(recorded, str) or not Path(recorded).is_absolute():
        raise PreflightError(f"{label} must be an absolute path under {data}")
    try:
        resolved = Path(recorded).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PreflightError(f"Missing {label}: {recorded}") from exc
    if not resolved.is_relative_to(data):
        raise PreflightError(f"{label} is outside the project data root: {recorded}")
    return _file(data, resolved.relative_to(data), label)


def _check_examples(data: Path) -> tuple[Path, Path]:
    example = {}
    deployed = {}
    for name, expected_sha in EXAMPLE_SHA256.items():
        example[name] = _file(PROJECT, EXAMPLES / name, f"packaged {name}")
        _pinned(example[name], expected_sha, f"packaged {name}")
        try:
            deployed[name] = _file(data, SCENARIO_REL / name, f"runtime {name}")
        except PreflightError as exc:
            raise PreflightError(
                f"{exc}. Run python3 tools/prepare_example.py to install the packaged example"
            ) from exc
        _pinned(deployed[name], expected_sha, f"runtime {name}")

    scenario = _json(example["scenario.json"], "packaged scenario")
    plan = _json(example["rule-plan.json"], "packaged rule plan")
    boxes = scenario.get("spec", {}).get("boxes")
    order = plan.get("order")
    placements = plan.get("packing", {}).get("placements")
    if (
        scenario.get("schema") != "depallet.scenario.v1"
        or scenario.get("scenario_id") != "v1_uniform"
        or not isinstance(scenario.get("cell"), dict)
        or scenario.get("capabilities", {}).get("unsupported_reasons") != []
        or not isinstance(boxes, list)
        or len(boxes) != 16
        or not isinstance(order, list)
        or len(order) != 16
        or not isinstance(placements, list)
        or {box.get("id") for box in boxes} != set(order)
        or {item.get("box_id") for item in placements} != set(order)
        or plan["packing"].get("order") != order
    ):
        raise PreflightError("Packaged V1 scenario and 16-box rule plan do not match")
    result = _json(_file(PROJECT, Path("evidence/full16-task-result.json"), "full16 evidence"),
                   "full16 evidence")
    if (
        result.get("task_complete") is not True
        or result.get("perception_source") != "simulation_oracle"
        or result.get("completed_ids") != order
    ):
        raise PreflightError("Full16 evidence does not match the packaged rule plan")
    return deployed["scenario.json"], deployed["rule-plan.json"]


def _check_profile_source() -> None:
    source = _file(PROJECT, Path("src/robot_collision_profiles.py"), "profile source")
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        values = {
            target.id: ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
            and target.id in {"PROFILE_NAMES", "LINK24_REL", "PINNED_SHA256", "LINK24_PINS"}
        }
    except (OSError, SyntaxError, ValueError) as exc:
        raise PreflightError(f"Cannot read profile pins from {source}") from exc
    expected_link2 = {
        "baseline": ASSET_SHA256["grasp-requests-v2/robot-slow.yml"],
        "candidate": ASSET_SHA256[str(LINK2 / "robot.yml")],
        "manifest": ASSET_SHA256[str(LINK2 / "manifest.json")],
        "validation": ASSET_SHA256[str(LINK2 / "validation.json")],
        "urdf": ASSET_SHA256["grasp-requests-v2/h2017_vgp20_velocity_limited.urdf"],
    }
    expected_link24 = {
        "candidate": ASSET_SHA256[str(LINK24 / "robot.yml")],
        "manifest": ASSET_SHA256[str(LINK24 / "manifest.json")],
        "validation": ASSET_SHA256[str(LINK24 / "validation.json")],
    }
    if (
        PROFILE not in values.get("PROFILE_NAMES", ())
        or values.get("LINK24_REL") != str(LINK24)
        or values.get("PINNED_SHA256") != expected_link2
        or values.get("LINK24_PINS") != expected_link24
    ):
        raise PreflightError("Runtime collision-profile pins differ from this 16/16 baseline")


def _check_assembly(data: Path, files: dict[str, Path]) -> int:
    manifest = _json(files["h2017-vgp20-v2/manifest.json"], "assembly manifest")
    _bound_path(manifest.get("usd"), files["h2017-vgp20-v2/h2017_vgp20.usda"],
                "assembly manifest USD")
    _bound_path(manifest.get("urdf"), files["h2017-vgp20-v2/h2017_vgp20.urdf"],
                "assembly manifest URDF")
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise PreflightError("Assembly manifest has no source hash map")
    references = set(re.findall(r"@([^@]+)@", files["h2017-vgp20-v2/h2017_vgp20.usda"].read_text()))
    expected_data_sources = {
        Path(recorded).resolve(strict=False): sha
        for recorded, sha in sources.items()
        if isinstance(recorded, str)
        and Path(recorded).resolve(strict=False).is_relative_to(data)
    }
    resolved_references = set()
    for recorded in references:
        path = _data_reference(recorded, data, "assembly USD reference")
        resolved_references.add(path)
        expected = expected_data_sources.get(path)
        if not isinstance(expected, str):
            raise PreflightError(f"Assembly USD reference lacks a manifest SHA: {recorded}")
        _pinned(path, expected, "assembly USD reference")
    if resolved_references != set(expected_data_sources):
        raise PreflightError("Assembly USD references differ from manifest data sources")
    return len(resolved_references)


def _check_meshes(data: Path, files: dict[str, Path]) -> int:
    manifest = _json(files["curobo-assets-v1/manifest.json"], "cuRobo asset manifest")
    planner = files["curobo-assets-v1/h2017_vgp20_planner.urdf"]
    _bound_path(manifest.get("planner_urdf"), planner, "cuRobo planner URDF")
    if manifest.get("planner_urdf_sha256") != ASSET_SHA256[
        "curobo-assets-v1/h2017_vgp20_planner.urdf"
    ]:
        raise PreflightError("cuRobo manifest planner URDF SHA binding differs")
    records = manifest.get("meshes")
    if not isinstance(records, list) or not records:
        raise PreflightError("cuRobo asset manifest has no mesh records")
    meshes: set[Path] = set()
    for record in records:
        if not isinstance(record, dict):
            raise PreflightError("Invalid cuRobo mesh manifest entry")
        path = _data_reference(record.get("copy"), data, "cuRobo mesh copy")
        if path in meshes:
            raise PreflightError(f"Duplicate cuRobo mesh manifest entry: {path}")
        expected = record.get("copy_sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise PreflightError(f"Invalid cuRobo mesh SHA binding: {path}")
        _pinned(path, expected, "cuRobo mesh")
        meshes.add(path)
    for rel in (
        "grasp-requests-v2/h2017_vgp20_velocity_limited.urdf",
        "curobo-assets-v1/h2017_vgp20_planner.urdf",
    ):
        try:
            root = ET.parse(files[rel]).getroot()
        except ET.ParseError as exc:
            raise PreflightError(f"Invalid URDF XML: {files[rel]}") from exc
        referenced = {
            _data_reference(node.attrib.get("filename"), data, f"{rel} mesh")
            for node in root.iter("mesh")
        }
        if referenced != meshes:
            raise PreflightError(f"{rel} mesh references differ from the cuRobo manifest")
    return len(meshes)


def _check_profile_bindings(files: dict[str, Path]) -> None:
    request = _json(files["grasp-requests-v2/rear_upper_3.request.json"], "grasp request")
    _bound_path(request.get("robot_config"), files["grasp-requests-v2/robot-slow.yml"],
                "grasp request robot_config")
    link2 = _json(files[str(LINK2 / "manifest.json")], "link2 manifest")
    for field, relative in (
        ("robot_config", LINK2 / "robot.yml"),
        ("baseline_config", Path("grasp-requests-v2/robot-slow.yml")),
        ("urdf", Path("grasp-requests-v2/h2017_vgp20_velocity_limited.urdf")),
        ("validation", LINK2 / "validation.json"),
    ):
        key = str(relative)
        _bound_path(link2.get(field), files[key], f"link2 manifest {field}")
        if link2.get(field + "_sha256") != ASSET_SHA256[key]:
            raise PreflightError(f"link2 manifest {field} SHA binding differs")
    link24 = _json(files[str(LINK24 / "manifest.json")], "link24 manifest")
    _bound_path(link24.get("parent_config"), files[str(LINK2 / "robot.yml")],
                "link24 manifest parent_config")
    if (
        link24.get("parent_sha256") != ASSET_SHA256[str(LINK2 / "robot.yml")]
        or link24.get("only_link4_append") is not True
        or link24.get("physical_execution_validated") is not False
    ):
        raise PreflightError("link24 parent hash or validation scope differs")
    validation = _json(files[str(LINK24 / "validation.json")], "link24 validation")
    if (
        validation.get("failure_pose_rejected") is not True
        or validation.get("physical_validated") is not False
    ):
        raise PreflightError("link24 validation scope differs")



def _check_upstream(workspace: Path, data: Path, files: dict[str, Path]) -> tuple[Path, int, str]:
    """Check the bound Doosan asset tree; its binary USD has no published pin."""
    upstream = _directory(workspace / UPSTREAM_REL, "Doosan upstream asset root")
    selected = files[str(LINK24 / "robot.yml")]
    roots = re.findall(r"(?m)^\s*asset_root_path:\s*(\S+)\s*$", selected.read_text())
    if len(roots) != 1:
        raise PreflightError("link24 robot.yml must declare one asset_root_path")
    recorded_root = roots[0].strip("\"'")
    if not Path(recorded_root).is_absolute():
        raise PreflightError("link24 asset_root_path must be absolute")
    try:
        actual_root = Path(recorded_root).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PreflightError(f"Missing link24 asset_root_path: {recorded_root}") from exc
    if actual_root != upstream:
        raise PreflightError(f"link24 asset_root_path points to {actual_root}, expected {upstream}")

    source_urdf = _file(upstream, Path("dsr_description2/urdf/h2017.urdf"),
                        "Doosan source URDF")
    assembly = _json(files["h2017-vgp20-v2/manifest.json"], "assembly manifest")
    sources = assembly.get("sources")
    if not isinstance(sources, dict):
        raise PreflightError("Assembly manifest has no source hash map")
    urdf_entries = [
        (recorded, sha) for recorded, sha in sources.items()
        if Path(recorded).resolve(strict=False) == source_urdf
    ]
    if len(urdf_entries) != 1:
        raise PreflightError("Assembly manifest lacks the Doosan source URDF hash binding")
    _bound_path(urdf_entries[0][0], source_urdf, "assembly Doosan source URDF")
    _pinned(source_urdf, urdf_entries[0][1], "Doosan source URDF")

    try:
        root = ET.parse(source_urdf).getroot()
    except ET.ParseError as exc:
        raise PreflightError(f"Invalid Doosan source URDF XML: {source_urdf}") from exc
    mesh_prefix = "package://dsr_description2/"
    referenced: set[Path] = set()
    for node in root.iter("mesh"):
        filename = node.attrib.get("filename", "")
        if not filename.startswith(mesh_prefix):
            raise PreflightError(f"Unexpected Doosan URDF mesh reference: {filename}")
        relative = Path("dsr_description2") / filename[len(mesh_prefix):]
        referenced.add(_file(upstream, relative, "Doosan source mesh"))
    if not referenced:
        raise PreflightError("Doosan source URDF has no mesh references")

    asset_manifest = _json(files["curobo-assets-v1/manifest.json"], "cuRobo asset manifest")
    records = asset_manifest.get("meshes")
    if not isinstance(records, list):
        raise PreflightError("cuRobo asset manifest has no source mesh records")
    recorded_meshes: set[Path] = set()
    for record in records:
        if not isinstance(record, dict):
            raise PreflightError("Invalid Doosan source mesh manifest entry")
        source = record.get("source")
        if not isinstance(source, str) or not Path(source).is_absolute():
            raise PreflightError("Doosan source mesh path must be absolute")
        try:
            resolved = Path(source).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise PreflightError(f"Missing Doosan source mesh: {source}") from exc
        if not resolved.is_relative_to(upstream):
            raise PreflightError(f"Doosan source mesh is outside the upstream checkout: {source}")
        path = _file(upstream, resolved.relative_to(upstream), "Doosan source mesh")
        _bound_path(source, path, "cuRobo manifest Doosan mesh source")
        expected = record.get("source_sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise PreflightError(f"Invalid Doosan source mesh SHA binding: {source}")
        _pinned(path, expected, "Doosan source mesh")
        if path in recorded_meshes:
            raise PreflightError(f"Duplicate Doosan source mesh record: {path}")
        recorded_meshes.add(path)
    if referenced != recorded_meshes:
        raise PreflightError("Doosan URDF mesh references differ from manifest source meshes")
    assembly_urdf = files["h2017-vgp20-v2/h2017_vgp20.urdf"]
    try:
        assembly_root = ET.parse(assembly_urdf).getroot()
    except ET.ParseError as exc:
        raise PreflightError(f"Invalid assembly URDF XML: {assembly_urdf}") from exc
    assembly_meshes: set[Path] = set()
    for node in assembly_root.iter("mesh"):
        filename = node.attrib.get("filename", "")
        if not Path(filename).is_absolute():
            raise PreflightError(f"Assembly URDF mesh path is not absolute: {filename}")
        try:
            resolved = Path(filename).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise PreflightError(f"Missing assembly URDF mesh: {filename}") from exc
        if not resolved.is_relative_to(upstream):
            raise PreflightError(f"Assembly URDF mesh is outside Doosan checkout: {filename}")
        assembly_meshes.add(_file(upstream, resolved.relative_to(upstream),
                                  "assembly URDF mesh"))
    if assembly_meshes != recorded_meshes:
        raise PreflightError("Assembly URDF mesh references differ from Doosan manifest meshes")

    # The selected asset root also contains the original vendor USD. Its exact
    # digest is not recorded by either project manifest, so only presence and
    # the binary USDC container signature are checked here.
    usd = _file(upstream, Path("dsr_description2/usd/h2017.usd"), "Doosan source USD")
    with usd.open("rb") as source:
        signature = source.read(8)
    if signature != b"PXR-USDC":
        raise PreflightError(f"Doosan source USD has an unexpected format: {usd}")
    dynamics = _file(data, Path("h2017-dynamics-v1/h2017_dynamics.usda"),
                     "assembly dynamics USD")
    dynamics_refs = re.findall(r"@([^@]+)@", dynamics.read_text())
    if len(dynamics_refs) != 1:
        raise PreflightError("Assembly dynamics USD must contain one Doosan USD reference")
    _bound_path(dynamics_refs[0], usd, "assembly dynamics Doosan USD reference")
    return upstream, len(recorded_meshes), _digest(usd)


def validate() -> dict[str, Any]:
    """Return a readiness report or raise PreflightError on the first blocker."""
    value = os.environ.get("JCLEE_WORKSPACE")
    if not value:
        raise PreflightError("JCLEE_WORKSPACE is unset; source workspace/setup/env.sh")
    workspace = _directory(Path(value), "JCLEE_WORKSPACE")
    data = _directory(workspace / DATA_REL, "project data root")
    scenario, rule_plan = _check_examples(data)
    _check_profile_source()
    files = {}
    for relative, expected in ASSET_SHA256.items():
        path = _file(data, Path(relative), f"required asset {relative}")
        _pinned(path, expected, f"required asset {relative}")
        files[relative] = path
    usd_source_count = _check_assembly(data, files)
    mesh_count = _check_meshes(data, files)
    _check_profile_bindings(files)
    upstream, upstream_mesh_count, upstream_usd_sha = _check_upstream(workspace, data, files)
    return {
        "schema": "ondevice_cobot.asset_preflight.v1",
        "passed": True,
        "gpu_launched": False,
        "physical_run_reproduced": False,
        "project": str(PROJECT),
        "asset_workspace": str(workspace),
        "data_root": str(data),
        "scenario": str(scenario),
        "rule_plan": str(rule_plan),
        "robot_collision_profile": PROFILE,
        "verified_asset_count": len(files) + usd_source_count + mesh_count + upstream_mesh_count + 2,
        "verified_mesh_count": mesh_count,
        "upstream_asset_root": str(upstream),
        "verified_upstream_mesh_count": upstream_mesh_count,
        "upstream_usd_sha256": upstream_usd_sha,
        "upstream_usd_sha256_pinned": False,
    }


def main() -> int:
    try:
        report = validate()
    except PreflightError as exc:
        print(f"asset preflight failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
