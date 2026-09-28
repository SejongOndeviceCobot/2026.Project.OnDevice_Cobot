#!/usr/bin/env python3
"""Check immutable V1 evidence and, optionally, the modular checkout without Isaac.

Archived run receipts prove their recorded scope only. A modular source manifest
checks current bytes and import layout; it does not rerun physics.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile


PUBLISHED_SOURCE_SHA256 = "70db570281186f184bdbf07b364f37a4f73f9b821e10ab53e04b4506efd528b8"
SNAPSHOT_MANIFEST_SHA256 = "8a54962c12f4e1cbee668c5172364de1aaeb0faec88e99a952638d9970ecec23"
SOURCE_ARCHIVE_SHA256 = "f09438c0322e3f655487b27e13e07706f8df1f0615d2672e340337b7702b1595"
TASK_RESULT_SHA256 = "943cad2d9f09d0c4feb87c43d445e73613851bf56aafbd8476f9463a7fcfa54a"
AUDIT_SUMMARY_SHA256 = "32e700fe7958de00d83d18ed887676d025e09168da1f9780b87c246a2bb116f5"
FULL_AUDIT_SHA256 = "f0c35e52439df5953cde1f5f2ac0d639112c3f7c753f4d2309748d0594acf727"
SCENARIO_SHA256 = "6de8ca5ea0832b77d577943ab534445c80c7696bff483edfb7057e85590e8e0e"
RULE_PLAN_SHA256 = "7b10b2caa7359608f1ce101a4a816a2851559c4cc112936f79e012c543d0da0c"
REPACKAGED_TASK_RESULT_SHA256 = "00f886b8027f0300c089e85959f593d6c44299c7b4c23139def35ae337550d1d"
REPACKAGED_EXIT_SHA256 = "e5d06bb19af78394543a39c9bfd00fafbe1fe07cdf60fed7896fb906ac62b949"
REPACKAGED_SOURCE_MANIFEST_SHA256 = "06b26b2af409a2069374018512f6c747206405dec1d28939f847087eed19e483"
REPACKAGED_FULL_AUDIT_SHA256 = "cdf8932ee5b1191e7928d4742c6137eb577c7d3b5580ca802ae600acf0247377"
REPACKAGED_EVALUATOR_SHA256 = "305a3c43214b6fdc7d960f51246c7285d89e4249f4a8bde71b6ef4875590b056"
REPACKAGED_AUDIT_SUMMARY_SHA256 = "0c7785b27679667c52f183938df0ad91c887298949c4362bffd10d9d46d8d816"
REPACKAGED_RUN = "full-20260927T192336_405591Z"
SOURCE_RUN = "20260921-current/physical-v1-link24-box13early-full16-overhead20-observerfinalize-repeat-v1"
# Immutable receipts for the completed modular run and its separate CPU audit.
MODULAR_RUN = "full-20260928T191017_646556Z"
MODULAR_TASK_RESULT_SHA256 = "bac1b97a9b1ea85fba199bab9fdea01244a054a0a991c6a995a54f36e65e2c97"
MODULAR_EXIT_SHA256 = "1bfcecd901a0fcaea9baaaee27b6988bb4ef5e2dd07162cd88bc8de9853ba8a5"
MODULAR_SOURCE_MANIFEST_SHA256 = "c54eb92b5f5fe54fc29ba7729fd79b02c65e6e2c73e0e0b73510943e6e0f2de3"
MODULAR_PREFLIGHT_SHA256 = "23541596663ebf9cb37dfea945985bbf81f939803a820be44d2b4defc259f48f"
MODULAR_AUDIT_SUMMARY_SHA256 = "4b9a90f5ea2bd5156eab15e2b2c60619324982e90c446fd16ed384105f44156a"
MODULAR_FULL_AUDIT_SHA256 = "12f440c47666fd1c40d6bdee42ab0ceb8d1ad3ec6ae705ed09c9089b7cc8f854"
MODULAR_EVALUATOR_SHA256 = "985845fd8ace574fdba8d79519ac186bd447824c7dc19220d8270645d52c7e49"
MODULAR_AUDIT_CHECKS = 490
MODULAR_RECEIPTS = (
    "evidence/modular-full16-task-result.json",
    "evidence/modular-full16-exit.json",
    "evidence/modular-full16-source-manifest.json",
    "evidence/modular-full16-preflight.json",
    "evidence/modular-full16-audit-summary.json",
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class EvidenceError(ValueError):
    """An evidence file is missing, changed, or internally inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceError(message)


def repository_file(root: Path, relative: str) -> Path:
    """Accept only ordinary files beneath the selected checkout."""
    require(isinstance(relative, str) and bool(relative), "빈 파일 경로")
    path = PurePosixPath(relative)
    require(not path.is_absolute() and all(part not in (".", "..") for part in path.parts),
            f"저장소 밖 경로: {relative}")
    candidate = root
    for part in path.parts:
        candidate = candidate / part
        require(not candidate.is_symlink(), f"심볼릭 링크 경로: {relative}")
    require(candidate.is_file(), f"파일 없음: {relative}")
    return candidate


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pinned_json(root: Path, relative: str, expected_sha256: str) -> dict:
    path = repository_file(root, relative)
    require(sha256_file(path) == expected_sha256, f"SHA256 불일치: {relative}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON 객체 필요: {relative}")
    return value


def file_index(value: dict, label: str) -> dict[str, str]:
    files = value.get("files")
    require(isinstance(files, list), f"파일 목록 없음: {label}")
    result: dict[str, str] = {}
    for entry in files:
        require(isinstance(entry, dict), f"잘못된 파일 항목: {label}")
        relative, digest = entry.get("path"), entry.get("sha256")
        require(isinstance(relative, str) and relative not in result,
                f"중복 또는 잘못된 파일 경로: {label}/{relative}")
        require(isinstance(digest, str) and SHA256_PATTERN.fullmatch(digest) is not None,
                f"잘못된 SHA256: {label}/{relative}")
        result[relative] = digest
    return result


def archived_sources(root: Path) -> dict[str, bytes]:
    """Read the immutable archive in memory, without extracting into the checkout."""
    relative = "evidence/full16-source.tar.gz"
    archive_path = repository_file(root, relative)
    require(sha256_file(archive_path) == SOURCE_ARCHIVE_SHA256,
            f"SHA256 불일치: {relative}")
    sources: dict[str, bytes] = {}
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            name = member.name
            path = PurePosixPath(name)
            require(member.isfile() and not path.is_absolute() and
                    all(part not in (".", "..") for part in path.parts) and
                    name not in sources,
                    f"보관 소스 항목 오류: {name}")
            stream = archive.extractfile(member)
            require(stream is not None, f"보관 소스를 읽을 수 없음: {name}")
            with stream:
                sources[name] = stream.read()
    return sources


def local_import_closure(sources: dict[str, bytes]) -> set[str]:
    """Recompute the flat local import graph, including the spawned cuRobo worker."""
    modules: dict[str, str] = {}
    for relative in sources:
        require(relative.startswith(("src/", "scripts/")) and relative.endswith(".py"),
                f"예상하지 못한 소스 위치: {relative}")
        name = PurePosixPath(relative).stem
        require(name not in modules, f"중복 Python 모듈명: {name}")
        modules[name] = relative
    imports: dict[str, set[str]] = {}
    for name, relative in modules.items():
        tree = ast.parse(sources[relative].decode("utf-8"), filename=relative)
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                targets = [node.module]
            else:
                continue
            names.update(target.split(".", 1)[0] for target in targets
                         if target.split(".", 1)[0] in modules)
        imports[name] = names
    seeds = {"run_task", "guarded_run", "curobo_worker"}
    require(seeds <= modules.keys(), "실행 진입점 누락")
    reached: set[str] = set()
    pending = list(seeds)
    while pending:
        name = pending.pop()
        if name not in reached:
            reached.add(name)
            pending.extend(imports[name] - reached)
    return {modules[name] for name in reached}


def verify_sources(root: Path, *, check_current: bool = False) -> int:
    published = pinned_json(root, "evidence/published-source.json", PUBLISHED_SOURCE_SHA256)
    snapshot = pinned_json(root, "evidence/code-snapshot-manifest.json", SNAPSHOT_MANIFEST_SHA256)
    require(published.get("schema") == "ondevice_cobot.published_source.v1", "게시 소스 스키마 불일치")
    require(published.get("source_run") == SOURCE_RUN, "성공 실행 출처 불일치")
    require(published.get("source_manifest_sha256") == SNAPSHOT_MANIFEST_SHA256,
            "게시 소스의 원본 manifest SHA256 불일치")
    require(snapshot.get("captured_before_simulation_start") is True,
            "실행 전 소스 스냅샷 표시 누락")
    original = file_index(snapshot, "code-snapshot-manifest.json")
    selected = file_index(published, "published-source.json")
    require(len(original) == 144 and len(selected) == 57, "스냅샷 또는 게시 소스 파일 수 불일치")
    archived = archived_sources(root)
    require(set(archived) == set(selected), "보관 소스 57개 목록 불일치")
    require(set(selected) == local_import_closure(archived), "보관 소스의 로컬 import 폐쇄 불일치")
    for relative, digest in selected.items():
        require(original.get(relative) == digest, f"원본 실행 소스와 SHA256 불일치: {relative}")
        require(hashlib.sha256(archived[relative]).hexdigest() == digest,
                f"보관된 소스 SHA256 불일치: {relative}")
    if check_current:
        verify_current_source(root, selected)
    return len(selected)


def verify_current_source(root: Path, historical: dict[str, str]) -> int:
    """Check the current modular tree, independently of pre-refactor run receipts.

    The manifest is an editable checkout inventory, unlike the pinned historical
    evidence above. A match means these *current* files have not changed since
    the manifest was made; it does not imply a successful post-refactor run.
    """
    relative_manifest = "evidence/modular-source-manifest.json"
    try:
        manifest = json.loads(repository_file(root, relative_manifest).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"현재 모듈 소스 manifest 오류: {relative_manifest}") from exc
    require(isinstance(manifest, dict) and
            manifest.get("schema") == "ondevice_cobot.modular_source.v1",
            "현재 모듈 소스 manifest 스키마 불일치")
    expected = file_index(manifest, relative_manifest)
    source_root = root / "src"
    require(source_root.is_dir() and not source_root.is_symlink(), "현재 src/ 폴더 없음")
    actual: set[str] = set()
    for candidate in source_root.rglob("*"):
        require(not candidate.is_symlink(),
                f"현재 src/ 심볼릭 링크 허용 안 함: {candidate.relative_to(root)}")
        if candidate.is_file() and candidate.suffix == ".py":
            actual.add(candidate.relative_to(root).as_posix())
    scripts = {f"scripts/{name}" for name in
               ("run_task.py", "guarded_run.py", "curobo_worker.py")}
    require(all(path.startswith("src/depallet/") for path in actual),
            "현재 src/에 depallet 패키지 밖 Python 파일이 있음")
    require(set(expected) == actual | scripts,
            "현재 모듈 소스 목록이 manifest 또는 고정 실행 스크립트와 다름")
    package_inits = {"src/depallet/__init__.py"}
    package_inits.update(f"src/depallet/{name}/__init__.py" for name in
                         ("scene", "observation", "planning", "motion",
                          "manipulation", "integration", "validation", "runtime"))
    require(package_inits <= actual, "현재 모듈 패키지 초기화 파일 누락")
    historic_names = {PurePosixPath(path).stem for path in historical
                      if path.startswith("src/")}
    current_names = [PurePosixPath(path).stem for path in actual
                     if not path.endswith("/__init__.py")]
    require(len(current_names) == len(set(current_names)) and
            historic_names <= set(current_names),
            "원본 54개 기능 모듈 중 누락 또는 중복이 있음")
    modules = {".".join(PurePosixPath(path).with_suffix("").parts[1:])
               for path in actual}
    modules = {name.removesuffix(".__init__") for name in modules}
    for relative, digest in expected.items():
        require(hashlib.sha256(repository_file(root, relative).read_bytes()).hexdigest() == digest,
                f"현재 모듈 소스 SHA256 불일치: {relative}")
        tree = ast.parse(repository_file(root, relative).read_text(encoding="utf-8"),
                         filename=relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                require(node.level == 0, f"상대 import를 사용한 현재 모듈: {relative}")
                targets = [node.module] if node.module else []
            else:
                continue
            for target in targets:
                if target is None:
                    continue
                require(target.split(".", 1)[0] not in historic_names,
                        f"이전 평면 모듈 import가 남음: {relative}: {target}")
                if target == "depallet" or target.startswith("depallet."):
                    require(target in modules,
                            f"현재 모듈 import 대상 없음: {relative}: {target}")
    return len(expected)


def verify_scenario_contract(scenario: dict, plan: dict) -> list[str]:
    require(scenario.get("schema") == "depallet.scenario.v1" and
            scenario.get("scenario_id") == "v1_uniform", "V1 예시 시나리오 불일치")
    spec = scenario.get("spec")
    require(isinstance(spec, dict) and isinstance(spec.get("boxes"), list), "시나리오 박스 목록 없음")
    boxes = spec["boxes"]
    box_ids = [box.get("id") for box in boxes if isinstance(box, dict)]
    require(len(boxes) == len(box_ids) == 16 and len(set(box_ids)) == 16 and
            all(isinstance(box_id, str) and box_id for box_id in box_ids),
            "시나리오에 고유한 박스 16개가 필요")
    require(plan.get("schema") == "depallet.scenario_rule_plan.v1" and
            plan.get("simulation_oracle_world") is True and
            plan.get("autonomous_model_executed") is False,
            "예시 계획의 시뮬레이터 정답 입력 표시 불일치")
    order = plan.get("order")
    packing = plan.get("packing")
    require(isinstance(order, list) and len(order) == 16 and len(set(order)) == 16 and
            set(order) == set(box_ids), "계획 순서가 시나리오 박스 16개와 다름")
    require(isinstance(packing, dict) and packing.get("order") == order and
            isinstance(packing.get("placements"), list), "계획 배치 또는 순서 불일치")
    placements = packing["placements"]
    placement_ids = [item.get("box_id") for item in placements if isinstance(item, dict)]
    require(len(placements) == len(placement_ids) == 16 and
            len(set(placement_ids)) == 16 and set(placement_ids) == set(box_ids),
            "계획 배치가 시나리오 박스 16개와 다름")
    return order


def verify_result_contract(result: dict, order: list[str]) -> None:
    require(result.get("schema") == "depallet.continuous_task_execution.v1", "실행 결과 스키마 불일치")
    for key in ("passed", "requested_prefix_passed", "task_complete", "physical_task_complete",
                "finalized", "physics_grasp_validated", "rgbd_integrity_passed", "continuous_simulation"):
        require(result.get(key) is True, f"실행 결과의 {key} 표시 불일치")
    require(result.get("state") == "COMPLETE" and result.get("requested_transfers") == 16 and
            result.get("total_boxes") == 16, "16개 전체 작업 완료 표시 불일치")
    require(result.get("completed_ids") == order, "완료 박스 순서가 예시 계획과 다름")
    transfers = result.get("transfer_results")
    require(isinstance(transfers, list) and len(transfers) == 16, "이송 결과 16개 필요")
    require([item.get("box_id") for item in transfers if isinstance(item, dict)] == order and
            all(isinstance(item, dict) and item.get("passed") is True for item in transfers),
            "16개 이송 중 누락 또는 실패가 있음")
    require(result.get("scene_resets_after_initialization") == 0 and
            result.get("joint_or_object_teleport_used") is False,
            "연속 물리 실행 표시 불일치")
    require(result.get("perception_source") == "simulation_oracle" and
            result.get("end_to_end_perception_pipeline_validated") is False and
            result.get("valid_sim_hours") == 0,
            "시뮬레이터 정답 입력 또는 인지 검증 한계 표시 불일치")
    contract = result.get("runtime_contract")
    require(isinstance(contract, dict) and contract.get("perception_source") == "simulation_oracle" and
            contract.get("inspection_ai_mode") == "off",
            "실행 계약의 인지 모드 표시 불일치")
    gate = result.get("final_whole_scene_gate")
    require(isinstance(gate, dict) and gate.get("passed") is True and
            gate.get("all_boxes_at_goal") is True and gate.get("completed_ids") == order and
            gate.get("poses_restored_or_fabricated") is False,
            "마지막 전체 장면 검증 결과 불일치")
    checks = gate.get("placement_checks")
    require(isinstance(checks, list) and len(checks) == 16 and
            all(isinstance(item, dict) and item.get("passed") is True for item in checks),
            "마지막 배치 검증 16개 중 누락 또는 실패가 있음")


def verify_audit_summary(summary: dict, order: list[str]) -> None:
    """Check the compact CPU audit receipt; the full raw audit stays external."""
    require(summary.get("schema") == "ondevice_cobot.full16_audit_summary.v1" and
            summary.get("source_run") == SOURCE_RUN, "감사 요약 출처 불일치")
    require(summary.get("task_result_sha256") == TASK_RESULT_SHA256 and
            summary.get("full_audit_sha256") == FULL_AUDIT_SHA256,
            "감사 요약의 원본 SHA256 불일치")
    require(summary.get("audit_passed") is True and
            summary.get("task_complete_verified") is True and
            summary.get("verified_commits") == order,
            "감사 요약의 16개 완료 ID 불일치")
    require(summary.get("passed_checks") == 473 and summary.get("total_checks") == 473,
            "감사 요약의 473개 검사 수 불일치")
    require(summary.get("physics_launched_by_audit") is False and
            summary.get("inference_executed_by_audit") is False,
            "감사 요약의 CPU 재계산 범위 불일치")


def verify_repackaged_audit_summary(summary: dict, order: list[str]) -> None:
    """Check the new CPU audit receipt without treating its 490 checks as the old 473."""
    require(summary.get("schema") == "ondevice_cobot.full16_audit_summary.v1" and
            summary.get("source_run") == REPACKAGED_RUN,
            "재패키징 감사 요약 출처 불일치")
    require(summary.get("full_audit_sha256") == REPACKAGED_FULL_AUDIT_SHA256 and
            summary.get("evaluator_sha256") == REPACKAGED_EVALUATOR_SHA256 and
            summary.get("task_result_sha256") == REPACKAGED_TASK_RESULT_SHA256,
            "재패키징 감사 원본·평가기·실행 결과 SHA256 불일치")
    require(summary.get("audit_passed") is True and
            summary.get("task_complete_verified") is True and
            summary.get("verified_commits") == order and
            summary.get("passed_checks") == 490 and summary.get("total_checks") == 490,
            "재패키징 감사의 16개 완료 또는 490개 검사 불일치")
    require(summary.get("perception_source") == "simulation_oracle" and
            summary.get("full_perception_pipeline_validated") is False and
            summary.get("physics_launched_by_audit") is False and
            summary.get("inference_executed_by_audit") is False,
            "재패키징 감사의 oracle 또는 CPU 검증 범위 불일치")


def verify_repackaged_contract(exit_receipt: dict, result: dict, source_manifest: dict,
                                published: dict[str, str], order: list[str]) -> None:
    """Tie the new guarded run's success receipt, task result, and exact source together."""
    require(exit_receipt.get("status") == "success" and
            type(exit_receipt.get("child_returncode")) is int and
            exit_receipt["child_returncode"] == 0 and
            exit_receipt.get("gpu_launch_performed") is True,
            "재패키징 실행 종료가 성공이 아님")
    output = exit_receipt.get("output")
    require(isinstance(output, str) and PurePosixPath(output).is_absolute() and
            PurePosixPath(output).name == REPACKAGED_RUN,
            "재패키징 실행 경로 불일치")
    cleanup = exit_receipt.get("cleanup")
    require(isinstance(cleanup, dict) and cleanup.get("remaining_pids") == [],
            "재패키징 실행 자식 프로세스 정리 기록 불일치")
    require(source_manifest.get("captured_before_simulation_start") is True,
            "재패키징 실행 전 소스 스냅샷 표시 누락")
    manifest_files = file_index(source_manifest, "repackaged-full16-source-manifest.json")
    require(len(manifest_files) == 57 and manifest_files == published,
            "재패키징 실행 소스 57개가 게시 원본과 다름")
    verify_result_contract(result, order)
    for index, transfer in enumerate(result["transfer_results"], start=1):
        expected = f"{output}/transfers/{index:02d}_{transfer['box_id']}"
        require(transfer.get("cycle_directory") == expected,
                f"재패키징 실행과 이송 경로 불일치: {index}")


def verify_repackaged_run(root: Path, order: list[str]) -> None:
    published = pinned_json(root, "evidence/published-source.json", PUBLISHED_SOURCE_SHA256)
    source_manifest = pinned_json(root, "evidence/repackaged-full16-source-manifest.json",
                                  REPACKAGED_SOURCE_MANIFEST_SHA256)
    exit_receipt = pinned_json(root, "evidence/repackaged-full16-exit.json",
                               REPACKAGED_EXIT_SHA256)
    result = pinned_json(root, "evidence/repackaged-full16-task-result.json",
                         REPACKAGED_TASK_RESULT_SHA256)
    verify_repackaged_contract(exit_receipt, result, source_manifest,
                                file_index(published, "published-source.json"), order)


def modular_receipts_present(root: Path) -> bool:
    """All five completed modular receipts are required in every release checkout."""
    found = [relative for relative in MODULAR_RECEIPTS
             if (root / relative).exists() or (root / relative).is_symlink()]
    require(len(found) == len(MODULAR_RECEIPTS),
            "모듈 실행 영수증 5개가 모두 필요")
    return True


def verify_modular_contract(exit_receipt: dict, result: dict, source_manifest: dict,
                            preflight: dict, order: list[str], run_id: str) -> None:
    """Bind one guarded modular run to its command, source inventory and 16 transfers."""
    require(exit_receipt.get("status") == "success" and
            type(exit_receipt.get("child_returncode")) is int and
            exit_receipt["child_returncode"] == 0 and
            exit_receipt.get("gpu_launch_performed") is True,
            "모듈 실행 종료가 성공이 아님")
    output = exit_receipt.get("output")
    require(isinstance(output, str) and PurePosixPath(output).is_absolute() and
            PurePosixPath(output).name == run_id,
            "모듈 실행 경로 불일치")
    cleanup = exit_receipt.get("cleanup")
    require(isinstance(cleanup, dict) and cleanup.get("remaining_pids") == [],
            "모듈 실행 자식 프로세스 정리 기록 불일치")
    require(preflight.get("gpu_launch_performed") is True and
            type(preflight.get("child_pid")) is int and preflight["child_pid"] > 0 and
            preflight["child_pid"] == exit_receipt.get("child_pid"),
            "모듈 실행 사전 기록과 종료 PID 불일치")
    command = preflight.get("command")
    require(isinstance(command, list) and len(command) >= 5 and
            all(isinstance(part, str) for part in command),
            "모듈 실행 명령 없음")
    runner = PurePosixPath(command[1])
    require(runner.is_absolute() and
            runner.parts[-3:] == ("2026.Project.OnDevice_Cobot-org", "scripts", "run_task.py") and
            PurePosixPath(command[0]).is_absolute(),
            "모듈 실행 명령의 진입점 불일치")

    def option(name: str) -> str:
        require(command.count(name) == 1, f"모듈 실행 옵션 누락 또는 중복: {name}")
        index = command.index(name)
        require(index + 1 < len(command) and not command[index + 1].startswith("--"),
                f"모듈 실행 옵션 값 없음: {name}")
        return command[index + 1]

    require(option("--max-transfers") == "16" and command.count("--record-rollout") == 1,
            "모듈 실행의 16개 이송 또는 기록 옵션 불일치")
    scenario = PurePosixPath(option("--scenario"))
    require(scenario.is_absolute() and scenario.name == "scenario.json",
            "모듈 실행 시나리오 경로 불일치")
    expected_options = {
        "--inspection-view": "wrist_pallet_v1",
        "--survey-scope": "highest_layer",
        "--survey-candidate": "top_180_clear",
        "--robot-collision-profile": "link24_hull_margin10mm",
        "--payload-cover-profile": "grid10x7x7",
        "--camera-rig": "overhead_wrist_v2",
        "--motion-profile": "brisk",
    }
    for name, expected in expected_options.items():
        require(option(name) == expected, f"모듈 실행 옵션 불일치: {name}")
    require(float(option("--transport-overhead-m")) == 0.2,
            "모듈 실행 이송 높이 옵션 불일치")

    require(source_manifest.get("captured_before_simulation_start") is True,
            "모듈 실행 전 소스 스냅샷 표시 누락")
    source_files = file_index(source_manifest, "modular-full16-source-manifest.json")
    scripts = {"scripts/run_task.py", "scripts/guarded_run.py", "scripts/curobo_worker.py"}
    package_inits = {"src/depallet/__init__.py"}
    package_inits.update(f"src/depallet/{name}/__init__.py" for name in
                         ("scene", "observation", "planning", "motion", "manipulation",
                          "integration", "validation", "runtime"))
    require(len(source_files) == 66 and scripts | package_inits <= source_files.keys() and
            all(path in scripts or path.startswith("src/depallet/") for path in source_files) and
            all(not PurePosixPath(path).is_absolute() and
                all(part not in (".", "..") for part in PurePosixPath(path).parts)
                for path in source_files),
            "모듈 실행 소스 66개 목록 불일치")

    verify_result_contract(result, order)
    contract = result["runtime_contract"]
    command_contract_keys = {
        "--inspection-view": "inspection_view",
        "--survey-scope": "survey_scope",
        "--survey-candidate": "survey_candidate",
        "--robot-collision-profile": "robot_collision_profile",
        "--payload-cover-profile": "payload_cover_profile",
        "--camera-rig": "camera_rig",
        "--motion-profile": "motion_profile",
    }
    require(all(contract.get(key) == option(flag)
                for flag, key in command_contract_keys.items()) and
            contract.get("transport_overhead_m") == 0.2,
            "모듈 실행 명령과 최종 실행 계약 불일치")
    for index, transfer in enumerate(result["transfer_results"], start=1):
        expected = f"{output}/transfers/{index:02d}_{transfer['box_id']}"
        require(transfer.get("cycle_directory") == expected,
                f"모듈 실행과 이송 경로 불일치: {index}")
        require(all(isinstance(transfer.get(key), str) and
                    SHA256_PATTERN.fullmatch(transfer[key]) is not None
                    for key in ("execution_sha256", "snapshot_sha256")),
                f"모듈 실행 이송 원본 해시 누락: {index}")


def verify_modular_audit_summary(summary: dict, order: list[str], *, run_id: str,
                                 task_result_sha256: str, full_audit_sha256: str,
                                 evaluator_sha256: str, expected_checks: int) -> None:
    """Check a separate modular CPU audit, without inheriting the old 473/490 counts."""
    require(summary.get("schema") == "ondevice_cobot.full16_audit_summary.v1" and
            summary.get("source_run") == run_id,
            "모듈 실행 감사 요약 출처 불일치")
    require(summary.get("task_result_sha256") == task_result_sha256 and
            summary.get("full_audit_sha256") == full_audit_sha256 and
            summary.get("evaluator_sha256") == evaluator_sha256 and
            summary.get("scenario_sha256") == SCENARIO_SHA256 and
            summary.get("task_plan_sha256") == RULE_PLAN_SHA256,
            "모듈 실행 감사 원본·평가기·입력 SHA256 불일치")
    require(summary.get("audit_passed") is True and
            summary.get("task_complete_verified") is True and
            summary.get("verified_commits") == order and
            type(expected_checks) is int and expected_checks > 0 and
            summary.get("passed_checks") == expected_checks and
            summary.get("total_checks") == expected_checks,
            "모듈 실행 감사의 16개 완료 또는 검사 수 불일치")
    require(summary.get("perception_source") == "simulation_oracle" and
            summary.get("full_perception_pipeline_validated") is False and
            summary.get("physics_launched_by_audit") is False and
            summary.get("inference_executed_by_audit") is False,
            "모듈 실행 감사의 oracle 또는 CPU 검증 범위 불일치")


def verify_modular_run(root: Path, order: list[str]) -> None:
    hashes = (MODULAR_TASK_RESULT_SHA256, MODULAR_EXIT_SHA256,
              MODULAR_SOURCE_MANIFEST_SHA256, MODULAR_PREFLIGHT_SHA256,
              MODULAR_AUDIT_SUMMARY_SHA256, MODULAR_FULL_AUDIT_SHA256,
              MODULAR_EVALUATOR_SHA256)
    require(MODULAR_RUN.startswith("full-") and
            all(SHA256_PATTERN.fullmatch(value) is not None for value in hashes) and
            MODULAR_AUDIT_CHECKS > 0,
            "모듈 실행 성공·감사 확정 후 TODO 검증 상수를 채워야 함")
    result, exit_receipt, source_manifest, preflight, summary = (
        pinned_json(root, relative, digest)
        for relative, digest in zip(MODULAR_RECEIPTS, hashes[:5]))
    verify_modular_contract(exit_receipt, result, source_manifest, preflight, order,
                            MODULAR_RUN)
    verify_modular_audit_summary(
        summary, order, run_id=MODULAR_RUN,
        task_result_sha256=MODULAR_TASK_RESULT_SHA256,
        full_audit_sha256=MODULAR_FULL_AUDIT_SHA256,
        evaluator_sha256=MODULAR_EVALUATOR_SHA256,
        expected_checks=MODULAR_AUDIT_CHECKS,
    )


def verify_release_source_equivalence(root: Path, source_manifest: dict) -> None:
    """Opt-in: compare the currently editable tree to the captured modular run."""
    relative = "evidence/modular-source-manifest.json"
    current_manifest = json.loads(repository_file(root, relative).read_text(encoding="utf-8"))
    require(isinstance(current_manifest, dict) and
            current_manifest.get("schema") == "ondevice_cobot.modular_source.v1",
            "현재 모듈 소스 manifest 스키마 불일치")
    released = file_index(source_manifest, "modular-full16-source-manifest.json")
    current = file_index(current_manifest, relative)
    require(current == released, "현재 모듈 소스와 실행 당시 소스 manifest 불일치")
    for path, digest in released.items():
        require(sha256_file(repository_file(root, path)) == digest,
                f"현재 모듈 소스와 실행 당시 SHA256 불일치: {path}")


def verify_repository(root: Path, *, check_current: bool = False) -> tuple[int, int]:
    root = root.resolve()
    require(root.is_dir(), f"저장소 폴더 없음: {root}")
    source_count = verify_sources(root, check_current=check_current)
    scenario = pinned_json(root, "examples/v1_uniform/scenario.json", SCENARIO_SHA256)
    plan = pinned_json(root, "examples/v1_uniform/rule-plan.json", RULE_PLAN_SHA256)
    result = pinned_json(root, "evidence/full16-task-result.json", TASK_RESULT_SHA256)
    summary = pinned_json(root, "evidence/full16-audit-summary.json", AUDIT_SUMMARY_SHA256)
    order = verify_scenario_contract(scenario, plan)
    verify_result_contract(result, order)
    verify_audit_summary(summary, order)
    verify_repackaged_run(root, order)
    repackaged_summary = pinned_json(root, "evidence/repackaged-full16-audit-summary.json",
                                     REPACKAGED_AUDIT_SUMMARY_SHA256)
    verify_repackaged_audit_summary(repackaged_summary, order)
    modular_receipts_present(root)
    verify_modular_run(root, order)
    return source_count, len(order)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="검증할 저장소 루트 (기본값: 이 스크립트의 상위 저장소)")
    parser.add_argument("--check-current", action="store_true",
                        help="현재 모듈식 src/와 실행 스크립트의 manifest·import 일치 검사 (물리 재실행 증명 아님)")
    parser.add_argument("--check-release-source", action="store_true",
                        help="완료된 모듈 실행의 소스 66개와 현재 체크아웃 동등성 검사")
    args = parser.parse_args(argv)
    try:
        source_count, box_count = verify_repository(
            args.root, check_current=args.check_current or args.check_release_source)
        if args.check_release_source:
            source_manifest = pinned_json(
                args.root, MODULAR_RECEIPTS[2], MODULAR_SOURCE_MANIFEST_SHA256)
            verify_release_source_equivalence(args.root, source_manifest)
    except (EvidenceError, OSError, ValueError, TypeError, KeyError, SyntaxError, tarfile.TarError) as error:
        print(f"근거 검증 실패: {error}", file=sys.stderr)
        return 1
    print(f"근거 검증 통과: 보관 소스 {source_count}개 SHA256, V1 예시 {box_count}개, "
          "원본·재패키징 저장 결과 각각 16/16, 원본 CPU 감사 473/473, "
          "재패키징 실행 CPU 감사 490/490 (시뮬레이터 정답 입력).")
    print(f"모듈 실행 저장 결과 16/16 및 별도 CPU 감사 "
          f"{MODULAR_AUDIT_CHECKS}/{MODULAR_AUDIT_CHECKS} 검증 통과 "
          "(시뮬레이터 정답 입력).")
    if args.check_current:
        print("현재 모듈식 소스 manifest·import 검사 통과. "
              "이 검사는 물리 재실행이나 과거 실행 소스와의 동등성 증명이 아닙니다.")
    if args.check_release_source:
        print("현재 모듈 소스 66개가 저장된 모듈 실행의 시작 전 소스와 SHA256으로 일치합니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
