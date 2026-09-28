"""Tampering checks for the archived full16 release evidence."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import verify_evidence  # noqa: E402


def load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def copy_release(destination: Path, *, include_current: bool = False) -> None:
    paths = [
        "evidence/full16-source.tar.gz",
        "evidence/published-source.json",
        "evidence/code-snapshot-manifest.json",
        "evidence/full16-task-result.json",
        "evidence/full16-audit-summary.json",
        "evidence/repackaged-full16-task-result.json",
        "evidence/repackaged-full16-exit.json",
        "evidence/repackaged-full16-source-manifest.json",
        "evidence/repackaged-full16-audit-summary.json",
        *verify_evidence.MODULAR_RECEIPTS,
        "examples/v1_uniform/scenario.json",
        "examples/v1_uniform/rule-plan.json",
    ]
    if include_current:
        paths += [file.relative_to(ROOT).as_posix()
                  for file in sorted((ROOT / "src/depallet").rglob("*.py"))]
        paths += ["scripts/run_task.py", "scripts/guarded_run.py",
                  "scripts/curobo_worker.py"]
    for relative in paths:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    if include_current:
        entries = [{"path": relative,
                    "sha256": hashlib.sha256((destination / relative).read_bytes()).hexdigest()}
                   for relative in sorted(set(paths))
                   if relative.startswith(("src/", "scripts/"))]
        (destination / "evidence/modular-source-manifest.json").write_text(
            json.dumps({"schema": "ondevice_cobot.modular_source.v1",
                        "files": entries}, indent=2) + "\n", encoding="utf-8")


def modular_fixture() -> tuple[dict, dict, dict, dict, list[str], str]:
    """Synthetic receipt set: contract tests need no unfinished GPU run."""
    run_id = "full-test-modular"
    exit_receipt = copy.deepcopy(load("evidence/repackaged-full16-exit.json"))
    result = copy.deepcopy(load("evidence/repackaged-full16-task-result.json"))
    old_output = exit_receipt["output"]
    output = f"/DATA/jclee/workspace/runs/2026.Project.OnDevice_Cobot-org/{run_id}"
    exit_receipt["output"] = output
    for transfer in result["transfer_results"]:
        transfer["cycle_directory"] = transfer["cycle_directory"].replace(old_output, output, 1)
    source_manifest = {
        "captured_before_simulation_start": True,
        "files": copy.deepcopy(load("evidence/modular-full16-source-manifest.json")["files"]),
    }
    preflight = {
        "gpu_launch_performed": True,
        "child_pid": exit_receipt["child_pid"],
        "command": [
            "/home/jclee/workspace/cache/2026.Project.OnDevice_Cobot-org/venv/bin/python",
            "/home/jclee/workspace/repos/own/2026.Project.OnDevice_Cobot-org/scripts/run_task.py",
            "--scenario", "/DATA/jclee/workspace/data/v1_uniform/scenario.json",
            "--max-transfers", "16",
            "--inspection-view", "wrist_pallet_v1",
            "--survey-scope", "highest_layer",
            "--survey-candidate", "top_180_clear",
            "--robot-collision-profile", "link24_hull_margin10mm",
            "--payload-cover-profile", "grid10x7x7",
            "--transport-overhead-m", ".20",
            "--camera-rig", "overhead_wrist_v2",
            "--motion-profile", "brisk",
            "--record-rollout",
        ],
    }
    order = load("examples/v1_uniform/rule-plan.json")["order"]
    return exit_receipt, result, source_manifest, preflight, order, run_id


class EvidenceTests(unittest.TestCase):
    def test_archived_release_is_consistent(self) -> None:
        self.assertEqual(verify_evidence.verify_repository(ROOT), (57, 16))

    def test_archive_tampering_fails_default_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            copy_release(checkout)
            with (checkout / "evidence/full16-source.tar.gz").open("ab") as stream:
                stream.write(b"changed")
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "tools/verify_evidence.py"),
                 "--root", str(checkout)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("full16-source.tar.gz", result.stderr)
            self.assertIn("SHA256", result.stderr)

    def test_live_edit_is_allowed_by_default_but_caught_on_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            copy_release(checkout, include_current=True)
            with (checkout / "src/depallet/runtime/task_runtime.py").open("a", encoding="utf-8") as stream:
                stream.write("\n# changed after the successful run\n")
            base = [sys.executable, "-B", str(ROOT / "tools/verify_evidence.py"),
                    "--root", str(checkout)]
            default = subprocess.run(base, capture_output=True, text=True, check=False)
            current = subprocess.run([*base, "--check-current"],
                                     capture_output=True, text=True, check=False)
            self.assertEqual(default.returncode, 0, default.stderr)
            self.assertEqual(current.returncode, 1)
            self.assertIn("src/depallet/runtime/task_runtime.py", current.stderr)
            self.assertIn("SHA256", current.stderr)

    def test_modular_checkout_matches_current_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            copy_release(checkout, include_current=True)
            self.assertEqual(
                verify_evidence.verify_repository(checkout, check_current=True),
                (57, 16),
            )

    def test_current_manifest_requires_all_modular_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            copy_release(checkout, include_current=True)
            manifest = checkout / "evidence/modular-source-manifest.json"
            value = json.loads(manifest.read_text(encoding="utf-8"))
            value["files"] = [entry for entry in value["files"]
                              if entry["path"] != "src/depallet/runtime/task_runtime.py"]
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(verify_evidence.EvidenceError, "소스 목록"):
                verify_evidence.verify_repository(checkout, check_current=True)

    def test_current_manifest_cannot_hide_flat_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            copy_release(checkout, include_current=True)
            source = checkout / "src/depallet/runtime/task_runtime.py"
            with source.open("a", encoding="utf-8") as stream:
                stream.write("\nfrom task_runtime import main\n")
            manifest = checkout / "evidence/modular-source-manifest.json"
            value = json.loads(manifest.read_text(encoding="utf-8"))
            for entry in value["files"]:
                if entry["path"] == "src/depallet/runtime/task_runtime.py":
                    entry["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(verify_evidence.EvidenceError, "이전 평면 모듈 import"):
                verify_evidence.verify_repository(checkout, check_current=True)

    def test_repackaged_exit_tampering_fails_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            copy_release(checkout)
            receipt = checkout / "evidence/repackaged-full16-exit.json"
            value = json.loads(receipt.read_text(encoding="utf-8"))
            value["status"] = "failed"
            receipt.write_text(json.dumps(value), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "tools/verify_evidence.py"),
                 "--root", str(checkout)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("repackaged-full16-exit.json", result.stderr)

    def test_repackaged_success_requires_exit_and_matching_source(self) -> None:
        exit_receipt = load("evidence/repackaged-full16-exit.json")
        result = load("evidence/repackaged-full16-task-result.json")
        manifest = load("evidence/repackaged-full16-source-manifest.json")
        published = {entry["path"]: entry["sha256"]
                     for entry in load("evidence/published-source.json")["files"]}
        order = load("examples/v1_uniform/rule-plan.json")["order"]

        failed_exit = copy.deepcopy(exit_receipt)
        failed_exit["child_returncode"] = 1
        with self.assertRaisesRegex(verify_evidence.EvidenceError, "종료"):
            verify_evidence.verify_repackaged_contract(
                failed_exit, result, manifest, published, order)

        wrong_source = copy.deepcopy(manifest)
        wrong_source["files"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(verify_evidence.EvidenceError, "소스 57개"):
            verify_evidence.verify_repackaged_contract(
                exit_receipt, result, wrong_source, published, order)

    def test_failed_transfer_cannot_be_reported_as_full_success(self) -> None:
        result = copy.deepcopy(load("evidence/full16-task-result.json"))
        order = load("examples/v1_uniform/rule-plan.json")["order"]
        result["transfer_results"][7]["passed"] = False
        with self.assertRaisesRegex(verify_evidence.EvidenceError, "이송"):
            verify_evidence.verify_result_contract(result, order)

    def test_audit_summary_cannot_claim_wrong_verified_count(self) -> None:
        summary = copy.deepcopy(load("evidence/full16-audit-summary.json"))
        order = load("examples/v1_uniform/rule-plan.json")["order"]
        summary["passed_checks"] = 472
        with self.assertRaisesRegex(verify_evidence.EvidenceError, "473"):
            verify_evidence.verify_audit_summary(summary, order)

    def test_new_audit_cannot_claim_wrong_verified_count(self) -> None:
        summary = copy.deepcopy(load("evidence/repackaged-full16-audit-summary.json"))
        order = load("examples/v1_uniform/rule-plan.json")["order"]
        summary["passed_checks"] = 489
        with self.assertRaisesRegex(verify_evidence.EvidenceError, "490"):
            verify_evidence.verify_repackaged_audit_summary(summary, order)

    def test_plan_cannot_silently_omit_a_scenario_box(self) -> None:
        scenario = load("examples/v1_uniform/scenario.json")
        plan = copy.deepcopy(load("examples/v1_uniform/rule-plan.json"))
        plan["order"][-1] = plan["order"][0]
        with self.assertRaisesRegex(verify_evidence.EvidenceError, "계획 순서"):
            verify_evidence.verify_scenario_contract(scenario, plan)

    def test_modular_audit_summary_tampering_fails_default_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            copy_release(checkout)
            summary = checkout / "evidence/modular-full16-audit-summary.json"
            value = json.loads(summary.read_text(encoding="utf-8"))
            value["passed_checks"] = 489
            summary.write_text(json.dumps(value), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-B", str(ROOT / "tools/verify_evidence.py"),
                 "--root", str(checkout)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("modular-full16-audit-summary.json", result.stderr)
            self.assertIn("SHA256", result.stderr)

    def test_modular_contract_binds_command_and_transfer_paths(self) -> None:
        exit_receipt, result, source, preflight, order, run_id = modular_fixture()
        verify_evidence.verify_modular_contract(
            exit_receipt, result, source, preflight, order, run_id)

        wrong_command = copy.deepcopy(preflight)
        index = wrong_command["command"].index("--max-transfers")
        wrong_command["command"][index + 1] = "8"
        with self.assertRaisesRegex(verify_evidence.EvidenceError, "16개 이송"):
            verify_evidence.verify_modular_contract(
                exit_receipt, result, source, wrong_command, order, run_id)

        wrong_contract = copy.deepcopy(result)
        wrong_contract["runtime_contract"]["motion_profile"] = "other"
        with self.assertRaisesRegex(verify_evidence.EvidenceError, "실행 계약"):
            verify_evidence.verify_modular_contract(
                exit_receipt, wrong_contract, source, preflight, order, run_id)

        wrong_result = copy.deepcopy(result)
        wrong_result["transfer_results"][7]["cycle_directory"] = "/another-run/transfers/08_box_15"
        with self.assertRaisesRegex(verify_evidence.EvidenceError, "이송 경로"):
            verify_evidence.verify_modular_contract(
                exit_receipt, wrong_result, source, preflight, order, run_id)

        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            copy_release(checkout)
            (checkout / verify_evidence.MODULAR_RECEIPTS[0]).unlink()
            with self.assertRaisesRegex(verify_evidence.EvidenceError, "영수증 5개"):
                verify_evidence.verify_repository(checkout)

    def test_modular_contract_rejects_failed_exit_and_bad_source_inventory(self) -> None:
        exit_receipt, result, source, preflight, order, run_id = modular_fixture()
        failed_exit = copy.deepcopy(exit_receipt)
        failed_exit["child_returncode"] = 1
        with self.assertRaisesRegex(verify_evidence.EvidenceError, "종료"):
            verify_evidence.verify_modular_contract(
                failed_exit, result, source, preflight, order, run_id)

        missing_source = copy.deepcopy(source)
        missing_source["files"].pop()
        with self.assertRaisesRegex(verify_evidence.EvidenceError, "소스 66개"):
            verify_evidence.verify_modular_contract(
                exit_receipt, result, missing_source, preflight, order, run_id)

        late_snapshot = copy.deepcopy(source)
        late_snapshot["captured_before_simulation_start"] = False
        with self.assertRaisesRegex(verify_evidence.EvidenceError, "실행 전"):
            verify_evidence.verify_modular_contract(
                exit_receipt, result, late_snapshot, preflight, order, run_id)

    def test_modular_audit_requires_exact_input_hashes_and_check_count(self) -> None:
        _, _, _, _, order, run_id = modular_fixture()
        summary = {
            "schema": "ondevice_cobot.full16_audit_summary.v1",
            "source_run": run_id,
            "task_result_sha256": "1" * 64,
            "full_audit_sha256": "2" * 64,
            "evaluator_sha256": "3" * 64,
            "scenario_sha256": verify_evidence.SCENARIO_SHA256,
            "task_plan_sha256": verify_evidence.RULE_PLAN_SHA256,
            "audit_passed": True,
            "task_complete_verified": True,
            "verified_commits": order,
            "passed_checks": 17,
            "total_checks": 17,
            "perception_source": "simulation_oracle",
            "full_perception_pipeline_validated": False,
            "physics_launched_by_audit": False,
            "inference_executed_by_audit": False,
        }
        expected = dict(run_id=run_id, task_result_sha256="1" * 64,
                        full_audit_sha256="2" * 64, evaluator_sha256="3" * 64,
                        expected_checks=17)
        verify_evidence.verify_modular_audit_summary(summary, order, **expected)
        wrong_input = copy.deepcopy(summary)
        wrong_input["task_plan_sha256"] = "0" * 64
        with self.assertRaisesRegex(verify_evidence.EvidenceError, "입력 SHA256"):
            verify_evidence.verify_modular_audit_summary(wrong_input, order, **expected)
        wrong_count = copy.deepcopy(summary)
        wrong_count["passed_checks"] = 16
        with self.assertRaisesRegex(verify_evidence.EvidenceError, "검사 수"):
            verify_evidence.verify_modular_audit_summary(wrong_count, order, **expected)

    def test_release_source_catches_joint_source_and_manifest_edit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            copy_release(checkout, include_current=True)
            manifest = checkout / "evidence/modular-source-manifest.json"
            value = json.loads(manifest.read_text(encoding="utf-8"))
            run_snapshot = {"captured_before_simulation_start": True,
                            "files": copy.deepcopy(value["files"])}
            verify_evidence.verify_release_source_equivalence(checkout, run_snapshot)

            source = checkout / "src/depallet/runtime/task_runtime.py"
            with source.open("a", encoding="utf-8") as stream:
                stream.write("\n# intentional future edit\n")
            for entry in value["files"]:
                if entry["path"] == "src/depallet/runtime/task_runtime.py":
                    entry["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest.write_text(json.dumps(value), encoding="utf-8")
            self.assertEqual(verify_evidence.verify_repository(checkout, check_current=True),
                             (57, 16))
            with self.assertRaisesRegex(verify_evidence.EvidenceError,
                                        "실행 당시 소스 manifest"):
                verify_evidence.verify_release_source_equivalence(checkout, run_snapshot)


if __name__ == "__main__":
    unittest.main()
