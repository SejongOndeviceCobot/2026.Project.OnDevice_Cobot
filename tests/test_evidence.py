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


if __name__ == "__main__":
    unittest.main()
