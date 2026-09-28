"""The packaged input must pass the same complete-plan gate as the Isaac runner."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from task_runtime import validate_runtime_plan_catalog  # noqa: E402


class PlanGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        example = ROOT / "examples/v1_uniform"
        cls.scenario = json.loads((example / "scenario.json").read_text())
        cls.plan = json.loads((example / "rule-plan.json").read_text())

    def test_packaged_full_plan_is_admitted(self) -> None:
        count = validate_runtime_plan_catalog(
            self.scenario["spec"], self.plan["packing"], self.plan["order"])
        self.assertEqual(count, 16)

    def test_missing_box_is_rejected_before_isaac(self) -> None:
        damaged = copy.deepcopy(self.plan)
        damaged["order"].pop()
        with self.assertRaisesRegex(ValueError, "complete 16-box goal layout"):
            validate_runtime_plan_catalog(self.scenario["spec"], damaged["packing"], damaged["order"])


if __name__ == "__main__":
    unittest.main()
