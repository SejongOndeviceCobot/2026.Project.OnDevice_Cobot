#!/usr/bin/env python3
"""Run the fixed, successful V1 task recipe through the original GPU guard."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from check_assets import PreflightError, validate

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parents[2]
CACHE = WORKSPACE / "cache" / PROJECT.name
RUNS = WORKSPACE / "runs" / PROJECT.name


def command_for(stage: str, scenario: Path, gpu: str, shared: bool, slot: int | None, output: Path) -> list[str]:
    command = [sys.executable, str(PROJECT / "scripts/guarded_run.py"), "--gpu", gpu,
               "--output", str(output)]
    command += ["--task-seconds", "7200"] if stage == "full" else ["--seconds", "600"]
    if shared:
        command.append("--allow-shared-gpu")
    if slot is not None:
        command += ["--parallel-slot", str(slot)]
    command += ["--", str(CACHE / "venv/bin/python"), str(PROJECT / "scripts/run_task.py"),
                "--scenario", str(scenario), "--max-transfers", "16" if stage == "full" else "1",
                "--inspection-view", "wrist_pallet_v1", "--survey-scope", "highest_layer",
                "--survey-candidate", "top_180_clear", "--robot-collision-profile", "link24_hull_margin10mm",
                "--payload-cover-profile", "grid10x7x7", "--transport-overhead-m", ".20",
                "--camera-rig", "overhead_wrist_v2", "--motion-profile", "brisk",
                "--preview-hz", "1", "--recording-compression-level", "1",
                "--recording-workers", "2", "--record-rollout", "--max-steps", "432000",
                "--frames", "54000"]
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "full"))
    parser.add_argument("--gpu", default="0", help="Exact GPU index or UUID")
    parser.add_argument("--allow-shared-gpu", action="store_true")
    parser.add_argument("--parallel-slot", type=int, choices=(0, 1), help="Original run used shared slot 0")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print command; no GPU launch")
    args = parser.parse_args(argv)
    if args.parallel_slot is not None and not args.allow_shared_gpu:
        parser.error("--parallel-slot requires --allow-shared-gpu")
    if Path(os.environ.get("JCLEE_WORKSPACE", str(WORKSPACE))).resolve() != WORKSPACE:
        parser.error("JCLEE_WORKSPACE must match the checkout workspace")
    try:
        assets = validate()
    except (PreflightError, OSError, ValueError) as error:
        print(f"asset preflight: {error}", file=sys.stderr)
        return 2
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = RUNS / f"{args.stage}-{stamp}"
    command = command_for(args.stage, Path(assets["scenario"]), args.gpu, args.allow_shared_gpu, args.parallel_slot, output)
    interpreter = CACHE / "venv/bin/python"
    curobo = CACHE / "curobo-venv/bin/python"
    report = {"stage": args.stage, "scenario": assets["scenario"], "output": str(output),
              "command": command, "launch_requested": not args.dry_run, "dry_run": args.dry_run,
              "isaac_venv_present": interpreter.is_file(), "curobo_venv_present": curobo.is_file(),
              "asset_preflight_passed": True}
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if args.dry_run:
        return 0
    missing = [name for name in ("nvidia-smi", "ffmpeg", "ffprobe") if not shutil.which(name)]
    if missing:
        parser.error("Missing system tools: " + ", ".join(missing))
    if not interpreter.is_file() or not curobo.is_file():
        parser.error("Install the Isaac and cuRobo environments before GPU execution")
    environment = dict(os.environ, JCLEE_WORKSPACE=str(WORKSPACE), ISAAC_P0_PROJECT=str(PROJECT),
                       ISAAC_P0_CACHE=str(CACHE), ISAAC_P0_RUNS=str(RUNS),
                       ISAAC_P0_DATA=str(assets["data_root"]), PYTHONPATH=str(PROJECT / "src"),
                       PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run(command, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
