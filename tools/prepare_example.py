#!/usr/bin/env python3
"""Copy the reviewed V1 scenario into the external data root without replacing files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parents[2]
EXAMPLE = PROJECT / "examples/v1_uniform"
RELATIVE = Path("scenario-suite-v1/v1_uniform_box16_first_box13_early_final_layer_v1")


def prepare(*, check: bool = False) -> dict[str, object]:
    configured = Path(os.environ.get("JCLEE_WORKSPACE", str(WORKSPACE))).resolve()
    if configured != WORKSPACE:
        raise ValueError("JCLEE_WORKSPACE must match the checkout workspace")
    data = WORKSPACE / "data/depallet_isaac_p0"
    destination = data / RELATIVE
    if not check:
        destination.mkdir(parents=True, exist_ok=True)
    if not destination.is_dir() or destination.is_symlink():
        raise ValueError(f"Example data directory is unavailable: {destination}")
    files: dict[str, str] = {}
    written = 0
    for name in ("scenario.json", "rule-plan.json"):
        source, target = EXAMPLE / name, destination / name
        if source.is_symlink() or not source.is_file() or target.is_symlink():
            raise ValueError(f"Unsafe example file: {name}")
        expected = source.read_bytes()
        if target.exists():
            if not target.is_file() or target.read_bytes() != expected:
                raise ValueError(f"Existing data differs; refusing to overwrite: {target}")
        elif check:
            raise ValueError(f"Prepare the example first: {target}")
        else:
            with target.open("xb") as output:
                output.write(expected)
            written += 1
        files[name] = hashlib.sha256(expected).hexdigest()
    return {"scenario": str(destination / "scenario.json"), "rule_plan": str(destination / "rule-plan.json"),
            "sha256": files, "wrote_files": written}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify that the example is already prepared")
    args = parser.parse_args()
    try:
        result = prepare(check=args.check)
    except (OSError, ValueError) as error:
        print(f"example: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
