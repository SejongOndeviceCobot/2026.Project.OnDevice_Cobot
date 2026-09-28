#!/usr/bin/env python3
"""Record the current modular source layout and hashes for review and CI."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAUNCHERS = ("run_task.py", "guarded_run.py", "curobo_worker.py")


def main() -> int:
    module_sources = sorted((ROOT / "src" / "depallet").rglob("*.py"))
    if not module_sources:
        raise FileNotFoundError("src/depallet Python sources are missing")
    sources = module_sources + [ROOT / "scripts" / name for name in LAUNCHERS]
    files = []
    for path in sources:
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"Invalid source path: {path}")
        files.append({"path": path.relative_to(ROOT).as_posix(),
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    manifest = {"schema": "ondevice_cobot.modular_source.v1", "files": files}
    target = ROOT / "evidence" / "modular-source-manifest.json"
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Updated {target.relative_to(ROOT)}: {len(files)} Python files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
