#!/usr/bin/env python3
"""Install isolated, pinned Isaac Sim and cuRobo environments.

The command downloads Python packages but never launches a GPU process or
changes an NVIDIA driver.  It requires an explicitly pinned cuRobo checkout.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
REQUIREMENTS = PROJECT / "requirements"
CUROBO_COMMIT = "78fd485fa82d9b9a063fb4985e371814587e666a"
MIN_FREE_BYTES = 100 * 1024**3


def _pinned_curobo(source: Path) -> Path:
    try:
        source = source.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"cuRobo source does not exist: {source}") from exc
    if not source.is_dir() or source.is_symlink():
        raise ValueError(f"cuRobo source must be a regular directory: {source}")
    command = ["git", "-C", str(source), "rev-parse", "HEAD"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ValueError("cuRobo source must be a Git checkout")
    if result.stdout.strip() != CUROBO_COMMIT:
        raise ValueError(f"cuRobo must be checked out at {CUROBO_COMMIT}")
    status = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise ValueError("cuRobo source must have a clean Git worktree")
    return source


def _regular_lock(name: str) -> Path:
    path = REQUIREMENTS / name
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing pinned requirement lock: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curobo-source", type=Path, required=True)
    parser.add_argument(
        "--accept-nvidia-eula",
        action="store_true",
        help="Confirm that the operator has read and accepts the NVIDIA Isaac Sim EULA.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        source = _pinned_curobo(args.curobo_source)
        isaac_lock = _regular_lock("isaac-requirements.lock")
        curobo_lock = _regular_lock("curobo-requirements.lock")
    except ValueError as exc:
        parser.error(str(exc))
    if not args.dry_run and not args.accept_nvidia_eula:
        parser.error("Read and accept the NVIDIA Isaac Sim EULA, then pass --accept-nvidia-eula.")
    uv = shutil.which("uv")
    if not uv:
        parser.error("uv is required to install the pinned environments.")

    local = PROJECT.parents[2] / "cache" / PROJECT.name
    environments = local
    if not args.dry_run and shutil.disk_usage(local.parent).free < MIN_FREE_BYTES:
        parser.error("Keep at least 100 GiB free for the GPU environments and runtime cache.")
    indexes = [
        "--extra-index-url", "https://pypi.nvidia.com",
        "--extra-index-url", "https://download.pytorch.org/whl/cu130",
        "--index-strategy", "unsafe-best-match",
    ]
    commands: list[list[str]] = []
    for name, lock in (("venv", isaac_lock), ("curobo-venv", curobo_lock)):
        environment = environments / name
        if environment.is_symlink():
            parser.error(f"Refusing to install into symlinked environment: {environment}")
        python = environment / "bin/python"
        if not (environment / "pyvenv.cfg").is_file():
            commands.append([uv, "venv", "--python", "3.12", str(environment)])
        commands.append([uv, "pip", "sync", "--python", str(python), str(lock), *indexes])
        if name == "curobo-venv":
            commands.append([
                uv, "pip", "install", "--python", str(python), "--no-deps",
                "--no-build-isolation", str(source),
            ])
        commands.append([uv, "pip", "check", "--python", str(python)])

    print(json.dumps({
        "project": str(PROJECT),
        "curobo_source": str(source),
        "curobo_commit": CUROBO_COMMIT,
        "gpu_launched": False,
        "dry_run": args.dry_run,
        "commands": commands,
    }, indent=2))
    if args.dry_run:
        return 0
    environments.mkdir(parents=True, exist_ok=True)
    environment = dict(
        os.environ,
        UV_CACHE_DIR=str(local / "uv-cache"),
        UV_PYTHON_INSTALL_DIR=str(local / "python"),
        UV_CONCURRENT_DOWNLOADS="2",
        UV_CONCURRENT_BUILDS="2",
        UV_CONCURRENT_INSTALLS="2",
        OMNI_KIT_ACCEPT_EULA="YES",
        PYTHONNOUSERSITE="1",
        SETUPTOOLS_SCM_PRETEND_VERSION="0.8.0.post1.dev43",
    )
    for command in commands:
        subprocess.run(command, env=environment, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
