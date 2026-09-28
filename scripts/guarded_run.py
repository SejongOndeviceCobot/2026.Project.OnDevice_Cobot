#!/usr/bin/env python3
"""Conservative, bounded launch of this project's one Isaac P0 runner.

No GPU work is performed by this module or its --help command. NVIDIA queries
are read-only. This is cooperative monitoring, not a GPU reservation or cgroup.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import contextmanager, ExitStack
from functools import partial
import fcntl
import io
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Callable

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parents[2]
CACHE = WORKSPACE / "cache" / PROJECT.name
RUNS = WORKSPACE / "runs" / PROJECT.name
GIB = 1024 ** 3
MIB = 1024 ** 2
MAX_GPU_MIB = 12 * 1024
MAX_RSS_BYTES = 16 * GIB
MIN_DISK_BYTES = 30 * GIB
QUERY_TIMEOUT = 2.0
PAGE_BYTES = os.sysconf("SC_PAGE_SIZE")
CLK_TCK = os.sysconf("SC_CLK_TCK")


class GuardError(RuntimeError):
    pass


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    total_mib: int
    used_mib: int
    utilization: int


@dataclass(frozen=True)
class GPUProcess:
    uuid: str
    pid: int
    used_mib: int


@dataclass(frozen=True)
class Proc:
    pid: int
    uid: int
    pgid: int
    sid: int
    start_ticks: int
    rss_bytes: int
    state: str


@dataclass(frozen=True)
class Group:
    pgid: int
    uid: int
    min_start_ticks: int


def _number(value: str, label: str) -> int:
    value = value.strip()
    if not value.isascii() or not value.isdigit():
        raise GuardError(f"Invalid NVIDIA {label}: {value!r}")
    return int(value)


def parse_gpus(raw: str) -> list[GPU]:
    rows = list(csv.reader(io.StringIO(raw)))
    result = []
    for row in rows:
        if not row or not any(part.strip() for part in row):
            continue
        if len(row) != 5:
            raise GuardError("Unexpected NVIDIA device CSV schema")
        index, uuid, total, used, utilization = (part.strip() for part in row)
        gpu = GPU(_number(index, "index"), uuid, _number(total, "total memory"),
                  _number(used, "used memory"), _number(utilization, "utilization"))
        if not uuid.startswith("GPU-") or len(uuid) < 8:
            raise GuardError("Invalid NVIDIA device UUID")
        if gpu.total_mib <= 0 or gpu.used_mib > gpu.total_mib or gpu.utilization > 100:
            raise GuardError("Invalid NVIDIA device counters")
        result.append(gpu)
    if not result or len({g.index for g in result}) != len(result) or len({g.uuid for g in result}) != len(result):
        raise GuardError("Missing or duplicate NVIDIA devices")
    return result


def parse_apps(raw: str) -> list[GPUProcess]:
    result = []
    for row in csv.reader(io.StringIO(raw)):
        if not row or not any(part.strip() for part in row):
            continue
        if len(row) != 3:
            raise GuardError("Unexpected NVIDIA process CSV schema")
        uuid, pid, memory = (part.strip() for part in row)
        app = GPUProcess(uuid, _number(pid, "PID"), _number(memory, "process memory"))
        if not uuid.startswith("GPU-") or len(uuid) < 8 or app.pid <= 0:
            raise GuardError("Invalid NVIDIA process identity")
        result.append(app)
    if len({(p.uuid, p.pid) for p in result}) != len(result):
        raise GuardError("Duplicate NVIDIA process rows")
    return result


def query_gpu(selector: str, run: Callable = subprocess.run) -> tuple[GPU, list[GPUProcess]]:
    def query(fields: str) -> str:
        try:
            out = run(["nvidia-smi", fields, "--format=csv,noheader,nounits"],
                      capture_output=True, text=True, timeout=QUERY_TIMEOUT, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GuardError(f"NVIDIA query unavailable: {type(exc).__name__}") from exc
        if out.returncode != 0:
            raise GuardError(f"NVIDIA query returned {out.returncode}")
        return out.stdout
    gpus = parse_gpus(query("--query-gpu=index,uuid,memory.total,memory.used,utilization.gpu"))
    apps = parse_apps(query("--query-compute-apps=gpu_uuid,pid,used_gpu_memory"))
    matches = [g for g in gpus if selector == str(g.index) or selector == g.uuid]
    if len(matches) != 1:
        raise GuardError("GPU selector must be an exact existing index or UUID")
    if any(p.uuid not in {g.uuid for g in gpus} for p in apps):
        raise GuardError("A process references an unknown GPU UUID")
    return matches[0], [p for p in apps if p.uuid == matches[0].uuid]


def load_memory_override(path: Path, now: float | None = None) -> dict | None:
    """Read an explicit, local, expiring user authorization; never renew it."""
    try:
        value = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise GuardError("Temporary memory authorization cannot be verified") from exc
    if not isinstance(value, dict):
        raise GuardError("Temporary memory authorization must be an object")
    start, end = value.get("starts_at_unix"), value.get("expires_at_unix")
    if (value.get("schema") != "depallet.memory_override.v1"
            or value.get("scope") != "memory_thresholds_only"
            or type(value.get("authorized_uid")) is not int or value["authorized_uid"] != os.getuid()
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in (start, end))
            or not 0 < end-start <= 86400):
        raise GuardError("Invalid temporary memory authorization")
    current = time.time() if now is None else now
    if not math.isfinite(current):
        raise GuardError("Authorization clock cannot be verified")
    return value if start <= current < end else None


def idle_reason(gpu: GPU, apps: list[GPUProcess], *, memory_checks: bool = True) -> str | None:
    if apps:
        return "existing_compute_process"
    if gpu.utilization > 5:
        return "gpu_utilization_above_5_percent"
    if memory_checks and gpu.used_mib > 512:
        return "gpu_memory_above_512_mib"
    if memory_checks and gpu.total_mib - gpu.used_mib < 24 * 1024:
        return "gpu_free_memory_below_24_gib"
    return None


def system_available_ram() -> int:
    """Linux MemAvailable; fail closed if the kernel counter is unavailable."""
    try:
        lines = Path("/proc/meminfo").read_text().splitlines()
        fields = next(line.split() for line in lines if line.startswith("MemAvailable:"))
        if len(fields) != 3 or fields[2] != "kB":
            raise ValueError("unexpected memory unit")
        value = int(fields[1]) * 1024
        if value <= 0:
            raise ValueError("nonpositive memory availability")
        return value
    except (OSError, ValueError, StopIteration) as exc:
        raise GuardError("System available memory cannot be verified") from exc


def shared_admission_reason(gpu: GPU, available_ram_bytes: int | None, *, memory_checks: bool = True, parallel: bool = False) -> str | None:
    if not memory_checks:
        return None
    # Explicit user opt-in on 2026-09-14. This is headroom monitoring, not a
    # reservation: another job can still grow between observations.
    if gpu.total_mib - gpu.used_mib < MAX_GPU_MIB * (2 if parallel else 1) + 24 * 1024:
        return "pool_gpu_free_memory_below_48_gib" if parallel else "shared_gpu_free_memory_below_36_gib"
    if isinstance(available_ram_bytes, bool) or not isinstance(available_ram_bytes, int) or available_ram_bytes < 32 * GIB:
        return "shared_system_available_memory_below_32_gib_or_unverified"
    return None


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)


def proc_info(pid: int) -> Proc | None:
    """Read Linux identity. A vanished PID is normal; other read errors fail closed."""
    directory = Path("/proc") / str(pid)
    try:
        uid_before = directory.stat().st_uid
        raw = (directory / "stat").read_text()
        uid_after = directory.stat().st_uid
    except (FileNotFoundError, ProcessLookupError):
        return None
    except OSError as exc:
        raise GuardError(f"Cannot inspect process {pid}: {type(exc).__name__}") from exc
    try:
        head, rest = raw.rsplit(") ", 1)
        actual_pid = int(head.split("(", 1)[0])
        parts = rest.split()
        info = Proc(actual_pid, uid_after, int(parts[2]), int(parts[3]),
                    int(parts[19]), max(0, int(parts[21])) * PAGE_BYTES, parts[0])
    except (ValueError, IndexError) as exc:
        raise GuardError(f"Malformed /proc identity for {pid}") from exc
    if actual_pid != pid or uid_before != uid_after:
        raise GuardError("Process identity changed while reading")
    return info


def belongs(info: Proc | None, group: Group) -> bool:
    return bool(info and info.uid == group.uid and info.pgid == group.pgid
                and info.sid == group.pgid and info.start_ticks >= group.min_start_ticks)


def group_members(group: Group) -> list[Proc]:
    members = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        # Avoid reading unrelated users' process data beyond directory ownership.
        try:
            if path.stat().st_uid != group.uid:
                continue
        except (FileNotFoundError, ProcessLookupError):
            continue
        info = proc_info(int(path.name))
        if belongs(info, group) and info.state not in {"Z", "X"}:
            members.append(info)
    return members


def same_identity(left: Proc | None, right: Proc) -> bool:
    return bool(left and (left.pid, left.uid, left.pgid, left.sid, left.start_ticks)
                == (right.pid, right.uid, right.pgid, right.sid, right.start_ticks))


def signal_member(member: Proc, group: Group, sig: int) -> bool:
    """pidfd + identity recheck avoids signalling a recycled PID or another group."""
    if not belongs(member, group):
        return False
    try:
        fd = os.pidfd_open(member.pid, 0)
    except ProcessLookupError:
        return False
    try:
        current = proc_info(member.pid)
        if not same_identity(current, member) or not belongs(current, group):
            return False
        signal.pidfd_send_signal(fd, sig)
        return True
    except ProcessLookupError:
        return False
    finally:
        os.close(fd)


def stop_group(group: Group) -> dict:
    """Only our still-verified session/group is signalled; never killpg or pkill."""
    sent: list[dict] = []
    for sig, duration in ((signal.SIGTERM, 5.0), (signal.SIGKILL, 2.0)):
        deadline = time.monotonic() + duration
        signalled: set[tuple[int, int]] = set()
        while time.monotonic() < deadline:
            members = group_members(group)
            if not members:
                return {"signals": sent, "remaining_pids": []}
            for member in members:
                identity = (member.pid, member.start_ticks)
                if identity not in signalled and signal_member(member, group, sig):
                    sent.append({"pid": member.pid, "signal": int(sig)})
                    signalled.add(identity)
            time.sleep(0.1)
    return {"signals": sent, "remaining_pids": [p.pid for p in group_members(group)]}


def monitor_reason(gpu: GPU, apps: list[GPUProcess], group: Group, members: list[Proc],
                   free_disk: int, elapsed: float, seconds: float,
                   baseline_used_mib: int, lookup: Callable = proc_info,
                   allow_shared: bool = False, available_ram_bytes: int | None = None,
                   memory_checks: bool = True, parallel: bool = False) -> str | None:
    if elapsed >= seconds:
        return "wall_timeout"
    if free_disk < MIN_DISK_BYTES:
        return "disk_free_below_30_gib"
    if memory_checks and sum(p.rss_bytes for p in members) > MAX_RSS_BYTES:
        return "group_rss_above_16_gib"
    owned_apps = []
    for app in apps:
        if belongs(lookup(app.pid), group):
            owned_apps.append(app)
        elif not allow_shared:
            return "foreign_compute_process"
    if allow_shared and memory_checks:
        if gpu.total_mib - gpu.used_mib < 24 * 1024:
            return "shared_gpu_free_memory_below_24_gib"
        if isinstance(available_ram_bytes, bool) or not isinstance(available_ram_bytes, int) or available_ram_bytes < 16 * GIB:
            return "shared_system_available_memory_below_16_gib_or_unverified"
    if memory_checks and sum(app.used_mib for app in owned_apps) > MAX_GPU_MIB:
        return "own_gpu_memory_above_12_gib"
    # Graphics allocations may not appear in compute-app accounting. In the
    # two-slot pool this conservative DEVICE fallback covers both workers;
    # the owned compute-app cap above remains 12 GiB per worker. Unattributed
    # foreign growth can still stop us, never the foreign process.
    if memory_checks and gpu.used_mib - baseline_used_mib > MAX_GPU_MIB * (2 if parallel else 1):
        return "device_memory_growth_above_24_gib_pool" if parallel else "device_memory_growth_above_12_gib"
    return None


def validate_command(command: list[str], project: Path = PROJECT, cache: Path = CACHE) -> list[str]:
    if len(command) < 2:
        raise GuardError("Command must be project venv Python followed by scripts/run_p0.py")
    expected_python = cache / "venv/bin/python"
    requested_python = Path(command[0]).expanduser()
    expected_script = project / "scripts/run_p0.py"
    requested_script = Path(command[1]).expanduser()
    # Exact interpreter/script pairs preserve dependency isolation.
    runner_envs = {"preview_assets.py": "venv", "run_p0.py": "venv", "run_depallet.py": "venv", "run_task.py": "venv", "validate_scene_usd.py": "venv",
                   "point2pose_worker.py": "point2pose-venv", "sam31_worker.py": "sam3-venv",
                   "graspgen_worker.py": "graspgen-venv",
                   "cutamp_worker.py": "cutamp-venv",
                   "curobo_worker.py": "curobo-venv", "vlm_worker.py": "vlm-venv"}
    if requested_script.name not in runner_envs:
        raise GuardError("Runner is not in the explicit project allowlist")
    expected_python = cache / runner_envs[requested_script.name] / "bin" / "python"
    expected_script = project / "scripts" / requested_script.name
    if requested_script.is_symlink() or requested_script.parent.resolve() != project.resolve() / "scripts":
        raise GuardError("Runner must be a direct regular file in the project scripts directory")
    if not requested_python.is_absolute() or requested_python.name != "python":
        raise GuardError("Use the absolute project venv/bin/python path")
    if requested_python.parent.resolve() != expected_python.parent.resolve():
        raise GuardError("Only the project venv Python is allowed")
    if not expected_python.is_file() or not os.access(expected_python, os.X_OK):
        raise GuardError("Project venv Python does not exist or is not executable")
    if not requested_script.is_absolute() or requested_script.resolve() != expected_script.resolve():
        raise GuardError("Only the exact allowlisted project runner may be launched")
    if expected_script.is_symlink() or not expected_script.is_file():
        raise GuardError("P0 runner must be an existing regular project file")
    if expected_script.parent.resolve() != project.resolve() / "scripts":
        raise GuardError("P0 scripts directory cannot point outside this project")
    if any("\x00" in arg for arg in command):
        raise GuardError("NUL bytes in arguments are forbidden")
    return [str(expected_python), str(expected_script), *command[2:]]


def new_output(value: str, runs: Path = RUNS) -> Path:
    output = Path(value).expanduser().resolve()
    root = runs.resolve()
    if output == root or not output.is_relative_to(root):
        raise GuardError("Output must be a new directory under this project's runs root")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    return output


@contextmanager
def launch_lease(cache: Path, slot: int | None):
    """Legacy exclusive lock, or one of exactly two shared-mode slot leases.

    FDs are also inherited by the child so a killed guard does not immediately
    free a slot while its simulator is still alive. Never unlink lock files.
    """
    if slot is not None and (type(slot) is not int or slot not in (0, 1)):
        raise GuardError("Parallel slot must be 0 or 1")
    cache.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        project_lock = stack.enter_context((cache / "guard.lock").open("a+"))
        try:
            fcntl.flock(project_lock, (fcntl.LOCK_EX if slot is None else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GuardError("Another jclee Isaac P0 guard holds the project lock") from exc
        locks = [project_lock]
        if slot is not None:
            slot_lock = stack.enter_context((cache / f"guard.slot{slot}.lock").open("a+"))
            try:
                fcntl.flock(slot_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise GuardError(f"Parallel slot {slot} is occupied") from exc
            locks.append(slot_lock)
        yield tuple(lock.fileno() for lock in locks)


def worker_cpus(slot: int | None, allowed=None) -> list[int]:
    cpus = sorted(os.sched_getaffinity(0) if allowed is None else allowed)
    if slot is None:
        return cpus[:4]
    if type(slot) is not int or slot not in (0, 1) or len(cpus) < 8:
        raise GuardError("Two-worker mode requires slots 0/1 and at least eight available CPUs")
    return cpus[4*slot:4*slot+4]


def _shared_diagnostic_plan(command: list[str], guard_output: str | Path | None,
                            workspace: Path) -> None:
    """Validate the one non-task command admitted to a shared slot."""
    if guard_output is None:
        raise GuardError("Shared diagnostic planning requires the guard output path")
    values: dict[str, str] = {}
    diagnostic_plan = False
    index = 2
    while index < len(command):
        argument = command[index]
        if argument == "--diagnostic-plan":
            if diagnostic_plan:
                raise GuardError("Shared diagnostic planning flags may not be repeated")
            diagnostic_plan = True
            index += 1
            continue
        if argument not in ("--request", "--output") or index + 1 >= len(command):
            raise GuardError("Shared cuRobo slots allow only --diagnostic-plan, --request, and --output")
        if argument in values or command[index + 1].startswith("--"):
            raise GuardError("Shared diagnostic planning requires one value for each path flag")
        values[argument] = command[index + 1]
        index += 2
    if not diagnostic_plan or set(values) != {"--request", "--output"}:
        raise GuardError("Shared cuRobo slots require explicit --diagnostic-plan, --request, and --output")

    # `runs/` can be a project-owned symlink into /DATA while the repository
    # itself remains mounted under /home.  Accept either canonical project
    # workspace path or the canonical project runs root; guard-owned outputs
    # are still separately constrained by new_output() and equality below.
    roots = (workspace.resolve(), RUNS.resolve())
    request_path = Path(values["--request"]).expanduser()
    output_path = Path(values["--output"]).expanduser()
    if not request_path.is_absolute() or not output_path.is_absolute():
        raise GuardError("Shared diagnostic request and output paths must be absolute")
    try:
        request_path = request_path.resolve(strict=True)
    except OSError as exc:
        raise GuardError("Shared diagnostic request must be an existing file") from exc
    output_path = output_path.resolve()
    if (not request_path.is_file()
            or not any(request_path.is_relative_to(root) for root in roots)
            or not any(output_path.is_relative_to(root) for root in roots)):
        raise GuardError("Shared diagnostic request and output paths must stay under the project workspace or runs root")
    if output_path != Path(guard_output).expanduser().resolve():
        raise GuardError("cuRobo --output must equal the guard --output")
    try:
        request = json.loads(request_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GuardError("Shared diagnostic request must be readable JSON") from exc
    if (not isinstance(request, dict)
            or request.get("robot_execution_authorized") is not False
            or request.get("physical_execution_validated") is not False):
        raise GuardError(
            "Shared diagnostic planning requires explicit false execution and physical-validation fields")


def validate_parallel_command(command: list[str], slot: int | None, allow_shared: bool,
                              guard_output: str | Path | None = None,
                              workspace: Path = WORKSPACE) -> None:
    if slot is None:
        return
    if type(slot) is not int or slot not in (0, 1):
        raise GuardError("Parallel slot must be 0 or 1")
    if not allow_shared:
        raise GuardError("Parallel mode requires --allow-shared-gpu")
    runner = Path(command[1]).name
    if runner == "curobo_worker.py":
        _shared_diagnostic_plan(command, guard_output, workspace)
        return
    if runner != "run_task.py":
        raise GuardError("Parallel mode requires run_task.py or guarded cuRobo diagnostic planning")
    if any(arg.startswith("--") and any(target.startswith(arg.split("=",1)[0])
           for target in ("--live", "--port")) for arg in command[2:]):
        raise GuardError("Parallel tasks use static output monitoring; live server/port arguments are disabled")


def child_environment(gpu: GPU, output: Path, slot: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    state = CACHE if slot is None else CACHE / "workers" / f"slot{slot}"
    env.update({
        "CUDA_VISIBLE_DEVICES": gpu.uuid, "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "ISAAC_P0_GPU_UUID": gpu.uuid, "ISAAC_P0_GPU_INDEX": str(gpu.index),
        "ISAAC_P0_OUTPUT": str(output), "ISAAC_P0_GUARDED": "1",
        "ISAAC_P0_PROJECT": str(PROJECT), "ISAAC_P0_CACHE": str(CACHE),
        "ISAAC_P0_RUNS": str(RUNS), "ISAAC_P0_RUNTIME_CACHE": str(state),
        "ISAAC_P0_PARALLEL_SLOT": "" if slot is None else str(slot),
        "XDG_CACHE_HOME": str(state / "xdg/cache"),
        "XDG_CONFIG_HOME": str(state / "xdg/config"),
        "XDG_DATA_HOME": str(state / "xdg/data"),
        "__GL_SHADER_DISK_CACHE_PATH": str(state / "shader"),
        "CUDA_CACHE_PATH": str(state / "cuda"),
        "TMPDIR": str(state / "tmp"),
        "PYTHONPATH": str(PROJECT / "src"), "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1", "HF_HUB_OFFLINE": "1", "WANDB_MODE": "disabled",
    })
    if slot is not None:
        env.update({"TORCH_EXTENSIONS_DIR": str(state / "torch-extensions"),
                    "TORCHINDUCTOR_CACHE_DIR": str(state / "torchinductor"),
                    "TRITON_CACHE_DIR": str(state / "triton"),
                    "WARP_CACHE_PATH": str(state / "warp")})
    # A terminal/desktop inherited through SSH must never accidentally enable GUI.
    env.pop("DISPLAY", None)
    env.pop("WAYLAND_DISPLAY", None)
    for name in ("XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "TMPDIR",
                 "CUDA_CACHE_PATH", "__GL_SHADER_DISK_CACHE_PATH"):
        Path(env[name]).mkdir(parents=True, exist_ok=True)
    return env


def _child_limits(cpus=None) -> None:
    os.sched_setaffinity(0, worker_cpus(None) if cpus is None else cpus)
    os.nice(10)


def run_guard(args: argparse.Namespace) -> int:
    command = validate_command(args.command, PROJECT, CACHE)
    task_seconds = getattr(args, "task_seconds", None)
    if task_seconds is not None:
        if Path(command[1]).name != "run_task.py":
            raise GuardError("Extended task budget is restricted to the full-pallet runtime")
        args.seconds = bounded_task_seconds(str(task_seconds))
    allow_shared = bool(getattr(args, "allow_shared_gpu", False))
    slot = getattr(args, "parallel_slot", None)
    validate_parallel_command(command, slot, allow_shared, args.output, WORKSPACE)
    cpus = worker_cpus(slot)
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise GuardError("Linux pidfd support is required for safe process cleanup")
    CACHE.mkdir(parents=True, exist_ok=True)
    with launch_lease(CACHE, slot) as lease_fds:
        output = new_output(args.output, RUNS)
        preflight = {"command": command, "seconds": args.seconds, "samples": [],
                     "gpu_launch_performed": False, "cooperative_monitor_only": True,
                     "allow_shared_gpu": allow_shared, "user_authorized_shared_memory_policy": allow_shared,
                     "parallel_slot": slot, "cpu_affinity": cpus, "max_parallel_tasks": 2 if slot is not None else 1,
                     "device_growth_fallback_mib": MAX_GPU_MIB * (2 if slot is not None else 1)}
        result = {"status": "preflight_pending", "child_returncode": None,
                  "gpu_launch_performed": False, "output": str(output)}
        child = None
        group = None
        interrupted = {"signal": None}
        previous = {}
        def on_signal(sig, _frame):
            interrupted["signal"] = sig
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[sig] = signal.signal(sig, on_signal)
        start = None
        try:
            selected_uuid = None
            gpu = None
            for sample_index in range(3):
                if interrupted["signal"] is not None:
                    raise GuardError("Guard interrupted before launch")
                gpu, apps = query_gpu(args.gpu)
                if selected_uuid is not None and gpu.uuid != selected_uuid:
                    raise GuardError("Selected GPU changed between preflight samples")
                selected_uuid = gpu.uuid
                available_ram = system_available_ram() if allow_shared else None
                override = load_memory_override(RUNS / 'control/memory-policy-override.json')
                memory_checks = override is None
                reason = shared_admission_reason(gpu, available_ram, memory_checks=memory_checks, parallel=slot is not None) if allow_shared else idle_reason(gpu, apps, memory_checks=memory_checks)
                if shutil.disk_usage(output).free < MIN_DISK_BYTES:
                    reason = "disk_free_below_30_gib"
                preflight["samples"].append({"unix_time": time.time(), "gpu": asdict(gpu),
                                              "compute_apps": [asdict(p) for p in apps],
                                              "system_available_ram_bytes": available_ram, "reason": reason,
                                              "memory_thresholds_enforced": memory_checks, "memory_override": override})
                write_json(output / "preflight.json", preflight)
                if reason:
                    result["status"] = "preflight_blocked"
                    result["reason"] = reason
                    return 75
                if sample_index < 2:
                    time.sleep(1.0)
            assert gpu is not None
            if interrupted["signal"] is not None:
                raise GuardError("Guard interrupted before launch")
            # Probe our own PID only: ensure this kernel permits race-safe signalling.
            probe_fd = os.pidfd_open(os.getpid(), 0)
            os.close(probe_fd)
            env = child_environment(gpu, output, slot)
            env["ISAAC_P0_SHARED_GPU"] = "1" if allow_shared else "0"
            env["ISAAC_P0_WALL_SECONDS"] = str(args.seconds)
            preflight["runtime_cache"] = env["ISAAC_P0_RUNTIME_CACHE"]
            preflight["mutable_environment"] = {key: env[key] for key in (
                "XDG_CACHE_HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "TMPDIR",
                "CUDA_CACHE_PATH", "__GL_SHADER_DISK_CACHE_PATH", "ISAAC_P0_RUNTIME_CACHE")}

            with (output / "child.log").open("x", encoding="utf-8") as log, \
                    (output / "monitor.jsonl").open("x", encoding="utf-8") as monitor:
                born = int(time.clock_gettime(time.CLOCK_BOOTTIME) * CLK_TCK) - 1
                start = time.monotonic()
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                         cwd=PROJECT, env=env, start_new_session=True,
                                         preexec_fn=partial(_child_limits, cpus), close_fds=True, pass_fds=lease_fds)
                group = Group(child.pid, os.getuid(), born)
                preflight["gpu_launch_performed"] = True
                preflight["child_pid"] = child.pid
                write_json(output / "preflight.json", preflight)
                result.update(status="running", gpu_launch_performed=True, child_pid=child.pid,
                              gpu_uuid=gpu.uuid, gpu_index=gpu.index)
                write_json(output / "exit.json", result)
                read_failures = 0
                while child.poll() is None:
                    elapsed = time.monotonic() - start
                    record = {"unix_time": time.time(), "elapsed_seconds": elapsed}
                    reason = None
                    if interrupted["signal"] is not None:
                        reason = "guard_interrupted"
                    elif elapsed >= args.seconds:
                        reason = "wall_timeout"
                    else:
                        try:
                            observed, apps = query_gpu(gpu.uuid)
                            members = group_members(group)
                            disk_free = shutil.disk_usage(output).free
                            available_ram = system_available_ram() if allow_shared else None
                            override = load_memory_override(RUNS / 'control/memory-policy-override.json')
                            memory_checks = override is None
                            record.update(memory_thresholds_enforced=memory_checks,memory_override=override,gpu=asdict(observed), compute_apps=[asdict(p) for p in apps],
                                          group_pids=[p.pid for p in members],
                                          group_rss_bytes=sum(p.rss_bytes for p in members),
                                          disk_free_bytes=disk_free, system_available_ram_bytes=available_ram,
                                          allow_shared_gpu=allow_shared)
                            reason = monitor_reason(observed, apps, group, members, disk_free,
                                                    time.monotonic() - start, args.seconds, gpu.used_mib,
                                                    allow_shared=allow_shared, available_ram_bytes=available_ram, memory_checks=memory_checks, parallel=slot is not None)
                            read_failures = 0
                        except (GuardError, OSError) as exc:
                            read_failures += 1
                            record.update(read_error=str(exc), consecutive_read_failures=read_failures)
                            if allow_shared:
                                reason = "shared_monitor_unverified"
                            elif read_failures >= 3:
                                reason = "monitor_read_failed_three_times"
                    record["stop_reason"] = reason
                    monitor.write(json.dumps(record, allow_nan=False) + "\n")
                    monitor.flush()
                    if reason:
                        result.update(status="stopped", reason=reason)
                        break
                    deadline = min(start + args.seconds, time.monotonic() + (1.0 if allow_shared else 2.0))
                    while child.poll() is None and time.monotonic() < deadline and interrupted["signal"] is None:
                        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
                if result["status"] == "running":
                    result["status"] = "success" if child.returncode == 0 else "child_failed"
                result["cleanup"] = stop_group(group)
                if result["cleanup"]["remaining_pids"]:
                    result.update(status="cleanup_incomplete", reason="own_group_members_remain")
                try:
                    result["child_returncode"] = child.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    result.update(status="cleanup_incomplete", reason="child_wait_timeout")
                return 0 if result["status"] == "success" else 74
        except (GuardError, OSError, ValueError) as exc:
            result.update(status="guard_error", reason=str(exc))
            return 74
        finally:
            if group is not None and "cleanup" not in result:
                try:
                    result["cleanup"] = stop_group(group)
                    if child is not None:
                        result["child_returncode"] = child.wait(timeout=2.0)
                except (GuardError, OSError, subprocess.TimeoutExpired) as exc:
                    result["cleanup_error"] = str(exc)
                    result["status"] = "cleanup_incomplete"
            result["elapsed_seconds"] = time.monotonic() - start if start is not None else 0.0
            write_json(output / "preflight.json", preflight)
            write_json(output / "exit.json", result)
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            print(json.dumps(result, indent=2, allow_nan=False))


def bounded_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seconds must be a number") from exc
    if not math.isfinite(seconds) or not 1 <= seconds <= 600:
        raise argparse.ArgumentTypeError("seconds must be finite and between 1 and 600")
    return seconds


def bounded_task_seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("task seconds must be a number") from exc
    if not math.isfinite(seconds) or not 1 <= seconds <= 7200:
        raise argparse.ArgumentTypeError("task seconds must be finite and within 1..7200")
    return seconds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=bounded_seconds, default=180.0)
    parser.add_argument("--task-seconds", type=bounded_task_seconds, help="Full-pallet run_task.py only: bounded continuous task up to 7200 seconds; all resource limits unchanged")
    parser.add_argument("--gpu", default="0", help="Exact GPU index or GPU UUID")
    parser.add_argument("--allow-shared-gpu", action="store_true", help="Explicit opt-in: require 36 GiB free at admission and keep 24 GiB free; our limits remain unchanged")
    parser.add_argument("--parallel-slot", type=int, choices=[0, 1], help="Opt-in bounded two-task mode; requires shared GPU and run_task.py, no live server")
    parser.add_argument("--output", required=True, help="New directory under project runs root")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    try:
        return run_guard(args)
    except (GuardError, OSError) as exc:
        print(f"guard: {exc}", file=sys.stderr)
        return 74


if __name__ == "__main__":
    raise SystemExit(main())
