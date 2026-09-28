"""CPU-only RGB-D diagnostic recording for Isaac P0, never training data."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import zipfile
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

FORMAT = "isaac_p0_rgbd_diagnostic_v1"
MIN_FRAMES = 100
MIN_DEPTH_VALID_FRACTION = 0.05
_RESERVED = ("rgb", "depth", "frames.jsonl", "metadata.json", "summary.json", "manifest.json", ".recording.lock")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def _write_json(path: Path, value: Any) -> None:
    text = _json(value) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_vector(value: Any, count: int, name: str) -> None:
    array = np.asarray(value)
    if array.shape != (count,) or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {count} finite numbers")


def _check_row(row: dict, previous: dict | None) -> None:
    if not isinstance(row, dict):
        raise ValueError("row must be a dictionary")
    timestamp = row.get("sim_time")
    step = row.get("physics_step")
    if isinstance(timestamp, bool) or not isinstance(timestamp, (float, int)) or not math.isfinite(timestamp) or timestamp <= 0:
        raise ValueError("sim_time must be finite and strictly positive")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("physics_step must be a positive integer")
    if previous is not None and (timestamp <= previous["sim_time"] or step <= previous["physics_step"]):
        raise ValueError("sim_time and physics_step must strictly increase")
    if not isinstance(row.get("phase"), str) or not row["phase"]:
        raise ValueError("phase must be a nonempty string")
    _finite_vector(row.get("box_position_m"), 3, "box_position_m")
    unavailable = row.get("robot_present") is False or (row.get("robot_state_available") is False and row.get("joint_state_source") == "unavailable_in_static_render")
    if not (unavailable and row.get("joints_rad") is None):
        _finite_vector(row.get("joints_rad"), 6, "joints_rad")
    objects = row.get("gripped_objects")
    if not isinstance(objects, list) or any(not isinstance(item, str) or not item for item in objects):
        raise ValueError("gripped_objects must be a list of nonempty path strings")
    _json(row)


def _check_arrays(rgb: np.ndarray, depth: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape not in ((height, width, 3), (height, width, 4)):
        raise ValueError("RGB must be uint8 with configured H,W and 3 or 4 channels")
    rgb = np.ascontiguousarray(rgb[:, :, :3])
    if not np.any(np.var(rgb.astype(np.float32), axis=(0, 1)) > 0):
        raise ValueError("RGB is spatially uniform")
    if not isinstance(depth, np.ndarray) or depth.shape != (height, width) or depth.dtype.kind != "f":
        raise ValueError("depth_m must be a floating array with configured H,W")
    original_valid = np.isfinite(depth) & (depth > 0)
    with np.errstate(over="ignore", under="ignore"):
        depth32 = np.ascontiguousarray(depth, dtype=np.float32)
    valid = np.isfinite(depth32) & (depth32 > 0)
    if not np.array_equal(original_valid, valid):
        raise ValueError("depth values cannot be represented as positive finite float32 metres")
    if float(valid.mean()) < MIN_DEPTH_VALID_FRACTION:
        raise ValueError("valid positive depth fraction is below 0.05")
    return rgb, depth32, valid


def _write_frame(output: Path, record: dict, rgb: np.ndarray, depth: np.ndarray,
                 valid: np.ndarray, png_compress_level: int,
                 depth_compress_level: int) -> None:
    """Write one immutable RGB-D pair; safe to execute in a worker thread."""
    with (output / record["rgb_path"]).open("xb") as stream:
        Image.fromarray(rgb).save(
            stream, format="PNG", compress_level=png_compress_level)
    with (output / record["depth_path"]).open("xb") as stream:
        with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=depth_compress_level) as archive:
            for name, array in (("depth_m", depth), ("valid", valid)):
                with archive.open(name + ".npy", mode="w", force_zip64=True) as member:
                    np.lib.format.write_array(member, array, allow_pickle=False)


class Recorder:
    """Append actual post-step sensor observations to an immutable diagnostic run.

    ``sim_time`` and ``physics_step`` must come from the simulator. This class
    records observations only and makes no observation/action pairing claim.
    Incomplete or crashed output is preserved and cannot be silently reused.
    """

    def __init__(self, output: Path, width: int = 320, height: int = 240,
                 png_compress_level: int = 6, depth_compress_level: int = 6,
                 writer_workers: int = 0):
        if type(width) is not int or type(height) is not int or not (2 <= width <= 1920 and 2 <= height <= 1080):
            raise ValueError("invalid recording dimensions")
        if (type(png_compress_level) is not int or not 0 <= png_compress_level <= 9
                or type(depth_compress_level) is not int or not 0 <= depth_compress_level <= 9):
            raise ValueError("compression levels must be integers from 0 through 9")
        if type(writer_workers) is not int or not 0 <= writer_workers <= 8:
            raise ValueError("writer_workers must be an integer from 0 through 8")
        output = Path(output)
        if output.is_symlink():
            raise ValueError("output directory must not be a symlink")
        output.mkdir(parents=True, exist_ok=True)
        self.output = output.resolve()
        if any((self.output / name).exists() or (self.output / name).is_symlink() for name in _RESERVED):
            raise FileExistsError("recording output already exists; use a fresh output directory")
        with (self.output / ".recording.lock").open("x") as stream:
            stream.write(str(os.getpid()) + "\n")
        (self.output / "rgb").mkdir()
        (self.output / "depth").mkdir()
        (self.output / "frames.jsonl").touch(exist_ok=False)
        self.width, self.height = width, height
        self.png_compress_level = png_compress_level
        self.depth_compress_level = depth_compress_level
        self.writer_workers = writer_workers
        self._writer = (ThreadPoolExecutor(max_workers=writer_workers,
                                           thread_name_prefix="rgbd-writer")
                        if writer_workers else None)
        self._pending: deque[Future] = deque()
        self._max_pending = writer_workers * 2
        self._writer_failure: BaseException | None = None
        self.count = 0
        self.previous = None
        self.finished = False

    def _raise_writer_failure(self) -> None:
        if self._writer_failure is not None:
            raise RuntimeError("asynchronous frame writer previously failed") from self._writer_failure

    def _await_oldest(self) -> None:
        future = self._pending.popleft()
        try:
            future.result()
        except BaseException as error:
            self._writer_failure = error
            self._pending.clear()
            assert self._writer is not None
            self._writer.shutdown(wait=True)
            self._writer = None
            raise

    def _drain_writer(self) -> None:
        self._raise_writer_failure()
        while self._pending:
            self._await_oldest()

    def append(self, rgb: np.ndarray, depth_m: np.ndarray, row: dict) -> int:
        if self.finished:
            raise RuntimeError("recording is already finalized")
        self._raise_writer_failure()
        _check_row(row, self.previous)
        if any(key in row for key in ("frame_index", "rgb_path", "depth_path")):
            raise ValueError("row contains reserved recorder fields")
        rgb, depth, valid = _check_arrays(rgb, depth_m, self.width, self.height)
        # Roundtrip makes a private snapshot and rejects non-JSON/NaN metadata.
        record = json.loads(_json(row))
        index = self.count
        record.update(frame_index=index, rgb_path=f"rgb/{index:06d}.png", depth_path=f"depth/{index:06d}.npz")
        if self._writer is None:
            _write_frame(self.output, record, rgb, depth, valid,
                         self.png_compress_level, self.depth_compress_level)
        else:
            if len(self._pending) >= self._max_pending:
                self._await_oldest()
            # Async mode must detach from render buffers before append returns.
            future = self._writer.submit(
                _write_frame, self.output, record, rgb.copy(), depth.copy(), valid.copy(),
                self.png_compress_level, self.depth_compress_level)
            self._pending.append(future)
        with (self.output / "frames.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(_json(record) + "\n")
            stream.flush()
        self.previous = record
        self.count += 1
        return index

    def finish(self, metadata: dict) -> dict:
        if self.finished:
            raise RuntimeError("recording is already finalized")
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")
        if metadata.get("valid_sim_hours", 0) != 0:
            raise ValueError("P0 diagnostics cannot count as accepted Sim20h data")
        _json(metadata)
        self._drain_writer()
        summary, files = _inspect(self.output, self.width, self.height)
        if summary["frame_count"] != self.count:
            raise ValueError("on-disk frame count differs from this recorder")
        summary.update(valid_sim_hours=0, training_dataset=False, format=FORMAT)
        _write_json(self.output / "metadata.json", {
            "format": FORMAT, "width": self.width, "height": self.height,
            "valid_sim_hours": 0, "training_dataset": False,
            "description": "P0 development RGB-D observations; no action contract or accepted collection hours",
            "run_metadata": metadata,
        })
        _write_json(self.output / "summary.json", summary)
        files.extend(["metadata.json", "summary.json"])
        _write_json(self.output / "manifest.json", {
            "format": FORMAT, "frame_count": self.count,
            "files": {name: _sha(self.output / name) for name in sorted(files)},
        })
        if self._writer is not None:
            self._writer.shutdown(wait=True)
            self._writer = None
        self.finished = True
        return summary


def _safe_file(output: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("unsafe manifest path")
    path = output / relative
    if path.is_symlink() or any((output / Path(*relative.parts[:i])).is_symlink() for i in range(1, len(relative.parts))):
        raise ValueError("recorded files must not be symlinks")
    if not path.is_file():
        raise ValueError(f"missing recorded file: {name}")
    return path


def _inspect(output: Path, width: int, height: int) -> tuple[dict, list[str]]:
    rows_path = _safe_file(output, "frames.jsonl")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    files = ["frames.jsonl"]
    previous = None
    depth_fractions = []
    timestamps = []
    for index, row in enumerate(rows):
        _check_row(row, previous)
        rgb_name, depth_name = f"rgb/{index:06d}.png", f"depth/{index:06d}.npz"
        if row.get("frame_index") != index or row.get("rgb_path") != rgb_name or row.get("depth_path") != depth_name:
            raise ValueError("frame index or image path is inconsistent")
        with Image.open(_safe_file(output, rgb_name)) as image:
            if image.mode != "RGB":
                raise ValueError("stored image must be RGB")
            rgb = np.array(image)
        with np.load(_safe_file(output, depth_name), allow_pickle=False) as bundle:
            if set(bundle.files) != {"depth_m", "valid"}:
                raise ValueError("unexpected depth payload")
            depth, valid = bundle["depth_m"], bundle["valid"]
        if depth.dtype != np.float32 or valid.dtype != np.bool_ or valid.shape != (height, width):
            raise ValueError("depth payload dtype or shape is invalid")
        _, _, expected = _check_arrays(rgb, depth, width, height)
        if not np.array_equal(valid, expected):
            raise ValueError("depth validity mask disagrees with finite positive metres")
        files.extend([rgb_name, depth_name])
        depth_fractions.append(float(valid.mean()))
        timestamps.append(row["sim_time"])
        previous = row
    expected_rgb = {output / f"rgb/{i:06d}.png" for i in range(len(rows))}
    expected_depth = {output / f"depth/{i:06d}.npz" for i in range(len(rows))}
    if set((output / "rgb").iterdir()) != expected_rgb or set((output / "depth").iterdir()) != expected_depth:
        raise ValueError("unindexed or missing RGB-D files")
    deltas = np.diff(timestamps)
    regular = bool(len(deltas) and np.allclose(deltas, deltas[0], rtol=0, atol=1e-6))
    flags = {
        "at_least_100_frames": len(rows) >= MIN_FRAMES,
        "strict_sim_time_and_physics_steps": True,
        "rgb_shape_and_nonuniform": True,
        "depth_shape_dtype_and_mask": True,
        "depth_valid_fraction_at_least_005": bool(rows),
        "recorded_file_integrity": True,
    }
    return {
        "passed": bool(all(flags.values())), "checks": flags,
        "frame_count": len(rows), "width": width, "height": height,
        "first_sim_time": timestamps[0] if timestamps else None,
        "last_sim_time": timestamps[-1] if timestamps else None,
        "observed_sim_span_seconds": timestamps[-1] - timestamps[0] if timestamps else 0.0,
        "fixed_sim_timestep": regular,
        "observed_frame_dt": float(deltas[0]) if regular else None,
        "min_depth_valid_fraction": min(depth_fractions) if depth_fractions else None,
    }, files


def validate_recording(output: Path) -> dict:
    """Re-read all diagnostic files, verify checksums and observation contracts."""
    output = Path(output).resolve()
    manifest = json.loads(_safe_file(output, "manifest.json").read_text())
    if manifest.get("format") != FORMAT or not isinstance(manifest.get("files"), dict):
        raise ValueError("unsupported recording manifest")
    for name, digest in manifest["files"].items():
        if _sha(_safe_file(output, name)) != digest:
            raise ValueError(f"checksum mismatch: {name}")
    metadata = json.loads(_safe_file(output, "metadata.json").read_text())
    stored = json.loads(_safe_file(output, "summary.json").read_text())
    if metadata.get("format") != FORMAT or metadata.get("valid_sim_hours") != 0 or metadata.get("training_dataset") is not False:
        raise ValueError("invalid diagnostic provenance")
    computed, files = _inspect(output, metadata["width"], metadata["height"])
    computed.update(valid_sim_hours=0, training_dataset=False, format=FORMAT)
    if stored != computed or manifest.get("frame_count") != computed["frame_count"]:
        raise ValueError("summary or frame count disagrees with the recorded observations")
    if set(manifest["files"]) != set(files + ["metadata.json", "summary.json"]):
        raise ValueError("manifest does not cover exactly all recording files")
    return computed


def build_video(output: Path, fps: float = 20) -> dict:
    """CPU H.264 encoding only when actual timestamps match the requested FPS."""
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or not 0 < fps <= 120:
        raise ValueError("invalid video FPS")
    output = Path(output).resolve()
    summary = validate_recording(output)
    if not summary["passed"]:
        raise ValueError("recording has not passed the 100-frame diagnostic gate")
    if not summary["fixed_sim_timestep"] or abs(summary["observed_frame_dt"] - 1 / fps) > 1e-6:
        raise ValueError("recorded simulation timestamps do not match the requested video FPS")
    if summary["width"] % 2 or summary["height"] % 2:
        raise ValueError("H.264 yuv420p requires even dimensions")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("CPU ffmpeg and ffprobe must already be available")
    for name in ("demo.mp4", "video.json", "demo.partial.mp4", ".video.lock"):
        if (output / name).exists() or (output / name).is_symlink():
            raise FileExistsError("video output already exists")
    (output / ".video.lock").touch(exist_ok=False)
    temporary = output / "demo.partial.mp4"
    subprocess.run([
        ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
        "-threads", "2", "-framerate", str(fps), "-i", str(output / "rgb/%06d.png"),
        "-frames:v", str(summary["frame_count"]), "-an", "-c:v", "libx264",
        "-threads", "2", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(temporary),
    ], check=True, timeout=300, capture_output=True)
    probe = json.loads(subprocess.run([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=nb_read_frames,avg_frame_rate,duration,width,height", "-of", "json", str(temporary),
    ], check=True, timeout=120, capture_output=True, text=True).stdout)["streams"][0]
    if int(probe["nb_read_frames"]) != summary["frame_count"] or abs(float(Fraction(probe["avg_frame_rate"])) - fps) > 1e-6:
        raise ValueError("encoded frame count or FPS differs from recording")
    if (probe["width"], probe["height"]) != (summary["width"], summary["height"]):
        raise ValueError("encoded video dimensions differ from recording")
    if abs(float(probe["duration"]) - summary["frame_count"] / fps) > 0.001:
        raise ValueError("encoded video duration differs from recording")
    # Link fails if a destination appeared concurrently; no overwrites.
    os.link(temporary, output / "demo.mp4")
    temporary.unlink()
    result = {
        "path": str(output / "demo.mp4"), "fps": float(fps),
        "frame_count": summary["frame_count"], "duration_seconds": float(probe["duration"]),
        "fixed_sim_timestep_verified": True, "interpolated_frames": 0,
        "encoder": "CPU libx264", "sha256": _sha(output / "demo.mp4"),
        "source_manifest_sha256": _sha(output / "manifest.json"), "valid_sim_hours": 0,
    }
    _write_json(output / "video.json", result)
    return result
