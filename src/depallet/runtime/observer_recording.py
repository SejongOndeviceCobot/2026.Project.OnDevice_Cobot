"""CPU-only streamed observer video for Isaac P0; never a policy input."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

import numpy as np


FORMAT = "isaac_p0_observer_video_v1"
ENCODER_TIMEOUT_SECONDS = 30
PROBE_TIMEOUT_SECONDS = 30
_RESERVED = (
    "observer.mp4", "observer.partial.mp4", "frames.jsonl", "metadata.json",
    "summary.json", "manifest.json", "ffmpeg.log", ".observer.lock",
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8
            or rgb.shape not in ((height, width, 3), (height, width, 4))):
        raise ValueError("RGB must be uint8 with configured H,W and 3 or 4 channels")
    value = np.ascontiguousarray(rgb[:, :, :3])
    if not np.any(value != value[0, 0]):
        raise ValueError("RGB is spatially uniform")
    return value


def _validate_frame_ledger(path: Path, expected_count: int, fps: int,
                           allow_frame_gaps: bool) -> int:
    """Re-read the recorder ledger and return its verified missing-frame count."""
    previous_time: float | None = None
    missing_source_frames = 0
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for count, line in enumerate(stream, start=1):
            if count > expected_count:
                raise ValueError("frames.jsonl frame count differs from recorder state")
            try:
                row = json.loads(line)
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError("frames.jsonl contains invalid JSON") from exc
            index = count - 1
            if not isinstance(row, dict) or type(row.get("frame_index")) is not int:
                raise ValueError("frames.jsonl contains invalid frame metadata")
            if row["frame_index"] != index:
                raise ValueError("frames.jsonl frame indices are not contiguous")
            timestamp = row.get("sim_time")
            if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                    or not math.isfinite(timestamp) or timestamp <= 0):
                raise ValueError("frames.jsonl contains invalid sim_time")
            timestamp = float(timestamp)
            missing = 0
            if previous_time is not None:
                delta = timestamp - previous_time
                intervals = round(delta * fps)
                if (intervals < 1 or abs(delta - intervals / fps) > 1e-6
                        or (not allow_frame_gaps and intervals != 1)):
                    raise ValueError("frames.jsonl violates observer sim_time cadence")
                missing = intervals - 1
            if allow_frame_gaps:
                if type(row.get("missing_source_frames_before")) is not int:
                    raise ValueError("frames.jsonl contains invalid gap metadata")
                if row["missing_source_frames_before"] != missing:
                    raise ValueError("frames.jsonl gap metadata disagrees with sim_time")
            elif "missing_source_frames_before" in row:
                raise ValueError("frames.jsonl contains unexpected gap metadata")
            missing_source_frames += missing
            previous_time = timestamp
    if count != expected_count:
        raise ValueError("frames.jsonl frame count differs from recorder state")
    return missing_source_frames


def _probe_observer(ffprobe: str, path: Path, width: int, height: int,
                    fps: int, frame_count: int) -> None:
    """Validate MP4 container metadata without a duration-scaled frame scan."""
    probe_result = subprocess.run([
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=nb_frames,avg_frame_rate,width,height,duration_ts,time_base",
        "-of", "json", str(path),
    ], check=True, timeout=PROBE_TIMEOUT_SECONDS, capture_output=True, text=True)
    streams = json.loads(probe_result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ValueError("observer output must contain exactly one video stream")
    probe = streams[0]
    try:
        encoded_count = int(probe["nb_frames"])
        encoded_fps = Fraction(probe["avg_frame_rate"])
        encoded_size = (int(probe["width"]), int(probe["height"]))
        encoded_duration = int(probe["duration_ts"]) * Fraction(probe["time_base"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("ffprobe returned invalid observer stream metadata") from exc
    if encoded_size != (width, height):
        raise ValueError("encoded observer resolution differs from 1920x1080")
    if encoded_fps != fps:
        raise ValueError("encoded observer FPS differs from 30")
    if encoded_count != frame_count:
        raise ValueError("encoded observer frame count differs from frames.jsonl")
    if encoded_duration != Fraction(frame_count, fps):
        raise ValueError("encoded observer duration differs from frames.jsonl")


class ObserverRecorder:
    """Stream exact 1080p/30 simulator frames to a CPU libx264 child.

    The encoder stays in this process group. ``close`` stops only that direct
    child with bounded waits and leaves incomplete evidence for diagnosis.
    """

    def __init__(self, output: Path, width: int = 1920, height: int = 1080,
                 fps: int = 30, allow_frame_gaps: bool = False):
        if type(width) is not int or type(height) is not int or type(fps) is not int:
            raise ValueError("observer dimensions and FPS must be integers")
        if (width, height, fps) != (1920, 1080, 30):
            raise ValueError("observer recording is fixed at 1920x1080 and 30 FPS")
        ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
        if not ffmpeg or not ffprobe:
            raise RuntimeError("CPU ffmpeg and ffprobe must already be available")
        output = Path(output)
        if output.is_symlink():
            raise ValueError("output directory must not be a symlink")
        output.mkdir(parents=True, exist_ok=True)
        self.output = output.resolve()
        if any((self.output / name).exists() or (self.output / name).is_symlink()
               for name in _RESERVED):
            raise FileExistsError("observer output already exists; use a fresh directory")
        self.width, self.height, self.fps = width, height, fps
        if type(allow_frame_gaps) is not bool:raise ValueError("Boolean gap policy required")
        self.allow_frame_gaps=allow_frame_gaps
        self.missing_source_frames=0
        self.count = 0
        self.finished = False
        self._closed = False
        self._previous_time: float | None = None
        self._ffprobe = ffprobe
        self._temporary = self.output / "observer.partial.mp4"
        self._lock = self.output / ".observer.lock"
        with self._lock.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()) + "\n")
        (self.output / "frames.jsonl").touch(exist_ok=False)
        self._log = (self.output / "ffmpeg.log").open("xb")
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-n",
            "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", f"{width}x{height}", "-framerate", str(fps),
            "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2",
            "-filter_threads", "2", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(self._temporary),
        ]
        try:
            self._process = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=self._log, bufsize=0, close_fds=True,
                start_new_session=False,
            )
        except BaseException:
            self._log.close()
            self._closed = True
            self._lock.unlink(missing_ok=True)
            raise
        if self._process.stdin is None:
            self.close()
            raise RuntimeError("ffmpeg stdin pipe was not created")

    def _check_row(self, row: dict) -> tuple[dict, float]:
        if not isinstance(row, dict):
            raise ValueError("row must be a dictionary")
        if "frame_index" in row:
            raise ValueError("row contains reserved recorder fields")
        timestamp = row.get("sim_time")
        if (isinstance(timestamp, bool) or not isinstance(timestamp, (int, float))
                or not math.isfinite(timestamp) or timestamp <= 0):
            raise ValueError("sim_time must be finite and strictly positive")
        timestamp = float(timestamp)
        missing=0
        if self._previous_time is not None:
            delta=timestamp-self._previous_time
            intervals=round(delta*self.fps)
            if (intervals<1 or abs(delta-intervals/self.fps)>1e-6
                    or (not self.allow_frame_gaps and intervals!=1)):
                raise ValueError("observer sim_time must follow strict 30 FPS cadence")
            missing=intervals-1
        record=json.loads(_json(row))
        if self.allow_frame_gaps:record['missing_source_frames_before']=missing
        return record, timestamp

    def append(self, rgb: np.ndarray, row: dict) -> int:
        if self.finished or self._closed:
            raise RuntimeError("observer recording is already closed")
        image = _checked_rgb(rgb, self.width, self.height)
        record, timestamp = self._check_row(row)
        if self._process.poll() is not None:
            self.close()
            raise RuntimeError("ffmpeg exited before observer recording finished")
        payload = memoryview(image).cast("B")
        try:
            while payload:
                written = self._process.stdin.write(payload)
                if not written:
                    raise BrokenPipeError("ffmpeg accepted no RGB bytes")
                payload = payload[written:]
        except (BrokenPipeError, OSError) as exc:
            self.close()
            raise RuntimeError("ffmpeg stopped accepting observer RGB frames") from exc
        index = self.count
        record["frame_index"] = index
        with (self.output / "frames.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(_json(record) + "\n")
            stream.flush()
        self.missing_source_frames+=record.get("missing_source_frames_before",0)
        self._previous_time = timestamp
        self.count += 1
        return index

    def _close_input_and_wait(self) -> int:
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        try:
            return self._process.wait(timeout=ENCODER_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self._process.kill()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("ffmpeg did not stop after bounded cleanup") from exc
            raise RuntimeError("ffmpeg timed out while finalizing observer video")

    def finish(self, metadata: dict) -> dict:
        if self.finished or self._closed:
            raise RuntimeError("observer recording is already closed")
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")
        if (metadata.get("training_dataset", False) is not False
                or metadata.get("policy_input", False) is not False
                or metadata.get("valid_sim_hours", 0) != 0
                or metadata.get("accepted_sim_hours", 0) != 0):
            raise ValueError("observer video cannot be training data, policy input, or accepted hours")
        _json(metadata)
        if self.count == 0:
            raise ValueError("observer video requires at least one frame")
        try:
            returncode = self._close_input_and_wait()
        finally:
            self._log.close()
            self._closed = True
            self._lock.unlink(missing_ok=True)
        if returncode != 0:
            raise RuntimeError(f"CPU ffmpeg exited with status {returncode}")
        frames_path = self.output / "frames.jsonl"
        with frames_path.open("a", encoding="utf-8") as stream:
            stream.flush()
            os.fsync(stream.fileno())
        ledger_missing = _validate_frame_ledger(
            frames_path, self.count, self.fps, self.allow_frame_gaps)
        if ledger_missing != self.missing_source_frames:
            raise ValueError("frames.jsonl gap count differs from recorder state")
        _probe_observer(
            self._ffprobe, self._temporary, self.width, self.height,
            self.fps, self.count)
        final = self.output / "observer.mp4"
        os.link(self._temporary, final)
        self._temporary.unlink()
        summary = {
            "format": FORMAT, "passed": True, "width": self.width,
            "height": self.height, "fps": self.fps, "frame_count": self.count,
            "duration_seconds": self.count / self.fps, "encoder": "CPU libx264",
            "threads": 2, "strict_sim_cadence": self.missing_source_frames==0,
            "missing_source_frames":self.missing_source_frames,
            "gap_policy":"record_actual_frames_only" if self.allow_frame_gaps else "strict",
            "video_time_matches_sim_time":self.missing_source_frames==0,
            "training_dataset": False, "policy_input": False,
            "valid_sim_hours": 0, "accepted_sim_hours": 0,
            "video_sha256": _sha(final),
        }
        _write_json(self.output / "metadata.json", {
            "format": FORMAT, "width": self.width, "height": self.height,
            "fps": self.fps, "training_dataset": False, "policy_input": False,
            "valid_sim_hours": 0, "accepted_sim_hours": 0,
            "run_metadata": metadata,
        })
        _write_json(self.output / "summary.json", summary)
        files = ["observer.mp4", "frames.jsonl", "ffmpeg.log", "metadata.json", "summary.json"]
        _write_json(self.output / "manifest.json", {
            "format": FORMAT, "frame_count": self.count,
            "files": {name: _sha(self.output / name) for name in files},
            "training_dataset": False, "policy_input": False,
            "valid_sim_hours": 0, "accepted_sim_hours": 0,
        })
        self.finished = True
        return summary

    def close(self) -> None:
        if self.finished or self._closed:
            return
        if self._process.stdin is not None and not self._process.stdin.closed:
            try:
                self._process.stdin.close()
            except OSError:
                pass
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError("ffmpeg did not stop after bounded cleanup") from exc
        self._log.close()
        self._closed = True
