"""Transcription pipeline: disk gate → ffmpeg extract → mlx-whisper → markdown."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from transcribe.md import (
    TranscriptValidationError,
    validate_transcript,
    write_markdown_from_json,
)

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_LANGUAGE = "en"
MIN_FREE_GB = 5.0
OUTPUT_SUFFIXES = ("md", "srt", "vtt", "txt", "tsv", "json")


class TranscribeError(Exception):
    """User-facing pipeline failure."""


@dataclass(frozen=True)
class TranscribeResult:
    source: Path
    out_dir: Path
    base: str
    skipped: bool
    md_path: Path | None
    srt_path: Path | None


def require_apple_silicon() -> None:
    if sys.platform != "darwin":
        raise TranscribeError(
            f"transcribe v0.1 is macOS + Apple Silicon only (got {sys.platform})."
        )
    machine = platform.machine().lower()
    if machine not in {"arm64", "aarch64"}:
        raise TranscribeError(
            f"transcribe v0.1 requires Apple Silicon (got machine={machine})."
        )


def require_bin(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise TranscribeError(
            f"`{name}` not found on PATH. Install ffmpeg (e.g. nix or Homebrew)."
        )
    return path


def free_gb(path: Path | None = None) -> float:
    target = path or Path.home()
    usage = shutil.disk_usage(target)
    return usage.free / (1024**3)


def disk_gate(min_gb: float = MIN_FREE_GB, path: Path | None = None) -> float:
    free = free_gb(path)
    if free < min_gb:
        raise TranscribeError(
            f"Free disk {free:.1f} GB < {min_gb:.1f} GB — free space or move HF cache "
            f"(HF_HOME) before running."
        )
    return free


def resolve_base(source: Path, name: str | None) -> str:
    base = source.stem if name is None else name
    if not base or base in {".", ".."} or any(c in base for c in ("/", "\\", "\0")):
        raise TranscribeError("Output name must be a single basename; use --out for a directory.")
    return base


def artifact_paths(out_dir: Path, base: str) -> dict[str, Path]:
    paths = {suffix: out_dir / f"{base}.{suffix}" for suffix in OUTPUT_SUFFIXES}
    paths["audio.m4a"] = out_dir / f"{base}.audio.m4a"
    return paths


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_completion(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError):
        return None
    return record if isinstance(record, dict) else None


def outputs_exist(out_dir: Path, base: str) -> bool:
    """Only a completion record and unchanged required artifacts count as complete."""
    record = read_completion(out_dir / f"{base}.complete.json")
    if not record or record.get("version") != 1:
        return False
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict) or not set(OUTPUT_SUFFIXES) <= artifacts.keys():
        return False
    paths = artifact_paths(out_dir, base)
    if not artifacts.keys() <= paths.keys():
        return False
    return all(
        isinstance(digest, str) and paths[suffix].is_file()
        and file_digest(paths[suffix]) == digest
        for suffix, digest in artifacts.items()
    )


def reject_source_aliases(source: Path, paths: list[Path]) -> None:
    for path in paths:
        if (
            str(path.resolve()).casefold() == str(source).casefold()
            or (source.exists() and path.exists() and path.samefile(source))
        ):
            raise TranscribeError(f"Output path aliases the source recording: {path}")


@contextmanager
def output_lock(path: Path, source: Path) -> Iterator[None]:
    # Import here so ordinary imports and unit tests also work on non-POSIX CI.
    import fcntl

    # Keep this inode: deleting it could split queued writers across two locks.
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "r+b") as stream:
        source_stat, lock_stat = source.stat(), os.fstat(stream.fileno())
        if (source_stat.st_dev, source_stat.st_ino) == (lock_stat.st_dev, lock_stat.st_ino):
            raise TranscribeError(f"Lock path aliases the source recording: {path}")
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def validate_backend_outputs(stage: Path) -> None:
    """Check promised UTF-8 formats before any final output is replaced."""
    contents = {}
    for suffix in ("json", "srt", "vtt", "txt", "tsv"):
        path = stage / f"result.{suffix}"
        if not path.is_file():
            raise TranscribeError(f"Expected {suffix.upper()} output missing: {path}")
        contents[suffix] = path.read_text(encoding="utf-8")

    try:
        data = json.loads(contents["json"])
    except ValueError as exc:
        raise TranscriptValidationError(f"Invalid JSON from mlx_whisper: {exc}") from exc
    validate_transcript(data)
    has_speech = any(segment["text"].strip() for segment in data["segments"])
    if has_speech and (not contents["srt"].strip() or not contents["txt"].strip()):
        raise TranscribeError("Incomplete caption or text output from mlx_whisper")

    def caption_time(value: str) -> int:
        match = re.fullmatch(r"(?:(\d+):)?([0-5]\d):([0-5]\d)[.,](\d{3})", value)
        if not match:
            raise TranscribeError("Invalid caption timestamp from mlx_whisper")
        try:
            hours, minutes, seconds, milliseconds = (int(part or 0) for part in match.groups())
        except ValueError as exc:
            raise TranscribeError("Invalid caption timestamp from mlx_whisper") from exc
        return ((hours * 60 + minutes) * 60 + seconds) * 1000 + milliseconds

    def captions(content: str, indexed: bool) -> None:
        seen_cue = False
        has_payload = False
        for block in re.split(r"\n\s*\n", content.strip()):
            if not block:
                continue
            lines = block.splitlines()
            offset = 1 if indexed else 0
            new_cue = not seen_cue or "-->" in lines[0] or (
                indexed and len(lines) > 1 and "-->" in lines[1]
            )
            if new_cue:
                if len(lines) <= offset or (indexed and not lines[0].isdigit()):
                    raise TranscribeError("Invalid caption cue from mlx_whisper")
                times = lines[offset].split(" --> ")
                if len(times) != 2 or caption_time(times[1]) < caption_time(times[0]):
                    raise TranscribeError("Invalid caption timing from mlx_whisper")
                seen_cue = True
                lines = lines[offset + 1:]
            # Paragraphs inside multiline speech continue the preceding cue.
            has_payload |= any(line.strip() for line in lines)
        if has_speech and not has_payload:
            raise TranscribeError("Incomplete caption speech output from mlx_whisper")

    captions(contents["srt"], indexed=True)
    vtt = contents["vtt"].splitlines()
    if not vtt or vtt[0].strip() != "WEBVTT":
        raise TranscribeError("Invalid VTT output from mlx_whisper")
    captions("\n".join(vtt[1:]), indexed=False)
    # mlx-whisper prints literal text, rather than CSV-escaped fields.
    rows = [line.split("\t", 2) for line in contents["tsv"].splitlines()]
    if not rows or rows[0] != ["start", "end", "text"]:
        raise TranscribeError("Invalid TSV header from mlx_whisper")
    seen_row = False
    has_tsv_payload = False
    for row in rows[1:]:
        if row == [""]:
            continue
        if len(row) == 1 and seen_row:
            # The backend writes embedded newlines in segment text literally.
            has_tsv_payload |= bool(row[0].strip())
            continue
        if len(row) != 3 or not row[0].isascii() or not row[1].isascii():
            raise TranscribeError("Invalid TSV row from mlx_whisper")
        try:
            valid = row[0].isdigit() and row[1].isdigit() and int(row[1]) >= int(row[0])
        except ValueError:
            valid = False
        if not valid:
            raise TranscribeError("Invalid TSV row from mlx_whisper")
        seen_row = True
        has_tsv_payload |= bool(row[2].strip())
    if has_speech and not has_tsv_payload:
        raise TranscribeError("Incomplete TSV speech output from mlx_whisper")


def publish_outputs(stage: Path, paths: dict[str, Path], completion: Path, keep_audio: bool) -> None:
    """Publish completion last; restore backups if a replacement fails."""
    backups = Path(tempfile.mkdtemp(prefix=".transcribe-recovery-", dir=completion.parent))
    targets = [*paths.values(), completion]
    saved = []
    changed = []
    retain_backups = False
    try:
        # Finish every backup before invalidating the old completion record.
        for target in targets:
            if target.exists():
                if not target.is_file():
                    raise TranscribeError(f"Output path is not a regular file: {target}")
                shutil.copy2(target, backups / target.name)
                saved.append(target)
        try:
            completion.unlink(missing_ok=True)
            changed.append(completion)
            for suffix, target in paths.items():
                staged = stage / ("audio.m4a" if suffix == "audio.m4a" else f"result.{suffix}")
                if suffix == "audio.m4a" and not keep_audio:
                    if target.exists():
                        target.unlink()
                        changed.append(target)
                    continue
                staged.replace(target)
                changed.append(target)
            (stage / "complete.json").replace(completion)
        except (OSError, KeyboardInterrupt) as publication_error:
            if not changed:
                raise  # Invalidating completion failed before any artifact changed.
            try:
                # The completion record is restored only after every artifact.
                for target in [p for p in changed if p != completion] + [completion]:
                    if target in saved:
                        restored = stage / f"restore-{target.name}"
                        shutil.copy2(backups / target.name, restored)
                        restored.replace(target)
                    else:
                        target.unlink(missing_ok=True)
            except (OSError, KeyboardInterrupt) as restore_error:
                retain_backups = True
                recovery_message = f"Publication and restore failed; recover previous outputs from {backups}: {restore_error}"
                try:
                    completion.unlink(missing_ok=True)
                except OSError as invalidation_error:
                    recovery_message += f"; completion removal failed: {invalidation_error}"
                if isinstance(publication_error, KeyboardInterrupt) or isinstance(restore_error, KeyboardInterrupt):
                    interrupt = publication_error if isinstance(publication_error, KeyboardInterrupt) else restore_error
                    interrupt.add_note(recovery_message)
                    raise interrupt
                raise TranscribeError(recovery_message) from restore_error
            raise
    finally:
        if not retain_backups:
            shutil.rmtree(backups)


def extract_audio(source: Path, audio_path: Path) -> None:
    """Extract mono 16 kHz AAC (small; not full WAV)."""
    ffmpeg = require_bin("ffmpeg")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        str(audio_path),
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        err = (exc.stderr or exc.stdout or str(exc)).strip()
        raise TranscribeError(f"ffmpeg extract failed:\n{err}") from exc


def resolve_mlx_whisper() -> list[str]:
    """Return argv prefix that runs the mlx_whisper CLI.

    Prefer the console script next to ``sys.executable`` (uv tool / venv installs
    put ``transcribe`` on PATH but not always ``mlx_whisper``). Fall back to PATH,
    then to ``mlx_whisper.cli:main`` via the same interpreter.
    """
    # Do not Path.resolve() sys.executable — uv/venv pythons are often symlinks
    # into the system interpreter; resolve() leaves the venv bin/ directory.
    exe_dir = Path(sys.executable).parent
    sibling = exe_dir / "mlx_whisper"
    if sibling.is_file():
        return [str(sibling)]
    on_path = shutil.which("mlx_whisper")
    if on_path:
        return [on_path]
    # Library present but no console script on PATH (e.g. stripped tool env)
    return [
        sys.executable,
        "-c",
        (
            "import sys; from mlx_whisper.cli import main; "
            "sys.argv = ['mlx_whisper'] + sys.argv[1:]; main()"
        ),
    ]


def run_mlx_whisper(
    audio_path: Path,
    *,
    out_dir: Path,
    base: str,
    model: str,
    language: str,
    initial_prompt: str | None,
) -> None:
    """Invoke mlx_whisper CLI (same flags as the proven spike)."""
    cmd = resolve_mlx_whisper()
    cmd.extend(
        [
            str(audio_path),
            "--model",
            model,
            "--language",
            language,
            "--task",
            "transcribe",
            "--output-dir",
            str(out_dir),
            "--output-name",
            base,
            "--output-format",
            "all",
            "--condition-on-previous-text",
            "False",
        ]
    )
    if initial_prompt:
        cmd.extend(["--initial-prompt", initial_prompt])

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise TranscribeError(
            "mlx_whisper not found. Reinstall with: "
            "uv tool install git+https://github.com/ngpestelos/transcribe"
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise TranscribeError(
            f"mlx_whisper failed with exit code {exc.returncode}."
        ) from exc


def check_environment(*, min_gb: float = MIN_FREE_GB) -> list[str]:
    """Return human-readable status lines; raise TranscribeError on hard fails."""
    lines: list[str] = []
    require_apple_silicon()
    lines.append(f"arch: {platform.machine()} (ok)")
    lines.append(f"python: {sys.version.split()[0]}")
    for name in ("ffmpeg", "ffprobe"):
        p = require_bin(name)
        lines.append(f"{name}: {p}")
    mlx_cmd = resolve_mlx_whisper()
    try:
        subprocess.run([*mlx_cmd, "--help"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        details = (getattr(exc, "stderr", None) or str(exc)).strip()
        raise TranscribeError(
            f"mlx_whisper cannot start: {details}. Reinstall with: "
            "uv tool install git+https://github.com/ngpestelos/transcribe"
        ) from exc
    lines.append(f"mlx_whisper: {' '.join(mlx_cmd)}")
    free = disk_gate(min_gb=min_gb)
    lines.append(f"free disk: {free:.1f} GB (min {min_gb:.1f})")
    hf = Path.home() / ".cache" / "huggingface"
    lines.append(f"HF cache (default): {hf}")
    return lines


def transcribe_file(
    source: Path,
    *,
    out_dir: Path | None = None,
    force: bool = False,
    model: str = DEFAULT_MODEL,
    language: str = DEFAULT_LANGUAGE,
    initial_prompt: str | None = None,
    name: str | None = None,
    min_free_gb: float = MIN_FREE_GB,
    keep_audio: bool = False,
) -> TranscribeResult:
    """Transcribe one source, preserving originals and previously completed outputs."""
    try:
        require_apple_silicon()
        require_bin("ffmpeg")
        require_bin("ffprobe")
        source = source.expanduser().resolve()
        if not source.is_file():
            raise TranscribeError(f"Source not found: {source}")
        base = resolve_base(source, name)
        dest = (out_dir or source.parent).expanduser().resolve()
        paths = artifact_paths(dest, base)
        completion = dest / f"{base}.complete.json"
        lock = dest / f"{base}.lock"
        protected = [*paths.values(), completion, lock]
        reject_source_aliases(source, protected)
        dest.mkdir(parents=True, exist_ok=True)

        with output_lock(lock, source):
            reject_source_aliases(source, protected)
            source_identity = {"path": str(source), "sha256": file_digest(source)}
            record = read_completion(completion)
            record_source = record.get("source") if record else None
            if (
                isinstance(record_source, dict)
                and isinstance(record_source.get("path"), str)
                and record_source["path"] != str(source)
            ):
                raise TranscribeError(
                    f"Output basename {base!r} belongs to a different source: "
                    f"{record_source.get('path')}. Use --name or another --out directory."
                )
            if not force:
                complete = (
                    record_source == source_identity and outputs_exist(dest, base)
                    and (not keep_audio or "audio.m4a" in record["artifacts"])
                )
                if complete:
                    return TranscribeResult(source, dest, base, True, paths["md"], paths["srt"])
                if completion.exists() or any(p.exists() or p.is_symlink() for p in paths.values()):
                    raise TranscribeError(
                        f"Existing outputs for {base!r} are incomplete, changed or unverified; "
                        "use --force to rebuild them."
                    )

            disk_gate(min_gb=min_free_gb, path=dest)
            with tempfile.TemporaryDirectory(prefix=".transcribe-", dir=dest) as tmp:
                stage = Path(tmp)
                audio = stage / "audio.m4a"
                extract_audio(source, audio)
                run_mlx_whisper(
                    audio, out_dir=stage, base="result", model=model,
                    language=language, initial_prompt=initial_prompt,
                )
                validate_backend_outputs(stage)
                write_markdown_from_json(
                    stage / "result.json", stage / "result.md",
                    source_name=source.name, model=model,
                )
                artifacts = {
                    suffix: file_digest(stage / f"result.{suffix}")
                    for suffix in OUTPUT_SUFFIXES
                }
                if keep_audio:
                    artifacts["audio.m4a"] = file_digest(audio)
                (stage / "complete.json").write_text(
                    json.dumps({"version": 1, "source": source_identity, "artifacts": artifacts}),
                    encoding="utf-8",
                )
                # Check again immediately before any final file or backup is opened.
                reject_source_aliases(source, protected)
                if file_digest(source) != source_identity["sha256"]:
                    raise TranscribeError(f"Source changed during transcription: {source}; rerun it.")
                publish_outputs(stage, paths, completion, keep_audio)
        return TranscribeResult(source, dest, base, False, paths["md"], paths["srt"])
    except (OSError, json.JSONDecodeError, UnicodeError, TranscriptValidationError) as exc:
        raise TranscribeError(f"Cannot transcribe {source}: {exc}") from exc
