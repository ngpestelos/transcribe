"""Transcription pipeline: disk gate → ffmpeg extract → mlx-whisper → markdown."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from transcribe.md import write_markdown_from_json

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
DEFAULT_LANGUAGE = "en"
MIN_FREE_GB = 5.0
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".webm", ".m4v", ".avi"}
AUDIO_EXTS = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac", ".wma"}


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
    if name:
        return name
    return source.stem


def outputs_exist(out_dir: Path, base: str) -> bool:
    return (out_dir / f"{base}.md").is_file() and (out_dir / f"{base}.srt").is_file()


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
    sibling = Path(sys.executable).resolve().parent / "mlx_whisper"
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


def ensure_json(out_dir: Path, base: str) -> Path:
    jp = out_dir / f"{base}.json"
    if not jp.is_file():
        raise TranscribeError(
            f"Expected JSON output missing: {jp}. mlx-whisper may have failed silently."
        )
    return jp


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
    """Transcribe one media file. Skip when .md and .srt already exist unless force."""
    require_apple_silicon()
    require_bin("ffmpeg")
    require_bin("ffprobe")

    source = source.expanduser().resolve()
    if not source.is_file():
        raise TranscribeError(f"Source not found: {source}")

    base = resolve_base(source, name)
    dest = (out_dir or source.parent).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    md_path = dest / f"{base}.md"
    srt_path = dest / f"{base}.srt"

    if not force and outputs_exist(dest, base):
        return TranscribeResult(
            source=source,
            out_dir=dest,
            base=base,
            skipped=True,
            md_path=md_path,
            srt_path=srt_path,
        )

    disk_gate(min_gb=min_free_gb, path=dest)

    suffix = source.suffix.lower()
    # Re-encode video always; pure audio still normalized to mono AAC for consistency.
    if suffix not in VIDEO_EXTS | AUDIO_EXTS and suffix:
        # Still try — ffmpeg may handle other containers.
        pass

    with tempfile.TemporaryDirectory(prefix="transcribe-") as tmp:
        tmp_path = Path(tmp)
        audio_path = tmp_path / f"{base.replace(' ', '_')}.m4a"
        extract_audio(source, audio_path)

        if keep_audio:
            kept = dest / f"{base}.m4a"
            shutil.copy2(audio_path, kept)

        run_mlx_whisper(
            audio_path,
            out_dir=dest,
            base=base,
            model=model,
            language=language,
            initial_prompt=initial_prompt,
        )

    jp = ensure_json(dest, base)
    write_markdown_from_json(
        jp,
        md_path,
        source_name=source.name,
        model=model,
    )

    if not srt_path.is_file():
        raise TranscribeError(f"Expected SRT output missing: {srt_path}")

    return TranscribeResult(
        source=source,
        out_dir=dest,
        base=base,
        skipped=False,
        md_path=md_path,
        srt_path=srt_path,
    )
