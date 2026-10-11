"""Build human-readable markdown transcripts from mlx-whisper JSON."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


class TranscriptValidationError(ValueError):
    """Expected malformed transcript data."""


def validate_transcript(data: Any) -> None:
    """Validate fields used by the renderer; empty segments represent silence."""
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        raise TranscriptValidationError("JSON must contain a segments list")

    def timestamp(value: Any, field: str) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TranscriptValidationError(f"{field} must be a finite nonnegative number")
        try:
            valid = math.isfinite(value) and value >= 0
        except OverflowError:
            valid = False
        if not valid:
            raise TranscriptValidationError(f"{field} must be a finite nonnegative number")

    if data.get("duration") is not None:
        timestamp(data["duration"], "duration")
    for field in ("text", "language"):
        if field in data and not isinstance(data[field], str):
            raise TranscriptValidationError(f"{field} must be a string")
    for index, segment in enumerate(data["segments"]):
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            raise TranscriptValidationError(f"segment {index} must contain string text")
        for field in ("start", "end"):
            timestamp(segment.get(field), f"segment {index} {field}")
        if segment["end"] < segment["start"]:
            raise TranscriptValidationError(f"segment {index} end precedes start")


def format_ts(seconds: float | int | None) -> str:
    """Format seconds as HH:MM:SS (floor, never negative)."""
    s = max(0, int(float(seconds or 0)))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def duration_seconds(data: dict[str, Any]) -> float:
    """Best-effort duration from whisper JSON."""
    if data.get("duration") is not None:
        return float(data["duration"])
    segments = data.get("segments") or []
    if not segments:
        return 0.0
    return float(segments[-1].get("end") or 0)


def json_to_markdown(
    data: dict[str, Any],
    *,
    source_name: str,
    model: str,
) -> str:
    """Render [HH:MM:SS] line transcript with a short header."""
    validate_transcript(data)
    dur_s = duration_seconds(data)
    lines = [
        f"# Transcript — {source_name}",
        "",
        f"- **Model:** `{model}`",
        f"- **Duration:** {format_ts(dur_s)} ({dur_s / 60:.1f} min)",
        "",
        "---",
        "",
    ]
    for seg in data.get("segments") or []:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"[{format_ts(seg.get('start'))}] {text}")
    lines.append("")
    return "\n".join(lines)


def write_markdown_from_json(
    json_path: Path,
    md_path: Path,
    *,
    source_name: str,
    model: str,
) -> Path:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    md_path.write_text(
        json_to_markdown(data, source_name=source_name, model=model),
        encoding="utf-8",
    )
    return md_path
