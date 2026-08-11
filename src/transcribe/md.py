"""Build human-readable markdown transcripts from mlx-whisper JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


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
