from pathlib import Path

import pytest

from transcribe.pipeline import (
    TranscribeError,
    outputs_exist,
    resolve_base,
    require_apple_silicon,
)


def test_resolve_base():
    assert resolve_base(Path("/tmp/Foo Bar.mp4"), None) == "Foo Bar"
    assert resolve_base(Path("/tmp/Foo Bar.mp4"), "custom") == "custom"


def test_outputs_exist(tmp_path: Path):
    base = "talk"
    assert not outputs_exist(tmp_path, base)
    (tmp_path / f"{base}.md").write_text("x")
    assert not outputs_exist(tmp_path, base)
    (tmp_path / f"{base}.srt").write_text("y")
    assert outputs_exist(tmp_path, base)


def test_require_apple_silicon_on_this_host():
    # CI on non-Mac should skip; this host is expected to be arm64 macOS.
    try:
        require_apple_silicon()
    except TranscribeError as exc:
        pytest.skip(str(exc))
