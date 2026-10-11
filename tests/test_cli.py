import json
from pathlib import Path
import tomllib

import pytest

from transcribe import __version__, cli, pipeline


@pytest.mark.parametrize("failure", ["json", "schema", "write"])
def test_batch_continues_after_expected_failure(tmp_path, monkeypatch, capsys, failure):
    monkeypatch.setattr(pipeline, "require_apple_silicon", lambda: None)
    monkeypatch.setattr(pipeline, "require_bin", lambda name: name)
    monkeypatch.setattr(pipeline, "disk_gate", lambda **kwargs: 10)
    monkeypatch.setattr(pipeline, "extract_audio", lambda source, target: target.write_bytes(b"audio"))
    first, second = tmp_path / "first.mp4", tmp_path / "second.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    attempted = []
    original_write = Path.write_text
    def write(path, content, **kwargs):
        if failure == "write" and path.name == "result.md" and len(attempted) == 1:
            raise PermissionError("Markdown write denied")
        return original_write(path, content, **kwargs)
    monkeypatch.setattr(Path, "write_text", write)
    def run(audio, *, out_dir, base, **kwargs):
        attempted.append(base)
        contents = {"json": json.dumps({"segments": []}), "srt": "", "txt": "", "vtt": "WEBVTT\n\n", "tsv": "start\tend\ttext\n"}
        if len(attempted) == 1:
            if failure in {"json", "schema"}:
                contents["json"] = "{" if failure == "json" else '{"segments": [3]}'
        for suffix, text in contents.items():
            (out_dir / base).with_suffix(f".{suffix}").write_text(text)
    monkeypatch.setattr(pipeline, "run_mlx_whisper", run)
    assert cli.main([str(first), str(second)]) == 1
    assert len(attempted) == 2
    captured = capsys.readouterr()
    assert "error:" in captured.err and "first.mp4" in captured.err
    assert "done: second.mp4" in captured.out
    assert (tmp_path / "second.md").is_file()


def test_cli_and_build_version_agree(capsys):
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    if "version" in project["project"]:
        assert project["project"]["version"] == __version__
    else:
        assert "version" in project["project"]["dynamic"]
        assert project["tool"]["hatch"]["version"]["path"] == "src/transcribe/__init__.py"
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"transcribe {__version__}"
