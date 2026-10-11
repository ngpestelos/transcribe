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


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("keep_audio", [False, True])
@pytest.mark.parametrize("force", [False, True])
def test_batch_preserves_every_named_input(tmp_path, monkeypatch, capsys, reverse, keep_audio, force):
    monkeypatch.setattr(pipeline, "require_apple_silicon", lambda: None)
    monkeypatch.setattr(pipeline, "require_bin", lambda name: name)
    monkeypatch.setattr(pipeline, "disk_gate", lambda **kwargs: 10)
    monkeypatch.setattr(pipeline, "extract_audio", lambda source, target: target.write_bytes(b"normalized"))
    video, audio = tmp_path / "talk.mp4", tmp_path / "talk.audio.m4a"
    video.write_bytes(b"ORIGINAL VIDEO")
    audio.write_bytes(b"ORIGINAL AUDIO")
    attempted = []

    def run(extracted, *, out_dir, base, **kwargs):
        attempted.append(base)
        contents = {
            "json": '{"segments": []}', "srt": "", "txt": "",
            "vtt": "WEBVTT\n\n", "tsv": "start\tend\ttext\n",
        }
        for suffix, text in contents.items():
            (out_dir / f"{base}.{suffix}").write_text(text)

    monkeypatch.setattr(pipeline, "run_mlx_whisper", run)
    inputs = [audio, video] if reverse else [video, audio]
    options = (["--keep-audio"] if keep_audio else []) + (["--force"] if force else [])
    assert cli.main([*(str(source) for source in inputs), *options]) == 1
    assert video.read_bytes() == b"ORIGINAL VIDEO"
    assert audio.read_bytes() == b"ORIGINAL AUDIO"
    captured = capsys.readouterr()
    assert "aliases the source" in captured.err
    assert "done: talk.audio.m4a" in captured.out
    assert "done: talk.mp4" not in captured.out
    assert len(attempted) == 1


def test_missing_named_input_does_not_block_a_safe_completed_source(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "require_apple_silicon", lambda: None)
    monkeypatch.setattr(pipeline, "require_bin", lambda name: name)
    monkeypatch.setattr(pipeline, "disk_gate", lambda **kwargs: 10)
    monkeypatch.setattr(pipeline, "extract_audio", lambda source, target: target.write_bytes(b"audio"))

    def run(audio, *, out_dir, base, **kwargs):
        for suffix, text in {
            "json": '{"segments": []}', "srt": "", "txt": "",
            "vtt": "WEBVTT\n\n", "tsv": "start\tend\ttext\n",
        }.items():
            (out_dir / f"{base}.{suffix}").write_text(text)

    monkeypatch.setattr(pipeline, "run_mlx_whisper", run)
    source = tmp_path / "safe.mp4"
    source.write_bytes(b"original")
    pipeline.transcribe_file(source)
    assert cli.main([str(source), str(tmp_path / "missing.mp4")]) == 1
    captured = capsys.readouterr()
    assert "skip: safe.mp4" in captured.out
    assert "missing.mp4" in captured.err
    assert source.read_bytes() == b"original"
