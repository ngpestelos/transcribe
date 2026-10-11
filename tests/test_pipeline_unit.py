from pathlib import Path
import json
import subprocess
import threading

import pytest
from transcribe import pipeline

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
    assert not outputs_exist(tmp_path, base)  # legacy existence is not completion


def test_require_apple_silicon_on_this_host():
    # CI on non-Mac should skip; this host is expected to be arm64 macOS.
    try:
        require_apple_silicon()
    except TranscribeError as exc:
        pytest.skip(str(exc))


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(pipeline, "require_apple_silicon", lambda: None)
    monkeypatch.setattr(pipeline, "require_bin", lambda name: name)
    monkeypatch.setattr(pipeline, "disk_gate", lambda **kwargs: 10)
    monkeypatch.setattr(
        pipeline, "extract_audio", lambda source, target: target.write_bytes(b"normalized")
    )
    calls = []

    def run(audio, *, out_dir, base, **kwargs):
        calls.append((audio, out_dir, base))
        data = {"segments": [{"start": 0, "end": 1, "text": "hello"}]}
        contents = {
            "json": json.dumps(data),
            "txt": "hello\n",
            "srt": "1\n00:00:00,000 --> 00:00:01,000\nhello\n\n",
            "vtt": "WEBVTT\n\n00:00.000 --> 00:01.000\nhello\n\n",
            "tsv": "start\tend\ttext\n0\t1000\thello\n",
        }
        # Match the backend writer's suffix replacement behavior.
        for suffix, content in contents.items():
            (out_dir / base).with_suffix(f".{suffix}").write_text(content)

    monkeypatch.setattr(pipeline, "run_mlx_whisper", run)
    return run, calls


def source_file(tmp_path, name="talk.mp4"):
    source = tmp_path / name
    source.write_bytes(b"original recording")
    return source


def output_bytes(dest):
    return {p.name: p.read_bytes() for p in dest.iterdir() if p.is_file()}


@pytest.mark.parametrize("fails", [False, True])
def test_kept_audio_preserves_source(tmp_path, backend, monkeypatch, fails):
    source = source_file(tmp_path, "lecture.m4a")
    if fails:
        def fail(*args, **kwargs):
            raise TranscribeError("backend failed")
        monkeypatch.setattr(pipeline, "run_mlx_whisper", fail)
        with pytest.raises(TranscribeError):
            pipeline.transcribe_file(source, keep_audio=True, force=True)
        assert not (tmp_path / "lecture.audio.m4a").exists()
    else:
        pipeline.transcribe_file(source, keep_audio=True)
        assert (tmp_path / "lecture.audio.m4a").read_bytes() == b"normalized"
    assert source.read_bytes() == b"original recording"


@pytest.mark.parametrize("name", ["/tmp/escape", "../escape", "a/b", "a\\b", ".", "..", ""])
def test_invalid_name_rejected_before_writes(tmp_path, backend, name):
    source = source_file(tmp_path)
    dest = tmp_path / "out"
    with pytest.raises(TranscribeError, match="basename"):
        pipeline.transcribe_file(source, name=name, out_dir=dest)
    assert not dest.exists()


@pytest.mark.parametrize("name", [None, "Lecture 会议.02"])
def test_complete_dotted_unicode_names(tmp_path, backend, name):
    source = source_file(tmp_path, "Lecture 会议.01.mp4")
    result = pipeline.transcribe_file(source, name=name)
    base = name or source.stem
    for suffix in ("md", "srt", "vtt", "txt", "tsv", "json"):
        assert (tmp_path / f"{base}.{suffix}").is_file()
    assert result.base == base
    audio, staged, internal_base = backend[1][0]
    assert audio.parent == staged
    assert "." not in internal_base
    assert staged.parent == tmp_path


@pytest.mark.parametrize("force", [False, True])
def test_shared_output_stem_refuses_different_sources(tmp_path, backend, force):
    first_dir, second_dir = tmp_path / "first", tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    first = source_file(first_dir)
    second = source_file(second_dir)
    dest = tmp_path / "out"
    pipeline.transcribe_file(first, out_dir=dest)
    before = output_bytes(dest)
    with pytest.raises(TranscribeError, match="different source"):
        pipeline.transcribe_file(second, out_dir=dest, force=force)
    assert output_bytes(dest) == before


@pytest.mark.parametrize("suffix", ["md", "srt", "vtt", "txt", "tsv", "json", "audio.m4a", "complete.json", "lock"])
def test_direct_source_alias_rejected(tmp_path, backend, suffix):
    source = source_file(tmp_path, f"talk.{suffix}")
    with pytest.raises(TranscribeError, match="source"):
        pipeline.transcribe_file(source, name="talk", keep_audio=True, force=True)
    assert source.read_bytes() == b"original recording"
    assert backend[1] == []


@pytest.mark.parametrize("suffix", ["md", "complete.json", "lock"])
@pytest.mark.parametrize("link", ["symlink", "hardlink"])
def test_link_source_alias_rejected_before_sidecar_open(tmp_path, backend, suffix, link):
    source = source_file(tmp_path)
    alias = tmp_path / f"talk.{suffix}"
    if link == "symlink":
        alias.symlink_to(source)
    else:
        alias.hardlink_to(source)
    with pytest.raises(TranscribeError, match="source"):
        pipeline.transcribe_file(source, keep_audio=True, force=True)
    assert source.read_bytes() == b"original recording"
    assert backend[1] == []


def test_case_insensitive_source_alias_rejected(tmp_path, backend):
    source = source_file(tmp_path, "TALK.MD")
    with pytest.raises(TranscribeError, match="source"):
        pipeline.transcribe_file(source, name="talk", force=True)
    assert source.read_bytes() == b"original recording"


def test_failed_forced_rerun_preserves_completed_outputs(tmp_path, backend, monkeypatch):
    source = source_file(tmp_path)
    dest = tmp_path / "out"
    pipeline.transcribe_file(source, out_dir=dest)
    before = output_bytes(dest)
    def fail(audio, *, out_dir, base, **kwargs):
        (out_dir / f"{base}.srt").write_text("partial")
        raise TranscribeError("backend failed")
    monkeypatch.setattr(pipeline, "run_mlx_whisper", fail)
    with pytest.raises(TranscribeError):
        pipeline.transcribe_file(source, out_dir=dest, force=True)
    assert output_bytes(dest) == before
    assert pipeline.transcribe_file(source, out_dir=dest).skipped


@pytest.mark.parametrize("missing", ["json", "srt", "vtt", "txt", "tsv", "all"])
def test_missing_new_output_cannot_reuse_old_outputs(tmp_path, backend, monkeypatch, missing):
    source = source_file(tmp_path)
    dest = tmp_path / "out"
    pipeline.transcribe_file(source, out_dir=dest)
    before = output_bytes(dest)
    def incomplete(audio, *, out_dir, base, **kwargs):
        if missing != "all":
            backend[0](audio, out_dir=out_dir, base=base, **kwargs)
            (out_dir / base).with_suffix(f".{missing}").unlink()
    monkeypatch.setattr(pipeline, "run_mlx_whisper", incomplete)
    with pytest.raises(TranscribeError, match="missing"):
        pipeline.transcribe_file(source, out_dir=dest, force=True, model="new-model")
    assert output_bytes(dest) == before


def test_legacy_and_tampered_outputs_require_force(tmp_path, backend):
    source = source_file(tmp_path)
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "talk.md").write_text("legacy")
    (dest / "talk.srt").write_text("legacy")
    with pytest.raises(TranscribeError, match="--force"):
        pipeline.transcribe_file(source, out_dir=dest)
    pipeline.transcribe_file(source, out_dir=dest, force=True)
    assert outputs_exist(dest, "talk")
    assert pipeline.transcribe_file(source, out_dir=dest).skipped
    (dest / "talk.txt").write_text("changed")
    with pytest.raises(TranscribeError, match="--force"):
        pipeline.transcribe_file(source, out_dir=dest)


def test_changed_source_needs_a_fresh_forced_attempt(tmp_path, backend):
    source = source_file(tmp_path)
    pipeline.transcribe_file(source)
    source.write_bytes(b"revised recording")
    with pytest.raises(TranscribeError, match="--force"):
        pipeline.transcribe_file(source)
    pipeline.transcribe_file(source, force=True)
    assert pipeline.transcribe_file(source).skipped


def test_check_probes_failing_fallback(tmp_path, backend, monkeypatch):
    monkeypatch.setattr(pipeline.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(pipeline.shutil, "which", lambda name: None)
    attempted = []
    def fail(cmd, **kwargs):
        attempted.append(cmd)
        raise subprocess.CalledProcessError(1, cmd, stderr="No module named mlx_whisper")
    monkeypatch.setattr(pipeline.subprocess, "run", fail)
    with pytest.raises(TranscribeError, match="mlx_whisper"):
        pipeline.check_environment()
    assert attempted[0][0] == str(tmp_path / "python")
    assert attempted[0][-1] == "--help"


@pytest.mark.parametrize("fails", [False, True])
def test_check_probes_sibling(tmp_path, backend, monkeypatch, fails):
    monkeypatch.setattr(pipeline.sys, "executable", str(tmp_path / "python"))
    sibling = tmp_path / "mlx_whisper"
    sibling.write_text("script")
    attempted = []
    def run(cmd, **kwargs):
        attempted.append(cmd)
        assert kwargs["check"]
        if fails:
            raise OSError("cannot start")
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(pipeline.subprocess, "run", run)
    if fails:
        with pytest.raises(TranscribeError, match="mlx_whisper"):
            pipeline.check_environment()
    else:
        assert any("mlx_whisper" in line for line in pipeline.check_environment())
    assert attempted == [[str(sibling), "--help"]]


@pytest.mark.parametrize("interrupt", [False, True])
def test_publication_failure_restores_previous_set(tmp_path, backend, monkeypatch, interrupt):
    source = source_file(tmp_path)
    dest = tmp_path / "out"
    pipeline.transcribe_file(source, out_dir=dest, keep_audio=True)
    before = output_bytes(dest)
    original = Path.replace
    def replace(path, target):
        if Path(target) == dest / "talk.srt" and path.name == "result.srt":
            if interrupt:
                raise KeyboardInterrupt()
            raise PermissionError("publish denied")
        return original(path, target)
    monkeypatch.setattr(Path, "replace", replace)
    expected = KeyboardInterrupt if interrupt else TranscribeError
    with pytest.raises(expected):
        pipeline.transcribe_file(source, out_dir=dest, force=True, model="new-model")
    assert output_bytes(dest) == before
    assert pipeline.transcribe_file(source, out_dir=dest, keep_audio=True).skipped


def test_failed_rollback_retains_backups_and_invalidates_completion(tmp_path, backend, monkeypatch):
    source = source_file(tmp_path)
    dest = tmp_path / "out"
    pipeline.transcribe_file(source, out_dir=dest)
    before = output_bytes(dest)
    original_replace, original_copy = Path.replace, pipeline.shutil.copy2
    def replace(path, target):
        if Path(target) == dest / "talk.srt" and path.name == "result.srt":
            raise PermissionError("publish denied")
        return original_replace(path, target)
    def copy(source, target, **kwargs):
        if Path(source).parent.name.startswith(".transcribe-recovery-"):
            raise PermissionError("restore denied")
        return original_copy(source, target, **kwargs)
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(pipeline.shutil, "copy2", copy)
    with pytest.raises(TranscribeError, match="recover") as exc:
        pipeline.transcribe_file(source, out_dir=dest, force=True, model="new-model")
    backups = list(dest.glob(".transcribe-recovery-*"))
    assert len(backups) == 1
    assert str(backups[0]) in str(exc.value)
    assert (backups[0] / "talk.md").read_bytes() == before["talk.md"]
    assert (backups[0] / "talk.complete.json").read_bytes() == before["talk.complete.json"]
    assert not (dest / "talk.complete.json").exists()
    with pytest.raises(TranscribeError, match="--force"):
        pipeline.transcribe_file(source, out_dir=dest)


def test_source_alias_created_during_inference_is_rejected(tmp_path, backend, monkeypatch):
    source = source_file(tmp_path)
    def run(audio, *, out_dir, base, **kwargs):
        backend[0](audio, out_dir=out_dir, base=base, **kwargs)
        (tmp_path / "talk.md").hardlink_to(source)
    monkeypatch.setattr(pipeline, "run_mlx_whisper", run)
    with pytest.raises(TranscribeError, match="source"):
        pipeline.transcribe_file(source, force=True)
    assert source.read_bytes() == b"original recording"
    assert not (tmp_path / "talk.complete.json").exists()


@pytest.mark.parametrize("suffix,content", [
    ("srt", "not subtitles"), ("vtt", "not subtitles"),
    ("tsv", "wrong\theader\n"), ("txt", b"\xff"),
    ("srt", ""), ("vtt", "WEBVTT\n\n"), ("txt", ""),
    ("tsv", "start\tend\ttext\n"),
])
def test_invalid_backend_formats_fail_before_publication(tmp_path, backend, monkeypatch, suffix, content):
    source = source_file(tmp_path)
    def run(audio, *, out_dir, base, **kwargs):
        backend[0](audio, out_dir=out_dir, base=base, **kwargs)
        (out_dir / f"{base}.{suffix}").write_bytes(content if isinstance(content, bytes) else content.encode())
    monkeypatch.setattr(pipeline, "run_mlx_whisper", run)
    with pytest.raises(TranscribeError):
        pipeline.transcribe_file(source)
    assert not (tmp_path / "talk.md").exists()


def test_empty_text_segments_remain_valid_silence(tmp_path, backend, monkeypatch):
    source = source_file(tmp_path)
    def run(audio, *, out_dir, base, **kwargs):
        contents = {
            "json": json.dumps({"segments": [{"start": 0, "end": 1, "text": ""}]}),
            "srt": "1\n00:00:00,000 --> 00:00:01,000\n\n\n",
            "vtt": "WEBVTT\n\n00:00.000 --> 00:01.000\n\n\n",
            "txt": "\n", "tsv": "start\tend\ttext\n0\t1000\t\n",
        }
        for suffix, text in contents.items():
            (out_dir / f"{base}.{suffix}").write_text(text)
    monkeypatch.setattr(pipeline, "run_mlx_whisper", run)
    assert not pipeline.transcribe_file(source).skipped


def test_same_stem_writers_serialize_and_skip(tmp_path, backend, monkeypatch):
    source = source_file(tmp_path)
    started, release = threading.Event(), threading.Event()
    results, errors = [], []
    def run(*args, **kwargs):
        started.set()
        assert release.wait(5)
        backend[0](*args, **kwargs)
    monkeypatch.setattr(pipeline, "run_mlx_whisper", run)
    def transcribe():
        try:
            results.append(pipeline.transcribe_file(source))
        except BaseException as exc:
            errors.append(exc)
    first, second = threading.Thread(target=transcribe), threading.Thread(target=transcribe)
    first.start()
    try:
        assert started.wait(5)
        second.start()
    finally:
        release.set()
        first.join(5)
        if second.ident is not None:
            second.join(5)
    assert not first.is_alive() and not second.is_alive()
    assert not errors
    assert len(backend[1]) == 1
    assert sorted(result.skipped for result in results) == [False, True]


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("suffix,content", [
    ("srt", "1\n00:00:00,000 --> 00:00:01,000\n\n"),
    ("vtt", "WEBVTT\n\n00:00.000 --> 00:01.000\n\n"),
    ("tsv", "start\tend\ttext\n0\t1000\t \n"),
    ("srt", "1\n00:99:00,000 --> 00:99:01,000\nhello\n\n"),
    ("srt", "1\n00:00:99,000 --> 00:01:00,000\nhello\n\n"),
    ("srt", "1\n00:00:02,000 --> 00:00:01,000\nhello\n\n"),
    ("vtt", "WEBVTT\n\n99:00.000 --> 99:01.000\nhello\n\n"),
    ("vtt", "WEBVTT\n\n00:99.000 --> 01:00.000\nhello\n\n"),
    ("vtt", "WEBVTT\n\n00:02.000 --> 00:01.000\nhello\n\n"),
])
def test_caption_review_blockers_cannot_publish(tmp_path, backend, monkeypatch, suffix, content, prior):
    source = source_file(tmp_path)
    dest = tmp_path / "out"
    if prior:
        pipeline.transcribe_file(source, out_dir=dest)
        before = output_bytes(dest)

    def run(audio, *, out_dir, base, **kwargs):
        backend[0](audio, out_dir=out_dir, base=base, **kwargs)
        (out_dir / f"{base}.{suffix}").write_text(content)

    monkeypatch.setattr(pipeline, "run_mlx_whisper", run)
    with pytest.raises(TranscribeError):
        pipeline.transcribe_file(source, out_dir=dest, force=prior)
    if prior:
        assert output_bytes(dest) == before
        assert pipeline.transcribe_file(source, out_dir=dest).skipped
    else:
        for extension in (*pipeline.OUTPUT_SUFFIXES, "complete.json"):
            assert not (dest / f"talk.{extension}").exists()


def test_writer_multiline_quotes_blank_cues_long_hours_and_equal_times(tmp_path, backend, monkeypatch):
    source = source_file(tmp_path)
    text = '"Hello 世界"\nSecond line\n\nEnd'
    data = {"segments": [
        {"start": 0, "end": 0, "text": ""},
        {"start": 360001, "end": 360001, "text": text},
    ]}

    def run(audio, *, out_dir, base, **kwargs):
        contents = {
            "json": json.dumps(data), "txt": text + "\n",
            "srt": "1\n00:00:00,000 --> 00:00:00,000\n\n\n2\n100:00:01,000 --> 100:00:01,000\n" + text + "\n\n",
            "vtt": "WEBVTT\n\n00:00.000 --> 00:00.000\n\n\n100:00:01.000 --> 100:00:01.000\n" + text + "\n\n",
            "tsv": "start\tend\ttext\n0\t0\t\n360001000\t360001000\t" + text + "\n",
        }
        for suffix, content in contents.items():
            (out_dir / f"{base}.{suffix}").write_text(content)

    monkeypatch.setattr(pipeline, "run_mlx_whisper", run)
    assert not pipeline.transcribe_file(source).skipped
    assert text in (tmp_path / "talk.md").read_text()


def test_null_duration_uses_segment_fallback_in_pipeline(tmp_path, backend, monkeypatch):
    source = source_file(tmp_path)

    def run(audio, *, out_dir, base, **kwargs):
        backend[0](audio, out_dir=out_dir, base=base, **kwargs)
        path = out_dir / f"{base}.json"
        data = json.loads(path.read_text())
        data["duration"] = None
        path.write_text(json.dumps(data))

    monkeypatch.setattr(pipeline, "run_mlx_whisper", run)
    assert not pipeline.transcribe_file(source).skipped
    assert "00:00:01" in (tmp_path / "talk.md").read_text()
