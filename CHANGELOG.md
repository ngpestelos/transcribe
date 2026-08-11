# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

## [0.1.2] — 2026-08-11

### Fixed

- Find `mlx_whisper` beside the venv `python` without `Path.resolve()` on `sys.executable` (uv/venv pythons symlink into Homebrew/system; resolve jumped out of the tool env). Prefer the real sibling console script over the `-c` fallback.

## [0.1.1] — 2026-08-11

### Fixed

- Resolve `mlx_whisper` from the same environment as the `transcribe` binary (sibling of `sys.executable`). `uv tool install` only links `transcribe` onto `PATH`, so the previous PATH/`python -m` fallback failed on a clean tool install.

## [0.1.0] — 2026-08-11

### Added

- Initial public release of the `transcribe` CLI for Apple Silicon.
- Pipeline: disk gate → ffmpeg mono 16 kHz AAC extract → mlx-whisper → custom `[HH:MM:SS]` markdown, plus mlx SRT/VTT/JSON/TXT/TSV outputs.
- Default model: `mlx-community/whisper-large-v3-turbo`.
- Idempotent skip when both `.md` and `.srt` already exist for the basename (`--force` to rebuild).
- `transcribe --check` environment probe (arch, ffmpeg/ffprobe, disk, mlx_whisper).
- Optional `--prompt` (`initial_prompt`), `--model`, `--language`, `--out`, `--name`, `--keep-audio`, `--min-free-gb`.
- Multi-file invocation: `transcribe a.mp4 b.mp4`.
- Install via `uv tool install` / editable install; console script entry point `transcribe`.

[Unreleased]: https://github.com/ngpestelos/transcribe/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/ngpestelos/transcribe/releases/tag/v0.1.2
[0.1.1]: https://github.com/ngpestelos/transcribe/releases/tag/v0.1.1
[0.1.0]: https://github.com/ngpestelos/transcribe/releases/tag/v0.1.0
