# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

### Fixed

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

[Unreleased]: https://github.com/ngpestelos/transcribe/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ngpestelos/transcribe/releases/tag/v0.1.0
