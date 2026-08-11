# transcribe

Local lecture/video speech-to-text CLI for **Apple Silicon**.

Wraps `ffmpeg` + [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) into one command that writes searchable markdown (`[HH:MM:SS]` lines), SRT/VTT for players, and JSON for reprocessing.

**macOS + Apple Silicon only** in v0.1. Offline after the first model download from Hugging Face.

## Install

Prerequisites:

- Apple Silicon Mac
- Python 3.11+
- [`ffmpeg`](https://ffmpeg.org/) on `PATH` (Homebrew, nix, etc.)
- [`uv`](https://github.com/astral-sh/uv) (recommended) or pip

```bash
# from GitHub (recommended)
uv tool install git+https://github.com/ngpestelos/transcribe

# or clone + editable
git clone https://github.com/ngpestelos/transcribe.git
cd transcribe
uv tool install -e .
```

Confirm the environment:

```bash
transcribe --check
transcribe --version
```

## Usage

```bash
# write .md/.srt/.json/… next to the source
transcribe lecture.mp4

# custom output directory
transcribe lecture.mp4 -o ~/Transcripts/

# domain vocabulary hint (optional)
transcribe lecture.mp4 --prompt "Lecture: markets, volume buzz, relative momentum."

# rebuild even if outputs exist
transcribe lecture.mp4 --force

# several files (skips any that already have .md + .srt)
transcribe part1.mp4 part2.mp4 -o ./out
```

### Flags

| Flag | Meaning |
|------|---------|
| `-o`, `--out DIR` | Output directory (default: source directory) |
| `--force` | Rebuild even when `.md` and `.srt` exist |
| `--model ID` | mlx-whisper model (default: `mlx-community/whisper-large-v3-turbo`) |
| `--language CODE` | Language code (default: `en`) |
| `--prompt TEXT` | `initial_prompt` for jargon / proper nouns |
| `--name BASE` | Output basename (single file only) |
| `--min-free-gb N` | Disk gate (default: `5`) |
| `--keep-audio` | Keep extracted mono AAC beside outputs |
| `--check` | Verify arch, ffmpeg, disk, mlx_whisper |
| `-V`, `--version` | Print version |

### Outputs

| File | Role |
|------|------|
| `.md` | Human read/search — `[HH:MM:SS] text` |
| `.srt` / `.vtt` | Video players |
| `.json` | Segments for reformat / reprocess |
| `.txt` / `.tsv` | Plain / tabular (from mlx-whisper) |

**Skip rule:** if both `.md` and `.srt` already exist for the basename, the file is skipped unless `--force`.

## How it works

1. Refuse to run off Apple Silicon or with less than ~5 GB free disk.
2. Extract mono 16 kHz AAC with ffmpeg (small; not full WAV).
3. Run mlx-whisper with a pinned default model.
4. Build a markdown transcript from the JSON segments.

First run downloads the model into the Hugging Face cache (`~/.cache/huggingface` by default, overridable with `HF_HOME`). That needs network once; later runs are local.

## Quality expectations

ASR drops small words and mangles jargon. Treat output as a **searchable index**, not publication-clean copy. Spot-check:

1. Opening topic matches audio
2. Domain proper nouns appear
3. Closing / Q&A is not pure hallucination
4. No extreme identical long-segment loops
5. SRT tracks speech within ~1 s on a few samples
6. Duration roughly matches the source

## Non-goals (v0.1)

- Linux / CUDA backends
- Mid-file resume / segment checkpoints
- Speaker diarization
- LLM chapter titles
- Cloud STT

## Development

```bash
git clone https://github.com/ngpestelos/transcribe.git
cd transcribe
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
pytest -q
transcribe --check
```

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE)
