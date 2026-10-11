# transcribe

[![CI](https://github.com/ngpestelos/transcribe/actions/workflows/ci.yml/badge.svg)](https://github.com/ngpestelos/transcribe/actions/workflows/ci.yml)

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

# several files (skips verified completed output sets)
transcribe part1.mp4 part2.mp4 -o ./out
```

### Flags

| Flag | Meaning |
|------|---------|
| `-o`, `--out DIR` | Output directory (default: source directory) |
| `--force` | Rebuild existing or unverified outputs |
| `--model ID` | mlx-whisper model (default: `mlx-community/whisper-large-v3-turbo`) |
| `--language CODE` | Language code (default: `en`) |
| `--prompt TEXT` | `initial_prompt` for jargon / proper nouns |
| `--name BASE` | Single basename; spaces, Unicode and dots are supported |
| `--min-free-gb N` | Disk gate (default: `5`) |
| `--keep-audio` | Keep normalized AAC as `<base>.audio.m4a` |
| `--check` | Verify arch, ffmpeg, disk and backend startup with `--help` |
| `-V`, `--version` | Print version |

### Outputs

| File | Role |
|------|------|
| `.md` | Human read/search — `[HH:MM:SS] text` |
| `.srt` / `.vtt` | Video players |
| `.json` | Segments for reformat / reprocess |
| `.txt` / `.tsv` | Plain / tabular (from mlx-whisper) |
| `.audio.m4a` | Normalized audio, when `--keep-audio` is used |
| `.complete.json` | Source identity and hashes of the completed artifacts |
| `.lock` | Persistent file lock for writers sharing this basename |

**Skip rule:** a matching source and unchanged completed output set are skipped unless `--force`.
Older outputs without a completion record, changed files and incomplete sets require `--force`.
Requesting `--keep-audio` requires that the completed set contains retained audio.

Different sources sharing a basename in one destination are refused, including with `--force`.
Choose a different `--name` or `--out` directory. Names with path components are rejected.
Any output or sidecar that aliases the source recording is rejected before it is opened for writing.

## How it works

1. Refuse to run off Apple Silicon or with less than ~5 GB free disk.
2. Lock the destination basename and stage mono 16 kHz AAC on that filesystem.
3. Run mlx-whisper in staging and validate its JSON, captions and tabular output.
4. Render Markdown, publish the output set and write its completion record last.

Failed reruns preserve the previous completed set. Detected publication errors trigger a rollback.
If rollback fails, the error names the recovery directory and completion is invalidated.
Keep that directory until you have recovered its files. Publication across flat files is not atomic
and does not guarantee durability after power loss.

Expected JSON and filesystem failures are reported per source. Later files in a batch continue;
any failed source makes the command exit nonzero. Empty segments are valid for silent recordings.

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

### CI and Dependabot

- **CI** (`.github/workflows/ci.yml`): unit tests on Python 3.11–3.13 (no mlx/torch install — those are runtime Mac deps), plus sdist/wheel build. Gate job: `ci-success`.
- **Dependabot** (weekly): `pip` and `github-actions`.
- **Auto-merge**: Dependabot PRs enable squash auto-merge; they land only after `ci-success` is green and `main` is up to date (branch protection).

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE)
