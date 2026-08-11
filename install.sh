#!/usr/bin/env bash
# Install the `transcribe` CLI via uv tool install (preferred) or pipx.
set -euo pipefail

REPO_URL="${TRANSCRIBE_REPO_URL:-https://github.com/ngpestelos/transcribe}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "error: ffmpeg not on PATH. Install ffmpeg first (Homebrew, nix, etc.)." >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  echo "Installing with uv tool install from ${REPO_URL} …"
  uv tool install --force "git+${REPO_URL}"
elif command -v pipx >/dev/null 2>&1; then
  echo "Installing with pipx from ${REPO_URL} …"
  pipx install --force "git+${REPO_URL}"
else
  echo "error: need uv (recommended) or pipx on PATH." >&2
  exit 1
fi

echo
transcribe --version
transcribe --check
echo "Install OK. Try: transcribe lecture.mp4"
