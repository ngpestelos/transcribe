"""CLI entry point for `transcribe`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transcribe import __version__
from transcribe.pipeline import (
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL,
    MIN_FREE_GB,
    TranscribeError,
    check_environment,
    transcribe_file,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcribe",
        description=(
            "Local lecture/video speech-to-text for Apple Silicon "
            "(ffmpeg + mlx-whisper). Offline after first model download."
        ),
    )
    p.add_argument(
        "sources",
        nargs="*",
        type=Path,
        help="Video or audio file(s) to transcribe",
    )
    p.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: same directory as each source)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when .md and .srt already exist",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"mlx-whisper model id (default: {DEFAULT_MODEL})",
    )
    p.add_argument(
        "--language",
        default=DEFAULT_LANGUAGE,
        help=f"Language code (default: {DEFAULT_LANGUAGE})",
    )
    p.add_argument(
        "--prompt",
        default=None,
        help="Optional initial_prompt for domain vocabulary / proper nouns",
    )
    p.add_argument(
        "--name",
        default=None,
        help="Output basename (single source only; default: source stem)",
    )
    p.add_argument(
        "--min-free-gb",
        type=float,
        default=MIN_FREE_GB,
        help=f"Refuse to run below this free disk (default: {MIN_FREE_GB})",
    )
    p.add_argument(
        "--keep-audio",
        action="store_true",
        help="Keep extracted mono AAC next to outputs",
    )
    p.add_argument(
        "--check",
        action="store_true",
        help="Verify ffmpeg, arch, disk, and mlx_whisper; exit",
    )
    p.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.check:
            for line in check_environment(min_gb=args.min_free_gb):
                print(line)
            print("ok")
            return 0

        if not args.sources:
            parser.error("provide at least one source file, or use --check")

        if args.name and len(args.sources) > 1:
            parser.error("--name requires a single source file")

        exit_code = 0
        for source in args.sources:
            try:
                result = transcribe_file(
                    source,
                    out_dir=args.out,
                    force=args.force,
                    model=args.model,
                    language=args.language,
                    initial_prompt=args.prompt,
                    name=args.name,
                    min_free_gb=args.min_free_gb,
                    keep_audio=args.keep_audio,
                )
            except TranscribeError as exc:
                print(f"error: {source}: {exc}", file=sys.stderr)
                exit_code = 1
                continue

            if result.skipped:
                print(
                    f"skip: {result.source.name} "
                    f"(exists: {result.md_path} + {result.srt_path})"
                )
            else:
                print(f"done: {result.source.name}")
                print(f"  md:  {result.md_path}")
                print(f"  srt: {result.srt_path}")
                print(f"  dir: {result.out_dir}")

        return exit_code

    except TranscribeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
