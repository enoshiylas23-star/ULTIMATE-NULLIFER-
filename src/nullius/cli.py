"""Command-line interface for Nullius.

Phase 1 exposes ``nullius data profile <file>``. Subsequent phases add their
own subcommands behind the same two-level dispatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from nullius.data import load_dataset, profile_dataframe


def _ensure_utf8_stdout() -> None:
    """Reports use Unicode (…, –, ⚠); write UTF-8 even on cp1252 consoles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # not a TextIOWrapper / closed
            pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nullius",
        description=(
            "Nullius — open autonomous data scientist. "
            "Correctness, reproducibility, and statistical validity over "
            "impressive-looking answers."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    data = sub.add_parser("data", help="Data Understanding Engine (Phase 1)")
    data_sub = data.add_subparsers(dest="data_command", required=True)

    prof = data_sub.add_parser(
        "profile",
        help="Produce a Data Understanding Report for a dataset (never modifies it).",
    )
    prof.add_argument("file", type=Path, help="CSV/TSV/parquet file to profile")
    prof.add_argument("--name", default=None, help="dataset name (default: file stem)")
    prof.add_argument("--target", default=None, help="name of the target column")
    prof.add_argument("--sep", default=None, help="delimiter for CSV files")
    prof.add_argument("--encoding", default=None, help="text encoding for CSV files")
    prof.add_argument(
        "--out",
        type=Path,
        default=None,
        help="also write the complete machine-readable report as JSON here",
    )
    prof.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="stdout format (default: markdown)",
    )
    prof.add_argument(
        "--max-corr-cols",
        type=int,
        default=200,
        help="cap for the full pairwise correlation matrix",
    )
    return parser


def _cmd_profile(args: argparse.Namespace) -> int:
    sep = args.sep
    if sep is None and args.file.suffix.lower() == ".tsv":
        sep = "\t"
    kwargs = {}
    if sep is not None:
        kwargs["sep"] = sep
    if args.encoding is not None:
        kwargs["encoding"] = args.encoding

    df, digest = load_dataset(args.file, **kwargs)
    report = profile_dataframe(
        df,
        name=args.name,
        source_path=str(args.file),
        source_sha256=digest,
        target=args.target,
        max_correlation_columns=args.max_corr_cols,
    )

    if args.format == "json" or args.out is not None:
        payload = json.dumps(report.to_dict(), indent=2, sort_keys=False)
        if args.format == "json":
            print(payload)
        if args.out is not None:
            args.out.write_text(payload + "\n", encoding="utf-8")
            print(f"\nJSON report written to {args.out}", file=sys.stderr)
    if args.format == "markdown":
        print(report.render_markdown())
    return 0


_COMMANDS = {
    ("data", "profile"): _cmd_profile,
}


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdout()
    parser = _build_parser()
    args = parser.parse_args(argv)
    handler = _COMMANDS.get((args.command, args.data_command))
    if handler is None:  # pragma: no cover - argparse prevents this
        parser.print_help()
        return 2
    try:
        return handler(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # pragma: no cover - unexpected failures
        print(f"unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
