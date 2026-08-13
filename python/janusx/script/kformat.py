"""Fast packed kfile filtering and conversion."""

from __future__ import annotations

import argparse
import logging
import socket
from pathlib import Path

from janusx import janusx as jxrs

from ._common.cli_args import (
    add_common_out_arg,
    add_common_prefix_arg,
    add_common_thread_arg,
)
from ._common.cli_core import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.config_render import emit_cli_configuration
from ._common.log import setup_logging
from ._common.outprefix import apply_output_prefix_compat
from ._common.progress import log_success
from ._common.threads import detect_effective_threads, format_requested_thread_usage


def build_parser() -> CliArgumentParser:
    parser = CliArgumentParser(
        prog="jx kformat",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx kformat -kfile panel_k31 -maf 0.02 -fmt bfile -o out/panel",
                "jx kformat -kfile panel_k31 -extract keep_kmers.txt -fmt kfile -o out/panel_subset",
            ]
        ),
        description=(
            "Filter an existing JanusX kfile by MAF or a k-mer list and convert it "
            "to PLINK bfile or another kfile using packed bitwise Rust kernels."
        ),
    )
    req = parser.add_argument_group("Input Arguments")
    req.add_argument(
        "-kfile",
        "--kfile",
        required=True,
        type=str,
        help="Input JanusX kfile prefix (.meta.json/.bkmer/.bsite/.idv).",
    )
    opt = parser.add_argument_group("Optional Arguments")
    opt.add_argument(
        "-fmt",
        "--fmt",
        dest="format",
        choices=["bfile", "kfile"],
        default="bfile",
        help="Output format: bfile or kfile (default: %(default)s).",
    )
    opt.add_argument(
        "-maf",
        "--maf",
        type=float,
        default=0.0,
        help="Keep k-mer rows with MAF >= threshold (default: %(default)s).",
    )
    opt.add_argument(
        "-extract",
        "--extract",
        type=str,
        default=None,
        metavar="FILE",
        help="Keep k-mers listed one per line; blank lines and # comments are ignored.",
    )
    add_common_thread_arg(
        opt,
        default_threads=detect_effective_threads(),
        help_profile="packed_rust",
    )
    add_common_out_arg(opt, default=".", help_profile="converted_results")
    add_common_prefix_arg(opt, default=None, help_profile="input_basename")
    opt.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output components.",
    )
    return parser


def read_extract_list(path: str) -> list[str]:
    values: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            values.append(value)
    if not values:
        raise ValueError(f"--extract file is empty: {path}")
    return values


def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 <= float(args.maf) <= 0.5):
        raise ValueError("--maf must be within [0, 0.5].")
    if int(args.thread) <= 0:
        raise ValueError("-t/--thread must be > 0.")
    if args.extract is not None and not Path(str(args.extract)).is_file():
        raise ValueError(f"--extract file not found: {args.extract}")


def _output_components(outprefix: str, output_format: str) -> str:
    suffixes = ".bed/.bim/.fam" if output_format == "bfile" else ".bkmer/.bsite/.idv/.meta.json"
    return f"{outprefix} ({suffixes})"


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))

    detected_threads = int(detect_effective_threads())
    requested_threads = int(args.thread)
    threads = min(requested_threads, detected_threads)
    out_dir, outprefix, _out_stem = apply_output_prefix_compat(
        args,
        "kformat",
        fallback_prefix="kformat",
    )
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log_path = f"{outprefix}.kformat.log"
    logger: logging.Logger = setup_logging(log_path)
    extract_kmers = read_extract_list(str(args.extract)) if args.extract is not None else None
    emit_cli_configuration(
        logger,
        app_title="JanusX - kformat",
        config_title="KFILE FORMAT CONFIG",
        host=socket.gethostname(),
        sections=[
            (
                "Input/Filter",
                [
                    ("Kfile", str(args.kfile)),
                    ("MAF threshold", float(args.maf)),
                    ("Extract rows", len(extract_kmers) if extract_kmers is not None else "None"),
                ],
            ),
            (
                "Runtime",
                [
                    (
                        "Threads",
                        format_requested_thread_usage(
                            requested_threads=requested_threads,
                            using_threads=threads,
                            detected_threads=detected_threads,
                        ),
                    ),
                    ("Output format", str(args.format)),
                    ("Force", bool(args.force)),
                    ("Log file", log_path),
                ],
            ),
        ],
        footer_rows=[("Output", _output_components(outprefix, str(args.format)))],
    )
    try:
        summary = jxrs.kformat_run(
            input=str(args.kfile),
            out=str(outprefix),
            format=str(args.format),
            maf=float(args.maf),
            extract_kmers=extract_kmers,
            thread=int(threads),
            force=bool(args.force),
        )
    except Exception as exc:
        logger.error(str(exc))
        return 1
    log_success(
        logger,
        f"Converted {int(summary['n_kmers_in'])} to {int(summary['n_kmers_out'])} k-mers "
        f"across {int(summary['n_samples'])} samples.",
    )
    logger.info(f"Output: {_output_components(outprefix, str(args.format))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
