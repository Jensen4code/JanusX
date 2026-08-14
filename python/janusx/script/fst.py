# -*- coding: utf-8 -*-
"""Per-site population differentiation from PLINK BED files."""

from __future__ import annotations

import argparse
import os
import socket
import time
from pathlib import Path

from ._common.cli_args import (
    add_common_out_arg,
    add_common_prefix_arg,
    add_common_thread_arg,
)
from ._common.cli_core import CliArgumentParser, cli_help_formatter, minimal_help_epilog
from ._common.config_render import emit_cli_configuration
from ._common.log import setup_logging
from ._common.outprefix import apply_output_prefix_compat
from ._common.pathcheck import format_path_for_display
from ._common.progress import CliStatus, format_elapsed, log_success
from ._common.threads import detect_effective_threads

try:
    from janusx import janusx as jxrs
except Exception:
    jxrs = None


def _normalize_prefix(value: str) -> str:
    raw = str(value).strip()
    low = raw.lower()
    if low.endswith((".bed", ".bim", ".fam")):
        return raw[:-4]
    return raw


def _require_backend() -> None:
    if jxrs is None or not hasattr(jxrs, "fst_bed_to_tsv"):
        raise RuntimeError(
            "Rust FST backend is unavailable. Rebuild/install JanusX before running fst."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = CliArgumentParser(
        prog="jx fst",
        formatter_class=cli_help_formatter(),
        epilog=minimal_help_epilog(
            [
                "jx fst -bfile cohort -within groups.tsv -o results/cohort",
                "jx fst -bfile cohort -within groups.tsv -method hudson -t 8",
            ]
        ),
        description=(
            "Bitwise per-site FST from PLINK SNP-major BED. "
            "Default method: Weir-Cockerham (wc)."
        ),
    )
    data = parser.add_argument_group("Input")
    data.add_argument(
        "-bfile",
        "--bfile",
        required=True,
        help="PLINK BED/BIM/FAM prefix.",
    )
    data.add_argument(
        "-within",
        "--within",
        required=True,
        help="PLINK-compatible cluster file: FID IID GROUP.",
    )
    data.add_argument(
        "-method",
        "--method",
        choices=("wc", "hudson"),
        default="wc",
        help="FST estimator (default: wc).",
    )
    out = parser.add_argument_group("Output / runtime")
    add_common_out_arg(out, default=".", help_profile="current_dir")
    add_common_prefix_arg(out, default=None, help_profile="input_basename")
    add_common_thread_arg(
        out,
        default_threads=max(1, int(detect_effective_threads())),
        help_profile="rust_auto",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    _require_backend()
    if int(args.thread) <= 0:
        raise ValueError("--thread must be > 0")
    bfile = _normalize_prefix(args.bfile)
    missing = [
        f"{bfile}{suffix}"
        for suffix in (".bed", ".bim", ".fam")
        if not Path(f"{bfile}{suffix}").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "incomplete PLINK input prefix; missing: " + ", ".join(missing)
        )
    if not os.path.isfile(args.within):
        raise FileNotFoundError(args.within)

    stem = os.path.basename(os.path.normpath(bfile))
    out_dir, output_prefix, _out_stem = apply_output_prefix_compat(
        args,
        stem,
        fallback_prefix="fst",
    )
    os.makedirs(out_dir, exist_ok=True)
    output_path = f"{output_prefix}.{args.method}.fst"
    log_path = f"{output_prefix}.fst.log"
    logger = setup_logging(log_path)
    emit_cli_configuration(
        logger,
        app_title="JanusX fst",
        config_title="Bitwise FST",
        host=socket.gethostname(),
        sections=[
            ("Input", [("BFILE", bfile), ("Within", args.within)]),
            ("Estimator", [("Method", args.method), ("Default method", "wc")]),
            ("Runtime", [("Threads", int(args.thread)), ("Output", output_path)]),
        ],
    )

    start = time.monotonic()
    with CliStatus("Computing FST...", enabled=True) as task:
        n_sites, n_pairs = jxrs.fst_bed_to_tsv(
            bfile,
            args.within,
            output_path,
            method=args.method,
            threads=int(args.thread),
        )
        task.complete("Computing FST ...Finished")
    elapsed = max(0.0, time.monotonic() - start)
    log_success(
        logger,
        f"FST finished [{format_elapsed(elapsed)}]; "
        f"sites={int(n_sites)}, population_pairs={int(n_pairs)}",
    )
    log_success(logger, f"FST table saved: {format_path_for_display(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
