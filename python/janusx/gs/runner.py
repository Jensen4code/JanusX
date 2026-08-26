# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib
import os
import threading
from typing import Optional

from .config import GsConfig

_GS_MODULE_LOCK = threading.Lock()
_GS_MODULE = None


def _get_gs_script():
    global _GS_MODULE
    mod = _GS_MODULE
    if mod is not None:
        return mod
    with _GS_MODULE_LOCK:
        mod = _GS_MODULE
        if mod is None:
            mod = importlib.import_module("janusx.gs.workflow")
            _GS_MODULE = mod
    return mod


def prewarm_gs_runner() -> None:
    """
    Preload and cache GS script module for low-latency first call.
    """
    _ = _get_gs_script()


def run_gs_cli(
    argv: list[str],
    *,
    log: bool = True,
    return_result: bool = True,
):
    return run_gs_argv(
        argv=argv,
        log=log,
        return_result=return_result,
    )


def run_gs_args(
    args,
    *,
    log: bool = True,
    return_result: bool = True,
):
    gs_script = _get_gs_script()
    parsed_args = args
    if args is None:
        parsed_args = gs_script.parse_args(None)  # type: ignore[attr-defined]
    elif isinstance(args, (list, tuple)):
        parsed_args = gs_script.parse_args([str(x) for x in args])  # type: ignore[attr-defined]
    return gs_script._run_gs_pipeline(  # type: ignore[attr-defined]
        args=parsed_args,
        log=log,
        return_result=return_result,
    )


def run_gs_argv(
    argv: list[str] | None = None,
    *,
    log: bool = True,
    return_result: bool = True,
):
    gs_script = _get_gs_script()
    args = gs_script.parse_args(argv)  # type: ignore[attr-defined]
    return run_gs_args(
        args=args,
        log=bool(log),
        return_result=bool(return_result),
    )


def run_gs_config(
    config: GsConfig,
    *,
    out: Optional[str] = None,
    prefix: Optional[str] = None,
    log: bool = True,
):
    argv = config.build_gs_argv(out_dir=out, out_prefix=prefix)
    # Keep the Python wrapper's explicit GBLUP/rrBLUP methods compatible with
    # the unified CLI without leaking a route override into the caller's
    # process.  `blup()` intentionally leaves the environment untouched so
    # the normal automatic dispatch and an explicit GS_BLUP override remain
    # available.
    force_mode: str | None = None
    kernels = tuple(str(x).strip().lower() for x in config.gblup_kernels)
    if kernels == ("a",) and not bool(config.rrblup) and not bool(config.blup):
        force_mode = "0"
    elif (
        len(kernels) == 0
        and bool(config.rrblup)
        and not bool(config.blup)
    ):
        force_mode = "1"

    had_previous = "GS_BLUP" in os.environ
    previous = os.environ.get("GS_BLUP")
    if force_mode is not None:
        os.environ["GS_BLUP"] = force_mode
    try:
        return run_gs_cli(argv, log=log, return_result=True)
    finally:
        if had_previous:
            assert previous is not None
            os.environ["GS_BLUP"] = previous
        else:
            os.environ.pop("GS_BLUP", None)
