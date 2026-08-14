# FastPop FST/XP-CLR Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add unified `jx fastpop fst` and `jx fastpop xpclr` sliding-window commands while preserving `jx fastpop structure`.

**Architecture:** Keep the existing FastPop/Admixture structure implementation as the default `structure` subcommand. Add Rust/PyO3 streaming backends for windowed FST and XP-CLR, with packed BED decoding and Rayon window parallelism. Python wrappers own CLI validation, population-list parsing, output/log naming, and dispatch.

**Tech Stack:** Rust 2021, PyO3 0.27, memmap2, Rayon, libm/statrs, Python argparse, PLINK SNP-major BED/BIM/FAM.

## Global Constraints

- The public entry points are `jx fastpop structure`, `jx fastpop fst`, and `jx fastpop xpclr`.
- `-window` and `-step` are required for both new analyses and use base-pair coordinates.
- FST defaults to Weir–Cockerham (`-method wc`) and supports `-matrix` with a `FID IID GROUP` file for all pairwise comparisons.
- XP-CLR defaults are `-map` omitted, `-maxsnps 600`, `-minsnps 10`, `-ld 0.95`, and `-rrate 1e-8`.
- XP-CLR follows hardingnj/xpclr v1.1.2: population 2 must not be fixed, multiallelic/all-missing sites are excluded, and the result includes the reference window fields plus `xpclr` and `xpclr_norm`.
- Do not add or stage files under `test/`, `.janusx_test/`, `benchmark/`, or `benchmarks/`.
- Work remains isolated in `.worktrees/fastpop-xpclr`; do not merge or push in this task.

---

### Task 1: Core formulas and window contracts

**Files:**
- Modify: `src/stats/fst.rs`
- Create: `src/stats/xpclr.rs`
- Test: Rust unit tests in the two source modules only; keep external fixtures in `/tmp`.

**Interfaces:**
- Preserve the existing per-site FST Rust functions for compatibility.
- Add tested helpers for window boundaries, WC window aggregation, Rogers–Huff correlation, XP-CLR likelihood integration, and normalization.

- [x] Write failing unit tests for weighted/mean WC FST windows and XP-CLR likelihood/window behavior.
- [x] Run the focused Rust tests and confirm failure because the new helpers do not exist.
- [x] Implement the minimal tested helpers.
- [x] Run focused tests and confirm they pass.

### Task 2: Rust streaming backends

**Files:**
- Modify: `src/stats/fst.rs`
- Create: `src/stats/xpclr.rs`
- Modify: `src/lib.rs`

**Interfaces:**
- Add `fst_bed_window_to_tsv(prefix, population1, population2_or_within, window, step, method, matrix, threads, output)`.
- Add `xpclr_bed_to_tsv(prefix, population1, population2, map, chromosome, window, step, maxsnps, minsnps, ld_cutoff, recombination_rate, threads, output)`.
- Both APIs return the number of windows/rows written and release the Python GIL while scanning.

- [x] Add direct BED/BIM/FAM population alignment and map validation.
- [x] Implement windowed BED decoding with packed genotype planes and Rayon-parallel window jobs.
- [x] Implement XP-CLR preprocessing, LD weights, selection-coefficient search, and streaming TSV output.
- [x] Run `cargo fmt`, `cargo check --lib`, and focused Rust tests.

### Task 3: FastPop CLI integration

**Files:**
- Modify: `python/janusx/script/fastpop.py`
- Create: `python/janusx/script/fastpop_fst.py`
- Create: `python/janusx/script/fastpop_xpclr.py`
- Modify: `python/janusx/script/JanusX.py`

**Interfaces:**
- `jx fastpop structure ...` dispatches to the existing Admixture parser.
- `jx fastpop fst -bfile PREFIX -p1 FILE -p2 FILE -window N -step N [-method wc|hudson] [-matrix] -o PREFIX`.
- `jx fastpop xpclr -bfile PREFIX -p1 FILE -p2 FILE -window N -step N [-map FILE] [-maxsnps N] [-minsnps N] [-ld FLOAT] [-rrate FLOAT] [-chr CHROM] -o PREFIX`.

- [x] Add parser smoke tests/commands for all three help paths.
- [x] Parse exact FID/IID or IID population lists and reject overlaps/missing IDs.
- [x] Create output directories and write predictable `.windowed.fst`/`.xpclr.tsv` plus logs.
- [x] Run Python compilation and CLI help checks.

### Task 4: Reference comparison and handoff

**Files:**
- No tracked test or benchmark fixtures; use `/tmp` only.

- [x] Compare WC window values against PLINK/VCFtools-compatible calculations, including weighted and mean columns.
- [x] Compare XP-CLR against hardingnj/xpclr v1.1.2 fixture/reference output within documented numerical tolerance.
- [x] Run direct PyO3 and CLI smoke tests with multiple threads.
- [x] Inspect staged paths, commit only source/CLI changes, and verify `main` remains untouched.
