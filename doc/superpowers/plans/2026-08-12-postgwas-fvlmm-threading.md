# PostGWAS FvLMM Threading Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the apparent single-thread stall after FvLMM fine-mapping preflight and make its dense effective-LD work obey `-t`, without changing variant identity, numerical results, memory limits, or output order.

**Architecture:** First replace the repeated 5.69-million-row BIM search with a one-pass target resolver shared by all fine-mapping routes. Then expose Python and Rust stage diagnostics, route dense products through the existing controlled f64 BLAS dispatcher, and use a bounded Rayon pool only for independent SNP-row residual work. Validate every optimization against the current scalar equations and the Rice6048 locus before integration.

**Tech Stack:** Python 3.13, pandas, NumPy, Rust, PyO3/numpy, nalgebra SVD, Rayon, CBLAS/Accelerate/OpenBLAS, pytest, cargo test, maturin, `/usr/bin/time -l`.

## Global Constraints

- Preserve the statistical model, sidecar schema, clumping threshold, SuSiE algorithm, output schemas, allele convention, SNP order, folded-SNP restoration, and credible-set membership.
- BIM resolution must be `O(BIM rows + prepared rows + matching candidates)` time and `O(prepared rows + matching candidates)` Python object memory.
- Exact SNP ID is preferred when supplied; allele-pair identity is enforced when supplied; coordinate fallback is accepted only when unambiguous.
- `valid_indices` must remain sorted by original SNP row, and parallel execution must be deterministic for `-t 1,2,4,8`.
- Effective-LD agreement uses `atol=1e-10, rtol=1e-10`; downstream PIP and posterior-mean agreement uses `atol=1e-6, rtol=1e-5`.
- Dense BLAS regions own the requested thread budget; Rayon work must not run concurrently with multithreaded BLAS.
- macOS Accelerate is a CPU backend, not a GPU path; Linux must continue to use the repository's OpenBLAS/backend dispatch.
- Keep the rank-revealing SVD unchanged unless measured evidence identifies it as a bottleneck.
- Peak RSS must remain below both the preflight estimate and the requested `-mem` limit.
- Rice6048 outputs and benchmark logs stay under `/tmp`; do not commit them and do not push any branch.
- Use `/Users/jingxianfu/mamba/envs/jxfu/bin/python` and its maturin installation. Do not export `DYLD_LIBRARY_PATH` for Python or `jx` runs on macOS.

## File Map

- Modify `python/janusx/script/postgwas.py`: streaming BIM identity resolution, call-site migration, stage timing, and kernel diagnostic logging.
- Modify `src/stats/mixed_ld.rs`: controlled f64 GEMM products, bounded Rayon row work, deterministic Gram construction, and kernel timing metadata.
- Modify `test/test_postgwas_fvlmm_effective_ld.py`: resolver contracts, timing contracts, numerical parity, thread determinism, and route integration tests.
- No new production module is needed: the BIM resolver belongs beside the existing fine-mapping identity rules, and the numerical kernels remain isolated in `mixed_ld.rs`.

---

### Task 1: Replace full-BIM materialization with a one-pass target resolver

**Files:**
- Modify: `python/janusx/script/postgwas.py:2019-2145,2842-2888,3763-3795,4232-4260`
- Test: `test/test_postgwas_fvlmm_effective_ld.py`

**Interfaces:**
- Produces: `_postgwas_resolve_prepared_bim_rows(genotype_prefix, prepared) -> tuple[list[int], list[tuple[str, int, str | None, str | None, str | None]]]`.
- Internal seam for deterministic scan testing: `_postgwas_scan_bim_candidates(handle, target_sites) -> tuple[dict[tuple[str, int], list[tuple[int, tuple[str, int, str | None, str | None, str | None]]]], int]`.
- Later tasks consume the returned indices and metadata in prepared-row order; no later task may re-read the full BIM.

- [ ] **Step 1: Write failing tests for identity parity and a single scan**

  Add synthetic fixtures covering duplicate coordinates, distinct SNP IDs/alleles, unambiguous coordinate fallback, named-ID mismatch, allele ambiguity, duplicate row reuse, malformed BIM rows, and an instrumented iterable that raises if iterated twice. Assert that the new resolver returns source BIM indices plus metadata in prepared order and retains only target-coordinate candidates.

  ```python
  class SinglePassLines:
      def __init__(self, lines):
          self.lines = lines
          self.iterations = 0

      def __iter__(self):
          self.iterations += 1
          if self.iterations != 1:
              raise AssertionError("BIM stream was scanned more than once")
          yield from self.lines


  def test_streaming_bim_candidate_scan_is_single_pass_and_bounded():
      lines = SinglePassLines(
          ["1 rs0 0 10 A G\n"]
          + [f"1 junk{i} 0 {1000 + i} A C\n" for i in range(10_000)]
          + ["3 rs_target 0 18414420 C T\n"]
      )
      candidates, row_count = postgwas._postgwas_scan_bim_candidates(
          lines, {("3", 18414420)}
      )
      assert lines.iterations == 1
      assert row_count == 10_002
      assert sum(map(len, candidates.values())) == 1
      assert candidates[("3", 18414420)][0][0] == 10_001
  ```

- [ ] **Step 2: Run the resolver tests and verify RED**

  Run:

  ```bash
  /Users/jingxianfu/mamba/envs/jxfu/bin/python -m pytest -q \
    test/test_postgwas_fvlmm_effective_ld.py -k 'streaming_bim or resolve_prepared_bim'
  ```

  Expected: FAIL because `_postgwas_scan_bim_candidates` and `_postgwas_resolve_prepared_bim_rows` do not exist.

- [ ] **Step 3: Implement target preparation and the streaming scan**

  Validate every prepared position and optional identity before opening the BIM; build `target_sites: set[tuple[str, int]]`; enumerate every nonempty BIM line so returned indices remain exact PLINK row indices; parse and retain only target sites. Return the total source-row count for diagnostics, and translate `OSError`/malformed rows into the existing `FineMapSkip` wording.

  ```python
  def _postgwas_scan_bim_candidates(handle, target_sites):
      candidates = {site: [] for site in target_sites}
      source_index = 0
      for line_number, line in enumerate(handle, start=1):
          tokens = line.split()
          if not tokens:
              continue
          if len(tokens) < 6:
              raise ValueError(f"Malformed BIM row at line {line_number}")
          chrom = _normalize_chr(tokens[0])
          pos = int(float(tokens[3]))
          metadata = (
              chrom,
              pos,
              _postgwas_variant_token(tokens[1]),
              _postgwas_variant_token(tokens[4], allele=True),
              _postgwas_variant_token(tokens[5], allele=True),
          )
          site = (chrom, pos)
          if site in candidates:
              candidates[site].append((source_index, metadata))
          source_index += 1
      return candidates, source_index
  ```

- [ ] **Step 4: Apply the existing identity rules to retained candidates**

  Reuse `_postgwas_variant_token` and `_postgwas_variant_allele_pair`. Preserve the current ID-first, allele-pair, ambiguity, and duplicate-resolution errors. Return `selected_indices` and `selected_metadata` with equal lengths and prepared-row ordering.

- [ ] **Step 5: Migrate every fine-mapping call site and delete the full-list path**

  Replace the FvLMM effective-LD route, raw canonicalization route, and sample-matched raw-LD route. Build `selected_bim_meta`, signatures, and ambiguous-site counts directly from returned metadata. Remove `_postgwas_read_bim_identity_rows` and `_postgwas_match_prepared_variants_to_bim` only after `rg` shows no callers.

- [ ] **Step 6: Run focused and route tests**

  ```bash
  /Users/jingxianfu/mamba/envs/jxfu/bin/python -m pytest -q \
    test/test_postgwas_fvlmm_effective_ld.py -k \
    'streaming_bim or resolve_prepared_bim or duplicate_coordinates or raw_route or sample_matched'
  ```

  Expected: PASS; the counting fixture reports one iteration and one retained candidate despite 10,002 BIM rows.

- [ ] **Step 7: Commit the independently reviewable resolver change**

  ```bash
  git add python/janusx/script/postgwas.py test/test_postgwas_fvlmm_effective_ld.py
  git commit -m "perf(postgwas): stream BIM identity resolution"
  ```

### Task 2: Add Python and Rust stage diagnostics

**Files:**
- Modify: `python/janusx/script/postgwas.py:2708-3225,4550-4680`
- Modify: `src/stats/mixed_ld.rs:1-350`
- Test: `test/test_postgwas_fvlmm_effective_ld.py`

**Interfaces:**
- Rust result mapping adds: `backend`, `requested_threads`, `using_threads`, `rotation_seconds`, `svd_seconds`, `residual_seconds`, `gram_seconds`, and `total_seconds`.
- Python emits one INFO record per stage with prefix `FvLMM fine-map stage:` and fields `stage=`, `seconds=`, `requested_threads=`, `using_threads=`, plus `backend=` where relevant.
- Timing fields are diagnostic metadata only; they do not enter PIP/CS output files or hashes.

- [ ] **Step 1: Write failing tests for diagnostic shape and logging**

  Extend the direct kernel test to require finite nonnegative timing values, `requested_threads == 2`, `using_threads == 2`, and `backend` in `{"accelerate", "openblas", "blas", "rust"}`. In a mocked FvLMM adapter test, use `caplog` to assert ordered stage names: `context`, `bim_scan`, `genotype_decode`, `grm_evd`, `effective_ld`.

- [ ] **Step 2: Run the diagnostic tests and verify RED**

  ```bash
  /Users/jingxianfu/mamba/envs/jxfu/bin/python -m pytest -q \
    test/test_postgwas_fvlmm_effective_ld.py -k 'diagnostic or stage_timing'
  ```

  Expected: FAIL because timing/backend keys and stage records are absent.

- [ ] **Step 3: Instrument the Rust kernel without changing arithmetic**

  Add `std::time::Instant`, timing fields to `EffectiveLdResult`, and `rust_sgemm_backend_tag`. Measure current rotation/whitening, SVD, residual/filter, and Gram blocks. Record `threads.max(1)` as both requested and using until Task 4 introduces a local Rayon pool. Include all keys in the PyDict.

- [ ] **Step 4: Instrument the Python route with `time.perf_counter()`**

  Add a small local logger helper rather than repeating format strings. Time context reconstruction, the resolver, genotype decode/alignment, EVD, and the Rust call. At the outer fine-map route, time effective clumping and SuSiE fitting. For EVD, log `_eigh_backend`; for Rust, consume and validate all diagnostic values before logging them.

  ```python
  def _postgwas_log_fvlmm_stage(logger, stage, started, *, requested_threads, using_threads, backend=None):
      elapsed = max(0.0, time.perf_counter() - started)
      logger.info(
          "FvLMM fine-map stage: stage=%s seconds=%.6f requested_threads=%d using_threads=%d%s",
          stage, elapsed, requested_threads, using_threads,
          "" if backend is None else f" backend={backend}",
      )
  ```

- [ ] **Step 5: Run diagnostics and regression tests**

  ```bash
  /Users/jingxianfu/mamba/envs/jxfu/bin/python -m pytest -q \
    test/test_postgwas_fvlmm_effective_ld.py -k \
    'diagnostic or stage_timing or effective_ld_adapter or calls_effective'
  ```

  Expected: PASS; timing fields are finite and logs expose the actual backend and thread count.

- [ ] **Step 6: Commit diagnostics separately**

  ```bash
  git add python/janusx/script/postgwas.py src/stats/mixed_ld.rs \
    test/test_postgwas_fvlmm_effective_ld.py
  git commit -m "perf(postgwas): report FvLMM fine-map stages"
  ```

### Task 3: Route rotation and fixed-effect projection through controlled f64 BLAS

**Files:**
- Modify: `src/stats/mixed_ld.rs:1-205,279-350`
- Test: `src/stats/mixed_ld.rs` test module
- Test: `test/test_postgwas_fvlmm_effective_ld.py:50-146`

**Interfaces:**
- Consumes: `crate::blas::{cblas_dgemm_dispatch, BlasThreadGuard, CblasInt, CBLAS_NO_TRANS, CBLAS_ROW_MAJOR, CBLAS_TRANS}`.
- Produces internal checked helpers `checked_dgemm(...) -> Result<(), String>` and `fvlmm_effective_ld_spectral_core(..., threads: usize) -> Result<EffectiveLdResult, String>`.
- `Gw = G U`, `Cw = U^T C`, and `projected = Gw Uc` each execute inside `BlasThreadGuard::enter(threads.max(1))`.

- [ ] **Step 1: Preserve the scalar implementation as a test-only oracle**

  Move the current arithmetic into `#[cfg(test)] fn fvlmm_effective_ld_scalar_reference(...)` before replacing it. Add deterministic matrices with nontrivial rotation, two fixed effects, and a near-rank-deficient fixed design. Compare valid indices, rank, and effective LD at `1e-10` absolute/relative tolerance.

- [ ] **Step 2: Add RED tests for thread-count parity**

  Call the production core with `threads=1,2,4,8`; assert identical valid indices and allclose matrices against the scalar oracle. Run:

  ```bash
  PYO3_PYTHON=/Users/jingxianfu/mamba/envs/jxfu/bin/python \
    cargo test mixed_ld::tests --lib -- --nocapture
  ```

  Expected: compile failure because the core does not yet accept `threads` and the checked BLAS helper is absent.

- [ ] **Step 3: Implement dimension-checked row-major dGEMM wrappers**

  Convert all dimensions and leading dimensions with `try_into()`, validate slice lengths with `checked_mul`, and return descriptive errors before entering unsafe code. Use row-major operands directly; do not transpose by allocating a second dense matrix.

  ```rust
  let _blas_guard = BlasThreadGuard::enter(threads.max(1));
  unsafe {
      cblas_dgemm_dispatch(
          CBLAS_ROW_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
          m_blas, n_blas, k_blas, 1.0,
          left.as_ptr(), k_blas, right.as_ptr(), k_blas,
          0.0, out.as_mut_ptr(), n_blas,
      );
  }
  ```

- [ ] **Step 4: Replace the three dense products and preserve SVD semantics**

  Compute `Gw` from row-major `G × U` using `u_t` with `TRANS`; compute `Cw` as `U^T × C`; apply whitening in deterministic loops; retain `SVD::try_new`; compute `projected = Gw × Uc` with dGEMM. Pass `threads` from the PyO3 entry point into the core and remove the single outer guard that currently does not control nalgebra multiplication.

- [ ] **Step 5: Run Rust and Python numerical gates**

  ```bash
  PYO3_PYTHON=/Users/jingxianfu/mamba/envs/jxfu/bin/python \
    cargo test mixed_ld::tests --lib -- --nocapture
  /Users/jingxianfu/mamba/envs/jxfu/bin/python -m maturin develop \
    --release --locked --features python-extension
  /Users/jingxianfu/mamba/envs/jxfu/bin/python -m pytest -q \
    test/test_postgwas_fvlmm_effective_ld.py -k 'rust_effective_ld'
  ```

  Expected: PASS at the declared `1e-10` LD tolerance for all thread counts.

- [ ] **Step 6: Commit BLAS products**

  ```bash
  git add src/stats/mixed_ld.rs test/test_postgwas_fvlmm_effective_ld.py
  git commit -m "perf(postgwas): use controlled BLAS for FvLMM LD"
  ```

### Task 4: Parallelize row-local residual work and compute the final Gram with BLAS

**Files:**
- Modify: `src/stats/mixed_ld.rs:1-350`
- Test: `src/stats/mixed_ld.rs` test module
- Test: `test/test_postgwas_fvlmm_effective_ld.py:50-190,2497-2645`

**Interfaces:**
- Uses `crate::stats::common::get_cached_pool(threads)` only if its PyResult boundary is suitable; otherwise constructs one local `rayon::ThreadPoolBuilder` per public kernel call.
- Rayon produces one indexed row result `(original_index, diagonal, normalized_row_or_none)`; collection order is original SNP order.
- Final `R = residuals residuals^T` uses the checked dGEMM helper and requested BLAS threads, followed by explicit symmetry averaging and exact unit diagonal.

- [ ] **Step 1: Write failing tests for filtering order and deterministic Gram output**

  Add a fixture containing constant, nearly degenerate, finite valid, and nonfinite-rejected rows. Assert that `threads=1,2,4,8` return the same sorted `valid_indices`, finite symmetric `R`, and exactly-one diagonal. Compare every result with the scalar oracle at `1e-10`.

- [ ] **Step 2: Run tests before implementation and capture the current diagnostic behavior**

  ```bash
  PYO3_PYTHON=/Users/jingxianfu/mamba/envs/jxfu/bin/python \
    cargo test mixed_ld::tests --lib -- --nocapture
  ```

  Expected: the new test fails because the residual stage still reports one serial execution path and Gram is still hand-written.

- [ ] **Step 3: Implement stable indexed Rayon residualization**

  Hold BLAS to one thread while the Rayon pool is active. Use `(0..snp_count).into_par_iter()` and collect a `Vec<RowResidual>`; do not mutate shared rows or accumulate a shared maximum. First collect each finite diagonal, then compute the global maximum serially, derive the tolerance, retain rows in source order, and normalize retained rows in a second indexed parallel pass.

  ```rust
  let rows = pool.install(|| {
      (0..snp_count)
          .into_par_iter()
          .map(|snp| residualize_one_row(snp, &gw, &projected, &u_covariates))
          .collect::<Vec<_>>()
  });
  let valid_indices = rows.iter().enumerate()
      .filter_map(|(index, row)| row.is_valid(diagonal_tolerance).then_some(index))
      .collect::<Vec<_>>();
  ```

- [ ] **Step 4: Pack normalized rows and compute the symmetric Gram via dGEMM**

  Allocate exactly `valid_count × sample_count` normalized residual values, run `R = X × X^T`, reject nonfinite entries, replace each off-diagonal pair by `0.5 * (Rij + Rji)`, and set every diagonal to exactly `1.0`. Do not run Rayon during this multithreaded BLAS call.

- [ ] **Step 5: Return accurate substage diagnostics**

  Set `using_threads` to the bounded pool/BLAS budget and include residual and Gram elapsed seconds. Preserve all existing rank, tolerance, and minimum-diagonal diagnostics.

- [ ] **Step 6: Run kernel, end-to-end, and susieR parity tests**

  ```bash
  PYO3_PYTHON=/Users/jingxianfu/mamba/envs/jxfu/bin/python \
    cargo test mixed_ld::tests --lib -- --nocapture
  /Users/jingxianfu/mamba/envs/jxfu/bin/python -m maturin develop \
    --release --locked --features python-extension
  /Users/jingxianfu/mamba/envs/jxfu/bin/python -m pytest -q \
    test/test_postgwas_fvlmm_effective_ld.py -k \
    'rust_effective_ld or synthetic_fvlmm_effective_ld'
  ```

  Expected: PASS; direct LD remains within `1e-10`, and PIP/posterior means remain within `atol=1e-6, rtol=1e-5`.

- [ ] **Step 7: Commit row parallelism and Gram optimization**

  ```bash
  git add src/stats/mixed_ld.rs test/test_postgwas_fvlmm_effective_ld.py
  git commit -m "perf(postgwas): parallelize FvLMM LD residuals"
  ```

### Task 5: Validate the Rice6048 command and broader regressions

**Files:**
- Modify: none unless verification exposes an in-scope regression
- Temporary only: `/tmp/jx-postgwas-fvlmm-threading-*`

**Interfaces:**
- Input prefix: `/Users/jingxianfu/script/JanusX/test.atlas/Rice6048`.
- Summary input: `/Users/jingxianfu/script/JanusX/test.20260805/test.blup/Rice6048.Plant_height.fvlmm.tsv` after its sidecar fingerprint is verified; regenerate a fresh `/tmp` GWAS result if it does not match.
- Benchmark command uses locus `3:18.414420:18.461203`, `-finemap susie`, `-finemap-max-iter 500`, `-finemap-L 5`, `-mem 8`, and `-t 1,2,4,8`.

- [ ] **Step 1: Run formatting, static, Rust, and focused Python gates**

  ```bash
  cargo fmt --all -- --check
  git diff --check
  PYO3_PYTHON=/Users/jingxianfu/mamba/envs/jxfu/bin/python \
    cargo test mixed_ld --lib -- --nocapture
  /Users/jingxianfu/mamba/envs/jxfu/bin/python -m py_compile \
    python/janusx/script/postgwas.py
  /Users/jingxianfu/mamba/envs/jxfu/bin/python -m pytest -q \
    test/test_postgwas_fvlmm_effective_ld.py
  ```

  Expected: all focused gates PASS.

- [ ] **Step 2: Build one release extension for the benchmark**

  ```bash
  /Users/jingxianfu/mamba/envs/jxfu/bin/python -m maturin develop \
    --release --locked --features python-extension
  ```

  Expected: the extension installs into `jxfu`; Python-facing commands run without `DYLD_LIBRARY_PATH`.

- [ ] **Step 3: Verify the existing FvLMM sidecar or generate a fresh `/tmp` input**

  Parse the sidecar through JanusX's own sidecar reader and verify genotype/GRM/sample/result fingerprints. If any fingerprint is stale, run the matching FvLMM GWAS once with output under `/tmp/jx-postgwas-fvlmm-threading-input`; do not alter the user's existing files.

- [ ] **Step 4: Run the four thread counts sequentially with fresh prefixes**

  For each `t` in `1 2 4 8`, run `/usr/bin/time -l` around:

  ```bash
  JX_RUST_LAPACK_BACKEND=accelerate JX_RUST_EIGH_LAPACK_BACKEND=accelerate \
    /Users/jingxianfu/mamba/envs/jxfu/bin/jx postgwas \
    -i <verified-fvlmm-tsv> \
    -o /tmp/jx-postgwas-fvlmm-threading-t${t}/Rice6048.Plant_height \
    -bfile /Users/jingxianfu/script/JanusX/test.atlas/Rice6048 \
    -bimrange 3:18.414420:18.461203 -finemap susie \
    -finemap-max-iter 500 -finemap-L 5 -mem 8 -t ${t}
  ```

  Keep runs sequential to avoid CPU and memory contention. Record total wall time, CPU utilization, maximum RSS, BIM rows scanned/candidates retained, every stage time, EVD/BLAS backend, requested/using threads, SuSiE iterations, convergence, and maximum absolute posterior mean.

- [ ] **Step 5: Compare all numerical outputs**

  Load every `.susie.pip.tsv` and credible-set table by SNP identity. Require identical SNP/CS membership and compare PIP/posterior means with `atol=1e-6, rtol=1e-5`. Record SHA-256 hashes when files are byte-identical; do not fail solely on harmless formatting differences.

- [ ] **Step 6: Run relevant broader regression tests**

  ```bash
  /Users/jingxianfu/mamba/envs/jxfu/bin/python -m pytest -q \
    test/test_gwas_null_model_sidecar.py \
    test/test_postgwas_fvlmm_effective_ld.py \
    test/test_fvlmm_scan_optimization.py \
    test/test_reml_interface.py \
    test/test_reml_integration.py
  ```

  Expected: no new failures. If `test_varying_fixed_covariate_reaches_blue_stage` fails exactly as on base `e7f6300`, record it as the known pre-existing REML failure rather than editing REML code.

- [ ] **Step 7: Review repository safety and summarize evidence**

  ```bash
  git status --short
  git log --oneline --decorate -6
  find /tmp -maxdepth 1 -name 'jx-postgwas-fvlmm-threading-*' -print
  ```

  Confirm that only planned source/tests/docs are tracked, all Rice6048 outputs remain under `/tmp`, the main checkout's two untracked user documents are untouched, and no remote push occurred. Report per-stage scaling honestly; the 169-variant locus is not required to show a fixed speedup if overhead dominates.

---

## Final Acceptance Checklist

- [ ] A 5.69-million-row BIM is scanned once, with retained Python metadata bounded by regional candidate count.
- [ ] `-t 1,2,4,8` is visible in EVD and effective-LD diagnostics and controls the dense backend.
- [ ] No Rayon region overlaps a multithreaded BLAS region.
- [ ] Effective LD matches the scalar equations at `1e-10`; PIP/posterior means match at `1e-6/1e-5`.
- [ ] Rice6048 converges with the same credible sets and approximately the established maximum posterior magnitude (`~21.82`, subject to the declared numerical tolerance).
- [ ] Peak RSS stays below the preflight estimate and `-mem 8`.
- [ ] Focused tests pass and broader tests introduce no new failure.
- [ ] Generated outputs are not committed, and nothing is pushed remotely.
