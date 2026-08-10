# Standardized SuSiE-RSS Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a stable dense-LD standardized z-score SuSiE-RSS solver in Rust, expose it through PyO3, and compare consistency, runtime, and peak memory with the equivalent susieR fit.

**Architecture:** `src/stats/susie.rs` owns the Python-independent SER/IBSS solver, prior-variance optimization, incremental fitted-value cache, ELBO, and PIP. `src/lib.rs` only declares and registers a thin NumPy/PyDict wrapper. Local scripts and generated results remain under ignored `.janusx_test/`.

**Tech Stack:** Rust 2021, PyO3 0.27, numpy 0.27, JanusX Brent and BLAS dispatch, Python 3.13 in micromamba `jxfu`, R/susieR.

## Global Constraints

- Implement exactly `Z = R sum_l(lambda_l) + epsilon`, with `epsilon ~ N(0,R)` and residual variance fixed at 1.
- Do not add `n`, residual-variance estimation, null weights, low-rank LD, credible sets, refinement, or a `jx` CLI.
- Use f64 and reject non-finite inputs; singular positive-semidefinite LD remains valid.
- Optimize `V_l` on `[0, max(0, max(z_tilde^2)-1)]`; never call Brent on a collapsed interval.
- Maintain `z_expected = sum_l R b_l` incrementally and resynchronize it from cached `R b_l` after every sweep.
- Compute PIP using `ln_1p` and `exp_m1`; exclude effects with `V_l <= prior_tol`.
- Generated fixtures, R environments, scripts, and benchmark output stay under `.janusx_test/` and are never staged.
- Preserve unrelated changes in `src/stats/grm.rs`, `src/stats/splmm.rs`, and existing untracked documents.
- Every behavior follows RED-GREEN-REFACTOR and ends in a focused commit.

---

### Task 1: Single-effect regression and variance boundary

**Files:**
- Create: `src/stats/susie.rs`
- Modify: `src/lib.rs` (module declaration only)
- Test: inline `susie::tests`

**Interfaces:**
- Produces `SingleEffectResult { alpha, mu, mu2, logbf, v, lbf }`.
- Produces `single_effect_regression(z_tilde, prior, previous_v) -> Result<SingleEffectResult, String>`.
- Produces normalized finite prior weights and a stable weighted log-sum-exp helper.

- [ ] **Step 1: Write failing analytic tests**

Add the module declaration and tests for these exact cases:

```rust
#[test]
fn ser_one_variant_has_analytic_solution() {
    let fit = single_effect_regression_for_test(&[3.0], &[1.0], 0.2).unwrap();
    assert!((fit.v - 8.0).abs() < 1e-8);
    assert_eq!(fit.alpha, vec![1.0]);
    assert!((fit.mu[0] - 8.0 / 3.0).abs() < 1e-8);
    assert!((fit.mu2[0] - 8.0).abs() < 1e-8);
}

#[test]
fn ser_noise_returns_exact_zero_effect() {
    let fit = single_effect_regression_for_test(&[0.2, -0.9], &[0.5, 0.5], 0.2).unwrap();
    assert_eq!(fit.v, 0.0);
    assert_eq!(fit.alpha, vec![0.5, 0.5]);
    assert_eq!(fit.mu, vec![0.0, 0.0]);
    assert_eq!(fit.mu2, vec![0.0, 0.0]);
}

#[test]
fn ser_equal_z_preserves_nonuniform_prior() {
    let fit = single_effect_regression_for_test(&[2.0, 2.0], &[0.8, 0.2], 0.2).unwrap();
    assert_vec_close(&fit.alpha, &[0.8, 0.2], 1e-12);
}
```

- [ ] **Step 2: Run RED**

Run `DYLD_LIBRARY_PATH=/Users/jingxianfu/mamba/envs/jxfu/lib PYO3_PYTHON=/Users/jingxianfu/mamba/envs/jxfu/bin/python cargo test --lib susie::tests::ser_ -- --nocapture`.

Expected: compilation fails because the SER API does not exist.

- [ ] **Step 3: Implement the minimum SER path**

Use:

```rust
#[inline]
fn logbf_at(z2: f64, v: f64) -> f64 {
    -0.5 * v.ln_1p() + 0.5 * (v / (1.0 + v)) * z2
}
```

Compute `max_abs_z` before squaring and reject `max_abs_z > f64::MAX.sqrt()`. If `v_upper <= 16.0 * f64::EPSILON`, return the exact zero-effect posterior without Brent. Otherwise call `brent_minimize_with_init` on the closed interval and compare the lower endpoint, Brent candidate, and upper endpoint. Normalize alpha by log-sum-exp and calculate `mu`, `mu2` from the approved formulas.

- [ ] **Step 4: Verify GREEN, then add validation RED/GREEN cycles**

Run the focused tests. Add one test at a time for empty z, non-finite z, negative/non-finite prior weights, zero prior sum, and overflow. Watch each fail before adding minimal validation.

- [ ] **Step 5: Commit**

Run `git add src/stats/susie.rs src/lib.rs && git commit -m "feat(susie): add stable single-effect regression"`.

---

### Task 2: Dense IBSS with incremental fitted-value cache

**Files:**
- Modify: `src/stats/susie.rs`
- Test: inline `susie::tests`

**Interfaces:**
- Produces `SusieRssConfig { l, max_iter, tol, prior_tol, threads }`.
- Produces `SusieRssFit` with row-major effect matrices and diagnostics.
- Produces `fit_susie_rss_f64(z, r, p, prior_weights, config) -> Result<SusieRssFit, String>`.

- [ ] **Step 1: Write failing cache and identity-LD tests**

```rust
#[test]
fn incremental_z_expected_matches_sum_of_cached_rb() {
    let r = vec![1.0, 0.3, 0.3, 1.0];
    let fit = fit_test(&[4.0, -3.0], &r, 2, 2, 1).unwrap();
    let mut direct = vec![0.0; 2];
    for l in 0..fit.l {
        for i in 0..2 { direct[i] += fit.rb[l * 2 + i]; }
    }
    assert_vec_close(&fit.z_expected, &direct, 1e-12);
}

#[test]
fn identity_ld_matches_scalar_ibss_reference() {
    let z = [4.0, 0.5, -3.0];
    let fit = fit_test(&z, &identity(3), 3, 2, 1).unwrap();
    assert_vec_close(&fit.posterior_mean, &direct_ibss_reference(&z, 2), 1e-9);
}
```

- [ ] **Step 2: Run RED**

Run `cargo test --lib susie::tests::incremental_z_expected_matches_sum_of_cached_rb -- --nocapture`, then `cargo test --lib susie::tests::identity_ld_matches_scalar_ibss_reference -- --nocapture`. Expected: missing fit structs/functions.

- [ ] **Step 3: Implement result storage and one-matvec updates**

`SusieRssFit` stores `p`, `l`, `alpha`, `mu`, `mu2`, `b`, `rb`, `lbf`, `lbf_variable`, `prior_variance`, `posterior_mean`, `z_expected`, `pip`, `elbo`, `max_alpha_delta`, `converged`, and `n_iter`.

Initialize `alpha[l]` to prior weights and all moments/caches to zero. Per effect:

```text
z_tilde    = z - z_expected + rb_old[l]
SER        = single_effect_regression(z_tilde, prior, V_old[l])
b_new      = alpha_new * mu_new
rb_new     = R @ b_new
z_expected = z_expected + rb_new - rb_old[l]
```

After every sweep, clear `z_expected` and sum cached `rb` rows. Start with a scalar row-major matvec to make correctness green.

- [ ] **Step 4: Verify GREEN and singular LD**

Run all SuSiE tests. Add a perfectly correlated `[[1,1],[1,1]]` test and ensure no Cholesky, inverse, or eigendecomposition is introduced.

- [ ] **Step 5: Replace production matvec with BLAS after tests are green**

Call `cblas_dgemm_dispatch` as row-major `(p x p) * (p x 1)` under `BlasThreadGuard::enter(threads.max(1))`. Check all `usize -> CblasInt` conversions. Retain the scalar implementation under tests and compare 1-thread and 2-thread output within `1e-10`.

- [ ] **Step 6: Commit**

Run `git add src/stats/susie.rs && git commit -m "feat(susie): add cached dense IBSS updates"`.

---

### Task 3: ELBO convergence and stable PIP

**Files:**
- Modify: `src/stats/susie.rs`
- Test: inline `susie::tests`

**Interfaces:**
- Produces `compute_elbo(...) -> Result<f64, String>`.
- Produces `compute_pip(alpha, prior_variance, l, p, prior_tol) -> Vec<f64>`.
- Completes convergence diagnostics in `SusieRssFit`.

- [ ] **Step 1: Write failing PIP tests**

```rust
#[test]
fn pip_is_stable_near_and_at_one() {
    let alpha = vec![0.999_999_999_999, 1.0, 1e-12, 0.0];
    let pip = compute_pip(&alpha, &[1.0, 1.0], 2, 2, 1e-9);
    assert!(pip[0] >= 0.999_999_999_999 && pip[0] <= 1.0);
    assert_eq!(pip[1], 1.0);
}

#[test]
fn pip_excludes_zero_variance_effects() {
    let pip = compute_pip(&[0.9, 0.1, 0.2, 0.8], &[1.0, 0.0], 2, 2, 1e-9);
    assert_vec_close(&pip, &[0.9, 0.1], 1e-15);
}
```

- [ ] **Step 2: Run RED, then implement stable PIP**

For each SNP, sum `(-alpha).ln_1p()` over active effects and set `pip = -log_not_pip.exp_m1()`, clamped to `[0,1]`. Run tests to GREEN.

- [ ] **Step 3: Write a failing direct ELBO test**

For `p=2,L=2`, enumerate all effect-indicator combinations to calculate `E[b'Rb]` independently. Include one active effect and one exact `V=0` effect. Compare to `compute_elbo` within `1e-12`.

- [ ] **Step 4: Implement ELBO**

Implement the expected log likelihood and categorical-plus-normal KL from the design. Skip `alpha == 0` terms to avoid `0*log(0)`. Define the exact `V=0` effect's KL as zero.

- [ ] **Step 5: Add convergence RED/GREEN cycles**

Require a strong two-signal fixture to converge before `max_iter`, `elbo.len() == n_iter + 1`, and ELBO differences no smaller than `-1e-9 * max(1, abs(previous))`. A separate `max_iter=1` test must return `converged=false`, `n_iter=1`. Record initial and per-sweep ELBO and `max_alpha_delta`; converge only on a full sweep.

- [ ] **Step 6: Commit**

Run `git add src/stats/susie.rs && git commit -m "feat(susie): add ELBO convergence and stable PIP"`.

---

### Task 4: Validation and PyO3 result contract

**Files:**
- Modify: `src/stats/susie.rs`
- Modify: `src/lib.rs`
- Test locally: `.janusx_test/susie/test_susie_py_api.py`

**Interfaces:**
- Produces `janusx.janusx.susie_rss_f64` with the approved signature.
- Returns all 12 approved named fields in a Python dictionary.

- [ ] **Step 1: Add validation RED/GREEN tests**

Test empty/mismatched shapes, asymmetric LD, diagonal not one, NaN/Inf, zero L, zero max_iter, negative tolerance, and prior length mismatch separately. Implement checked `p*p`, finite checks, mixed symmetry tolerance `1e-10 + 1e-8*scale`, and unit-diagonal checks. Do not add a PSD factorization.

- [ ] **Step 2: Add a failing Python export test**

Create the ignored local script with `assert hasattr(janusx.janusx, "susie_rss_f64")`, build the extension before registration, and verify it fails.

- [ ] **Step 3: Implement and register the wrapper**

Use this exact signature:

```rust
#[pyfunction]
#[pyo3(signature = (z, r, l=10, prior_weights=None, max_iter=100,
                    tol=1e-4, prior_tol=1e-9, threads=1))]
pub fn susie_rss_f64<'py>(
    py: Python<'py>,
    z: PyReadonlyArray1<'py, f64>,
    r: PyReadonlyArray2<'py, f64>,
    l: usize,
    prior_weights: Option<PyReadonlyArray1<'py, f64>>,
    max_iter: usize,
    tol: f64,
    prior_tol: f64,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>>;
```

Require C-contiguous z/R/prior buffers. Return NumPy arrays named `alpha`, `mu`, `mu2`, `prior_variance`, `lbf`, `lbf_variable`, `pip`, `posterior_mean`, `elbo`, `max_alpha_delta`, plus scalar `converged`, `n_iter`.

- [ ] **Step 4: Build and test GREEN**

Run `mamba run -n jxfu maturin develop --release --locked`, then the local Python API script. Assert keys, shapes, f64 dtypes, finite values, rejection of non-contiguous R, and agreement with a small Rust fixture.

- [ ] **Step 5: Run regressions and commit**

Run focused SuSiE tests, `cargo fmt --all -- --check`, and `cargo test --lib --no-fail-fast`. Record unrelated baseline failures without editing their modules. Commit only `src/stats/susie.rs src/lib.rs` as `feat(susie): expose standardized RSS solver`.

---

### Task 5: susieR consistency validation

**Files:**
- Local only: `.janusx_test/susie/create_fixtures.py`
- Local only: `.janusx_test/susie/run_susier.R`
- Local only: `.janusx_test/susie/compare_results.py`
- Modify only after a new RED regression test: `src/stats/susie.rs`

**Interfaces:**
- Produces local NPY/TSV/JSON parity reports; no tracked benchmark files.

- [ ] **Step 1: Build an isolated R/susieR environment**

Install `r-base` and `r-susier` locally. Record R version, susieR version, BLAS backend, and thread variables.

- [ ] **Step 2: Generate deterministic shared fixtures**

Cover identity, AR(1), block, exact-duplicate, and near-singular LD; one strong signal, two separated signals, and all-noise z. Both implementations read the same files and settings.

- [ ] **Step 3: Run equivalent susieR settings**

Use `susie_rss(z=z,R=R,n=NULL,L=L,estimate_prior_variance=TRUE,estimate_prior_method="optim",estimate_residual_variance=FALSE,null_weight=0,tol=1e-4,max_iter=100)` and export all comparable fields.

- [ ] **Step 4: Compare robustly**

Compare PIP and posterior mean directly. Match effect rows by minimum `1-abs(cor(alpha_rust,alpha_r))`, then report max/RMS differences in alpha, mu, and V. Align ELBO by subtracting each initial objective constant.

- [ ] **Step 5: Resolve discrepancies through TDD**

Every material discrepancy first becomes the smallest deterministic failing Rust test. Verify RED, implement the minimal correction, verify GREEN, and rerun every fixture. Never relax tolerance to hide systematic disagreement.

- [ ] **Step 6: Commit only tracked fixes**

If needed, commit only Rust changes as `fix(susie): align standardized RSS updates with susieR`. Keep `.janusx_test/` ignored.

---

### Task 6: Runtime, memory, and final verification

**Files:**
- Local only: `.janusx_test/susie/benchmark_janusx.py`
- Local only: `.janusx_test/susie/benchmark_susier.R`
- Local only: `.janusx_test/susie/results/`

**Interfaces:**
- Produces a local comparison at `p=1000,2000,5000`, `L=10`, at least three repeats each.

- [ ] **Step 1: Generate inputs outside timed regions**

Create deterministic positive-semidefinite dense LD and z files once under `.janusx_test/susie/data/`.

- [ ] **Step 2: Benchmark separate clean processes**

Use `/usr/bin/time -l` on macOS. Record wall/user/system time, maximum RSS, iteration count, and PIP/posterior-mean checksums. Never time both methods in one process.

- [ ] **Step 3: Audit thread fairness**

Record both BLAS backends, requested/effective threads, CPU utilization, and thread environment variables. Report Accelerate as host-managed/best-effort; do not claim strict macOS thread enforcement.

- [ ] **Step 4: Run final verification**

Run `cargo fmt --all -- --check`, focused SuSiE Rust tests, a release `maturin develop`, the Python API test, `git diff --check`, and `git status --short`.

- [ ] **Step 5: Audit scope and report evidence**

Confirm no `.janusx_test` artifact is staged and original GRM/SPLMM changes remain untouched. Report test counts, susieR version, parity metrics, runtime medians/ranges, peak RSS, backend/thread metadata, and remaining limitations.
