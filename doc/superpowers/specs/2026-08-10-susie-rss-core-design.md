# Standardized SuSiE-RSS Core Design

**Date:** 2026-08-10  
**Status:** Approved for implementation planning  
**Scope:** A stable, dense-LD, standardized z-score SuSiE-RSS core in Rust with a thin PyO3 interface

## 1. Objective

Implement the following standardized summary-statistics model in JanusX:

\[
Z = R\sum_{l=1}^{L}\lambda_l + \varepsilon,
\qquad \varepsilon\sim\mathcal N(0,R),
\]

where each \(\lambda_l\) is a single-effect vector. The first release accepts only a z-score vector and a dense LD correlation matrix. The residual variance is fixed at one.

The implementation must:

- reproduce the fixed point of the equivalent `susieR::susie_rss` standardized z-score path;
- estimate one non-negative prior variance \(V_l\) per effect;
- expose posterior inclusion probabilities, posterior effect summaries, PIPs, and convergence diagnostics;
- avoid avoidable \(O(Lp^2)\) work inside each individual effect update;
- support controlled BLAS threading where the selected backend exposes thread control;
- provide reproducible consistency, runtime, and peak-memory comparisons with susieR.

## 2. Explicit non-goals

The first implementation does not include:

- sample size `n`, `bhat`, `shat`, or phenotype variance conversion;
- residual variance estimation;
- a null component or `null_weight`;
- z-score PVE adjustment;
- low-rank or blockwise LD input;
- finite-reference or LD-mismatch correction;
- credible-set purity filtering or refinement;
- a new `jx` command-line workflow.

These features may be layered on the stable core later. They must not complicate or silently alter the model defined above.

## 3. Architecture

### 3.1 Rust core

Add `src/stats/susie.rs` containing a Python-independent fitting core:

```rust
pub(crate) struct SusieRssConfig {
    pub l: usize,
    pub max_iter: usize,
    pub tol: f64,
    pub prior_tol: f64,
    pub threads: usize,
}

pub(crate) struct SusieRssFit {
    // Dense row-major result buffers and diagnostics.
}

pub(crate) fn fit_susie_rss_f64(
    z: &[f64],
    r: &[f64],
    p: usize,
    prior_weights: Option<&[f64]>,
    config: &SusieRssConfig,
) -> Result<SusieRssFit, String>;
```

The statistical core must not depend on Python objects. This allows direct Rust unit tests and prevents Python conversion details from determining the numerical algorithm.

### 3.2 PyO3 interface

Register a thin function from `src/lib.rs`:

```python
janusx.susie_rss_f64(
    z,
    R,
    L=10,
    prior_weights=None,
    max_iter=100,
    tol=1e-4,
    prior_tol=1e-9,
    threads=1,
)
```

The wrapper validates NumPy dimensions and dtypes, calls the Rust core, and returns a named Python dictionary containing NumPy arrays and scalar diagnostics. A named result is preferred to a long positional tuple because the result contains several matrices with the same shape.

The wrapper should avoid copying the \(p\times p\) LD matrix when it is already C-contiguous. Non-contiguous input may be rejected with a clear error in the first version instead of creating a hidden second dense LD matrix.

## 4. Statistical update

For effect \(l\), define its residual z-score as

\[
\widetilde Z_l
= Z-R\sum_{k\ne l}E(\lambda_k).
\]

For a candidate prior variance \(V\geq 0\), the single-variable log Bayes factor is

\[
\log BF_{li}(V)
=-\frac12\log(1+V)
+\frac12\frac{V}{1+V}\widetilde Z_{li}^2.
\]

With normalized prior weights \(\pi_i\), the single-effect marginal log likelihood is

\[
f_l(V)=\log\sum_i \pi_i BF_{li}(V).
\]

After selecting \(\widehat V_l\), update

\[
\alpha_{li}
=\operatorname{softmax}_i
\left(\log\pi_i+\log BF_{li}(\widehat V_l)\right),
\]

\[
\mu_{li}
=\frac{\widehat V_l}{1+\widehat V_l}\widetilde Z_{li},
\]

\[
\mu^{(2)}_{li}
=\frac{\widehat V_l}{1+\widehat V_l}+\mu_{li}^2,
\qquad
b_{li}=E(\lambda_{li})=\alpha_{li}\mu_{li}.
\]

All posterior normalization is performed with log-sum-exp.

## 5. Prior-variance optimization

The derivative of the single-effect objective is

\[
f_l'(V)=
\frac{E_{\alpha(V)}[\widetilde Z_{li}^2]-(1+V)}
{2(1+V)^2}.
\]

Therefore, because the posterior-weighted mean cannot exceed the largest squared residual z-score, a global maximizer cannot exceed

\[
V_{\max}=\max\left(0,\max_i\widetilde Z_{li}^2-1\right).
\]

The optimizer follows these rules:

1. If `V_max <= v_boundary_tol`, return the exact boundary solution `V = 0`; never call Brent on a collapsed interval.
2. Otherwise minimize `-f_l(V)` with the existing JanusX bounded Brent implementation on `[0, V_max]`.
3. Explicitly evaluate and compare the lower endpoint, the Brent candidate, and the upper endpoint.
4. Use the previous iteration's \(V_l\), clipped into the interval, as Brent's optional initial point.
5. Reject non-finite residual z-scores before optimization. Detect overflow while forming squared z-scores and report a numerical input error rather than passing an infinite bound to Brent.

No arbitrary positive lower bound or inflated upper bound is used. A positive lower bound would turn a noise-only effect into a non-zero effect, and the derivative above proves that LD sharing cannot move the optimum past `V_max` within this model.

## 6. Incremental fitted-value cache

Maintain, for every effect,

\[
b_l=\alpha_l\odot\mu_l,
\qquad rb_l=Rb_l,
\]

and one global fitted z-score vector

\[
Z_{\mathrm{expected}}=\sum_l rb_l.
\]

An effect update is:

```text
z_tilde    = z - z_expected + rb_old[l]
b_new      = alpha_new * mu_new
rb_new     = R @ b_new
z_expected = z_expected + rb_new - rb_old[l]
```

Thus each effect update performs exactly one dense matrix-vector multiplication. It does not recompute \(\sum_k b_k\) or multiply the full sum by \(R\).

At the end of every outer sweep, rebuild `z_expected` by summing the cached `rb[l]` vectors. This \(O(Lp)\) resynchronization prevents incremental floating-point drift without adding another \(O(p^2)\) matrix-vector operation.

The dense matrix-vector operation uses the common JanusX BLAS dispatch and `BlasThreadGuard`. OpenBLAS builds are expected to honor `threads`. Apple Accelerate does not expose the same strict runtime thread setter, so macOS thread control is best-effort and this limitation must be reported in benchmark metadata.

## 7. ELBO and convergence

Let

\[
\bar b=\sum_l b_l,
\qquad \overline{Rb}=R\bar b=Z_{\mathrm{expected}}.
\]

Ignoring constants independent of the variational parameters, the expected log likelihood is

\[
E_q[\log p(Z\mid b)]
=Z^T\bar b
-\frac12\left[
\bar b^TR\bar b
+\sum_l\left(
\sum_i\alpha_{li}\mu^{(2)}_{li}-b_l^TRb_l
\right)
\right].
\]

For \(V_l>0\), the KL contribution is

\[
KL_l
=\sum_i\alpha_{li}\left[
\log\frac{\alpha_{li}}{\pi_i}
+\frac12\left(
\frac{s_l^2+\mu_{li}^2}{V_l}-1+\log\frac{V_l}{s_l^2}
\right)
\right],
\]

where \(s_l^2=V_l/(1+V_l)\). For the exact boundary solution \(V_l=0\), define `alpha = prior_weights`, `mu = 0`, `mu2 = 0`, and `KL_l = 0` by continuity.

The tracked objective is

\[
ELBO=E_q[\log p(Z\mid b)]-\sum_l KL_l.
\]

Convergence is checked after complete outer sweeps. The fit converges when the absolute ELBO change is below `tol`, allowing only a scale-aware numerical monotonicity tolerance. A decrease larger than that tolerance is retained as a diagnostic and treated as a numerical failure during tests rather than silently declared converged.

Also record `max_alpha_delta` to diagnose apparent ELBO convergence with unstable effect allocations.

## 8. Stable PIP calculation

Effects with \(V_l\leq\text{prior_tol}\) are excluded from PIP calculation. For active effects,

\[
\log(1-PIP_i)=\sum_l\log(1-\alpha_{li}).
\]

The Rust implementation must use

```rust
let log_not_pip = active_effects
    .map(|l| (-alpha[l * p + i]).ln_1p())
    .sum::<f64>();
let pip = -log_not_pip.exp_m1();
```

This avoids cancellation both in `log(1 - alpha)` and in the final `1 - exp(log_not_pip)`. `alpha == 1` naturally produces `PIP == 1`.

## 9. Result contract

The Python result contains:

| Field | Shape/type | Meaning |
|---|---:|---|
| `alpha` | `L x p` | Posterior inclusion probabilities |
| `mu` | `L x p` | Conditional posterior means |
| `mu2` | `L x p` | Conditional posterior second moments |
| `prior_variance` | `L` | Estimated \(V_l\) |
| `lbf` | `L` | Single-effect log marginal Bayes factors |
| `lbf_variable` | `L x p` | Per-variable log Bayes factors |
| `pip` | `p` | Stable posterior inclusion probabilities across active effects |
| `posterior_mean` | `p` | \(\sum_l\alpha_l\odot\mu_l\) |
| `elbo` | `n_iter + 1` | Initial and per-sweep ELBO values |
| `max_alpha_delta` | `n_iter` | Largest allocation change per sweep |
| `converged` | `bool` | Whether the convergence criterion was met |
| `n_iter` | `int` | Number of completed outer sweeps |

`L` is clamped to `p`, matching the natural maximum number of single effects.

## 10. Input validation

The core validates:

- `p > 0`, `L > 0`, `max_iter > 0`, `tol >= 0`, and `prior_tol >= 0`;
- `z.len() == p` and `R.len() == p * p` without integer overflow;
- all z-scores and LD entries are finite;
- `R` is symmetric within a documented mixed absolute/relative tolerance;
- every diagonal entry of `R` is close to one;
- prior weights have length `p`, are finite and non-negative, and have a positive finite sum.

The implementation accepts positive-semidefinite and singular LD matrices because no inverse or factorization is needed. It does not run an \(O(p^3)\) positive-semidefinite check. Material ELBO decreases provide an additional runtime diagnostic for invalid or inconsistent input.

## 11. Testing strategy

Implementation follows test-driven development.

### 11.1 Rust unit tests

Tests in `src/stats/susie.rs` cover:

- one-variable analytic posterior and optimized \(V\);
- `R = I` against a small scalar reference implementation;
- exact `V = 0` behavior when `max(z^2) <= 1`;
- a maximizer at or near the upper endpoint;
- non-uniform prior weights;
- perfectly correlated and near-singular LD;
- stable PIP when alpha is zero, near one, or exactly one;
- incremental `z_expected` updates against direct recomputation;
- ELBO calculation against a direct small-matrix reference;
- ELBO monotonicity across accepted coordinate updates;
- convergence and maximum-iteration reporting;
- input-validation failures;
- numerical equality across supported thread counts.

### 11.2 Python boundary tests

Tests verify:

- returned names, shapes, dtypes, and C-order interpretation;
- rejection of non-contiguous or incorrectly shaped `R`;
- optional and non-uniform prior weights;
- parity between the Python wrapper and direct Rust fixtures.

### 11.3 susieR consistency

Create an isolated R environment because `Rscript` is not currently available in the active local environment. Run susieR with the equivalent restricted settings:

```r
susie_rss(
  z = z,
  R = R,
  n = NULL,
  L = L,
  estimate_prior_variance = TRUE,
  estimate_prior_method = "optim",
  estimate_residual_variance = FALSE,
  null_weight = 0,
  tol = 1e-4,
  max_iter = 100
)
```

The comparison reports:

- maximum and RMS PIP difference;
- maximum and RMS posterior-mean difference;
- prior-variance difference;
- ELBO trajectory/final-objective difference after aligning objective constants;
- alpha and mu differences after matching exchangeable effect labels.

Raw alpha rows are not compared without effect matching because the SuSiE factorization is invariant to permutations of the effect labels.

### 11.4 Runtime and memory benchmark

Benchmark dense LD inputs at representative sizes such as `p = 1,000`, `2,000`, and `5,000`, with `L = 10`, identical stopping settings, and separate clean processes.

Measure:

- wall time;
- CPU time and effective CPU utilization;
- iteration count;
- peak resident memory using `/usr/bin/time -l` on macOS or `/usr/bin/time -v` on Linux;
- BLAS backend and requested/effective thread settings.

The JanusX and susieR runs must use the same input files and comparable BLAS thread settings. Benchmark scripts and generated results belong under the ignored `.janusx_test/` tree and must not be committed or pushed.

## 12. Acceptance criteria

The implementation is ready when:

1. all analytic and numerical Rust tests pass;
2. the Python wrapper passes shape, validation, and parity tests;
3. optimized effects do not call Brent with a collapsed interval;
4. PIPs remain accurate for alpha values near or equal to one;
5. cached and direct fitted z-scores agree within floating-point tolerance;
6. ELBO is monotone up to the documented numerical tolerance on valid fixtures;
7. susieR comparisons meet tolerances defined from observed deterministic fixtures, with effect-label matching;
8. runtime and peak-memory results are collected in separate processes with backend/thread metadata;
9. no benchmark data, temporary environment, or generated result is added to Git.
