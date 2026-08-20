//! Bayesian marker-effect solvers over additive PLINK BED blocks.
//!
//! Let `Z` denote the standardized additive genotype matrix after marker
//! filtering and sample subsetting, with markers in rows and samples in
//! columns. This module fits
//!
//! `y = X alpha + Z' beta + e,  e ~ N(0, sigma_e^2 I)`.
//!
//! The residual inverse-chi-square prior uses `prior_ss_e = nu_0 * S_0^2` as
//! its sum-of-squares parameter. Consequently, each update has the form
//! `(RSS + prior_ss_e) / chi2(n + nu_0)`; `prior_ss_e` is not `S_0^2` alone.
//!
//! The three maintained priors are:
//!
//! - `BayesA`: each marker has its own variance,
//!   `beta_j | sigma_bj^2 ~ N(0, sigma_bj^2)`,
//!   `sigma_bj^2 ~ scaled-inv-chi^2`.
//! - `BayesB`: marker inclusion indicator
//!   `delta_j ~ Bernoulli(pi)`, inactive markers have `beta_j = 0`, and active
//!   markers follow the BayesA variance hierarchy.
//! - `BayesC`: the same spike-and-slab inclusion structure as BayesB, but
//!   active markers share a common marker variance and `pi` is updated from its
//!   Beta-Binomial posterior.
//!
//! All maintained paths stream standardized BED blocks from either resident
//! packed rows or `WindowedBedMatrix`. Small resident packed inputs may be
//! predecoded once, but the streaming path remains the default maintained route
//! for GS. Each MCMC iteration updates fixed effects, residual variance, marker
//! effects, and inclusion/variance hyperparameters. Production kernels monitor
//! split-chain R-hat up to a hard iteration limit, discard pre-convergence
//! samples once the threshold is reached, and return means from the next 1000
//! posterior samples. If the limit is reached without convergence, the next
//! 1000 iterations form the fallback posterior window. Trace-only kernels
//! retain explicit thinning for diagnostic plots.

use numpy::ndarray::Array2;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::types::{PyAny, PyDict};
use pyo3::{prelude::*, BoundObject};
use rand::rngs::{OsRng, StdRng};
use rand::{Rng, SeedableRng, TryRngCore};
use rand_distr::{Beta, ChiSquared, Gamma, StandardNormal};
use std::borrow::Cow;
use std::collections::HashMap;
use std::env;
use std::sync::mpsc::sync_channel;
use std::sync::{Arc, OnceLock};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{vdupq_n_f32, vfmaq_f32, vld1q_f32, vst1q_f32};

#[cfg(all(test, target_arch = "aarch64"))]
use std::arch::aarch64::{
    vcvt_f64_f32, vdupq_n_f64, vfmaq_f64, vget_high_f32, vget_low_f32, vld1q_f64, vst1q_f64,
};

use crate::bedmath::{decode_standardized_packed_block_f32, is_identity_indices};
#[cfg(test)]
use crate::blas::cblas_daxpy_dispatch;
use crate::blas::{
    cblas_ddot_dispatch, cblas_dgemm_dispatch, CblasInt, OpenBlasThreadGuard, CBLAS_COL_MAJOR,
    CBLAS_NO_TRANS, CBLAS_TRANS,
};
use crate::decode::decode_prepared_additive_block_packed_f32;
use crate::gload::WindowedBedMatrix;
use crate::stats_common::{get_cached_pool, parse_index_vec_i64_value_error};

/// Dense marker input held for the duration of a native sampler call.
///
/// A contiguous float32 NumPy array is kept as a `PyReadonlyArray2` so its
/// borrow remains active while the GIL is detached.  The sampler can then
/// read the marker rows directly without first allocating and copying a
/// second `Vec<f32>`.  Non-contiguous and float64 inputs use the owned
/// fallback, preserving the old logical row-major conversion semantics.
pub(crate) enum DenseF32Input<'py> {
    Borrowed {
        array: PyReadonlyArray2<'py, f32>,
        rows: usize,
        cols: usize,
    },
    Owned {
        data: Vec<f32>,
        rows: usize,
        cols: usize,
    },
}

impl DenseF32Input<'_> {
    #[inline]
    pub(crate) fn rows(&self) -> usize {
        match self {
            Self::Borrowed { rows, .. } | Self::Owned { rows, .. } => *rows,
        }
    }

    #[inline]
    pub(crate) fn cols(&self) -> usize {
        match self {
            Self::Borrowed { cols, .. } | Self::Owned { cols, .. } => *cols,
        }
    }

    #[inline]
    pub(crate) fn as_slice(&self) -> &[f32] {
        match self {
            Self::Borrowed { array, .. } => array
                .as_slice()
                .expect("borrowed dense float32 input must be contiguous"),
            Self::Owned { data, .. } => data,
        }
    }

    #[cfg(test)]
    #[inline]
    pub(crate) fn is_borrowed(&self) -> bool {
        matches!(self, Self::Borrowed { .. })
    }
}

pub(crate) fn array2_to_f32_input<'py>(
    obj: &Bound<'py, PyAny>,
    label: &str,
) -> PyResult<DenseF32Input<'py>> {
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'py, f32>>() {
        let view = arr.as_array();
        let (rows, cols) = view.dim();
        if arr.as_slice().is_ok() {
            return Ok(DenseF32Input::Borrowed {
                array: arr,
                rows,
                cols,
            });
        }
        return Ok(DenseF32Input::Owned {
            data: view.iter().copied().collect(),
            rows,
            cols,
        });
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray2<'py, f64>>() {
        let view = arr.as_array();
        let (rows, cols) = view.dim();
        return Ok(DenseF32Input::Owned {
            data: view.iter().map(|value| *value as f32).collect(),
            rows,
            cols,
        });
    }
    Err(PyValueError::new_err(format!(
        "{label} must be a 2D float32 or float64 numpy array"
    )))
}

fn array1_to_vec(arr: &PyReadonlyArray1<f64>) -> Vec<f64> {
    arr.as_array().iter().copied().collect()
}

fn array2_to_vec(arr: &PyReadonlyArray2<f64>) -> Vec<f64> {
    let view = arr.as_array();
    let (n, p) = view.dim();
    let mut out = Vec::with_capacity(n * p);
    for i in 0..n {
        for j in 0..p {
            out.push(view[[i, j]]);
        }
    }
    out
}

#[inline]
pub(crate) fn copy_f64_to_f32(src: &[f64], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    for (out, value) in dst.iter_mut().zip(src.iter()) {
        *out = *value as f32;
    }
}

#[inline]
pub(crate) fn copy_f32_to_f64(src: &[f32], dst: &mut [f64]) {
    debug_assert_eq!(src.len(), dst.len());
    for (out, value) in dst.iter_mut().zip(src.iter()) {
        *out = *value as f64;
    }
}

fn parse_optional_index_vec_i64(
    indices: Option<&PyReadonlyArray1<i64>>,
    upper_bound: usize,
    label: &str,
) -> PyResult<Vec<usize>> {
    match indices {
        Some(arr) => parse_index_vec_i64_value_error(arr.as_slice()?, upper_bound, label),
        None => Ok(Vec::new()),
    }
}

fn parse_index_vec_i64_string(
    raw: &[i64],
    upper_bound: usize,
    label: &str,
) -> Result<Vec<usize>, String> {
    parse_index_vec_i64_value_error(raw, upper_bound, label).map_err(|e| e.to_string())
}

#[derive(Debug)]
struct PackedBayesTraceResult {
    beta: Vec<f64>,
    alpha: Vec<f64>,
    vare: f64,
    h2_mean: f64,
    var_h2: f64,
    prob_in_mean: f64,
    n_active_mean: f64,
    iter_trace: Vec<i64>,
    h2_trace: Vec<f64>,
    var_e_trace: Vec<f64>,
    prob_in_trace: Vec<f64>,
    n_active_trace: Vec<f64>,
    full_iter_trace: Vec<i64>,
    full_h2_trace: Vec<f64>,
    full_var_e_trace: Vec<f64>,
    full_prob_in_trace: Vec<f64>,
    full_n_active_trace: Vec<f64>,
    beta_trace_indices: Vec<i64>,
    beta_trace: Vec<f64>,
}

fn posterior_keep_iters(n_iter: usize, burnin: usize, thin: usize) -> Vec<i64> {
    let mut out = Vec::new();
    for it in 0..n_iter {
        if it >= burnin && ((it - burnin) % thin == 0) {
            out.push((it + 1) as i64);
        }
    }
    out
}

// A single MCMC chain does not support the classical multi-chain Gelman--Rubin
// diagnostic.  We therefore report the standard split-chain R-hat calculated
// from the retained posterior h2 samples.  The two consecutive halves are
// treated as two chains; this is useful for detecting a persistent drift in the
// scalar variance component while keeping the production kernels streaming.
// Keep the early-stop criterion aligned with the GS user-facing contract.
// Keep the conventional conservative convergence cutoff for split-chain R-hat.
const BAYES_RHAT_THRESHOLD: f64 = 1.10;
const BAYES_RHAT_MIN_KEEP: usize = 500;
const BAYES_RHAT_CHECK_EVERY: usize = 50;
const BAYES_RHAT_STABLE_CHECKS: usize = 3;
pub(crate) const BAYES_POSTERIOR_SAMPLES: usize = 1000;

#[derive(Debug, Default)]
struct BayesRhatState {
    // Prefix sums let us evaluate the two split-chain moments in O(1) at
    // each check.  Keeping the prefix moments rather than rescanning all
    // retained samples avoids O(n_iter^2) work for long, non-converging runs.
    h2_sum: Vec<f64>,
    h2_sq_sum: Vec<f64>,
    stable_checks: usize,
}

impl BayesRhatState {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            h2_sum: {
                let mut values = Vec::with_capacity(capacity + 1);
                values.push(0.0);
                values
            },
            h2_sq_sum: {
                let mut values = Vec::with_capacity(capacity + 1);
                values.push(0.0);
                values
            },
            stable_checks: 0,
        }
    }

    #[inline]
    fn reset(&mut self) {
        self.h2_sum.clear();
        self.h2_sum.push(0.0);
        self.h2_sq_sum.clear();
        self.h2_sq_sum.push(0.0);
        self.stable_checks = 0;
    }

    #[inline]
    fn observe(&mut self, h2: f64) -> bool {
        if h2.is_finite() {
            let sum = *self.h2_sum.last().unwrap_or(&0.0);
            let sq_sum = *self.h2_sq_sum.last().unwrap_or(&0.0);
            self.h2_sum.push(sum + h2);
            self.h2_sq_sum.push(sq_sum + h2 * h2);
        }
        let n = self.h2_sum.len().saturating_sub(1);
        if n < BAYES_RHAT_MIN_KEEP || n % BAYES_RHAT_CHECK_EVERY != 0 {
            return false;
        }
        let rhat = split_rhat_from_prefix(&self.h2_sum, &self.h2_sq_sum);
        if rhat.is_finite() && rhat < BAYES_RHAT_THRESHOLD {
            self.stable_checks += 1;
        } else {
            self.stable_checks = 0;
        }
        self.stable_checks >= BAYES_RHAT_STABLE_CHECKS
    }

    #[inline]
    fn value(&self) -> f64 {
        split_rhat_from_prefix(&self.h2_sum, &self.h2_sq_sum)
    }
}

/// Controls R-hat monitoring and posterior collection. `n_iter` is the upper
/// bound for the monitoring phase. Once R-hat stabilizes, the pre-trigger
/// summaries are discarded and exactly `BAYES_POSTERIOR_SAMPLES` subsequent
/// samples are retained. If the monitoring phase reaches `n_iter` without
/// convergence, the same posterior collection phase starts as a fallback.
#[derive(Debug)]
pub(crate) struct BayesSamplingController {
    n_iter: usize,
    thin: usize,
    actual_iterations: usize,
    posterior_samples: usize,
    rhat_state: BayesRhatState,
    convergence_iteration: usize,
    collecting_posterior: bool,
}

impl BayesSamplingController {
    pub(crate) fn new(n_iter: usize, _burnin: usize, thin: usize) -> Self {
        let thin = thin.max(1);
        let target_keep = BAYES_POSTERIOR_SAMPLES / thin + 1;
        Self {
            n_iter,
            thin,
            actual_iterations: 0,
            posterior_samples: 0,
            rhat_state: BayesRhatState::with_capacity(target_keep + 1),
            convergence_iteration: 0,
            collecting_posterior: false,
        }
    }

    #[inline]
    pub(crate) fn should_run(&self) -> bool {
        if self.collecting_posterior {
            self.posterior_samples < BAYES_POSTERIOR_SAMPLES
        } else {
            // A low-level caller may request `thin > 1`. Allow the monitor
            // to reach the first retained iteration at or after `n_iter`, so
            // the fallback transition is still observed instead of exiting
            // with zero posterior samples. Production GS fixes thin=1.
            let remainder = self.n_iter.saturating_sub(1) % self.thin;
            let monitor_end = self.n_iter + (self.thin - remainder) % self.thin;
            self.actual_iterations < monitor_end
        }
    }

    /// Advance one MCMC iteration and report whether its summary is retained.
    #[inline]
    pub(crate) fn begin_iteration(&mut self) -> bool {
        self.actual_iterations += 1;
        let it = self.actual_iterations - 1;
        it % self.thin == 0
    }

    /// Observe a retained h2 sample. Returns true when the monitoring summary
    /// must be cleared before the next iteration starts formal posterior
    /// collection.
    #[inline]
    pub(crate) fn observe(&mut self, h2: f64) -> bool {
        if self.collecting_posterior {
            self.posterior_samples += 1;
            let _ = self.rhat_state.observe(h2);
            return false;
        }

        if self.rhat_state.observe(h2) {
            self.collecting_posterior = true;
            self.convergence_iteration = self.actual_iterations;
            self.posterior_samples = 0;
            self.rhat_state.reset();
            return true;
        }

        // No convergence by the monitoring upper bound: use the following
        // 1000 iterations as the fallback posterior window. There is no
        // convergence iteration to report in this branch.
        if self.actual_iterations >= self.n_iter {
            self.collecting_posterior = true;
            self.convergence_iteration = 0;
            self.posterior_samples = 0;
            self.rhat_state.reset();
            return true;
        }

        false
    }

    #[inline]
    pub(crate) fn posterior_samples(&self) -> usize {
        self.posterior_samples
    }

    #[inline]
    pub(crate) fn rhat_value(&self) -> f64 {
        self.rhat_state.value()
    }

    #[inline]
    pub(crate) fn actual_iterations(&self) -> usize {
        self.actual_iterations
    }

    #[inline]
    pub(crate) fn convergence_iteration(&self) -> usize {
        self.convergence_iteration
    }
}

fn split_rhat_from_prefix(sum: &[f64], sq_sum: &[f64]) -> f64 {
    let n = sum.len().saturating_sub(1);
    if sq_sum.len() != sum.len() {
        return f64::NAN;
    }
    if n < 4 {
        return f64::NAN;
    }
    // Use equal halves; an odd final sample is intentionally ignored.
    let chain_n = n / 2;
    if chain_n < 2 {
        return f64::NAN;
    }
    let left_sum = sum[chain_n];
    let left_sq_sum = sq_sum[chain_n];
    let split_end = chain_n * 2;
    let total_sum = sum[split_end];
    let total_sq_sum = sq_sum[split_end];
    let right_sum = total_sum - left_sum;
    let right_sq_sum = total_sq_sum - left_sq_sum;
    let mean_left = left_sum / chain_n as f64;
    let mean_right = right_sum / chain_n as f64;
    let var_left =
        ((left_sq_sum - left_sum * left_sum / chain_n as f64) / (chain_n - 1) as f64).max(0.0);
    let var_right =
        ((right_sq_sum - right_sum * right_sum / chain_n as f64) / (chain_n - 1) as f64).max(0.0);
    let within = 0.5 * (var_left + var_right);
    let between = chain_n as f64 * (mean_left - mean_right) * (mean_left - mean_right) / 2.0;
    let var_hat = ((chain_n - 1) as f64 / chain_n as f64) * within + between / chain_n as f64;
    if !within.is_finite() || !var_hat.is_finite() {
        return f64::NAN;
    }
    if within <= f64::MIN_POSITIVE {
        return if var_hat <= f64::MIN_POSITIVE {
            1.0
        } else {
            f64::INFINITY
        };
    }
    (var_hat / within).sqrt().max(1.0)
}

fn fill_beta_trace_row(
    beta_trace: &mut [f64],
    row_idx: usize,
    n_trace_snps: usize,
    trace_snp_indices: &[usize],
    beta: &[f64],
    d: Option<&[u8]>,
) {
    if n_trace_snps == 0 {
        return;
    }
    let row_off = row_idx * n_trace_snps;
    for (k, &j) in trace_snp_indices.iter().enumerate() {
        let val = if let Some(mask) = d {
            (mask[j] as f64) * beta[j]
        } else {
            beta[j]
        };
        beta_trace[row_off + k] = val;
    }
}

fn packed_trace_result_to_pydict<'py>(
    py: Python<'py>,
    res: PackedBayesTraceResult,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py).into_bound();
    let beta_py = res.beta.into_pyarray(py);
    let alpha_py = res.alpha.into_pyarray(py);
    let iter_trace_py = res.iter_trace.into_pyarray(py);
    let h2_trace_py = res.h2_trace.into_pyarray(py);
    let var_e_trace_py = res.var_e_trace.into_pyarray(py);
    let prob_in_trace_py = res.prob_in_trace.into_pyarray(py);
    let n_active_trace_py = res.n_active_trace.into_pyarray(py);
    let full_iter_trace_py = res.full_iter_trace.into_pyarray(py);
    let full_h2_trace_py = res.full_h2_trace.into_pyarray(py);
    let full_var_e_trace_py = res.full_var_e_trace.into_pyarray(py);
    let full_prob_in_trace_py = res.full_prob_in_trace.into_pyarray(py);
    let full_n_active_trace_py = res.full_n_active_trace.into_pyarray(py);
    let beta_trace_indices_py = res.beta_trace_indices.into_pyarray(py);
    let beta_trace_arr = Array2::from_shape_vec(
        (iter_trace_py.len(), beta_trace_indices_py.len()),
        res.beta_trace,
    )
    .map_err(|e| PyValueError::new_err(format!("invalid beta trace shape: {e}")))?;
    let beta_trace_py = PyArray2::from_owned_array(py, beta_trace_arr);
    out.set_item("beta", beta_py)?;
    out.set_item("alpha", alpha_py)?;
    out.set_item("vare", res.vare)?;
    out.set_item("h2_mean", res.h2_mean)?;
    out.set_item("var_h2", res.var_h2)?;
    out.set_item("prob_in_mean", res.prob_in_mean)?;
    out.set_item("n_active_mean", res.n_active_mean)?;
    out.set_item("iter_trace", iter_trace_py)?;
    out.set_item("h2_trace", h2_trace_py)?;
    out.set_item("var_e_trace", var_e_trace_py)?;
    out.set_item("prob_in_trace", prob_in_trace_py)?;
    out.set_item("n_active_trace", n_active_trace_py)?;
    out.set_item("full_iter_trace", full_iter_trace_py)?;
    out.set_item("full_h2_trace", full_h2_trace_py)?;
    out.set_item("full_var_e_trace", full_var_e_trace_py)?;
    out.set_item("full_prob_in_trace", full_prob_in_trace_py)?;
    out.set_item("full_n_active_trace", full_n_active_trace_py)?;
    out.set_item("beta_trace_indices", beta_trace_indices_py)?;
    out.set_item("beta_trace", beta_trace_py)?;
    Ok(out)
}

#[inline]
fn row_major_block_mul_vec_f64(
    block: &[f64],
    rows: usize,
    cols: usize,
    vec: &[f64],
    out: &mut [f64],
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(vec.len(), cols);
    debug_assert_eq!(out.len(), rows);
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    unsafe {
        cblas_dgemm_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_TRANS,
            CBLAS_NO_TRANS,
            rows as CblasInt,
            1 as CblasInt,
            cols as CblasInt,
            1.0_f64,
            block.as_ptr(),
            cols as CblasInt,
            vec.as_ptr(),
            cols as CblasInt,
            0.0_f64,
            out.as_mut_ptr(),
            rows as CblasInt,
        );
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        for r in 0..rows {
            let row = &block[r * cols..(r + 1) * cols];
            let mut acc = 0.0_f64;
            for c in 0..cols {
                acc += row[c] * vec[c];
            }
            out[r] = acc;
        }
    }
}

#[inline]
fn row_major_block_t_mul_vec_f64(
    block: &[f64],
    rows: usize,
    cols: usize,
    vec: &[f64],
    out: &mut [f64],
) {
    debug_assert_eq!(block.len(), rows.saturating_mul(cols));
    debug_assert_eq!(vec.len(), rows);
    debug_assert_eq!(out.len(), cols);
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    unsafe {
        cblas_dgemm_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            cols as CblasInt,
            1 as CblasInt,
            rows as CblasInt,
            1.0_f64,
            block.as_ptr(),
            cols as CblasInt,
            vec.as_ptr(),
            rows as CblasInt,
            0.0_f64,
            out.as_mut_ptr(),
            cols as CblasInt,
        );
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        for c in 0..cols {
            let mut acc = 0.0_f64;
            for r in 0..rows {
                acc += block[r * cols + c] * vec[r];
            }
            out[c] = acc;
        }
    }
}

#[inline]
fn row_major_xtx_f64(x: &[f64], n: usize, q: usize, xtx_out: &mut [f64]) {
    debug_assert_eq!(x.len(), n.saturating_mul(q));
    debug_assert_eq!(xtx_out.len(), q.saturating_mul(q));
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    unsafe {
        cblas_dgemm_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            q as CblasInt,
            q as CblasInt,
            n as CblasInt,
            1.0_f64,
            x.as_ptr(),
            q as CblasInt,
            x.as_ptr(),
            q as CblasInt,
            0.0_f64,
            xtx_out.as_mut_ptr(),
            q as CblasInt,
        );
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        for i in 0..q {
            for j in 0..q {
                let mut acc = 0.0_f64;
                for r in 0..n {
                    acc += x[r * q + i] * x[r * q + j];
                }
                xtx_out[i * q + j] = acc;
            }
        }
    }
}

#[inline]
pub(crate) fn ddot_f64(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    if x.is_empty() {
        return 0.0_f64;
    }
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        if x.len() <= (CblasInt::MAX as usize) {
            unsafe {
                return cblas_ddot_dispatch(
                    x.len() as CblasInt,
                    x.as_ptr(),
                    1 as CblasInt,
                    y.as_ptr(),
                    1 as CblasInt,
                );
            }
        }
    }
    x.iter()
        .zip(y.iter())
        .map(|(xi, yi)| (*xi) * (*yi))
        .sum::<f64>()
}

#[inline]
#[cfg(test)]
pub(crate) fn daxpy_inplace_f64(alpha: f64, x: &[f64], y: &mut [f64]) {
    debug_assert_eq!(x.len(), y.len());
    if alpha == 0.0_f64 || x.is_empty() {
        return;
    }
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    {
        if x.len() <= (CblasInt::MAX as usize) {
            unsafe {
                cblas_daxpy_dispatch(
                    x.len() as CblasInt,
                    alpha,
                    x.as_ptr(),
                    1 as CblasInt,
                    y.as_mut_ptr(),
                    1 as CblasInt,
                );
            }
            return;
        }
    }
    for (yi, xi) in y.iter_mut().zip(x.iter()) {
        *yi = alpha.mul_add(*xi, *yi);
    }
}

/// Return the conditional marker score without materializing `r + x_j beta_j`.
///
/// The maintained residual is `r = y - X beta`, so it still contains the
/// current marker effect.  Adding `beta_j * d_j` recovers
/// `x_j^T (r + x_j beta_j)` with a single dot product over the samples.
#[inline]
#[cfg(test)]
pub(crate) fn marker_conditional_dot_f64(
    residual: &[f64],
    marker: &[f64],
    old_beta: f64,
    marker_ss: f64,
) -> f64 {
    ddot_f64(residual, marker) + old_beta * marker_ss
}

/// Mixed-precision marker score: f32 genotype storage with an f64 residual
/// and f64 accumulation.  This is intentionally separate from the all-f64
/// helper so callers cannot accidentally downcast the residual or statistic.
#[inline]
#[cfg(test)]
pub(crate) fn marker_conditional_dot_f32_f64(
    residual: &[f64],
    marker: &[f32],
    old_beta: f64,
    marker_ss: f64,
) -> f64 {
    debug_assert_eq!(residual.len(), marker.len());
    // ARM64 NEON converts four f32 marker values to two f64x2 vectors and
    // accumulates in f64.  This avoids a scalar cast on every product while
    // preserving the mixed-precision accumulator contract.
    let (mut dot, mut i) = {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { marker_dot_f32_f64_neon(residual, marker) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            (0.0_f64, 0usize)
        }
    };
    while i + 4 <= marker.len() {
        dot = residual[i].mul_add(marker[i] as f64, dot);
        dot = residual[i + 1].mul_add(marker[i + 1] as f64, dot);
        dot = residual[i + 2].mul_add(marker[i + 2] as f64, dot);
        dot = residual[i + 3].mul_add(marker[i + 3] as f64, dot);
        i += 4;
    }
    while i < marker.len() {
        dot = residual[i].mul_add(marker[i] as f64, dot);
        i += 1;
    }
    dot + old_beta * marker_ss
}

/// Marker score for the hot Gibbs loop when both the standardized marker and
/// maintained marker residual are f32.  Genotypes and the repeatedly streamed
/// residual stay compact, while the score is accumulated in f64 so marker
/// ordering and posterior draws remain stable under mixed precision.
#[inline]
#[cfg(test)]
pub(crate) fn marker_conditional_dot_f32_f32(
    residual: &[f32],
    marker: &[f32],
    old_beta: f64,
    marker_ss: f64,
) -> f64 {
    marker_conditional_dot_f32_f32_blocked(residual, marker, old_beta, marker_ss)
}

#[inline]
#[cfg(test)]
pub(crate) fn marker_conditional_dot_f32_f32_fast(
    residual: &[f32],
    marker: &[f32],
    old_beta: f64,
    marker_ss: f64,
) -> f64 {
    debug_assert_eq!(residual.len(), marker.len());
    let (mut dot, mut i) = {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { marker_dot_f32_f32_f64_neon(residual, marker) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            (0.0_f64, 0usize)
        }
    };
    while i < marker.len() {
        dot = (residual[i] as f64).mul_add(marker[i] as f64, dot);
        i += 1;
    }
    dot + old_beta * marker_ss
}

/// Compute the conditional marker score with f32 vector accumulators and a
/// f64 accumulator between SIMD blocks.  The previous kernel converted every
/// f32 lane to f64 before the FMA; that preserved more dot-product precision
/// than the marker input carries, but made the hot path conversion-bound on
/// Apple NEON.  Summing each 4-lane block in f32 and promoting only the block
/// total retains a bounded mixed-precision error while avoiding per-lane f64
/// conversion.
#[inline]
pub(crate) fn marker_conditional_dot_f32_f32_blocked(
    residual: &[f32],
    marker: &[f32],
    old_beta: f64,
    marker_ss: f64,
) -> f64 {
    debug_assert_eq!(residual.len(), marker.len());
    let (mut dot, mut i) = {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { marker_dot_f32_f32_blocked_neon(residual, marker) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let mut acc0 = 0.0_f32;
            let mut acc1 = 0.0_f32;
            let mut acc2 = 0.0_f32;
            let mut acc3 = 0.0_f32;
            let limit = marker.len() / 4 * 4;
            let mut i = 0usize;
            while i < limit {
                acc0 += residual[i] * marker[i];
                acc1 += residual[i + 1] * marker[i + 1];
                acc2 += residual[i + 2] * marker[i + 2];
                acc3 += residual[i + 3] * marker[i + 3];
                i += 4;
            }
            (
                f64::from(acc0) + f64::from(acc1) + f64::from(acc2) + f64::from(acc3),
                i,
            )
        }
    };
    while i < marker.len() {
        dot += f64::from(residual[i] * marker[i]);
        i += 1;
    }
    dot + old_beta * marker_ss
}

#[cfg(all(test, target_arch = "aarch64"))]
#[inline]
unsafe fn marker_dot_f32_f32_f64_neon(residual: &[f32], marker: &[f32]) -> (f64, usize) {
    let limit = marker.len() / 8 * 8;
    let mut i = 0usize;
    let mut acc0_lo = vdupq_n_f64(0.0);
    let mut acc0_hi = vdupq_n_f64(0.0);
    let mut acc1_lo = vdupq_n_f64(0.0);
    let mut acc1_hi = vdupq_n_f64(0.0);
    while i < limit {
        let m0 = vld1q_f32(marker.as_ptr().add(i));
        let r0 = vld1q_f32(residual.as_ptr().add(i));
        let m1 = vld1q_f32(marker.as_ptr().add(i + 4));
        let r1 = vld1q_f32(residual.as_ptr().add(i + 4));
        acc0_lo = vfmaq_f64(
            acc0_lo,
            vcvt_f64_f32(vget_low_f32(r0)),
            vcvt_f64_f32(vget_low_f32(m0)),
        );
        acc0_hi = vfmaq_f64(
            acc0_hi,
            vcvt_f64_f32(vget_high_f32(r0)),
            vcvt_f64_f32(vget_high_f32(m0)),
        );
        acc1_lo = vfmaq_f64(
            acc1_lo,
            vcvt_f64_f32(vget_low_f32(r1)),
            vcvt_f64_f32(vget_low_f32(m1)),
        );
        acc1_hi = vfmaq_f64(
            acc1_hi,
            vcvt_f64_f32(vget_high_f32(r1)),
            vcvt_f64_f32(vget_high_f32(m1)),
        );
        i += 8;
    }
    if i + 4 <= marker.len() {
        let m = vld1q_f32(marker.as_ptr().add(i));
        let r = vld1q_f32(residual.as_ptr().add(i));
        acc0_lo = vfmaq_f64(
            acc0_lo,
            vcvt_f64_f32(vget_low_f32(r)),
            vcvt_f64_f32(vget_low_f32(m)),
        );
        acc0_hi = vfmaq_f64(
            acc0_hi,
            vcvt_f64_f32(vget_high_f32(r)),
            vcvt_f64_f32(vget_high_f32(m)),
        );
        i += 4;
    }
    let mut lanes0_lo = [0.0_f64; 2];
    let mut lanes0_hi = [0.0_f64; 2];
    let mut lanes1_lo = [0.0_f64; 2];
    let mut lanes1_hi = [0.0_f64; 2];
    vst1q_f64(lanes0_lo.as_mut_ptr(), acc0_lo);
    vst1q_f64(lanes0_hi.as_mut_ptr(), acc0_hi);
    vst1q_f64(lanes1_lo.as_mut_ptr(), acc1_lo);
    vst1q_f64(lanes1_hi.as_mut_ptr(), acc1_hi);
    (
        lanes0_lo[0]
            + lanes0_lo[1]
            + lanes0_hi[0]
            + lanes0_hi[1]
            + lanes1_lo[0]
            + lanes1_lo[1]
            + lanes1_hi[0]
            + lanes1_hi[1],
        i,
    )
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn marker_dot_f32_f32_blocked_neon(residual: &[f32], marker: &[f32]) -> (f64, usize) {
    let limit = marker.len() / 16 * 16;
    let mut i = 0usize;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    let mut acc2 = vdupq_n_f32(0.0);
    let mut acc3 = vdupq_n_f32(0.0);
    while i < limit {
        acc0 = vfmaq_f32(
            acc0,
            vld1q_f32(residual.as_ptr().add(i)),
            vld1q_f32(marker.as_ptr().add(i)),
        );
        acc1 = vfmaq_f32(
            acc1,
            vld1q_f32(residual.as_ptr().add(i + 4)),
            vld1q_f32(marker.as_ptr().add(i + 4)),
        );
        acc2 = vfmaq_f32(
            acc2,
            vld1q_f32(residual.as_ptr().add(i + 8)),
            vld1q_f32(marker.as_ptr().add(i + 8)),
        );
        acc3 = vfmaq_f32(
            acc3,
            vld1q_f32(residual.as_ptr().add(i + 12)),
            vld1q_f32(marker.as_ptr().add(i + 12)),
        );
        i += 16;
    }
    let mut lanes0 = [0.0_f32; 4];
    let mut lanes1 = [0.0_f32; 4];
    let mut lanes2 = [0.0_f32; 4];
    let mut lanes3 = [0.0_f32; 4];
    vst1q_f32(lanes0.as_mut_ptr(), acc0);
    vst1q_f32(lanes1.as_mut_ptr(), acc1);
    vst1q_f32(lanes2.as_mut_ptr(), acc2);
    vst1q_f32(lanes3.as_mut_ptr(), acc3);
    let mut dot = lanes0.iter().copied().map(f64::from).sum::<f64>()
        + lanes1.iter().copied().map(f64::from).sum::<f64>()
        + lanes2.iter().copied().map(f64::from).sum::<f64>()
        + lanes3.iter().copied().map(f64::from).sum::<f64>();

    if i + 4 <= marker.len() {
        let acc = vfmaq_f32(
            vdupq_n_f32(0.0),
            vld1q_f32(residual.as_ptr().add(i)),
            vld1q_f32(marker.as_ptr().add(i)),
        );
        let mut lanes = [0.0_f32; 4];
        vst1q_f32(lanes.as_mut_ptr(), acc);
        dot += lanes.iter().copied().map(f64::from).sum::<f64>();
        i += 4;
    }
    (dot, i)
}

#[cfg(all(test, target_arch = "aarch64"))]
#[inline]
unsafe fn marker_dot_f32_f64_neon(residual: &[f64], marker: &[f32]) -> (f64, usize) {
    let limit = marker.len() / 4 * 4;
    let mut i = 0usize;
    let mut acc_lo = vdupq_n_f64(0.0);
    let mut acc_hi = vdupq_n_f64(0.0);
    while i < limit {
        let m = vld1q_f32(marker.as_ptr().add(i));
        let m_lo = vcvt_f64_f32(vget_low_f32(m));
        let m_hi = vcvt_f64_f32(vget_high_f32(m));
        let r_lo = vld1q_f64(residual.as_ptr().add(i));
        let r_hi = vld1q_f64(residual.as_ptr().add(i + 2));
        acc_lo = vfmaq_f64(acc_lo, r_lo, m_lo);
        acc_hi = vfmaq_f64(acc_hi, r_hi, m_hi);
        i += 4;
    }
    let mut lanes_lo = [0.0_f64; 2];
    let mut lanes_hi = [0.0_f64; 2];
    vst1q_f64(lanes_lo.as_mut_ptr(), acc_lo);
    vst1q_f64(lanes_hi.as_mut_ptr(), acc_hi);
    (lanes_lo[0] + lanes_lo[1] + lanes_hi[0] + lanes_hi[1], i)
}

/// Update a maintained residual after replacing one marker effect.
///
/// Since `r = y - X beta`, changing `beta_j` from `old_beta` to `new_beta`
/// requires `r += x_j * (old_beta - new_beta)`.  Keeping this as one BLAS
/// AXPY avoids separate remove/add scans of the sample vector.
#[inline]
#[cfg(test)]
pub(crate) fn update_marker_residual_f64(
    old_beta: f64,
    new_beta: f64,
    marker: &[f64],
    residual: &mut [f64],
) {
    daxpy_inplace_f64(old_beta - new_beta, marker, residual);
}

/// Mixed-precision residual update.  The residual remains f64 for numerical
/// stability while the marker is read from the f32 backend buffer.
#[inline]
#[cfg(test)]
pub(crate) fn update_marker_residual_f32_f64(
    old_beta: f64,
    new_beta: f64,
    marker: &[f32],
    residual: &mut [f64],
) {
    debug_assert_eq!(marker.len(), residual.len());
    let alpha = old_beta - new_beta;
    let mut i = {
        #[cfg(all(test, target_arch = "aarch64"))]
        {
            unsafe { update_marker_residual_f32_f64_neon(alpha, marker, residual) }
        }
        #[cfg(not(all(test, target_arch = "aarch64")))]
        {
            0usize
        }
    };
    while i + 4 <= marker.len() {
        residual[i] = alpha.mul_add(marker[i] as f64, residual[i]);
        residual[i + 1] = alpha.mul_add(marker[i + 1] as f64, residual[i + 1]);
        residual[i + 2] = alpha.mul_add(marker[i + 2] as f64, residual[i + 2]);
        residual[i + 3] = alpha.mul_add(marker[i + 3] as f64, residual[i + 3]);
        i += 4;
    }
    while i < marker.len() {
        residual[i] = alpha.mul_add(marker[i] as f64, residual[i]);
        i += 1;
    }
}

/// f32 marker residual update.  `alpha` is retained in f64 because sampled
/// effects are f64; the per-sample update is intentionally rounded once when
/// written back to the f32 residual buffer.
#[inline]
pub(crate) fn update_marker_residual_f32(
    old_beta: f64,
    new_beta: f64,
    marker: &[f32],
    residual: &mut [f32],
) {
    debug_assert_eq!(marker.len(), residual.len());
    let alpha = (old_beta - new_beta) as f32;
    // Spike components frequently leave a marker at exactly zero.  Avoid a
    // full sample-vector traversal when replacing zero by zero; this is the
    // dominant case for BayesB/C/R and is also safe for all backends.
    if alpha == 0.0_f32 || marker.is_empty() {
        return;
    }
    let mut i = {
        #[cfg(target_arch = "aarch64")]
        {
            unsafe { update_marker_residual_f32_neon(alpha, marker, residual) }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            0usize
        }
    };
    while i + 4 <= marker.len() {
        residual[i] = alpha.mul_add(marker[i], residual[i]);
        residual[i + 1] = alpha.mul_add(marker[i + 1], residual[i + 1]);
        residual[i + 2] = alpha.mul_add(marker[i + 2], residual[i + 2]);
        residual[i + 3] = alpha.mul_add(marker[i + 3], residual[i + 3]);
        i += 4;
    }
    while i < marker.len() {
        residual[i] = alpha.mul_add(marker[i], residual[i]);
        i += 1;
    }
}

/// Unified marker Gibbs kernel.  The conditional score is reduced once, the
/// caller samples the model-specific new effect from that score, and the
/// maintained f32 residual is updated before returning.  Keeping this
/// transition in one inlined function makes all BayesA/B/C/R backends share
/// the same SIMD dot/update path.  The two vector traversals are intentional:
/// a Gibbs draw cannot update the residual until its conditional draw is known.
#[inline]
pub(crate) fn bayes_marker_update_f32<F>(
    residual: &mut [f32],
    marker: &[f32],
    old_beta: f64,
    marker_ss: f64,
    choose_beta: F,
) -> (f64, f64)
where
    F: FnOnce(f64) -> f64,
{
    bayes_marker_update_f32_fused(residual, marker, old_beta, marker_ss, choose_beta)
}

/// Fast marker transition for models whose conditional draw cannot fail.
/// Keeping score, callback, and residual replacement in one inline function
/// removes the Result/closure adapter from BayesA/B/C while retaining the
/// required two streaming passes: the new beta is not known until the score
/// reduction has completed.
#[inline]
pub(crate) fn bayes_marker_update_f32_fused<F>(
    residual: &mut [f32],
    marker: &[f32],
    old_beta: f64,
    marker_ss: f64,
    choose_beta: F,
) -> (f64, f64)
where
    F: FnOnce(f64) -> f64,
{
    let u = marker_conditional_dot_f32_f32_blocked(residual, marker, old_beta, marker_ss);
    let new_beta = choose_beta(u);
    update_marker_residual_f32(old_beta, new_beta, marker, residual);
    (u, new_beta)
}

/// Result-returning form used by BayesR, whose component posterior calculation
/// performs input validation inside the marker callback.
#[inline]
pub(crate) fn bayes_marker_update_f32_result<F>(
    residual: &mut [f32],
    marker: &[f32],
    old_beta: f64,
    marker_ss: f64,
    choose_beta: F,
) -> Result<(f64, f64), String>
where
    F: FnOnce(f64) -> Result<f64, String>,
{
    let u = marker_conditional_dot_f32_f32_blocked(residual, marker, old_beta, marker_ss);
    let new_beta = choose_beta(u)?;
    update_marker_residual_f32(old_beta, new_beta, marker, residual);
    Ok((u, new_beta))
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn update_marker_residual_f32_neon(
    alpha: f32,
    marker: &[f32],
    residual: &mut [f32],
) -> usize {
    let limit = marker.len() / 4 * 4;
    let alpha_v = vdupq_n_f32(alpha);
    let mut i = 0usize;
    while i < limit {
        let m = vld1q_f32(marker.as_ptr().add(i));
        let r = vld1q_f32(residual.as_ptr().add(i));
        let out = vfmaq_f32(r, m, alpha_v);
        vst1q_f32(residual.as_mut_ptr().add(i), out);
        i += 4;
    }
    i
}

#[cfg(all(test, target_arch = "aarch64"))]
#[inline]
unsafe fn update_marker_residual_f32_f64_neon(
    alpha: f64,
    marker: &[f32],
    residual: &mut [f64],
) -> usize {
    let limit = marker.len() / 4 * 4;
    let alpha_v = vdupq_n_f64(alpha);
    let mut i = 0usize;
    while i < limit {
        let m = vld1q_f32(marker.as_ptr().add(i));
        let m_lo = vcvt_f64_f32(vget_low_f32(m));
        let m_hi = vcvt_f64_f32(vget_high_f32(m));
        let r_lo = vld1q_f64(residual.as_ptr().add(i));
        let r_hi = vld1q_f64(residual.as_ptr().add(i + 2));
        let out_lo = vfmaq_f64(r_lo, m_lo, alpha_v);
        let out_hi = vfmaq_f64(r_hi, m_hi, alpha_v);
        vst1q_f64(residual.as_mut_ptr().add(i), out_lo);
        vst1q_f64(residual.as_mut_ptr().add(i + 2), out_hi);
        i += 4;
    }
    i
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn update_alpha_gauss_seidel_blas(
    x: &[f64],
    n: usize,
    q: usize,
    inv_var_e: f64,
    inv_var_b_fixed: f64,
    x2_x: &[f64],
    xtx: &[f64],
    alpha: &mut [f64],
    r: &mut [f64],
    xtr_buf: &mut [f64],
    delta_alpha: &mut [f64],
    tmp_n: &mut [f64],
    rng: &mut StdRng,
) {
    debug_assert_eq!(x.len(), n.saturating_mul(q));
    debug_assert_eq!(x2_x.len(), q);
    debug_assert_eq!(xtx.len(), q.saturating_mul(q));
    debug_assert_eq!(alpha.len(), q);
    debug_assert_eq!(r.len(), n);
    debug_assert_eq!(xtr_buf.len(), q);
    debug_assert_eq!(delta_alpha.len(), q);
    debug_assert_eq!(tmp_n.len(), n);

    row_major_block_t_mul_vec_f64(x, n, q, r, xtr_buf);
    delta_alpha.fill(0.0_f64);

    for k in 0..q {
        let rhs = xtr_buf[k].mul_add(inv_var_e, x2_x[k] * alpha[k] * inv_var_e);
        let c = x2_x[k].mul_add(inv_var_e, inv_var_b_fixed);
        let z_alpha: f64 = rng.sample(StandardNormal);
        let new_alpha = rhs / c + (1.0_f64 / c).sqrt() * z_alpha;
        let delta = alpha[k] - new_alpha;
        alpha[k] = new_alpha;
        delta_alpha[k] = delta;
        if delta != 0.0_f64 {
            for t in 0..q {
                xtr_buf[t] += delta * xtx[t * q + k];
            }
        }
    }

    row_major_block_mul_vec_f64(x, n, q, delta_alpha, tmp_n);
    for i in 0..n {
        r[i] += tmp_n[i];
    }
}

pub(crate) struct PackedByteLut {
    code4: [[u8; 4]; 256],
}

pub(crate) fn packed_byte_lut() -> &'static PackedByteLut {
    static LUT: OnceLock<PackedByteLut> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut code4 = [[0u8; 4]; 256];
        for b in 0u16..=255 {
            let byte = b as u8;
            for lane in 0..4usize {
                code4[byte as usize][lane] = (byte >> (lane * 2)) & 0b11;
            }
        }
        PackedByteLut { code4 }
    })
}

impl PackedByteLut {
    pub(crate) fn code4(&self) -> &[[u8; 4]; 256] {
        &self.code4
    }
}

fn parse_env_truthy(raw: &str) -> Option<bool> {
    let s = raw.trim().to_ascii_lowercase();
    if s.is_empty() {
        return None;
    }
    if matches!(s.as_str(), "1" | "true" | "yes" | "y" | "on") {
        return Some(true);
    }
    if matches!(s.as_str(), "0" | "false" | "no" | "n" | "off") {
        return Some(false);
    }
    None
}

#[inline]
fn bayes_packed_double_buffer_enabled() -> bool {
    env::var("JX_BAYES_PACKED_DOUBLE_BUFFER")
        .ok()
        .and_then(|raw| parse_env_truthy(&raw))
        .unwrap_or(true)
}

#[inline]
pub(crate) fn bayes_packed_block_rows(n: usize, p: usize) -> usize {
    if n == 0 || p == 0 {
        return 1;
    }
    let default_target_bytes: usize = 32 * 1024 * 1024;
    let bytes_per_row = n.saturating_mul(std::mem::size_of::<f32>()).max(1);
    let mut rows = (default_target_bytes / bytes_per_row).max(1);
    rows = rows.clamp(8, 2048);
    if let Ok(raw) = env::var("JX_BAYES_PACKED_ROW_BLOCK") {
        if let Ok(v) = raw.trim().parse::<usize>() {
            if v > 0 {
                rows = v;
            }
        }
    }
    rows.clamp(1, p)
}

#[inline]
fn bayes_packed_predecode_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        if let Ok(raw) = env::var("JX_BAYES_PACKED_PREDECODE") {
            if let Some(v) = parse_env_truthy(&raw) {
                return v;
            }
        }
        true
    })
}

#[inline]
fn bayes_packed_predecode_max_mb() -> usize {
    static MAX_MB: OnceLock<usize> = OnceLock::new();
    *MAX_MB.get_or_init(|| {
        if let Ok(raw) = env::var("JX_BAYES_PACKED_PREDECODE_MAX_MB") {
            if let Ok(v) = raw.trim().parse::<usize>() {
                if v > 0 {
                    return v;
                }
            }
        }
        768usize
    })
}

#[inline]
pub(crate) fn bayes_packed_blas_threads() -> usize {
    // Default policy stays "BLAS single-thread + Rayon parallel decode".
    // Env override for experiments:
    //   JX_BAYES_PACKED_BLAS_THREADS=4  -> force BLAS 4 threads in Bayes packed kernels
    //   JX_BAYES_PACKED_BLAS_THREADS=0  -> do not override BLAS threads
    if let Ok(raw) = env::var("JX_BAYES_PACKED_BLAS_THREADS") {
        if let Ok(v) = raw.trim().parse::<usize>() {
            return v;
        }
    }
    1usize
}

#[inline]
fn bayes_packed_should_predecode_dense(n: usize, p: usize) -> bool {
    if !bayes_packed_predecode_enabled() {
        return false;
    }
    let elem_cnt = n.saturating_mul(p);
    if elem_cnt == 0 {
        return false;
    }
    let dense_bytes = elem_cnt.saturating_mul(std::mem::size_of::<f32>());
    let cap_bytes = bayes_packed_predecode_max_mb().saturating_mul(1024 * 1024);
    if dense_bytes > cap_bytes {
        return false;
    }
    // Keep a lightweight guard on huge dimensions even if memory cap allows it.
    elem_cnt <= 80_000_000usize
}

#[inline]
fn bayes_stream_window_mb(n_samples: usize, block_rows: usize) -> usize {
    let bytes_per_snp = n_samples.div_ceil(4).max(1);
    let target_bytes = block_rows
        .max(1)
        .saturating_mul(bytes_per_snp)
        .saturating_mul(2);
    target_bytes.div_ceil(1024 * 1024).max(1)
}

pub(crate) enum BayesPackedSource<'a> {
    Resident {
        packed_flat: &'a [u8],
        bytes_per_snp: usize,
    },
    Windowed(WindowedBedMatrix),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BayesDecodeMode {
    Dense,
    Single,
    Double,
}

#[inline]
pub(crate) fn bayes_decode_mode(p: usize, block_rows: usize) -> BayesDecodeMode {
    if p == 0 || block_rows >= p {
        BayesDecodeMode::Dense
    } else {
        BayesDecodeMode::Double
    }
}

/// Common marker access contract used by every Bayesian marker sampler.
///
/// Marker rows are returned in standardized f32 form.  The fixed-effect
/// residual is synchronized in f64 outside the marker sweep, while the hot
/// marker residual and genotype buffers are f32 and all reductions remain f64.
/// Resident packed BED, windowed BED, and dense inputs therefore share one
/// compact marker path.
pub(crate) trait BayesMarkerBackend {
    fn n_samples(&self) -> usize;
    fn n_markers(&self) -> usize;
    fn block_rows(&self) -> usize;
    fn fill_block(
        &mut self,
        row_start: usize,
        row_end: usize,
        out_block: &mut [f32],
        n: usize,
    ) -> Result<(), String>;

    fn for_each_block<F>(&mut self, mut f: F) -> Result<(), String>
    where
        F: FnMut(usize, usize, &[f32]) -> Result<(), String>,
        Self: Sized,
    {
        let n = self.n_samples();
        let p = self.n_markers();
        let block_rows = self.block_rows().max(1).min(p.max(1));
        let mut block = vec![0.0_f32; block_rows * n];
        for row_start in (0..p).step_by(block_rows) {
            let row_end = (row_start + block_rows).min(p);
            let block_len = (row_end - row_start) * n;
            self.fill_block(row_start, row_end, &mut block[..block_len], n)?;
            f(row_start, row_end, &block[..block_len])?;
        }
        Ok(())
    }
}

/// Dense marker backend used by the in-memory Bayes entry points.
pub(crate) struct DenseBayesBackend<'a> {
    matrix: &'a [f32],
    n: usize,
    p: usize,
    block_rows: usize,
}

impl<'a> DenseBayesBackend<'a> {
    pub(crate) fn new(matrix: &'a [f32], n: usize, p: usize) -> Result<Self, String> {
        if matrix.len() != p.saturating_mul(n) {
            return Err("Bayes dense genotype dimensions are incompatible".to_string());
        }
        Ok(Self {
            matrix,
            n,
            p,
            block_rows: p.min(2048).max(1),
        })
    }
}

impl BayesMarkerBackend for DenseBayesBackend<'_> {
    fn n_samples(&self) -> usize {
        self.n
    }

    fn n_markers(&self) -> usize {
        self.p
    }

    fn block_rows(&self) -> usize {
        self.block_rows
    }

    fn fill_block(
        &mut self,
        row_start: usize,
        row_end: usize,
        out_block: &mut [f32],
        n: usize,
    ) -> Result<(), String> {
        if row_start > row_end || row_end > self.p || n != self.n {
            return Err("Bayes dense block bounds are invalid".to_string());
        }
        let expected = (row_end - row_start).saturating_mul(n);
        if out_block.len() != expected {
            return Err("Bayes dense block buffer has an invalid length".to_string());
        }
        out_block.copy_from_slice(&self.matrix[row_start * n..row_end * n]);
        Ok(())
    }

    /// Dense rows are already resident and contiguous.  Passing a borrowed
    /// slice avoids copying every marker block into a staging buffer on every
    /// MCMC iteration; packed/stream backends keep the default fill path.
    fn for_each_block<F>(&mut self, mut f: F) -> Result<(), String>
    where
        F: FnMut(usize, usize, &[f32]) -> Result<(), String>,
        Self: Sized,
    {
        let n = self.n;
        let p = self.p;
        let block_rows = self.block_rows.max(1).min(p.max(1));
        for row_start in (0..p).step_by(block_rows) {
            let row_end = (row_start + block_rows).min(p);
            f(row_start, row_end, &self.matrix[row_start * n..row_end * n])?;
        }
        Ok(())
    }
}

/// Shared packed/stream BED backend.
///
/// This owns all source-specific state (row/sample mapping, decode scratch,
/// thread pool and source window).  An explicit memory-derived block uses two
/// reusable decode buffers by default; when that block covers every active
/// marker, the source is materialized once and the dense backend is used.
/// Keeping this policy here makes it apply uniformly to every Bayes model.
pub(crate) struct PackedBayesBackend<'a, 'source> {
    source: &'a mut BayesPackedSource<'source>,
    n_samples_total: usize,
    p: usize,
    row_flip: &'a [bool],
    row_mean: &'a [f32],
    row_inv_sd: &'a [f32],
    packed_row_indices: Option<&'a [usize]>,
    sample_indices: &'a [usize],
    full_sample_fast: bool,
    block_rows: usize,
    pool: Option<&'a Arc<rayon::ThreadPool>>,
    scratch_f32: Vec<f32>,
    decode_mode: BayesDecodeMode,
}

impl<'a, 'source> PackedBayesBackend<'a, 'source> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        source: &'a mut BayesPackedSource<'source>,
        n_samples_total: usize,
        p: usize,
        row_flip: &'a [bool],
        row_mean: &'a [f32],
        row_inv_sd: &'a [f32],
        packed_row_indices: Option<&'a [usize]>,
        sample_indices: &'a [usize],
        block_rows: Option<usize>,
        pool: Option<&'a Arc<rayon::ThreadPool>>,
    ) -> Result<Self, String> {
        if p == 0 {
            return Err("Bayes packed genotype has no markers".to_string());
        }
        if row_flip.len() != p || row_mean.len() != p || row_inv_sd.len() != p {
            return Err("Bayes packed row metadata length mismatch".to_string());
        }
        if let Some(indices) = packed_row_indices {
            if indices.len() != p {
                return Err("Bayes packed source-row index length mismatch".to_string());
            }
        }
        if sample_indices.is_empty() || sample_indices.iter().any(|&i| i >= n_samples_total) {
            return Err("Bayes packed sample indices are invalid".to_string());
        }
        let explicit_block_rows = block_rows.is_some();
        let resolved_block_rows = block_rows
            .unwrap_or_else(|| bayes_packed_block_rows(sample_indices.len(), p))
            .max(1)
            .min(p);
        let decode_mode = if explicit_block_rows {
            let base_mode = bayes_decode_mode(p, resolved_block_rows);
            if base_mode == BayesDecodeMode::Double && !bayes_packed_double_buffer_enabled() {
                BayesDecodeMode::Single
            } else {
                base_mode
            }
        } else {
            BayesDecodeMode::Single
        };
        Ok(Self {
            source,
            n_samples_total,
            p,
            row_flip,
            row_mean,
            row_inv_sd,
            packed_row_indices,
            sample_indices,
            full_sample_fast: is_identity_indices(sample_indices, n_samples_total),
            block_rows: resolved_block_rows,
            pool,
            scratch_f32: Vec::new(),
            decode_mode,
        })
    }

    /// Materialize a packed source when the common predecode policy allows it.
    /// An explicit block covering all active markers forces this dense path,
    /// including for a windowed BED source.
    pub(crate) fn maybe_predecode_dense_f32(&mut self) -> Result<Option<Vec<f32>>, String> {
        let force_dense = self.decode_mode == BayesDecodeMode::Dense;
        let dense = maybe_predecode_source_dense_f32(
            &mut self.source,
            self.n_samples_total,
            self.row_flip,
            self.row_mean,
            self.row_inv_sd,
            self.sample_indices,
            self.sample_indices.len(),
            self.p,
            self.packed_row_indices,
            packed_byte_lut().code4(),
            self.pool,
            Some(self.block_rows),
            force_dense,
        )?;
        if dense.is_some() {
            self.decode_mode = BayesDecodeMode::Dense;
        }
        Ok(dense)
    }

    fn duplicate_source(&self) -> Result<BayesPackedSource<'source>, String> {
        match &*self.source {
            BayesPackedSource::Resident {
                packed_flat,
                bytes_per_snp,
            } => Ok(BayesPackedSource::Resident {
                packed_flat,
                bytes_per_snp: *bytes_per_snp,
            }),
            BayesPackedSource::Windowed(matrix) => {
                Ok(BayesPackedSource::Windowed(matrix.duplicate()?))
            }
        }
    }

    pub(crate) fn for_each_block<F>(&mut self, mut f: F) -> Result<(), String>
    where
        F: FnMut(usize, usize, &[f32]) -> Result<(), String>,
    {
        if self.decode_mode == BayesDecodeMode::Double {
            return self.for_each_block_double(&mut f);
        }
        let n = self.sample_indices.len();
        let block_rows = self.block_rows.max(1).min(self.p.max(1));
        let mut block = vec![0.0_f32; block_rows * n];
        for row_start in (0..self.p).step_by(block_rows) {
            let row_end = (row_start + block_rows).min(self.p);
            let block_len = (row_end - row_start) * n;
            self.fill_block(row_start, row_end, &mut block[..block_len], n)?;
            f(row_start, row_end, &block[..block_len])?;
        }
        Ok(())
    }

    fn for_each_block_double<F>(&mut self, f: &mut F) -> Result<(), String>
    where
        F: FnMut(usize, usize, &[f32]) -> Result<(), String>,
    {
        let worker_source = self.duplicate_source()?;
        let n = self.sample_indices.len();
        let p = self.p;
        let block_rows = self.block_rows.max(1).min(p.max(1));
        let n_samples_total = self.n_samples_total;
        let row_flip = self.row_flip;
        let row_mean = self.row_mean;
        let row_inv_sd = self.row_inv_sd;
        let packed_row_indices = self.packed_row_indices;
        let sample_indices = self.sample_indices;
        let full_sample_fast = self.full_sample_fast;
        let pool = self.pool;
        let code4_lut = packed_byte_lut().code4();

        std::thread::scope(|scope| {
            let (request_tx, request_rx) = sync_channel::<(usize, usize, Vec<f32>)>(2);
            let (ready_tx, ready_rx) = sync_channel::<Result<(usize, usize, Vec<f32>), String>>(2);

            scope.spawn(move || {
                let mut source = worker_source;
                let mut scratch_f32 = Vec::<f32>::new();
                while let Ok((row_start, row_end, mut block)) = request_rx.recv() {
                    let block_len = (row_end - row_start) * n;
                    let result = decode_source_block_standardized_into(
                        &mut source,
                        n_samples_total,
                        row_start,
                        row_end,
                        sample_indices,
                        full_sample_fast,
                        packed_row_indices,
                        row_flip,
                        row_mean,
                        row_inv_sd,
                        code4_lut,
                        &mut block[..block_len],
                        n,
                        pool,
                        &mut scratch_f32,
                    )
                    .map(|_| (row_start, row_end, block));
                    if ready_tx.send(result).is_err() {
                        break;
                    }
                }
            });

            let mut blocks = (0..p).step_by(block_rows).map(|row_start| {
                let row_end = (row_start + block_rows).min(p);
                (row_start, row_end)
            });
            let send_request =
                |row_start: usize, row_end: usize, mut block: Vec<f32>| -> Result<(), String> {
                    let block_len = (row_end - row_start) * n;
                    if block.len() != block_len {
                        block.resize(block_len, 0.0);
                    }
                    request_tx
                        .send((row_start, row_end, block))
                        .map_err(|_| "Bayes double-buffer worker stopped".to_string())
                };

            let mut in_flight = 0usize;
            for _ in 0..2 {
                if let Some((row_start, row_end)) = blocks.next() {
                    send_request(row_start, row_end, Vec::new())?;
                    in_flight += 1;
                }
            }

            while in_flight > 0 {
                let ready = ready_rx
                    .recv()
                    .map_err(|_| "Bayes double-buffer worker stopped".to_string())??;
                let (row_start, row_end, block) = ready;
                in_flight -= 1;
                f(row_start, row_end, &block)?;
                if let Some((next_start, next_end)) = blocks.next() {
                    send_request(next_start, next_end, block)?;
                    in_flight += 1;
                }
            }
            Ok(())
        })
    }
}

impl BayesMarkerBackend for PackedBayesBackend<'_, '_> {
    fn n_samples(&self) -> usize {
        self.sample_indices.len()
    }

    fn n_markers(&self) -> usize {
        self.p
    }

    fn block_rows(&self) -> usize {
        self.block_rows
    }

    fn fill_block(
        &mut self,
        row_start: usize,
        row_end: usize,
        out_block: &mut [f32],
        n: usize,
    ) -> Result<(), String> {
        if row_start > row_end || row_end > self.p || n != self.sample_indices.len() {
            return Err("Bayes packed block bounds are invalid".to_string());
        }
        decode_source_block_standardized_into(
            &mut self.source,
            self.n_samples_total,
            row_start,
            row_end,
            self.sample_indices,
            self.full_sample_fast,
            self.packed_row_indices,
            self.row_flip,
            self.row_mean,
            self.row_inv_sd,
            packed_byte_lut().code4(),
            out_block,
            n,
            self.pool,
            &mut self.scratch_f32,
        )
    }
}

pub(crate) fn build_bayes_source<'a>(
    resident_packed_flat: Option<&'a [u8]>,
    prefix: &str,
    n_samples: usize,
    block_rows: usize,
    mmap_window_mb: Option<usize>,
) -> Result<BayesPackedSource<'a>, String> {
    if let Some(packed_flat) = resident_packed_flat {
        return Ok(BayesPackedSource::Resident {
            packed_flat,
            bytes_per_snp: n_samples.div_ceil(4),
        });
    }
    if prefix.trim().is_empty() {
        return Err("Bayes streaming path requires non-empty prefix".to_string());
    }
    let window_mb = mmap_window_mb
        .map(|v| v.max(1))
        .unwrap_or_else(|| bayes_stream_window_mb(n_samples, block_rows));
    let matrix = WindowedBedMatrix::open(prefix, window_mb)?;
    Ok(BayesPackedSource::Windowed(matrix))
}

#[inline]
pub(crate) fn decode_source_block_standardized_into(
    source: &mut BayesPackedSource<'_>,
    n_samples: usize,
    row_start: usize,
    row_end: usize,
    sample_idx: &[usize],
    full_sample_fast: bool,
    packed_row_indices: Option<&[usize]>,
    row_flip: &[bool],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    code4_lut: &[[u8; 4]; 256],
    out_block: &mut [f32],
    n: usize,
    pool: Option<&Arc<rayon::ThreadPool>>,
    scratch_f32: &mut Vec<f32>,
) -> Result<(), String> {
    let block_rows = row_end - row_start;
    debug_assert_eq!(out_block.len(), block_rows * n);
    if scratch_f32.len() < block_rows * n {
        scratch_f32.resize(block_rows * n, 0.0_f32);
    }
    let tmp = &mut scratch_f32[..block_rows * n];
    match source {
        BayesPackedSource::Resident {
            packed_flat,
            bytes_per_snp,
        } => {
            decode_standardized_packed_block_f32(
                packed_flat,
                *bytes_per_snp,
                n_samples,
                row_flip,
                row_mean,
                row_inv_sd,
                sample_idx,
                full_sample_fast,
                row_start,
                tmp,
                code4_lut,
                pool,
            )?;
        }
        BayesPackedSource::Windowed(matrix) => {
            let cur_rows = row_end - row_start;
            let bytes_per_snp = matrix.bytes_per_snp();
            let packed_slice = if let Some(indices) = packed_row_indices {
                let source_rows = &indices[row_start..row_end];
                let mut rel_indices = Vec::with_capacity(cur_rows);
                let packed_slice = matrix.prepare_source_rows(source_rows, &mut rel_indices)?;
                decode_prepared_additive_block_packed_f32(
                    packed_slice,
                    bytes_per_snp,
                    n_samples,
                    &row_flip[row_start..row_end],
                    &row_mean[row_start..row_end],
                    &row_inv_sd[row_start..row_end],
                    sample_idx,
                    full_sample_fast,
                    None,
                    Some(rel_indices.as_slice()),
                    0usize,
                    cur_rows,
                    tmp,
                    pool,
                )?;
                out_block.copy_from_slice(tmp);
                return Ok(());
            } else {
                matrix.read_source_range(row_start, row_end)?
            };
            decode_prepared_additive_block_packed_f32(
                packed_slice,
                bytes_per_snp,
                n_samples,
                &row_flip[row_start..row_end],
                &row_mean[row_start..row_end],
                &row_inv_sd[row_start..row_end],
                sample_idx,
                full_sample_fast,
                None,
                None,
                0usize,
                cur_rows,
                tmp,
                pool,
            )?;
        }
    }
    out_block.copy_from_slice(tmp);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn maybe_predecode_source_dense_f32(
    source: &mut BayesPackedSource<'_>,
    n_samples: usize,
    row_flip: &[bool],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    n: usize,
    p: usize,
    packed_row_indices: Option<&[usize]>,
    code4_lut: &[[u8; 4]; 256],
    pool: Option<&Arc<rayon::ThreadPool>>,
    block_rows_override: Option<usize>,
    force_dense: bool,
) -> Result<Option<Vec<f32>>, String> {
    if !force_dense && matches!(source, BayesPackedSource::Windowed(_)) {
        return Ok(None);
    }
    if !force_dense && !bayes_packed_should_predecode_dense(n, p) {
        return Ok(None);
    }
    let full_sample_fast = is_identity_indices(sample_idx, n_samples);
    let block_rows = block_rows_override
        .unwrap_or_else(|| bayes_packed_block_rows(n, p))
        .max(1)
        .min(p.max(1));
    let mut dense_f32 = vec![0.0_f32; p * n];
    let mut scratch_f32 = Vec::<f32>::new();
    for st in (0..p).step_by(block_rows) {
        let ed = (st + block_rows).min(p);
        let br = ed - st;
        decode_source_block_standardized_into(
            source,
            n_samples,
            st,
            ed,
            sample_idx,
            full_sample_fast,
            packed_row_indices,
            row_flip,
            row_mean,
            row_inv_sd,
            code4_lut,
            &mut dense_f32[st * n..(st + br) * n],
            n,
            pool,
            &mut scratch_f32,
        )?;
    }
    Ok(Some(dense_f32))
}

pub(crate) fn genetic_variance_from_residual(
    y: &[f64],
    r: &[f64],
    x: &[f64],
    alpha: &[f64],
    n: usize,
    q: usize,
) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    let mut mean_g = 0.0;
    let mut m2 = 0.0;
    for i in 0..n {
        let mut xa = 0.0;
        for k in 0..q {
            xa += x[i * q + k] * alpha[k];
        }
        let g = y[i] - r[i] - xa;
        let delta = g - mean_g;
        mean_g += delta / (i as f64 + 1.0);
        let delta2 = g - mean_g;
        m2 += delta * delta2;
    }
    m2 / (n as f64 - 1.0)
}

#[inline]
fn bayesb_rate0(shape0: f64, rate0_opt: Option<f64>, s0_b: f64) -> Result<f64, String> {
    let rate0 = match rate0_opt {
        Some(v) => v,
        None => {
            if shape0 <= 1.0 {
                return Err("shape0 must be > 1 when rate0 is not provided".to_string());
            }
            (shape0 - 1.0) / s0_b
        }
    };
    if rate0 <= 0.0 {
        return Err("rate0 must be positive".to_string());
    }
    Ok(rate0)
}

#[inline]
fn bayes_inclusion_prior_shapes(prob_in_init: f64, counts: f64) -> (f64, f64) {
    let counts_in = (counts * prob_in_init).max(f64::MIN_POSITIVE);
    let counts_out = (counts * (1.0 - prob_in_init)).max(f64::MIN_POSITIVE);
    (counts_in, counts_out)
}

#[inline]
fn sample_gamma_with_rate<R: Rng + ?Sized>(
    rng: &mut R,
    shape: f64,
    rate: f64,
) -> Result<f64, String> {
    if !(shape.is_finite() && shape > 0.0) {
        return Err("Gamma shape must be positive and finite".to_string());
    }
    if !(rate.is_finite() && rate > 0.0) {
        return Err("Gamma rate must be positive and finite".to_string());
    }
    let gamma = Gamma::new(shape, 1.0).map_err(|e| e.to_string())?;
    Ok(rng.sample(gamma) / rate)
}

#[inline]
fn bayes_positive_floor(x: f64) -> f64 {
    x.max(1e-300_f64)
}

fn bayesb_core_impl(
    y: &[f64],
    m: &[f32],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0_opt: Option<f64>,
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    fixed_prob_in_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    seed: Option<u64>,
) -> Result<
    (
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        f64,
        f64,
        f64,
        f64,
        f64,
        Vec<f64>,
        f64,
        usize,
        usize,
        usize,
    ),
    String,
> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    let prob_in_base = fixed_prob_in_opt.unwrap_or(prob_in_init);
    if !(prob_in_base > 0.0 && prob_in_base < 1.0) {
        return Err("prob_in must be in (0, 1)".to_string());
    }
    if counts < 0.0 {
        return Err("counts must be >= 0".to_string());
    }

    if m.len() != n * p {
        return Err("M has incompatible dimensions".to_string());
    }
    if x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32]; // 生成 32 字节随机种子（适配 StdRng）
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes) // 用随机种子初始化 StdRng（无 Result，直接返回）
        }
    };

    let mut x2 = vec![0.0_f64; p];
    let mut mean_x = vec![0.0_f64; p];
    for j in 0..p {
        let mut s = 0.0;
        let mut msum = 0.0;
        for i in 0..n {
            let v = m[j * n + i];
            s += (v as f64) * (v as f64);
            msum += v as f64;
        }
        x2[j] = s;
        mean_x[j] = msum / n_f;
    }
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let mut y_mean = 0.0;
    for v in y {
        y_mean += *v;
    }
    y_mean /= n_f;
    let mut var_y = 0.0;
    for v in y {
        let d = *v - y_mean;
        var_y += d * d;
    }
    var_y /= n_f - 1.0;

    let s0_b = match s0_b_opt {
        Some(v) => v,
        None => {
            if msx <= 0.0 {
                return Err("MSx must be positive to compute S0_b".to_string());
            }
            var_y * r2 / msx * (df0_b + 2.0) / prob_in_base
        }
    };
    if s0_b <= 0.0 {
        return Err("S0_b must be positive".to_string());
    }

    let rate0 = bayesb_rate0(shape0, rate0_opt, s0_b)?;

    let mut var_e = var_y * (1.0 - r2);
    if var_e <= 0.0 {
        return Err("varE must be positive; check R2".to_string());
    }

    let prior_ss_e = match prior_ss_e_opt {
        Some(v) => v,
        None => var_e * (df0_e + 2.0),
    };
    if prior_ss_e <= 0.0 {
        return Err("prior_ss_e must be positive".to_string());
    }

    let mut beta = vec![0.0; p];
    let mut d = vec![0u8; p];
    let mut var_b = vec![s0_b / (df0_b + 2.0); p];
    let mut s = s0_b;
    let mut prob_in = prob_in_base;
    let (counts_in, counts_out) = bayes_inclusion_prior_shapes(prob_in_base, counts);

    let mut alpha = vec![0.0; q];
    let mut x2_x = vec![0.0; q];
    for k in 0..q {
        let mut s2 = 0.0;
        for i in 0..n {
            let v = x[i * q + k];
            s2 += v * v;
        }
        x2_x[k] = s2;
    }

    let mut r = y.to_vec();
    let mut marker_residual = vec![0.0_f32; n];
    let mut beta_sum = vec![0.0; p];
    let mut pip_sum = vec![0.0; p];
    let mut varb_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut prob_in_sum = 0.0;
    let mut n_active_sum = 0.0;
    let mut schedule = BayesSamplingController::new(n_iter, burnin, thin);
    let var_b_fixed = 1e10_f64;

    let chi_b_active = ChiSquared::new(df0_b + 1.0).map_err(|e| e.to_string())?;
    let chi_b_inactive = ChiSquared::new(df0_b).map_err(|e| e.to_string())?;
    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    while schedule.should_run() {
        let retain_sample = schedule.begin_iteration();
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        for k in 0..q {
            let mut rhs = 0.0;
            for i in 0..n {
                rhs += x[i * q + k] * r[i];
            }
            rhs = rhs * inv_var_e + x2_x[k] * alpha[k] * inv_var_e;
            let c = x2_x[k] * inv_var_e + inv_var_b_fixed;
            let z_alpha: f64 = rng.sample(StandardNormal);
            let new_alpha = rhs / c + (1.0 / c).sqrt() * z_alpha;

            let delta = alpha[k] - new_alpha;
            for i in 0..n {
                r[i] += delta * x[i * q + k];
            }
            alpha[k] = new_alpha;
        }

        copy_f64_to_f32(&r, &mut marker_residual);
        let log_odds_prior = (prob_in / (1.0 - prob_in)).ln();

        for j in 0..p {
            let m_j = &m[j * n..(j + 1) * n];
            let b_old = beta[j];
            let c = x2[j] * inv_var_e + 1.0 / var_b[j];
            if !(c.is_finite() && c > 0.0) {
                return Err("Non-positive posterior precision in BayesB beta update".to_string());
            }
            let mut new_d = 0u8;
            let (_, new_beta) =
                bayes_marker_update_f32(&mut marker_residual, m_j, b_old, x2[j], |xe| {
                    let rhs = xe * inv_var_e;
                    // Collapsed d_j sampler: integrate out beta_j when evaluating d_j.
                    let log_bf10 = 0.5 * rhs * rhs / c - 0.5 * (var_b[j] * c).ln();
                    let log_odds = log_odds_prior + log_bf10;
                    let p_in = if log_odds >= 0.0 {
                        1.0 / (1.0 + (-log_odds).exp())
                    } else {
                        let e = log_odds.exp();
                        e / (1.0 + e)
                    };
                    new_d = if rng.random::<f64>() < p_in { 1u8 } else { 0u8 };
                    if new_d == 1 {
                        let z_beta: f64 = rng.sample(StandardNormal);
                        rhs / c + (1.0 / c).sqrt() * z_beta
                    } else {
                        0.0
                    }
                });
            d[j] = new_d;
            beta[j] = new_beta;
        }

        copy_f32_to_f64(&marker_residual, &mut r);
        let mut n_active = 0usize;
        for j in 0..p {
            if d[j] == 1 {
                let beta2 = beta[j] * beta[j];
                var_b[j] = bayes_positive_floor((s + beta2) / rng.sample(chi_b_active));
                if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                    return Err("BayesB var_b became non-finite or non-positive".to_string());
                }
                n_active += 1;
            } else {
                var_b[j] = bayes_positive_floor(s / rng.sample(chi_b_inactive));
                if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                    return Err(
                        "BayesB inactive var_b became non-finite or non-positive".to_string()
                    );
                }
            }
        }

        let mut tmp_rate = 0.0;
        for vb in &var_b {
            tmp_rate += 1.0 / *vb;
        }
        tmp_rate = tmp_rate / 2.0 + rate0;
        if tmp_rate <= 0.0 {
            return Err("Gamma rate became non-positive while updating BayesB S".to_string());
        }
        let tmp_shape = p as f64 * df0_b / 2.0 + shape0;
        s = bayes_positive_floor(sample_gamma_with_rate(&mut rng, tmp_shape, tmp_rate)?);
        if !(s.is_finite() && s > 0.0) {
            return Err("BayesB S became non-finite or non-positive".to_string());
        }

        let mrk_in = n_active as f64;
        let a = mrk_in + counts_in + 1.0;
        let b = (p as f64 - mrk_in) + counts_out + 1.0;
        if fixed_prob_in_opt.is_none() {
            let beta_dist = Beta::new(a, b).map_err(|e| e.to_string())?;
            prob_in = rng.sample(beta_dist);
        }

        let ss_e = ddot_f64(&r, &r) + prior_ss_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesB var_e became non-finite or non-positive".to_string());
        }

        if retain_sample {
            for j in 0..p {
                // Posterior marker effect should be E[d_j * beta_j], not E[beta_j].
                beta_sum[j] += (d[j] as f64) * beta[j];
                pip_sum[j] += d[j] as f64;
                varb_sum[j] += var_b[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            prob_in_sum += prob_in;
            n_active_sum += mrk_in;
            if schedule.observe(h2) {
                beta_sum.fill(0.0);
                pip_sum.fill(0.0);
                varb_sum.fill(0.0);
                alpha_sum.fill(0.0);
                var_e_sum = 0.0;
                h2_sum = 0.0;
                h2_sq_sum = 0.0;
                prob_in_sum = 0.0;
                n_active_sum = 0.0;
            }
        }
    }

    let n_keep = schedule.posterior_samples();
    if n_keep == 0 {
        return Err("No posterior samples kept after the R-hat monitoring phase".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for j in 0..p {
        beta_sum[j] *= inv_keep;
        pip_sum[j] *= inv_keep;
        varb_sum[j] *= inv_keep;
    }
    for k in 0..q {
        alpha_sum[k] *= inv_keep;
    }
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    let prob_in_mean = prob_in_sum * inv_keep;
    let n_active_mean = n_active_sum * inv_keep;
    let rhat_h2 = schedule.rhat_value();
    let actual_iterations = schedule.actual_iterations;
    let convergence_iteration = schedule.convergence_iteration;
    let posterior_samples = schedule.posterior_samples();

    Ok((
        beta_sum,
        alpha_sum,
        varb_sum,
        var_e_sum,
        h2_mean,
        var_h2,
        prob_in_mean,
        n_active_mean,
        pip_sum,
        rhat_h2,
        actual_iterations,
        convergence_iteration,
        posterior_samples,
    ))
}

fn bayesc_core_impl(
    y: &[f64],
    m: &[f32],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    fixed_prob_in_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    seed: Option<u64>,
) -> Result<
    (
        Vec<f64>,
        Vec<f64>,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        Vec<f64>,
        f64,
        usize,
        usize,
        usize,
    ),
    String,
> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    let prob_in_base = fixed_prob_in_opt.unwrap_or(prob_in_init);
    if !(prob_in_base > 0.0 && prob_in_base < 1.0) {
        return Err("prob_in must be in (0, 1)".to_string());
    }
    if counts < 0.0 {
        return Err("counts must be >= 0".to_string());
    }

    if m.len() != n * p {
        return Err("M has incompatible dimensions".to_string());
    }
    if x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32]; // 生成 32 字节随机种子（适配 StdRng）
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes) // 用随机种子初始化 StdRng（无 Result，直接返回）
        }
    };

    let mut x2 = vec![0.0_f64; p];
    let mut mean_x = vec![0.0_f64; p];
    for j in 0..p {
        let mut s = 0.0;
        let mut msum = 0.0;
        for i in 0..n {
            let v = m[j * n + i];
            s += (v as f64) * (v as f64);
            msum += v as f64;
        }
        x2[j] = s;
        mean_x[j] = msum / n_f;
    }
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let mut y_mean = 0.0;
    for v in y {
        y_mean += *v;
    }
    y_mean /= n_f;
    let mut var_y = 0.0;
    for v in y {
        let d = *v - y_mean;
        var_y += d * d;
    }
    var_y /= n_f - 1.0;

    let s0_b = match s0_b_opt {
        Some(v) => v,
        None => {
            if msx <= 0.0 {
                return Err("MSx must be positive to compute S0_b".to_string());
            }
            var_y * r2 / msx * (df0_b + 2.0) / prob_in_base
        }
    };
    if s0_b <= 0.0 {
        return Err("S0_b must be positive".to_string());
    }

    let mut var_e = var_y * (1.0 - r2);
    if var_e <= 0.0 {
        return Err("varE must be positive; check R2".to_string());
    }

    let prior_ss_e = match prior_ss_e_opt {
        Some(v) => v,
        None => var_e * (df0_e + 2.0),
    };
    if prior_ss_e <= 0.0 {
        return Err("prior_ss_e must be positive".to_string());
    }

    let mut beta = vec![0.0; p];
    let mut d = vec![0u8; p];
    let mut var_b = s0_b;
    let mut prob_in = prob_in_base;

    let counts_in = counts * prob_in_base;
    let counts_out = counts - counts_in;

    let mut alpha = vec![0.0; q];
    let mut x2_x = vec![0.0; q];
    for k in 0..q {
        let mut s2 = 0.0;
        for i in 0..n {
            let v = x[i * q + k];
            s2 += v * v;
        }
        x2_x[k] = s2;
    }

    let mut r = y.to_vec();
    let mut marker_residual = vec![0.0_f32; n];
    let mut beta_sum = vec![0.0; p];
    let mut pip_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut varb_sum = 0.0;
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut prob_in_sum = 0.0;
    let mut n_active_sum = 0.0;
    let mut schedule = BayesSamplingController::new(n_iter, burnin, thin);
    let var_b_fixed = 1e10_f64;

    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;
    let mut chi_b_cache: HashMap<usize, ChiSquared<f64>> = HashMap::new();

    while schedule.should_run() {
        let retain_sample = schedule.begin_iteration();
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        for k in 0..q {
            let mut rhs = 0.0;
            for i in 0..n {
                rhs += x[i * q + k] * r[i];
            }
            rhs = rhs * inv_var_e + x2_x[k] * alpha[k] * inv_var_e;
            let c = x2_x[k] * inv_var_e + inv_var_b_fixed;
            let z_alpha: f64 = rng.sample(StandardNormal);
            let new_alpha = rhs / c + (1.0 / c).sqrt() * z_alpha;

            let delta = alpha[k] - new_alpha;
            for i in 0..n {
                r[i] += delta * x[i * q + k];
            }
            alpha[k] = new_alpha;
        }

        copy_f64_to_f32(&r, &mut marker_residual);
        let log_odds_prior = (prob_in / (1.0 - prob_in)).ln();

        for j in 0..p {
            let m_j = &m[j * n..(j + 1) * n];
            let b_old = beta[j];
            let c = x2[j] * inv_var_e + 1.0 / var_b;
            if !(c.is_finite() && c > 0.0) {
                return Err("Non-positive posterior precision in BayesC beta update".to_string());
            }
            let mut new_d = 0u8;
            let (_, new_beta) =
                bayes_marker_update_f32(&mut marker_residual, m_j, b_old, x2[j], |xe| {
                    let rhs = xe * inv_var_e;
                    // Collapsed d_j sampler: integrate out beta_j when evaluating d_j.
                    let log_bf10 = 0.5 * rhs * rhs / c - 0.5 * (var_b * c).ln();
                    let log_odds = log_odds_prior + log_bf10;
                    let p_in = if log_odds >= 0.0 {
                        1.0 / (1.0 + (-log_odds).exp())
                    } else {
                        let e = log_odds.exp();
                        e / (1.0 + e)
                    };
                    new_d = if rng.random::<f64>() < p_in { 1u8 } else { 0u8 };
                    if new_d == 1 {
                        let z_beta: f64 = rng.sample(StandardNormal);
                        rhs / c + (1.0 / c).sqrt() * z_beta
                    } else {
                        0.0
                    }
                });
            d[j] = new_d;
            beta[j] = new_beta;
        }

        copy_f32_to_f64(&marker_residual, &mut r);
        let mut mrk_in_usize = 0usize;
        let mut ss_b = 0.0;
        for j in 0..p {
            if d[j] == 1 {
                ss_b += beta[j] * beta[j];
                mrk_in_usize += 1;
            }
        }
        ss_b += s0_b;
        let chi_b_eff = if let Some(dist) = chi_b_cache.get(&mrk_in_usize) {
            *dist
        } else {
            let dist = ChiSquared::new(df0_b + mrk_in_usize as f64).map_err(|e| e.to_string())?;
            chi_b_cache.insert(mrk_in_usize, dist);
            dist
        };
        var_b = ss_b / rng.sample(chi_b_eff);
        if !(var_b.is_finite() && var_b > 0.0) {
            return Err("BayesC var_b became non-finite or non-positive".to_string());
        }

        let mrk_in = mrk_in_usize as f64;
        let a = mrk_in + counts_in + 1.0;
        let b = (p as f64 - mrk_in) + counts_out + 1.0;
        if fixed_prob_in_opt.is_none() {
            let beta_dist = Beta::new(a, b).map_err(|e| e.to_string())?;
            prob_in = rng.sample(beta_dist);
        }

        let ss_e = ddot_f64(&r, &r) + prior_ss_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesC var_e became non-finite or non-positive".to_string());
        }

        if retain_sample {
            for j in 0..p {
                // Posterior marker effect should be E[d_j * beta_j], not E[beta_j].
                beta_sum[j] += (d[j] as f64) * beta[j];
                pip_sum[j] += d[j] as f64;
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            varb_sum += var_b;
            var_e_sum += var_e;
            let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            prob_in_sum += prob_in;
            n_active_sum += mrk_in;
            if schedule.observe(h2) {
                beta_sum.fill(0.0);
                pip_sum.fill(0.0);
                alpha_sum.fill(0.0);
                varb_sum = 0.0;
                var_e_sum = 0.0;
                h2_sum = 0.0;
                h2_sq_sum = 0.0;
                prob_in_sum = 0.0;
                n_active_sum = 0.0;
            }
        }
    }

    let n_keep = schedule.posterior_samples();
    if n_keep == 0 {
        return Err("No posterior samples kept after the R-hat monitoring phase".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for j in 0..p {
        beta_sum[j] *= inv_keep;
        pip_sum[j] *= inv_keep;
    }
    for k in 0..q {
        alpha_sum[k] *= inv_keep;
    }
    varb_sum *= inv_keep;
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    let prob_in_mean = prob_in_sum * inv_keep;
    let n_active_mean = n_active_sum * inv_keep;
    let rhat_h2 = schedule.rhat_value();
    let actual_iterations = schedule.actual_iterations;
    let convergence_iteration = schedule.convergence_iteration;
    let posterior_samples = schedule.posterior_samples();

    Ok((
        beta_sum,
        alpha_sum,
        varb_sum,
        var_e_sum,
        h2_mean,
        var_h2,
        prob_in_mean,
        n_active_mean,
        pip_sum,
        rhat_h2,
        actual_iterations,
        convergence_iteration,
        posterior_samples,
    ))
}

fn bayesa_core_impl(
    y: &[f64],
    m: &[f32],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0_opt: Option<f64>,
    s0_b_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    _min_abs_beta: f64,
    seed: Option<u64>,
) -> Result<
    (
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        f64,
        f64,
        f64,
        f64,
        usize,
        usize,
        usize,
    ),
    String,
> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }

    if m.len() != n * p {
        return Err("M has incompatible dimensions".to_string());
    }
    if x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32]; // 生成 32 字节随机种子（适配 StdRng）
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes) // 用随机种子初始化 StdRng（无 Result，直接返回）
        }
    };

    let mut x2 = vec![0.0_f64; p];
    let mut mean_x = vec![0.0_f64; p];
    for j in 0..p {
        let mut s = 0.0;
        let mut msum = 0.0;
        for i in 0..n {
            let v = m[j * n + i];
            s += (v as f64) * (v as f64);
            msum += v as f64;
        }
        x2[j] = s;
        mean_x[j] = msum / n_f;
    }
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let mut y_mean = 0.0;
    for v in y {
        y_mean += *v;
    }
    y_mean /= n_f;
    let mut var_y = 0.0;
    for v in y {
        let d = *v - y_mean;
        var_y += d * d;
    }
    var_y /= n_f - 1.0;

    let s0_b = match s0_b_opt {
        Some(v) => v,
        None => {
            if msx <= 0.0 {
                return Err("MSx must be positive to compute S0_b".to_string());
            }
            var_y * r2 / msx * (df0_b + 2.0)
        }
    };
    if s0_b <= 0.0 {
        return Err("S0_b must be positive".to_string());
    }

    let rate0 = match rate0_opt {
        Some(v) => v,
        None => {
            if shape0 <= 1.0 {
                return Err("shape0 must be > 1 when rate0 is not provided".to_string());
            }
            (shape0 - 1.0) / s0_b
        }
    };
    if rate0 <= 0.0 {
        return Err("rate0 must be positive".to_string());
    }

    let mut var_e = var_y * (1.0 - r2);
    if var_e <= 0.0 {
        return Err("varE must be positive; check R2".to_string());
    }

    let prior_ss_e = match prior_ss_e_opt {
        Some(v) => v,
        None => var_e * (df0_e + 2.0),
    };
    if prior_ss_e <= 0.0 {
        return Err("prior_ss_e must be positive".to_string());
    }

    let mut beta = vec![0.0; p];
    let mut var_b = vec![s0_b / (df0_b + 2.0); p];
    let mut s = s0_b;

    let mut alpha = vec![0.0; q];
    let mut x2_x = vec![0.0; q];
    for k in 0..q {
        let mut s2 = 0.0;
        for i in 0..n {
            let v = x[i * q + k];
            s2 += v * v;
        }
        x2_x[k] = s2;
    }

    let mut r = y.to_vec();
    let mut marker_residual = vec![0.0_f32; n];
    let mut beta_sum = vec![0.0; p];
    let mut varb_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut schedule = BayesSamplingController::new(n_iter, burnin, thin);
    let var_b_fixed = 1e10_f64;

    let chi_b = ChiSquared::new(df0_b + 1.0).map_err(|e| e.to_string())?;
    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    while schedule.should_run() {
        let retain_sample = schedule.begin_iteration();
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        for k in 0..q {
            let mut rhs = 0.0;
            for i in 0..n {
                rhs += x[i * q + k] * r[i];
            }
            rhs = rhs * inv_var_e + x2_x[k] * alpha[k] * inv_var_e;
            let c = x2_x[k] * inv_var_e + inv_var_b_fixed;
            let z_alpha: f64 = rng.sample(StandardNormal);
            let new_alpha = rhs / c + (1.0 / c).sqrt() * z_alpha;

            let delta = alpha[k] - new_alpha;
            for i in 0..n {
                r[i] += delta * x[i * q + k];
            }
            alpha[k] = new_alpha;
        }

        copy_f64_to_f32(&r, &mut marker_residual);
        for j in 0..p {
            let m_j = &m[j * n..(j + 1) * n];
            let c = x2[j] * inv_var_e + 1.0 / var_b[j];
            let z_beta: f64 = rng.sample(StandardNormal);
            let old_beta = beta[j];
            let (_, new_beta) =
                bayes_marker_update_f32(&mut marker_residual, m_j, old_beta, x2[j], |u| {
                    u * inv_var_e / c + (1.0 / c).sqrt() * z_beta
                });
            beta[j] = new_beta;
        }

        copy_f32_to_f64(&marker_residual, &mut r);
        for j in 0..p {
            var_b[j] = (s + beta[j] * beta[j]) / rng.sample(chi_b);
            if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                return Err("BayesA var_b became non-finite or non-positive".to_string());
            }
        }

        let mut tmp_rate = 0.0;
        for j in 0..p {
            tmp_rate += 1.0 / var_b[j];
        }
        tmp_rate = tmp_rate / 2.0 + rate0;
        if tmp_rate <= 0.0 {
            return Err("Gamma rate became non-positive while updating S".to_string());
        }
        let tmp_shape = p as f64 * df0_b / 2.0 + shape0;
        let gamma = Gamma::new(tmp_shape, 1.0 / tmp_rate).map_err(|e| e.to_string())?;
        s = rng.sample(gamma);
        if !(s.is_finite() && s > 0.0) {
            return Err("BayesA S became non-finite or non-positive".to_string());
        }

        let ss_e = ddot_f64(&r, &r) + prior_ss_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesA var_e became non-finite or non-positive".to_string());
        }

        if retain_sample {
            for j in 0..p {
                beta_sum[j] += beta[j];
                varb_sum[j] += var_b[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            if schedule.observe(h2) {
                beta_sum.fill(0.0);
                varb_sum.fill(0.0);
                alpha_sum.fill(0.0);
                var_e_sum = 0.0;
                h2_sum = 0.0;
                h2_sq_sum = 0.0;
            }
        }
    }

    let n_keep = schedule.posterior_samples();
    if n_keep == 0 {
        return Err("No posterior samples kept after the R-hat monitoring phase".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for j in 0..p {
        beta_sum[j] *= inv_keep;
        varb_sum[j] *= inv_keep;
    }
    for k in 0..q {
        alpha_sum[k] *= inv_keep;
    }
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    let rhat_h2 = schedule.rhat_value();
    let actual_iterations = schedule.actual_iterations;
    let convergence_iteration = schedule.convergence_iteration;
    let posterior_samples = schedule.posterior_samples();
    Ok((
        beta_sum,
        alpha_sum,
        varb_sum,
        var_e_sum,
        h2_mean,
        var_h2,
        rhat_h2,
        actual_iterations,
        convergence_iteration,
        posterior_samples,
    ))
}

fn bayesa_packed_core_impl(
    y: &[f64],
    source: &mut BayesPackedSource<'_>,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    packed_row_indices: Option<&[usize]>,
    sample_idx: &[usize],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0_opt: Option<f64>,
    s0_b_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    _min_abs_beta: f64,
    seed: Option<u64>,
    backend_block_rows: Option<usize>,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<
    (
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        f64,
        f64,
        f64,
        f64,
        usize,
        usize,
        usize,
    ),
    String,
> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    if y.len() != n {
        return Err("y length mismatch with sample_indices".to_string());
    }
    if x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }
    if row_flip.len() != p || row_maf.len() != p || row_mean.len() != p || row_inv_sd.len() != p {
        return Err("row metadata length mismatch with packed rows".to_string());
    }

    let mut backend = PackedBayesBackend::new(
        source,
        n_samples,
        p,
        row_flip,
        row_mean,
        row_inv_sd,
        packed_row_indices,
        sample_idx,
        backend_block_rows,
        pool,
    )?;
    if let Some(m_dense) = backend.maybe_predecode_dense_f32()? {
        return bayesa_core_impl(
            y,
            &m_dense,
            x,
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            shape0,
            rate0_opt,
            s0_b_opt,
            df0_e,
            prior_ss_e_opt,
            0.0,
            seed,
        );
    }
    let _blas_guard = OpenBlasThreadGuard::enter(bayes_packed_blas_threads());

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32];
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes)
        }
    };

    let mut x2 = vec![0.0_f64; p];
    let mut mean_x = vec![0.0_f64; p];
    backend.for_each_block(|st, ed, m_block| {
        let br = ed - st;
        for off in 0..br {
            let row = &m_block[off * n..(off + 1) * n];
            let mut s = 0.0;
            let mut msum = 0.0;
            for &v in row {
                s += (v as f64) * (v as f64);
                msum += v as f64;
            }
            x2[st + off] = s;
            mean_x[st + off] = msum / n_f;
        }
        Ok(())
    })?;
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let mut y_mean = 0.0;
    for v in y {
        y_mean += *v;
    }
    y_mean /= n_f;
    let mut var_y = 0.0;
    for v in y {
        let d = *v - y_mean;
        var_y += d * d;
    }
    var_y /= n_f - 1.0;

    let s0_b = match s0_b_opt {
        Some(v) => v,
        None => {
            if msx <= 0.0 {
                return Err("MSx must be positive to compute S0_b".to_string());
            }
            var_y * r2 / msx * (df0_b + 2.0)
        }
    };
    if s0_b <= 0.0 {
        return Err("S0_b must be positive".to_string());
    }

    let rate0 = match rate0_opt {
        Some(v) => v,
        None => {
            if shape0 <= 1.0 {
                return Err("shape0 must be > 1 when rate0 is not provided".to_string());
            }
            (shape0 - 1.0) / s0_b
        }
    };
    if rate0 <= 0.0 {
        return Err("rate0 must be positive".to_string());
    }

    let mut var_e = var_y * (1.0 - r2);
    if var_e <= 0.0 {
        return Err("varE must be positive; check R2".to_string());
    }

    let prior_ss_e = match prior_ss_e_opt {
        Some(v) => v,
        None => var_e * (df0_e + 2.0),
    };
    if prior_ss_e <= 0.0 {
        return Err("prior_ss_e must be positive".to_string());
    }

    let mut beta = vec![0.0; p];
    let mut var_b = vec![s0_b / (df0_b + 2.0); p];
    let mut s = s0_b;

    let mut alpha = vec![0.0; q];
    let mut xtx = vec![0.0_f64; q * q];
    row_major_xtx_f64(x, n, q, &mut xtx);
    let mut x2_x = vec![0.0_f64; q];
    for k in 0..q {
        x2_x[k] = xtx[k * q + k];
    }
    let mut alpha_xtr = vec![0.0_f64; q];
    let mut alpha_delta = vec![0.0_f64; q];
    let mut alpha_tmp_n = vec![0.0_f64; n];

    let mut r = y.to_vec();
    let mut marker_residual = vec![0.0_f32; n];
    let mut beta_sum = vec![0.0; p];
    let mut varb_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut schedule = BayesSamplingController::new(n_iter, burnin, thin);
    let var_b_fixed = 1e10_f64;

    let chi_b = ChiSquared::new(df0_b + 1.0).map_err(|e| e.to_string())?;
    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    while schedule.should_run() {
        let retain_sample = schedule.begin_iteration();
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        update_alpha_gauss_seidel_blas(
            x,
            n,
            q,
            inv_var_e,
            inv_var_b_fixed,
            &x2_x,
            &xtx,
            &mut alpha,
            &mut r,
            &mut alpha_xtr,
            &mut alpha_delta,
            &mut alpha_tmp_n,
            &mut rng,
        );

        copy_f64_to_f32(&r, &mut marker_residual);
        backend.for_each_block(|st, ed, m_block| {
            let br = ed - st;
            for off in 0..br {
                let j = st + off;
                let m_row = &m_block[off * n..(off + 1) * n];
                let c = x2[j] * inv_var_e + 1.0 / var_b[j];
                let z_beta: f64 = rng.sample(StandardNormal);
                let old_beta = beta[j];
                let (_, new_beta) =
                    bayes_marker_update_f32(&mut marker_residual, m_row, old_beta, x2[j], |u| {
                        u * inv_var_e / c + (1.0 / c).sqrt() * z_beta
                    });
                beta[j] = new_beta;
            }
            Ok(())
        })?;

        copy_f32_to_f64(&marker_residual, &mut r);
        for j in 0..p {
            var_b[j] = (s + beta[j] * beta[j]) / rng.sample(chi_b);
            if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                return Err("BayesA packed var_b became non-finite or non-positive".to_string());
            }
        }

        let mut tmp_rate = 0.0;
        for j in 0..p {
            tmp_rate += 1.0 / var_b[j];
        }
        tmp_rate = tmp_rate / 2.0 + rate0;
        if tmp_rate <= 0.0 {
            return Err("Gamma rate became non-positive while updating S".to_string());
        }
        let tmp_shape = p as f64 * df0_b / 2.0 + shape0;
        let gamma = Gamma::new(tmp_shape, 1.0 / tmp_rate).map_err(|e| e.to_string())?;
        s = rng.sample(gamma);
        if !(s.is_finite() && s > 0.0) {
            return Err("BayesA packed S became non-finite or non-positive".to_string());
        }

        let ss_e = ddot_f64(&r, &r) + prior_ss_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesA packed var_e became non-finite or non-positive".to_string());
        }

        if retain_sample {
            for j in 0..p {
                beta_sum[j] += beta[j];
                varb_sum[j] += var_b[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            if schedule.observe(h2) {
                beta_sum.fill(0.0);
                varb_sum.fill(0.0);
                alpha_sum.fill(0.0);
                var_e_sum = 0.0;
                h2_sum = 0.0;
                h2_sq_sum = 0.0;
            }
        }
    }

    let n_keep = schedule.posterior_samples();
    if n_keep == 0 {
        return Err("No posterior samples kept after the R-hat monitoring phase".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for j in 0..p {
        beta_sum[j] *= inv_keep;
        varb_sum[j] *= inv_keep;
    }
    for k in 0..q {
        alpha_sum[k] *= inv_keep;
    }
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    let rhat_h2 = schedule.rhat_value();
    let actual_iterations = schedule.actual_iterations;
    let convergence_iteration = schedule.convergence_iteration;
    let posterior_samples = schedule.posterior_samples();
    Ok((
        beta_sum,
        alpha_sum,
        varb_sum,
        var_e_sum,
        h2_mean,
        var_h2,
        rhat_h2,
        actual_iterations,
        convergence_iteration,
        posterior_samples,
    ))
}

fn bayesb_packed_core_impl(
    y: &[f64],
    source: &mut BayesPackedSource<'_>,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    packed_row_indices: Option<&[usize]>,
    sample_idx: &[usize],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0_opt: Option<f64>,
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    fixed_prob_in_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    seed: Option<u64>,
    backend_block_rows: Option<usize>,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<
    (
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        f64,
        f64,
        f64,
        f64,
        f64,
        Vec<f64>,
        f64,
        usize,
        usize,
        usize,
    ),
    String,
> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    let prob_in_base = fixed_prob_in_opt.unwrap_or(prob_in_init);
    if !(prob_in_base > 0.0 && prob_in_base < 1.0) {
        return Err("prob_in must be in (0, 1)".to_string());
    }
    if counts < 0.0 {
        return Err("counts must be >= 0".to_string());
    }
    if y.len() != n {
        return Err("y length mismatch with sample_indices".to_string());
    }
    if x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }
    if row_flip.len() != p || row_maf.len() != p || row_mean.len() != p || row_inv_sd.len() != p {
        return Err("row metadata length mismatch with packed rows".to_string());
    }

    let mut backend = PackedBayesBackend::new(
        source,
        n_samples,
        p,
        row_flip,
        row_mean,
        row_inv_sd,
        packed_row_indices,
        sample_idx,
        backend_block_rows,
        pool,
    )?;
    if let Some(m_dense) = backend.maybe_predecode_dense_f32()? {
        return bayesb_core_impl(
            y,
            &m_dense,
            x,
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            shape0,
            rate0_opt,
            s0_b_opt,
            prob_in_init,
            counts,
            fixed_prob_in_opt,
            df0_e,
            prior_ss_e_opt,
            seed,
        );
    }
    let _blas_guard = OpenBlasThreadGuard::enter(bayes_packed_blas_threads());

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32];
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes)
        }
    };

    let mut x2 = vec![0.0_f64; p];
    let mut mean_x = vec![0.0_f64; p];
    backend.for_each_block(|st, ed, m_block| {
        let br = ed - st;
        for off in 0..br {
            let row = &m_block[off * n..(off + 1) * n];
            let mut s = 0.0;
            let mut msum = 0.0;
            for &v in row {
                s += (v as f64) * (v as f64);
                msum += v as f64;
            }
            x2[st + off] = s;
            mean_x[st + off] = msum / n_f;
        }
        Ok(())
    })?;
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let mut y_mean = 0.0;
    for v in y {
        y_mean += *v;
    }
    y_mean /= n_f;
    let mut var_y = 0.0;
    for v in y {
        let d = *v - y_mean;
        var_y += d * d;
    }
    var_y /= n_f - 1.0;

    let s0_b = match s0_b_opt {
        Some(v) => v,
        None => {
            if msx <= 0.0 {
                return Err("MSx must be positive to compute S0_b".to_string());
            }
            var_y * r2 / msx * (df0_b + 2.0) / prob_in_base
        }
    };
    if s0_b <= 0.0 {
        return Err("S0_b must be positive".to_string());
    }

    let rate0 = bayesb_rate0(shape0, rate0_opt, s0_b)?;

    let mut var_e = var_y * (1.0 - r2);
    if var_e <= 0.0 {
        return Err("varE must be positive; check R2".to_string());
    }

    let prior_ss_e = match prior_ss_e_opt {
        Some(v) => v,
        None => var_e * (df0_e + 2.0),
    };
    if prior_ss_e <= 0.0 {
        return Err("prior_ss_e must be positive".to_string());
    }

    let mut beta = vec![0.0; p];
    let mut d = vec![0u8; p];
    let mut var_b = vec![s0_b / (df0_b + 2.0); p];
    let mut prob_in = prob_in_base;
    let mut s = s0_b;
    let (counts_in, counts_out) = bayes_inclusion_prior_shapes(prob_in_base, counts);

    let mut alpha = vec![0.0; q];
    let mut xtx = vec![0.0_f64; q * q];
    row_major_xtx_f64(x, n, q, &mut xtx);
    let mut x2_x = vec![0.0_f64; q];
    for k in 0..q {
        x2_x[k] = xtx[k * q + k];
    }
    let mut alpha_xtr = vec![0.0_f64; q];
    let mut alpha_delta = vec![0.0_f64; q];
    let mut alpha_tmp_n = vec![0.0_f64; n];

    let mut r = y.to_vec();
    let mut marker_residual = vec![0.0_f32; n];
    let mut beta_sum = vec![0.0; p];
    let mut pip_sum = vec![0.0; p];
    let mut varb_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut prob_in_sum = 0.0;
    let mut n_active_sum = 0.0;
    let mut schedule = BayesSamplingController::new(n_iter, burnin, thin);
    let var_b_fixed = 1e10_f64;

    let chi_b_active = ChiSquared::new(df0_b + 1.0).map_err(|e| e.to_string())?;
    let chi_b_inactive = ChiSquared::new(df0_b).map_err(|e| e.to_string())?;
    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    while schedule.should_run() {
        let retain_sample = schedule.begin_iteration();
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        update_alpha_gauss_seidel_blas(
            x,
            n,
            q,
            inv_var_e,
            inv_var_b_fixed,
            &x2_x,
            &xtx,
            &mut alpha,
            &mut r,
            &mut alpha_xtr,
            &mut alpha_delta,
            &mut alpha_tmp_n,
            &mut rng,
        );

        copy_f64_to_f32(&r, &mut marker_residual);
        let log_odds_prior = (prob_in / (1.0 - prob_in)).ln();

        backend.for_each_block(|st, ed, m_block| {
            let br = ed - st;
            for off in 0..br {
                let j = st + off;
                let m_row = &m_block[off * n..(off + 1) * n];
                let b_old = beta[j];
                let c = x2[j] * inv_var_e + 1.0 / var_b[j];
                if !(c.is_finite() && c > 0.0) {
                    return Err(
                        "Non-positive posterior precision in BayesB packed beta update".to_string(),
                    );
                }
                let mut new_d = 0u8;
                let (_, new_beta) =
                    bayes_marker_update_f32(&mut marker_residual, m_row, b_old, x2[j], |xe| {
                        let rhs = xe * inv_var_e;
                        // Collapsed d_j sampler: integrate out beta_j when evaluating d_j.
                        let log_bf10 = 0.5 * rhs * rhs / c - 0.5 * (var_b[j] * c).ln();
                        let log_odds = log_odds_prior + log_bf10;
                        let p_in = if log_odds >= 0.0 {
                            1.0 / (1.0 + (-log_odds).exp())
                        } else {
                            let e = log_odds.exp();
                            e / (1.0 + e)
                        };
                        new_d = if rng.random::<f64>() < p_in { 1u8 } else { 0u8 };
                        if new_d == 1 {
                            let z_beta: f64 = rng.sample(StandardNormal);
                            rhs / c + (1.0 / c).sqrt() * z_beta
                        } else {
                            0.0
                        }
                    });
                d[j] = new_d;
                beta[j] = new_beta;
            }
            Ok(())
        })?;

        copy_f32_to_f64(&marker_residual, &mut r);
        let mut n_active = 0usize;
        for j in 0..p {
            if d[j] == 1 {
                let beta2 = beta[j] * beta[j];
                var_b[j] = bayes_positive_floor((s + beta2) / rng.sample(chi_b_active));
                if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                    return Err("BayesB packed var_b became non-finite or non-positive".to_string());
                }
                n_active += 1;
            } else {
                var_b[j] = bayes_positive_floor(s / rng.sample(chi_b_inactive));
                if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                    return Err(
                        "BayesB packed inactive var_b became non-finite or non-positive"
                            .to_string(),
                    );
                }
            }
        }

        let mut tmp_rate = 0.0;
        for vb in &var_b {
            tmp_rate += 1.0 / *vb;
        }
        tmp_rate = tmp_rate / 2.0 + rate0;
        if tmp_rate <= 0.0 {
            return Err(
                "Gamma rate became non-positive while updating BayesB packed S".to_string(),
            );
        }
        let tmp_shape = p as f64 * df0_b / 2.0 + shape0;
        s = bayes_positive_floor(sample_gamma_with_rate(&mut rng, tmp_shape, tmp_rate)?);
        if !(s.is_finite() && s > 0.0) {
            return Err("BayesB packed S became non-finite or non-positive".to_string());
        }

        let mrk_in = n_active as f64;
        let a = mrk_in + counts_in + 1.0;
        let b = (p as f64 - mrk_in) + counts_out + 1.0;
        if fixed_prob_in_opt.is_none() {
            let beta_dist = Beta::new(a, b).map_err(|e| e.to_string())?;
            prob_in = rng.sample(beta_dist);
        }

        let ss_e = ddot_f64(&r, &r) + prior_ss_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesB packed var_e became non-finite or non-positive".to_string());
        }

        if retain_sample {
            for j in 0..p {
                // Posterior marker effect should be E[d_j * beta_j], not E[beta_j].
                beta_sum[j] += (d[j] as f64) * beta[j];
                pip_sum[j] += d[j] as f64;
                varb_sum[j] += var_b[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            prob_in_sum += prob_in;
            n_active_sum += mrk_in;
            if schedule.observe(h2) {
                beta_sum.fill(0.0);
                pip_sum.fill(0.0);
                varb_sum.fill(0.0);
                alpha_sum.fill(0.0);
                var_e_sum = 0.0;
                h2_sum = 0.0;
                h2_sq_sum = 0.0;
                prob_in_sum = 0.0;
                n_active_sum = 0.0;
            }
        }
    }

    let n_keep = schedule.posterior_samples();
    if n_keep == 0 {
        return Err("No posterior samples kept after the R-hat monitoring phase".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for j in 0..p {
        beta_sum[j] *= inv_keep;
        pip_sum[j] *= inv_keep;
        varb_sum[j] *= inv_keep;
    }
    for k in 0..q {
        alpha_sum[k] *= inv_keep;
    }
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    let prob_in_mean = prob_in_sum * inv_keep;
    let n_active_mean = n_active_sum * inv_keep;
    let rhat_h2 = schedule.rhat_value();
    let actual_iterations = schedule.actual_iterations;
    let convergence_iteration = schedule.convergence_iteration;
    let posterior_samples = schedule.posterior_samples();

    Ok((
        beta_sum,
        alpha_sum,
        varb_sum,
        var_e_sum,
        h2_mean,
        var_h2,
        prob_in_mean,
        n_active_mean,
        pip_sum,
        rhat_h2,
        actual_iterations,
        convergence_iteration,
        posterior_samples,
    ))
}

fn bayesc_packed_core_impl(
    y: &[f64],
    source: &mut BayesPackedSource<'_>,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    packed_row_indices: Option<&[usize]>,
    sample_idx: &[usize],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    fixed_prob_in_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    seed: Option<u64>,
    backend_block_rows: Option<usize>,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<
    (
        Vec<f64>,
        Vec<f64>,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        Vec<f64>,
        f64,
        usize,
        usize,
        usize,
    ),
    String,
> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    let prob_in_base = fixed_prob_in_opt.unwrap_or(prob_in_init);
    if !(prob_in_base > 0.0 && prob_in_base < 1.0) {
        return Err("prob_in must be in (0, 1)".to_string());
    }
    if counts < 0.0 {
        return Err("counts must be >= 0".to_string());
    }
    if y.len() != n {
        return Err("y length mismatch with sample_indices".to_string());
    }
    if x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }
    if row_flip.len() != p || row_maf.len() != p || row_mean.len() != p || row_inv_sd.len() != p {
        return Err("row metadata length mismatch with packed rows".to_string());
    }

    let mut backend = PackedBayesBackend::new(
        source,
        n_samples,
        p,
        row_flip,
        row_mean,
        row_inv_sd,
        packed_row_indices,
        sample_idx,
        backend_block_rows,
        pool,
    )?;
    if let Some(m_dense) = backend.maybe_predecode_dense_f32()? {
        return bayesc_core_impl(
            y,
            &m_dense,
            x,
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            s0_b_opt,
            prob_in_init,
            counts,
            fixed_prob_in_opt,
            df0_e,
            prior_ss_e_opt,
            seed,
        );
    }
    let _blas_guard = OpenBlasThreadGuard::enter(bayes_packed_blas_threads());

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32];
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes)
        }
    };

    let mut x2 = vec![0.0_f64; p];
    let mut mean_x = vec![0.0_f64; p];
    backend.for_each_block(|st, ed, m_block| {
        let br = ed - st;
        for off in 0..br {
            let row = &m_block[off * n..(off + 1) * n];
            let mut s = 0.0;
            let mut msum = 0.0;
            for &v in row {
                s += (v as f64) * (v as f64);
                msum += v as f64;
            }
            x2[st + off] = s;
            mean_x[st + off] = msum / n_f;
        }
        Ok(())
    })?;
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let mut y_mean = 0.0;
    for v in y {
        y_mean += *v;
    }
    y_mean /= n_f;
    let mut var_y = 0.0;
    for v in y {
        let d = *v - y_mean;
        var_y += d * d;
    }
    var_y /= n_f - 1.0;

    let s0_b = match s0_b_opt {
        Some(v) => v,
        None => {
            if msx <= 0.0 {
                return Err("MSx must be positive to compute S0_b".to_string());
            }
            var_y * r2 / msx * (df0_b + 2.0) / prob_in_base
        }
    };
    if s0_b <= 0.0 {
        return Err("S0_b must be positive".to_string());
    }

    let mut var_e = var_y * (1.0 - r2);
    if var_e <= 0.0 {
        return Err("varE must be positive; check R2".to_string());
    }

    let prior_ss_e = match prior_ss_e_opt {
        Some(v) => v,
        None => var_e * (df0_e + 2.0),
    };
    if prior_ss_e <= 0.0 {
        return Err("prior_ss_e must be positive".to_string());
    }

    let mut beta = vec![0.0; p];
    let mut d = vec![0u8; p];
    let mut var_b = s0_b;
    let mut prob_in = prob_in_base;
    let counts_in = counts * prob_in_base;
    let counts_out = counts - counts_in;

    let mut alpha = vec![0.0; q];
    let mut xtx = vec![0.0_f64; q * q];
    row_major_xtx_f64(x, n, q, &mut xtx);
    let mut x2_x = vec![0.0_f64; q];
    for k in 0..q {
        x2_x[k] = xtx[k * q + k];
    }
    let mut alpha_xtr = vec![0.0_f64; q];
    let mut alpha_delta = vec![0.0_f64; q];
    let mut alpha_tmp_n = vec![0.0_f64; n];

    let mut r = y.to_vec();
    let mut marker_residual = vec![0.0_f32; n];
    let mut beta_sum = vec![0.0; p];
    let mut pip_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut varb_sum = 0.0;
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut prob_in_sum = 0.0;
    let mut n_active_sum = 0.0;
    let mut schedule = BayesSamplingController::new(n_iter, burnin, thin);
    let var_b_fixed = 1e10_f64;

    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;
    let mut chi_b_cache: HashMap<usize, ChiSquared<f64>> = HashMap::new();

    while schedule.should_run() {
        let retain_sample = schedule.begin_iteration();
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        update_alpha_gauss_seidel_blas(
            x,
            n,
            q,
            inv_var_e,
            inv_var_b_fixed,
            &x2_x,
            &xtx,
            &mut alpha,
            &mut r,
            &mut alpha_xtr,
            &mut alpha_delta,
            &mut alpha_tmp_n,
            &mut rng,
        );

        copy_f64_to_f32(&r, &mut marker_residual);
        let log_odds_prior = (prob_in / (1.0 - prob_in)).ln();

        backend.for_each_block(|st, ed, m_block| {
            let br = ed - st;
            for off in 0..br {
                let j = st + off;
                let m_row = &m_block[off * n..(off + 1) * n];
                let b_old = beta[j];
                let c = x2[j] * inv_var_e + 1.0 / var_b;
                if !(c.is_finite() && c > 0.0) {
                    return Err(
                        "Non-positive posterior precision in BayesC packed beta update".to_string(),
                    );
                }
                let mut new_d = 0u8;
                let (_, new_beta) =
                    bayes_marker_update_f32(&mut marker_residual, m_row, b_old, x2[j], |xe| {
                        let rhs = xe * inv_var_e;
                        // Collapsed d_j sampler: integrate out beta_j when evaluating d_j.
                        let log_bf10 = 0.5 * rhs * rhs / c - 0.5 * (var_b * c).ln();
                        let log_odds = log_odds_prior + log_bf10;
                        let p_in = if log_odds >= 0.0 {
                            1.0 / (1.0 + (-log_odds).exp())
                        } else {
                            let e = log_odds.exp();
                            e / (1.0 + e)
                        };
                        new_d = if rng.random::<f64>() < p_in { 1u8 } else { 0u8 };
                        if new_d == 1 {
                            let z_beta: f64 = rng.sample(StandardNormal);
                            rhs / c + (1.0 / c).sqrt() * z_beta
                        } else {
                            0.0
                        }
                    });
                d[j] = new_d;
                beta[j] = new_beta;
            }
            Ok(())
        })?;

        copy_f32_to_f64(&marker_residual, &mut r);
        let mut mrk_in_usize = 0usize;
        let mut ss_b = 0.0;
        for j in 0..p {
            if d[j] == 1 {
                ss_b += beta[j] * beta[j];
                mrk_in_usize += 1;
            }
        }
        ss_b += s0_b;
        let chi_b_eff = if let Some(dist) = chi_b_cache.get(&mrk_in_usize) {
            *dist
        } else {
            let dist = ChiSquared::new(df0_b + mrk_in_usize as f64).map_err(|e| e.to_string())?;
            chi_b_cache.insert(mrk_in_usize, dist);
            dist
        };
        var_b = ss_b / rng.sample(chi_b_eff);
        if !(var_b.is_finite() && var_b > 0.0) {
            return Err("BayesC packed var_b became non-finite or non-positive".to_string());
        }

        let mrk_in = mrk_in_usize as f64;
        let a = mrk_in + counts_in + 1.0;
        let b = (p as f64 - mrk_in) + counts_out + 1.0;
        if fixed_prob_in_opt.is_none() {
            let beta_dist = Beta::new(a, b).map_err(|e| e.to_string())?;
            prob_in = rng.sample(beta_dist);
        }

        let ss_e = ddot_f64(&r, &r) + prior_ss_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesC packed var_e became non-finite or non-positive".to_string());
        }

        if retain_sample {
            for j in 0..p {
                // Posterior marker effect should be E[d_j * beta_j], not E[beta_j].
                beta_sum[j] += (d[j] as f64) * beta[j];
                pip_sum[j] += d[j] as f64;
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            varb_sum += var_b;
            var_e_sum += var_e;
            let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            prob_in_sum += prob_in;
            n_active_sum += mrk_in;
            if schedule.observe(h2) {
                beta_sum.fill(0.0);
                pip_sum.fill(0.0);
                alpha_sum.fill(0.0);
                varb_sum = 0.0;
                var_e_sum = 0.0;
                h2_sum = 0.0;
                h2_sq_sum = 0.0;
                prob_in_sum = 0.0;
                n_active_sum = 0.0;
            }
        }
    }

    let n_keep = schedule.posterior_samples();
    if n_keep == 0 {
        return Err("No posterior samples kept after the R-hat monitoring phase".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for j in 0..p {
        beta_sum[j] *= inv_keep;
        pip_sum[j] *= inv_keep;
    }
    for k in 0..q {
        alpha_sum[k] *= inv_keep;
    }
    varb_sum *= inv_keep;
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    let prob_in_mean = prob_in_sum * inv_keep;
    let n_active_mean = n_active_sum * inv_keep;
    let rhat_h2 = schedule.rhat_value();
    let actual_iterations = schedule.actual_iterations;
    let convergence_iteration = schedule.convergence_iteration;
    let posterior_samples = schedule.posterior_samples();

    Ok((
        beta_sum,
        alpha_sum,
        varb_sum,
        var_e_sum,
        h2_mean,
        var_h2,
        prob_in_mean,
        n_active_mean,
        pip_sum,
        rhat_h2,
        actual_iterations,
        convergence_iteration,
        posterior_samples,
    ))
}

#[pyfunction]
#[pyo3(signature = (
    y,
    m,
    x = None,
    n_iter = 3000,
    burnin = 1000,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    shape0 = 1.1,
    rate0 = None,
    s0_b = None,
    df0_e = 5.0,
    prior_ss_e = None,
    min_abs_beta = 1e-9,
    seed = None
))]
pub fn bayesa(
    py: Python,
    y: PyReadonlyArray1<f64>,
    m: Bound<'_, PyAny>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0: Option<f64>,
    s0_b: Option<f64>,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    min_abs_beta: f64,
    seed: Option<u64>,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    f64,
    f64,
    usize,
    usize,
    usize,
)> {
    if n_iter == 0 {
        return Err(PyValueError::new_err("n_iter must be > 0"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !min_abs_beta.is_finite() || min_abs_beta < 0.0 {
        return Err(PyValueError::new_err(
            "min_abs_beta is deprecated/ignored; keep it finite and >= 0 for compatibility",
        ));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if shape0 <= 0.0 {
        return Err(PyValueError::new_err("shape0 must be > 0"));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let m_input = array2_to_f32_input(&m, "M")?;
    if m_input.cols() != n {
        return Err(PyValueError::new_err("M cols must match len(y)"));
    }
    let p = m_input.rows();
    let m_slice = m_input.as_slice();

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let result = py.detach(|| {
        bayesa_core_impl(
            y_vec.as_ref(),
            m_slice,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            shape0,
            rate0,
            s0_b,
            df0_e,
            prior_ss_e,
            min_abs_beta,
            seed,
        )
    });

    match result {
        Ok((
            beta,
            alpha,
            varb,
            vare,
            h2_mean,
            var_h2,
            rhat_h2,
            actual_iterations,
            convergence_iteration,
            posterior_samples,
        )) => {
            let beta_py = beta.into_pyarray(py).into_bound().unbind();
            let alpha_py = alpha.into_pyarray(py).into_bound().unbind();
            let varb_py = varb.into_pyarray(py).into_bound().unbind();
            Ok((
                beta_py,
                alpha_py,
                varb_py,
                vare,
                h2_mean,
                var_h2,
                rhat_h2,
                actual_iterations,
                convergence_iteration,
                posterior_samples,
            ))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y,
    m,
    x = None,
    n_iter = 3000,
    burnin = 1000,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    shape0 = 1.1,
    rate0 = None,
    s0_b = None,
    prob_in = 0.5,
    counts = 10.0,
    fixed_pi = None,
    df0_e = 5.0,
    prior_ss_e = None,
    seed = None
))]
pub fn bayesb(
    py: Python,
    y: PyReadonlyArray1<f64>,
    m: Bound<'_, PyAny>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0: Option<f64>,
    s0_b: Option<f64>,
    prob_in: f64,
    counts: f64,
    fixed_pi: Option<f64>,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    seed: Option<u64>,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    f64,
    f64,
    f64,
    Py<PyArray1<f64>>,
    f64,
    usize,
    usize,
)> {
    if n_iter == 0 {
        return Err(PyValueError::new_err("n_iter must be > 0"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if shape0 <= 0.0 {
        return Err(PyValueError::new_err("shape0 must be > 0"));
    }
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let m_input = array2_to_f32_input(&m, "M")?;
    if m_input.cols() != n {
        return Err(PyValueError::new_err("M cols must match len(y)"));
    }
    let p = m_input.rows();
    let m_slice = m_input.as_slice();

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let result = py.detach(|| {
        bayesb_core_impl(
            y_vec.as_ref(),
            m_slice,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            shape0,
            rate0,
            s0_b,
            prob_in,
            counts,
            fixed_pi,
            df0_e,
            prior_ss_e,
            seed,
        )
    });

    match result {
        Ok((
            beta,
            alpha,
            varb,
            vare,
            h2_mean,
            var_h2,
            prob_in_mean,
            n_active_mean,
            pip,
            rhat_h2,
            actual_iterations,
            convergence_iteration,
            _posterior_samples,
        )) => {
            let beta_py = beta.into_pyarray(py).into_bound().unbind();
            let alpha_py = alpha.into_pyarray(py).into_bound().unbind();
            let varb_py = varb.into_pyarray(py).into_bound().unbind();
            let pip_py = pip.into_pyarray(py).into_bound().unbind();
            Ok((
                beta_py,
                alpha_py,
                varb_py,
                vare,
                h2_mean,
                var_h2,
                prob_in_mean,
                n_active_mean,
                pip_py,
                rhat_h2,
                actual_iterations,
                convergence_iteration,
            ))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y,
    m,
    x = None,
    n_iter = 3000,
    burnin = 1000,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    s0_b = None,
    prob_in = 0.5,
    counts = 10.0,
    fixed_pi = None,
    df0_e = 5.0,
    prior_ss_e = None,
    seed = None
))]
pub fn bayesc(
    py: Python,
    y: PyReadonlyArray1<f64>,
    m: Bound<'_, PyAny>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b: Option<f64>,
    prob_in: f64,
    counts: f64,
    fixed_pi: Option<f64>,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    seed: Option<u64>,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    Py<PyArray1<f64>>,
    f64,
    usize,
    usize,
)> {
    if n_iter == 0 {
        return Err(PyValueError::new_err("n_iter must be > 0"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }
    if let Some(v) = s0_b {
        if v <= 0.0 {
            return Err(PyValueError::new_err("s0_b must be > 0"));
        }
    }
    if let Some(v) = prior_ss_e {
        if v <= 0.0 {
            return Err(PyValueError::new_err("prior_ss_e must be > 0"));
        }
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let m_input = array2_to_f32_input(&m, "M")?;
    if m_input.cols() != n {
        return Err(PyValueError::new_err("M cols must match len(y)"));
    }
    let p = m_input.rows();
    let m_slice = m_input.as_slice();

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let result = py.detach(|| {
        bayesc_core_impl(
            y_vec.as_ref(),
            m_slice,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            s0_b,
            prob_in,
            counts,
            fixed_pi,
            df0_e,
            prior_ss_e,
            seed,
        )
    });

    match result {
        Ok((
            beta,
            alpha,
            varb_mean,
            vare,
            h2_mean,
            var_h2,
            prob_in_mean,
            n_active_mean,
            pip,
            rhat_h2,
            actual_iterations,
            convergence_iteration,
            _posterior_samples,
        )) => {
            let beta_py = beta.into_pyarray(py).into_bound().unbind();
            let alpha_py = alpha.into_pyarray(py).into_bound().unbind();
            let pip_py = pip.into_pyarray(py).into_bound().unbind();
            Ok((
                beta_py,
                alpha_py,
                varb_mean,
                vare,
                h2_mean,
                var_h2,
                prob_in_mean,
                n_active_mean,
                pip_py,
                rhat_h2,
                actual_iterations,
                convergence_iteration,
            ))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y,
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 3000,
    burnin = 1000,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    shape0 = 1.1,
    rate0 = None,
    s0_b = None,
    df0_e = 5.0,
    prior_ss_e = None,
    min_abs_beta = 1e-9,
    threads = 0,
    seed = None,
    block_rows = None
))]
pub fn bayesa_packed(
    py: Python,
    y: PyReadonlyArray1<f64>,
    packed: PyReadonlyArray2<u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0: Option<f64>,
    s0_b: Option<f64>,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    min_abs_beta: f64,
    threads: usize,
    seed: Option<u64>,
    block_rows: Option<usize>,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    f64,
    f64,
    usize,
    usize,
    usize,
)> {
    if n_iter == 0 {
        return Err(PyValueError::new_err("n_iter must be > 0"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !min_abs_beta.is_finite() || min_abs_beta < 0.0 {
        return Err(PyValueError::new_err(
            "min_abs_beta is deprecated/ignored; keep it finite and >= 0 for compatibility",
        ));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if shape0 <= 0.0 {
        return Err(PyValueError::new_err("shape0 must be > 0"));
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyValueError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let p = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyValueError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<'_, [bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<'_, [f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    let row_mean_vec: Cow<'_, [f32]> = match row_mean.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_mean.as_array().iter().copied().collect()),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> = match row_inv_sd.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_inv_sd.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match packed rows",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let packed_flat: Cow<'_, [u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();

    let result = py.detach(|| {
        let mut source = BayesPackedSource::Resident {
            packed_flat: packed_flat.as_ref(),
            bytes_per_snp,
        };
        bayesa_packed_core_impl(
            y_vec.as_ref(),
            &mut source,
            n_samples,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            None,
            &sample_idx,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            shape0,
            rate0,
            s0_b,
            df0_e,
            prior_ss_e,
            min_abs_beta,
            seed,
            block_rows,
            pool_ref,
        )
    });

    match result {
        Ok((
            beta,
            alpha,
            varb,
            vare,
            h2_mean,
            var_h2,
            rhat_h2,
            actual_iterations,
            convergence_iteration,
            posterior_samples,
        )) => {
            let beta_py = beta.into_pyarray(py).into_bound().unbind();
            let alpha_py = alpha.into_pyarray(py).into_bound().unbind();
            let varb_py = varb.into_pyarray(py).into_bound().unbind();
            Ok((
                beta_py,
                alpha_py,
                varb_py,
                vare,
                h2_mean,
                var_h2,
                rhat_h2,
                actual_iterations,
                convergence_iteration,
                posterior_samples,
            ))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y,
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 3000,
    burnin = 1000,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    shape0 = 1.1,
    rate0 = None,
    s0_b = None,
    prob_in = 0.5,
    counts = 10.0,
    fixed_pi = None,
    df0_e = 5.0,
    prior_ss_e = None,
    threads = 0,
    seed = None,
    block_rows = None
))]
pub fn bayesb_packed(
    py: Python,
    y: PyReadonlyArray1<f64>,
    packed: PyReadonlyArray2<u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0: Option<f64>,
    s0_b: Option<f64>,
    prob_in: f64,
    counts: f64,
    fixed_pi: Option<f64>,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    threads: usize,
    seed: Option<u64>,
    block_rows: Option<usize>,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    f64,
    f64,
    f64,
    Py<PyArray1<f64>>,
    f64,
    usize,
    usize,
)> {
    if n_iter == 0 {
        return Err(PyValueError::new_err("n_iter must be > 0"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if shape0 <= 0.0 {
        return Err(PyValueError::new_err("shape0 must be > 0"));
    }
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyValueError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let p = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyValueError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<'_, [bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<'_, [f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    let row_mean_vec: Cow<'_, [f32]> = match row_mean.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_mean.as_array().iter().copied().collect()),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> = match row_inv_sd.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_inv_sd.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match packed rows",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let packed_flat: Cow<'_, [u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();

    let result = py.detach(|| {
        let mut source = BayesPackedSource::Resident {
            packed_flat: packed_flat.as_ref(),
            bytes_per_snp,
        };
        bayesb_packed_core_impl(
            y_vec.as_ref(),
            &mut source,
            n_samples,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            None,
            &sample_idx,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            shape0,
            rate0,
            s0_b,
            prob_in,
            counts,
            fixed_pi,
            df0_e,
            prior_ss_e,
            seed,
            block_rows,
            pool_ref,
        )
    });

    match result {
        Ok((
            beta,
            alpha,
            varb,
            vare,
            h2_mean,
            var_h2,
            prob_in_mean,
            n_active_mean,
            pip,
            rhat_h2,
            actual_iterations,
            convergence_iteration,
            _posterior_samples,
        )) => {
            let beta_py = beta.into_pyarray(py).into_bound().unbind();
            let alpha_py = alpha.into_pyarray(py).into_bound().unbind();
            let varb_py = varb.into_pyarray(py).into_bound().unbind();
            let pip_py = pip.into_pyarray(py).into_bound().unbind();
            Ok((
                beta_py,
                alpha_py,
                varb_py,
                vare,
                h2_mean,
                var_h2,
                prob_in_mean,
                n_active_mean,
                pip_py,
                rhat_h2,
                actual_iterations,
                convergence_iteration,
            ))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y,
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 3000,
    burnin = 1000,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    s0_b = None,
    prob_in = 0.5,
    counts = 10.0,
    fixed_pi = None,
    df0_e = 5.0,
    prior_ss_e = None,
    threads = 0,
    seed = None,
    block_rows = None
))]
pub fn bayesc_packed(
    py: Python,
    y: PyReadonlyArray1<f64>,
    packed: PyReadonlyArray2<u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b: Option<f64>,
    prob_in: f64,
    counts: f64,
    fixed_pi: Option<f64>,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    threads: usize,
    seed: Option<u64>,
    block_rows: Option<usize>,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    Py<PyArray1<f64>>,
    f64,
    usize,
    usize,
)> {
    if n_iter == 0 {
        return Err(PyValueError::new_err("n_iter must be > 0"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }
    if let Some(v) = s0_b {
        if v <= 0.0 {
            return Err(PyValueError::new_err("s0_b must be > 0"));
        }
    }
    if let Some(v) = prior_ss_e {
        if v <= 0.0 {
            return Err(PyValueError::new_err("prior_ss_e must be > 0"));
        }
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyValueError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let p = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyValueError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<'_, [bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<'_, [f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    let row_mean_vec: Cow<'_, [f32]> = match row_mean.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_mean.as_array().iter().copied().collect()),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> = match row_inv_sd.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_inv_sd.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match packed rows",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let packed_flat: Cow<'_, [u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();

    let result = py.detach(|| {
        let mut source = BayesPackedSource::Resident {
            packed_flat: packed_flat.as_ref(),
            bytes_per_snp,
        };
        bayesc_packed_core_impl(
            y_vec.as_ref(),
            &mut source,
            n_samples,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            None,
            &sample_idx,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            s0_b,
            prob_in,
            counts,
            fixed_pi,
            df0_e,
            prior_ss_e,
            seed,
            block_rows,
            pool_ref,
        )
    });

    match result {
        Ok((
            beta,
            alpha,
            varb_mean,
            vare,
            h2_mean,
            var_h2,
            prob_in_mean,
            n_active_mean,
            pip,
            rhat_h2,
            actual_iterations,
            convergence_iteration,
            _posterior_samples,
        )) => {
            let beta_py = beta.into_pyarray(py).into_bound().unbind();
            let alpha_py = alpha.into_pyarray(py).into_bound().unbind();
            let pip_py = pip.into_pyarray(py).into_bound().unbind();
            Ok((
                beta_py,
                alpha_py,
                varb_mean,
                vare,
                h2_mean,
                var_h2,
                prob_in_mean,
                n_active_mean,
                pip_py,
                rhat_h2,
                actual_iterations,
                convergence_iteration,
            ))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    y,
    n_samples,
    row_indices,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 3000,
    burnin = 1000,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    shape0 = 1.1,
    rate0 = None,
    s0_b = None,
    df0_e = 5.0,
    prior_ss_e = None,
    min_abs_beta = 1e-9,
    threads = 0,
    seed = None,
    block_rows = None,
    mmap_window_mb = None
))]
pub fn bayesa_stream_bed(
    py: Python,
    prefix: String,
    y: PyReadonlyArray1<f64>,
    n_samples: usize,
    row_indices: PyReadonlyArray1<i64>,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0: Option<f64>,
    s0_b: Option<f64>,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    min_abs_beta: f64,
    threads: usize,
    seed: Option<u64>,
    block_rows: Option<usize>,
    mmap_window_mb: Option<usize>,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    f64,
    f64,
    usize,
    usize,
    usize,
)> {
    if n_iter == 0 {
        return Err(PyValueError::new_err("n_iter must be > 0"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !min_abs_beta.is_finite() || min_abs_beta < 0.0 {
        return Err(PyValueError::new_err(
            "min_abs_beta is deprecated/ignored; keep it finite and >= 0 for compatibility",
        ));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if shape0 <= 0.0 {
        return Err(PyValueError::new_err("shape0 must be > 0"));
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let row_idx_raw = row_indices.as_slice()?.to_vec();
    let p = row_idx_raw.len();
    if p == 0 {
        return Err(PyValueError::new_err("row_indices must not be empty"));
    }
    let row_flip_vec: Cow<'_, [bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<'_, [f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    let row_mean_vec: Cow<'_, [f32]> = match row_mean.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_mean.as_array().iter().copied().collect()),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> = match row_inv_sd.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_inv_sd.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match row_indices",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }
    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();
    let result = py.detach(|| {
        let block_rows = block_rows
            .unwrap_or_else(|| bayes_packed_block_rows(n, p))
            .max(1);
        let mut source = build_bayes_source(None, &prefix, n_samples, block_rows, mmap_window_mb)?;
        let n_source = match &source {
            BayesPackedSource::Resident { .. } => 0usize,
            BayesPackedSource::Windowed(matrix) => matrix.n_source_snps(),
        };
        let packed_row_indices =
            parse_index_vec_i64_string(row_idx_raw.as_slice(), n_source, "row_indices")?;
        bayesa_packed_core_impl(
            y_vec.as_ref(),
            &mut source,
            n_samples,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            Some(packed_row_indices.as_slice()),
            &sample_idx,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            shape0,
            rate0,
            s0_b,
            df0_e,
            prior_ss_e,
            min_abs_beta,
            seed,
            Some(block_rows),
            pool_ref,
        )
    });

    match result {
        Ok((
            beta,
            alpha,
            varb,
            vare,
            h2_mean,
            var_h2,
            rhat_h2,
            actual_iterations,
            convergence_iteration,
            posterior_samples,
        )) => {
            let beta_py = beta.into_pyarray(py).into_bound().unbind();
            let alpha_py = alpha.into_pyarray(py).into_bound().unbind();
            let varb_py = varb.into_pyarray(py).into_bound().unbind();
            Ok((
                beta_py,
                alpha_py,
                varb_py,
                vare,
                h2_mean,
                var_h2,
                rhat_h2,
                actual_iterations,
                convergence_iteration,
                posterior_samples,
            ))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    y,
    n_samples,
    row_indices,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 3000,
    burnin = 1000,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    shape0 = 1.1,
    rate0 = None,
    s0_b = None,
    prob_in = 0.5,
    counts = 10.0,
    fixed_pi = None,
    df0_e = 5.0,
    prior_ss_e = None,
    threads = 0,
    seed = None,
    block_rows = None,
    mmap_window_mb = None
))]
pub fn bayesb_stream_bed(
    py: Python,
    prefix: String,
    y: PyReadonlyArray1<f64>,
    n_samples: usize,
    row_indices: PyReadonlyArray1<i64>,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0: Option<f64>,
    s0_b: Option<f64>,
    prob_in: f64,
    counts: f64,
    fixed_pi: Option<f64>,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    threads: usize,
    seed: Option<u64>,
    block_rows: Option<usize>,
    mmap_window_mb: Option<usize>,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    f64,
    f64,
    f64,
    Py<PyArray1<f64>>,
    f64,
    usize,
    usize,
)> {
    if n_iter == 0 {
        return Err(PyValueError::new_err("n_iter must be > 0"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if shape0 <= 0.0 {
        return Err(PyValueError::new_err("shape0 must be > 0"));
    }
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let row_idx_raw = row_indices.as_slice()?.to_vec();
    let p = row_idx_raw.len();
    if p == 0 {
        return Err(PyValueError::new_err("row_indices must not be empty"));
    }
    let row_flip_vec: Cow<'_, [bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<'_, [f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    let row_mean_vec: Cow<'_, [f32]> = match row_mean.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_mean.as_array().iter().copied().collect()),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> = match row_inv_sd.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_inv_sd.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match row_indices",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }
    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();
    let result = py.detach(|| {
        let block_rows = block_rows
            .unwrap_or_else(|| bayes_packed_block_rows(n, p))
            .max(1);
        let mut source = build_bayes_source(None, &prefix, n_samples, block_rows, mmap_window_mb)?;
        let n_source = match &source {
            BayesPackedSource::Resident { .. } => 0usize,
            BayesPackedSource::Windowed(matrix) => matrix.n_source_snps(),
        };
        let packed_row_indices =
            parse_index_vec_i64_string(row_idx_raw.as_slice(), n_source, "row_indices")?;
        bayesb_packed_core_impl(
            y_vec.as_ref(),
            &mut source,
            n_samples,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            Some(packed_row_indices.as_slice()),
            &sample_idx,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            shape0,
            rate0,
            s0_b,
            prob_in,
            counts,
            fixed_pi,
            df0_e,
            prior_ss_e,
            seed,
            Some(block_rows),
            pool_ref,
        )
    });

    match result {
        Ok((
            beta,
            alpha,
            varb,
            vare,
            h2_mean,
            var_h2,
            prob_in_mean,
            n_active_mean,
            pip,
            rhat_h2,
            actual_iterations,
            convergence_iteration,
            _posterior_samples,
        )) => {
            let beta_py = beta.into_pyarray(py).into_bound().unbind();
            let alpha_py = alpha.into_pyarray(py).into_bound().unbind();
            let varb_py = varb.into_pyarray(py).into_bound().unbind();
            let pip_py = pip.into_pyarray(py).into_bound().unbind();
            Ok((
                beta_py,
                alpha_py,
                varb_py,
                vare,
                h2_mean,
                var_h2,
                prob_in_mean,
                n_active_mean,
                pip_py,
                rhat_h2,
                actual_iterations,
                convergence_iteration,
            ))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    prefix,
    y,
    n_samples,
    row_indices,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 3000,
    burnin = 1000,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    s0_b = None,
    prob_in = 0.5,
    counts = 10.0,
    fixed_pi = None,
    df0_e = 5.0,
    prior_ss_e = None,
    threads = 0,
    seed = None,
    block_rows = None,
    mmap_window_mb = None
))]
pub fn bayesc_stream_bed(
    py: Python,
    prefix: String,
    y: PyReadonlyArray1<f64>,
    n_samples: usize,
    row_indices: PyReadonlyArray1<i64>,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b: Option<f64>,
    prob_in: f64,
    counts: f64,
    fixed_pi: Option<f64>,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    threads: usize,
    seed: Option<u64>,
    block_rows: Option<usize>,
    mmap_window_mb: Option<usize>,
) -> PyResult<(
    Py<PyArray1<f64>>,
    Py<PyArray1<f64>>,
    f64,
    f64,
    f64,
    f64,
    f64,
    f64,
    Py<PyArray1<f64>>,
    f64,
    usize,
    usize,
)> {
    if n_iter == 0 {
        return Err(PyValueError::new_err("n_iter must be > 0"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }
    if let Some(v) = s0_b {
        if v <= 0.0 {
            return Err(PyValueError::new_err("s0_b must be > 0"));
        }
    }
    if let Some(v) = prior_ss_e {
        if v <= 0.0 {
            return Err(PyValueError::new_err("prior_ss_e must be > 0"));
        }
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let row_idx_raw = row_indices.as_slice()?.to_vec();
    let p = row_idx_raw.len();
    if p == 0 {
        return Err(PyValueError::new_err("row_indices must not be empty"));
    }
    let row_flip_vec: Cow<'_, [bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<'_, [f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    let row_mean_vec: Cow<'_, [f32]> = match row_mean.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_mean.as_array().iter().copied().collect()),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> = match row_inv_sd.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_inv_sd.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match row_indices",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }
    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();
    let result = py.detach(|| {
        let block_rows = block_rows
            .unwrap_or_else(|| bayes_packed_block_rows(n, p))
            .max(1);
        let mut source = build_bayes_source(None, &prefix, n_samples, block_rows, mmap_window_mb)?;
        let n_source = match &source {
            BayesPackedSource::Resident { .. } => 0usize,
            BayesPackedSource::Windowed(matrix) => matrix.n_source_snps(),
        };
        let packed_row_indices =
            parse_index_vec_i64_string(row_idx_raw.as_slice(), n_source, "row_indices")?;
        bayesc_packed_core_impl(
            y_vec.as_ref(),
            &mut source,
            n_samples,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            Some(packed_row_indices.as_slice()),
            &sample_idx,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            s0_b,
            prob_in,
            counts,
            fixed_pi,
            df0_e,
            prior_ss_e,
            seed,
            Some(block_rows),
            pool_ref,
        )
    });

    match result {
        Ok((
            beta,
            alpha,
            varb_mean,
            vare,
            h2_mean,
            var_h2,
            prob_in_mean,
            n_active_mean,
            pip,
            rhat_h2,
            actual_iterations,
            convergence_iteration,
            _posterior_samples,
        )) => {
            let beta_py = beta.into_pyarray(py).into_bound().unbind();
            let alpha_py = alpha.into_pyarray(py).into_bound().unbind();
            let pip_py = pip.into_pyarray(py).into_bound().unbind();
            Ok((
                beta_py,
                alpha_py,
                varb_mean,
                vare,
                h2_mean,
                var_h2,
                prob_in_mean,
                n_active_mean,
                pip_py,
                rhat_h2,
                actual_iterations,
                convergence_iteration,
            ))
        }
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[allow(clippy::too_many_arguments)]
fn bayesa_packed_trace_core_impl(
    y: &[f64],
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0_opt: Option<f64>,
    s0_b_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    seed: Option<u64>,
    trace_snp_indices: &[usize],
    block_rows: Option<usize>,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<PackedBayesTraceResult, String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    if y.len() != n {
        return Err("y length mismatch with sample_indices".to_string());
    }
    if x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }
    if row_flip.len() != p || row_maf.len() != p || row_mean.len() != p || row_inv_sd.len() != p {
        return Err("row metadata length mismatch with packed rows".to_string());
    }

    let _blas_guard = OpenBlasThreadGuard::enter(bayes_packed_blas_threads());

    let mut source = BayesPackedSource::Resident {
        packed_flat,
        bytes_per_snp,
    };
    let mut backend = PackedBayesBackend::new(
        &mut source,
        n_samples,
        p,
        row_flip,
        row_mean,
        row_inv_sd,
        None,
        sample_idx,
        block_rows,
        pool,
    )?;

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32];
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes)
        }
    };

    let mut x2 = vec![0.0_f64; p];
    let mut mean_x = vec![0.0_f64; p];
    backend.for_each_block(|st, ed, m_block| {
        let br = ed - st;
        for off in 0..br {
            let row = &m_block[off * n..(off + 1) * n];
            let mut s = 0.0;
            let mut msum = 0.0;
            for &v in row {
                s += (v as f64) * (v as f64);
                msum += v as f64;
            }
            x2[st + off] = s;
            mean_x[st + off] = msum / n_f;
        }
        Ok(())
    })?;
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let mut y_mean = 0.0;
    for v in y {
        y_mean += *v;
    }
    y_mean /= n_f;
    let mut var_y = 0.0;
    for v in y {
        let d = *v - y_mean;
        var_y += d * d;
    }
    var_y /= n_f - 1.0;

    let s0_b = match s0_b_opt {
        Some(v) => v,
        None => {
            if msx <= 0.0 {
                return Err("MSx must be positive to compute S0_b".to_string());
            }
            var_y * r2 / msx * (df0_b + 2.0)
        }
    };
    if s0_b <= 0.0 {
        return Err("S0_b must be positive".to_string());
    }

    let rate0 = match rate0_opt {
        Some(v) => v,
        None => {
            if shape0 <= 1.0 {
                return Err("shape0 must be > 1 when rate0 is not provided".to_string());
            }
            (shape0 - 1.0) / s0_b
        }
    };
    if rate0 <= 0.0 {
        return Err("rate0 must be positive".to_string());
    }

    let mut var_e = var_y * (1.0 - r2);
    if var_e <= 0.0 {
        return Err("varE must be positive; check R2".to_string());
    }

    let prior_ss_e = match prior_ss_e_opt {
        Some(v) => v,
        None => var_e * (df0_e + 2.0),
    };
    if prior_ss_e <= 0.0 {
        return Err("prior_ss_e must be positive".to_string());
    }

    let mut beta = vec![0.0; p];
    let mut var_b = vec![s0_b / (df0_b + 2.0); p];
    let mut s = s0_b;

    let mut alpha = vec![0.0; q];
    let mut xtx = vec![0.0_f64; q * q];
    row_major_xtx_f64(x, n, q, &mut xtx);
    let mut x2_x = vec![0.0_f64; q];
    for k in 0..q {
        x2_x[k] = xtx[k * q + k];
    }
    let mut alpha_xtr = vec![0.0_f64; q];
    let mut alpha_delta = vec![0.0_f64; q];
    let mut alpha_tmp_n = vec![0.0_f64; n];

    let mut r = y.to_vec();
    let mut marker_residual = vec![0.0_f32; n];
    let mut beta_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let keep_iters = posterior_keep_iters(n_iter, burnin, thin);
    let n_keep_target = keep_iters.len();
    let n_trace_snps = trace_snp_indices.len();
    let mut h2_trace = Vec::with_capacity(n_keep_target);
    let mut var_e_trace = Vec::with_capacity(n_keep_target);
    let prob_in_trace = vec![f64::NAN; n_keep_target];
    let n_active_trace = vec![f64::NAN; n_keep_target];
    let full_iter_trace: Vec<i64> = (1..=n_iter).map(|v| v as i64).collect();
    let mut full_h2_trace = Vec::with_capacity(n_iter);
    let mut full_var_e_trace = Vec::with_capacity(n_iter);
    let full_prob_in_trace = vec![f64::NAN; n_iter];
    let full_n_active_trace = vec![f64::NAN; n_iter];
    let mut beta_trace = vec![0.0_f64; n_keep_target * n_trace_snps];
    let mut keep_row = 0usize;
    let mut n_keep = 0usize;
    let var_b_fixed = 1e10_f64;

    let chi_b = ChiSquared::new(df0_b + 1.0).map_err(|e| e.to_string())?;
    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    for it in 0..n_iter {
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        update_alpha_gauss_seidel_blas(
            x,
            n,
            q,
            inv_var_e,
            inv_var_b_fixed,
            &x2_x,
            &xtx,
            &mut alpha,
            &mut r,
            &mut alpha_xtr,
            &mut alpha_delta,
            &mut alpha_tmp_n,
            &mut rng,
        );

        copy_f64_to_f32(&r, &mut marker_residual);
        backend.for_each_block(|st, ed, m_block| {
            let br = ed - st;
            for off in 0..br {
                let j = st + off;
                let m_row = &m_block[off * n..(off + 1) * n];
                let c = x2[j] * inv_var_e + 1.0 / var_b[j];
                let z_beta: f64 = rng.sample(StandardNormal);
                let old_beta = beta[j];
                let (_, new_beta) =
                    bayes_marker_update_f32(&mut marker_residual, m_row, old_beta, x2[j], |u| {
                        u * inv_var_e / c + (1.0 / c).sqrt() * z_beta
                    });
                beta[j] = new_beta;
            }
            Ok(())
        })?;

        copy_f32_to_f64(&marker_residual, &mut r);
        for j in 0..p {
            var_b[j] = (s + beta[j] * beta[j]) / rng.sample(chi_b);
            if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                return Err("BayesA packed var_b became non-finite or non-positive".to_string());
            }
        }

        let mut tmp_rate = 0.0;
        for vb in &var_b {
            tmp_rate += 1.0 / *vb;
        }
        tmp_rate = tmp_rate / 2.0 + rate0;
        if tmp_rate <= 0.0 {
            return Err("Gamma rate became non-positive while updating S".to_string());
        }
        let tmp_shape = p as f64 * df0_b / 2.0 + shape0;
        let gamma = Gamma::new(tmp_shape, 1.0 / tmp_rate).map_err(|e| e.to_string())?;
        s = rng.sample(gamma);
        if !(s.is_finite() && s > 0.0) {
            return Err("BayesA packed S became non-finite or non-positive".to_string());
        }

        let ss_e = ddot_f64(&r, &r) + prior_ss_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesA packed var_e became non-finite or non-positive".to_string());
        }

        let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
        let h2 = var_g / (var_g + var_e);
        full_h2_trace.push(h2);
        full_var_e_trace.push(var_e);

        if it >= burnin && ((it - burnin) % thin == 0) {
            for j in 0..p {
                beta_sum[j] += beta[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            h2_trace.push(h2);
            var_e_trace.push(var_e);
            fill_beta_trace_row(
                &mut beta_trace,
                keep_row,
                n_trace_snps,
                trace_snp_indices,
                &beta,
                None,
            );
            keep_row += 1;
            n_keep += 1;
        }
    }

    if n_keep == 0 {
        return Err("No posterior samples kept for the requested trace iterations".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for bj in &mut beta_sum {
        *bj *= inv_keep;
    }
    for ak in &mut alpha_sum {
        *ak *= inv_keep;
    }
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }

    Ok(PackedBayesTraceResult {
        beta: beta_sum,
        alpha: alpha_sum,
        vare: var_e_sum,
        h2_mean,
        var_h2,
        prob_in_mean: f64::NAN,
        n_active_mean: f64::NAN,
        iter_trace: keep_iters,
        h2_trace,
        var_e_trace,
        prob_in_trace,
        n_active_trace,
        full_iter_trace,
        full_h2_trace,
        full_var_e_trace,
        full_prob_in_trace,
        full_n_active_trace,
        beta_trace_indices: trace_snp_indices.iter().map(|&v| v as i64).collect(),
        beta_trace,
    })
}

#[allow(clippy::too_many_arguments)]
fn bayesb_packed_trace_core_impl(
    y: &[f64],
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0_opt: Option<f64>,
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    fixed_prob_in_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    seed: Option<u64>,
    trace_snp_indices: &[usize],
    block_rows: Option<usize>,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<PackedBayesTraceResult, String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    let prob_in_base = fixed_prob_in_opt.unwrap_or(prob_in_init);
    if !(prob_in_base > 0.0 && prob_in_base < 1.0) {
        return Err("prob_in must be in (0, 1)".to_string());
    }
    if counts < 0.0 {
        return Err("counts must be >= 0".to_string());
    }
    if y.len() != n {
        return Err("y length mismatch with sample_indices".to_string());
    }
    if x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }
    if row_flip.len() != p || row_maf.len() != p || row_mean.len() != p || row_inv_sd.len() != p {
        return Err("row metadata length mismatch with packed rows".to_string());
    }

    let _blas_guard = OpenBlasThreadGuard::enter(bayes_packed_blas_threads());

    let mut source = BayesPackedSource::Resident {
        packed_flat,
        bytes_per_snp,
    };
    let mut backend = PackedBayesBackend::new(
        &mut source,
        n_samples,
        p,
        row_flip,
        row_mean,
        row_inv_sd,
        None,
        sample_idx,
        block_rows,
        pool,
    )?;

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32];
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes)
        }
    };

    let mut x2 = vec![0.0_f64; p];
    let mut mean_x = vec![0.0_f64; p];
    backend.for_each_block(|st, ed, m_block| {
        let br = ed - st;
        for off in 0..br {
            let row = &m_block[off * n..(off + 1) * n];
            let mut s = 0.0;
            let mut msum = 0.0;
            for &v in row {
                s += (v as f64) * (v as f64);
                msum += v as f64;
            }
            x2[st + off] = s;
            mean_x[st + off] = msum / n_f;
        }
        Ok(())
    })?;
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let mut y_mean = 0.0;
    for v in y {
        y_mean += *v;
    }
    y_mean /= n_f;
    let mut var_y = 0.0;
    for v in y {
        let d = *v - y_mean;
        var_y += d * d;
    }
    var_y /= n_f - 1.0;

    let s0_b = match s0_b_opt {
        Some(v) => v,
        None => {
            if msx <= 0.0 {
                return Err("MSx must be positive to compute S0_b".to_string());
            }
            var_y * r2 / msx * (df0_b + 2.0) / prob_in_base
        }
    };
    if s0_b <= 0.0 {
        return Err("S0_b must be positive".to_string());
    }

    let rate0 = bayesb_rate0(shape0, rate0_opt, s0_b)?;

    let mut var_e = var_y * (1.0 - r2);
    if var_e <= 0.0 {
        return Err("varE must be positive; check R2".to_string());
    }

    let prior_ss_e = match prior_ss_e_opt {
        Some(v) => v,
        None => var_e * (df0_e + 2.0),
    };
    if prior_ss_e <= 0.0 {
        return Err("prior_ss_e must be positive".to_string());
    }

    let mut beta = vec![0.0; p];
    let mut d = vec![0u8; p];
    let mut var_b = vec![s0_b / (df0_b + 2.0); p];
    let mut prob_in = prob_in_base;
    let mut s = s0_b;
    let (counts_in, counts_out) = bayes_inclusion_prior_shapes(prob_in_base, counts);

    let mut alpha = vec![0.0; q];
    let mut xtx = vec![0.0_f64; q * q];
    row_major_xtx_f64(x, n, q, &mut xtx);
    let mut x2_x = vec![0.0_f64; q];
    for k in 0..q {
        x2_x[k] = xtx[k * q + k];
    }
    let mut alpha_xtr = vec![0.0_f64; q];
    let mut alpha_delta = vec![0.0_f64; q];
    let mut alpha_tmp_n = vec![0.0_f64; n];

    let mut r = y.to_vec();
    let mut marker_residual = vec![0.0_f32; n];
    let mut beta_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut prob_in_sum = 0.0;
    let mut n_active_sum = 0.0;
    let keep_iters = posterior_keep_iters(n_iter, burnin, thin);
    let n_keep_target = keep_iters.len();
    let n_trace_snps = trace_snp_indices.len();
    let mut h2_trace = Vec::with_capacity(n_keep_target);
    let mut var_e_trace = Vec::with_capacity(n_keep_target);
    let mut prob_in_trace = Vec::with_capacity(n_keep_target);
    let mut n_active_trace = Vec::with_capacity(n_keep_target);
    let full_iter_trace: Vec<i64> = (1..=n_iter).map(|v| v as i64).collect();
    let mut full_h2_trace = Vec::with_capacity(n_iter);
    let mut full_var_e_trace = Vec::with_capacity(n_iter);
    let mut full_prob_in_trace = Vec::with_capacity(n_iter);
    let mut full_n_active_trace = Vec::with_capacity(n_iter);
    let mut beta_trace = vec![0.0_f64; n_keep_target * n_trace_snps];
    let mut keep_row = 0usize;
    let mut n_keep = 0usize;
    let var_b_fixed = 1e10_f64;

    let chi_b_active = ChiSquared::new(df0_b + 1.0).map_err(|e| e.to_string())?;
    let chi_b_inactive = ChiSquared::new(df0_b).map_err(|e| e.to_string())?;
    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;

    for it in 0..n_iter {
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        update_alpha_gauss_seidel_blas(
            x,
            n,
            q,
            inv_var_e,
            inv_var_b_fixed,
            &x2_x,
            &xtx,
            &mut alpha,
            &mut r,
            &mut alpha_xtr,
            &mut alpha_delta,
            &mut alpha_tmp_n,
            &mut rng,
        );

        copy_f64_to_f32(&r, &mut marker_residual);
        let log_odds_prior = (prob_in / (1.0 - prob_in)).ln();

        backend.for_each_block(|st, ed, m_block| {
            let br = ed - st;
            for off in 0..br {
                let j = st + off;
                let m_row = &m_block[off * n..(off + 1) * n];
                let b_old = beta[j];
                let c = x2[j] * inv_var_e + 1.0 / var_b[j];
                if !(c.is_finite() && c > 0.0) {
                    return Err(
                        "Non-positive posterior precision in BayesB packed beta update".to_string(),
                    );
                }
                let mut new_d = 0u8;
                let (_, new_beta) =
                    bayes_marker_update_f32(&mut marker_residual, m_row, b_old, x2[j], |xe| {
                        let rhs = xe * inv_var_e;
                        let log_bf10 = 0.5 * rhs * rhs / c - 0.5 * (var_b[j] * c).ln();
                        let log_odds = log_odds_prior + log_bf10;
                        let p_in = if log_odds >= 0.0 {
                            1.0 / (1.0 + (-log_odds).exp())
                        } else {
                            let e = log_odds.exp();
                            e / (1.0 + e)
                        };
                        new_d = if rng.random::<f64>() < p_in { 1u8 } else { 0u8 };
                        if new_d == 1 {
                            let z_beta: f64 = rng.sample(StandardNormal);
                            rhs / c + (1.0 / c).sqrt() * z_beta
                        } else {
                            0.0
                        }
                    });
                d[j] = new_d;
                beta[j] = new_beta;
            }
            Ok(())
        })?;

        copy_f32_to_f64(&marker_residual, &mut r);
        let mut n_active = 0usize;
        for j in 0..p {
            if d[j] == 1 {
                let beta2 = beta[j] * beta[j];
                var_b[j] = bayes_positive_floor((s + beta2) / rng.sample(chi_b_active));
                if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                    return Err("BayesB packed var_b became non-finite or non-positive".to_string());
                }
                n_active += 1;
            } else {
                var_b[j] = bayes_positive_floor(s / rng.sample(chi_b_inactive));
                if !(var_b[j].is_finite() && var_b[j] > 0.0) {
                    return Err(
                        "BayesB packed inactive var_b became non-finite or non-positive"
                            .to_string(),
                    );
                }
            }
        }

        let mut tmp_rate = 0.0;
        for vb in &var_b {
            tmp_rate += 1.0 / *vb;
        }
        tmp_rate = tmp_rate / 2.0 + rate0;
        if tmp_rate <= 0.0 {
            return Err(
                "Gamma rate became non-positive while updating BayesB packed trace S".to_string(),
            );
        }
        let tmp_shape = p as f64 * df0_b / 2.0 + shape0;
        s = bayes_positive_floor(sample_gamma_with_rate(&mut rng, tmp_shape, tmp_rate)?);
        if !(s.is_finite() && s > 0.0) {
            return Err("BayesB packed trace S became non-finite or non-positive".to_string());
        }

        let mrk_in = n_active as f64;
        let a = mrk_in + counts_in;
        let b = (p as f64 - mrk_in) + counts_out;
        if fixed_prob_in_opt.is_none() {
            let beta_dist = Beta::new(a, b).map_err(|e| e.to_string())?;
            prob_in = rng.sample(beta_dist);
        }

        let ss_e = ddot_f64(&r, &r) + prior_ss_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesB packed var_e became non-finite or non-positive".to_string());
        }

        let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
        let h2 = var_g / (var_g + var_e);
        full_h2_trace.push(h2);
        full_var_e_trace.push(var_e);
        full_prob_in_trace.push(prob_in);
        full_n_active_trace.push(mrk_in);

        if it >= burnin && ((it - burnin) % thin == 0) {
            for j in 0..p {
                beta_sum[j] += (d[j] as f64) * beta[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            prob_in_sum += prob_in;
            n_active_sum += mrk_in;
            h2_trace.push(h2);
            var_e_trace.push(var_e);
            prob_in_trace.push(prob_in);
            n_active_trace.push(mrk_in);
            fill_beta_trace_row(
                &mut beta_trace,
                keep_row,
                n_trace_snps,
                trace_snp_indices,
                &beta,
                Some(&d),
            );
            keep_row += 1;
            n_keep += 1;
        }
    }

    if n_keep == 0 {
        return Err("No posterior samples kept for the requested trace iterations".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for bj in &mut beta_sum {
        *bj *= inv_keep;
    }
    for ak in &mut alpha_sum {
        *ak *= inv_keep;
    }
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    let prob_in_mean = prob_in_sum * inv_keep;
    let n_active_mean = n_active_sum * inv_keep;

    Ok(PackedBayesTraceResult {
        beta: beta_sum,
        alpha: alpha_sum,
        vare: var_e_sum,
        h2_mean,
        var_h2,
        prob_in_mean,
        n_active_mean,
        iter_trace: keep_iters,
        h2_trace,
        var_e_trace,
        prob_in_trace,
        n_active_trace,
        full_iter_trace,
        full_h2_trace,
        full_var_e_trace,
        full_prob_in_trace,
        full_n_active_trace,
        beta_trace_indices: trace_snp_indices.iter().map(|&v| v as i64).collect(),
        beta_trace,
    })
}

#[allow(clippy::too_many_arguments)]
fn bayesc_packed_trace_core_impl(
    y: &[f64],
    packed_flat: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    row_flip: &[bool],
    row_maf: &[f32],
    row_mean: &[f32],
    row_inv_sd: &[f32],
    sample_idx: &[usize],
    x: &[f64],
    n: usize,
    p: usize,
    q: usize,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b_opt: Option<f64>,
    prob_in_init: f64,
    counts: f64,
    fixed_prob_in_opt: Option<f64>,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    seed: Option<u64>,
    trace_snp_indices: &[usize],
    block_rows: Option<usize>,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Result<PackedBayesTraceResult, String> {
    let n_f = n as f64;
    if n_f <= 1.0 {
        return Err("n must be > 1".to_string());
    }
    let prob_in_base = fixed_prob_in_opt.unwrap_or(prob_in_init);
    if !(prob_in_base > 0.0 && prob_in_base < 1.0) {
        return Err("prob_in must be in (0, 1)".to_string());
    }
    if counts < 0.0 {
        return Err("counts must be >= 0".to_string());
    }
    if y.len() != n {
        return Err("y length mismatch with sample_indices".to_string());
    }
    if x.len() != n * q {
        return Err("X has incompatible dimensions".to_string());
    }
    if row_flip.len() != p || row_maf.len() != p || row_mean.len() != p || row_inv_sd.len() != p {
        return Err("row metadata length mismatch with packed rows".to_string());
    }

    let _blas_guard = OpenBlasThreadGuard::enter(bayes_packed_blas_threads());

    let mut source = BayesPackedSource::Resident {
        packed_flat,
        bytes_per_snp,
    };
    let mut backend = PackedBayesBackend::new(
        &mut source,
        n_samples,
        p,
        row_flip,
        row_mean,
        row_inv_sd,
        None,
        sample_idx,
        block_rows,
        pool,
    )?;

    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut seed_bytes = [0u8; 32];
            if let Err(err) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("False to generate random seed: {err}, try to use fixed seed");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes)
        }
    };

    let mut x2 = vec![0.0_f64; p];
    let mut mean_x = vec![0.0_f64; p];
    backend.for_each_block(|st, ed, m_block| {
        let br = ed - st;
        for off in 0..br {
            let row = &m_block[off * n..(off + 1) * n];
            let mut s = 0.0;
            let mut msum = 0.0;
            for &v in row {
                s += (v as f64) * (v as f64);
                msum += v as f64;
            }
            x2[st + off] = s;
            mean_x[st + off] = msum / n_f;
        }
        Ok(())
    })?;
    let mut sum_x2 = 0.0;
    let mut sum_mean_x2 = 0.0;
    for j in 0..p {
        sum_x2 += x2[j];
        sum_mean_x2 += mean_x[j] * mean_x[j];
    }
    let msx = sum_x2 / n_f - sum_mean_x2;

    let mut y_mean = 0.0;
    for v in y {
        y_mean += *v;
    }
    y_mean /= n_f;
    let mut var_y = 0.0;
    for v in y {
        let d = *v - y_mean;
        var_y += d * d;
    }
    var_y /= n_f - 1.0;

    let s0_b = match s0_b_opt {
        Some(v) => v,
        None => {
            if msx <= 0.0 {
                return Err("MSx must be positive to compute S0_b".to_string());
            }
            var_y * r2 / msx * (df0_b + 2.0) / prob_in_base
        }
    };
    if s0_b <= 0.0 {
        return Err("S0_b must be positive".to_string());
    }

    let mut var_e = var_y * (1.0 - r2);
    if var_e <= 0.0 {
        return Err("varE must be positive; check R2".to_string());
    }

    let prior_ss_e = match prior_ss_e_opt {
        Some(v) => v,
        None => var_e * (df0_e + 2.0),
    };
    if prior_ss_e <= 0.0 {
        return Err("prior_ss_e must be positive".to_string());
    }

    let mut beta = vec![0.0; p];
    let mut d = vec![0u8; p];
    let mut var_b = s0_b;
    let mut prob_in = prob_in_base;
    let counts_in = counts * prob_in_base;
    let counts_out = counts - counts_in;

    let mut alpha = vec![0.0; q];
    let mut xtx = vec![0.0_f64; q * q];
    row_major_xtx_f64(x, n, q, &mut xtx);
    let mut x2_x = vec![0.0_f64; q];
    for k in 0..q {
        x2_x[k] = xtx[k * q + k];
    }
    let mut alpha_xtr = vec![0.0_f64; q];
    let mut alpha_delta = vec![0.0_f64; q];
    let mut alpha_tmp_n = vec![0.0_f64; n];

    let mut r = y.to_vec();
    let mut marker_residual = vec![0.0_f32; n];
    let mut beta_sum = vec![0.0; p];
    let mut alpha_sum = vec![0.0; q];
    let mut var_e_sum = 0.0;
    let mut h2_sum = 0.0;
    let mut h2_sq_sum = 0.0;
    let mut prob_in_sum = 0.0;
    let mut n_active_sum = 0.0;
    let keep_iters = posterior_keep_iters(n_iter, burnin, thin);
    let n_keep_target = keep_iters.len();
    let n_trace_snps = trace_snp_indices.len();
    let mut h2_trace = Vec::with_capacity(n_keep_target);
    let mut var_e_trace = Vec::with_capacity(n_keep_target);
    let mut prob_in_trace = Vec::with_capacity(n_keep_target);
    let mut n_active_trace = Vec::with_capacity(n_keep_target);
    let full_iter_trace: Vec<i64> = (1..=n_iter).map(|v| v as i64).collect();
    let mut full_h2_trace = Vec::with_capacity(n_iter);
    let mut full_var_e_trace = Vec::with_capacity(n_iter);
    let mut full_prob_in_trace = Vec::with_capacity(n_iter);
    let mut full_n_active_trace = Vec::with_capacity(n_iter);
    let mut beta_trace = vec![0.0_f64; n_keep_target * n_trace_snps];
    let mut keep_row = 0usize;
    let mut n_keep = 0usize;
    let var_b_fixed = 1e10_f64;

    let chi_e = ChiSquared::new(n_f + df0_e).map_err(|e| e.to_string())?;
    let mut chi_b_cache: HashMap<usize, ChiSquared<f64>> = HashMap::new();

    for it in 0..n_iter {
        let inv_var_e = 1.0 / var_e;
        let inv_var_b_fixed = 1.0 / var_b_fixed;

        update_alpha_gauss_seidel_blas(
            x,
            n,
            q,
            inv_var_e,
            inv_var_b_fixed,
            &x2_x,
            &xtx,
            &mut alpha,
            &mut r,
            &mut alpha_xtr,
            &mut alpha_delta,
            &mut alpha_tmp_n,
            &mut rng,
        );

        copy_f64_to_f32(&r, &mut marker_residual);
        let log_odds_prior = (prob_in / (1.0 - prob_in)).ln();

        backend.for_each_block(|st, ed, m_block| {
            let br = ed - st;
            for off in 0..br {
                let j = st + off;
                let m_row = &m_block[off * n..(off + 1) * n];
                let b_old = beta[j];
                let c = x2[j] * inv_var_e + 1.0 / var_b;
                if !(c.is_finite() && c > 0.0) {
                    return Err(
                        "Non-positive posterior precision in BayesC packed beta update".to_string(),
                    );
                }
                let mut new_d = 0u8;
                let (_, new_beta) =
                    bayes_marker_update_f32(&mut marker_residual, m_row, b_old, x2[j], |xe| {
                        let rhs = xe * inv_var_e;
                        let log_bf10 = 0.5 * rhs * rhs / c - 0.5 * (var_b * c).ln();
                        let log_odds = log_odds_prior + log_bf10;
                        let p_in = if log_odds >= 0.0 {
                            1.0 / (1.0 + (-log_odds).exp())
                        } else {
                            let e = log_odds.exp();
                            e / (1.0 + e)
                        };
                        new_d = if rng.random::<f64>() < p_in { 1u8 } else { 0u8 };
                        if new_d == 1 {
                            let z_beta: f64 = rng.sample(StandardNormal);
                            rhs / c + (1.0 / c).sqrt() * z_beta
                        } else {
                            0.0
                        }
                    });
                d[j] = new_d;
                beta[j] = new_beta;
            }
            Ok(())
        })?;

        copy_f32_to_f64(&marker_residual, &mut r);
        let mut mrk_in_usize = 0usize;
        let mut ss_b = 0.0;
        for j in 0..p {
            if d[j] == 1 {
                ss_b += beta[j] * beta[j];
                mrk_in_usize += 1;
            }
        }
        ss_b += s0_b;
        let chi_b_eff = if let Some(dist) = chi_b_cache.get(&mrk_in_usize) {
            *dist
        } else {
            let dist = ChiSquared::new(df0_b + mrk_in_usize as f64).map_err(|e| e.to_string())?;
            chi_b_cache.insert(mrk_in_usize, dist);
            dist
        };
        var_b = ss_b / rng.sample(chi_b_eff);
        if !(var_b.is_finite() && var_b > 0.0) {
            return Err("BayesC packed var_b became non-finite or non-positive".to_string());
        }

        let mrk_in = mrk_in_usize as f64;
        let a = mrk_in + counts_in + 1.0;
        let b = (p as f64 - mrk_in) + counts_out + 1.0;
        if fixed_prob_in_opt.is_none() {
            let beta_dist = Beta::new(a, b).map_err(|e| e.to_string())?;
            prob_in = rng.sample(beta_dist);
        }

        let ss_e = ddot_f64(&r, &r) + prior_ss_e;
        var_e = ss_e / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesC packed var_e became non-finite or non-positive".to_string());
        }

        let var_g = genetic_variance_from_residual(y, &r, x, &alpha, n, q);
        let h2 = var_g / (var_g + var_e);
        full_h2_trace.push(h2);
        full_var_e_trace.push(var_e);
        full_prob_in_trace.push(prob_in);
        full_n_active_trace.push(mrk_in);

        if it >= burnin && ((it - burnin) % thin == 0) {
            for j in 0..p {
                beta_sum[j] += (d[j] as f64) * beta[j];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            prob_in_sum += prob_in;
            n_active_sum += mrk_in;
            h2_trace.push(h2);
            var_e_trace.push(var_e);
            prob_in_trace.push(prob_in);
            n_active_trace.push(mrk_in);
            fill_beta_trace_row(
                &mut beta_trace,
                keep_row,
                n_trace_snps,
                trace_snp_indices,
                &beta,
                Some(&d),
            );
            keep_row += 1;
            n_keep += 1;
        }
    }

    if n_keep == 0 {
        return Err("No posterior samples kept for the requested trace iterations".to_string());
    }

    let inv_keep = 1.0 / n_keep as f64;
    for bj in &mut beta_sum {
        *bj *= inv_keep;
    }
    for ak in &mut alpha_sum {
        *ak *= inv_keep;
    }
    var_e_sum *= inv_keep;
    let h2_mean = h2_sum * inv_keep;
    let mut var_h2 = h2_sq_sum * inv_keep - h2_mean * h2_mean;
    if var_h2 < 0.0 {
        var_h2 = 0.0;
    }
    let prob_in_mean = prob_in_sum * inv_keep;
    let n_active_mean = n_active_sum * inv_keep;

    Ok(PackedBayesTraceResult {
        beta: beta_sum,
        alpha: alpha_sum,
        vare: var_e_sum,
        h2_mean,
        var_h2,
        prob_in_mean,
        n_active_mean,
        iter_trace: keep_iters,
        h2_trace,
        var_e_trace,
        prob_in_trace,
        n_active_trace,
        full_iter_trace,
        full_h2_trace,
        full_var_e_trace,
        full_prob_in_trace,
        full_n_active_trace,
        beta_trace_indices: trace_snp_indices.iter().map(|&v| v as i64).collect(),
        beta_trace,
    })
}

#[pyfunction]
#[pyo3(signature = (
    y,
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 200,
    burnin = 100,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    shape0 = 1.1,
    rate0 = None,
    s0_b = None,
    df0_e = 5.0,
    prior_ss_e = None,
    min_abs_beta = 1e-9,
    trace_snp_indices = None,
    threads = 0,
    seed = None,
    block_rows = None
))]
pub fn bayesa_packed_trace<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    packed: PyReadonlyArray2<u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0: Option<f64>,
    s0_b: Option<f64>,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    min_abs_beta: f64,
    trace_snp_indices: Option<PyReadonlyArray1<i64>>,
    threads: usize,
    seed: Option<u64>,
    block_rows: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    if n_iter <= burnin {
        return Err(PyValueError::new_err("n_iter must be > burnin"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !min_abs_beta.is_finite() || min_abs_beta < 0.0 {
        return Err(PyValueError::new_err(
            "min_abs_beta is deprecated/ignored; keep it finite and >= 0 for compatibility",
        ));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if shape0 <= 0.0 {
        return Err(PyValueError::new_err("shape0 must be > 0"));
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyValueError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let p = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyValueError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<'_, [bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<'_, [f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    let row_mean_vec: Cow<'_, [f32]> = match row_mean.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_mean.as_array().iter().copied().collect()),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> = match row_inv_sd.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_inv_sd.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match packed rows",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }
    let trace_idx =
        parse_optional_index_vec_i64(trace_snp_indices.as_ref(), p, "trace_snp_indices")?;

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let packed_flat: Cow<'_, [u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();
    let result = py.detach(|| {
        bayesa_packed_trace_core_impl(
            y_vec.as_ref(),
            packed_flat.as_ref(),
            bytes_per_snp,
            n_samples,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            &sample_idx,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            shape0,
            rate0,
            s0_b,
            df0_e,
            prior_ss_e,
            seed,
            &trace_idx,
            block_rows,
            pool_ref,
        )
    });
    match result {
        Ok(res) => packed_trace_result_to_pydict(py, res),
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y,
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 200,
    burnin = 100,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    shape0 = 1.1,
    rate0 = None,
    s0_b = None,
    prob_in = 0.5,
    counts = 10.0,
    fixed_pi = None,
    df0_e = 5.0,
    prior_ss_e = None,
    trace_snp_indices = None,
    threads = 0,
    seed = None,
    block_rows = None
))]
pub fn bayesb_packed_trace<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    packed: PyReadonlyArray2<u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    shape0: f64,
    rate0: Option<f64>,
    s0_b: Option<f64>,
    prob_in: f64,
    counts: f64,
    fixed_pi: Option<f64>,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    trace_snp_indices: Option<PyReadonlyArray1<i64>>,
    threads: usize,
    seed: Option<u64>,
    block_rows: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    if n_iter <= burnin {
        return Err(PyValueError::new_err("n_iter must be > burnin"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if shape0 <= 0.0 {
        return Err(PyValueError::new_err("shape0 must be > 0"));
    }
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyValueError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let p = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyValueError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<'_, [bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<'_, [f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    let row_mean_vec: Cow<'_, [f32]> = match row_mean.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_mean.as_array().iter().copied().collect()),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> = match row_inv_sd.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_inv_sd.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match packed rows",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }
    let trace_idx =
        parse_optional_index_vec_i64(trace_snp_indices.as_ref(), p, "trace_snp_indices")?;

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let packed_flat: Cow<'_, [u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();
    let result = py.detach(|| {
        bayesb_packed_trace_core_impl(
            y_vec.as_ref(),
            packed_flat.as_ref(),
            bytes_per_snp,
            n_samples,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            &sample_idx,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            shape0,
            rate0,
            s0_b,
            prob_in,
            counts,
            fixed_pi,
            df0_e,
            prior_ss_e,
            seed,
            &trace_idx,
            block_rows,
            pool_ref,
        )
    });
    match result {
        Ok(res) => packed_trace_result_to_pydict(py, res),
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[pyfunction]
#[pyo3(signature = (
    y,
    packed,
    n_samples,
    row_flip,
    row_maf,
    row_mean,
    row_inv_sd,
    sample_indices,
    x = None,
    n_iter = 200,
    burnin = 100,
    thin = 1,
    r2 = 0.5,
    df0_b = 5.0,
    s0_b = None,
    prob_in = 0.5,
    counts = 10.0,
    fixed_pi = None,
    df0_e = 5.0,
    prior_ss_e = None,
    trace_snp_indices = None,
    threads = 0,
    seed = None,
    block_rows = None
))]
pub fn bayesc_packed_trace<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
    packed: PyReadonlyArray2<u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<bool>,
    row_maf: PyReadonlyArray1<f32>,
    row_mean: PyReadonlyArray1<f32>,
    row_inv_sd: PyReadonlyArray1<f32>,
    sample_indices: PyReadonlyArray1<i64>,
    x: Option<PyReadonlyArray2<f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_b: f64,
    s0_b: Option<f64>,
    prob_in: f64,
    counts: f64,
    fixed_pi: Option<f64>,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    trace_snp_indices: Option<PyReadonlyArray1<i64>>,
    threads: usize,
    seed: Option<u64>,
    block_rows: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    if n_iter <= burnin {
        return Err(PyValueError::new_err("n_iter must be > burnin"));
    }
    if thin == 0 {
        return Err(PyValueError::new_err("thin must be >= 1"));
    }
    if !(r2 > 0.0 && r2 < 1.0) {
        return Err(PyValueError::new_err("R2 must be in (0, 1)"));
    }
    if df0_b <= 0.0 || df0_e <= 0.0 {
        return Err(PyValueError::new_err("df0_b and df0_e must be > 0"));
    }
    if !(prob_in > 0.0 && prob_in < 1.0) {
        return Err(PyValueError::new_err("prob_in must be in (0, 1)"));
    }
    if counts < 0.0 {
        return Err(PyValueError::new_err("counts must be >= 0"));
    }
    if let Some(v) = s0_b {
        if v <= 0.0 {
            return Err(PyValueError::new_err("s0_b must be > 0"));
        }
    }
    if let Some(v) = prior_ss_e {
        if v <= 0.0 {
            return Err(PyValueError::new_err("prior_ss_e must be > 0"));
        }
    }
    if n_samples == 0 {
        return Err(PyValueError::new_err("n_samples must be > 0"));
    }

    let packed_arr = packed.as_array();
    if packed_arr.ndim() != 2 {
        return Err(PyValueError::new_err(
            "packed must be 2D (m, bytes_per_snp)",
        ));
    }
    let p = packed_arr.shape()[0];
    let bytes_per_snp = packed_arr.shape()[1];
    let expected_bps = (n_samples + 3) / 4;
    if bytes_per_snp != expected_bps {
        return Err(PyValueError::new_err(format!(
            "packed second dimension mismatch: got {bytes_per_snp}, expected {expected_bps} for n_samples={n_samples}"
        )));
    }

    let row_flip_vec: Cow<'_, [bool]> = match row_flip.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_flip.as_array().iter().copied().collect()),
    };
    let row_maf_vec: Cow<'_, [f32]> = match row_maf.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_maf.as_array().iter().copied().collect()),
    };
    let row_mean_vec: Cow<'_, [f32]> = match row_mean.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_mean.as_array().iter().copied().collect()),
    };
    let row_inv_sd_vec: Cow<'_, [f32]> = match row_inv_sd.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(row_inv_sd.as_array().iter().copied().collect()),
    };
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "row_flip/row_maf/row_mean/row_inv_sd length must match packed rows",
        ));
    }

    let y_vec: Cow<'_, [f64]> = match y.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(array1_to_vec(&y)),
    };
    let n = y_vec.len();
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != n {
        return Err(PyValueError::new_err(format!(
            "sample_indices length mismatch: got {}, expected len(y)={n}",
            sample_idx.len()
        )));
    }
    let trace_idx =
        parse_optional_index_vec_i64(trace_snp_indices.as_ref(), p, "trace_snp_indices")?;

    let (x_vec, q): (Cow<'_, [f64]>, usize) = match &x {
        Some(arr) => {
            let x_shape = arr.shape();
            if x_shape[0] != n {
                return Err(PyValueError::new_err("X rows must match len(y)"));
            }
            let q = x_shape[1];
            let xv = match arr.as_slice() {
                Ok(s) => Cow::Borrowed(s),
                Err(_) => Cow::Owned(array2_to_vec(arr)),
            };
            (xv, q)
        }
        None => (Cow::Owned(vec![1.0; n]), 1usize),
    };

    let packed_flat: Cow<'_, [u8]> = match packed.as_slice() {
        Ok(s) => Cow::Borrowed(s),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool_owned = get_cached_pool(threads)?;
    let pool_ref = pool_owned.as_ref();
    let result = py.detach(|| {
        bayesc_packed_trace_core_impl(
            y_vec.as_ref(),
            packed_flat.as_ref(),
            bytes_per_snp,
            n_samples,
            row_flip_vec.as_ref(),
            row_maf_vec.as_ref(),
            row_mean_vec.as_ref(),
            row_inv_sd_vec.as_ref(),
            &sample_idx,
            x_vec.as_ref(),
            n,
            p,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_b,
            s0_b,
            prob_in,
            counts,
            fixed_pi,
            df0_e,
            prior_ss_e,
            seed,
            &trace_idx,
            block_rows,
            pool_ref,
        )
    });
    match result {
        Ok(res) => packed_trace_result_to_pydict(py, res),
        Err(msg) => Err(PyValueError::new_err(msg)),
    }
}

#[cfg(test)]
mod backend_tests {
    use super::*;

    #[test]
    fn marker_conditional_update_matches_remove_dot_add_reference() {
        let marker = [0.0_f64, 1.0, 2.0, 1.0, 0.0];
        let residual = [1.5_f64, -0.5, 2.0, 0.25, -1.0];
        let old_beta = -0.37_f64;
        let new_beta = 0.82_f64;
        let d = ddot_f64(&marker, &marker);

        let mut reference_residual = residual;
        daxpy_inplace_f64(old_beta, &marker, &mut reference_residual);
        let reference_u = ddot_f64(&reference_residual, &marker);
        daxpy_inplace_f64(-new_beta, &marker, &mut reference_residual);

        let fast_u = marker_conditional_dot_f64(&residual, &marker, old_beta, d);
        let mut fast_residual = residual;
        update_marker_residual_f64(old_beta, new_beta, &marker, &mut fast_residual);

        assert!((fast_u - reference_u).abs() < 1e-12);
        for (fast, reference) in fast_residual.iter().zip(reference_residual.iter()) {
            assert!((fast - reference).abs() < 1e-12);
        }
    }

    #[test]
    fn bayes_decode_plan_uses_dense_when_one_block_covers_all_markers() {
        assert_eq!(bayes_decode_mode(4, 4), BayesDecodeMode::Dense);
        assert_eq!(bayes_decode_mode(3, 4), BayesDecodeMode::Dense);
        assert_eq!(bayes_decode_mode(5, 4), BayesDecodeMode::Double);
    }

    #[test]
    fn dense_backend_reads_requested_marker_block() {
        let matrix = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut backend = DenseBayesBackend::new(&matrix, 4, 2).expect("valid dense backend");
        assert_eq!(backend.n_samples(), 4);
        assert_eq!(backend.n_markers(), 2);
        assert_eq!(backend.block_rows(), 2);

        let mut block = vec![0.0_f32; 4];
        backend
            .fill_block(1, 2, &mut block, 4)
            .expect("valid dense block");
        assert_eq!(block, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn dense_backend_iterates_borrowed_marker_rows() {
        let matrix = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let matrix_ptr = matrix.as_ptr();
        let mut backend = DenseBayesBackend::new(&matrix, 4, 2).expect("valid dense backend");
        let mut blocks = 0usize;
        backend
            .for_each_block(|row_start, row_end, block| {
                assert_eq!(block.as_ptr(), unsafe { matrix_ptr.add(row_start * 4) });
                assert_eq!(block.len(), (row_end - row_start) * 4);
                blocks += 1;
                Ok(())
            })
            .expect("dense borrowed iteration succeeds");
        assert_eq!(blocks, 1);
    }

    #[test]
    fn mixed_precision_marker_helpers_match_f64_reference() {
        let marker_f32 = [0.0_f32, 1.0, 2.0, 1.0, 0.0];
        let marker_f64 = marker_f32.map(f64::from);
        let residual = [1.5_f64, -0.5, 2.0, 0.25, -1.0];
        let old_beta = -0.37_f64;
        let new_beta = 0.82_f64;
        let d = ddot_f64(&marker_f64, &marker_f64);

        let reference_u = marker_conditional_dot_f64(&residual, &marker_f64, old_beta, d);
        let mixed_u = marker_conditional_dot_f32_f64(&residual, &marker_f32, old_beta, d);
        assert!((mixed_u - reference_u).abs() < 1e-12);

        let mut reference_residual = residual;
        update_marker_residual_f64(old_beta, new_beta, &marker_f64, &mut reference_residual);
        let mut mixed_residual = residual;
        update_marker_residual_f32_f64(old_beta, new_beta, &marker_f32, &mut mixed_residual);
        for (mixed, reference) in mixed_residual.iter().zip(reference_residual.iter()) {
            assert!((mixed - reference).abs() < 1e-12);
        }
    }

    #[test]
    fn f32_residual_marker_helpers_match_f64_reference_with_rounding_bound() {
        let marker = [0.0_f32, 1.0, 2.0, 1.0, 0.0];
        let residual_f64 = [1.5_f64, -0.5, 2.0, 0.25, -1.0];
        let residual_f32 = residual_f64.map(|value| value as f32);
        let marker_f64 = marker.map(f64::from);
        let old_beta = -0.37_f64;
        let new_beta = 0.82_f64;
        let d = ddot_f64(&marker_f64, &marker_f64);

        let reference_u = marker_conditional_dot_f64(&residual_f64, &marker_f64, old_beta, d);
        let f32_u = marker_conditional_dot_f32_f32(&residual_f32, &marker, old_beta, d);
        assert!((f32_u - reference_u).abs() < 1e-6);

        let mut reference_residual = residual_f64;
        update_marker_residual_f64(old_beta, new_beta, &marker_f64, &mut reference_residual);
        let mut f32_out = residual_f32;
        update_marker_residual_f32(old_beta, new_beta, &marker, &mut f32_out);
        for (actual, expected) in f32_out.iter().zip(reference_residual.iter()) {
            assert!((*actual as f64 - *expected).abs() < 2e-6);
        }
    }

    #[test]
    fn fast_f32_marker_dot_matches_f64_reference_with_float32_bound() {
        let marker = [
            0.03125_f32,
            1.0,
            2.0,
            1.0,
            -0.1171875,
            -0.5,
            0.25,
            0.6875,
            -1.375,
            0.203125,
            0.8125,
        ];
        let residual = [
            1.5_f32,
            -0.5,
            2.0,
            0.25,
            -1.0,
            0.75,
            -0.125,
            0.33333334,
            -0.7777778,
            1.2345679,
            -0.44444445,
        ];
        let marker_f64 = marker.map(f64::from);
        let residual_f64 = residual.map(f64::from);
        let marker_ss = marker_f64.iter().map(|v| v * v).sum::<f64>();
        let expected = marker_conditional_dot_f64(&residual_f64, &marker_f64, -0.37, marker_ss);
        let actual = marker_conditional_dot_f32_f32_fast(&residual, &marker, -0.37, marker_ss);
        assert!((actual - expected).abs() < 1e-10);
    }

    #[test]
    fn blocked_f32_marker_dot_matches_f64_reference() {
        let marker: Vec<f32> = (0..257).map(|i| ((i as f32) * 0.03125).sin()).collect();
        let residual: Vec<f32> = (0..257)
            .map(|i| ((i as f32) * 0.017).cos() * 0.75)
            .collect();
        let marker_f64: Vec<f64> = marker.iter().copied().map(f64::from).collect();
        let residual_f64: Vec<f64> = residual.iter().copied().map(f64::from).collect();
        let marker_ss = marker_f64.iter().map(|v| v * v).sum::<f64>();
        let expected = marker_conditional_dot_f64(&residual_f64, &marker_f64, -0.37, marker_ss);
        let actual = marker_conditional_dot_f32_f32_blocked(&residual, &marker, -0.37, marker_ss);
        assert!((actual - expected).abs() < 2e-5);
    }

    #[test]
    fn fused_marker_update_matches_dot_then_axpy_reference() {
        let marker = [0.0_f32, 1.0, 2.0, 1.0, 0.0, -0.5, 0.25, 1.25];
        let initial = [1.5_f32, -0.5, 2.0, 0.25, -1.0, 0.75, -0.125, 0.33333334];
        let old_beta = -0.37_f64;
        let marker_ss = marker
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>();
        let expected_u =
            marker_conditional_dot_f32_f32_blocked(&initial, &marker, old_beta, marker_ss);
        let expected_beta = 0.125 + 0.03125 * expected_u;
        let mut expected_residual = initial;
        update_marker_residual_f32(old_beta, expected_beta, &marker, &mut expected_residual);

        let mut actual_residual = initial;
        let (actual_u, actual_beta) = bayes_marker_update_f32_fused(
            &mut actual_residual,
            &marker,
            old_beta,
            marker_ss,
            |u| 0.125 + 0.03125 * u,
        );
        assert_eq!(actual_u, expected_u);
        assert_eq!(actual_beta, expected_beta);
        assert_eq!(actual_residual, expected_residual);
    }

    #[test]
    fn dense_f32_input_keeps_contiguous_python_storage_borrowed() {
        Python::initialize();
        Python::attach(|py| {
            let array = numpy::PyArray2::<f32>::zeros(py, (2, 3), false);
            let any = array.as_any();
            let input = array2_to_f32_input(&any, "M").expect("valid float32 matrix");
            assert!(input.is_borrowed());
            assert_eq!(input.rows(), 2);
            assert_eq!(input.cols(), 3);
            assert_eq!(input.as_slice(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        });
    }

    #[test]
    fn dense_f64_input_uses_owned_float32_fallback() {
        Python::initialize();
        Python::attach(|py| {
            let array = numpy::PyArray2::<f64>::zeros(py, (2, 3), false);
            let any = array.as_any();
            let input = array2_to_f32_input(&any, "M").expect("valid float64 matrix");
            assert!(!input.is_borrowed());
            assert_eq!(input.rows(), 2);
            assert_eq!(input.cols(), 3);
            assert_eq!(input.as_slice(), &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        });
    }

    #[test]
    fn unified_marker_update_returns_score_and_updates_residual() {
        let marker = [0.0_f32, 1.0, 2.0, 1.0, 0.0];
        let mut residual = [1.5_f32, -0.5, 2.0, 0.25, -1.0];
        let old_beta = -0.37_f64;
        let new_beta = 0.82_f64;
        let marker_ss = marker
            .iter()
            .map(|value| f64::from(*value) * f64::from(*value))
            .sum::<f64>();
        let expected_u = marker_conditional_dot_f32_f32(&residual, &marker, old_beta, marker_ss);
        let (u, actual_beta) =
            bayes_marker_update_f32(&mut residual, &marker, old_beta, marker_ss, |_| new_beta);
        assert_eq!(actual_beta, new_beta);
        assert_eq!(u, expected_u);
    }

    #[test]
    fn zero_marker_update_is_a_noop() {
        let marker = [0.0_f32, 1.0, 2.0, 1.0, 0.0];
        let mut residual = [1.5_f32, -0.5, 2.0, 0.25, -1.0];
        let expected = residual;
        update_marker_residual_f32(0.0, 0.0, &marker, &mut residual);
        assert_eq!(residual, expected);
    }

    #[test]
    fn packed_backend_rejects_incomplete_row_metadata() {
        let packed = [0_u8; 2];
        let mut source = BayesPackedSource::Resident {
            packed_flat: &packed,
            bytes_per_snp: 1,
        };
        let result = PackedBayesBackend::new(
            &mut source,
            4,
            2,
            &[false],
            &[0.0, 0.0],
            &[1.0, 1.0],
            None,
            &[0, 1],
            None,
            None,
        );
        assert!(result.is_err());
    }

    #[test]
    fn packed_backend_double_buffer_preserves_marker_order() {
        // Four samples, three SNP rows, two-bit BED codes.  The values are
        // intentionally distinct so an out-of-order prefetch is observable.
        let packed = [0b00_11_10_00_u8, 0b11_00_10_11_u8, 0b10_10_10_10_u8];
        let mut source = BayesPackedSource::Resident {
            packed_flat: &packed,
            bytes_per_snp: 1,
        };
        let mut backend = PackedBayesBackend::new(
            &mut source,
            4,
            3,
            &[false, false, false],
            &[0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0],
            None,
            &[0, 1, 2, 3],
            Some(1),
            None,
        )
        .expect("valid packed backend");

        let mut starts = Vec::new();
        let mut rows = Vec::new();
        backend
            .for_each_block(|start, end, block| {
                starts.push((start, end));
                rows.push(block.to_vec());
                Ok(())
            })
            .expect("double-buffered scan succeeds");

        assert_eq!(starts, vec![(0, 1), (1, 2), (2, 3)]);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], vec![0.0, 1.0, 2.0, 0.0]);
        assert_eq!(rows[1], vec![2.0, 1.0, 0.0, 2.0]);
        assert_eq!(rows[2], vec![1.0, 1.0, 1.0, 1.0]);
    }
}
