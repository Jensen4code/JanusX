//! BayesR marker-effect sampler.
//!
//! The residual inverse-chi-square prior accepts `prior_ss_e = nu_0 * S_0^2`
//! (the prior sum-of-squares term), rather than the scale `S_0^2` alone.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::types::PyDict;
use pyo3::{prelude::*, BoundObject};
use rand::rngs::{OsRng, StdRng};
use rand::{Rng, SeedableRng, TryRngCore};
use rand_distr::{ChiSquared, Gamma, StandardNormal};
use std::borrow::Cow;
use std::sync::Arc;

use crate::bayes::{
    bayes_packed_blas_threads, bayes_packed_block_rows, ddot_f64, genetic_variance_from_residual,
    marker_conditional_dot_f64, update_alpha_gauss_seidel_blas, update_marker_residual_f64,
    BayesMarkerBackend, BayesPackedSource, BayesSamplingController, DenseBayesBackend,
    PackedBayesBackend, BAYES_POSTERIOR_SAMPLES,
};
use crate::blas::OpenBlasThreadGuard;
use crate::stats_common::{get_cached_pool, parse_index_vec_i64_value_error};

const BAYESR_COMPONENTS: usize = 4;

fn validate_bayesr_prior(pi: &[f64], gamma: &[f64]) -> Result<(), String> {
    if pi.len() != BAYESR_COMPONENTS || gamma.len() != BAYESR_COMPONENTS {
        return Err(format!(
            "BayesR requires exactly {BAYESR_COMPONENTS} pi and gamma values"
        ));
    }
    if !pi.iter().all(|v| v.is_finite() && *v > 0.0) {
        return Err("BayesR pi values must be finite and > 0".to_string());
    }
    let pi_sum: f64 = pi.iter().sum();
    if !pi_sum.is_finite() || pi_sum <= 0.0 {
        return Err("BayesR pi values must have a positive finite sum".to_string());
    }
    if !gamma[0].is_finite() || gamma[0].abs() > 1e-15 {
        return Err("BayesR gamma[0] must be zero for the spike component".to_string());
    }
    if !gamma[1..].iter().all(|v| v.is_finite() && *v > 0.0) {
        return Err("BayesR non-spike gamma values must be finite and > 0".to_string());
    }
    Ok(())
}

fn component_log_weights(
    pi: &[f64],
    gamma: &[f64],
    u: f64,
    d: f64,
    sigma_lambda2: f64,
    sigma_e2: f64,
) -> Result<[f64; BAYESR_COMPONENTS], String> {
    validate_bayesr_prior(pi, gamma)?;
    if !(u.is_finite() && d.is_finite() && d >= 0.0) {
        return Err("BayesR marker sufficient statistics must be finite and d >= 0".to_string());
    }
    if !(sigma_lambda2.is_finite() && sigma_lambda2 > 0.0) {
        return Err("BayesR sigma_lambda2 must be finite and > 0".to_string());
    }
    if !(sigma_e2.is_finite() && sigma_e2 > 0.0) {
        return Err("BayesR sigma_e2 must be finite and > 0".to_string());
    }
    let mut out = [0.0; BAYESR_COMPONENTS];
    out[0] = pi[0].ln();
    for k in 1..BAYESR_COMPONENTS {
        let tau2 = gamma[k] * sigma_lambda2;
        let denom = sigma_e2 + tau2 * d;
        out[k] = pi[k].ln() - 0.5 * (1.0 + tau2 * d / sigma_e2).ln()
            + tau2 * u * u / (2.0 * sigma_e2 * denom);
    }
    Ok(out)
}

fn normalize_component_log_weights(
    log_weights: &[f64],
) -> Result<[f64; BAYESR_COMPONENTS], String> {
    if log_weights.len() != BAYESR_COMPONENTS || !log_weights.iter().all(|v| v.is_finite()) {
        return Err("BayesR component log weights must contain four finite values".to_string());
    }
    let max_log = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut q = [0.0; BAYESR_COMPONENTS];
    for (dst, value) in q.iter_mut().zip(log_weights.iter()) {
        *dst = (*value - max_log).exp();
    }
    let total: f64 = q.iter().sum();
    if !(total.is_finite() && total > 0.0) {
        return Err("BayesR component weights are not normalizable".to_string());
    }
    for value in &mut q {
        *value /= total;
    }
    Ok(q)
}

#[inline]
fn sample_component<R: Rng + ?Sized>(q: &[f64; BAYESR_COMPONENTS], rng: &mut R) -> usize {
    let draw = rng.random::<f64>();
    let mut cumulative = 0.0;
    for (k, probability) in q.iter().enumerate() {
        cumulative += *probability;
        if draw < cumulative || k + 1 == BAYESR_COMPONENTS {
            return k;
        }
    }
    BAYESR_COMPONENTS - 1
}

struct BayesRResult {
    beta: Vec<f64>,
    alpha: Vec<f64>,
    varbeta: Vec<f64>,
    vare: f64,
    h2_mean: f64,
    var_h2: f64,
    pip: Vec<f64>,
    component_prob: Vec<f32>,
    pi: Vec<f64>,
    sigma_lambda2: f64,
    rhat_h2: f64,
    actual_iterations: usize,
    convergence_iteration: usize,
    posterior_samples: usize,
}

#[allow(clippy::too_many_arguments)]
fn bayesr_core_impl<B: BayesMarkerBackend>(
    backend: &mut B,
    y: &[f64],
    x: &[f64],
    q: usize,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_e: f64,
    prior_ss_e_opt: Option<f64>,
    pi_init: &[f64],
    gamma: &[f64],
    df0_lambda: f64,
    s0_lambda2: f64,
    seed: Option<u64>,
) -> Result<BayesRResult, String> {
    let n = backend.n_samples();
    let p = backend.n_markers();
    if n <= 1 {
        return Err("BayesR requires n > 1".to_string());
    }
    if y.len() != n {
        return Err("BayesR phenotype length does not match genotype samples".to_string());
    }
    if x.len() != n.saturating_mul(q) || q == 0 {
        return Err("BayesR covariate dimensions are incompatible".to_string());
    }
    if n_iter == 0 || thin == 0 {
        return Err("BayesR n_iter must be > 0 and thin must be >= 1".to_string());
    }
    if !(r2.is_finite() && r2 > 0.0 && r2 < 1.0) {
        return Err("BayesR r2 must be finite and in (0, 1)".to_string());
    }
    if !(df0_e.is_finite() && df0_e > 0.0) {
        return Err("BayesR df0_e must be finite and > 0".to_string());
    }
    if !(df0_lambda.is_finite() && df0_lambda > 0.0) {
        return Err("BayesR df0_lambda must be finite and > 0".to_string());
    }
    if !(s0_lambda2.is_finite() && s0_lambda2 > 0.0) {
        return Err("BayesR s0_lambda2 must be finite and > 0".to_string());
    }
    validate_bayesr_prior(pi_init, gamma)?;

    let mut rng = match seed {
        Some(value) => StdRng::seed_from_u64(value),
        None => {
            let mut seed_bytes = [0u8; 32];
            if let Err(error) = OsRng.try_fill_bytes(&mut seed_bytes) {
                eprintln!("Failed to generate random seed: {error}; using a fixed fallback");
                seed_bytes = [42u8; 32];
            }
            StdRng::from_seed(seed_bytes)
        }
    };

    let mut x2 = vec![0.0_f64; p];
    let mut mean_x = vec![0.0_f64; p];
    backend.for_each_block(|row_start, row_end, marker_block| {
        for offset in 0..(row_end - row_start) {
            let row = &marker_block[offset * n..(offset + 1) * n];
            let mut sum_sq = 0.0;
            let mut sum = 0.0;
            for &value in row {
                sum_sq += value * value;
                sum += value;
            }
            x2[row_start + offset] = sum_sq;
            mean_x[row_start + offset] = sum / n as f64;
        }
        Ok(())
    })?;
    let msx =
        x2.iter().sum::<f64>() / n as f64 - mean_x.iter().map(|value| value * value).sum::<f64>();
    if !(msx.is_finite() && msx > 0.0) {
        return Err("BayesR marker mean square must be positive".to_string());
    }

    let y_mean = y.iter().sum::<f64>() / n as f64;
    let var_y = y
        .iter()
        .map(|value| {
            let delta = *value - y_mean;
            delta * delta
        })
        .sum::<f64>()
        / (n - 1) as f64;
    if !(var_y.is_finite() && var_y > 0.0) {
        return Err("BayesR phenotype variance must be positive".to_string());
    }

    let mut pi = [0.0_f64; BAYESR_COMPONENTS];
    let pi_total = pi_init.iter().sum::<f64>();
    for (dst, value) in pi.iter_mut().zip(pi_init.iter()) {
        *dst = *value / pi_total;
    }
    let gamma_array: [f64; BAYESR_COMPONENTS] = [gamma[0], gamma[1], gamma[2], gamma[3]];
    let mut sigma_lambda2 = s0_lambda2;
    let mut var_e = var_y * (1.0 - r2);
    if !(var_e.is_finite() && var_e > 0.0) {
        return Err("BayesR initial residual variance must be positive".to_string());
    }
    let prior_ss_e = prior_ss_e_opt.unwrap_or(var_e * (df0_e + 2.0));
    if !(prior_ss_e.is_finite() && prior_ss_e > 0.0) {
        return Err("BayesR prior_ss_e must be finite and > 0 (nu_0 * S_0^2)".to_string());
    }

    let mut alpha = vec![0.0_f64; q];
    let mut x2_x = vec![0.0_f64; q];
    for k in 0..q {
        let mut sum = 0.0;
        for i in 0..n {
            let value = x[i * q + k];
            sum += value * value;
        }
        x2_x[k] = sum;
    }
    let mut xtx = vec![0.0_f64; q * q];
    for i in 0..n {
        for a in 0..q {
            let xa = x[i * q + a];
            for b in 0..q {
                xtx[a * q + b] += xa * x[i * q + b];
            }
        }
    }
    let mut alpha_xtr = vec![0.0_f64; q];
    let mut alpha_delta = vec![0.0_f64; q];
    let mut alpha_tmp_n = vec![0.0_f64; n];

    let mut beta = vec![0.0_f64; p];
    let mut component = vec![0usize; p];
    let mut residual = y.to_vec();
    let mut beta_sum = vec![0.0_f64; p];
    let mut beta_second_sum = vec![0.0_f64; p];
    let mut pip_sum = vec![0.0_f64; p];
    let mut component_sum = vec![0.0_f32; p.saturating_mul(BAYESR_COMPONENTS)];
    let mut alpha_sum = vec![0.0_f64; q];
    let mut var_e_sum = 0.0_f64;
    let mut sigma_lambda2_sum = 0.0_f64;
    let mut pi_sum = [0.0_f64; BAYESR_COMPONENTS];
    let mut h2_sum = 0.0_f64;
    let mut h2_sq_sum = 0.0_f64;
    let mut schedule = BayesSamplingController::new(n_iter, burnin, thin);
    let chi_e = ChiSquared::new(n as f64 + df0_e).map_err(|error| error.to_string())?;

    while schedule.should_run() {
        let retain_sample = schedule.begin_iteration();
        let inv_var_e = 1.0 / var_e;
        update_alpha_gauss_seidel_blas(
            x,
            n,
            q,
            inv_var_e,
            1.0e-10,
            &x2_x,
            &xtx,
            &mut alpha,
            &mut residual,
            &mut alpha_xtr,
            &mut alpha_delta,
            &mut alpha_tmp_n,
            &mut rng,
        );

        backend.for_each_block(|row_start, row_end, marker_block| {
            for offset in 0..(row_end - row_start) {
                let j = row_start + offset;
                let marker = &marker_block[offset * n..(offset + 1) * n];
                let old_beta = beta[j];
                let u = marker_conditional_dot_f64(&residual, marker, old_beta, x2[j]);
                let log_weights =
                    component_log_weights(&pi, &gamma_array, u, x2[j], sigma_lambda2, var_e)?;
                let q_component = normalize_component_log_weights(&log_weights)?;
                let selected = sample_component(&q_component, &mut rng);
                component[j] = selected;
                let new_beta = if selected == 0 {
                    0.0
                } else {
                    let tau2 = gamma_array[selected] * sigma_lambda2;
                    let precision = x2[j] * inv_var_e + 1.0 / tau2;
                    let posterior_var = 1.0 / precision;
                    let posterior_mean = posterior_var * u * inv_var_e;
                    let z: f64 = rng.sample(StandardNormal);
                    posterior_mean + posterior_var.sqrt() * z
                };
                beta[j] = new_beta;
                update_marker_residual_f64(old_beta, new_beta, marker, &mut residual);

                if retain_sample {
                    beta_sum[j] += new_beta;
                    beta_second_sum[j] += new_beta * new_beta;
                    pip_sum[j] += 1.0 - q_component[0];
                    for k in 0..BAYESR_COMPONENTS {
                        component_sum[j * BAYESR_COMPONENTS + k] += q_component[k] as f32;
                    }
                }
            }
            Ok(())
        })?;

        let mut active_count = 0usize;
        let mut scaled_sum = 0.0_f64;
        let mut component_counts = [0usize; BAYESR_COMPONENTS];
        for j in 0..p {
            let selected = component[j];
            component_counts[selected] += 1;
            if selected > 0 {
                active_count += 1;
                scaled_sum += beta[j] * beta[j] / gamma_array[selected];
            }
        }
        let lambda_df = df0_lambda + active_count as f64;
        let lambda_numerator = df0_lambda * s0_lambda2 + scaled_sum;
        sigma_lambda2 =
            lambda_numerator / rng.sample(ChiSquared::new(lambda_df).map_err(|e| e.to_string())?);
        if !(sigma_lambda2.is_finite() && sigma_lambda2 > 0.0) {
            return Err("BayesR sigma_lambda2 became non-finite or non-positive".to_string());
        }

        let mut dirichlet_draw = [0.0_f64; BAYESR_COMPONENTS];
        let mut dirichlet_total = 0.0_f64;
        for k in 0..BAYESR_COMPONENTS {
            let shape = 1.0 + component_counts[k] as f64;
            let draw = rng.sample(Gamma::new(shape, 1.0).map_err(|e| e.to_string())?);
            dirichlet_draw[k] = draw;
            dirichlet_total += draw;
        }
        for k in 0..BAYESR_COMPONENTS {
            pi[k] = dirichlet_draw[k] / dirichlet_total;
        }

        var_e = (ddot_f64(&residual, &residual) + prior_ss_e) / rng.sample(chi_e);
        if !(var_e.is_finite() && var_e > 0.0) {
            return Err("BayesR residual variance became non-finite or non-positive".to_string());
        }

        if retain_sample {
            for k in 0..BAYESR_COMPONENTS {
                pi_sum[k] += pi[k];
            }
            for k in 0..q {
                alpha_sum[k] += alpha[k];
            }
            var_e_sum += var_e;
            sigma_lambda2_sum += sigma_lambda2;
            let var_g = genetic_variance_from_residual(y, &residual, x, &alpha, n, q);
            let h2 = var_g / (var_g + var_e);
            h2_sum += h2;
            h2_sq_sum += h2 * h2;
            if schedule.observe(h2) {
                beta_sum.fill(0.0);
                beta_second_sum.fill(0.0);
                pip_sum.fill(0.0);
                component_sum.fill(0.0);
                alpha_sum.fill(0.0);
                var_e_sum = 0.0;
                sigma_lambda2_sum = 0.0;
                pi_sum = [0.0; BAYESR_COMPONENTS];
                h2_sum = 0.0;
                h2_sq_sum = 0.0;
            }
        }
    }

    let n_keep = schedule.posterior_samples();
    if n_keep != BAYES_POSTERIOR_SAMPLES {
        return Err(format!(
            "BayesR retained {n_keep} posterior samples; expected {BAYES_POSTERIOR_SAMPLES}"
        ));
    }
    let inv_keep = 1.0 / n_keep as f64;
    let mut varbeta = vec![0.0_f64; p];
    for j in 0..p {
        beta_sum[j] *= inv_keep;
        let second = beta_second_sum[j] * inv_keep;
        varbeta[j] = (second - beta_sum[j] * beta_sum[j]).max(0.0);
        pip_sum[j] *= inv_keep;
    }
    for value in &mut component_sum {
        *value *= inv_keep as f32;
    }
    for value in &mut alpha_sum {
        *value *= inv_keep;
    }
    let h2_mean = h2_sum * inv_keep;
    let var_h2 = (h2_sq_sum * inv_keep - h2_mean * h2_mean).max(0.0);
    let pi_mean = pi_sum
        .iter()
        .map(|value| value * inv_keep)
        .collect::<Vec<_>>();
    Ok(BayesRResult {
        beta: beta_sum,
        alpha: alpha_sum,
        varbeta,
        vare: var_e_sum * inv_keep,
        h2_mean,
        var_h2,
        pip: pip_sum,
        component_prob: component_sum,
        pi: pi_mean,
        sigma_lambda2: sigma_lambda2_sum * inv_keep,
        rhat_h2: schedule.rhat_value(),
        actual_iterations: schedule.actual_iterations(),
        convergence_iteration: schedule.convergence_iteration(),
        posterior_samples: n_keep,
    })
}

const BAYESR_DEFAULT_PI: [f64; BAYESR_COMPONENTS] = [0.90, 0.06, 0.03, 0.01];
const BAYESR_DEFAULT_GAMMA: [f64; BAYESR_COMPONENTS] = [0.0, 0.01, 0.1, 1.0];

fn array1_to_vec(arr: &PyReadonlyArray1<f64>) -> Vec<f64> {
    arr.as_array().iter().copied().collect()
}

fn array2_to_vec(arr: &PyReadonlyArray2<f64>) -> Vec<f64> {
    let view = arr.as_array();
    let (n, p) = view.dim();
    let mut out = Vec::with_capacity(n.saturating_mul(p));
    for i in 0..n {
        for j in 0..p {
            out.push(view[[i, j]]);
        }
    }
    out
}

fn parse_covariates(x: Option<&PyReadonlyArray2<f64>>, n: usize) -> PyResult<(Vec<f64>, usize)> {
    match x {
        Some(arr) => {
            let shape = arr.shape();
            if shape.len() != 2 || shape[0] != n || shape[1] == 0 {
                return Err(PyValueError::new_err(
                    "BayesR X must be a non-empty matrix with rows matching len(y)",
                ));
            }
            let values = if let Ok(slice) = arr.as_slice() {
                slice.to_vec()
            } else {
                array2_to_vec(arr)
            };
            Ok((values, shape[1]))
        }
        None => Ok((vec![1.0; n], 1usize)),
    }
}

fn parse_bayesr_prior(
    values: Option<&PyReadonlyArray1<f64>>,
    default: &[f64; BAYESR_COMPONENTS],
    label: &str,
) -> PyResult<Vec<f64>> {
    let out = match values {
        Some(arr) => array1_to_vec(arr),
        None => default.to_vec(),
    };
    validate_bayesr_prior(
        if label == "pi" {
            &out
        } else {
            &BAYESR_DEFAULT_PI
        },
        if label == "gamma" {
            &out
        } else {
            &BAYESR_DEFAULT_GAMMA
        },
    )
    .map_err(PyValueError::new_err)?;
    Ok(out)
}

fn bayesr_result_to_pydict<'py>(
    py: Python<'py>,
    result: BayesRResult,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py).into_bound();
    let beta = result.beta.into_pyarray(py);
    let alpha = result.alpha.into_pyarray(py);
    let varbeta = result.varbeta.into_pyarray(py);
    let pip = result.pip.into_pyarray(py);
    let pi = result.pi.into_pyarray(py);
    let component_prob =
        Array2::from_shape_vec((pip.len(), BAYESR_COMPONENTS), result.component_prob).map_err(
            |error| {
                PyValueError::new_err(format!(
                    "invalid BayesR component probability shape: {error}"
                ))
            },
        )?;
    let component_prob = PyArray2::from_owned_array(py, component_prob);
    out.set_item("beta", beta)?;
    out.set_item("alpha", alpha)?;
    out.set_item("varbeta", varbeta)?;
    out.set_item("vare", result.vare)?;
    out.set_item("h2_mean", result.h2_mean)?;
    out.set_item("var_h2", result.var_h2)?;
    out.set_item("pip", pip)?;
    out.set_item("component_prob", component_prob)?;
    out.set_item("pi", pi)?;
    out.set_item("sigma_lambda2", result.sigma_lambda2)?;
    out.set_item("rhat_h2", result.rhat_h2)?;
    out.set_item("actual_iterations", result.actual_iterations)?;
    out.set_item("convergence_iteration", result.convergence_iteration)?;
    out.set_item("posterior_samples", result.posterior_samples)?;
    Ok(out)
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
    df0_e = 5.0,
    prior_ss_e = None,
    pi = None,
    gamma = None,
    df0_lambda = 1.0,
    s0_lambda2 = 1.0,
    seed = None
))]
pub fn bayesr<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    m: PyReadonlyArray2<'py, f64>,
    x: Option<PyReadonlyArray2<'py, f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    pi: Option<PyReadonlyArray1<'py, f64>>,
    gamma: Option<PyReadonlyArray1<'py, f64>>,
    df0_lambda: f64,
    s0_lambda2: f64,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyDict>> {
    let y_vec = if let Ok(slice) = y.as_slice() {
        slice.to_vec()
    } else {
        array1_to_vec(&y)
    };
    let m_shape = m.shape();
    if m_shape.len() != 2 || m_shape[1] != y_vec.len() {
        return Err(PyValueError::new_err("BayesR M cols must match len(y)"));
    }
    let m_vec = if let Ok(slice) = m.as_slice() {
        slice.to_vec()
    } else {
        array2_to_vec(&m)
    };
    let (x_vec, q) = match x.as_ref() {
        Some(arr) => {
            let shape = arr.shape();
            if shape.len() != 2 || shape[0] != y_vec.len() {
                return Err(PyValueError::new_err("BayesR X rows must match len(y)"));
            }
            let values = if let Ok(slice) = arr.as_slice() {
                slice.to_vec()
            } else {
                array2_to_vec(arr)
            };
            (values, shape[1])
        }
        None => (vec![1.0; y_vec.len()], 1usize),
    };
    let pi_vec = parse_bayesr_prior(pi.as_ref(), &BAYESR_DEFAULT_PI, "pi")?;
    let gamma_vec = parse_bayesr_prior(gamma.as_ref(), &BAYESR_DEFAULT_GAMMA, "gamma")?;
    let result = py.detach(|| {
        let mut backend = DenseBayesBackend::new(&m_vec, y_vec.len(), m_shape[0])?;
        bayesr_core_impl(
            &mut backend,
            &y_vec,
            &x_vec,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_e,
            prior_ss_e,
            &pi_vec,
            &gamma_vec,
            df0_lambda,
            s0_lambda2,
            seed,
        )
    });
    match result {
        Ok(result) => bayesr_result_to_pydict(py, result),
        Err(error) => Err(PyValueError::new_err(error)),
    }
}

#[allow(clippy::too_many_arguments)]
fn bayesr_packed_core_impl<'a>(
    source: &'a mut BayesPackedSource<'a>,
    n_samples_total: usize,
    y: &[f64],
    row_flip: &'a [bool],
    row_mean: &'a [f32],
    row_inv_sd: &'a [f32],
    packed_row_indices: Option<&'a [usize]>,
    sample_indices: &'a [usize],
    x: &[f64],
    q: usize,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    pi_init: &[f64],
    gamma: &[f64],
    df0_lambda: f64,
    s0_lambda2: f64,
    seed: Option<u64>,
    block_rows: Option<usize>,
    pool: Option<&'a Arc<rayon::ThreadPool>>,
) -> Result<BayesRResult, String> {
    let p = row_flip.len();
    let mut backend = PackedBayesBackend::new(
        source,
        n_samples_total,
        p,
        row_flip,
        row_mean,
        row_inv_sd,
        packed_row_indices,
        sample_indices,
        block_rows,
        pool,
    )?;
    // Keep the backend policy shared with BayesA/B/C: when the memory budget
    // supplies a block covering every active marker, decode once and run the
    // dense sampler rather than repeatedly decoding the same BED rows for
    // every MCMC iteration.
    if let Some(m_dense) = backend.maybe_predecode_dense_f64()? {
        let mut dense_backend = DenseBayesBackend::new(&m_dense, sample_indices.len(), p)?;
        return bayesr_core_impl(
            &mut dense_backend,
            y,
            x,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_e,
            prior_ss_e,
            pi_init,
            gamma,
            df0_lambda,
            s0_lambda2,
            seed,
        );
    }
    bayesr_core_impl(
        &mut backend,
        y,
        x,
        q,
        n_iter,
        burnin,
        thin,
        r2,
        df0_e,
        prior_ss_e,
        pi_init,
        gamma,
        df0_lambda,
        s0_lambda2,
        seed,
    )
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
    df0_e = 5.0,
    prior_ss_e = None,
    pi = None,
    gamma = None,
    df0_lambda = 1.0,
    s0_lambda2 = 1.0,
    threads = 0,
    seed = None,
    block_rows = None
))]
pub fn bayesr_packed<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    packed: PyReadonlyArray2<'py, u8>,
    n_samples: usize,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    row_mean: PyReadonlyArray1<'py, f32>,
    row_inv_sd: PyReadonlyArray1<'py, f32>,
    sample_indices: PyReadonlyArray1<'py, i64>,
    x: Option<PyReadonlyArray2<'py, f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    pi: Option<PyReadonlyArray1<'py, f64>>,
    gamma: Option<PyReadonlyArray1<'py, f64>>,
    df0_lambda: f64,
    s0_lambda2: f64,
    threads: usize,
    seed: Option<u64>,
    block_rows: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    if n_samples == 0 {
        return Err(PyValueError::new_err("BayesR n_samples must be > 0"));
    }
    let packed_arr = packed.as_array();
    let shape = packed_arr.shape();
    if shape.len() != 2 || shape[1] != n_samples.div_ceil(4) {
        return Err(PyValueError::new_err(
            "BayesR packed shape is incompatible with n_samples",
        ));
    }
    let p = shape[0];
    if p == 0 {
        return Err(PyValueError::new_err(
            "BayesR packed genotype has no markers",
        ));
    }
    let row_flip_vec = row_flip.as_array().iter().copied().collect::<Vec<_>>();
    let row_maf_vec = row_maf.as_array().iter().copied().collect::<Vec<_>>();
    let row_mean_vec = row_mean.as_array().iter().copied().collect::<Vec<_>>();
    let row_inv_sd_vec = row_inv_sd.as_array().iter().copied().collect::<Vec<_>>();
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "BayesR packed row metadata must match packed marker count",
        ));
    }
    let y_vec = if let Ok(slice) = y.as_slice() {
        slice.to_vec()
    } else {
        array1_to_vec(&y)
    };
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != y_vec.len() {
        return Err(PyValueError::new_err(
            "BayesR sample_indices length must match len(y)",
        ));
    }
    let (x_vec, q) = parse_covariates(x.as_ref(), y_vec.len())?;
    let pi_vec = parse_bayesr_prior(pi.as_ref(), &BAYESR_DEFAULT_PI, "pi")?;
    let gamma_vec = parse_bayesr_prior(gamma.as_ref(), &BAYESR_DEFAULT_GAMMA, "gamma")?;
    let packed_flat: Cow<'_, [u8]> = match packed.as_slice() {
        Ok(slice) => Cow::Borrowed(slice),
        Err(_) => Cow::Owned(packed_arr.iter().copied().collect()),
    };
    let pool_owned = get_cached_pool(threads)?;
    let pool = pool_owned.as_ref();
    let result = py.detach(|| {
        let _blas_guard = OpenBlasThreadGuard::enter(bayes_packed_blas_threads());
        let mut source = BayesPackedSource::Resident {
            packed_flat: packed_flat.as_ref(),
            bytes_per_snp: n_samples.div_ceil(4),
        };
        bayesr_packed_core_impl(
            &mut source,
            n_samples,
            &y_vec,
            &row_flip_vec,
            &row_mean_vec,
            &row_inv_sd_vec,
            None,
            &sample_idx,
            &x_vec,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_e,
            prior_ss_e,
            &pi_vec,
            &gamma_vec,
            df0_lambda,
            s0_lambda2,
            seed,
            block_rows,
            pool,
        )
    });
    match result {
        Ok(result) => bayesr_result_to_pydict(py, result),
        Err(error) => Err(PyValueError::new_err(error)),
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
    df0_e = 5.0,
    prior_ss_e = None,
    pi = None,
    gamma = None,
    df0_lambda = 1.0,
    s0_lambda2 = 1.0,
    threads = 0,
    seed = None,
    block_rows = None,
    mmap_window_mb = None
))]
pub fn bayesr_stream_bed<'py>(
    py: Python<'py>,
    prefix: String,
    y: PyReadonlyArray1<'py, f64>,
    n_samples: usize,
    row_indices: PyReadonlyArray1<'py, i64>,
    row_flip: PyReadonlyArray1<'py, bool>,
    row_maf: PyReadonlyArray1<'py, f32>,
    row_mean: PyReadonlyArray1<'py, f32>,
    row_inv_sd: PyReadonlyArray1<'py, f32>,
    sample_indices: PyReadonlyArray1<'py, i64>,
    x: Option<PyReadonlyArray2<'py, f64>>,
    n_iter: usize,
    burnin: usize,
    thin: usize,
    r2: f64,
    df0_e: f64,
    prior_ss_e: Option<f64>,
    pi: Option<PyReadonlyArray1<'py, f64>>,
    gamma: Option<PyReadonlyArray1<'py, f64>>,
    df0_lambda: f64,
    s0_lambda2: f64,
    threads: usize,
    seed: Option<u64>,
    block_rows: Option<usize>,
    mmap_window_mb: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    if n_samples == 0 {
        return Err(PyValueError::new_err("BayesR n_samples must be > 0"));
    }
    let row_indices_raw = row_indices.as_slice()?.to_vec();
    let p = row_indices_raw.len();
    if p == 0 {
        return Err(PyValueError::new_err(
            "BayesR row_indices must not be empty",
        ));
    }
    let row_flip_vec = row_flip.as_array().iter().copied().collect::<Vec<_>>();
    let row_maf_vec = row_maf.as_array().iter().copied().collect::<Vec<_>>();
    let row_mean_vec = row_mean.as_array().iter().copied().collect::<Vec<_>>();
    let row_inv_sd_vec = row_inv_sd.as_array().iter().copied().collect::<Vec<_>>();
    if row_flip_vec.len() != p
        || row_maf_vec.len() != p
        || row_mean_vec.len() != p
        || row_inv_sd_vec.len() != p
    {
        return Err(PyValueError::new_err(
            "BayesR stream row metadata must match row_indices",
        ));
    }
    let y_vec = if let Ok(slice) = y.as_slice() {
        slice.to_vec()
    } else {
        array1_to_vec(&y)
    };
    let sample_idx =
        parse_index_vec_i64_value_error(sample_indices.as_slice()?, n_samples, "sample_indices")?;
    if sample_idx.len() != y_vec.len() {
        return Err(PyValueError::new_err(
            "BayesR sample_indices length must match len(y)",
        ));
    }
    let (x_vec, q) = parse_covariates(x.as_ref(), y_vec.len())?;
    let pi_vec = parse_bayesr_prior(pi.as_ref(), &BAYESR_DEFAULT_PI, "pi")?;
    let gamma_vec = parse_bayesr_prior(gamma.as_ref(), &BAYESR_DEFAULT_GAMMA, "gamma")?;
    let pool_owned = get_cached_pool(threads)?;
    let pool = pool_owned.as_ref();
    let result = py.detach(|| {
        let rows = block_rows
            .unwrap_or_else(|| bayes_packed_block_rows(y_vec.len(), p))
            .max(1);
        let mut source =
            crate::bayes::build_bayes_source(None, &prefix, n_samples, rows, mmap_window_mb)?;
        let n_source = match &source {
            BayesPackedSource::Resident { .. } => p,
            BayesPackedSource::Windowed(matrix) => matrix.n_source_snps(),
        };
        let packed_row_indices =
            parse_index_vec_i64_value_error(row_indices_raw.as_slice(), n_source, "row_indices")
                .map_err(|error| error.to_string())?;
        let _blas_guard = OpenBlasThreadGuard::enter(bayes_packed_blas_threads());
        bayesr_packed_core_impl(
            &mut source,
            n_samples,
            &y_vec,
            &row_flip_vec,
            &row_mean_vec,
            &row_inv_sd_vec,
            Some(packed_row_indices.as_slice()),
            &sample_idx,
            &x_vec,
            q,
            n_iter,
            burnin,
            thin,
            r2,
            df0_e,
            prior_ss_e,
            &pi_vec,
            &gamma_vec,
            df0_lambda,
            s0_lambda2,
            seed,
            Some(rows),
            pool,
        )
    });
    match result {
        Ok(result) => bayesr_result_to_pydict(py, result),
        Err(error) => Err(PyValueError::new_err(error)),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn component_log_weights_match_closed_form() {
        let pi: [f64; 4] = [0.90, 0.06, 0.03, 0.01];
        let gamma: [f64; 4] = [0.0, 0.01, 0.1, 1.0];
        let got = super::component_log_weights(&pi, &gamma, 2.5, 3.0, 1.7, 0.8)
            .expect("valid BayesR component parameters");

        assert_eq!(got.len(), 4);
        assert!((got[0] - pi[0].ln()).abs() < 1e-12);
        for k in 1..4 {
            let tau2 = gamma[k] * 1.7;
            let expected = pi[k].ln() - 0.5 * (1.0 + tau2 * 3.0 / 0.8).ln()
                + tau2 * 2.5_f64.powi(2) / (2.0 * 0.8 * (0.8 + tau2 * 3.0));
            assert!((got[k] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn component_probabilities_are_normalized_and_pip_is_one_minus_spike() {
        let log_weights: [f64; 4] = [0.2_f64.ln(), 0.3_f64.ln(), 0.1_f64.ln(), 0.4_f64.ln()];
        let q =
            super::normalize_component_log_weights(&log_weights).expect("finite component weights");
        let total: f64 = q.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert!((1.0 - q[0] - (q[1] + q[2] + q[3])).abs() < 1e-12);
        assert!((q[3] - 0.4).abs() < 1e-12);
    }

    #[test]
    fn bayesr_prior_validation_requires_four_components_and_zero_spike() {
        assert!(
            super::validate_bayesr_prior(&[0.9, 0.06, 0.03, 0.01], &[0.0, 0.01, 0.1, 1.0]).is_ok()
        );
        assert!(super::validate_bayesr_prior(&[0.9, 0.1], &[0.0, 1.0]).is_err());
        assert!(
            super::validate_bayesr_prior(&[0.9, 0.06, 0.03, 0.01], &[0.1, 0.01, 0.1, 1.0]).is_err()
        );
    }
}
