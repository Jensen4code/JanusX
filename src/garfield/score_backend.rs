#![allow(dead_code)]

use super::score::{
    binary_maf_from_n_hit, score_cont_centered_gain_from_sum_and_n_hit,
    score_cont_centered_gain_packed_with_n_hit, validate_continuous_y, ContinuousRuleScore,
};
use crate::stats_common::{check_ctrlc, interrupt_requested, INTERRUPTED_MSG};
use numpy::ndarray::Array1;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::BoundObject;
use std::env;
use std::time::Instant;

#[derive(Clone, Debug)]
pub(crate) struct BatchScoreParts {
    raw_score: Vec<f64>,
    mean_hit: Vec<f64>,
    mean_miss: Vec<f64>,
    support_frac: Vec<f64>,
    n_hit: Vec<u32>,
    n_miss: Vec<u32>,
    sum_hit: Vec<f64>,
    total_sum: f64,
    n_rows: usize,
    row_words: usize,
    n_samples: usize,
}

#[inline]
fn validate_batch_shape(
    y: &[f64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
    ctx: &str,
) -> Result<(), String> {
    if n_rows == 0 {
        return Err(format!("{ctx}: n_rows must be > 0"));
    }
    if n_samples == 0 {
        return Err(format!("{ctx}: n_samples must be > 0"));
    }
    if row_words == 0 {
        return Err(format!("{ctx}: row_words must be > 0"));
    }
    let bit_cap = row_words.saturating_mul(64);
    if n_samples > bit_cap {
        return Err(format!(
            "{ctx}: n_samples={} exceeds bit capacity={} (row_words={})",
            n_samples, bit_cap, row_words
        ));
    }
    validate_continuous_y(y, n_samples, ctx)?;
    Ok(())
}

#[inline]
fn count_sum_y_where_bit1_row(bits: &[u64], y: &[f64], n_samples: usize) -> (u32, f64) {
    let full_words = n_samples >> 6;
    let rem = n_samples & 63;
    let mut n1 = 0u32;
    let mut s1 = 0.0f64;

    for (w_idx, &word) in bits.iter().take(full_words).enumerate() {
        n1 = n1.saturating_add(word.count_ones());
        let mut w = word;
        let base = w_idx << 6;
        while w != 0 {
            let tz = w.trailing_zeros() as usize;
            s1 += y[base + tz];
            w &= w - 1;
        }
    }

    if rem != 0 {
        let mask = (1u64 << rem) - 1u64;
        let mut w = bits[full_words] & mask;
        n1 = n1.saturating_add(w.count_ones());
        let base = full_words << 6;
        while w != 0 {
            let tz = w.trailing_zeros() as usize;
            s1 += y[base + tz];
            w &= w - 1;
        }
    }

    (n1, s1)
}

fn build_score_parts_from_sum_hit(
    sum_hit: Vec<f64>,
    n_hit: Vec<u32>,
    n_samples: usize,
    total_sum: f64,
    row_words: usize,
) -> Result<BatchScoreParts, String> {
    let n_rows = sum_hit.len();
    let mut raw_score = Vec::with_capacity(n_rows);
    let mut mean_hit = Vec::with_capacity(n_rows);
    let mut mean_miss = Vec::with_capacity(n_rows);
    let mut support_frac = Vec::with_capacity(n_rows);
    let mut n_miss = Vec::with_capacity(n_rows);
    for (idx, (&sum_hit_i, &n_hit_i)) in sum_hit.iter().zip(n_hit.iter()).enumerate() {
        if (idx & 255) == 0 {
            check_interrupt_fast()?;
        }
        let sc = score_cont_centered_gain_from_sum_and_n_hit(
            total_sum,
            sum_hit_i,
            n_samples,
            n_hit_i as usize,
        );
        raw_score.push(sc.raw_score);
        mean_hit.push(sc.mean_hit);
        mean_miss.push(sc.mean_miss);
        support_frac.push(sc.support_frac);
        n_miss.push(sc.n_miss as u32);
    }
    Ok(BatchScoreParts {
        raw_score,
        mean_hit,
        mean_miss,
        support_frac,
        n_hit,
        n_miss,
        sum_hit,
        total_sum,
        n_rows,
        row_words,
        n_samples,
    })
}

fn score_cont_centered_gain_batch_packed_cpu_impl(
    y: &[f64],
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
) -> Result<BatchScoreParts, String> {
    const CTX: &str = "garfield_score_cont_centered_gain_batch_packed_cpu";
    check_ctrlc()?;
    validate_batch_shape(y, row_words, n_rows, n_samples, CTX)?;
    if bits_flat.len() != row_words.saturating_mul(n_rows) {
        return Err(format!(
            "{CTX}: bits size mismatch: got {}, expected {} (n_rows={} row_words={})",
            bits_flat.len(),
            row_words.saturating_mul(n_rows),
            n_rows,
            row_words
        ));
    }
    let total_sum = y.iter().take(n_samples).copied().sum::<f64>();
    let mut sum_hit = Vec::with_capacity(n_rows);
    let mut n_hit = Vec::with_capacity(n_rows);
    for rid in 0..n_rows {
        if (rid & 63) == 0 {
            check_interrupt_fast()?;
        }
        let row = &bits_flat[rid * row_words..(rid + 1) * row_words];
        let (n1, s1) = count_sum_y_where_bit1_row(row, y, n_samples);
        n_hit.push(n1);
        sum_hit.push(s1);
    }
    build_score_parts_from_sum_hit(sum_hit, n_hit, n_samples, total_sum, row_words)
}

fn batch_score_parts_to_pydict<'py>(
    py: Python<'py>,
    parts: BatchScoreParts,
    backend: &str,
    elapsed_ms: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py).into_bound();
    out.set_item(
        "raw_score",
        PyArray1::from_owned_array(py, Array1::from_vec(parts.raw_score)).into_bound(),
    )?;
    out.set_item(
        "mean_hit",
        PyArray1::from_owned_array(py, Array1::from_vec(parts.mean_hit)).into_bound(),
    )?;
    out.set_item(
        "mean_miss",
        PyArray1::from_owned_array(py, Array1::from_vec(parts.mean_miss)).into_bound(),
    )?;
    out.set_item(
        "support_frac",
        PyArray1::from_owned_array(py, Array1::from_vec(parts.support_frac)).into_bound(),
    )?;
    out.set_item(
        "n_hit",
        PyArray1::from_owned_array(py, Array1::from_vec(parts.n_hit)).into_bound(),
    )?;
    out.set_item(
        "n_miss",
        PyArray1::from_owned_array(py, Array1::from_vec(parts.n_miss)).into_bound(),
    )?;
    out.set_item(
        "sum_hit",
        PyArray1::from_owned_array(py, Array1::from_vec(parts.sum_hit)).into_bound(),
    )?;
    out.set_item("backend", backend)?;
    out.set_item("total_sum", parts.total_sum)?;
    out.set_item("n_rows", parts.n_rows)?;
    out.set_item("row_words", parts.row_words)?;
    out.set_item("n_samples", parts.n_samples)?;
    if let Some(v) = elapsed_ms {
        out.set_item("elapsed_ms", v)?;
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GarfieldCenteredGainBackendMode {
    Legacy,
    Cpu,
}

#[inline]
fn words_for_samples(n_samples: usize) -> usize {
    n_samples.div_ceil(64).max(1)
}

#[inline]
fn tail_mask(n_samples: usize) -> Option<u64> {
    let rem = n_samples & 63;
    if rem == 0 {
        None
    } else {
        Some((1u64 << rem) - 1u64)
    }
}

#[inline]
fn apply_tail_mask(bits: &mut [u64], mask: Option<u64>) {
    if let Some(m) = mask {
        if let Some(last) = bits.last_mut() {
            *last &= m;
        }
    }
}

#[inline]
fn complement_first_literal(bits: &[u64], n_samples: usize) -> Vec<u64> {
    let needed_words = words_for_samples(n_samples);
    let mut out = bits[..needed_words].to_vec();
    for word in out.iter_mut() {
        *word = !*word;
    }
    apply_tail_mask(&mut out, tail_mask(n_samples));
    out
}

#[inline]
fn batch_parts_row_to_score(parts: &BatchScoreParts, row_idx: usize) -> ContinuousRuleScore {
    ContinuousRuleScore {
        score: parts.raw_score[row_idx],
        raw_score: parts.raw_score[row_idx],
        mean_hit: parts.mean_hit[row_idx],
        mean_miss: parts.mean_miss[row_idx],
        support_frac: parts.support_frac[row_idx],
        dosage_maf: binary_maf_from_n_hit(parts.n_samples, parts.n_hit[row_idx] as usize),
        n_hit: parts.n_hit[row_idx] as usize,
        n_ge2: 0,
        n_miss: parts.n_miss[row_idx] as usize,
    }
}

#[inline]
fn singleton_scores_from_positive_parts(
    parts: &BatchScoreParts,
) -> Result<Vec<ContinuousRuleScore>, String> {
    let mut out = Vec::with_capacity(parts.n_rows.saturating_mul(2));
    for row_idx in 0..parts.n_rows {
        if (row_idx & 255) == 0 {
            check_interrupt_fast()?;
        }
        out.push(batch_parts_row_to_score(parts, row_idx));
        out.push(score_cont_centered_gain_from_sum_and_n_hit(
            parts.total_sum,
            parts.total_sum - parts.sum_hit[row_idx],
            parts.n_samples,
            parts
                .n_samples
                .saturating_sub(parts.n_hit[row_idx] as usize),
        ));
    }
    Ok(out)
}

pub(crate) fn score_cont_centered_gain_singletons_packed_legacy_impl(
    y: &[f64],
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
) -> Result<Vec<ContinuousRuleScore>, String> {
    const CTX: &str = "garfield_score_cont_centered_gain_singletons_packed_legacy";
    check_ctrlc()?;
    validate_batch_shape(y, row_words, n_rows, n_samples, CTX)?;
    if bits_flat.len() != row_words.saturating_mul(n_rows) {
        return Err(format!(
            "{CTX}: bits size mismatch: got {}, expected {} (n_rows={} row_words={})",
            bits_flat.len(),
            row_words.saturating_mul(n_rows),
            n_rows,
            row_words
        ));
    }
    let total_sum = y.iter().take(n_samples).copied().sum::<f64>();
    let needed_words = words_for_samples(n_samples);
    let mut out = Vec::with_capacity(n_rows.saturating_mul(2));
    for row_idx in 0..n_rows {
        if (row_idx & 63) == 0 {
            check_interrupt_fast()?;
        }
        let start = row_idx * row_words;
        let row = &bits_flat[start..start + needed_words];
        let n_hit = count_sum_y_where_bit1_row(row, y, n_samples).0 as usize;
        out.push(score_cont_centered_gain_packed_with_n_hit(
            y, row, n_samples, total_sum, n_hit,
        ));
        let neg_bits = complement_first_literal(row, n_samples);
        let n_hit_neg = n_samples.saturating_sub(n_hit);
        out.push(score_cont_centered_gain_packed_with_n_hit(
            y,
            neg_bits.as_slice(),
            n_samples,
            total_sum,
            n_hit_neg,
        ));
    }
    Ok(out)
}

#[inline]
fn check_interrupt_fast() -> Result<(), String> {
    if interrupt_requested() {
        Err(INTERRUPTED_MSG.to_string())
    } else {
        Ok(())
    }
}

pub(crate) fn score_cont_centered_gain_singletons_packed_cpu_impl(
    y: &[f64],
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
) -> Result<Vec<ContinuousRuleScore>, String> {
    singleton_scores_from_positive_parts(&score_cont_centered_gain_batch_packed_cpu_impl(
        y, bits_flat, row_words, n_rows, n_samples,
    )?)
}

#[inline]
fn parse_centered_gain_backend_mode(raw: &str) -> Result<GarfieldCenteredGainBackendMode, String> {
    let mode = raw.trim().to_ascii_lowercase();
    match mode.as_str() {
        "" => Ok(GarfieldCenteredGainBackendMode::Cpu),
        "legacy" => Ok(GarfieldCenteredGainBackendMode::Legacy),
        "cpu" => Ok(GarfieldCenteredGainBackendMode::Cpu),
        _ => Err(format!(
            "JX_GARFIELD_SCORE_BACKEND must be one of: cpu, legacy; got '{}'",
            raw
        )),
    }
}

#[inline]
pub(crate) fn parse_centered_gain_backend_mode_from_env(
) -> Result<GarfieldCenteredGainBackendMode, String> {
    let raw = env::var("JX_GARFIELD_SCORE_BACKEND").unwrap_or_default();
    parse_centered_gain_backend_mode(&raw)
}

pub(crate) fn score_cont_centered_gain_singletons_packed_with_backend(
    mode: GarfieldCenteredGainBackendMode,
    y: &[f64],
    bits_flat: &[u64],
    row_words: usize,
    n_rows: usize,
    n_samples: usize,
) -> Result<(Vec<ContinuousRuleScore>, &'static str), String> {
    match mode {
        GarfieldCenteredGainBackendMode::Legacy => Ok((
            score_cont_centered_gain_singletons_packed_legacy_impl(
                y, bits_flat, row_words, n_rows, n_samples,
            )?,
            "legacy",
        )),
        GarfieldCenteredGainBackendMode::Cpu => Ok((
            score_cont_centered_gain_singletons_packed_cpu_impl(
                y, bits_flat, row_words, n_rows, n_samples,
            )?,
            "cpu",
        )),
    }
}

#[pyfunction(name = "garfield_score_cont_centered_gain_batch_packed_cpu")]
pub fn garfield_score_cont_centered_gain_batch_packed_cpu_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<'py, f64>,
    bits: PyReadonlyArray2<'py, u64>,
    n_samples: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let yv = y.as_slice()?;
    let shape = bits.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err("bits must be a 2D uint64 array"));
    }
    let n_rows = shape[0];
    let row_words = shape[1];
    let bits_flat = bits
        .as_slice()
        .map_err(|_| PyValueError::new_err("bits must be contiguous (C-order)"))?;
    let t0 = Instant::now();
    let parts = py
        .detach(|| {
            score_cont_centered_gain_batch_packed_cpu_impl(
                yv, bits_flat, row_words, n_rows, n_samples,
            )
        })
        .map_err(PyRuntimeError::new_err)?;
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    batch_score_parts_to_pydict(py, parts, "cpu", Some(elapsed_ms))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_backend_accepts_only_cpu_or_legacy() {
        assert_eq!(
            parse_centered_gain_backend_mode("cpu").expect("cpu backend"),
            GarfieldCenteredGainBackendMode::Cpu
        );
        assert_eq!(
            parse_centered_gain_backend_mode("legacy").expect("legacy backend"),
            GarfieldCenteredGainBackendMode::Legacy
        );
        for raw in ["auto", "gpu", "metal", "new_gpu"] {
            assert!(parse_centered_gain_backend_mode(raw).is_err());
        }
    }

    #[test]
    fn test_batch_cpu_matches_scalar() {
        pyo3::Python::initialize();
        let y = vec![5.0, 4.0, -1.0, -2.0];
        let bits = vec![0b0011_u64, 0b0101_u64];
        let got = score_cont_centered_gain_batch_packed_cpu_impl(&y, &bits, 1, 2, y.len())
            .expect("cpu batch");
        assert_eq!(got.n_hit, vec![2, 2]);
        assert!((got.mean_hit[0] - 4.5).abs() < 1e-12);
        assert!((got.raw_score[0] - 36.0).abs() < 1e-12);
        let sc2 = score_cont_centered_gain_from_sum_and_n_hit(6.0, 4.0, 4, 2);
        assert!((got.raw_score[1] - sc2.raw_score).abs() < 1e-12);
    }
}
