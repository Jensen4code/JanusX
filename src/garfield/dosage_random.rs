//! Fixed-covariance GRM-aware dosage interaction oracle.
//!
//! This module is intentionally separate from the ordinary dosage scorer. It
//! implements the first random-effect layer only: callers provide a GRM and
//! already-fitted variance components, then a single Cholesky factor of
//! `V = sigma_g2 * K + sigma_e2 * I` is reused for pairwise GLS/FWL scores.
//!
//! The scan currently favors correctness and a bounded memory footprint over
//! the eventual low-rank/blocked implementation. It precomputes `V^-1 g_j`
//! once per marker and solves only the interaction column for each pair. No
//! explicit inverse of `V` is formed.

use nalgebra::{Cholesky, DMatrix, DVector, Dyn};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

const RANK_TOL: f64 = 1.0e-11;
const VARIANCE_TOL: f64 = 1.0e-12;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GrmPairFit {
    pub(crate) beta: Vec<f64>,
    pub(crate) interaction_beta: f64,
    pub(crate) interaction_variance: f64,
    pub(crate) interaction_score: f64,
    pub(crate) delta_q: f64,
    pub(crate) null_q: f64,
    pub(crate) full_q: f64,
    pub(crate) residual_df: usize,
    pub(crate) fixed_rank: usize,
    pub(crate) n_valid: usize,
    pub(crate) sigma_g2: f64,
    pub(crate) sigma_e2: f64,
}

#[derive(Clone, Copy, Debug)]
struct GrmPairStatistic {
    interaction_beta: f64,
    interaction_variance: f64,
    interaction_score: f64,
    delta_q: f64,
    null_q: f64,
    full_q: f64,
    residual_df: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GrmPairCandidate {
    pub(crate) first: usize,
    pub(crate) second: usize,
    pub(crate) fit: GrmPairFit,
}

#[derive(Clone, Debug)]
struct GrmHeapEntry {
    first: usize,
    second: usize,
    statistic: GrmPairStatistic,
}

impl PartialEq for GrmHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.first == other.first
            && self.second == other.second
            && self.statistic.interaction_score.to_bits()
                == other.statistic.interaction_score.to_bits()
    }
}

impl Eq for GrmHeapEntry {}

fn candidate_cmp(a: &GrmHeapEntry, b: &GrmHeapEntry) -> Ordering {
    a.statistic
        .interaction_score
        .total_cmp(&b.statistic.interaction_score)
        .then_with(|| b.first.cmp(&a.first))
        .then_with(|| b.second.cmp(&a.second))
}

impl Ord for GrmHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        candidate_cmp(self, other)
    }
}

impl PartialOrd for GrmHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Default)]
struct GrmPairAccumulator {
    heap: BinaryHeap<Reverse<GrmHeapEntry>>,
    pairs_evaluated: usize,
    pairs_skipped: usize,
}

impl GrmPairAccumulator {
    fn push(&mut self, first: usize, second: usize, statistic: GrmPairStatistic, top_k: usize) {
        if top_k == 0 {
            return;
        }
        let entry = Reverse(GrmHeapEntry {
            first,
            second,
            statistic,
        });
        if self.heap.len() < top_k {
            self.heap.push(entry);
        } else if self
            .heap
            .peek()
            .map(|worst| entry.0.cmp(&worst.0) == Ordering::Greater)
            .unwrap_or(true)
        {
            let _ = self.heap.pop();
            self.heap.push(entry);
        }
    }

    fn merge(&mut self, other: Self, top_k: usize) {
        self.pairs_evaluated = self.pairs_evaluated.saturating_add(other.pairs_evaluated);
        self.pairs_skipped = self.pairs_skipped.saturating_add(other.pairs_skipped);
        for Reverse(entry) in other.heap {
            self.push(entry.first, entry.second, entry.statistic, top_k);
        }
    }
}

struct PairWorkspace {
    interaction: Vec<f64>,
    vinv_interaction: Vec<f64>,
    lower_gram: Vec<f64>,
    rhs_interaction: Vec<f64>,
    rhs_y: Vec<f64>,
    solved_interaction: Vec<f64>,
    solved_y: Vec<f64>,
}

impl PairWorkspace {
    fn new(n: usize, lower_dim: usize) -> Self {
        Self {
            interaction: vec![0.0; n],
            vinv_interaction: vec![0.0; n],
            lower_gram: vec![0.0; lower_dim * lower_dim],
            rhs_interaction: vec![0.0; lower_dim],
            rhs_y: vec![0.0; lower_dim],
            solved_interaction: vec![0.0; lower_dim],
            solved_y: vec![0.0; lower_dim],
        }
    }

    fn clear_lower(&mut self, lower_dim: usize) {
        self.lower_gram[..lower_dim * lower_dim].fill(0.0);
        self.rhs_interaction[..lower_dim].fill(0.0);
        self.rhs_y[..lower_dim].fill(0.0);
        self.solved_interaction[..lower_dim].fill(0.0);
        self.solved_y[..lower_dim].fill(0.0);
    }
}

struct FixedVContext {
    n: usize,
    fixed_rank: usize,
    fixed_columns: Vec<Vec<f64>>,
    fixed_gram: Vec<f64>,
    fixed_y_cross: Vec<f64>,
    vinv_y: Vec<f64>,
    y_vinv_y: f64,
    chol_v: Cholesky<f64, Dyn>,
    sigma_g2: f64,
    sigma_e2: f64,
}

impl FixedVContext {
    fn new(
        y: &[f64],
        grm: &[f64],
        n: usize,
        sigma_g2: f64,
        sigma_e2: f64,
        covariates: &[f64],
        n_covariates: usize,
    ) -> Result<Self, String> {
        if n == 0 {
            return Err("GRM-aware interaction requires n > 0".to_string());
        }
        if y.len() != n {
            return Err(format!("y length={} but expected {n}", y.len()));
        }
        if grm.len() != n.saturating_mul(n) {
            return Err(format!(
                "GRM length={} but expected {}",
                grm.len(),
                n.saturating_mul(n)
            ));
        }
        if !sigma_g2.is_finite() || sigma_g2 < 0.0 {
            return Err(format!("sigma_g2 must be finite and >= 0; got {sigma_g2}"));
        }
        if !sigma_e2.is_finite() || sigma_e2 <= 0.0 {
            return Err(format!("sigma_e2 must be finite and > 0; got {sigma_e2}"));
        }
        if y.iter().any(|value| !value.is_finite()) {
            return Err("y contains non-finite values".to_string());
        }
        let expected_covariates = n
            .checked_mul(n_covariates)
            .ok_or_else(|| "covariate design size overflow".to_string())?;
        if covariates.len() != expected_covariates {
            return Err(format!(
                "covariate length={} but expected {}",
                covariates.len(),
                expected_covariates
            ));
        }
        if covariates.iter().any(|value| !value.is_finite()) {
            return Err("covariates contain non-finite values".to_string());
        }

        let fixed_rank = n_covariates
            .checked_add(1)
            .ok_or_else(|| "fixed-effect column count overflow".to_string())?;
        if n <= fixed_rank + 3 {
            return Err(format!(
                "need more than fixed rank + 3 samples; got n={n}, fixed columns={fixed_rank}"
            ));
        }

        let mut fixed_columns = Vec::with_capacity(fixed_rank);
        fixed_columns.push(vec![1.0; n]);
        for column in 0..n_covariates {
            let mut values = Vec::with_capacity(n);
            for row in 0..n {
                values.push(covariates[row * n_covariates + column]);
            }
            fixed_columns.push(values);
        }
        let fixed_design =
            DMatrix::from_fn(n, fixed_rank, |row, column| fixed_columns[column][row]);
        check_full_rank(&fixed_design, "fixed-effect design")?;

        let mut v = DMatrix::zeros(n, n);
        for row in 0..n {
            for column in 0..n {
                let left = grm[row * n + column];
                let right = grm[column * n + row];
                if !left.is_finite() || !right.is_finite() {
                    return Err("GRM contains non-finite values".to_string());
                }
                let scale = left.abs().max(right.abs()).max(1.0);
                if (left - right).abs() > 1.0e-10 * scale {
                    return Err(format!(
                        "GRM is not symmetric at ({row}, {column}): {left} vs {right}"
                    ));
                }
                v[(row, column)] = sigma_g2 * left + if row == column { sigma_e2 } else { 0.0 };
            }
        }
        let chol_v = Cholesky::new(v).ok_or_else(|| {
            "GRM covariance V = sigma_g2*K + sigma_e2*I is not positive definite".to_string()
        })?;

        let mut vinv_y = vec![0.0; n];
        solve_cholesky_into(&chol_v, y, &mut vinv_y)?;
        let y_vinv_y = dot(y, &vinv_y);
        let mut fixed_vinv_columns = Vec::with_capacity(fixed_rank);
        for column in &fixed_columns {
            let mut solved = vec![0.0; n];
            solve_cholesky_into(&chol_v, column, &mut solved)?;
            fixed_vinv_columns.push(solved);
        }
        let mut fixed_gram = vec![0.0; fixed_rank * fixed_rank];
        let mut fixed_y_cross = vec![0.0; fixed_rank];
        for row in 0..fixed_rank {
            fixed_y_cross[row] = dot(&fixed_columns[row], &vinv_y);
            for column in 0..fixed_rank {
                fixed_gram[row * fixed_rank + column] =
                    dot(&fixed_columns[row], &fixed_vinv_columns[column]);
            }
        }
        Ok(Self {
            n,
            fixed_rank,
            fixed_columns,
            fixed_gram,
            fixed_y_cross,
            vinv_y,
            y_vinv_y,
            chol_v,
            sigma_g2,
            sigma_e2,
        })
    }

    fn solve_v_into(&self, rhs: &[f64], out: &mut [f64]) -> Result<(), String> {
        solve_cholesky_into(&self.chol_v, rhs, out)
    }

    fn precompute_marker_vinv(
        &self,
        genotypes: &[f64],
        n_markers: usize,
    ) -> Result<Vec<Vec<f64>>, String> {
        let mut result = Vec::with_capacity(n_markers);
        for marker in 0..n_markers {
            let start = marker * self.n;
            let values = &genotypes[start..start + self.n];
            if values.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "genotype marker {marker} contains non-finite values"
                ));
            }
            let mut solved = vec![0.0; self.n];
            self.solve_v_into(values, &mut solved)?;
            result.push(solved);
        }
        Ok(result)
    }

    fn score_pair_with_workspace(
        &self,
        g1: &[f64],
        g2: &[f64],
        vinv_g1: &[f64],
        vinv_g2: &[f64],
        workspace: &mut PairWorkspace,
        beta_out: Option<&mut Vec<f64>>,
    ) -> Result<GrmPairStatistic, String> {
        if g1.len() != self.n || g2.len() != self.n {
            return Err("genotype vector length does not match GRM".to_string());
        }
        if vinv_g1.len() != self.n || vinv_g2.len() != self.n {
            return Err("V^-1 marker length does not match GRM".to_string());
        }
        for row in 0..self.n {
            workspace.interaction[row] = g1[row] * g2[row];
        }
        self.solve_v_into(&workspace.interaction, &mut workspace.vinv_interaction)?;
        let lower_dim = self.fixed_rank + 2;
        workspace.clear_lower(lower_dim);

        for row in 0..self.fixed_rank {
            for column in 0..self.fixed_rank {
                workspace.lower_gram[row * lower_dim + column] =
                    self.fixed_gram[row * self.fixed_rank + column];
            }
            workspace.lower_gram[row * lower_dim + self.fixed_rank] =
                dot(&self.fixed_columns[row], vinv_g1);
            workspace.lower_gram[row * lower_dim + self.fixed_rank + 1] =
                dot(&self.fixed_columns[row], vinv_g2);
            workspace.rhs_y[row] = self.fixed_y_cross[row];
            workspace.rhs_interaction[row] =
                dot(&self.fixed_columns[row], &workspace.vinv_interaction);
        }
        for row in 0..self.fixed_rank {
            workspace.lower_gram[self.fixed_rank * lower_dim + row] =
                workspace.lower_gram[row * lower_dim + self.fixed_rank];
            workspace.lower_gram[(self.fixed_rank + 1) * lower_dim + row] =
                workspace.lower_gram[row * lower_dim + self.fixed_rank + 1];
        }
        workspace.lower_gram[self.fixed_rank * lower_dim + self.fixed_rank] = dot(g1, vinv_g1);
        workspace.lower_gram[self.fixed_rank * lower_dim + self.fixed_rank + 1] = dot(g1, vinv_g2);
        workspace.lower_gram[(self.fixed_rank + 1) * lower_dim + self.fixed_rank] =
            workspace.lower_gram[self.fixed_rank * lower_dim + self.fixed_rank + 1];
        workspace.lower_gram[(self.fixed_rank + 1) * lower_dim + self.fixed_rank + 1] =
            dot(g2, vinv_g2);
        workspace.rhs_y[self.fixed_rank] = dot(g1, &self.vinv_y);
        workspace.rhs_y[self.fixed_rank + 1] = dot(g2, &self.vinv_y);
        workspace.rhs_interaction[self.fixed_rank] = dot(g1, &workspace.vinv_interaction);
        workspace.rhs_interaction[self.fixed_rank + 1] = dot(g2, &workspace.vinv_interaction);
        let lower = DMatrix::from_row_slice(
            lower_dim,
            lower_dim,
            &workspace.lower_gram[..lower_dim * lower_dim],
        );
        let lower_chol = Cholesky::new(lower).ok_or_else(|| {
            "interaction unidentifiable: lower GLS design is rank-deficient".to_string()
        })?;
        let rhs_interaction = DVector::from_column_slice(&workspace.rhs_interaction[..lower_dim]);
        let rhs_y = DVector::from_column_slice(&workspace.rhs_y[..lower_dim]);
        let solved_interaction = lower_chol.solve(&rhs_interaction);
        let solved_y = lower_chol.solve(&rhs_y);
        workspace.solved_interaction[..lower_dim].copy_from_slice(solved_interaction.as_slice());
        workspace.solved_y[..lower_dim].copy_from_slice(solved_y.as_slice());
        let interaction_raw_q = dot(&workspace.interaction, &workspace.vinv_interaction);
        let interaction_cross_y = dot(&workspace.interaction, &self.vinv_y);
        let projected_variance = interaction_raw_q
            - dot(
                &workspace.rhs_interaction[..lower_dim],
                &workspace.solved_interaction[..lower_dim],
            );
        if !projected_variance.is_finite()
            || projected_variance <= VARIANCE_TOL * interaction_raw_q.abs().max(1.0)
        {
            return Err(
                "interaction unidentifiable: residualized interaction variance is non-positive"
                    .to_string(),
            );
        }
        let projected_covariance = interaction_cross_y
            - dot(
                &workspace.rhs_interaction[..lower_dim],
                &workspace.solved_y[..lower_dim],
            );
        let interaction_beta = projected_covariance / projected_variance;
        let delta_q = projected_covariance * interaction_beta;
        let null_q = (self.y_vinv_y
            - dot(
                &workspace.rhs_y[..lower_dim],
                &workspace.solved_y[..lower_dim],
            ))
        .max(0.0);
        let full_q = (null_q - delta_q).max(0.0);
        if ![interaction_beta, delta_q, null_q, full_q]
            .iter()
            .all(|value| value.is_finite())
        {
            return Err("non-finite GLS interaction statistic".to_string());
        }
        if let Some(beta_out) = beta_out {
            beta_out.clear();
            beta_out.extend(
                workspace.solved_y[..lower_dim]
                    .iter()
                    .zip(workspace.solved_interaction[..lower_dim].iter())
                    .map(|(null_beta, correction)| null_beta - correction * interaction_beta),
            );
            beta_out.push(interaction_beta);
        }
        Ok(GrmPairStatistic {
            interaction_beta,
            interaction_variance: projected_variance,
            interaction_score: delta_q,
            delta_q,
            null_q,
            full_q,
            residual_df: self.n - self.fixed_rank - 3,
        })
    }

    fn fit_pair(&self, g1: &[f64], g2: &[f64]) -> Result<GrmPairFit, String> {
        if g1.iter().any(|value| !value.is_finite()) || g2.iter().any(|value| !value.is_finite()) {
            return Err(
                "missing/non-finite genotype values are not supported by fixed-V scan".to_string(),
            );
        }
        if g1.len() != self.n || g2.len() != self.n {
            return Err("genotype vector length does not match GRM".to_string());
        }
        let mut vinv_g1 = vec![0.0; self.n];
        let mut vinv_g2 = vec![0.0; self.n];
        self.solve_v_into(g1, &mut vinv_g1)?;
        self.solve_v_into(g2, &mut vinv_g2)?;
        let mut workspace = PairWorkspace::new(self.n, self.fixed_rank + 2);
        let mut beta = Vec::new();
        let statistic = self.score_pair_with_workspace(
            g1,
            g2,
            &vinv_g1,
            &vinv_g2,
            &mut workspace,
            Some(&mut beta),
        )?;
        Ok(GrmPairFit {
            beta,
            interaction_beta: statistic.interaction_beta,
            interaction_variance: statistic.interaction_variance,
            interaction_score: statistic.interaction_score,
            delta_q: statistic.delta_q,
            null_q: statistic.null_q,
            full_q: statistic.full_q,
            residual_df: statistic.residual_df,
            fixed_rank: self.fixed_rank,
            n_valid: self.n,
            sigma_g2: self.sigma_g2,
            sigma_e2: self.sigma_e2,
        })
    }
}

fn check_full_rank(matrix: &DMatrix<f64>, name: &str) -> Result<(), String> {
    let qr = matrix.clone().col_piv_qr();
    let r = qr.r();
    let scale = (0..matrix.ncols())
        .map(|column| r[(column, column)].abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let tolerance = RANK_TOL * (matrix.nrows().max(matrix.ncols()) as f64) * scale;
    let rank = (0..matrix.ncols())
        .filter(|&column| r[(column, column)].abs() > tolerance)
        .count();
    if rank != matrix.ncols() {
        Err(format!(
            "{name} is rank-deficient: rank={rank}, columns={}, tolerance={tolerance:.3e}",
            matrix.ncols()
        ))
    } else {
        Ok(())
    }
}

#[inline]
fn dot(left: &[f64], right: &[f64]) -> f64 {
    debug_assert_eq!(left.len(), right.len());
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

fn solve_cholesky_into(
    chol: &Cholesky<f64, Dyn>,
    rhs: &[f64],
    out: &mut [f64],
) -> Result<(), String> {
    if rhs.len() != out.len() || rhs.len() != chol.l_dirty().nrows() {
        return Err("Cholesky solve dimension mismatch".to_string());
    }
    out.copy_from_slice(rhs);
    let lower = chol.l_dirty();
    for row in 0..rhs.len() {
        let mut value = out[row];
        for column in 0..row {
            value -= lower[(row, column)] * out[column];
        }
        let diagonal = lower[(row, row)];
        if !diagonal.is_finite() || diagonal <= 0.0 {
            return Err("non-positive Cholesky diagonal".to_string());
        }
        out[row] = value / diagonal;
    }
    for row in (0..rhs.len()).rev() {
        let mut value = out[row];
        for column in (row + 1)..rhs.len() {
            value -= lower[(column, row)] * out[column];
        }
        let diagonal = lower[(row, row)];
        out[row] = value / diagonal;
    }
    if out.iter().any(|value| !value.is_finite()) {
        Err("non-finite Cholesky solution".to_string())
    } else {
        Ok(())
    }
}

fn validate_scan_inputs(
    genotypes: &[f64],
    n_markers: usize,
    n_samples: usize,
    y: &[f64],
    grm: &[f64],
) -> Result<(), String> {
    let expected = n_markers
        .checked_mul(n_samples)
        .ok_or_else(|| "genotype matrix size overflow".to_string())?;
    if n_markers == 0 || n_samples == 0 {
        return Err("n_markers and n_samples must be > 0".to_string());
    }
    if genotypes.len() != expected {
        return Err(format!(
            "genotype length={} but expected {expected}",
            genotypes.len()
        ));
    }
    if y.len() != n_samples {
        return Err(format!("y length={} but expected {n_samples}", y.len()));
    }
    if grm.len() != n_samples.saturating_mul(n_samples) {
        return Err(format!(
            "GRM length={} but expected {}",
            grm.len(),
            n_samples.saturating_mul(n_samples)
        ));
    }
    Ok(())
}

fn fit_grm_pair(
    g1: &[f64],
    g2: &[f64],
    y: &[f64],
    grm: &[f64],
    n: usize,
    sigma_g2: f64,
    sigma_e2: f64,
    covariates: &[f64],
    n_covariates: usize,
) -> Result<GrmPairFit, String> {
    if g1.len() != n || g2.len() != n || y.len() != n {
        return Err("g1, g2, and y lengths must match n".to_string());
    }
    let context = FixedVContext::new(y, grm, n, sigma_g2, sigma_e2, covariates, n_covariates)?;
    context.fit_pair(g1, g2)
}

fn scan_grm_pairs(
    genotypes: &[f64],
    n_markers: usize,
    n_samples: usize,
    y: &[f64],
    grm: &[f64],
    sigma_g2: f64,
    sigma_e2: f64,
    top_k: usize,
    threads: usize,
    covariates: &[f64],
    n_covariates: usize,
) -> Result<(Vec<GrmPairCandidate>, usize, usize), String> {
    validate_scan_inputs(genotypes, n_markers, n_samples, y, grm)?;
    let context = FixedVContext::new(
        y,
        grm,
        n_samples,
        sigma_g2,
        sigma_e2,
        covariates,
        n_covariates,
    )?;
    let vinv_markers = context.precompute_marker_vinv(genotypes, n_markers)?;
    let first_count = n_markers.saturating_sub(1);
    let scan_range = |range: std::ops::Range<usize>| -> Result<GrmPairAccumulator, String> {
        let mut accumulator = GrmPairAccumulator::default();
        let mut workspace = PairWorkspace::new(n_samples, context.fixed_rank + 2);
        for first in range {
            let first_start = first * n_samples;
            let first_values = &genotypes[first_start..first_start + n_samples];
            let first_vinv = &vinv_markers[first];
            for second in (first + 1)..n_markers {
                let second_start = second * n_samples;
                let second_values = &genotypes[second_start..second_start + n_samples];
                let second_vinv = &vinv_markers[second];
                match context.score_pair_with_workspace(
                    first_values,
                    second_values,
                    first_vinv,
                    second_vinv,
                    &mut workspace,
                    None,
                ) {
                    Ok(statistic) => {
                        accumulator.pairs_evaluated = accumulator.pairs_evaluated.saturating_add(1);
                        accumulator.push(first, second, statistic, top_k);
                    }
                    Err(error) if error.contains("unidentifiable") => {
                        accumulator.pairs_skipped = accumulator.pairs_skipped.saturating_add(1);
                    }
                    Err(error) => return Err(error),
                }
            }
        }
        Ok(accumulator)
    };

    let accumulator = if threads <= 1 || first_count <= 1 {
        scan_range(0..first_count)?
    } else {
        let chunk_count = threads.saturating_mul(4).max(1);
        let chunk_len = (first_count + chunk_count - 1) / chunk_count;
        let ranges = (0..first_count)
            .step_by(chunk_len.max(1))
            .map(|start| start..(start + chunk_len).min(first_count))
            .collect::<Vec<_>>();
        let partials = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|error| format!("failed to build GRM dosage Rayon pool: {error}"))?
            .install(|| {
                ranges
                    .par_iter()
                    .map(|range| scan_range(range.clone()))
                    .collect::<Result<Vec<_>, String>>()
            })?;
        let mut merged = GrmPairAccumulator::default();
        for partial in partials {
            merged.merge(partial, top_k);
        }
        merged
    };

    let mut entries = accumulator
        .heap
        .into_iter()
        .map(|Reverse(entry)| entry)
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| candidate_cmp(right, left));
    let mut candidates = Vec::with_capacity(entries.len());
    for entry in entries {
        let first_start = entry.first * n_samples;
        let second_start = entry.second * n_samples;
        let fit = context.fit_pair(
            &genotypes[first_start..first_start + n_samples],
            &genotypes[second_start..second_start + n_samples],
        )?;
        candidates.push(GrmPairCandidate {
            first: entry.first,
            second: entry.second,
            fit,
        });
    }
    Ok((
        candidates,
        accumulator.pairs_evaluated,
        accumulator.pairs_skipped,
    ))
}

fn array1_to_vec(array: &PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    let view = array.as_array();
    if view.is_standard_layout() {
        view.as_slice()
            .expect("standard-layout array must expose a contiguous slice")
            .to_vec()
    } else {
        view.indexed_iter().map(|(_, value)| *value).collect()
    }
}

fn array2_to_vec(array: &PyReadonlyArray2<'_, f64>) -> Vec<f64> {
    let view = array.as_array();
    if view.is_standard_layout() {
        view.as_slice()
            .expect("standard-layout array must expose a contiguous slice")
            .to_vec()
    } else {
        // `iter()` follows the underlying strided traversal for non-standard
        // views.  The Rust kernels below expect a row-major flat buffer, so
        // materialize by logical (row, column) coordinates instead.  This is
        // important for Fortran-order and sliced NumPy inputs.
        let (n_rows, n_columns) = view.dim();
        let mut values = Vec::with_capacity(n_rows.saturating_mul(n_columns));
        for row in 0..n_rows {
            for column in 0..n_columns {
                values.push(view[[row, column]]);
            }
        }
        values
    }
}

fn set_fit_items<'py>(py: Python<'py>, out: &Bound<'py, PyDict>, fit: &GrmPairFit) -> PyResult<()> {
    out.set_item("beta", PyArray1::from_vec(py, fit.beta.clone()))?;
    out.set_item("interaction_beta", fit.interaction_beta)?;
    out.set_item("interaction_variance", fit.interaction_variance)?;
    out.set_item("interaction_score", fit.interaction_score)?;
    out.set_item("delta_q", fit.delta_q)?;
    out.set_item("null_q", fit.null_q)?;
    out.set_item("full_q", fit.full_q)?;
    out.set_item("residual_df", fit.residual_df)?;
    out.set_item("fixed_rank", fit.fixed_rank)?;
    out.set_item("n_valid", fit.n_valid)?;
    out.set_item("sigma_g2", fit.sigma_g2)?;
    out.set_item("sigma_e2", fit.sigma_e2)?;
    Ok(())
}

#[pyfunction(name = "garfield_dosage_grm_pair_fit")]
#[pyo3(signature = (g1, g2, y, grm, sigma_g2, sigma_e2, covariates=None))]
pub fn garfield_dosage_grm_pair_fit_py<'py>(
    py: Python<'py>,
    g1: PyReadonlyArray1<'py, f64>,
    g2: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    grm: PyReadonlyArray2<'py, f64>,
    sigma_g2: f64,
    sigma_e2: f64,
    covariates: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let g1_vec = array1_to_vec(&g1);
    let g2_vec = array1_to_vec(&g2);
    let y_vec = array1_to_vec(&y);
    let shape = grm.shape();
    if shape.len() != 2 || shape[0] != shape[1] || shape[0] != y_vec.len() {
        return Err(PyValueError::new_err(format!(
            "grm must have shape (n_samples, n_samples); got {:?}, expected ({}, {})",
            shape,
            y_vec.len(),
            y_vec.len()
        )));
    }
    let grm_vec = array2_to_vec(&grm);
    let (covariate_vec, n_covariates) = if let Some(covariates) = covariates {
        let cov_shape = covariates.shape();
        if cov_shape.len() != 2 || cov_shape[0] != y_vec.len() {
            return Err(PyValueError::new_err(format!(
                "covariates must have shape (n_samples, n_covariates); got {:?}, expected first dimension {}",
                cov_shape,
                y_vec.len()
            )));
        }
        (array2_to_vec(&covariates), cov_shape[1])
    } else {
        (Vec::new(), 0)
    };
    let fit = fit_grm_pair(
        &g1_vec,
        &g2_vec,
        &y_vec,
        &grm_vec,
        y_vec.len(),
        sigma_g2,
        sigma_e2,
        &covariate_vec,
        n_covariates,
    )
    .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    set_fit_items(py, &out, &fit)?;
    Ok(out)
}

#[pyfunction(name = "garfield_dosage_grm_pair_scan")]
#[pyo3(signature = (genotypes, y, grm, sigma_g2, sigma_e2, top_k=100, threads=0, covariates=None))]
pub fn garfield_dosage_grm_pair_scan_py<'py>(
    py: Python<'py>,
    genotypes: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    grm: PyReadonlyArray2<'py, f64>,
    sigma_g2: f64,
    sigma_e2: f64,
    top_k: usize,
    threads: usize,
    covariates: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let genotype_shape = genotypes.shape();
    if genotype_shape.len() != 2 {
        return Err(PyValueError::new_err(
            "genotypes must be a 2D marker-major array",
        ));
    }
    let n_markers = genotype_shape[0];
    let n_samples = genotype_shape[1];
    let y_vec = array1_to_vec(&y);
    let grm_shape = grm.shape();
    if grm_shape.len() != 2 || grm_shape[0] != grm_shape[1] || grm_shape[0] != n_samples {
        return Err(PyValueError::new_err(format!(
            "grm must have shape ({n_samples}, {n_samples}); got {:?}",
            grm_shape
        )));
    }
    let genotype_vec = array2_to_vec(&genotypes);
    let grm_vec = array2_to_vec(&grm);
    let (covariate_vec, n_covariates) = if let Some(covariates) = covariates {
        let shape = covariates.shape();
        if shape.len() != 2 || shape[0] != n_samples {
            return Err(PyValueError::new_err(format!(
                "covariates must have shape (n_samples, n_covariates); got {:?}, expected first dimension {n_samples}",
                shape
            )));
        }
        (array2_to_vec(&covariates), shape[1])
    } else {
        (Vec::new(), 0)
    };
    let effective_threads = if threads == 0 {
        rayon::current_num_threads().max(1)
    } else {
        threads
    };
    let (candidates, pairs_evaluated, pairs_skipped) = scan_grm_pairs(
        &genotype_vec,
        n_markers,
        n_samples,
        &y_vec,
        &grm_vec,
        sigma_g2,
        sigma_e2,
        top_k,
        effective_threads,
        &covariate_vec,
        n_covariates,
    )
    .map_err(PyValueError::new_err)?;
    let output_candidates = PyList::empty(py);
    for candidate in candidates {
        let item = PyDict::new(py);
        item.set_item("first", candidate.first)?;
        item.set_item("second", candidate.second)?;
        set_fit_items(py, &item, &candidate.fit)?;
        output_candidates.append(item)?;
    }
    let out = PyDict::new(py);
    out.set_item("candidates", output_candidates)?;
    out.set_item("pairs_evaluated", pairs_evaluated)?;
    out.set_item("pairs_skipped", pairs_skipped)?;
    out.set_item("n_markers", n_markers)?;
    out.set_item("n_samples", n_samples)?;
    out.set_item("threads", effective_threads)?;
    out.set_item("sigma_g2", sigma_g2)?;
    out.set_item("sigma_e2", sigma_e2)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_v_pair_matches_direct_gls_reference() {
        let g1 = [0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0];
        let g2 = [1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let y = [1.2, 0.4, 2.1, 3.7, 1.5, 0.8, 2.4, 2.9];
        let grm = [
            1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.1, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.1, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.1, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 1.0,
        ];
        let covariates = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75];
        let fit = fit_grm_pair(&g1, &g2, &y, &grm, 8, 0.7, 0.3, &covariates, 1)
            .expect("fixed-V pair should be identifiable");
        let v = DMatrix::from_row_slice(
            8,
            8,
            &(0..64)
                .map(|index| {
                    let row = index / 8;
                    let column = index % 8;
                    0.7 * grm[index] + if row == column { 0.3 } else { 0.0 }
                })
                .collect::<Vec<_>>(),
        );
        let chol = Cholesky::new(v).expect("reference V should be SPD");
        let design = DMatrix::from_fn(8, 5, |row, column| match column {
            0 => 1.0,
            1 => covariates[row],
            2 => g1[row],
            3 => g2[row],
            _ => g1[row] * g2[row],
        });
        let vinv_design = chol.solve(&design);
        let vinv_y = chol.solve(&DVector::from_column_slice(&y));
        let reference_beta = (design.transpose() * &vinv_design)
            .cholesky()
            .expect("reference design should be full rank")
            .solve(&(design.transpose() * vinv_y.clone()));
        let residual = DVector::from_column_slice(&y) - &design * &reference_beta;
        let full_q = (residual.transpose() * chol.solve(&residual))[(0, 0)];
        let lower = design.columns(0, 4).into_owned();
        let lower_beta = (lower.transpose() * chol.solve(&lower))
            .cholesky()
            .expect("reference lower design should be full rank")
            .solve(&(lower.transpose() * vinv_y));
        let lower_residual = DVector::from_column_slice(&y) - lower * lower_beta;
        let null_q = (lower_residual.transpose() * chol.solve(&lower_residual))[(0, 0)];
        assert!(fit
            .beta
            .iter()
            .zip(reference_beta.iter())
            .all(|(left, right)| (left - right).abs() < 1.0e-10));
        assert!((fit.full_q - full_q).abs() < 1.0e-10);
        assert!((fit.null_q - null_q).abs() < 1.0e-10);
        assert!((fit.delta_q - (null_q - full_q)).abs() < 1.0e-10);
        assert!(fit.interaction_beta.is_finite());
        assert!(fit.interaction_score.is_finite());
        assert_eq!(fit.residual_df, 3);
    }

    #[test]
    fn fixed_v_scan_evaluates_each_pair_once() {
        let genotypes = [
            0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0,
            2.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0,
        ];
        let y = [1.2, 0.4, 2.1, 3.7, 1.5, 0.8, 2.4, 2.9];
        let grm = (0..64)
            .map(|index| if index / 8 == index % 8 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let (candidates, evaluated, skipped) =
            scan_grm_pairs(&genotypes, 3, 8, &y, &grm, 0.0, 1.0, 10, 2, &[], 0)
                .expect("identity V scan should succeed");
        assert_eq!(evaluated + skipped, 3);
        assert_eq!(candidates.len(), 3);
    }

    #[test]
    fn rank_deficient_fixed_effects_are_rejected() {
        let g1 = [0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0];
        let g2 = [1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let y = [1.2, 0.4, 2.1, 3.7, 1.5, 0.8, 2.4, 2.9];
        let grm = (0..64)
            .map(|index| if index / 8 == index % 8 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let covariates = [1.0; 8];
        let error = fit_grm_pair(&g1, &g2, &y, &grm, 8, 0.0, 1.0, &covariates, 1)
            .expect_err("duplicate intercept must be rejected");
        assert!(error.contains("rank-deficient"));
    }
}
