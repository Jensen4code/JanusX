//! Fixed-covariance GRM-aware dosage interaction oracle.
//!
//! This module is intentionally separate from the ordinary dosage scorer. It
//! implements the first random-effect layer only: callers provide a GRM and
//! already-fitted variance components, then a single Cholesky factor of
//! `V = sigma_g2 * K + sigma_e2 * I` is reused for pairwise GLS/FWL scores.
//!
//! The scan keeps the covariance factorized: it precomputes `V^-1 g_j` once
//! per marker and solves interaction columns in cache-sized blocks with a
//! BLAS triangular-solve kernel. No explicit inverse of `V` is formed.

use crate::blas::{
    cblas_ddot_dispatch, cblas_dtrsm_dispatch, rust_sgemm_backend_tag, BlasThreadGuard, CblasInt,
    CBLAS_COL_MAJOR, CBLAS_DIAG_NON_UNIT, CBLAS_LEFT, CBLAS_LOWER, CBLAS_NO_TRANS, CBLAS_TRANS,
};
use nalgebra::{Cholesky, DMatrix, Dyn};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ops::AddAssign;
use std::time::{Duration, Instant};

use super::residual::{garfield_residualize_exact_from_grm_rust, GarfieldResidualResult};

const RANK_TOL: f64 = 1.0e-11;
const VARIANCE_TOL: f64 = 1.0e-12;
const GRM_INTERACTION_BLOCK_SIZE: usize = 256;

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
    pair_index: usize,
    first: usize,
    second: usize,
    statistic: GrmPairStatistic,
    /// Geometry and interaction response cross-product are retained only for
    /// scan winners.  This lets the final Top-K refit reuse work already done
    /// in the scan; other accumulator users keep these fields as `None`.
    geometry: Option<PairGeometry>,
    interaction_cross_y: Option<f64>,
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

/// Optional diagnostics for the fixed-V scanner.  The profiler is deliberately
/// environment-gated so normal production scans do not pay for timing calls or
/// emit anything on stderr.
#[derive(Clone, Copy, Default)]
struct GrmScanTiming {
    context: Duration,
    marker_vinv: Duration,
    interaction_build: Duration,
    interaction_solve: Duration,
    pair_geometry_score: Duration,
    final_fit: Duration,
    blocks: usize,
}

impl AddAssign for GrmScanTiming {
    fn add_assign(&mut self, other: Self) {
        self.context += other.context;
        self.marker_vinv += other.marker_vinv;
        self.interaction_build += other.interaction_build;
        self.interaction_solve += other.interaction_solve;
        self.pair_geometry_score += other.pair_geometry_score;
        self.final_fit += other.final_fit;
        self.blocks = self.blocks.saturating_add(other.blocks);
    }
}

#[derive(Default)]
struct GrmScanRangeResult {
    accumulator: GrmPairAccumulator,
    timing: GrmScanTiming,
}

#[inline]
fn grm_profile_enabled() -> bool {
    std::env::var("JX_GRM_PROFILE")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
}

impl GrmPairAccumulator {
    #[inline]
    fn accepts(
        &self,
        first: usize,
        second: usize,
        statistic: GrmPairStatistic,
        top_k: usize,
    ) -> bool {
        if top_k == 0 || self.heap.len() < top_k {
            return top_k > 0;
        }
        let candidate = GrmHeapEntry {
            pair_index: 0,
            first,
            second,
            statistic,
            geometry: None,
            interaction_cross_y: None,
        };
        self.heap
            .peek()
            .map(|worst| candidate.cmp(&worst.0) == Ordering::Greater)
            .unwrap_or(true)
    }

    #[inline]
    fn push_entry(&mut self, entry: Reverse<GrmHeapEntry>, top_k: usize) {
        if top_k == 0 {
            return;
        }
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

    fn push(
        &mut self,
        pair_index: usize,
        first: usize,
        second: usize,
        statistic: GrmPairStatistic,
        top_k: usize,
    ) {
        if !self.accepts(first, second, statistic, top_k) {
            return;
        }
        self.push_entry(
            Reverse(GrmHeapEntry {
                pair_index,
                first,
                second,
                statistic,
                geometry: None,
                interaction_cross_y: None,
            }),
            top_k,
        );
    }

    fn push_with_geometry(
        &mut self,
        pair_index: usize,
        first: usize,
        second: usize,
        statistic: GrmPairStatistic,
        geometry: PairGeometryView<'_>,
        interaction_cross_y: f64,
        top_k: usize,
    ) {
        if !self.accepts(first, second, statistic, top_k) {
            return;
        }
        self.push_entry(
            Reverse(GrmHeapEntry {
                pair_index,
                first,
                second,
                statistic,
                geometry: Some(PairGeometry {
                    lower_chol: geometry.lower_chol.to_vec(),
                    rhs_interaction: geometry.rhs_interaction.to_vec(),
                    solved_interaction: geometry.solved_interaction.to_vec(),
                    projected_variance: geometry.projected_variance,
                }),
                interaction_cross_y: Some(interaction_cross_y),
            }),
            top_k,
        );
    }

    fn merge(&mut self, other: Self, top_k: usize) {
        self.pairs_evaluated = self.pairs_evaluated.saturating_add(other.pairs_evaluated);
        self.pairs_skipped = self.pairs_skipped.saturating_add(other.pairs_skipped);
        for entry in other.heap {
            self.push_entry(entry, top_k);
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

/// Marker-level cross-products that do not depend on the second marker in a
/// pair.  Computing these once removes repeated O(n) reductions from the
/// fixed-V scan while keeping the statistical formula unchanged.
struct MarkerVinvStats {
    /// Marker-by-fixed-effect cross-products, row-major by marker.
    fixed_cross: Vec<f64>,
    /// Marker-by-V^-1 y cross-products.
    y_cross: Vec<f64>,
    /// Marker self cross-products g_j^T V^-1 g_j.
    self_cross: Vec<f64>,
}

#[derive(Clone, Copy)]
struct MarkerPairVinvStats<'a> {
    first_fixed_cross: &'a [f64],
    second_fixed_cross: &'a [f64],
    first_self_cross: f64,
    cross: f64,
    second_self_cross: f64,
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
}

/// Genotype-only geometry for one pair under a fixed covariance matrix.
///
/// The interaction column is formed and solved once while preparing a scan.
/// Later phenotypes only update the response cross-products, so repeated
/// scans sharing the same genotype window and variance components do not
/// redo the expensive (V^{-1}(g_i g_j)) solve.
#[derive(Clone, Debug)]
struct PairGeometry {
    lower_chol: Vec<f64>,
    rhs_interaction: Vec<f64>,
    solved_interaction: Vec<f64>,
    projected_variance: f64,
}

#[derive(Clone, Copy)]
struct PairGeometryView<'a> {
    lower_chol: &'a [f64],
    rhs_interaction: &'a [f64],
    solved_interaction: &'a [f64],
    projected_variance: f64,
}

struct PreparedPair {
    first: usize,
    second: usize,
    geometry_index: usize,
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

    fn set_response(&mut self, y: &[f64]) -> Result<(), String> {
        if y.len() != self.n {
            return Err(format!("y length={} but expected {}", y.len(), self.n));
        }
        if y.iter().any(|value| !value.is_finite()) {
            return Err("y contains non-finite values".to_string());
        }
        solve_cholesky_into(&self.chol_v, y, &mut self.vinv_y)?;
        self.y_vinv_y = dot(y, &self.vinv_y);
        for row in 0..self.fixed_rank {
            self.fixed_y_cross[row] = dot(&self.fixed_columns[row], &self.vinv_y);
        }
        Ok(())
    }

    fn precompute_marker_vinv_with_threads(
        &self,
        genotypes: &[f64],
        n_markers: usize,
        threads: usize,
    ) -> Result<Vec<f64>, String> {
        let expected_len = n_markers
            .checked_mul(self.n)
            .ok_or_else(|| "genotype matrix size overflow".to_string())?;
        if genotypes.len() != expected_len {
            return Err(format!(
                "genotype length={} but expected {expected_len}",
                genotypes.len()
            ));
        }

        let validate_marker = |marker: usize| -> Result<(), String> {
            let start = marker * self.n;
            let values = &genotypes[start..start + self.n];
            if values.iter().any(|value| !value.is_finite()) {
                return Err(format!(
                    "genotype marker {marker} contains non-finite values"
                ));
            }
            Ok(())
        };

        // The CBLAS path treats marker-major genotype storage as an
        // `n x n_markers` column-major RHS matrix, so all marker solves share
        // one batched triangular solve.  Keep the existing Rayon scalar path
        // for builds whose BLAS dispatch is the portable Rust fallback.
        if rust_sgemm_backend_tag() != "rust" {
            let _blas_guard = BlasThreadGuard::enter(threads.max(1));
            let mut solved = genotypes.to_vec();
            solve_cholesky_block_blas_in_place(&self.chol_v, &mut solved, n_markers)?;
            Ok(solved)
        } else if threads <= 1 || n_markers <= 1 {
            let mut solved = vec![0.0; expected_len];
            for marker in 0..n_markers {
                validate_marker(marker)?;
                let start = marker * self.n;
                self.solve_v_into(
                    &genotypes[start..start + self.n],
                    &mut solved[start..start + self.n],
                )?;
            }
            Ok(solved)
        } else {
            let pool = ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .map_err(|error| format!("failed to build GRM dosage Rayon pool: {error}"))?;
            let mut solved = vec![0.0; expected_len];
            pool.install(|| {
                solved
                    .par_chunks_mut(self.n)
                    .enumerate()
                    .try_for_each(|(marker, output)| {
                        validate_marker(marker)?;
                        let start = marker * self.n;
                        self.solve_v_into(&genotypes[start..start + self.n], output)
                    })
            })?;
            Ok(solved)
        }
    }

    fn precompute_marker_vinv_stats(
        &self,
        genotypes: &[f64],
        vinv_markers: &[f64],
        n_markers: usize,
    ) -> Result<MarkerVinvStats, String> {
        let expected_len = n_markers
            .checked_mul(self.n)
            .ok_or_else(|| "genotype matrix size overflow".to_string())?;
        if genotypes.len() != expected_len || vinv_markers.len() != expected_len {
            return Err("marker V^-1 statistics dimension mismatch".to_string());
        }
        let mut fixed_cross = vec![0.0; n_markers * self.fixed_rank];
        let mut y_cross = vec![0.0; n_markers];
        let mut self_cross = vec![0.0; n_markers];
        for marker in 0..n_markers {
            let start = marker * self.n;
            let values = &genotypes[start..start + self.n];
            let vinv_values = &vinv_markers[start..start + self.n];
            for fixed in 0..self.fixed_rank {
                fixed_cross[marker * self.fixed_rank + fixed] =
                    dot(&self.fixed_columns[fixed], vinv_values);
            }
            y_cross[marker] = dot(values, &self.vinv_y);
            self_cross[marker] = dot(values, vinv_values);
        }
        Ok(MarkerVinvStats {
            fixed_cross,
            y_cross,
            self_cross,
        })
    }

    #[cfg(test)]
    fn build_pair_geometry(
        &self,
        g1: &[f64],
        g2: &[f64],
        vinv_g1: &[f64],
        vinv_g2: &[f64],
        workspace: &mut PairWorkspace,
    ) -> Result<PairGeometry, String> {
        let projected_variance =
            self.build_pair_geometry_in_place(g1, g2, vinv_g1, vinv_g2, workspace)?;
        let lower_dim = self.fixed_rank + 2;
        Ok(PairGeometry {
            lower_chol: workspace.lower_gram[..lower_dim * lower_dim].to_vec(),
            rhs_interaction: workspace.rhs_interaction[..lower_dim].to_vec(),
            solved_interaction: workspace.solved_interaction[..lower_dim].to_vec(),
            projected_variance,
        })
    }

    /// Build one pair's geometry directly in the caller-provided workspace.
    ///
    /// The scan path consumes the geometry immediately, so allocating three
    /// small vectors for every pair is unnecessary.  The lower Gram matrix is
    /// factorized in place and the projected interaction variance is returned;
    /// the owning wrapper is retained for the reference unit tests.
    fn build_pair_geometry_in_place(
        &self,
        g1: &[f64],
        g2: &[f64],
        vinv_g1: &[f64],
        vinv_g2: &[f64],
        workspace: &mut PairWorkspace,
    ) -> Result<f64, String> {
        self.build_pair_geometry_in_place_with_marker_stats(
            g1, g2, vinv_g1, vinv_g2, None, workspace,
        )
    }

    /// Build one pair's geometry while reusing marker-level cross-products.
    /// The interaction column is still solved exactly once; only the two
    /// marker solves and their fixed/y/self reductions are avoided.
    fn build_pair_geometry_in_place_with_marker_stats(
        &self,
        g1: &[f64],
        g2: &[f64],
        vinv_g1: &[f64],
        vinv_g2: &[f64],
        marker_stats: Option<MarkerPairVinvStats<'_>>,
        workspace: &mut PairWorkspace,
    ) -> Result<f64, String> {
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
        self.build_pair_geometry_from_solved_in_place(
            g1,
            g2,
            vinv_g1,
            vinv_g2,
            &workspace.interaction,
            &workspace.vinv_interaction,
            0,
            1,
            marker_stats,
            &mut workspace.lower_gram,
            &mut workspace.rhs_interaction,
            &mut workspace.solved_interaction,
        )
    }

    fn build_pair_geometry_from_solved_in_place(
        &self,
        g1: &[f64],
        g2: &[f64],
        vinv_g1: &[f64],
        vinv_g2: &[f64],
        interaction: &[f64],
        vinv_interaction: &[f64],
        offset: usize,
        stride: usize,
        marker_stats: Option<MarkerPairVinvStats<'_>>,
        lower_gram: &mut [f64],
        rhs_interaction: &mut [f64],
        solved_interaction: &mut [f64],
    ) -> Result<f64, String> {
        if g1.len() != self.n || g2.len() != self.n {
            return Err("genotype vector length does not match GRM".to_string());
        }
        if stride == 0
            || interaction
                .len()
                .checked_sub(offset)
                .and_then(|length| length.checked_add(stride - 1))
                .map(|length| length / stride)
                .unwrap_or(0)
                < self.n
            || vinv_interaction
                .len()
                .checked_sub(offset)
                .and_then(|length| length.checked_add(stride - 1))
                .map(|length| length / stride)
                .unwrap_or(0)
                < self.n
        {
            return Err("V^-1 interaction length does not match GRM".to_string());
        }
        let interaction_raw_q =
            dot_strided_both(interaction, vinv_interaction, offset, stride, self.n);
        self.build_pair_geometry_with_raw_q_in_place(
            g1,
            g2,
            vinv_g1,
            vinv_g2,
            vinv_interaction,
            interaction_raw_q,
            offset,
            stride,
            marker_stats,
            lower_gram,
            rhs_interaction,
            solved_interaction,
        )
    }

    /// Build pair geometry when the caller has already reduced the raw
    /// interaction column to its quadratic form.  This is used by the blocked
    /// scanner to keep only one interaction buffer: the raw column is consumed
    /// before its storage is overwritten by `V^-1`.
    fn build_pair_geometry_with_raw_q_in_place(
        &self,
        g1: &[f64],
        g2: &[f64],
        vinv_g1: &[f64],
        vinv_g2: &[f64],
        vinv_interaction: &[f64],
        interaction_raw_q: f64,
        offset: usize,
        stride: usize,
        marker_stats: Option<MarkerPairVinvStats<'_>>,
        lower_gram: &mut [f64],
        rhs_interaction: &mut [f64],
        solved_interaction: &mut [f64],
    ) -> Result<f64, String> {
        if g1.len() != self.n || g2.len() != self.n {
            return Err("genotype vector length does not match GRM".to_string());
        }
        if let Some(stats) = marker_stats {
            if stats.first_fixed_cross.len() != self.fixed_rank
                || stats.second_fixed_cross.len() != self.fixed_rank
            {
                return Err("marker V^-1 statistics fixed-effect dimension mismatch".to_string());
            }
        }
        if stride == 0
            || vinv_interaction
                .len()
                .checked_sub(offset)
                .and_then(|length| length.checked_add(stride - 1))
                .map(|length| length / stride)
                .unwrap_or(0)
                < self.n
        {
            return Err("V^-1 interaction length does not match GRM".to_string());
        }
        let lower_dim = self.fixed_rank + 2;
        if lower_gram.len() < lower_dim * lower_dim
            || rhs_interaction.len() < lower_dim
            || solved_interaction.len() < lower_dim
        {
            return Err("pair workspace dimension mismatch".to_string());
        }
        lower_gram[..lower_dim * lower_dim].fill(0.0);
        rhs_interaction[..lower_dim].fill(0.0);
        solved_interaction[..lower_dim].fill(0.0);

        for row in 0..self.fixed_rank {
            for column in 0..self.fixed_rank {
                lower_gram[row * lower_dim + column] =
                    self.fixed_gram[row * self.fixed_rank + column];
            }
            lower_gram[row * lower_dim + self.fixed_rank] = if let Some(stats) = marker_stats {
                stats.first_fixed_cross[row]
            } else {
                dot(&self.fixed_columns[row], vinv_g1)
            };
            lower_gram[row * lower_dim + self.fixed_rank + 1] = if let Some(stats) = marker_stats {
                stats.second_fixed_cross[row]
            } else {
                dot(&self.fixed_columns[row], vinv_g2)
            };
            rhs_interaction[row] =
                dot_strided(&self.fixed_columns[row], vinv_interaction, offset, stride);
        }
        for row in 0..self.fixed_rank {
            lower_gram[self.fixed_rank * lower_dim + row] =
                lower_gram[row * lower_dim + self.fixed_rank];
            lower_gram[(self.fixed_rank + 1) * lower_dim + row] =
                lower_gram[row * lower_dim + self.fixed_rank + 1];
        }
        lower_gram[self.fixed_rank * lower_dim + self.fixed_rank] = marker_stats
            .map(|stats| stats.first_self_cross)
            .unwrap_or_else(|| dot(g1, vinv_g1));
        lower_gram[self.fixed_rank * lower_dim + self.fixed_rank + 1] = marker_stats
            .map(|stats| stats.cross)
            .unwrap_or_else(|| dot(g1, vinv_g2));
        lower_gram[(self.fixed_rank + 1) * lower_dim + self.fixed_rank] =
            lower_gram[self.fixed_rank * lower_dim + self.fixed_rank + 1];
        lower_gram[(self.fixed_rank + 1) * lower_dim + self.fixed_rank + 1] = marker_stats
            .map(|stats| stats.second_self_cross)
            .unwrap_or_else(|| dot(g2, vinv_g2));
        rhs_interaction[self.fixed_rank] = dot_strided(g1, vinv_interaction, offset, stride);
        rhs_interaction[self.fixed_rank + 1] = dot_strided(g2, vinv_interaction, offset, stride);

        cholesky_factor_lower_in_place(&mut lower_gram[..lower_dim * lower_dim], lower_dim)?;
        solve_small_cholesky(
            &lower_gram[..lower_dim * lower_dim],
            lower_dim,
            &rhs_interaction[..lower_dim],
            &mut solved_interaction[..lower_dim],
        )?;
        let projected_variance = interaction_raw_q
            - dot(
                &rhs_interaction[..lower_dim],
                &solved_interaction[..lower_dim],
            );
        if !projected_variance.is_finite()
            || projected_variance <= VARIANCE_TOL * interaction_raw_q.abs().max(1.0)
        {
            return Err(
                "interaction unidentifiable: residualized interaction variance is non-positive"
                    .to_string(),
            );
        }
        Ok(projected_variance)
    }

    fn score_pair_with_geometry_view(
        &self,
        g1: &[f64],
        g2: &[f64],
        geometry: PairGeometryView<'_>,
        workspace: &mut PairWorkspace,
        beta_out: Option<&mut Vec<f64>>,
    ) -> Result<GrmPairStatistic, String> {
        for row in 0..self.n {
            workspace.interaction[row] = g1[row] * g2[row];
        }
        let interaction_cross_y = dot(&workspace.interaction, &self.vinv_y);
        self.score_pair_with_geometry_view_and_cross_y(
            g1,
            g2,
            geometry,
            interaction_cross_y,
            None,
            workspace,
            beta_out,
        )
    }

    fn score_pair_with_geometry_view_and_cross_y(
        &self,
        g1: &[f64],
        g2: &[f64],
        geometry: PairGeometryView<'_>,
        interaction_cross_y: f64,
        marker_y_cross: Option<(f64, f64)>,
        workspace: &mut PairWorkspace,
        beta_out: Option<&mut Vec<f64>>,
    ) -> Result<GrmPairStatistic, String> {
        self.score_pair_with_scratch(
            g1,
            g2,
            &geometry.lower_chol,
            &geometry.rhs_interaction,
            &geometry.solved_interaction,
            geometry.projected_variance,
            interaction_cross_y,
            marker_y_cross,
            &mut workspace.rhs_y,
            &mut workspace.solved_y,
            beta_out,
        )
    }

    fn score_pair_with_scratch(
        &self,
        g1: &[f64],
        g2: &[f64],
        lower_chol: &[f64],
        rhs_interaction: &[f64],
        solved_interaction: &[f64],
        projected_variance: f64,
        interaction_cross_y: f64,
        marker_y_cross: Option<(f64, f64)>,
        rhs_y: &mut [f64],
        solved_y: &mut [f64],
        beta_out: Option<&mut Vec<f64>>,
    ) -> Result<GrmPairStatistic, String> {
        if g1.len() != self.n || g2.len() != self.n {
            return Err("genotype vector length does not match GRM".to_string());
        }
        let lower_dim = self.fixed_rank + 2;
        if lower_chol.len() != lower_dim * lower_dim
            || rhs_interaction.len() != lower_dim
            || solved_interaction.len() != lower_dim
        {
            return Err("pair geometry dimension mismatch".to_string());
        }
        if rhs_y.len() < lower_dim || solved_y.len() < lower_dim {
            return Err("pair workspace dimension mismatch".to_string());
        }
        rhs_y[..lower_dim].fill(0.0);
        rhs_y[..self.fixed_rank].copy_from_slice(&self.fixed_y_cross);
        let (g1_y_cross, g2_y_cross) =
            marker_y_cross.unwrap_or_else(|| (dot(g1, &self.vinv_y), dot(g2, &self.vinv_y)));
        rhs_y[self.fixed_rank] = g1_y_cross;
        rhs_y[self.fixed_rank + 1] = g2_y_cross;
        solve_small_cholesky(
            lower_chol,
            lower_dim,
            &rhs_y[..lower_dim],
            &mut solved_y[..lower_dim],
        )?;
        let projected_covariance =
            interaction_cross_y - dot(rhs_interaction, &solved_y[..lower_dim]);
        let interaction_beta = projected_covariance / projected_variance;
        let delta_q = (projected_covariance * interaction_beta).max(0.0);
        let null_q = (self.y_vinv_y - dot(&rhs_y[..lower_dim], &solved_y[..lower_dim])).max(0.0);
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
                solved_y[..lower_dim]
                    .iter()
                    .zip(solved_interaction.iter())
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

    fn fit_pair_with_geometry_view_into(
        &self,
        g1: &[f64],
        g2: &[f64],
        geometry: PairGeometryView<'_>,
        workspace: &mut PairWorkspace,
        beta_out: &mut Vec<f64>,
    ) -> Result<GrmPairFit, String> {
        let statistic =
            self.score_pair_with_geometry_view(g1, g2, geometry, workspace, Some(beta_out))?;
        Ok(GrmPairFit {
            beta: beta_out.clone(),
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

    fn fit_pair_with_geometry_view_and_cross_y_into(
        &self,
        g1: &[f64],
        g2: &[f64],
        geometry: PairGeometryView<'_>,
        interaction_cross_y: f64,
        marker_y_cross: Option<(f64, f64)>,
        workspace: &mut PairWorkspace,
        beta_out: &mut Vec<f64>,
    ) -> Result<GrmPairFit, String> {
        let statistic = self.score_pair_with_geometry_view_and_cross_y(
            g1,
            g2,
            geometry,
            interaction_cross_y,
            marker_y_cross,
            workspace,
            Some(beta_out),
        )?;
        Ok(GrmPairFit {
            beta: beta_out.clone(),
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

    /// Fit one pair using marker `V^-1` vectors and cross-products prepared by
    /// the scanner.  This is used only for the final Top-K refit, where the
    /// scan has already ranked candidates and the marker solves must not be
    /// repeated merely to materialize coefficient output.
    fn fit_pair_with_precomputed_marker_vinv(
        &self,
        g1: &[f64],
        g2: &[f64],
        vinv_g1: &[f64],
        vinv_g2: &[f64],
        marker_stats: MarkerPairVinvStats<'_>,
        marker_y_cross: (f64, f64),
        workspace: &mut PairWorkspace,
        beta_out: &mut Vec<f64>,
    ) -> Result<GrmPairFit, String> {
        if g1.iter().any(|value| !value.is_finite()) || g2.iter().any(|value| !value.is_finite()) {
            return Err(
                "missing/non-finite genotype values are not supported by fixed-V scan".to_string(),
            );
        }
        if g1.len() != self.n
            || g2.len() != self.n
            || vinv_g1.len() != self.n
            || vinv_g2.len() != self.n
        {
            return Err("precomputed marker fit dimension mismatch".to_string());
        }
        let projected_variance = self.build_pair_geometry_in_place_with_marker_stats(
            g1,
            g2,
            vinv_g1,
            vinv_g2,
            Some(marker_stats),
            workspace,
        )?;
        let lower_dim = self.fixed_rank + 2;
        let interaction_cross_y = dot(&workspace.interaction, &self.vinv_y);
        let statistic = self.score_pair_with_scratch(
            g1,
            g2,
            &workspace.lower_gram[..lower_dim * lower_dim],
            &workspace.rhs_interaction[..lower_dim],
            &workspace.solved_interaction[..lower_dim],
            projected_variance,
            interaction_cross_y,
            Some(marker_y_cross),
            &mut workspace.rhs_y,
            &mut workspace.solved_y,
            Some(beta_out),
        )?;
        Ok(GrmPairFit {
            beta: beta_out.clone(),
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

    fn score_pair_with_workspace(
        &self,
        g1: &[f64],
        g2: &[f64],
        vinv_g1: &[f64],
        vinv_g2: &[f64],
        workspace: &mut PairWorkspace,
        beta_out: Option<&mut Vec<f64>>,
    ) -> Result<GrmPairStatistic, String> {
        let projected_variance =
            self.build_pair_geometry_in_place(g1, g2, vinv_g1, vinv_g2, workspace)?;
        let lower_dim = self.fixed_rank + 2;
        let interaction_cross_y = dot(&workspace.interaction, &self.vinv_y);
        self.score_pair_with_scratch(
            g1,
            g2,
            &workspace.lower_gram[..lower_dim * lower_dim],
            &workspace.rhs_interaction[..lower_dim],
            &workspace.solved_interaction[..lower_dim],
            projected_variance,
            interaction_cross_y,
            None,
            &mut workspace.rhs_y,
            &mut workspace.solved_y,
            beta_out,
        )
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

fn cholesky_factor_lower_in_place(matrix: &mut [f64], dim: usize) -> Result<(), String> {
    if matrix.len() != dim.saturating_mul(dim) {
        return Err("small Cholesky factor dimension mismatch".to_string());
    }
    for row in 0..dim {
        for column in 0..=row {
            let mut value = matrix[row * dim + column];
            for k in 0..column {
                value -= matrix[row * dim + k] * matrix[column * dim + k];
            }
            if row == column {
                if !value.is_finite() || value <= VARIANCE_TOL {
                    return Err(
                        "interaction unidentifiable: lower GLS design is rank-deficient"
                            .to_string(),
                    );
                }
                matrix[row * dim + column] = value.sqrt();
            } else {
                let diagonal = matrix[column * dim + column];
                if !diagonal.is_finite() || diagonal <= 0.0 {
                    return Err(
                        "interaction unidentifiable: lower GLS design is rank-deficient"
                            .to_string(),
                    );
                }
                matrix[row * dim + column] = value / diagonal;
            }
        }
        for column in (row + 1)..dim {
            matrix[row * dim + column] = 0.0;
        }
    }
    Ok(())
}

fn solve_small_cholesky(
    lower: &[f64],
    dim: usize,
    rhs: &[f64],
    out: &mut [f64],
) -> Result<(), String> {
    if lower.len() != dim.saturating_mul(dim) || rhs.len() != dim || out.len() != dim {
        return Err("small Cholesky solve dimension mismatch".to_string());
    }
    out.copy_from_slice(rhs);
    for row in 0..dim {
        let mut value = out[row];
        for column in 0..row {
            value -= lower[row * dim + column] * out[column];
        }
        let diagonal = lower[row * dim + row];
        if !diagonal.is_finite() || diagonal <= 0.0 {
            return Err("non-positive small Cholesky diagonal".to_string());
        }
        out[row] = value / diagonal;
    }
    for row in (0..dim).rev() {
        let mut value = out[row];
        for column in (row + 1)..dim {
            value -= lower[column * dim + row] * out[column];
        }
        let diagonal = lower[row * dim + row];
        out[row] = value / diagonal;
    }
    if out.iter().any(|value| !value.is_finite()) {
        Err("non-finite small Cholesky solution".to_string())
    } else {
        Ok(())
    }
}

#[inline]
fn dot(left: &[f64], right: &[f64]) -> f64 {
    debug_assert_eq!(left.len(), right.len());
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    if left.len() <= CblasInt::MAX as usize {
        // The fixed-V scanner performs many long reductions.  Use the
        // platform BLAS dot kernel for those reductions; the scalar fallback
        // remains available for non-BLAS targets and oversized slices.
        return unsafe {
            cblas_ddot_dispatch(left.len() as CblasInt, left.as_ptr(), 1, right.as_ptr(), 1)
        };
    }
    left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
}

#[inline]
fn dot_strided(left: &[f64], right: &[f64], offset: usize, stride: usize) -> f64 {
    debug_assert!(stride > 0);
    debug_assert!(
        offset.saturating_add(left.len().saturating_sub(1).saturating_mul(stride)) < right.len()
    );
    let mut result = 0.0;
    for (index, value) in left.iter().enumerate() {
        result += *value * right[offset + index * stride];
    }
    result
}

#[inline]
fn dot_strided_both(
    left: &[f64],
    right: &[f64],
    offset: usize,
    stride: usize,
    length: usize,
) -> f64 {
    debug_assert!(stride > 0);
    debug_assert!(
        offset.saturating_add(length.saturating_sub(1).saturating_mul(stride)) < left.len()
    );
    debug_assert!(
        offset.saturating_add(length.saturating_sub(1).saturating_mul(stride)) < right.len()
    );
    let mut result = 0.0;
    for index in 0..length {
        let position = offset + index * stride;
        result += left[position] * right[position];
    }
    result
}

#[inline]
fn pair_index(n_markers: usize, first: usize, second: usize) -> usize {
    debug_assert!(first < second && second < n_markers);
    first * (2 * n_markers - first - 1) / 2 + (second - first - 1)
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

/// Solve multiple right-hand sides stored row-major as `n_rows x n_rhs`.
///
/// The scalar path above is intentionally kept as the reference operation.
/// This blocked variant performs the same forward/back substitutions while
/// keeping a small group of right-hand sides adjacent in memory, improving
/// cache reuse when the pair scanner processes several interactions together.
#[cfg(test)]
#[inline]
fn solve_cholesky_block_into(
    chol: &Cholesky<f64, Dyn>,
    rhs: &[f64],
    out: &mut [f64],
    n_rhs: usize,
) -> Result<(), String> {
    let n = chol.l_dirty().nrows();
    if n_rhs == 0 || rhs.len() != n.saturating_mul(n_rhs) || out.len() != rhs.len() {
        return Err("blocked Cholesky solve dimension mismatch".to_string());
    }
    out.copy_from_slice(rhs);
    let lower = chol.l_dirty();
    for pivot in 0..n {
        let diagonal = lower[(pivot, pivot)];
        if !diagonal.is_finite() || diagonal <= 0.0 {
            return Err("non-positive Cholesky diagonal".to_string());
        }
        for column in 0..n_rhs {
            out[pivot * n_rhs + column] /= diagonal;
        }
        for row in (pivot + 1)..n {
            let coefficient = lower[(row, pivot)];
            for column in 0..n_rhs {
                out[row * n_rhs + column] -= coefficient * out[pivot * n_rhs + column];
            }
        }
    }
    for pivot in (0..n).rev() {
        let diagonal = lower[(pivot, pivot)];
        for column in 0..n_rhs {
            out[pivot * n_rhs + column] /= diagonal;
        }
        for row in 0..pivot {
            let coefficient = lower[(pivot, row)];
            for column in 0..n_rhs {
                out[row * n_rhs + column] -= coefficient * out[pivot * n_rhs + column];
            }
        }
    }
    if out.iter().any(|value| !value.is_finite()) {
        Err("non-finite blocked Cholesky solution".to_string())
    } else {
        Ok(())
    }
}

/// Solve a column-major block of right-hand sides with the selected CBLAS
/// triangular-solve kernel, preserving the copying API used by the reference
/// path.  `rhs` and `out` are `n x n_rhs` matrices with leading dimension `n`;
/// callers that already own a mutable RHS buffer should use the in-place
/// variant below to avoid this copy.
#[cfg(test)]
#[inline]
fn solve_cholesky_block_blas_into(
    chol: &Cholesky<f64, Dyn>,
    rhs: &[f64],
    out: &mut [f64],
    n_rhs: usize,
) -> Result<(), String> {
    let n = chol.l_dirty().nrows();
    if n_rhs == 0 || rhs.len() != n.saturating_mul(n_rhs) || out.len() != rhs.len() {
        return Err("blocked Cholesky solve dimension mismatch".to_string());
    }
    out.copy_from_slice(rhs);
    solve_cholesky_block_blas_in_place(chol, out, n_rhs)
}

/// Solve a column-major block of right-hand sides in place with the selected
/// CBLAS triangular-solve kernel.  The caller must already have populated
/// `rhs_and_out` with an `n x n_rhs` block.  Keeping this variant separate
/// lets block scanners avoid copying an interaction block into a second
/// buffer before the two triangular solves.
#[inline]
fn solve_cholesky_block_blas_in_place(
    chol: &Cholesky<f64, Dyn>,
    rhs_and_out: &mut [f64],
    n_rhs: usize,
) -> Result<(), String> {
    let n = chol.l_dirty().nrows();
    if n_rhs == 0 || rhs_and_out.len() != n.saturating_mul(n_rhs) {
        return Err("in-place blocked Cholesky solve dimension mismatch".to_string());
    }
    let n_blas = CblasInt::try_from(n)
        .map_err(|_| "blocked Cholesky dimension exceeds CBLAS integer range".to_string())?;
    let n_rhs_blas = CblasInt::try_from(n_rhs)
        .map_err(|_| "blocked Cholesky RHS count exceeds CBLAS integer range".to_string())?;
    let lower = chol.l_dirty();
    for index in 0..n {
        let diagonal = lower[(index, index)];
        if !diagonal.is_finite() || diagonal <= 0.0 {
            return Err("non-positive Cholesky diagonal".to_string());
        }
    }
    // BLAS is intentionally held at one thread by the caller while the outer
    // scanner distributes independent first-marker ranges with Rayon.
    unsafe {
        cblas_dtrsm_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_LEFT,
            CBLAS_LOWER,
            CBLAS_NO_TRANS,
            CBLAS_DIAG_NON_UNIT,
            n_blas,
            n_rhs_blas,
            1.0,
            lower.as_slice().as_ptr(),
            n_blas,
            rhs_and_out.as_mut_ptr(),
            n_blas,
        );
        cblas_dtrsm_dispatch(
            CBLAS_COL_MAJOR,
            CBLAS_LEFT,
            CBLAS_LOWER,
            CBLAS_TRANS,
            CBLAS_DIAG_NON_UNIT,
            n_blas,
            n_rhs_blas,
            1.0,
            lower.as_slice().as_ptr(),
            n_blas,
            rhs_and_out.as_mut_ptr(),
            n_blas,
        );
    }
    if rhs_and_out.iter().any(|value| !value.is_finite()) {
        Err("non-finite blocked Cholesky solution".to_string())
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
    let profile = grm_profile_enabled();
    let scan_started = profile.then(Instant::now);
    validate_scan_inputs(genotypes, n_markers, n_samples, y, grm)?;
    let context_started = profile.then(Instant::now);
    let context = FixedVContext::new(
        y,
        grm,
        n_samples,
        sigma_g2,
        sigma_e2,
        covariates,
        n_covariates,
    )?;
    let context_elapsed = context_started.map_or(Duration::ZERO, |started| started.elapsed());
    let marker_started = profile.then(Instant::now);
    let vinv_markers =
        context.precompute_marker_vinv_with_threads(genotypes, n_markers, threads)?;
    let marker_stats = context.precompute_marker_vinv_stats(genotypes, &vinv_markers, n_markers)?;
    let marker_elapsed = marker_started.map_or(Duration::ZERO, |started| started.elapsed());
    // The outer Rayon scanner owns the parallelism; keep each BLAS triangular
    // solve single-threaded to avoid Rayon × BLAS oversubscription.
    let _blas_guard = BlasThreadGuard::enter(1);
    let first_count = n_markers.saturating_sub(1);
    let scan_range = |range: std::ops::Range<usize>| -> Result<GrmScanRangeResult, String> {
        let mut accumulator = GrmPairAccumulator::default();
        let mut timing = GrmScanTiming::default();
        let mut workspace = PairWorkspace::new(n_samples, context.fixed_rank + 2);
        let block_capacity = GRM_INTERACTION_BLOCK_SIZE;
        let mut block_interaction = vec![0.0; n_samples * block_capacity];
        let mut block_interaction_cross_y = vec![0.0; block_capacity];
        for first in range {
            let first_start = first * n_samples;
            let first_values = &genotypes[first_start..first_start + n_samples];
            let first_vinv_start = first * n_samples;
            let first_vinv = &vinv_markers[first_vinv_start..first_vinv_start + n_samples];
            let mut second_start = first + 1;
            while second_start < n_markers {
                let block_width = (n_markers - second_start).min(block_capacity);
                let interaction_build_started = profile.then(Instant::now);
                for local in 0..block_width {
                    let second = second_start + local;
                    let block_column = local * n_samples;
                    let second_values = &genotypes[second * n_samples..(second + 1) * n_samples];
                    let mut interaction_cross_y = 0.0;
                    for row in 0..n_samples {
                        let value = first_values[row] * second_values[row];
                        block_interaction[block_column + row] = value;
                        interaction_cross_y += value * context.vinv_y[row];
                    }
                    block_interaction_cross_y[local] = interaction_cross_y;
                }
                if let Some(started) = interaction_build_started {
                    timing.interaction_build += started.elapsed();
                }
                timing.blocks = timing.blocks.saturating_add(1);
                let interaction_solve_started = profile.then(Instant::now);
                solve_cholesky_block_blas_in_place(
                    &context.chol_v,
                    &mut block_interaction[..n_samples * block_width],
                    block_width,
                )?;
                if let Some(started) = interaction_solve_started {
                    timing.interaction_solve += started.elapsed();
                }
                let pair_geometry_started = profile.then(Instant::now);
                for local in 0..block_width {
                    let second = second_start + local;
                    let second_start_offset = second * n_samples;
                    let second_values =
                        &genotypes[second_start_offset..second_start_offset + n_samples];
                    let second_vinv_start = second * n_samples;
                    let second_vinv =
                        &vinv_markers[second_vinv_start..second_vinv_start + n_samples];
                    let fixed_rank = context.fixed_rank;
                    let pair_stats = MarkerPairVinvStats {
                        first_fixed_cross: &marker_stats.fixed_cross
                            [first * fixed_rank..(first + 1) * fixed_rank],
                        second_fixed_cross: &marker_stats.fixed_cross
                            [second * fixed_rank..(second + 1) * fixed_rank],
                        first_self_cross: marker_stats.self_cross[first],
                        cross: dot(first_values, second_vinv),
                        second_self_cross: marker_stats.self_cross[second],
                    };
                    let interaction_offset = local * n_samples;
                    let mut interaction_raw_q = 0.0;
                    for row in 0..n_samples {
                        interaction_raw_q += first_values[row]
                            * second_values[row]
                            * block_interaction[interaction_offset + row];
                    }
                    let projected_variance = context.build_pair_geometry_with_raw_q_in_place(
                        first_values,
                        second_values,
                        first_vinv,
                        second_vinv,
                        &block_interaction[..n_samples * block_width],
                        interaction_raw_q,
                        interaction_offset,
                        1,
                        Some(pair_stats),
                        &mut workspace.lower_gram,
                        &mut workspace.rhs_interaction,
                        &mut workspace.solved_interaction,
                    );
                    match projected_variance {
                        Ok(projected_variance) => {
                            let lower_dim = context.fixed_rank + 2;
                            let interaction_cross_y = block_interaction_cross_y[local];
                            let statistic = context.score_pair_with_scratch(
                                first_values,
                                second_values,
                                &workspace.lower_gram[..lower_dim * lower_dim],
                                &workspace.rhs_interaction[..lower_dim],
                                &workspace.solved_interaction[..lower_dim],
                                projected_variance,
                                interaction_cross_y,
                                Some((marker_stats.y_cross[first], marker_stats.y_cross[second])),
                                &mut workspace.rhs_y,
                                &mut workspace.solved_y,
                                None,
                            );
                            match statistic {
                                Ok(statistic) => {
                                    accumulator.pairs_evaluated =
                                        accumulator.pairs_evaluated.saturating_add(1);
                                    let geometry = PairGeometryView {
                                        lower_chol: &workspace.lower_gram[..lower_dim * lower_dim],
                                        rhs_interaction: &workspace.rhs_interaction[..lower_dim],
                                        solved_interaction: &workspace.solved_interaction
                                            [..lower_dim],
                                        projected_variance,
                                    };
                                    accumulator.push_with_geometry(
                                        pair_index(n_markers, first, second),
                                        first,
                                        second,
                                        statistic,
                                        geometry,
                                        interaction_cross_y,
                                        top_k,
                                    );
                                }
                                Err(error) if error.contains("unidentifiable") => {
                                    accumulator.pairs_skipped =
                                        accumulator.pairs_skipped.saturating_add(1);
                                }
                                Err(error) => return Err(error),
                            }
                        }
                        Err(error) if error.contains("unidentifiable") => {
                            accumulator.pairs_skipped = accumulator.pairs_skipped.saturating_add(1);
                        }
                        Err(error) => return Err(error),
                    }
                }
                if let Some(started) = pair_geometry_started {
                    timing.pair_geometry_score += started.elapsed();
                }
                second_start += block_width;
            }
        }
        Ok(GrmScanRangeResult {
            accumulator,
            timing,
        })
    };

    let (accumulator, mut timing) = if threads <= 1 || first_count <= 1 {
        let result = scan_range(0..first_count)?;
        (result.accumulator, result.timing)
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
        let mut timing = GrmScanTiming::default();
        for partial in partials {
            timing += partial.timing;
            merged.merge(partial.accumulator, top_k);
        }
        (merged, timing)
    };

    let mut entries = accumulator
        .heap
        .into_iter()
        .map(|Reverse(entry)| entry)
        .collect::<Vec<_>>();
    entries.sort_by(|left, right| candidate_cmp(right, left));
    let mut candidates = Vec::with_capacity(entries.len());
    let mut final_workspace = PairWorkspace::new(n_samples, context.fixed_rank + 2);
    let mut final_beta = Vec::new();
    let final_fit_started = profile.then(Instant::now);
    for entry in entries {
        let first_start = entry.first * n_samples;
        let second_start = entry.second * n_samples;
        let first_values = &genotypes[first_start..first_start + n_samples];
        let second_values = &genotypes[second_start..second_start + n_samples];
        let fit = if let (Some(geometry), Some(interaction_cross_y)) =
            (entry.geometry.as_ref(), entry.interaction_cross_y)
        {
            let geometry = PairGeometryView {
                lower_chol: &geometry.lower_chol,
                rhs_interaction: &geometry.rhs_interaction,
                solved_interaction: &geometry.solved_interaction,
                projected_variance: geometry.projected_variance,
            };
            context.fit_pair_with_geometry_view_and_cross_y_into(
                first_values,
                second_values,
                geometry,
                interaction_cross_y,
                Some((
                    marker_stats.y_cross[entry.first],
                    marker_stats.y_cross[entry.second],
                )),
                &mut final_workspace,
                &mut final_beta,
            )?
        } else {
            // All entries produced by `scan_grm_pairs` carry geometry.  Keep a
            // defensive fallback for accumulator users that do not, while
            // still reusing marker solves and cached marker reductions.
            let first_vinv = &vinv_markers[first_start..first_start + n_samples];
            let second_vinv = &vinv_markers[second_start..second_start + n_samples];
            let fixed_rank = context.fixed_rank;
            let pair_stats = MarkerPairVinvStats {
                first_fixed_cross: &marker_stats.fixed_cross
                    [entry.first * fixed_rank..(entry.first + 1) * fixed_rank],
                second_fixed_cross: &marker_stats.fixed_cross
                    [entry.second * fixed_rank..(entry.second + 1) * fixed_rank],
                first_self_cross: marker_stats.self_cross[entry.first],
                cross: dot(first_values, second_vinv),
                second_self_cross: marker_stats.self_cross[entry.second],
            };
            context.fit_pair_with_precomputed_marker_vinv(
                first_values,
                second_values,
                first_vinv,
                second_vinv,
                pair_stats,
                (
                    marker_stats.y_cross[entry.first],
                    marker_stats.y_cross[entry.second],
                ),
                &mut final_workspace,
                &mut final_beta,
            )?
        };
        candidates.push(GrmPairCandidate {
            first: entry.first,
            second: entry.second,
            fit,
        });
    }
    if let Some(started) = final_fit_started {
        timing.final_fit += started.elapsed();
    }
    if let Some(started) = scan_started {
        let total = started.elapsed();
        eprintln!(
            "[grm-profile] n={} m={} threads={} total={:.6}s context={:.6}s marker_vinv={:.6}s interaction_build={:.6}s interaction_solve={:.6}s pair_geometry_score={:.6}s final_fit={:.6}s blocks={} pairs={} skipped={}",
            n_samples,
            n_markers,
            threads,
            total.as_secs_f64(),
            context_elapsed.as_secs_f64(),
            marker_elapsed.as_secs_f64(),
            timing.interaction_build.as_secs_f64(),
            timing.interaction_solve.as_secs_f64(),
            timing.pair_geometry_score.as_secs_f64(),
            timing.final_fit.as_secs_f64(),
            timing.blocks,
            accumulator.pairs_evaluated,
            accumulator.pairs_skipped,
        );
    }
    Ok((
        candidates,
        accumulator.pairs_evaluated,
        accumulator.pairs_skipped,
    ))
}

/// Fit the null GRM model once and scan all pairs with the resulting fixed
/// covariance.  This is deliberately an internal helper first: callers can
/// compare the automatic path with `scan_grm_pairs` supplied with the same
/// variance components before any CLI defaults are changed.
#[allow(clippy::too_many_arguments)]
fn scan_grm_pairs_reml(
    genotypes: &[f64],
    n_markers: usize,
    n_samples: usize,
    y: &[f64],
    grm: &[f64],
    top_k: usize,
    threads: usize,
    covariates: &[f64],
    n_covariates: usize,
) -> Result<
    (
        (Vec<GrmPairCandidate>, usize, usize),
        GarfieldResidualResult,
    ),
    String,
> {
    validate_scan_inputs(genotypes, n_markers, n_samples, y, grm)?;
    let x_cov = if n_covariates == 0 {
        None
    } else {
        Some(covariates.to_vec())
    };
    let reml = garfield_residualize_exact_from_grm_rust(
        grm.to_vec(),
        n_samples,
        y.to_vec(),
        x_cov,
        n_covariates,
        threads,
        -5.0,
        5.0,
        50,
        1.0e-3,
        true,
        15_000,
        false,
        None,
        0,
        0,
    )?;
    let scan = scan_grm_pairs(
        genotypes,
        n_markers,
        n_samples,
        y,
        grm,
        reml.sigma_g2,
        reml.sigma_e2,
        top_k,
        threads,
        covariates,
        n_covariates,
    )?;
    Ok((scan, reml))
}

/// Reusable fixed-\( V \) scan state.  The pair geometry is genotype-only,
/// so one preparation can score many response vectors (for example, the
/// phenotype replicates in a matched benchmark) without recomputing
/// (V^{-1}(g_i g_j)).
struct PreparedGrmScan {
    context: FixedVContext,
    genotypes: Vec<f64>,
    n_samples: usize,
    pairs: Vec<PreparedPair>,
    geometry_dim: usize,
    geometry_lower_chol: Vec<f64>,
    geometry_rhs_interaction: Vec<f64>,
    geometry_solved_interaction: Vec<f64>,
    geometry_projected_variance: Vec<f64>,
    pairs_skipped: usize,
}

struct PreparedRangeResult {
    pairs: Vec<PreparedPair>,
    geometry_lower_chol: Vec<f64>,
    geometry_rhs_interaction: Vec<f64>,
    geometry_solved_interaction: Vec<f64>,
    geometry_projected_variance: Vec<f64>,
    pairs_skipped: usize,
}

impl PreparedGrmScan {
    #[allow(clippy::too_many_arguments)]
    fn new(
        genotypes: &[f64],
        n_markers: usize,
        n_samples: usize,
        y: &[f64],
        grm: &[f64],
        sigma_g2: f64,
        sigma_e2: f64,
        _top_k: usize,
        threads: usize,
        covariates: &[f64],
        n_covariates: usize,
    ) -> Result<Self, String> {
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
        let vinv_markers =
            context.precompute_marker_vinv_with_threads(genotypes, n_markers, threads)?;
        let marker_stats =
            context.precompute_marker_vinv_stats(genotypes, &vinv_markers, n_markers)?;
        let _blas_guard = BlasThreadGuard::enter(1);
        let first_count = n_markers.saturating_sub(1);
        let geometry_dim = context.fixed_rank + 2;
        let build_range = |range: std::ops::Range<usize>| -> Result<PreparedRangeResult, String> {
            let mut result = PreparedRangeResult {
                pairs: Vec::new(),
                geometry_lower_chol: Vec::new(),
                geometry_rhs_interaction: Vec::new(),
                geometry_solved_interaction: Vec::new(),
                geometry_projected_variance: Vec::new(),
                pairs_skipped: 0,
            };
            let mut workspace = PairWorkspace::new(n_samples, context.fixed_rank + 2);
            let block_capacity = GRM_INTERACTION_BLOCK_SIZE;
            let mut block_interaction = vec![0.0; n_samples * block_capacity];
            for first in range {
                let first_start = first * n_samples;
                let first_values = &genotypes[first_start..first_start + n_samples];
                let first_vinv_start = first * n_samples;
                let first_vinv = &vinv_markers[first_vinv_start..first_vinv_start + n_samples];
                let mut second_start = first + 1;
                while second_start < n_markers {
                    let block_width = (n_markers - second_start).min(block_capacity);
                    for local in 0..block_width {
                        let second = second_start + local;
                        let block_column = local * n_samples;
                        let second_values =
                            &genotypes[second * n_samples..(second + 1) * n_samples];
                        for row in 0..n_samples {
                            block_interaction[block_column + row] =
                                first_values[row] * second_values[row];
                        }
                    }
                    solve_cholesky_block_blas_in_place(
                        &context.chol_v,
                        &mut block_interaction[..n_samples * block_width],
                        block_width,
                    )?;
                    for local in 0..block_width {
                        let second = second_start + local;
                        let second_offset = second * n_samples;
                        let second_values = &genotypes[second_offset..second_offset + n_samples];
                        let pair_stats = MarkerPairVinvStats {
                            first_fixed_cross: &marker_stats.fixed_cross
                                [first * context.fixed_rank..(first + 1) * context.fixed_rank],
                            second_fixed_cross: &marker_stats.fixed_cross
                                [second * context.fixed_rank..(second + 1) * context.fixed_rank],
                            first_self_cross: marker_stats.self_cross[first],
                            cross: dot(
                                first_values,
                                &vinv_markers[second * n_samples..(second + 1) * n_samples],
                            ),
                            second_self_cross: marker_stats.self_cross[second],
                        };
                        let interaction_offset = local * n_samples;
                        let mut interaction_raw_q = 0.0;
                        for row in 0..n_samples {
                            interaction_raw_q += first_values[row]
                                * second_values[row]
                                * block_interaction[interaction_offset + row];
                        }
                        let geometry = context.build_pair_geometry_with_raw_q_in_place(
                            first_values,
                            second_values,
                            first_vinv,
                            &vinv_markers[second * n_samples..(second + 1) * n_samples],
                            &block_interaction[..n_samples * block_width],
                            interaction_raw_q,
                            interaction_offset,
                            1,
                            Some(pair_stats),
                            &mut workspace.lower_gram,
                            &mut workspace.rhs_interaction,
                            &mut workspace.solved_interaction,
                        );
                        match geometry {
                            Ok(projected_variance) => {
                                let geometry_index = result.pairs.len();
                                result.geometry_lower_chol.extend_from_slice(
                                    &workspace.lower_gram[..geometry_dim * geometry_dim],
                                );
                                result
                                    .geometry_rhs_interaction
                                    .extend_from_slice(&workspace.rhs_interaction[..geometry_dim]);
                                result.geometry_solved_interaction.extend_from_slice(
                                    &workspace.solved_interaction[..geometry_dim],
                                );
                                result.geometry_projected_variance.push(projected_variance);
                                result.pairs.push(PreparedPair {
                                    first,
                                    second,
                                    geometry_index,
                                });
                            }
                            Err(error) if error.contains("unidentifiable") => {
                                result.pairs_skipped = result.pairs_skipped.saturating_add(1)
                            }
                            Err(error) => return Err(error),
                        }
                    }
                    second_start += block_width;
                }
            }
            Ok(result)
        };

        let effective_threads = threads.max(1);
        let partials = if effective_threads <= 1 || first_count <= 1 {
            vec![build_range(0..first_count)?]
        } else {
            let chunk_count = effective_threads.saturating_mul(4).max(1);
            let chunk_len = (first_count + chunk_count - 1) / chunk_count;
            let ranges = (0..first_count)
                .step_by(chunk_len.max(1))
                .map(|start| start..(start + chunk_len).min(first_count))
                .collect::<Vec<_>>();
            let partials = ThreadPoolBuilder::new()
                .num_threads(effective_threads)
                .build()
                .map_err(|error| format!("failed to build GRM dosage Rayon pool: {error}"))?
                .install(|| {
                    ranges
                        .par_iter()
                        .map(|range| build_range(range.clone()))
                        .collect::<Result<Vec<_>, String>>()
                })?;
            partials
        };

        let mut combined = PreparedRangeResult {
            pairs: Vec::new(),
            geometry_lower_chol: Vec::new(),
            geometry_rhs_interaction: Vec::new(),
            geometry_solved_interaction: Vec::new(),
            geometry_projected_variance: Vec::new(),
            pairs_skipped: 0,
        };
        for mut partial in partials {
            let geometry_base = combined.pairs.len();
            for pair in &mut partial.pairs {
                pair.geometry_index += geometry_base;
            }
            combined.pairs.extend(partial.pairs);
            combined
                .geometry_lower_chol
                .extend(partial.geometry_lower_chol);
            combined
                .geometry_rhs_interaction
                .extend(partial.geometry_rhs_interaction);
            combined
                .geometry_solved_interaction
                .extend(partial.geometry_solved_interaction);
            combined
                .geometry_projected_variance
                .extend(partial.geometry_projected_variance);
            combined.pairs_skipped = combined.pairs_skipped.saturating_add(partial.pairs_skipped);
        }
        let mut pairs = combined.pairs;
        pairs.sort_unstable_by_key(|pair| (pair.first, pair.second));
        Ok(Self {
            context,
            genotypes: genotypes.to_vec(),
            n_samples,
            pairs,
            geometry_dim,
            geometry_lower_chol: combined.geometry_lower_chol,
            geometry_rhs_interaction: combined.geometry_rhs_interaction,
            geometry_solved_interaction: combined.geometry_solved_interaction,
            geometry_projected_variance: combined.geometry_projected_variance,
            pairs_skipped: combined.pairs_skipped,
        })
    }

    #[cfg(test)]
    fn geometry_entries(&self) -> usize {
        self.pairs.len()
    }

    #[cfg(test)]
    fn geometry_storage_len(&self) -> usize {
        self.geometry_lower_chol.len()
            + self.geometry_rhs_interaction.len()
            + self.geometry_solved_interaction.len()
    }

    #[inline]
    fn geometry_view(&self, pair: &PreparedPair) -> PairGeometryView<'_> {
        let lower_start = pair.geometry_index * self.geometry_dim * self.geometry_dim;
        let vector_start = pair.geometry_index * self.geometry_dim;
        PairGeometryView {
            lower_chol: &self.geometry_lower_chol
                [lower_start..lower_start + self.geometry_dim * self.geometry_dim],
            rhs_interaction: &self.geometry_rhs_interaction
                [vector_start..vector_start + self.geometry_dim],
            solved_interaction: &self.geometry_solved_interaction
                [vector_start..vector_start + self.geometry_dim],
            projected_variance: self.geometry_projected_variance[pair.geometry_index],
        }
    }

    fn scan_many(
        &mut self,
        responses: &[&[f64]],
        top_k: usize,
        threads: usize,
    ) -> Result<Vec<(Vec<GrmPairCandidate>, usize, usize)>, String> {
        let mut results = Vec::with_capacity(responses.len());
        for response in responses {
            self.context.set_response(response)?;
            results.push(self.scan_one(top_k, threads)?);
        }
        Ok(results)
    }

    fn scan_one(
        &self,
        top_k: usize,
        threads: usize,
    ) -> Result<(Vec<GrmPairCandidate>, usize, usize), String> {
        let scan_range = |range: std::ops::Range<usize>| -> Result<GrmPairAccumulator, String> {
            let mut accumulator = GrmPairAccumulator::default();
            let mut workspace = PairWorkspace::new(self.n_samples, self.context.fixed_rank + 2);
            for pair_index in range {
                let pair = &self.pairs[pair_index];
                let first_start = pair.first * self.n_samples;
                let second_start = pair.second * self.n_samples;
                let first_values = &self.genotypes[first_start..first_start + self.n_samples];
                let second_values = &self.genotypes[second_start..second_start + self.n_samples];
                match self.context.score_pair_with_geometry_view(
                    first_values,
                    second_values,
                    self.geometry_view(pair),
                    &mut workspace,
                    None,
                ) {
                    Ok(statistic) => {
                        accumulator.pairs_evaluated = accumulator.pairs_evaluated.saturating_add(1);
                        accumulator.push(pair_index, pair.first, pair.second, statistic, top_k);
                    }
                    Err(error) if error.contains("unidentifiable") => {
                        accumulator.pairs_skipped = accumulator.pairs_skipped.saturating_add(1);
                    }
                    Err(error) => return Err(error),
                }
            }
            Ok(accumulator)
        };
        let effective_threads = threads.max(1);
        let accumulator = if effective_threads <= 1 || self.pairs.len() <= 1 {
            scan_range(0..self.pairs.len())?
        } else {
            let chunk_count = effective_threads.saturating_mul(4).max(1);
            let chunk_len = (self.pairs.len() + chunk_count - 1) / chunk_count;
            let ranges = (0..self.pairs.len())
                .step_by(chunk_len.max(1))
                .map(|start| start..(start + chunk_len).min(self.pairs.len()))
                .collect::<Vec<_>>();
            let partials = ThreadPoolBuilder::new()
                .num_threads(effective_threads)
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
        let mut final_workspace = PairWorkspace::new(self.n_samples, self.context.fixed_rank + 2);
        let mut final_beta = Vec::new();
        for entry in entries {
            let pair = &self.pairs[entry.pair_index];
            let first_start = pair.first * self.n_samples;
            let second_start = pair.second * self.n_samples;
            let first_values = &self.genotypes[first_start..first_start + self.n_samples];
            let second_values = &self.genotypes[second_start..second_start + self.n_samples];
            let fit = self.context.fit_pair_with_geometry_view_into(
                first_values,
                second_values,
                self.geometry_view(pair),
                &mut final_workspace,
                &mut final_beta,
            )?;
            candidates.push(GrmPairCandidate {
                first: pair.first,
                second: pair.second,
                fit,
            });
        }
        Ok((candidates, accumulator.pairs_evaluated, self.pairs_skipped))
    }
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

fn set_scan_items<'py>(
    py: Python<'py>,
    out: &Bound<'py, PyDict>,
    candidates: &[GrmPairCandidate],
    pairs_evaluated: usize,
    pairs_skipped: usize,
    n_markers: usize,
    n_samples: usize,
    threads: usize,
    sigma_g2: f64,
    sigma_e2: f64,
) -> PyResult<()> {
    let output_candidates = PyList::empty(py);
    for candidate in candidates {
        let item = PyDict::new(py);
        item.set_item("first", candidate.first)?;
        item.set_item("second", candidate.second)?;
        set_fit_items(py, &item, &candidate.fit)?;
        output_candidates.append(item)?;
    }
    out.set_item("candidates", output_candidates)?;
    out.set_item("pairs_evaluated", pairs_evaluated)?;
    out.set_item("pairs_skipped", pairs_skipped)?;
    out.set_item("n_markers", n_markers)?;
    out.set_item("n_samples", n_samples)?;
    out.set_item("threads", threads)?;
    out.set_item("sigma_g2", sigma_g2)?;
    out.set_item("sigma_e2", sigma_e2)?;
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

/// Batch variant for matched responses sharing one genotype window, GRM and
/// pair-search geometry.  Each response still gets an independent fixed-V
/// GLS/FWL score; only genotype-only quantities are reused.
#[pyfunction(name = "garfield_dosage_grm_pair_scan_batch")]
#[pyo3(signature = (genotypes, phenotypes, grm, sigma_g2, sigma_e2, top_k=100, threads=0, covariates=None))]
pub fn garfield_dosage_grm_pair_scan_batch_py<'py>(
    py: Python<'py>,
    genotypes: PyReadonlyArray2<'py, f64>,
    phenotypes: PyReadonlyArray2<'py, f64>,
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
    let phenotype_shape = phenotypes.shape();
    if phenotype_shape.len() != 2 || phenotype_shape[0] == 0 || phenotype_shape[1] != n_samples {
        return Err(PyValueError::new_err(format!(
            "phenotypes must have shape (n_responses, {n_samples}); got {:?}",
            phenotype_shape
        )));
    }
    let grm_shape = grm.shape();
    if grm_shape.len() != 2 || grm_shape[0] != grm_shape[1] || grm_shape[0] != n_samples {
        return Err(PyValueError::new_err(format!(
            "grm must have shape ({n_samples}, {n_samples}); got {:?}",
            grm_shape
        )));
    }
    let genotype_vec = array2_to_vec(&genotypes);
    let phenotype_vec = array2_to_vec(&phenotypes);
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
    let first_response = &phenotype_vec[..n_samples];
    let mut prepared = PreparedGrmScan::new(
        &genotype_vec,
        n_markers,
        n_samples,
        first_response,
        &grm_vec,
        sigma_g2,
        sigma_e2,
        top_k,
        effective_threads,
        &covariate_vec,
        n_covariates,
    )
    .map_err(PyValueError::new_err)?;
    let responses = phenotype_vec.chunks_exact(n_samples).collect::<Vec<_>>();
    let scans = prepared
        .scan_many(&responses, top_k, effective_threads)
        .map_err(PyValueError::new_err)?;
    let output_scans = PyList::empty(py);
    for (candidates, pairs_evaluated, pairs_skipped) in scans {
        let item = PyDict::new(py);
        set_scan_items(
            py,
            &item,
            &candidates,
            pairs_evaluated,
            pairs_skipped,
            n_markers,
            n_samples,
            effective_threads,
            sigma_g2,
            sigma_e2,
        )?;
        output_scans.append(item)?;
    }
    let out = PyDict::new(py);
    out.set_item("scans", output_scans)?;
    out.set_item("n_responses", phenotype_shape[0])?;
    out.set_item("n_markers", n_markers)?;
    out.set_item("n_samples", n_samples)?;
    out.set_item("threads", effective_threads)?;
    out.set_item("sigma_g2", sigma_g2)?;
    out.set_item("sigma_e2", sigma_e2)?;
    Ok(out)
}

/// Fit one null REML model and scan all marker pairs with its fixed
/// covariance.  The null is fitted exactly once; variance components are not
/// re-estimated inside the pair loop.  This API is intentionally separate
/// from the existing fixed-V scan so callers can validate the two stages
/// independently before wiring a production CLI path.
#[pyfunction(name = "garfield_dosage_grm_pair_scan_reml")]
#[pyo3(signature = (genotypes, y, grm, top_k=100, threads=0, covariates=None))]
pub fn garfield_dosage_grm_pair_scan_reml_py<'py>(
    py: Python<'py>,
    genotypes: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    grm: PyReadonlyArray2<'py, f64>,
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
    let ((candidates, pairs_evaluated, pairs_skipped), reml) = scan_grm_pairs_reml(
        &genotype_vec,
        n_markers,
        n_samples,
        &y_vec,
        &grm_vec,
        top_k,
        effective_threads,
        &covariate_vec,
        n_covariates,
    )
    .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    set_scan_items(
        py,
        &out,
        &candidates,
        pairs_evaluated,
        pairs_skipped,
        n_markers,
        n_samples,
        effective_threads,
        reml.sigma_g2,
        reml.sigma_e2,
    )?;
    out.set_item("pve", reml.pve)?;
    out.set_item("reml_lbd", reml.lbd)?;
    out.set_item("reml_loglike", reml.reml)?;
    out.set_item("reml_n_fixed_effects", reml.n_fixed_effects)?;
    out.set_item("reml_eigh_backend", reml.eigh_backend)?;
    out.set_item("reml_eigh_elapsed", reml.eigh_elapsed)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

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
        let context = FixedVContext::new(&y, &grm, 8, 0.0, 1.0, &[], 0)
            .expect("identity covariance should be valid");
        for candidate in candidates {
            let first_start = candidate.first * 8;
            let second_start = candidate.second * 8;
            let reference = context
                .fit_pair(
                    &genotypes[first_start..first_start + 8],
                    &genotypes[second_start..second_start + 8],
                )
                .expect("direct pair fit should succeed");
            assert!(candidate
                .fit
                .beta
                .iter()
                .zip(reference.beta.iter())
                .all(|(left, right)| (left - right).abs() < 1.0e-10));
            assert!(
                (candidate.fit.interaction_score - reference.interaction_score).abs() < 1.0e-10
            );
            assert!((candidate.fit.delta_q - reference.delta_q).abs() < 1.0e-10);
            assert!((candidate.fit.null_q - reference.null_q).abs() < 1.0e-10);
            assert!((candidate.fit.full_q - reference.full_q).abs() < 1.0e-10);
        }
    }

    #[test]
    fn precomputed_marker_fit_matches_pairwise_reference() {
        let n_samples = 8;
        let n_markers = 3;
        let genotypes = [
            0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0,
            2.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0,
        ];
        let y = [1.2, 0.4, 2.1, 3.7, 1.5, 0.8, 2.4, 2.9];
        let grm = (0..n_samples * n_samples)
            .map(|index| {
                if index / n_samples == index % n_samples {
                    1.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let context = FixedVContext::new(&y, &grm, n_samples, 0.4, 0.6, &[], 0)
            .expect("identity covariance should be valid");
        let vinv_markers = context
            .precompute_marker_vinv_with_threads(&genotypes, n_markers, 1)
            .expect("marker solves should succeed");
        let marker_stats = context
            .precompute_marker_vinv_stats(&genotypes, &vinv_markers, n_markers)
            .expect("marker statistics should succeed");
        let first = &genotypes[..n_samples];
        let second = &genotypes[n_samples..2 * n_samples];
        let first_vinv = &vinv_markers[..n_samples];
        let second_vinv = &vinv_markers[n_samples..2 * n_samples];
        let marker_pair_stats = MarkerPairVinvStats {
            first_fixed_cross: &marker_stats.fixed_cross[..context.fixed_rank],
            second_fixed_cross: &marker_stats.fixed_cross
                [context.fixed_rank..2 * context.fixed_rank],
            first_self_cross: marker_stats.self_cross[0],
            cross: dot(first, second_vinv),
            second_self_cross: marker_stats.self_cross[1],
        };
        let mut workspace = PairWorkspace::new(n_samples, context.fixed_rank + 2);
        let mut beta = Vec::new();
        let optimized = context
            .fit_pair_with_precomputed_marker_vinv(
                first,
                second,
                first_vinv,
                second_vinv,
                marker_pair_stats,
                (marker_stats.y_cross[0], marker_stats.y_cross[1]),
                &mut workspace,
                &mut beta,
            )
            .expect("precomputed marker fit should succeed");
        let reference = context
            .fit_pair(first, second)
            .expect("reference pair fit should succeed");
        assert_eq!(optimized.residual_df, reference.residual_df);
        assert_eq!(optimized.fixed_rank, reference.fixed_rank);
        assert!(optimized
            .beta
            .iter()
            .zip(reference.beta.iter())
            .all(|(left, right)| (left - right).abs() < 1.0e-12));
        assert!((optimized.interaction_score - reference.interaction_score).abs() < 1.0e-12);
        assert!((optimized.delta_q - reference.delta_q).abs() < 1.0e-12);
        assert!((optimized.null_q - reference.null_q).abs() < 1.0e-12);
        assert!((optimized.full_q - reference.full_q).abs() < 1.0e-12);
    }

    #[test]
    fn prepared_scan_reuses_pair_geometry_for_multiple_responses() {
        let genotypes = [
            0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0,
            2.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0,
        ];
        let y1 = [1.2, 0.4, 2.1, 3.7, 1.5, 0.8, 2.4, 2.9];
        let y2 = [-0.2, 1.4, 0.1, 2.7, 0.5, 1.8, 1.4, 3.9];
        let grm = (0..64)
            .map(|index| if index / 8 == index % 8 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let mut prepared =
            PreparedGrmScan::new(&genotypes, 3, 8, &y1, &grm, 0.0, 1.0, 10, 1, &[], 0)
                .expect("prepared scan should succeed");
        assert_eq!(prepared.geometry_entries(), 3);
        assert_eq!(prepared.geometry_storage_len(), 3 * 9 + 3 * 3 + 3 * 3);
        let batch = prepared
            .scan_many(&[&y1, &y2], 10, 1)
            .expect("batch scan should succeed");
        assert_eq!(batch.len(), 2);
        let one = scan_grm_pairs(&genotypes, 3, 8, &y1, &grm, 0.0, 1.0, 10, 1, &[], 0)
            .expect("single scan should succeed");
        assert_eq!(batch[0].1, one.1);
        assert_eq!(batch[0].2, one.2);
        assert_eq!(batch[0].0, one.0);
    }

    #[test]
    fn reml_scan_matches_explicit_fixed_v_scan() {
        let genotypes = [
            0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0,
            2.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0,
        ];
        let y = [1.2, 0.4, 2.1, 3.7, 1.5, 0.8, 2.4, 2.9];
        let grm = (0..64)
            .map(|index| if index / 8 == index % 8 { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let ((automatic_candidates, automatic_evaluated, automatic_skipped), reml) =
            scan_grm_pairs_reml(&genotypes, 3, 8, &y, &grm, 10, 1, &[], 0)
                .expect("automatic REML scan should succeed");
        let explicit = scan_grm_pairs(
            &genotypes,
            3,
            8,
            &y,
            &grm,
            reml.sigma_g2,
            reml.sigma_e2,
            10,
            1,
            &[],
            0,
        )
        .expect("explicit fixed-V scan should succeed");
        assert_eq!(automatic_evaluated, explicit.1);
        assert_eq!(automatic_skipped, explicit.2);
        assert_eq!(automatic_candidates, explicit.0);
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

    #[test]
    fn marker_vinv_parallel_matches_serial() {
        let n_samples = 8;
        let n_markers = 3;
        let genotypes = [
            0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0,
            2.0, 1.0, 1.0, 0.0, 1.0, 2.0, 0.0,
        ];
        let y = [1.2, 0.4, 2.1, 3.7, 1.5, 0.8, 2.4, 2.9];
        let grm = (0..n_samples * n_samples)
            .map(|index| {
                if index / n_samples == index % n_samples {
                    1.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let context = FixedVContext::new(&y, &grm, n_samples, 0.4, 0.6, &[], 0)
            .expect("identity covariance should be valid");
        let serial = context
            .precompute_marker_vinv_with_threads(&genotypes, n_markers, 1)
            .expect("serial marker solve should succeed");
        let parallel = context
            .precompute_marker_vinv_with_threads(&genotypes, n_markers, 2)
            .expect("parallel marker solve should succeed");
        assert_eq!(serial.len(), n_markers * n_samples);
        assert_eq!(serial.len(), parallel.len());
        for (left, right) in serial.iter().zip(parallel.iter()) {
            assert!((left - right).abs() < 1.0e-12);
        }
    }

    #[test]
    fn in_place_pair_geometry_matches_owned_geometry() {
        let n_samples = 8;
        let g1 = [0.0, 1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0];
        let g2 = [1.0, 0.0, 2.0, 1.0, 2.0, 0.0, 1.0, 2.0];
        let y = [1.2, 0.4, 2.1, 3.7, 1.5, 0.8, 2.4, 2.9];
        let grm = (0..n_samples * n_samples)
            .map(|index| {
                if index / n_samples == index % n_samples {
                    1.0
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let context = FixedVContext::new(&y, &grm, n_samples, 0.0, 1.0, &[], 0)
            .expect("identity covariance should be valid");
        let mut owned_workspace = PairWorkspace::new(n_samples, context.fixed_rank + 2);
        let owned = context
            .build_pair_geometry(&g1, &g2, &g1, &g2, &mut owned_workspace)
            .expect("owned geometry should succeed");
        let mut in_place_workspace = PairWorkspace::new(n_samples, context.fixed_rank + 2);
        let variance = context
            .build_pair_geometry_in_place(&g1, &g2, &g1, &g2, &mut in_place_workspace)
            .expect("in-place geometry should succeed");
        let lower_dim = context.fixed_rank + 2;
        assert!((variance - owned.projected_variance).abs() < 1.0e-14);
        assert!(in_place_workspace.lower_gram[..lower_dim * lower_dim]
            .iter()
            .zip(owned.lower_chol.iter())
            .all(|(left, right)| (left - right).abs() < 1.0e-14));
        assert!(in_place_workspace.rhs_interaction[..lower_dim]
            .iter()
            .zip(owned.rhs_interaction.iter())
            .all(|(left, right)| (left - right).abs() < 1.0e-14));
        assert!(in_place_workspace.solved_interaction[..lower_dim]
            .iter()
            .zip(owned.solved_interaction.iter())
            .all(|(left, right)| (left - right).abs() < 1.0e-14));
    }

    #[test]
    fn blocked_cholesky_solve_matches_columnwise_solves() {
        let matrix =
            DMatrix::from_row_slice(3, 3, &[4.0, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 2.0]);
        let chol = Cholesky::new(matrix).expect("test matrix should be positive definite");
        let rhs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut expected = rhs;
        for column in 0..2 {
            let mut column_rhs = [rhs[column], rhs[2 + column], rhs[4 + column]];
            let original_rhs = column_rhs;
            solve_cholesky_into(&chol, &original_rhs, &mut column_rhs)
                .expect("column solve should succeed");
            expected[column] = column_rhs[0];
            expected[2 + column] = column_rhs[1];
            expected[4 + column] = column_rhs[2];
        }
        let mut blocked = vec![0.0; rhs.len()];
        solve_cholesky_block_into(&chol, &rhs, &mut blocked, 2)
            .expect("blocked solve should succeed");
        assert!(blocked
            .iter()
            .zip(expected.iter())
            .all(|(left, right)| (left - right).abs() < 1.0e-14));
    }

    #[test]
    fn blas_blocked_cholesky_solve_matches_columnwise_solves() {
        let matrix =
            DMatrix::from_row_slice(3, 3, &[4.0, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 2.0]);
        let chol = Cholesky::new(matrix).expect("test matrix should be positive definite");
        // The BLAS path uses column-major `n x n_rhs` right-hand sides.
        let rhs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut expected = rhs;
        for column in 0..2 {
            let original_rhs = [rhs[column * 3], rhs[column * 3 + 1], rhs[column * 3 + 2]];
            let mut solved = [0.0; 3];
            solve_cholesky_into(&chol, &original_rhs, &mut solved)
                .expect("column solve should succeed");
            expected[column * 3..column * 3 + 3].copy_from_slice(&solved);
        }
        let mut blocked = vec![0.0; rhs.len()];
        solve_cholesky_block_blas_into(&chol, &rhs, &mut blocked, 2)
            .expect("BLAS blocked solve should succeed");
        assert!(blocked
            .iter()
            .zip(expected.iter())
            .all(|(left, right)| (left - right).abs() < 1.0e-14));
    }

    #[test]
    fn blas_blocked_cholesky_solve_in_place_matches_copying_variant() {
        let matrix =
            DMatrix::from_row_slice(3, 3, &[4.0, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 2.0]);
        let chol = Cholesky::new(matrix).expect("test matrix should be positive definite");
        let rhs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut copied = vec![0.0; rhs.len()];
        solve_cholesky_block_blas_into(&chol, &rhs, &mut copied, 2)
            .expect("BLAS blocked solve should succeed");
        let mut in_place = rhs;
        solve_cholesky_block_blas_in_place(&chol, &mut in_place, 2)
            .expect("in-place BLAS blocked solve should succeed");
        assert!(in_place
            .iter()
            .zip(copied.iter())
            .all(|(left, right)| (left - right).abs() < 1.0e-14));
    }
}
