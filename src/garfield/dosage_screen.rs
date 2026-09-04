//! Experimental screen-only backends for the GRM dosage interaction scan.
//!
//! These scanners are deliberately separate from the exact GRM scorer.  They
//! return only compact Top-L candidates and are intended to measure whether a
//! cheap screen can preserve the exact GRM ranking before an exact refine.

use crate::blas::{
    cblas_dgemm_dispatch, BlasThreadGuard, CblasInt, CBLAS_COL_MAJOR, CBLAS_NO_TRANS, CBLAS_TRANS,
};
use crate::eigh::symmetric_eigh_f64_row_major;
use nalgebra::DMatrix;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::cmp::{Ordering, Reverse};
use std::collections::{HashMap, HashSet};

const SCREEN_SOLVE_EPS: f64 = 1.0e-12;
const SCREEN_VARIANCE_TOL: f64 = 1.0e-12;
const SCREEN_BLOCK_DEFAULT: usize = 256;

/// A canonical owner plan for one half-open genomic window.
///
/// A pair is emitted only when this window has the smallest grid window id
/// that contains both endpoints.  `pair_window_ids`/`pair_window_offsets`
/// retain every later window containing an emitted pair, which lets a caller
/// reconstruct legacy per-window Top-K results without rescoring the pair.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct CanonicalOwnerPlan {
    pub(crate) pairs: Vec<(usize, usize)>,
    pub(crate) pair_window_offsets: Vec<usize>,
    pub(crate) pair_window_ids: Vec<u64>,
    pub(crate) pairs_raw_window: usize,
    pub(crate) pairs_owner_skipped: usize,
    pub(crate) pairs_unique_screened: usize,
}

#[inline]
fn floor_div_i128(value: i128, divisor: i128) -> i128 {
    debug_assert!(divisor > 0);
    let quotient = value / divisor;
    let remainder = value % divisor;
    if remainder != 0 && value < 0 {
        quotient - 1
    } else {
        quotient
    }
}

#[inline]
fn ceil_div_i128(value: i128, divisor: i128) -> i128 {
    debug_assert!(divisor > 0);
    -floor_div_i128(-value, divisor)
}

#[inline]
fn stable_window_id(chromosome_id: u32, grid_index: i128) -> Result<u64, String> {
    if !(0..=i128::from(u32::MAX)).contains(&grid_index) {
        return Err(format!(
            "window grid index {grid_index} cannot be represented in a stable u64 window id"
        ));
    }
    Ok((u64::from(chromosome_id) << 32) | grid_index as u64)
}

/// Build the unique-owner pair plan for one sorted, half-open window.
///
/// The grid is `chromosome_origin + q * step`, with `q >= 0`; `window_start`
/// must be one of those starts.  For a pair `(left, right)`, all containing
/// grid windows satisfy
///
/// `right - window_size < start <= left`.
///
/// The smallest such `q` is the canonical owner.  This remains correct when
/// a window overlaps two, four, or any other number of grid windows.
fn canonical_owner_plan(
    positions: &[i64],
    chromosome_id: u32,
    window_start: i64,
    chromosome_origin: i64,
    window_size: i64,
    step: i64,
) -> Result<CanonicalOwnerPlan, String> {
    if window_size <= 0 {
        return Err("window_size must be > 0".to_string());
    }
    if step <= 0 {
        return Err("step must be > 0".to_string());
    }
    if window_start < chromosome_origin {
        return Err(format!(
            "window_start {window_start} is before chromosome_origin {chromosome_origin}"
        ));
    }
    let start_delta = i128::from(window_start) - i128::from(chromosome_origin);
    let step_i = i128::from(step);
    if start_delta % step_i != 0 {
        return Err(format!(
            "window_start {window_start} is not aligned to origin {chromosome_origin} and step {step}"
        ));
    }
    let current_grid = start_delta / step_i;
    let _current_id = stable_window_id(chromosome_id, current_grid)?;
    let window_end = i128::from(window_start) + i128::from(window_size);
    if positions.windows(2).any(|pair| pair[0] > pair[1]) {
        return Err("window positions must be sorted in nondecreasing order".to_string());
    }
    if positions.iter().any(|&position| {
        let position = i128::from(position);
        position < i128::from(window_start) || position >= window_end
    }) {
        return Err(format!(
            "window positions must lie in half-open interval [{window_start}, {window_end})"
        ));
    }

    let pairs_raw_window = positions
        .len()
        .saturating_mul(positions.len().saturating_sub(1))
        / 2;
    let mut pairs = Vec::new();
    let mut pair_window_offsets = Vec::with_capacity(pairs_raw_window.saturating_add(1));
    let mut pair_window_ids = Vec::new();
    pair_window_offsets.push(0);

    for left_index in 0..positions.len() {
        let left = i128::from(positions[left_index]);
        let left_grid = floor_div_i128(left - i128::from(chromosome_origin), step_i);
        for right_index in (left_index + 1)..positions.len() {
            let right = i128::from(positions[right_index]);
            // A half-open window [start, start + W) contains right iff
            // start >= right - W + 1 for integer marker coordinates.
            let first_start = right - i128::from(window_size) + 1;
            let first_grid =
                ceil_div_i128(first_start - i128::from(chromosome_origin), step_i).max(0);
            let last_grid = left_grid;
            if first_grid > last_grid || first_grid != current_grid {
                continue;
            }
            let first_grid_u = u64::try_from(first_grid)
                .map_err(|_| format!("canonical owner grid index {first_grid} is out of range"))?;
            let last_grid_u = u64::try_from(last_grid)
                .map_err(|_| format!("canonical owner grid index {last_grid} is out of range"))?;
            let _ = stable_window_id(chromosome_id, first_grid)?;
            let _ = stable_window_id(chromosome_id, last_grid)?;
            pairs.push((left_index, right_index));
            for grid in first_grid_u..=last_grid_u {
                pair_window_ids.push((u64::from(chromosome_id) << 32) | grid);
            }
            pair_window_offsets.push(pair_window_ids.len());
        }
    }

    let pairs_unique_screened = pairs.len();
    let pairs_owner_skipped = pairs_raw_window.saturating_sub(pairs_unique_screened);
    Ok(CanonicalOwnerPlan {
        pairs,
        pair_window_offsets,
        pair_window_ids,
        pairs_raw_window,
        pairs_owner_skipped,
        pairs_unique_screened,
    })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ScreenPairCandidate {
    pub(crate) first: usize,
    pub(crate) second: usize,
    pub(crate) score: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ScreenScanResult {
    pub(crate) candidates: Vec<ScreenPairCandidate>,
    pub(crate) pairs_evaluated: usize,
    pub(crate) pairs_skipped: usize,
    pub(crate) skipped_lower_geometry: usize,
    pub(crate) skipped_interaction_variance: usize,
    pub(crate) skipped_nonfinite: usize,
    pub(crate) rank: usize,
    pub(crate) baseline_weight: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct GroupedScreenScanResult {
    first: Vec<ScreenPairCandidate>,
    second: Vec<ScreenPairCandidate>,
    pairs_evaluated: usize,
    pairs_skipped: usize,
    skipped_lower_geometry: usize,
    skipped_interaction_variance: usize,
    skipped_nonfinite: usize,
    rank: usize,
    baseline_weight: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct MembershipScreenScanResult {
    window_ids: Vec<u64>,
    candidates: Vec<Vec<ScreenPairCandidate>>,
    pairs_evaluated: usize,
    pairs_skipped: usize,
    skipped_lower_geometry: usize,
    skipped_interaction_variance: usize,
    skipped_nonfinite: usize,
    rank: usize,
    baseline_weight: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScreenSkipReason {
    LowerGeometry,
    InteractionVariance,
    NonFinite,
}

enum ScreenPairScore {
    Valid(f64),
    Skipped(ScreenSkipReason),
}

#[derive(Clone, Copy, Debug)]
struct ScreenHeapEntry {
    first: usize,
    second: usize,
    score: f64,
}

impl PartialEq for ScreenHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.first == other.first
            && self.second == other.second
            && self.score.to_bits() == other.score.to_bits()
    }
}

impl Eq for ScreenHeapEntry {}

impl PartialOrd for ScreenHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScreenHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.first.cmp(&other.first))
            .then_with(|| self.second.cmp(&other.second))
    }
}

#[derive(Default)]
struct ScreenAccumulator {
    heap: std::collections::BinaryHeap<Reverse<ScreenHeapEntry>>,
    pairs_evaluated: usize,
    pairs_skipped: usize,
    skipped_lower_geometry: usize,
    skipped_interaction_variance: usize,
    skipped_nonfinite: usize,
}

impl ScreenAccumulator {
    #[inline]
    fn push(&mut self, first: usize, second: usize, score: f64, top_k: usize) {
        if top_k == 0 || !score.is_finite() {
            return;
        }
        let entry = ScreenHeapEntry {
            first,
            second,
            score,
        };
        if self.heap.len() < top_k {
            self.heap.push(Reverse(entry));
        } else if self.heap.peek().is_some_and(|smallest| entry > smallest.0) {
            self.heap.pop();
            self.heap.push(Reverse(entry));
        }
    }

    fn merge(&mut self, other: Self, top_k: usize) {
        self.pairs_evaluated = self.pairs_evaluated.saturating_add(other.pairs_evaluated);
        self.pairs_skipped = self.pairs_skipped.saturating_add(other.pairs_skipped);
        self.skipped_lower_geometry = self
            .skipped_lower_geometry
            .saturating_add(other.skipped_lower_geometry);
        self.skipped_interaction_variance = self
            .skipped_interaction_variance
            .saturating_add(other.skipped_interaction_variance);
        self.skipped_nonfinite = self
            .skipped_nonfinite
            .saturating_add(other.skipped_nonfinite);
        for Reverse(entry) in other.heap {
            self.push(entry.first, entry.second, entry.score, top_k);
        }
    }

    #[inline]
    fn skip(&mut self, reason: ScreenSkipReason) {
        self.pairs_skipped = self.pairs_skipped.saturating_add(1);
        match reason {
            ScreenSkipReason::LowerGeometry => {
                self.skipped_lower_geometry = self.skipped_lower_geometry.saturating_add(1)
            }
            ScreenSkipReason::InteractionVariance => {
                self.skipped_interaction_variance =
                    self.skipped_interaction_variance.saturating_add(1)
            }
            ScreenSkipReason::NonFinite => {
                self.skipped_nonfinite = self.skipped_nonfinite.saturating_add(1)
            }
        }
    }

    fn into_sorted(self) -> Vec<ScreenPairCandidate> {
        let mut values = self
            .heap
            .into_iter()
            .map(|Reverse(entry)| ScreenPairCandidate {
                first: entry.first,
                second: entry.second,
                score: entry.score,
            })
            .collect::<Vec<_>>();
        values.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.first.cmp(&right.first))
                .then_with(|| left.second.cmp(&right.second))
        });
        values
    }
}

#[derive(Clone, Copy, Debug)]
struct GlobalTopKEntry {
    key: u64,
    score: f64,
}

impl PartialEq for GlobalTopKEntry {
    fn eq(&self, other: &Self) -> bool {
        self.key == other.key && self.score.to_bits() == other.score.to_bits()
    }
}

impl Eq for GlobalTopKEntry {}

impl PartialOrd for GlobalTopKEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GlobalTopKEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .total_cmp(&other.score)
            .then_with(|| self.key.cmp(&other.key))
    }
}

/// Streaming global Top-K merge with a duplicate-safe key map.
///
/// The owner path should never send a key twice.  We still retain the map as
/// a release-build safety net: duplicate occurrences are counted and a later
/// higher score replaces the earlier value.  Callers can assert the returned
/// duplicate counter is zero when canonical ownership is expected.
#[derive(Clone, Debug)]
struct CanonicalGlobalTopK {
    top_k: usize,
    // This is an indexed min-heap rather than a `BinaryHeap<Reverse<_>>` so
    // that a later higher score for an already-retained key updates the one
    // live heap entry in place.  Lazy duplicate entries would otherwise make
    // a bounded Top-K lose slots when a key is updated after it ceased to be
    // the heap minimum.
    heap: Vec<GlobalTopKEntry>,
    heap_positions: HashMap<u64, usize>,
    seen_keys: HashSet<u64>,
    candidates_before_global_topk: usize,
    global_duplicate_pairs: usize,
}

impl CanonicalGlobalTopK {
    fn new(top_k: usize) -> Self {
        Self {
            top_k,
            heap: Vec::new(),
            heap_positions: HashMap::new(),
            seen_keys: HashSet::new(),
            candidates_before_global_topk: 0,
            global_duplicate_pairs: 0,
        }
    }

    #[inline]
    fn swap_heap_entries(&mut self, left: usize, right: usize) {
        self.heap.swap(left, right);
        self.heap_positions.insert(self.heap[left].key, left);
        self.heap_positions.insert(self.heap[right].key, right);
    }

    #[inline]
    fn sift_up(&mut self, mut index: usize) {
        while index > 0 {
            let parent = (index - 1) / 2;
            if self.heap[parent] <= self.heap[index] {
                break;
            }
            self.swap_heap_entries(parent, index);
            index = parent;
        }
    }

    #[inline]
    fn sift_down(&mut self, mut index: usize) {
        loop {
            let left = index.saturating_mul(2).saturating_add(1);
            if left >= self.heap.len() {
                break;
            }
            let right = left + 1;
            let mut smallest = left;
            if right < self.heap.len() && self.heap[right] < self.heap[left] {
                smallest = right;
            }
            if self.heap[index] <= self.heap[smallest] {
                break;
            }
            self.swap_heap_entries(index, smallest);
            index = smallest;
        }
    }

    #[inline]
    fn push(&mut self, key: u64, score: f64) {
        if !score.is_finite() {
            return;
        }
        self.candidates_before_global_topk = self.candidates_before_global_topk.saturating_add(1);
        if !self.seen_keys.insert(key) {
            self.global_duplicate_pairs = self.global_duplicate_pairs.saturating_add(1);
        }
        if self.top_k == 0 {
            return;
        }

        if let Some(&index) = self.heap_positions.get(&key) {
            if score <= self.heap[index].score {
                return;
            }
            self.heap[index].score = score;
            // Increasing a key's score can only move it down in a min-heap.
            self.sift_down(index);
            return;
        }

        let entry = GlobalTopKEntry { key, score };
        if self.heap.len() < self.top_k {
            let index = self.heap.len();
            self.heap.push(entry);
            self.heap_positions.insert(key, index);
            self.sift_up(index);
        } else if self.heap.first().is_some_and(|smallest| entry > *smallest) {
            let evicted = self.heap[0].key;
            self.heap[0] = entry;
            self.heap_positions.remove(&evicted);
            self.heap_positions.insert(key, 0);
            self.sift_down(0);
        }
    }

    fn into_sorted(self) -> Vec<(u64, f64)> {
        let mut values = self
            .heap
            .into_iter()
            .map(|entry| (entry.key, entry.score))
            .collect::<Vec<_>>();
        values.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        values
    }
}

#[derive(Clone, Debug)]
struct ScreenMetric {
    n: usize,
    baseline_weight: f64,
    eigenvectors: Vec<f64>, // n x rank, column-major
    correction_weights: Vec<f64>,
}

/// The eigendecomposition shared by all explicit and adaptive low-rank
/// metrics.  Building this once is important for `--screen-rank auto`: a
/// rank audit should compare several approximations of the same covariance,
/// not repeatedly decompose the GRM.
#[derive(Clone, Debug)]
struct ScreenSpectrum {
    n: usize,
    inverse_eigenvalues: Vec<f64>,
    eigenvectors_row_major: Vec<f64>,
    diagnostic_baseline: f64,
}

const ADAPTIVE_SCREEN_RANKS: &[usize] = &[32, 64, 128, 256, 512];

#[derive(Clone, Debug)]
struct AdaptiveRankCandidate {
    rank: usize,
    baseline_weight: f64,
    diagnostic_correction_energy_c0: f64,
    diagnostic_spectral_tail_c0: f64,
    geometry_valid_pairs: usize,
    geometry_mean_relative_error: f64,
    geometry_p95_relative_error: f64,
    geometry_q99_relative_error: f64,
    geometry_q999_relative_error: f64,
    geometry_max_relative_error: f64,
    geometry_pass: bool,
    score_pairs_compared: usize,
    score_spearman: f64,
    score_top_k_overlap: f64,
}

#[derive(Clone, Debug)]
struct AdaptiveRankReport {
    selected_rank: usize,
    full_rank: usize,
    pilot_pair_count: usize,
    full_geometry_valid_pairs: usize,
    sigma_g2: f64,
    sigma_e2: f64,
    diagnostic_baseline: f64,
    geometry_rtol: f64,
    geometry_max_rtol: f64,
    score_audit_top_k: usize,
    used_largest_rank_fallback: bool,
    candidates: Vec<AdaptiveRankCandidate>,
}

impl ScreenMetric {
    #[inline]
    fn rank(&self) -> usize {
        self.correction_weights.len()
    }

    #[inline]
    fn weighted(&self, raw: f64, left_q: &[f64], right_q: &[f64]) -> f64 {
        debug_assert_eq!(left_q.len(), self.rank());
        debug_assert_eq!(right_q.len(), self.rank());
        let mut value = self.baseline_weight * raw;
        for index in 0..self.rank() {
            value += self.correction_weights[index] * left_q[index] * right_q[index];
        }
        value
    }
}

struct ScreenFixedContext {
    n: usize,
    fixed_columns: Vec<Vec<f64>>, // fixed_rank vectors, including intercept
    fixed_rank: usize,
    metric: ScreenMetric,
    q_fixed: Vec<f64>, // fixed_rank x rank, row-major
    q_y: Vec<f64>,
    fixed_gram: Vec<f64>,
    fixed_y_cross: Vec<f64>,
}

struct ScreenMarkerStats {
    q_markers: Vec<f64>, // marker-major, m x rank
    raw_fixed_cross: Vec<f64>,
    raw_y_cross: Vec<f64>,
    raw_sq_sum: Vec<f64>,
}

struct ScreenWorkspace {
    lower_gram: Vec<f64>,
    rhs_interaction: Vec<f64>,
    solved_interaction: Vec<f64>,
    rhs_y: Vec<f64>,
    solved_y: Vec<f64>,
    raw_fixed_interaction: Vec<f64>,
}

#[derive(Default)]
struct GroupedScreenAccumulator {
    first: ScreenAccumulator,
    second: ScreenAccumulator,
    counts: ScreenAccumulator,
}

impl GroupedScreenAccumulator {
    fn merge(&mut self, other: Self, top_k: usize) {
        self.counts.merge(other.counts, 0);
        self.first.merge(other.first, top_k);
        self.second.merge(other.second, top_k);
    }
}

impl ScreenWorkspace {
    fn new(fixed_rank: usize) -> Self {
        let dimension = fixed_rank + 2;
        Self {
            lower_gram: vec![0.0; dimension * dimension],
            rhs_interaction: vec![0.0; dimension],
            solved_interaction: vec![0.0; dimension],
            rhs_y: vec![0.0; dimension],
            solved_y: vec![0.0; dimension],
            raw_fixed_interaction: vec![0.0; fixed_rank],
        }
    }
}

fn parse_screen_block_size() -> usize {
    std::env::var("JX_GRM_INTERACTION_BLOCK_SIZE")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(SCREEN_BLOCK_DEFAULT)
}

fn parse_screen_blas_threads() -> usize {
    std::env::var("JX_GRM_BLAS_THREADS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1)
}

fn validate_dense_inputs(
    genotypes: &[f64],
    n_markers: usize,
    n_samples: usize,
    y: &[f64],
) -> Result<(), String> {
    if n_markers == 0 || n_samples == 0 {
        return Err("n_markers and n_samples must be > 0".to_string());
    }
    let expected = n_markers
        .checked_mul(n_samples)
        .ok_or_else(|| "genotype matrix size overflow".to_string())?;
    if genotypes.len() != expected {
        return Err(format!(
            "genotypes length={} but expected {expected}",
            genotypes.len()
        ));
    }
    if y.len() != n_samples {
        return Err(format!("y length={} but expected {n_samples}", y.len()));
    }
    if y.iter().any(|value| !value.is_finite()) {
        return Err("y contains non-finite values".to_string());
    }
    if genotypes.iter().any(|value| !value.is_finite()) {
        return Err("genotypes contain non-finite values".to_string());
    }
    Ok(())
}

fn build_fixed_columns(
    n_samples: usize,
    covariates: &[f64],
    n_covariates: usize,
) -> Result<Vec<Vec<f64>>, String> {
    let expected = n_samples
        .checked_mul(n_covariates)
        .ok_or_else(|| "covariate matrix size overflow".to_string())?;
    if covariates.len() != expected {
        return Err(format!(
            "covariates length={} but expected {expected}",
            covariates.len()
        ));
    }
    if covariates.iter().any(|value| !value.is_finite()) {
        return Err("covariates contain non-finite values".to_string());
    }
    let mut columns = Vec::with_capacity(n_covariates + 1);
    columns.push(vec![1.0; n_samples]);
    for column in 0..n_covariates {
        columns.push(
            (0..n_samples)
                .map(|row| covariates[row * n_covariates + column])
                .collect(),
        );
    }
    let rank = columns.len();
    let design = DMatrix::from_fn(n_samples, rank, |row, column| columns[column][row]);
    check_full_rank(&design, "fixed-effect design")?;
    Ok(columns)
}

fn check_full_rank(matrix: &DMatrix<f64>, name: &str) -> Result<(), String> {
    let qr = matrix.clone().col_piv_qr();
    let r = qr.r();
    let scale = (0..matrix.ncols())
        .map(|column| r[(column, column)].abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let tolerance = SCREEN_SOLVE_EPS * (matrix.nrows().max(matrix.ncols()) as f64) * scale;
    let rank = (0..matrix.ncols())
        .filter(|&column| r[(column, column)].abs() > tolerance)
        .count();
    if rank != matrix.ncols() {
        Err(format!(
            "{name} is rank-deficient: rank={rank}, columns={}",
            matrix.ncols()
        ))
    } else {
        Ok(())
    }
}

impl ScreenSpectrum {
    fn from_grm(grm: &[f64], n: usize, sigma_g2: f64, sigma_e2: f64) -> Result<Self, String> {
        if n == 0 {
            return Err("GRM dimension must be > 0".to_string());
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
        let mut covariance = vec![0.0; n * n];
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
                covariance[row * n + column] =
                    sigma_g2 * left + if row == column { sigma_e2 } else { 0.0 };
            }
        }
        let (eigenvalues, eigenvectors_row_major, _) =
            symmetric_eigh_f64_row_major(&covariance, n)?;
        let inverse_eigenvalues = eigenvalues
            .iter()
            .enumerate()
            .map(|(index, value)| {
                if !value.is_finite() || *value <= SCREEN_VARIANCE_TOL {
                    Err(format!(
                        "GRM covariance eigenvalue {index} is not positive: {value}"
                    ))
                } else {
                    Ok(1.0 / value)
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut sorted_inverse = inverse_eigenvalues.clone();
        sorted_inverse.sort_by(f64::total_cmp);
        let diagnostic_baseline = sorted_inverse[sorted_inverse.len() / 2];
        Ok(Self {
            n,
            inverse_eigenvalues,
            eigenvectors_row_major,
            diagnostic_baseline,
        })
    }

    /// Construct the same rank-specific omitted-space baseline used by the
    /// existing explicit low-rank backend.  The adaptive selector only
    /// chooses `requested_rank`; it never replaces this baseline with the
    /// diagnostic fixed `c0`.
    fn metric(&self, requested_rank: usize) -> Result<ScreenMetric, String> {
        let take = requested_rank.min(self.n);
        let mut selected = (0..self.n).collect::<Vec<_>>();
        selected.sort_by(|left, right| {
            (self.inverse_eigenvalues[*right] - self.diagnostic_baseline)
                .abs()
                .total_cmp(&(self.inverse_eigenvalues[*left] - self.diagnostic_baseline).abs())
                .then_with(|| left.cmp(right))
        });
        selected.truncate(take);
        let mut selected_mask = vec![false; self.n];
        for &index in &selected {
            selected_mask[index] = true;
        }
        let mut omitted = (0..self.n)
            .filter(|&index| !selected_mask[index])
            .collect::<Vec<_>>();
        let mut baseline = if omitted.is_empty() {
            self.diagnostic_baseline
        } else {
            omitted
                .iter()
                .map(|&index| self.inverse_eigenvalues[index])
                .sum::<f64>()
                / omitted.len() as f64
        };

        // Re-rank once using the actual omitted-space baseline.  This keeps
        // explicit-rank behavior unchanged while making `c_r` rank-specific.
        selected = (0..self.n).collect::<Vec<_>>();
        selected.sort_by(|left, right| {
            (self.inverse_eigenvalues[*right] - baseline)
                .abs()
                .total_cmp(&(self.inverse_eigenvalues[*left] - baseline).abs())
                .then_with(|| left.cmp(right))
        });
        selected.truncate(take);
        selected_mask.fill(false);
        for &index in &selected {
            selected_mask[index] = true;
        }
        omitted = (0..self.n)
            .filter(|&index| !selected_mask[index])
            .collect::<Vec<_>>();
        baseline = if omitted.is_empty() {
            self.diagnostic_baseline
        } else {
            omitted
                .iter()
                .map(|&index| self.inverse_eigenvalues[index])
                .sum::<f64>()
                / omitted.len() as f64
        };
        if !baseline.is_finite() || baseline <= 0.0 {
            return Err(format!(
                "invalid scalar baseline inverse weight: {baseline}"
            ));
        }

        let rank = selected.len();
        let mut eigenvectors = vec![0.0; self.n * rank];
        let mut correction_weights = vec![0.0; rank];
        for (out, index) in selected.iter().copied().enumerate() {
            correction_weights[out] = self.inverse_eigenvalues[index] - baseline;
            for row in 0..self.n {
                eigenvectors[row + out * self.n] =
                    self.eigenvectors_row_major[row * self.n + index];
            }
        }
        Ok(ScreenMetric {
            n: self.n,
            baseline_weight: baseline,
            eigenvectors,
            correction_weights,
        })
    }

    /// A diagnostic only: cumulative correction energy relative to a fixed
    /// scalar `c0`.  The actual metric uses `metric()` and its rank-specific
    /// omitted-space baseline instead.
    fn diagnostic_correction_energy(&self, requested_rank: usize) -> f64 {
        let total = self
            .inverse_eigenvalues
            .iter()
            .map(|value| (value - self.diagnostic_baseline).powi(2))
            .sum::<f64>();
        if total <= SCREEN_VARIANCE_TOL {
            return 1.0;
        }
        let mut indices = (0..self.n).collect::<Vec<_>>();
        indices.sort_by(|left, right| {
            (self.inverse_eigenvalues[*right] - self.diagnostic_baseline)
                .abs()
                .total_cmp(&(self.inverse_eigenvalues[*left] - self.diagnostic_baseline).abs())
                .then_with(|| left.cmp(right))
        });
        let selected = indices.into_iter().take(requested_rank.min(self.n));
        selected
            .map(|index| (self.inverse_eigenvalues[index] - self.diagnostic_baseline).powi(2))
            .sum::<f64>()
            / total
    }
}

fn build_lowrank_metric(
    grm: &[f64],
    n: usize,
    sigma_g2: f64,
    sigma_e2: f64,
    requested_rank: usize,
) -> Result<ScreenMetric, String> {
    let spectrum = ScreenSpectrum::from_grm(grm, n, sigma_g2, sigma_e2)?;
    spectrum.metric(requested_rank)
}

impl ScreenFixedContext {
    fn new(y: &[f64], fixed_columns: Vec<Vec<f64>>, metric: ScreenMetric) -> Result<Self, String> {
        let n = y.len();
        if metric.n != n {
            return Err("screen metric dimension does not match phenotype".to_string());
        }
        if fixed_columns.iter().any(|column| column.len() != n) {
            return Err("fixed-effect column dimension mismatch".to_string());
        }
        let fixed_rank = fixed_columns.len();
        if n <= fixed_rank + 3 {
            return Err(format!(
                "need more than fixed rank + 3 samples; got n={n}, fixed columns={fixed_rank}"
            ));
        }
        let rank = metric.rank();
        let mut q_y = vec![0.0; rank];
        let mut q_fixed = vec![0.0; fixed_rank * rank];
        for component in 0..rank {
            let eigenvector = &metric.eigenvectors[component * n..(component + 1) * n];
            q_y[component] = dot(eigenvector, y);
            for fixed in 0..fixed_rank {
                q_fixed[fixed * rank + component] = dot(eigenvector, &fixed_columns[fixed]);
            }
        }
        let mut fixed_gram = vec![0.0; fixed_rank * fixed_rank];
        let mut fixed_y_cross = vec![0.0; fixed_rank];
        for row in 0..fixed_rank {
            for column in 0..fixed_rank {
                fixed_gram[row * fixed_rank + column] = metric.weighted(
                    dot(&fixed_columns[row], &fixed_columns[column]),
                    &q_fixed[row * rank..(row + 1) * rank],
                    &q_fixed[column * rank..(column + 1) * rank],
                );
            }
            fixed_y_cross[row] = metric.weighted(
                dot(&fixed_columns[row], y),
                &q_fixed[row * rank..(row + 1) * rank],
                &q_y,
            );
        }
        Ok(Self {
            n,
            fixed_columns,
            fixed_rank,
            metric,
            q_fixed,
            q_y,
            fixed_gram,
            fixed_y_cross,
        })
    }

    fn prepare_markers(
        &self,
        genotypes: &[f64],
        n_markers: usize,
        y: &[f64],
    ) -> Result<ScreenMarkerStats, String> {
        let expected = n_markers
            .checked_mul(self.n)
            .ok_or_else(|| "genotype matrix size overflow".to_string())?;
        if genotypes.len() != expected {
            return Err("screen marker matrix dimension mismatch".to_string());
        }
        let rank = self.metric.rank();
        let mut q_markers = vec![0.0; n_markers * rank];
        if rank > 0 {
            let n_blas = CblasInt::try_from(self.n)
                .map_err(|_| "screen sample count exceeds CBLAS range".to_string())?;
            let markers_blas = CblasInt::try_from(n_markers)
                .map_err(|_| "screen marker count exceeds CBLAS range".to_string())?;
            let rank_blas = CblasInt::try_from(rank)
                .map_err(|_| "screen rank exceeds CBLAS range".to_string())?;
            let _blas_guard = BlasThreadGuard::enter(parse_screen_blas_threads());
            unsafe {
                cblas_dgemm_dispatch(
                    CBLAS_COL_MAJOR,
                    CBLAS_TRANS,
                    CBLAS_NO_TRANS,
                    rank_blas,
                    markers_blas,
                    n_blas,
                    1.0,
                    self.metric.eigenvectors.as_ptr(),
                    n_blas,
                    genotypes.as_ptr(),
                    n_blas,
                    0.0,
                    q_markers.as_mut_ptr(),
                    rank_blas,
                );
            }
        }
        let mut raw_fixed_cross = vec![0.0; n_markers * self.fixed_rank];
        let mut raw_y_cross = vec![0.0; n_markers];
        let mut raw_sq_sum = vec![0.0; n_markers];
        for marker in 0..n_markers {
            let values = &genotypes[marker * self.n..(marker + 1) * self.n];
            raw_sq_sum[marker] = values.iter().map(|value| value * value).sum();
            raw_y_cross[marker] = dot(values, y);
            for fixed in 0..self.fixed_rank {
                raw_fixed_cross[marker * self.fixed_rank + fixed] =
                    dot(values, &self.fixed_columns[fixed]);
            }
        }
        Ok(ScreenMarkerStats {
            q_markers,
            raw_fixed_cross,
            raw_y_cross,
            raw_sq_sum,
        })
    }

    #[inline]
    fn weighted_marker(&self, raw: f64, marker_q: &[f64], fixed_q: &[f64]) -> f64 {
        self.metric.weighted(raw, marker_q, fixed_q)
    }

    fn score_pair(
        &self,
        first: usize,
        second: usize,
        first_values: &[f64],
        second_values: &[f64],
        interaction_values: &[f64],
        y: &[f64],
        marker_stats: &ScreenMarkerStats,
        q_interaction: &[f64],
        workspace: &mut ScreenWorkspace,
    ) -> ScreenPairScore {
        let lower_dim = self.fixed_rank + 2;
        let rank = self.metric.rank();
        let first_q = &marker_stats.q_markers[first * rank..(first + 1) * rank];
        let second_q = &marker_stats.q_markers[second * rank..(second + 1) * rank];
        let first_fixed_raw =
            &marker_stats.raw_fixed_cross[first * self.fixed_rank..(first + 1) * self.fixed_rank];
        let second_fixed_raw =
            &marker_stats.raw_fixed_cross[second * self.fixed_rank..(second + 1) * self.fixed_rank];
        let raw_xx = marker_stats.raw_sq_sum[first];
        let raw_zz = marker_stats.raw_sq_sum[second];
        let raw_xy = marker_stats.raw_y_cross[first];
        let raw_zy = marker_stats.raw_y_cross[second];
        let mut raw_xz = 0.0;
        let mut raw_xw = 0.0;
        let mut raw_zw = 0.0;
        let mut raw_ww = 0.0;
        let mut raw_yw = 0.0;
        workspace.raw_fixed_interaction.fill(0.0);
        debug_assert_eq!(interaction_values.len(), self.n);
        for row in 0..self.n {
            let x = first_values[row];
            let z = second_values[row];
            let w = interaction_values[row];
            raw_xz += w;
            raw_xw += x * w;
            raw_zw += z * w;
            raw_ww += w * w;
            raw_yw += y[row] * w;
            for fixed in 0..self.fixed_rank {
                workspace.raw_fixed_interaction[fixed] += self.fixed_columns[fixed][row] * w;
            }
        }
        workspace.lower_gram[..lower_dim * lower_dim].fill(0.0);
        workspace.rhs_interaction[..lower_dim].fill(0.0);
        workspace.rhs_y[..lower_dim].fill(0.0);
        for row in 0..self.fixed_rank {
            let row_q = &self.q_fixed[row * rank..(row + 1) * rank];
            for column in 0..self.fixed_rank {
                workspace.lower_gram[row * lower_dim + column] =
                    self.fixed_gram[row * self.fixed_rank + column];
            }
            workspace.lower_gram[row * lower_dim + self.fixed_rank] =
                self.weighted_marker(first_fixed_raw[row], first_q, row_q);
            workspace.lower_gram[row * lower_dim + self.fixed_rank + 1] =
                self.weighted_marker(second_fixed_raw[row], second_q, row_q);
            workspace.rhs_interaction[row] =
                self.metric
                    .weighted(workspace.raw_fixed_interaction[row], row_q, q_interaction);
            workspace.rhs_y[row] = self.fixed_y_cross[row];
        }
        for row in 0..self.fixed_rank {
            workspace.lower_gram[self.fixed_rank * lower_dim + row] =
                workspace.lower_gram[row * lower_dim + self.fixed_rank];
            workspace.lower_gram[(self.fixed_rank + 1) * lower_dim + row] =
                workspace.lower_gram[row * lower_dim + self.fixed_rank + 1];
        }
        let first_self = self.weighted_marker(raw_xx, first_q, first_q);
        let cross = self.weighted_marker(raw_xz, first_q, second_q);
        let second_self = self.weighted_marker(raw_zz, second_q, second_q);
        workspace.lower_gram[self.fixed_rank * lower_dim + self.fixed_rank] = first_self;
        workspace.lower_gram[self.fixed_rank * lower_dim + self.fixed_rank + 1] = cross;
        workspace.lower_gram[(self.fixed_rank + 1) * lower_dim + self.fixed_rank] = cross;
        workspace.lower_gram[(self.fixed_rank + 1) * lower_dim + self.fixed_rank + 1] = second_self;
        workspace.rhs_interaction[self.fixed_rank] =
            self.metric.weighted(raw_xw, first_q, q_interaction);
        workspace.rhs_interaction[self.fixed_rank + 1] =
            self.metric.weighted(raw_zw, second_q, q_interaction);
        workspace.rhs_y[self.fixed_rank] = self.metric.weighted(raw_xy, first_q, &self.q_y);
        workspace.rhs_y[self.fixed_rank + 1] = self.metric.weighted(raw_zy, second_q, &self.q_y);
        let interaction_raw = self.metric.weighted(raw_ww, q_interaction, q_interaction);
        let interaction_y = self.metric.weighted(raw_yw, q_interaction, &self.q_y);
        if !cholesky_lower_in_place(
            &mut workspace.lower_gram[..lower_dim * lower_dim],
            lower_dim,
        ) {
            return ScreenPairScore::Skipped(ScreenSkipReason::LowerGeometry);
        }
        if !solve_cholesky(
            &workspace.lower_gram[..lower_dim * lower_dim],
            lower_dim,
            &workspace.rhs_interaction[..lower_dim],
            &mut workspace.solved_interaction[..lower_dim],
        ) || !solve_cholesky(
            &workspace.lower_gram[..lower_dim * lower_dim],
            lower_dim,
            &workspace.rhs_y[..lower_dim],
            &mut workspace.solved_y[..lower_dim],
        ) {
            return ScreenPairScore::Skipped(ScreenSkipReason::NonFinite);
        }
        let projected_variance = interaction_raw
            - dot(
                &workspace.rhs_interaction[..lower_dim],
                &workspace.solved_interaction[..lower_dim],
            );
        if !projected_variance.is_finite()
            || projected_variance <= SCREEN_VARIANCE_TOL * interaction_raw.abs().max(1.0)
        {
            return ScreenPairScore::Skipped(ScreenSkipReason::InteractionVariance);
        }
        let projected_covariance = interaction_y
            - dot(
                &workspace.rhs_interaction[..lower_dim],
                &workspace.solved_y[..lower_dim],
            );
        let score = projected_covariance * projected_covariance / projected_variance;
        if score.is_finite() {
            ScreenPairScore::Valid(score.max(0.0))
        } else {
            ScreenPairScore::Skipped(ScreenSkipReason::NonFinite)
        }
    }
}

/// Return the residualized interaction information
/// `w' P_[fixed, x, z] w` for one marker pair.  This deliberately mirrors the
/// lower-geometry part of `score_pair` but does not touch the phenotype.  It
/// is used by the adaptive rank selector as the primary, outcome-independent
/// criterion for deciding whether a low-rank metric is safe for a population.
fn interaction_geometry_with_context(
    context: &ScreenFixedContext,
    first: usize,
    second: usize,
    first_values: &[f64],
    second_values: &[f64],
    marker_stats: &ScreenMarkerStats,
    workspace: &mut ScreenWorkspace,
) -> Result<f64, ScreenSkipReason> {
    let lower_dim = context.fixed_rank + 2;
    let rank = context.metric.rank();
    let first_q = &marker_stats.q_markers[first * rank..(first + 1) * rank];
    let second_q = &marker_stats.q_markers[second * rank..(second + 1) * rank];
    let first_fixed_raw =
        &marker_stats.raw_fixed_cross[first * context.fixed_rank..(first + 1) * context.fixed_rank];
    let second_fixed_raw = &marker_stats.raw_fixed_cross
        [second * context.fixed_rank..(second + 1) * context.fixed_rank];
    let mut raw_xz = 0.0;
    let mut raw_xw = 0.0;
    let mut raw_zw = 0.0;
    let mut raw_ww = 0.0;
    let mut q_interaction = vec![0.0; rank];
    workspace.raw_fixed_interaction.fill(0.0);
    for row in 0..context.n {
        let x = first_values[row];
        let z = second_values[row];
        let w = x * z;
        raw_xz += w;
        raw_xw += x * w;
        raw_zw += z * w;
        raw_ww += w * w;
        for fixed in 0..context.fixed_rank {
            workspace.raw_fixed_interaction[fixed] += context.fixed_columns[fixed][row] * w;
        }
    }
    for component in 0..rank {
        let eigenvector =
            &context.metric.eigenvectors[component * context.n..(component + 1) * context.n];
        q_interaction[component] = dot_product(eigenvector, first_values, second_values);
    }
    workspace.lower_gram[..lower_dim * lower_dim].fill(0.0);
    workspace.rhs_interaction[..lower_dim].fill(0.0);
    for row in 0..context.fixed_rank {
        let row_q = &context.q_fixed[row * rank..(row + 1) * rank];
        for column in 0..context.fixed_rank {
            workspace.lower_gram[row * lower_dim + column] =
                context.fixed_gram[row * context.fixed_rank + column];
        }
        workspace.lower_gram[row * lower_dim + context.fixed_rank] =
            context
                .metric
                .weighted(first_fixed_raw[row], first_q, row_q);
        workspace.lower_gram[row * lower_dim + context.fixed_rank + 1] =
            context
                .metric
                .weighted(second_fixed_raw[row], second_q, row_q);
        workspace.rhs_interaction[row] =
            context
                .metric
                .weighted(workspace.raw_fixed_interaction[row], row_q, &q_interaction);
    }
    for row in 0..context.fixed_rank {
        workspace.lower_gram[context.fixed_rank * lower_dim + row] =
            workspace.lower_gram[row * lower_dim + context.fixed_rank];
        workspace.lower_gram[(context.fixed_rank + 1) * lower_dim + row] =
            workspace.lower_gram[row * lower_dim + context.fixed_rank + 1];
    }
    let first_self = context
        .metric
        .weighted(marker_stats.raw_sq_sum[first], first_q, first_q);
    let cross = context.metric.weighted(raw_xz, first_q, second_q);
    let second_self = context
        .metric
        .weighted(marker_stats.raw_sq_sum[second], second_q, second_q);
    workspace.lower_gram[context.fixed_rank * lower_dim + context.fixed_rank] = first_self;
    workspace.lower_gram[context.fixed_rank * lower_dim + context.fixed_rank + 1] = cross;
    workspace.lower_gram[(context.fixed_rank + 1) * lower_dim + context.fixed_rank] = cross;
    workspace.lower_gram[(context.fixed_rank + 1) * lower_dim + context.fixed_rank + 1] =
        second_self;
    workspace.rhs_interaction[context.fixed_rank] =
        context.metric.weighted(raw_xw, first_q, &q_interaction);
    workspace.rhs_interaction[context.fixed_rank + 1] =
        context.metric.weighted(raw_zw, second_q, &q_interaction);
    let interaction_raw = context
        .metric
        .weighted(raw_ww, &q_interaction, &q_interaction);
    if !cholesky_lower_in_place(
        &mut workspace.lower_gram[..lower_dim * lower_dim],
        lower_dim,
    ) {
        return Err(ScreenSkipReason::LowerGeometry);
    }
    if !solve_cholesky(
        &workspace.lower_gram[..lower_dim * lower_dim],
        lower_dim,
        &workspace.rhs_interaction[..lower_dim],
        &mut workspace.solved_interaction[..lower_dim],
    ) {
        return Err(ScreenSkipReason::NonFinite);
    }
    let projected_variance = interaction_raw
        - dot(
            &workspace.rhs_interaction[..lower_dim],
            &workspace.solved_interaction[..lower_dim],
        );
    if !projected_variance.is_finite()
        || projected_variance <= SCREEN_VARIANCE_TOL * interaction_raw.abs().max(1.0)
    {
        return Err(ScreenSkipReason::InteractionVariance);
    }
    Ok(projected_variance)
}

/// Compute `sum eigenvector[row] * x[row] * z[row]` without materializing the
/// interaction vector.  The helper keeps the geometry calculation in the
/// same eigenvector convention as the production screen scorer.
#[inline]
fn dot_product(eigenvector: &[f64], first: &[f64], second: &[f64]) -> f64 {
    debug_assert_eq!(eigenvector.len(), first.len());
    debug_assert_eq!(first.len(), second.len());
    eigenvector
        .iter()
        .zip(first.iter().zip(second))
        .map(|(eigen, (left, right))| eigen * left * right)
        .sum()
}

#[inline]
fn dot(left: &[f64], right: &[f64]) -> f64 {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

#[inline]
fn cholesky_lower_in_place(matrix: &mut [f64], dimension: usize) -> bool {
    for row in 0..dimension {
        for column in 0..=row {
            let mut value = matrix[row * dimension + column];
            for index in 0..column {
                value -= matrix[row * dimension + index] * matrix[column * dimension + index];
            }
            if row == column {
                if !value.is_finite() || value <= SCREEN_VARIANCE_TOL {
                    return false;
                }
                matrix[row * dimension + column] = value.sqrt();
            } else {
                let diagonal = matrix[column * dimension + column];
                if !diagonal.is_finite() || diagonal <= 0.0 {
                    return false;
                }
                matrix[row * dimension + column] = value / diagonal;
            }
        }
        for column in (row + 1)..dimension {
            matrix[row * dimension + column] = 0.0;
        }
    }
    true
}

#[inline]
fn solve_cholesky(lower: &[f64], dimension: usize, rhs: &[f64], out: &mut [f64]) -> bool {
    out[..dimension].copy_from_slice(&rhs[..dimension]);
    for row in 0..dimension {
        let mut value = out[row];
        for column in 0..row {
            value -= lower[row * dimension + column] * out[column];
        }
        let diagonal = lower[row * dimension + row];
        if !diagonal.is_finite() || diagonal <= 0.0 {
            return false;
        }
        out[row] = value / diagonal;
    }
    for row in (0..dimension).rev() {
        let mut value = out[row];
        for column in (row + 1)..dimension {
            value -= lower[column * dimension + row] * out[column];
        }
        let diagonal = lower[row * dimension + row];
        out[row] = value / diagonal;
    }
    out[..dimension].iter().all(|value| value.is_finite())
}

fn quantile(values: &[f64], probability: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    debug_assert!((0.0..=1.0).contains(&probability));
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() as f64 * probability).ceil() as usize).saturating_sub(1);
    sorted[index.min(sorted.len() - 1)]
}

fn ordinal_ranks(values: &[f64]) -> Vec<f64> {
    let mut order = (0..values.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        values[*left]
            .total_cmp(&values[*right])
            .then_with(|| left.cmp(right))
    });
    let mut ranks = vec![0.0; values.len()];
    for (rank, index) in order.into_iter().enumerate() {
        ranks[index] = rank as f64;
    }
    ranks
}

fn pearson_correlation(left: &[f64], right: &[f64]) -> f64 {
    if left.len() != right.len() || left.len() < 2 {
        return f64::NAN;
    }
    let left_mean = left.iter().sum::<f64>() / left.len() as f64;
    let right_mean = right.iter().sum::<f64>() / right.len() as f64;
    let mut numerator = 0.0;
    let mut left_ss = 0.0;
    let mut right_ss = 0.0;
    for (&left_value, &right_value) in left.iter().zip(right) {
        let left_delta = left_value - left_mean;
        let right_delta = right_value - right_mean;
        numerator += left_delta * right_delta;
        left_ss += left_delta * left_delta;
        right_ss += right_delta * right_delta;
    }
    if left_ss <= SCREEN_VARIANCE_TOL || right_ss <= SCREEN_VARIANCE_TOL {
        if left
            .iter()
            .zip(right)
            .all(|(a, b)| (a - b).abs() <= 1.0e-12)
        {
            1.0
        } else {
            f64::NAN
        }
    } else {
        numerator / (left_ss * right_ss).sqrt()
    }
}

fn score_pairs_for_audit(
    context: &ScreenFixedContext,
    genotypes: &[f64],
    n_markers: usize,
    y: &[f64],
    pairs: &[(usize, usize)],
) -> Result<HashMap<(usize, usize), f64>, String> {
    if pairs.is_empty() {
        return Ok(HashMap::new());
    }
    let result =
        scan_selected_with_context(context, genotypes, n_markers, y, pairs, pairs.len(), 1)?;
    Ok(result
        .candidates
        .into_iter()
        .map(|candidate| ((candidate.first, candidate.second), candidate.score))
        .collect())
}

fn geometry_for_pairs(
    context: &ScreenFixedContext,
    genotypes: &[f64],
    n_markers: usize,
    y: &[f64],
    pairs: &[(usize, usize)],
) -> Result<HashMap<(usize, usize), f64>, String> {
    let marker_stats = context.prepare_markers(genotypes, n_markers, y)?;
    let mut workspace = ScreenWorkspace::new(context.fixed_rank);
    let mut output = HashMap::with_capacity(pairs.len());
    for &(first, second) in pairs {
        let first_values = &genotypes[first * context.n..(first + 1) * context.n];
        let second_values = &genotypes[second * context.n..(second + 1) * context.n];
        if let Ok(geometry) = interaction_geometry_with_context(
            context,
            first,
            second,
            first_values,
            second_values,
            &marker_stats,
            &mut workspace,
        ) {
            output.insert((first, second), geometry);
        }
    }
    Ok(output)
}

/// Select the smallest standard rank whose residualized interaction geometry
/// is stable on a deterministic pilot set.  Score/rank agreement with the
/// full-rank metric is recorded as an audit, but is intentionally not used as
/// the selector: it depends on the observed phenotype and therefore must not
/// silently alter a null-calibrated search space.
fn choose_adaptive_rank(
    genotypes: &[f64],
    n_markers: usize,
    n_samples: usize,
    y: &[f64],
    grm: &[f64],
    sigma_g2: f64,
    sigma_e2: f64,
    pairs: &[(usize, usize)],
    covariates: &[f64],
    n_covariates: usize,
    candidate_ranks: Vec<usize>,
    geometry_rtol: f64,
    geometry_max_rtol: f64,
    score_audit_top_k: usize,
) -> Result<AdaptiveRankReport, String> {
    let spectrum = ScreenSpectrum::from_grm(grm, n_samples, sigma_g2, sigma_e2)?;
    choose_adaptive_rank_with_spectrum(
        genotypes,
        n_markers,
        n_samples,
        y,
        pairs,
        covariates,
        n_covariates,
        candidate_ranks,
        geometry_rtol,
        geometry_max_rtol,
        score_audit_top_k,
        &spectrum,
        sigma_g2,
        sigma_e2,
    )
}

fn choose_adaptive_rank_with_spectrum(
    genotypes: &[f64],
    n_markers: usize,
    n_samples: usize,
    y: &[f64],
    pairs: &[(usize, usize)],
    covariates: &[f64],
    n_covariates: usize,
    candidate_ranks: Vec<usize>,
    geometry_rtol: f64,
    geometry_max_rtol: f64,
    score_audit_top_k: usize,
    spectrum: &ScreenSpectrum,
    sigma_g2: f64,
    sigma_e2: f64,
) -> Result<AdaptiveRankReport, String> {
    validate_dense_inputs(genotypes, n_markers, n_samples, y)?;
    if pairs.is_empty() {
        return Err("adaptive rank pilot requires at least one marker pair".to_string());
    }
    if !geometry_rtol.is_finite() || geometry_rtol < 0.0 {
        return Err(format!(
            "geometry_rtol must be finite and >= 0; got {geometry_rtol}"
        ));
    }
    if !geometry_max_rtol.is_finite() || geometry_max_rtol < geometry_rtol {
        return Err(format!(
            "geometry_max_rtol must be finite and >= geometry_rtol; got {geometry_max_rtol}"
        ));
    }
    let mut unique_pairs = HashSet::with_capacity(pairs.len());
    for (index, &(first, second)) in pairs.iter().enumerate() {
        if first >= n_markers || second >= n_markers || first >= second {
            return Err(format!(
                "pilot_pairs[{index}] = ({first}, {second}) is not canonical or is out of range"
            ));
        }
        if !unique_pairs.insert((first, second)) {
            return Err(format!(
                "pilot_pairs[{index}] = ({first}, {second}) is a duplicate"
            ));
        }
    }
    let fixed_columns = build_fixed_columns(n_samples, covariates, n_covariates)?;
    let full_context =
        ScreenFixedContext::new(y, fixed_columns.clone(), spectrum.metric(n_samples)?)?;
    let full_geometry = geometry_for_pairs(&full_context, genotypes, n_markers, y, pairs)?;
    if full_geometry.is_empty() {
        return Err("adaptive rank pilot has no full-rank geometry-valid pairs".to_string());
    }
    let full_scores = score_pairs_for_audit(&full_context, genotypes, n_markers, y, pairs)?;
    let mut requested_ranks = if candidate_ranks.is_empty() {
        ADAPTIVE_SCREEN_RANKS.to_vec()
    } else {
        candidate_ranks
    };
    requested_ranks.retain(|rank| *rank > 0);
    if requested_ranks.is_empty() {
        return Err("candidate_ranks must contain at least one positive rank".to_string());
    }
    requested_ranks.sort_unstable();
    requested_ranks.dedup();
    let mut effective_ranks = requested_ranks
        .into_iter()
        .map(|rank| rank.min(n_samples))
        .filter(|rank| *rank > 0)
        .collect::<Vec<_>>();
    effective_ranks.sort_unstable();
    effective_ranks.dedup();
    let mut candidates = Vec::with_capacity(effective_ranks.len());
    for rank in effective_ranks {
        let metric = spectrum.metric(rank)?;
        let context = ScreenFixedContext::new(y, fixed_columns.clone(), metric.clone())?;
        let geometry = geometry_for_pairs(&context, genotypes, n_markers, y, pairs)?;
        let errors = full_geometry
            .iter()
            .filter_map(|(pair, &full_value)| {
                geometry.get(pair).map(|&value| {
                    (value - full_value).abs() / full_value.abs().max(SCREEN_VARIANCE_TOL)
                })
            })
            .filter(|error| error.is_finite())
            .collect::<Vec<_>>();
        let geometry_mean_relative_error = if errors.is_empty() {
            f64::NAN
        } else {
            errors.iter().sum::<f64>() / errors.len() as f64
        };
        let geometry_p95_relative_error = quantile(&errors, 0.95);
        let geometry_q99_relative_error = quantile(&errors, 0.99);
        let geometry_q999_relative_error = quantile(&errors, 0.999);
        let geometry_max_relative_error = errors.iter().copied().fold(0.0, f64::max);
        let candidate_scores = score_pairs_for_audit(&context, genotypes, n_markers, y, pairs)?;
        let mut common_pairs = Vec::new();
        let mut full_values = Vec::new();
        let mut candidate_values = Vec::new();
        for pair in pairs {
            if let (Some(&full_score), Some(&candidate_score)) =
                (full_scores.get(pair), candidate_scores.get(pair))
            {
                common_pairs.push(*pair);
                full_values.push(full_score);
                candidate_values.push(candidate_score);
            }
        }
        let score_spearman = pearson_correlation(
            &ordinal_ranks(&full_values),
            &ordinal_ranks(&candidate_values),
        );
        let score_top_k_overlap =
            top_score_overlap(&full_scores, &candidate_scores, score_audit_top_k);
        let geometry_pass = errors.len() == full_geometry.len()
            && geometry_p95_relative_error.is_finite()
            && geometry_p95_relative_error <= geometry_rtol
            && geometry_max_relative_error <= geometry_max_rtol;
        let diagnostic_correction_energy_c0 = spectrum.diagnostic_correction_energy(rank);
        candidates.push(AdaptiveRankCandidate {
            rank,
            baseline_weight: metric.baseline_weight,
            diagnostic_correction_energy_c0,
            diagnostic_spectral_tail_c0: (1.0 - diagnostic_correction_energy_c0).max(0.0),
            geometry_valid_pairs: errors.len(),
            geometry_mean_relative_error,
            geometry_p95_relative_error,
            geometry_q99_relative_error,
            geometry_q999_relative_error,
            geometry_max_relative_error,
            geometry_pass,
            score_pairs_compared: common_pairs.len(),
            score_spearman,
            score_top_k_overlap,
        });
    }
    let mut used_largest_rank_fallback = false;
    let selected_index = if let Some(index) = candidates
        .iter()
        .position(|candidate| candidate.geometry_pass)
    {
        index
    } else {
        // A requested rank that misses the declared geometry tolerances must
        // never be returned as if it were safe.  Add a full-rank exact
        // candidate (the reference metric is already available) and make the
        // fallback explicit in the report.  Callers that want a bounded
        // approximation despite this condition can still use the explicit
        // `Context::new(..., rank)` API.
        used_largest_rank_fallback = true;
        let full_rank = spectrum.n;
        if !candidates
            .iter()
            .any(|candidate| candidate.rank == full_rank)
        {
            let full_metric = spectrum.metric(full_rank)?;
            let full_score_pairs = full_scores.len();
            candidates.push(AdaptiveRankCandidate {
                rank: full_rank,
                baseline_weight: full_metric.baseline_weight,
                diagnostic_correction_energy_c0: spectrum.diagnostic_correction_energy(full_rank),
                diagnostic_spectral_tail_c0: 0.0,
                geometry_valid_pairs: full_geometry.len(),
                geometry_mean_relative_error: 0.0,
                geometry_p95_relative_error: 0.0,
                geometry_q99_relative_error: 0.0,
                geometry_q999_relative_error: 0.0,
                geometry_max_relative_error: 0.0,
                geometry_pass: true,
                score_pairs_compared: full_score_pairs,
                score_spearman: if full_score_pairs >= 2 { 1.0 } else { f64::NAN },
                score_top_k_overlap: if full_score_pairs > 0 { 1.0 } else { f64::NAN },
            });
        }
        candidates
            .iter()
            .position(|candidate| candidate.rank == full_rank)
            .expect("full-rank fallback candidate must be present")
    };
    Ok(AdaptiveRankReport {
        selected_rank: candidates[selected_index].rank,
        full_rank: n_samples,
        pilot_pair_count: pairs.len(),
        full_geometry_valid_pairs: full_geometry.len(),
        sigma_g2,
        sigma_e2,
        diagnostic_baseline: spectrum.diagnostic_baseline,
        geometry_rtol,
        geometry_max_rtol,
        score_audit_top_k,
        used_largest_rank_fallback,
        candidates,
    })
}

fn top_score_overlap(
    full_scores: &HashMap<(usize, usize), f64>,
    candidate_scores: &HashMap<(usize, usize), f64>,
    top_k: usize,
) -> f64 {
    if top_k == 0 || full_scores.is_empty() {
        return f64::NAN;
    }
    let top = |scores: &HashMap<(usize, usize), f64>| {
        let mut entries = scores.iter().collect::<Vec<_>>();
        entries.sort_by(|(left_pair, left_score), (right_pair, right_score)| {
            right_score
                .total_cmp(left_score)
                .then_with(|| left_pair.cmp(right_pair))
        });
        entries
            .into_iter()
            .take(top_k)
            .map(|(pair, _)| *pair)
            .collect::<HashSet<_>>()
    };
    let full_top = top(full_scores);
    let candidate_top = top(candidate_scores);
    if full_top.is_empty() {
        f64::NAN
    } else {
        full_top.intersection(&candidate_top).count() as f64 / full_top.len() as f64
    }
}

fn scan_with_context(
    context: &ScreenFixedContext,
    genotypes: &[f64],
    n_markers: usize,
    y: &[f64],
    top_k: usize,
    threads: usize,
) -> Result<ScreenScanResult, String> {
    let marker_stats = context.prepare_markers(genotypes, n_markers, y)?;
    let rank = context.metric.rank();
    let block_capacity = parse_screen_block_size();
    // Validate all dimensions before entering the Rayon closure.  The scan
    // itself has no fallible return channel per pair, so silently substituting
    // `CblasInt::MAX` here would turn a malformed request into undefined
    // BLAS dimensions.
    let n_blas = CblasInt::try_from(context.n)
        .map_err(|_| "screen sample count exceeds CBLAS range".to_string())?;
    let rank_blas =
        CblasInt::try_from(rank).map_err(|_| "screen rank exceeds CBLAS range".to_string())?;
    let _block_capacity_blas = CblasInt::try_from(block_capacity)
        .map_err(|_| "screen block size exceeds CBLAS range".to_string())?;
    let _blas_guard = BlasThreadGuard::enter(parse_screen_blas_threads());
    let first_count = n_markers.saturating_sub(1);
    let scan_range = |range: std::ops::Range<usize>| -> ScreenAccumulator {
        let mut accumulator = ScreenAccumulator::default();
        let mut workspace = ScreenWorkspace::new(context.fixed_rank);
        let mut block_interaction = vec![0.0; context.n * block_capacity];
        let mut block_q_interaction = vec![0.0; rank * block_capacity];
        for first in range {
            let first_values = &genotypes[first * context.n..(first + 1) * context.n];
            let mut second_start = first + 1;
            while second_start < n_markers {
                let block_width = (n_markers - second_start).min(block_capacity);
                for local in 0..block_width {
                    let second = second_start + local;
                    let second_values = &genotypes[second * context.n..(second + 1) * context.n];
                    block_interaction[local * context.n..(local + 1) * context.n]
                        .iter_mut()
                        .zip(first_values.iter().zip(second_values))
                        .for_each(|(out, (left, right))| *out = left * right);
                }
                if rank > 0 {
                    let width_blas = CblasInt::try_from(block_width)
                        .expect("validated screen block width exceeds CBLAS range");
                    unsafe {
                        cblas_dgemm_dispatch(
                            CBLAS_COL_MAJOR,
                            CBLAS_TRANS,
                            CBLAS_NO_TRANS,
                            rank_blas,
                            width_blas,
                            n_blas,
                            1.0,
                            context.metric.eigenvectors.as_ptr(),
                            n_blas,
                            block_interaction.as_ptr(),
                            n_blas,
                            0.0,
                            block_q_interaction.as_mut_ptr(),
                            rank_blas,
                        );
                    }
                }
                for local in 0..block_width {
                    let second = second_start + local;
                    let second_values = &genotypes[second * context.n..(second + 1) * context.n];
                    let q_interaction = &block_q_interaction[local * rank..(local + 1) * rank];
                    let interaction_values =
                        &block_interaction[local * context.n..(local + 1) * context.n];
                    // `interaction_values` is already the product vector, so
                    // the scorer only traverses it once for the pair-specific
                    // moments.  The argument is retained as `second_values`
                    // to keep the marker-major API explicit.
                    let score = context.score_pair(
                        first,
                        second,
                        first_values,
                        second_values,
                        interaction_values,
                        y,
                        &marker_stats,
                        q_interaction,
                        &mut workspace,
                    );
                    match score {
                        ScreenPairScore::Valid(score) => {
                            accumulator.pairs_evaluated =
                                accumulator.pairs_evaluated.saturating_add(1);
                            accumulator.push(first, second, score, top_k);
                        }
                        ScreenPairScore::Skipped(reason) => accumulator.skip(reason),
                    }
                }
                second_start += block_width;
            }
        }
        accumulator
    };
    let accumulator = if threads <= 1 || first_count <= 1 {
        scan_range(0..first_count)
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
            .map_err(|error| format!("failed to build screen Rayon pool: {error}"))?
            .install(|| {
                ranges
                    .par_iter()
                    .map(|range| scan_range(range.clone()))
                    .collect::<Vec<_>>()
            });
        let mut merged = ScreenAccumulator::default();
        for partial in partials {
            merged.merge(partial, top_k);
        }
        merged
    };
    let pairs_evaluated = accumulator.pairs_evaluated;
    let pairs_skipped = accumulator.pairs_skipped;
    let skipped_lower_geometry = accumulator.skipped_lower_geometry;
    let skipped_interaction_variance = accumulator.skipped_interaction_variance;
    let skipped_nonfinite = accumulator.skipped_nonfinite;
    let candidates = accumulator.into_sorted();
    Ok(ScreenScanResult {
        candidates,
        pairs_evaluated,
        pairs_skipped,
        skipped_lower_geometry,
        skipped_interaction_variance,
        skipped_nonfinite,
        rank,
        baseline_weight: context.metric.baseline_weight,
    })
}

/// Scan an explicit, canonicalized list of marker pairs.
///
/// This is used by the genome-wide owner-window path: each biological pair is
/// assigned to exactly one window before entering the scorer.  Keeping the
/// selected-pair scanner separate from the all-pairs implementation preserves
/// the latter's hot loop while allowing the caller to avoid rescoring pairs in
/// overlapping windows.
fn scan_selected_with_context(
    context: &ScreenFixedContext,
    genotypes: &[f64],
    n_markers: usize,
    y: &[f64],
    pairs: &[(usize, usize)],
    top_k: usize,
    threads: usize,
) -> Result<ScreenScanResult, String> {
    let marker_stats = context.prepare_markers(genotypes, n_markers, y)?;
    for (index, &(first, second)) in pairs.iter().enumerate() {
        if first >= n_markers || second >= n_markers || first >= second {
            return Err(format!(
                "selected pair {index} ({first}, {second}) is not canonical for n_markers={n_markers}"
            ));
        }
    }
    let rank = context.metric.rank();
    let block_capacity = parse_screen_block_size();
    let n_blas = CblasInt::try_from(context.n)
        .map_err(|_| "screen sample count exceeds CBLAS range".to_string())?;
    let rank_blas =
        CblasInt::try_from(rank).map_err(|_| "screen rank exceeds CBLAS range".to_string())?;
    let _block_capacity_blas = CblasInt::try_from(block_capacity)
        .map_err(|_| "screen block size exceeds CBLAS range".to_string())?;
    let _blas_guard = BlasThreadGuard::enter(parse_screen_blas_threads());
    let scan_range = |range: std::ops::Range<usize>| -> ScreenAccumulator {
        let mut accumulator = ScreenAccumulator::default();
        let mut workspace = ScreenWorkspace::new(context.fixed_rank);
        let mut block_interaction = vec![0.0; context.n * block_capacity];
        let mut block_q_interaction = vec![0.0; rank * block_capacity];
        for block_start in (range.start..range.end).step_by(block_capacity.max(1)) {
            let block_width = (range.end - block_start).min(block_capacity);
            for local in 0..block_width {
                let (first, second) = pairs[block_start + local];
                let first_values = &genotypes[first * context.n..(first + 1) * context.n];
                let second_values = &genotypes[second * context.n..(second + 1) * context.n];
                block_interaction[local * context.n..(local + 1) * context.n]
                    .iter_mut()
                    .zip(first_values.iter().zip(second_values))
                    .for_each(|(out, (left, right))| *out = left * right);
            }
            if rank > 0 {
                let width_blas = CblasInt::try_from(block_width)
                    .expect("validated screen block width exceeds CBLAS range");
                unsafe {
                    cblas_dgemm_dispatch(
                        CBLAS_COL_MAJOR,
                        CBLAS_TRANS,
                        CBLAS_NO_TRANS,
                        rank_blas,
                        width_blas,
                        n_blas,
                        1.0,
                        context.metric.eigenvectors.as_ptr(),
                        n_blas,
                        block_interaction.as_ptr(),
                        n_blas,
                        0.0,
                        block_q_interaction.as_mut_ptr(),
                        rank_blas,
                    );
                }
            }
            for local in 0..block_width {
                let (first, second) = pairs[block_start + local];
                let first_values = &genotypes[first * context.n..(first + 1) * context.n];
                let second_values = &genotypes[second * context.n..(second + 1) * context.n];
                let q_interaction = &block_q_interaction[local * rank..(local + 1) * rank];
                let interaction_values =
                    &block_interaction[local * context.n..(local + 1) * context.n];
                match context.score_pair(
                    first,
                    second,
                    first_values,
                    second_values,
                    interaction_values,
                    y,
                    &marker_stats,
                    q_interaction,
                    &mut workspace,
                ) {
                    ScreenPairScore::Valid(score) => {
                        accumulator.pairs_evaluated = accumulator.pairs_evaluated.saturating_add(1);
                        accumulator.push(first, second, score, top_k);
                    }
                    ScreenPairScore::Skipped(reason) => accumulator.skip(reason),
                }
            }
        }
        accumulator
    };
    let pair_count = pairs.len();
    let accumulator = if threads <= 1 || pair_count <= 1 {
        scan_range(0..pair_count)
    } else {
        let chunk_count = threads.saturating_mul(4).max(1);
        let chunk_len = (pair_count + chunk_count - 1) / chunk_count;
        let ranges = (0..pair_count)
            .step_by(chunk_len.max(1))
            .map(|start| start..(start + chunk_len).min(pair_count))
            .collect::<Vec<_>>();
        let partials = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|error| format!("failed to build screen Rayon pool: {error}"))?
            .install(|| {
                ranges
                    .par_iter()
                    .map(|range| scan_range(range.clone()))
                    .collect::<Vec<_>>()
            });
        let mut merged = ScreenAccumulator::default();
        for partial in partials {
            merged.merge(partial, top_k);
        }
        merged
    };
    let pairs_evaluated = accumulator.pairs_evaluated;
    let pairs_skipped = accumulator.pairs_skipped;
    let skipped_lower_geometry = accumulator.skipped_lower_geometry;
    let skipped_interaction_variance = accumulator.skipped_interaction_variance;
    let skipped_nonfinite = accumulator.skipped_nonfinite;
    let candidates = accumulator.into_sorted();
    Ok(ScreenScanResult {
        candidates,
        pairs_evaluated,
        pairs_skipped,
        skipped_lower_geometry,
        skipped_interaction_variance,
        skipped_nonfinite,
        rank,
        baseline_weight: context.metric.baseline_weight,
    })
}

/// Scan an explicit pair list once while maintaining two independent Top-K
/// heaps.  A bit in `groups[index]` sends a pair's score to the corresponding
/// heap (bit 0 = first, bit 1 = second).  This is used by the canonical-owner
/// window path: an owner pair is scored once, then its score can contribute to
/// both the owner window and the immediately preceding overlapping window.
fn scan_selected_grouped_with_context(
    context: &ScreenFixedContext,
    genotypes: &[f64],
    n_markers: usize,
    y: &[f64],
    pairs: &[(usize, usize)],
    groups: &[u8],
    top_k: usize,
    threads: usize,
) -> Result<GroupedScreenScanResult, String> {
    if pairs.len() != groups.len() {
        return Err(format!(
            "selected pair/group lengths differ: pairs={}, groups={}",
            pairs.len(),
            groups.len()
        ));
    }
    let marker_stats = context.prepare_markers(genotypes, n_markers, y)?;
    for (index, &(first, second)) in pairs.iter().enumerate() {
        if first >= n_markers || second >= n_markers || first >= second {
            return Err(format!(
                "selected pair {index} ({first}, {second}) is not canonical for n_markers={n_markers}"
            ));
        }
    }
    let rank = context.metric.rank();
    let block_capacity = parse_screen_block_size();
    let n_blas = CblasInt::try_from(context.n)
        .map_err(|_| "screen sample count exceeds CBLAS range".to_string())?;
    let rank_blas =
        CblasInt::try_from(rank).map_err(|_| "screen rank exceeds CBLAS range".to_string())?;
    let _block_capacity_blas = CblasInt::try_from(block_capacity)
        .map_err(|_| "screen block size exceeds CBLAS range".to_string())?;
    let _blas_guard = BlasThreadGuard::enter(parse_screen_blas_threads());
    let scan_range = |range: std::ops::Range<usize>| -> GroupedScreenAccumulator {
        let mut accumulator = GroupedScreenAccumulator::default();
        let mut workspace = ScreenWorkspace::new(context.fixed_rank);
        let mut block_interaction = vec![0.0; context.n * block_capacity];
        let mut block_q_interaction = vec![0.0; rank * block_capacity];
        for block_start in (range.start..range.end).step_by(block_capacity.max(1)) {
            let block_width = (range.end - block_start).min(block_capacity);
            for local in 0..block_width {
                let (first, second) = pairs[block_start + local];
                let first_values = &genotypes[first * context.n..(first + 1) * context.n];
                let second_values = &genotypes[second * context.n..(second + 1) * context.n];
                block_interaction[local * context.n..(local + 1) * context.n]
                    .iter_mut()
                    .zip(first_values.iter().zip(second_values))
                    .for_each(|(out, (left, right))| *out = left * right);
            }
            if rank > 0 {
                let width_blas = CblasInt::try_from(block_width)
                    .expect("validated screen block width exceeds CBLAS range");
                unsafe {
                    cblas_dgemm_dispatch(
                        CBLAS_COL_MAJOR,
                        CBLAS_TRANS,
                        CBLAS_NO_TRANS,
                        rank_blas,
                        width_blas,
                        n_blas,
                        1.0,
                        context.metric.eigenvectors.as_ptr(),
                        n_blas,
                        block_interaction.as_ptr(),
                        n_blas,
                        0.0,
                        block_q_interaction.as_mut_ptr(),
                        rank_blas,
                    );
                }
            }
            for local in 0..block_width {
                let pair_index = block_start + local;
                let (first, second) = pairs[pair_index];
                let first_values = &genotypes[first * context.n..(first + 1) * context.n];
                let second_values = &genotypes[second * context.n..(second + 1) * context.n];
                let q_interaction = &block_q_interaction[local * rank..(local + 1) * rank];
                let interaction_values =
                    &block_interaction[local * context.n..(local + 1) * context.n];
                match context.score_pair(
                    first,
                    second,
                    first_values,
                    second_values,
                    interaction_values,
                    y,
                    &marker_stats,
                    q_interaction,
                    &mut workspace,
                ) {
                    ScreenPairScore::Valid(score) => {
                        accumulator.counts.pairs_evaluated =
                            accumulator.counts.pairs_evaluated.saturating_add(1);
                        let group = groups[pair_index];
                        if group & 1 != 0 {
                            accumulator.first.push(first, second, score, top_k);
                        }
                        if group & 2 != 0 {
                            accumulator.second.push(first, second, score, top_k);
                        }
                    }
                    ScreenPairScore::Skipped(reason) => accumulator.counts.skip(reason),
                }
            }
        }
        accumulator
    };
    let pair_count = pairs.len();
    let accumulator = if threads <= 1 || pair_count <= 1 {
        scan_range(0..pair_count)
    } else {
        let chunk_count = threads.saturating_mul(4).max(1);
        let chunk_len = (pair_count + chunk_count - 1) / chunk_count;
        let ranges = (0..pair_count)
            .step_by(chunk_len.max(1))
            .map(|start| start..(start + chunk_len).min(pair_count))
            .collect::<Vec<_>>();
        let partials = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|error| format!("failed to build screen Rayon pool: {error}"))?
            .install(|| {
                ranges
                    .par_iter()
                    .map(|range| scan_range(range.clone()))
                    .collect::<Vec<_>>()
            });
        let mut merged = GroupedScreenAccumulator::default();
        for partial in partials {
            merged.merge(partial, top_k);
        }
        merged
    };
    let counts = accumulator.counts;
    Ok(GroupedScreenScanResult {
        first: accumulator.first.into_sorted(),
        second: accumulator.second.into_sorted(),
        pairs_evaluated: counts.pairs_evaluated,
        pairs_skipped: counts.pairs_skipped,
        skipped_lower_geometry: counts.skipped_lower_geometry,
        skipped_interaction_variance: counts.skipped_interaction_variance,
        skipped_nonfinite: counts.skipped_nonfinite,
        rank,
        baseline_weight: context.metric.baseline_weight,
    })
}

/// Score owner pairs once while maintaining a local Top-K heap for every
/// window membership.  Unlike the historical two-group helper, this accepts
/// an arbitrary number of overlapping windows per pair.
fn scan_selected_memberships_with_context(
    context: &ScreenFixedContext,
    genotypes: &[f64],
    n_markers: usize,
    y: &[f64],
    pairs: &[(usize, usize)],
    pair_window_offsets: &[usize],
    pair_window_ids: &[u64],
    top_k: usize,
    threads: usize,
) -> Result<MembershipScreenScanResult, String> {
    if pair_window_offsets.len() != pairs.len().saturating_add(1) {
        return Err(format!(
            "pair/window offsets must have n_pairs + 1 entries: pairs={}, offsets={}",
            pairs.len(),
            pair_window_offsets.len()
        ));
    }
    if pair_window_offsets.first().copied().unwrap_or(0) != 0
        || pair_window_offsets.last().copied().unwrap_or(0) != pair_window_ids.len()
        || pair_window_offsets
            .windows(2)
            .any(|window| window[0] > window[1])
    {
        return Err("pair/window offsets are not a valid monotone partition".to_string());
    }
    let mut window_ids = pair_window_ids.to_vec();
    window_ids.sort_unstable();
    window_ids.dedup();
    if !pairs.is_empty() && window_ids.is_empty() {
        return Err("non-empty pair set must have at least one window membership".to_string());
    }
    let window_lookup = window_ids
        .iter()
        .enumerate()
        .map(|(index, &window_id)| (window_id, index))
        .collect::<HashMap<_, _>>();

    let marker_stats = context.prepare_markers(genotypes, n_markers, y)?;
    for (index, &(first, second)) in pairs.iter().enumerate() {
        if first >= n_markers || second >= n_markers || first >= second {
            return Err(format!(
                "selected pair {index} ({first}, {second}) is not canonical for n_markers={n_markers}"
            ));
        }
        let start = pair_window_offsets[index];
        let end = pair_window_offsets[index + 1];
        if start == end {
            return Err(format!(
                "pair {index} must have at least one window membership"
            ));
        }
        if pair_window_ids[start..end]
            .windows(2)
            .any(|window| window[0] >= window[1])
        {
            return Err(format!(
                "pair {index} has duplicate or unsorted window memberships"
            ));
        }
    }
    let rank = context.metric.rank();
    let block_capacity = parse_screen_block_size();
    let n_blas = CblasInt::try_from(context.n)
        .map_err(|_| "screen sample count exceeds CBLAS range".to_string())?;
    let rank_blas =
        CblasInt::try_from(rank).map_err(|_| "screen rank exceeds CBLAS range".to_string())?;
    let _block_capacity_blas = CblasInt::try_from(block_capacity)
        .map_err(|_| "screen block size exceeds CBLAS range".to_string())?;
    let _blas_guard = BlasThreadGuard::enter(parse_screen_blas_threads());
    let scan_range = |range: std::ops::Range<usize>| -> Vec<ScreenAccumulator> {
        let mut accumulators = (0..window_ids.len())
            .map(|_| ScreenAccumulator::default())
            .collect::<Vec<_>>();
        let mut workspace = ScreenWorkspace::new(context.fixed_rank);
        let mut block_interaction = vec![0.0; context.n * block_capacity];
        let mut block_q_interaction = vec![0.0; rank * block_capacity];
        for block_start in (range.start..range.end).step_by(block_capacity.max(1)) {
            let block_width = (range.end - block_start).min(block_capacity);
            for local in 0..block_width {
                let pair_index = block_start + local;
                let (first, second) = pairs[pair_index];
                let first_values = &genotypes[first * context.n..(first + 1) * context.n];
                let second_values = &genotypes[second * context.n..(second + 1) * context.n];
                block_interaction[local * context.n..(local + 1) * context.n]
                    .iter_mut()
                    .zip(first_values.iter().zip(second_values))
                    .for_each(|(out, (left, right))| *out = left * right);
            }
            if rank > 0 {
                let width_blas = CblasInt::try_from(block_width)
                    .expect("validated screen block width exceeds CBLAS range");
                unsafe {
                    cblas_dgemm_dispatch(
                        CBLAS_COL_MAJOR,
                        CBLAS_TRANS,
                        CBLAS_NO_TRANS,
                        rank_blas,
                        width_blas,
                        n_blas,
                        1.0,
                        context.metric.eigenvectors.as_ptr(),
                        n_blas,
                        block_interaction.as_ptr(),
                        n_blas,
                        0.0,
                        block_q_interaction.as_mut_ptr(),
                        rank_blas,
                    );
                }
            }
            for local in 0..block_width {
                let pair_index = block_start + local;
                let (first, second) = pairs[pair_index];
                let q_interaction = &block_q_interaction[local * rank..(local + 1) * rank];
                let interaction_values =
                    &block_interaction[local * context.n..(local + 1) * context.n];
                match context.score_pair(
                    first,
                    second,
                    &genotypes[first * context.n..(first + 1) * context.n],
                    &genotypes[second * context.n..(second + 1) * context.n],
                    interaction_values,
                    y,
                    &marker_stats,
                    q_interaction,
                    &mut workspace,
                ) {
                    ScreenPairScore::Valid(score) => {
                        // Count each owner pair once, irrespective of the
                        // number of windows in its membership range.
                        accumulators[0].pairs_evaluated =
                            accumulators[0].pairs_evaluated.saturating_add(1);
                        let start = pair_window_offsets[pair_index];
                        let end = pair_window_offsets[pair_index + 1];
                        for &window_id in &pair_window_ids[start..end] {
                            if let Some(&group) = window_lookup.get(&window_id) {
                                accumulators[group].push(first, second, score, top_k);
                            }
                        }
                    }
                    ScreenPairScore::Skipped(reason) => accumulators[0].skip(reason),
                }
            }
        }
        accumulators
    };
    let pair_count = pairs.len();
    let accumulators = if threads <= 1 || pair_count <= 1 {
        scan_range(0..pair_count)
    } else {
        let chunk_count = threads.saturating_mul(4).max(1);
        let chunk_len = (pair_count + chunk_count - 1) / chunk_count;
        let ranges = (0..pair_count)
            .step_by(chunk_len.max(1))
            .map(|start| start..(start + chunk_len).min(pair_count))
            .collect::<Vec<_>>();
        let partials = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|error| format!("failed to build screen Rayon pool: {error}"))?
            .install(|| {
                ranges
                    .par_iter()
                    .map(|range| scan_range(range.clone()))
                    .collect::<Vec<_>>()
            });
        let mut merged = (0..window_ids.len())
            .map(|_| ScreenAccumulator::default())
            .collect::<Vec<_>>();
        for partial in partials {
            for (target, source) in merged.iter_mut().zip(partial) {
                target.merge(source, top_k);
            }
        }
        merged
    };
    let mut counts = ScreenAccumulator::default();
    let candidates = accumulators
        .into_iter()
        .enumerate()
        .map(|(index, accumulator)| {
            if index == 0 {
                counts.pairs_evaluated = accumulator.pairs_evaluated;
                counts.pairs_skipped = accumulator.pairs_skipped;
                counts.skipped_lower_geometry = accumulator.skipped_lower_geometry;
                counts.skipped_interaction_variance = accumulator.skipped_interaction_variance;
                counts.skipped_nonfinite = accumulator.skipped_nonfinite;
            }
            accumulator.into_sorted()
        })
        .collect::<Vec<_>>();
    Ok(MembershipScreenScanResult {
        window_ids,
        candidates,
        pairs_evaluated: counts.pairs_evaluated,
        pairs_skipped: counts.pairs_skipped,
        skipped_lower_geometry: counts.skipped_lower_geometry,
        skipped_interaction_variance: counts.skipped_interaction_variance,
        skipped_nonfinite: counts.skipped_nonfinite,
        rank,
        baseline_weight: context.metric.baseline_weight,
    })
}

fn scan_lm_screen_selected_grouped(
    genotypes: &[f64],
    n_markers: usize,
    n_samples: usize,
    y: &[f64],
    pairs: &[(usize, usize)],
    groups: &[u8],
    top_k: usize,
    threads: usize,
    covariates: &[f64],
    n_covariates: usize,
) -> Result<GroupedScreenScanResult, String> {
    validate_dense_inputs(genotypes, n_markers, n_samples, y)?;
    let fixed_columns = build_fixed_columns(n_samples, covariates, n_covariates)?;
    let metric = ScreenMetric {
        n: n_samples,
        baseline_weight: 1.0,
        eigenvectors: Vec::new(),
        correction_weights: Vec::new(),
    };
    let context = ScreenFixedContext::new(y, fixed_columns, metric)?;
    scan_selected_grouped_with_context(
        &context,
        genotypes,
        n_markers,
        y,
        pairs,
        groups,
        top_k,
        threads.max(1),
    )
}

pub(crate) fn scan_lm_screen(
    genotypes: &[f64],
    n_markers: usize,
    n_samples: usize,
    y: &[f64],
    top_k: usize,
    threads: usize,
    covariates: &[f64],
    n_covariates: usize,
) -> Result<ScreenScanResult, String> {
    validate_dense_inputs(genotypes, n_markers, n_samples, y)?;
    let fixed_columns = build_fixed_columns(n_samples, covariates, n_covariates)?;
    let metric = ScreenMetric {
        n: n_samples,
        baseline_weight: 1.0,
        eigenvectors: Vec::new(),
        correction_weights: Vec::new(),
    };
    let context = ScreenFixedContext::new(y, fixed_columns, metric)?;
    scan_with_context(&context, genotypes, n_markers, y, top_k, threads.max(1))
}

pub(crate) fn scan_lm_screen_selected(
    genotypes: &[f64],
    n_markers: usize,
    n_samples: usize,
    y: &[f64],
    pairs: &[(usize, usize)],
    top_k: usize,
    threads: usize,
    covariates: &[f64],
    n_covariates: usize,
) -> Result<ScreenScanResult, String> {
    validate_dense_inputs(genotypes, n_markers, n_samples, y)?;
    let fixed_columns = build_fixed_columns(n_samples, covariates, n_covariates)?;
    let metric = ScreenMetric {
        n: n_samples,
        baseline_weight: 1.0,
        eigenvectors: Vec::new(),
        correction_weights: Vec::new(),
    };
    let context = ScreenFixedContext::new(y, fixed_columns, metric)?;
    scan_selected_with_context(
        &context,
        genotypes,
        n_markers,
        y,
        pairs,
        top_k,
        threads.max(1),
    )
}

pub(crate) fn scan_lowrank_grm_screen(
    genotypes: &[f64],
    n_markers: usize,
    n_samples: usize,
    y: &[f64],
    grm: &[f64],
    sigma_g2: f64,
    sigma_e2: f64,
    rank: usize,
    top_k: usize,
    threads: usize,
    covariates: &[f64],
    n_covariates: usize,
) -> Result<ScreenScanResult, String> {
    validate_dense_inputs(genotypes, n_markers, n_samples, y)?;
    let fixed_columns = build_fixed_columns(n_samples, covariates, n_covariates)?;
    let metric = build_lowrank_metric(grm, n_samples, sigma_g2, sigma_e2, rank)?;
    let context = ScreenFixedContext::new(y, fixed_columns, metric)?;
    scan_with_context(&context, genotypes, n_markers, y, top_k, threads.max(1))
}

fn array1_to_vec(array: &PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    let view = array.as_array();
    view.iter().copied().collect()
}

fn array1_to_u8(array: &PyReadonlyArray1<'_, u8>) -> Vec<u8> {
    let view = array.as_array();
    view.iter().copied().collect()
}

fn array2_to_vec(array: &PyReadonlyArray2<'_, f64>) -> Vec<f64> {
    let view = array.as_array();
    let (rows, columns) = view.dim();
    let mut values = Vec::with_capacity(rows.saturating_mul(columns));
    for row in 0..rows {
        for column in 0..columns {
            values.push(view[[row, column]]);
        }
    }
    values
}

fn array2_to_pairs(array: &PyReadonlyArray2<'_, i64>) -> PyResult<Vec<(usize, usize)>> {
    let shape = array.shape();
    if shape.len() != 2 || shape[1] != 2 {
        return Err(PyValueError::new_err(format!(
            "pairs must have shape (n_pairs, 2); got {:?}",
            shape
        )));
    }
    let view = array.as_array();
    let mut pairs = Vec::with_capacity(shape[0]);
    for row in 0..shape[0] {
        let first = view[[row, 0]];
        let second = view[[row, 1]];
        if first < 0 || second < 0 || first >= second {
            return Err(PyValueError::new_err(format!(
                "pairs[{row}] = ({first}, {second}) is not canonical"
            )));
        }
        pairs.push((first as usize, second as usize));
    }
    Ok(pairs)
}

fn screen_result_to_py<'py>(
    py: Python<'py>,
    result: ScreenScanResult,
    mode: &'static str,
) -> PyResult<Bound<'py, PyDict>> {
    let first = result
        .candidates
        .iter()
        .map(|candidate| candidate.first as i64)
        .collect::<Vec<_>>();
    let second = result
        .candidates
        .iter()
        .map(|candidate| candidate.second as i64)
        .collect::<Vec<_>>();
    let scores = result
        .candidates
        .iter()
        .map(|candidate| candidate.score)
        .collect::<Vec<_>>();
    let out = PyDict::new(py);
    out.set_item("first", PyArray1::from_vec(py, first))?;
    out.set_item("second", PyArray1::from_vec(py, second))?;
    out.set_item("score", PyArray1::from_vec(py, scores))?;
    out.set_item("pairs_evaluated", result.pairs_evaluated)?;
    out.set_item("pairs_skipped", result.pairs_skipped)?;
    out.set_item("skipped_lower_geometry", result.skipped_lower_geometry)?;
    out.set_item(
        "skipped_interaction_variance",
        result.skipped_interaction_variance,
    )?;
    out.set_item("skipped_nonfinite", result.skipped_nonfinite)?;
    out.set_item("rank", result.rank)?;
    out.set_item("baseline_weight", result.baseline_weight)?;
    out.set_item("mode", mode)?;
    Ok(out)
}

fn adaptive_rank_report_to_py<'py>(
    py: Python<'py>,
    report: &AdaptiveRankReport,
) -> PyResult<Bound<'py, PyDict>> {
    let candidates = PyList::empty(py);
    for candidate in &report.candidates {
        let item = PyDict::new(py);
        item.set_item("rank", candidate.rank)?;
        item.set_item("baseline_weight", candidate.baseline_weight)?;
        item.set_item(
            "diagnostic_correction_energy_c0",
            candidate.diagnostic_correction_energy_c0,
        )?;
        item.set_item(
            "diagnostic_spectral_tail_c0",
            candidate.diagnostic_spectral_tail_c0,
        )?;
        item.set_item("geometry_valid_pairs", candidate.geometry_valid_pairs)?;
        item.set_item(
            "geometry_mean_relative_error",
            candidate.geometry_mean_relative_error,
        )?;
        item.set_item(
            "geometry_p95_relative_error",
            candidate.geometry_p95_relative_error,
        )?;
        item.set_item(
            "geometry_q99_relative_error",
            candidate.geometry_q99_relative_error,
        )?;
        item.set_item(
            "geometry_q999_relative_error",
            candidate.geometry_q999_relative_error,
        )?;
        item.set_item(
            "geometry_max_relative_error",
            candidate.geometry_max_relative_error,
        )?;
        item.set_item("geometry_pass", candidate.geometry_pass)?;
        item.set_item("score_pairs_compared", candidate.score_pairs_compared)?;
        item.set_item("score_spearman", candidate.score_spearman)?;
        item.set_item("score_top_k_overlap", candidate.score_top_k_overlap)?;
        candidates.append(item)?;
    }
    let out = PyDict::new(py);
    out.set_item("mode", "adaptive")?;
    out.set_item("selected_rank", report.selected_rank)?;
    out.set_item("full_rank", report.full_rank)?;
    out.set_item("pilot_pair_count", report.pilot_pair_count)?;
    out.set_item(
        "full_geometry_valid_pairs",
        report.full_geometry_valid_pairs,
    )?;
    out.set_item("sigma_g2", report.sigma_g2)?;
    out.set_item("sigma_e2", report.sigma_e2)?;
    out.set_item("variance_ratio", report.sigma_g2 / report.sigma_e2)?;
    out.set_item("diagnostic_baseline", report.diagnostic_baseline)?;
    out.set_item("geometry_rtol", report.geometry_rtol)?;
    out.set_item("geometry_max_rtol", report.geometry_max_rtol)?;
    out.set_item("score_audit_top_k", report.score_audit_top_k)?;
    out.set_item("selection_criterion", "geometry_p95_and_max_relative_error")?;
    out.set_item(
        "used_largest_rank_fallback",
        report.used_largest_rank_fallback,
    )?;
    out.set_item(
        "fallback_mode",
        if report.used_largest_rank_fallback {
            "full_rank_exact"
        } else {
            "geometry_candidate"
        },
    )?;
    out.set_item("candidates", candidates)?;
    Ok(out)
}

fn grouped_screen_result_to_py<'py>(
    py: Python<'py>,
    result: GroupedScreenScanResult,
    mode: &'static str,
) -> PyResult<Bound<'py, PyDict>> {
    let (first_a, second_a, scores_a) = result
        .first
        .iter()
        .map(|candidate| {
            (
                candidate.first as i64,
                candidate.second as i64,
                candidate.score,
            )
        })
        .fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |mut output, (first, second, score)| {
                output.0.push(first);
                output.1.push(second);
                output.2.push(score);
                output
            },
        );
    let (first_b, second_b, scores_b) = result
        .second
        .iter()
        .map(|candidate| {
            (
                candidate.first as i64,
                candidate.second as i64,
                candidate.score,
            )
        })
        .fold(
            (Vec::new(), Vec::new(), Vec::new()),
            |mut output, (first, second, score)| {
                output.0.push(first);
                output.1.push(second);
                output.2.push(score);
                output
            },
        );
    let out = PyDict::new(py);
    out.set_item("first_a", PyArray1::from_vec(py, first_a))?;
    out.set_item("second_a", PyArray1::from_vec(py, second_a))?;
    out.set_item("score_a", PyArray1::from_vec(py, scores_a))?;
    out.set_item("first_b", PyArray1::from_vec(py, first_b))?;
    out.set_item("second_b", PyArray1::from_vec(py, second_b))?;
    out.set_item("score_b", PyArray1::from_vec(py, scores_b))?;
    out.set_item("pairs_evaluated", result.pairs_evaluated)?;
    out.set_item("pairs_skipped", result.pairs_skipped)?;
    out.set_item("skipped_lower_geometry", result.skipped_lower_geometry)?;
    out.set_item(
        "skipped_interaction_variance",
        result.skipped_interaction_variance,
    )?;
    out.set_item("skipped_nonfinite", result.skipped_nonfinite)?;
    out.set_item("rank", result.rank)?;
    out.set_item("baseline_weight", result.baseline_weight)?;
    out.set_item("mode", mode)?;
    Ok(out)
}

fn membership_screen_result_to_py<'py>(
    py: Python<'py>,
    result: MembershipScreenScanResult,
    mode: &'static str,
) -> PyResult<Bound<'py, PyDict>> {
    let mut first = Vec::new();
    let mut second = Vec::new();
    let mut scores = Vec::new();
    let mut offsets = Vec::with_capacity(result.candidates.len().saturating_add(1));
    offsets.push(0_i64);
    for candidates in result.candidates {
        for candidate in candidates {
            first.push(candidate.first as i64);
            second.push(candidate.second as i64);
            scores.push(candidate.score);
        }
        offsets.push(first.len() as i64);
    }
    let out = PyDict::new(py);
    out.set_item("first", PyArray1::from_vec(py, first))?;
    out.set_item("second", PyArray1::from_vec(py, second))?;
    out.set_item("score", PyArray1::from_vec(py, scores))?;
    out.set_item("window_ids", PyArray1::from_vec(py, result.window_ids))?;
    out.set_item("window_candidate_offsets", PyArray1::from_vec(py, offsets))?;
    out.set_item("pairs_evaluated", result.pairs_evaluated)?;
    out.set_item("pairs_skipped", result.pairs_skipped)?;
    out.set_item("skipped_lower_geometry", result.skipped_lower_geometry)?;
    out.set_item(
        "skipped_interaction_variance",
        result.skipped_interaction_variance,
    )?;
    out.set_item("skipped_nonfinite", result.skipped_nonfinite)?;
    out.set_item("rank", result.rank)?;
    out.set_item("baseline_weight", result.baseline_weight)?;
    out.set_item("mode", mode)?;
    Ok(out)
}

fn canonical_owner_plan_to_py<'py>(
    py: Python<'py>,
    plan: &CanonicalOwnerPlan,
    chromosome_id: u32,
    window_start: i64,
    chromosome_origin: i64,
    window_size: i64,
    step: i64,
) -> PyResult<Bound<'py, PyDict>> {
    let first = plan
        .pairs
        .iter()
        .map(|pair| pair.0 as i64)
        .collect::<Vec<_>>();
    let second = plan
        .pairs
        .iter()
        .map(|pair| pair.1 as i64)
        .collect::<Vec<_>>();
    let out = PyDict::new(py);
    out.set_item("first", PyArray1::from_vec(py, first))?;
    out.set_item("second", PyArray1::from_vec(py, second))?;
    out.set_item(
        "pair_window_offsets",
        PyArray1::from_vec(
            py,
            plan.pair_window_offsets
                .iter()
                .map(|&value| value as i64)
                .collect(),
        ),
    )?;
    out.set_item(
        "pair_window_ids",
        PyArray1::from_vec(py, plan.pair_window_ids.clone()),
    )?;
    let current_grid =
        (i128::from(window_start) - i128::from(chromosome_origin)) / i128::from(step);
    let current_window_id =
        stable_window_id(chromosome_id, current_grid).map_err(PyValueError::new_err)?;
    out.set_item("current_window_id", current_window_id)?;
    out.set_item("chromosome_id", chromosome_id)?;
    out.set_item("window_start", window_start)?;
    out.set_item("chromosome_origin", chromosome_origin)?;
    out.set_item("window_size", window_size)?;
    out.set_item("step", step)?;
    out.set_item("pairs_raw_window", plan.pairs_raw_window)?;
    out.set_item("pairs_owner_skipped", plan.pairs_owner_skipped)?;
    out.set_item("pairs_unique_screened", plan.pairs_unique_screened)?;
    Ok(out)
}

fn add_canonical_screen_counts<'py>(
    out: &Bound<'py, PyDict>,
    plan: &CanonicalOwnerPlan,
    pairs_valid_scored: usize,
) -> PyResult<()> {
    out.set_item("pairs_raw_window", plan.pairs_raw_window)?;
    out.set_item("pairs_owner_skipped", plan.pairs_owner_skipped)?;
    out.set_item("pairs_unique_screened", plan.pairs_unique_screened)?;
    out.set_item("pairs_valid_scored", pairs_valid_scored)?;
    Ok(())
}

#[pyfunction(name = "garfield_dosage_canonical_owner_plan")]
#[pyo3(signature = (positions, chromosome_id, window_start, chromosome_origin, window_size, step))]
pub fn garfield_dosage_canonical_owner_plan_py<'py>(
    py: Python<'py>,
    positions: PyReadonlyArray1<'py, i64>,
    chromosome_id: u32,
    window_start: i64,
    chromosome_origin: i64,
    window_size: i64,
    step: i64,
) -> PyResult<Bound<'py, PyDict>> {
    let position_vec = positions.as_array().iter().copied().collect::<Vec<_>>();
    let plan = canonical_owner_plan(
        &position_vec,
        chromosome_id,
        window_start,
        chromosome_origin,
        window_size,
        step,
    )
    .map_err(PyValueError::new_err)?;
    canonical_owner_plan_to_py(
        py,
        &plan,
        chromosome_id,
        window_start,
        chromosome_origin,
        window_size,
        step,
    )
}

#[pyfunction(name = "garfield_dosage_lm_screen_scan_canonical_owner")]
#[pyo3(signature = (genotypes, y, positions, chromosome_id, window_start, chromosome_origin, window_size, step, top_k=100, threads=0, covariates=None))]
pub fn garfield_dosage_lm_screen_scan_canonical_owner_py<'py>(
    py: Python<'py>,
    genotypes: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    positions: PyReadonlyArray1<'py, i64>,
    chromosome_id: u32,
    window_start: i64,
    chromosome_origin: i64,
    window_size: i64,
    step: i64,
    top_k: usize,
    threads: usize,
    covariates: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let shape = genotypes.shape();
    if shape.len() != 2 || shape[0] != positions.shape()[0] {
        return Err(PyValueError::new_err(format!(
            "genotypes must be marker-major with one position per marker; got shape={shape:?}, positions={}",
            positions.shape()[0]
        )));
    }
    let position_vec = positions.as_array().iter().copied().collect::<Vec<_>>();
    let plan = canonical_owner_plan(
        &position_vec,
        chromosome_id,
        window_start,
        chromosome_origin,
        window_size,
        step,
    )
    .map_err(PyValueError::new_err)?;
    let genotype_vec = array2_to_vec(&genotypes);
    let y_vec = array1_to_vec(&y);
    let (covariate_vec, n_covariates) = if let Some(covariates) = covariates {
        let cov_shape = covariates.shape();
        if cov_shape.len() != 2 || cov_shape[0] != y_vec.len() {
            return Err(PyValueError::new_err(format!(
                "covariates must have shape (n_samples, n_covariates); got {:?}",
                cov_shape
            )));
        }
        (array2_to_vec(&covariates), cov_shape[1])
    } else {
        (Vec::new(), 0)
    };
    let fixed_columns = build_fixed_columns(shape[1], &covariate_vec, n_covariates)
        .map_err(PyValueError::new_err)?;
    let context = ScreenFixedContext::new(
        &y_vec,
        fixed_columns,
        ScreenMetric {
            n: shape[1],
            baseline_weight: 1.0,
            eigenvectors: Vec::new(),
            correction_weights: Vec::new(),
        },
    )
    .map_err(PyValueError::new_err)?;
    let result = scan_selected_memberships_with_context(
        &context,
        &genotype_vec,
        shape[0],
        &y_vec,
        &plan.pairs,
        &plan.pair_window_offsets,
        &plan.pair_window_ids,
        top_k,
        if threads == 0 {
            rayon::current_num_threads().max(1)
        } else {
            threads.max(1)
        },
    )
    .map_err(PyValueError::new_err)?;
    let pairs_valid_scored = result.pairs_evaluated;
    let out = membership_screen_result_to_py(py, result, "lm-canonical-owner")?;
    add_canonical_screen_counts(&out, &plan, pairs_valid_scored)?;
    out.set_item(
        "owner_first",
        PyArray1::from_vec(py, plan.pairs.iter().map(|pair| pair.0 as i64).collect()),
    )?;
    out.set_item(
        "owner_second",
        PyArray1::from_vec(py, plan.pairs.iter().map(|pair| pair.1 as i64).collect()),
    )?;
    out.set_item(
        "pair_window_offsets",
        PyArray1::from_vec(
            py,
            plan.pair_window_offsets
                .iter()
                .map(|&value| value as i64)
                .collect(),
        ),
    )?;
    out.set_item(
        "pair_window_ids",
        PyArray1::from_vec(py, plan.pair_window_ids),
    )?;
    Ok(out)
}

/// Python-facing streaming global Top-K merge for canonical pair candidates.
/// The key is an opaque globally canonical pair id supplied by the caller.
#[pyclass(name = "GarfieldDosageCanonicalTopK")]
pub struct GarfieldDosageCanonicalTopK {
    inner: CanonicalGlobalTopK,
    enforce_unique: bool,
}

#[pymethods]
impl GarfieldDosageCanonicalTopK {
    #[new]
    #[pyo3(signature = (top_k, enforce_unique=true))]
    fn new(top_k: usize, enforce_unique: bool) -> Self {
        Self {
            inner: CanonicalGlobalTopK::new(top_k),
            enforce_unique,
        }
    }

    #[pyo3(signature = (keys, scores))]
    fn extend<'py>(
        &mut self,
        keys: PyReadonlyArray1<'py, i64>,
        scores: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<()> {
        let key_values = keys.as_array();
        let score_values = scores.as_array();
        if key_values.len() != score_values.len() {
            return Err(PyValueError::new_err(format!(
                "keys and scores lengths differ: {} vs {}",
                key_values.len(),
                score_values.len()
            )));
        }
        for (&key, &score) in key_values.iter().zip(score_values.iter()) {
            if key < 0 {
                return Err(PyValueError::new_err(
                    "global pair keys must be non-negative",
                ));
            }
            if !score.is_finite() {
                return Err(PyValueError::new_err("global pair scores must be finite"));
            }
            self.inner.push(key as u64, score);
        }
        Ok(())
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if self.enforce_unique {
            debug_assert_eq!(self.inner.global_duplicate_pairs, 0);
        }
        let candidates = self.inner.clone().into_sorted();
        let keys = candidates
            .iter()
            .map(|entry| entry.0 as i64)
            .collect::<Vec<_>>();
        let scores = candidates.iter().map(|entry| entry.1).collect::<Vec<_>>();
        let out = PyDict::new(py);
        out.set_item("key", PyArray1::from_vec(py, keys))?;
        out.set_item("score", PyArray1::from_vec(py, scores))?;
        out.set_item(
            "candidates_before_global_topk",
            self.inner.candidates_before_global_topk,
        )?;
        out.set_item("candidates_retained", candidates.len())?;
        out.set_item("global_duplicate_pairs", self.inner.global_duplicate_pairs)?;
        Ok(out)
    }

    #[getter]
    fn candidates_before_global_topk(&self) -> usize {
        self.inner.candidates_before_global_topk
    }

    #[getter]
    fn global_duplicate_pairs(&self) -> usize {
        self.inner.global_duplicate_pairs
    }
}

#[pyfunction(name = "garfield_dosage_lm_screen_scan")]
#[pyo3(signature = (genotypes, y, top_k=100, threads=0, covariates=None))]
pub fn garfield_dosage_lm_screen_scan_py<'py>(
    py: Python<'py>,
    genotypes: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    top_k: usize,
    threads: usize,
    covariates: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let shape = genotypes.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err(
            "genotypes must be a 2D marker-major array",
        ));
    }
    let n_markers = shape[0];
    let n_samples = shape[1];
    let y_vec = array1_to_vec(&y);
    let (covariate_vec, n_covariates) = if let Some(covariates) = covariates {
        let cov_shape = covariates.shape();
        if cov_shape.len() != 2 || cov_shape[0] != n_samples {
            return Err(PyValueError::new_err(format!(
                "covariates must have shape (n_samples, n_covariates); got {:?}",
                cov_shape
            )));
        }
        (array2_to_vec(&covariates), cov_shape[1])
    } else {
        (Vec::new(), 0)
    };
    let genotype_vec = array2_to_vec(&genotypes);
    let effective_threads = if threads == 0 {
        rayon::current_num_threads().max(1)
    } else {
        threads
    };
    let result = scan_lm_screen(
        &genotype_vec,
        n_markers,
        n_samples,
        &y_vec,
        top_k,
        effective_threads,
        &covariate_vec,
        n_covariates,
    )
    .map_err(PyValueError::new_err)?;
    screen_result_to_py(py, result, "lm")
}

#[pyfunction(name = "garfield_dosage_lm_screen_scan_selected")]
#[pyo3(signature = (genotypes, y, pairs, top_k=100, threads=0, covariates=None))]
pub fn garfield_dosage_lm_screen_scan_selected_py<'py>(
    py: Python<'py>,
    genotypes: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    pairs: PyReadonlyArray2<'py, i64>,
    top_k: usize,
    threads: usize,
    covariates: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let shape = genotypes.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err(
            "genotypes must be a 2D marker-major array",
        ));
    }
    let n_markers = shape[0];
    let n_samples = shape[1];
    let y_vec = array1_to_vec(&y);
    let (covariate_vec, n_covariates) = if let Some(covariates) = covariates {
        let cov_shape = covariates.shape();
        if cov_shape.len() != 2 || cov_shape[0] != n_samples {
            return Err(PyValueError::new_err(format!(
                "covariates must have shape (n_samples, n_covariates); got {:?}",
                cov_shape
            )));
        }
        (array2_to_vec(&covariates), cov_shape[1])
    } else {
        (Vec::new(), 0)
    };
    let pair_vec = array2_to_pairs(&pairs)?;
    let genotype_vec = array2_to_vec(&genotypes);
    let effective_threads = if threads == 0 {
        rayon::current_num_threads().max(1)
    } else {
        threads
    };
    let result = scan_lm_screen_selected(
        &genotype_vec,
        n_markers,
        n_samples,
        &y_vec,
        &pair_vec,
        top_k,
        effective_threads,
        &covariate_vec,
        n_covariates,
    )
    .map_err(PyValueError::new_err)?;
    screen_result_to_py(py, result, "lm-selected")
}

#[pyfunction(name = "garfield_dosage_lm_screen_scan_selected_grouped")]
#[pyo3(signature = (genotypes, y, pairs, groups, top_k=100, threads=0, covariates=None))]
pub fn garfield_dosage_lm_screen_scan_selected_grouped_py<'py>(
    py: Python<'py>,
    genotypes: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    pairs: PyReadonlyArray2<'py, i64>,
    groups: PyReadonlyArray1<'py, u8>,
    top_k: usize,
    threads: usize,
    covariates: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let shape = genotypes.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err(
            "genotypes must be a 2D marker-major array",
        ));
    }
    let n_markers = shape[0];
    let n_samples = shape[1];
    let y_vec = array1_to_vec(&y);
    let pair_vec = array2_to_pairs(&pairs)?;
    let group_vec = array1_to_u8(&groups);
    let (covariate_vec, n_covariates) = if let Some(covariates) = covariates {
        let cov_shape = covariates.shape();
        if cov_shape.len() != 2 || cov_shape[0] != n_samples {
            return Err(PyValueError::new_err(format!(
                "covariates must have shape (n_samples, n_covariates); got {:?}",
                cov_shape
            )));
        }
        (array2_to_vec(&covariates), cov_shape[1])
    } else {
        (Vec::new(), 0)
    };
    let genotype_vec = array2_to_vec(&genotypes);
    let effective_threads = if threads == 0 {
        rayon::current_num_threads().max(1)
    } else {
        threads
    };
    let result = scan_lm_screen_selected_grouped(
        &genotype_vec,
        n_markers,
        n_samples,
        &y_vec,
        &pair_vec,
        &group_vec,
        top_k,
        effective_threads,
        &covariate_vec,
        n_covariates,
    )
    .map_err(PyValueError::new_err)?;
    grouped_screen_result_to_py(py, result, "lm-selected-grouped")
}

#[pyfunction(name = "garfield_dosage_lowrank_grm_screen_scan")]
#[pyo3(signature = (genotypes, y, grm, sigma_g2, sigma_e2, rank=32, top_k=100, threads=0, covariates=None))]
pub fn garfield_dosage_lowrank_grm_screen_scan_py<'py>(
    py: Python<'py>,
    genotypes: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    grm: PyReadonlyArray2<'py, f64>,
    sigma_g2: f64,
    sigma_e2: f64,
    rank: usize,
    top_k: usize,
    threads: usize,
    covariates: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Bound<'py, PyDict>> {
    let shape = genotypes.shape();
    if shape.len() != 2 {
        return Err(PyValueError::new_err(
            "genotypes must be a 2D marker-major array",
        ));
    }
    let n_markers = shape[0];
    let n_samples = shape[1];
    let grm_shape = grm.shape();
    if grm_shape.len() != 2 || grm_shape[0] != n_samples || grm_shape[1] != n_samples {
        return Err(PyValueError::new_err(format!(
            "grm must have shape ({n_samples}, {n_samples}); got {:?}",
            grm_shape
        )));
    }
    let y_vec = array1_to_vec(&y);
    let genotype_vec = array2_to_vec(&genotypes);
    let grm_vec = array2_to_vec(&grm);
    let (covariate_vec, n_covariates) = if let Some(covariates) = covariates {
        let cov_shape = covariates.shape();
        if cov_shape.len() != 2 || cov_shape[0] != n_samples {
            return Err(PyValueError::new_err(format!(
                "covariates must have shape (n_samples, n_covariates); got {:?}",
                cov_shape
            )));
        }
        (array2_to_vec(&covariates), cov_shape[1])
    } else {
        (Vec::new(), 0)
    };
    let effective_threads = if threads == 0 {
        rayon::current_num_threads().max(1)
    } else {
        threads
    };
    let result = scan_lowrank_grm_screen(
        &genotype_vec,
        n_markers,
        n_samples,
        &y_vec,
        &grm_vec,
        sigma_g2,
        sigma_e2,
        rank,
        top_k,
        effective_threads,
        &covariate_vec,
        n_covariates,
    )
    .map_err(PyValueError::new_err)?;
    screen_result_to_py(py, result, "lowrank-grm")
}

/// Choose a low-rank GRM screen rank from a genotype/pair pilot.
///
/// The selector compares residualized interaction geometry against the same
/// covariance represented at full rank.  Score/rank agreement is returned as
/// an audit only; it does not select a rank from the observed phenotype.  The
/// chosen rank can therefore be fixed and recorded in a matched null
/// calibration artifact before a genome-wide scan.
#[pyfunction(name = "garfield_dosage_lowrank_choose_rank")]
#[pyo3(signature = (pilot_genotypes, y, grm, sigma_g2, sigma_e2, pilot_pairs, covariates=None, candidate_ranks=None, geometry_rtol=0.02, geometry_max_rtol=0.10, score_top_k=100))]
pub fn garfield_dosage_lowrank_choose_rank_py<'py>(
    py: Python<'py>,
    pilot_genotypes: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    grm: PyReadonlyArray2<'py, f64>,
    sigma_g2: f64,
    sigma_e2: f64,
    pilot_pairs: PyReadonlyArray2<'py, i64>,
    covariates: Option<PyReadonlyArray2<'py, f64>>,
    candidate_ranks: Option<Vec<usize>>,
    geometry_rtol: f64,
    geometry_max_rtol: f64,
    score_top_k: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let genotype_shape = pilot_genotypes.shape();
    if genotype_shape.len() != 2 {
        return Err(PyValueError::new_err(
            "pilot_genotypes must be a 2D marker-major array",
        ));
    }
    let y_vec = array1_to_vec(&y);
    let n_markers = genotype_shape[0];
    let n_samples = genotype_shape[1];
    let grm_shape = grm.shape();
    if grm_shape.len() != 2 || grm_shape[0] != n_samples || grm_shape[1] != n_samples {
        return Err(PyValueError::new_err(format!(
            "grm must have shape ({n_samples}, {n_samples}); got {:?}",
            grm_shape
        )));
    }
    let pilot_pairs_vec = array2_to_pairs(&pilot_pairs)?;
    let genotype_vec = array2_to_vec(&pilot_genotypes);
    let grm_vec = array2_to_vec(&grm);
    let (covariate_vec, n_covariates) = if let Some(covariates) = covariates {
        let cov_shape = covariates.shape();
        if cov_shape.len() != 2 || cov_shape[0] != n_samples {
            return Err(PyValueError::new_err(format!(
                "covariates must have shape (n_samples, n_covariates); got {:?}",
                cov_shape
            )));
        }
        (array2_to_vec(&covariates), cov_shape[1])
    } else {
        (Vec::new(), 0)
    };
    let report = choose_adaptive_rank(
        &genotype_vec,
        n_markers,
        n_samples,
        &y_vec,
        &grm_vec,
        sigma_g2,
        sigma_e2,
        &pilot_pairs_vec,
        &covariate_vec,
        n_covariates,
        candidate_ranks.unwrap_or_else(|| ADAPTIVE_SCREEN_RANKS.to_vec()),
        geometry_rtol,
        geometry_max_rtol,
        score_top_k,
    )
    .map_err(PyValueError::new_err)?;
    adaptive_rank_report_to_py(py, &report)
}

/// Reusable low-rank GRM screen context for window scans.
///
/// The ordinary low-rank function intentionally has a simple, stateless API,
/// but rebuilding the GRM eigensystem for every overlapping window would make
/// a genome-wide screen impractical.  This opt-in context constructs the
/// metric and fixed-effect projection once, then reuses it for each window.
#[pyclass(name = "GarfieldDosageLowrankScreenContext")]
pub struct GarfieldDosageLowrankScreenContext {
    context: ScreenFixedContext,
    y: Vec<f64>,
    adaptive_report: Option<AdaptiveRankReport>,
}

#[pymethods]
impl GarfieldDosageLowrankScreenContext {
    #[new]
    #[pyo3(signature = (y, grm, sigma_g2, sigma_e2, rank=32, covariates=None))]
    fn new<'py>(
        y: PyReadonlyArray1<'py, f64>,
        grm: PyReadonlyArray2<'py, f64>,
        sigma_g2: f64,
        sigma_e2: f64,
        rank: usize,
        covariates: Option<PyReadonlyArray2<'py, f64>>,
    ) -> PyResult<Self> {
        let y_vec = array1_to_vec(&y);
        let n_samples = y_vec.len();
        let grm_shape = grm.shape();
        if grm_shape.len() != 2 || grm_shape[0] != n_samples || grm_shape[1] != n_samples {
            return Err(PyValueError::new_err(format!(
                "grm must have shape ({n_samples}, {n_samples}); got {:?}",
                grm_shape
            )));
        }
        let grm_vec = array2_to_vec(&grm);
        let (covariate_vec, n_covariates) = if let Some(covariates) = covariates {
            let cov_shape = covariates.shape();
            if cov_shape.len() != 2 || cov_shape[0] != n_samples {
                return Err(PyValueError::new_err(format!(
                    "covariates must have shape (n_samples, n_covariates); got {:?}",
                    cov_shape
                )));
            }
            (array2_to_vec(&covariates), cov_shape[1])
        } else {
            (Vec::new(), 0)
        };
        if rank == 0 {
            return Err(PyValueError::new_err("rank must be positive"));
        }
        let fixed_columns = build_fixed_columns(n_samples, &covariate_vec, n_covariates)
            .map_err(PyValueError::new_err)?;
        let metric = build_lowrank_metric(&grm_vec, n_samples, sigma_g2, sigma_e2, rank)
            .map_err(PyValueError::new_err)?;
        let context = ScreenFixedContext::new(&y_vec, fixed_columns, metric)
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            context,
            y: y_vec,
            adaptive_report: None,
        })
    }

    /// Construct a context using the adaptive geometry pilot.  The pilot is
    /// deliberately explicit: callers must provide representative marker
    /// pairs, and the resulting rank/report should be persisted with the
    /// calibration metadata before scanning the rest of the genome.
    #[staticmethod]
    #[pyo3(signature = (pilot_genotypes, y, grm, sigma_g2, sigma_e2, pilot_pairs, covariates=None, candidate_ranks=None, geometry_rtol=0.02, geometry_max_rtol=0.10, score_top_k=100))]
    fn auto<'py>(
        pilot_genotypes: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
        grm: PyReadonlyArray2<'py, f64>,
        sigma_g2: f64,
        sigma_e2: f64,
        pilot_pairs: PyReadonlyArray2<'py, i64>,
        covariates: Option<PyReadonlyArray2<'py, f64>>,
        candidate_ranks: Option<Vec<usize>>,
        geometry_rtol: f64,
        geometry_max_rtol: f64,
        score_top_k: usize,
    ) -> PyResult<Self> {
        let genotype_shape = pilot_genotypes.shape();
        if genotype_shape.len() != 2 {
            return Err(PyValueError::new_err(
                "pilot_genotypes must be a 2D marker-major array",
            ));
        }
        let y_vec = array1_to_vec(&y);
        let n_markers = genotype_shape[0];
        let n_samples = genotype_shape[1];
        let grm_shape = grm.shape();
        if grm_shape.len() != 2 || grm_shape[0] != n_samples || grm_shape[1] != n_samples {
            return Err(PyValueError::new_err(format!(
                "grm must have shape ({n_samples}, {n_samples}); got {:?}",
                grm_shape
            )));
        }
        let pilot_pairs_vec = array2_to_pairs(&pilot_pairs)?;
        let genotype_vec = array2_to_vec(&pilot_genotypes);
        let grm_vec = array2_to_vec(&grm);
        let (covariate_vec, n_covariates) = if let Some(covariates) = covariates {
            let cov_shape = covariates.shape();
            if cov_shape.len() != 2 || cov_shape[0] != n_samples {
                return Err(PyValueError::new_err(format!(
                    "covariates must have shape (n_samples, n_covariates); got {:?}",
                    cov_shape
                )));
            }
            (array2_to_vec(&covariates), cov_shape[1])
        } else {
            (Vec::new(), 0)
        };
        let spectrum = ScreenSpectrum::from_grm(&grm_vec, n_samples, sigma_g2, sigma_e2)
            .map_err(PyValueError::new_err)?;
        let report = choose_adaptive_rank_with_spectrum(
            &genotype_vec,
            n_markers,
            n_samples,
            &y_vec,
            &pilot_pairs_vec,
            &covariate_vec,
            n_covariates,
            candidate_ranks.unwrap_or_else(|| ADAPTIVE_SCREEN_RANKS.to_vec()),
            geometry_rtol,
            geometry_max_rtol,
            score_top_k,
            &spectrum,
            sigma_g2,
            sigma_e2,
        )
        .map_err(PyValueError::new_err)?;
        let fixed_columns = build_fixed_columns(n_samples, &covariate_vec, n_covariates)
            .map_err(PyValueError::new_err)?;
        let metric = spectrum
            .metric(report.selected_rank)
            .map_err(PyValueError::new_err)?;
        let context = ScreenFixedContext::new(&y_vec, fixed_columns, metric)
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            context,
            y: y_vec,
            adaptive_report: Some(report),
        })
    }

    /// Return the rank-selection audit.  Explicit-rank contexts return a
    /// compact explicit-mode record so callers can serialize one stable shape.
    fn rank_diagnostics<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if let Some(report) = &self.adaptive_report {
            adaptive_rank_report_to_py(py, report)
        } else {
            let out = PyDict::new(py);
            out.set_item("mode", "explicit")?;
            out.set_item("selected_rank", self.context.metric.rank())?;
            out.set_item("full_rank", self.context.n)?;
            out.set_item("baseline_weight", self.context.metric.baseline_weight)?;
            Ok(out)
        }
    }

    #[pyo3(signature = (genotypes, top_k=100, threads=0))]
    fn scan<'py>(
        &self,
        py: Python<'py>,
        genotypes: PyReadonlyArray2<'py, f64>,
        top_k: usize,
        threads: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let shape = genotypes.shape();
        if shape.len() != 2 {
            return Err(PyValueError::new_err(
                "genotypes must be a 2D marker-major array",
            ));
        }
        let n_markers = shape[0];
        let n_samples = shape[1];
        if n_samples != self.y.len() {
            return Err(PyValueError::new_err(format!(
                "genotypes has {n_samples} samples but context expects {}",
                self.y.len()
            )));
        }
        let genotype_vec = array2_to_vec(&genotypes);
        validate_dense_inputs(&genotype_vec, n_markers, n_samples, &self.y)
            .map_err(PyValueError::new_err)?;
        let effective_threads = if threads == 0 {
            rayon::current_num_threads().max(1)
        } else {
            threads
        };
        let result = scan_with_context(
            &self.context,
            &genotype_vec,
            n_markers,
            &self.y,
            top_k,
            effective_threads,
        )
        .map_err(PyValueError::new_err)?;
        screen_result_to_py(py, result, "lowrank-grm-context")
    }

    #[pyo3(signature = (genotypes, pairs, top_k=100, threads=0))]
    fn scan_selected<'py>(
        &self,
        py: Python<'py>,
        genotypes: PyReadonlyArray2<'py, f64>,
        pairs: PyReadonlyArray2<'py, i64>,
        top_k: usize,
        threads: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let shape = genotypes.shape();
        if shape.len() != 2 {
            return Err(PyValueError::new_err(
                "genotypes must be a 2D marker-major array",
            ));
        }
        let n_markers = shape[0];
        let n_samples = shape[1];
        if n_samples != self.y.len() {
            return Err(PyValueError::new_err(format!(
                "genotypes has {n_samples} samples but context expects {}",
                self.y.len()
            )));
        }
        let pair_vec = array2_to_pairs(&pairs)?;
        let genotype_vec = array2_to_vec(&genotypes);
        validate_dense_inputs(&genotype_vec, n_markers, n_samples, &self.y)
            .map_err(PyValueError::new_err)?;
        let effective_threads = if threads == 0 {
            rayon::current_num_threads().max(1)
        } else {
            threads
        };
        let result = scan_selected_with_context(
            &self.context,
            &genotype_vec,
            n_markers,
            &self.y,
            &pair_vec,
            top_k,
            effective_threads,
        )
        .map_err(PyValueError::new_err)?;
        screen_result_to_py(py, result, "lowrank-grm-context-selected")
    }

    /// Score only the pairs owned by this half-open window.  The plan is
    /// computed in Rust; Python callers only provide marker positions and
    /// window-grid metadata.  Membership arrays describe later windows that
    /// contain the same pair and are returned for local-Top-K compatibility.
    #[pyo3(signature = (genotypes, positions, chromosome_id, window_start, chromosome_origin, window_size, step, top_k=100, threads=0))]
    fn scan_canonical_owner<'py>(
        &self,
        py: Python<'py>,
        genotypes: PyReadonlyArray2<'py, f64>,
        positions: PyReadonlyArray1<'py, i64>,
        chromosome_id: u32,
        window_start: i64,
        chromosome_origin: i64,
        window_size: i64,
        step: i64,
        top_k: usize,
        threads: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let shape = genotypes.shape();
        if shape.len() != 2 || shape[0] != positions.shape()[0] {
            return Err(PyValueError::new_err(format!(
                "genotypes must be marker-major with one position per marker; got shape={shape:?}, positions={}",
                positions.shape()[0]
            )));
        }
        let position_vec = positions.as_array().iter().copied().collect::<Vec<_>>();
        let plan = canonical_owner_plan(
            &position_vec,
            chromosome_id,
            window_start,
            chromosome_origin,
            window_size,
            step,
        )
        .map_err(PyValueError::new_err)?;
        let genotype_vec = array2_to_vec(&genotypes);
        validate_dense_inputs(&genotype_vec, shape[0], shape[1], &self.y)
            .map_err(PyValueError::new_err)?;
        let result = scan_selected_memberships_with_context(
            &self.context,
            &genotype_vec,
            shape[0],
            &self.y,
            &plan.pairs,
            &plan.pair_window_offsets,
            &plan.pair_window_ids,
            top_k,
            if threads == 0 {
                rayon::current_num_threads().max(1)
            } else {
                threads.max(1)
            },
        )
        .map_err(PyValueError::new_err)?;
        let pairs_valid_scored = result.pairs_evaluated;
        let out =
            membership_screen_result_to_py(py, result, "lowrank-grm-context-canonical-owner")?;
        add_canonical_screen_counts(&out, &plan, pairs_valid_scored)?;
        out.set_item(
            "owner_first",
            PyArray1::from_vec(py, plan.pairs.iter().map(|pair| pair.0 as i64).collect()),
        )?;
        out.set_item(
            "owner_second",
            PyArray1::from_vec(py, plan.pairs.iter().map(|pair| pair.1 as i64).collect()),
        )?;
        out.set_item(
            "pair_window_offsets",
            PyArray1::from_vec(
                py,
                plan.pair_window_offsets
                    .iter()
                    .map(|&value| value as i64)
                    .collect(),
            ),
        )?;
        out.set_item(
            "pair_window_ids",
            PyArray1::from_vec(py, plan.pair_window_ids),
        )?;
        Ok(out)
    }

    /// Score selected pairs once and return independent Top-K lists for two
    /// overlapping windows.  `groups` is a bit mask per pair: bit 0 sends the
    /// pair to `*_a`, bit 1 sends it to `*_b`.
    #[pyo3(signature = (genotypes, pairs, groups, top_k=100, threads=0))]
    fn scan_selected_grouped<'py>(
        &self,
        py: Python<'py>,
        genotypes: PyReadonlyArray2<'py, f64>,
        pairs: PyReadonlyArray2<'py, i64>,
        groups: PyReadonlyArray1<'py, u8>,
        top_k: usize,
        threads: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let shape = genotypes.shape();
        if shape.len() != 2 {
            return Err(PyValueError::new_err(
                "genotypes must be a 2D marker-major array",
            ));
        }
        let n_markers = shape[0];
        let n_samples = shape[1];
        if n_samples != self.y.len() {
            return Err(PyValueError::new_err(format!(
                "genotypes has {n_samples} samples but context expects {}",
                self.y.len()
            )));
        }
        let pair_vec = array2_to_pairs(&pairs)?;
        let group_vec = array1_to_u8(&groups);
        let genotype_vec = array2_to_vec(&genotypes);
        validate_dense_inputs(&genotype_vec, n_markers, n_samples, &self.y)
            .map_err(PyValueError::new_err)?;
        let effective_threads = if threads == 0 {
            rayon::current_num_threads().max(1)
        } else {
            threads
        };
        let result = scan_selected_grouped_with_context(
            &self.context,
            &genotype_vec,
            n_markers,
            &self.y,
            &pair_vec,
            &group_vec,
            top_k,
            effective_threads,
        )
        .map_err(PyValueError::new_err)?;
        grouped_screen_result_to_py(py, result, "lowrank-grm-context-selected-grouped")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lm_screen_finds_the_known_interaction_and_evaluates_all_pairs() {
        let genotypes = vec![
            0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
        ];
        let y = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let scan = scan_lm_screen(&genotypes, 4, 8, &y, 6, 1, &[], 0).unwrap();
        assert_eq!(scan.pairs_evaluated, 6);
        assert_eq!(scan.pairs_skipped, 0);
        let causal = scan
            .candidates
            .iter()
            .find(|candidate| (candidate.first, candidate.second) == (0, 1))
            .expect("known interaction should be present in the full Top-K");
        assert!(causal.score.is_finite() && causal.score > 0.0);
    }

    #[test]
    fn selected_screen_evaluates_only_requested_pairs() {
        let genotypes = vec![
            0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
        ];
        let y = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let pairs = vec![(0, 1), (2, 3)];
        let scan = scan_lm_screen_selected(&genotypes, 4, 8, &y, &pairs, 2, 1, &[], 0).unwrap();
        assert_eq!(scan.pairs_evaluated + scan.pairs_skipped, pairs.len());
        assert!(scan
            .candidates
            .iter()
            .all(|candidate| pairs.contains(&(candidate.first, candidate.second))));
    }

    #[test]
    fn grouped_selected_screen_evaluates_each_pair_once_for_multiple_windows() {
        let genotypes = vec![
            0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0,
        ];
        let y = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let pairs = vec![(0, 1), (2, 3)];
        let groups = vec![3_u8, 1_u8];
        let result =
            scan_lm_screen_selected_grouped(&genotypes, 4, 8, &y, &pairs, &groups, 2, 1, &[], 0)
                .unwrap();
        assert_eq!(result.pairs_evaluated + result.pairs_skipped, pairs.len());
        assert!(result
            .first
            .iter()
            .all(|candidate| pairs.contains(&(candidate.first, candidate.second))));
        assert!(result
            .second
            .iter()
            .all(|candidate| pairs.contains(&(candidate.first, candidate.second))));
        assert!(result
            .second
            .iter()
            .any(|candidate| (candidate.first, candidate.second) == pairs[0]));
        assert!(!result
            .second
            .iter()
            .any(|candidate| (candidate.first, candidate.second) == pairs[1]));
    }

    #[test]
    fn lowrank_rank_zero_matches_lm_screen_up_to_scalar_weight() {
        let genotypes = vec![
            0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0,
        ];
        let y = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let lm = scan_lm_screen(&genotypes, 3, 8, &y, 3, 1, &[], 0).unwrap();
        let grm = vec![
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let lowrank =
            scan_lowrank_grm_screen(&genotypes, 3, 8, &y, &grm, 0.0, 1.0, 0, 3, 1, &[], 0).unwrap();
        assert_eq!(
            lm.candidates
                .iter()
                .map(|candidate| (candidate.first, candidate.second))
                .collect::<Vec<_>>(),
            lowrank
                .candidates
                .iter()
                .map(|candidate| (candidate.first, candidate.second))
                .collect::<Vec<_>>()
        );
        for (left, right) in lm.candidates.iter().zip(lowrank.candidates.iter()) {
            assert!((left.score - right.score).abs() < 1.0e-10);
        }
    }

    #[test]
    fn full_rank_lowrank_metric_matches_direct_inverse_covariance() {
        let n = 8;
        let mut grm = vec![0.0; n * n];
        for row in 0..n {
            for column in 0..n {
                grm[row * n + column] = if row == column {
                    1.0
                } else if (row + column) % 3 == 0 {
                    0.08
                } else {
                    0.0
                };
            }
        }
        let metric = build_lowrank_metric(&grm, n, 0.4, 0.6, n).unwrap();
        let covariance = DMatrix::from_row_slice(n, n, &grm) * 0.4 + DMatrix::identity(n, n) * 0.6;
        let inverse = covariance
            .try_inverse()
            .expect("positive-definite covariance");
        let left = (0..n)
            .map(|row| (row as f64 - 2.5) * 0.7)
            .collect::<Vec<_>>();
        let right = (0..n)
            .map(|row| (row as f64 + 1.0).sin())
            .collect::<Vec<_>>();
        let left_q = (0..metric.rank())
            .map(|component| {
                dot(
                    &metric.eigenvectors[component * n..(component + 1) * n],
                    &left,
                )
            })
            .collect::<Vec<_>>();
        let right_q = (0..metric.rank())
            .map(|component| {
                dot(
                    &metric.eigenvectors[component * n..(component + 1) * n],
                    &right,
                )
            })
            .collect::<Vec<_>>();
        let direct = (0..n)
            .map(|row| {
                (0..n)
                    .map(|column| left[row] * inverse[(row, column)] * right[column])
                    .sum::<f64>()
            })
            .sum::<f64>();
        let approximated = metric.weighted(dot(&left, &right), &left_q, &right_q);
        assert!((approximated - direct).abs() < 1.0e-10);
    }

    #[test]
    fn adaptive_rank_uses_geometry_and_allows_full_eigenspace_rank() {
        let n = 8;
        let grm = (0..n * n)
            .map(|index| if index / n == index % n { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let genotypes = vec![
            0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, // marker 0
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, // marker 1
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, // marker 2
        ];
        let y = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let pairs = vec![(0, 1), (0, 2), (1, 2)];
        let report = choose_adaptive_rank(
            &genotypes,
            3,
            n,
            &y,
            &grm,
            0.4,
            0.6,
            &pairs,
            &[],
            0,
            vec![n],
            0.01,
            0.10,
            10,
        )
        .expect("adaptive rank should accept a full eigenspace candidate");
        assert_eq!(report.selected_rank, n);
        assert_eq!(report.candidates[0].rank, n);
        assert!(report.candidates[0].geometry_valid_pairs > 0);
        // The eigenspace rank is independent of the regression residual-df
        // condition used when fitting a particular pair.
        assert_eq!(
            build_lowrank_metric(&grm, n, 0.4, 0.6, n).unwrap().rank(),
            n
        );
    }

    #[test]
    fn adaptive_rank_reports_zero_geometry_error_for_scalar_covariance() {
        let n = 8;
        let grm = (0..n * n)
            .map(|index| if index / n == index % n { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let genotypes = vec![
            0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, // marker 0
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, // marker 1
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, // marker 2
        ];
        let y = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let pairs = vec![(0, 1), (0, 2), (1, 2)];
        let report = choose_adaptive_rank(
            &genotypes,
            3,
            n,
            &y,
            &grm,
            0.4,
            0.6,
            &pairs,
            &[],
            0,
            vec![1, 2, 4],
            0.01,
            0.10,
            10,
        )
        .expect("scalar covariance should be geometry-stable");
        assert_eq!(report.selected_rank, 1);
        assert!(report.candidates[0].geometry_p95_relative_error < 1.0e-12);
        assert!(report.candidates[0].geometry_q99_relative_error < 1.0e-12);
        assert!(report.candidates[0].geometry_q999_relative_error < 1.0e-12);
        assert!(report.candidates[0].diagnostic_spectral_tail_c0 < 1.0e-12);
        assert!(report.candidates[0].score_pairs_compared > 0);
        assert!(report.candidates[0].score_spearman.is_finite());
    }

    #[test]
    fn adaptive_rank_uses_rank_specific_omitted_baselines() {
        let n = 6;
        let grm = (0..n * n)
            .map(|index| {
                let row = index / n;
                let column = index % n;
                if row == column {
                    1.0 + row as f64 * 0.1
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let metric_1 = build_lowrank_metric(&grm, n, 0.8, 0.4, 1).unwrap();
        let metric_2 = build_lowrank_metric(&grm, n, 0.8, 0.4, 2).unwrap();
        assert_ne!(
            metric_1.baseline_weight.to_bits(),
            metric_2.baseline_weight.to_bits(),
            "the actual approximation must retain the rank-specific omitted-space baseline"
        );
    }

    #[test]
    fn adaptive_rank_falls_back_to_full_rank_when_candidates_fail() {
        let n = 8;
        let grm = (0..n * n)
            .map(|index| {
                let row = index / n;
                let column = index % n;
                if row == column {
                    1.0 + row as f64 * 0.2
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();
        let genotypes = vec![
            0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, // marker 0
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, // marker 1
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, // marker 2
        ];
        let y = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let pairs = vec![(0, 1), (0, 2), (1, 2)];
        let report = choose_adaptive_rank(
            &genotypes,
            3,
            n,
            &y,
            &grm,
            0.8,
            0.4,
            &pairs,
            &[],
            0,
            vec![1, 2],
            0.0,
            0.0,
            10,
        )
        .expect("adaptive rank should use a safe exact fallback");
        assert_eq!(report.selected_rank, n);
        assert!(report.used_largest_rank_fallback);
        let full = report
            .candidates
            .last()
            .expect("full-rank fallback should be reported");
        assert_eq!(full.rank, n);
        assert!(full.geometry_pass);
        assert!(full.geometry_p95_relative_error.abs() < 1.0e-12);
        assert!(full.geometry_q999_relative_error.abs() < 1.0e-12);
    }

    #[test]
    fn adaptive_rank_rejects_duplicate_pilot_pairs() {
        let n = 4;
        let grm = (0..n * n)
            .map(|index| if index / n == index % n { 1.0 } else { 0.0 })
            .collect::<Vec<_>>();
        let genotypes = vec![
            0.0, 0.0, 1.0, 1.0, // marker 0
            0.0, 1.0, 0.0, 1.0, // marker 1
        ];
        let y = vec![0.0, 1.0, 0.0, 1.0];
        let duplicate_pairs = vec![(0, 1), (0, 1)];
        let error = choose_adaptive_rank(
            &genotypes,
            2,
            n,
            &y,
            &grm,
            0.4,
            0.6,
            &duplicate_pairs,
            &[],
            0,
            vec![1, 2],
            0.01,
            0.10,
            10,
        )
        .expect_err("duplicate pilot pairs must not silently reweight the audit");
        assert!(error.contains("duplicate"));
    }

    #[test]
    fn canonical_owner_plan_uses_minimum_window_and_all_memberships() {
        // W/S = 2.  The current half-open window is [50, 150), and the pair
        // (50, 90) belongs to starts 0 and 50.  It is owned by window 0 and
        // therefore must not be scored again from the current window.
        let plan = canonical_owner_plan(&[50, 90, 120, 140], 3, 50, 0, 100, 50)
            .expect("canonical owner plan should accept a sorted half-open window");
        assert_eq!(plan.pairs_raw_window, 6);
        assert_eq!(plan.pairs_owner_skipped, 1);
        assert_eq!(plan.pairs_unique_screened, 5);
        assert_eq!(plan.pairs[0], (0, 2));
        assert_eq!(plan.pairs[4], (2, 3));
        assert_eq!(plan.pair_window_offsets, vec![0, 1, 2, 3, 4, 6]);
        assert_eq!(
            plan.pair_window_ids,
            vec![
                (3_u64 << 32) | 1,
                (3_u64 << 32) | 1,
                (3_u64 << 32) | 1,
                (3_u64 << 32) | 1,
                (3_u64 << 32) | 1,
                (3_u64 << 32) | 2,
            ]
        );
    }

    #[test]
    fn canonical_owner_plan_supports_four_overlapping_windows() {
        // W/S = 4.  The first pair is contained in starts 0, 25, 50 and 75;
        // the current window starts at 75, so it is owned by an earlier
        // window and is excluded from this owner scan.
        let plan = canonical_owner_plan(&[75, 76, 77, 78], 1, 75, 0, 100, 25)
            .expect("four-way overlap should be supported");
        assert_eq!(plan.pairs_raw_window, 6);
        assert_eq!(plan.pairs_unique_screened, 0);
        assert_eq!(plan.pairs_owner_skipped, 6);

        // At the first grid window the same positions are owned, and every
        // owner pair carries all of its future memberships.
        let first = canonical_owner_plan(&[75, 76, 77], 1, 0, 0, 100, 25)
            .expect("first grid window should be valid");
        assert_eq!(first.pairs_unique_screened, 3);
        assert!(first
            .pair_window_offsets
            .windows(2)
            .all(|window| window[1] - window[0] == 4));
    }

    #[test]
    fn canonical_owner_plan_rejects_unsorted_and_closed_end_positions() {
        assert!(canonical_owner_plan(&[10, 9], 0, 0, 0, 100, 50).is_err());
        assert!(canonical_owner_plan(&[0, 100], 0, 0, 0, 100, 50).is_err());
        assert!(canonical_owner_plan(&[0, 1], 0, 1, 0, 100, 50).is_err());
    }

    #[test]
    fn canonical_owner_plan_handles_nonoverlapping_windows_and_duplicate_positions() {
        let plan = canonical_owner_plan(&[0, 10, 20, 90], 7, 0, 0, 100, 100).unwrap();
        assert_eq!(plan.pairs_raw_window, 6);
        assert_eq!(plan.pairs_owner_skipped, 0);
        assert_eq!(plan.pairs_unique_screened, 6);
        assert!(plan
            .pair_window_offsets
            .windows(2)
            .all(|window| window[1] - window[0] == 1));

        // Pair identity is marker-index based, not coordinate based.  Two
        // markers at one coordinate are still a valid distinct pair.
        let duplicate_positions = canonical_owner_plan(&[10, 10, 20], 7, 0, 0, 100, 100).unwrap();
        assert_eq!(duplicate_positions.pairs, vec![(0, 1), (0, 2), (1, 2)]);
        assert_eq!(duplicate_positions.pairs_unique_screened, 3);
    }

    #[test]
    fn canonical_owner_plan_matches_unique_pairs_for_any_overlap_ratio() {
        let all_positions = [0_i64, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220];
        for step in [100_i64, 50, 25] {
            let mut raw_pairs = 0_usize;
            let mut skipped_pairs = 0_usize;
            let mut expected = HashSet::new();
            let mut owners = HashSet::new();
            for window_start in (0_i64..=200).step_by(step as usize) {
                let global_indices = all_positions
                    .iter()
                    .enumerate()
                    .filter_map(|(index, &position)| {
                        (window_start <= position && position < window_start + 100).then_some(index)
                    })
                    .collect::<Vec<_>>();
                let window_positions = global_indices
                    .iter()
                    .map(|&index| all_positions[index])
                    .collect::<Vec<_>>();
                let plan =
                    canonical_owner_plan(&window_positions, 2, window_start, 0, 100, step).unwrap();
                raw_pairs += plan.pairs_raw_window;
                skipped_pairs += plan.pairs_owner_skipped;
                for left in 0..global_indices.len() {
                    for right in (left + 1)..global_indices.len() {
                        expected.insert((global_indices[left], global_indices[right]));
                    }
                }
                for &(left, right) in &plan.pairs {
                    assert!(owners.insert((global_indices[left], global_indices[right])));
                }
            }
            assert_eq!(owners, expected, "unexpected owner set for step={step}");
            assert_eq!(raw_pairs, skipped_pairs + owners.len());
        }
    }

    #[test]
    fn canonical_global_topk_counts_duplicate_occurrences_and_keeps_best() {
        let mut merge = CanonicalGlobalTopK::new(2);
        merge.push(11, 1.0);
        merge.push(22, 3.0);
        merge.push(11, 4.0);
        merge.push(33, 2.0);
        let candidates_before = merge.candidates_before_global_topk;
        let candidates_retained = merge.clone().into_sorted().len();
        let duplicate_pairs = merge.global_duplicate_pairs;
        let result = merge.into_sorted();
        assert_eq!(result, vec![(11, 4.0), (22, 3.0)]);
        assert_eq!(candidates_before, 4);
        assert_eq!(candidates_retained, 2);
        assert_eq!(duplicate_pairs, 1);

        // Updating a key that is not currently the heap minimum must not
        // leave a stale entry consuming one of the bounded Top-K slots.
        let mut updated = CanonicalGlobalTopK::new(2);
        updated.push(101, 100.0);
        updated.push(202, 90.0);
        updated.push(101, 105.0);
        assert_eq!(updated.into_sorted(), vec![(101, 105.0), (202, 90.0)]);
    }

    #[test]
    fn canonical_memberships_score_owner_pair_once_and_route_to_all_windows() {
        // Marker-major binary genotype columns.  The first four markers span
        // all two-way cells, so every pair has an identifiable interaction
        // geometry under the LM screen.
        let genotypes = vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, // marker 0
            0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, // marker 1
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, // marker 2
            0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, // marker 3
        ];
        let y = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let positions = [50, 90, 120, 140];
        let plan = canonical_owner_plan(&positions, 3, 50, 0, 100, 50).unwrap();
        let fixed_columns = build_fixed_columns(8, &[], 0).unwrap();
        let context = ScreenFixedContext::new(
            &y,
            fixed_columns,
            ScreenMetric {
                n: 8,
                baseline_weight: 1.0,
                eigenvectors: Vec::new(),
                correction_weights: Vec::new(),
            },
        )
        .unwrap();
        let result = scan_selected_memberships_with_context(
            &context,
            &genotypes,
            4,
            &y,
            &plan.pairs,
            &plan.pair_window_offsets,
            &plan.pair_window_ids,
            16,
            1,
        )
        .unwrap();
        assert_eq!(result.pairs_evaluated + result.pairs_skipped, 5);
        assert_eq!(
            result.window_ids,
            vec![(3_u64 << 32) | 1, (3_u64 << 32) | 2]
        );
        // The (120, 140) pair has two memberships and must occur in both
        // independent local Top-K lists, despite being scored only once.
        for candidates in &result.candidates {
            assert!(candidates
                .iter()
                .any(|candidate| (candidate.first, candidate.second) == (2, 3)));
        }
    }

    #[test]
    fn canonical_memberships_reject_pairs_without_a_window_membership() {
        let genotypes = vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, // marker 0
            0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, // marker 1
        ];
        let y = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let fixed_columns = build_fixed_columns(8, &[], 0).unwrap();
        let context = ScreenFixedContext::new(
            &y,
            fixed_columns,
            ScreenMetric {
                n: 8,
                baseline_weight: 1.0,
                eigenvectors: Vec::new(),
                correction_weights: Vec::new(),
            },
        )
        .unwrap();
        let error = scan_selected_memberships_with_context(
            &context,
            &genotypes,
            2,
            &y,
            &[(0, 1)],
            &[0, 0],
            &[],
            4,
            1,
        )
        .expect_err("a pair without a membership must be rejected");
        assert!(error.contains("at least one window membership"));
    }
}
