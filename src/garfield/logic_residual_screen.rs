//! Packed four-cell sufficient statistics for residual screening.
//!
//! This module is intentionally independent from the Logic5 discovery path.
//! It will provide a GRAMMAR-like screen from an already residualized
//! phenotype, while exact GARFIELD/GLS inference remains a separate step.

use super::score::{validate_continuous_y, PackedYSumLookup};
use crate::breader::load_bin01_as_u64_words;
use crate::bstats::{tail_mask, words_for_samples};
use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Cell order is always `[00, 01, 10, 11]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct FourCellSufficientStats {
    pub(crate) counts: [usize; 4],
    pub(crate) sums: [f64; 4],
    pub(crate) n_valid: usize,
}

/// One residual-screen gate derived from four-cell sufficient statistics.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ResidualGateScore {
    pub(crate) gate: ResidualGate,
    pub(crate) score: f64,
    pub(crate) signed_score: f64,
    pub(crate) n_hit: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResidualGate {
    And00,
    And01,
    And10,
    And11,
    Xor,
}

impl ResidualGate {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::And00 => "AND_00",
            Self::And01 => "AND_01",
            Self::And10 => "AND_10",
            Self::And11 => "AND_11",
            Self::Xor => "XOR",
        }
    }
}

/// Proposal policy used by the residual screen. This is a proposal-layer
/// choice; four-cell summaries and downstream exact inference are unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResidualProposalMode {
    /// Historical best-of-four-ANDs plus optional XOR behavior.
    Max5,
    /// Gate-independent between-cell score, followed by post-hoc template
    /// classification.
    Omnibus4Cell,
    /// Separate AND and XOR proposal heaps, then union and deduplicate.
    SplitHeap,
}

impl ResidualProposalMode {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "max5" | "max-5" | "baseline" => Ok(Self::Max5),
            "omnibus" | "4cell" | "4-cell" | "omnibus4cell" => Ok(Self::Omnibus4Cell),
            "split" | "split-heap" | "and-xor" => Ok(Self::SplitHeap),
            other => Err(format!(
                "proposal_mode must be one of max5, omnibus, or split; got {other}"
            )),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Max5 => "max5",
            Self::Omnibus4Cell => "omnibus4cell",
            Self::SplitHeap => "split_heap",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResidualProposalSource {
    Max5,
    Omnibus4Cell,
    And,
    Xor,
}

impl ResidualProposalSource {
    fn as_str(self) -> &'static str {
        match self {
            Self::Max5 => "max5",
            Self::Omnibus4Cell => "omnibus4cell",
            Self::And => "and",
            Self::Xor => "xor",
        }
    }

    fn rank(self) -> u8 {
        match self {
            Self::Max5 | Self::Omnibus4Cell => 0,
            Self::And => 1,
            Self::Xor => 2,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn four_cell_sufficient_stats_packed(
    first: &[u64],
    second: &[u64],
    first_valid: Option<&[u64]>,
    second_valid: Option<&[u64]>,
    residual: &[f64],
    n_samples: usize,
    lookup: &PackedYSumLookup,
) -> Result<FourCellSufficientStats, String> {
    if n_samples == 0 {
        return Err("four-cell residual screen requires n_samples > 0".to_string());
    }
    validate_continuous_y(residual, n_samples, "four-cell residual screen")?;
    validate_packed_cell_rows(first, second, first_valid, second_valid, n_samples)?;
    Ok(four_cell_sufficient_stats_packed_unchecked(
        first,
        second,
        first_valid,
        second_valid,
        n_samples,
        lookup,
    ))
}

#[inline]
fn validate_packed_cell_rows(
    first: &[u64],
    second: &[u64],
    first_valid: Option<&[u64]>,
    second_valid: Option<&[u64]>,
    n_samples: usize,
) -> Result<(), String> {
    let words = words_for_samples(n_samples);
    if first.len() < words || second.len() < words {
        return Err(format!(
            "four-cell residual screen packed row is shorter than {words} words"
        ));
    }
    for (name, row) in [("first_valid", first_valid), ("second_valid", second_valid)] {
        if let Some(row) = row {
            if row.len() < words {
                return Err(format!(
                    "four-cell residual screen {name} row is shorter than {words} words"
                ));
            }
        }
    }
    Ok(())
}

/// Fast inner-loop form of [`four_cell_sufficient_stats_packed`].  The caller
/// must have validated the packed rows, sample count, residual, and lookup.
#[inline]
fn four_cell_sufficient_stats_packed_unchecked(
    first: &[u64],
    second: &[u64],
    first_valid: Option<&[u64]>,
    second_valid: Option<&[u64]>,
    n_samples: usize,
    lookup: &PackedYSumLookup,
) -> FourCellSufficientStats {
    debug_assert!(first.len() >= words_for_samples(n_samples));
    debug_assert!(second.len() >= words_for_samples(n_samples));
    debug_assert!(first_valid
        .map(|row| row.len() >= words_for_samples(n_samples))
        .unwrap_or(true));
    debug_assert!(second_valid
        .map(|row| row.len() >= words_for_samples(n_samples))
        .unwrap_or(true));
    let words = words_for_samples(n_samples);
    let tail = tail_mask(n_samples);
    let mut counts = [0usize; 4];
    let mut sums = [0.0f64; 4];
    for word_idx in 0..words {
        let x = first[word_idx];
        let z = second[word_idx];
        let mut valid = match (first_valid, second_valid) {
            (Some(lhs), Some(rhs)) => lhs[word_idx] & rhs[word_idx],
            (Some(lhs), None) => lhs[word_idx],
            (None, Some(rhs)) => rhs[word_idx],
            (None, None) => u64::MAX,
        };
        if word_idx + 1 == words {
            if let Some(mask) = tail {
                valid &= mask;
            }
        }
        let cells = [
            (!x & !z) & valid,
            (!x & z) & valid,
            (x & !z) & valid,
            (x & z) & valid,
        ];
        for (cell, &mask) in cells.iter().enumerate() {
            counts[cell] = counts[cell].saturating_add(mask.count_ones() as usize);
            sums[cell] += lookup.sum_word(word_idx, mask);
        }
    }
    FourCellSufficientStats {
        counts,
        sums,
        n_valid: counts.iter().sum(),
    }
}

#[inline]
pub(crate) fn derive_residual_gate_scores(
    stats: FourCellSufficientStats,
) -> [ResidualGateScore; 5] {
    let total = stats.sums.iter().sum::<f64>();
    let n_total = stats.n_valid as f64;
    let gate_data = [
        (ResidualGate::And00, stats.counts[0], stats.sums[0]),
        (ResidualGate::And01, stats.counts[1], stats.sums[1]),
        (ResidualGate::And10, stats.counts[2], stats.sums[2]),
        (ResidualGate::And11, stats.counts[3], stats.sums[3]),
        (
            ResidualGate::Xor,
            stats.counts[1].saturating_add(stats.counts[2]),
            stats.sums[1] + stats.sums[2],
        ),
    ];
    std::array::from_fn(|idx| {
        let (gate, n_hit, sum_hit) = gate_data[idx];
        if stats.n_valid == 0 || n_hit == 0 || n_hit >= stats.n_valid {
            return ResidualGateScore {
                gate,
                score: f64::NAN,
                signed_score: f64::NAN,
                n_hit,
            };
        }
        let n_miss = stats.n_valid - n_hit;
        let hit_f = n_hit as f64;
        let miss_f = n_miss as f64;
        let denominator = hit_f * miss_f / n_total;
        if !denominator.is_finite() || denominator <= 0.0 {
            return ResidualGateScore {
                gate,
                score: f64::NAN,
                signed_score: f64::NAN,
                n_hit,
            };
        }
        // Center within the pair-valid samples.  For an intercept-containing
        // residual vector this is exactly g' r; the centering also keeps the
        // masked path consistent when the valid subset has a nonzero sum.
        let centered_sum = sum_hit - (n_hit as f64 / n_total) * total;
        let signed_score = centered_sum / denominator.sqrt();
        // Use the same multiplication order as the existing centered-Gain
        // scorer so the missing-free residual screen is bitwise comparable
        // (not merely numerically close) to the Logic5 baseline.
        let score = n_total * centered_sum * centered_sum / (hit_f * miss_f);
        ResidualGateScore {
            gate,
            score,
            signed_score,
            n_hit,
        }
    })
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResidualScreenCandidate {
    pub(crate) first: usize,
    pub(crate) second: usize,
    pub(crate) best_gate: ResidualGateScore,
    pub(crate) proposal_score: f64,
    pub(crate) proposal_source: ResidualProposalSource,
    pub(crate) stats: FourCellSufficientStats,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ResidualScreenResult {
    pub(crate) candidates: Vec<ResidualScreenCandidate>,
    pub(crate) pairs_evaluated: usize,
    pub(crate) pairs_scored: usize,
    pub(crate) n_rows: usize,
    pub(crate) n_samples: usize,
}

#[inline]
fn best_gate_score(
    scores: &[ResidualGateScore; 5],
    include_xor: bool,
) -> Option<ResidualGateScore> {
    // Logic5 historically evaluates positive-first AND11, then AND01,
    // AND10, AND00.  Retain that priority for exact ties even though the
    // sufficient-statistics arrays are stored in natural cell order
    // [00, 01, 10, 11].
    let order = if include_xor {
        [3usize, 1, 2, 0, 4]
    } else {
        [3usize, 1, 2, 0, usize::MAX]
    };
    // Keep the first gate on exact ties.  This makes the derived gate label
    // deterministic and preserves the historical Logic5 ordering (AND gates
    // before XOR) without changing the score used for ranking.
    let mut best = None;
    for &gate_idx in order.iter().take(if include_xor { 5 } else { 4 }) {
        let score = scores[gate_idx];
        if !score.score.is_finite() {
            continue;
        }
        if best
            .map(|current: ResidualGateScore| score.score > current.score)
            .unwrap_or(true)
        {
            best = Some(score);
        }
    }
    best
}

#[inline]
fn best_and_gate_score(scores: &[ResidualGateScore; 5]) -> Option<ResidualGateScore> {
    let mut best = None;
    for &gate_idx in &[3usize, 1, 2, 0] {
        let score = scores[gate_idx];
        if !score.score.is_finite() {
            continue;
        }
        if best
            .map(|current: ResidualGateScore| score.score > current.score)
            .unwrap_or(true)
        {
            best = Some(score);
        }
    }
    best
}

/// Gate-independent between-cell residual sum of squares. It has at most
/// three degrees of freedom and is used only for proposal ranking.
#[inline]
fn four_cell_omnibus_score(stats: FourCellSufficientStats) -> Option<f64> {
    if stats.n_valid == 0 {
        return None;
    }
    let total = stats.sums.iter().sum::<f64>();
    let grand_mean = total / stats.n_valid as f64;
    let mut score = 0.0;
    for (&count, &sum) in stats.counts.iter().zip(stats.sums.iter()) {
        if count == 0 {
            continue;
        }
        let delta = sum / count as f64 - grand_mean;
        score += count as f64 * delta * delta;
    }
    score.is_finite().then_some(score)
}

#[inline]
fn candidate_cmp(lhs: &ResidualScreenCandidate, rhs: &ResidualScreenCandidate) -> Ordering {
    lhs.proposal_score
        .total_cmp(&rhs.proposal_score)
        .then_with(|| rhs.first.cmp(&lhs.first))
        .then_with(|| rhs.second.cmp(&lhs.second))
        .then_with(|| rhs.proposal_source.rank().cmp(&lhs.proposal_source.rank()))
}

#[inline]
fn proposal_candidate(
    first: usize,
    second: usize,
    stats: FourCellSufficientStats,
    scores: &[ResidualGateScore; 5],
    proposal_score: f64,
    proposal_source: ResidualProposalSource,
    include_xor: bool,
) -> Option<ResidualScreenCandidate> {
    if !proposal_score.is_finite() {
        return None;
    }
    let best_gate = best_gate_score(scores, include_xor)?;
    Some(ResidualScreenCandidate {
        first,
        second,
        best_gate,
        proposal_score,
        proposal_source,
        stats,
    })
}

#[inline]
fn merge_split_heaps(
    and_heap: BinaryHeap<Reverse<CandidateHeapEntry>>,
    xor_heap: BinaryHeap<Reverse<CandidateHeapEntry>>,
    top_k: usize,
) -> Vec<ResidualScreenCandidate> {
    let mut candidates = and_heap
        .into_iter()
        .chain(xor_heap)
        .map(|Reverse(entry)| entry.0)
        .collect::<Vec<_>>();
    // Sort by pair first so all representations of a pair are adjacent;
    // sorting only by score would allow the same pair to survive twice when
    // its AND and XOR scores are separated by another pair's score.
    candidates.sort_unstable_by(|lhs, rhs| {
        (lhs.first, lhs.second)
            .cmp(&(rhs.first, rhs.second))
            .then_with(|| candidate_cmp(rhs, lhs))
    });
    let mut unique = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        if unique
            .last()
            .map(|previous: &ResidualScreenCandidate| {
                previous.first == candidate.first && previous.second == candidate.second
            })
            .unwrap_or(false)
        {
            continue;
        }
        unique.push(candidate);
    }
    unique.sort_unstable_by(|lhs, rhs| candidate_cmp(rhs, lhs));
    let mut candidates = unique;
    candidates.truncate(top_k);
    candidates
}

#[derive(Clone, Debug)]
struct CandidateHeapEntry(ResidualScreenCandidate);

impl PartialEq for CandidateHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0.first == other.0.first
            && self.0.second == other.0.second
            && self.0.best_gate.gate == other.0.best_gate.gate
            && self.0.best_gate.score.to_bits() == other.0.best_gate.score.to_bits()
            && self.0.proposal_score.to_bits() == other.0.proposal_score.to_bits()
            && self.0.proposal_source == other.0.proposal_source
    }
}

impl Eq for CandidateHeapEntry {}

impl Ord for CandidateHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        candidate_cmp(&self.0, &other.0)
    }
}

impl PartialOrd for CandidateHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[inline]
fn push_top_k(
    heap: &mut BinaryHeap<Reverse<CandidateHeapEntry>>,
    candidate: ResidualScreenCandidate,
    top_k: usize,
) {
    if top_k == 0 {
        return;
    }
    let item = Reverse(CandidateHeapEntry(candidate));
    if heap.len() < top_k {
        heap.push(item);
    } else if heap
        .peek()
        .map(|worst| item.0.cmp(&worst.0) == Ordering::Greater)
        .unwrap_or(true)
    {
        let _ = heap.pop();
        heap.push(item);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_residual_pairs_packed(
    bits_flat: &[u64],
    valid_flat: Option<&[u64]>,
    row_words: usize,
    n_rows: usize,
    residual: &[f64],
    n_samples: usize,
    top_k: usize,
    include_xor: bool,
) -> Result<ResidualScreenResult, String> {
    scan_residual_pairs_packed_with_parallel(
        bits_flat,
        valid_flat,
        row_words,
        n_rows,
        residual,
        n_samples,
        top_k,
        include_xor,
        true,
    )
}

/// Internal variant used by callers that already parallelize over windows.
/// Standalone Python calls use Rayon; nested callers can pass `false` to keep
/// one level of thread-local Top-K heaps.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_residual_pairs_packed_with_parallel(
    bits_flat: &[u64],
    valid_flat: Option<&[u64]>,
    row_words: usize,
    n_rows: usize,
    residual: &[f64],
    n_samples: usize,
    top_k: usize,
    include_xor: bool,
    allow_parallel: bool,
) -> Result<ResidualScreenResult, String> {
    scan_residual_pairs_packed_with_mode(
        bits_flat,
        valid_flat,
        row_words,
        n_rows,
        residual,
        n_samples,
        top_k,
        include_xor,
        ResidualProposalMode::Max5,
        allow_parallel,
    )
}

/// Internal scanner with an explicit proposal policy. The default public
/// residual-screen API remains [`ResidualProposalMode::Max5`] for regression
/// compatibility; the other modes are diagnostic/experimental alternatives.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_residual_pairs_packed_with_mode(
    bits_flat: &[u64],
    valid_flat: Option<&[u64]>,
    row_words: usize,
    n_rows: usize,
    residual: &[f64],
    n_samples: usize,
    top_k: usize,
    include_xor: bool,
    proposal_mode: ResidualProposalMode,
    allow_parallel: bool,
) -> Result<ResidualScreenResult, String> {
    if n_rows < 2 {
        return Err("four-cell residual screen requires at least two markers".to_string());
    }
    let needed_words = words_for_samples(n_samples);
    if row_words < needed_words {
        return Err(format!(
            "four-cell residual screen row_words={row_words} smaller than required {needed_words}"
        ));
    }
    let required = n_rows
        .checked_mul(row_words)
        .ok_or_else(|| "four-cell residual screen packed size overflow".to_string())?;
    if bits_flat.len() < required {
        return Err(format!(
            "four-cell residual screen bits length={} smaller than required {required}",
            bits_flat.len()
        ));
    }
    if let Some(valid) = valid_flat {
        if valid.len() < required {
            return Err(format!(
                "four-cell residual screen valid bits length={} smaller than required {required}",
                valid.len()
            ));
        }
    }
    validate_continuous_y(residual, n_samples, "four-cell residual screen")?;
    let lookup = PackedYSumLookup::build(residual, n_samples)?;
    let heap_capacity = top_k.min(1024);
    let and_budget = if include_xor {
        top_k.div_ceil(2)
    } else {
        top_k
    };
    let xor_budget = if include_xor { top_k / 2 } else { 0 };
    let primary_budget = match proposal_mode {
        ResidualProposalMode::SplitHeap => and_budget,
        _ => top_k,
    };
    let first_rows = 0..n_rows.saturating_sub(1);
    let scan_first = |first: usize,
                      mut and_heap: BinaryHeap<Reverse<CandidateHeapEntry>>,
                      mut xor_heap: BinaryHeap<Reverse<CandidateHeapEntry>>,
                      mut pairs_scored: usize|
     -> (
        BinaryHeap<Reverse<CandidateHeapEntry>>,
        BinaryHeap<Reverse<CandidateHeapEntry>>,
        usize,
    ) {
        let first_start = first * row_words;
        let first_row = &bits_flat[first_start..first_start + needed_words];
        let first_valid = valid_flat.map(|valid| &valid[first_start..first_start + needed_words]);
        for second in (first + 1)..n_rows {
            let second_start = second * row_words;
            let second_row = &bits_flat[second_start..second_start + needed_words];
            let second_valid =
                valid_flat.map(|valid| &valid[second_start..second_start + needed_words]);
            let stats = four_cell_sufficient_stats_packed_unchecked(
                first_row,
                second_row,
                first_valid,
                second_valid,
                n_samples,
                &lookup,
            );
            let scores = derive_residual_gate_scores(stats);
            match proposal_mode {
                ResidualProposalMode::Max5 => {
                    let Some(best_gate) = best_gate_score(&scores, include_xor) else {
                        continue;
                    };
                    if let Some(candidate) = proposal_candidate(
                        first,
                        second,
                        stats,
                        &scores,
                        best_gate.score,
                        ResidualProposalSource::Max5,
                        include_xor,
                    ) {
                        pairs_scored = pairs_scored.saturating_add(1);
                        push_top_k(&mut and_heap, candidate, primary_budget);
                    }
                }
                ResidualProposalMode::Omnibus4Cell => {
                    let Some(proposal_score) = four_cell_omnibus_score(stats) else {
                        continue;
                    };
                    if let Some(candidate) = proposal_candidate(
                        first,
                        second,
                        stats,
                        &scores,
                        proposal_score,
                        ResidualProposalSource::Omnibus4Cell,
                        include_xor,
                    ) {
                        pairs_scored = pairs_scored.saturating_add(1);
                        push_top_k(&mut and_heap, candidate, primary_budget);
                    }
                }
                ResidualProposalMode::SplitHeap => {
                    let mut retained = false;
                    if let Some(and_gate) = best_and_gate_score(&scores) {
                        if let Some(candidate) = proposal_candidate(
                            first,
                            second,
                            stats,
                            &scores,
                            and_gate.score,
                            ResidualProposalSource::And,
                            include_xor,
                        ) {
                            push_top_k(&mut and_heap, candidate, and_budget);
                            retained = true;
                        }
                    }
                    if include_xor && xor_budget > 0 && scores[4].score.is_finite() {
                        if let Some(candidate) = proposal_candidate(
                            first,
                            second,
                            stats,
                            &scores,
                            scores[4].score,
                            ResidualProposalSource::Xor,
                            include_xor,
                        ) {
                            push_top_k(&mut xor_heap, candidate, xor_budget);
                            retained = true;
                        }
                    }
                    if retained {
                        pairs_scored = pairs_scored.saturating_add(1);
                    }
                }
            }
        }
        (and_heap, xor_heap, pairs_scored)
    };
    let empty_heaps = || {
        (
            BinaryHeap::<Reverse<CandidateHeapEntry>>::with_capacity(heap_capacity),
            BinaryHeap::<Reverse<CandidateHeapEntry>>::with_capacity(heap_capacity),
        )
    };
    let merge_heaps = |((mut left_and, mut left_xor), left_scored): (
        (
            BinaryHeap<Reverse<CandidateHeapEntry>>,
            BinaryHeap<Reverse<CandidateHeapEntry>>,
        ),
        usize,
    ),
                       ((right_and, right_xor), right_scored): (
        (
            BinaryHeap<Reverse<CandidateHeapEntry>>,
            BinaryHeap<Reverse<CandidateHeapEntry>>,
        ),
        usize,
    )| {
        for Reverse(entry) in right_and {
            push_top_k(&mut left_and, entry.0, primary_budget);
        }
        let xor_budget = if proposal_mode == ResidualProposalMode::SplitHeap {
            xor_budget
        } else {
            0
        };
        for Reverse(entry) in right_xor {
            push_top_k(&mut left_xor, entry.0, xor_budget);
        }
        (
            (left_and, left_xor),
            left_scored.saturating_add(right_scored),
        )
    };
    let ((and_heap, xor_heap), pairs_scored) =
        if allow_parallel && rayon::current_num_threads() > 1 && n_rows >= 32 {
            first_rows
                .into_par_iter()
                .fold(
                    || {
                        let (and_heap, xor_heap) = empty_heaps();
                        ((and_heap, xor_heap), 0usize)
                    },
                    |((and_heap, xor_heap), pairs_scored), first| {
                        let (and_heap, xor_heap, pairs_scored) =
                            scan_first(first, and_heap, xor_heap, pairs_scored);
                        ((and_heap, xor_heap), pairs_scored)
                    },
                )
                .reduce(
                    || {
                        let (and_heap, xor_heap) = empty_heaps();
                        ((and_heap, xor_heap), 0usize)
                    },
                    merge_heaps,
                )
        } else {
            let (mut serial_and, mut serial_xor) = empty_heaps();
            let mut serial_scored = 0usize;
            for first in 0..n_rows.saturating_sub(1) {
                let (row_and, row_xor, row_scored) =
                    scan_first(first, empty_heaps().0, empty_heaps().1, 0);
                for Reverse(entry) in row_and {
                    push_top_k(&mut serial_and, entry.0, primary_budget);
                }
                let xor_budget = if proposal_mode == ResidualProposalMode::SplitHeap {
                    xor_budget
                } else {
                    0
                };
                for Reverse(entry) in row_xor {
                    push_top_k(&mut serial_xor, entry.0, xor_budget);
                }
                serial_scored = serial_scored.saturating_add(row_scored);
            }
            ((serial_and, serial_xor), serial_scored)
        };
    let pairs_evaluated = n_rows
        .checked_mul(n_rows.saturating_sub(1))
        .and_then(|value| value.checked_div(2))
        .ok_or_else(|| "four-cell residual screen pair counter overflow".to_string())?;
    let candidates = if proposal_mode == ResidualProposalMode::SplitHeap {
        merge_split_heaps(and_heap, xor_heap, top_k)
    } else {
        let mut candidates = and_heap
            .into_iter()
            .map(|Reverse(entry)| entry.0)
            .collect::<Vec<_>>();
        candidates.sort_unstable_by(|lhs, rhs| candidate_cmp(rhs, lhs));
        candidates
    };
    Ok(ResidualScreenResult {
        candidates,
        pairs_evaluated,
        pairs_scored,
        n_rows,
        n_samples,
    })
}

fn u64_array2_to_vec(array: &PyReadonlyArray2<'_, u64>) -> Vec<u64> {
    array.as_array().iter().copied().collect()
}

fn set_screen_columns<'py>(
    out: &Bound<'py, PyDict>,
    candidates: &[ResidualScreenCandidate],
) -> PyResult<()> {
    let cell_names = ["00", "01", "10", "11"];
    for (cell, name) in cell_names.iter().enumerate() {
        out.set_item(
            format!("n{name}"),
            candidates
                .iter()
                .map(|candidate| candidate.stats.counts[cell])
                .collect::<Vec<_>>(),
        )?;
        out.set_item(
            format!("sum{name}"),
            candidates
                .iter()
                .map(|candidate| candidate.stats.sums[cell])
                .collect::<Vec<_>>(),
        )?;
        out.set_item(
            format!("residual_mean{name}"),
            candidates
                .iter()
                .map(|candidate| {
                    let count = candidate.stats.counts[cell];
                    if count == 0 {
                        f64::NAN
                    } else {
                        candidate.stats.sums[cell] / count as f64
                    }
                })
                .collect::<Vec<_>>(),
        )?;
    }
    out.set_item(
        "n_valid",
        candidates
            .iter()
            .map(|candidate| candidate.stats.n_valid)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "best_gate",
        candidates
            .iter()
            .map(|candidate| candidate.best_gate.gate.as_str())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "screen_score",
        candidates
            .iter()
            .map(|candidate| candidate.proposal_score)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "proposal_score",
        candidates
            .iter()
            .map(|candidate| candidate.proposal_score)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "proposal_source",
        candidates
            .iter()
            .map(|candidate| candidate.proposal_source.as_str())
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "signed_score",
        candidates
            .iter()
            .map(|candidate| candidate.best_gate.signed_score)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "n_hit",
        candidates
            .iter()
            .map(|candidate| candidate.best_gate.n_hit)
            .collect::<Vec<_>>(),
    )?;
    let gate_names = ["and00", "and01", "and10", "and11", "xor"];
    let all_scores = candidates
        .iter()
        .map(|candidate| derive_residual_gate_scores(candidate.stats))
        .collect::<Vec<_>>();
    for (gate_idx, name) in gate_names.iter().enumerate() {
        out.set_item(
            format!("{name}_score"),
            all_scores
                .iter()
                .map(|scores| scores[gate_idx].score)
                .collect::<Vec<_>>(),
        )?;
        out.set_item(
            format!("{name}_signed_score"),
            all_scores
                .iter()
                .map(|scores| scores[gate_idx].signed_score)
                .collect::<Vec<_>>(),
        )?;
        out.set_item(
            format!("{name}_n_hit"),
            all_scores
                .iter()
                .map(|scores| scores[gate_idx].n_hit)
                .collect::<Vec<_>>(),
        )?;
    }
    Ok(())
}

/// Scan every packed marker pair with a residual-only four-cell primitive.
///
/// `bin_path` stores marker-major 0/1 bits.  `valid_bits`, when provided, is
/// a marker-major `(n_rows, row_words)` uint64 matrix; pair validity is the
/// intersection of the two marker rows.  Omitting it is the missing-free fast
/// path.  The returned screen scores are for ranking only; exact GRM-aware
/// inference must use the existing exact backend on retained candidates.
#[pyfunction(name = "garfield_logic_residual_screen_bin")]
#[pyo3(signature = (bin_path, residual, top_k=100, valid_bits=None, xor_search=true, proposal_mode="max5"))]
pub fn garfield_logic_residual_screen_bin_py<'py>(
    py: Python<'py>,
    bin_path: String,
    residual: PyReadonlyArray1<'py, f64>,
    top_k: usize,
    valid_bits: Option<PyReadonlyArray2<'py, u64>>,
    xor_search: bool,
    proposal_mode: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let (bits, row_words, n_rows, n_samples) =
        load_bin01_as_u64_words(&bin_path, "garfield_logic_residual_screen_bin")
            .map_err(PyRuntimeError::new_err)?;
    let residual_slice = residual
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("garfield_logic_residual_screen_bin: {e}")))?;
    let valid_owned = if let Some(valid) = valid_bits.as_ref() {
        let shape = valid.shape();
        if shape != [n_rows, row_words] {
            return Err(PyValueError::new_err(format!(
                "valid_bits shape=({},{}) expected ({n_rows},{row_words})",
                shape[0], shape[1]
            )));
        }
        Some(u64_array2_to_vec(valid))
    } else {
        None
    };
    let proposal_mode =
        ResidualProposalMode::parse(proposal_mode).map_err(PyValueError::new_err)?;
    let result = scan_residual_pairs_packed_with_mode(
        &bits,
        valid_owned.as_deref(),
        row_words,
        n_rows,
        residual_slice,
        n_samples,
        top_k,
        xor_search,
        proposal_mode,
        true,
    )
    .map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    out.set_item(
        "first",
        result
            .candidates
            .iter()
            .map(|candidate| candidate.first)
            .collect::<Vec<_>>(),
    )?;
    out.set_item(
        "second",
        result
            .candidates
            .iter()
            .map(|candidate| candidate.second)
            .collect::<Vec<_>>(),
    )?;
    set_screen_columns(&out, result.candidates.as_slice())?;
    out.set_item("pairs_evaluated", result.pairs_evaluated)?;
    out.set_item("pairs_scored", result.pairs_scored)?;
    out.set_item("n_rows", result.n_rows)?;
    out.set_item("n_samples", result.n_samples)?;
    out.set_item("valid_mask_supplied", valid_bits.is_some())?;
    out.set_item("xor_search", xor_search)?;
    out.set_item("proposal_mode", proposal_mode.as_str())?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::garfield::score::score_cont_centered_gain_from_sum_and_n_hit;

    fn pack(values: &[bool]) -> Vec<u64> {
        let mut out = vec![0u64; values.len().div_ceil(64).max(1)];
        for (idx, &value) in values.iter().enumerate() {
            if value {
                out[idx >> 6] |= 1u64 << (idx & 63);
            }
        }
        out
    }

    #[test]
    fn four_cell_stats_match_dense_reference_and_xor_derivation() {
        let first = pack(&[false, false, true, true, false, true, true, false, true]);
        let second = pack(&[false, true, false, true, false, true, false, true, true]);
        let residual = [-4.0, 1.0, 2.0, 3.0, -2.0, 0.0, 4.0, -1.0, -3.0];
        let lookup = PackedYSumLookup::build(&residual, residual.len()).unwrap();
        let stats = four_cell_sufficient_stats_packed(
            &first,
            &second,
            None,
            None,
            &residual,
            residual.len(),
            &lookup,
        )
        .unwrap();
        assert_eq!(stats.counts, [2, 2, 2, 3]);
        assert_eq!(stats.sums, [-6.0, 0.0, 6.0, 0.0]);
        assert_eq!(stats.n_valid, 9);
        assert_eq!(stats.sums[1] + stats.sums[2], 6.0);
        assert_eq!(stats.counts[1] + stats.counts[2], 4);
    }

    #[test]
    fn strict_pair_valid_mask_excludes_missing_or_heterozygous_samples() {
        let first = pack(&[false, false, true, true, false, true, true, false]);
        let second = pack(&[false, true, false, true, false, true, false, true]);
        let first_valid = pack(&[true, true, false, true, true, true, true, true]);
        let second_valid = pack(&[true, true, true, true, false, true, true, true]);
        let residual = [-3.0, 1.0, 8.0, 2.0, 9.0, -1.0, 4.0, -2.0];
        let lookup = PackedYSumLookup::build(&residual, residual.len()).unwrap();
        let stats = four_cell_sufficient_stats_packed(
            &first,
            &second,
            Some(&first_valid),
            Some(&second_valid),
            &residual,
            residual.len(),
            &lookup,
        )
        .unwrap();
        assert_eq!(stats.counts, [1, 2, 1, 2]);
        assert_eq!(stats.sums, [-3.0, -1.0, 4.0, 1.0]);
        assert_eq!(stats.n_valid, 6);
    }

    #[test]
    fn residual_gate_scores_match_centered_gain_reference() {
        let stats = FourCellSufficientStats {
            counts: [2, 2, 2, 3],
            sums: [-6.0, 0.0, 6.0, 0.0],
            n_valid: 9,
        };
        let gates = derive_residual_gate_scores(stats);
        let total = stats.sums.iter().sum::<f64>();
        let expected = [
            (2usize, -6.0),
            (2usize, 0.0),
            (2usize, 6.0),
            (3usize, 0.0),
            (4usize, 6.0),
        ];
        for (gate, (n_hit, sum_hit)) in gates.iter().zip(expected) {
            let reference =
                score_cont_centered_gain_from_sum_and_n_hit(total, sum_hit, stats.n_valid, n_hit);
            assert!(
                (gate.score - reference.raw_score).abs() < 1e-12,
                "gate={:?} n_hit={} signed={} got={} expected={}",
                gate.gate,
                n_hit,
                gate.signed_score,
                gate.score,
                reference.raw_score
            );
            assert_eq!(gate.n_hit, n_hit);
        }
    }

    #[test]
    fn zero_variance_gate_is_unidentifiable() {
        let stats = FourCellSufficientStats {
            counts: [4, 0, 0, 0],
            sums: [0.0, 0.0, 0.0, 0.0],
            n_valid: 4,
        };
        let gates = derive_residual_gate_scores(stats);
        assert!(gates[0].score.is_nan());
        assert!(gates[0].signed_score.is_nan());
    }

    #[test]
    fn pair_scan_returns_all_pairs_in_deterministic_score_order() {
        let rows = [
            pack(&[false, false, true, true, false]),
            pack(&[false, true, false, true, false]),
            pack(&[true, true, true, false, false]),
        ];
        let bits_flat = rows.concat();
        let residual = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let result = scan_residual_pairs_packed(
            &bits_flat,
            None,
            1,
            rows.len(),
            &residual,
            residual.len(),
            3,
            true,
        )
        .unwrap();
        assert_eq!(result.pairs_evaluated, 3);
        assert_eq!(result.pairs_scored, 3);
        assert_eq!(result.candidates.len(), 3);

        // The scanner and the public primitive must agree on every returned
        // candidate, including the one-pass four-cell statistics.
        let lookup = PackedYSumLookup::build(&residual, residual.len()).unwrap();
        for candidate in &result.candidates {
            let first = &rows[candidate.first];
            let second = &rows[candidate.second];
            let expected = four_cell_sufficient_stats_packed(
                first,
                second,
                None,
                None,
                &residual,
                residual.len(),
                &lookup,
            )
            .unwrap();
            assert_eq!(candidate.stats, expected);
            let expected_gate = best_gate_score(&derive_residual_gate_scores(expected), true)
                .expect("at least one gate should have variance");
            assert_eq!(candidate.best_gate, expected_gate);
        }
        for pair in result.candidates.windows(2) {
            assert!(candidate_cmp(&pair[0], &pair[1]) != Ordering::Less);
        }
    }

    #[test]
    fn pair_scan_intersects_strict_valid_masks() {
        let rows = [
            pack(&[false, false, true, true, false]),
            pack(&[false, true, false, true, false]),
        ];
        let valid_rows = [
            pack(&[true, true, false, true, true]),
            pack(&[true, true, true, true, false]),
        ];
        let bits_flat = rows.concat();
        let valid_flat = valid_rows.concat();
        let residual = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let result = scan_residual_pairs_packed(
            &bits_flat,
            Some(&valid_flat),
            1,
            rows.len(),
            &residual,
            residual.len(),
            10,
            true,
        )
        .unwrap();
        assert_eq!(result.pairs_evaluated, 1);
        assert_eq!(result.pairs_scored, 1);
        assert_eq!(result.candidates[0].stats.n_valid, 3);
        assert_eq!(result.candidates[0].stats.counts.iter().sum::<usize>(), 3);
    }

    #[test]
    fn missing_free_screen_matches_legacy_logic5_pair_scores() {
        let rows = [
            pack(&[false, false, true, true, false]),
            pack(&[false, true, false, true, false]),
            pack(&[true, true, true, false, false]),
        ];
        let bits_flat = rows.concat();
        let residual = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let legacy =
            crate::garfield::all_pair::scan_all_pairs_continuous_packed_with_parallel_and_xor(
                &bits_flat,
                1,
                rows.len(),
                &residual,
                residual.len(),
                3,
                false,
                true,
            )
            .unwrap();
        let screen = scan_residual_pairs_packed(
            &bits_flat,
            None,
            1,
            rows.len(),
            &residual,
            residual.len(),
            3,
            true,
        )
        .unwrap();
        assert_eq!(legacy.candidates.len(), screen.candidates.len());
        for (old, current) in legacy.candidates.iter().zip(screen.candidates.iter()) {
            assert_eq!((old.first, old.second), (current.first, current.second));
            let expected_gate = match (old.gate, old.polarity) {
                (crate::garfield::all_pair::AllPairGate::And, 0) => ResidualGate::And11,
                (crate::garfield::all_pair::AllPairGate::And, 1) => ResidualGate::And01,
                (crate::garfield::all_pair::AllPairGate::And, 2) => ResidualGate::And10,
                (crate::garfield::all_pair::AllPairGate::And, 3) => ResidualGate::And00,
                (crate::garfield::all_pair::AllPairGate::Xor, 0) => ResidualGate::Xor,
                _ => panic!("unexpected legacy gate/polarity"),
            };
            assert_eq!(current.best_gate.gate, expected_gate);
            // AND masks use exactly the same scalar score expression.  XOR
            // is derived from the two cell sums, so allow only the rounding
            // from that final two-term addition.
            if matches!(expected_gate, ResidualGate::Xor) {
                assert!((old.raw_score - current.best_gate.score).abs() < 1e-12);
            } else {
                assert_eq!(old.raw_score.to_bits(), current.best_gate.score.to_bits());
            }
        }
    }

    #[test]
    fn four_cell_omnibus_score_uses_all_cells_without_gate_selection() {
        let stats = FourCellSufficientStats {
            counts: [2, 2, 2, 2],
            sums: [4.0, 0.0, 0.0, 4.0],
            n_valid: 8,
        };
        let score = four_cell_omnibus_score(stats).expect("non-flat four-cell pattern");
        // Cell means are [2, 0, 0, 2], grand mean is 1, so the between-cell
        // sum of squares is 4 cells * 2 samples * 1^2 = 8.
        assert!((score - 8.0).abs() < 1.0e-12);
    }

    #[test]
    fn split_heap_returns_unique_pairs_within_total_budget() {
        let rows = [
            pack(&[false, false, true, true, false, true]),
            pack(&[false, true, false, true, false, true]),
            pack(&[true, true, true, false, false, false]),
            pack(&[true, false, true, false, true, false]),
        ];
        let bits_flat = rows.concat();
        let residual = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let result = scan_residual_pairs_packed_with_mode(
            &bits_flat,
            None,
            1,
            rows.len(),
            &residual,
            residual.len(),
            4,
            true,
            ResidualProposalMode::SplitHeap,
            false,
        )
        .unwrap();
        assert!(result.candidates.len() <= 4);
        for left in 0..result.candidates.len() {
            for right in (left + 1)..result.candidates.len() {
                assert_ne!(
                    (
                        result.candidates[left].first,
                        result.candidates[left].second
                    ),
                    (
                        result.candidates[right].first,
                        result.candidates[right].second
                    )
                );
            }
        }
        assert!(result
            .candidates
            .iter()
            .all(|candidate| candidate.proposal_score.is_finite()));
    }

    #[test]
    fn split_heap_deduplicates_interleaved_and_xor_scores() {
        let stats = FourCellSufficientStats {
            counts: [1, 1, 1, 1],
            sums: [0.0, 0.0, 0.0, 0.0],
            n_valid: 4,
        };
        let gate = ResidualGateScore {
            gate: ResidualGate::And11,
            score: 0.0,
            signed_score: 0.0,
            n_hit: 1,
        };
        let candidate = |first, second, score, source| ResidualScreenCandidate {
            first,
            second,
            best_gate: gate,
            proposal_score: score,
            proposal_source: source,
            stats,
        };
        let mut and_heap = BinaryHeap::new();
        let mut xor_heap = BinaryHeap::new();
        and_heap.push(Reverse(CandidateHeapEntry(candidate(
            0,
            1,
            10.0,
            ResidualProposalSource::And,
        ))));
        xor_heap.push(Reverse(CandidateHeapEntry(candidate(
            2,
            3,
            9.5,
            ResidualProposalSource::Xor,
        ))));
        xor_heap.push(Reverse(CandidateHeapEntry(candidate(
            0,
            1,
            9.0,
            ResidualProposalSource::Xor,
        ))));
        let merged = merge_split_heaps(and_heap, xor_heap, 3);
        assert_eq!(merged.len(), 2);
        assert_eq!((merged[0].first, merged[0].second), (0, 1));
        assert_eq!((merged[1].first, merged[1].second), (2, 3));
    }
}
