//! Conditional two-locus dosage interaction backend.
//!
//! The implementation is deliberately split into three layers:
//!
//! 1. marker representation/decoding (binary, ternary, or floating dosage);
//! 2. pair sufficient statistics (PairMoments);
//! 3. one FWL interaction scorer.
//!
//! All representations fit the same model
//!
//!     y = mu + beta1*g1 + beta2*g2 + beta12*g1*g2 + e
//!
//! The public fit is currently an OLS/FWL backend.  An LMM/PCG adapter can
//! later provide transformed moments without changing the pair scorer.

#[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
use crate::blas::{
    cblas_dgemm_dispatch, BlasThreadGuard, CblasInt, CBLAS_NO_TRANS, CBLAS_ROW_MAJOR, CBLAS_TRANS,
};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, FisherSnedecor};
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::ops::Range;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

const DESIGN_COLS: usize = 4;
const DOSAGE_LEVELS: usize = 3;
const DOSAGE_CELLS: usize = DOSAGE_LEVELS * DOSAGE_LEVELS;
const SOLVE_EPS: f64 = 1.0e-12;
const DOSAGE_TOL: f64 = 1.0e-8;
const LOGIC_TOL: f64 = 1.0e-8;
const FLOAT_GEMM_BLOCK_MARKERS: usize = 64;

#[cfg(test)]
static PVALUE_CALLS: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum DosageLogicLabel {
    Additive,
    High00,
    High10,
    High01,
    High11,
    Low00,
    Low10,
    Low01,
    Low11,
    XorHigh,
    XorLow,
    Unresolved,
}

impl DosageLogicLabel {
    fn as_str(self) -> &'static str {
        match self {
            Self::Additive => "additive",
            Self::High00 => "high_00",
            Self::High10 => "high_10",
            Self::High01 => "high_01",
            Self::High11 => "high_11",
            Self::Low00 => "low_00",
            Self::Low10 => "low_10",
            Self::Low01 => "low_01",
            Self::Low11 => "low_11",
            Self::XorHigh => "xor_high",
            Self::XorLow => "xor_low",
            Self::Unresolved => "unresolved",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct MarkerEncoding {
    pub(crate) offset: f64,
    pub(crate) scale: f64,
}

#[derive(Clone, Debug)]
struct BinaryMarker {
    encoding: MarkerEncoding,
    levels: [f64; 2],
    high_bits: Vec<u64>,
    missing_bits: Vec<u64>,
    has_missing: bool,
    n_samples: usize,
}

#[derive(Clone, Debug)]
struct TernaryMarker {
    encoding: MarkerEncoding,
    plane_one: Vec<u64>,
    plane_two: Vec<u64>,
    missing_bits: Vec<u64>,
    has_missing: bool,
    n_samples: usize,
}

#[derive(Clone, Debug)]
struct FloatMarker {
    encoding: MarkerEncoding,
    values: Vec<f64>,
    missing: Vec<bool>,
    has_missing: bool,
    has_integer_values: bool,
    n_samples: usize,
}

#[derive(Clone, Debug)]
enum MarkerRepresentation {
    Binary(BinaryMarker),
    Ternary(TernaryMarker),
    Float(FloatMarker),
}

impl MarkerRepresentation {
    fn encoding(&self) -> MarkerEncoding {
        match self {
            Self::Binary(marker) => marker.encoding,
            Self::Ternary(marker) => marker.encoding,
            Self::Float(marker) => marker.encoding,
        }
    }

    fn n_samples(&self) -> usize {
        match self {
            Self::Binary(marker) => marker.n_samples,
            Self::Ternary(marker) => marker.n_samples,
            Self::Float(marker) => marker.n_samples,
        }
    }

    fn n_levels(&self) -> usize {
        match self {
            Self::Binary(_) => 2,
            Self::Ternary(_) => 3,
            Self::Float(_) => 0,
        }
    }

    fn is_discrete(&self) -> bool {
        !matches!(self, Self::Float(_))
    }

    fn has_missing(&self) -> bool {
        match self {
            Self::Binary(marker) => marker.has_missing,
            Self::Ternary(marker) => marker.has_missing,
            Self::Float(marker) => marker.has_missing,
        }
    }

    fn binary_levels(&self) -> Option<[f64; 2]> {
        match self {
            Self::Binary(marker) => Some(marker.levels),
            _ => None,
        }
    }

    fn value_at(&self, index: usize) -> f64 {
        match self {
            Self::Binary(marker) => {
                let word = index / 64;
                let bit = 1u64 << (index % 64);
                if marker.high_bits[word] & bit != 0 {
                    marker.levels[1]
                } else {
                    marker.levels[0]
                }
            }
            Self::Ternary(marker) => {
                let word = index / 64;
                let bit = 1u64 << (index % 64);
                if marker.plane_two[word] & bit != 0 {
                    2.0
                } else if marker.plane_one[word] & bit != 0 {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Float(marker) => marker.values[index],
        }
    }

    fn missing_at(&self, index: usize) -> bool {
        match self {
            Self::Binary(marker) => bit_is_set(&marker.missing_bits, index),
            Self::Ternary(marker) => bit_is_set(&marker.missing_bits, index),
            Self::Float(marker) => marker.missing[index],
        }
    }

    fn category_count(&self) -> usize {
        self.n_levels()
    }

    fn category_word(&self, category: usize, word: usize) -> u64 {
        let n_words = words_for_samples(self.n_samples());
        if word >= n_words || category >= self.category_count() {
            return 0;
        }
        let active = active_word_mask(self.n_samples(), word);
        match self {
            Self::Binary(marker) => {
                let high = marker.high_bits[word];
                let bits = if category == 0 { !high } else { high };
                bits & !marker.missing_bits[word] & active
            }
            Self::Ternary(marker) => {
                let bits = match category {
                    0 => !(marker.plane_one[word] | marker.plane_two[word]),
                    1 => marker.plane_one[word],
                    2 => marker.plane_two[word],
                    _ => 0,
                };
                bits & !marker.missing_bits[word] & active
            }
            Self::Float(_) => 0,
        }
    }

    fn missing_word(&self, word: usize) -> u64 {
        match self {
            Self::Binary(marker) => marker.missing_bits.get(word).copied().unwrap_or(0),
            Self::Ternary(marker) => marker.missing_bits.get(word).copied().unwrap_or(0),
            Self::Float(marker) => marker
                .missing
                .chunks(64)
                .nth(word)
                .map(|chunk| {
                    chunk
                        .iter()
                        .enumerate()
                        .fold(0u64, |bits, (offset, missing)| {
                            bits | (u64::from(*missing) << offset)
                        })
                })
                .unwrap_or(0),
        }
    }
}

fn words_for_samples(n_samples: usize) -> usize {
    n_samples.saturating_add(63) / 64
}

fn active_word_mask(n_samples: usize, word: usize) -> u64 {
    let start = word.saturating_mul(64);
    if start >= n_samples {
        return 0;
    }
    let remaining = n_samples - start;
    if remaining >= 64 {
        u64::MAX
    } else {
        (1u64 << remaining) - 1
    }
}

#[inline]
fn bit_is_set(bits: &[u64], index: usize) -> bool {
    bits.get(index / 64)
        .map(|word| word & (1u64 << (index % 64)) != 0)
        .unwrap_or(false)
}

#[inline]
fn set_bit(bits: &mut [u64], index: usize) {
    bits[index / 64] |= 1u64 << (index % 64);
}

/// Byte-indexed lookup tables for summing a fixed phenotype over arbitrary
/// 64-bit masks.  Discrete pair moments are accumulated from category
/// intersections; looking up eight bytes per word avoids visiting every set
/// sample bit in the pair hot loop.
struct MaskYLookup {
    y: Vec<f64>,
    y2: Vec<f64>,
    n_words: usize,
    total_y: f64,
    total_y2: f64,
}

impl MaskYLookup {
    const BYTE_TABLE_SIZE: usize = 256;
    const BYTES_PER_WORD: usize = 8;

    fn new(y: &[f64]) -> Self {
        let n_words = words_for_samples(y.len());
        let total_y = y.iter().copied().sum::<f64>();
        let total_y2 = y.iter().map(|value| value * value).sum::<f64>();
        let table_len = n_words * Self::BYTES_PER_WORD * Self::BYTE_TABLE_SIZE;
        let mut y_table = vec![0.0; table_len];
        let mut y2_table = vec![0.0; table_len];
        for word in 0..n_words {
            for byte in 0..Self::BYTES_PER_WORD {
                let base = (word * Self::BYTES_PER_WORD + byte) * Self::BYTE_TABLE_SIZE;
                let sample_base = word * 64 + byte * 8;
                for mask in 1..Self::BYTE_TABLE_SIZE {
                    let low_bit = mask.trailing_zeros() as usize;
                    let previous = mask & (mask - 1);
                    let sample = sample_base + low_bit;
                    let (value, square) = y
                        .get(sample)
                        .copied()
                        .map(|value| (value, value * value))
                        .unwrap_or((0.0, 0.0));
                    y_table[base + mask] = y_table[base + previous] + value;
                    y2_table[base + mask] = y2_table[base + previous] + square;
                }
            }
        }
        Self {
            y: y_table,
            y2: y2_table,
            n_words,
            total_y,
            total_y2,
        }
    }

    #[inline]
    fn sum_table(table: &[f64], mask: u64, word: usize) -> f64 {
        let base = word * Self::BYTES_PER_WORD * Self::BYTE_TABLE_SIZE;
        table[base + (mask as usize & 0xff)]
            + table[base + Self::BYTE_TABLE_SIZE + ((mask >> 8) as usize & 0xff)]
            + table[base + 2 * Self::BYTE_TABLE_SIZE + ((mask >> 16) as usize & 0xff)]
            + table[base + 3 * Self::BYTE_TABLE_SIZE + ((mask >> 24) as usize & 0xff)]
            + table[base + 4 * Self::BYTE_TABLE_SIZE + ((mask >> 32) as usize & 0xff)]
            + table[base + 5 * Self::BYTE_TABLE_SIZE + ((mask >> 40) as usize & 0xff)]
            + table[base + 6 * Self::BYTE_TABLE_SIZE + ((mask >> 48) as usize & 0xff)]
            + table[base + 7 * Self::BYTE_TABLE_SIZE + ((mask >> 56) as usize & 0xff)]
    }

    #[inline]
    fn sum_y(&self, mask: u64, word: usize) -> f64 {
        debug_assert!(word < self.n_words);
        Self::sum_table(&self.y, mask, word)
    }

    #[inline]
    fn sum_y2(&self, mask: u64, word: usize) -> f64 {
        debug_assert!(word < self.n_words);
        Self::sum_table(&self.y2, mask, word)
    }
}

fn integer_dosage_code(value: f64) -> Option<usize> {
    if !value.is_finite() {
        return None;
    }
    let rounded = value.round();
    if (value - rounded).abs() <= DOSAGE_TOL && (0.0..=2.0).contains(&rounded) {
        Some(rounded as usize)
    } else {
        None
    }
}

fn finite_distinct_levels(values: &[f64]) -> Vec<f64> {
    let mut levels = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    levels.sort_by(f64::total_cmp);
    levels.dedup_by(|left, right| (*left - *right).abs() <= DOSAGE_TOL);
    levels
}

fn validate_dosage_values(values: &[f64], name: &str) -> Result<(), String> {
    if values.is_empty() {
        return Err(format!("{name} must not be empty"));
    }
    let mut finite = 0usize;
    for (index, value) in values.iter().copied().enumerate() {
        if value.is_infinite() {
            return Err(format!("{name}[{index}] is infinite"));
        }
        if value.is_nan() {
            continue;
        }
        if !(-DOSAGE_TOL..=2.0 + DOSAGE_TOL).contains(&value) {
            return Err(format!(
                "{name}[{index}] is outside dosage range [0, 2]: {value}"
            ));
        }
        finite += 1;
    }
    if finite == 0 {
        return Err(format!("{name} has no observed dosage values"));
    }
    Ok(())
}

fn build_marker_representation(values: &[f64], name: &str) -> Result<MarkerRepresentation, String> {
    validate_dosage_values(values, name)?;
    let mut codes = Vec::with_capacity(values.len());
    let mut integer_levels = [false; DOSAGE_LEVELS];
    let mut all_integer = true;
    let mut has_missing = false;
    for value in values.iter().copied() {
        let code = integer_dosage_code(value);
        if let Some(code) = code {
            integer_levels[code] = true;
        } else if value.is_finite() {
            all_integer = false;
        } else {
            has_missing = true;
        }
        codes.push(code);
    }
    let levels = integer_levels
        .iter()
        .enumerate()
        .filter_map(|(level, present)| present.then_some(level as f64))
        .collect::<Vec<_>>();
    let finite_levels = finite_distinct_levels(values);
    if finite_levels.len() <= 1 {
        return Err(format!("{name} is monomorphic"));
    }
    let n_words = words_for_samples(values.len());
    if all_integer && levels.len() == 2 {
        let mut high_bits = vec![0u64; n_words];
        let mut missing_bits = vec![0u64; n_words];
        for (index, code) in codes.iter().enumerate() {
            match code {
                Some(code) if (*code as f64) == levels[1] => set_bit(&mut high_bits, index),
                Some(_) => {}
                None => set_bit(&mut missing_bits, index),
            }
        }
        return Ok(MarkerRepresentation::Binary(BinaryMarker {
            encoding: MarkerEncoding {
                offset: levels[0],
                scale: levels[1] - levels[0],
            },
            levels: [levels[0], levels[1]],
            high_bits,
            missing_bits,
            has_missing,
            n_samples: values.len(),
        }));
    }
    if all_integer && levels.len() == 3 {
        let mut plane_one = vec![0u64; n_words];
        let mut plane_two = vec![0u64; n_words];
        let mut missing_bits = vec![0u64; n_words];
        for (index, code) in codes.iter().enumerate() {
            match code {
                Some(1) => set_bit(&mut plane_one, index),
                Some(2) => set_bit(&mut plane_two, index),
                Some(0) => {}
                Some(_) => unreachable!("integer dosage code is restricted to 0, 1, 2"),
                None => set_bit(&mut missing_bits, index),
            }
        }
        return Ok(MarkerRepresentation::Ternary(TernaryMarker {
            encoding: MarkerEncoding {
                offset: 0.0,
                scale: 1.0,
            },
            plane_one,
            plane_two,
            missing_bits,
            has_missing,
            n_samples: values.len(),
        }));
    }
    let missing = values
        .iter()
        .map(|value| value.is_nan())
        .collect::<Vec<_>>();
    let has_integer_values = integer_levels.iter().any(|present| *present);
    Ok(MarkerRepresentation::Float(FloatMarker {
        encoding: MarkerEncoding {
            offset: 0.0,
            scale: 1.0,
        },
        values: values.to_vec(),
        missing,
        has_missing,
        has_integer_values,
        n_samples: values.len(),
    }))
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PairMoments {
    pub(crate) xtx: [[f64; DESIGN_COLS]; DESIGN_COLS],
    pub(crate) xty: [f64; DESIGN_COLS],
    pub(crate) yty: f64,
    pub(crate) n_valid: usize,
    pub(crate) cell_counts: [usize; DOSAGE_CELLS],
    pub(crate) cell_y_sums: [f64; DOSAGE_CELLS],
}

impl PairMoments {
    fn new() -> Self {
        Self {
            xtx: [[0.0; DESIGN_COLS]; DESIGN_COLS],
            xty: [0.0; DESIGN_COLS],
            yty: 0.0,
            n_valid: 0,
            cell_counts: [0; DOSAGE_CELLS],
            cell_y_sums: [0.0; DOSAGE_CELLS],
        }
    }

    fn add_observation(
        &mut self,
        encoded_g1: f64,
        encoded_g2: f64,
        raw_g1: f64,
        raw_g2: f64,
        y: f64,
    ) {
        let x = [1.0, encoded_g1, encoded_g2, encoded_g1 * encoded_g2];
        self.n_valid += 1;
        self.yty += y * y;
        for column in 0..DESIGN_COLS {
            self.xty[column] += x[column] * y;
            for row in 0..DESIGN_COLS {
                self.xtx[column][row] += x[column] * x[row];
            }
        }
        if let (Some(g1), Some(g2)) = (integer_dosage_code(raw_g1), integer_dosage_code(raw_g2)) {
            let cell = dosage_cell_index(g1, g2);
            self.cell_counts[cell] += 1;
            self.cell_y_sums[cell] += y;
        }
    }

    #[inline]
    fn add_float_moments(&mut self, encoded_g1: f64, encoded_g2: f64, y: f64) {
        let interaction = encoded_g1 * encoded_g2;
        self.n_valid += 1;
        self.yty += y * y;
        self.xty[0] += y;
        self.xty[1] += encoded_g1 * y;
        self.xty[2] += encoded_g2 * y;
        self.xty[3] += interaction * y;

        self.xtx[0][0] += 1.0;
        self.xtx[0][1] += encoded_g1;
        self.xtx[0][2] += encoded_g2;
        self.xtx[0][3] += interaction;
        self.xtx[1][0] += encoded_g1;
        self.xtx[1][1] += encoded_g1 * encoded_g1;
        self.xtx[1][2] += interaction;
        self.xtx[1][3] += encoded_g1 * interaction;
        self.xtx[2][0] += encoded_g2;
        self.xtx[2][1] += interaction;
        self.xtx[2][2] += encoded_g2 * encoded_g2;
        self.xtx[2][3] += encoded_g2 * interaction;
        self.xtx[3][0] += interaction;
        self.xtx[3][1] += encoded_g1 * interaction;
        self.xtx[3][2] += encoded_g2 * interaction;
        self.xtx[3][3] += interaction * interaction;
    }

    #[inline]
    fn add_float_observation(
        &mut self,
        encoded_g1: f64,
        encoded_g2: f64,
        raw_g1: f64,
        raw_g2: f64,
        y: f64,
    ) {
        self.add_float_moments(encoded_g1, encoded_g2, y);

        if let (Some(g1), Some(g2)) = (integer_dosage_code(raw_g1), integer_dosage_code(raw_g2)) {
            let cell = dosage_cell_index(g1, g2);
            self.cell_counts[cell] += 1;
            self.cell_y_sums[cell] += y;
        }
    }

    #[inline]
    fn add_discrete_cell(
        &mut self,
        encoded_g1: f64,
        encoded_g2: f64,
        raw_g1: usize,
        raw_g2: usize,
        count: usize,
        y_sum: f64,
        y2_sum: f64,
    ) {
        if count == 0 {
            return;
        }
        let count = count as f64;
        let interaction = encoded_g1 * encoded_g2;
        self.n_valid += count as usize;
        self.yty += y2_sum;
        self.xty[0] += y_sum;
        self.xty[1] += encoded_g1 * y_sum;
        self.xty[2] += encoded_g2 * y_sum;
        self.xty[3] += interaction * y_sum;

        // The design columns are [1, g1, g2, g1*g2].  Fill the symmetric
        // cross-product matrix directly; this is called once per genotype
        // cell, rather than once per sample.
        self.xtx[0][0] += count;
        self.xtx[0][1] += count * encoded_g1;
        self.xtx[0][2] += count * encoded_g2;
        self.xtx[0][3] += count * interaction;
        self.xtx[1][0] += count * encoded_g1;
        self.xtx[1][1] += count * encoded_g1 * encoded_g1;
        self.xtx[1][2] += count * encoded_g1 * encoded_g2;
        self.xtx[1][3] += count * encoded_g1 * interaction;
        self.xtx[2][0] += count * encoded_g2;
        self.xtx[2][1] += count * encoded_g2 * encoded_g1;
        self.xtx[2][2] += count * encoded_g2 * encoded_g2;
        self.xtx[2][3] += count * encoded_g2 * interaction;
        self.xtx[3][0] += count * interaction;
        self.xtx[3][1] += count * interaction * encoded_g1;
        self.xtx[3][2] += count * interaction * encoded_g2;
        self.xtx[3][3] += count * interaction * interaction;

        let cell = dosage_cell_index(raw_g1, raw_g2);
        self.cell_counts[cell] += count as usize;
        self.cell_y_sums[cell] += y_sum;
    }
}

fn accumulate_mask(
    moments: &mut PairMoments,
    mask: u64,
    word: usize,
    n_samples: usize,
    y: &[f64],
    representation_g1: &MarkerRepresentation,
    representation_g2: &MarkerRepresentation,
    category_g1: usize,
    category_g2: usize,
) {
    let mut remaining = mask & active_word_mask(n_samples, word);
    while remaining != 0 {
        let offset = remaining.trailing_zeros() as usize;
        let sample = word * 64 + offset;
        if sample >= n_samples {
            break;
        }
        let raw_g1 = raw_category_value(representation_g1, category_g1);
        let raw_g2 = raw_category_value(representation_g2, category_g2);
        let encoded_g1 =
            (raw_g1 - representation_g1.encoding().offset) / representation_g1.encoding().scale;
        let encoded_g2 =
            (raw_g2 - representation_g2.encoding().offset) / representation_g2.encoding().scale;
        moments.add_observation(encoded_g1, encoded_g2, raw_g1, raw_g2, y[sample]);
        remaining &= remaining - 1;
    }
}

fn raw_category_value(representation: &MarkerRepresentation, category: usize) -> f64 {
    match representation {
        MarkerRepresentation::Binary(marker) => marker.levels[category],
        MarkerRepresentation::Ternary(_) => category as f64,
        MarkerRepresentation::Float(_) => unreachable!("float representations have no categories"),
    }
}

fn pair_moments_discrete_complete(
    representation_g1: &MarkerRepresentation,
    representation_g2: &MarkerRepresentation,
    y: &[f64],
) -> Result<PairMoments, String> {
    if !representation_g1.is_discrete()
        || !representation_g2.is_discrete()
        || representation_g1.has_missing()
        || representation_g2.has_missing()
    {
        return Err(
            "complete discrete path requires two discrete missing-free markers".to_string(),
        );
    }
    let n_samples = y.len();
    let mut moments = PairMoments::new();
    for category_g1 in 0..representation_g1.category_count() {
        for category_g2 in 0..representation_g2.category_count() {
            for word in 0..words_for_samples(n_samples) {
                let mask = representation_g1.category_word(category_g1, word)
                    & representation_g2.category_word(category_g2, word);
                accumulate_mask(
                    &mut moments,
                    mask,
                    word,
                    n_samples,
                    y,
                    representation_g1,
                    representation_g2,
                    category_g1,
                    category_g2,
                );
            }
        }
    }
    Ok(moments)
}

/// Missing-free binary×binary moments. For encoded 0/1 columns `x`, `z`,
/// and `w=x*z`, all powers collapse (`x²=x`, `z²=z`, and `w²=w`). Accumulate
/// the four cell counts and phenotype sums directly, then fill the sufficient
/// statistics once. This avoids enum/category dispatch and replaces four
/// disjoint y² lookups with one lookup over the active word.
#[inline]
fn pair_moments_binary_binary_with_lookup(
    marker_g1: &BinaryMarker,
    marker_g2: &BinaryMarker,
    y: &[f64],
    lookup: &MaskYLookup,
) -> Result<PairMoments, String> {
    if marker_g1.n_samples != y.len() || marker_g2.n_samples != y.len() {
        return Err("marker and phenotype lengths do not match".to_string());
    }
    if marker_g1.has_missing || marker_g2.has_missing {
        return Err("binary fast path requires missing-free markers".to_string());
    }
    if lookup.n_words != words_for_samples(y.len()) {
        return Err("phenotype lookup length does not match markers".to_string());
    }

    let mut count_00 = 0usize;
    let mut count_10 = 0usize;
    let mut count_01 = 0usize;
    let mut count_11 = 0usize;
    let mut y_00 = 0.0;
    let mut y_10 = 0.0;
    let mut y_01 = 0.0;
    for word in 0..words_for_samples(y.len()) {
        let active = active_word_mask(y.len(), word);
        let high_1 = marker_g1.high_bits[word] & active;
        let high_2 = marker_g2.high_bits[word] & active;
        let low_1 = !high_1 & active;
        let low_2 = !high_2 & active;
        let mask_00 = low_1 & low_2;
        let mask_10 = high_1 & low_2;
        let mask_01 = low_1 & high_2;
        let mask_11 = high_1 & high_2;

        count_00 += mask_00.count_ones() as usize;
        count_10 += mask_10.count_ones() as usize;
        count_01 += mask_01.count_ones() as usize;
        count_11 += mask_11.count_ones() as usize;
        y_00 += lookup.sum_y(mask_00, word);
        y_10 += lookup.sum_y(mask_10, word);
        y_01 += lookup.sum_y(mask_01, word);
    }

    let n = (count_00 + count_10 + count_01 + count_11) as f64;
    let sx = (count_10 + count_11) as f64;
    let sz = (count_01 + count_11) as f64;
    let sw = count_11 as f64;
    // The cell masks partition all samples on this missing-free path.  Keep
    // only three byte-table y lookups per word and recover the fourth cell and
    // the total y² from the lookup's global totals.
    let y_11 = lookup.total_y - y_00 - y_10 - y_01;
    let sy = lookup.total_y;
    let sxy = y_10 + y_11;
    let szy = y_01 + y_11;
    let swy = y_11;

    let mut cell_counts = [0usize; DOSAGE_CELLS];
    let mut cell_y_sums = [0.0; DOSAGE_CELLS];
    let high_code_g1 = integer_dosage_code(marker_g1.levels[1])
        .expect("binary marker high level must be an integer dosage");
    let low_code_g1 = integer_dosage_code(marker_g1.levels[0])
        .expect("binary marker low level must be an integer dosage");
    let high_code_g2 = integer_dosage_code(marker_g2.levels[1])
        .expect("binary marker high level must be an integer dosage");
    let low_code_g2 = integer_dosage_code(marker_g2.levels[0])
        .expect("binary marker low level must be an integer dosage");
    let cells = [
        (low_code_g1, low_code_g2, count_00, y_00),
        (high_code_g1, low_code_g2, count_10, y_10),
        (low_code_g1, high_code_g2, count_01, y_01),
        (high_code_g1, high_code_g2, count_11, y_11),
    ];
    for (g1, g2, count, y_sum) in cells {
        let cell = dosage_cell_index(g1, g2);
        cell_counts[cell] = count;
        cell_y_sums[cell] = y_sum;
    }

    Ok(PairMoments {
        xtx: [
            [n, sx, sz, sw],
            [sx, sx, sw, sw],
            [sz, sw, sz, sw],
            [sw, sw, sw, sw],
        ],
        xty: [sy, sxy, szy, swy],
        yty: lookup.total_y2,
        n_valid: n as usize,
        cell_counts,
        cell_y_sums,
    })
}

/// Build moments from a small, complete genotype-cell table.  The caller has
/// already accumulated the category intersections, so this helper performs
/// only the fixed-width (at most nine cells) reconstruction of the saturated
/// design.  In particular, it never visits samples and it does not perform a
/// y² lookup for each cell: on a missing-free path `total_y2` is known once.
#[inline]
fn pair_moments_from_discrete_cells(
    encoded_levels_g1: &[f64],
    raw_codes_g1: &[usize],
    encoded_levels_g2: &[f64],
    raw_codes_g2: &[usize],
    cell_counts: &[usize],
    cell_y_sums: &[f64],
    total_y2: f64,
) -> PairMoments {
    debug_assert_eq!(
        cell_counts.len(),
        encoded_levels_g1.len() * encoded_levels_g2.len()
    );
    debug_assert_eq!(cell_y_sums.len(), cell_counts.len());
    let mut moments = PairMoments::new();
    for (cell, (&count, &y_sum)) in cell_counts.iter().zip(cell_y_sums).enumerate() {
        let g1_index = cell / encoded_levels_g2.len();
        let g2_index = cell % encoded_levels_g2.len();
        moments.add_discrete_cell(
            encoded_levels_g1[g1_index],
            encoded_levels_g2[g2_index],
            raw_codes_g1[g1_index],
            raw_codes_g2[g2_index],
            count,
            y_sum,
            0.0,
        );
    }
    moments.yty = total_y2;
    moments
}

/// Missing-free binary×ternary moments.  The binary marker contributes one
/// non-zero bit plane and the ternary marker contributes two.  Only the two
/// non-zero intersections are formed; cells containing zero are recovered
/// from the two marker marginals.  This keeps the hot loop at two bitset
/// intersections while retaining the complete cell table for materialized
/// results.
#[inline]
fn pair_moments_binary_ternary_with_lookup(
    marker_g1: &BinaryMarker,
    marker_g2: &TernaryMarker,
    y: &[f64],
    lookup: &MaskYLookup,
) -> Result<PairMoments, String> {
    if marker_g1.n_samples != y.len() || marker_g2.n_samples != y.len() {
        return Err("marker and phenotype lengths do not match".to_string());
    }
    if marker_g1.has_missing || marker_g2.has_missing {
        return Err("binary×ternary fast path requires missing-free markers".to_string());
    }
    if lookup.n_words != words_for_samples(y.len()) {
        return Err("phenotype lookup length does not match markers".to_string());
    }

    let mut cell_counts = [0usize; 2 * DOSAGE_LEVELS];
    let mut cell_y_sums = [0.0; 2 * DOSAGE_LEVELS];
    let mut binary_counts = [0usize; 2];
    let mut binary_y_sums = [0.0; 2];
    let mut ternary_counts = [0usize; DOSAGE_LEVELS];
    let mut ternary_y_sums = [0.0; DOSAGE_LEVELS];
    for word in 0..words_for_samples(y.len()) {
        let active = active_word_mask(y.len(), word);
        let binary_high = marker_g1.high_bits[word] & active;
        let ternary_one = marker_g2.plane_one[word];
        let ternary_two = marker_g2.plane_two[word];
        let ternary_one = ternary_one & active;
        let ternary_two = ternary_two & active;
        let intersections = [binary_high & ternary_one, binary_high & ternary_two];
        let intersection_counts = [
            intersections[0].count_ones() as usize,
            intersections[1].count_ones() as usize,
        ];
        let intersection_y_sums = [
            lookup.sum_y(intersections[0], word),
            lookup.sum_y(intersections[1], word),
        ];
        cell_counts[dosage_cell_index(1, 1)] += intersection_counts[0];
        cell_counts[dosage_cell_index(1, 2)] += intersection_counts[1];
        cell_y_sums[dosage_cell_index(1, 1)] += intersection_y_sums[0];
        cell_y_sums[dosage_cell_index(1, 2)] += intersection_y_sums[1];

        let binary_high_count = binary_high.count_ones() as usize;
        binary_counts[1] += binary_high_count;
        binary_y_sums[1] += lookup.sum_y(binary_high, word);
        let active_count = active.count_ones() as usize;
        // The zero category is restored from the global complement after
        // all words have been visited.  Do not subtract from total_y per
        // word: total_y spans the complete phenotype, not this word.
        debug_assert!(active_count >= binary_high_count);

        let ternary_one_count = ternary_one.count_ones() as usize;
        let ternary_two_count = ternary_two.count_ones() as usize;
        ternary_counts[1] += ternary_one_count;
        ternary_counts[2] += ternary_two_count;
        ternary_y_sums[1] += lookup.sum_y(ternary_one, word);
        ternary_y_sums[2] += lookup.sum_y(ternary_two, word);
        debug_assert!(active_count >= ternary_one_count + ternary_two_count);
    }

    binary_counts[0] = y.len() - binary_counts[1];
    binary_y_sums[0] = lookup.total_y - binary_y_sums[1];
    ternary_counts[0] = y.len() - ternary_counts[1] - ternary_counts[2];
    ternary_y_sums[0] = lookup.total_y - ternary_y_sums[1] - ternary_y_sums[2];

    // Recover all cells involving at least one zero from marker marginals.
    // The four direct intersections above provide the non-zero/non-zero
    // cells; the remaining values are exact complements on this
    // missing-free path.
    cell_counts[dosage_cell_index(1, 0)] = binary_counts[1]
        - cell_counts[dosage_cell_index(1, 1)]
        - cell_counts[dosage_cell_index(1, 2)];
    cell_counts[dosage_cell_index(0, 1)] = ternary_counts[1] - cell_counts[dosage_cell_index(1, 1)];
    cell_counts[dosage_cell_index(0, 2)] = ternary_counts[2] - cell_counts[dosage_cell_index(1, 2)];
    cell_counts[dosage_cell_index(0, 0)] = y.len()
        - cell_counts[dosage_cell_index(1, 0)]
        - cell_counts[dosage_cell_index(1, 1)]
        - cell_counts[dosage_cell_index(1, 2)]
        - cell_counts[dosage_cell_index(0, 1)]
        - cell_counts[dosage_cell_index(0, 2)];
    cell_y_sums[dosage_cell_index(1, 0)] = binary_y_sums[1]
        - cell_y_sums[dosage_cell_index(1, 1)]
        - cell_y_sums[dosage_cell_index(1, 2)];
    cell_y_sums[dosage_cell_index(0, 1)] = ternary_y_sums[1] - cell_y_sums[dosage_cell_index(1, 1)];
    cell_y_sums[dosage_cell_index(0, 2)] = ternary_y_sums[2] - cell_y_sums[dosage_cell_index(1, 2)];
    cell_y_sums[dosage_cell_index(0, 0)] = lookup.total_y
        - cell_y_sums[dosage_cell_index(1, 0)]
        - cell_y_sums[dosage_cell_index(1, 1)]
        - cell_y_sums[dosage_cell_index(1, 2)]
        - cell_y_sums[dosage_cell_index(0, 1)]
        - cell_y_sums[dosage_cell_index(0, 2)];

    let raw_codes_g1 = [
        integer_dosage_code(marker_g1.levels[0])
            .expect("binary marker low level must be an integer dosage"),
        integer_dosage_code(marker_g1.levels[1])
            .expect("binary marker high level must be an integer dosage"),
    ];
    let encoded_levels_g1 = [
        (marker_g1.levels[0] - marker_g1.encoding.offset) / marker_g1.encoding.scale,
        (marker_g1.levels[1] - marker_g1.encoding.offset) / marker_g1.encoding.scale,
    ];
    let encoded_levels_g2 = [0.0, 1.0, 2.0];
    let raw_codes_g2 = [0usize, 1usize, 2usize];
    Ok(pair_moments_from_discrete_cells(
        &encoded_levels_g1,
        &raw_codes_g1,
        &encoded_levels_g2,
        &raw_codes_g2,
        &cell_counts,
        &cell_y_sums,
        lookup.total_y2,
    ))
}

/// Missing-free ternary×binary moments.  Reusing the binary×ternary kernel
/// keeps one hot loop while swapping the two design columns and cell axes in
/// a fixed-size post-processing step.
#[inline]
fn pair_moments_ternary_binary_with_lookup(
    marker_g1: &TernaryMarker,
    marker_g2: &BinaryMarker,
    y: &[f64],
    lookup: &MaskYLookup,
) -> Result<PairMoments, String> {
    let moments = pair_moments_binary_ternary_with_lookup(marker_g2, marker_g1, y, lookup)?;
    Ok(swap_pair_moments(moments))
}

/// Swap the two marker columns in a saturated two-locus moment table.  This
/// is used only for the ternary×binary orientation and is intentionally
/// allocation-free.
#[inline]
fn swap_pair_moments(moments: PairMoments) -> PairMoments {
    let permutation = [0usize, 2, 1, 3];
    let old_xtx = moments.xtx;
    let old_xty = moments.xty;
    let xtx = std::array::from_fn(|row| {
        std::array::from_fn(|column| old_xtx[permutation[row]][permutation[column]])
    });
    let xty = std::array::from_fn(|column| old_xty[permutation[column]]);
    let mut cell_counts = [0usize; DOSAGE_CELLS];
    let mut cell_y_sums = [0.0; DOSAGE_CELLS];
    for g1 in 0..DOSAGE_LEVELS {
        for g2 in 0..DOSAGE_LEVELS {
            let target = dosage_cell_index(g1, g2);
            let source = dosage_cell_index(g2, g1);
            cell_counts[target] = moments.cell_counts[source];
            cell_y_sums[target] = moments.cell_y_sums[source];
        }
    }
    PairMoments {
        xtx,
        xty,
        yty: moments.yty,
        n_valid: moments.n_valid,
        cell_counts,
        cell_y_sums,
    }
}

/// Missing-free ternary×ternary moments.  The pair hot loop consists of four
/// direct non-zero bit-plane intersections.  Cells containing zero are
/// recovered from marker marginals, so the complete cell table is retained
/// without the nine-way category dispatch used by the generic fallback.
#[inline]
fn pair_moments_ternary_ternary_with_lookup(
    marker_g1: &TernaryMarker,
    marker_g2: &TernaryMarker,
    y: &[f64],
    lookup: &MaskYLookup,
) -> Result<PairMoments, String> {
    if marker_g1.n_samples != y.len() || marker_g2.n_samples != y.len() {
        return Err("marker and phenotype lengths do not match".to_string());
    }
    if marker_g1.has_missing || marker_g2.has_missing {
        return Err("ternary fast path requires missing-free markers".to_string());
    }
    if lookup.n_words != words_for_samples(y.len()) {
        return Err("phenotype lookup length does not match markers".to_string());
    }

    let mut cell_counts = [0usize; DOSAGE_CELLS];
    let mut cell_y_sums = [0.0; DOSAGE_CELLS];
    let mut g1_counts = [0usize; DOSAGE_LEVELS];
    let mut g1_y_sums = [0.0; DOSAGE_LEVELS];
    let mut g2_counts = [0usize; DOSAGE_LEVELS];
    let mut g2_y_sums = [0.0; DOSAGE_LEVELS];
    for word in 0..words_for_samples(y.len()) {
        let active = active_word_mask(y.len(), word);
        let one_g1 = marker_g1.plane_one[word];
        let two_g1 = marker_g1.plane_two[word];
        let one_g2 = marker_g2.plane_one[word];
        let two_g2 = marker_g2.plane_two[word];
        let one_g1 = one_g1 & active;
        let two_g1 = two_g1 & active;
        let one_g2 = one_g2 & active;
        let two_g2 = two_g2 & active;
        let intersections = [
            one_g1 & one_g2,
            one_g1 & two_g2,
            two_g1 & one_g2,
            two_g1 & two_g2,
        ];
        let intersection_cells = [
            dosage_cell_index(1, 1),
            dosage_cell_index(1, 2),
            dosage_cell_index(2, 1),
            dosage_cell_index(2, 2),
        ];
        for (mask, cell) in intersections.into_iter().zip(intersection_cells) {
            cell_counts[cell] += mask.count_ones() as usize;
            cell_y_sums[cell] += lookup.sum_y(mask, word);
        }

        let masks_g1 = [one_g1, two_g1];
        let masks_g2 = [one_g2, two_g2];
        for (offset, mask) in masks_g1.into_iter().enumerate() {
            let count = mask.count_ones() as usize;
            g1_counts[offset + 1] += count;
            g1_y_sums[offset + 1] += lookup.sum_y(mask, word);
        }
        for (offset, mask) in masks_g2.into_iter().enumerate() {
            let count = mask.count_ones() as usize;
            g2_counts[offset + 1] += count;
            g2_y_sums[offset + 1] += lookup.sum_y(mask, word);
        }
    }

    g1_counts[0] = y.len() - g1_counts[1] - g1_counts[2];
    g2_counts[0] = y.len() - g2_counts[1] - g2_counts[2];
    g1_y_sums[0] = lookup.total_y - g1_y_sums[1] - g1_y_sums[2];
    g2_y_sums[0] = lookup.total_y - g2_y_sums[1] - g2_y_sums[2];

    cell_counts[dosage_cell_index(1, 0)] =
        g1_counts[1] - cell_counts[dosage_cell_index(1, 1)] - cell_counts[dosage_cell_index(1, 2)];
    cell_counts[dosage_cell_index(2, 0)] =
        g1_counts[2] - cell_counts[dosage_cell_index(2, 1)] - cell_counts[dosage_cell_index(2, 2)];
    cell_counts[dosage_cell_index(0, 1)] =
        g2_counts[1] - cell_counts[dosage_cell_index(1, 1)] - cell_counts[dosage_cell_index(2, 1)];
    cell_counts[dosage_cell_index(0, 2)] =
        g2_counts[2] - cell_counts[dosage_cell_index(1, 2)] - cell_counts[dosage_cell_index(2, 2)];
    cell_counts[dosage_cell_index(0, 0)] = y.len()
        - cell_counts[dosage_cell_index(1, 0)]
        - cell_counts[dosage_cell_index(1, 1)]
        - cell_counts[dosage_cell_index(1, 2)]
        - cell_counts[dosage_cell_index(2, 0)]
        - cell_counts[dosage_cell_index(2, 1)]
        - cell_counts[dosage_cell_index(2, 2)]
        - cell_counts[dosage_cell_index(0, 1)]
        - cell_counts[dosage_cell_index(0, 2)];
    cell_y_sums[dosage_cell_index(1, 0)] =
        g1_y_sums[1] - cell_y_sums[dosage_cell_index(1, 1)] - cell_y_sums[dosage_cell_index(1, 2)];
    cell_y_sums[dosage_cell_index(2, 0)] =
        g1_y_sums[2] - cell_y_sums[dosage_cell_index(2, 1)] - cell_y_sums[dosage_cell_index(2, 2)];
    cell_y_sums[dosage_cell_index(0, 1)] =
        g2_y_sums[1] - cell_y_sums[dosage_cell_index(1, 1)] - cell_y_sums[dosage_cell_index(2, 1)];
    cell_y_sums[dosage_cell_index(0, 2)] =
        g2_y_sums[2] - cell_y_sums[dosage_cell_index(1, 2)] - cell_y_sums[dosage_cell_index(2, 2)];
    cell_y_sums[dosage_cell_index(0, 0)] = lookup.total_y
        - cell_y_sums[dosage_cell_index(1, 0)]
        - cell_y_sums[dosage_cell_index(1, 1)]
        - cell_y_sums[dosage_cell_index(1, 2)]
        - cell_y_sums[dosage_cell_index(2, 0)]
        - cell_y_sums[dosage_cell_index(2, 1)]
        - cell_y_sums[dosage_cell_index(2, 2)]
        - cell_y_sums[dosage_cell_index(0, 1)]
        - cell_y_sums[dosage_cell_index(0, 2)];
    let encoded_levels = [0.0, 1.0, 2.0];
    let raw_codes = [0usize, 1usize, 2usize];
    Ok(pair_moments_from_discrete_cells(
        &encoded_levels,
        &raw_codes,
        &encoded_levels,
        &raw_codes,
        &cell_counts,
        &cell_y_sums,
        lookup.total_y2,
    ))
}

/// Accumulate discrete moments from category intersections using a shared
/// phenotype lookup table.  The lookup path is deliberately separate from the
/// scalar reference implementation above: it preserves the missing-free and
/// pairwise-missing semantics while replacing one set-bit visit per sample by
/// eight cached byte sums per 64-sample word.
fn pair_moments_discrete_with_lookup(
    representation_g1: &MarkerRepresentation,
    representation_g2: &MarkerRepresentation,
    y: &[f64],
    lookup: &MaskYLookup,
    masked: bool,
) -> Result<PairMoments, String> {
    if !representation_g1.is_discrete() || !representation_g2.is_discrete() {
        return Err("lookup path requires two discrete markers".to_string());
    }
    if lookup.n_words != words_for_samples(y.len()) {
        return Err("phenotype lookup length does not match markers".to_string());
    }

    let n_samples = y.len();
    let n_words = words_for_samples(n_samples);
    let mut moments = PairMoments::new();
    for category_g1 in 0..representation_g1.category_count() {
        let raw_g1 = raw_category_value(representation_g1, category_g1);
        let raw_code_g1 = integer_dosage_code(raw_g1)
            .ok_or_else(|| "discrete marker category is not an integer dosage".to_string())?;
        let encoding_g1 = representation_g1.encoding();
        let encoded_g1 = (raw_g1 - encoding_g1.offset) / encoding_g1.scale;
        for category_g2 in 0..representation_g2.category_count() {
            let raw_g2 = raw_category_value(representation_g2, category_g2);
            let raw_code_g2 = integer_dosage_code(raw_g2)
                .ok_or_else(|| "discrete marker category is not an integer dosage".to_string())?;
            let encoding_g2 = representation_g2.encoding();
            let encoded_g2 = (raw_g2 - encoding_g2.offset) / encoding_g2.scale;
            let mut count = 0usize;
            let mut y_sum = 0.0;
            let mut y2_sum = 0.0;
            for word in 0..n_words {
                let mut mask = representation_g1.category_word(category_g1, word)
                    & representation_g2.category_word(category_g2, word);
                if masked {
                    mask &= !representation_g1.missing_word(word)
                        & !representation_g2.missing_word(word)
                        & active_word_mask(n_samples, word);
                }
                count += mask.count_ones() as usize;
                y_sum += lookup.sum_y(mask, word);
                y2_sum += lookup.sum_y2(mask, word);
            }
            moments.add_discrete_cell(
                encoded_g1,
                encoded_g2,
                raw_code_g1,
                raw_code_g2,
                count,
                y_sum,
                y2_sum,
            );
        }
    }
    Ok(moments)
}

fn pair_moments_discrete_masked(
    representation_g1: &MarkerRepresentation,
    representation_g2: &MarkerRepresentation,
    y: &[f64],
) -> Result<PairMoments, String> {
    if !representation_g1.is_discrete() || !representation_g2.is_discrete() {
        return Err("masked discrete path requires two discrete markers".to_string());
    }
    let n_samples = y.len();
    let mut moments = PairMoments::new();
    for category_g1 in 0..representation_g1.category_count() {
        for category_g2 in 0..representation_g2.category_count() {
            for word in 0..words_for_samples(n_samples) {
                let valid = active_word_mask(n_samples, word)
                    & !representation_g1.missing_word(word)
                    & !representation_g2.missing_word(word);
                let mask = representation_g1.category_word(category_g1, word)
                    & representation_g2.category_word(category_g2, word)
                    & valid;
                accumulate_mask(
                    &mut moments,
                    mask,
                    word,
                    n_samples,
                    y,
                    representation_g1,
                    representation_g2,
                    category_g1,
                    category_g2,
                );
            }
        }
    }
    Ok(moments)
}

fn pair_moments_float_generic(
    representation_g1: &MarkerRepresentation,
    representation_g2: &MarkerRepresentation,
    y: &[f64],
) -> Result<PairMoments, String> {
    let mut moments = PairMoments::new();
    let encoding_g1 = representation_g1.encoding();
    let encoding_g2 = representation_g2.encoding();
    for sample in 0..y.len() {
        if representation_g1.missing_at(sample) || representation_g2.missing_at(sample) {
            continue;
        }
        let raw_g1 = representation_g1.value_at(sample);
        let raw_g2 = representation_g2.value_at(sample);
        let encoded_g1 = (raw_g1 - encoding_g1.offset) / encoding_g1.scale;
        let encoded_g2 = (raw_g2 - encoding_g2.offset) / encoding_g2.scale;
        moments.add_float_observation(encoded_g1, encoded_g2, raw_g1, raw_g2, y[sample]);
    }
    Ok(moments)
}

/// Specialized float×float loop used by the hot pair scan.  Matching the
/// enum variants once outside the sample loop avoids three virtual-looking
/// dispatches (`missing_at`, `value_at`, and `encoding`) for every sample.
#[inline]
fn pair_moments_float_float(
    marker_g1: &FloatMarker,
    marker_g2: &FloatMarker,
    y: &[f64],
) -> PairMoments {
    debug_assert_eq!(marker_g1.n_samples, y.len());
    debug_assert_eq!(marker_g2.n_samples, y.len());
    let mut moments = PairMoments::new();
    let encoding_g1 = marker_g1.encoding;
    let encoding_g2 = marker_g2.encoding;
    if marker_g1.has_integer_values && marker_g2.has_integer_values {
        for sample in 0..y.len() {
            if marker_g1.missing[sample] || marker_g2.missing[sample] {
                continue;
            }
            let raw_g1 = marker_g1.values[sample];
            let raw_g2 = marker_g2.values[sample];
            let encoded_g1 = (raw_g1 - encoding_g1.offset) / encoding_g1.scale;
            let encoded_g2 = (raw_g2 - encoding_g2.offset) / encoding_g2.scale;
            moments.add_float_observation(encoded_g1, encoded_g2, raw_g1, raw_g2, y[sample]);
        }
    } else {
        for sample in 0..y.len() {
            if marker_g1.missing[sample] || marker_g2.missing[sample] {
                continue;
            }
            let raw_g1 = marker_g1.values[sample];
            let raw_g2 = marker_g2.values[sample];
            let encoded_g1 = (raw_g1 - encoding_g1.offset) / encoding_g1.scale;
            let encoded_g2 = (raw_g2 - encoding_g2.offset) / encoding_g2.scale;
            moments.add_float_moments(encoded_g1, encoded_g2, y[sample]);
        }
    }
    moments
}

fn pair_moments_float(
    representation_g1: &MarkerRepresentation,
    representation_g2: &MarkerRepresentation,
    y: &[f64],
) -> Result<PairMoments, String> {
    if let (MarkerRepresentation::Float(marker_g1), MarkerRepresentation::Float(marker_g2)) =
        (representation_g1, representation_g2)
    {
        return Ok(pair_moments_float_float(marker_g1, marker_g2, y));
    }
    pair_moments_float_generic(representation_g1, representation_g2, y)
}

#[inline]
fn dgemm_row_major_f64(
    left: &[f64],
    rows: usize,
    inner: usize,
    right: &[f64],
    columns: usize,
    out: &mut [f64],
) {
    debug_assert!(left.len() >= rows.saturating_mul(inner));
    debug_assert!(right.len() >= columns.saturating_mul(inner));
    debug_assert!(out.len() >= rows.saturating_mul(columns));
    if rows == 0 || inner == 0 || columns == 0 {
        return;
    }

    // `right` is stored as [column][sample], so exposing its transpose to a
    // row-major GEMM computes left[sample] × right[sample] for every pair of
    // marker rows.  BLAS is used where available; matrixmultiply provides the
    // same stride-aware operation on other targets.
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    unsafe {
        cblas_dgemm_dispatch(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_TRANS,
            rows as CblasInt,
            columns as CblasInt,
            inner as CblasInt,
            1.0,
            left.as_ptr(),
            inner as CblasInt,
            right.as_ptr(),
            inner as CblasInt,
            0.0,
            out.as_mut_ptr(),
            columns as CblasInt,
        );
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    unsafe {
        matrixmultiply::dgemm(
            rows,
            inner,
            columns,
            1.0,
            left.as_ptr(),
            inner as isize,
            1,
            right.as_ptr(),
            1,
            inner as isize,
            0.0,
            out.as_mut_ptr(),
            columns as isize,
            1,
        );
    }
}

#[inline]
fn fill_float_block(
    markers: &[&FloatMarker],
    start: usize,
    end: usize,
    n_samples: usize,
    y: &[f64],
    values: &mut [f64],
    squares: &mut [f64],
    mut weighted: Option<&mut [f64]>,
) {
    let rows = end - start;
    debug_assert!(values.len() >= rows.saturating_mul(n_samples));
    debug_assert!(squares.len() >= rows.saturating_mul(n_samples));
    if let Some(weighted) = weighted.as_ref() {
        debug_assert!(weighted.len() >= rows.saturating_mul(n_samples));
    }
    for (local_row, marker) in markers[start..end].iter().copied().enumerate() {
        let offset = marker.encoding.offset;
        let scale = marker.encoding.scale;
        let row_offset = local_row * n_samples;
        for sample in 0..n_samples {
            let value = (marker.values[sample] - offset) / scale;
            values[row_offset + sample] = value;
            squares[row_offset + sample] = value * value;
            if let Some(weighted) = weighted.as_deref_mut() {
                weighted[row_offset + sample] = value * y[sample];
            }
        }
    }
}

#[inline]
fn float_marker_summaries(markers: &[&FloatMarker], y: &[f64]) -> Vec<[f64; 3]> {
    markers
        .iter()
        .map(|marker| {
            let mut sx = 0.0;
            let mut sxx = 0.0;
            let mut sxy = 0.0;
            for sample in 0..y.len() {
                let value =
                    (marker.values[sample] - marker.encoding.offset) / marker.encoding.scale;
                sx += value;
                sxx += value * value;
                sxy += value * y[sample];
            }
            [sx, sxx, sxy]
        })
        .collect()
}

#[inline]
fn float_pair_moments_from_products(
    summary_g1: [f64; 3],
    summary_g2: [f64; 3],
    n_samples: usize,
    sum_g1g2: f64,
    sum_g1sq_g2: f64,
    sum_g1_g2sq: f64,
    sum_g1sq_g2sq: f64,
    sum_y_g1g2: f64,
    total_y: f64,
    total_y2: f64,
) -> PairMoments {
    let (sx, sxx, sxy) = (summary_g1[0], summary_g1[1], summary_g1[2]);
    let (sz, szz, szy) = (summary_g2[0], summary_g2[1], summary_g2[2]);
    let sw = sum_g1g2;
    PairMoments {
        xtx: [
            [n_samples as f64, sx, sz, sw],
            [sx, sxx, sw, sum_g1sq_g2],
            [sz, sw, szz, sum_g1_g2sq],
            [sw, sum_g1sq_g2, sum_g1_g2sq, sum_g1sq_g2sq],
        ],
        xty: [total_y, sxy, szy, sum_y_g1g2],
        yty: total_y2,
        n_valid: n_samples,
        cell_counts: [0; DOSAGE_CELLS],
        cell_y_sums: [0.0; DOSAGE_CELLS],
    }
}

/// Scan complete float markers by marker blocks.  Five GEMMs provide all
/// pair moments (`xz`, `x²z`, `xz²`, `x²z²`, and `yxz`) for a block; the
/// scalar FWL scorer and bounded Top-K heap then run once per pair.  Product
/// blocks are discarded before moving to the next tile, so memory is O(BN)
/// rather than O(M²).
fn scan_float_pairs_blocked(
    markers: &[&FloatMarker],
    n_samples: usize,
    y: &[f64],
    top_k: usize,
) -> Result<PairScanAccumulator, String> {
    if y.len() != n_samples {
        return Err(format!("y length={} but expected {n_samples}", y.len()));
    }
    if markers
        .iter()
        .any(|marker| marker.n_samples != n_samples || marker.has_missing)
    {
        return Err("blocked float path requires complete markers".to_string());
    }
    let mut accumulator = PairScanAccumulator::default();
    if markers.len() < 2 {
        return Ok(accumulator);
    }

    let summaries = float_marker_summaries(markers, y);
    let block = FLOAT_GEMM_BLOCK_MARKERS;
    let mut left = vec![0.0; block * n_samples];
    let mut left_squares = vec![0.0; block * n_samples];
    let mut right = vec![0.0; block * n_samples];
    let mut right_squares = vec![0.0; block * n_samples];
    let mut right_weighted = vec![0.0; block * n_samples];
    let mut product_xz = vec![0.0; block * block];
    let mut product_x2z = vec![0.0; block * block];
    let mut product_xz2 = vec![0.0; block * block];
    let mut product_x2z2 = vec![0.0; block * block];
    let mut product_yxz = vec![0.0; block * block];
    let total_y = y.iter().copied().sum::<f64>();
    let total_y2 = y.iter().map(|value| value * value).sum::<f64>();

    // Avoid nested BLAS threading: this routine is currently a single
    // blocked stream, while the discrete scanner owns its Rayon parallelism.
    #[cfg(any(target_os = "macos", target_os = "linux", target_os = "windows"))]
    let _blas_guard = BlasThreadGuard::enter(1);

    for left_start in (0..markers.len()).step_by(block) {
        let left_end = (left_start + block).min(markers.len());
        let left_rows = left_end - left_start;
        fill_float_block(
            markers,
            left_start,
            left_end,
            n_samples,
            y,
            &mut left[..left_rows * n_samples],
            &mut left_squares[..left_rows * n_samples],
            None,
        );
        for right_start in (left_start..markers.len()).step_by(block) {
            let right_end = (right_start + block).min(markers.len());
            let right_rows = right_end - right_start;
            fill_float_block(
                markers,
                right_start,
                right_end,
                n_samples,
                y,
                &mut right[..right_rows * n_samples],
                &mut right_squares[..right_rows * n_samples],
                Some(&mut right_weighted[..right_rows * n_samples]),
            );
            let product_len = left_rows * right_rows;
            dgemm_row_major_f64(
                &left[..left_rows * n_samples],
                left_rows,
                n_samples,
                &right[..right_rows * n_samples],
                right_rows,
                &mut product_xz[..product_len],
            );
            dgemm_row_major_f64(
                &left_squares[..left_rows * n_samples],
                left_rows,
                n_samples,
                &right[..right_rows * n_samples],
                right_rows,
                &mut product_x2z[..product_len],
            );
            dgemm_row_major_f64(
                &left[..left_rows * n_samples],
                left_rows,
                n_samples,
                &right_squares[..right_rows * n_samples],
                right_rows,
                &mut product_xz2[..product_len],
            );
            dgemm_row_major_f64(
                &left_squares[..left_rows * n_samples],
                left_rows,
                n_samples,
                &right_squares[..right_rows * n_samples],
                right_rows,
                &mut product_x2z2[..product_len],
            );
            dgemm_row_major_f64(
                &left[..left_rows * n_samples],
                left_rows,
                n_samples,
                &right_weighted[..right_rows * n_samples],
                right_rows,
                &mut product_yxz[..product_len],
            );

            for local_left in 0..left_rows {
                let first = left_start + local_left;
                let first_product = local_left * right_rows;
                for local_right in 0..right_rows {
                    let second = right_start + local_right;
                    if first >= second {
                        continue;
                    }
                    let product = first_product + local_right;
                    let moments = float_pair_moments_from_products(
                        summaries[first],
                        summaries[second],
                        n_samples,
                        product_xz[product],
                        product_x2z[product],
                        product_xz2[product],
                        product_x2z2[product],
                        product_yxz[product],
                        total_y,
                        total_y2,
                    );
                    let score = match score_pair_moments_scalar(&moments) {
                        Ok(score) => score,
                        Err(error) if error.contains("unidentifiable") => {
                            accumulator.pairs_skipped = accumulator.pairs_skipped.saturating_add(1);
                            continue;
                        }
                        Err(error) => return Err(error),
                    };
                    accumulator.pairs_evaluated = accumulator.pairs_evaluated.saturating_add(1);
                    push_top_k_score(
                        &mut accumulator.heap,
                        ScoredPair {
                            first,
                            second,
                            score,
                        },
                        top_k,
                    );
                }
            }
        }
    }
    Ok(accumulator)
}

#[cfg(test)]
fn pair_moments(
    representation_g1: &MarkerRepresentation,
    representation_g2: &MarkerRepresentation,
    y: &[f64],
) -> Result<PairMoments, String> {
    pair_moments_with_lookup(representation_g1, representation_g2, y, None)
}

fn pair_moments_with_lookup(
    representation_g1: &MarkerRepresentation,
    representation_g2: &MarkerRepresentation,
    y: &[f64],
    lookup: Option<&MaskYLookup>,
) -> Result<PairMoments, String> {
    if representation_g1.n_samples() != y.len() || representation_g2.n_samples() != y.len() {
        return Err("marker and phenotype lengths do not match".to_string());
    }
    if representation_g1.is_discrete() && representation_g2.is_discrete() {
        if let Some(lookup) = lookup {
            let masked = representation_g1.has_missing() || representation_g2.has_missing();
            if !masked {
                match (representation_g1, representation_g2) {
                    (
                        MarkerRepresentation::Binary(marker_g1),
                        MarkerRepresentation::Binary(marker_g2),
                    ) => {
                        // Binary01 is the frozen fast path; keep its behavior
                        // and numerical ordering unchanged.
                        return pair_moments_binary_binary_with_lookup(
                            marker_g1, marker_g2, y, lookup,
                        );
                    }
                    (
                        MarkerRepresentation::Binary(marker_g1),
                        MarkerRepresentation::Ternary(marker_g2),
                    ) => {
                        return pair_moments_binary_ternary_with_lookup(
                            marker_g1, marker_g2, y, lookup,
                        );
                    }
                    (
                        MarkerRepresentation::Ternary(marker_g1),
                        MarkerRepresentation::Binary(marker_g2),
                    ) => {
                        return pair_moments_ternary_binary_with_lookup(
                            marker_g1, marker_g2, y, lookup,
                        );
                    }
                    (
                        MarkerRepresentation::Ternary(marker_g1),
                        MarkerRepresentation::Ternary(marker_g2),
                    ) => {
                        return pair_moments_ternary_ternary_with_lookup(
                            marker_g1, marker_g2, y, lookup,
                        );
                    }
                    _ => unreachable!("discrete representation variants are exhaustive"),
                }
            }
            return pair_moments_discrete_with_lookup(
                representation_g1,
                representation_g2,
                y,
                lookup,
                masked,
            );
        }
        if !representation_g1.has_missing() && !representation_g2.has_missing() {
            pair_moments_discrete_complete(representation_g1, representation_g2, y)
        } else {
            pair_moments_discrete_masked(representation_g1, representation_g2, y)
        }
    } else {
        pair_moments_float(representation_g1, representation_g2, y)
    }
}

#[cfg(test)]
fn flatten_matrix(matrix: &[[f64; DESIGN_COLS]; DESIGN_COLS]) -> Vec<f64> {
    matrix.iter().flat_map(|row| row.iter().copied()).collect()
}

#[cfg(test)]
fn solve_linear(matrix: &[f64], rhs: &[f64], p: usize) -> Result<Vec<f64>, String> {
    if matrix.len() != p * p || rhs.len() != p {
        return Err("linear-system dimensions do not match".to_string());
    }
    let mut left = matrix.to_vec();
    let mut right = rhs.to_vec();
    let matrix_scale = left
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max)
        .max(1.0);
    let pivot_tol = SOLVE_EPS * matrix_scale;
    for column in 0..p {
        let mut pivot = column;
        let mut pivot_abs = left[column * p + column].abs();
        for row in (column + 1)..p {
            let value = left[row * p + column].abs();
            if value > pivot_abs {
                pivot = row;
                pivot_abs = value;
            }
        }
        if !(pivot_abs > pivot_tol) || !pivot_abs.is_finite() {
            return Err(format!("singular design matrix at column {column}"));
        }
        if pivot != column {
            for index in 0..p {
                left.swap(column * p + index, pivot * p + index);
            }
            right.swap(column, pivot);
        }
        let diagonal = left[column * p + column];
        for row in (column + 1)..p {
            let factor = left[row * p + column] / diagonal;
            if factor == 0.0 {
                continue;
            }
            left[row * p + column] = 0.0;
            for index in (column + 1)..p {
                left[row * p + index] -= factor * left[column * p + index];
            }
            right[row] -= factor * right[column];
        }
    }
    let mut solution = vec![0.0; p];
    for row in (0..p).rev() {
        let tail = ((row + 1)..p)
            .map(|column| left[row * p + column] * solution[column])
            .sum::<f64>();
        let diagonal = left[row * p + row];
        if !(diagonal.abs() > pivot_tol) || !diagonal.is_finite() {
            return Err(format!("singular design matrix at column {row}"));
        }
        solution[row] = (right[row] - tail) / diagonal;
    }
    if solution.iter().any(|value| !value.is_finite()) {
        return Err("linear-system solution is not finite".to_string());
    }
    Ok(solution)
}

#[cfg(test)]
fn rss_from_moments(moments: &PairMoments, beta: &[f64]) -> f64 {
    let fitted_cross = beta
        .iter()
        .enumerate()
        .map(|(column, coefficient)| coefficient * moments.xty[column])
        .sum::<f64>();
    let quadratic = beta
        .iter()
        .enumerate()
        .map(|(column, coefficient)| {
            coefficient
                * beta
                    .iter()
                    .enumerate()
                    .map(|(row, other)| moments.xtx[column][row] * other)
                    .sum::<f64>()
        })
        .sum::<f64>();
    moments.yty - 2.0 * fitted_cross + quadratic
}

#[derive(Clone, Copy, Debug)]
struct FwlScore {
    beta_encoded: [f64; DESIGN_COLS],
    interaction_se_encoded: f64,
    interaction_score: f64,
    interaction_delta_rss: f64,
    sigma_e2: f64,
}

#[cfg(test)]
fn score_pair_moments_generic_reference(moments: &PairMoments) -> Result<FwlScore, String> {
    if moments.n_valid <= DESIGN_COLS {
        return Err("interaction unidentifiable: insufficient valid samples".to_string());
    }
    let full_matrix = flatten_matrix(&moments.xtx);
    let beta_vec = solve_linear(&full_matrix, &moments.xty, DESIGN_COLS)
        .map_err(|error| format!("interaction unidentifiable: {error}"))?;
    let beta_encoded = [beta_vec[0], beta_vec[1], beta_vec[2], beta_vec[3]];
    let null_matrix = [
        moments.xtx[0][0],
        moments.xtx[0][1],
        moments.xtx[0][2],
        moments.xtx[1][0],
        moments.xtx[1][1],
        moments.xtx[1][2],
        moments.xtx[2][0],
        moments.xtx[2][1],
        moments.xtx[2][2],
    ];
    let null_rhs = [moments.xty[0], moments.xty[1], moments.xty[2]];
    let null_beta = solve_linear(&null_matrix, &null_rhs, 3)
        .map_err(|error| format!("interaction unidentifiable: {error}"))?;
    let full_rss = rss_from_moments(moments, &beta_vec).max(0.0);
    let null_beta_full = [null_beta[0], null_beta[1], null_beta[2], 0.0];
    let null_rss = rss_from_moments(moments, &null_beta_full).max(0.0);
    let delta_rss = (null_rss - full_rss).max(0.0);
    let mut interaction_rhs = vec![0.0; DESIGN_COLS];
    interaction_rhs[3] = 1.0;
    let covariance_column = solve_linear(&full_matrix, &interaction_rhs, DESIGN_COLS)
        .map_err(|error| format!("interaction unidentifiable: {error}"))?;
    let covariance_factor = covariance_column[3];
    if !(covariance_factor > 0.0) || !covariance_factor.is_finite() {
        return Err("interaction unidentifiable: non-positive interaction covariance".to_string());
    }
    let degrees_of_freedom = moments.n_valid.saturating_sub(DESIGN_COLS);
    let sigma_e2 = if degrees_of_freedom > 0 {
        full_rss / degrees_of_freedom as f64
    } else {
        0.0
    };
    let interaction_se_encoded = if sigma_e2 > 0.0 {
        (sigma_e2 * covariance_factor).sqrt()
    } else {
        0.0
    };
    let interaction_score = if delta_rss > SOLVE_EPS {
        if sigma_e2 > SOLVE_EPS {
            delta_rss / sigma_e2
        } else {
            f64::INFINITY
        }
    } else {
        0.0
    };
    Ok(FwlScore {
        beta_encoded,
        interaction_se_encoded,
        interaction_score,
        interaction_delta_rss: delta_rss,
        sigma_e2,
    })
}

/// Score the saturated two-locus dosage model with scalar FWL arithmetic.
///
/// The scan evaluates this function once per pair, so it intentionally does
/// not allocate a matrix or call a generic linear-system solver.  The
/// intercept is removed analytically and the interaction column is residualized
/// against the two main-effect columns using the closed-form 2x2 solution.
fn score_pair_moments_scalar(moments: &PairMoments) -> Result<FwlScore, String> {
    if moments.n_valid <= DESIGN_COLS {
        return Err("interaction unidentifiable: insufficient valid samples".to_string());
    }

    let n = moments.xtx[0][0];
    if !(n > 0.0) || !n.is_finite() {
        return Err("interaction unidentifiable: invalid sample count".to_string());
    }

    // Raw moments of the intercept, x, z, w=xz, and y.  All genotype
    // columns are already in their internal encoded scale at this point.
    let sx = moments.xtx[0][1];
    let sz = moments.xtx[0][2];
    let sw = moments.xtx[0][3];
    let sy = moments.xty[0];
    let sxx = moments.xtx[1][1];
    let sxz = moments.xtx[1][2];
    let sxw = moments.xtx[1][3];
    let szz = moments.xtx[2][2];
    let szw = moments.xtx[2][3];
    let sww = moments.xtx[3][3];
    let sxy = moments.xty[1];
    let szy = moments.xty[2];
    let swy = moments.xty[3];
    let syy = moments.yty;

    let inv_n = 1.0 / n;
    let cxx = sxx - sx * sx * inv_n;
    let cxz = sxz - sx * sz * inv_n;
    let cxw = sxw - sx * sw * inv_n;
    let czz = szz - sz * sz * inv_n;
    let czw = szw - sz * sw * inv_n;
    let cww = sww - sw * sw * inv_n;
    let cxy = sxy - sx * sy * inv_n;
    let czy = szy - sz * sy * inv_n;
    let cyw = swy - sy * sw * inv_n;
    let cyy = syy - sy * sy * inv_n;

    let centered = [cxx, cxz, cxw, czz, czw, cww, cxy, czy, cyw, cyy];
    if centered.iter().any(|value| !value.is_finite()) {
        return Err("interaction unidentifiable: non-finite centered moments".to_string());
    }

    // The 2x2 main-effect determinant is the only denominator needed to
    // residualize w against x and z.
    let determinant = cxx * czz - cxz * cxz;
    let determinant_scale = (cxx * czz).abs().max(cxz * cxz).max(1.0);
    let determinant_tol = SOLVE_EPS * determinant_scale;
    if !(determinant > determinant_tol) || !determinant.is_finite() {
        return Err("interaction unidentifiable: rank-deficient main effects".to_string());
    }

    let null_beta_x = (czz * cxy - cxz * czy) / determinant;
    let null_beta_z = (cxx * czy - cxz * cxy) / determinant;

    // w = ax*x + az*z + w_perp.  This is the scalar form of X'X^{-1}X'w.
    let w_projection_x = (czz * cxw - cxz * czw) / determinant;
    let w_projection_z = (cxx * czw - cxz * cxw) / determinant;
    let interaction_variance = cww - w_projection_x * cxw - w_projection_z * czw;
    let interaction_variance_tol = SOLVE_EPS * cww.abs().max(1.0);
    if !(interaction_variance > interaction_variance_tol) || !interaction_variance.is_finite() {
        return Err("interaction unidentifiable: rank-deficient interaction".to_string());
    }

    let interaction_covariance = cyw - w_projection_x * cxy - w_projection_z * czy;
    let interaction_beta = interaction_covariance / interaction_variance;
    let beta_x = null_beta_x - w_projection_x * interaction_beta;
    let beta_z = null_beta_z - w_projection_z * interaction_beta;
    let intercept =
        sy * inv_n - beta_x * sx * inv_n - beta_z * sz * inv_n - interaction_beta * sw * inv_n;
    let beta_encoded = [intercept, beta_x, beta_z, interaction_beta];
    if beta_encoded.iter().any(|value| !value.is_finite()) {
        return Err("interaction unidentifiable: non-finite coefficient".to_string());
    }

    let null_explained = null_beta_x * cxy + null_beta_z * czy;
    let null_rss = (cyy - null_explained).max(0.0);
    let interaction_delta_rss = (interaction_covariance * interaction_beta).max(0.0);
    let full_rss = (null_rss - interaction_delta_rss).max(0.0);
    let degrees_of_freedom = moments.n_valid.saturating_sub(DESIGN_COLS);
    let sigma_e2 = if degrees_of_freedom > 0 {
        full_rss / degrees_of_freedom as f64
    } else {
        0.0
    };
    let interaction_se_encoded = if sigma_e2 > 0.0 {
        (sigma_e2 / interaction_variance).sqrt()
    } else {
        0.0
    };
    let interaction_score = if interaction_delta_rss > SOLVE_EPS {
        if sigma_e2 > SOLVE_EPS {
            interaction_delta_rss / sigma_e2
        } else {
            f64::INFINITY
        }
    } else {
        0.0
    };
    Ok(FwlScore {
        beta_encoded,
        interaction_se_encoded,
        interaction_score,
        interaction_delta_rss,
        sigma_e2,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DosagePairFit {
    pub(crate) beta: [f64; DESIGN_COLS],
    pub(crate) dosage_cell_means: [f64; DOSAGE_CELLS],
    pub(crate) observed_dosage_cell_means: [f64; DOSAGE_CELLS],
    pub(crate) dosage_cell_counts: [usize; DOSAGE_CELLS],
    pub(crate) binary_cell_means: Option<[f64; 4]>,
    pub(crate) observed_binary_cell_means: Option<[f64; 4]>,
    pub(crate) binary_cell_counts: Option<[usize; 4]>,
    pub(crate) n_levels_g1: usize,
    pub(crate) n_levels_g2: usize,
    pub(crate) n_valid: usize,
    pub(crate) encoding_offset_g1: f64,
    pub(crate) encoding_offset_g2: f64,
    pub(crate) encoding_scale_g1: f64,
    pub(crate) encoding_scale_g2: f64,
    pub(crate) interaction_identifiable: bool,
    pub(crate) interaction_beta: f64,
    pub(crate) interaction_se: f64,
    pub(crate) interaction_score: f64,
    pub(crate) interaction_delta_rss: f64,
    pub(crate) sigma_e2: f64,
    pub(crate) logic_label: DosageLogicLabel,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DosagePairCandidate {
    pub(crate) first: usize,
    pub(crate) second: usize,
    pub(crate) fit: DosagePairFit,
}

/// Minimal pair state retained while scanning.  Full cell summaries and
/// logic labels are materialized only for the bounded Top-K output after the
/// hot loop has finished.
#[derive(Clone, Copy, Debug)]
struct ScoredPair {
    first: usize,
    second: usize,
    score: FwlScore,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DosagePairScanResult {
    pub(crate) candidates: Vec<DosagePairCandidate>,
    pub(crate) pairs_evaluated: usize,
    pub(crate) pairs_skipped: usize,
    pub(crate) n_markers: usize,
    pub(crate) n_samples: usize,
}

#[derive(Default)]
struct PairScanAccumulator {
    heap: BinaryHeap<Reverse<ScoreHeapEntry>>,
    pairs_evaluated: usize,
    pairs_skipped: usize,
}

impl PairScanAccumulator {
    #[inline]
    fn scan_first_range(
        first_range: Range<usize>,
        representations: &[Option<MarkerRepresentation>],
        n_markers: usize,
        n_samples: usize,
        y: &[f64],
        lookup: &MaskYLookup,
        top_k: usize,
    ) -> Result<Self, String> {
        let mut accumulator = Self::default();
        for first in first_range {
            let Some(representation_g1) = representations[first].as_ref() else {
                continue;
            };
            for second in (first + 1)..n_markers {
                let Some(representation_g2) = representations[second].as_ref() else {
                    continue;
                };
                let score = match score_marker_pair_with_lookup(
                    representation_g1,
                    representation_g2,
                    y,
                    Some(lookup),
                ) {
                    Ok(score) => score,
                    Err(error) if error.contains("unidentifiable") => {
                        accumulator.pairs_skipped = accumulator.pairs_skipped.saturating_add(1);
                        continue;
                    }
                    Err(error) => return Err(error),
                };
                accumulator.pairs_evaluated = accumulator.pairs_evaluated.saturating_add(1);
                push_top_k_score(
                    &mut accumulator.heap,
                    ScoredPair {
                        first,
                        second,
                        score,
                    },
                    top_k,
                );
            }
        }
        debug_assert_eq!(lookup.n_words, words_for_samples(n_samples));
        Ok(accumulator)
    }

    #[inline]
    fn merge(&mut self, other: Self, top_k: usize) {
        self.pairs_evaluated = self.pairs_evaluated.saturating_add(other.pairs_evaluated);
        self.pairs_skipped = self.pairs_skipped.saturating_add(other.pairs_skipped);
        for Reverse(entry) in other.heap {
            push_top_k_score(&mut self.heap, entry.0, top_k);
        }
    }
}

fn score_cmp(
    a_score: f64,
    a_first: usize,
    a_second: usize,
    b_score: f64,
    b_first: usize,
    b_second: usize,
) -> Ordering {
    a_score
        .total_cmp(&b_score)
        .then_with(|| b_first.cmp(&a_first))
        .then_with(|| b_second.cmp(&a_second))
}

#[derive(Clone, Copy, Debug)]
struct ScoreHeapEntry(ScoredPair);

impl PartialEq for ScoreHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.0.first == other.0.first
            && self.0.second == other.0.second
            && self.0.score.interaction_score.to_bits() == other.0.score.interaction_score.to_bits()
    }
}

impl Eq for ScoreHeapEntry {}

impl Ord for ScoreHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        score_cmp(
            self.0.score.interaction_score,
            self.0.first,
            self.0.second,
            other.0.score.interaction_score,
            other.0.first,
            other.0.second,
        )
    }
}

impl PartialOrd for ScoreHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

fn push_top_k_score(
    heap: &mut BinaryHeap<Reverse<ScoreHeapEntry>>,
    candidate: ScoredPair,
    top_k: usize,
) {
    if top_k == 0 {
        return;
    }
    let entry = Reverse(ScoreHeapEntry(candidate));
    if heap.len() < top_k {
        heap.push(entry);
    } else if heap
        .peek()
        .map(|worst| entry.0.cmp(&worst.0) == Ordering::Greater)
        .unwrap_or(true)
    {
        let _ = heap.pop();
        heap.push(entry);
    }
}

#[inline]
fn dosage_cell_index(g1: usize, g2: usize) -> usize {
    g1 * DOSAGE_LEVELS + g2
}

#[inline]
fn fitted_dosage_cell_mean(beta: &[f64; DESIGN_COLS], g1: f64, g2: f64) -> f64 {
    beta[0] + beta[1] * g1 + beta[2] * g2 + beta[3] * g1 * g2
}

fn classify_logic(cell_means: &[f64; 4], interaction_beta: f64) -> DosageLogicLabel {
    if cell_means.iter().any(|value| !value.is_finite()) {
        return DosageLogicLabel::Unresolved;
    }
    let scale = cell_means
        .iter()
        .map(|value| value.abs())
        .fold(1.0, f64::max);
    if interaction_beta.abs() <= LOGIC_TOL * scale {
        return DosageLogicLabel::Additive;
    }
    let max_value = cell_means.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let min_value = cell_means.iter().copied().fold(f64::INFINITY, f64::min);
    let range = max_value - min_value;
    let tolerance = LOGIC_TOL * scale.max(range);
    let max_indices = cell_means
        .iter()
        .enumerate()
        .filter(|(_, value)| **value >= max_value - tolerance)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if max_indices.len() == 1 {
        return match max_indices[0] {
            0 => DosageLogicLabel::High00,
            1 => DosageLogicLabel::High10,
            2 => DosageLogicLabel::High01,
            _ => DosageLogicLabel::High11,
        };
    }
    let min_indices = cell_means
        .iter()
        .enumerate()
        .filter(|(_, value)| **value <= min_value + tolerance)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if min_indices.len() == 1 {
        return match min_indices[0] {
            0 => DosageLogicLabel::Low00,
            1 => DosageLogicLabel::Low10,
            2 => DosageLogicLabel::Low01,
            _ => DosageLogicLabel::Low11,
        };
    }
    let odd_mean = (cell_means[1] + cell_means[2]) / 2.0;
    let even_mean = (cell_means[0] + cell_means[3]) / 2.0;
    let odd_spread = (cell_means[1] - cell_means[2]).abs();
    let even_spread = (cell_means[0] - cell_means[3]).abs();
    let parity_tolerance = (range * 0.10).max(tolerance);
    if odd_mean > even_mean + tolerance
        && odd_spread <= parity_tolerance
        && even_spread <= parity_tolerance
    {
        DosageLogicLabel::XorHigh
    } else if even_mean > odd_mean + tolerance
        && odd_spread <= parity_tolerance
        && even_spread <= parity_tolerance
    {
        DosageLogicLabel::XorLow
    } else {
        DosageLogicLabel::Unresolved
    }
}

fn observed_cell_means(moments: &PairMoments) -> [f64; DOSAGE_CELLS] {
    std::array::from_fn(|cell| {
        if moments.cell_counts[cell] == 0 {
            f64::NAN
        } else {
            moments.cell_y_sums[cell] / moments.cell_counts[cell] as f64
        }
    })
}

fn binary_summaries(
    beta: &[f64; DESIGN_COLS],
    moments: &PairMoments,
    representation_g1: &MarkerRepresentation,
    representation_g2: &MarkerRepresentation,
) -> (Option<[f64; 4]>, Option<[f64; 4]>, Option<[usize; 4]>) {
    let (Some(levels_g1), Some(levels_g2)) = (
        representation_g1.binary_levels(),
        representation_g2.binary_levels(),
    ) else {
        return (None, None, None);
    };
    let fitted = [
        fitted_dosage_cell_mean(beta, levels_g1[0], levels_g2[0]),
        fitted_dosage_cell_mean(beta, levels_g1[1], levels_g2[0]),
        fitted_dosage_cell_mean(beta, levels_g1[0], levels_g2[1]),
        fitted_dosage_cell_mean(beta, levels_g1[1], levels_g2[1]),
    ];
    let codes_g1 = [
        integer_dosage_code(levels_g1[0]).expect("binary levels are integer"),
        integer_dosage_code(levels_g1[1]).expect("binary levels are integer"),
    ];
    let codes_g2 = [
        integer_dosage_code(levels_g2[0]).expect("binary levels are integer"),
        integer_dosage_code(levels_g2[1]).expect("binary levels are integer"),
    ];
    let mut counts = [0usize; 4];
    let mut sums = [0.0; 4];
    for first in 0..2 {
        for second in 0..2 {
            let raw_cell = dosage_cell_index(codes_g1[first], codes_g2[second]);
            let binary_cell = first + 2 * second;
            counts[binary_cell] = moments.cell_counts[raw_cell];
            sums[binary_cell] = moments.cell_y_sums[raw_cell];
        }
    }
    let observed = std::array::from_fn(|cell| {
        if counts[cell] == 0 {
            f64::NAN
        } else {
            sums[cell] / counts[cell] as f64
        }
    });
    (Some(fitted), Some(observed), Some(counts))
}

fn back_transform_beta(
    beta_encoded: &[f64; DESIGN_COLS],
    encoding_g1: MarkerEncoding,
    encoding_g2: MarkerEncoding,
) -> [f64; DESIGN_COLS] {
    let interaction = beta_encoded[3] / (encoding_g1.scale * encoding_g2.scale);
    let beta1 = beta_encoded[1] / encoding_g1.scale - encoding_g2.offset * interaction;
    let beta2 = beta_encoded[2] / encoding_g2.scale - encoding_g1.offset * interaction;
    let intercept = beta_encoded[0]
        - encoding_g1.offset * beta1
        - encoding_g2.offset * beta2
        - encoding_g1.offset * encoding_g2.offset * interaction;
    [intercept, beta1, beta2, interaction]
}

fn fit_marker_pair(
    representation_g1: &MarkerRepresentation,
    representation_g2: &MarkerRepresentation,
    y: &[f64],
) -> Result<DosagePairFit, String> {
    if y.iter().any(|value| !value.is_finite()) {
        return Err("y contains non-finite values".to_string());
    }
    fit_marker_pair_with_lookup(representation_g1, representation_g2, y, None)
}

fn fit_marker_pair_with_lookup(
    representation_g1: &MarkerRepresentation,
    representation_g2: &MarkerRepresentation,
    y: &[f64],
    lookup: Option<&MaskYLookup>,
) -> Result<DosagePairFit, String> {
    let moments = pair_moments_with_lookup(representation_g1, representation_g2, y, lookup)?;
    let score = score_pair_moments_scalar(&moments)?;
    let encoding_g1 = representation_g1.encoding();
    let encoding_g2 = representation_g2.encoding();
    let beta = back_transform_beta(&score.beta_encoded, encoding_g1, encoding_g2);
    let interaction_scale = encoding_g1.scale * encoding_g2.scale;
    let interaction_se = score.interaction_se_encoded / interaction_scale.abs();
    let dosage_cell_means = std::array::from_fn(|cell| {
        let g1 = (cell / DOSAGE_LEVELS) as f64;
        let g2 = (cell % DOSAGE_LEVELS) as f64;
        fitted_dosage_cell_mean(&beta, g1, g2)
    });
    let observed_dosage_cell_means = observed_cell_means(&moments);
    let (binary_cell_means, observed_binary_cell_means, binary_cell_counts) =
        binary_summaries(&beta, &moments, representation_g1, representation_g2);
    let logic_label = binary_cell_means
        .as_ref()
        .map(|means| classify_logic(means, beta[3]))
        .unwrap_or(DosageLogicLabel::Unresolved);
    Ok(DosagePairFit {
        beta,
        dosage_cell_means,
        observed_dosage_cell_means,
        dosage_cell_counts: moments.cell_counts,
        binary_cell_means,
        observed_binary_cell_means,
        binary_cell_counts,
        n_levels_g1: representation_g1.n_levels(),
        n_levels_g2: representation_g2.n_levels(),
        n_valid: moments.n_valid,
        encoding_offset_g1: encoding_g1.offset,
        encoding_offset_g2: encoding_g2.offset,
        encoding_scale_g1: encoding_g1.scale,
        encoding_scale_g2: encoding_g2.scale,
        interaction_identifiable: true,
        interaction_beta: beta[3],
        interaction_se,
        interaction_score: score.interaction_score,
        interaction_delta_rss: score.interaction_delta_rss,
        sigma_e2: score.sigma_e2,
        logic_label,
    })
}

#[inline]
fn score_marker_pair_with_lookup(
    representation_g1: &MarkerRepresentation,
    representation_g2: &MarkerRepresentation,
    y: &[f64],
    lookup: Option<&MaskYLookup>,
) -> Result<FwlScore, String> {
    let moments = pair_moments_with_lookup(representation_g1, representation_g2, y, lookup)?;
    score_pair_moments_scalar(&moments)
}

/// Compute the interaction p-value after selection.
///
/// This helper is intentionally not called by scan_dosage_pairs: the all-pair
/// hot loop only computes moments and the FWL score.  Callers should invoke
/// it for the bounded candidate list after ranking/selection.
pub(crate) fn interaction_p_value(fit: &DosagePairFit) -> Option<f64> {
    #[cfg(test)]
    PVALUE_CALLS.fetch_add(1, AtomicOrdering::Relaxed);
    let residual_df = fit.n_valid.saturating_sub(DESIGN_COLS);
    if residual_df == 0 || fit.interaction_score.is_nan() {
        return None;
    }
    if fit.interaction_score.is_infinite() {
        return Some(0.0);
    }
    if fit.interaction_score < 0.0 {
        return None;
    }
    FisherSnedecor::new(1.0, residual_df as f64)
        .ok()
        .map(|distribution| (1.0 - distribution.cdf(fit.interaction_score)).clamp(0.0, 1.0))
}

pub(crate) fn fit_dosage_pair(g1: &[f64], g2: &[f64], y: &[f64]) -> Result<DosagePairFit, String> {
    if g1.len() != g2.len() || g1.len() != y.len() {
        return Err(format!(
            "g1, g2, and y lengths must match; got {}, {}, {}",
            g1.len(),
            g2.len(),
            y.len()
        ));
    }
    if y.len() <= DESIGN_COLS {
        return Err(format!("need more than {DESIGN_COLS} samples"));
    }
    let representation_g1 = build_marker_representation(g1, "g1")?;
    let representation_g2 = build_marker_representation(g2, "g2")?;
    fit_marker_pair(&representation_g1, &representation_g2, y)
}

fn validate_scan_inputs(
    genotypes_len: usize,
    n_markers: usize,
    n_samples: usize,
    y_len: usize,
) -> Result<(), String> {
    if n_markers == 0 || n_samples == 0 {
        return Err("n_markers and n_samples must be > 0".to_string());
    }
    let expected = n_markers
        .checked_mul(n_samples)
        .ok_or_else(|| "genotype matrix size overflow".to_string())?;
    if genotypes_len != expected {
        return Err(format!(
            "genotypes length={} but expected {} (n_markers*n_samples)",
            genotypes_len, expected
        ));
    }
    if y_len != n_samples {
        return Err(format!("y length={} but expected {n_samples}", y_len));
    }
    Ok(())
}

fn build_marker_representations(
    genotypes: &[f64],
    n_markers: usize,
    n_samples: usize,
) -> Vec<Option<MarkerRepresentation>> {
    (0..n_markers)
        .map(|marker| {
            build_marker_representation(
                &genotypes[marker * n_samples..(marker + 1) * n_samples],
                &format!("genotype marker {marker}"),
            )
            .ok()
        })
        .collect()
}

fn scan_dosage_pairs_from_representations(
    representations: Vec<Option<MarkerRepresentation>>,
    n_markers: usize,
    n_samples: usize,
    y: &[f64],
    top_k: usize,
) -> Result<DosagePairScanResult, String> {
    scan_dosage_pairs_from_representations_with_threads(
        representations,
        n_markers,
        n_samples,
        y,
        top_k,
        1,
    )
}

fn scan_dosage_pairs_from_representations_with_threads(
    representations: Vec<Option<MarkerRepresentation>>,
    n_markers: usize,
    n_samples: usize,
    y: &[f64],
    top_k: usize,
    threads: usize,
) -> Result<DosagePairScanResult, String> {
    if representations.len() != n_markers {
        return Err("marker representation count does not match n_markers".to_string());
    }
    if y.len() != n_samples {
        return Err(format!("y length={} but expected {n_samples}", y.len()));
    }
    if y.iter().any(|value| !value.is_finite()) {
        return Err("y contains non-finite values".to_string());
    }
    let lookup = MaskYLookup::new(y);
    let first_count = n_markers.saturating_sub(1);
    let all_complete_float = representations.iter().all(|representation| {
        matches!(
            representation,
            Some(MarkerRepresentation::Float(marker)) if !marker.has_missing
        )
    });
    let mut accumulator = if all_complete_float {
        let float_markers = representations
            .iter()
            .map(|representation| match representation.as_ref() {
                Some(MarkerRepresentation::Float(marker)) => Ok(marker),
                _ => Err("complete float representation was not available".to_string()),
            })
            .collect::<Result<Vec<_>, _>>()?;
        scan_float_pairs_blocked(&float_markers, n_samples, y, top_k)?
    } else if threads <= 1 || first_count <= 1 {
        PairScanAccumulator::scan_first_range(
            0..first_count,
            &representations,
            n_markers,
            n_samples,
            y,
            &lookup,
            top_k,
        )?
    } else {
        let chunk_count = threads.saturating_mul(4).max(1);
        let chunk_len = (first_count + chunk_count - 1) / chunk_count;
        let ranges = (0..first_count)
            .step_by(chunk_len.max(1))
            .map(|start| start..(start + chunk_len).min(first_count))
            .collect::<Vec<_>>();
        let run = || {
            ranges
                .par_iter()
                .map(|range| {
                    PairScanAccumulator::scan_first_range(
                        range.clone(),
                        &representations,
                        n_markers,
                        n_samples,
                        y,
                        &lookup,
                        top_k,
                    )
                })
                .collect::<Result<Vec<_>, String>>()
        };
        let partials = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|error| format!("failed to build dosage Rayon pool: {error}"))?
            .install(run)?;
        let mut merged = PairScanAccumulator::default();
        for partial in partials {
            merged.merge(partial, top_k);
        }
        merged
    };
    let mut heap = std::mem::take(&mut accumulator.heap);
    let pairs_evaluated = accumulator.pairs_evaluated;
    let pairs_skipped = accumulator.pairs_skipped;
    let mut scored_pairs = heap
        .drain()
        .map(|Reverse(entry)| entry.0)
        .collect::<Vec<_>>();
    scored_pairs.sort_by(|a, b| {
        score_cmp(
            b.score.interaction_score,
            b.first,
            b.second,
            a.score.interaction_score,
            a.first,
            a.second,
        )
    });
    let mut candidates = Vec::with_capacity(scored_pairs.len());
    for scored in scored_pairs {
        let Some(representation_g1) = representations[scored.first].as_ref() else {
            continue;
        };
        let Some(representation_g2) = representations[scored.second].as_ref() else {
            continue;
        };
        let fit =
            fit_marker_pair_with_lookup(representation_g1, representation_g2, y, Some(&lookup))
                .map_err(|error| format!("failed to materialize retained pair: {error}"))?;
        candidates.push(DosagePairCandidate {
            first: scored.first,
            second: scored.second,
            fit,
        });
    }
    Ok(DosagePairScanResult {
        candidates,
        pairs_evaluated,
        pairs_skipped,
        n_markers,
        n_samples,
    })
}

#[cfg(test)]
fn scan_dosage_pairs(
    genotypes: &[f64],
    n_markers: usize,
    n_samples: usize,
    y: &[f64],
    top_k: usize,
) -> Result<DosagePairScanResult, String> {
    validate_scan_inputs(genotypes.len(), n_markers, n_samples, y.len())?;
    let representations = build_marker_representations(genotypes, n_markers, n_samples);
    scan_dosage_pairs_from_representations(representations, n_markers, n_samples, y, top_k)
}

/// Owned-input variant used by the Python boundary.  Building the compact
/// marker representations first and then releasing the temporary flattened
/// genotype matrix avoids retaining two full marker-major copies during the
/// expensive pair scan.
fn scan_dosage_pairs_owned(
    genotypes: Vec<f64>,
    n_markers: usize,
    n_samples: usize,
    y: &[f64],
    top_k: usize,
) -> Result<DosagePairScanResult, String> {
    scan_dosage_pairs_owned_with_threads(genotypes, n_markers, n_samples, y, top_k, 1)
}

fn scan_dosage_pairs_owned_with_threads(
    genotypes: Vec<f64>,
    n_markers: usize,
    n_samples: usize,
    y: &[f64],
    top_k: usize,
    threads: usize,
) -> Result<DosagePairScanResult, String> {
    validate_scan_inputs(genotypes.len(), n_markers, n_samples, y.len())?;
    let representations = build_marker_representations(&genotypes, n_markers, n_samples);
    drop(genotypes);
    scan_dosage_pairs_from_representations_with_threads(
        representations,
        n_markers,
        n_samples,
        y,
        top_k,
        threads,
    )
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
        let shape = view.shape();
        let mut out = Vec::with_capacity(view.len());
        for row in 0..shape[0] {
            for column in 0..shape[1] {
                out.push(view[[row, column]]);
            }
        }
        out
    }
}

fn set_fit_items<'py>(
    py: Python<'py>,
    out: &Bound<'py, PyDict>,
    fit: &DosagePairFit,
) -> PyResult<()> {
    out.set_item("beta", PyArray1::from_vec(py, fit.beta.to_vec()))?;
    out.set_item(
        "dosage_cell_means",
        PyArray1::from_vec(py, fit.dosage_cell_means.to_vec()),
    )?;
    out.set_item(
        "observed_dosage_cell_means",
        PyArray1::from_vec(py, fit.observed_dosage_cell_means.to_vec()),
    )?;
    out.set_item(
        "dosage_cell_counts",
        PyArray1::from_vec(py, fit.dosage_cell_counts.to_vec()),
    )?;
    if let Some(values) = fit.binary_cell_means {
        out.set_item("binary_cell_means", PyArray1::from_vec(py, values.to_vec()))?;
    } else {
        out.set_item("binary_cell_means", py.None())?;
    }
    if let Some(values) = fit.observed_binary_cell_means {
        out.set_item(
            "observed_binary_cell_means",
            PyArray1::from_vec(py, values.to_vec()),
        )?;
    } else {
        out.set_item("observed_binary_cell_means", py.None())?;
    }
    if let Some(values) = fit.binary_cell_counts {
        out.set_item(
            "binary_cell_counts",
            PyArray1::from_vec(py, values.to_vec()),
        )?;
    } else {
        out.set_item("binary_cell_counts", py.None())?;
    }
    out.set_item("n_levels_g1", fit.n_levels_g1)?;
    out.set_item("n_levels_g2", fit.n_levels_g2)?;
    out.set_item("n_valid", fit.n_valid)?;
    out.set_item("encoding_offset_g1", fit.encoding_offset_g1)?;
    out.set_item("encoding_offset_g2", fit.encoding_offset_g2)?;
    out.set_item("encoding_scale_g1", fit.encoding_scale_g1)?;
    out.set_item("encoding_scale_g2", fit.encoding_scale_g2)?;
    out.set_item("interaction_identifiable", fit.interaction_identifiable)?;
    out.set_item("interaction_beta", fit.interaction_beta)?;
    out.set_item("interaction_se", fit.interaction_se)?;
    out.set_item("interaction_score", fit.interaction_score)?;
    out.set_item("interaction_pvalue", interaction_p_value(fit))?;
    out.set_item("interaction_delta_rss", fit.interaction_delta_rss)?;
    out.set_item("sigma_e2", fit.sigma_e2)?;
    out.set_item("logic_label", fit.logic_label.as_str())?;
    Ok(())
}

#[pyfunction(name = "garfield_dosage_pair_fit")]
#[pyo3(signature = (g1, g2, y))]
pub fn garfield_dosage_pair_fit_py<'py>(
    py: Python<'py>,
    g1: PyReadonlyArray1<'py, f64>,
    g2: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let g1_vec = array1_to_vec(&g1);
    let g2_vec = array1_to_vec(&g2);
    let y_vec = array1_to_vec(&y);
    let fit = fit_dosage_pair(&g1_vec, &g2_vec, &y_vec).map_err(PyValueError::new_err)?;
    let out = PyDict::new(py);
    set_fit_items(py, &out, &fit)?;
    out.set_item("n_samples", y_vec.len())?;
    Ok(out)
}

#[pyfunction(name = "garfield_dosage_pair_scan")]
#[pyo3(signature = (genotypes, y, top_k=100, threads=0))]
pub fn garfield_dosage_pair_scan_py<'py>(
    py: Python<'py>,
    genotypes: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
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
    let genotype_vec = array2_to_vec(&genotypes);
    let y_vec = array1_to_vec(&y);
    let effective_threads = if threads == 0 {
        rayon::current_num_threads().max(1)
    } else {
        threads
    };
    let scan = scan_dosage_pairs_owned_with_threads(
        genotype_vec,
        n_markers,
        n_samples,
        &y_vec,
        top_k,
        effective_threads,
    )
    .map_err(PyValueError::new_err)?;
    let candidates = PyList::empty(py);
    for candidate in scan.candidates {
        let item = PyDict::new(py);
        item.set_item("first", candidate.first)?;
        item.set_item("second", candidate.second)?;
        set_fit_items(py, &item, &candidate.fit)?;
        candidates.append(item)?;
    }
    let out = PyDict::new(py);
    out.set_item("candidates", candidates)?;
    out.set_item("pairs_evaluated", scan.pairs_evaluated)?;
    out.set_item("pairs_skipped", scan.pairs_skipped)?;
    out.set_item("n_markers", scan.n_markers)?;
    out.set_item("n_samples", scan.n_samples)?;
    out.set_item("threads", effective_threads)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_mask_lookup_matches_scalar_mask_sums() {
        let y = [1.25, -2.0, 3.5, 4.0, -5.25, 6.0, 7.75, -8.5, 9.0, 10.5];
        let lookup = MaskYLookup::new(&y);
        let mask = (1u64 << 0) | (1u64 << 2) | (1u64 << 7) | (1u64 << 9);
        let expected = y[0] + y[2] + y[7] + y[9];
        let expected_sq = y[0] * y[0] + y[2] * y[2] + y[7] * y[7] + y[9] * y[9];
        assert_close(lookup.sum_y(mask, 0), expected);
        assert_close(lookup.sum_y2(mask, 0), expected_sq);
        assert_close(lookup.sum_y(1u64 << 63, 0), 0.0);
    }

    #[test]
    fn aggregated_discrete_moments_match_scalar_intersection_accumulation() {
        let g1 = [0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0];
        let g2 = [0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 2.0, 0.0, 1.0, 2.0];
        let y = [1.25, -2.0, 3.5, 4.0, -5.25, 6.0, 7.75, -8.5, 9.0, 10.5];
        let representation_g1 = build_marker_representation(&g1, "g1").unwrap();
        let representation_g2 = build_marker_representation(&g2, "g2").unwrap();
        let lookup = MaskYLookup::new(&y);
        let expected =
            pair_moments_discrete_complete(&representation_g1, &representation_g2, &y).unwrap();
        let got = pair_moments_discrete_with_lookup(
            &representation_g1,
            &representation_g2,
            &y,
            &lookup,
            false,
        )
        .unwrap();
        assert_eq!(got.n_valid, expected.n_valid);
        assert_eq!(got.cell_counts, expected.cell_counts);
        for (got, expected) in got.xtx.iter().flatten().zip(expected.xtx.iter().flatten()) {
            assert_close(*got, *expected);
        }
        for (got, expected) in got.xty.iter().zip(expected.xty.iter()) {
            assert_close(*got, *expected);
        }
        assert_close(got.yty, expected.yty);
        for (got, expected) in got.cell_y_sums.iter().zip(expected.cell_y_sums.iter()) {
            assert_close(*got, *expected);
        }
    }

    #[test]
    fn binary_lookup_fast_path_matches_scalar_reference() {
        let g1 = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0];
        let g2 = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let y = [1.25, -2.0, 3.5, 4.0, -5.25, 6.0, 7.75, -8.5, 9.0, 10.5];
        let representation_g1 = build_marker_representation(&g1, "g1").unwrap();
        let representation_g2 = build_marker_representation(&g2, "g2").unwrap();
        let (MarkerRepresentation::Binary(marker_g1), MarkerRepresentation::Binary(marker_g2)) =
            (&representation_g1, &representation_g2)
        else {
            panic!("binary markers must use the binary representation");
        };
        let lookup = MaskYLookup::new(&y);
        let expected =
            pair_moments_discrete_complete(&representation_g1, &representation_g2, &y).unwrap();
        let got =
            pair_moments_binary_binary_with_lookup(marker_g1, marker_g2, &y, &lookup).unwrap();
        assert_eq!(got.n_valid, expected.n_valid);
        assert_eq!(got.cell_counts, expected.cell_counts);
        for (got, expected) in got.xtx.iter().flatten().zip(expected.xtx.iter().flatten()) {
            assert_close(*got, *expected);
        }
        for (got, expected) in got.xty.iter().zip(expected.xty.iter()) {
            assert_close(*got, *expected);
        }
        assert_close(got.yty, expected.yty);
        for (got, expected) in got.cell_y_sums.iter().zip(expected.cell_y_sums.iter()) {
            assert_close(*got, *expected);
        }
    }

    #[test]
    fn ternary_lookup_fast_path_matches_scalar_reference() {
        let g1 = [
            0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 2.0,
        ];
        let g2 = [
            0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 1.0, 2.0, 0.0, 1.0,
        ];
        let y = [
            1.25, -2.0, 3.5, 4.0, -5.25, 6.0, 7.75, -8.5, 9.0, 10.5, -11.0, 12.5, 13.0,
        ];
        let representation_g1 = build_marker_representation(&g1, "g1").unwrap();
        let representation_g2 = build_marker_representation(&g2, "g2").unwrap();
        let (MarkerRepresentation::Ternary(marker_g1), MarkerRepresentation::Ternary(marker_g2)) =
            (&representation_g1, &representation_g2)
        else {
            panic!("three-level markers must use the ternary representation");
        };
        let lookup = MaskYLookup::new(&y);
        let expected =
            pair_moments_discrete_complete(&representation_g1, &representation_g2, &y).unwrap();
        let got =
            pair_moments_ternary_ternary_with_lookup(marker_g1, marker_g2, &y, &lookup).unwrap();
        assert_eq!(got.n_valid, expected.n_valid);
        assert_eq!(got.cell_counts, expected.cell_counts);
        for (got, expected) in got.xtx.iter().flatten().zip(expected.xtx.iter().flatten()) {
            assert_close(*got, *expected);
        }
        for (got, expected) in got.xty.iter().zip(expected.xty.iter()) {
            assert_close(*got, *expected);
        }
        assert_close(got.yty, expected.yty);
        for (got, expected) in got.cell_y_sums.iter().zip(expected.cell_y_sums.iter()) {
            assert_close(*got, *expected);
        }
    }

    #[test]
    fn binary_ternary_lookup_fast_path_matches_scalar_reference() {
        let binary = [
            0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 2.0, 0.0,
        ];
        let ternary = [
            0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0,
        ];
        let y = [
            1.25, -2.0, 3.5, 4.0, -5.25, 6.0, 7.75, -8.5, 9.0, 10.5, -11.0, 12.5, 13.0,
        ];
        let representation_binary = build_marker_representation(&binary, "binary").unwrap();
        let representation_ternary = build_marker_representation(&ternary, "ternary").unwrap();
        let (
            MarkerRepresentation::Binary(binary_marker),
            MarkerRepresentation::Ternary(ternary_marker),
        ) = (&representation_binary, &representation_ternary)
        else {
            panic!("expected binary and ternary representations");
        };
        let lookup = MaskYLookup::new(&y);
        let expected =
            pair_moments_discrete_complete(&representation_binary, &representation_ternary, &y)
                .unwrap();
        let got =
            pair_moments_binary_ternary_with_lookup(binary_marker, ternary_marker, &y, &lookup)
                .unwrap();
        assert_eq!(got.n_valid, expected.n_valid);
        assert_eq!(got.cell_counts, expected.cell_counts);
        for (got, expected) in got.xtx.iter().flatten().zip(expected.xtx.iter().flatten()) {
            assert_close(*got, *expected);
        }
        for (got, expected) in got.xty.iter().zip(expected.xty.iter()) {
            assert_close(*got, *expected);
        }
        assert_close(got.yty, expected.yty);
        for (got, expected) in got.cell_y_sums.iter().zip(expected.cell_y_sums.iter()) {
            assert_close(*got, *expected);
        }

        let expected_swapped =
            pair_moments_discrete_complete(&representation_ternary, &representation_binary, &y)
                .unwrap();
        let got_swapped =
            pair_moments_ternary_binary_with_lookup(ternary_marker, binary_marker, &y, &lookup)
                .unwrap();
        assert_eq!(got_swapped.n_valid, expected_swapped.n_valid);
        assert_eq!(got_swapped.cell_counts, expected_swapped.cell_counts);
        for (got, expected) in got_swapped
            .xtx
            .iter()
            .flatten()
            .zip(expected_swapped.xtx.iter().flatten())
        {
            assert_close(*got, *expected);
        }
        for (got, expected) in got_swapped.xty.iter().zip(expected_swapped.xty.iter()) {
            assert_close(*got, *expected);
        }
        assert_close(got_swapped.yty, expected_swapped.yty);
        for (got, expected) in got_swapped
            .cell_y_sums
            .iter()
            .zip(expected_swapped.cell_y_sums.iter())
        {
            assert_close(*got, *expected);
        }
    }

    #[test]
    fn aggregated_masked_moments_match_pairwise_missing_reference() {
        let g1 = [0.0, 2.0, f64::NAN, 2.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0];
        let g2 = [0.0, 1.0, 2.0, f64::NAN, 0.0, 2.0, 2.0, 0.0, 1.0, 2.0];
        let y = [1.25, -2.0, 3.5, 4.0, -5.25, 6.0, 7.75, -8.5, 9.0, 10.5];
        let representation_g1 = build_marker_representation(&g1, "g1").unwrap();
        let representation_g2 = build_marker_representation(&g2, "g2").unwrap();
        let lookup = MaskYLookup::new(&y);
        let expected =
            pair_moments_discrete_masked(&representation_g1, &representation_g2, &y).unwrap();
        let got = pair_moments_discrete_with_lookup(
            &representation_g1,
            &representation_g2,
            &y,
            &lookup,
            true,
        )
        .unwrap();
        assert_eq!(got.n_valid, expected.n_valid);
        assert_eq!(got.cell_counts, expected.cell_counts);
        for (got, expected) in got.xtx.iter().flatten().zip(expected.xtx.iter().flatten()) {
            assert_close(*got, *expected);
        }
        for (got, expected) in got.xty.iter().zip(expected.xty.iter()) {
            assert_close(*got, *expected);
        }
        assert_close(got.yty, expected.yty);
        for (got, expected) in got.cell_y_sums.iter().zip(expected.cell_y_sums.iter()) {
            assert_close(*got, *expected);
        }
    }

    #[test]
    fn unrolled_float_moments_match_observation_accumulation() {
        let observations = [
            (0.15, 1.25, 1.0, 0.0, 2.0),
            (0.75, 0.50, 0.25, 1.0, -1.5),
            (1.50, 1.75, 1.5, 2.0, 3.25),
            (0.25, 0.25, 2.0, 0.25, -4.0),
        ];
        let mut expected = PairMoments::new();
        let mut got = PairMoments::new();
        for (encoded_g1, encoded_g2, raw_g1, raw_g2, y) in observations {
            expected.add_observation(encoded_g1, encoded_g2, raw_g1, raw_g2, y);
            got.add_float_observation(encoded_g1, encoded_g2, raw_g1, raw_g2, y);
        }
        assert_eq!(got.n_valid, expected.n_valid);
        assert_eq!(got.cell_counts, expected.cell_counts);
        for (got, expected) in got.xtx.iter().flatten().zip(expected.xtx.iter().flatten()) {
            assert_close(*got, *expected);
        }
        for (got, expected) in got.xty.iter().zip(expected.xty.iter()) {
            assert_close(*got, *expected);
        }
        assert_close(got.yty, expected.yty);
        for (got, expected) in got.cell_y_sums.iter().zip(expected.cell_y_sums.iter()) {
            assert_close(*got, *expected);
        }
    }

    #[test]
    fn specialized_float_float_moments_match_generic_reference() {
        let g1 = [0.15, 0.75, 1.50, 0.25, 1.10, 0.40];
        let g2 = [1.25, 0.50, 1.75, 0.25, 0.80, 1.90];
        let y = [1.0, -1.5, 3.25, -4.0, 2.0, 0.75];
        let representation_g1 = build_marker_representation(&g1, "g1").unwrap();
        let representation_g2 = build_marker_representation(&g2, "g2").unwrap();
        let (MarkerRepresentation::Float(marker_g1), MarkerRepresentation::Float(marker_g2)) =
            (&representation_g1, &representation_g2)
        else {
            panic!("fractional marker must use the float representation");
        };
        let expected =
            pair_moments_float_generic(&representation_g1, &representation_g2, &y).unwrap();
        let got = pair_moments_float_float(marker_g1, marker_g2, &y);
        assert_eq!(got.n_valid, expected.n_valid);
        assert_eq!(got.cell_counts, expected.cell_counts);
        for (got, expected) in got.xtx.iter().flatten().zip(expected.xtx.iter().flatten()) {
            assert_close(*got, *expected);
        }
        for (got, expected) in got.xty.iter().zip(expected.xty.iter()) {
            assert_close(*got, *expected);
        }
        assert_close(got.yty, expected.yty);
        for (got, expected) in got.cell_y_sums.iter().zip(expected.cell_y_sums.iter()) {
            assert_close(*got, *expected);
        }
    }

    #[test]
    fn saturated_binary_model_recovers_coefficients_and_cells() {
        let g1 = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let g2 = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let y = [1.0, 3.0, 4.0, 10.0, 1.0, 3.0, 4.0, 10.0];
        let fit = fit_dosage_pair(&g1, &g2, &y).unwrap();
        assert_close(fit.beta[0], 1.0);
        assert_close(fit.beta[1], 3.0);
        assert_close(fit.beta[2], 2.0);
        assert_close(fit.beta[3], 4.0);
        assert_eq!(fit.dosage_cell_counts, [2, 2, 0, 2, 2, 0, 0, 0, 0]);
        for (got, expected) in fit
            .binary_cell_means
            .unwrap()
            .iter()
            .zip([1.0, 4.0, 3.0, 10.0])
        {
            assert_close(*got, expected);
        }
        assert!(fit.dosage_cell_means[4].is_finite());
        assert!(fit.interaction_score > 1.0e8);
        assert_eq!(fit.logic_label, DosageLogicLabel::High11);
    }

    #[test]
    fn additive_model_has_zero_interaction_statistic() {
        let g1 = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let g2 = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let y = [2.0, 5.0, 7.0, 10.0, 2.0, 5.0, 7.0, 10.0];
        let fit = fit_dosage_pair(&g1, &g2, &y).unwrap();
        assert_close(fit.beta[3], 0.0);
        assert_close(fit.interaction_delta_rss, 0.0);
        assert_close(fit.interaction_score, 0.0);
        assert!(fit.binary_cell_means.is_some());
        assert_eq!(fit.logic_label, DosageLogicLabel::Additive);
    }

    #[test]
    fn inbred_zero_two_coding_has_the_same_binary_cell_model() {
        let g1 = [0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0];
        let g2 = [0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0];
        let y = [1.0, 3.0, 4.0, 10.0, 1.0, 3.0, 4.0, 10.0];
        let fit = fit_dosage_pair(&g1, &g2, &y).unwrap();
        assert_eq!(fit.dosage_cell_counts, [2, 0, 2, 0, 0, 0, 2, 0, 2]);
        for (got, expected) in fit
            .binary_cell_means
            .unwrap()
            .iter()
            .zip([1.0, 4.0, 3.0, 10.0])
        {
            assert_close(*got, expected);
        }
        assert_close(fit.beta[0], 1.0);
        assert_close(fit.beta[1], 1.5);
        assert_close(fit.beta[2], 1.0);
        assert_close(fit.interaction_beta, 1.0);
        assert_close(fit.encoding_scale_g1, 2.0);
        assert_close(fit.encoding_scale_g2, 2.0);
        assert_eq!(fit.n_valid, 8);
    }

    #[test]
    fn pairwise_missing_uses_a_masked_path_without_imputation() {
        let g1 = [0.0, 0.0, 2.0, 2.0, f64::NAN, 0.0, 2.0, 2.0, 0.0];
        let g2 = [0.0, 2.0, 0.0, 2.0, 0.0, f64::NAN, 0.0, 2.0, 2.0];
        let y = [1.0, 3.0, 4.0, 10.0, 99.0, 99.0, 4.0, 10.0, 1.0];
        let fit = fit_dosage_pair(&g1, &g2, &y).unwrap();
        assert_eq!(fit.n_valid, 7);
        assert_eq!(fit.dosage_cell_counts.iter().sum::<usize>(), 7);
        assert_eq!(fit.dosage_cell_counts[0], 1);
        assert_eq!(fit.dosage_cell_counts[8], 2);
        assert_close(fit.observed_dosage_cell_means[8], 10.0);
    }

    #[test]
    fn rank_deficient_interaction_is_reported_as_unidentifiable() {
        let g1 = [0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 2.0];
        let g2 = g1;
        let y = [1.0, 3.0, 4.0, 10.0, 1.0, 3.0, 4.0, 10.0];
        let error = fit_dosage_pair(&g1, &g2, &y).unwrap_err();
        assert!(
            error.contains("unidentifiable"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn fractional_dosage_uses_the_float_representation_backend() {
        let g1 = [0.0, 0.25, 0.75, 1.25, 1.75, 2.0, 0.5, 1.5];
        let g2 = [0.0, 1.0, 0.5, 2.0, 1.5, 0.25, 1.75, 0.75];
        let y = [1.0, 2.0, 4.0, 7.0, 6.0, 3.0, 5.0, 8.0];
        let fit = fit_dosage_pair(&g1, &g2, &y).unwrap();
        assert_close(fit.encoding_scale_g1, 1.0);
        assert_close(fit.encoding_scale_g2, 1.0);
        assert_eq!(fit.logic_label, DosageLogicLabel::Unresolved);
        assert!(fit.interaction_score.is_finite());
    }

    #[test]
    fn ternary_dosage_is_fitted_but_not_logic_labeled() {
        let mut g1 = Vec::new();
        let mut g2 = Vec::new();
        let mut y = Vec::new();
        for a in 0..=2 {
            for b in 0..=2 {
                for _ in 0..2 {
                    let a = a as f64;
                    let b = b as f64;
                    g1.push(a);
                    g2.push(b);
                    y.push(2.0 + 3.0 * a - 2.0 * b + 0.5 * a * b);
                }
            }
        }
        let fit = fit_dosage_pair(&g1, &g2, &y).unwrap();
        assert_close(fit.beta[0], 2.0);
        assert_close(fit.beta[1], 3.0);
        assert_close(fit.beta[2], -2.0);
        assert_close(fit.beta[3], 0.5);
        assert_eq!(fit.dosage_cell_counts, [2; 9]);
        assert!(fit.binary_cell_means.is_none());
        assert_eq!(fit.logic_label, DosageLogicLabel::Unresolved);
    }

    #[test]
    fn cell_means_identify_xor_patterns() {
        let g1 = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let g2 = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let high = [5.0, 10.0, 10.0, 5.0, 5.0, 10.0, 10.0, 5.0];
        let low = [10.0, 5.0, 5.0, 10.0, 10.0, 5.0, 5.0, 10.0];
        let high_fit = fit_dosage_pair(&g1, &g2, &high).unwrap();
        let low_fit = fit_dosage_pair(&g1, &g2, &low).unwrap();
        assert_eq!(high_fit.logic_label, DosageLogicLabel::XorHigh);
        assert_eq!(low_fit.logic_label, DosageLogicLabel::XorLow);
    }

    #[test]
    fn repeated_fwl_scoring_is_deterministic_without_matrix_inverse() {
        let g1 = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let g2 = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let y = [1.0, 3.0, 4.0, 10.0, 1.0, 3.0, 4.0, 10.0];
        let ordinary = fit_dosage_pair(&g1, &g2, &y).unwrap();
        let repeated = fit_dosage_pair(&g1, &g2, &y).unwrap();
        for (got, expected) in repeated.beta.iter().zip(ordinary.beta) {
            assert_close(*got, expected);
        }
        assert_close(repeated.interaction_score, ordinary.interaction_score);
    }

    #[test]
    fn scalar_fwl_matches_the_generic_reference_on_nonorthogonal_data() {
        let g1 = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0, 2.0];
        let g2 = [0.0, 0.0, 1.0, 2.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 2.0, 0.0];
        let y = [1.0, 2.5, 5.0, 3.0, 7.5, 8.0, 2.0, 4.5, 10.0, 5.5, 6.0, 4.0];
        let representation_g1 = build_marker_representation(&g1, "g1").unwrap();
        let representation_g2 = build_marker_representation(&g2, "g2").unwrap();
        let moments = pair_moments(&representation_g1, &representation_g2, &y).unwrap();
        let reference = score_pair_moments_generic_reference(&moments).unwrap();
        let scalar = score_pair_moments_scalar(&moments).unwrap();
        for (got, expected) in scalar.beta_encoded.iter().zip(reference.beta_encoded) {
            assert_close(*got, expected);
        }
        assert_close(scalar.interaction_score, reference.interaction_score);
        assert_close(
            scalar.interaction_delta_rss,
            reference.interaction_delta_rss,
        );
        assert_close(
            scalar.interaction_se_encoded,
            reference.interaction_se_encoded,
        );
        assert_close(scalar.sigma_e2, reference.sigma_e2);
    }

    #[test]
    fn all_pair_hot_loop_defers_p_value_until_candidates_are_retained() {
        PVALUE_CALLS.store(0, AtomicOrdering::Relaxed);
        let genotypes = [
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
        ];
        let y = [1.0, 1.0, 4.0, 4.0, 3.0, 3.0, 10.0, 10.0];
        let scan = scan_dosage_pairs(&genotypes, 2, 8, &y, 1).unwrap();
        assert_eq!(PVALUE_CALLS.load(AtomicOrdering::Relaxed), 0);
        let p_value = interaction_p_value(&scan.candidates[0].fit).unwrap();
        assert!((0.0..=1.0).contains(&p_value));
        assert_eq!(PVALUE_CALLS.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn all_pair_scan_returns_the_strongest_pair() {
        let genotypes = [
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        ];
        let y = [1.0, 1.0, 4.0, 4.0, 3.0, 3.0, 10.0, 10.0];
        let scan = scan_dosage_pairs(&genotypes, 3, 8, &y, 2).unwrap();
        assert_eq!(scan.pairs_evaluated, 3);
        assert_eq!(scan.candidates.len(), 2);
        assert_eq!(
            (scan.candidates[0].first, scan.candidates[0].second),
            (0, 1)
        );
        assert!(
            scan.candidates[0].fit.interaction_score > scan.candidates[1].fit.interaction_score
        );
    }

    #[test]
    fn owned_scan_matches_borrowed_scan() {
        let genotypes = vec![
            0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
        ];
        let y = [1.0, 1.0, 4.0, 4.0, 3.0, 3.0, 10.0, 10.0];
        let borrowed = scan_dosage_pairs(&genotypes, 2, 8, &y, 2).unwrap();
        let owned = scan_dosage_pairs_owned(genotypes, 2, 8, &y, 2).unwrap();
        assert_eq!(owned.pairs_evaluated, borrowed.pairs_evaluated);
        assert_eq!(owned.pairs_skipped, borrowed.pairs_skipped);
        assert_eq!(owned.candidates.len(), borrowed.candidates.len());
        for (got, expected) in owned.candidates.iter().zip(borrowed.candidates.iter()) {
            assert_eq!((got.first, got.second), (expected.first, expected.second));
            assert_close(got.fit.interaction_score, expected.fit.interaction_score);
        }
    }

    #[test]
    fn parallel_scan_matches_serial_top_k_and_counts() {
        let n_markers = 9usize;
        let n_samples = 37usize;
        let mut state = 0x123456789abcdef0u64;
        let mut genotypes = Vec::with_capacity(n_markers * n_samples);
        for marker in 0..n_markers {
            for sample in 0..n_samples {
                state ^= state << 7;
                state ^= state >> 9;
                state ^= state << 8;
                genotypes.push(
                    (state
                        .wrapping_add((marker as u64).wrapping_mul(0x9e3779b9))
                        .wrapping_add(sample as u64)
                        % 3) as f64,
                );
            }
        }
        let y = (0..n_samples)
            .map(|sample| (sample as f64 * 0.17).sin() + sample as f64 * 0.003)
            .collect::<Vec<_>>();
        let serial = scan_dosage_pairs(&genotypes, n_markers, n_samples, &y, 7).unwrap();
        let representations = build_marker_representations(&genotypes, n_markers, n_samples);
        let parallel = scan_dosage_pairs_from_representations_with_threads(
            representations,
            n_markers,
            n_samples,
            &y,
            7,
            3,
        )
        .unwrap();
        assert_eq!(parallel.pairs_evaluated, serial.pairs_evaluated);
        assert_eq!(parallel.pairs_skipped, serial.pairs_skipped);
        assert_eq!(parallel.candidates.len(), serial.candidates.len());
        for (got, expected) in parallel.candidates.iter().zip(serial.candidates.iter()) {
            assert_eq!((got.first, got.second), (expected.first, expected.second));
            assert_close(got.fit.interaction_score, expected.fit.interaction_score);
        }
    }

    #[test]
    fn all_pair_scan_skips_rank_deficient_linked_pairs() {
        let marker = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];
        let other = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let genotypes = [
            marker[0], marker[1], marker[2], marker[3], marker[4], marker[5], marker[6], marker[7],
            marker[0], marker[1], marker[2], marker[3], marker[4], marker[5], marker[6], marker[7],
            other[0], other[1], other[2], other[3], other[4], other[5], other[6], other[7],
        ];
        let y = [1.0, 3.0, 4.0, 10.0, 1.0, 3.0, 4.0, 10.0];
        let scan = scan_dosage_pairs(&genotypes, 3, 8, &y, 10).unwrap();
        assert_eq!(scan.pairs_evaluated, 2);
        assert_eq!(scan.pairs_skipped, 1);
        assert_eq!(scan.candidates.len(), 2);
    }

    #[test]
    fn blocked_float_scan_matches_pairwise_reference_top_k() {
        let n_markers = 7usize;
        let n_samples = 37usize;
        let mut genotypes = Vec::with_capacity(n_markers * n_samples);
        let mut state = 0x4d595df4d0f33173u64;
        for marker in 0..n_markers {
            for sample in 0..n_samples {
                state ^= state << 7;
                state ^= state >> 9;
                state ^= state << 8;
                let value = ((state
                    .wrapping_add((marker as u64).wrapping_mul(0x9e3779b9))
                    .wrapping_add(sample as u64)
                    % 2001) as f64)
                    / 1000.0;
                genotypes.push(value);
            }
        }
        let y = (0..n_samples)
            .map(|sample| (sample as f64 * 0.17).sin() + sample as f64 * 0.003)
            .collect::<Vec<_>>();
        let representations = build_marker_representations(&genotypes, n_markers, n_samples);
        let float_markers = representations
            .iter()
            .map(|representation| match representation.as_ref().unwrap() {
                MarkerRepresentation::Float(marker) => marker,
                _ => panic!("fractional test marker must use float representation"),
            })
            .collect::<Vec<_>>();
        let blocked = scan_float_pairs_blocked(&float_markers, n_samples, &y, 10).unwrap();
        let mut reference = PairScanAccumulator::default();
        for first in 0..n_markers - 1 {
            for second in first + 1..n_markers {
                let moments =
                    pair_moments_float_float(float_markers[first], float_markers[second], &y);
                let score = score_pair_moments_scalar(&moments).unwrap();
                reference.pairs_evaluated += 1;
                push_top_k_score(
                    &mut reference.heap,
                    ScoredPair {
                        first,
                        second,
                        score,
                    },
                    10,
                );
            }
        }
        let mut blocked_pairs = blocked
            .heap
            .into_iter()
            .map(|Reverse(entry)| entry.0)
            .collect::<Vec<_>>();
        let mut reference_pairs = reference
            .heap
            .into_iter()
            .map(|Reverse(entry)| entry.0)
            .collect::<Vec<_>>();
        let sort_pairs = |pairs: &mut Vec<ScoredPair>| {
            pairs.sort_by(|a, b| {
                score_cmp(
                    b.score.interaction_score,
                    b.first,
                    b.second,
                    a.score.interaction_score,
                    a.first,
                    a.second,
                )
            });
        };
        sort_pairs(&mut blocked_pairs);
        sort_pairs(&mut reference_pairs);
        assert_eq!(blocked.pairs_evaluated, reference.pairs_evaluated);
        assert_eq!(blocked_pairs.len(), reference_pairs.len());
        for (got, expected) in blocked_pairs.iter().zip(reference_pairs.iter()) {
            assert_eq!((got.first, got.second), (expected.first, expected.second));
            assert_close(
                got.score.interaction_score,
                expected.score.interaction_score,
            );
            assert_close(
                got.score.interaction_delta_rss,
                expected.score.interaction_delta_rss,
            );
        }
    }

    fn assert_close(got: f64, expected: f64) {
        assert!(
            (got - expected).abs() < 1.0e-8 || (got.is_infinite() && expected.is_infinite()),
            "got {got}, expected {expected}"
        );
    }
}
