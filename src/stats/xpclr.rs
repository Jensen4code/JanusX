//! XP-CLR selection scans for PLINK SNP-major BED files.
//!
//! The implementation follows the likelihood model used by
//! `hardingnj/xpclr` (v1.1.2).  BED rows remain packed while allele counts are
//! collected; only the bounded set of sites in the current window is decoded
//! to dosages for Rogers--Huff LD weighting.  Windows are evaluated in a
//! Rayon pool and results are written after normalization.

use crate::stats_common::{
    emit_progress_callback, get_cached_pool, map_err_string_to_py, progress_step,
};
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::seq::index::sample as sample_indices;
use rand::SeedableRng;
use rayon::prelude::*;
use statrs::function::gamma::ln_gamma;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;

const BED_HEADER_LEN: usize = 3;
const XPCLR_BLOCK_SITES: usize = 4096;
const XPCLR_HEADER: &str =
    "chrom\tstart\tend\tmodelL\tnullL\tsel_coef\tnsnp\tnsnp_avail\tXPCLR\tnorm_XPCLR";
const SELECTION_COEFFICIENTS: [f64; 16] = [
    0.0, 0.00001, 0.00005, 0.0001, 0.0002, 0.0004, 0.0006, 0.0008, 0.001, 0.003, 0.005, 0.01, 0.05,
    0.08, 0.1, 0.15,
];

#[derive(Clone, Debug)]
struct BimSite {
    chrom: String,
    snp: String,
    pos: i64,
}

#[derive(Clone, Debug)]
struct FamSample {
    fid: String,
    iid: String,
}

#[derive(Clone, Debug)]
struct SiteCounts {
    pos: i64,
    alt1: usize,
    calls1: usize,
    q2: f64,
    genetic_distance: f64,
    null_likelihood: f64,
    dosages: Vec<i8>,
}

#[derive(Clone, Debug)]
struct RawWindowResult {
    chrom: String,
    start: i64,
    stop: i64,
    pos_start: i64,
    pos_stop: i64,
    n_snps: usize,
    n_snps_avail: usize,
    model_l: f64,
    null_l: f64,
    sel_coef: f64,
}

struct BedInput {
    bed: File,
    n_samples: usize,
    n_sites: usize,
    bytes_per_snp: usize,
    fam: Vec<FamSample>,
}

#[derive(Clone, Debug)]
struct XpclrWindowWork {
    start: i64,
    stop: i64,
    n_snps_avail: usize,
    sites: Vec<SiteCounts>,
}

#[derive(Default)]
struct XpclrResultReorder {
    next_sequence: usize,
    pending: BTreeMap<usize, RawWindowResult>,
}

impl XpclrResultReorder {
    fn push(&mut self, sequence_id: usize, result: RawWindowResult) -> Vec<RawWindowResult> {
        self.pending.insert(sequence_id, result);
        let mut ready = Vec::new();
        while let Some(result) = self.pending.remove(&self.next_sequence) {
            ready.push(result);
            self.next_sequence = self.next_sequence.saturating_add(1);
        }
        ready
    }
}

struct XpclrWindowTaskResult {
    sequence_id: usize,
    result: Result<RawWindowResult, String>,
}

struct XpclrWindowPipeline {
    sender: SyncSender<XpclrWindowTaskResult>,
    receiver: Receiver<XpclrWindowTaskResult>,
    reorder: XpclrResultReorder,
    next_submit: usize,
    in_flight: usize,
    max_in_flight: usize,
}

impl XpclrWindowPipeline {
    fn new(max_in_flight: usize) -> Self {
        let capacity = max_in_flight.max(1);
        let (sender, receiver) = sync_channel(capacity);
        Self {
            sender,
            receiver,
            reorder: XpclrResultReorder::default(),
            next_submit: 0,
            in_flight: 0,
            max_in_flight: capacity,
        }
    }

    fn submit(
        &mut self,
        pool: &rayon::ThreadPool,
        work: XpclrWindowWork,
        omega: f64,
        ld_cutoff: f64,
        minsnps: usize,
        chrom: &str,
        spool: &mut XpclrRawSpool,
    ) -> Result<(), String> {
        if self.in_flight >= self.max_in_flight {
            self.drain_one(spool)?;
        }
        let sequence_id = self.next_submit;
        self.next_submit = self.next_submit.saturating_add(1);
        self.in_flight = self.in_flight.saturating_add(1);
        let sender = self.sender.clone();
        let chrom = chrom.to_string();
        pool.spawn(move || {
            // A panic in a detached Rayon task must become a receiver error;
            // otherwise the bounded producer could wait forever for a result
            // that will never be sent.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                compute_window_work(&work, omega, ld_cutoff, minsnps, &chrom)
            }))
            .map_err(|_| "XP-CLR window task panicked".to_string());
            let _ = sender.send(XpclrWindowTaskResult {
                sequence_id,
                result,
            });
        });
        Ok(())
    }

    fn drain_one(&mut self, spool: &mut XpclrRawSpool) -> Result<(), String> {
        let task = self.receiver.recv().map_err(|error| {
            format!("XP-CLR window pipeline disconnected before result was received: {error}")
        })?;
        self.in_flight = self.in_flight.checked_sub(1).ok_or_else(|| {
            "XP-CLR window pipeline received more results than submitted".to_string()
        })?;
        let result = task.result?;
        let ready = self.reorder.push(task.sequence_id, result);
        spool.append(&ready)
    }

    fn drain_all(&mut self, spool: &mut XpclrRawSpool) -> Result<(), String> {
        while self.in_flight > 0 {
            self.drain_one(spool)?;
        }
        if !self.reorder.pending.is_empty() {
            return Err("XP-CLR window pipeline has a gap in result sequence".to_string());
        }
        self.reorder = XpclrResultReorder::default();
        self.next_submit = 0;
        Ok(())
    }
}

struct XpclrRawSpool {
    path: PathBuf,
    writer: BufWriter<File>,
    count: usize,
}

impl XpclrRawSpool {
    fn new(output: &Path, chrom_idx: usize) -> Result<Self, String> {
        let parent = output.parent().unwrap_or_else(|| Path::new("."));
        let stem = output
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("xpclr.tsv");
        let path = parent.join(format!(
            ".{stem}.janusx-raw.{}.{}",
            std::process::id(),
            chrom_idx
        ));
        let file = File::create(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Self {
            path,
            writer: BufWriter::new(file),
            count: 0,
        })
    }

    fn append(&mut self, results: &[RawWindowResult]) -> Result<(), String> {
        for result in results {
            for value in [result.start, result.stop, result.pos_start, result.pos_stop] {
                self.writer
                    .write_all(&value.to_le_bytes())
                    .map_err(|e| format!("{}: {e}", self.path.display()))?;
            }
            for value in [result.n_snps as u64, result.n_snps_avail as u64] {
                self.writer
                    .write_all(&value.to_le_bytes())
                    .map_err(|e| format!("{}: {e}", self.path.display()))?;
            }
            for value in [result.model_l, result.null_l, result.sel_coef] {
                self.writer
                    .write_all(&value.to_le_bytes())
                    .map_err(|e| format!("{}: {e}", self.path.display()))?;
            }
            self.count += 1;
        }
        Ok(())
    }

    fn finalize<W: Write>(
        &mut self,
        writer: &mut W,
        chrom: &str,
        output: &str,
    ) -> Result<(usize, usize), String> {
        self.writer
            .flush()
            .map_err(|e| format!("{}: {e}", self.path.display()))?;
        let mut reader = BufReader::new(
            File::open(&self.path).map_err(|e| format!("{}: {e}", self.path.display()))?,
        );
        let mut finite_count = 0usize;
        let mut sum = 0.0;
        while let Some(result) = read_raw_window(&mut reader, chrom)? {
            if let Some(value) = raw_xpclr_value(&result) {
                sum += value;
                finite_count += 1;
            }
        }
        let mean = if finite_count > 0 {
            sum / finite_count as f64
        } else {
            f64::NAN
        };

        let mut variance = 0.0;
        if finite_count > 0 {
            let mut reader = BufReader::new(
                File::open(&self.path).map_err(|e| format!("{}: {e}", self.path.display()))?,
            );
            while let Some(result) = read_raw_window(&mut reader, chrom)? {
                if let Some(value) = raw_xpclr_value(&result) {
                    variance += (value - mean).powi(2);
                }
            }
            variance /= finite_count as f64;
        }
        let sd = variance.sqrt();
        let mut valid = 0usize;
        let mut reader = BufReader::new(
            File::open(&self.path).map_err(|e| format!("{}: {e}", self.path.display()))?,
        );
        while let Some(result) = read_raw_window(&mut reader, chrom)? {
            let xpclr = raw_xpclr_value(&result).unwrap_or(f64::NAN);
            let xpclr_norm = if xpclr.is_finite() && sd > 0.0 {
                (xpclr - mean) / sd
            } else {
                f64::NAN
            };
            if xpclr.is_finite() {
                valid += 1;
            }
            write_xpclr_row(writer, &result, xpclr, xpclr_norm, output)?;
        }
        std::fs::remove_file(&self.path).map_err(|e| format!("{}: {e}", self.path.display()))?;
        Ok((self.count, valid))
    }
}

impl Drop for XpclrRawSpool {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn read_raw_window<R: Read>(
    reader: &mut R,
    chrom: &str,
) -> Result<Option<RawWindowResult>, String> {
    let mut first = [0u8; 8];
    let read = reader
        .read(&mut first)
        .map_err(|e| format!("XP-CLR raw spool ({chrom}): {e}"))?;
    if read == 0 {
        return Ok(None);
    }
    if read != first.len() {
        return Err(format!("XP-CLR raw spool ({chrom}) is truncated"));
    }
    let read_i64 = |reader: &mut R, first: Option<[u8; 8]>| -> Result<i64, String> {
        let mut bytes = first.unwrap_or([0u8; 8]);
        if first.is_none() {
            reader
                .read_exact(&mut bytes)
                .map_err(|e| format!("XP-CLR raw spool ({chrom}): {e}"))?;
        }
        Ok(i64::from_le_bytes(bytes))
    };
    let start = read_i64(reader, Some(first))?;
    let stop = read_i64(reader, None)?;
    let pos_start = read_i64(reader, None)?;
    let pos_stop = read_i64(reader, None)?;
    let mut u64_bytes = [0u8; 8];
    reader
        .read_exact(&mut u64_bytes)
        .map_err(|e| format!("XP-CLR raw spool ({chrom}): {e}"))?;
    let n_snps = u64::from_le_bytes(u64_bytes) as usize;
    reader
        .read_exact(&mut u64_bytes)
        .map_err(|e| format!("XP-CLR raw spool ({chrom}): {e}"))?;
    let n_snps_avail = u64::from_le_bytes(u64_bytes) as usize;
    let mut f64_bytes = [0u8; 8];
    let mut read_f64 = || -> Result<f64, String> {
        reader
            .read_exact(&mut f64_bytes)
            .map_err(|e| format!("XP-CLR raw spool ({chrom}): {e}"))?;
        Ok(f64::from_le_bytes(f64_bytes))
    };
    Ok(Some(RawWindowResult {
        chrom: chrom.to_string(),
        start,
        stop,
        pos_start,
        pos_stop,
        n_snps,
        n_snps_avail,
        model_l: read_f64()?,
        null_l: read_f64()?,
        sel_coef: read_f64()?,
    }))
}

fn raw_xpclr_value(result: &RawWindowResult) -> Option<f64> {
    (result.model_l.is_finite() && result.null_l.is_finite())
        .then(|| 2.0 * (result.model_l - result.null_l))
}

fn format_xpclr_value(value: f64) -> String {
    if value.is_finite() && value.abs() < 0.0001 {
        format!("{value:.4e}")
    } else {
        format!("{value:.4}")
    }
}

fn write_xpclr_row<W: Write>(
    writer: &mut W,
    raw: &RawWindowResult,
    xpclr: f64,
    xpclr_norm: f64,
    output: &str,
) -> Result<(), String> {
    writeln!(
        writer,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        raw.chrom,
        raw.start,
        raw.stop,
        format_xpclr_value(raw.model_l),
        format_xpclr_value(raw.null_l),
        format_xpclr_value(raw.sel_coef),
        raw.n_snps,
        raw.n_snps_avail,
        format_xpclr_value(xpclr),
        format_xpclr_value(xpclr_norm),
    )
    .map_err(|e| format!("{output}: {e}"))
}

struct XpclrWindowAccumulator {
    next_start: i64,
    pending: VecDeque<SiteCounts>,
}

impl XpclrWindowAccumulator {
    fn new() -> Self {
        Self {
            next_start: 1,
            pending: VecDeque::new(),
        }
    }

    fn push(
        &mut self,
        site: SiteCounts,
        window: i64,
        step: i64,
        maxsnps: usize,
        seed: u64,
        chrom_idx: usize,
    ) -> Vec<XpclrWindowWork> {
        let mut emitted = Vec::new();
        while self.next_start < site.pos {
            let stop = self.next_start.saturating_add(window - 1);
            // XP-CLR follows searchsorted(stop), so `stop` is exclusive.
            if site.pos < stop {
                break;
            }
            if self.pending.is_empty() {
                emitted.push(self.make_work(window, maxsnps, seed, chrom_idx));
                self.advance(step);
                continue;
            }
            emitted.push(self.make_work(window, maxsnps, seed, chrom_idx));
            self.advance(step);
        }
        let stop = self.next_start.saturating_add(window - 1);
        if site.pos >= self.next_start && site.pos < stop {
            self.pending.push_back(site);
        }
        emitted
    }

    fn finish(
        &mut self,
        max_pos: i64,
        window: i64,
        step: i64,
        maxsnps: usize,
        seed: u64,
        chrom_idx: usize,
    ) -> Vec<XpclrWindowWork> {
        let mut emitted = Vec::new();
        while self.next_start < max_pos {
            emitted.push(self.make_work(window, maxsnps, seed, chrom_idx));
            self.advance(step);
        }
        emitted
    }

    fn make_work(
        &self,
        window: i64,
        maxsnps: usize,
        seed: u64,
        chrom_idx: usize,
    ) -> XpclrWindowWork {
        let stop = self.next_start.saturating_add(window - 1);
        let available = self
            .pending
            .iter()
            .take_while(|site| site.pos < stop)
            .count();
        let indices = (0..available).collect::<Vec<_>>();
        let selected = select_sites_seeded(
            &indices,
            maxsnps,
            window_seed(
                seed,
                chrom_idx,
                self.next_start,
                self.next_start.saturating_add(window - 1),
            ),
        );
        XpclrWindowWork {
            start: self.next_start,
            stop,
            n_snps_avail: available,
            sites: selected
                .into_iter()
                .map(|idx| self.pending[idx].clone())
                .collect(),
        }
    }

    fn advance(&mut self, step: i64) {
        self.next_start = self.next_start.saturating_add(step);
        while self
            .pending
            .front()
            .is_some_and(|site| site.pos < self.next_start)
        {
            self.pending.pop_front();
        }
    }
}

#[derive(Clone, Debug)]
struct SampleMask {
    words: Vec<u64>,
}

#[derive(Clone, Copy, Debug, Default)]
struct BitPlanes {
    ref_bits: u64,
    het_bits: u64,
    alt_bits: u64,
}

#[inline]
fn words_for_samples(n_samples: usize) -> usize {
    n_samples.div_ceil(64)
}

#[inline]
fn set_sample_bit(words: &mut [u64], sample_idx: usize) {
    words[sample_idx >> 6] |= 1u64 << (sample_idx & 63);
}

fn sample_mask(indices: &[usize], n_samples: usize) -> Result<SampleMask, String> {
    let mut words = vec![0u64; words_for_samples(n_samples)];
    let mut seen = HashSet::new();
    for &sample_idx in indices {
        if sample_idx >= n_samples {
            return Err(format!(
                "sample index {sample_idx} >= the BED sample count {n_samples}"
            ));
        }
        if !seen.insert(sample_idx) {
            return Err(format!("duplicate sample index {sample_idx}"));
        }
        set_sample_bit(&mut words, sample_idx);
    }
    if indices.is_empty() {
        return Err("population sample list is empty".to_string());
    }
    Ok(SampleMask { words })
}

const fn build_byte_plane_lut(code: u8) -> [u8; 256] {
    let mut lut = [0u8; 256];
    let mut packed = 0usize;
    while packed < 256 {
        let mut lane = 0usize;
        let mut bits = 0u8;
        while lane < 4 {
            if ((packed as u8 >> (lane * 2)) & 0b11) == code {
                bits |= 1u8 << lane;
            }
            lane += 1;
        }
        lut[packed] = bits;
        packed += 1;
    }
    lut
}

const REF_BYTE_PLANE_LUT: [u8; 256] = build_byte_plane_lut(0b00);
const HET_BYTE_PLANE_LUT: [u8; 256] = build_byte_plane_lut(0b10);
const ALT_BYTE_PLANE_LUT: [u8; 256] = build_byte_plane_lut(0b11);

#[inline]
fn transpose_bed_word(row: &[u8], sample_start: usize, n_samples: usize) -> BitPlanes {
    let end = (sample_start + 64).min(n_samples);
    let valid = end - sample_start;
    let byte_count = valid.div_ceil(4);
    let byte_start = sample_start / 4;
    let mut out = BitPlanes::default();
    for byte_idx in 0..byte_count {
        let packed = row[byte_start + byte_idx] as usize;
        let shift = byte_idx * 4;
        out.ref_bits |= u64::from(REF_BYTE_PLANE_LUT[packed]) << shift;
        out.het_bits |= u64::from(HET_BYTE_PLANE_LUT[packed]) << shift;
        out.alt_bits |= u64::from(ALT_BYTE_PLANE_LUT[packed]) << shift;
    }
    if valid < 64 {
        let valid_mask = (1u64 << valid) - 1;
        out.ref_bits &= valid_mask;
        out.het_bits &= valid_mask;
        out.alt_bits &= valid_mask;
    }
    out
}

fn count_population_bitwise(row: &[u8], n_samples: usize, mask: &SampleMask) -> (usize, usize) {
    // PLINK BED encodes 00/10/11 as A1/A1, A1/A2, A2/A2.  PLINK's VCF
    // representation writes A2 as REF and A1 as ALT, which is the allele
    // orientation used by hardingnj/xpclr.  Therefore XP-CLR's alternate
    // allele count is the A1 dosage (2 * A1/A1 + A1/A2), not the A2 dosage.
    let mut allele1_count = 0usize;
    let mut het_count = 0usize;
    let mut allele2_count = 0usize;
    for (word_idx, sample_start) in (0..n_samples).step_by(64).enumerate() {
        let planes = transpose_bed_word(row, sample_start, n_samples);
        let mask_word = mask.words[word_idx];
        allele1_count += (planes.ref_bits & mask_word).count_ones() as usize;
        het_count += (planes.het_bits & mask_word).count_ones() as usize;
        allele2_count += (planes.alt_bits & mask_word).count_ones() as usize;
    }
    (
        het_count + 2 * allele1_count,
        allele1_count + het_count + allele2_count,
    )
}

/// XP-CLR's conversion from recombination distance `r` and selection
/// coefficient `s` to the correlation parameter `c`.
pub(crate) fn determine_c(r: f64, s: f64) -> f64 {
    if s <= 0.0 {
        return 1.0;
    }
    let exponent = -libm::log(2.0 * 20_000.0) * r.max(1.0e-7) / s;
    (1.0 - libm::exp(exponent)).mul_add(100_000.0, 0.0).round() / 100_000.0
}

/// Mean XP-CLR drift statistic used as the variance multiplier.
pub(crate) fn determine_omega(q1: &[f64], q2: &[f64]) -> f64 {
    if q1.len() != q2.len() || q1.is_empty() {
        return f64::NAN;
    }
    q1.iter()
        .zip(q2)
        .map(|(&p1, &p2)| {
            let denominator = p2 * (1.0 - p2);
            if denominator <= 0.0 {
                f64::NAN
            } else {
                (p1 - p2).powi(2) / denominator
            }
        })
        .sum::<f64>()
        / q1.len() as f64
}

/// Return half-open genomic window starts as inclusive `(start, stop)`
/// coordinates.  `stop` is the exclusive scan bound, matching the reference
/// implementation's `arange(start, stop, step)` behavior.
pub(crate) fn window_starts(start: i64, stop: i64, window: i64, step: i64) -> Vec<(i64, i64)> {
    if stop <= start || window <= 0 || step <= 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut current = start;
    while current < stop {
        out.push((current, current.saturating_add(window - 1)));
        current = current.saturating_add(step);
    }
    out
}

/// Normalize finite XP-CLR scores while preserving invalid windows as NaN.
pub(crate) fn normalize_scores(values: &[f64]) -> Vec<f64> {
    let finite = values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return values.iter().map(|_| f64::NAN).collect();
    }
    let mean = finite.iter().sum::<f64>() / finite.len() as f64;
    let variance = finite.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / finite.len() as f64;
    let sd = variance.sqrt();
    values
        .iter()
        .map(|v| {
            if v.is_finite() && sd > 0.0 {
                (v - mean) / sd
            } else {
                f64::NAN
            }
        })
        .collect()
}

#[inline]
fn binomial_log_coefficient(k: usize, n: usize) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    ln_gamma((n + 1) as f64) - ln_gamma((k + 1) as f64) - ln_gamma((n - k + 1) as f64)
}

#[inline]
fn binomial_pmf_with_log_coefficient(log_coefficient: f64, k: usize, n: usize, p: f64) -> f64 {
    if k > n || !(0.0..=1.0).contains(&p) {
        return 0.0;
    }
    if p == 0.0 {
        return usize::from(k == 0) as f64;
    }
    if p == 1.0 {
        return usize::from(k == n) as f64;
    }
    let log_probability =
        log_coefficient + k as f64 * libm::log(p) + (n - k) as f64 * libm::log1p(-p);
    libm::exp(log_probability)
}

#[inline]
fn xpclr_pdf_with_constants(
    p1: f64,
    c: f64,
    p2: f64,
    c2: f64,
    normalizer: f64,
    denominator: f64,
) -> f64 {
    let mut result = 0.0;
    // The reference implementation adds both tails when c > 0.5; these are
    // deliberately independent conditions rather than an if/else chain.
    if p1 < c {
        let b = (c - p1) / c2;
        let centered = p1 - c * p2;
        result += normalizer * b * libm::exp(-(centered * centered) / denominator);
    }
    // ``hardingnj/xpclr`` uses ``bisect_right`` on the scalar sample, so the
    // right tail starts strictly after ``1-c`` (not at the boundary).  This
    // matters because the adaptive integrator evaluates split endpoints.
    if p1 > 1.0 - c {
        let shifted = p1 + c - 1.0;
        let b = shifted / c2;
        let centered = shifted - c * p2;
        result += normalizer * b * libm::exp(-(centered * centered) / denominator);
    }
    result
}

fn adaptive_simpson<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64, eps: f64, depth: u32) -> f64 {
    if !(b > a) {
        return 0.0;
    }
    let c = (a + b) * 0.5;
    let fa = f(a);
    let fb = f(b);
    let fc = f(c);
    let whole = (b - a) * (fa + 4.0 * fc + fb) / 6.0;
    adaptive_simpson_inner(f, a, b, fa, fb, fc, whole, eps, depth)
}

fn adaptive_simpson_inner<F: Fn(f64) -> f64>(
    f: &F,
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
    fc: f64,
    whole: f64,
    eps: f64,
    depth: u32,
) -> f64 {
    if !whole.is_finite() {
        return 0.0;
    }
    let c = (a + b) * 0.5;
    let left_mid = (a + c) * 0.5;
    let right_mid = (c + b) * 0.5;
    let f_left_mid = f(left_mid);
    let f_right_mid = f(right_mid);
    let left = (c - a) * (fa + 4.0 * f_left_mid + fc) / 6.0;
    let right = (b - c) * (fc + 4.0 * f_right_mid + fb) / 6.0;
    let delta = left + right - whole;
    if depth == 0 || delta.abs() <= 15.0 * eps {
        return left + right + delta / 15.0;
    }
    adaptive_simpson_inner(f, a, c, fa, fc, f_left_mid, left, eps * 0.5, depth - 1)
        + adaptive_simpson_inner(f, c, b, fc, fb, f_right_mid, right, eps * 0.5, depth - 1)
}

fn integrate_split<F: Fn(f64) -> f64>(f: F, c: f64, extra_points: &[f64]) -> f64 {
    const A: f64 = 0.001;
    const B: f64 = 0.999;
    let mut points = vec![A, B];
    for point in [c, 1.0 - c].into_iter().chain(extra_points.iter().copied()) {
        if point > A && point < B {
            points.push(point);
        }
    }
    points.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(Ordering::Equal));
    points.dedup_by(|lhs, rhs| (*lhs - *rhs).abs() < 1.0e-14);
    points
        .windows(2)
        // The reference uses scipy.quad(epsrel=1e-3, epsabs=0).  A fixed
        // 1e-8 absolute tolerance loses the numerator when its integral is
        // itself around 1e-8, so use a tighter absolute floor and more depth.
        .map(|pair| adaptive_simpson(&f, pair[0], pair[1], 1.0e-12, 28))
        .sum()
}

/// The Chen likelihood ratio for one SNP/window observation.  The tuple is
/// `(alternate alleles in pop1, called alleles in pop1, c, pop2 frequency,
/// variance)` and mirrors `xpclr.methods.chen_likelihood`.
pub(crate) fn chen_likelihood(data: (usize, usize, f64, f64, f64)) -> f64 {
    let (xj, nj, c, p2, variance) = data;
    if nj == 0 || !c.is_finite() || !p2.is_finite() || variance <= 0.0 {
        return 0.0;
    }
    let sample_frequency = xj as f64 / nj as f64;
    // The binomial coefficient is independent of the quadrature point.  The
    // previous implementation recomputed three lgamma values at every
    // adaptive-Simpson evaluation, which dominated large XP-CLR windows.
    // Hoisting it preserves the exact arithmetic expression while removing
    // that redundant work from the hot loop.
    let log_coefficient = binomial_log_coefficient(xj, nj);
    let c2 = c * c;
    if c2 == 0.0 {
        return -1800.0;
    }
    let normalizer = (2.0 * std::f64::consts::PI * variance).sqrt().recip();
    let denominator = 2.0 * c2 * variance;
    let marginal = integrate_split(
        |p| xpclr_pdf_with_constants(p, c, p2, c2, normalizer, denominator),
        c,
        &[],
    );
    let integrated = integrate_split(
        |p| {
            xpclr_pdf_with_constants(p, c, p2, c2, normalizer, denominator)
                * binomial_pmf_with_log_coefficient(log_coefficient, xj, nj, p)
        },
        c,
        &[sample_frequency],
    );
    if marginal <= 0.0 || integrated <= 0.0 || !marginal.is_finite() || !integrated.is_finite() {
        -1800.0
    } else {
        libm::log(integrated) - libm::log(marginal)
    }
}

fn calculate_xpclr(
    rows: &[(usize, usize, f64, f64, f64)],
    omega: f64,
    null_likelihood: f64,
) -> (f64, f64, f64) {
    if rows.is_empty() || !omega.is_finite() || omega < 0.0 || !null_likelihood.is_finite() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let mut maximum_likelihood = null_likelihood;
    let mut maximum_selection = 0.0;
    // The null likelihood is independent of the window distance and selection
    // grid.  It is supplied by the caller after being cached once per site.
    for &selection in SELECTION_COEFFICIENTS.iter().skip(1) {
        let mut likelihood = 0.0;
        for &(alt1, n1, distance, q2, weight) in rows {
            let variance = omega * q2 * (1.0 - q2);
            let c = determine_c(distance, selection);
            likelihood += weight * chen_likelihood((alt1, n1, c, q2, variance));
        }
        if likelihood > maximum_likelihood {
            maximum_likelihood = likelihood;
            maximum_selection = selection;
        } else {
            // `hardingnj/xpclr` stops once its monotonic selection grid starts
            // decreasing, so retain the same early-exit behavior.
            break;
        }
    }
    (maximum_likelihood, null_likelihood, maximum_selection)
}

#[inline]
fn bed_dosage(row: &[u8], sample_idx: usize) -> i8 {
    match (row[sample_idx >> 2] >> ((sample_idx & 3) * 2)) & 0b11 {
        0b00 => 2,
        0b10 => 1,
        0b11 => 0,
        _ => -1,
    }
}

fn decode_population_into(row: &[u8], sample_indices: &[usize], destination: &mut [i8]) {
    debug_assert_eq!(sample_indices.len(), destination.len());
    for (&sample_idx, dosage) in sample_indices.iter().zip(destination.iter_mut()) {
        *dosage = bed_dosage(row, sample_idx);
    }
}

fn rogers_huff_r_squared(lhs: &[i8], rhs: &[i8]) -> Option<f64> {
    let mut n = 0.0;
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;
    for (&x, &y) in lhs.iter().zip(rhs) {
        // scikit-allel's ``to_n_alt(fill=0)`` is used by the reference
        // implementation for unphased LD, so an individual missing call is
        // represented as dosage zero rather than removed pairwise.
        let x = x.max(0) as f64;
        let y = y.max(0) as f64;
        n += 1.0;
        sx += x;
        sy += y;
        sxx += x * x;
        syy += y * y;
        sxy += x * y;
    }
    if n == 0.0 {
        return None;
    }
    let covariance = sxy - sx * sy / n;
    let x_variance = sxx - sx * sx / n;
    let y_variance = syy - sy * sy / n;
    if x_variance <= 0.0 || y_variance <= 0.0 {
        None
    } else {
        Some((covariance * covariance) / (x_variance * y_variance))
    }
}

fn determine_weights_flat(
    dosages: &[i8],
    n_sites: usize,
    n_samples: usize,
    ld_cutoff: f64,
) -> Vec<f64> {
    debug_assert_eq!(dosages.len(), n_sites * n_samples);
    let mut highly_correlated = vec![0usize; n_sites];
    for lhs in 0..n_sites {
        let lhs_start = lhs * n_samples;
        let lhs_dosages = &dosages[lhs_start..lhs_start + n_samples];
        for rhs in lhs + 1..n_sites {
            let rhs_start = rhs * n_samples;
            let above_cutoff =
                rogers_huff_r_squared(lhs_dosages, &dosages[rhs_start..rhs_start + n_samples])
                    .map(|r_squared| r_squared > ld_cutoff)
                    .unwrap_or(true);
            if above_cutoff {
                highly_correlated[lhs] += 1;
                highly_correlated[rhs] += 1;
            }
        }
    }
    highly_correlated
        .into_iter()
        .map(|count| 1.0 / (1.0 + count as f64))
        .collect()
}

fn determine_weights_for_sites(sites: &[SiteCounts], ld_cutoff: f64) -> Vec<f64> {
    let mut highly_correlated = vec![0usize; sites.len()];
    for lhs in 0..sites.len() {
        for rhs in lhs + 1..sites.len() {
            let above_cutoff = rogers_huff_r_squared(&sites[lhs].dosages, &sites[rhs].dosages)
                .map(|r_squared| r_squared > ld_cutoff)
                .unwrap_or(true);
            if above_cutoff {
                highly_correlated[lhs] += 1;
                highly_correlated[rhs] += 1;
            }
        }
    }
    highly_correlated
        .into_iter()
        .map(|count| 1.0 / (1.0 + count as f64))
        .collect()
}

fn read_fam(path: &str) -> Result<Vec<FamSample>, String> {
    let file = File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut samples = Vec::new();
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("{path}: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let columns = line.split_whitespace().collect::<Vec<_>>();
        if columns.len() < 2 {
            return Err(format!("malformed FAM line {}: {}", line_no + 1, line));
        }
        samples.push(FamSample {
            fid: columns[0].to_string(),
            iid: columns[1].to_string(),
        });
    }
    if samples.is_empty() {
        return Err(format!("{path}: no samples found"));
    }
    Ok(samples)
}

struct BimReader {
    path: String,
    reader: BufReader<File>,
    line_no: usize,
    remaining_sites: Option<usize>,
}

impl BimReader {
    fn open_at(
        prefix: &str,
        bim_offset: u64,
        remaining_sites: Option<usize>,
    ) -> Result<Self, String> {
        let path = format!("{prefix}.bim");
        let file = File::open(&path).map_err(|e| format!("{path}: {e}"))?;
        let mut reader = BufReader::new(file);
        if bim_offset > 0 {
            reader
                .seek(SeekFrom::Start(bim_offset))
                .map_err(|e| format!("{path}: {e}"))?;
        }
        Ok(Self {
            path,
            reader,
            line_no: 0,
            remaining_sites,
        })
    }

    fn next_site(&mut self) -> Result<Option<BimSite>, String> {
        if self.remaining_sites == Some(0) {
            return Ok(None);
        }
        let mut line = String::new();
        loop {
            line.clear();
            if self
                .reader
                .read_line(&mut line)
                .map_err(|e| format!("{}: {e}", self.path))?
                == 0
            {
                return Ok(None);
            }
            self.line_no += 1;
            if !line.trim().is_empty() {
                break;
            }
        }
        let raw = line.trim_end();
        let columns = raw.split_whitespace().collect::<Vec<_>>();
        if columns.len() < 4 {
            return Err(format!(
                "malformed BIM line {}:{}: {raw}",
                self.path, self.line_no
            ));
        }
        let pos = columns[3].parse::<i64>().map_err(|e| {
            format!(
                "invalid BIM position at {}:{}: {e}",
                self.path, self.line_no
            )
        })?;
        if let Some(remaining) = self.remaining_sites.as_mut() {
            *remaining -= 1;
        }
        Ok(Some(BimSite {
            chrom: columns[0].to_string(),
            snp: columns[1].to_string(),
            pos,
        }))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct BimScanPlan {
    site_start: usize,
    n_sites: usize,
    bim_offset: u64,
    bed_offset: u64,
}

fn scan_bim_range(
    prefix: &str,
    chrom_filter: Option<&str>,
    total_sites: usize,
    bytes_per_snp: usize,
) -> Result<BimScanPlan, String> {
    let prefix = normalize_prefix(prefix);
    let path = format!("{prefix}.bim");
    let file = File::open(&path).map_err(|e| format!("{path}: {e}"))?;
    let mut reader = BufReader::new(file);
    let Some(target) = chrom_filter
        .map(str::trim)
        .filter(|chrom| !chrom.is_empty())
    else {
        return Ok(BimScanPlan {
            site_start: 0,
            n_sites: total_sites,
            bim_offset: 0,
            bed_offset: BED_HEADER_LEN as u64,
        });
    };

    let mut line = String::new();
    let mut byte_offset = 0u64;
    let mut site_idx = 0usize;
    let mut site_start = None::<usize>;
    let mut bim_offset = None::<u64>;
    let mut n_sites = 0usize;
    let mut left_target = false;
    loop {
        line.clear();
        let bytes_read = reader
            .read_line(&mut line)
            .map_err(|e| format!("{path}: {e}"))?;
        if bytes_read == 0 {
            break;
        }
        let line_start = byte_offset;
        byte_offset = byte_offset.saturating_add(bytes_read as u64);
        if line.trim().is_empty() {
            continue;
        }
        let chrom = line
            .split_whitespace()
            .next()
            .ok_or_else(|| format!("malformed BIM line {}:{}", path, site_idx + 1))?;
        if site_idx >= total_sites {
            return Err(format!(
                "BIM contains more variants than BED: BIM>{total_sites} at {}:{}",
                path,
                site_idx + 1
            ));
        }
        if chrom == target {
            if left_target {
                return Err(format!(
                    "BIM chromosome {target} reappears after a later chromosome; XP-CLR requires chromosome-sorted BIM"
                ));
            }
            if site_start.is_none() {
                site_start = Some(site_idx);
                bim_offset = Some(line_start);
            }
            n_sites += 1;
        } else if site_start.is_some() {
            left_target = true;
        }
        site_idx += 1;
    }
    if site_idx != total_sites {
        return Err(format!(
            "BED/BIM variant count mismatch: BED={total_sites}, BIM={site_idx}"
        ));
    }
    let site_start = site_start.unwrap_or(total_sites);
    let bim_offset = bim_offset.unwrap_or(byte_offset);
    Ok(BimScanPlan {
        site_start,
        n_sites,
        bim_offset,
        bed_offset: BED_HEADER_LEN as u64 + site_start as u64 * bytes_per_snp as u64,
    })
}

fn normalize_prefix(prefix: &str) -> String {
    let trimmed = prefix.trim();
    let lower = trimmed.to_ascii_lowercase();
    if lower.ends_with(".bed") || lower.ends_with(".bim") || lower.ends_with(".fam") {
        trimmed[..trimmed.len() - 4].to_string()
    } else {
        trimmed.to_string()
    }
}

fn open_bed(prefix: &str) -> Result<BedInput, String> {
    let prefix = normalize_prefix(prefix);
    let fam = read_fam(&format!("{prefix}.fam"))?;
    let bed_path = format!("{prefix}.bed");
    let bed = File::open(&bed_path).map_err(|e| format!("{bed_path}: {e}"))?;
    let metadata = bed.metadata().map_err(|e| format!("{bed_path}: {e}"))?;
    if metadata.len() < BED_HEADER_LEN as u64 {
        return Err(format!("{bed_path}: BED file is shorter than its header"));
    }
    let mut header = [0u8; BED_HEADER_LEN];
    let mut header_reader = bed.try_clone().map_err(|e| format!("{bed_path}: {e}"))?;
    header_reader
        .read_exact(&mut header)
        .map_err(|e| format!("{bed_path}: {e}"))?;
    if header != [0x6c, 0x1b, 0x01] {
        return Err(format!("{bed_path}: expected SNP-major PLINK BED header"));
    }
    let n_samples = fam.len();
    let bytes_per_snp = n_samples.div_ceil(4);
    let payload = metadata.len() as usize - BED_HEADER_LEN;
    if bytes_per_snp == 0 || payload == 0 || payload % bytes_per_snp != 0 {
        return Err(format!(
            "{bed_path}: payload has {payload} bytes, not divisible by {bytes_per_snp} bytes/SNP for {n_samples} samples",
        ));
    }
    let n_sites = payload / bytes_per_snp;
    Ok(BedInput {
        bed,
        n_samples,
        n_sites,
        bytes_per_snp,
        fam,
    })
}

fn read_population(path: &str, fam: &[FamSample]) -> Result<Vec<usize>, String> {
    let file = File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut by_key = HashMap::<(&str, &str), usize>::new();
    let mut by_iid = HashMap::<&str, Vec<usize>>::new();
    for (idx, sample) in fam.iter().enumerate() {
        by_key.insert((&sample.fid, &sample.iid), idx);
        by_iid.entry(&sample.iid).or_default().push(idx);
    }
    let mut selected = Vec::new();
    let mut seen = HashSet::new();
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("{path}: {e}"))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let columns = trimmed.split_whitespace().collect::<Vec<_>>();
        let sample_idx = if columns.len() >= 2 {
            by_key
                .get(&(columns[0], columns[1]))
                .copied()
                .ok_or_else(|| {
                    format!(
                        "{path}: sample FID/IID {} {} at line {} is absent from FAM",
                        columns[0],
                        columns[1],
                        line_no + 1
                    )
                })?
        } else {
            let matches = by_iid.get(columns[0]).ok_or_else(|| {
                format!(
                    "{path}: sample IID {} at line {} is absent from FAM",
                    columns[0],
                    line_no + 1
                )
            })?;
            if matches.len() != 1 {
                return Err(format!(
                    "{path}: IID {} at line {} is ambiguous; provide FID IID",
                    columns[0],
                    line_no + 1
                ));
            }
            matches[0]
        };
        if !seen.insert(sample_idx) {
            return Err(format!("{path}: duplicate sample at line {}", line_no + 1));
        }
        selected.push(sample_idx);
    }
    if selected.is_empty() {
        return Err(format!("{path}: no samples found"));
    }
    Ok(selected)
}

fn read_genetic_map(path: &str) -> Result<HashMap<(String, i64), f64>, String> {
    let file = File::open(path).map_err(|e| format!("{path}: {e}"))?;
    let mut map = HashMap::new();
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("{path}: {e}"))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let columns = trimmed.split_whitespace().collect::<Vec<_>>();
        if columns.len() < 4 {
            return Err(format!("malformed map line {}: {}", line_no + 1, line));
        }
        let genetic_distance = columns[2]
            .parse::<f64>()
            .map_err(|e| format!("invalid genetic distance at {}:{}: {e}", path, line_no + 1))?;
        let position = columns[3]
            .parse::<i64>()
            .map_err(|e| format!("invalid map position at {}:{}: {e}", path, line_no + 1))?;
        map.insert((columns[1].to_string(), position), genetic_distance);
    }
    if map.is_empty() {
        return Err(format!("{path}: no map records found"));
    }
    Ok(map)
}

fn select_sites_seeded(sites: &[usize], maxsnps: usize, seed: u64) -> Vec<usize> {
    if sites.len() <= maxsnps {
        return sites.to_vec();
    }
    if maxsnps == 0 {
        return Vec::new();
    }
    if maxsnps == 1 {
        return vec![sites[sites.len() / 2]];
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut selected = sample_indices(&mut rng, sites.len(), maxsnps)
        .into_iter()
        .map(|idx| sites[idx])
        .collect::<Vec<_>>();
    selected.sort_unstable();
    selected
}

#[inline]
fn window_seed(seed: u64, chrom_idx: usize, start: i64, stop: i64) -> u64 {
    let mut value = seed ^ (chrom_idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    value ^= (start as u64).rotate_left(17);
    value ^= (stop as u64).rotate_right(11);
    // SplitMix64 finalizer; the mapping is deterministic and independent of
    // Rayon scheduling.
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn compute_window_work(
    work: &XpclrWindowWork,
    omega: f64,
    ld_cutoff: f64,
    minsnps: usize,
    chrom: &str,
) -> RawWindowResult {
    let n_snps = work.sites.len();
    let n_snps_avail = work.n_snps_avail;
    if n_snps < minsnps {
        return RawWindowResult {
            chrom: chrom.to_string(),
            start: work.start,
            stop: work.stop,
            pos_start: 0,
            pos_stop: 0,
            n_snps,
            n_snps_avail,
            model_l: f64::NAN,
            null_l: f64::NAN,
            sel_coef: f64::NAN,
        };
    }

    let pos_start = work.sites[0].pos;
    let pos_stop = work.sites.last().unwrap().pos;
    let mean_gdist = work
        .sites
        .iter()
        .map(|site| site.genetic_distance)
        .sum::<f64>()
        / work.sites.len() as f64;
    let weights = determine_weights_for_sites(&work.sites, ld_cutoff);
    let rows = work
        .sites
        .iter()
        .zip(weights.iter().copied())
        .map(|(site, weight)| {
            (
                site.alt1,
                site.calls1 * 2,
                (site.genetic_distance - mean_gdist).abs(),
                site.q2,
                weight,
            )
        })
        .collect::<Vec<_>>();
    let null_l = work
        .sites
        .iter()
        .zip(weights.iter())
        .map(|(site, &weight)| weight * site.null_likelihood)
        .sum::<f64>();
    let (model_l, _, sel_coef) = calculate_xpclr(&rows, omega, null_l);
    RawWindowResult {
        chrom: chrom.to_string(),
        start: work.start,
        stop: work.stop,
        pos_start,
        pos_stop,
        n_snps,
        n_snps_avail,
        model_l,
        null_l,
        sel_coef,
    }
}

fn scan_site_counts(
    row: &[u8],
    n_samples: usize,
    bim: &BimSite,
    pop1_mask: &SampleMask,
    pop2_mask: &SampleMask,
    pop2_indices: &[usize],
    map: Option<&HashMap<(String, i64), f64>>,
    rrate: f64,
    omega: Option<f64>,
    decode_dosages: bool,
) -> Result<Option<SiteCounts>, String> {
    let (alt1, calls1) = count_population_bitwise(row, n_samples, pop1_mask);
    let (alt2, calls2) = count_population_bitwise(row, n_samples, pop2_mask);
    if calls1 == 0 || calls2 == 0 {
        return Ok(None);
    }
    let total2 = calls2 * 2;
    // Match xpclr's biallelic, non-fixed, non-singleton pop2 filter.
    if alt2 == 0 || alt2 == total2 || alt2 == 1 || alt2 + 1 == total2 {
        return Ok(None);
    }
    let q2 = alt2 as f64 / total2 as f64;
    let genetic_distance = map
        .map(|map| {
            map.get(&(bim.chrom.clone(), bim.pos))
                .copied()
                .ok_or_else(|| {
                    format!(
                        "map has no record for BIM variant {}:{} ({})",
                        bim.chrom, bim.pos, bim.snp
                    )
                })
        })
        .transpose()?
        .unwrap_or(bim.pos as f64 * rrate);
    let null_likelihood = omega.map(|omega| {
        let variance = omega * q2 * (1.0 - q2);
        chen_likelihood((alt1, calls1 * 2, 1.0, q2, variance))
    });
    let dosages = if decode_dosages {
        let mut dosages = vec![0i8; pop2_indices.len()];
        decode_population_into(row, pop2_indices, &mut dosages);
        dosages
    } else {
        Vec::new()
    };
    Ok(Some(SiteCounts {
        pos: bim.pos,
        alt1,
        calls1,
        q2,
        genetic_distance,
        null_likelihood: null_likelihood.unwrap_or(f64::NAN),
        dosages,
    }))
}

fn scan_block_sites(
    block_payload: &[u8],
    bytes_per_snp: usize,
    n_samples: usize,
    block_sites: &[BimSite],
    pop1_mask: &SampleMask,
    pop2_mask: &SampleMask,
    pop2_indices: &[usize],
    map: Option<&HashMap<(String, i64), f64>>,
    rrate: f64,
    omega: Option<f64>,
    decode_dosages: bool,
    pool: Option<&Arc<rayon::ThreadPool>>,
) -> Vec<Result<Option<SiteCounts>, String>> {
    debug_assert_eq!(block_payload.len(), block_sites.len() * bytes_per_snp);
    let scan = || {
        block_sites
            .par_iter()
            .enumerate()
            .map(|(offset, bim)| {
                let row = &block_payload[offset * bytes_per_snp..(offset + 1) * bytes_per_snp];
                scan_site_counts(
                    row,
                    n_samples,
                    bim,
                    pop1_mask,
                    pop2_mask,
                    pop2_indices,
                    map,
                    rrate,
                    omega,
                    decode_dosages,
                )
            })
            .collect::<Vec<_>>()
    };
    match pool {
        Some(pool) => pool.install(scan),
        None => scan(),
    }
}

fn finish_xpclr_chrom_pipeline(
    pipeline: &mut XpclrWindowPipeline,
    pool: &rayon::ThreadPool,
    accumulator: &mut Option<XpclrWindowAccumulator>,
    max_pos: i64,
    window: i64,
    step: i64,
    maxsnps: usize,
    sample_seed: u64,
    chrom_idx: usize,
    spool: &mut XpclrRawSpool,
    writer: &mut BufWriter<File>,
    omega: f64,
    ld_cutoff: f64,
    minsnps: usize,
    chrom: &str,
    output: &str,
) -> Result<(usize, usize), String> {
    if let Some(mut accumulator) = accumulator.take() {
        for work in accumulator.finish(max_pos, window, step, maxsnps, sample_seed, chrom_idx) {
            pipeline.submit(pool, work, omega, ld_cutoff, minsnps, chrom, spool)?;
        }
    }
    pipeline.drain_all(spool)?;
    spool.finalize(writer, chrom, output)
}

fn run_xpclr(
    prefix: &str,
    pop1_path: &str,
    pop2_path: &str,
    map_path: Option<&str>,
    chrom_filter: Option<&str>,
    output: &str,
    window: i64,
    step: i64,
    maxsnps: usize,
    minsnps: usize,
    ld_cutoff: f64,
    rrate: f64,
    sample_seed: u64,
    threads: usize,
    progress_callback: Option<&Py<PyAny>>,
    progress_every: usize,
) -> Result<(usize, usize), String> {
    if window <= 0 || step <= 0 {
        return Err("window and step must be positive".to_string());
    }
    if maxsnps == 0 || minsnps < 2 {
        return Err("maxsnps must be positive and minsnps must be at least 2".to_string());
    }
    if !(0.0..=1.0).contains(&ld_cutoff) {
        return Err("ld cutoff must be between 0 and 1".to_string());
    }
    if !rrate.is_finite() || rrate < 0.0 {
        return Err("rrate must be finite and non-negative".to_string());
    }
    let input = open_bed(prefix)?;
    if input.n_samples != input.fam.len() {
        return Err("BED metadata length mismatch".to_string());
    }
    let pop1 = read_population(pop1_path, &input.fam)?;
    let pop2 = read_population(pop2_path, &input.fam)?;
    let pop1_set = pop1.iter().copied().collect::<HashSet<_>>();
    if pop2.iter().any(|idx| pop1_set.contains(idx)) {
        return Err("pop1 and pop2 contain overlapping samples".to_string());
    }
    let pop1_mask = sample_mask(&pop1, input.n_samples)?;
    let pop2_mask = sample_mask(&pop2, input.n_samples)?;
    let map = match map_path.map(str::trim).filter(|path| !path.is_empty()) {
        Some(path) => Some(read_genetic_map(path)?),
        None => None,
    };

    let bed_prefix = normalize_prefix(prefix);
    let scan_plan = scan_bim_range(
        &bed_prefix,
        chrom_filter,
        input.n_sites,
        input.bytes_per_snp,
    )?;
    let scan_sites = scan_plan.n_sites;
    // XP-CLR uses the same exact Rayon worker budget for the site scan and
    // the window pipeline.  The Python wrapper validates positive `-t`, while
    // the direct Rust/PyO3 API historically allowed zero to mean "default";
    // make that case deterministic and serial rather than silently falling
    // back to an unrelated global pool.
    let effective_threads = threads.max(1);
    let pool = get_cached_pool(effective_threads)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "XP-CLR failed to create its Rayon thread pool".to_string())?;
    let progress_step = progress_step(scan_sites, progress_every);
    let mut next_progress = progress_step;
    if progress_callback.is_some() {
        emit_progress_callback(progress_callback, 0, 0, scan_sites)?;
    }

    // Pass 1: stream BIM/BED once to estimate the global omega.  No per-site
    // metadata or window index is retained.
    let mut bim_reader = BimReader::open_at(&bed_prefix, scan_plan.bim_offset, Some(scan_sites))?;
    let mut bed = input
        .bed
        .try_clone()
        .map_err(|e| format!("{prefix}.bed: {e}"))?;
    bed.seek(SeekFrom::Start(scan_plan.bed_offset))
        .map_err(|e| format!("{prefix}.bed: {e}"))?;
    let mut omega_sum = 0.0;
    let mut omega_sites = 0usize;
    let mut usable_sites = 0usize;
    let mut block_start = 0usize;
    while block_start < scan_sites {
        let block_len = (scan_sites - block_start).min(XPCLR_BLOCK_SITES);
        let mut block_payload = vec![0u8; block_len * input.bytes_per_snp];
        bed.read_exact(&mut block_payload)
            .map_err(|e| format!("{prefix}.bed: {e}"))?;
        let block_sites = (0..block_len)
            .map(|_| {
                bim_reader.next_site()?.ok_or_else(|| {
                    "BIM ended before all BED variants during XP-CLR omega scan".to_string()
                })
            })
            .collect::<Result<Vec<_>, String>>()?;
        let block_results = scan_block_sites(
            &block_payload,
            input.bytes_per_snp,
            input.n_samples,
            &block_sites,
            &pop1_mask,
            &pop2_mask,
            &pop2,
            map.as_ref(),
            rrate,
            None,
            false,
            Some(&pool),
        );
        for result in block_results {
            let Some(site) = result? else { continue };
            let q1 = site.alt1 as f64 / (site.calls1 * 2) as f64;
            let denominator = site.q2 * (1.0 - site.q2);
            omega_sum += (q1 - site.q2).powi(2) / denominator;
            omega_sites += 1;
            usable_sites += 1;
        }
        if progress_callback.is_some()
            && (block_start + block_len >= next_progress || block_start + block_len == scan_sites)
        {
            let done = block_start + block_len;
            emit_progress_callback(progress_callback, 0, done, scan_sites)?;
            while next_progress <= done {
                next_progress = next_progress.saturating_add(progress_step);
            }
        }
        block_start += block_len;
    }
    if bim_reader.next_site()?.is_some() {
        return Err(format!(
            "BED/BIM variant count mismatch during XP-CLR omega scan: BED={}",
            scan_sites
        ));
    }
    if usable_sites == 0 || omega_sites == 0 {
        return Err("no usable biallelic sites remain after XP-CLR filtering".to_string());
    }
    let omega = omega_sum / omega_sites as f64;

    if progress_callback.is_some() {
        emit_progress_callback(progress_callback, 0, scan_sites, scan_sites)?;
        emit_progress_callback(progress_callback, 1, 0, scan_sites)?;
        next_progress = progress_step;
    }

    let output_path = Path::new(output);
    if let Some(parent) = output_path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let file = File::create(output).map_err(|e| format!("{output}: {e}"))?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "{XPCLR_HEADER}").map_err(|e| format!("{output}: {e}"))?;
    let (total_windows, valid) = {
        // Keep only a bounded number of window payloads in the Rayon task
        // graph.  The producer can continue decoding while workers calculate;
        // once the bound is reached it drains one completed result, which is
        // both backpressure and dynamic scheduling for uneven windows.
        let max_in_flight = effective_threads.saturating_mul(4).max(32);
        let mut pipeline = XpclrWindowPipeline::new(max_in_flight);
        let mut valid = 0usize;
        let mut total_windows = 0usize;
        let mut bim_reader =
            BimReader::open_at(&bed_prefix, scan_plan.bim_offset, Some(scan_sites))?;
        let mut bed = input
            .bed
            .try_clone()
            .map_err(|e| format!("{prefix}.bed: {e}"))?;
        bed.seek(SeekFrom::Start(scan_plan.bed_offset))
            .map_err(|e| format!("{prefix}.bed: {e}"))?;
        let mut current_chrom = None::<String>;
        let mut current_chrom_idx = 0usize;
        let mut seen_chromosome = false;
        let mut completed_chroms = HashSet::<String>::new();
        let mut accumulator = None::<XpclrWindowAccumulator>;
        let mut max_pos = 0i64;
        let mut last_pos = None::<i64>;
        let mut spool = None::<XpclrRawSpool>;

        let mut block_start = 0usize;
        while block_start < scan_sites {
            let block_len = (scan_sites - block_start).min(XPCLR_BLOCK_SITES);
            let mut block_payload = vec![0u8; block_len * input.bytes_per_snp];
            bed.read_exact(&mut block_payload)
                .map_err(|e| format!("{prefix}.bed: {e}"))?;
            let block_sites = (0..block_len)
                .map(|_| {
                    bim_reader.next_site()?.ok_or_else(|| {
                        "BIM ended before all BED variants during XP-CLR scan".to_string()
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            let block_results = scan_block_sites(
                &block_payload,
                input.bytes_per_snp,
                input.n_samples,
                &block_sites,
                &pop1_mask,
                &pop2_mask,
                &pop2,
                map.as_ref(),
                rrate,
                Some(omega),
                true,
                Some(&pool),
            );
            for (bim, result) in block_sites.into_iter().zip(block_results) {
                let Some(site) = result? else { continue };
                let site_chrom = bim.chrom.clone();
                if current_chrom.as_deref() != Some(site_chrom.as_str()) {
                    if let Some(chrom) = current_chrom.take() {
                        let mut chrom_spool = spool.take().ok_or_else(|| {
                            "internal XP-CLR error: missing chromosome spool".to_string()
                        })?;
                        let (count, chrom_valid) = finish_xpclr_chrom_pipeline(
                            &mut pipeline,
                            &pool,
                            &mut accumulator,
                            max_pos,
                            window,
                            step,
                            maxsnps,
                            sample_seed,
                            current_chrom_idx,
                            &mut chrom_spool,
                            &mut writer,
                            omega,
                            ld_cutoff,
                            minsnps,
                            &chrom,
                            output,
                        )?;
                        total_windows += count;
                        valid += chrom_valid;
                        completed_chroms.insert(chrom);
                    }
                    if completed_chroms.contains(&site_chrom) {
                        return Err(format!(
                            "BIM chromosome {} reappears after a later chromosome; XP-CLR requires chromosome-sorted BIM",
                            site_chrom
                        ));
                    }
                    current_chrom = Some(site_chrom.clone());
                    if seen_chromosome {
                        current_chrom_idx += 1;
                    }
                    seen_chromosome = true;
                    accumulator = Some(XpclrWindowAccumulator::new());
                    spool = Some(XpclrRawSpool::new(output_path, current_chrom_idx)?);
                    max_pos = 0;
                    last_pos = None;
                }
                if last_pos.is_some_and(|previous| site.pos < previous) {
                    return Err(format!(
                        "BIM positions are not sorted within chromosome {}",
                        site_chrom
                    ));
                }
                last_pos = Some(site.pos);
                max_pos = max_pos.max(site.pos);
                let emitted = accumulator.as_mut().unwrap().push(
                    site,
                    window,
                    step,
                    maxsnps,
                    sample_seed,
                    current_chrom_idx,
                );
                let chrom = current_chrom.as_deref().unwrap().to_string();
                for work in emitted {
                    pipeline.submit(
                        &pool,
                        work,
                        omega,
                        ld_cutoff,
                        minsnps,
                        &chrom,
                        spool.as_mut().ok_or_else(|| {
                            "internal XP-CLR error: missing chromosome spool".to_string()
                        })?,
                    )?;
                }
            }
            if progress_callback.is_some()
                && (block_start + block_len >= next_progress
                    || block_start + block_len == scan_sites)
            {
                let done = block_start + block_len;
                let progress_done = if done == scan_sites {
                    done.saturating_sub(1)
                } else {
                    done
                };
                emit_progress_callback(progress_callback, 1, progress_done, scan_sites)?;
                while next_progress <= done {
                    next_progress = next_progress.saturating_add(progress_step);
                }
            }
            block_start += block_len;
        }
        if bim_reader.next_site()?.is_some() {
            return Err(format!(
                "BED/BIM variant count mismatch during XP-CLR scan: BED={}",
                scan_sites
            ));
        }
        if let Some(chrom) = current_chrom.take() {
            let mut chrom_spool = spool
                .take()
                .ok_or_else(|| "internal XP-CLR error: missing chromosome spool".to_string())?;
            let (count, chrom_valid) = finish_xpclr_chrom_pipeline(
                &mut pipeline,
                &pool,
                &mut accumulator,
                max_pos,
                window,
                step,
                maxsnps,
                sample_seed,
                current_chrom_idx,
                &mut chrom_spool,
                &mut writer,
                omega,
                ld_cutoff,
                minsnps,
                &chrom,
                output,
            )?;
            total_windows += count;
            valid += chrom_valid;
        }
        (total_windows, valid)
    };
    writer.flush().map_err(|e| format!("{output}: {e}"))?;
    if progress_callback.is_some() {
        emit_progress_callback(progress_callback, 1, scan_sites, scan_sites)?;
    }
    Ok((total_windows, valid))
}

#[pyfunction]
#[pyo3(signature = (prefix, pop1, pop2, output, map_path=None, chrom=None, window=50000, step=50000, maxsnps=600, minsnps=10, ld_cutoff=0.95, rrate=1e-8, seed=42, threads=0, progress_callback=None, progress_every=0))]
pub fn xpclr_bed_to_tsv(
    py: Python<'_>,
    prefix: String,
    pop1: String,
    pop2: String,
    output: String,
    map_path: Option<String>,
    chrom: Option<String>,
    window: i64,
    step: i64,
    maxsnps: usize,
    minsnps: usize,
    ld_cutoff: f64,
    rrate: f64,
    seed: u64,
    threads: usize,
    progress_callback: Option<Py<PyAny>>,
    progress_every: usize,
) -> PyResult<(usize, usize)> {
    py.detach(move || {
        run_xpclr(
            &prefix,
            &pop1,
            &pop2,
            map_path.as_deref(),
            chrom.as_deref(),
            &output,
            window,
            step,
            maxsnps,
            minsnps,
            ld_cutoff,
            rrate,
            seed,
            threads,
            progress_callback.as_ref(),
            progress_every,
        )
    })
    .map_err(map_err_string_to_py)
}

#[cfg(test)]
mod tests {
    use super::{
        calculate_xpclr, chen_likelihood, determine_c, determine_omega, determine_weights_flat,
        normalize_scores, select_sites_seeded, window_starts,
    };

    #[test]
    fn xpclr_reference_defaults_and_window_contract_are_stable() {
        assert_eq!(determine_c(0.0, 0.0), 1.0);
        assert_eq!(determine_c(0.01, 0.001), 1.0);
        assert!((determine_omega(&[0.25, 0.5], &[0.2, 0.4]) - 0.028645833333333332).abs() < 1e-12);
        assert_eq!(window_starts(1, 101, 50, 50), vec![(1, 50), (51, 100)]);
        let value = chen_likelihood((2, 8, 1.0, 0.25, 0.03125 * 0.25));
        assert!(value.is_finite());
    }

    #[test]
    fn xpclr_normalization_ignores_nan_windows() {
        let values = normalize_scores(&[1.0, 2.0, f64::NAN, 3.0]);
        assert!(values[2].is_nan());
        assert!((values[0] + 1.224744871391589).abs() < 1e-12);
        assert!(values[1].abs() < 1e-12);
        assert!((values[3] - 1.224744871391589).abs() < 1e-12);
    }

    #[test]
    fn xpclr_uses_a_precomputed_null_likelihood() {
        let rows = [(2, 8, 0.0, 0.25, 1.0)];
        let supplied_null = 12.5;
        let (_, null_likelihood, _) = calculate_xpclr(&rows, 0.03125, supplied_null);
        assert_eq!(null_likelihood, supplied_null);
    }

    #[test]
    fn xpclr_flat_ld_weights_match_the_pairwise_contract() {
        let dosages = [0, 1, 2, 1, 0, 1, 2, 1, 2, 2, 2, 2];
        let weights = determine_weights_flat(&dosages, 3, 4, 0.95);
        assert_eq!(weights, vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
    }

    #[test]
    fn xpclr_maxsnps_sampling_is_seeded_sorted_and_without_replacement() {
        let sites = (0usize..100).collect::<Vec<_>>();
        let first = select_sites_seeded(&sites, 12, 42);
        let second = select_sites_seeded(&sites, 12, 42);
        let other = select_sites_seeded(&sites, 12, 43);
        assert_eq!(first, second);
        assert_ne!(first, other);
        assert_eq!(first.len(), 12);
        assert!(first.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(
            first
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            12
        );
    }

    #[test]
    fn streaming_window_accumulator_keeps_empty_and_overlapping_windows() {
        let mut accumulator = super::XpclrWindowAccumulator::new();
        let site = |pos| super::SiteCounts {
            pos,
            alt1: 2,
            calls1: 4,
            q2: 0.25,
            genetic_distance: pos as f64,
            null_likelihood: -1.0,
            dosages: vec![0, 1],
        };
        let mut emitted = Vec::new();
        emitted.extend(accumulator.push(site(10), 20, 10, 600, 42, 0));
        emitted.extend(accumulator.push(site(25), 20, 10, 600, 42, 0));
        emitted.extend(accumulator.finish(25, 20, 10, 600, 42, 0));
        assert_eq!(
            emitted
                .iter()
                .map(|work| (work.start, work.stop, work.n_snps_avail))
                .collect::<Vec<_>>(),
            vec![(1, 20, 1), (11, 30, 1), (21, 40, 1)],
        );
    }
}
