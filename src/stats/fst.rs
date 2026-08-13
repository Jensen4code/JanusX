//! Bitwise population differentiation statistics for PLINK SNP-major BED files.
//!
//! The implementation deliberately keeps the genotype matrix packed.  Each BED
//! row is transposed into three sample-bit planes and all population counts are
//! obtained with the SIMD-aware bitwise popcount kernels.

use crate::bitwise::and_popcount;
use crate::gfcore::SiteInfo;
use crate::stats_common::{get_cached_pool, map_err_string_to_py};
use memmap2::Mmap;
use numpy::PyArray2;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::BoundObject;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};

const BED_HEADER_LEN: usize = 3;
const FST_BLOCK_SITES: usize = 4096;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GroupStats {
    /// Number of diploid samples with a non-missing call.
    pub(crate) n: usize,
    /// Number of alternate alleles across those samples.
    pub(crate) alt_alleles: usize,
    /// Number of heterozygous diploid calls.
    pub(crate) het: usize,
}

impl GroupStats {
    #[cfg(test)]
    fn from_counts(n: usize, alt_alleles: usize, het: usize) -> Self {
        Self {
            n,
            alt_alleles,
            het,
        }
    }
}

#[derive(Clone, Debug)]
struct GroupMask {
    name: String,
    words: Vec<u64>,
    n_samples: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct BitPlanes {
    ref_bits: u64,
    het_bits: u64,
    alt_bits: u64,
}

#[derive(Clone, Debug, Default)]
struct BitPlaneWords {
    ref_bits: Vec<u64>,
    het_bits: Vec<u64>,
    alt_bits: Vec<u64>,
}

struct FstScratch {
    planes: BitPlaneWords,
    stats: Vec<GroupStats>,
}

impl FstScratch {
    fn new(n_samples: usize, n_groups: usize) -> Self {
        let n_words = words_for_samples(n_samples);
        Self {
            planes: BitPlaneWords {
                ref_bits: vec![0u64; n_words],
                het_bits: vec![0u64; n_words],
                alt_bits: vec![0u64; n_words],
            },
            stats: vec![GroupStats::default(); n_groups],
        }
    }
}

#[derive(Clone, Debug)]
struct FstOutput {
    sites: Vec<SiteInfo>,
    values: Vec<f64>,
    nmiss_left: Vec<usize>,
    nmiss_right: Vec<usize>,
    pair_names: Vec<(String, String)>,
}

struct FstInput {
    mmap: Mmap,
    n_samples: usize,
    n_sites: usize,
    bytes_per_snp: usize,
    masks: Vec<GroupMask>,
    pair_names: Vec<(String, String)>,
    method: FstMethod,
}

struct FstBimReader {
    path: String,
    reader: BufReader<File>,
    line_no: usize,
}

impl FstBimReader {
    fn open(prefix: &str) -> Result<Self, String> {
        let path = format!("{prefix}.bim");
        let file = File::open(&path).map_err(|e| format!("{path}: {e}"))?;
        Ok(Self {
            path,
            reader: BufReader::new(file),
            line_no: 0,
        })
    }

    fn next_site(&mut self) -> Result<Option<SiteInfo>, String> {
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
        if columns.len() < 6 {
            return Err(format!(
                "malformed BIM line at {}:{}; expected 6 columns: {raw}",
                self.path, self.line_no
            ));
        }
        let pos = columns[3].parse::<i32>().map_err(|e| {
            format!(
                "invalid BIM position at {}:{}: {e}: {raw}",
                self.path, self.line_no
            )
        })?;
        Ok(Some(SiteInfo {
            chrom: columns[0].to_string(),
            pos,
            snp: columns[1].to_string(),
            ref_allele: columns[4].to_string(),
            alt_allele: columns[5].to_string(),
        }))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FstMethod {
    Wc,
    Hudson,
}

impl FstMethod {
    fn parse(method: &str) -> Result<Self, String> {
        match method.trim().to_ascii_lowercase().as_str() {
            "wc" | "weir-cockerham" | "weir_cockerham" => Ok(Self::Wc),
            "hudson" => Ok(Self::Hudson),
            other => Err(format!(
                "unsupported FST method {other:?}; expected 'wc' or 'hudson'"
            )),
        }
    }
}

#[inline]
fn words_for_samples(n_samples: usize) -> usize {
    n_samples.div_ceil(64)
}

#[inline]
fn set_bit(words: &mut [u64], sample_idx: usize) {
    words[sample_idx >> 6] |= 1u64 << (sample_idx & 63);
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

// One BED byte stores four 2-bit calls.  These tables compact the selected
// genotype code into four one-bit sample lanes, avoiding scalar genotype
// decoding in the hot path.
const REF_BYTE_PLANE_LUT: [u8; 256] = build_byte_plane_lut(0b00);
const HET_BYTE_PLANE_LUT: [u8; 256] = build_byte_plane_lut(0b10);
const ALT_BYTE_PLANE_LUT: [u8; 256] = build_byte_plane_lut(0b11);

/// Transpose up to 64 BED genotype calls into sample-bit planes.
///
/// PLINK uses 00/10/11 for 0/1/2 and 01 for missing. Padding calls in the
/// last BED byte never enter a count. A BED byte is expanded with a small
/// compile-time lookup table rather than decoding each genotype to a numeric
/// dosage.
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

fn build_group_masks(
    group_by_sample: &[Option<String>],
    n_samples: usize,
) -> Result<Vec<GroupMask>, String> {
    if group_by_sample.len() != n_samples {
        return Err(format!(
            "group assignment length mismatch: got {}, expected {n_samples}",
            group_by_sample.len()
        ));
    }
    let mut group_names = Vec::<String>::new();
    let mut group_lookup = HashMap::<String, usize>::new();
    let mut sample_group_idx = vec![None; n_samples];
    for (sample_idx, group) in group_by_sample.iter().enumerate() {
        let Some(group) = group else { continue };
        let group = group.trim();
        if group.is_empty() {
            continue;
        }
        let group_idx = if let Some(&idx) = group_lookup.get(group) {
            idx
        } else {
            let idx = group_names.len();
            group_names.push(group.to_string());
            group_lookup.insert(group.to_string(), idx);
            idx
        };
        sample_group_idx[sample_idx] = Some(group_idx);
    }
    if group_names.len() < 2 {
        return Err("FST requires at least two non-empty groups".to_string());
    }

    let words = words_for_samples(n_samples);
    let mut masks = group_names
        .into_iter()
        .map(|name| GroupMask {
            name,
            words: vec![0u64; words],
            n_samples: 0,
        })
        .collect::<Vec<_>>();
    for (sample_idx, group_idx) in sample_group_idx.into_iter().enumerate() {
        if let Some(group_idx) = group_idx {
            set_bit(&mut masks[group_idx].words, sample_idx);
            masks[group_idx].n_samples += 1;
        }
    }
    Ok(masks)
}

fn read_fam_keys(fam_path: &str) -> Result<Vec<(String, String)>, String> {
    let file = File::open(fam_path).map_err(|e| format!("{fam_path}: {e}"))?;
    let mut fam_ids = Vec::new();
    let mut seen = HashSet::<(String, String)>::new();
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("{fam_path}: {e}"))?;
        if line.trim().is_empty() {
            continue;
        }
        let mut cols = line.split_whitespace();
        let fid = cols
            .next()
            .ok_or_else(|| format!("malformed FAM line at {fam_path}:{}: {line}", line_no + 1))?;
        let iid = cols
            .next()
            .ok_or_else(|| format!("malformed FAM line at {fam_path}:{}: {line}", line_no + 1))?;
        let key = (fid.to_string(), iid.to_string());
        if !seen.insert(key.clone()) {
            return Err(format!(
                "duplicate FID/IID in FAM at {fam_path}:{}: {} {}",
                line_no + 1,
                key.0,
                key.1
            ));
        }
        fam_ids.push(key);
    }
    if fam_ids.is_empty() {
        return Err(format!("FAM contains no samples: {fam_path}"));
    }
    Ok(fam_ids)
}

fn read_within_groups(
    within_path: &str,
    fam_ids: &[(String, String)],
) -> Result<Vec<Option<String>>, String> {
    let file = File::open(within_path).map_err(|e| format!("{within_path}: {e}"))?;
    let reader = BufReader::new(file);

    let mut by_key = HashMap::<(String, String), String>::new();
    let mut seen_keys = HashSet::<(String, String)>::new();
    for (line_no, line) in reader.lines().enumerate() {
        let line = line.map_err(|e| format!("{within_path}:{}: {e}", line_no + 1))?;
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        let cols = line.split_whitespace().collect::<Vec<_>>();
        if cols.len() < 3 {
            return Err(format!(
                "malformed --within line at {within_path}:{}; expected FID IID GROUP",
                line_no + 1
            ));
        }
        let key = (cols[0].to_string(), cols[1].to_string());
        if !seen_keys.insert(key.clone()) {
            return Err(format!(
                "duplicate sample assignment in {within_path}:{} for {} {}",
                line_no + 1,
                cols[0],
                cols[1]
            ));
        }
        let group = cols[2].to_string();
        by_key.insert(key, group);
    }

    let mut out = Vec::with_capacity(fam_ids.len());
    let mut matched = HashSet::<(String, String)>::new();
    for key in fam_ids {
        if let Some(group) = by_key.get(&key) {
            matched.insert(key.clone());
            out.push(Some(group.clone()));
        } else {
            out.push(None);
        }
    }
    if let Some((key, _)) = by_key.iter().find(|(key, _)| !matched.contains(key)) {
        return Err(format!(
            "--within contains sample {} {} not present in FAM; FID/IID must match exactly",
            key.0, key.1
        ));
    }
    Ok(out)
}

fn open_bed(prefix: &str) -> Result<(Mmap, usize, usize, usize), String> {
    let fam_path = format!("{prefix}.fam");
    let n_samples = read_fam_keys(&fam_path)?.len();
    let path = format!("{prefix}.bed");
    let file = File::open(&path).map_err(|e| format!("{path}: {e}"))?;
    let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("{path}: {e}"))?;
    if mmap.len() < BED_HEADER_LEN || mmap[0] != 0x6c || mmap[1] != 0x1b || mmap[2] != 0x01 {
        return Err(format!(
            "{path}: expected SNP-major PLINK BED header 0x6c 0x1b 0x01"
        ));
    }
    let bytes_per_snp = (n_samples + 3) / 4;
    let payload = mmap.len() - BED_HEADER_LEN;
    if bytes_per_snp == 0 || payload == 0 || payload % bytes_per_snp != 0 {
        return Err(format!(
            "{path}: invalid payload length {payload} for {bytes_per_snp} bytes/SNP"
        ));
    }
    Ok((mmap, n_samples, payload / bytes_per_snp, bytes_per_snp))
}

#[inline]
fn group_stats_from_planes_bitwise(planes: &BitPlaneWords, mask: &GroupMask) -> GroupStats {
    let ref_count = and_popcount(mask.words.as_slice(), &planes.ref_bits) as usize;
    let het_count = and_popcount(mask.words.as_slice(), &planes.het_bits) as usize;
    let alt_count = and_popcount(mask.words.as_slice(), &planes.alt_bits) as usize;
    GroupStats {
        n: ref_count + het_count + alt_count,
        alt_alleles: het_count + 2 * alt_count,
        het: het_count,
    }
}

fn transpose_row_into(row: &[u8], n_samples: usize, out: &mut BitPlaneWords) {
    for (word_idx, sample_start) in (0..n_samples).step_by(64).enumerate() {
        let p = transpose_bed_word(row, sample_start, n_samples);
        out.ref_bits[word_idx] = p.ref_bits;
        out.het_bits[word_idx] = p.het_bits;
        out.alt_bits[word_idx] = p.alt_bits;
    }
}

fn stats_for_row_into(row: &[u8], n_samples: usize, masks: &[GroupMask], scratch: &mut FstScratch) {
    transpose_row_into(row, n_samples, &mut scratch.planes);
    for (stats, mask) in scratch.stats.iter_mut().zip(masks) {
        *stats = group_stats_from_planes_bitwise(&scratch.planes, mask);
    }
}

#[cfg(test)]
fn stats_for_row(row: &[u8], n_samples: usize, masks: &[GroupMask]) -> Vec<GroupStats> {
    let mut scratch = FstScratch::new(n_samples, masks.len());
    stats_for_row_into(row, n_samples, masks, &mut scratch);
    scratch.stats
}

#[inline]
fn allele_frequency(stats: &GroupStats) -> Option<f64> {
    (stats.n > 0).then(|| stats.alt_alleles as f64 / (2.0 * stats.n as f64))
}

/// Weir--Cockerham's per-site theta estimator.
pub(crate) fn wc_fst(groups: &[GroupStats]) -> Option<f64> {
    if groups.len() < 2 || groups.iter().any(|g| g.n < 2) {
        return None;
    }
    let r = groups.len() as f64;
    let total_n: usize = groups.iter().map(|g| g.n).sum();
    let n_bar = total_n as f64 / r;
    if n_bar <= 1.0 {
        return None;
    }
    let p_bar = groups
        .iter()
        .filter_map(|g| allele_frequency(g).map(|p| g.n as f64 * p))
        .sum::<f64>()
        / (r * n_bar);
    let s_squared = groups
        .iter()
        .filter_map(|g| allele_frequency(g).map(|p| g.n as f64 * (p - p_bar).powi(2)))
        .sum::<f64>()
        / ((r - 1.0) * n_bar);
    let h_bar = groups.iter().map(|g| g.het as f64).sum::<f64>() / (r * n_bar);
    let n_c = (r * n_bar - groups.iter().map(|g| (g.n as f64).powi(2)).sum::<f64>() / (r * n_bar))
        / (r - 1.0);
    if n_c <= 0.0 {
        return None;
    }
    let q = p_bar * (1.0 - p_bar) - ((r - 1.0) / r) * s_squared;
    let a = (n_bar / n_c) * (s_squared - (q - h_bar / 4.0) / (n_bar - 1.0));
    let b = (n_bar / (n_bar - 1.0)) * (q - ((2.0 * n_bar - 1.0) / (4.0 * n_bar)) * h_bar);
    let c = h_bar / 2.0;
    let denominator = a + b + c;
    if !denominator.is_finite() || denominator == 0.0 {
        None
    } else {
        Some(a / denominator)
    }
}

/// Hudson's unbiased two-population per-site estimator.
pub(crate) fn hudson_fst(lhs: &GroupStats, rhs: &GroupStats) -> Option<f64> {
    if lhs.n < 2 || rhs.n < 2 {
        return None;
    }
    let p_lhs = allele_frequency(lhs)?;
    let p_rhs = allele_frequency(rhs)?;
    let between = p_lhs * (1.0 - p_rhs) + p_rhs * (1.0 - p_lhs);
    if !between.is_finite() || between <= 0.0 {
        return None;
    }
    let within_lhs = 2.0 * (lhs.alt_alleles as f64) * ((2 * lhs.n - lhs.alt_alleles) as f64)
        / ((2 * lhs.n) as f64 * (2 * lhs.n - 1) as f64);
    let within_rhs = 2.0 * (rhs.alt_alleles as f64) * ((2 * rhs.n - rhs.alt_alleles) as f64)
        / ((2 * rhs.n) as f64 * (2 * rhs.n - 1) as f64);
    let numerator = between - 0.5 * (within_lhs + within_rhs);
    Some(numerator / between)
}

fn normalize_plink_prefix(prefix: &str) -> String {
    let trimmed = prefix.trim();
    let lower = trimmed.to_ascii_lowercase();
    if lower.ends_with(".bed") || lower.ends_with(".bim") || lower.ends_with(".fam") {
        trimmed[..trimmed.len() - 4].to_string()
    } else {
        trimmed.to_string()
    }
}

fn pair_names(masks: &[GroupMask], method: FstMethod) -> Vec<(String, String)> {
    match method {
        FstMethod::Wc => vec![("ALL".to_string(), "ALL".to_string())],
        FstMethod::Hudson => {
            let mut out = Vec::new();
            for lhs in 0..masks.len() {
                for rhs in (lhs + 1)..masks.len() {
                    out.push((masks[lhs].name.clone(), masks[rhs].name.clone()));
                }
            }
            out
        }
    }
}

fn open_fst_input(prefix: &str, within_path: &str, method_name: &str) -> Result<FstInput, String> {
    let method = FstMethod::parse(method_name)?;
    let bed_prefix = normalize_plink_prefix(prefix);
    let fam_ids = read_fam_keys(&format!("{bed_prefix}.fam"))?;
    let n_samples = fam_ids.len();
    let groups_by_sample = read_within_groups(within_path, &fam_ids)?;
    let masks = build_group_masks(&groups_by_sample, n_samples)?;
    for mask in &masks {
        if mask.n_samples < 2 {
            return Err(format!(
                "group {:?} has only {} assigned samples; at least 2 are required",
                mask.name, mask.n_samples
            ));
        }
    }

    let (mmap, bed_n_samples, n_sites, bytes_per_snp) = open_bed(&bed_prefix)?;
    if bed_n_samples != n_samples {
        return Err(format!(
            "BED/FAM sample count mismatch: BED/FAM={bed_n_samples}, FAM metadata={n_samples}"
        ));
    }
    let mut bim_reader = FstBimReader::open(&bed_prefix)?;
    let mut bim_sites = 0usize;
    while bim_reader.next_site()?.is_some() {
        bim_sites += 1;
    }
    if bim_sites != n_sites {
        return Err(format!(
            "BED/BIM variant count mismatch: BED={n_sites}, BIM={bim_sites}"
        ));
    }
    let pair_names = pair_names(&masks, method);
    Ok(FstInput {
        mmap,
        n_samples,
        n_sites,
        bytes_per_snp,
        masks,
        pair_names,
        method,
    })
}

fn compute_fst_block(
    input: &FstInput,
    start: usize,
    end: usize,
    threads: usize,
) -> Result<FstOutput, String> {
    if start > end || end > input.n_sites {
        return Err(format!(
            "invalid FST block range [{start}, {end}) for {} sites",
            input.n_sites
        ));
    }
    let n_sites = end - start;
    let pair_names = input.pair_names.clone();
    let pair_count = pair_names.len();
    let mut values = vec![f64::NAN; n_sites * pair_count];
    let mut nmiss_left = vec![0usize; n_sites * pair_count];
    let mut nmiss_right = vec![0usize; n_sites * pair_count];
    let payload = &input.mmap[BED_HEADER_LEN..];

    let mut run = || {
        values
            .par_chunks_mut(pair_count)
            .zip(nmiss_left.par_chunks_mut(pair_count))
            .zip(nmiss_right.par_chunks_mut(pair_count))
            .enumerate()
            .for_each_init(
                || FstScratch::new(input.n_samples, input.masks.len()),
                |scratch, (site_idx, ((value_row, left_row), right_row))| {
                    let source_site_idx = start + site_idx;
                    let row = &payload[source_site_idx * input.bytes_per_snp
                        ..(source_site_idx + 1) * input.bytes_per_snp];
                    stats_for_row_into(row, input.n_samples, &input.masks, scratch);
                    let stats = &scratch.stats;
                    match input.method {
                        FstMethod::Wc => {
                            left_row[0] = stats.iter().map(|group| group.n).sum();
                            value_row[0] = wc_fst(&stats).unwrap_or(f64::NAN);
                        }
                        FstMethod::Hudson => {
                            let mut pair_idx = 0usize;
                            for lhs in 0..stats.len() {
                                for rhs in (lhs + 1)..stats.len() {
                                    left_row[pair_idx] = stats[lhs].n;
                                    right_row[pair_idx] = stats[rhs].n;
                                    value_row[pair_idx] =
                                        hudson_fst(&stats[lhs], &stats[rhs]).unwrap_or(f64::NAN);
                                    pair_idx += 1;
                                }
                            }
                        }
                    }
                },
            );
    };
    let pool = get_cached_pool(threads).map_err(|e| e.to_string())?;
    if let Some(pool) = pool {
        pool.install(&mut run);
    } else {
        run();
    }

    // Keep the row-major [site, population-pair] layout.  This is both the
    // natural output order and the shape exposed to Python.  The mmap remains
    // borrowed by `payload` until this point, so all worker reads are complete.
    Ok(FstOutput {
        sites: Vec::new(),
        values,
        nmiss_left,
        nmiss_right,
        pair_names,
    })
}

fn compute_fst_core(
    prefix: &str,
    within_path: &str,
    method_name: &str,
    threads: usize,
) -> Result<FstOutput, String> {
    let input = open_fst_input(prefix, within_path, method_name)?;
    let mut output = compute_fst_block(&input, 0, input.n_sites, threads)?;
    let mut bim_reader = FstBimReader::open(&normalize_plink_prefix(prefix))?;
    output.sites = read_fst_sites(&mut bim_reader, input.n_sites)?;
    if bim_reader.next_site()?.is_some() {
        return Err("BIM contains more rows than BED while loading FST".to_string());
    }
    Ok(output)
}

fn write_fst_header<W: Write>(writer: &mut W, method: FstMethod, path: &str) -> Result<(), String> {
    match method {
        FstMethod::Wc => {
            writeln!(writer, "CHR\tSNP\tPOS\tNMISS\tFST").map_err(|e| format!("{path}: {e}"))?;
        }
        FstMethod::Hudson => {
            writeln!(writer, "POP1\tPOP2\tCHR\tSNP\tPOS\tNMISS1\tNMISS2\tFST")
                .map_err(|e| format!("{path}: {e}"))?;
        }
    }
    Ok(())
}

fn write_fst_rows<W: Write>(
    writer: &mut W,
    output: &FstOutput,
    method: FstMethod,
    path: &str,
) -> Result<(), String> {
    match method {
        FstMethod::Wc => {
            for (site_idx, site) in output.sites.iter().enumerate() {
                writeln!(
                    writer,
                    "{}\t{}\t{}\t{}\t{}",
                    site.chrom,
                    site.snp,
                    site.pos,
                    output.nmiss_left[site_idx],
                    output.values[site_idx]
                )
                .map_err(|e| format!("{path}: {e}"))?;
            }
        }
        FstMethod::Hudson => {
            let pair_count = output.pair_names.len();
            for (site_idx, site) in output.sites.iter().enumerate() {
                for (pair_idx, (pop1, pop2)) in output.pair_names.iter().enumerate() {
                    let offset = site_idx * pair_count + pair_idx;
                    writeln!(
                        writer,
                        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                        pop1,
                        pop2,
                        site.chrom,
                        site.snp,
                        site.pos,
                        output.nmiss_left[offset],
                        output.nmiss_right[offset],
                        output.values[offset]
                    )
                    .map_err(|e| format!("{path}: {e}"))?;
                }
            }
        }
    }
    Ok(())
}

fn read_fst_sites(reader: &mut FstBimReader, count: usize) -> Result<Vec<SiteInfo>, String> {
    let mut sites = Vec::with_capacity(count);
    for _ in 0..count {
        sites.push(
            reader
                .next_site()?
                .ok_or_else(|| "BIM ended while loading FST site metadata".to_string())?,
        );
    }
    Ok(sites)
}

fn write_fst_tsv_streaming(
    prefix: &str,
    within_path: &str,
    output_path: &str,
    method_name: &str,
    threads: usize,
) -> Result<(usize, usize), String> {
    let input = open_fst_input(prefix, within_path, method_name)?;
    let mut bim_reader = FstBimReader::open(&normalize_plink_prefix(prefix))?;
    let method = input.method;
    let file = File::create(output_path).map_err(|e| format!("{output_path}: {e}"))?;
    let mut writer = BufWriter::new(file);
    write_fst_header(&mut writer, method, output_path)?;

    let mut start = 0usize;
    while start < input.n_sites {
        let end = (start + FST_BLOCK_SITES).min(input.n_sites);
        let mut block = compute_fst_block(&input, start, end, threads)?;
        block.sites = read_fst_sites(&mut bim_reader, end - start)?;
        write_fst_rows(&mut writer, &block, method, output_path)?;
        start = end;
    }
    if bim_reader.next_site()?.is_some() {
        return Err(format!(
            "BED/BIM variant count mismatch while streaming: BED={}",
            input.n_sites
        ));
    }
    writer.flush().map_err(|e| format!("{output_path}: {e}"))?;
    Ok((input.n_sites, input.pair_names.len()))
}

#[pyfunction]
#[pyo3(signature = (prefix, within, method="wc", threads=0))]
pub fn fst_bed<'py>(
    py: Python<'py>,
    prefix: String,
    within: String,
    method: &str,
    threads: usize,
) -> PyResult<(
    Vec<String>,
    Vec<i64>,
    Vec<String>,
    Bound<'py, numpy::PyArray2<f64>>,
    Bound<'py, numpy::PyArray2<i64>>,
    Bound<'py, numpy::PyArray2<i64>>,
    Vec<String>,
    Vec<String>,
)> {
    let method_owned = method.to_string();
    let output = py
        .detach(move || compute_fst_core(&prefix, &within, &method_owned, threads))
        .map_err(map_err_string_to_py)?;
    let n_sites = output.sites.len();
    let n_pairs = output.pair_names.len();
    let chrom = output
        .sites
        .iter()
        .map(|site| site.chrom.clone())
        .collect::<Vec<_>>();
    let pos = output
        .sites
        .iter()
        .map(|site| i64::from(site.pos))
        .collect::<Vec<_>>();
    let snp = output
        .sites
        .iter()
        .map(|site| site.snp.clone())
        .collect::<Vec<_>>();
    let fst = PyArray2::from_owned_array(
        py,
        numpy::ndarray::Array2::from_shape_vec((n_sites, n_pairs), output.values)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    let nmiss_left = PyArray2::from_owned_array(
        py,
        numpy::ndarray::Array2::from_shape_vec(
            (n_sites, n_pairs),
            output
                .nmiss_left
                .into_iter()
                .map(|value| value as i64)
                .collect(),
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    let nmiss_right = PyArray2::from_owned_array(
        py,
        numpy::ndarray::Array2::from_shape_vec(
            (n_sites, n_pairs),
            output
                .nmiss_right
                .into_iter()
                .map(|value| value as i64)
                .collect(),
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
    )
    .into_bound();
    let pop1 = output
        .pair_names
        .iter()
        .map(|(lhs, _)| lhs.clone())
        .collect::<Vec<_>>();
    let pop2 = output
        .pair_names
        .iter()
        .map(|(_, rhs)| rhs.clone())
        .collect::<Vec<_>>();
    Ok((chrom, pos, snp, fst, nmiss_left, nmiss_right, pop1, pop2))
}

#[pyfunction]
#[pyo3(signature = (prefix, within, output, method="wc", threads=0))]
pub fn fst_bed_to_tsv(
    py: Python<'_>,
    prefix: String,
    within: String,
    output: String,
    method: &str,
    threads: usize,
) -> PyResult<(usize, usize)> {
    let method_owned = method.to_string();
    py.detach(move || write_fst_tsv_streaming(&prefix, &within, &output, &method_owned, threads))
        .map_err(map_err_string_to_py)
}

#[cfg(test)]
mod tests {
    use super::{
        build_group_masks, hudson_fst, read_within_groups, stats_for_row, wc_fst, GroupStats,
    };

    fn close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "actual={actual:.15e}, expected={expected:.15e}"
        );
    }

    #[test]
    fn wc_matches_two_population_reference_case() {
        let groups = [
            GroupStats::from_counts(4, 0, 0),
            GroupStats::from_counts(4, 8, 0),
        ];
        close(wc_fst(&groups).unwrap(), 1.0);
    }

    #[test]
    fn wc_supports_unequal_multi_population_sample_sizes() {
        let groups = [
            GroupStats::from_counts(4, 0, 0),
            GroupStats::from_counts(4, 4, 0),
            GroupStats::from_counts(4, 8, 0),
        ];
        let value = wc_fst(&groups).unwrap();
        assert!(value.is_finite());
        close(value, 2.0 / 3.0);
    }

    #[test]
    fn wc_matches_plink_with_heterozygotes() {
        let groups = [
            GroupStats::from_counts(2, 4, 0),
            GroupStats::from_counts(2, 1, 1),
            GroupStats::from_counts(2, 2, 0),
        ];
        close(wc_fst(&groups).unwrap(), 1.0 / 7.0);
    }

    #[test]
    fn hudson_uses_unbiased_within_population_diversity() {
        let groups = [
            GroupStats::from_counts(2, 0, 0),
            GroupStats::from_counts(2, 4, 0),
        ];
        close(hudson_fst(&groups[0], &groups[1]).unwrap(), 1.0);
    }

    #[test]
    fn hudson_returns_none_when_between_diversity_is_zero() {
        let group = GroupStats::from_counts(4, 0, 0);
        assert!(hudson_fst(&group, &group).is_none());
    }

    #[test]
    fn bitwise_bed_decode_ignores_missing_and_padding() {
        // Calls are: 0/0, 0/1, 1/1, missing, 0/0.  The final four BED
        // lanes are padding and must not contribute to any count.
        let row = [0b01_11_10_00u8, 0b00u8];
        let masks = build_group_masks(
            &[
                Some("A".to_string()),
                Some("A".to_string()),
                Some("B".to_string()),
                Some("B".to_string()),
                Some("B".to_string()),
            ],
            5,
        )
        .unwrap();
        let stats = stats_for_row(&row, 5, &masks);
        assert_eq!(
            stats,
            vec![
                GroupStats::from_counts(2, 1, 1),
                GroupStats::from_counts(2, 2, 0),
            ]
        );
    }

    #[test]
    fn bitwise_bed_decode_handles_multiple_sample_words() {
        let n_samples = 65;
        let mut row = vec![0u8; (n_samples + 3) / 4];
        // Sample 0: hom-ref; sample 63: het; sample 64: hom-alt.
        row[0] = 0b00;
        row[15] |= 0b10 << 6;
        row[16] |= 0b11;
        let mut assignments = vec![Some("B".to_string()); n_samples];
        assignments[0] = Some("A".to_string());
        assignments[64] = Some("C".to_string());
        let masks = build_group_masks(&assignments, n_samples).unwrap();
        let stats = stats_for_row(&row, n_samples, &masks);
        assert_eq!(stats[0], GroupStats::from_counts(1, 0, 0));
        assert_eq!(stats[1], GroupStats::from_counts(63, 1, 1));
        assert_eq!(stats[2], GroupStats::from_counts(1, 2, 0));
    }

    #[test]
    fn within_requires_exact_fid_iid_matches() {
        let path = std::env::temp_dir().join(format!(
            "janusx_fst_within_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::write(&path, "F2 S1 A\nF1 S2 B\n").unwrap();
        let fam = [
            ("F1".to_string(), "S1".to_string()),
            ("F1".to_string(), "S2".to_string()),
        ];
        let err = read_within_groups(path.to_str().unwrap(), &fam).unwrap_err();
        assert!(err.contains("F2 S1"));
        let _ = std::fs::remove_file(path);
    }
}
