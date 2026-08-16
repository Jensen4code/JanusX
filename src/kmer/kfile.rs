use crate::breader::parse_bsite_header;
use crate::kmer::format::{KmergeMeta, SampleEntry, BSITE_HEADER_SIZE};
use anyhow::{bail, Context, Result};
use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::BoundObject;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::Instant;

#[derive(Clone, Debug)]
struct KfileLayout {
    meta_path: PathBuf,
    bsite_path: PathBuf,
    idv_path: PathBuf,
    meta: KmergeMeta,
    samples: Vec<SampleEntry>,
}

fn resolve_meta_path(prefix_or_meta: &Path) -> PathBuf {
    let raw = prefix_or_meta.to_string_lossy();
    if raw.ends_with(".meta.json") {
        prefix_or_meta.to_path_buf()
    } else {
        PathBuf::from(format!("{raw}.meta.json"))
    }
}

fn validate_meta(meta: &KmergeMeta, meta_path: &Path) -> Result<()> {
    if meta.format.trim() != "janusx-kmer-bitmatrix-v1" {
        bail!("unsupported kmerge format in {}", meta_path.display());
    }
    if meta.n_samples == 0 {
        bail!("kmerge metadata has zero samples: {}", meta_path.display());
    }
    if meta.n_kmers == 0 {
        bail!("kmerge metadata has zero k-mers: {}", meta_path.display());
    }
    if meta.matrix_layout.trim() != "column_major_bitset" {
        bail!(
            "unsupported matrix layout in {}: {}",
            meta_path.display(),
            meta.matrix_layout
        );
    }
    if meta.bit_order.trim() != "little_bit_order" {
        bail!(
            "unsupported bit order in {}: {}",
            meta_path.display(),
            meta.bit_order
        );
    }
    if meta.compression.trim() != "none" {
        bail!(
            "compressed bitmatrices are not supported in {}: {}",
            meta_path.display(),
            meta.compression
        );
    }
    let expected_bytes = meta
        .n_samples
        .checked_add(7)
        .ok_or_else(|| anyhow::anyhow!("bytes_per_col overflow in {}", meta_path.display()))?
        / 8;
    if meta.bytes_per_col != expected_bytes {
        bail!(
            "bytes_per_col mismatch in {}: metadata={} expected={}",
            meta_path.display(),
            meta.bytes_per_col,
            expected_bytes
        );
    }
    Ok(())
}

fn read_idv_file(path: &Path) -> Result<Vec<SampleEntry>> {
    let mut rows = Vec::new();
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(path)
        .with_context(|| format!("failed to open idv file: {}", path.display()))?;
    let headers = reader
        .headers()
        .with_context(|| format!("failed to read idv header: {}", path.display()))?
        .clone();
    if headers.len() < 3
        || headers.get(0) != Some("#idx")
        || headers.get(1) != Some("sample_id")
        || headers.get(2) != Some("kmc_prefix")
    {
        bail!("invalid idv header in {}", path.display());
    }
    for rec in reader.records() {
        let rec = rec.with_context(|| format!("failed to parse idv row: {}", path.display()))?;
        let index = rec
            .get(0)
            .ok_or_else(|| anyhow::anyhow!("missing idx in {}", path.display()))?
            .parse::<u32>()
            .with_context(|| format!("invalid idx in {}", path.display()))?;
        let sample_id = rec
            .get(1)
            .ok_or_else(|| anyhow::anyhow!("missing sample_id in {}", path.display()))?
            .trim()
            .to_string();
        if sample_id.is_empty() {
            bail!("empty sample_id in {}", path.display());
        }
        let kmc_prefix = rec
            .get(2)
            .ok_or_else(|| anyhow::anyhow!("missing kmc_prefix in {}", path.display()))?
            .to_string();
        rows.push(SampleEntry {
            index,
            sample_id,
            kmc_prefix,
        });
    }
    rows.sort_by_key(|row| row.index);
    let mut seen = HashSet::with_capacity(rows.len());
    for (expected, row) in rows.iter().enumerate() {
        if row.index as usize != expected {
            bail!(
                "idv indices must be contiguous from 0 in {} (saw {} at row {})",
                path.display(),
                row.index,
                expected
            );
        }
        if !seen.insert(row.sample_id.clone()) {
            bail!(
                "duplicate sample_id in {}: {}",
                path.display(),
                row.sample_id
            );
        }
    }
    Ok(rows)
}

fn load_layout(prefix_or_meta: &Path) -> Result<KfileLayout> {
    let meta_path = resolve_meta_path(prefix_or_meta);
    let meta_file = File::open(&meta_path)
        .with_context(|| format!("failed to open kmerge meta file: {}", meta_path.display()))?;
    let meta: KmergeMeta = serde_json::from_reader(meta_file)
        .with_context(|| format!("failed to parse kmerge meta file: {}", meta_path.display()))?;
    validate_meta(&meta, &meta_path)?;

    let base_dir = meta_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let bsite_path = base_dir.join(&meta.bsite_file);
    let idv_path = base_dir.join(&meta.idv_file);
    let mut bsite = File::open(&bsite_path)
        .with_context(|| format!("failed to open bsite file: {}", bsite_path.display()))?;
    let expected_payload = meta
        .n_kmers
        .checked_mul(meta.bytes_per_col)
        .ok_or_else(|| anyhow::anyhow!("bsite payload size overflow"))?;
    let expected_len = (BSITE_HEADER_SIZE as u64)
        .checked_add(expected_payload)
        .ok_or_else(|| anyhow::anyhow!("bsite file size overflow"))?;
    let actual_len = bsite
        .metadata()
        .with_context(|| format!("failed to stat bsite file: {}", bsite_path.display()))?
        .len();
    if actual_len != expected_len {
        bail!(
            "bsite payload length mismatch for {}: actual={} expected={}",
            bsite_path.display(),
            actual_len,
            expected_len
        );
    }
    let mut header = [0u8; BSITE_HEADER_SIZE];
    bsite
        .read_exact(&mut header)
        .with_context(|| format!("failed to read bsite header: {}", bsite_path.display()))?;
    let parsed = parse_bsite_header(&header, &bsite_path.display().to_string())
        .map_err(anyhow::Error::msg)?;
    if parsed.n_samples != meta.n_samples
        || parsed.n_kmers != meta.n_kmers
        || parsed.bytes_per_col != meta.bytes_per_col
    {
        bail!(
            "bsite header does not match metadata: {}",
            bsite_path.display()
        );
    }

    let samples = read_idv_file(&idv_path)?;
    if samples.len() != usize::try_from(meta.n_samples).context("n_samples does not fit usize")? {
        bail!(
            "idv sample count does not match metadata: {}",
            idv_path.display()
        );
    }
    Ok(KfileLayout {
        meta_path,
        bsite_path,
        idv_path,
        meta,
        samples,
    })
}

fn build_sample_selection(
    samples: &[SampleEntry],
    sample_indices: Option<Vec<usize>>,
) -> Result<(Vec<usize>, Vec<String>)> {
    let indices = sample_indices.unwrap_or_else(|| (0..samples.len()).collect());
    let mut seen = HashSet::with_capacity(indices.len());
    let mut ids = Vec::with_capacity(indices.len());
    for &idx in &indices {
        let sample = samples
            .get(idx)
            .ok_or_else(|| anyhow::anyhow!("sample index out of range: {idx}"))?;
        if !seen.insert(idx) {
            bail!("duplicate sample index: {idx}");
        }
        ids.push(sample.sample_id.clone());
    }
    if indices.is_empty() {
        bail!("sample selection is empty");
    }
    Ok((indices, ids))
}

fn decode_bsite_chunk(
    bytes: &[u8],
    n_samples: usize,
    n_rows: usize,
    _row_offset: u64,
    selected_indices: &[usize],
) -> Result<(Vec<f32>, Vec<u64>)> {
    let bytes_per_col = n_samples
        .checked_add(7)
        .ok_or_else(|| anyhow::anyhow!("sample count overflow"))?
        / 8;
    let expected_len = n_rows
        .checked_mul(bytes_per_col)
        .ok_or_else(|| anyhow::anyhow!("chunk payload size overflow"))?;
    if bytes.len() != expected_len {
        bail!(
            "bsite chunk length mismatch: actual={} expected={}",
            bytes.len(),
            expected_len
        );
    }
    let indices: Vec<usize> = if selected_indices.is_empty() {
        (0..n_samples).collect()
    } else {
        selected_indices.to_vec()
    };
    for &idx in &indices {
        if idx >= n_samples {
            bail!("selected sample index out of range: {idx}");
        }
    }
    let mut dosage = vec![0.0f32; n_rows * indices.len()];
    let mut presence = vec![0u64; n_rows];
    for row in 0..n_rows {
        let col = &bytes[row * bytes_per_col..(row + 1) * bytes_per_col];
        let mut row_presence = 0u64;
        for (out_idx, &sample_idx) in indices.iter().enumerate() {
            let present = ((col[sample_idx >> 3] >> (sample_idx & 7)) & 1) != 0;
            if present {
                dosage[row * indices.len() + out_idx] = 2.0;
                row_presence += 1;
            }
        }
        presence[row] = row_presence;
    }
    Ok((dosage, presence))
}

#[inline]
fn maf_from_presence(presence: u64, n_samples: usize) -> f32 {
    if n_samples == 0 {
        return f32::NAN;
    }
    let p = presence as f32 / n_samples as f32;
    p.min(1.0 - p)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct KfileGrmStageTiming {
    pub read_ns: u64,
    pub filter_ns: u64,
    pub decode_ns: u64,
}

impl KfileGrmStageTiming {
    fn add_read(&mut self, elapsed: std::time::Duration) {
        self.read_ns = self
            .read_ns
            .saturating_add(elapsed.as_nanos().min(u64::MAX as u128) as u64);
    }

    fn add_filter(&mut self, elapsed: std::time::Duration) {
        self.filter_ns = self
            .filter_ns
            .saturating_add(elapsed.as_nanos().min(u64::MAX as u128) as u64);
    }

    fn add_decode(&mut self, elapsed: std::time::Duration) {
        self.decode_ns = self
            .decode_ns
            .saturating_add(elapsed.as_nanos().min(u64::MAX as u128) as u64);
    }
}

#[derive(Serialize, Deserialize)]
pub(crate) struct KfileGrmPreparedBlock {
    packed: Vec<u8>,
    retained_row_offsets: Vec<usize>,
    row_center: Vec<f32>,
    row_scale: Vec<f32>,
    pub scanned_rows: usize,
    pub denominator: f64,
}

#[derive(Clone, Copy, Debug)]
struct KfileBitsetDecodeEntry {
    bit: usize,
    column: usize,
}

#[derive(Clone, Debug)]
struct KfileBitsetDecodeGroup {
    byte: usize,
    mask: u8,
    entries: Vec<KfileBitsetDecodeEntry>,
}

/// Precomputed bit positions for one decoded sample stripe.
///
/// The plan groups requested samples by source byte. This removes repeated
/// division/masking from the hot row loop while retaining the caller's sample
/// order in the output columns.
#[derive(Clone, Debug)]
pub(crate) struct KfileBitsetDecodePlan {
    width: usize,
    full_identity: bool,
    groups: Vec<KfileBitsetDecodeGroup>,
}

impl KfileBitsetDecodePlan {
    pub(crate) fn new(sample_indices: &[usize], n_samples_full: usize) -> Result<Self> {
        validate_grm_selection(sample_indices, n_samples_full)?;
        let bytes_per_col = n_samples_full
            .checked_add(7)
            .ok_or_else(|| anyhow::anyhow!("sample count overflow"))?
            / 8;
        let mut grouped = (0..bytes_per_col)
            .map(|_| Vec::<KfileBitsetDecodeEntry>::new())
            .collect::<Vec<_>>();
        for (column, &sample_index) in sample_indices.iter().enumerate() {
            grouped[sample_index >> 3].push(KfileBitsetDecodeEntry {
                bit: sample_index & 7,
                column,
            });
        }
        let groups = grouped
            .into_iter()
            .enumerate()
            .filter_map(|(byte, entries)| {
                if entries.is_empty() {
                    return None;
                }
                let mask = entries
                    .iter()
                    .fold(0u8, |mask, entry| mask | (1u8 << entry.bit));
                Some(KfileBitsetDecodeGroup {
                    byte,
                    mask,
                    entries,
                })
            })
            .collect::<Vec<_>>();
        let full_identity = sample_indices.len() == n_samples_full
            && sample_indices
                .iter()
                .enumerate()
                .all(|(expected, &actual)| expected == actual);
        Ok(Self {
            width: sample_indices.len(),
            full_identity,
            groups,
        })
    }

    pub(crate) fn width(&self) -> usize {
        self.width
    }
}

fn bitset_dosage_lut() -> &'static [[f32; 8]; 256] {
    static LUT: OnceLock<[[f32; 8]; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut table = [[0.0_f32; 8]; 256];
        for (value, row) in table.iter_mut().enumerate() {
            for (bit, dosage) in row.iter_mut().enumerate() {
                if (value & (1usize << bit)) != 0 {
                    *dosage = 2.0;
                }
            }
        }
        table
    })
}

#[inline]
fn full_bitset_presence(row: &[u8], n_samples: usize) -> usize {
    let full_bytes = n_samples / 8;
    let mut count = 0u32;
    let whole_bytes = full_bytes - (full_bytes % 8);
    for chunk in row[..whole_bytes].chunks_exact(8) {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(chunk);
        count += u64::from_le_bytes(bytes).count_ones();
    }
    for &value in &row[whole_bytes..full_bytes] {
        count += value.count_ones();
    }
    let tail = n_samples % 8;
    if tail != 0 {
        count += (row[full_bytes] & ((1u8 << tail) - 1)).count_ones();
    }
    count as usize
}

impl KfileGrmPreparedBlock {
    pub(crate) fn retained_rows(&self) -> usize {
        self.retained_row_offsets.len()
    }

    pub(crate) fn packed_len(&self) -> usize {
        self.packed.len()
    }
}

fn validate_grm_parameters(method: usize, maf_threshold: f32) -> Result<()> {
    if method != 1 && method != 2 {
        bail!("GRM method must be 1 or 2, got {method}");
    }
    if !maf_threshold.is_finite() || !(0.0..=0.5).contains(&maf_threshold) {
        bail!("GRM MAF threshold must be finite and within 0..=0.5, got {maf_threshold}");
    }
    Ok(())
}

fn validate_grm_value_type(meta: &KmergeMeta, meta_path: &Path) -> Result<()> {
    if meta.value_type.trim() != "binary_presence" {
        bail!(
            "unsupported GRM value type in {}: {}",
            meta_path.display(),
            meta.value_type
        );
    }
    Ok(())
}

fn validate_grm_selection(sample_indices: &[usize], n_samples_full: usize) -> Result<()> {
    if sample_indices.is_empty() {
        bail!("sample selection is empty");
    }
    let mut seen = HashSet::with_capacity(sample_indices.len());
    for &index in sample_indices {
        if index >= n_samples_full {
            bail!("sample index out of range: {index}");
        }
        if !seen.insert(index) {
            bail!("duplicate sample index: {index}");
        }
    }
    Ok(())
}

pub(crate) fn prepare_grm_bsite_block(
    bytes: &[u8],
    n_samples_full: usize,
    n_rows: usize,
    sample_indices: &[usize],
    method: usize,
    maf_threshold: f32,
) -> Result<KfileGrmPreparedBlock> {
    validate_grm_parameters(method, maf_threshold)?;
    validate_grm_selection(sample_indices, n_samples_full)?;
    let bytes_per_col = n_samples_full
        .checked_add(7)
        .ok_or_else(|| anyhow::anyhow!("sample count overflow"))?
        / 8;
    let expected_len = n_rows
        .checked_mul(bytes_per_col)
        .ok_or_else(|| anyhow::anyhow!("GRM block payload size overflow"))?;
    if bytes.len() != expected_len {
        bail!(
            "GRM block length mismatch: actual={} expected={}",
            bytes.len(),
            expected_len
        );
    }

    let selected_n = sample_indices.len();
    let decode_plan = KfileBitsetDecodePlan::new(sample_indices, n_samples_full)?;
    let mut retained_row_offsets = Vec::new();
    let mut row_center = Vec::new();
    let mut row_scale = Vec::new();
    let mut denominator = 0.0f64;
    let full_identity = decode_plan.full_identity;
    for row in 0..n_rows {
        let packed_row = &bytes[row * bytes_per_col..(row + 1) * bytes_per_col];
        let presence = if full_identity {
            full_bitset_presence(packed_row, n_samples_full)
        } else {
            decode_plan
                .groups
                .iter()
                .map(|group| (packed_row[group.byte] & group.mask).count_ones() as usize)
                .sum()
        };
        let p = presence as f64 / selected_n as f64;
        let maf = p.min(1.0 - p);
        let variance = 2.0 * p * (1.0 - p);
        if maf >= maf_threshold as f64 && variance > 1e-12 {
            retained_row_offsets.push(row);
            row_center.push((2.0 * p) as f32);
            if method == 1 {
                row_scale.push(1.0);
                denominator += variance;
            } else {
                row_scale.push((1.0 / variance.sqrt()) as f32);
                denominator += 1.0;
            }
        }
    }
    Ok(KfileGrmPreparedBlock {
        packed: bytes.to_vec(),
        retained_row_offsets,
        row_center,
        row_scale,
        scanned_rows: n_rows,
        denominator,
    })
}

pub(crate) fn decode_grm_prepared_block_into(
    block: &KfileGrmPreparedBlock,
    sample_indices: &[usize],
    out: &mut [f32],
) -> Result<()> {
    if sample_indices.is_empty() {
        return Ok(());
    }
    let n_samples_full = sample_indices
        .iter()
        .copied()
        .max()
        .and_then(|index| index.checked_add(1))
        .unwrap_or(0);
    let plan = KfileBitsetDecodePlan::new(sample_indices, n_samples_full)?;
    decode_grm_prepared_block_into_with_plan(block, &plan, out)
}

pub(crate) fn decode_grm_prepared_block_into_with_plan(
    block: &KfileGrmPreparedBlock,
    plan: &KfileBitsetDecodePlan,
    out: &mut [f32],
) -> Result<()> {
    let required_len = block
        .retained_rows()
        .checked_mul(plan.width())
        .ok_or_else(|| anyhow::anyhow!("GRM decoded output size overflow"))?;
    if out.len() < required_len {
        bail!(
            "GRM decode output is too small: actual={} required={}",
            out.len(),
            required_len
        );
    }
    if block.scanned_rows == 0 {
        if !block.packed.is_empty() {
            bail!("GRM prepared block has bytes but zero scanned rows");
        }
        return Ok(());
    }
    if block.packed.len() % block.scanned_rows != 0 {
        bail!("GRM prepared block has an invalid packed layout");
    }
    let bytes_per_col = block.packed.len() / block.scanned_rows;
    let lut = bitset_dosage_lut();
    for (decoded_row, &packed_row) in block.retained_row_offsets.iter().enumerate() {
        if packed_row >= block.scanned_rows {
            bail!("GRM prepared block has an invalid retained row offset");
        }
        let row_bytes = &block.packed[packed_row * bytes_per_col..(packed_row + 1) * bytes_per_col];
        let output_row = &mut out[decoded_row * plan.width()..(decoded_row + 1) * plan.width()];
        if plan.full_identity {
            for group in &plan.groups {
                if group.byte >= bytes_per_col {
                    bail!(
                        "bitset decode plan byte out of range: {} >= {}",
                        group.byte,
                        bytes_per_col
                    );
                }
                let expanded = &lut[row_bytes[group.byte] as usize];
                let output_start = group.byte * 8;
                for (offset, &genotype) in expanded.iter().take(group.entries.len()).enumerate() {
                    output_row[output_start + offset] =
                        (genotype - block.row_center[decoded_row]) * block.row_scale[decoded_row];
                }
            }
        } else {
            for group in &plan.groups {
                if group.byte >= bytes_per_col {
                    bail!(
                        "bitset decode plan byte out of range: {} >= {}",
                        group.byte,
                        bytes_per_col
                    );
                }
                let expanded = &lut[row_bytes[group.byte] as usize];
                for entry in &group.entries {
                    output_row[entry.column] = (expanded[entry.bit]
                        - block.row_center[decoded_row])
                        * block.row_scale[decoded_row];
                }
            }
        }
    }
    Ok(())
}

pub(crate) struct KfileGrmSource {
    layout: KfileLayout,
    bsite: File,
    sample_indices: Vec<usize>,
    sample_ids: Vec<String>,
    method: usize,
    maf_threshold: f32,
    next_row: u64,
    effective_rows: usize,
    denominator: f64,
    stage_timing: KfileGrmStageTiming,
    full_decode_plan: KfileBitsetDecodePlan,
}

#[allow(dead_code)]
impl KfileGrmSource {
    pub(crate) fn open(
        prefix: &Path,
        sample_indices: Option<&[usize]>,
        method: usize,
        maf_threshold: f32,
    ) -> Result<Self> {
        validate_grm_parameters(method, maf_threshold)?;
        let layout = load_layout(prefix)?;
        validate_grm_value_type(&layout.meta, &layout.meta_path)?;
        let requested_indices = sample_indices.map(|indices| indices.to_vec());
        let (sample_indices, sample_ids) =
            build_sample_selection(&layout.samples, requested_indices)?;
        let n_samples_full =
            usize::try_from(layout.meta.n_samples).context("n_samples does not fit usize")?;
        let full_decode_plan = KfileBitsetDecodePlan::new(&sample_indices, n_samples_full)?;
        let bsite = File::open(&layout.bsite_path).with_context(|| {
            format!("failed to open bsite file: {}", layout.bsite_path.display())
        })?;
        Ok(Self {
            layout,
            bsite,
            sample_indices,
            sample_ids,
            method,
            maf_threshold,
            next_row: 0,
            effective_rows: 0,
            denominator: 0.0,
            stage_timing: KfileGrmStageTiming::default(),
            full_decode_plan,
        })
    }

    pub(crate) fn next_prepared_block(
        &mut self,
        block_rows: usize,
    ) -> Result<Option<KfileGrmPreparedBlock>> {
        if block_rows == 0 {
            bail!("GRM block_rows must be > 0");
        }
        if self.next_row >= self.layout.meta.n_kmers {
            return Ok(None);
        }
        let rows_left = self.layout.meta.n_kmers - self.next_row;
        let n_rows = usize::try_from(rows_left.min(block_rows as u64))
            .context("GRM block row count does not fit usize")?;
        let bytes_per_col = usize::try_from(self.layout.meta.bytes_per_col)
            .context("bytes_per_col does not fit usize")?;
        let byte_len = n_rows
            .checked_mul(bytes_per_col)
            .ok_or_else(|| anyhow::anyhow!("GRM block payload size overflow"))?;
        let offset = (BSITE_HEADER_SIZE as u64)
            .checked_add(
                self.next_row
                    .checked_mul(self.layout.meta.bytes_per_col)
                    .ok_or_else(|| anyhow::anyhow!("GRM bsite offset overflow"))?,
            )
            .ok_or_else(|| anyhow::anyhow!("GRM bsite offset overflow"))?;
        self.bsite
            .seek(SeekFrom::Start(offset))
            .with_context(|| format!("failed to seek {}", self.layout.bsite_path.display()))?;
        let mut bytes = vec![0u8; byte_len];
        let read_started = Instant::now();
        self.bsite
            .read_exact(&mut bytes)
            .with_context(|| format!("failed to read {}", self.layout.bsite_path.display()))?;
        self.stage_timing.add_read(read_started.elapsed());

        let filter_started = Instant::now();
        let block = prepare_grm_bsite_block(
            &bytes,
            self.n_samples_full(),
            n_rows,
            &self.sample_indices,
            self.method,
            self.maf_threshold,
        )?;
        self.stage_timing.add_filter(filter_started.elapsed());
        self.next_row += n_rows as u64;
        self.effective_rows = self
            .effective_rows
            .checked_add(block.retained_rows())
            .ok_or_else(|| anyhow::anyhow!("effective GRM row count overflow"))?;
        self.denominator += block.denominator;
        Ok(Some(block))
    }

    pub(crate) fn decode_block_into(
        &mut self,
        block: &KfileGrmPreparedBlock,
        selected_positions: Range<usize>,
        out: &mut [f32],
    ) -> Result<()> {
        if selected_positions.start > selected_positions.end
            || selected_positions.end > self.sample_indices.len()
        {
            bail!("GRM selected sample positions are out of range");
        }
        let decode_started = Instant::now();
        if selected_positions.start == 0 && selected_positions.end == self.sample_indices.len() {
            decode_grm_prepared_block_into_with_plan(block, &self.full_decode_plan, out)?;
        } else {
            let selected = &self.sample_indices[selected_positions.clone()];
            let plan = KfileBitsetDecodePlan::new(selected, self.n_samples_full())?;
            decode_grm_prepared_block_into_with_plan(block, &plan, out)?;
        }
        self.stage_timing.add_decode(decode_started.elapsed());
        Ok(())
    }

    pub(crate) fn decode_block_into_with_plan(
        &mut self,
        block: &KfileGrmPreparedBlock,
        plan: &KfileBitsetDecodePlan,
        out: &mut [f32],
    ) -> Result<()> {
        let decode_started = Instant::now();
        decode_grm_prepared_block_into_with_plan(block, plan, out)?;
        self.stage_timing.add_decode(decode_started.elapsed());
        Ok(())
    }

    pub(crate) fn reset_scan(&mut self) -> Result<()> {
        self.bsite
            .seek(SeekFrom::Start(BSITE_HEADER_SIZE as u64))
            .with_context(|| format!("failed to reset {}", self.layout.bsite_path.display()))?;
        self.next_row = 0;
        self.effective_rows = 0;
        self.denominator = 0.0;
        Ok(())
    }

    pub(crate) fn reset_timing(&mut self) {
        self.stage_timing = KfileGrmStageTiming::default();
    }

    pub(crate) fn n_samples(&self) -> usize {
        self.sample_indices.len()
    }

    pub(crate) fn n_samples_full(&self) -> usize {
        self.layout.meta.n_samples as usize
    }

    pub(crate) fn n_kmers(&self) -> u64 {
        self.layout.meta.n_kmers
    }

    pub(crate) fn sample_indices(&self) -> &[usize] {
        &self.sample_indices
    }

    pub(crate) fn sample_ids(&self) -> &[String] {
        &self.sample_ids
    }

    pub(crate) fn scanned_rows(&self) -> usize {
        self.next_row as usize
    }

    pub(crate) fn effective_rows(&self) -> usize {
        self.effective_rows
    }

    pub(crate) fn denominator(&self) -> f64 {
        self.denominator
    }

    pub(crate) fn stage_timing(&self) -> KfileGrmStageTiming {
        self.stage_timing
    }
}

#[pyclass]
pub struct KfileChunkReader {
    bsite: File,
    bsite_path: PathBuf,
    n_samples_full: usize,
    n_kmers: u64,
    bytes_per_col: usize,
    sample_indices: Vec<usize>,
    sample_ids: Vec<String>,
    next_row: u64,
}

#[pymethods]
impl KfileChunkReader {
    #[new]
    #[pyo3(signature = (prefix, sample_indices=None, mmap_window_mb=None))]
    fn new(
        prefix: String,
        sample_indices: Option<Vec<usize>>,
        mmap_window_mb: Option<usize>,
    ) -> PyResult<Self> {
        let _ = mmap_window_mb;
        let layout =
            load_layout(Path::new(&prefix)).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let n_samples_full = usize::try_from(layout.meta.n_samples)
            .map_err(|_| PyRuntimeError::new_err("n_samples does not fit usize"))?;
        let (sample_indices, sample_ids) = build_sample_selection(&layout.samples, sample_indices)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let bsite = File::open(&layout.bsite_path)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to open bsite file: {e}")))?;
        Ok(Self {
            bsite,
            bsite_path: layout.bsite_path,
            n_samples_full,
            n_kmers: layout.meta.n_kmers,
            bytes_per_col: usize::try_from(layout.meta.bytes_per_col)
                .map_err(|_| PyRuntimeError::new_err("bytes_per_col does not fit usize"))?,
            sample_indices,
            sample_ids,
            next_row: 0,
        })
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.sample_indices.len()
    }

    #[getter]
    fn n_samples_full(&self) -> usize {
        self.n_samples_full
    }

    #[getter]
    fn n_kmers(&self) -> u64 {
        self.n_kmers
    }

    #[getter]
    fn bytes_per_col(&self) -> usize {
        self.bytes_per_col
    }

    #[getter]
    fn sample_ids(&self) -> Vec<String> {
        self.sample_ids.clone()
    }

    #[getter]
    fn next_row(&self) -> u64 {
        self.next_row
    }

    #[pyo3(signature = (chunk_size))]
    fn next_chunk<'py>(
        &mut self,
        py: Python<'py>,
        chunk_size: usize,
    ) -> PyResult<
        Option<(
            Bound<'py, PyArray2<f32>>,
            Bound<'py, PyArray1<f32>>,
            Bound<'py, PyArray1<i64>>,
        )>,
    > {
        if chunk_size == 0 {
            return Err(PyValueError::new_err("chunk_size must be > 0"));
        }
        if self.next_row >= self.n_kmers {
            return Ok(None);
        }
        let rows_left = self.n_kmers - self.next_row;
        let n_rows = usize::try_from(rows_left.min(chunk_size as u64))
            .map_err(|_| PyRuntimeError::new_err("chunk row count does not fit usize"))?;
        let byte_len = n_rows
            .checked_mul(self.bytes_per_col)
            .ok_or_else(|| PyRuntimeError::new_err("chunk payload size overflow"))?;
        let offset = (BSITE_HEADER_SIZE as u64)
            .checked_add(
                self.next_row
                    .checked_mul(self.bytes_per_col as u64)
                    .ok_or_else(|| PyRuntimeError::new_err("bsite offset overflow"))?,
            )
            .ok_or_else(|| PyRuntimeError::new_err("bsite offset overflow"))?;
        self.bsite.seek(SeekFrom::Start(offset)).map_err(|e| {
            PyRuntimeError::new_err(format!("failed to seek {}: {e}", self.bsite_path.display()))
        })?;
        let mut bytes = vec![0u8; byte_len];
        self.bsite.read_exact(&mut bytes).map_err(|e| {
            PyRuntimeError::new_err(format!("failed to read {}: {e}", self.bsite_path.display()))
        })?;
        let (dosage, presence) = decode_bsite_chunk(
            &bytes,
            self.n_samples_full,
            n_rows,
            self.next_row,
            &self.sample_indices,
        )
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let n_selected = self.sample_indices.len();
        let maf = presence
            .iter()
            .map(|&count| maf_from_presence(count, n_selected))
            .collect::<Vec<_>>();
        let row_indices = (self.next_row..self.next_row + n_rows as u64)
            .map(|idx| idx as i64)
            .collect::<Vec<_>>();
        self.next_row += n_rows as u64;
        let geno = Array2::from_shape_vec((n_rows, n_selected), dosage)
            .map_err(|e| PyRuntimeError::new_err(format!("failed to shape dosage chunk: {e}")))?;
        Ok(Some((
            PyArray2::from_owned_array(py, geno).into_bound(),
            PyArray1::from_owned_array(py, Array1::from_vec(maf)).into_bound(),
            PyArray1::from_owned_array(py, Array1::from_vec(row_indices)).into_bound(),
        )))
    }
}

#[pyfunction(name = "kfile_inspect")]
pub fn kfile_inspect_py<'py>(py: Python<'py>, prefix: String) -> PyResult<Bound<'py, PyDict>> {
    let layout =
        load_layout(Path::new(&prefix)).map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let out = PyDict::new(py);
    out.set_item("prefix", prefix)?;
    out.set_item("meta", layout.meta_path.to_string_lossy().as_ref())?;
    out.set_item("bsite", layout.bsite_path.to_string_lossy().as_ref())?;
    out.set_item("idv", layout.idv_path.to_string_lossy().as_ref())?;
    out.set_item("k", layout.meta.k)?;
    out.set_item("n_samples", layout.meta.n_samples)?;
    out.set_item("n_kmers", layout.meta.n_kmers)?;
    out.set_item("bytes_per_col", layout.meta.bytes_per_col)?;
    out.set_item(
        "sample_ids",
        layout
            .samples
            .iter()
            .map(|sample| sample.sample_id.clone())
            .collect::<Vec<_>>(),
    )?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::{
        decode_bsite_chunk, decode_grm_prepared_block_into,
        decode_grm_prepared_block_into_with_plan, maf_from_presence, prepare_grm_bsite_block,
        KfileBitsetDecodePlan, KfileChunkReader, KfileGrmSource,
    };
    use crate::kmer::format::{BsiteHeader, KmergeMeta};
    use std::fs::{self, File};
    use std::io::Write;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static FIXTURE_COUNTER: AtomicUsize = AtomicUsize::new(0);

    fn grm_fixture_dir() -> PathBuf {
        let unique = FIXTURE_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "janusx-kfile-grm-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time")
                .as_nanos(),
            unique
        ));
        fs::create_dir(&path).expect("create fixture directory");
        path
    }

    fn write_grm_fixture(dir: &Path, meta: &KmergeMeta, idv_rows: &str, payload: &[u8]) {
        let prefix = dir.join("fixture");
        fs::write(
            PathBuf::from(format!("{}.meta.json", prefix.display())),
            serde_json::to_vec(meta).expect("serialize metadata"),
        )
        .expect("write metadata");
        fs::write(dir.join(&meta.idv_file), idv_rows).expect("write idv");
        let mut bsite = File::create(dir.join(&meta.bsite_file)).expect("create bsite");
        BsiteHeader {
            n_samples: meta.n_samples,
            n_kmers: meta.n_kmers,
            bytes_per_col: meta.bytes_per_col,
        }
        .write_to(&mut bsite)
        .expect("write bsite header");
        bsite.write_all(payload).expect("write bsite payload");
    }

    fn valid_grm_meta() -> KmergeMeta {
        KmergeMeta {
            format: "janusx-kmer-bitmatrix-v1".to_string(),
            k: 31,
            n_samples: 4,
            n_kmers: 3,
            bytes_per_col: 1,
            encoding: "acgt_2bit".to_string(),
            canonical: true,
            matrix_layout: "column_major_bitset".to_string(),
            value_type: "binary_presence".to_string(),
            bit_order: "little_bit_order".to_string(),
            bkmer_file: "fixture.bkmer".to_string(),
            bsite_file: "fixture.bsite".to_string(),
            idv_file: "fixture.idv".to_string(),
            min_count: 1,
            min_presence_rate: 0.0,
            max_presence_rate: 1.0,
            bucket_bits: 8,
            compression: "none".to_string(),
        }
    }

    const VALID_IDV: &str =
        "#idx\tsample_id\tkmc_prefix\n0\ts0\tp0\n1\ts1\tp1\n2\ts2\tp2\n3\ts3\tp3\n";

    #[test]
    fn grm_block_filters_on_selected_sample_maf_and_centers_zero_two_dosage() {
        // rows: 0011, 1111, 0001 over four selected samples
        let bytes = [0b0000_0011, 0b0000_1111, 0b0000_0001];
        let block = prepare_grm_bsite_block(&bytes, 4, 3, &[0, 1, 2, 3], 1, 0.25).unwrap();
        let mut values = vec![0.0; block.retained_rows() * 4];
        decode_grm_prepared_block_into(&block, &[0, 1, 2, 3], &mut values).unwrap();
        assert_eq!(block.retained_rows(), 2);
        assert_eq!(block.scanned_rows, 3);
        assert_eq!(values, vec![1.0, 1.0, -1.0, -1.0, 1.5, -0.5, -0.5, -0.5]);
        assert!((block.denominator - 0.875).abs() < 1e-12);
    }

    #[test]
    fn grm_block_method_two_excludes_monomorphic_rows_and_standardizes() {
        let bytes = [0b0000_0011, 0b0000_1111];
        let block = prepare_grm_bsite_block(&bytes, 4, 2, &[3, 1, 0, 2], 2, 0.0).unwrap();
        let mut values = vec![0.0; block.retained_rows() * 4];
        decode_grm_prepared_block_into(&block, &[3, 1, 0, 2], &mut values).unwrap();
        assert_eq!(block.retained_rows(), 1);
        assert_eq!(block.denominator, 1.0);
        assert_eq!(values.len(), 4);
        assert!((values.iter().map(|x| x * x).sum::<f32>() - 8.0).abs() < 1e-5);
    }

    #[test]
    fn grm_bitset_lut_plan_matches_scalar_for_all_bytes_and_reordered_subset() {
        let n_samples = 13usize;
        let bytes_per_col = n_samples.div_ceil(8);
        let mut bytes = Vec::<u8>::new();
        for value in 1u8..=254u8 {
            bytes.push(value);
            bytes.push((!value) & 0x1f);
        }
        let n_rows = bytes.len() / bytes_per_col;
        let selections = vec![
            (0..n_samples).collect::<Vec<_>>(),
            vec![12usize, 0, 7, 3, 10],
        ];
        for selected in selections {
            let block = prepare_grm_bsite_block(
                bytes.as_slice(),
                n_samples,
                n_rows,
                selected.as_slice(),
                1,
                0.0,
            )
            .expect("prepare bitset block");
            let plan = KfileBitsetDecodePlan::new(selected.as_slice(), n_samples)
                .expect("build bitset decode plan");
            let mut got = vec![0.0_f32; block.retained_rows() * selected.len()];
            decode_grm_prepared_block_into_with_plan(&block, &plan, &mut got)
                .expect("decode with bitset plan");
            let mut expected = vec![0.0_f32; got.len()];
            for (decoded_row, &raw_row) in block.retained_row_offsets.iter().enumerate() {
                let row_bytes = &bytes[raw_row * bytes_per_col..(raw_row + 1) * bytes_per_col];
                for (column, &sample_index) in selected.iter().enumerate() {
                    let genotype =
                        if ((row_bytes[sample_index >> 3] >> (sample_index & 7)) & 1) != 0 {
                            2.0_f32
                        } else {
                            0.0_f32
                        };
                    expected[decoded_row * selected.len() + column] =
                        genotype - block.row_center[decoded_row];
                }
            }
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn decodes_little_endian_presence_bits_to_dosage() {
        let (dosage, presence) =
            decode_bsite_chunk(&[0b0101_0011], 6, 1, 0, &[]).expect("decode chunk");
        assert_eq!(dosage, vec![2.0, 2.0, 0.0, 0.0, 2.0, 0.0]);
        assert_eq!(presence, vec![3]);
    }

    #[test]
    fn decodes_selected_samples_in_requested_order() {
        let (dosage, presence) =
            decode_bsite_chunk(&[0b0101_0011, 0b0000_1000], 6, 2, 0, &[5, 0, 3])
                .expect("decode selected chunk");
        assert_eq!(dosage, vec![0.0, 2.0, 0.0, 0.0, 0.0, 2.0]);
        assert_eq!(presence, vec![1, 1]);
    }

    #[test]
    fn maf_uses_presence_frequency_without_filtering() {
        assert_eq!(maf_from_presence(0, 4), 0.0);
        assert_eq!(maf_from_presence(1, 4), 0.25);
        assert_eq!(maf_from_presence(3, 4), 0.25);
        assert_eq!(maf_from_presence(4, 4), 0.0);
    }

    #[test]
    fn chunk_row_offset_is_preserved() {
        let (dosage, presence) =
            decode_bsite_chunk(&[0b0000_0001, 0b0000_0010], 2, 2, 7, &[]).expect("decode rows");
        assert_eq!(dosage, vec![2.0, 0.0, 0.0, 2.0]);
        assert_eq!(presence, vec![1, 1]);
    }

    #[test]
    fn grm_source_accepts_prefix_or_metadata_and_resets_scan_without_resetting_timing() {
        let dir = grm_fixture_dir();
        let meta = valid_grm_meta();
        write_grm_fixture(
            &dir,
            &meta,
            VALID_IDV,
            &[0b0000_0011, 0b0000_1111, 0b0000_0001],
        );
        let prefix = dir.join("fixture");
        let meta_path = PathBuf::from(format!("{}.meta.json", prefix.display()));
        let mut from_prefix = KfileGrmSource::open(&prefix, None, 1, 0.25).unwrap();
        let from_meta = KfileGrmSource::open(&meta_path, None, 1, 0.25).unwrap();
        assert_eq!(from_prefix.sample_ids(), from_meta.sample_ids());
        assert_eq!(from_prefix.n_samples(), from_meta.n_samples());

        let first = from_prefix.next_prepared_block(3).unwrap().unwrap();
        let mut first_values = vec![0.0; first.retained_rows() * 4];
        from_prefix
            .decode_block_into(&first, 0..4, &mut first_values)
            .unwrap();
        let timing_before_reset = from_prefix.stage_timing();
        from_prefix.reset_scan().unwrap();
        assert_eq!(from_prefix.stage_timing(), timing_before_reset);
        assert_eq!(from_prefix.scanned_rows(), 0);
        assert_eq!(from_prefix.effective_rows(), 0);
        assert_eq!(from_prefix.denominator(), 0.0);
        let repeated = from_prefix.next_prepared_block(3).unwrap().unwrap();
        let mut repeated_values = vec![0.0; repeated.retained_rows() * 4];
        from_prefix
            .decode_block_into(&repeated, 0..4, &mut repeated_values)
            .unwrap();
        assert_eq!(first_values, repeated_values);
        assert_eq!(first.denominator, repeated.denominator);
        fs::remove_dir_all(dir).expect("remove fixture directory");
    }

    #[test]
    fn grm_source_decodes_selected_position_stripes_in_grm_sample_order() {
        let dir = grm_fixture_dir();
        let meta = valid_grm_meta();
        write_grm_fixture(
            &dir,
            &meta,
            VALID_IDV,
            &[0b0000_0011, 0b0000_0001, 0b0000_1110],
        );
        let prefix = dir.join("fixture");
        let mut source = KfileGrmSource::open(&prefix, Some(&[3, 1, 0, 2]), 1, 0.0).unwrap();
        let block = source.next_prepared_block(3).unwrap().unwrap();
        let rows = block.retained_rows();
        let mut full = vec![0.0; rows * 4];
        let mut left = vec![0.0; rows * 2];
        let mut right = vec![0.0; rows * 2];
        source.decode_block_into(&block, 0..4, &mut full).unwrap();
        source.decode_block_into(&block, 0..2, &mut left).unwrap();
        source.decode_block_into(&block, 2..4, &mut right).unwrap();
        let mut combined = vec![0.0; rows * 4];
        for row in 0..rows {
            combined[row * 4..row * 4 + 2].copy_from_slice(&left[row * 2..row * 2 + 2]);
            combined[row * 4 + 2..row * 4 + 4].copy_from_slice(&right[row * 2..row * 2 + 2]);
        }
        assert_eq!(combined, full);
        fs::remove_dir_all(dir).expect("remove fixture directory");
    }

    #[test]
    fn grm_source_requires_binary_presence_without_changing_chunk_reader() {
        let dir = grm_fixture_dir();
        let mut meta = valid_grm_meta();
        meta.value_type = "dosage".to_string();
        write_grm_fixture(&dir, &meta, VALID_IDV, &[3, 15, 1]);
        let prefix = dir.join("fixture");
        assert!(KfileChunkReader::new(prefix.to_string_lossy().into_owned(), None, None).is_ok());
        assert!(KfileGrmSource::open(&prefix, None, 1, 0.0).is_err());
        fs::remove_dir_all(dir).expect("remove fixture directory");
    }

    #[test]
    fn grm_source_rejects_invalid_metadata_and_selection() {
        let idv_cases = [
            (
                "duplicate idv sample",
                "#idx\tsample_id\tkmc_prefix\n0\ts0\tp0\n1\ts0\tp1\n2\ts2\tp2\n3\ts3\tp3\n",
            ),
            (
                "non-contiguous idv index",
                "#idx\tsample_id\tkmc_prefix\n0\ts0\tp0\n2\ts1\tp1\n3\ts2\tp2\n4\ts3\tp3\n",
            ),
        ];
        for (_name, idv_rows) in idv_cases {
            let dir = grm_fixture_dir();
            let meta = valid_grm_meta();
            write_grm_fixture(&dir, &meta, idv_rows, &[3, 15, 1]);
            assert!(KfileGrmSource::open(&dir.join("fixture"), None, 1, 0.0).is_err());
            fs::remove_dir_all(dir).expect("remove fixture directory");
        }

        for (name, field, value) in [
            ("non-binary value type", "value_type", "dosage"),
            ("unsupported bit order", "bit_order", "big_bit_order"),
            ("unsupported compression", "compression", "zstd"),
        ] {
            let dir = grm_fixture_dir();
            let mut meta = valid_grm_meta();
            match field {
                "value_type" => meta.value_type = value.to_string(),
                "bit_order" => meta.bit_order = value.to_string(),
                "compression" => meta.compression = value.to_string(),
                _ => unreachable!("unknown metadata field"),
            }
            write_grm_fixture(&dir, &meta, VALID_IDV, &[3, 15, 1]);
            assert!(
                KfileGrmSource::open(&dir.join("fixture"), None, 1, 0.0).is_err(),
                "{name} must fail"
            );
            fs::remove_dir_all(dir).expect("remove fixture directory");
        }

        let dir = grm_fixture_dir();
        let meta = valid_grm_meta();
        write_grm_fixture(&dir, &meta, VALID_IDV, &[3, 15, 1]);
        let prefix = dir.join("fixture");
        assert!(KfileGrmSource::open(&prefix, None, 3, 0.0).is_err());
        assert!(KfileGrmSource::open(&prefix, None, 1, -0.01).is_err());
        assert!(KfileGrmSource::open(&prefix, None, 1, 0.51).is_err());
        assert!(KfileGrmSource::open(&prefix, Some(&[0, 0]), 1, 0.0).is_err());
        assert!(KfileGrmSource::open(&prefix, Some(&[4]), 1, 0.0).is_err());
        fs::remove_file(dir.join("fixture.bsite")).expect("remove bsite");
        write_grm_fixture(&dir, &meta, VALID_IDV, &[3, 15]);
        assert!(KfileGrmSource::open(&prefix, None, 1, 0.0).is_err());
        fs::remove_dir_all(dir).expect("remove fixture directory");
    }
}
