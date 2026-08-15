use crate::breader::{parse_bkmer_header, parse_bsite_header};
use crate::kmer::encode::{canonical_code, decode_kmer_u64, encode_kmer_u64, ENCODING_NAME};
use crate::kmer::format::{
    BkmerHeader, BsiteHeader, KmergeMeta, SampleEntry, BKMER_HEADER_SIZE, BSITE_HEADER_SIZE,
};
use crate::kmer::writer::{write_idv_file, write_meta_json};
use anyhow::{bail, Context, Result};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::{BTreeMap, HashSet};
use std::fs::{self, File};
#[cfg(not(unix))]
use std::io::Read;
use std::io::{BufWriter, Seek, SeekFrom, Write};
#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;
#[cfg(unix)]
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::{mpsc::sync_channel, Arc, Mutex};
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_SCAN_BLOCK_ROWS: usize = 16_384;

#[derive(Clone, Debug)]
struct KformatLayout {
    bkmer_path: PathBuf,
    bsite_path: PathBuf,
    meta: KmergeMeta,
    samples: Vec<SampleEntry>,
}

struct KformatInput {
    bkmer: File,
    bsite: File,
    n_kmers: usize,
    n_samples: usize,
    bytes_per_col: usize,
}

struct InputBlock {
    block_idx: usize,
    start: usize,
    bkmer: Vec<u8>,
    bsite: Vec<u8>,
    selected: Vec<u32>,
}

fn read_exact_at(file: &File, buffer: &mut [u8], offset: u64) -> Result<()> {
    #[cfg(unix)]
    {
        file.read_exact_at(buffer, offset)?;
        return Ok(());
    }

    #[cfg(not(unix))]
    {
        let mut handle = file.try_clone()?;
        handle.seek(SeekFrom::Start(offset))?;
        handle.read_exact(buffer)?;
        Ok(())
    }
}

#[cfg(target_os = "linux")]
fn discard_file_range(file: &File, offset: u64, length: u64) {
    if length == 0 {
        return;
    }
    let _ = unsafe {
        libc::posix_fadvise(
            file.as_raw_fd(),
            offset as libc::off_t,
            length as libc::off_t,
            libc::POSIX_FADV_DONTNEED,
        )
    };
}

#[cfg(not(target_os = "linux"))]
fn discard_file_range(_file: &File, _offset: u64, _length: u64) {}

impl KformatInput {
    fn open(layout: &KformatLayout) -> Result<Self> {
        let bkmer = File::open(&layout.bkmer_path)
            .with_context(|| format!("failed to open bkmer: {}", layout.bkmer_path.display()))?;
        let bsite = File::open(&layout.bsite_path)
            .with_context(|| format!("failed to open bsite: {}", layout.bsite_path.display()))?;
        let mut bkmer_header = [0u8; BKMER_HEADER_SIZE];
        read_exact_at(&bkmer, &mut bkmer_header, 0).with_context(|| {
            format!(
                "failed to read bkmer header: {}",
                layout.bkmer_path.display()
            )
        })?;
        let bkmer_header =
            parse_bkmer_header(&bkmer_header, "kformat bkmer").map_err(anyhow::Error::msg)?;
        if bkmer_header.k != layout.meta.k || bkmer_header.n_kmers != layout.meta.n_kmers {
            bail!("bkmer header does not match kfile metadata");
        }
        let expected_bkmer_len = (BKMER_HEADER_SIZE as u64)
            .checked_add(
                layout
                    .meta
                    .n_kmers
                    .checked_mul(8)
                    .ok_or_else(|| anyhow::anyhow!("bkmer payload length overflow"))?,
            )
            .ok_or_else(|| anyhow::anyhow!("bkmer file length overflow"))?;
        let actual_bkmer_len = bkmer
            .metadata()
            .with_context(|| format!("failed to stat bkmer: {}", layout.bkmer_path.display()))?
            .len();
        if actual_bkmer_len != expected_bkmer_len {
            bail!(
                "bkmer payload length mismatch: expected {}, got {}",
                expected_bkmer_len,
                actual_bkmer_len
            );
        }

        let mut bsite_header = [0u8; BSITE_HEADER_SIZE];
        read_exact_at(&bsite, &mut bsite_header, 0).with_context(|| {
            format!(
                "failed to read bsite header: {}",
                layout.bsite_path.display()
            )
        })?;
        let bsite_header =
            parse_bsite_header(&bsite_header, "kformat bsite").map_err(anyhow::Error::msg)?;
        if bsite_header.n_samples != layout.meta.n_samples
            || bsite_header.n_kmers != layout.meta.n_kmers
            || bsite_header.bytes_per_col != layout.meta.bytes_per_col
        {
            bail!("bsite header does not match kfile metadata");
        }
        let expected_bsite_len = (BSITE_HEADER_SIZE as u64)
            .checked_add(
                layout
                    .meta
                    .n_kmers
                    .checked_mul(layout.meta.bytes_per_col)
                    .ok_or_else(|| anyhow::anyhow!("bsite payload length overflow"))?,
            )
            .ok_or_else(|| anyhow::anyhow!("bsite file length overflow"))?;
        let actual_bsite_len = bsite
            .metadata()
            .with_context(|| format!("failed to stat bsite: {}", layout.bsite_path.display()))?
            .len();
        if actual_bsite_len != expected_bsite_len {
            bail!(
                "bsite payload length mismatch: expected {}, got {}",
                expected_bsite_len,
                actual_bsite_len
            );
        }

        Ok(Self {
            bkmer,
            bsite,
            n_kmers: usize::try_from(layout.meta.n_kmers)
                .map_err(|_| anyhow::anyhow!("k-mer count does not fit platform usize"))?,
            n_samples: usize::try_from(layout.meta.n_samples)
                .map_err(|_| anyhow::anyhow!("sample count does not fit platform usize"))?,
            bytes_per_col: usize::try_from(layout.meta.bytes_per_col)
                .map_err(|_| anyhow::anyhow!("bytes_per_col does not fit platform usize"))?,
        })
    }

    fn read_block(&self, block_idx: usize, start: usize, end: usize) -> Result<InputBlock> {
        if start > end || end > self.n_kmers {
            bail!(
                "invalid kformat block range: block={} start={} end={} n_kmers={}",
                block_idx,
                start,
                end,
                self.n_kmers
            );
        }
        let rows = end - start;
        let bkmer_len = rows
            .checked_mul(8)
            .ok_or_else(|| anyhow::anyhow!("bkmer block length overflow"))?;
        let bsite_len = rows
            .checked_mul(self.bytes_per_col)
            .ok_or_else(|| anyhow::anyhow!("bsite block length overflow"))?;
        let start_u64 =
            u64::try_from(start).map_err(|_| anyhow::anyhow!("block start does not fit u64"))?;
        let bkmer_offset = (BKMER_HEADER_SIZE as u64)
            .checked_add(
                start_u64
                    .checked_mul(8)
                    .ok_or_else(|| anyhow::anyhow!("bkmer block offset overflow"))?,
            )
            .ok_or_else(|| anyhow::anyhow!("bkmer block offset overflow"))?;
        let bsite_offset = (BSITE_HEADER_SIZE as u64)
            .checked_add(
                start_u64
                    .checked_mul(self.bytes_per_col as u64)
                    .ok_or_else(|| anyhow::anyhow!("bsite block offset overflow"))?,
            )
            .ok_or_else(|| anyhow::anyhow!("bsite block offset overflow"))?;
        let mut bkmer = vec![0u8; bkmer_len];
        let mut bsite = vec![0u8; bsite_len];
        read_exact_at(&self.bkmer, &mut bkmer, bkmer_offset)
            .with_context(|| format!("failed to read bkmer block {block_idx}"))?;
        read_exact_at(&self.bsite, &mut bsite, bsite_offset)
            .with_context(|| format!("failed to read bsite block {block_idx}"))?;
        Ok(InputBlock {
            block_idx,
            start,
            bkmer,
            bsite,
            selected: Vec::new(),
        })
    }
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
        bail!("unsupported kfile format in {}", meta_path.display());
    }
    if meta.n_samples == 0 || meta.n_kmers == 0 {
        bail!(
            "kfile metadata must contain samples and k-mers: {}",
            meta_path.display()
        );
    }
    if meta.k == 0 || meta.k > 31 {
        bail!("kfile k must be within 1..=31: {}", meta.k);
    }
    if meta.matrix_layout.trim() != "column_major_bitset"
        || meta.bit_order.trim() != "little_bit_order"
        || meta.compression.trim() != "none"
    {
        bail!(
            "unsupported kfile bitmatrix layout in {}",
            meta_path.display()
        );
    }
    let expected_bytes = meta
        .n_samples
        .checked_add(7)
        .ok_or_else(|| anyhow::anyhow!("kfile bytes_per_col overflow"))?
        / 8;
    if meta.bytes_per_col != expected_bytes {
        bail!(
            "kfile bytes_per_col mismatch: metadata={} expected={}",
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
    let headers = reader.headers()?.clone();
    if headers.len() < 3
        || headers.get(0) != Some("#idx")
        || headers.get(1) != Some("sample_id")
        || headers.get(2) != Some("kmc_prefix")
    {
        bail!("invalid idv header in {}", path.display());
    }
    for rec in reader.records() {
        let rec = rec?;
        let index = rec
            .get(0)
            .ok_or_else(|| anyhow::anyhow!("missing idv index"))?
            .parse::<u32>()?;
        let sample_id = rec
            .get(1)
            .ok_or_else(|| anyhow::anyhow!("missing sample_id"))?
            .trim()
            .to_string();
        if sample_id.is_empty() {
            bail!("empty sample_id in {}", path.display());
        }
        rows.push(SampleEntry {
            index,
            sample_id,
            kmc_prefix: rec.get(2).unwrap_or_default().to_string(),
        });
    }
    rows.sort_by_key(|row| row.index);
    let mut seen = HashSet::with_capacity(rows.len());
    for (expected, row) in rows.iter().enumerate() {
        if row.index as usize != expected || !seen.insert(row.sample_id.clone()) {
            bail!(
                "idv indices must be contiguous and sample IDs unique: {}",
                path.display()
            );
        }
    }
    Ok(rows)
}

fn load_layout(prefix_or_meta: &Path) -> Result<KformatLayout> {
    let meta_path = resolve_meta_path(prefix_or_meta);
    let meta: KmergeMeta = serde_json::from_reader(
        File::open(&meta_path)
            .with_context(|| format!("failed to open meta: {}", meta_path.display()))?,
    )?;
    validate_meta(&meta, &meta_path)?;
    let base = meta_path.parent().unwrap_or_else(|| Path::new("."));
    let bkmer_path = base.join(&meta.bkmer_file);
    let bsite_path = base.join(&meta.bsite_file);
    let idv_path = base.join(&meta.idv_file);
    for path in [&bkmer_path, &bsite_path, &idv_path] {
        if !path.is_file() {
            bail!("required kfile component not found: {}", path.display());
        }
    }

    let samples = read_idv_file(&idv_path)?;
    if samples.len() as u64 != meta.n_samples {
        bail!("idv sample count does not match kfile metadata");
    }
    Ok(KformatLayout {
        bkmer_path,
        bsite_path,
        meta,
        samples,
    })
}

#[inline]
fn valid_sample_mask(n_samples: usize) -> u8 {
    let rem = n_samples & 7;
    if rem == 0 {
        0xff
    } else {
        (1u16 << rem) as u8 - 1
    }
}

#[inline]
pub(crate) fn maf_from_bitset(bytes: &[u8], n_samples: usize) -> f64 {
    if n_samples == 0 {
        return f64::NAN;
    }
    let mut count = bytes
        .chunks_exact(8)
        .map(|chunk| {
            u64::from_le_bytes(chunk.try_into().expect("popcount chunk is 8 bytes")).count_ones()
                as usize
        })
        .sum::<usize>();
    count += bytes
        .chunks_exact(8)
        .remainder()
        .iter()
        .map(|byte| byte.count_ones() as usize)
        .sum::<usize>();
    if let Some(last) = bytes.last() {
        let invalid = last & !valid_sample_mask(n_samples);
        count = count.saturating_sub(invalid.count_ones() as usize);
    }
    let p = count as f64 / n_samples as f64;
    p.min(1.0 - p)
}

fn normalize_extract_codes(values: &[String], k: u32, canonical: bool) -> Result<HashSet<u64>> {
    let mut out = HashSet::with_capacity(values.len());
    for value in values {
        let seq = value.trim();
        if seq.len() != k as usize {
            bail!(
                "extract k-mer has length {}, expected {}: {}",
                seq.len(),
                k,
                seq
            );
        }
        let code = encode_kmer_u64(seq)?;
        out.insert(if canonical {
            canonical_code(code, k)
        } else {
            code
        });
    }
    Ok(out)
}

#[inline]
fn bkmer_code_at(body: &[u8], row: usize) -> u64 {
    let start = row * 8;
    u64::from_le_bytes(
        body[start..start + 8]
            .try_into()
            .expect("bkmer row is 8 bytes"),
    )
}

#[inline]
fn row_is_selected(
    bkmer_body: &[u8],
    bsite_body: &[u8],
    row: usize,
    n_samples: usize,
    bytes_per_col: usize,
    maf_threshold: f64,
    extract: Option<&HashSet<u64>>,
) -> bool {
    let code = bkmer_code_at(bkmer_body, row);
    if extract.is_some_and(|set| !set.contains(&code)) {
        return false;
    }
    let col = &bsite_body[row * bytes_per_col..(row + 1) * bytes_per_col];
    maf_from_bitset(col, n_samples) + f64::EPSILON >= maf_threshold
}

fn selected_rows_in_block(
    bkmer_body: &[u8],
    bsite_body: &[u8],
    n_samples: usize,
    bytes_per_col: usize,
    maf_threshold: f64,
    extract: Option<&HashSet<u64>>,
) -> Vec<u32> {
    let mut rows = Vec::new();
    for row in 0..bkmer_body.len() / 8 {
        if row_is_selected(
            bkmer_body,
            bsite_body,
            row,
            n_samples,
            bytes_per_col,
            maf_threshold,
            extract,
        ) {
            rows.push(row as u32);
        }
    }
    rows
}

/// Scan selected rows with bounded parallel workers and visit them in input order.
///
/// Workers retain only a block-local `u32` offset list.  The ordered writer keeps
/// at most a small number of completed blocks in the `BTreeMap`, so memory does
/// not grow with the number of selected k-mers.
fn stream_selected_rows<F>(
    input: &KformatInput,
    maf_threshold: f64,
    extract: Option<&HashSet<u64>>,
    threads: usize,
    mut visit: F,
) -> Result<u64>
where
    F: FnMut(usize, u64, &[u8], &[u8]) -> Result<()>,
{
    if input.n_samples == 0 {
        bail!("kformat requires at least one sample");
    }
    if !(0.0..=0.5).contains(&maf_threshold) {
        bail!("maf must be within [0, 0.5]");
    }
    if input.n_kmers == 0 {
        return Ok(0);
    }

    let block = DEFAULT_SCAN_BLOCK_ROWS;
    let n_blocks = input.n_kmers.div_ceil(block);
    let available_workers = thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1);
    let worker_count = threads.max(1).min(n_blocks).min(available_workers);

    if worker_count == 1 {
        let mut count = 0u64;
        for block_idx in 0..n_blocks {
            let start = block_idx * block;
            let end = (start + block).min(input.n_kmers);
            let mut input_block = input.read_block(block_idx, start, end)?;
            input_block.selected = selected_rows_in_block(
                &input_block.bkmer,
                &input_block.bsite,
                input.n_samples,
                input.bytes_per_col,
                maf_threshold,
                extract,
            );
            for local_row in input_block.selected.iter().copied() {
                let local_row = local_row as usize;
                let kmer_bytes = &input_block.bkmer[local_row * 8..(local_row + 1) * 8];
                let code = bkmer_code_at(&input_block.bkmer, local_row);
                let col = &input_block.bsite
                    [local_row * input.bytes_per_col..(local_row + 1) * input.bytes_per_col];
                visit(input_block.start + local_row, code, kmer_bytes, col)?;
                count += 1;
            }
            discard_file_range(
                &input.bkmer,
                BKMER_HEADER_SIZE as u64 + (start as u64) * 8,
                input_block.bkmer.len() as u64,
            );
            discard_file_range(
                &input.bsite,
                BSITE_HEADER_SIZE as u64 + (start as u64) * input.bytes_per_col as u64,
                input_block.bsite.len() as u64,
            );
        }
        return Ok(count);
    }

    let channel_capacity = worker_count.saturating_mul(2).max(1);
    let (job_tx, job_rx) = sync_channel::<(usize, usize, usize)>(channel_capacity);
    let (result_tx, result_rx) = sync_channel::<Result<InputBlock>>(channel_capacity);
    let shared_job_rx = Arc::new(Mutex::new(job_rx));

    thread::scope(|scope| -> Result<u64> {
        for _ in 0..worker_count {
            let worker_job_rx = Arc::clone(&shared_job_rx);
            let worker_result_tx = result_tx.clone();
            scope.spawn(move || loop {
                let job = match worker_job_rx.lock() {
                    Ok(receiver) => receiver.recv(),
                    Err(_) => return,
                };
                let Ok((block_idx, start, end)) = job else {
                    return;
                };
                let result = input
                    .read_block(block_idx, start, end)
                    .and_then(|mut block| {
                        block.selected = selected_rows_in_block(
                            &block.bkmer,
                            &block.bsite,
                            input.n_samples,
                            input.bytes_per_col,
                            maf_threshold,
                            extract,
                        );
                        Ok(block)
                    });
                if worker_result_tx.send(result).is_err() {
                    return;
                }
            });
        }
        drop(result_tx);

        let mut pending: BTreeMap<usize, InputBlock> = BTreeMap::new();
        let mut next_block = 0usize;
        let mut next_to_dispatch = 0usize;
        let mut count = 0u64;

        // Keep only a bounded number of blocks in flight.  This prevents a
        // slow early block from allowing all later results to accumulate in
        // `pending` while preserving deterministic input order.
        let window = channel_capacity;
        while next_to_dispatch < n_blocks && next_to_dispatch - next_block < window {
            let start = next_to_dispatch * block;
            let end = (start + block).min(input.n_kmers);
            job_tx
                .send((next_to_dispatch, start, end))
                .map_err(|_| anyhow::anyhow!("kformat worker pipeline stopped early"))?;
            next_to_dispatch += 1;
        }

        for _ in 0..n_blocks {
            let result_block = result_rx
                .recv()
                .map_err(|_| anyhow::anyhow!("kformat worker pipeline stopped early"))?;
            let input_block = result_block?;
            pending.insert(input_block.block_idx, input_block);
            while let Some(input_block) = pending.remove(&next_block) {
                for local_row in input_block.selected.iter().copied() {
                    let local_row = local_row as usize;
                    let kmer_bytes = &input_block.bkmer[local_row * 8..(local_row + 1) * 8];
                    let code = bkmer_code_at(&input_block.bkmer, local_row);
                    let col = &input_block.bsite
                        [local_row * input.bytes_per_col..(local_row + 1) * input.bytes_per_col];
                    visit(input_block.start + local_row, code, kmer_bytes, col)?;
                    count += 1;
                }
                discard_file_range(
                    &input.bkmer,
                    BKMER_HEADER_SIZE as u64 + (input_block.start as u64) * 8,
                    input_block.bkmer.len() as u64,
                );
                discard_file_range(
                    &input.bsite,
                    BSITE_HEADER_SIZE as u64
                        + (input_block.start as u64) * input.bytes_per_col as u64,
                    input_block.bsite.len() as u64,
                );
                next_block += 1;
            }

            while next_to_dispatch < n_blocks && next_to_dispatch - next_block < window {
                let start = next_to_dispatch * block;
                let end = (start + block).min(input.n_kmers);
                job_tx
                    .send((next_to_dispatch, start, end))
                    .map_err(|_| anyhow::anyhow!("kformat worker pipeline stopped early"))?;
                next_to_dispatch += 1;
            }
        }
        drop(job_tx);
        if next_block != n_blocks || !pending.is_empty() {
            bail!("kformat worker pipeline did not preserve all input blocks");
        }
        Ok(count)
    })
}

#[cfg(test)]
#[inline]
pub(crate) fn plink_row_from_presence(presence: &[bool]) -> Vec<u8> {
    let mut row = vec![0u8; presence.len().div_ceil(4)];
    for (idx, present) in presence.iter().copied().enumerate() {
        let code = if present { 0b11 } else { 0b00 };
        row[idx >> 2] |= code << ((idx & 3) * 2);
    }
    row
}

fn plink_row_from_bitset(col: &[u8], n_samples: usize) -> Vec<u8> {
    const NIBBLE_TO_PLINK: [u8; 16] = [
        0, 3, 12, 15, 48, 51, 60, 63, 192, 195, 204, 207, 240, 243, 252, 255,
    ];
    let mut row = vec![0u8; n_samples.div_ceil(4)];
    let valid_mask = valid_sample_mask(n_samples);
    for (src_idx, &raw) in col.iter().enumerate() {
        let byte = if src_idx + 1 == col.len() {
            raw & valid_mask
        } else {
            raw
        };
        let dst = src_idx * 2;
        if dst < row.len() {
            row[dst] = NIBBLE_TO_PLINK[(byte & 0x0f) as usize];
        }
        if dst + 1 < row.len() {
            row[dst + 1] = NIBBLE_TO_PLINK[(byte >> 4) as usize];
        }
    }
    row
}

fn write_bfile(
    out_prefix: &Path,
    layout: &KformatLayout,
    input: &KformatInput,
    maf_threshold: f64,
    extract: Option<&HashSet<u64>>,
    threads: usize,
) -> Result<u64> {
    let out_prefix = out_prefix.to_string_lossy();
    let mut fam =
        BufWriter::with_capacity(4 * 1024 * 1024, File::create(format!("{out_prefix}.fam"))?);
    for sample in &layout.samples {
        writeln!(
            fam,
            "{}\t{}\t0\t0\t1\t-9",
            sample.sample_id, sample.sample_id
        )?;
    }
    fam.flush()?;
    let mut bim =
        BufWriter::with_capacity(4 * 1024 * 1024, File::create(format!("{out_prefix}.bim"))?);
    let mut bed =
        BufWriter::with_capacity(8 * 1024 * 1024, File::create(format!("{out_prefix}.bed"))?);
    bed.write_all(&[0x6c, 0x1b, 0x01])?;
    let mut out_idx = 0u64;
    let written = stream_selected_rows(
        input,
        maf_threshold,
        extract,
        threads,
        |_row, code, _kmer_bytes, col| {
            let kmer = decode_kmer_u64(code, layout.meta.k);
            writeln!(bim, "0\t{kmer}\t0\t{}\tA\tG", out_idx + 1)?;
            bed.write_all(&plink_row_from_bitset(col, layout.samples.len()))?;
            out_idx += 1;
            Ok(())
        },
    )?;
    bim.flush()?;
    bed.flush()?;
    Ok(written)
}

fn write_kfile(
    out_prefix: &Path,
    output_name: &str,
    layout: &KformatLayout,
    input: &KformatInput,
    maf_threshold: f64,
    extract: Option<&HashSet<u64>>,
    threads: usize,
) -> Result<u64> {
    let prefix = out_prefix.to_string_lossy();
    let idv_path = PathBuf::from(format!("{prefix}.idv"));
    let bkmer_path = PathBuf::from(format!("{prefix}.bkmer"));
    let bsite_path = PathBuf::from(format!("{prefix}.bsite"));
    let meta_path = PathBuf::from(format!("{prefix}.meta.json"));
    write_idv_file(&idv_path, &layout.samples)?;

    let mut bkmer_writer = BufWriter::with_capacity(8 * 1024 * 1024, File::create(&bkmer_path)?);
    BkmerHeader {
        k: layout.meta.k,
        n_kmers: 0,
        canonical: if layout.meta.canonical { 1 } else { 0 },
    }
    .write_to(&mut bkmer_writer)?;

    let mut bsite_writer = BufWriter::with_capacity(8 * 1024 * 1024, File::create(&bsite_path)?);
    BsiteHeader {
        n_samples: layout.meta.n_samples,
        n_kmers: 0,
        bytes_per_col: layout.meta.bytes_per_col,
    }
    .write_to(&mut bsite_writer)?;
    let written = stream_selected_rows(
        input,
        maf_threshold,
        extract,
        threads,
        |_row, _code, kmer_bytes, col| {
            bkmer_writer.write_all(kmer_bytes)?;
            bsite_writer.write_all(col)?;
            Ok(())
        },
    )?;
    bkmer_writer.flush()?;
    bsite_writer.flush()?;

    let mut bkmer_file = bkmer_writer
        .into_inner()
        .map_err(|err| anyhow::anyhow!("failed to finalize bkmer: {err}"))?;
    bkmer_file.seek(SeekFrom::Start(0))?;
    BkmerHeader {
        k: layout.meta.k,
        n_kmers: written,
        canonical: if layout.meta.canonical { 1 } else { 0 },
    }
    .write_to(&mut bkmer_file)?;
    bkmer_file.flush()?;

    let mut bsite_file = bsite_writer
        .into_inner()
        .map_err(|err| anyhow::anyhow!("failed to finalize bsite: {err}"))?;
    bsite_file.seek(SeekFrom::Start(0))?;
    BsiteHeader {
        n_samples: layout.meta.n_samples,
        n_kmers: written,
        bytes_per_col: layout.meta.bytes_per_col,
    }
    .write_to(&mut bsite_file)?;
    bsite_file.flush()?;

    let mut meta = layout.meta.clone();
    meta.n_kmers = written;
    meta.bkmer_file = format!("{output_name}.bkmer");
    meta.bsite_file = format!("{output_name}.bsite");
    meta.idv_file = format!("{output_name}.idv");
    meta.min_presence_rate = 0.0;
    meta.max_presence_rate = 1.0;
    meta.encoding = ENCODING_NAME.to_string();
    write_meta_json(&meta_path, &meta)?;
    Ok(written)
}

fn append_suffix(prefix: &Path, suffix: &str) -> PathBuf {
    let mut raw = prefix.as_os_str().to_os_string();
    raw.push(suffix);
    PathBuf::from(raw)
}

fn output_exists(prefix: &Path, format: &str) -> bool {
    let suffixes = if format == "bfile" {
        ["bed", "bim", "fam"]
    } else {
        ["bkmer", "bsite", "idv"]
    };
    suffixes
        .iter()
        .any(|suffix| append_suffix(prefix, &format!(".{suffix}")).exists())
        || (format == "kfile" && append_suffix(prefix, ".meta.json").exists())
}

fn make_temp_output_prefix(output: &Path) -> Result<(PathBuf, PathBuf)> {
    let parent = output.parent().unwrap_or_else(|| Path::new("."));
    let name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| anyhow::anyhow!("output prefix must have a valid file name"))?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| anyhow::anyhow!("system clock error: {e}"))?
        .as_nanos();
    for attempt in 0..100u32 {
        let dir = parent.join(format!(
            ".{name}.kformat.tmp.{}.{}.{}",
            std::process::id(),
            stamp,
            attempt
        ));
        match fs::create_dir(&dir) {
            Ok(()) => return Ok((dir.clone(), dir.join(name))),
            Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(err) => return Err(err.into()),
        }
    }
    bail!(
        "could not create a unique kformat temporary directory beside {}",
        output.display()
    )
}

fn commit_output_files(temp_prefix: &Path, output: &Path, format: &str) -> Result<()> {
    let suffixes: &[&str] = if format == "bfile" {
        &[".fam", ".bim", ".bed"]
    } else {
        &[".idv", ".bkmer", ".bsite", ".meta.json"]
    };
    for suffix in suffixes {
        fs::rename(
            append_suffix(temp_prefix, suffix),
            append_suffix(output, suffix),
        )?;
    }
    Ok(())
}

#[pyfunction(
    name = "kformat_run",
    signature = (input, out, format="bfile", maf=0.0, extract_kmers=None, thread=1, force=false)
)]
pub fn kformat_run_py(
    py: Python<'_>,
    input: String,
    out: String,
    format: &str,
    maf: f64,
    extract_kmers: Option<Vec<String>>,
    thread: usize,
    force: bool,
) -> PyResult<Py<PyDict>> {
    let result = py
        .detach(|| {
            run_kformat(
                &input,
                &out,
                format,
                maf,
                extract_kmers.as_deref(),
                thread,
                force,
            )
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
    let dict = PyDict::new(py);
    dict.set_item("input", result.input)?;
    dict.set_item("output", result.output)?;
    dict.set_item("format", result.format)?;
    dict.set_item("n_samples", result.n_samples)?;
    dict.set_item("n_kmers_in", result.n_kmers_in)?;
    dict.set_item("n_kmers_out", result.n_kmers_out)?;
    dict.set_item("maf", result.maf)?;
    Ok(dict.unbind())
}

#[derive(Debug)]
struct KformatSummary {
    input: String,
    output: String,
    format: String,
    n_samples: u64,
    n_kmers_in: u64,
    n_kmers_out: u64,
    maf: f64,
}

fn run_kformat(
    input: &str,
    out: &str,
    format: &str,
    maf: f64,
    extract_kmers: Option<&[String]>,
    thread: usize,
    force: bool,
) -> Result<KformatSummary> {
    let format = format.trim().to_ascii_lowercase();
    if format != "bfile" && format != "kfile" {
        bail!("unsupported kformat output format: {format}");
    }
    if !(0.0..=0.5).contains(&maf) {
        bail!("maf must be within [0, 0.5]");
    }
    let input_path = Path::new(input);
    let layout = load_layout(input_path)?;
    let output = PathBuf::from(out);
    if output.as_os_str().is_empty() {
        bail!("output prefix is empty");
    }
    if output_exists(&output, &format) && !force {
        bail!("output already exists: {} (use --force)", output.display());
    }
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    let input_reader = KformatInput::open(&layout)?;
    let output_name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| anyhow::anyhow!("output prefix must have a valid file name"))?
        .to_string();
    let (temp_dir, temp_prefix) = make_temp_output_prefix(&output)?;
    let extract = extract_kmers
        .map(|items| normalize_extract_codes(items, layout.meta.k, layout.meta.canonical))
        .transpose()?;

    let n_kmers_out = if format == "bfile" {
        write_bfile(
            &temp_prefix,
            &layout,
            &input_reader,
            maf,
            extract.as_ref(),
            thread,
        )
    } else {
        write_kfile(
            &temp_prefix,
            &output_name,
            &layout,
            &input_reader,
            maf,
            extract.as_ref(),
            thread,
        )
    };
    let n_kmers_out = match n_kmers_out {
        Ok(count) if count > 0 => count,
        Ok(_) => {
            let _ = fs::remove_dir_all(&temp_dir);
            bail!("no k-mers survived kformat filters");
        }
        Err(err) => {
            let _ = fs::remove_dir_all(&temp_dir);
            return Err(err);
        }
    };
    if let Err(err) = commit_output_files(&temp_prefix, &output, &format) {
        let _ = fs::remove_dir_all(&temp_dir);
        return Err(err.into());
    }
    fs::remove_dir_all(&temp_dir)?;
    Ok(KformatSummary {
        input: input.to_string(),
        output: out.to_string(),
        format: format.to_string(),
        n_samples: layout.meta.n_samples,
        n_kmers_in: layout.meta.n_kmers,
        n_kmers_out,
        maf,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::fs::{self, File, OpenOptions};
    use std::io::Write;
    use std::path::PathBuf;

    use super::{
        maf_from_bitset, normalize_extract_codes, plink_row_from_presence, stream_selected_rows,
        write_kfile, KformatInput, KformatLayout, DEFAULT_SCAN_BLOCK_ROWS,
    };
    use crate::kmer::encode::{canonical_code, encode_kmer_u64};
    use crate::kmer::format::{BkmerHeader, BsiteHeader, KmergeMeta, SampleEntry};
    use crate::kmer::writer::{write_idv_file, write_meta_json};

    fn test_fixture(
        n_kmers: usize,
        n_samples: usize,
    ) -> (PathBuf, KformatLayout, Vec<u8>, Vec<u8>) {
        let dir = std::env::temp_dir().join(format!(
            "janusx_kformat_windowed_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("t")
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp directory");
        let bytes_per_col = n_samples.div_ceil(8);
        let mut bkmer_body = Vec::with_capacity(n_kmers * 8);
        for code in 0u64..n_kmers as u64 {
            bkmer_body.extend_from_slice(&code.to_le_bytes());
        }
        let bsite_body = vec![1u8; n_kmers * bytes_per_col];
        let bkmer_path = dir.join("input.bkmer");
        let bsite_path = dir.join("input.bsite");
        let mut bkmer = File::create(&bkmer_path).expect("create bkmer");
        BkmerHeader {
            k: 4,
            n_kmers: n_kmers as u64,
            canonical: 1,
        }
        .write_to(&mut bkmer)
        .expect("write bkmer header");
        bkmer.write_all(&bkmer_body).expect("write bkmer body");
        let mut bsite = File::create(&bsite_path).expect("create bsite");
        BsiteHeader {
            n_samples: n_samples as u64,
            n_kmers: n_kmers as u64,
            bytes_per_col: bytes_per_col as u64,
        }
        .write_to(&mut bsite)
        .expect("write bsite header");
        bsite.write_all(&bsite_body).expect("write bsite body");
        let samples = (0..n_samples)
            .map(|idx| SampleEntry {
                index: idx as u32,
                sample_id: format!("S{}", idx + 1),
                kmc_prefix: String::new(),
            })
            .collect();
        let layout = KformatLayout {
            bkmer_path,
            bsite_path,
            meta: KmergeMeta {
                format: "janusx-kmer-bitmatrix-v1".to_string(),
                k: 4,
                n_samples: n_samples as u64,
                n_kmers: n_kmers as u64,
                bytes_per_col: bytes_per_col as u64,
                encoding: "ACGT_2BIT_A00_C01_G10_T11".to_string(),
                canonical: true,
                matrix_layout: "column_major_bitset".to_string(),
                value_type: "binary_presence".to_string(),
                bit_order: "little_bit_order".to_string(),
                bkmer_file: "input.bkmer".to_string(),
                bsite_file: "input.bsite".to_string(),
                idv_file: "input.idv".to_string(),
                min_count: 1,
                min_presence_rate: 0.0,
                max_presence_rate: 1.0,
                bucket_bits: 0,
                compression: "none".to_string(),
            },
            samples,
        };
        (dir, layout, bkmer_body, bsite_body)
    }

    #[test]
    fn maf_uses_valid_sample_bits_only() {
        assert!((maf_from_bitset(&[0b0000_1011], 5) - 0.4).abs() < 1e-12);
        assert_eq!(maf_from_bitset(&[0b1111_1111], 5), 0.0);
    }

    #[test]
    fn extract_list_is_encoded_and_canonicalized() {
        let codes = normalize_extract_codes(&["ACGT".to_string()], 4, true).expect("codes");
        assert!(codes.contains(&canonical_code(encode_kmer_u64("ACGT").unwrap(), 4)));
    }

    #[test]
    fn packed_selection_keeps_source_order() {
        let (dir, layout, mut bkmer_body, _bsite_body) = test_fixture(4, 3);
        let codes = [11u64, 22, 33, 44];
        bkmer_body.clear();
        for code in codes {
            bkmer_body.extend_from_slice(&code.to_le_bytes());
        }
        let mut rewritten = File::create(&layout.bkmer_path).expect("rewrite bkmer");
        BkmerHeader {
            k: 4,
            n_kmers: 4,
            canonical: 1,
        }
        .write_to(&mut rewritten)
        .expect("write bkmer header");
        rewritten.write_all(&bkmer_body).expect("write bkmer body");
        let mut extract = HashSet::new();
        extract.insert(44u64);
        let mut keep = Vec::new();
        let input = KformatInput::open(&layout).expect("open input");
        let written = stream_selected_rows(
            &input,
            0.0,
            Some(&extract),
            2,
            |row, code, _kmer_bytes, _presence| {
                assert_eq!(code, 44);
                keep.push(row);
                Ok(())
            },
        )
        .expect("select");
        assert_eq!(written, 1);
        assert_eq!(keep, vec![3]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn streaming_selection_counts_rows_without_materializing_indices() {
        let (dir, layout, _bkmer_body, _bsite_body) = test_fixture(4, 3);
        let input = KformatInput::open(&layout).expect("open input");
        let mut count = 0usize;
        let written = stream_selected_rows(&input, 0.0, None, 2, |_, _, _, _| {
            count += 1;
            Ok(())
        })
        .expect("count selected rows");
        assert_eq!(written, count as u64);
        assert_eq!(count, 4);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn streaming_selection_preserves_order_across_parallel_blocks() {
        let n_kmers = DEFAULT_SCAN_BLOCK_ROWS * 3 + 17;
        let (dir, layout, _bkmer_body, _bsite_body) = test_fixture(n_kmers, 8);
        let input = KformatInput::open(&layout).expect("open input");
        let mut keep = Vec::with_capacity(n_kmers);
        let written = stream_selected_rows(&input, 0.0, None, 3, |row, _, _, _| {
            keep.push(row);
            Ok(())
        })
        .expect("stream parallel blocks");

        assert_eq!(written, n_kmers as u64);
        assert_eq!(keep.len(), n_kmers);
        assert!(keep.iter().copied().eq(0..n_kmers));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn window_reader_reads_middle_block() {
        let n_kmers = DEFAULT_SCAN_BLOCK_ROWS * 2 + 7;
        let (dir, layout, bkmer_body, bsite_body) = test_fixture(n_kmers, 8);
        let input = KformatInput::open(&layout).expect("open input");
        let start = DEFAULT_SCAN_BLOCK_ROWS;
        let end = DEFAULT_SCAN_BLOCK_ROWS * 2;
        let block = input.read_block(1, start, end).expect("read middle block");
        assert_eq!(block.block_idx, 1);
        assert_eq!(&block.bkmer[..8], &bkmer_body[start * 8..start * 8 + 8]);
        assert_eq!(&block.bsite[..8], &bsite_body[start..start + 8]);
        assert_eq!(block.bkmer.len(), (end - start) * 8);
        assert_eq!(block.bsite.len(), end - start);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn window_reader_rejects_truncated_payload() {
        let (dir, layout, _bkmer_body, bsite_body) = test_fixture(32, 8);
        let truncated_len = (super::BSITE_HEADER_SIZE + bsite_body.len() - 1) as u64;
        OpenOptions::new()
            .write(true)
            .open(&layout.bsite_path)
            .expect("open bsite")
            .set_len(truncated_len)
            .expect("truncate bsite");
        let error = match KformatInput::open(&layout) {
            Ok(_) => panic!("truncated bsite must fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("payload length"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn kfile_writer_patches_streamed_row_count_and_preserves_payload_order() {
        let (dir, layout, mut bkmer_body, _fixture_bsite_body) = test_fixture(4, 3);
        let codes = [11u64, 22, 33, 44];
        bkmer_body.clear();
        for code in codes {
            bkmer_body.extend_from_slice(&code.to_le_bytes());
        }
        let bsite_body = vec![0b0000_0001, 0b0000_0011, 0b0000_0000, 0b0000_0111];
        let mut bkmer = File::create(&layout.bkmer_path).expect("rewrite bkmer");
        BkmerHeader {
            k: 4,
            n_kmers: 4,
            canonical: 1,
        }
        .write_to(&mut bkmer)
        .expect("write bkmer header");
        bkmer.write_all(&bkmer_body).expect("write bkmer body");
        let mut bsite = File::create(&layout.bsite_path).expect("rewrite bsite");
        BsiteHeader {
            n_samples: 3,
            n_kmers: 4,
            bytes_per_col: 1,
        }
        .write_to(&mut bsite)
        .expect("write bsite header");
        bsite.write_all(&bsite_body).expect("write bsite body");
        let input = KformatInput::open(&layout).expect("open input");

        let written = write_kfile(&dir.join("out"), "out", &layout, &input, 0.25, None, 2)
            .expect("write streamed kfile");
        assert_eq!(written, 2);

        let bkmer = fs::read(dir.join("out.bkmer")).expect("read bkmer");
        assert_eq!(u64::from_le_bytes(bkmer[16..24].try_into().unwrap()), 2);
        assert_eq!(&bkmer[64..80], &bkmer_body[0..16]);
        let bsite = fs::read(dir.join("out.bsite")).expect("read bsite");
        assert_eq!(u64::from_le_bytes(bsite[24..32].try_into().unwrap()), 2);
        assert_eq!(&bsite[80..82], &bsite_body[0..2]);
        let meta: KmergeMeta =
            serde_json::from_slice(&fs::read(dir.join("out.meta.json")).expect("read meta"))
                .expect("parse meta");
        assert_eq!(meta.n_kmers, 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_kformat_writes_bfile_from_windowed_input() {
        let (dir, layout, _bkmer_body, _bsite_body) = test_fixture(4, 3);
        write_idv_file(&dir.join("input.idv"), &layout.samples).expect("write idv");
        write_meta_json(&dir.join("input.meta.json"), &layout.meta).expect("write meta");

        let output = dir.join("out");
        let summary = super::run_kformat(
            &dir.join("input.meta.json").to_string_lossy(),
            &output.to_string_lossy(),
            "bfile",
            0.25,
            None,
            3,
            false,
        )
        .expect("run windowed bfile conversion");
        assert_eq!(summary.n_kmers_in, 4);
        assert_eq!(summary.n_kmers_out, 4);
        assert_eq!(
            fs::read(output.with_extension("bed"))
                .expect("read bed")
                .len(),
            3 + 4
        );
        assert_eq!(
            fs::read_to_string(output.with_extension("bim"))
                .expect("read bim")
                .lines()
                .count(),
            4
        );
        assert_eq!(
            fs::read_to_string(output.with_extension("fam"))
                .expect("read fam")
                .lines()
                .count(),
            3
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn plink_presence_uses_zero_or_two_dosage_codes() {
        assert_eq!(
            plink_row_from_presence(&[true, false, true]),
            vec![0b11_00_11]
        );
    }

    #[test]
    fn plink_bitset_uses_packed_two_bit_presence_codes() {
        assert_eq!(
            super::plink_row_from_bitset(&[0b0000_0101], 3),
            vec![0b110011]
        );
    }
}
