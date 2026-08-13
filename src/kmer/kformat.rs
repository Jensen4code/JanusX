use crate::breader::{open_bkmer_mmap, open_bsite_mmap, BkmerMmapView, BsiteMmapView};
use crate::kmer::encode::{canonical_code, decode_kmer_u64, encode_kmer_u64, ENCODING_NAME};
use crate::kmer::format::{
    BkmerHeader, BsiteHeader, KmergeMeta, SampleEntry, BKMER_HEADER_SIZE, BSITE_HEADER_SIZE,
};
use crate::kmer::writer::{write_idv_file, write_meta_json};
use anyhow::{bail, Context, Result};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const DEFAULT_SCAN_BLOCK_ROWS: usize = 16_384;

#[derive(Clone, Debug)]
struct KformatLayout {
    bkmer_path: PathBuf,
    bsite_path: PathBuf,
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

    let bkmer = open_bkmer_mmap(&bkmer_path, "kformat bkmer").map_err(anyhow::Error::msg)?;
    if bkmer.header.k != meta.k || bkmer.header.n_kmers != meta.n_kmers {
        bail!("bkmer header does not match kfile metadata");
    }
    let bkmer_len = BKMER_HEADER_SIZE as u64 + meta.n_kmers * 8;
    if bkmer.mmap.len() as u64 != bkmer_len {
        bail!("bkmer payload length mismatch for {}", bkmer_path.display());
    }
    let bsite = open_bsite_mmap(&bsite_path, "kformat bsite").map_err(anyhow::Error::msg)?;
    if bsite.header.n_samples != meta.n_samples
        || bsite.header.n_kmers != meta.n_kmers
        || bsite.header.bytes_per_col != meta.bytes_per_col
    {
        bail!("bsite header does not match kfile metadata");
    }
    let bsite_len = BSITE_HEADER_SIZE as u64 + meta.n_kmers * meta.bytes_per_col;
    if bsite.mmap.len() as u64 != bsite_len {
        bail!("bsite payload length mismatch for {}", bsite_path.display());
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

fn select_rows(
    bkmer_body: &[u8],
    bsite_body: &[u8],
    n_kmers: usize,
    n_samples: usize,
    maf_threshold: f64,
    extract: Option<&HashSet<u64>>,
    threads: usize,
) -> Result<Vec<usize>> {
    let bytes_per_col = n_samples.div_ceil(8);
    if bkmer_body.len() != n_kmers * 8 || bsite_body.len() != n_kmers * bytes_per_col {
        bail!("bsite body and bkmer row counts do not match");
    }
    if !(0.0..=0.5).contains(&maf_threshold) {
        bail!("maf must be within [0, 0.5]");
    }
    let block = DEFAULT_SCAN_BLOCK_ROWS;
    let jobs = (0..n_kmers)
        .step_by(block)
        .map(|start| {
            let end = (start + block).min(n_kmers);
            (start, end)
        })
        .collect::<Vec<_>>();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .map_err(|e| anyhow::anyhow!("kformat rayon pool: {e}"))?;
    let selected = pool.install(|| {
        jobs.par_iter()
            .map(|&(start, end)| {
                let mut local = Vec::new();
                for row in start..end {
                    let code = bkmer_code_at(bkmer_body, row);
                    if extract.is_some_and(|set| !set.contains(&code)) {
                        continue;
                    }
                    let col = &bsite_body[row * bytes_per_col..(row + 1) * bytes_per_col];
                    if maf_from_bitset(col, n_samples) + f64::EPSILON >= maf_threshold {
                        local.push(row);
                    }
                }
                local
            })
            .collect::<Vec<_>>()
    });
    Ok(selected.into_iter().flatten().collect())
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
    bkmer: &BkmerMmapView,
    bsite: &BsiteMmapView,
    selected: &[usize],
) -> Result<()> {
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
    let bkmer_body = bkmer
        .body_prefix_for_rows(bkmer.header.n_kmers)
        .map_err(anyhow::Error::msg)?;
    let body = bsite.body().map_err(anyhow::Error::msg)?;
    let bytes_per_col = layout.meta.bytes_per_col as usize;
    for (out_idx, &row) in selected.iter().enumerate() {
        let kmer = decode_kmer_u64(bkmer_code_at(bkmer_body, row), layout.meta.k);
        writeln!(bim, "0\t{kmer}\t0\t{}\tA\tG", out_idx + 1)?;
        let col = &body[row * bytes_per_col..(row + 1) * bytes_per_col];
        bed.write_all(&plink_row_from_bitset(col, layout.samples.len()))?;
    }
    bim.flush()?;
    bed.flush()?;
    Ok(())
}

fn write_kfile(
    out_prefix: &Path,
    output_name: &str,
    layout: &KformatLayout,
    bkmer: &BkmerMmapView,
    bsite: &BsiteMmapView,
    selected: &[usize],
) -> Result<()> {
    let prefix = out_prefix.to_string_lossy();
    let idv_path = PathBuf::from(format!("{prefix}.idv"));
    let bkmer_path = PathBuf::from(format!("{prefix}.bkmer"));
    let bsite_path = PathBuf::from(format!("{prefix}.bsite"));
    let meta_path = PathBuf::from(format!("{prefix}.meta.json"));
    write_idv_file(&idv_path, &layout.samples)?;

    let mut bkmer_writer = BufWriter::with_capacity(8 * 1024 * 1024, File::create(&bkmer_path)?);
    BkmerHeader {
        k: layout.meta.k,
        n_kmers: selected.len() as u64,
        canonical: if layout.meta.canonical { 1 } else { 0 },
    }
    .write_to(&mut bkmer_writer)?;
    let bkmer_body = bkmer
        .body_prefix_for_rows(bkmer.header.n_kmers)
        .map_err(anyhow::Error::msg)?;
    for &row in selected {
        bkmer_writer.write_all(&bkmer_body[row * 8..(row + 1) * 8])?;
    }
    bkmer_writer.flush()?;

    let mut bsite_writer = BufWriter::with_capacity(8 * 1024 * 1024, File::create(&bsite_path)?);
    BsiteHeader {
        n_samples: layout.meta.n_samples,
        n_kmers: selected.len() as u64,
        bytes_per_col: layout.meta.bytes_per_col,
    }
    .write_to(&mut bsite_writer)?;
    let bsite_body = bsite.body().map_err(anyhow::Error::msg)?;
    let bytes_per_col = layout.meta.bytes_per_col as usize;
    for &row in selected {
        bsite_writer.write_all(&bsite_body[row * bytes_per_col..(row + 1) * bytes_per_col])?;
    }
    bsite_writer.flush()?;

    let mut meta = layout.meta.clone();
    meta.n_kmers = selected.len() as u64;
    meta.bkmer_file = format!("{output_name}.bkmer");
    meta.bsite_file = format!("{output_name}.bsite");
    meta.idv_file = format!("{output_name}.idv");
    meta.min_presence_rate = 0.0;
    meta.max_presence_rate = 1.0;
    meta.encoding = ENCODING_NAME.to_string();
    write_meta_json(&meta_path, &meta)?;
    Ok(())
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
    let output_name = output
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| anyhow::anyhow!("output prefix must have a valid file name"))?
        .to_string();
    let (temp_dir, temp_prefix) = make_temp_output_prefix(&output)?;
    let bkmer = open_bkmer_mmap(&layout.bkmer_path, "kformat bkmer").map_err(anyhow::Error::msg)?;
    let bsite = open_bsite_mmap(&layout.bsite_path, "kformat bsite").map_err(anyhow::Error::msg)?;
    let bkmer_body = bkmer
        .body_prefix_for_rows(bkmer.header.n_kmers)
        .map_err(anyhow::Error::msg)?;
    let bsite_body = bsite.body().map_err(anyhow::Error::msg)?;
    let extract = extract_kmers
        .map(|items| normalize_extract_codes(items, layout.meta.k, layout.meta.canonical))
        .transpose()?;
    let selected = select_rows(
        bkmer_body,
        bsite_body,
        layout.meta.n_kmers as usize,
        layout.samples.len(),
        maf,
        extract.as_ref(),
        thread,
    )?;
    if selected.is_empty() {
        let _ = fs::remove_dir_all(&temp_dir);
        bail!("no k-mers survived kformat filters");
    }

    let write_result = if format == "bfile" {
        write_bfile(&temp_prefix, &layout, &bkmer, &bsite, &selected)
    } else {
        write_kfile(
            &temp_prefix,
            &output_name,
            &layout,
            &bkmer,
            &bsite,
            &selected,
        )
    };
    if let Err(err) = write_result {
        let _ = fs::remove_dir_all(&temp_dir);
        return Err(err);
    }
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
        n_kmers_out: selected.len() as u64,
        maf,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::{maf_from_bitset, normalize_extract_codes, plink_row_from_presence, select_rows};
    use crate::kmer::encode::{canonical_code, encode_kmer_u64};

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
        let codes = vec![11u64, 22, 33, 44];
        let mut bkmer = Vec::with_capacity(codes.len() * 8);
        for code in &codes {
            bkmer.extend_from_slice(&code.to_le_bytes());
        }
        let bsite = vec![0b0000_0001, 0b0000_0011, 0b0000_0000, 0b0000_0111];
        let mut extract = HashSet::new();
        extract.insert(44u64);
        let keep =
            select_rows(&bkmer, &bsite, codes.len(), 3, 0.0, Some(&extract), 2).expect("select");
        assert_eq!(keep, vec![3]);
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
