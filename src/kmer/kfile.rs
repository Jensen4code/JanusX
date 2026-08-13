use crate::breader::parse_bsite_header;
use crate::kmer::format::{KmergeMeta, SampleEntry, BSITE_HEADER_SIZE};
use anyhow::{bail, Context, Result};
use numpy::ndarray::{Array1, Array2};
use numpy::{PyArray1, PyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::BoundObject;
use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

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
    use super::{decode_bsite_chunk, maf_from_presence};

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
}
