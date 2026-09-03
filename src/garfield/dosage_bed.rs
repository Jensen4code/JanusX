//! Persistent packed-BED reader for fixed-V GRM dosage scans.
//!
//! The reader opens FAM/BED/BIM once, keeps only a compact BIM index and a
//! bounded BED mmap window, and feeds each requested window to the existing
//! fixed-V GRM/GLS scanner.  It is intentionally separate from the legacy
//! Boolean GARFIELD path and from the dense-array dosage APIs.

use crate::garfield::dosage_random::{scan_grm_pairs, set_scan_items};
use crate::gfcore::{
    process_snp_row_with_precomputed_counts, process_snp_row_with_precomputed_counts_preserve_alt,
    read_fam, BedSnpIter, SiteInfo,
};
use crate::gfreader::build_sample_selection;
use numpy::ndarray::Array2;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Return the source-row range for an inclusive BIM interval.
#[cfg(test)]
fn window_bounds_for_sites(
    sites: &[SiteInfo],
    chrom: &str,
    start: i32,
    end: i32,
) -> Option<(usize, usize)> {
    if start > end {
        return None;
    }
    let mut first = None;
    let mut last = None;
    for (idx, site) in sites.iter().enumerate() {
        if site.chrom == chrom && site.pos >= start && site.pos <= end {
            first.get_or_insert(idx);
            last = Some(idx + 1);
        } else if first.is_some() && site.chrom == chrom && site.pos > end {
            break;
        }
    }
    first.zip(last)
}

struct BimIndex {
    path: PathBuf,
    offsets: Vec<u64>,
    positions: Vec<i32>,
    chrom_names: Vec<String>,
    chrom_ranges: HashMap<String, (usize, usize)>,
    reader: BufReader<File>,
}

impl BimIndex {
    fn open(prefix: &str, expected_rows: usize) -> Result<Self, String> {
        let path = PathBuf::from(format!("{prefix}.bim"));
        let file = File::open(&path).map_err(|e| e.to_string())?;
        let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
        let mut offsets = Vec::with_capacity(expected_rows.saturating_add(1));
        let mut positions = Vec::with_capacity(expected_rows);
        let mut chrom_names = Vec::<String>::new();
        let mut chrom_lookup = HashMap::<String, u32>::new();
        let mut chrom_ranges = HashMap::<String, (usize, usize)>::new();
        let mut seen_chroms = HashMap::<u32, usize>::new();
        let mut last_chrom_id = None::<u32>;
        let mut last_pos = None::<i32>;
        let mut offset = 0u64;
        let mut line = Vec::<u8>::new();

        loop {
            offsets.push(offset);
            line.clear();
            let bytes = reader
                .read_until(b'\n', &mut line)
                .map_err(|e| format!("{}: {e}", path.display()))?;
            if bytes == 0 {
                offsets.pop();
                break;
            }
            offset = offset.saturating_add(bytes as u64);
            let text = std::str::from_utf8(&line)
                .map_err(|e| format!("{}: invalid UTF-8 in BIM: {e}", path.display()))?;
            let (chrom, pos) = parse_bim_index_line(&path, positions.len() + 1, text.trim_end())?;
            let chrom_id = if let Some(&id) = chrom_lookup.get(chrom.as_str()) {
                id
            } else {
                let id = u32::try_from(chrom_names.len())
                    .map_err(|_| "too many chromosomes for compact BIM index".to_string())?;
                chrom_lookup.insert(chrom.clone(), id);
                chrom_names.push(chrom.clone());
                id
            };
            if let Some(previous) = last_chrom_id {
                if previous == chrom_id {
                    if let Some(previous_pos) = last_pos {
                        if pos < previous_pos {
                            return Err(format!(
                                "BIM positions are not sorted within chromosome {chrom} at row {}",
                                positions.len() + 1
                            ));
                        }
                    }
                } else if seen_chroms.contains_key(&chrom_id) {
                    return Err(format!(
                        "BIM chromosome {chrom} is not contiguous; packed window reader requires chromosome-grouped BIM"
                    ));
                }
            }
            let row = positions.len();
            seen_chroms.entry(chrom_id).or_insert(row);
            chrom_ranges
                .entry(chrom.clone())
                .and_modify(|range| range.1 = row + 1)
                .or_insert((row, row + 1));
            positions.push(pos);
            last_chrom_id = Some(chrom_id);
            last_pos = Some(pos);
        }

        if positions.is_empty() {
            return Err(format!("no site rows found in {}", path.display()));
        }
        if positions.len() != expected_rows {
            return Err(format!(
                "BED/BIM SNP count mismatch: bed={expected_rows}, bim={}",
                positions.len()
            ));
        }
        offsets.push(offset);
        Ok(Self {
            path,
            offsets,
            positions,
            chrom_names,
            chrom_ranges,
            reader,
        })
    }

    fn bounds(&self, chrom: &str, start: i32, end: i32) -> Option<(usize, usize)> {
        if start > end {
            return None;
        }
        let &(first, last) = self.chrom_ranges.get(chrom)?;
        let row_start = lower_bound(&self.positions, first, last, start);
        let row_end = upper_bound(&self.positions, first, last, end);
        if row_start < row_end {
            Some((row_start, row_end))
        } else {
            None
        }
    }

    fn read_sites(&mut self, start: usize, end: usize) -> Result<Vec<SiteInfo>, String> {
        if start > end || end > self.positions.len() {
            return Err(format!("invalid BIM row range [{start}, {end})"));
        }
        if start == end {
            return Ok(Vec::new());
        }
        self.reader
            .seek(SeekFrom::Start(self.offsets[start]))
            .map_err(|e| format!("{}: {e}", self.path.display()))?;
        let mut sites = Vec::with_capacity(end - start);
        let mut line = String::new();
        for row in start..end {
            line.clear();
            let bytes = self
                .reader
                .read_line(&mut line)
                .map_err(|e| format!("{}: {e}", self.path.display()))?;
            if bytes == 0 {
                return Err(format!(
                    "{} ended before BIM row {}",
                    self.path.display(),
                    row + 1
                ));
            }
            sites.push(parse_bim_site_line(&self.path, row + 1, line.trim_end())?);
        }
        Ok(sites)
    }

    fn n_snps(&self) -> usize {
        self.positions.len()
    }

    fn chrom_count(&self) -> usize {
        self.chrom_names.len()
    }

    fn sanity_checksum(&self) -> usize {
        self.positions.len()
    }
}

fn lower_bound(values: &[i32], mut first: usize, mut last: usize, target: i32) -> usize {
    while first < last {
        let middle = first + (last - first) / 2;
        if values[middle] < target {
            first = middle + 1;
        } else {
            last = middle;
        }
    }
    first
}

fn upper_bound(values: &[i32], mut first: usize, mut last: usize, target: i32) -> usize {
    while first < last {
        let middle = first + (last - first) / 2;
        if values[middle] <= target {
            first = middle + 1;
        } else {
            last = middle;
        }
    }
    first
}

fn parse_bim_index_line(path: &Path, line_no: usize, line: &str) -> Result<(String, i32), String> {
    let mut cols = line.split_whitespace();
    let chrom = cols
        .next()
        .ok_or_else(|| format!("Malformed BIM line at {}:{line_no}", path.display()))?;
    let _snp = cols
        .next()
        .ok_or_else(|| format!("Malformed BIM line at {}:{line_no}", path.display()))?;
    let _cm = cols
        .next()
        .ok_or_else(|| format!("Malformed BIM line at {}:{line_no}", path.display()))?;
    let pos = cols
        .next()
        .ok_or_else(|| format!("Malformed BIM line at {}:{line_no}", path.display()))?
        .parse::<i32>()
        .map_err(|e| {
            format!(
                "Malformed BIM position at {}:{line_no}: {e}",
                path.display()
            )
        })?;
    Ok((chrom.to_string(), pos))
}

fn parse_bim_site_line(path: &Path, line_no: usize, line: &str) -> Result<SiteInfo, String> {
    let mut cols = line.split_whitespace();
    let chrom = cols
        .next()
        .ok_or_else(|| format!("Malformed BIM line at {}:{line_no}", path.display()))?;
    let snp = cols
        .next()
        .ok_or_else(|| format!("Malformed BIM line at {}:{line_no}", path.display()))?;
    let _cm = cols
        .next()
        .ok_or_else(|| format!("Malformed BIM line at {}:{line_no}", path.display()))?;
    let pos = cols
        .next()
        .ok_or_else(|| format!("Malformed BIM line at {}:{line_no}", path.display()))?
        .parse::<i32>()
        .map_err(|e| {
            format!(
                "Malformed BIM position at {}:{line_no}: {e}",
                path.display()
            )
        })?;
    let ref_allele = cols
        .next()
        .ok_or_else(|| format!("Malformed BIM line at {}:{line_no}", path.display()))?;
    let alt_allele = cols
        .next()
        .ok_or_else(|| format!("Malformed BIM line at {}:{line_no}", path.display()))?;
    Ok(SiteInfo {
        chrom: chrom.to_string(),
        pos,
        snp: snp.to_string(),
        ref_allele: ref_allele.to_string(),
        alt_allele: alt_allele.to_string(),
    })
}

#[derive(Debug)]
struct WindowData {
    genotypes: Vec<f64>,
    sites: Vec<SiteInfo>,
    source_indices: Vec<usize>,
    n_source_rows: usize,
}

#[pyclass]
pub struct GarfieldPackedBedDosageReader {
    prefix: String,
    bed: BedSnpIter,
    bim: BimIndex,
    sample_indices: Vec<usize>,
    sample_ids: Vec<String>,
    maf_threshold: f32,
    max_missing_rate: f32,
    fill_missing: bool,
    preserve_alt_orientation: bool,
    mmap_window_mb: usize,
    scratch: Vec<f32>,
}

impl GarfieldPackedBedDosageReader {
    fn read_window_data(
        &mut self,
        chrom: &str,
        start: i32,
        end: i32,
    ) -> Result<(WindowData, f64), String> {
        let decode_started = Instant::now();
        let Some((row_start, row_end)) = self.bim.bounds(chrom, start, end) else {
            return Ok((
                WindowData {
                    genotypes: Vec::new(),
                    sites: Vec::new(),
                    source_indices: Vec::new(),
                    n_source_rows: 0,
                },
                decode_started.elapsed().as_secs_f64(),
            ));
        };
        let sites = self.bim.read_sites(row_start, row_end)?;
        let n_samples = self.sample_indices.len();
        if n_samples == 0 {
            return Err("selected sample set is empty".to_string());
        }
        self.scratch.resize(n_samples, -9.0);
        let mut genotypes = Vec::with_capacity((row_end - row_start).saturating_mul(n_samples));
        let mut source_indices = Vec::with_capacity(row_end - row_start);
        let mut kept_sites = Vec::with_capacity(row_end - row_start);
        for source_idx in row_start..row_end {
            self.bed.ensure_window_for_snp(source_idx)?;
            let counts = self
                .bed
                .decode_snp_selected_raw_into_with_counts_at(
                    source_idx,
                    &self.sample_indices,
                    &mut self.scratch,
                )
                .ok_or_else(|| format!("failed to decode BED SNP row {source_idx}"))?;
            let site = sites
                .get(source_idx - row_start)
                .ok_or_else(|| "BIM/site range mismatch".to_string())?;
            let mut site = site.clone();
            let keep = if self.preserve_alt_orientation {
                process_snp_row_with_precomputed_counts_preserve_alt(
                    &mut self.scratch,
                    &mut site.ref_allele,
                    &mut site.alt_allele,
                    counts,
                    self.maf_threshold,
                    self.max_missing_rate,
                    self.fill_missing,
                    false,
                    1.0,
                )
                .is_some()
            } else {
                process_snp_row_with_precomputed_counts(
                    &mut self.scratch,
                    &mut site.ref_allele,
                    &mut site.alt_allele,
                    counts,
                    self.maf_threshold,
                    self.max_missing_rate,
                    self.fill_missing,
                    false,
                    1.0,
                )
                .is_some()
            };
            if keep {
                genotypes.extend(self.scratch.iter().map(|value| *value as f64));
                source_indices.push(source_idx);
                kept_sites.push(site);
            }
        }
        Ok((
            WindowData {
                genotypes,
                sites: kept_sites,
                source_indices,
                n_source_rows: row_end - row_start,
            },
            decode_started.elapsed().as_secs_f64(),
        ))
    }
}

#[pymethods]
impl GarfieldPackedBedDosageReader {
    #[new]
    #[pyo3(signature = (
        prefix,
        maf_threshold=0.0,
        max_missing_rate=1.0,
        fill_missing=true,
        sample_ids=None,
        sample_indices=None,
        mmap_window_mb=128,
        preserve_alt_orientation=false,
    ))]
    fn new(
        prefix: String,
        maf_threshold: f32,
        max_missing_rate: f32,
        fill_missing: bool,
        sample_ids: Option<Vec<String>>,
        sample_indices: Option<Vec<usize>>,
        mmap_window_mb: usize,
        preserve_alt_orientation: bool,
    ) -> PyResult<Self> {
        if mmap_window_mb == 0 {
            return Err(PyValueError::new_err("mmap_window_mb must be > 0"));
        }
        if !(0.0..=1.0).contains(&maf_threshold) {
            return Err(PyValueError::new_err("maf_threshold must be within [0, 1]"));
        }
        if !(0.0..=1.0).contains(&max_missing_rate) {
            return Err(PyValueError::new_err(
                "max_missing_rate must be within [0, 1]",
            ));
        }
        let mut normalized_prefix = prefix.clone();
        for suffix in [".bed", ".bim", ".fam"] {
            if normalized_prefix.to_ascii_lowercase().ends_with(suffix) {
                normalized_prefix.truncate(normalized_prefix.len() - suffix.len());
                break;
            }
        }
        let samples = read_fam(&normalized_prefix).map_err(PyRuntimeError::new_err)?;
        let bed = BedSnpIter::new_for_grm_window(&normalized_prefix, mmap_window_mb)
            .map_err(PyRuntimeError::new_err)?;
        let bim =
            BimIndex::open(&normalized_prefix, bed.n_snps()).map_err(PyRuntimeError::new_err)?;
        let (sample_indices, sample_ids) =
            build_sample_selection(&samples, sample_ids, sample_indices)
                .map_err(PyValueError::new_err)?;
        if sample_indices.iter().any(|&idx| idx >= samples.len()) {
            return Err(PyValueError::new_err("sample index out of range"));
        }
        Ok(Self {
            prefix: normalized_prefix,
            bed,
            bim,
            scratch: vec![-9.0; sample_indices.len()],
            sample_indices,
            sample_ids,
            maf_threshold,
            max_missing_rate,
            fill_missing,
            preserve_alt_orientation,
            mmap_window_mb,
        })
    }

    #[getter]
    fn n_samples(&self) -> usize {
        self.sample_indices.len()
    }

    #[getter]
    fn n_snps(&self) -> usize {
        self.bim.n_snps()
    }

    #[getter]
    fn sample_ids(&self) -> Vec<String> {
        self.sample_ids.clone()
    }

    #[getter]
    fn prefix(&self) -> &str {
        &self.prefix
    }

    fn window_bounds(&self, chrom: String, start: i32, end: i32) -> Option<(usize, usize)> {
        self.bim.bounds(&chrom, start, end)
    }

    fn read_window<'py>(
        &mut self,
        py: Python<'py>,
        chrom: String,
        start: i32,
        end: i32,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (data, decode_seconds) = self
            .read_window_data(&chrom, start, end)
            .map_err(PyValueError::new_err)?;
        let n_markers = data.sites.len();
        let array = Array2::from_shape_vec((n_markers, self.sample_indices.len()), data.genotypes)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        let out = PyDict::new(py);
        out.set_item("genotypes", PyArray2::from_owned_array(py, array))?;
        out.set_item("sites", sites_to_pylist(py, &data.sites)?)?;
        out.set_item("source_indices", data.source_indices)?;
        out.set_item("chrom", chrom)?;
        out.set_item("start", start)?;
        out.set_item("end", end)?;
        out.set_item("n_source_rows", data.n_source_rows)?;
        out.set_item("n_markers", n_markers)?;
        out.set_item("n_samples", self.sample_indices.len())?;
        out.set_item("decode_seconds", decode_seconds)?;
        out.set_item("mmap_window_mb", self.mmap_window_mb)?;
        Ok(out)
    }

    #[pyo3(signature = (chrom, start, end, y, grm, sigma_g2, sigma_e2, top_k=100, threads=0, covariates=None))]
    fn scan_window<'py>(
        &mut self,
        py: Python<'py>,
        chrom: String,
        start: i32,
        end: i32,
        y: PyReadonlyArray1<'py, f64>,
        grm: PyReadonlyArray2<'py, f64>,
        sigma_g2: f64,
        sigma_e2: f64,
        top_k: usize,
        threads: usize,
        covariates: Option<PyReadonlyArray2<'py, f64>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (data, decode_seconds) = self
            .read_window_data(&chrom, start, end)
            .map_err(PyValueError::new_err)?;
        if data.sites.is_empty() {
            return Err(PyValueError::new_err(format!(
                "no markers remain in requested BED window {chrom}:{start}-{end}"
            )));
        }
        if !self.fill_missing
            && data
                .genotypes
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(PyValueError::new_err(
                "GRM dosage BED scan requires fill_missing=True when missing genotypes remain",
            ));
        }
        let y_vec = array1_to_vec(&y);
        let n_samples = self.sample_indices.len();
        if y_vec.len() != n_samples {
            return Err(PyValueError::new_err(format!(
                "y length={} but expected {n_samples}",
                y_vec.len()
            )));
        }
        let grm_shape = grm.shape();
        if grm_shape.len() != 2 || grm_shape[0] != n_samples || grm_shape[1] != n_samples {
            return Err(PyValueError::new_err(format!(
                "grm must have shape ({n_samples}, {n_samples}); got {:?}",
                grm_shape
            )));
        }
        let grm_vec = array2_to_vec(&grm);
        let (covariate_vec, n_covariates) = if let Some(covariates) = covariates {
            let shape = covariates.shape();
            if shape.len() != 2 || shape[0] != n_samples {
                return Err(PyValueError::new_err(format!(
                    "covariates must have shape (n_samples, n_covariates); got {:?}",
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
        let scan_started = Instant::now();
        let (candidates, pairs_evaluated, pairs_skipped) = scan_grm_pairs(
            &data.genotypes,
            data.sites.len(),
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
        let scan_seconds = scan_started.elapsed().as_secs_f64();
        let out = PyDict::new(py);
        set_scan_items(
            py,
            &out,
            &candidates,
            pairs_evaluated,
            pairs_skipped,
            data.sites.len(),
            n_samples,
            effective_threads,
            sigma_g2,
            sigma_e2,
        )?;
        out.set_item("sites", sites_to_pylist(py, &data.sites)?)?;
        out.set_item("source_indices", data.source_indices)?;
        out.set_item("chrom", chrom)?;
        out.set_item("start", start)?;
        out.set_item("end", end)?;
        out.set_item("n_source_rows", data.n_source_rows)?;
        out.set_item("n_markers", data.sites.len())?;
        out.set_item("n_samples", n_samples)?;
        out.set_item("decode_seconds", decode_seconds)?;
        out.set_item("scan_seconds", scan_seconds)?;
        out.set_item("total_seconds", decode_seconds + scan_seconds)?;
        out.set_item("mmap_window_mb", self.mmap_window_mb)?;
        out.set_item("packed_bed", true)?;
        out.set_item("bim_chromosomes", self.bim.chrom_count())?;
        out.set_item("bim_index_rows", self.bim.sanity_checksum())?;
        Ok(out)
    }
}

fn sites_to_pylist<'py>(py: Python<'py>, sites: &[SiteInfo]) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for site in sites {
        let item = PyDict::new(py);
        item.set_item("chrom", &site.chrom)?;
        item.set_item("pos", site.pos)?;
        item.set_item("snp", &site.snp)?;
        item.set_item("ref", &site.ref_allele)?;
        item.set_item("alt", &site.alt_allele)?;
        out.append(item)?;
    }
    Ok(out)
}

fn array1_to_vec(array: &PyReadonlyArray1<'_, f64>) -> Vec<f64> {
    array.as_array().iter().copied().collect()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn site(chrom: &str, pos: i32) -> SiteInfo {
        SiteInfo {
            chrom: chrom.to_string(),
            pos,
            snp: format!("{chrom}:{pos}"),
            ref_allele: "A".to_string(),
            alt_allele: "C".to_string(),
        }
    }

    #[test]
    fn window_bounds_use_inclusive_coordinates_and_chromosome_ranges() {
        let sites = vec![
            site("1", 100),
            site("1", 200),
            site("1", 300),
            site("2", 100),
        ];

        assert_eq!(window_bounds_for_sites(&sites, "1", 100, 300), Some((0, 3)));
        assert_eq!(window_bounds_for_sites(&sites, "1", 201, 301), Some((2, 3)));
        assert_eq!(window_bounds_for_sites(&sites, "2", 0, 101), Some((3, 4)));
        assert_eq!(window_bounds_for_sites(&sites, "1", 301, 400), None);
        assert_eq!(window_bounds_for_sites(&sites, "3", 0, 1000), None);
    }
}
