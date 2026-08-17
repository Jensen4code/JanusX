use crate::kmer::format::{KmergeMeta, SampleEntry};
use anyhow::{Context, Result};
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::Path;

pub fn bytes_per_col(n_samples: usize) -> usize {
    n_samples.div_ceil(8)
}

pub fn bitset_column(n_samples: usize, present_sample_ids: &[u32]) -> Vec<u8> {
    let mut buf = vec![0u8; bytes_per_col(n_samples)];
    for &sample_id in present_sample_ids {
        let idx = sample_id as usize;
        let byte = idx >> 3;
        let bit = idx & 7;
        if let Some(slot) = buf.get_mut(byte) {
            *slot |= 1u8 << bit;
        }
    }
    buf
}

pub fn write_idv_file(path: &Path, samples: &[SampleEntry]) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("failed to create idv file: {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for sample in samples {
        writeln!(writer, "{}", sample.sample_id)?;
    }
    writer.flush()?;
    Ok(())
}

/// Read the compact one-sample-ID-per-line sidecar used by current kfiles.
///
/// Older JanusX releases wrote a three-column TSV with an ``#idx`` header.
/// Accept that layout as a read-only compatibility path so existing kfiles
/// remain usable while all new writers emit the compact representation.
pub fn read_idv_file(path: &Path) -> Result<Vec<SampleEntry>> {
    let file =
        File::open(path).with_context(|| format!("failed to open idv file: {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut rows = Vec::new();
    let mut legacy = false;
    let mut saw_data = false;
    for (line_no, raw) in reader.lines().enumerate() {
        let line = raw.with_context(|| {
            format!("failed to read idv row {}:{}", path.display(), line_no + 1)
        })?;
        let text = line.trim();
        if text.is_empty() {
            continue;
        }
        if !saw_data && text.starts_with("#idx") {
            legacy = true;
            saw_data = true;
            continue;
        }
        saw_data = true;
        let fields = text.split_whitespace().collect::<Vec<_>>();
        let (index, sample_id, kmc_prefix) = if legacy {
            if fields.len() < 2 {
                anyhow::bail!(
                    "invalid legacy idv row at {}:{}",
                    path.display(),
                    line_no + 1
                );
            }
            let index = fields[0].parse::<u32>().with_context(|| {
                format!(
                    "invalid legacy idv index at {}:{}",
                    path.display(),
                    line_no + 1
                )
            })?;
            (
                index,
                fields[1].to_string(),
                fields.get(2).copied().unwrap_or_default().to_string(),
            )
        } else {
            if fields.len() != 1 {
                anyhow::bail!(
                    "idv must contain one sample ID per line at {}:{}",
                    path.display(),
                    line_no + 1
                );
            }
            (rows.len() as u32, fields[0].to_string(), String::new())
        };
        if sample_id.is_empty() {
            anyhow::bail!("empty sample ID at {}:{}", path.display(), line_no + 1);
        }
        rows.push(SampleEntry {
            index,
            sample_id,
            kmc_prefix,
        });
    }
    rows.sort_by_key(|row| row.index);
    let mut seen = std::collections::HashSet::with_capacity(rows.len());
    for (expected, row) in rows.iter().enumerate() {
        if row.index as usize != expected {
            anyhow::bail!(
                "idv indices must be contiguous from 0 in {} (saw {} at row {})",
                path.display(),
                row.index,
                expected
            );
        }
        if !seen.insert(row.sample_id.clone()) {
            anyhow::bail!(
                "duplicate sample ID in {}: {}",
                path.display(),
                row.sample_id
            );
        }
    }
    Ok(rows)
}

pub fn append_path_to_writer<W: Write>(writer: &mut W, path: &Path) -> Result<u64> {
    let mut reader = File::open(path)
        .with_context(|| format!("failed to open part file: {}", path.display()))?;
    let copied = io::copy(&mut reader, writer)
        .with_context(|| format!("failed appending part file: {}", path.display()))?;
    Ok(copied)
}

#[cfg(test)]
mod tests {
    use super::{read_idv_file, write_idv_file};
    use crate::kmer::format::SampleEntry;
    use std::fs;

    #[test]
    fn idv_writer_uses_one_sample_id_per_line_and_reader_accepts_legacy_tsv() {
        let dir = std::env::temp_dir().join(format!(
            "janusx_idv_format_{}_{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("t")
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create temp directory");
        let compact = dir.join("compact.idv");
        let samples = vec![
            SampleEntry {
                index: 0,
                sample_id: "s1".to_string(),
                kmc_prefix: "p1".to_string(),
            },
            SampleEntry {
                index: 1,
                sample_id: "s2".to_string(),
                kmc_prefix: "p2".to_string(),
            },
        ];
        write_idv_file(&compact, &samples).expect("write compact idv");
        assert_eq!(
            fs::read_to_string(&compact).expect("read compact"),
            "s1\ns2\n"
        );
        assert_eq!(read_idv_file(&compact).expect("read compact").len(), 2);

        let legacy = dir.join("legacy.idv");
        fs::write(
            &legacy,
            "#idx\tsample_id\tkmc_prefix\n0\ts1\tp1\n1\ts2\tp2\n",
        )
        .expect("write legacy idv");
        let parsed = read_idv_file(&legacy).expect("read legacy");
        assert_eq!(parsed[0].sample_id, "s1");
        assert_eq!(parsed[1].kmc_prefix, "p2");
        let _ = fs::remove_dir_all(&dir);
    }
}

pub fn write_meta_json(path: &Path, meta: &KmergeMeta) -> Result<()> {
    let file = File::create(path)
        .with_context(|| format!("failed to create meta file: {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, meta)
        .with_context(|| format!("failed to serialize meta json: {}", path.display()))?;
    writer.write_all(b"\n")?;
    writer.flush()?;
    Ok(())
}
