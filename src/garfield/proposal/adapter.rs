use super::types::{ChannelEvidence, ProposalChannelId, ProposalContext, SiteProposal};
use crate::ml::common::topk_indices;

pub(crate) fn validate_context(context: &ProposalContext<'_>) -> Result<(), String> {
    if context.marker_ids.is_empty() {
        return Err("proposal context has no markers".to_string());
    }
    if context.marker_ids.len() != context.genotype_rows.len() {
        return Err(format!(
            "proposal marker/genotype length mismatch: {} vs {}",
            context.marker_ids.len(),
            context.genotype_rows.len()
        ));
    }
    if context.residual.is_empty() {
        return Err("proposal context has no residual samples".to_string());
    }
    if context.maf.len() != context.marker_ids.len() {
        return Err(format!(
            "proposal marker/maf length mismatch: {} vs {}",
            context.marker_ids.len(),
            context.maf.len()
        ));
    }
    for (row_idx, row) in context.genotype_rows.iter().enumerate() {
        if row.len() != context.residual.len() {
            return Err(format!(
                "proposal genotype row {} has {} samples, expected {}",
                row_idx,
                row.len(),
                context.residual.len()
            ));
        }
    }
    if !context.ld_r2.is_finite() || !(0.0..=1.0).contains(&context.ld_r2) {
        return Err(format!(
            "proposal ld_r2 must be in [0,1], got {}",
            context.ld_r2
        ));
    }
    if context.max_order == 0 {
        return Err("proposal max_order must be > 0".to_string());
    }
    Ok(())
}

pub(crate) fn normalize_binary_rows(rows: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, String> {
    let first_len = rows
        .first()
        .ok_or_else(|| "proposal genotype rows are empty".to_string())?
        .len();
    if first_len == 0 {
        return Err("proposal genotype rows have zero samples".to_string());
    }
    let mut out = Vec::with_capacity(rows.len());
    for (idx, row) in rows.iter().enumerate() {
        if row.len() != first_len {
            return Err(format!(
                "proposal genotype row {} has {} samples, expected {}",
                idx,
                row.len(),
                first_len
            ));
        }
        out.push(row.iter().map(|&value| u8::from(value > 0)).collect());
    }
    Ok(out)
}

pub(crate) fn ranked_site_proposals<F>(
    marker_ids: &[usize],
    scores: &[f64],
    channel: ProposalChannelId,
    top_k: usize,
    mut evidence: F,
) -> Vec<SiteProposal>
where
    F: FnMut(f64) -> ChannelEvidence,
{
    if marker_ids.len() != scores.len() || top_k == 0 {
        return Vec::new();
    }
    let selected = topk_indices(scores, top_k.min(marker_ids.len()));
    let n = selected.len();
    selected
        .into_iter()
        .enumerate()
        .map(|(rank_offset, local_idx)| {
            let rank = rank_offset + 1;
            let percentile = if n <= 1 {
                Some(1.0)
            } else {
                Some(((n - rank) as f64 / ((n - 1) as f64)).clamp(0.0, 1.0))
            };
            let score = scores[local_idx];
            SiteProposal {
                marker_id: marker_ids[local_idx],
                channel,
                rank,
                percentile,
                evidence: evidence(score),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_rows_retain_feature_major_shape() {
        let rows = vec![vec![0_u8, 2, 1], vec![2, 0, 1]];
        let normalized = normalize_binary_rows(&rows).unwrap();
        assert_eq!(normalized, vec![vec![0, 1, 1], vec![1, 0, 1]]);
    }

    #[test]
    fn binary_rows_reject_empty_samples() {
        let rows = vec![Vec::<u8>::new()];
        assert!(normalize_binary_rows(&rows).is_err());
    }
}
