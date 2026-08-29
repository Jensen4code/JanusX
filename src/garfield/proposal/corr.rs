use super::adapter::{normalize_binary_rows, ranked_site_proposals, validate_context};
use super::types::{
    ChannelEvidence, ChannelProposalBatch, ProposalChannel, ProposalChannelId, ProposalContext,
};
use crate::ml::univariate::feature_scores_abs_corr_dosage_x;

#[derive(Clone, Copy, Debug)]
pub(crate) struct CorrChannel {
    top_k: usize,
}

impl CorrChannel {
    pub(crate) const fn new(top_k: usize) -> Self {
        Self { top_k }
    }
}

impl Default for CorrChannel {
    fn default() -> Self {
        Self::new(64)
    }
}

impl ProposalChannel for CorrChannel {
    fn id(&self) -> ProposalChannelId {
        ProposalChannelId::Corr
    }

    fn propose(&self, context: &ProposalContext<'_>) -> Result<ChannelProposalBatch, String> {
        validate_context(context)?;
        let rows = normalize_binary_rows(context.genotype_rows)?;
        let scores = feature_scores_abs_corr_dosage_x(rows.as_slice(), context.residual);
        let site_proposals = ranked_site_proposals(
            context.marker_ids,
            scores.as_slice(),
            self.id(),
            self.top_k,
            |score| ChannelEvidence::Corr { score },
        );
        Ok(ChannelProposalBatch {
            channel: self.id(),
            window_id: context.window_id.to_string(),
            site_proposals,
            rule_proposals: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::garfield::proposal::types::ProposalContext;

    fn context<'a>(
        marker_ids: &'a [usize],
        genotype_rows: &'a [Vec<u8>],
        residual: &'a [f64],
        maf: &'a [f64],
    ) -> ProposalContext<'a> {
        ProposalContext {
            window_id: "corr-test",
            marker_ids,
            genotype_rows,
            residual,
            maf,
            ld_r2: 0.8,
            max_order: 3,
            replicate_seed: 19,
            replicate_id: 0,
            is_null: false,
            provenance_token: 19,
        }
    }

    #[test]
    fn corr_returns_global_marker_ids_and_local_ranks() {
        let marker_ids = [101, 205, 309];
        let genotype_rows = vec![vec![0_u8, 1, 0, 1], vec![0, 0, 1, 1], vec![1, 0, 1, 0]];
        let residual = [1.0, -1.0, 1.0, -1.0];
        let maf = [0.5, 0.5, 0.5];
        let batch = CorrChannel::new(2)
            .propose(&context(&marker_ids, &genotype_rows, &residual, &maf))
            .unwrap();
        assert!(batch
            .site_proposals
            .iter()
            .all(|p| marker_ids.contains(&p.marker_id)));
        assert!(batch
            .site_proposals
            .windows(2)
            .all(|w| w[0].rank < w[1].rank));
        assert!(batch
            .site_proposals
            .iter()
            .all(|p| p.channel == ProposalChannelId::Corr));
    }
}
