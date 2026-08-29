use std::collections::BTreeMap;

use super::config::{ProposalConfig, ProposalMode};
use super::corr::CorrChannel;
use super::merge::merge_and_select_site_pool;
use super::rf::RfChannel;
use super::types::{
    MergedSitePool, ProposalChannel, ProposalChannelId, ProposalContext, SiteProposal,
};

/// Execute development proposal channels and return the final LD-
/// representative site pool. This module owns channel dispatch so the scan
/// code only sees a stable pool contract.
pub(crate) fn build_proposal_site_pool(
    config: ProposalConfig,
    window_id: &str,
    marker_ids: &[usize],
    genotype_rows: &[Vec<u8>],
    residual: &[f64],
    maf: &[f64],
    ld_r2: f64,
    replicate_seed: u64,
) -> Result<MergedSitePool, String> {
    build_proposal_site_pool_for_replicate(
        config,
        window_id,
        marker_ids,
        genotype_rows,
        residual,
        maf,
        ld_r2,
        replicate_seed,
        0,
        false,
    )
}

pub(crate) fn build_proposal_site_pool_for_replicate(
    config: ProposalConfig,
    window_id: &str,
    marker_ids: &[usize],
    genotype_rows: &[Vec<u8>],
    residual: &[f64],
    maf: &[f64],
    ld_r2: f64,
    replicate_seed: u64,
    replicate_id: usize,
    is_null: bool,
) -> Result<MergedSitePool, String> {
    if matches!(config.mode, ProposalMode::Off) {
        return Err("proposal site pool requested while proposal mode is off".to_string());
    }
    let context = ProposalContext {
        window_id,
        marker_ids,
        genotype_rows,
        residual,
        maf,
        ld_r2,
        max_order: config.max_order,
        replicate_seed,
        replicate_id,
        is_null,
        provenance_token: replicate_seed
            ^ (replicate_id as u64).rotate_left(11)
            ^ u64::from(is_null).wrapping_mul(0xD1B5_4A32_D192_ED03),
    };
    let mut proposals = Vec::<SiteProposal>::new();
    match config.mode {
        ProposalMode::Off => unreachable!("checked above"),
        ProposalMode::Corr => {
            proposals.extend(CorrChannel::default().propose(&context)?.site_proposals)
        }
        ProposalMode::Rf => {
            proposals.extend(RfChannel::default().propose(&context)?.site_proposals)
        }
        ProposalMode::CorrRf => {
            proposals.extend(CorrChannel::default().propose(&context)?.site_proposals);
            proposals.extend(RfChannel::default().propose(&context)?.site_proposals);
        }
    }
    let marker_rows = marker_ids
        .iter()
        .copied()
        .zip(genotype_rows.iter().cloned())
        .collect::<BTreeMap<_, _>>();
    merge_and_select_site_pool(
        proposals,
        &marker_rows,
        ld_r2,
        default_channel_quotas(config.mode, config.k_site),
        config.k_site,
    )
}

fn default_channel_quotas(mode: ProposalMode, k_site: usize) -> super::merge::ChannelQuota {
    let channels = match mode {
        ProposalMode::Off => Vec::new(),
        ProposalMode::Corr => vec![(ProposalChannelId::Corr, k_site)],
        ProposalMode::Rf => vec![(ProposalChannelId::Rf, k_site)],
        ProposalMode::CorrRf => vec![
            (ProposalChannelId::Corr, k_site.div_ceil(2)),
            (ProposalChannelId::Rf, k_site / 2),
        ],
    };
    super::merge::ChannelQuota::from_entries(channels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn off_mode_is_not_accidentally_a_proposal_run() {
        let result = build_proposal_site_pool(
            ProposalConfig::default(),
            "test",
            &[0],
            &[vec![0, 1]],
            &[0.0, 1.0],
            &[0.5],
            0.8,
            1,
        );
        assert!(result.is_err());
    }
}
