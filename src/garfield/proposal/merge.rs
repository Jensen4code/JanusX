use std::collections::{BTreeMap, HashSet};

use super::ld::{channel_priority, cluster_proxy_map, cluster_site_proposals};
use super::types::{LdCluster, MergedSitePool, ProposalChannelId, SiteProposal};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ChannelQuota {
    entries: Vec<(ProposalChannelId, usize)>,
}

impl ChannelQuota {
    pub(crate) fn new<const N: usize>(entries: [(ProposalChannelId, usize); N]) -> Self {
        Self::from_entries(entries.into_iter().collect())
    }

    pub(crate) fn from_entries(entries: Vec<(ProposalChannelId, usize)>) -> Self {
        let mut by_channel = BTreeMap::<ProposalChannelId, usize>::new();
        for (channel, quota) in entries {
            by_channel
                .entry(channel)
                .and_modify(|current| *current = current.saturating_add(quota))
                .or_insert(quota);
        }
        Self {
            entries: by_channel.into_iter().collect(),
        }
    }

    pub(crate) fn entries(&self) -> &[(ProposalChannelId, usize)] {
        &self.entries
    }
}

fn cluster_queue_key(cluster: &LdCluster, channel: ProposalChannelId) -> (usize, usize, usize) {
    let evidence = cluster
        .channel_evidence
        .get(&channel)
        .expect("queue only contains clusters with channel evidence");
    (
        evidence.best_rank,
        cluster.representative_marker_id,
        cluster.cluster_id,
    )
}

fn channel_queues(clusters: &[LdCluster]) -> BTreeMap<ProposalChannelId, Vec<usize>> {
    let mut queues = BTreeMap::<ProposalChannelId, Vec<usize>>::new();
    for cluster in clusters {
        for channel in cluster.channel_evidence.keys().copied() {
            queues.entry(channel).or_default().push(cluster.cluster_id);
        }
    }
    for (channel, queue) in &mut queues {
        queue.sort_by_key(|cluster_id| cluster_queue_key(&clusters[*cluster_id], *channel));
    }
    queues
}

fn select_channel_quota(
    selected: &mut Vec<usize>,
    selected_set: &mut HashSet<usize>,
    queue: &[usize],
    quota: usize,
    k_site: usize,
) {
    let mut taken_for_channel = 0usize;
    for &cluster_id in queue {
        if selected.len() >= k_site || taken_for_channel >= quota {
            break;
        }
        if selected_set.insert(cluster_id) {
            selected.push(cluster_id);
            taken_for_channel += 1;
        }
    }
}

/// Merge channel proposals, cluster high-LD proxies, then allocate a fixed
/// representative pool with deterministic channel quotas and refill.
pub(crate) fn merge_and_select_site_pool(
    proposals: Vec<SiteProposal>,
    marker_rows: &BTreeMap<usize, Vec<u8>>,
    ld_r2: f64,
    quotas: ChannelQuota,
    k_site: usize,
) -> Result<MergedSitePool, String> {
    let clusters = cluster_site_proposals(&proposals, marker_rows, ld_r2)?;
    let k_pre = clusters.iter().map(|cluster| cluster.members.len()).sum();
    if k_site == 0 || clusters.is_empty() {
        return Ok(MergedSitePool {
            representatives: Vec::new(),
            proxy_to_representative: cluster_proxy_map(&clusters),
            clusters,
            k_pre,
            k_site: 0,
        });
    }

    let queues = channel_queues(&clusters);
    let mut selected = Vec::with_capacity(k_site.min(clusters.len()));
    let mut selected_set = HashSet::with_capacity(k_site.min(clusters.len()));

    // First pass honors each configured channel quota.  A shared LD cluster
    // consumes a slot only in the channel that actually selected it.
    for &(channel, quota) in quotas.entries() {
        if let Some(queue) = queues.get(&channel) {
            select_channel_quota(&mut selected, &mut selected_set, queue, quota, k_site);
        }
        if selected.len() >= k_site {
            break;
        }
    }

    // Deterministic refill uses the best remaining cluster evidence, without
    // comparing raw Corr and RF score scales.
    let mut remaining = clusters
        .iter()
        .filter(|cluster| !selected_set.contains(&cluster.cluster_id))
        .map(|cluster| {
            let (channel, evidence) = cluster
                .channel_evidence
                .iter()
                .min_by_key(|(channel, evidence)| {
                    (
                        evidence.best_rank,
                        channel_priority(**channel),
                        cluster.representative_marker_id,
                    )
                })
                .expect("cluster has at least one channel");
            (
                cluster.cluster_id,
                evidence.best_rank,
                channel_priority(*channel),
                cluster.representative_marker_id,
            )
        })
        .collect::<Vec<_>>();
    remaining.sort_unstable_by_key(|(_, best_rank, priority, marker_id)| {
        (*best_rank, *priority, *marker_id)
    });
    for (cluster_id, _, _, _) in remaining {
        if selected.len() >= k_site {
            break;
        }
        selected.push(cluster_id);
    }

    let representatives = selected
        .iter()
        .map(|&cluster_id| clusters[cluster_id].representative_marker_id)
        .collect::<Vec<_>>();
    Ok(MergedSitePool {
        k_site: representatives.len(),
        representatives,
        proxy_to_representative: cluster_proxy_map(&clusters),
        clusters,
        k_pre,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::garfield::proposal::types::{ChannelEvidence, ProposalChannelId, SiteProposal};

    fn site_proposals_for_rows(ids: [usize; 3], scores: [f64; 3]) -> Vec<SiteProposal> {
        ids.into_iter()
            .zip(scores)
            .enumerate()
            .map(|(idx, (marker_id, score))| SiteProposal {
                marker_id,
                channel: if idx == 0 {
                    ProposalChannelId::Rf
                } else {
                    ProposalChannelId::Corr
                },
                rank: idx + 1,
                percentile: Some(1.0 - idx as f64 / 3.0),
                evidence: if idx == 0 {
                    ChannelEvidence::Rf {
                        importance: score,
                        tree_prevalence: None,
                        leaf_mass: None,
                        supporting_trees: None,
                        min_depth: None,
                    }
                } else {
                    ChannelEvidence::Corr { score }
                },
            })
            .collect()
    }

    fn binary_rows_for_high_ld_pair() -> BTreeMap<usize, Vec<u8>> {
        BTreeMap::from([
            (11, vec![0, 1, 0, 1, 0, 1]),
            (17, vec![0, 1, 0, 1, 0, 1]),
            (29, vec![0, 0, 1, 1, 0, 0]),
        ])
    }

    fn default_quota() -> ChannelQuota {
        ChannelQuota::new([(ProposalChannelId::Rf, 1), (ProposalChannelId::Corr, 1)])
    }

    #[test]
    fn quota_is_allocated_to_clusters_not_raw_rows() {
        let proposals = site_proposals_for_rows([11, 17, 29], [0.9, 0.8, 0.7]);
        let selected = merge_and_select_site_pool(
            proposals,
            &binary_rows_for_high_ld_pair(),
            0.8,
            ChannelQuota::new([(ProposalChannelId::Rf, 2)]),
            2,
        )
        .unwrap();
        assert_eq!(selected.k_site, 2);
        assert_eq!(selected.representatives.len(), 2);
    }

    #[test]
    fn refill_is_deterministic_when_a_channel_quota_collapses() {
        let proposals = site_proposals_for_rows([11, 17, 29], [0.9, 0.8, 0.7]);
        let first = merge_and_select_site_pool(
            proposals.clone(),
            &binary_rows_for_high_ld_pair(),
            0.8,
            default_quota(),
            3,
        )
        .unwrap();
        let second = merge_and_select_site_pool(
            proposals,
            &binary_rows_for_high_ld_pair(),
            0.8,
            default_quota(),
            3,
        )
        .unwrap();
        assert_eq!(first.representatives, second.representatives);
    }
}
