#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ProposalMode {
    Off,
    Corr,
    Rf,
    CorrRf,
}

impl ProposalMode {
    fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "off" | "" => Ok(Self::Off),
            "corr" => Ok(Self::Corr),
            "rf" => Ok(Self::Rf),
            "corr_rf" => Ok(Self::CorrRf),
            other => Err(format!(
                "JX_GARFIELD_PROPOSAL_MODE must be off|corr|rf|corr_rf, got '{other}'"
            )),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ProposalConfig {
    pub(crate) mode: ProposalMode,
    pub(crate) k_site: usize,
    pub(crate) k2: usize,
    pub(crate) max_order: usize,
}

impl Default for ProposalConfig {
    fn default() -> Self {
        Self {
            mode: ProposalMode::Off,
            k_site: 64,
            k2: 100,
            max_order: 2,
        }
    }
}

fn parse_positive_usize(name: &str, raw: Option<String>, default: usize) -> Result<usize, String> {
    let Some(raw) = raw else {
        return Ok(default);
    };
    let parsed = raw
        .trim()
        .parse::<usize>()
        .map_err(|_| format!("{name} must be a positive integer, got '{raw}'"))?;
    if parsed == 0 {
        return Err(format!("{name} must be > 0"));
    }
    Ok(parsed)
}

/// Resolve development-only proposal settings.  Parsing is intentionally
/// performed at the scan boundary instead of cached globally, so separate
/// null/observed runs cannot accidentally inherit a previous test setting.
pub(crate) fn resolve_proposal_config() -> Result<ProposalConfig, String> {
    let mode = ProposalMode::parse(
        std::env::var("JX_GARFIELD_PROPOSAL_MODE")
            .unwrap_or_else(|_| "off".to_string())
            .as_str(),
    )?;
    let defaults = ProposalConfig {
        mode,
        ..ProposalConfig::default()
    };
    let k_site = parse_positive_usize(
        "JX_GARFIELD_PROPOSAL_K_SITE",
        std::env::var("JX_GARFIELD_PROPOSAL_K_SITE").ok(),
        defaults.k_site,
    )?;
    let k2 = parse_positive_usize(
        "JX_GARFIELD_PROPOSAL_K2",
        std::env::var("JX_GARFIELD_PROPOSAL_K2").ok(),
        defaults.k2,
    )?;
    let max_order = parse_positive_usize(
        "JX_GARFIELD_PROPOSAL_MAX_ORDER",
        std::env::var("JX_GARFIELD_PROPOSAL_MAX_ORDER").ok(),
        defaults.max_order,
    )?;
    if !matches!(max_order, 2 | 3) {
        return Err(format!(
            "JX_GARFIELD_PROPOSAL_MAX_ORDER must be 2 or 3, got {max_order}"
        ));
    }
    Ok(ProposalConfig {
        mode,
        k_site,
        k2,
        max_order,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn clear_env() {
        std::env::remove_var("JX_GARFIELD_PROPOSAL_MODE");
        std::env::remove_var("JX_GARFIELD_PROPOSAL_K_SITE");
        std::env::remove_var("JX_GARFIELD_PROPOSAL_K2");
        std::env::remove_var("JX_GARFIELD_PROPOSAL_MAX_ORDER");
    }

    #[test]
    fn proposal_mode_defaults_to_off() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        assert_eq!(resolve_proposal_config().unwrap().mode, ProposalMode::Off);
    }

    #[test]
    fn proposal_mode_parses_strict_development_values() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        std::env::set_var("JX_GARFIELD_PROPOSAL_MODE", "corr_rf");
        std::env::set_var("JX_GARFIELD_PROPOSAL_K_SITE", "128");
        std::env::set_var("JX_GARFIELD_PROPOSAL_K2", "200");
        std::env::set_var("JX_GARFIELD_PROPOSAL_MAX_ORDER", "3");
        let cfg = resolve_proposal_config().unwrap();
        assert_eq!(
            cfg,
            ProposalConfig {
                mode: ProposalMode::CorrRf,
                k_site: 128,
                k2: 200,
                max_order: 3
            }
        );
        clear_env();
    }

    #[test]
    fn proposal_mode_rejects_invalid_order() {
        let _guard = ENV_LOCK.lock().unwrap();
        clear_env();
        std::env::set_var("JX_GARFIELD_PROPOSAL_MAX_ORDER", "4");
        assert!(resolve_proposal_config().is_err());
        clear_env();
    }
}
