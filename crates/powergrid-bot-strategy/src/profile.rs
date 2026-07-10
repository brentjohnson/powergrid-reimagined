use powergrid_core::types::BotDifficulty;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

// Embedded at compile time; overridden at runtime by pointing the
// BOT_PROFILES_FILE env var at an alternative TOML (read once, at first use).
const DEFAULT_PROFILES_TOML: &str = include_str!("../../../assets/bots/default.toml");

// ---------------------------------------------------------------------------
// Weight structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuctionWeights {
    /// Elektro reserved per resource-consuming plant for fuel (see `auction_reserve`),
    /// plus a flat allowance kept aside for ~2 city builds.
    pub city_reserve: f32,
    /// Flat elektro safety buffer on top of `city_reserve` and fuel reserves.
    pub safety_buffer: f32,
    /// Minimum total `PlantValue` (Elektro) improvement required to justify
    /// replacing a rack plant when full — compared against `evaluate_plant(...).total`,
    /// which already nets out the forced discard.
    pub upgrade_margin: f32,
    /// Minimum `PlantValue` (Elektro) for a plant to be worth opening an auction
    /// for; doubles as the `Pass` baseline score.
    pub min_open_score: f32,
    /// How many cities beyond currently owned are considered "usefully planned
    /// for" when capping projected income (`useful_city_target`). 2 = mild
    /// planning ahead; 1 = tight (hard); 3+ = relaxed (easy / early game). The
    /// target is always capped at `end_game_cities`.
    pub buildable_lookahead: u8,
    /// Weight on the endgame capacity premium (Elektro per city of gap closed,
    /// scaled by how close the leader is to the end-game city count). Higher →
    /// the bot pays more for capacity that helps it cross the finish line first.
    pub endgame_weight: f32,
    /// Weight on denial value: `denial_weight × (best opponent's projected gain
    /// from this plant)`. 0.0 disables opponent-aware bidding (easy/normal);
    /// >0.0 makes the bot pay extra to keep strong plants away from rivals.
    pub denial_weight: f32,
    /// Weight on the operating-cost term, applied as `operating_cost_weight ×
    /// fuel_feasibility × expected_firing_cost × remaining_rounds` — turns
    /// gross income into net income by charging the plant's forward,
    /// demand/replenishment-aware fuel spend over the rounds it actually runs
    /// (the unfed rounds are priced separately, by `fuel_risk`). Without this,
    /// a 1-coal plant and a thirstier 2-gas plant of equal capacity would be
    /// valued identically — fuel type would be invisible (see `expected_firing_cost`).
    pub operating_cost_weight: f32,
    /// Weight on the fuel-risk penalty, applied as `fuel_risk_weight ×
    /// (1 − fuel_feasibility) × remaining_rounds × (income_gain + fuel_price)`
    /// — a thirsty plant the player likely can't keep fed is worth less in
    /// Elektro, in proportion to the income it won't reliably earn and the
    /// absolute cost of the fuel it burns (see `fuel_feasibility`).
    pub fuel_risk_weight: f32,
    /// Weight on the replacement-waste penalty: `replacement_waste_weight ×
    /// remaining_rounds × (income the discarded plant still contributes)`.
    /// Applies only when a full-rack purchase forces a discard.
    pub replacement_waste_weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuyWeights {
    /// Fuel reserve multiplier: spend this many ×plant.cost on fuel per plant.
    pub fuel_reserve_multiplier: f32,
    /// Target rounds of fuel to hold in storage when the fuel is currently
    /// cheaper than its forward-expected price (the buy-resources stockpile
    /// pass). `1.0` (the default) disables stockpiling — the bot only buys the
    /// coming firing's worth. Higher tiers pre-buy cheap/scarce fuel up to this
    /// many rounds, capped by real storage.
    #[serde(default = "default_stockpile_rounds")]
    pub stockpile_rounds: f32,
}

fn default_stockpile_rounds() -> f32 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildWeights {
    /// Bonus for cities that opponents already occupy (0.0 = ignore, >0 = block earlier).
    pub block_weight: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BureaucracyWeights {
    /// 1.0 = always prefer oil for hybrid plants (conserves coal).
    /// 0.0 = always prefer coal for hybrids.
    pub oil_preference: f32,
}

// ---------------------------------------------------------------------------
// Profile and registry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BotProfile {
    pub display_name: String,
    /// Boltzmann temperature: 0.0 = pure argmax; higher = more random sampling.
    pub temperature: f32,
    /// Probability of applying bid jitter (0.0–1.0).
    pub jitter: f32,
    /// Maximum elektro added by bid jitter.
    pub max_jitter: u8,
    pub auction: AuctionWeights,
    pub buy: BuyWeights,
    pub build: BuildWeights,
    pub bureaucracy: BureaucracyWeights,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileRegistry {
    pub easy: BotProfile,
    pub normal: BotProfile,
    pub hard: BotProfile,
    /// Fallback/valuation profile for the RL-driven Expert difficulty — the
    /// policy itself lives in `crate::policy`, not here.
    pub expert: BotProfile,
}

impl ProfileRegistry {
    pub fn profile_for(&self, difficulty: BotDifficulty) -> &BotProfile {
        match difficulty {
            BotDifficulty::Easy => &self.easy,
            BotDifficulty::Normal => &self.normal,
            BotDifficulty::Hard => &self.hard,
            BotDifficulty::Expert => &self.expert,
        }
    }
}

/// Parse the compiled-in `assets/bots/default.toml`, ignoring any
/// `BOT_PROFILES_FILE` override. Use this when you specifically want the
/// pristine shipped profiles — e.g. the evolutionary search reads `hard` as its
/// init mean and `normal` as the fixed opponent yardstick, and must not have
/// those perturbed by whatever champion file the env var points at.
///
/// Panics on malformed embedded data (a compile-time asset, so this is a
/// build/programmer error, not runtime input).
pub fn embedded_registry() -> ProfileRegistry {
    toml::from_str(DEFAULT_PROFILES_TOML).expect("invalid default bot profiles TOML")
}

/// The process-wide bot profile registry.
///
/// Resolved once, at first use, and cached for the lifetime of the process:
/// if `BOT_PROFILES_FILE` names a readable, valid TOML file it wins; otherwise
/// the compiled-in `assets/bots/default.toml` is used. Caching also removes the
/// per-decision `toml::from_str` cost the bridge used to pay on every bot move.
///
/// Because the override is read only once, set `BOT_PROFILES_FILE` before the
/// first bot decision (i.e. at process start). This is how an evolved champion
/// profile is deployed to the lobby, client, and Python eval without recompiling.
pub fn default_registry() -> &'static ProfileRegistry {
    static REGISTRY: OnceLock<ProfileRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| resolve_registry(std::env::var_os("BOT_PROFILES_FILE")))
}

/// Pure resolution of the registry from an optional override path: read+parse
/// the file if given and valid, otherwise fall back to the embedded defaults
/// (logging why). Factored out of [`default_registry`] so the override logic is
/// testable without touching the process-global `OnceLock` or env vars.
fn resolve_registry(override_path: Option<std::ffi::OsString>) -> ProfileRegistry {
    let Some(path) = override_path else {
        return embedded_registry();
    };
    match std::fs::read_to_string(&path) {
        Ok(contents) => match toml::from_str(&contents) {
            Ok(registry) => {
                tracing::info!("loaded bot profiles from {:?}", path);
                registry
            }
            Err(e) => {
                tracing::error!(
                    "BOT_PROFILES_FILE {:?} is not valid profile TOML ({e}); \
                     falling back to embedded defaults",
                    path
                );
                embedded_registry()
            }
        },
        Err(e) => {
            tracing::error!(
                "could not read BOT_PROFILES_FILE {:?} ({e}); \
                 falling back to embedded defaults",
                path
            );
            embedded_registry()
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_profiles_parse_correctly() {
        let registry = embedded_registry();
        assert_eq!(registry.easy.display_name, "Easy");
        assert_eq!(registry.normal.display_name, "Normal");
        assert_eq!(registry.hard.display_name, "Hard");
        assert_eq!(registry.expert.display_name, "Expert");
    }

    #[test]
    fn resolve_registry_none_uses_embedded() {
        let reg = resolve_registry(None);
        assert_eq!(reg.hard.display_name, "Hard");
    }

    #[test]
    fn resolve_registry_reads_override_file() {
        // Round-trip the embedded registry with one distinctive edit, then
        // confirm the override path is read and parsed (not the embedded copy).
        let mut reg = embedded_registry();
        reg.hard.auction.city_reserve = 12345.0;
        let toml = toml::to_string(&reg).unwrap();
        let path =
            std::env::temp_dir().join(format!("pg_profiles_override_{}.toml", std::process::id()));
        std::fs::write(&path, toml).unwrap();

        let loaded = resolve_registry(Some(path.clone().into_os_string()));
        assert_eq!(loaded.hard.auction.city_reserve, 12345.0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn resolve_registry_bad_path_falls_back() {
        let reg = resolve_registry(Some("/nonexistent/does/not/exist.toml".into()));
        // Falls back to embedded rather than panicking.
        assert_eq!(reg.hard.display_name, "Hard");
        assert_eq!(reg.hard.auction.city_reserve, 30.0);
    }

    #[test]
    fn normal_profile_auction_weights() {
        let registry = embedded_registry();
        let w = &registry.normal.auction;
        assert_eq!(w.city_reserve, 30.0);
        assert_eq!(w.safety_buffer, 5.0);
        assert_eq!(w.upgrade_margin, 20.0);
        assert_eq!(w.min_open_score, 20.0);
        assert_eq!(w.buildable_lookahead, 2);
        // Normal is opponent-blind: denial is a hard-only feature.
        assert_eq!(w.denial_weight, 0.0);
        assert_eq!(w.operating_cost_weight, 1.0);
        assert_eq!(registry.normal.jitter, 0.3);
        assert_eq!(registry.normal.max_jitter, 3);
        assert_eq!(registry.normal.buy.fuel_reserve_multiplier, 4.0);
    }

    #[test]
    fn hard_profile_has_nonzero_opponent_features() {
        let registry = embedded_registry();
        let w = &registry.hard.auction;
        assert!(w.denial_weight > 0.0, "hard should value denying opponents");
        assert!(w.endgame_weight > 0.0, "hard should value endgame capacity");
        assert!(w.fuel_risk_weight > 0.0, "hard should price fuel risk");
        assert!(registry.hard.build.block_weight > 0.0);
    }

    #[test]
    fn easy_and_normal_profiles_are_opponent_blind() {
        let registry = embedded_registry();
        // Denial requires reasoning about every opponent's board state — reserved
        // for the hard tier. Easy/normal must keep it disabled.
        assert_eq!(registry.easy.auction.denial_weight, 0.0);
        assert_eq!(registry.normal.auction.denial_weight, 0.0);
    }

    #[test]
    fn weight_tiers_escalate_with_difficulty() {
        let registry = embedded_registry();
        let (easy, normal, hard) = (
            &registry.easy.auction,
            &registry.normal.auction,
            &registry.hard.auction,
        );
        // Net-income thinking is fundamental economics, not a sophistication
        // tier — every difficulty nets fuel operating cost out the same way.
        assert_eq!(easy.operating_cost_weight, normal.operating_cost_weight);
        assert_eq!(normal.operating_cost_weight, hard.operating_cost_weight);
        // Harder bots price fuel risk and replacement waste more aggressively...
        assert!(easy.fuel_risk_weight <= normal.fuel_risk_weight);
        assert!(normal.fuel_risk_weight <= hard.fuel_risk_weight);
        assert!(easy.replacement_waste_weight <= normal.replacement_waste_weight);
        assert!(normal.replacement_waste_weight <= hard.replacement_waste_weight);
        // ...and plan capacity further ahead of their current city count (lower
        // lookahead = tighter ceiling = more disciplined).
        assert!(hard.buildable_lookahead <= normal.buildable_lookahead);
        assert!(normal.buildable_lookahead <= easy.buildable_lookahead);
    }
}
