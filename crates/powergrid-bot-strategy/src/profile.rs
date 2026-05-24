use powergrid_core::types::BotDifficulty;
use serde::Deserialize;

// Embedded at compile time; override via BOT_PROFILES_FILE env var at startup
// (not yet wired — added for future runtime customisation).
const DEFAULT_PROFILES_TOML: &str = include_str!("../../../assets/bots/default.toml");

// ---------------------------------------------------------------------------
// Weight structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct AuctionWeights {
    pub cities_weight: f32,
    pub green_bonus: f32,
    pub efficiency_weight: f32,
    /// Fuel reserve per resource-consuming plant (multiplied by plant.cost).
    pub city_reserve: f32,
    pub safety_buffer: f32,
    /// Minimum plant-score improvement to justify replacing a rack plant.
    pub upgrade_margin: f32,
    /// Minimum plant score to be worth opening an auction for.
    pub min_open_score: f32,
    /// Extra elektro per city of capacity gained when computing bid ceiling.
    pub capacity_premium: f32,
    /// Capacity threshold at/above which the high-capacity premium activates.
    pub high_capacity_threshold: u8,
    /// Score-space bonus per step at/above the threshold (graduated). 0.0 disables.
    pub high_capacity_bonus: f32,
    /// Elektro added to the bid ceiling per step at/above the threshold. 0.0 disables.
    pub high_capacity_bid_premium: f32,
    // Hard-only features (0.0 in easy/normal):
    pub opponent_gap_weight: f32,
    pub endgame_weight: f32,
    pub pipeline_weight: f32,
    /// Penalty per city lost when forced to discard at full capacity.
    /// Applied as (capacity_bump - plant.cities) * weight — always ≤ 0 when rack is full.
    pub upgrade_efficiency_weight: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BuyWeights {
    /// Fuel reserve multiplier: spend this many ×plant.cost on fuel per plant.
    pub fuel_reserve_multiplier: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BuildWeights {
    /// Bonus for cities that opponents already occupy (0.0 = ignore, >0 = block earlier).
    pub block_weight: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BureaucracyWeights {
    /// 1.0 = always prefer oil for hybrid plants (conserves coal).
    /// 0.0 = always prefer coal for hybrids.
    pub oil_preference: f32,
}

// ---------------------------------------------------------------------------
// Profile and registry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
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

#[derive(Debug, Clone, Deserialize)]
pub struct ProfileRegistry {
    pub easy: BotProfile,
    pub normal: BotProfile,
    pub hard: BotProfile,
}

impl ProfileRegistry {
    pub fn profile_for(&self, difficulty: BotDifficulty) -> &BotProfile {
        match difficulty {
            BotDifficulty::Easy => &self.easy,
            BotDifficulty::Normal => &self.normal,
            BotDifficulty::Hard => &self.hard,
        }
    }
}

pub fn default_registry() -> ProfileRegistry {
    toml::from_str(DEFAULT_PROFILES_TOML).expect("invalid default bot profiles TOML")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_profiles_parse_correctly() {
        let registry = default_registry();
        assert_eq!(registry.easy.display_name, "Easy");
        assert_eq!(registry.normal.display_name, "Normal");
        assert_eq!(registry.hard.display_name, "Hard");
    }

    #[test]
    fn normal_profile_matches_legacy_constants() {
        let registry = default_registry();
        let w = &registry.normal.auction;
        // Verify the normal profile reproduces the original magic numbers.
        assert_eq!(w.cities_weight, 15.0);
        assert_eq!(w.green_bonus, 25.0);
        assert_eq!(w.efficiency_weight, 10.0);
        assert_eq!(w.city_reserve, 30.0);
        assert_eq!(w.safety_buffer, 5.0);
        assert_eq!(w.upgrade_margin, 10.0);
        assert_eq!(w.min_open_score, 20.0);
        assert_eq!(w.capacity_premium, 2.0);
        assert_eq!(w.opponent_gap_weight, 0.0);
        assert_eq!(w.endgame_weight, 0.0);
        assert_eq!(registry.normal.jitter, 0.3);
        assert_eq!(registry.normal.max_jitter, 3);
        assert_eq!(registry.normal.buy.fuel_reserve_multiplier, 4.0);
    }

    #[test]
    fn hard_profile_has_nonzero_opponent_features() {
        let registry = default_registry();
        let w = &registry.hard.auction;
        assert!(w.opponent_gap_weight > 0.0);
        assert!(w.endgame_weight > 0.0);
        assert!(w.pipeline_weight > 0.0);
        assert!(registry.hard.build.block_weight > 0.0);
    }

    #[test]
    fn high_capacity_premium_tiers() {
        let registry = default_registry();
        assert!(
            registry.normal.auction.high_capacity_bonus > 0.0,
            "normal should have high-capacity score bonus"
        );
        assert!(
            registry.normal.auction.high_capacity_bid_premium > 0.0,
            "normal should have high-capacity bid premium"
        );
        assert!(
            registry.hard.auction.high_capacity_bonus > 0.0,
            "hard should have high-capacity score bonus"
        );
        assert_eq!(
            registry.easy.auction.high_capacity_bonus, 0.0,
            "easy should have no high-capacity bonus"
        );
        assert_eq!(
            registry.easy.auction.high_capacity_bid_premium, 0.0,
            "easy should have no high-capacity bid premium"
        );
    }
}
