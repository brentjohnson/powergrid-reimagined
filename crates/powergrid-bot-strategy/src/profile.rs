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
    /// Weight on the fuel-risk penalty, applied as
    /// `fuel_risk_weight × plant_fuel_scarcity × plant.cost × remaining_rounds`
    /// — a thirsty plant on a scarce/contested resource is worth less in Elektro.
    pub fuel_risk_weight: f32,
    /// Weight on the replacement-waste penalty: `replacement_waste_weight ×
    /// remaining_rounds × (income the discarded plant still contributes)`.
    /// Applies only when a full-rack purchase forces a discard.
    pub replacement_waste_weight: f32,
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
    fn normal_profile_auction_weights() {
        let registry = default_registry();
        let w = &registry.normal.auction;
        assert_eq!(w.city_reserve, 30.0);
        assert_eq!(w.safety_buffer, 5.0);
        assert_eq!(w.upgrade_margin, 20.0);
        assert_eq!(w.min_open_score, 20.0);
        assert_eq!(w.buildable_lookahead, 2);
        // Normal is opponent-blind: denial is a hard-only feature.
        assert_eq!(w.denial_weight, 0.0);
        assert_eq!(registry.normal.jitter, 0.3);
        assert_eq!(registry.normal.max_jitter, 3);
        assert_eq!(registry.normal.buy.fuel_reserve_multiplier, 4.0);
    }

    #[test]
    fn hard_profile_has_nonzero_opponent_features() {
        let registry = default_registry();
        let w = &registry.hard.auction;
        assert!(w.denial_weight > 0.0, "hard should value denying opponents");
        assert!(w.endgame_weight > 0.0, "hard should value endgame capacity");
        assert!(w.fuel_risk_weight > 0.0, "hard should price fuel risk");
        assert!(registry.hard.build.block_weight > 0.0);
    }

    #[test]
    fn easy_and_normal_profiles_are_opponent_blind() {
        let registry = default_registry();
        // Denial requires reasoning about every opponent's board state — reserved
        // for the hard tier. Easy/normal must keep it disabled.
        assert_eq!(registry.easy.auction.denial_weight, 0.0);
        assert_eq!(registry.normal.auction.denial_weight, 0.0);
    }

    #[test]
    fn weight_tiers_escalate_with_difficulty() {
        let registry = default_registry();
        let (easy, normal, hard) = (
            &registry.easy.auction,
            &registry.normal.auction,
            &registry.hard.auction,
        );
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
