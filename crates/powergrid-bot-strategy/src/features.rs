use powergrid_core::{
    state::GameState,
    types::{PlantKind, Player, PowerPlant},
};

use crate::profile::{AuctionWeights, BuyWeights};

// ---------------------------------------------------------------------------
// Plant helpers
// ---------------------------------------------------------------------------

/// Graduated step count for the high-capacity premium.
/// Returns 0 below the threshold, 1 at the threshold, +1 per extra city.
fn high_capacity_steps(cities: u8, threshold: u8) -> u32 {
    if cities < threshold {
        0
    } else {
        (cities - threshold + 1) as u32
    }
}

pub fn is_green(plant: &PowerPlant) -> bool {
    matches!(plant.kind, PlantKind::Wind)
}

/// Base desirability score for a plant, using profile weights.
pub fn plant_score(plant: &PowerPlant, w: &AuctionWeights) -> f32 {
    let city_value = plant.cities as f32 * w.cities_weight;
    let fuel_bonus = if is_green(plant) { w.green_bonus } else { 0.0 };
    let efficiency = if plant.cost == 0 {
        30.0
    } else {
        (plant.cities as f32 * w.efficiency_weight) / plant.cost as f32
    };
    let high_cap =
        w.high_capacity_bonus * high_capacity_steps(plant.cities, w.high_capacity_threshold) as f32;
    city_value + fuel_bonus + efficiency + high_cap
}

/// Score for a plant candidate including hard-only context features.
/// For normal/easy profiles, the extra weights are 0.0 so only the base score matters.
pub fn plant_score_contextual(
    plant: &PowerPlant,
    player: &Player,
    state: &GameState,
    w: &AuctionWeights,
) -> f32 {
    let mut score = plant_score(plant, w);

    if w.opponent_gap_weight > 0.0 {
        let my_cities = state.player_city_count(player.id) as f32;
        let max_opp = state
            .players
            .iter()
            .filter(|p| p.id != player.id)
            .map(|p| state.player_city_count(p.id) as f32)
            .fold(0.0f32, f32::max);
        score += w.opponent_gap_weight * (max_opp - my_cities).max(0.0);
    }

    if w.endgame_weight > 0.0 {
        let max_cities = state
            .players
            .iter()
            .map(|p| state.player_city_count(p.id) as u32)
            .max()
            .unwrap_or(0);
        let proximity = max_cities as f32 / state.end_game_cities as f32;
        score += w.endgame_weight * proximity;
    }

    if w.pipeline_weight > 0.0 && !state.market.future.is_empty() {
        let future_avg: f32 = state
            .market
            .future
            .iter()
            .map(|p| plant_score(p, w))
            .sum::<f32>()
            / state.market.future.len() as f32;
        let base = plant_score(plant, w);
        score += w.pipeline_weight * (base - future_avg).max(0.0);
    }

    if w.upgrade_efficiency_weight > 0.0 {
        let bump = capacity_bump(plant, player, w) as f32;
        score += (bump - plant.cities as f32) * w.upgrade_efficiency_weight;
    }

    score
}

/// Net cities-powered capacity gained by acquiring `plant`.
/// When the rack is full (3 plants) we'd discard the worst — bump = new minus worst.
pub fn capacity_bump(plant: &PowerPlant, player: &Player, w: &AuctionWeights) -> i32 {
    if player.plants.len() < 3 {
        return plant.cities as i32;
    }
    let worst_cities = player
        .plants
        .iter()
        .min_by(|a, b| {
            plant_score(a, w)
                .partial_cmp(&plant_score(b, w))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|p| p.cities as i32)
        .unwrap_or(0);
    plant.cities as i32 - worst_cities
}

/// True when acquiring a new plant would give little or no benefit.
pub fn should_skip_auction(
    player: &Player,
    candidate: &PowerPlant,
    w: &AuctionWeights,
    city_count: usize,
) -> bool {
    let powerable: u8 = player.plants.iter().map(|p| p.cities).sum();
    if powerable > city_count as u8 {
        return true;
    }
    if player.plants.len() >= 3 {
        if let Some(worst) = player.plants.iter().min_by(|a, b| {
            plant_score(a, w)
                .partial_cmp(&plant_score(b, w))
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            if plant_score(candidate, w) - plant_score(worst, w) < w.upgrade_margin {
                return true;
            }
        }
    }
    false
}

/// Cash to keep in reserve after winning an auction: fuel for all plants plus city builds.
pub fn auction_reserve(
    plant: &PowerPlant,
    player: &Player,
    w: &AuctionWeights,
    buy: &BuyWeights,
) -> u32 {
    let mut reserve = 0u32;
    for p in &player.plants {
        if p.kind.needs_resources() {
            reserve += (p.cost as f32 * buy.fuel_reserve_multiplier) as u32;
        }
    }
    if plant.kind.needs_resources() {
        reserve += (plant.cost as f32 * buy.fuel_reserve_multiplier) as u32;
    }
    reserve += w.city_reserve as u32;
    reserve += w.safety_buffer as u32;
    reserve
}

/// Deterministic bid ceiling for a plant.
/// `min_bid` is the effective minimum (1 if the discount token is on this plant,
/// else the printed plant number).
pub fn bid_ceiling(
    plant: &PowerPlant,
    player: &Player,
    round: u32,
    w: &AuctionWeights,
    buy: &BuyWeights,
    min_bid: u32,
) -> u32 {
    let listed = min_bid;
    let reserve = auction_reserve(plant, player, w, buy);

    let raw_ceiling = if round == 1 {
        listed
    } else {
        let bump = capacity_bump(plant, player, w);
        let premium = if bump > 0 {
            let base = bump as u32 * w.capacity_premium as u32;
            let high_cap = (w.high_capacity_bid_premium
                * high_capacity_steps(plant.cities, w.high_capacity_threshold) as f32)
                as u32;
            base + high_cap
        } else {
            0
        };
        let affordable = player.money.saturating_sub(reserve);
        (listed + premium).min(affordable).max(listed)
    };

    raw_ceiling.min(player.money)
}

/// Bonus for building in a contested city (already occupied by opponents).
pub fn city_contest_bonus(owner_count: usize, block_weight: f32) -> f32 {
    if block_weight <= 0.0 || owner_count == 0 {
        0.0
    } else {
        block_weight * owner_count as f32
    }
}
