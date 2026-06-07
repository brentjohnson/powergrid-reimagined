use powergrid_core::{
    rules::replenishment_amounts,
    state::GameState,
    types::{PlantKind, Player, PowerPlant, Resource, ResourceMarket},
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

// ---------------------------------------------------------------------------
// Resource-market helpers
// ---------------------------------------------------------------------------

/// Per-round fuel pressure for one resource, roughly in [0, 1+].
///
/// Combines three factors the user requested:
/// - **availability**: low market stock → denominator shrinks → scarcity rises
/// - **demand**: total units all players' plants need each round → shortfall rises
/// - **replenishment rate**: at this step/player-count → shortfall rises when slow
///
/// `GasOrOil` hybrids contribute `plant.cost / 2` to *each* of gas and oil
/// (caller is responsible for choosing which resource to query).
pub fn fuel_scarcity(state: &GameState, resource: Resource) -> f32 {
    let avail = state.resources.available(resource) as f32;
    let (coal_r, oil_r, gas_r, uranium_r) = replenishment_amounts(state.step, state.players.len());
    let replen = match resource {
        Resource::Coal => coal_r,
        Resource::Oil => oil_r,
        Resource::Gas => gas_r,
        Resource::Uranium => uranium_r,
    } as f32;

    // Sum per-round demand across every player's installed plants.
    let mut demand = 0.0f32;
    for player in &state.players {
        for plant in &player.plants {
            let cost = plant.cost as f32;
            match plant.kind {
                // Hybrid splits half/half between gas and oil.
                PlantKind::GasOrOil => match resource {
                    Resource::Gas | Resource::Oil => demand += cost / 2.0,
                    _ => {}
                },
                // Pure-fuel plant: only contributes if its resource matches.
                _ => {
                    if plant.kind.resources().contains(&resource) {
                        demand += cost;
                    }
                }
            }
        }
    }

    let shortfall = (demand - replen).max(0.0);
    // Denominator +1 prevents division by zero when both avail and replen are 0.
    shortfall / (avail + replen + 1.0)
}

/// Fuel scarcity for the resource a specific plant would burn.
///
/// - Wind (no fuel) → 0.0
/// - `GasOrOil` → min(gas_scarcity, oil_scarcity): the bot will buy whichever
///   is cheaper, so use the easier resource's pressure.
/// - All other kinds → scarcity of their single resource.
pub fn plant_fuel_scarcity(plant: &PowerPlant, state: &GameState) -> f32 {
    match plant.kind {
        PlantKind::Wind => 0.0,
        PlantKind::GasOrOil => {
            fuel_scarcity(state, Resource::Gas).min(fuel_scarcity(state, Resource::Oil))
        }
        _ => plant
            .kind
            .resources()
            .first()
            .map(|&r| fuel_scarcity(state, r))
            .unwrap_or(0.0),
    }
}

/// Estimated elektro cost to buy one full firing (`plant.cost` units) at
/// current market prices.
///
/// - Wind → 0.
/// - `GasOrOil` → prices the more-available (cheaper) resource.
/// - If the market cannot supply the full amount from any viable resource,
///   prices what is available and charges 9 elektro per missing unit (the
///   maximum slot price across all resource tables) as a conservative overrun.
pub fn estimate_firing_cost(plant: &PowerPlant, market: &ResourceMarket) -> u32 {
    if !plant.kind.needs_resources() {
        return 0;
    }
    let amount = plant.cost;

    // Determine the order of resources to try. Hybrids prefer the more-available one.
    let resources: Vec<Resource> = match plant.kind {
        PlantKind::GasOrOil => {
            if market.available(Resource::Gas) >= market.available(Resource::Oil) {
                vec![Resource::Gas, Resource::Oil]
            } else {
                vec![Resource::Oil, Resource::Gas]
            }
        }
        _ => plant.kind.resources(),
    };

    // Return the cost from the first resource that can fully supply the amount.
    for &resource in &resources {
        if let Some(cost) = market.price(resource, amount) {
            return cost;
        }
    }

    // Fallback: market too depleted to supply the full amount from any resource.
    // Price what's available from the preferred resource and charge max rate for the rest.
    let resource = resources[0];
    let avail = market.available(resource);
    let avail_cost = market.price(resource, avail).unwrap_or(0);
    let shortfall = amount.saturating_sub(avail) as u32;
    // 9 is the maximum per-unit price across all resource price tables.
    avail_cost + shortfall * 9
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

    // Penalise capacity that would overshoot the useful ceiling *after* buying this plant.
    // Uses capacity_bump (net gain, accounting for any discard) so a full-rack upgrade that
    // actually shrinks capacity produces zero or negative projected surplus.
    if w.overshoot_weight > 0.0 {
        let powerable: i32 = player.plants.iter().map(|p| p.cities as i32).sum();
        let target = useful_city_target(player, state, w) as i32;
        let projected_surplus = (powerable + capacity_bump(plant, player, w) - target).max(0);
        score -= w.overshoot_weight * projected_surplus as f32;
    }

    // Penalise plants whose fuel is scarce, heavily demanded, or slowly replenished.
    // Scaled by plant.cost so a thirsty plant on a contested resource loses more.
    if w.fuel_scarcity_weight > 0.0 && plant.kind.needs_resources() {
        score -= w.fuel_scarcity_weight * plant_fuel_scarcity(plant, state) * plant.cost as f32;
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

/// Cities worth owning given current progress and the rules of the game.
/// = min(owned + buildable_lookahead, end_game_cities).
/// Capacity beyond this is either surplus we already have planned for (lookahead) or pure waste
/// (beyond the game-ending threshold).
pub fn useful_city_target(player: &Player, state: &GameState, w: &AuctionWeights) -> u8 {
    let owned = state.player_city_count(player.id) as u8;
    (owned + w.buildable_lookahead).min(state.end_game_cities)
}

/// True when acquiring a new plant would give little or no benefit (full rack, low upgrade margin).
/// Capacity overshoot (capacity > useful ceiling) is handled by `plant_score_contextual`.
pub fn should_skip_auction(player: &Player, candidate: &PowerPlant, w: &AuctionWeights) -> bool {
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
///
/// When `market` is `Some`, uses live market prices (`estimate_firing_cost`) to
/// value one firing of each plant, so scarce fuels inflate the reserve and push the
/// bid ceiling down.  When `None`, falls back to the static `plant.cost ×
/// fuel_reserve_multiplier` estimate (used by isolated unit tests).
pub fn auction_reserve(
    plant: &PowerPlant,
    player: &Player,
    w: &AuctionWeights,
    buy: &BuyWeights,
    market: Option<&ResourceMarket>,
) -> u32 {
    let mut reserve = 0u32;
    for p in &player.plants {
        if p.kind.needs_resources() {
            let fuel_cost = match market {
                Some(m) => estimate_firing_cost(p, m) as f32,
                None => p.cost as f32,
            };
            reserve += (fuel_cost * buy.fuel_reserve_multiplier) as u32;
        }
    }
    if plant.kind.needs_resources() {
        let fuel_cost = match market {
            Some(m) => estimate_firing_cost(plant, m) as f32,
            None => plant.cost as f32,
        };
        reserve += (fuel_cost * buy.fuel_reserve_multiplier) as u32;
    }
    reserve += w.city_reserve as u32;
    reserve += w.safety_buffer as u32;
    reserve
}

/// Deterministic bid ceiling for a plant.
/// `min_bid` is the effective minimum (1 if the discount token is on this plant,
/// else the printed plant number).
///
/// Pass `market = Some(&state.resources)` in production to have the reserve
/// calculated from live market prices (scarce fuel → higher reserve → lower
/// ceiling).  Pass `None` to use the flat `plant.cost × multiplier` estimate
/// (preserves exact values expected by isolated unit tests).
pub fn bid_ceiling(
    plant: &PowerPlant,
    player: &Player,
    round: u32,
    w: &AuctionWeights,
    buy: &BuyWeights,
    min_bid: u32,
    market: Option<&ResourceMarket>,
) -> u32 {
    let listed = min_bid;
    let reserve = auction_reserve(plant, player, w, buy, market);

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
