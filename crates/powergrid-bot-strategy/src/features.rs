use powergrid_core::{
    rules::replenishment_amounts,
    state::GameState,
    types::{income_for, PlantKind, Player, PowerPlant, Resource, ResourceMarket},
};

use crate::profile::{AuctionWeights, BuyWeights};

// ---------------------------------------------------------------------------
// Plant helpers
// ---------------------------------------------------------------------------

/// Simple ranking score for an owned plant — used only to identify the "worst"
/// plant when the rack is full (capacity_bump / discard / replacement decisions).
/// Higher cities-powered ranks better; among equal capacity, cheaper-to-fire
/// (lower resource cost) ranks better. Deliberately weight-free: it's a tie-break
/// heuristic, not a valuation — `evaluate_plant` is the source of truth for value.
pub fn plant_score(plant: &PowerPlant) -> f32 {
    plant.cities as f32 * 100.0 - plant.cost as f32
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
    fuel_scarcity_with_extra(state, resource, 0.0)
}

/// `fuel_scarcity`, plus an `extra_demand` (in fuel units/round) folded into the
/// shortfall before scoring. Lets a not-yet-owned candidate plant's own appetite
/// be weighed against the existing market — see `plant_fuel_scarcity`, which is
/// the only caller that passes a nonzero `extra_demand`.
fn fuel_scarcity_with_extra(state: &GameState, resource: Resource, extra_demand: f32) -> f32 {
    let avail = state.resources.available(resource) as f32;
    let (coal_r, oil_r, gas_r, uranium_r) = replenishment_amounts(state.step, state.players.len());
    let replen = match resource {
        Resource::Coal => coal_r,
        Resource::Oil => oil_r,
        Resource::Gas => gas_r,
        Resource::Uranium => uranium_r,
    } as f32;

    // Sum per-round demand across every player's installed plants.
    let mut demand = extra_demand;
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

/// Fuel scarcity for the resource a specific plant would burn — *including that
/// plant's own per-round demand* in the shortfall. This matters most for a candidate
/// being evaluated in an auction: it isn't in `state.players[..].plants` yet, so
/// without folding its appetite in here, `fuel_scarcity` would be blind to the very
/// thing that makes a thirsty plant risky (e.g. a 2-uranium plant when uranium
/// replenishes at 1/round and the market is nearly empty would otherwise score 0).
///
/// - Wind (no fuel) → 0.0
/// - `GasOrOil` → min(gas_scarcity, oil_scarcity), each charged half the plant's
///   cost as extra demand (mirrors the hybrid split `fuel_scarcity` already applies
///   to owned plants): the bot will buy whichever is cheaper, so use the easier
///   resource's pressure.
/// - All other kinds → scarcity of their single resource, charged the full cost.
pub fn plant_fuel_scarcity(plant: &PowerPlant, state: &GameState) -> f32 {
    match plant.kind {
        PlantKind::Wind => 0.0,
        PlantKind::GasOrOil => {
            let half = plant.cost as f32 / 2.0;
            fuel_scarcity_with_extra(state, Resource::Gas, half).min(fuel_scarcity_with_extra(
                state,
                Resource::Oil,
                half,
            ))
        }
        _ => plant
            .kind
            .resources()
            .first()
            .map(|&r| fuel_scarcity_with_extra(state, r, plant.cost as f32))
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

// ---------------------------------------------------------------------------
// Capacity helpers
// ---------------------------------------------------------------------------

/// Net cities-powered capacity gained by acquiring `plant`.
/// When the rack is full (3 plants) we'd discard the worst — bump = new minus worst.
pub fn capacity_bump(plant: &PowerPlant, player: &Player) -> i32 {
    if player.plants.len() < 3 {
        return plant.cities as i32;
    }
    let worst_cities = player
        .plants
        .iter()
        .min_by(|a, b| {
            plant_score(a)
                .partial_cmp(&plant_score(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|p| p.cities as i32)
        .unwrap_or(0);
    plant.cities as i32 - worst_cities
}

/// Cities worth owning given current progress and the rules of the game.
/// = min(owned + buildable_lookahead, end_game_cities).
/// Capacity beyond this is either surplus we already have planned for (lookahead)
/// or pure waste (beyond the game-ending threshold). Used to cap projected income
/// so capacity nobody can use before the game ends doesn't inflate a plant's value.
pub fn useful_city_target(player: &Player, state: &GameState, w: &AuctionWeights) -> u8 {
    let owned = state.player_city_count(player.id) as u8;
    (owned + w.buildable_lookahead).min(state.end_game_cities)
}

/// True when acquiring `candidate` would give little or no benefit: the rack is
/// full and the plant's total Elektro value — which already nets out the forced
/// discard via `capacity_bump` — doesn't clear the upgrade margin.
pub fn should_skip_auction(
    player: &Player,
    candidate: &PowerPlant,
    state: &GameState,
    w: &AuctionWeights,
) -> bool {
    if player.plants.len() >= 3 {
        return evaluate_plant(candidate, player, state, w).total < w.upgrade_margin;
    }
    false
}

// ---------------------------------------------------------------------------
// Elektro-denominated plant valuation
// ---------------------------------------------------------------------------
//
// LOGIC.md's central idea: stop asking "how good is this plant?" and ask "how
// many Elektro is this plant worth *right now, in this exact game state*?" Every
// term below is converted into the same currency — expected future Elektro — so
// they can be summed, compared, and used directly as a bid ceiling:
//
//     PlantValue ≈ IncrementalIncome + FuelSavings + CapacityPremium + Denial
//                   - FuelRisk - ReplacementWaste
//     MaximumBid  = PlantValue

/// Rough estimate of how many productive rounds remain before the game ends.
/// Power Grid ends once a player reaches `end_game_cities`; the gap between the
/// leader and that threshold — divided by a typical ~1.5-cities-per-round build
/// pace — gives a serviceable horizon for discounting future income and risk.
/// Clamped to `[1, 8]` so neither the opening nor the closing turns produce
/// degenerate (zero or unbounded) valuations. (LOGIC.md §4B "How many rounds remain?")
pub fn remaining_rounds(state: &GameState) -> f32 {
    let leader_cities = state
        .players
        .iter()
        .map(|p| state.player_city_count(p.id) as f32)
        .fold(0.0f32, f32::max);
    let gap = (state.end_game_cities as f32 - leader_cities).max(0.0);
    (gap / 1.5).clamp(1.0, 8.0)
}

/// Projected per-round income gain from acquiring `plant`, and the net capacity
/// bump it represents. Powered-city counts are capped at `useful_city_target` so
/// capacity beyond what the player can realistically build into doesn't inflate
/// the valuation (LOGIC.md's "overshoot" concern, expressed directly in income).
fn projected_income_gain(
    plant: &PowerPlant,
    player: &Player,
    state: &GameState,
    w: &AuctionWeights,
) -> (f32, i32) {
    let bump = capacity_bump(plant, player);
    let target = useful_city_target(player, state, w) as i32;
    let current_capacity: i32 = player.plants.iter().map(|p| p.cities as i32).sum();
    let old_powered = current_capacity.clamp(0, target) as u8;
    let new_powered = (current_capacity + bump).clamp(0, target) as u8;
    let gain = income_for(new_powered) as f32 - income_for(old_powered) as f32;
    (gain, bump)
}

/// Bonus for capacity that helps close the gap to the game-ending city count
/// (LOGIC.md §3 "Capacity Premium" / §6C "Endgame Thresholds"). Scales with how
/// close the leader is to triggering game end: early on, raw capacity is cheap to
/// rearrange later; late game, crossing the threshold first decides the win.
fn endgame_capacity_premium(
    player: &Player,
    state: &GameState,
    w: &AuctionWeights,
    bump: i32,
) -> f32 {
    if w.endgame_weight <= 0.0 || bump <= 0 {
        return 0.0;
    }
    let end = state.end_game_cities as f32;
    let current_capacity: f32 = player.plants.iter().map(|p| p.cities as f32).sum();
    let new_capacity = current_capacity + bump as f32;

    // How much of the gap to the game-ending capacity threshold this purchase
    // closes (capped at the bump itself — overshooting the threshold doesn't
    // earn extra credit for the cities beyond it).
    let old_gap = (end - current_capacity).max(0.0);
    let new_gap = (end - new_capacity).max(0.0);
    let closed = (old_gap - new_gap).clamp(0.0, bump as f32);

    // The premium matters more as the leader nears the end-game city count —
    // early game, raw capacity is cheap to reorganize; late game it decides the win.
    let leader = state
        .players
        .iter()
        .map(|p| state.player_city_count(p.id) as f32)
        .fold(0.0f32, f32::max);
    let proximity = (leader / end).clamp(0.0, 1.0);

    w.endgame_weight * proximity * closed
}

/// Non-recursive estimate of how much `plant` would be worth to `opponent` —
/// income gain plus endgame premium only, deliberately excluding denial so the
/// term can't recurse through `evaluate_plant` (LOGIC.md §7 "Denial Value").
fn opponent_gain(
    plant: &PowerPlant,
    opponent: &Player,
    state: &GameState,
    w: &AuctionWeights,
) -> f32 {
    let (income_gain, bump) = projected_income_gain(plant, opponent, state, w);
    let premium = endgame_capacity_premium(opponent, state, w, bump);
    (income_gain * remaining_rounds(state) + premium).max(0.0)
}

/// Elektro-denominated breakdown of a plant's worth to a specific player, right
/// now, in this exact game state — the model described in LOGIC.md. Every term is
/// expressed in the same currency (expected future Elektro) so they sum directly,
/// and `total` is a principled bid ceiling: `MaximumBid = PlantValue`.
#[derive(Debug, Clone, Copy, Default)]
pub struct PlantValuation {
    /// Extra income earned over the rest of the game from powering more cities.
    /// (LOGIC.md §1 "Incremental Income")
    pub incremental_income: f32,
    /// Elektro saved (or lost, if negative) per firing vs. the plant this would
    /// replace. Zero unless the rack is full. (LOGIC.md §2 "Fuel Savings")
    pub fuel_savings: f32,
    /// Bonus for capacity that helps close the gap to the game-ending city count.
    /// (LOGIC.md §3 "Capacity Premium")
    pub capacity_premium: f32,
    /// Value of denying the plant to the opponent who'd benefit most from it.
    /// Zero unless `denial_weight > 0` (hard-only). (LOGIC.md §7 "Denial Value")
    pub denial: f32,
    /// Penalty for relying on fuel that's scarce, contested, or slow to
    /// replenish. (LOGIC.md §6 "Resource Risk")
    pub fuel_risk: f32,
    /// Value thrown away on a forced full-rack discard of a plant that still had
    /// useful income left to earn. (LOGIC.md §8 "Replacement Waste")
    pub replacement_waste: f32,
    /// Sum of every positive term above minus `fuel_risk` and
    /// `replacement_waste`, floored at 0 — a plant is never worth bidding
    /// *negative* Elektro for.
    pub total: f32,
}

/// Evaluate `plant` for `player` in expected-future-Elektro terms — "how many
/// Elektro is this plant worth right now, in this exact game state?" (LOGIC.md).
/// Bots follow the rule `MaximumBid = PlantValue = evaluate_plant(...).total`.
pub fn evaluate_plant(
    plant: &PowerPlant,
    player: &Player,
    state: &GameState,
    w: &AuctionWeights,
) -> PlantValuation {
    let rounds = remaining_rounds(state);
    let (income_gain, bump) = projected_income_gain(plant, player, state, w);
    let incremental_income = income_gain * rounds;

    // The plant that would have to be discarded to make room, if the rack is full.
    let worst_owned = if player.plants.len() >= 3 {
        player
            .plants
            .iter()
            .min_by(|a, b| {
                plant_score(a)
                    .partial_cmp(&plant_score(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .filter(|worst| worst.number != plant.number)
    } else {
        None
    };

    let fuel_savings = worst_owned
        .map(|worst| {
            let old_cost = estimate_firing_cost(worst, &state.resources) as f32;
            let new_cost = estimate_firing_cost(plant, &state.resources) as f32;
            (old_cost - new_cost) * rounds
        })
        .unwrap_or(0.0);

    let capacity_premium = endgame_capacity_premium(player, state, w, bump);

    let denial = if w.denial_weight > 0.0 {
        let best_opponent_gain = state
            .players
            .iter()
            .filter(|p| p.id != player.id)
            .map(|opp| opponent_gain(plant, opp, state, w))
            .fold(0.0f32, f32::max);
        w.denial_weight * best_opponent_gain
    } else {
        0.0
    };

    let fuel_risk = if plant.kind.needs_resources() {
        w.fuel_risk_weight * plant_fuel_scarcity(plant, state) * plant.cost as f32 * rounds
    } else {
        0.0
    };

    let replacement_waste = worst_owned
        .map(|worst| {
            let target = useful_city_target(player, state, w);
            let current: u8 = player.plants.iter().map(|p| p.cities).sum();
            let without_worst = current.saturating_sub(worst.cities);
            let marginal_income = (income_for(current.min(target)) as f32
                - income_for(without_worst.min(target)) as f32)
                .max(0.0);
            w.replacement_waste_weight * rounds * marginal_income
        })
        .unwrap_or(0.0);

    let total = (incremental_income + fuel_savings + capacity_premium + denial
        - fuel_risk
        - replacement_waste)
        .max(0.0);

    PlantValuation {
        incremental_income,
        fuel_savings,
        capacity_premium,
        denial,
        fuel_risk,
        replacement_waste,
        total,
    }
}

// ---------------------------------------------------------------------------
// Auction reserve
// ---------------------------------------------------------------------------

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

/// Bonus for building in a contested city (already occupied by opponents).
pub fn city_contest_bonus(owner_count: usize, block_weight: f32) -> f32 {
    if block_weight <= 0.0 || owner_count == 0 {
        0.0
    } else {
        block_weight * owner_count as f32
    }
}
