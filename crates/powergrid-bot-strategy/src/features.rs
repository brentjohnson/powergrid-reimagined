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

/// Per-round fuel demand a plant places on `resource`.
///
/// - Wind, and pure-fuel plants that don't burn `resource` → 0
/// - `GasOrOil` hybrids split `cost / 2` between gas and oil — the balanced-load
///   assumption `decide_buy_resources` follows when both pools are healthy
/// - Matching pure-fuel plants → their full `cost`
fn per_round_demand(plant: &PowerPlant, resource: Resource) -> f32 {
    let cost = plant.cost as f32;
    match plant.kind {
        PlantKind::GasOrOil => match resource {
            Resource::Gas | Resource::Oil => cost / 2.0,
            _ => 0.0,
        },
        _ => {
            if plant.kind.resources().contains(&resource) {
                cost
            } else {
                0.0
            }
        }
    }
}

/// Fraction of the remaining game, in `[0, 1]`, that `player` could realistically
/// keep `plant` (plus any rack-mates sharing its fuel) supplied with `resource`.
///
/// Two supply streams feed the player each round:
/// - **fair share of replenishment**: the per-round restock split evenly among
///   every player who burns this resource (this player included)
/// - **drawdown of existing market stock**: spread evenly over the rounds left
///
/// weighed against **total demand**: the candidate's own appetite *plus* everything
/// the player already owns that draws on the same resource — a player running a
/// heavy coal plant can't treat a second one as "fully fed" just because the new
/// plant alone would fit inside the fair share.
///
/// `1.0` when there's no demand on this resource at all (e.g. the unused half of
/// a hybrid's pool, or any plant that doesn't burn it).
fn resource_feasibility(
    plant: &PowerPlant,
    player: &Player,
    state: &GameState,
    resource: Resource,
) -> f32 {
    let (coal_r, oil_r, gas_r, uranium_r) = replenishment_amounts(state.step, state.players.len());
    let replen = match resource {
        Resource::Coal => coal_r,
        Resource::Oil => oil_r,
        Resource::Gas => gas_r,
        Resource::Uranium => uranium_r,
    } as f32;

    // This player plus everyone else whose rack draws on `resource` — they all
    // compete for the same replenishment stream.
    let competitors = 1.0
        + state
            .players
            .iter()
            .filter(|p| p.id != player.id)
            .filter(|p| {
                p.plants
                    .iter()
                    .any(|owned| per_round_demand(owned, resource) > 0.0)
            })
            .count() as f32;
    let fair_share = replen / competitors;

    let rounds = remaining_rounds(state);
    let market_drawdown = state.resources.available(resource) as f32 / rounds;
    let sustainable = fair_share + market_drawdown;

    let demand = per_round_demand(plant, resource)
        + player
            .plants
            .iter()
            .map(|owned| per_round_demand(owned, resource))
            .sum::<f32>();

    if demand <= 0.0 {
        1.0
    } else {
        (sustainable / demand).clamp(0.0, 1.0)
    }
}

/// How reliably `player` could keep `plant` fueled for the rest of the game, in
/// `[0, 1]` (`1.0` = always fed, `0.0` = essentially never). Feeds the fuel-risk
/// term of `evaluate_plant`: a thirsty plant on a contested, slow-replenishing
/// resource is worth less than its raw income suggests, because that income won't
/// reliably materialize.
///
/// - Wind → `1.0` (nothing to run out of)
/// - `GasOrOil` → the better of its two pools — the bot sources from whichever is
///   easier (mirrors `estimate_firing_cost`'s "prefer the more-available resource")
/// - All other kinds → feasibility of their single resource
pub fn fuel_feasibility(plant: &PowerPlant, player: &Player, state: &GameState) -> f32 {
    match plant.kind {
        PlantKind::Wind => 1.0,
        PlantKind::GasOrOil => resource_feasibility(plant, player, state, Resource::Gas)
            .max(resource_feasibility(plant, player, state, Resource::Oil)),
        _ => plant
            .kind
            .resources()
            .first()
            .map(|&r| resource_feasibility(plant, player, state, r))
            .unwrap_or(1.0),
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

/// Average elektro it would cost to fire `plant` per round, simulated forward
/// over the rest of the game as every player's rack draws the market down each
/// round and `replenishment_amounts` refills it.
///
/// `estimate_firing_cost` alone only sees a *snapshot* — a coal plant looks
/// cheap at the table's current price even if five players are about to drain
/// it dry. This walks the market forward `remaining_rounds` times: each round,
/// price one firing at the current table, then drain it by total per-round
/// demand (every player's rack plus this candidate, balanced-split for
/// hybrids — the same model `fuel_feasibility` uses) and replenish it. A
/// plentiful, lightly-contested fuel stays near its cheap snapshot price the
/// whole way; a contested, slow-refilling one drifts toward the table's
/// expensive end, and the average reflects that forward dearness. Market-wide,
/// so it doesn't depend on which player is asking.
pub fn expected_firing_cost(plant: &PowerPlant, state: &GameState) -> f32 {
    if !plant.kind.needs_resources() {
        return 0.0;
    }

    let rounds = (remaining_rounds(state).round() as usize).max(1);
    let (coal_r, oil_r, gas_r, uranium_r) = replenishment_amounts(state.step, state.players.len());
    let mut market = state.resources.clone();
    let mut total = 0.0f32;

    for _ in 0..rounds {
        total += estimate_firing_cost(plant, &market) as f32;

        for (resource, replen) in [
            (Resource::Coal, coal_r),
            (Resource::Oil, oil_r),
            (Resource::Gas, gas_r),
            (Resource::Uranium, uranium_r),
        ] {
            // Total per-round draw on `resource`: every player's rack (this
            // player's included) plus the candidate the player is evaluating.
            let demand = per_round_demand(plant, resource)
                + state
                    .players
                    .iter()
                    .flat_map(|p| p.plants.iter())
                    .map(|owned| per_round_demand(owned, resource))
                    .sum::<f32>();
            let drain = (demand.round() as u8).min(market.available(resource));
            market.take(resource, drain);
            market.replenish(resource, replen);
        }
    }

    total / rounds as f32
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
/// discard via `capacity_bump` — doesn't clear the upgrade margin. The margin
/// is scaled by `late_game_urgency`: hoarded cash loses value as the game nears
/// its end, so the bar to upgrade drops with it.
pub fn should_skip_auction(
    player: &Player,
    candidate: &PowerPlant,
    state: &GameState,
    w: &AuctionWeights,
) -> bool {
    if player.plants.len() >= 3 {
        let margin = w.upgrade_margin * late_game_urgency(state);
        return evaluate_plant(candidate, player, state, w).total < margin;
    }
    false
}

/// How much future value holding cash still has, in `(0, 1]`: 1.0 with the full
/// `remaining_rounds` horizon ahead, falling toward 1/8 as the game closes out.
/// Scales the auction buy thresholds (`min_open_score`, `upgrade_margin`) so
/// bots that would otherwise sit on a growing pile of Elektro keep buying
/// plants late game instead of passing every auction until the step cap.
pub fn late_game_urgency(state: &GameState) -> f32 {
    remaining_rounds(state) / 8.0
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
//                   - OperatingCost - FuelRisk - ReplacementWaste
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
    /// Expected fuel spend over the rounds the plant will actually run —
    /// `expected_firing_cost` (a forward, demand/replenishment-aware price)
    /// times the fed fraction of the remaining game (`fuel_feasibility`).
    /// Turns gross income into *net* income: a 1-coal plant and a 2-gas plant
    /// powering the same city are no longer worth the same. Zero for Wind.
    /// (LOGIC.md §2 "Fuel Savings" / "Resource Efficiency")
    pub operating_cost: f32,
    /// Penalty for relying on fuel the player likely can't keep supplied — the
    /// expected income and fuel-cost exposure of the rounds `fuel_feasibility`
    /// says will go unfed. Zero on flush, uncontested markets (or Wind).
    /// (LOGIC.md §6 "Resource Risk")
    pub fuel_risk: f32,
    /// Value thrown away on a forced full-rack discard of a plant that still had
    /// useful income left to earn. (LOGIC.md §8 "Replacement Waste")
    pub replacement_waste: f32,
    /// Sum of every positive term above minus `operating_cost`, `fuel_risk` and
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

    // Shared between the two fuel terms below: the fraction of the remaining
    // game the plant can realistically be kept fed.
    let feasibility = if plant.kind.needs_resources() {
        fuel_feasibility(plant, player, state)
    } else {
        1.0
    };

    // Gross income alone overstates a plant's worth — running it costs fuel.
    // Charge the forward, demand/replenishment-aware price (`expected_firing_cost`,
    // not just the table snapshot) over the *fed* rounds only — the unfed rounds
    // are already priced into `fuel_risk` below, so the two terms partition the
    // game's rounds with no double-count. This is what makes a 1-coal plant
    // worth more than an equal-capacity 2-gas plant: net, not gross, income.
    let operating_cost = if plant.kind.needs_resources() {
        let per_round = expected_firing_cost(plant, state);
        w.operating_cost_weight * feasibility * per_round * rounds
    } else {
        0.0
    };

    // Penalize plants whose fuel the player likely can't keep flowing — weighted
    // value-at-risk over the rounds `fuel_feasibility` says will go unfed:
    // the income those rounds would have earned, plus the absolute cost of the
    // fuel itself (so a thirsty plant on a *dear*, scarce resource — e.g. uranium
    // at ~9 Elektro/unit — is penalized harder than an equally infeasible one on
    // a cheap resource). Zero on flush/uncontested markets, where feasibility is 1.
    let fuel_risk = if plant.kind.needs_resources() {
        let infeasibility = 1.0 - feasibility;
        let fuel_price = estimate_firing_cost(plant, &state.resources) as f32;
        w.fuel_risk_weight * infeasibility * rounds * (income_gain + fuel_price)
    } else {
        0.0
    };

    // Note: `capacity_bump` already nets the forced discard out of
    // `incremental_income` (bump = new − worst), so charging the worst plant's
    // marginal income here again partially double-counts the loss. The term is
    // kept as a deliberate conservatism knob, with small default weights — see
    // the comments in assets/bots/default.toml.
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
        - operating_cost
        - fuel_risk
        - replacement_waste)
        .max(0.0);

    PlantValuation {
        incremental_income,
        fuel_savings,
        capacity_premium,
        denial,
        operating_cost,
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
    let mut reserve = fuel_reserve(player, buy, market);
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

/// Elektro to keep in hand for one firing of every resource-consuming plant the
/// player already owns. With a live `market`, each firing is priced via
/// `estimate_firing_cost` (scarce fuels inflate the reserve); without one, the
/// static `plant.cost` estimate is used (isolated unit tests).
pub fn fuel_reserve(player: &Player, buy: &BuyWeights, market: Option<&ResourceMarket>) -> u32 {
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
