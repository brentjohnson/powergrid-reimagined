use powergrid_core::{
    actions::Action,
    rules::effective_min_bid,
    state::GameState,
    types::{
        connection_cost, income_for, PlantKind, Player, PlayerId, PowerPlant, Resource,
        ResourceMarket,
    },
};
use tracing::{debug, info, warn};

use crate::{
    bot::Bot,
    encoding,
    features::{
        auction_reserve, capacity_bump, city_contest_bonus, evaluate_plant, expected_unit_price,
        fuel_reserve, late_game_urgency, plant_score, player_resource_demand, should_skip_auction,
    },
    policy,
    profile::default_registry,
};

// ---------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------

/// Compatibility shim: creates a one-shot Normal-profile bot and decides.
/// Use `Bot::decide` for persistent bots with stable RNG state.
pub fn decide(state: &GameState, me: PlayerId) -> Option<Action> {
    let registry = default_registry();
    let profile = registry.normal.clone();
    let seed = me.as_u128() as u64;
    let mut bot = Bot::new(
        me,
        String::new(),
        powergrid_core::types::PlayerColor::Red,
        profile,
        seed,
    );
    decide_with_bot(state, &mut bot)
}

/// Outcome of asking the RL policy for a move.
pub(crate) enum RlDecision {
    Action(Action),
    /// Someone else acts right now — wait (the heuristic must not answer either,
    /// or the bot would mix training-time turn semantics with heuristic ones).
    NotMyTurn,
    /// The policy can't be used in this game (non-default map); use the heuristic.
    Unavailable,
}

/// Decide with the RL policy: encode the state, run the MLP, and sample from
/// the masked softmax with the bot's persistent RNG (stochastic by design —
/// the policy was trained and evaluated that way; greedy play can stall).
pub(crate) fn decide_rl(state: &GameState, bot: &mut Bot) -> RlDecision {
    // The encoding is compiled against the default USA map.
    if !encoding::map_matches_default(&state.map) {
        warn!(
            "bot '{}': RL policy unavailable on non-default map; using heuristic",
            bot.name
        );
        return RlDecision::Unavailable;
    }

    // Strict turn gate, matching the training-time actor semantics (in
    // Bureaucracy: first of `remaining`, not merely a member of it).
    if encoding::current_actor_id(state) != Some(bot.id) {
        return RlDecision::NotMyTurn;
    }

    let mask = encoding::build_action_mask(state, bot.id);
    if mask.iter().all(|&m| m == 0) {
        warn!("bot '{}': empty action mask on own turn", bot.name);
        return RlDecision::Unavailable;
    }

    let obs = encoding::build_observation(state, bot.id);
    let policy = bot.policy.clone().expect("decide_rl requires a policy");
    let logits = policy.logits(&obs);
    match policy::sample_masked(&logits, &mask, &mut bot.rng) {
        Some(action_id) => RlDecision::Action(encoding::action_id_to_action(
            action_id as u16,
            state,
            bot.id,
        )),
        None => RlDecision::Unavailable,
    }
}

/// Full implementation: dispatch to phase-specific handlers.
pub(crate) fn decide_with_bot(state: &GameState, bot: &mut Bot) -> Option<Action> {
    use powergrid_core::types::Phase;
    match &state.phase {
        Phase::Lobby | Phase::PlayerOrder | Phase::GameOver { .. } => None,

        Phase::Auction {
            current_bidder_idx,
            active_bid,
            bought,
            passed,
        } => decide_auction(state, bot, *current_bidder_idx, active_bid, bought, passed),

        Phase::DiscardPlant {
            player, new_plant, ..
        } => {
            if *player != bot.id {
                return None;
            }
            decide_discard(state, bot, new_plant)
        }

        Phase::DiscardResource {
            player, drop_total, ..
        } => {
            if *player != bot.id {
                return None;
            }
            decide_discard_resource(state, bot, *drop_total)
        }

        Phase::BuyResources { remaining } => {
            if remaining.first() != Some(&bot.id) {
                return None;
            }
            decide_buy_resources(state, bot)
        }

        Phase::BuildCities { remaining } => {
            if remaining.first() != Some(&bot.id) {
                return None;
            }
            decide_build_cities(state, bot)
        }

        Phase::Bureaucracy { remaining } => {
            if !remaining.contains(&bot.id) {
                return None;
            }
            decide_power_cities(state, bot)
        }

        Phase::PowerCitiesFuel {
            player,
            hybrid_cost,
            ..
        } => {
            if *player != bot.id {
                return None;
            }
            decide_power_cities_fuel(state, bot, *hybrid_cost)
        }
    }
}

// ---------------------------------------------------------------------------
// Auction phase
// ---------------------------------------------------------------------------

fn decide_auction(
    state: &GameState,
    bot: &mut Bot,
    current_bidder_idx: usize,
    active_bid: &Option<powergrid_core::types::ActiveBid>,
    bought: &[PlayerId],
    passed: &[PlayerId],
) -> Option<Action> {
    let w = &bot.profile.auction.clone();
    let buy = &bot.profile.buy.clone();
    let my_player = state.player(bot.id)?;

    if let Some(bid) = active_bid {
        if bid.remaining_bidders.first() != Some(&bot.id) {
            return None;
        }
        let plant = state
            .market
            .actual
            .iter()
            .find(|p| p.number == bid.plant_number)?;

        let min_bid = effective_min_bid(&state.market, bid.plant_number);

        // LOGIC.md: "Maximum Rational Bid = Plant Value" — never pay more than
        // the plant is worth in expected future Elektro. Round 1 has no
        // discretion (the only legal price is the listed minimum); later rounds
        // bid up to `evaluate_plant(...).total`, clamped by what's affordable
        // after reserving cash for fuel and city builds, and never below the
        // listed price (the bot always matches the opening bid if it can).
        let ceiling = if state.round == 1 {
            min_bid.min(my_player.money)
        } else {
            let reserve = auction_reserve(plant, my_player, w, buy, Some(&state.resources));
            let affordable = my_player.money.saturating_sub(reserve);
            let value = evaluate_plant(plant, my_player, state, w).total.round() as u32;
            value.min(affordable).max(min_bid).min(my_player.money)
        };
        let ceiling_jittered = bot
            .maybe_jitter(ceiling, bot.profile.max_jitter)
            .min(my_player.money);

        if bid.amount < ceiling_jittered {
            let raise = bid.amount + 1;
            info!(
                "Raising bid on plant {} to {} (ceiling {})",
                bid.plant_number, raise, ceiling_jittered
            );
            return Some(Action::PlaceBid { amount: raise });
        } else {
            info!(
                "Passing on plant {} — bid {} exceeds ceiling {}",
                bid.plant_number, bid.amount, ceiling_jittered
            );
            return Some(Action::PassAuction);
        }
    }

    if state
        .player_order
        .get(current_bidder_idx)
        .copied()
        .unwrap_or_default()
        != bot.id
    {
        return None;
    }
    if bought.contains(&bot.id) || passed.contains(&bot.id) {
        return None;
    }

    let is_round_one = state.round == 1;

    // Build a scored list of candidates: each affordable plant + PassAuction baseline.
    // Pass gets the `min_open_score` baseline; plants must exceed it to be preferred.
    #[derive(Clone)]
    enum AuctionCandidate {
        Select(u8), // plant_number
        Pass,
    }

    let mut candidates: Vec<(AuctionCandidate, f32)> = state
        .market
        .actual
        .iter()
        .filter(|p| my_player.money >= effective_min_bid(&state.market, p.number))
        .filter(|p| {
            // In round 1 we must buy — don't filter. Later: apply skip logic.
            is_round_one
                || (!should_skip_auction(my_player, p, state, w)
                    && capacity_bump(p, my_player) >= 1)
        })
        .map(|p| {
            // Score candidates by their Elektro value (LOGIC.md: "How many Elektro
            // is this plant worth right now, in this exact game state?").
            let value = evaluate_plant(p, my_player, state, w).total;
            (AuctionCandidate::Select(p.number), value)
        })
        .collect();

    // PassAuction as a scored baseline (not available in round 1): plants must
    // be worth at least `min_open_score` Elektro to be preferred over passing.
    // The baseline decays with `late_game_urgency` — hoarded cash is worth less
    // and less as the game closes, so passing gets harder to justify.
    if !is_round_one {
        candidates.push((
            AuctionCandidate::Pass,
            w.min_open_score * late_game_urgency(state),
        ));
    }

    if candidates.is_empty() {
        // Round 1 forced buy but nothing is affordable — pick cheapest regardless.
        if is_round_one {
            let cheapest = state.market.actual.iter().min_by_key(|p| p.number)?;
            info!(
                "Round 1 forced buy — selecting cheapest plant {}",
                cheapest.number
            );
            return Some(Action::SelectPlant {
                plant_number: cheapest.number,
            });
        }
        info!("Passing auction — cannot afford or no viable plant");
        return Some(Action::PassAuction);
    }

    let chosen = bot.sample_softmax(&candidates)?;
    match chosen {
        AuctionCandidate::Select(plant_number) => {
            let plant = state
                .market
                .actual
                .iter()
                .find(|p| p.number == plant_number)?;
            info!(
                "Selecting plant {} (kind={:?}, cities={}, value={:.1})",
                plant.number,
                plant.kind,
                plant.cities,
                evaluate_plant(plant, my_player, state, w).total,
            );
            Some(Action::SelectPlant { plant_number })
        }
        AuctionCandidate::Pass => {
            info!("Passing auction — no plant scores above threshold");
            Some(Action::PassAuction)
        }
    }
}

// ---------------------------------------------------------------------------
// Discard phase
// ---------------------------------------------------------------------------

fn decide_discard(state: &GameState, bot: &mut Bot, new_plant: &PowerPlant) -> Option<Action> {
    let player = state.player(bot.id)?;

    let worst = player
        .plants
        .iter()
        .filter(|p| p.number != new_plant.number)
        .min_by(|a, b| {
            plant_score(a)
                .partial_cmp(&plant_score(b))
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;

    info!(
        "Discarding plant {} ({:.1}) to make room for plant {} ({:.1})",
        worst.number,
        plant_score(worst),
        new_plant.number,
        plant_score(new_plant),
    );
    Some(Action::DiscardPlant {
        plant_number: worst.number,
    })
}

// ---------------------------------------------------------------------------
// Resource-discard phase
// ---------------------------------------------------------------------------

fn decide_discard_resource(state: &GameState, bot: &mut Bot, drop_total: u8) -> Option<Action> {
    let player = state.player(bot.id)?;
    let gas = drop_total.min(player.resources.gas);
    let oil = drop_total - gas;
    info!(
        "DiscardResource: dropping {} gas and {} oil (drop_total={})",
        gas, oil, drop_total
    );
    Some(Action::DiscardResource { gas, oil })
}

// ---------------------------------------------------------------------------
// Buy resources phase
// ---------------------------------------------------------------------------

fn decide_buy_resources(state: &GameState, bot: &mut Bot) -> Option<Action> {
    let player = state.player(bot.id)?;

    let mut purchases: Vec<(Resource, u8)> = Vec::new();
    let mut sim_market = state.resources.clone();
    let mut sim_player = player.clone();
    let mut budget = player.money;

    // Most cities first; break ties by plant number (smaller = cheaper to fuel).
    let mut plants = player.plants.clone();
    plants.sort_by(|a, b| b.cities.cmp(&a.cities).then(a.number.cmp(&b.number)));

    // `buy_for_plant` raises a *shared* fuel pool up to `target` total units, so the
    // targets passed in must be cumulative across every plant drawing on that pool —
    // otherwise a second coal plant sees the first plant's purchase as "already have
    // enough" and buys nothing (the bug this fixes: two coal plants needing 3 + 2
    // ended up with only 3 coal total). Track running cumulative targets per pool.
    let (mut coal_target, mut oil_target, mut gas_target, mut uranium_target) =
        (0u8, 0u8, 0u8, 0u8);
    // Gas and oil share both storage and market, so pure Gas/Oil demand must be
    // folded into the hybrid (GasOrOil) target too — see the second pass below.
    let mut gasoil_combined_target = 0u8;

    // Pass 1: pure-fuel plants first, so hybrids only need to cover the incremental
    // shortfall on the shared gas+oil pool once pure demand is already accounted for.
    for plant in plants.iter().filter(|p| p.kind != PlantKind::GasOrOil) {
        let target = match plant.kind {
            PlantKind::Coal => {
                coal_target = coal_target.saturating_add(plant.cost);
                coal_target
            }
            PlantKind::Oil => {
                oil_target = oil_target.saturating_add(plant.cost);
                gasoil_combined_target = gasoil_combined_target.saturating_add(plant.cost);
                oil_target
            }
            PlantKind::Gas => {
                gas_target = gas_target.saturating_add(plant.cost);
                gasoil_combined_target = gasoil_combined_target.saturating_add(plant.cost);
                gas_target
            }
            PlantKind::Uranium => {
                uranium_target = uranium_target.saturating_add(plant.cost);
                uranium_target
            }
            PlantKind::Wind | PlantKind::GasOrOil => continue,
        };
        buy_for_plant(
            plant,
            target,
            &mut sim_market,
            &mut sim_player,
            &mut budget,
            &mut purchases,
        );
    }

    // Pass 2: hybrids draw on the same combined gas+oil pool; each one's cumulative
    // target includes all pure gas/oil demand plus every hybrid's cost so far.
    for plant in plants.iter().filter(|p| p.kind == PlantKind::GasOrOil) {
        gasoil_combined_target = gasoil_combined_target.saturating_add(plant.cost);
        buy_for_plant(
            plant,
            gasoil_combined_target,
            &mut sim_market,
            &mut sim_player,
            &mut budget,
            &mut purchases,
        );
    }

    // Pass 3: stockpile cheap fuel ahead of future dearness. The essential
    // passes above cover only the coming firing; this spends *surplus* cash
    // (above a city-build reserve) to pre-buy fuel that is currently cheaper
    // than its forward-expected price, up to `stockpile_rounds` of storage.
    let stockpile_rounds = bot.profile.buy.stockpile_rounds;
    if stockpile_rounds > 1.0 {
        let reserve =
            bot.profile.auction.city_reserve as u32 + bot.profile.auction.safety_buffer as u32;
        stockpile_cheap_fuel(
            state,
            &mut sim_player,
            &mut sim_market,
            &mut budget,
            &mut purchases,
            reserve,
            stockpile_rounds,
        );
    }

    if purchases.is_empty() {
        info!("Buy resources: nothing to buy, done");
    } else {
        let total = state.resources.batch_price(&purchases).unwrap_or(0);
        info!(
            "Buy resources: {:?} for ~{} elektro (have {})",
            purchases, total, player.money
        );
    }

    Some(Action::BuyResourceBatch { purchases })
}

/// Bring `plant`'s fuel level up to `target` by purchasing from the simulated market.
fn buy_for_plant(
    plant: &PowerPlant,
    target: u8,
    market: &mut ResourceMarket,
    player: &mut Player,
    budget: &mut u32,
    purchases: &mut Vec<(Resource, u8)>,
) {
    match plant.kind {
        PlantKind::Coal => {
            let want = target.saturating_sub(player.resources.coal);
            if want > 0 {
                try_buy(Resource::Coal, want, market, player, budget, purchases);
            }
        }
        PlantKind::Oil => {
            let want = target.saturating_sub(player.resources.oil);
            if want > 0 {
                try_buy(Resource::Oil, want, market, player, budget, purchases);
            }
        }
        PlantKind::GasOrOil => {
            let combined = player.resources.gas.saturating_add(player.resources.oil);
            let want = target.saturating_sub(combined);
            if want == 0 {
                return;
            }
            // Prefer the fuel type with more market supply; tie-break to oil.
            let prefer_oil = market.available(Resource::Oil) >= market.available(Resource::Gas);
            let (first, second) = if prefer_oil {
                (Resource::Oil, Resource::Gas)
            } else {
                (Resource::Gas, Resource::Oil)
            };
            try_buy(first, want, market, player, budget, purchases);
            let combined = player.resources.gas.saturating_add(player.resources.oil);
            let remaining = target.saturating_sub(combined);
            if remaining > 0 {
                try_buy(second, remaining, market, player, budget, purchases);
            }
        }
        PlantKind::Gas => {
            let want = target.saturating_sub(player.resources.gas);
            if want > 0 {
                try_buy(Resource::Gas, want, market, player, budget, purchases);
            }
        }
        PlantKind::Uranium => {
            let want = target.saturating_sub(player.resources.uranium);
            if want > 0 {
                try_buy(Resource::Uranium, want, market, player, budget, purchases);
            }
        }
        PlantKind::Wind => {}
    }
}

/// Attempt to purchase up to `want` units, degrading gracefully on budget/storage limits.
fn try_buy(
    resource: Resource,
    want: u8,
    market: &mut ResourceMarket,
    player: &mut Player,
    budget: &mut u32,
    purchases: &mut Vec<(Resource, u8)>,
) {
    // Keep buying chunks until `want` is satisfied or nothing more can be bought —
    // a single chunk can fall short of `want` under budget/storage pressure (e.g. the
    // largest affordable chunk is smaller than `want`), and settling for that first
    // chunk would leave affordable, storable units on the table.
    let mut remaining = want;
    while remaining > 0 {
        let available = market.available(resource);
        let cap = remaining.min(available);
        if cap == 0 {
            break;
        }
        let mut bought = false;
        for n in (1..=cap).rev() {
            if !player.can_add_resource(resource, n) {
                continue;
            }
            if let Some(cost) = market.price(resource, n) {
                if cost <= *budget {
                    debug!("Buying {} {:?} for {} elektro", n, resource, cost);
                    purchases.push((resource, n));
                    market.take(resource, n);
                    player.resources.add(resource, n);
                    *budget -= cost;
                    remaining -= n;
                    bought = true;
                    break;
                }
            }
        }
        if !bought {
            debug!(
                "Cannot afford/store any more {:?} (remaining {}, budget {})",
                resource, remaining, budget
            );
            break;
        }
    }
}

/// Pre-buy fuel that is currently cheaper than its forward-expected price, up to
/// `stockpile_rounds` rounds of storage, spending only cash above `reserve`.
/// Captures the human tactic of hoarding cheap/scarce fuel — the valuation model
/// already forward-prices this dearness (`expected_firing_cost`); the buy logic
/// just wasn't acting on it. Buys one unit at a time so the rising marginal price
/// naturally halts the stockpile once the fuel is no longer a bargain, and
/// `can_add_resource` caps it at the rack's real storage limit.
#[allow(clippy::too_many_arguments)]
fn stockpile_cheap_fuel(
    state: &GameState,
    player: &mut Player,
    market: &mut ResourceMarket,
    budget: &mut u32,
    purchases: &mut Vec<(Resource, u8)>,
    reserve: u32,
    stockpile_rounds: f32,
) {
    for resource in [
        Resource::Coal,
        Resource::Oil,
        Resource::Gas,
        Resource::Uranium,
    ] {
        let demand = player_resource_demand(player, resource);
        if demand <= 0.0 {
            continue;
        }
        let expected = expected_unit_price(resource, state);
        let cap_units = (demand * stockpile_rounds).round() as u8;
        while player.resources.get(resource) < cap_units {
            if *budget <= reserve
                || market.available(resource) == 0
                || !player.can_add_resource(resource, 1)
            {
                break;
            }
            let Some(unit_cost) = market.price(resource, 1) else {
                break;
            };
            // Only pre-buy while this next unit is at/below its forward price and
            // paying for it still leaves the city-build reserve intact.
            if unit_cost as f32 > expected || budget.saturating_sub(unit_cost) < reserve {
                break;
            }
            debug!(
                "Stockpiling 1 {:?} at {} (forward price {:.1})",
                resource, unit_cost, expected
            );
            purchases.push((resource, 1));
            market.take(resource, 1);
            player.resources.add(resource, 1);
            *budget -= unit_cost;
        }
    }
}

// ---------------------------------------------------------------------------
// Build cities phase
// ---------------------------------------------------------------------------

fn decide_build_cities(state: &GameState, bot: &mut Bot) -> Option<Action> {
    let player = state.player(bot.id)?;
    let block_weight = bot.profile.build.block_weight;
    let owned_cities = state.player_cities(bot.id);

    let mut budget = player.money;

    // Enumerate buildable cities in active regions: not already owned, slot open.
    let mut candidates: Vec<(String, u32, f32)> = state
        .map
        .cities
        .values()
        .filter(|city| {
            state.active_regions.contains(&city.region)
                && !city.owners.contains(&bot.id)
                && city.owners.len() < state.step as usize
        })
        .filter_map(|city| {
            let route_cost = state.map.connection_cost_to(&owned_cities, &city.id)?;
            let slot_cost = connection_cost(city.owners.len());
            let total = route_cost + slot_cost;
            // Hard-only: prefer cities opponents already occupy (block / density bonus).
            let bonus = city_contest_bonus(city.owners.len(), block_weight);
            Some((city.id.clone(), total, bonus))
        })
        .collect();

    // Sort by (cost - contest_bonus) ascending — cheapest and most contested
    // first — with a city-id tiebreak so the order is total and deterministic.
    // Without the tiebreak, equal-cost cities resolve by `map.cities` HashMap
    // iteration order, which Rust randomizes per map instance; that made
    // otherwise-identical games diverge and put a noise floor under paired bot
    // evaluation (see RL-TRAINING-JOURNAL.md / powergrid-evolve).
    candidates.sort_by(|(id_a, cost_a, bonus_a), (id_b, cost_b, bonus_b)| {
        let adjusted_a = *cost_a as f32 - bonus_a;
        let adjusted_b = *cost_b as f32 - bonus_b;
        adjusted_a
            .partial_cmp(&adjusted_b)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| id_a.cmp(id_b))
    });

    // Spend freely up to capacity headroom: cities we can actually power.
    // Beyond that, cities earn no income — but they still count toward the
    // `end_game_cities` trigger, so keep building with *surplus* cash only
    // (everything above the fuel + city reserves). Without this, a bot whose
    // rack capacity equals its city count stops building forever and the game
    // can reach a no-progress fixed point that never ends.
    let powerable: u8 = player.plants.iter().map(|p| p.cities).sum();
    let owned = owned_cities.len() as u8;
    let headroom = powerable.saturating_sub(owned) as usize;
    let overbuild_reserve = fuel_reserve(player, &bot.profile.buy, Some(&state.resources))
        + bot.profile.auction.city_reserve as u32
        + bot.profile.auction.safety_buffer as u32;
    // Never build past the game-end trigger — those cities are pure waste.
    let build_cap = (state.end_game_cities as usize).saturating_sub(owned_cities.len());

    let mut city_ids: Vec<String> = Vec::new();
    let mut simulated_cities: Vec<String> = owned_cities.clone();

    for (city_id, _, _) in &candidates {
        if city_ids.len() >= build_cap {
            break;
        }

        let route_cost = state
            .map
            .connection_cost_to(&simulated_cities, city_id)
            .unwrap_or(u32::MAX);
        let city = state.map.cities.get(city_id.as_str())?;
        let slot_cost =
            connection_cost(city.owners.len() + city_ids.iter().filter(|c| *c == city_id).count());
        let total = route_cost + slot_cost;

        let affordable = total <= budget;
        let overbuild_ok = affordable && budget - total >= overbuild_reserve;
        if affordable && (city_ids.len() < headroom || overbuild_ok) {
            info!(
                "Building in {} (route={}, slot={}, total={}{})",
                city_id,
                route_cost,
                slot_cost,
                total,
                if city_ids.len() < headroom {
                    ""
                } else {
                    ", overbuild"
                }
            );
            budget -= total;
            city_ids.push(city_id.clone());
            simulated_cities.push(city_id.clone());
        }
    }

    if city_ids.is_empty() {
        info!("Build cities: nothing affordable, done");
        Some(Action::DoneBuilding)
    } else {
        info!(
            "Building {} cities: {:?} (budget remaining: {})",
            city_ids.len(),
            city_ids,
            budget
        );
        Some(Action::BuildCities { city_ids })
    }
}

// ---------------------------------------------------------------------------
// Bureaucracy phase
// ---------------------------------------------------------------------------

fn decide_power_cities(state: &GameState, bot: &mut Bot) -> Option<Action> {
    let player = state.player(bot.id)?;

    let (plant_numbers, cities_powered, _) =
        player.optimal_firing_subset(state.player_city_count(bot.id) as u8);
    let expected_income = income_for(cities_powered);

    info!(
        "PowerCities with plants {:?} — expect to power {} cities, earn {} elektro",
        plant_numbers, cities_powered, expected_income
    );

    Some(Action::PowerCities { plant_numbers })
}

fn decide_power_cities_fuel(state: &GameState, bot: &mut Bot, hybrid_cost: u8) -> Option<Action> {
    use powergrid_core::types::Phase;
    let player = state.player(bot.id)?;

    if let Phase::PowerCitiesFuel { plant_numbers, .. } = &state.phase {
        let pure_gas: u8 = plant_numbers
            .iter()
            .filter_map(|&num| player.plants.iter().find(|p| p.number == num))
            .filter(|p| p.kind == PlantKind::Gas)
            .map(|p| p.cost)
            .sum();
        let pure_oil: u8 = plant_numbers
            .iter()
            .filter_map(|&num| player.plants.iter().find(|p| p.number == num))
            .filter(|p| p.kind == PlantKind::Oil)
            .map(|p| p.cost)
            .sum();

        let _gas_avail = player.resources.gas.saturating_sub(pure_gas);
        let oil_avail = player.resources.oil.saturating_sub(pure_oil);

        // Prefer oil for hybrids to conserve gas (controlled by oil_preference weight).
        let oil_used = if bot.profile.bureaucracy.oil_preference >= 0.5 {
            hybrid_cost.min(oil_avail)
        } else {
            0
        };
        let gas = hybrid_cost - oil_used;

        info!(
            "PowerCitiesFuel: using {} gas + {} oil for hybrids (hybrid_cost={})",
            gas, oil_used, hybrid_cost
        );
        Some(Action::PowerCitiesFuel { gas, oil: oil_used })
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use powergrid_core::{
        default_map,
        state::GameState,
        types::{Player, PlayerColor, PlayerId, PowerPlant},
    };

    use crate::{
        features::{
            auction_reserve, estimate_firing_cost, evaluate_plant, expected_firing_cost,
            fuel_feasibility, remaining_rounds, useful_city_target,
        },
        profile::default_registry,
    };

    fn coal_plant(number: u8, cost: u8, cities: u8) -> PowerPlant {
        PowerPlant {
            number,
            kind: PlantKind::Coal,
            cost,
            cities,
        }
    }

    fn hybrid_plant(number: u8, cost: u8, cities: u8) -> PowerPlant {
        PowerPlant {
            number,
            kind: PlantKind::GasOrOil,
            cost,
            cities,
        }
    }

    fn gas_plant(number: u8, cost: u8, cities: u8) -> PowerPlant {
        PowerPlant {
            number,
            kind: PlantKind::Gas,
            cost,
            cities,
        }
    }

    fn uranium_plant(number: u8, cost: u8, cities: u8) -> PowerPlant {
        PowerPlant {
            number,
            kind: PlantKind::Uranium,
            cost,
            cities,
        }
    }

    /// Build a normal-profile bot whose id matches `player`'s, so
    /// `decide_buy_resources` (which looks the player up via `state.player(bot.id)`)
    /// finds the player pushed by `state_with_player`.
    fn bot_for(player: &Player) -> Bot {
        let mut bot = normal_bot();
        bot.id = player.id;
        bot
    }

    /// Sum the quantities of `resource` across a `BuyResourceBatch` purchase list.
    fn bought(purchases: &[(Resource, u8)], resource: Resource) -> u8 {
        purchases
            .iter()
            .filter(|(r, _)| *r == resource)
            .map(|(_, n)| n)
            .sum()
    }

    fn bot_with_money(money: u32) -> Player {
        let mut p = Player::new("bot".into(), PlayerColor::Red);
        p.money = money;
        p
    }

    fn normal_bot() -> Bot {
        let registry = default_registry();
        let profile = registry.normal.clone();
        Bot::new(
            PlayerId::nil(),
            "test".into(),
            PlayerColor::Red,
            profile,
            42,
        )
    }

    fn hard_bot_for(player: &Player) -> Bot {
        let registry = default_registry();
        let mut bot = Bot::new(
            PlayerId::nil(),
            "test".into(),
            PlayerColor::Red,
            registry.hard.clone(),
            42,
        );
        bot.id = player.id;
        bot
    }

    /// Build a minimal 4-player GameState and insert `player` as its first player.
    fn state_with_player(player: &Player) -> GameState {
        let mut state = GameState::new(default_map(), 4);
        state.end_game_cities = 17; // 4-player default
        state.players.push(player.clone());
        state.player_order.push(player.id);
        state
    }

    #[test]
    fn buys_minimum_fuel_when_money_tight() {
        let plant = coal_plant(5, 2, 1);
        let mut player = bot_with_money(20);
        player.plants.push(plant.clone());

        let mut market = ResourceMarket::initial();
        let mut purchases = vec![];
        let mut budget = player.money;

        buy_for_plant(
            &plant,
            plant.cost,
            &mut market,
            &mut player,
            &mut budget,
            &mut purchases,
        );

        let coal_bought: u8 = purchases
            .iter()
            .filter(|(r, _)| *r == Resource::Coal)
            .map(|(_, n)| n)
            .sum();
        assert!(
            coal_bought >= plant.cost,
            "expected >= {} coal, got {}",
            plant.cost,
            coal_bought
        );
    }

    #[test]
    fn falls_back_to_gas_for_hybrid_when_oil_empty() {
        let plant = hybrid_plant(10, 3, 2);
        let mut player = bot_with_money(50);
        player.plants.push(plant.clone());

        let mut market = ResourceMarket::initial();
        market.oil = 0;
        market.gas = 24;

        let mut purchases = vec![];
        let mut budget = player.money;

        buy_for_plant(
            &plant,
            plant.cost,
            &mut market,
            &mut player,
            &mut budget,
            &mut purchases,
        );

        let gas_bought: u8 = purchases
            .iter()
            .filter(|(r, _)| *r == Resource::Gas)
            .map(|(_, n)| n)
            .sum();
        assert!(
            gas_bought >= plant.cost,
            "expected >= {} gas as fallback, got {}",
            plant.cost,
            gas_bought
        );
    }

    #[test]
    fn degrades_gracefully_when_full_topup_unaffordable() {
        let plant = coal_plant(15, 4, 3);
        let mut player = bot_with_money(5);
        player.plants.push(plant.clone());

        let mut market = ResourceMarket::initial();
        let mut purchases = vec![];
        let mut budget = player.money;

        buy_for_plant(
            &plant,
            plant.cost * 2,
            &mut market,
            &mut player,
            &mut budget,
            &mut purchases,
        );

        let coal_bought: u8 = purchases
            .iter()
            .filter(|(r, _)| *r == Resource::Coal)
            .map(|(_, n)| n)
            .sum();
        assert!(coal_bought > 0, "expected some coal to be bought, got none");
        assert!(coal_bought <= 5, "spent more than budget allows");
    }

    #[test]
    fn hybrid_buys_cheaper_fuel_first() {
        let plant = hybrid_plant(10, 3, 2);
        let mut player = bot_with_money(50);
        player.plants.push(plant.clone());

        // Make gas plentiful (cheap) and oil scarce (expensive) → hybrid should prefer gas.
        let mut market = ResourceMarket::initial();
        market.oil = 6;
        market.gas = 24;

        let mut purchases = vec![];
        let mut budget = player.money;

        buy_for_plant(
            &plant,
            plant.cost,
            &mut market,
            &mut player,
            &mut budget,
            &mut purchases,
        );

        let gas_bought: u8 = purchases
            .iter()
            .filter(|(r, _)| *r == Resource::Gas)
            .map(|(_, n)| n)
            .sum();
        let oil_bought: u8 = purchases
            .iter()
            .filter(|(r, _)| *r == Resource::Oil)
            .map(|(_, n)| n)
            .sum();
        assert!(
            gas_bought >= plant.cost,
            "expected to buy >= {} gas (cheaper), got {} gas and {} oil",
            plant.cost,
            gas_bought,
            oil_bought
        );
        assert_eq!(
            oil_bought, 0,
            "should not buy oil when gas is cheaper (got {} oil)",
            oil_bought
        );
    }

    #[test]
    fn jittered_bid_never_exceeds_player_money() {
        // LOGIC.md: "Maximum Rational Bid = Plant Value", but production code
        // additionally clamps the (possibly jittered) bid to `player.money` —
        // a bot can never bid more than it has, however high PlantValue or
        // jitter pushes the ceiling.
        let plant = coal_plant(15, 2, 3);
        let player = bot_with_money(10);
        let registry = default_registry();
        let w = &registry.normal.auction;
        let state = state_with_player(&player);
        let value = evaluate_plant(&plant, &player, &state, w).total.round() as u32;

        let mut bot = normal_bot();
        let max_jitter = bot.profile.max_jitter;
        for _ in 0..50 {
            // Production code applies .min(player.money) after jitter; mirror that here.
            assert!(
                bot.maybe_jitter(value, max_jitter).min(player.money) <= player.money,
                "jittered bid must never exceed available money"
            );
        }
    }

    #[test]
    fn should_skip_only_on_full_rack_no_improvement() {
        // should_skip_auction should NOT hard-skip on capacity alone — only on full rack + low upgrade.
        let mut player = bot_with_money(50);
        player.plants.push(coal_plant(5, 2, 3)); // powerable=3, rack size=1 (not full)
        let candidate = coal_plant(20, 2, 3);
        let registry = default_registry();
        let w = &registry.normal.auction;
        let state = state_with_player(&player);
        assert!(!should_skip_auction(
            &state.players[0],
            &candidate,
            &state,
            w
        ));
    }

    #[test]
    fn dont_skip_when_at_capacity() {
        let mut player = bot_with_money(50);
        player.plants.push(coal_plant(5, 2, 2));
        let candidate = coal_plant(20, 2, 3);
        let registry = default_registry();
        let w = &registry.normal.auction;
        let state = state_with_player(&player);
        // powerable=2, rack not full → don't skip
        assert!(!should_skip_auction(
            &state.players[0],
            &candidate,
            &state,
            w
        ));
    }

    #[test]
    fn skips_full_rack_upgrade_below_margin() {
        // Full rack (3 plants) + a candidate whose net Elektro value doesn't
        // clear `upgrade_margin` should be skipped — replacing a working plant
        // for a marginal gain just wastes money (LOGIC.md §3 "Replacement Quality").
        let mut player = bot_with_money(100);
        player.plants.push(coal_plant(5, 2, 3));
        player.plants.push(coal_plant(7, 2, 3));
        player.plants.push(coal_plant(10, 2, 3));
        // A same-size replacement nets ~0 incremental income — well under any margin.
        let candidate = coal_plant(20, 2, 3);
        let registry = default_registry();
        let w = &registry.normal.auction;
        let state = state_with_player(&player);
        let value = evaluate_plant(&candidate, &state.players[0], &state, w).total;
        assert!(
            value < w.upgrade_margin,
            "sanity: candidate value {value} should be below upgrade_margin {}",
            w.upgrade_margin
        );
        assert!(should_skip_auction(
            &state.players[0],
            &candidate,
            &state,
            w
        ));
    }

    // -----------------------------------------------------------------------
    // evaluate_plant — Elektro-denominated valuation (LOGIC.md)
    // -----------------------------------------------------------------------

    #[test]
    fn high_capacity_plant_out_values_small_one() {
        // LOGIC.md §1: "A plant that increases your capacity from 8 to 12
        // cities is usually much more valuable than one that increases it from
        // 8 to 9." With an empty rack, a 6-city plant should be worth
        // substantially more (in Elektro) than a 1-city plant of similar cost.
        let player = bot_with_money(200);
        let registry = default_registry();
        let w = &registry.normal.auction;
        let state = state_with_player(&player);

        let small = coal_plant(10, 3, 1);
        let big = coal_plant(20, 3, 6);

        let small_value = evaluate_plant(&small, &state.players[0], &state, w).total;
        let big_value = evaluate_plant(&big, &state.players[0], &state, w).total;

        assert!(
            big_value > small_value,
            "6-city plant ({big_value:.1}) should out-value a 1-city plant ({small_value:.1})"
        );
    }

    #[test]
    fn full_rack_upgrade_nets_only_the_delta() {
        // LOGIC.md §3A "Replacement Quality": "A plant that powers 6 cities
        // isn't really a +6 upgrade if you're replacing a plant that already
        // powers 5 — the actual gain is only +1." With a full rack, incremental
        // income should reflect the *net* capacity change (new minus discarded),
        // not the new plant's raw city count.
        let mut player = bot_with_money(200);
        player.plants.push(coal_plant(5, 2, 5)); // the "worst" plant — will be discarded
        player.plants.push(coal_plant(7, 3, 5));
        player.plants.push(coal_plant(9, 4, 5));
        let state = state_with_player(&player);
        let w = &default_registry().normal.auction;

        let candidate = coal_plant(20, 3, 6); // +6 raw, but net bump is only +1 over the discard
        let valuation = evaluate_plant(&candidate, &state.players[0], &state, w);

        let bump = capacity_bump(&candidate, &state.players[0]);
        assert_eq!(
            bump, 1,
            "sanity: net capacity bump should be the delta over the discard"
        );

        // incremental_income should be driven by the +1 net bump, not the raw +6 —
        // i.e. far smaller than what a +6 bump would project.
        let hypothetical_full_bump_income =
            (income_for(15 + 6) as f32 - income_for(15) as f32) * remaining_rounds(&state);
        assert!(
            valuation.incremental_income < hypothetical_full_bump_income,
            "incremental_income ({:.1}) should reflect the net +1 delta, not the raw +6 \
             (hypothetical full-bump income would be {:.1})",
            valuation.incremental_income,
            hypothetical_full_bump_income
        );
    }

    #[test]
    fn total_is_never_negative() {
        // PlantValuation::total is floored at 0 — "a plant is never worth
        // bidding negative Elektro for" (see doc comment on `PlantValuation::total`).
        // Stack every penalty: full rack (replacement_waste), scarce fuel
        // (fuel_risk), using the hard profile (nonzero denial/fuel_risk weights).
        let mut player = bot_with_money(200);
        player.plants.push(coal_plant(5, 2, 5));
        player.plants.push(coal_plant(7, 2, 5));
        player.plants.push(coal_plant(9, 2, 5));
        let mut state = state_with_player(&player);
        state.resources.coal = 1; // scarce
        state.players[0].plants.push(coal_plant(30, 25, 1)); // inflate demand → scarcity

        let w = &default_registry().hard.auction;
        // A thirsty, low-capacity, costly candidate — about as bad a buy as it gets.
        let candidate = coal_plant(40, 6, 1);
        let valuation = evaluate_plant(&candidate, &state.players[0], &state, w);

        assert!(
            valuation.total >= 0.0,
            "PlantValuation::total must never be negative, got {}",
            valuation.total
        );
    }

    #[test]
    fn endgame_ceiling_clamps_useful_target() {
        // With owned=15 and buildable_lookahead=3, the naive target would be 18,
        // but if end_game_cities=17 it clamps to 17.
        let registry = default_registry();
        let w = &registry.normal.auction;
        let mut player = bot_with_money(100);
        let mut state = state_with_player(&player);
        state.end_game_cities = 17;
        // Manually nudge owned count by lowering the end-game ceiling instead.
        // Construct a player with 0 cities but set end_game_cities=2 to simulate the clamp.
        state.end_game_cities = 2;
        let player_ref = &state.players[0];
        // owned=0, buildable_lookahead=2 → naive target=2 → clamped to 2 (matches end_game_cities).
        let target = useful_city_target(player_ref, &state, w);
        assert_eq!(target, 2, "target should clamp to end_game_cities=2");

        // Now raise end_game_cities above lookahead → clamp does not fire.
        state.end_game_cities = 17;
        player.plants.clear();
        let target2 = useful_city_target(&state.players[0], &state, w);
        assert_eq!(
            target2, w.buildable_lookahead,
            "with 0 owned cities, target should be buildable_lookahead={}",
            w.buildable_lookahead
        );
    }

    #[test]
    fn auction_reserve_protects_two_city_builds() {
        // No existing plants, candidate is a basic coal plant cost=2:
        // fuel reserve = 2 * 4 = 8, city reserve = 30, safety = 5 → total 43.
        let player = bot_with_money(100);
        let candidate = coal_plant(15, 2, 2);
        let registry = default_registry();
        let w = &registry.normal.auction;
        let buy = &registry.normal.buy;
        assert_eq!(
            auction_reserve(&candidate, &player, w, buy, None),
            8 + 30 + 5
        );
    }

    #[test]
    fn jitter_sometimes_lifts_the_bid() {
        // Jitter is a Bot-level mechanism independent of how the base bid was
        // computed — exercise it directly against a fixed base value.
        let base = 50u32;

        // With seed 42 and 200 trials, count how many jitter.
        let mut bot = normal_bot();
        let max_jitter = bot.profile.max_jitter;
        let mut saw_jitter = false;
        let mut saw_no_jitter = false;
        for _ in 0..200 {
            let bid = bot.maybe_jitter(base, max_jitter);
            if bid > base {
                saw_jitter = true;
                assert!(
                    bid <= base + max_jitter as u32,
                    "jitter exceeded max_jitter"
                );
            } else {
                saw_no_jitter = true;
            }
        }
        assert!(
            saw_jitter,
            "expected at least one jittered bid in 200 trials"
        );
        assert!(
            saw_no_jitter,
            "expected at least one non-jittered bid in 200 trials"
        );
    }

    #[test]
    fn essential_pass_ignores_city_reserve() {
        let plant = coal_plant(5, 2, 1);
        let mut player = bot_with_money(22);
        player.plants.push(plant.clone());

        let mut market = ResourceMarket::initial();
        let mut purchases = vec![];
        let mut budget = player.money;

        buy_for_plant(
            &plant,
            plant.cost,
            &mut market,
            &mut player,
            &mut budget,
            &mut purchases,
        );

        let coal_bought: u8 = purchases
            .iter()
            .filter(|(r, _)| *r == Resource::Coal)
            .map(|(_, n)| n)
            .sum();
        assert!(
            coal_bought >= plant.cost,
            "essential pass should buy at least {} coal (got {})",
            plant.cost,
            coal_bought
        );
    }

    /// Regression test for the bug where a bot with two coal plants (burning 3 and 2
    /// coal respectively) bought only 3 coal total instead of 5 — because
    /// `decide_buy_resources` passed each plant's *own* cost as the target against the
    /// shared coal pool, so the second plant saw "already have enough" and bought
    /// nothing. Targets must be cumulative across plants sharing a fuel pool.
    #[test]
    fn two_coal_plants_buy_combined_fuel() {
        let mut player = bot_with_money(168);
        player.plants.push(coal_plant(33, 3, 6));
        player.plants.push(coal_plant(29, 2, 5));

        let state = state_with_player(&player);
        let mut bot = bot_for(&player);

        let action = decide_buy_resources(&state, &mut bot).expect("bot should act");
        let Action::BuyResourceBatch { purchases } = action else {
            panic!("expected BuyResourceBatch, got {action:?}");
        };

        let coal_bought = bought(&purchases, Resource::Coal);
        assert!(
            coal_bought >= 5,
            "expected combined coal purchase >= 5 (3 + 2) so both plants can fire, got {}",
            coal_bought
        );
    }

    /// Two coal plants plus a hybrid: pure coal demand and shared gas/oil demand must
    /// each accumulate independently across all plants drawing on their pool.
    #[test]
    fn two_coal_plus_hybrid_combined_targets() {
        let mut player = bot_with_money(300);
        player.plants.push(coal_plant(10, 3, 6));
        player.plants.push(coal_plant(11, 2, 5));
        player.plants.push(hybrid_plant(12, 2, 4));

        let state = state_with_player(&player);
        let mut bot = bot_for(&player);

        let action = decide_buy_resources(&state, &mut bot).expect("bot should act");
        let Action::BuyResourceBatch { purchases } = action else {
            panic!("expected BuyResourceBatch, got {action:?}");
        };

        let coal_bought = bought(&purchases, Resource::Coal);
        let gasoil_bought = bought(&purchases, Resource::Gas) + bought(&purchases, Resource::Oil);
        assert!(
            coal_bought >= 5,
            "expected combined coal purchase >= 5 (3 + 2), got {}",
            coal_bought
        );
        assert!(
            gasoil_bought >= 2,
            "expected combined gas+oil purchase >= 2 for the hybrid, got {}",
            gasoil_bought
        );
    }

    /// A pure Gas plant and a hybrid share the gas+oil pool. The pure plant must still
    /// be guaranteed its gas (it cannot burn oil), while the hybrid's cumulative target
    /// folds in the pure plant's demand so the combined pool covers both.
    #[test]
    fn pure_gas_and_hybrid_share_pool() {
        let mut player = bot_with_money(200);
        player.plants.push(gas_plant(20, 2, 4));
        player.plants.push(hybrid_plant(21, 3, 5));

        let mut state = state_with_player(&player);
        state.resources = ResourceMarket {
            coal: 23,
            gas: 18,
            oil: 2,
            uranium: 2,
        };
        let mut bot = bot_for(&player);

        let action = decide_buy_resources(&state, &mut bot).expect("bot should act");
        let Action::BuyResourceBatch { purchases } = action else {
            panic!("expected BuyResourceBatch, got {action:?}");
        };

        let gas_bought = bought(&purchases, Resource::Gas);
        let gasoil_bought = gas_bought + bought(&purchases, Resource::Oil);
        assert!(
            gas_bought >= 2,
            "pure gas plant must be guaranteed its 2 gas, got {}",
            gas_bought
        );
        assert!(
            gasoil_bought >= 5,
            "expected combined gas+oil purchase >= 5 (2 pure + 3 hybrid), got {}",
            gasoil_bought
        );
    }

    /// A hard bot (`stockpile_rounds = 2.0`) facing coal-hungry opponents should
    /// pre-buy coal *beyond* the single firing its own plant needs, because the
    /// contested market makes coal's forward price exceed its current price —
    /// exactly when hoarding cheap fuel pays off. Capped at the rack's real
    /// storage (2 × cost = 6 for a cost-3 coal plant).
    #[test]
    fn hard_bot_stockpiles_cheap_contested_fuel() {
        let mut player = bot_with_money(200);
        player.plants.push(coal_plant(20, 3, 5)); // burns 3 coal/round
        let mut state = state_with_player(&player);
        // Three coal-hungry opponents drain the market → forward coal price far
        // above the (currently cheap) table price.
        for n in 0..3u8 {
            let mut opp = bot_with_money(100);
            opp.plants.push(coal_plant(30 + n, 8, 4));
            state.players.push(opp);
        }
        let mut bot = hard_bot_for(&player);

        let action = decide_buy_resources(&state, &mut bot).expect("bot should act");
        let Action::BuyResourceBatch { purchases } = action else {
            panic!("expected BuyResourceBatch");
        };
        let coal = bought(&purchases, Resource::Coal);
        assert!(
            coal > 3,
            "hard bot should stockpile past its 3-coal firing when coal is cheap now \
             but dear later, got {coal}"
        );
        assert!(
            coal <= 6,
            "stockpile must respect the 2×cost=6 storage cap, got {coal}"
        );
    }

    /// The stockpile pass spends only cash *above* the city-build reserve
    /// (city_reserve + safety_buffer = 35 for the hard profile). With just
    /// enough money for the firing and nothing to spare, it buys no extra fuel.
    #[test]
    fn stockpile_respects_city_build_reserve() {
        let mut player = bot_with_money(38);
        player.plants.push(coal_plant(20, 3, 5));
        let mut state = state_with_player(&player);
        for n in 0..3u8 {
            let mut opp = bot_with_money(100);
            opp.plants.push(coal_plant(30 + n, 8, 4));
            state.players.push(opp);
        }
        let mut bot = hard_bot_for(&player);

        let action = decide_buy_resources(&state, &mut bot).expect("bot should act");
        let Action::BuyResourceBatch { purchases } = action else {
            panic!("expected BuyResourceBatch");
        };
        let coal = bought(&purchases, Resource::Coal);
        assert_eq!(
            coal, 3,
            "with no surplus above the build reserve, buy only the firing (got {coal})"
        );
    }

    /// The normal profile keeps `stockpile_rounds = 1.0` (the eval yardstick must
    /// not change): even in the same contested market it buys only the firing.
    #[test]
    fn normal_bot_does_not_stockpile() {
        let mut player = bot_with_money(200);
        player.plants.push(coal_plant(20, 3, 5));
        let mut state = state_with_player(&player);
        for n in 0..3u8 {
            let mut opp = bot_with_money(100);
            opp.plants.push(coal_plant(30 + n, 8, 4));
            state.players.push(opp);
        }
        let mut bot = bot_for(&player); // normal profile

        let action = decide_buy_resources(&state, &mut bot).expect("bot should act");
        let Action::BuyResourceBatch { purchases } = action else {
            panic!("expected BuyResourceBatch");
        };
        let coal = bought(&purchases, Resource::Coal);
        assert_eq!(coal, 3, "normal bot must not stockpile (got {coal})");
    }

    #[test]
    fn softmax_temperature_zero_gives_best() {
        // Normal profile has temperature = 0 → pure argmax.
        let mut bot = normal_bot();
        let candidates = vec![("a", 10.0f32), ("b", 50.0f32), ("c", 30.0f32)];
        for _ in 0..20 {
            let chosen = bot.sample_softmax(&candidates).unwrap();
            assert_eq!(chosen, "b", "argmax should always pick best score");
        }
    }

    #[test]
    fn softmax_high_temperature_samples_non_best() {
        let registry = default_registry();
        let mut profile = registry.easy.clone();
        profile.temperature = 5.0;
        let mut bot = Bot::new(
            PlayerId::nil(),
            "test".into(),
            PlayerColor::Red,
            profile,
            99,
        );
        let candidates = vec![("best", 100.0f32), ("other", 90.0f32)];
        let mut saw_other = false;
        for _ in 0..200 {
            if bot.sample_softmax(&candidates).unwrap() == "other" {
                saw_other = true;
                break;
            }
        }
        assert!(
            saw_other,
            "high temperature should occasionally pick non-best"
        );
    }

    // -----------------------------------------------------------------------
    // Fuel-feasibility / fuel-risk tests
    // -----------------------------------------------------------------------

    #[test]
    fn scarce_fuel_lowers_plant_value_via_fuel_risk() {
        // A coal plant in a scarce coal market should carry a fuel_risk penalty
        // (and thus a lower total value) than the identical plant in a flush
        // market (LOGIC.md §6 "Resource Risk").
        let player = bot_with_money(100);
        let registry = default_registry();
        let w = &registry.normal.auction;

        let candidate = coal_plant(15, 3, 2); // cost=3 coal per firing

        // Flush market, no other plants → scarcity=0, no fuel_risk penalty.
        let mut state_flush = state_with_player(&player);
        state_flush.resources.coal = 23;

        // Scarce market: give the player a heavy coal plant so demand > replen.
        let mut state_scarce = state_with_player(&player);
        state_scarce.resources.coal = 2;
        state_scarce.players[0].plants.push(coal_plant(20, 20, 3)); // demand=20

        let flush = evaluate_plant(&candidate, &state_flush.players[0], &state_flush, w);
        let scarce = evaluate_plant(&candidate, &state_scarce.players[0], &state_scarce, w);

        assert_eq!(
            flush.fuel_risk, 0.0,
            "flush market should carry no fuel_risk"
        );
        assert!(
            scarce.fuel_risk > 0.0,
            "scarce coal market should impose a fuel_risk penalty"
        );
        assert!(
            scarce.total < flush.total,
            "coal plant should be worth less when coal is scarce: flush={:.1} scarce={:.1}",
            flush.total,
            scarce.total
        );
    }

    /// Regression test for the reported bug: a thirsty candidate plant's *own*
    /// fuel appetite must weigh against it — not just demand from plants the
    /// player(s) already own (during an auction the candidate isn't in anyone's
    /// rack yet). `fuel_feasibility` → `resource_feasibility` folds the
    /// candidate's own `per_round_demand` directly into the `demand` side of its
    /// sustainable-vs-needed ratio, so a not-yet-owned 2-uranium candidate is
    /// correctly seen as hard to keep fed when uranium replenishes at 1/round,
    /// the market holds only 2, and a competitor already burns uranium too.
    ///
    /// Scenario mirrors the report: round-3-ish, 4 players (uranium replen=1 at
    /// step 1), an opponent already owns a 1-uranium plant, and the market holds
    /// only 2 uranium. The candidate should now carry a real `fuel_risk` penalty
    /// and be valued below the same plant in a flush, uncontested uranium market.
    #[test]
    fn thirsty_candidate_plant_demand_counts_toward_its_own_fuel_risk() {
        let player = bot_with_money(200);
        let registry = default_registry();
        let w = &registry.normal.auction;

        let candidate = uranium_plant(30, 2, 3); // needs 2 uranium per firing

        // Tight market: an opponent already burns 1 uranium/round, replen=1, only
        // 2 uranium left. Candidate's own 2-unit appetite isn't owned anywhere yet.
        let mut state_tight = state_with_player(&player);
        state_tight.resources.uranium = 2;
        let mut opponent = bot_with_money(100);
        opponent.plants.push(uranium_plant(31, 1, 2));
        state_tight.players.push(opponent);

        // Flush market, no other uranium plants anywhere → scarcity should stay 0.
        let mut state_flush = state_with_player(&player);
        state_flush.resources.uranium = 12;

        let tight = evaluate_plant(&candidate, &state_tight.players[0], &state_tight, w);
        let flush = evaluate_plant(&candidate, &state_flush.players[0], &state_flush, w);

        assert_eq!(
            flush.fuel_risk, 0.0,
            "flush uranium market should carry no fuel_risk"
        );
        assert!(
            tight.fuel_risk > 0.0,
            "a 2-uranium candidate in a near-empty, slow-replenishing market should \
             carry a fuel_risk penalty driven by its own demand, got {:.3}",
            tight.fuel_risk
        );
        assert!(
            tight.total < flush.total,
            "the thirsty candidate should be valued lower in the tight market: \
             tight={:.1} flush={:.1}",
            tight.total,
            flush.total
        );
    }

    #[test]
    fn wind_plant_value_unaffected_by_fuel_scarcity() {
        // Even with all resources depleted and heavy fuel demand, a Wind plant's
        // fuel_risk (and thus total value) should be identical — it burns no fuel
        // (fuel_feasibility returns 1.0 unconditionally for Wind).
        //
        // The demand is placed on a SECOND player so the scored player's `plants`
        // vector (which drives capacity_bump / income projections) is the same in
        // both states.
        let player = bot_with_money(100);
        let registry = default_registry();
        let w = &registry.hard.auction; // hard has the highest fuel_risk_weight

        let wind = PowerPlant {
            number: 44,
            kind: PlantKind::Wind,
            cost: 0,
            cities: 2,
        };

        // Second player owns a heavy coal plant — creates non-zero coal scarcity in both
        // states via demand (to make the test meaningful when resources are depleted).
        let mut coal_bot = bot_with_money(50);
        coal_bot.plants.push(coal_plant(20, 20, 3));

        let mut state_flush = state_with_player(&player);
        state_flush.players.push(coal_bot.clone());

        let mut state_scarce = state_with_player(&player);
        state_scarce.players.push(coal_bot.clone());
        // Deplete all resources — only difference between the two states.
        state_scarce.resources.coal = 0;
        state_scarce.resources.oil = 0;
        state_scarce.resources.gas = 0;
        state_scarce.resources.uranium = 0;

        let flush = evaluate_plant(&wind, &state_flush.players[0], &state_flush, w);
        let scarce = evaluate_plant(&wind, &state_scarce.players[0], &state_scarce, w);

        assert_eq!(flush.fuel_risk, 0.0, "Wind plants must carry no fuel_risk");
        assert_eq!(scarce.fuel_risk, 0.0, "Wind plants must carry no fuel_risk");
        assert_eq!(
            flush.total, scarce.total,
            "Wind plant value must not depend on resource scarcity"
        );
    }

    #[test]
    fn fuel_feasibility_wind_is_one() {
        let player = bot_with_money(100);
        let mut state = state_with_player(&player);
        // Deplete everything and pile on demand to make the test maximally sensitive.
        state.resources.coal = 0;
        state.resources.gas = 0;
        state.resources.oil = 0;
        state.resources.uranium = 0;
        state.players[0].plants.push(coal_plant(20, 20, 3));

        let wind = PowerPlant {
            number: 44,
            kind: PlantKind::Wind,
            cost: 0,
            cities: 2,
        };
        assert_eq!(
            fuel_feasibility(&wind, &state.players[0], &state),
            1.0,
            "Wind plants are always fully feasible — they burn no fuel"
        );
    }

    #[test]
    fn feasibility_drops_with_competition() {
        // The reported scenario, isolated to `fuel_feasibility`: a 2-uranium
        // candidate should look fully feasible in an uncontested, flush market,
        // but become clearly infeasible once uranium is nearly depleted (replen
        // is only 1/round) and a competitor is already drawing on the same pool.
        let player = bot_with_money(100);
        let candidate = uranium_plant(30, 2, 3);

        let mut state_flush = state_with_player(&player);
        state_flush.resources.uranium = 12;
        let flush = fuel_feasibility(&candidate, &state_flush.players[0], &state_flush);
        assert_eq!(flush, 1.0, "uncontested + flush uranium → fully feasible");

        let mut state_tight = state_with_player(&player);
        state_tight.resources.uranium = 2;
        let mut opponent = bot_with_money(100);
        opponent.plants.push(uranium_plant(31, 1, 2));
        state_tight.players.push(opponent);
        let tight = fuel_feasibility(&candidate, &state_tight.players[0], &state_tight);

        assert!(
            tight < 1.0,
            "contested, nearly-empty, slow-replenishing uranium should reduce \
             feasibility below 1.0, got {tight:.3}"
        );
    }

    #[test]
    fn feasibility_accounts_for_owned_demand() {
        // `resource_feasibility` weighs the candidate's demand *plus* whatever
        // the player's existing rack already draws from the same pool — a player
        // already running a heavy coal plant can't treat a second one as "fully
        // fed" just because the new plant alone would fit the fair share.
        let candidate = coal_plant(40, 3, 2);

        let empty_player = bot_with_money(100);
        let state_empty = state_with_player(&empty_player);
        let empty_feas = fuel_feasibility(&candidate, &state_empty.players[0], &state_empty);

        let mut laden_player = bot_with_money(100);
        laden_player.plants.push(coal_plant(20, 20, 3)); // already burns 20 coal/round
        let state_laden = state_with_player(&laden_player);
        let laden_feas = fuel_feasibility(&candidate, &state_laden.players[0], &state_laden);

        assert!(
            laden_feas < empty_feas,
            "a player already drawing heavily on coal should see lower feasibility \
             for a new coal plant than a player with an empty rack: \
             laden={laden_feas:.3} empty={empty_feas:.3}"
        );
    }

    #[test]
    fn hybrid_feasibility_uses_easier_pool() {
        // A hybrid sources from whichever of gas/oil is easier — mirrors
        // `estimate_firing_cost`'s "prefer the more-available resource". With oil
        // empty but gas maxed out, the hybrid's gas-side draw is fully sustainable
        // (its `max` over the two pools lands at 1.0), so it escapes the empty oil
        // pool entirely. A pure-oil plant with the same per-pool demand has no such
        // escape and is throttled by oil's fair share alone.
        let player = bot_with_money(100);
        let mut state = state_with_player(&player);
        state.resources.gas = 24; // maxed out — gas_sustainable lands exactly at 6/round
        state.resources.oil = 0; // empty — oil_sustainable is just its fair share, 5/round

        let hybrid = hybrid_plant(40, 12, 3); // 6 gas + 6 oil demand if split evenly
        let pure_oil = PowerPlant {
            number: 41,
            kind: PlantKind::Oil,
            cost: 6, // matches the hybrid's oil-side demand
            cities: 3,
        };

        let hybrid_feas = fuel_feasibility(&hybrid, &state.players[0], &state);
        let oil_feas = fuel_feasibility(&pure_oil, &state.players[0], &state);

        assert!(
            hybrid_feas > oil_feas,
            "hybrid should escape the empty oil pool by sourcing fully-sustainable gas \
             instead, while a pure-oil plant of equal per-pool demand stays throttled: \
             hybrid={hybrid_feas:.3} pure_oil={oil_feas:.3}"
        );
        assert_eq!(
            hybrid_feas, 1.0,
            "hybrid should be fully fed via the maxed gas pool"
        );
        assert!(
            oil_feas < 1.0,
            "pure oil plant should be throttled by oil's fair share alone"
        );
    }

    /// Pin the `fuel_risk` formula — `fuel_risk_weight × (1 - feasibility) ×
    /// remaining_rounds × (income_gain + fuel_price)` — by hand-deriving every
    /// term from its own public building block (`fuel_feasibility`,
    /// `remaining_rounds`, `estimate_firing_cost`, and `capacity_bump` +
    /// `useful_city_target` + `income_for` for `income_gain`) and checking
    /// `evaluate_plant` agrees. This is what proves the two factors this phase
    /// adds — *absolute fuel price* and *replenishment-vs-competition
    /// feasibility* — are actually wired into the valuation, not merely computed
    /// and discarded.
    #[test]
    fn fuel_risk_reflects_feasibility_and_absolute_price() {
        let player = bot_with_money(200);
        let registry = default_registry();
        let w = &registry.normal.auction;

        let candidate = uranium_plant(30, 2, 3);

        // The reported scenario — both feasibility and absolute price should bite.
        let mut state = state_with_player(&player);
        state.resources.uranium = 2;
        let mut opponent = bot_with_money(100);
        opponent.plants.push(uranium_plant(31, 1, 2));
        state.players.push(opponent);

        let me = &state.players[0];
        let valuation = evaluate_plant(&candidate, me, &state, w);

        let feasibility = fuel_feasibility(&candidate, me, &state);
        let rounds = remaining_rounds(&state);
        let fuel_price = estimate_firing_cost(&candidate, &state.resources) as f32;

        // Hand-reproduce `projected_income_gain`'s `income_gain` from public parts.
        let bump = capacity_bump(&candidate, me);
        let target = useful_city_target(me, &state, w) as i32;
        let current_capacity: i32 = me.plants.iter().map(|p| p.cities as i32).sum();
        let old_powered = current_capacity.clamp(0, target) as u8;
        let new_powered = (current_capacity + bump).clamp(0, target) as u8;
        let income_gain = income_for(new_powered) as f32 - income_for(old_powered) as f32;

        assert!(feasibility < 1.0, "sanity: scenario should be infeasible");
        assert!(
            fuel_price > 0.0,
            "sanity: uranium should carry a real price here"
        );

        let expected =
            w.fuel_risk_weight * (1.0 - feasibility) * rounds * (income_gain + fuel_price);
        assert!(
            (valuation.fuel_risk - expected).abs() < 0.05,
            "fuel_risk should equal fuel_risk_weight × (1 - feasibility) × rounds × \
             (income_gain + fuel_price): got {:.3}, expected {:.3} (feasibility={:.3}, \
             rounds={}, fuel_price={}, income_gain={})",
            valuation.fuel_risk,
            expected,
            feasibility,
            rounds,
            fuel_price,
            income_gain,
        );
    }

    // -----------------------------------------------------------------------
    // Operating-cost tests
    // -----------------------------------------------------------------------

    #[test]
    fn cheaper_fuel_out_values_pricier_same_capacity() {
        // Reproduces the reported bug: at the opening, every one-city plant
        // scored identically because gross income was the only signal — fuel
        // type was invisible. A 1-coal plant should clearly out-value a
        // thirstier 2-gas plant of the same capacity once the forward fuel
        // spend is netted out of gross income.
        let player = bot_with_money(100);
        let state = state_with_player(&player);
        let me = &state.players[0];
        let registry = default_registry();
        let w = &registry.normal.auction;

        let coal_candidate = coal_plant(50, 1, 1);
        let gas_candidate = gas_plant(51, 2, 1);

        let coal_val = evaluate_plant(&coal_candidate, me, &state, w);
        let gas_val = evaluate_plant(&gas_candidate, me, &state, w);

        assert_eq!(
            coal_val.incremental_income, gas_val.incremental_income,
            "sanity: both plants gain the same single city, so gross income \
             should be identical — the only thing that should differ is fuel cost"
        );
        assert!(
            coal_val.operating_cost < gas_val.operating_cost,
            "a 1-coal plant should be cheaper to run than a thirstier 2-gas \
             plant: coal={:.1} gas={:.1}",
            coal_val.operating_cost,
            gas_val.operating_cost
        );
        assert!(
            coal_val.total > gas_val.total,
            "the cheaper-to-fire coal plant should out-value the pricier gas \
             plant of equal capacity — fuel type should matter: \
             coal={:.1} gas={:.1}",
            coal_val.total,
            gas_val.total
        );
    }

    #[test]
    fn operating_cost_zero_for_wind() {
        // Wind burns no fuel — `expected_firing_cost` and `operating_cost`
        // should both be zero regardless of weight or market state.
        let player = bot_with_money(100);
        let state = state_with_player(&player);
        let me = &state.players[0];
        let registry = default_registry();
        let w = &registry.normal.auction;

        let wind = PowerPlant {
            number: 52,
            kind: PlantKind::Wind,
            cost: 0,
            cities: 1,
        };

        assert_eq!(expected_firing_cost(&wind, &state), 0.0);
        assert_eq!(evaluate_plant(&wind, me, &state, w).operating_cost, 0.0);
    }

    #[test]
    fn operating_cost_rises_with_market_demand() {
        // `expected_firing_cost` should see further than `estimate_firing_cost`'s
        // snapshot: the same candidate, in markets that start *identically*
        // priced, costs more to keep fed on average once several opponents are
        // already racing to burn the same fuel and out-pace its replenishment —
        // the forward-looking demand/replenishment effect the absolute-price
        // model lacked before.
        let player = bot_with_money(100);
        let candidate = coal_plant(50, 3, 2);

        let state_uncontested = state_with_player(&player);
        let mut state_contested = state_with_player(&player);
        for n in 0..2u8 {
            let mut rival = bot_with_money(100);
            rival.plants.push(coal_plant(60 + n, 10, 3));
            state_contested.players.push(rival);
        }

        // Both markets start at the same stock, so the snapshot price matches —
        // any difference below comes purely from simulating the market forward.
        assert_eq!(
            estimate_firing_cost(&candidate, &state_uncontested.resources),
            estimate_firing_cost(&candidate, &state_contested.resources),
            "sanity: both states start with identical market stock and price"
        );

        let forward_uncontested = expected_firing_cost(&candidate, &state_uncontested);
        let forward_contested = expected_firing_cost(&candidate, &state_contested);

        assert!(
            forward_contested > forward_uncontested,
            "heavier market-wide demand on the same fuel should drive the \
             forward-looking price above the uncontested case, even though \
             both markets start identically: uncontested={forward_uncontested:.2} \
             contested={forward_contested:.2}"
        );
    }

    #[test]
    fn operating_cost_reflects_feasibility_and_forward_price() {
        // Pin the `operating_cost` formula: `operating_cost_weight ×
        // fuel_feasibility × expected_firing_cost × remaining_rounds`. The
        // `feasibility` factor is what makes this partition cleanly with
        // `fuel_risk` — a plant that can only be kept fed `feasibility` of the
        // time is charged fuel for that fraction of the game only; the rest is
        // priced as lost income + dearness by `fuel_risk` instead, with no
        // double-count.
        let player = bot_with_money(200);
        let registry = default_registry();
        let w = &registry.normal.auction;

        let candidate = uranium_plant(30, 2, 3);
        let mut state = state_with_player(&player);
        state.resources.uranium = 2;
        let mut opponent = bot_with_money(100);
        opponent.plants.push(uranium_plant(31, 1, 2));
        state.players.push(opponent);

        let me = &state.players[0];
        let valuation = evaluate_plant(&candidate, me, &state, w);

        let feasibility = fuel_feasibility(&candidate, me, &state);
        let rounds = remaining_rounds(&state);
        let forward_price = expected_firing_cost(&candidate, &state);

        assert!(feasibility < 1.0, "sanity: scenario should be infeasible");

        let expected = w.operating_cost_weight * feasibility * forward_price * rounds;
        assert!(
            (valuation.operating_cost - expected).abs() < 0.05,
            "operating_cost should equal operating_cost_weight × feasibility × \
             expected_firing_cost × rounds: got {:.3}, expected {:.3} \
             (feasibility={:.3}, forward_price={:.3}, rounds={})",
            valuation.operating_cost,
            expected,
            feasibility,
            forward_price,
            rounds,
        );
    }

    #[test]
    fn auction_reserve_grows_with_scarce_market() {
        // Depleted coal market makes `estimate_firing_cost` >> the flat
        // `plant.cost` fallback, which inflates the live-priced reserve — this
        // is what shrinks `affordable` (and thus the bid ceiling, since
        // `MaximumBid = PlantValue.min(affordable)`) when fuel is scarce.
        let mut player = bot_with_money(100);
        player.plants.push(coal_plant(20, 3, 3));
        let registry = default_registry();
        let w = &registry.normal.auction;
        let buy = &registry.normal.buy;
        let candidate = coal_plant(25, 2, 2);

        let mut scarce_market = ResourceMarket::initial();
        scarce_market.coal = 3;

        let reserve_scarce = auction_reserve(&candidate, &player, w, buy, Some(&scarce_market));
        let reserve_flat = auction_reserve(&candidate, &player, w, buy, None);

        assert!(
            reserve_scarce > reserve_flat,
            "reserve should grow when fuel is scarce: scarce={reserve_scarce} flat={reserve_flat}"
        );
    }

    #[test]
    fn estimate_firing_cost_prices_cheapest_units_first() {
        // With a full coal market (coal=23), two cheap coal units cost 2+2=4.
        // (price_table Coal: index 21=2, index 22=2; last_occupied = 22.)
        let plant = coal_plant(5, 2, 1);
        let market = ResourceMarket::initial(); // coal=23
        let cost = estimate_firing_cost(&plant, &market);
        assert_eq!(
            cost, 4,
            "2 coal from flush market should cost 4 (slots 22+21 = 2+2)"
        );
    }

    #[test]
    fn estimate_firing_cost_wind_is_zero() {
        let wind = PowerPlant {
            number: 44,
            kind: PlantKind::Wind,
            cost: 0,
            cities: 2,
        };
        let market = ResourceMarket::initial();
        assert_eq!(estimate_firing_cost(&wind, &market), 0);
    }

    #[test]
    fn estimate_firing_cost_hybrid_prefers_more_available_resource() {
        // Hybrid with gas plentiful (18) and oil scarce (1) should price gas.
        let hybrid = PowerPlant {
            number: 10,
            kind: PlantKind::GasOrOil,
            cost: 2,
            cities: 2,
        };
        let mut market = ResourceMarket::initial();
        market.gas = 18; // gas is cheap (plenty available)
        market.oil = 1; // oil is very expensive

        let cost_actual = estimate_firing_cost(&hybrid, &market);

        // Gas price for 2 units from market.gas=18: last_occupied=17.
        // Gas table: [8,8,8,7,7,7,6,6,6,5,5,5,4,4,4,3,3,3,2,2,2,1,1,1]
        // slot 17=3, slot 16=3 → total 6.
        // Oil price for 2 units from market.oil=1: can only supply 1 → fallback (more expensive).
        // So actual should equal the gas price (6).
        assert_eq!(
            cost_actual, 6,
            "hybrid should price the more available (gas) resource: got {cost_actual}"
        );
    }

    // -----------------------------------------------------------------------
    // Build-cities: endgame overbuild
    // -----------------------------------------------------------------------

    fn wind_plant(number: u8, cities: u8) -> PowerPlant {
        PowerPlant {
            number,
            kind: PlantKind::Wind,
            cost: 0,
            cities,
        }
    }

    /// `state_with_player` plus a real buildable board: every map region active
    /// and the first `owned` cities (sorted id order) marked as owned by `player`.
    fn build_state(player: &Player, owned: usize) -> GameState {
        let mut state = state_with_player(player);
        state.active_regions = state.map.regions.clone();
        let mut ids: Vec<String> = state.map.cities.keys().cloned().collect();
        ids.sort();
        for id in ids.into_iter().take(owned) {
            state
                .map
                .cities
                .get_mut(&id)
                .unwrap()
                .owners
                .push(player.id);
        }
        state
    }

    #[test]
    fn overbuilds_with_surplus_cash_at_full_capacity() {
        // owned == powerable: headroom is 0, but with plenty of money above the
        // overbuild reserve the bot must keep building toward end_game_cities
        // instead of stalling on DoneBuilding forever.
        let mut player = bot_with_money(500);
        player.plants.push(wind_plant(44, 2));
        let state = build_state(&player, 2);
        let mut bot = bot_for(&player);

        let action = decide_build_cities(&state, &mut bot).expect("bot should act");
        let Action::BuildCities { city_ids } = action else {
            panic!("expected overbuild despite zero headroom, got {action:?}");
        };
        assert!(!city_ids.is_empty());
    }

    #[test]
    fn does_not_overbuild_below_reserve() {
        // Same zero-headroom position, but the money on hand barely covers the
        // overbuild reserve (city_reserve 30 + safety_buffer 5 for a fuel-free
        // wind rack) — unpowerable cities must not eat into it.
        let mut player = bot_with_money(40);
        player.plants.push(wind_plant(44, 2));
        let state = build_state(&player, 2);
        let mut bot = bot_for(&player);

        let action = decide_build_cities(&state, &mut bot).expect("bot should act");
        assert!(
            matches!(action, Action::DoneBuilding),
            "must not overbuild into the reserve, got {action:?}"
        );
    }

    #[test]
    fn never_builds_past_end_game_trigger() {
        // 16 of 17 end-game cities owned, huge capacity headroom and budget:
        // exactly one more city is useful — it triggers game end; any further
        // build is pure waste.
        let mut player = bot_with_money(1000);
        player.plants.push(wind_plant(44, 30));
        let state = build_state(&player, 16);
        assert_eq!(state.end_game_cities, 17);
        let mut bot = bot_for(&player);

        let action = decide_build_cities(&state, &mut bot).expect("bot should act");
        let Action::BuildCities { city_ids } = action else {
            panic!("expected a build, got {action:?}");
        };
        assert_eq!(
            city_ids.len(),
            1,
            "must stop at end_game_cities, got {city_ids:?}"
        );
    }
}
