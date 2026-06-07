use powergrid_core::{
    actions::Action,
    rules::effective_min_bid,
    state::GameState,
    types::{
        connection_cost, income_for, PlantKind, Player, PlayerId, PowerPlant, Resource,
        ResourceMarket,
    },
};
use tracing::{debug, info};

use crate::{
    bot::Bot,
    features::{
        auction_reserve, capacity_bump, city_contest_bonus, evaluate_plant, plant_score,
        should_skip_auction,
    },
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
    if !is_round_one {
        candidates.push((AuctionCandidate::Pass, w.min_open_score));
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

    for plant in &plants {
        buy_for_plant(
            plant,
            plant.cost,
            &mut sim_market,
            &mut sim_player,
            &mut budget,
            &mut purchases,
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
    let available = market.available(resource);
    let cap = want.min(available);
    if cap == 0 {
        return;
    }
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
                return;
            }
        }
    }
    debug!(
        "Cannot afford any {:?} (want {}, budget {})",
        resource, want, budget
    );
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

    // Sort by (cost - contest_bonus) ascending — cheapest and most contested first.
    candidates.sort_by(|(_, cost_a, bonus_a), (_, cost_b, bonus_b)| {
        let adjusted_a = *cost_a as f32 - bonus_a;
        let adjusted_b = *cost_b as f32 - bonus_b;
        adjusted_a
            .partial_cmp(&adjusted_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Only buy up to capacity headroom: cities we can actually power.
    // Buying more than that never increases income and wastes the city-build budget.
    let powerable: u8 = player.plants.iter().map(|p| p.cities).sum();
    let owned = owned_cities.len() as u8;
    let headroom = powerable.saturating_sub(owned) as usize;

    let mut city_ids: Vec<String> = Vec::new();
    let mut simulated_cities: Vec<String> = owned_cities.clone();

    for (city_id, _, _) in &candidates {
        if city_ids.len() >= headroom {
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

        if total <= budget {
            info!(
                "Building in {} (route={}, slot={}, total={})",
                city_id, route_cost, slot_cost, total
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
            auction_reserve, estimate_firing_cost, evaluate_plant, fuel_scarcity,
            plant_fuel_scarcity, remaining_rounds, useful_city_target,
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
    // Resource-scarcity tests
    // -----------------------------------------------------------------------

    /// Build a 1-player state and give that player coal plants so total coal
    /// demand exceeds the 1-player (≈6-player) replenishment of 7/round.
    fn state_with_coal_demand(demand_cost: u8) -> GameState {
        let player = bot_with_money(100);
        let mut state = state_with_player(&player);
        // Give the single player a coal plant that burns `demand_cost` units.
        state.players[0].plants.push(coal_plant(20, demand_cost, 3));
        state
    }

    #[test]
    fn fuel_scarcity_rises_with_lower_availability() {
        // demand=10 > replen_coal=7 (step 1, 1 player → "6-player" bracket) → shortfall=3.
        // scarcity = shortfall / (avail + replen + 1).  Lower avail → lower denominator → higher.
        let mut state = state_with_coal_demand(10);

        state.resources.coal = 23; // flush market
        let flush = fuel_scarcity(&state, Resource::Coal);

        state.resources.coal = 3; // depleted market
        let scarce = fuel_scarcity(&state, Resource::Coal);

        assert!(
            scarce > flush,
            "scarcity should rise when availability drops: flush={flush:.3} scarce={scarce:.3}"
        );
    }

    #[test]
    fn fuel_scarcity_rises_with_higher_demand() {
        let player = bot_with_money(100);
        let mut state = state_with_player(&player);
        state.resources.coal = 10;

        // No coal plants → demand=0, shortfall=0, scarcity=0.
        let low = fuel_scarcity(&state, Resource::Coal);

        // Add a high-demand coal plant (cost=20 > replen=7).
        state.players[0].plants.push(coal_plant(20, 20, 3));
        let high = fuel_scarcity(&state, Resource::Coal);

        assert!(
            high > low,
            "scarcity should rise when demand exceeds replenishment: low={low:.3} high={high:.3}"
        );
    }

    #[test]
    fn hybrid_demand_splits_gas_oil_not_coal() {
        // A GasOrOil hybrid should add demand to gas and oil but NOT to coal.
        // We set coal very low — if hybrid mistakenly contributed coal demand,
        // coal scarcity would be nonzero (demand would exceed replenishment).
        let player = bot_with_money(100);
        let mut state = state_with_player(&player);

        // Add a hybrid plant with high cost so demand/2 exceeds replen for both gas and oil.
        // replen_gas=3, replen_oil=5 (1-player → "6-player" bracket, step 1).
        // cost=14 → gas_demand=7 > 3, oil_demand=7 > 5.  Coal demand stays 0.
        state.players[0].plants.push(PowerPlant {
            number: 10,
            kind: PlantKind::GasOrOil,
            cost: 14,
            cities: 2,
        });
        state.resources.coal = 1; // very scarce — would drive coal scarcity up if demand existed

        let coal_sc = fuel_scarcity(&state, Resource::Coal);
        let gas_sc = fuel_scarcity(&state, Resource::Gas);
        let oil_sc = fuel_scarcity(&state, Resource::Oil);

        assert_eq!(
            coal_sc, 0.0,
            "hybrid plant must not contribute to coal demand (scarcity={coal_sc:.4})"
        );
        assert!(
            gas_sc > 0.0,
            "hybrid must contribute to gas demand (scarcity={gas_sc:.4})"
        );
        assert!(
            oil_sc > 0.0,
            "hybrid must contribute to oil demand (scarcity={oil_sc:.4})"
        );
    }

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

    #[test]
    fn wind_plant_value_unaffected_by_fuel_scarcity() {
        // Even with all resources depleted and heavy fuel demand, a Wind plant's
        // fuel_risk (and thus total value) should be identical — it burns no fuel
        // (plant_fuel_scarcity returns 0.0).
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
    fn plant_fuel_scarcity_wind_is_zero() {
        let player = bot_with_money(100);
        let mut state = state_with_player(&player);
        // Deplete everything to make the test maximally sensitive.
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
            plant_fuel_scarcity(&wind, &state),
            0.0,
            "Wind plants have zero fuel scarcity by definition"
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
}
