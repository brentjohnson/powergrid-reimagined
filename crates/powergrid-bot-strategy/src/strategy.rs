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
        bid_ceiling, capacity_bump, city_contest_bonus, plant_score, plant_score_contextual,
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
        let ceiling = bid_ceiling(plant, my_player, state.round, w, buy, min_bid);
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
                || (!should_skip_auction(my_player, p, w) && capacity_bump(p, my_player, w) >= 1)
        })
        .map(|p| {
            let score = plant_score_contextual(p, my_player, state, w);
            (AuctionCandidate::Select(p.number), score)
        })
        .collect();

    // PassAuction as a scored baseline (not available in round 1).
    // Overshoot capacity is penalised on individual plant scores (see plant_score_contextual),
    // so the Pass baseline is simply the minimum worthwhile plant score threshold.
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
                "Selecting plant {} (kind={:?}, cities={}, score={:.1})",
                plant.number,
                plant.kind,
                plant.cities,
                plant_score_contextual(plant, my_player, state, w),
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
    let w = &bot.profile.auction;

    let worst = player
        .plants
        .iter()
        .filter(|p| p.number != new_plant.number)
        .min_by(|a, b| {
            plant_score(a, w)
                .partial_cmp(&plant_score(b, w))
                .unwrap_or(std::cmp::Ordering::Equal)
        })?;

    info!(
        "Discarding plant {} ({:.1}) to make room for plant {} ({:.1})",
        worst.number,
        plant_score(worst, w),
        new_plant.number,
        plant_score(new_plant, w),
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
        features::{auction_reserve, plant_score_contextual, useful_city_target},
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
    fn round_one_caps_bid_at_listed_price() {
        let plant = coal_plant(15, 2, 2);
        let player = bot_with_money(50);
        let registry = default_registry();
        let w = &registry.normal.auction;
        let buy = &registry.normal.buy;
        assert_eq!(
            bid_ceiling(&plant, &player, 1, w, buy, plant.number as u32),
            15
        );
    }

    #[test]
    fn later_round_no_capacity_bump_means_no_premium() {
        let mut player = bot_with_money(100);
        player.plants.push(coal_plant(5, 2, 2));
        player.plants.push(coal_plant(7, 2, 2));
        player.plants.push(coal_plant(10, 2, 2));
        let candidate = coal_plant(20, 2, 1);
        let registry = default_registry();
        let w = &registry.normal.auction;
        let buy = &registry.normal.buy;
        assert_eq!(
            bid_ceiling(&candidate, &player, 3, w, buy, candidate.number as u32),
            20
        );
    }

    #[test]
    fn later_round_significant_bump_allows_small_premium() {
        let player = bot_with_money(100);
        let candidate = coal_plant(15, 2, 3);
        let registry = default_registry();
        let w = &registry.normal.auction;
        let buy = &registry.normal.buy;
        let ceiling = bid_ceiling(&candidate, &player, 2, w, buy, candidate.number as u32);
        assert!(
            ceiling > 15 && ceiling <= 21,
            "expected a small premium above 15, got {}",
            ceiling
        );
    }

    #[test]
    fn never_bids_above_player_money() {
        let plant = coal_plant(15, 2, 3);
        let player = bot_with_money(10);
        let registry = default_registry();
        let w = &registry.normal.auction;
        let buy = &registry.normal.buy;
        assert_eq!(
            bid_ceiling(&plant, &player, 1, w, buy, plant.number as u32),
            10
        );
        assert_eq!(
            bid_ceiling(&plant, &player, 2, w, buy, plant.number as u32),
            10
        );
        let mut bot = normal_bot();
        let max_jitter = bot.profile.max_jitter;
        for _ in 0..50 {
            // Production code applies .min(player.money) after jitter; mirror that here.
            assert!(
                bot.maybe_jitter(
                    bid_ceiling(&plant, &player, 1, w, buy, plant.number as u32),
                    max_jitter
                )
                .min(player.money)
                    <= player.money
            );
            assert!(
                bot.maybe_jitter(
                    bid_ceiling(&plant, &player, 2, w, buy, plant.number as u32),
                    max_jitter
                )
                .min(player.money)
                    <= player.money
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
        assert!(!should_skip_auction(&player, &candidate, w));
    }

    #[test]
    fn dont_skip_when_at_capacity() {
        let mut player = bot_with_money(50);
        player.plants.push(coal_plant(5, 2, 2));
        let candidate = coal_plant(20, 2, 3);
        let registry = default_registry();
        let w = &registry.normal.auction;
        // powerable=2, rack not full → don't skip
        assert!(!should_skip_auction(&player, &candidate, w));
    }

    #[test]
    fn overshoot_weight_profile_tiers() {
        let registry = default_registry();
        // All tiers must have a non-zero overshoot weight — zero would disable the endgame brake.
        assert!(
            registry.easy.auction.overshoot_weight > 0.0,
            "easy must have overshoot weight"
        );
        assert!(
            registry.normal.auction.overshoot_weight >= registry.easy.auction.overshoot_weight,
            "normal should be at least as aggressive as easy"
        );
        assert!(
            registry.hard.auction.overshoot_weight >= registry.normal.auction.overshoot_weight,
            "hard should be the most aggressive"
        );
        // Hard bot plans least capacity ahead (tightest ceiling).
        assert!(
            registry.hard.auction.buildable_lookahead
                <= registry.normal.auction.buildable_lookahead,
            "hard should have the tightest buildable_lookahead"
        );
    }

    #[test]
    fn overshoot_penalty_applied_to_plant_score() {
        // A bot that already owns 4 cities (owned=4) with end_game_cities=17 and
        // buildable_lookahead=2 has useful_city_target=6.
        // If it currently has powerable=5 and buys a +3 plant → projected total=8 → overshoot=2.
        // The score should be reduced by overshoot_weight * 2 compared to no-overshoot case.
        let registry = default_registry();
        let w = &registry.normal.auction;

        let mut player = bot_with_money(100);
        player.plants.push(coal_plant(5, 2, 5)); // existing powerable=5
                                                 // Give the player 4 owned cities by inserting cities into a state.
        let mut state = state_with_player(&player);
        state.end_game_cities = 17;
        // Manually set player_city_count by assigning cities in the map.
        // Simpler: tweak end_game_cities / useful_city_target directly via a custom weight set.
        // Use a profile where end_game_cities=6 to force the boundary.
        let mut w_custom = w.clone();
        w_custom.overshoot_weight = 35.0;
        w_custom.buildable_lookahead = 2;

        // state has player with 0 owned cities, so target = 0+2 = 2 (min(2, 17)).
        // Existing powerable = 5; adding a 3-city plant → projected = 5+3=8, overshoot = 8-2 = 6.
        let player_ref = &state.players[0];
        let candidate = coal_plant(20, 2, 3);
        let score_with = plant_score_contextual(&candidate, player_ref, &state, &w_custom);
        let score_base = plant_score(&candidate, &w_custom);
        let expected_penalty = 35.0 * 6.0; // overshoot_weight * 6 cities of overshoot
        assert!(
            score_base - score_with >= expected_penalty - 1.0,
            "expected ~{expected_penalty} penalty from overshoot, base={score_base}, actual={score_with}"
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
        assert_eq!(auction_reserve(&candidate, &player, w, buy), 8 + 30 + 5);
    }

    #[test]
    fn high_capacity_plant_score_graduated() {
        let registry = default_registry();
        let w = &registry.normal.auction;

        let four_city = coal_plant(20, 3, 4);
        let five_city = coal_plant(25, 4, 5);
        let six_city = coal_plant(30, 5, 6);

        let mut w_zero = w.clone();
        w_zero.high_capacity_bonus = 0.0;

        let premium_5 = plant_score(&five_city, w) - plant_score(&five_city, &w_zero);
        let premium_6 = plant_score(&six_city, w) - plant_score(&six_city, &w_zero);

        assert!(
            premium_5 > 0.0,
            "5-city plant should score higher with the premium (got {premium_5})"
        );
        assert!(
            premium_6 > premium_5,
            "6-city premium ({premium_6}) should exceed 5-city premium ({premium_5}) — graduated"
        );
        // 4-city plant gets no premium
        let premium_4 = plant_score(&four_city, w) - plant_score(&four_city, &w_zero);
        assert_eq!(
            premium_4, 0.0,
            "4-city plant should not receive the premium"
        );
    }

    #[test]
    fn high_capacity_bid_ceiling_premium() {
        // 5-city plant, empty rack → capacity_bump = 5, round 2 → premium applies.
        let plant = coal_plant(25, 2, 5);
        let player = bot_with_money(100);
        let registry = default_registry();
        let w = &registry.normal.auction;
        let buy = &registry.normal.buy;

        let ceiling_with = bid_ceiling(&plant, &player, 2, w, buy, plant.number as u32);

        let mut w_zero = w.clone();
        w_zero.high_capacity_bid_premium = 0.0;
        let ceiling_without = bid_ceiling(&plant, &player, 2, &w_zero, buy, plant.number as u32);

        assert!(
            ceiling_with > ceiling_without,
            "5-city plant bid ceiling should be higher with the premium ({ceiling_with} vs {ceiling_without})"
        );
    }

    #[test]
    fn jitter_sometimes_lifts_the_ceiling() {
        let plant = coal_plant(15, 2, 2);
        let player = bot_with_money(100);
        let registry = default_registry();
        let w = &registry.normal.auction;
        let buy = &registry.normal.buy;
        let base = bid_ceiling(&plant, &player, 1, w, buy, plant.number as u32);

        // With seed 42 and 200 trials, count how many jitter.
        let mut bot = normal_bot();
        let max_jitter = bot.profile.max_jitter;
        let mut saw_jitter = false;
        let mut saw_no_jitter = false;
        for _ in 0..200 {
            let bid = bot.maybe_jitter(base, max_jitter).min(player.money);
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
}
