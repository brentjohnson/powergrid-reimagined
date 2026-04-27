use powergrid_core::{
    actions::Action,
    state::GameState,
    types::{
        connection_cost, income_for, PlantKind, Player, PlayerId, PowerPlant, Resource,
        ResourceMarket,
    },
};
use tracing::{debug, info};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Given the current game state and the bot's player id, decide what action to
/// take next.  Returns `None` when it is not the bot's turn to act.
pub fn decide(state: &GameState, me: PlayerId) -> Option<Action> {
    use powergrid_core::types::Phase;
    match &state.phase {
        Phase::Lobby | Phase::PlayerOrder | Phase::GameOver { .. } => None,

        Phase::Auction {
            current_bidder_idx,
            active_bid,
            bought,
            passed,
        } => decide_auction(state, me, *current_bidder_idx, active_bid, bought, passed),

        Phase::DiscardPlant {
            player, new_plant, ..
        } => {
            if *player != me {
                return None;
            }
            decide_discard(state, me, new_plant)
        }

        Phase::DiscardResource {
            player, drop_total, ..
        } => {
            if *player != me {
                return None;
            }
            decide_discard_resource(state, me, *drop_total)
        }

        Phase::BuyResources { remaining } => {
            if remaining.first() != Some(&me) {
                return None;
            }
            decide_buy_resources(state, me)
        }

        Phase::BuildCities { remaining } => {
            if remaining.first() != Some(&me) {
                return None;
            }
            decide_build_cities(state, me)
        }

        Phase::Bureaucracy { remaining } => {
            if !remaining.contains(&me) {
                return None;
            }
            decide_power_cities(state, me)
        }

        Phase::PowerCitiesFuel {
            player,
            hybrid_cost,
            ..
        } => {
            if *player != me {
                return None;
            }
            decide_power_cities_fuel(state, me, *hybrid_cost)
        }
    }
}

// ---------------------------------------------------------------------------
// Plant scoring
// ---------------------------------------------------------------------------

/// Score a plant for acquisition. Higher = more desirable.
fn plant_score(plant: &PowerPlant) -> i32 {
    let city_value = plant.cities as i32 * 15;
    let fuel_bonus = if matches!(plant.kind, PlantKind::Wind | PlantKind::Fusion) {
        25 // no running cost ever
    } else {
        0
    };
    // Efficiency: cities per resource consumed (wind/fusion cost=0, treat as very efficient)
    let efficiency = if plant.cost == 0 {
        30
    } else {
        (plant.cities as i32 * 10) / plant.cost as i32
    };
    city_value + fuel_bonus + efficiency
}

/// How much a plant is intrinsically worth, based on its listed price + city capacity.
fn plant_value_ceiling(plant: &PowerPlant) -> u32 {
    let base = plant.number as u32;
    let city_premium = plant.cities as u32 * 4;
    let no_fuel_bonus = if matches!(plant.kind, PlantKind::Wind | PlantKind::Fusion) {
        5
    } else {
        0
    };
    base + city_premium + no_fuel_bonus
}

/// How much cash to keep after winning an auction: fuel for all owned plants
/// (including the new one) plus enough for at least one city build.
fn auction_reserve(plant: &PowerPlant, player: &Player) -> u32 {
    let mut reserve = 0u32;
    for p in &player.plants {
        if p.kind.needs_resources() {
            reserve += p.cost as u32 * 4;
        }
    }
    if plant.kind.needs_resources() {
        reserve += plant.cost as u32 * 4;
    }
    reserve += 15; // at least one city build
    reserve += 5; // safety buffer
    reserve
}

/// True when acquiring a new plant would give little or no benefit.
///
/// Two cases: (1) we already have excess generation capacity relative to the
/// cities we own, or (2) we have a full rack (3 plants) and the candidate is
/// not a meaningful upgrade over our worst plant.
fn should_skip_auction(player: &Player, candidate: &PowerPlant) -> bool {
    let powerable: u8 = player.plants.iter().map(|p| p.cities).sum();
    let owned = player.cities.len() as u8;
    if powerable > owned.saturating_add(2) {
        return true;
    }
    if player.plants.len() >= 3 {
        if let Some(worst) = player.plants.iter().min_by_key(|p| plant_score(p)) {
            if plant_score(candidate) - plant_score(worst) < 10 {
                return true;
            }
        }
    }
    false
}

/// Max we are willing to pay for a plant (bid ceiling).
fn max_bid(plant: &PowerPlant, player: &Player) -> u32 {
    let value = plant_value_ceiling(plant);
    let affordable = player.money.saturating_sub(auction_reserve(plant, player));
    value.min(affordable).max(plant.number as u32)
}

// ---------------------------------------------------------------------------
// Auction phase
// ---------------------------------------------------------------------------

fn decide_auction(
    state: &GameState,
    me: PlayerId,
    current_bidder_idx: usize,
    active_bid: &Option<powergrid_core::types::ActiveBid>,
    bought: &[PlayerId],
    passed: &[PlayerId],
) -> Option<Action> {
    let my_player = state.player(me)?;

    if let Some(bid) = active_bid {
        // An active bid is in progress — act only if we are the next bidder.
        if bid.remaining_bidders.first() != Some(&me) {
            return None;
        }

        // Find the plant being auctioned.
        let plant = state
            .market
            .actual
            .iter()
            .find(|p| p.number == bid.plant_number)?;

        let ceiling = max_bid(plant, my_player);
        if bid.amount < ceiling {
            let raise = bid.amount + 1;
            info!(
                "Raising bid on plant {} to {} (ceiling {})",
                bid.plant_number, raise, ceiling
            );
            return Some(Action::PlaceBid { amount: raise });
        } else {
            info!(
                "Passing on plant {} — bid {} exceeds ceiling {}",
                bid.plant_number, bid.amount, ceiling
            );
            return Some(Action::PassAuction);
        }
    }

    // No active bid — it's our turn to select a plant (or pass).
    if state
        .player_order
        .get(current_bidder_idx)
        .copied()
        .unwrap_or_default()
        != me
    {
        return None;
    }

    // Already bought or passed this round?
    if bought.contains(&me) || passed.contains(&me) {
        return None;
    }

    let is_round_one = state.round == 1;

    // Score each plant in the actual market and pick the best affordable one.
    let best = state
        .market
        .actual
        .iter()
        .filter(|p| my_player.money >= p.number as u32)
        .max_by_key(|p| plant_score(p));

    match best {
        Some(plant) => {
            let score = plant_score(plant);
            // In round 1 we must buy; otherwise only buy if the plant is worth it
            // and we actually need more generation capacity.
            if is_round_one || (score >= 20 && !should_skip_auction(my_player, plant)) {
                info!(
                    "Selecting plant {} (kind={:?}, cities={}, score={})",
                    plant.number, plant.kind, plant.cities, score
                );
                Some(Action::SelectPlant {
                    plant_number: plant.number,
                })
            } else {
                info!(
                    "Passing auction — no plant worth buying (best score {})",
                    score
                );
                Some(Action::PassAuction)
            }
        }
        None => {
            if is_round_one {
                // Must buy in round 1 but can't afford anything — pick cheapest regardless.
                let cheapest = state.market.actual.iter().min_by_key(|p| p.number)?;
                info!(
                    "Round 1 forced buy — selecting cheapest plant {}",
                    cheapest.number
                );
                Some(Action::SelectPlant {
                    plant_number: cheapest.number,
                })
            } else {
                info!("Passing auction — cannot afford any plant");
                Some(Action::PassAuction)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Discard phase
// ---------------------------------------------------------------------------

fn decide_discard(state: &GameState, me: PlayerId, new_plant: &PowerPlant) -> Option<Action> {
    let player = state.player(me)?;

    // Discard the existing plant with the lowest score that is not the new plant.
    let worst = player
        .plants
        .iter()
        .filter(|p| p.number != new_plant.number)
        .min_by_key(|p| plant_score(p))?;

    info!(
        "Discarding plant {} (score {}) to make room for plant {} (score {})",
        worst.number,
        plant_score(worst),
        new_plant.number,
        plant_score(new_plant)
    );
    Some(Action::DiscardPlant {
        plant_number: worst.number,
    })
}

// ---------------------------------------------------------------------------
// Resource-discard phase
// ---------------------------------------------------------------------------

fn decide_discard_resource(state: &GameState, me: PlayerId, drop_total: u8) -> Option<Action> {
    let player = state.player(me)?;
    // Drop coal first to preserve oil for hybrid plants (mirrors resources_needed heuristic).
    let coal = drop_total.min(player.resources.coal);
    let oil = drop_total - coal;
    info!(
        "DiscardResource: dropping {} coal and {} oil (drop_total={})",
        coal, oil, drop_total
    );
    Some(Action::DiscardResource { coal, oil })
}

// ---------------------------------------------------------------------------
// Buy resources phase
// ---------------------------------------------------------------------------

/// Reserve enough for ~1-2 city builds so the bot can expand after buying fuel.
fn city_build_reserve(state: &GameState, me: PlayerId) -> u32 {
    let Some(player) = state.player(me) else {
        return 15;
    };
    let powerable = player.plants.iter().map(|p| p.cities).sum::<u8>() as usize;
    let owned = player.cities.len();
    let want = powerable.saturating_sub(owned).min(2) as u32;
    (want * 15).max(15)
}

fn decide_buy_resources(state: &GameState, me: PlayerId) -> Option<Action> {
    let player = state.player(me)?;

    let mut purchases: Vec<(Resource, u8)> = Vec::new();
    let mut sim_market = state.resources.clone();
    let mut sim_player = player.clone();
    let mut budget = player.money;

    // Most cities first; break ties by plant number (smaller = cheaper to fuel).
    let mut plants = player.plants.clone();
    plants.sort_by(|a, b| b.cities.cmp(&a.cities).then(a.number.cmp(&b.number)));

    // Pass 1 — essential: ensure each plant can fire at least once this round.
    // Use the full money budget so the city-build reserve does not starve fuel.
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

    // Pass 2 — subtract city-build reserve from whatever remains after essential fuel.
    let city_reserve = city_build_reserve(state, me);
    budget = budget.saturating_sub(city_reserve);

    // Pass 3 — top-up: fill remaining storage capacity with leftover cash.
    for plant in &plants {
        buy_for_plant(
            plant,
            plant.cost * 2,
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

/// Bring `plant`'s fuel level up to `target` by purchasing from the simulated
/// market.  Hybrid plants try oil first, then fall back to coal when oil is
/// unavailable or won't fit in storage.  Commits all changes to `sim_market`,
/// `sim_player`, `budget`, and `purchases`.
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
        PlantKind::CoalOrOil => {
            // The plant can fire on any mix of coal and oil; compare combined total.
            let combined = player.resources.coal.saturating_add(player.resources.oil);
            let want = target.saturating_sub(combined);
            if want == 0 {
                return;
            }
            // Oil first to conserve coal for pure-coal plants.
            try_buy(Resource::Oil, want, market, player, budget, purchases);
            // Re-check shortfall after oil purchase — oil may have been exhausted.
            let combined = player.resources.coal.saturating_add(player.resources.oil);
            let remaining = target.saturating_sub(combined);
            if remaining > 0 {
                try_buy(Resource::Coal, remaining, market, player, budget, purchases);
            }
        }
        PlantKind::Garbage => {
            let want = target.saturating_sub(player.resources.garbage);
            if want > 0 {
                try_buy(Resource::Garbage, want, market, player, budget, purchases);
            }
        }
        PlantKind::Uranium => {
            let want = target.saturating_sub(player.resources.uranium);
            if want > 0 {
                try_buy(Resource::Uranium, want, market, player, budget, purchases);
            }
        }
        PlantKind::Wind | PlantKind::Fusion => {}
    }
}

/// Attempt to purchase up to `want` units of `resource`, degrading gracefully
/// to smaller amounts when the full quantity is too expensive or would exceed
/// storage capacity.  Commits a successful purchase to all mutable state.
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
    // Find the largest affordable amount that also fits in storage.
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

fn decide_build_cities(state: &GameState, me: PlayerId) -> Option<Action> {
    let player = state.player(me)?;

    let mut budget = player.money;

    // Enumerate all cities in active regions that we could build in.
    let mut candidates: Vec<(String, u32)> = state
        .map
        .cities
        .values()
        .filter(|city| {
            state.active_regions.contains(&city.region)
                && !player.cities.contains(&city.id)
                && city.owners.len() < state.step as usize
        })
        .filter_map(|city| {
            let route_cost = state.map.connection_cost_to(&player.cities, &city.id)?;
            let slot_cost = connection_cost(city.owners.len());
            Some((city.id.clone(), route_cost + slot_cost))
        })
        .collect();

    // Sort cheapest first.
    candidates.sort_by_key(|(_, cost)| *cost);

    // Greedily build cheapest affordable cities. Budget is the only constraint —
    // building ahead of current powering capacity is correct since income tracks
    // cities owned and claiming spots early denies them to opponents.
    let mut city_ids: Vec<String> = Vec::new();
    let mut simulated_cities: Vec<String> = player.cities.clone();

    for (city_id, _) in &candidates {
        // Recompute cost using the growing simulated network (batch routing).
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

fn decide_power_cities(state: &GameState, me: PlayerId) -> Option<Action> {
    let player = state.player(me)?;

    // Fire all plants that have enough fuel (the server picks the optimal subset).
    let plant_numbers: Vec<u8> = player
        .plants
        .iter()
        .filter(|p| can_fire(p, &player.resources))
        .map(|p| p.number)
        .collect();

    let cities_powered = player.cities_powerable();
    let expected_income = income_for(cities_powered.min(player.city_count() as u8));

    info!(
        "PowerCities with plants {:?} — expect to power {} cities, earn {} elektro",
        plant_numbers, cities_powered, expected_income
    );

    Some(Action::PowerCities { plant_numbers })
}

fn decide_power_cities_fuel(state: &GameState, me: PlayerId, hybrid_cost: u8) -> Option<Action> {
    use powergrid_core::types::Phase;
    let player = state.player(me)?;

    if let Phase::PowerCitiesFuel { plant_numbers, .. } = &state.phase {
        // Compute pure-fuel obligations so we know what's left for hybrids.
        let pure_coal: u8 = plant_numbers
            .iter()
            .filter_map(|&num| player.plants.iter().find(|p| p.number == num))
            .filter(|p| p.kind == PlantKind::Coal)
            .map(|p| p.cost)
            .sum();
        let pure_oil: u8 = plant_numbers
            .iter()
            .filter_map(|&num| player.plants.iter().find(|p| p.number == num))
            .filter(|p| p.kind == PlantKind::Oil)
            .map(|p| p.cost)
            .sum();

        let _coal_avail = player.resources.coal.saturating_sub(pure_coal);
        let oil_avail = player.resources.oil.saturating_sub(pure_oil);

        // Prefer oil for hybrids (conserves coal for future pure-Coal plants).
        let oil = hybrid_cost.min(oil_avail);
        let coal = hybrid_cost - oil;

        info!(
            "PowerCitiesFuel: using {} coal + {} oil for hybrids (hybrid_cost={})",
            coal, oil, hybrid_cost
        );
        Some(Action::PowerCitiesFuel { coal, oil })
    } else {
        None
    }
}

fn can_fire(plant: &PowerPlant, resources: &powergrid_core::types::PlayerResources) -> bool {
    match plant.kind {
        PlantKind::Wind | PlantKind::Fusion => true,
        PlantKind::Coal => resources.coal >= plant.cost,
        PlantKind::Oil => resources.oil >= plant.cost,
        PlantKind::CoalOrOil => resources.coal + resources.oil >= plant.cost,
        PlantKind::Garbage => resources.garbage >= plant.cost,
        PlantKind::Uranium => resources.uranium >= plant.cost,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use powergrid_core::types::{Player, PlayerColor, PowerPlant};

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
            kind: PlantKind::CoalOrOil,
            cost,
            cities,
        }
    }

    fn bot_with_money(money: u32) -> Player {
        let mut p = Player::new("bot".into(), PlayerColor::Red);
        p.money = money;
        p
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
    fn falls_back_to_coal_for_hybrid_when_oil_empty() {
        let plant = hybrid_plant(10, 3, 2);
        let mut player = bot_with_money(50);
        player.plants.push(plant.clone());

        // Drain oil market completely.
        let mut market = ResourceMarket::initial();
        market.oil = 0;

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
            "expected >= {} coal as fallback, got {}",
            plant.cost,
            coal_bought
        );
    }

    #[test]
    fn degrades_gracefully_when_full_topup_unaffordable() {
        // Plant cost=4, so full top-up = 8 coal. Player only has $5 — can still
        // afford 1 coal at the market's cheapest slot (1 elektro each).
        let plant = coal_plant(15, 4, 3);
        let mut player = bot_with_money(5);
        player.plants.push(plant.clone());

        let mut market = ResourceMarket::initial(); // 24 coal at 1 elektro each
        let mut purchases = vec![];
        let mut budget = player.money;

        // Top-up target: cost * 2 = 8; should buy as much as $5 allows.
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
    fn essential_pass_ignores_city_reserve() {
        // Bot has $22. City reserve would be 15, leaving only $7 with old logic.
        // With new logic the essential pass gets the full $22, so it can buy 2
        // coal at 1 elektro each (cost 2) easily.
        let plant = coal_plant(5, 2, 1);
        let mut player = bot_with_money(22);
        player.plants.push(plant.clone());

        let mut market = ResourceMarket::initial();
        let mut purchases = vec![];
        // Simulate pass-1 budget = full money (no reserve yet).
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
}
