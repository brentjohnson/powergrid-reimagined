use powergrid_core::{
    actions::Action,
    state::GameState,
    types::{connection_cost, income_for, PlantKind, Player, PlayerId, PowerPlant, Resource},
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
    let market = &state.resources;

    // Build a purchase list: for each plant, how much fuel do we still need?
    let mut purchases: Vec<(Resource, u8)> = Vec::new();
    let mut simulated_market = market.clone();
    let city_reserve = city_build_reserve(state, me);
    let mut budget = player.money.saturating_sub(city_reserve);

    // Process plants ordered by most cities powered first (fill the best plants first).
    let mut plants_sorted = player.plants.clone();
    plants_sorted.sort_by(|a, b| b.cities.cmp(&a.cities));

    for plant in &plants_sorted {
        if !plant.kind.needs_resources() {
            continue;
        }

        let resources_for_plant = resources_needed(plant, &player.resources);

        for (resource, needed) in resources_for_plant {
            if needed == 0 {
                continue;
            }

            // Don't exceed capacity.
            let can_store = if player.can_add_resource(resource, needed) {
                needed
            } else {
                // Find how much we can actually fit.
                (1..=needed)
                    .rev()
                    .find(|&n| player.can_add_resource(resource, n))
                    .unwrap_or(0)
            };
            if can_store == 0 {
                continue;
            }

            let available = simulated_market.available(resource);
            let amount = can_store.min(available);
            if amount == 0 {
                continue;
            }

            if let Some(cost) = simulated_market.price(resource, amount) {
                if cost <= budget {
                    debug!("Buying {} {:?} for {} elektro", amount, resource, cost);
                    purchases.push((resource, amount));
                    simulated_market.take(resource, amount);
                    budget -= cost;
                } else {
                    debug!(
                        "Cannot afford {} {:?} (costs {}, have {})",
                        amount, resource, cost, budget
                    );
                }
            }
        }
    }

    // Log what we're buying.
    if purchases.is_empty() {
        info!("Buy resources: nothing to buy, done");
    } else {
        let total = market.batch_price(&purchases).unwrap_or(0);
        info!(
            "Buy resources: {:?} for {} total elektro (have {})",
            purchases, total, player.money
        );
    }

    Some(Action::BuyResourceBatch { purchases })
}

/// How many additional resources does this plant need to be fully fuelled?
fn resources_needed(
    plant: &PowerPlant,
    stored: &powergrid_core::types::PlayerResources,
) -> Vec<(Resource, u8)> {
    match plant.kind {
        PlantKind::Coal => {
            let have = stored.coal;
            let cap = plant.cost * 2;
            vec![(Resource::Coal, cap.saturating_sub(have))]
        }
        PlantKind::Oil => {
            let have = stored.oil;
            let cap = plant.cost * 2;
            vec![(Resource::Oil, cap.saturating_sub(have))]
        }
        PlantKind::CoalOrOil => {
            // For hybrids, prefer oil when buying (conserves coal for pure-coal plants).
            // Buy oil up to cap, then fall back to coal.
            let cap = plant.cost * 2;
            let have_oil = stored.oil;
            let oil_needed = cap.saturating_sub(have_oil);
            if oil_needed > 0 {
                vec![(Resource::Oil, oil_needed)]
            } else {
                vec![]
            }
        }
        PlantKind::Garbage => {
            let have = stored.garbage;
            let cap = plant.cost * 2;
            vec![(Resource::Garbage, cap.saturating_sub(have))]
        }
        PlantKind::Uranium => {
            let have = stored.uranium;
            let cap = plant.cost * 2;
            vec![(Resource::Uranium, cap.saturating_sub(have))]
        }
        PlantKind::Wind | PlantKind::Fusion => vec![],
    }
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
