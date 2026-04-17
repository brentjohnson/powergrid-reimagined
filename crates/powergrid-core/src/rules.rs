use crate::actions::{Action, ActionError};
use crate::state::GameState;
use crate::types::*;
use rand::seq::SliceRandom;
use rand::SeedableRng;

/// Entry point: validate and apply an action from `actor`.
/// Returns the mutated state on success, or an error (state unchanged).
pub fn apply_action(
    state: &mut GameState,
    actor: PlayerId,
    action: Action,
) -> Result<(), ActionError> {
    match action {
        Action::JoinGame { name, color } => handle_join(state, actor, name, color),
        Action::StartGame => handle_start(state, actor),
        Action::SelectPlant { plant_number } => handle_select_plant(state, actor, plant_number),
        Action::PlaceBid { amount } => handle_place_bid(state, actor, amount),
        Action::PassAuction => handle_pass_auction(state, actor),
        Action::BuyResources { resource, amount } => {
            handle_buy_resources(state, actor, resource, amount)
        }
        Action::DoneBuying => handle_done_buying(state, actor),
        Action::BuildCity { city_id } => handle_build_city(state, actor, city_id),
        Action::BuildCities { city_ids } => handle_build_cities(state, actor, city_ids),
        Action::DoneBuilding => handle_done_building(state, actor),
        Action::PowerCities { plant_numbers } => handle_power_cities(state, actor, plant_numbers),
    }
}

// ---------------------------------------------------------------------------
// Lobby
// ---------------------------------------------------------------------------

fn handle_join(
    state: &mut GameState,
    actor: PlayerId,
    name: String,
    color: PlayerColor,
) -> Result<(), ActionError> {
    if !matches!(state.phase, Phase::Lobby) {
        return Err(ActionError::WrongPhase);
    }
    if state.players.len() >= 6 {
        return Err(ActionError::GameFull);
    }
    if state.players.iter().any(|p| p.name == name) {
        return Err(ActionError::NameTaken);
    }
    if state.players.iter().any(|p| p.color == color) {
        return Err(ActionError::ColorTaken);
    }

    let mut player = Player::new(name.clone(), color);
    player.id = actor; // use the id assigned by the server
    state.players.push(player);
    state.log(format!("{} joined the game", name));
    Ok(())
}

fn handle_start(state: &mut GameState, actor: PlayerId) -> Result<(), ActionError> {
    if !matches!(state.phase, Phase::Lobby) {
        return Err(ActionError::WrongPhase);
    }
    if state.host_id() != Some(actor) {
        return Err(ActionError::NotHost);
    }
    if state.players.len() < 2 {
        return Err(ActionError::NotEnoughPlayers);
    }

    state.round = 1;

    let mut rng = match state.rng_seed {
        Some(seed) => rand::rngs::SmallRng::seed_from_u64(seed),
        None => rand::rngs::SmallRng::from_entropy(),
    };

    state.player_order = state.players.iter().map(|p| p.id).collect();
    state.player_order.shuffle(&mut rng);

    let player_count = state.players.len();
    state.market.setup_deck(&mut rng, player_count);

    begin_auction(state);
    state.log("Game started!".to_string());
    Ok(())
}

// ---------------------------------------------------------------------------
// Auction phase helpers
// ---------------------------------------------------------------------------

fn begin_auction(state: &mut GameState) {
    let bought = Vec::new();
    let passed = Vec::new();
    state.phase = Phase::Auction {
        current_bidder_idx: 0,
        active_bid: None,
        bought,
        passed,
    };
}

fn handle_select_plant(
    state: &mut GameState,
    actor: PlayerId,
    plant_number: u8,
) -> Result<(), ActionError> {
    let (current_bidder_idx, active_bid, bought, passed) = match &state.phase {
        Phase::Auction {
            current_bidder_idx,
            active_bid,
            bought,
            passed,
        } => (
            *current_bidder_idx,
            active_bid.clone(),
            bought.clone(),
            passed.clone(),
        ),
        _ => return Err(ActionError::WrongPhase),
    };

    if active_bid.is_some() {
        return Err(ActionError::WrongPhase); // bidding in progress
    }

    let acting_player = state.player_order[current_bidder_idx];
    if actor != acting_player {
        return Err(ActionError::NotYourTurn);
    }

    // Verify plant is in actual market.
    if !state.market.actual.iter().any(|p| p.number == plant_number) {
        return Err(ActionError::PlantNotInMarket(plant_number));
    }

    let player_money = state.player(actor).ok_or(ActionError::UnknownPlayer)?.money;
    if (plant_number as u32) > player_money {
        return Err(ActionError::CannotAfford);
    }

    // Start bid at the plant's number (minimum bid).
    // The selector has implicitly bid the minimum by selecting; exclude them so
    // other players respond first. They re-enter the rotation if outbid.
    let remaining_bidders: Vec<PlayerId> = state
        .player_order
        .iter()
        .filter(|&&id| !bought.contains(&id) && !passed.contains(&id) && id != actor)
        .cloned()
        .collect();

    // If no other players remain to bid, the selector wins at minimum bid immediately.
    if remaining_bidders.is_empty() {
        return award_plant(
            state,
            actor,
            plant_number,
            plant_number as u32,
            bought,
            passed,
        );
    }

    state.phase = Phase::Auction {
        current_bidder_idx,
        active_bid: Some(ActiveBid {
            plant_number,
            highest_bidder: actor,
            amount: plant_number as u32,
            remaining_bidders,
        }),
        bought,
        passed,
    };
    Ok(())
}

fn handle_place_bid(
    state: &mut GameState,
    actor: PlayerId,
    amount: u32,
) -> Result<(), ActionError> {
    let (current_bidder_idx, active_bid, bought, passed) = match &state.phase {
        Phase::Auction {
            current_bidder_idx,
            active_bid,
            bought,
            passed,
        } => (
            *current_bidder_idx,
            active_bid.clone(),
            bought.clone(),
            passed.clone(),
        ),
        _ => return Err(ActionError::WrongPhase),
    };

    let mut bid = active_bid.ok_or(ActionError::WrongPhase)?;

    // It must be this player's turn to bid.
    let next_bidder = bid
        .remaining_bidders
        .first()
        .copied()
        .ok_or(ActionError::WrongPhase)?;
    if actor != next_bidder {
        return Err(ActionError::NotYourTurn);
    }

    if amount <= bid.amount {
        return Err(ActionError::BidTooLow(amount, bid.amount + 1));
    }

    let player_money = state.player(actor).ok_or(ActionError::UnknownPlayer)?.money;
    if amount > player_money {
        return Err(ActionError::CannotAfford);
    }

    let old_highest = bid.highest_bidder;
    bid.highest_bidder = actor;
    bid.amount = amount;
    bid.remaining_bidders.remove(0);
    // Move this player to the end — they bid again only if others raise.
    bid.remaining_bidders.push(actor);
    // Give the previous highest bidder a chance to counter-bid,
    // but only after all other players who haven't bid yet get their turn.
    if old_highest != actor && !bid.remaining_bidders.contains(&old_highest) {
        let insert_pos = bid.remaining_bidders.len() - 1;
        bid.remaining_bidders.insert(insert_pos, old_highest);
    }

    state.phase = Phase::Auction {
        current_bidder_idx,
        active_bid: Some(bid),
        bought,
        passed,
    };
    Ok(())
}

fn handle_pass_auction(state: &mut GameState, actor: PlayerId) -> Result<(), ActionError> {
    let (current_bidder_idx, active_bid, bought, mut passed) = match &state.phase {
        Phase::Auction {
            current_bidder_idx,
            active_bid,
            bought,
            passed,
        } => (
            *current_bidder_idx,
            active_bid.clone(),
            bought.clone(),
            passed.clone(),
        ),
        _ => return Err(ActionError::WrongPhase),
    };

    if let Some(mut bid) = active_bid {
        // Passing on an active bid — remove this player from the rotation.
        let next_bidder = bid
            .remaining_bidders
            .first()
            .copied()
            .ok_or(ActionError::WrongPhase)?;
        if actor != next_bidder {
            return Err(ActionError::NotYourTurn);
        }

        bid.remaining_bidders.remove(0);

        if bid.remaining_bidders.is_empty() || bid.remaining_bidders == vec![bid.highest_bidder] {
            // Auction resolved — winner buys the plant.
            award_plant(
                state,
                bid.highest_bidder,
                bid.plant_number,
                bid.amount,
                bought,
                passed,
            )?;
        } else {
            state.phase = Phase::Auction {
                current_bidder_idx,
                active_bid: Some(bid),
                bought,
                passed,
            };
        }
    } else {
        // No active bid — player passes their turn to select a plant.
        let acting_player = state.player_order[current_bidder_idx];
        if actor != acting_player {
            return Err(ActionError::NotYourTurn);
        }

        passed.push(actor);
        advance_auction(state, current_bidder_idx, bought, passed);
    }
    Ok(())
}

fn award_plant(
    state: &mut GameState,
    winner: PlayerId,
    plant_number: u8,
    cost: u32,
    mut bought: Vec<PlayerId>,
    passed: Vec<PlayerId>,
) -> Result<(), ActionError> {
    let plant = state
        .market
        .take_from_actual(plant_number)
        .ok_or(ActionError::PlantNotInMarket(plant_number))?;

    let player = state.player_mut(winner).ok_or(ActionError::UnknownPlayer)?;
    player.money -= cost;

    // Discard lowest plant if player already has 3.
    if player.plants.len() >= 3 {
        player.plants.sort_by_key(|p| p.number);
        player.plants.remove(0);
    }
    player.plants.push(plant.clone());
    player.plants.sort_by_key(|p| p.number);

    state.log(format!(
        "{} bought plant {} for {}",
        state.player(winner).map(|p| p.name.as_str()).unwrap_or("?"),
        plant_number,
        cost
    ));

    bought.push(winner);

    // Find current_bidder_idx for the winner.
    let current_bidder_idx = state
        .player_order
        .iter()
        .position(|&id| id == winner)
        .unwrap_or(0);
    advance_auction(state, current_bidder_idx, bought, passed);
    Ok(())
}

fn advance_auction(
    state: &mut GameState,
    current_bidder_idx: usize,
    bought: Vec<PlayerId>,
    passed: Vec<PlayerId>,
) {
    let total = state.player_order.len();
    let all_done: Vec<PlayerId> = bought.iter().chain(passed.iter()).cloned().collect();

    // Check if everyone has acted.
    if all_done.len() >= total {
        // End of auction — remove lowest plant, transition to buy resources.
        state.market.remove_lowest();
        begin_buy_resources(state);
        return;
    }

    // Find next player who hasn't bought or passed.
    let mut next_idx = (current_bidder_idx + 1) % total;
    let mut iterations = 0;
    while all_done.contains(&state.player_order[next_idx]) {
        next_idx = (next_idx + 1) % total;
        iterations += 1;
        if iterations > total {
            state.market.remove_lowest();
            begin_buy_resources(state);
            return;
        }
    }

    state.phase = Phase::Auction {
        current_bidder_idx: next_idx,
        active_bid: None,
        bought,
        passed,
    };
}

// ---------------------------------------------------------------------------
// Buy resources
// ---------------------------------------------------------------------------

fn begin_buy_resources(state: &mut GameState) {
    // After the first auction, recalculate order based on plants purchased
    // (no cities exist yet, so it sorts by highest plant number).
    if state.round == 1 {
        recalculate_player_order(state);
    }
    // Reverse player order.
    let remaining: Vec<PlayerId> = state.player_order.iter().rev().cloned().collect();
    state.phase = Phase::BuyResources { remaining };
}

fn handle_buy_resources(
    state: &mut GameState,
    actor: PlayerId,
    resource: Resource,
    amount: u8,
) -> Result<(), ActionError> {
    let remaining = match &state.phase {
        Phase::BuyResources { remaining } => remaining.clone(),
        _ => return Err(ActionError::WrongPhase),
    };

    if remaining.first().copied() != Some(actor) {
        return Err(ActionError::NotYourTurn);
    }

    let cost = state
        .resources
        .price(resource, amount)
        .ok_or(ActionError::ResourceUnavailable)?;

    let player = state.player(actor).ok_or(ActionError::UnknownPlayer)?;
    if player.money < cost {
        return Err(ActionError::CannotAfford);
    }

    if !player.can_add_resource(resource, amount) {
        return Err(ActionError::OverCapacity);
    }

    state.resources.take(resource, amount);
    let player = state.player_mut(actor).ok_or(ActionError::UnknownPlayer)?;
    player.money -= cost;
    player.resources.add(resource, amount);

    // Don't advance — player may buy more resources.
    Ok(())
}

fn handle_done_buying(state: &mut GameState, actor: PlayerId) -> Result<(), ActionError> {
    let mut remaining = match &state.phase {
        Phase::BuyResources { remaining } => remaining.clone(),
        _ => return Err(ActionError::WrongPhase),
    };

    if remaining.first().copied() != Some(actor) {
        return Err(ActionError::NotYourTurn);
    }

    remaining.remove(0);
    if remaining.is_empty() {
        begin_build_cities(state);
    } else {
        state.phase = Phase::BuyResources { remaining };
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Build cities
// ---------------------------------------------------------------------------

fn begin_build_cities(state: &mut GameState) {
    let remaining: Vec<PlayerId> = state.player_order.iter().rev().cloned().collect();
    state.phase = Phase::BuildCities { remaining };
}

/// Validates and applies a single city build for `actor`. Does NOT advance the turn.
/// Assumes phase and turn ownership have already been checked.
fn apply_single_build(
    state: &mut GameState,
    actor: PlayerId,
    city_id: &str,
) -> Result<(), ActionError> {
    let city = state
        .map
        .cities
        .get(city_id)
        .ok_or_else(|| ActionError::CityNotFound(city_id.to_string()))?;

    if city.owners.len() >= 3 {
        return Err(ActionError::CityFull(city_id.to_string()));
    }
    if city.owners.contains(&actor) {
        return Err(ActionError::AlreadyBuiltThere);
    }

    let player = state.player(actor).ok_or(ActionError::UnknownPlayer)?;
    let owned_cities = player.cities.clone();
    let route_cost = state
        .map
        .connection_cost_to(&owned_cities, city_id)
        .unwrap_or(0);
    let city_slot_cost = connection_cost(state.map.cities[city_id].owners.len());
    let total_cost = route_cost + city_slot_cost;

    if player.money < total_cost {
        return Err(ActionError::CannotAffordCity);
    }

    let player = state.player_mut(actor).ok_or(ActionError::UnknownPlayer)?;
    player.money -= total_cost;
    player.cities.push(city_id.to_string());

    state
        .map
        .cities
        .get_mut(city_id)
        .unwrap()
        .owners
        .push(actor);
    state.log(format!(
        "{} built in {}",
        state.player(actor).map(|p| p.name.as_str()).unwrap_or("?"),
        city_id
    ));

    Ok(())
}

/// Removes the acting player from the build queue and transitions phase.
fn advance_build_phase(state: &mut GameState, mut remaining: Vec<PlayerId>) {
    remaining.remove(0);
    if remaining.is_empty() {
        begin_bureaucracy(state);
    } else {
        state.phase = Phase::BuildCities { remaining };
    }
}

fn check_end_game_trigger(state: &mut GameState) {
    let max_cities = state
        .players
        .iter()
        .map(|p| p.cities.len())
        .max()
        .unwrap_or(0);
    if max_cities >= state.end_game_cities as usize {
        state.log("End-game triggered! Finish the round.".to_string());
    }
}

fn handle_build_city(
    state: &mut GameState,
    actor: PlayerId,
    city_id: String,
) -> Result<(), ActionError> {
    let remaining = match &state.phase {
        Phase::BuildCities { remaining } => remaining.clone(),
        _ => return Err(ActionError::WrongPhase),
    };

    if remaining.first().copied() != Some(actor) {
        return Err(ActionError::NotYourTurn);
    }

    apply_single_build(state, actor, &city_id)?;
    check_end_game_trigger(state);

    Ok(())
}

fn handle_build_cities(
    state: &mut GameState,
    actor: PlayerId,
    city_ids: Vec<String>,
) -> Result<(), ActionError> {
    let remaining = match &state.phase {
        Phase::BuildCities { remaining } => remaining.clone(),
        _ => return Err(ActionError::WrongPhase),
    };

    if remaining.first().copied() != Some(actor) {
        return Err(ActionError::NotYourTurn);
    }

    if city_ids.is_empty() {
        return Err(ActionError::EmptyBuildList);
    }

    // Reject duplicates.
    let mut seen = std::collections::HashSet::new();
    for id in &city_ids {
        if !seen.insert(id.as_str()) {
            return Err(ActionError::DuplicateCityInBuild);
        }
    }

    // Apply all builds on a scratch copy for atomicity.
    let mut scratch = state.clone();
    for city_id in &city_ids {
        apply_single_build(&mut scratch, actor, city_id)?;
    }

    // All succeeded — commit and advance the phase.
    *state = scratch;
    check_end_game_trigger(state);
    advance_build_phase(state, remaining);

    Ok(())
}

fn handle_done_building(state: &mut GameState, actor: PlayerId) -> Result<(), ActionError> {
    let remaining = match &state.phase {
        Phase::BuildCities { remaining } => remaining.clone(),
        _ => return Err(ActionError::WrongPhase),
    };

    if remaining.first().copied() != Some(actor) {
        return Err(ActionError::NotYourTurn);
    }

    advance_build_phase(state, remaining);
    Ok(())
}

// ---------------------------------------------------------------------------
// Bureaucracy
// ---------------------------------------------------------------------------

fn begin_bureaucracy(state: &mut GameState) {
    let remaining: Vec<PlayerId> = state.player_order.clone();
    state.phase = Phase::Bureaucracy { remaining };
}

fn handle_power_cities(
    state: &mut GameState,
    actor: PlayerId,
    plant_numbers: Vec<u8>,
) -> Result<(), ActionError> {
    let remaining = match &state.phase {
        Phase::Bureaucracy { remaining } => remaining.clone(),
        _ => return Err(ActionError::WrongPhase),
    };

    if remaining.first().copied() != Some(actor) {
        return Err(ActionError::NotYourTurn);
    }

    // Validate that player owns all specified plants.
    {
        let player = state.player(actor).ok_or(ActionError::UnknownPlayer)?;
        for &num in &plant_numbers {
            if !player.plants.iter().any(|p| p.number == num) {
                return Err(ActionError::PlantNotOwned(num));
            }
        }
    }

    // Calculate cities powered and consume resources.
    let powered = {
        let player = state.player(actor).ok_or(ActionError::UnknownPlayer)?;
        let cities_owned = player.city_count() as u8;
        let mut powered = 0u8;
        let mut res = player.resources.clone();

        for &num in &plant_numbers {
            let plant = player.plants.iter().find(|p| p.number == num).unwrap();
            if plant.kind.needs_resources() {
                let r = plant.kind.resources()[0]; // simplified: use first resource type
                if !res.remove(r, plant.cost) {
                    // Try second resource for hybrid plants.
                    if plant.kind == PlantKind::CoalOrOil {
                        let r2 = Resource::Oil;
                        if !res.remove(r2, plant.cost) {
                            continue; // can't fire
                        }
                    } else {
                        continue; // can't fire
                    }
                }
            }
            powered += plant.cities;
        }
        powered.min(cities_owned)
    };

    // Apply resource consumption (simplified — consume in declared order).
    {
        let player = state.player_mut(actor).ok_or(ActionError::UnknownPlayer)?;
        for &num in &plant_numbers {
            let plant = player
                .plants
                .iter()
                .find(|p| p.number == num)
                .unwrap()
                .clone();
            if plant.kind.needs_resources() {
                let r = plant.kind.resources()[0];
                if !player.resources.remove(r, plant.cost) && plant.kind == PlantKind::CoalOrOil {
                    player.resources.remove(Resource::Oil, plant.cost);
                }
            }
        }
    }

    let income = income_for(powered);
    let player = state.player_mut(actor).ok_or(ActionError::UnknownPlayer)?;
    player.money += income;

    state.log(format!(
        "{} powered {} cities, earned {}",
        state.player(actor).map(|p| p.name.as_str()).unwrap_or("?"),
        powered,
        income
    ));

    // Advance.
    let mut remaining = remaining;
    remaining.remove(0);

    if remaining.is_empty() {
        end_of_round(state);
    } else {
        state.phase = Phase::Bureaucracy { remaining };
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// End of round
// ---------------------------------------------------------------------------

fn end_of_round(state: &mut GameState) {
    // Check for game-over condition.
    let winner = determine_winner(state);
    if let Some(winner_id) = winner {
        state.phase = Phase::GameOver { winner: winner_id };
        let name = state
            .player(winner_id)
            .map(|p| p.name.clone())
            .unwrap_or_default();
        state.log(format!("{} wins!", name));
        return;
    }

    // Replenish resource market (simplified: add back a fixed amount per resource).
    replenish_resources(state);

    // Recalculate player order: most cities first; ties broken by highest plant number.
    recalculate_player_order(state);

    state.round += 1;
    begin_auction(state);
    state.log(format!("Round {} begins", state.round));
}

fn determine_winner(state: &GameState) -> Option<PlayerId> {
    // Check if any player hit the end-game city threshold.
    let triggered = state
        .players
        .iter()
        .any(|p| p.cities.len() >= state.end_game_cities as usize);
    if !triggered {
        return None;
    }

    // Winner: player who can power the most cities.
    // Tie: most money.
    state
        .players
        .iter()
        .max_by_key(|p| (p.cities_powerable(), p.money))
        .map(|p| p.id)
}

fn recalculate_player_order(state: &mut GameState) {
    // Most cities → first. Tie: highest plant number → first.
    let mut order: Vec<PlayerId> = state.players.iter().map(|p| p.id).collect();
    order.sort_by(|&a, &b| {
        let pa = state.player(a).unwrap();
        let pb = state.player(b).unwrap();
        let ca = pa.city_count();
        let cb = pb.city_count();
        if ca != cb {
            return cb.cmp(&ca); // more cities = earlier
        }
        let plant_a = pa.plants.iter().map(|p| p.number).max().unwrap_or(0);
        let plant_b = pb.plants.iter().map(|p| p.number).max().unwrap_or(0);
        plant_b.cmp(&plant_a)
    });
    state.player_order = order;
}

fn replenish_resources(state: &mut GameState) {
    // Standard replenishment per round per player count (step 1 amounts).
    let n = state.players.len();
    let (coal, oil, garbage, uranium) = match n {
        2 => (3, 2, 1, 1),
        3 => (4, 2, 1, 1),
        4 => (5, 3, 2, 1),
        5 => (5, 4, 3, 2),
        _ => (7, 5, 3, 2),
    };
    let before = state.resources.clone();
    state.resources.replenish(Resource::Coal, coal);
    state.resources.replenish(Resource::Oil, oil);
    state.resources.replenish(Resource::Garbage, garbage);
    state.resources.replenish(Resource::Uranium, uranium);
    let dc = state.resources.coal - before.coal;
    let do_ = state.resources.oil - before.oil;
    let dg = state.resources.garbage - before.garbage;
    let du = state.resources.uranium - before.uranium;
    if dc + do_ + dg + du > 0 {
        state.log(format!(
            "Resources replenished: +{dc} coal, +{do_} oil, +{dg} garbage, +{du} uranium"
        ));
    } else {
        state.log("Resources: market already at capacity, nothing added");
    }
}

// ---------------------------------------------------------------------------
// Plant deck builder
// ---------------------------------------------------------------------------

/// Build the standard Powergrid base-game plant deck and initial market.
pub fn build_plant_deck() -> PlantMarket {
    let mut all_plants: Vec<PowerPlant> = vec![
        // number, kind, resource cost, cities powered
        PowerPlant {
            number: 3,
            kind: PlantKind::Oil,
            cost: 2,
            cities: 1,
        },
        PowerPlant {
            number: 4,
            kind: PlantKind::Coal,
            cost: 2,
            cities: 1,
        },
        PowerPlant {
            number: 5,
            kind: PlantKind::CoalOrOil,
            cost: 2,
            cities: 1,
        },
        PowerPlant {
            number: 6,
            kind: PlantKind::Garbage,
            cost: 1,
            cities: 1,
        },
        PowerPlant {
            number: 7,
            kind: PlantKind::Oil,
            cost: 3,
            cities: 2,
        },
        PowerPlant {
            number: 8,
            kind: PlantKind::Coal,
            cost: 3,
            cities: 2,
        },
        PowerPlant {
            number: 9,
            kind: PlantKind::Oil,
            cost: 1,
            cities: 1,
        },
        PowerPlant {
            number: 10,
            kind: PlantKind::Coal,
            cost: 2,
            cities: 2,
        },
        PowerPlant {
            number: 11,
            kind: PlantKind::Uranium,
            cost: 1,
            cities: 2,
        },
        PowerPlant {
            number: 12,
            kind: PlantKind::CoalOrOil,
            cost: 2,
            cities: 2,
        },
        PowerPlant {
            number: 13,
            kind: PlantKind::Wind,
            cost: 0,
            cities: 1,
        },
        PowerPlant {
            number: 14,
            kind: PlantKind::Garbage,
            cost: 2,
            cities: 2,
        },
        PowerPlant {
            number: 15,
            kind: PlantKind::Coal,
            cost: 2,
            cities: 3,
        },
        PowerPlant {
            number: 16,
            kind: PlantKind::Oil,
            cost: 2,
            cities: 3,
        },
        PowerPlant {
            number: 17,
            kind: PlantKind::Uranium,
            cost: 1,
            cities: 2,
        },
        PowerPlant {
            number: 18,
            kind: PlantKind::Wind,
            cost: 0,
            cities: 2,
        },
        PowerPlant {
            number: 19,
            kind: PlantKind::Garbage,
            cost: 2,
            cities: 3,
        },
        PowerPlant {
            number: 20,
            kind: PlantKind::Coal,
            cost: 3,
            cities: 5,
        },
        PowerPlant {
            number: 21,
            kind: PlantKind::CoalOrOil,
            cost: 2,
            cities: 4,
        },
        PowerPlant {
            number: 22,
            kind: PlantKind::Wind,
            cost: 0,
            cities: 2,
        },
        PowerPlant {
            number: 23,
            kind: PlantKind::Uranium,
            cost: 1,
            cities: 3,
        },
        PowerPlant {
            number: 24,
            kind: PlantKind::Garbage,
            cost: 2,
            cities: 4,
        },
        PowerPlant {
            number: 25,
            kind: PlantKind::Coal,
            cost: 2,
            cities: 5,
        },
        PowerPlant {
            number: 26,
            kind: PlantKind::Oil,
            cost: 2,
            cities: 5,
        },
        PowerPlant {
            number: 27,
            kind: PlantKind::Wind,
            cost: 0,
            cities: 3,
        },
        PowerPlant {
            number: 28,
            kind: PlantKind::Uranium,
            cost: 1,
            cities: 4,
        },
        PowerPlant {
            number: 29,
            kind: PlantKind::CoalOrOil,
            cost: 1,
            cities: 4,
        },
        PowerPlant {
            number: 30,
            kind: PlantKind::Garbage,
            cost: 3,
            cities: 6,
        },
        PowerPlant {
            number: 31,
            kind: PlantKind::Coal,
            cost: 3,
            cities: 6,
        },
        PowerPlant {
            number: 32,
            kind: PlantKind::Oil,
            cost: 3,
            cities: 6,
        },
        PowerPlant {
            number: 33,
            kind: PlantKind::Wind,
            cost: 0,
            cities: 4,
        },
        PowerPlant {
            number: 34,
            kind: PlantKind::Uranium,
            cost: 1,
            cities: 5,
        },
        PowerPlant {
            number: 35,
            kind: PlantKind::Oil,
            cost: 1,
            cities: 5,
        },
        PowerPlant {
            number: 36,
            kind: PlantKind::Coal,
            cost: 3,
            cities: 7,
        },
        PowerPlant {
            number: 38,
            kind: PlantKind::Garbage,
            cost: 3,
            cities: 7,
        },
        PowerPlant {
            number: 39,
            kind: PlantKind::Uranium,
            cost: 1,
            cities: 6,
        },
        PowerPlant {
            number: 40,
            kind: PlantKind::Oil,
            cost: 2,
            cities: 6,
        },
        PowerPlant {
            number: 42,
            kind: PlantKind::Coal,
            cost: 2,
            cities: 6,
        },
        PowerPlant {
            number: 44,
            kind: PlantKind::Wind,
            cost: 0,
            cities: 5,
        },
        PowerPlant {
            number: 46,
            kind: PlantKind::CoalOrOil,
            cost: 3,
            cities: 7,
        },
        PowerPlant {
            number: 50,
            kind: PlantKind::Fusion,
            cost: 0,
            cities: 6,
        },
    ];

    all_plants.sort_by_key(|p| p.number);

    // Plants 3–10 form the initial market visible at game start.
    let initial: Vec<PowerPlant> = all_plants
        .iter()
        .filter(|p| p.number <= 10)
        .cloned()
        .collect();

    // Plant 13 is set aside and placed on top of the deck at game start.
    let plant_13 = all_plants.iter().find(|p| p.number == 13).cloned();

    // Remaining plants (11–50, excluding 13) form the draw deck.
    // Reversed so that pop() draws in ascending order before shuffling.
    let deck: Vec<PowerPlant> = all_plants
        .iter()
        .filter(|p| p.number > 10 && p.number != 13)
        .rev()
        .cloned()
        .collect();

    let actual = initial.iter().take(4).cloned().collect();
    let future = initial.iter().skip(4).cloned().collect();

    PlantMarket {
        actual,
        future,
        deck,
        plant_13,
        step3_at_bottom: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::{CityData, ConnectionData, Map, MapData};

    fn test_map() -> Map {
        Map::from_data(MapData {
            name: "Test".into(),
            regions: vec!["r1".into()],
            image: None,
            cities: vec![
                CityData {
                    id: "a".into(),
                    name: "A".into(),
                    region: "r1".into(),
                    x: None,
                    y: None,
                },
                CityData {
                    id: "b".into(),
                    name: "B".into(),
                    region: "r1".into(),
                    x: None,
                    y: None,
                },
                CityData {
                    id: "c".into(),
                    name: "C".into(),
                    region: "r1".into(),
                    x: None,
                    y: None,
                },
            ],
            connections: vec![
                ConnectionData {
                    from: "a".into(),
                    to: "b".into(),
                    cost: 5,
                },
                ConnectionData {
                    from: "b".into(),
                    to: "c".into(),
                    cost: 3,
                },
            ],
            resource_slots: vec![],
            turn_order_slots: vec![],
        })
    }

    fn three_player_game() -> (GameState, PlayerId, PlayerId, PlayerId) {
        let mut state = GameState::new_with_seed(test_map(), 3, 42);
        let p1 = uuid::Uuid::new_v4();
        let p2 = uuid::Uuid::new_v4();
        let p3 = uuid::Uuid::new_v4();
        apply_action(
            &mut state,
            p1,
            Action::JoinGame {
                name: "Alice".into(),
                color: PlayerColor::Red,
            },
        )
        .unwrap();
        apply_action(
            &mut state,
            p2,
            Action::JoinGame {
                name: "Bob".into(),
                color: PlayerColor::Blue,
            },
        )
        .unwrap();
        apply_action(
            &mut state,
            p3,
            Action::JoinGame {
                name: "Carol".into(),
                color: PlayerColor::Yellow,
            },
        )
        .unwrap();
        (state, p1, p2, p3)
    }

    fn two_player_game() -> (GameState, PlayerId, PlayerId) {
        let mut state = GameState::new_with_seed(test_map(), 2, 42);
        let p1 = uuid::Uuid::new_v4();
        let p2 = uuid::Uuid::new_v4();
        apply_action(
            &mut state,
            p1,
            Action::JoinGame {
                name: "Alice".into(),
                color: PlayerColor::Red,
            },
        )
        .unwrap();
        apply_action(
            &mut state,
            p2,
            Action::JoinGame {
                name: "Bob".into(),
                color: PlayerColor::Blue,
            },
        )
        .unwrap();
        (state, p1, p2)
    }

    #[test]
    fn test_bid_order_after_overbid() {
        // Scenario: the first player selects a plant, the second overbids.
        // The next bidder should be the third player, not the first.
        let (mut state, p1, p2, p3) = three_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        // Derive the seeded turn order rather than assuming insertion order.
        let first = state.player_order[0];
        let second = state.player_order[1];
        let third = state.player_order[2];

        // First player selects the lowest-numbered plant in the actual market.
        let plant_number = {
            let Phase::Auction { .. } = &state.phase else {
                panic!("expected Auction phase");
            };
            state.market.actual[0].number
        };
        let min_bid = plant_number as u32;

        apply_action(&mut state, first, Action::SelectPlant { plant_number }).unwrap();

        // Second player overbids.
        apply_action(
            &mut state,
            second,
            Action::PlaceBid {
                amount: min_bid + 1,
            },
        )
        .unwrap();

        // Next bidder in remaining_bidders must be the third player.
        let Phase::Auction { active_bid, .. } = &state.phase else {
            panic!("expected Auction phase after second bid");
        };
        let bid = active_bid.as_ref().expect("should have active bid");
        assert_eq!(
            bid.remaining_bidders[0], third,
            "third player should bid next"
        );

        // Suppress unused variable warnings — all three player ids are needed for the game setup.
        let _ = (p1, p2, p3);
    }

    #[test]
    fn test_join_and_start() {
        let (mut state, p1, _p2) = two_player_game();
        assert_eq!(state.players.len(), 2);
        apply_action(&mut state, p1, Action::StartGame).unwrap();
        assert!(matches!(state.phase, Phase::Auction { .. }));
    }

    #[test]
    fn test_duplicate_name_rejected() {
        let mut state = GameState::new_with_seed(test_map(), 2, 42);
        let p1 = uuid::Uuid::new_v4();
        let p2 = uuid::Uuid::new_v4();
        apply_action(
            &mut state,
            p1,
            Action::JoinGame {
                name: "Alice".into(),
                color: PlayerColor::Red,
            },
        )
        .unwrap();
        let err = apply_action(
            &mut state,
            p2,
            Action::JoinGame {
                name: "Alice".into(),
                color: PlayerColor::Blue,
            },
        );
        assert!(matches!(err, Err(ActionError::NameTaken)));
    }

    #[test]
    fn test_non_host_cannot_start() {
        let (mut state, _p1, p2) = two_player_game();
        let err = apply_action(&mut state, p2, Action::StartGame);
        assert!(matches!(err, Err(ActionError::NotHost)));
    }

    #[test]
    fn test_wrong_phase_action() {
        let (mut state, p1, _p2) = two_player_game();
        // Can't build a city before game starts.
        let err = apply_action(
            &mut state,
            p1,
            Action::BuildCity {
                city_id: "a".into(),
            },
        );
        assert!(matches!(err, Err(ActionError::WrongPhase)));
    }

    #[test]
    fn test_last_player_auto_wins_plant_at_minimum() {
        let (mut state, p1, p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        // Derive actual turn order from the seeded shuffle.
        let first = state.player_order[0];
        let second = state.player_order[1];

        // First player passes their auction turn.
        apply_action(&mut state, first, Action::PassAuction).unwrap();

        // Now only the second player remains. Selecting a plant should immediately award it.
        apply_action(&mut state, second, Action::SelectPlant { plant_number: 3 }).unwrap();

        // Should have advanced past auction into BuyResources.
        assert!(
            matches!(state.phase, Phase::BuyResources { .. }),
            "expected BuyResources after last player auto-wins plant, got {:?}",
            state.phase
        );

        // Second player should own plant 3 and have been charged its minimum bid (3).
        let second_player = state.player(second).unwrap();
        assert!(second_player.plants.iter().any(|p| p.number == 3));
        assert_eq!(second_player.money, 50 - 3);

        let _ = (p1, p2);
    }

    #[test]
    fn test_resource_market_price() {
        let market = ResourceMarket::initial();
        // With full coal supply (24 units), the cheapest slots are occupied.
        let price = market.price(Resource::Coal, 1);
        assert!(price.is_some());
    }

    /// Set up a two-player game in the BuildCities phase with `first` as the current actor.
    fn two_player_build_phase() -> (GameState, PlayerId, PlayerId) {
        let (mut state, p1, p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();
        // Force the state directly into BuildCities so we don't need to replay the full
        // Auction + BuyResources flow. Give both players plenty of money.
        state.phase = Phase::BuildCities {
            remaining: state.player_order.iter().rev().cloned().collect(),
        };
        for player in &mut state.players {
            player.money = 500;
        }
        (state, p1, p2)
    }

    #[test]
    fn test_build_cities_empty_list_rejected() {
        let (mut state, p1, p2) = two_player_build_phase();
        let first = match &state.phase {
            Phase::BuildCities { remaining } => *remaining.first().unwrap(),
            _ => unreachable!(),
        };
        let err =
            apply_action(&mut state, first, Action::BuildCities { city_ids: vec![] }).unwrap_err();
        assert!(matches!(err, ActionError::EmptyBuildList), "got {err:?}");
        let _ = (p1, p2); // suppress unused warnings
    }

    #[test]
    fn test_build_cities_duplicate_rejected() {
        let (mut state, _p1, _p2) = two_player_build_phase();
        let first = match &state.phase {
            Phase::BuildCities { remaining } => *remaining.first().unwrap(),
            _ => unreachable!(),
        };
        let err = apply_action(
            &mut state,
            first,
            Action::BuildCities {
                city_ids: vec!["a".into(), "a".into()],
            },
        )
        .unwrap_err();
        assert!(
            matches!(err, ActionError::DuplicateCityInBuild),
            "got {err:?}"
        );
    }

    #[test]
    fn test_build_cities_single_equivalent_to_build_city_then_done() {
        let (mut state_batch, _, _) = two_player_build_phase();
        let (mut state_single, _, _) = two_player_build_phase();

        let first_batch = match &state_batch.phase {
            Phase::BuildCities { remaining } => *remaining.first().unwrap(),
            _ => unreachable!(),
        };
        let first_single = match &state_single.phase {
            Phase::BuildCities { remaining } => *remaining.first().unwrap(),
            _ => unreachable!(),
        };

        // Batch: build "a" and end turn.
        apply_action(
            &mut state_batch,
            first_batch,
            Action::BuildCities {
                city_ids: vec!["a".into()],
            },
        )
        .unwrap();

        // Single: build "a" then done.
        apply_action(
            &mut state_single,
            first_single,
            Action::BuildCity {
                city_id: "a".into(),
            },
        )
        .unwrap();
        apply_action(&mut state_single, first_single, Action::DoneBuilding).unwrap();

        // Both should have advanced to the same phase.
        let batch_phase = std::mem::discriminant(&state_batch.phase);
        let single_phase = std::mem::discriminant(&state_single.phase);
        assert_eq!(batch_phase, single_phase);

        // Player should own "a" and have been charged at least the slot fee.
        let player_batch = state_batch.player(first_batch).unwrap();
        let player_single = state_single.player(first_single).unwrap();
        assert!(player_batch.cities.contains(&"a".to_string()));
        assert!(player_single.cities.contains(&"a".to_string()));
        assert_eq!(player_batch.money, player_single.money);
    }

    #[test]
    fn test_build_cities_per_city_pricing() {
        // Map: a --5-- b --3-- c
        // Own {a}, build {b, c} in that order.
        // Build b: route=5 (a→b), slot=10. Total=15. Now own {a,b}.
        // Build c: route=3 (b→c), slot=10. Total=13. Now own {a,b,c}.
        // Grand total charged = 28.
        let (mut state, _p1, _p2) = two_player_build_phase();
        let first = match &state.phase {
            Phase::BuildCities { remaining } => *remaining.first().unwrap(),
            _ => unreachable!(),
        };
        let starting_money = state.player(first).unwrap().money;

        // Pre-seed city "a" as owned so the player has a network.
        state.player_mut(first).unwrap().cities.push("a".into());
        state.map.cities.get_mut("a").unwrap().owners.push(first);

        apply_action(
            &mut state,
            first,
            Action::BuildCities {
                city_ids: vec!["b".into(), "c".into()],
            },
        )
        .unwrap();

        let player = state.player(first).unwrap();
        assert!(player.cities.contains(&"b".to_string()));
        assert!(player.cities.contains(&"c".to_string()));
        // route costs: 5 + 3 = 8; slot costs: 10 + 10 = 20; total = 28.
        assert_eq!(player.money, starting_money - 28);
    }

    #[test]
    fn test_build_cities_atomic_rollback() {
        let (mut state, _p1, _p2) = two_player_build_phase();
        let first = match &state.phase {
            Phase::BuildCities { remaining } => *remaining.first().unwrap(),
            _ => unreachable!(),
        };
        // Give player only enough for the first city slot (10), not the second.
        state.player_mut(first).unwrap().money = 10;
        let money_before = 10u32;

        // Build a (first city, slot fee 10 = exactly affordable), then b (route 5 + slot 10 = 15, unaffordable).
        let err = apply_action(
            &mut state,
            first,
            Action::BuildCities {
                city_ids: vec!["a".into(), "b".into()],
            },
        )
        .unwrap_err();

        assert!(matches!(err, ActionError::CannotAffordCity), "got {err:?}");
        // State should be unchanged — player still has original money and no cities.
        let player = state.player(first).unwrap();
        assert_eq!(player.money, money_before);
        assert!(!player.cities.contains(&"a".to_string()));
    }

    #[test]
    fn test_build_cities_advances_phase() {
        let (mut state, _p1, _p2) = two_player_build_phase();
        let first = match &state.phase {
            Phase::BuildCities { remaining } => *remaining.first().unwrap(),
            _ => unreachable!(),
        };

        apply_action(
            &mut state,
            first,
            Action::BuildCities {
                city_ids: vec!["a".into()],
            },
        )
        .unwrap();

        // Turn should have advanced; first player should no longer be acting.
        match &state.phase {
            Phase::BuildCities { remaining } => {
                assert_ne!(
                    remaining.first().copied(),
                    Some(first),
                    "first player should no longer be current"
                );
            }
            Phase::Bureaucracy { .. } => {} // also valid if only one player left
            other => panic!("unexpected phase: {other:?}"),
        }
    }

    #[test]
    fn test_hybrid_plant_shared_capacity() {
        use crate::types::{PlantKind, PlayerColor, PowerPlant};

        // A CoalOrOil plant with cost=2 holds 4 total resources (coal + oil combined).
        let mut player = crate::types::Player::new("Test".into(), PlayerColor::Red);
        player.plants.push(PowerPlant {
            number: 5,
            kind: PlantKind::CoalOrOil,
            cost: 2,
            cities: 1,
        });

        // Can buy up to 4 coal with 0 oil stored.
        assert!(player.can_add_resource(Resource::Coal, 4));
        // Cannot buy 5 coal — exceeds total slots.
        assert!(!player.can_add_resource(Resource::Coal, 5));

        // After storing 4 coal, cannot buy any oil.
        player.resources.coal = 4;
        assert!(!player.can_add_resource(Resource::Oil, 1));

        // After storing 2 coal, can only buy 2 more oil.
        player.resources.coal = 2;
        assert!(player.can_add_resource(Resource::Oil, 2));
        assert!(!player.can_add_resource(Resource::Oil, 3));
    }

    #[test]
    fn test_resources_replenished_after_round() {
        use crate::types::{PlantKind, PowerPlant};

        let (mut state, p1, p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        // Give both players a coal plant (cost=2, powers 1 city) and money.
        for player in &mut state.players {
            player.plants.push(PowerPlant {
                number: 4,
                kind: PlantKind::Coal,
                cost: 2,
                cities: 1,
            });
            player.money = 500;
        }

        // Force into BuyResources (reversed player order).
        let buy_order: Vec<PlayerId> = state.player_order.iter().rev().cloned().collect();
        state.phase = Phase::BuyResources {
            remaining: buy_order,
        };

        // Each player buys 2 coal (4 total consumed from market).
        let buy_order_snapshot: Vec<PlayerId> = match &state.phase {
            Phase::BuyResources { remaining } => remaining.clone(),
            _ => unreachable!(),
        };
        for actor in &buy_order_snapshot {
            apply_action(
                &mut state,
                *actor,
                Action::BuyResources {
                    resource: Resource::Coal,
                    amount: 2,
                },
            )
            .unwrap();
            apply_action(&mut state, *actor, Action::DoneBuying).unwrap();
        }
        let coal_after_buy = state.resources.coal;
        assert_eq!(coal_after_buy, 20, "expected 4 coal bought from initial 24");

        // Both players skip building.
        assert!(matches!(state.phase, Phase::BuildCities { .. }));
        let build_order: Vec<PlayerId> = match &state.phase {
            Phase::BuildCities { remaining } => remaining.clone(),
            _ => unreachable!(),
        };
        for actor in &build_order {
            apply_action(&mut state, *actor, Action::DoneBuilding).unwrap();
        }

        // Bureaucracy: both players power their cities (no cities built, so 0 powered).
        assert!(matches!(state.phase, Phase::Bureaucracy { .. }));
        let power_order: Vec<PlayerId> = match &state.phase {
            Phase::Bureaucracy { remaining } => remaining.clone(),
            _ => unreachable!(),
        };
        for actor in &power_order {
            apply_action(
                &mut state,
                *actor,
                Action::PowerCities {
                    plant_numbers: vec![4],
                },
            )
            .unwrap();
        }

        // After end_of_round, 3 coal should have been replenished (2-player step-1 rate).
        assert!(
            state.resources.coal > coal_after_buy,
            "expected coal to increase after replenishment; got {} (was {})",
            state.resources.coal,
            coal_after_buy
        );
        assert_eq!(
            state.resources.coal,
            coal_after_buy + 3,
            "expected exactly 3 coal replenished for 2 players"
        );
        let _ = (p1, p2);
    }
}
