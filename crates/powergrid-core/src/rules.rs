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
        Action::BuyResourceBatch { purchases } => {
            handle_buy_resource_batch(state, actor, purchases)
        }
        Action::DoneBuying => handle_done_buying(state, actor),
        Action::BuildCity { city_id } => handle_build_city(state, actor, city_id),
        Action::BuildCities { city_ids } => handle_build_cities(state, actor, city_ids),
        Action::DoneBuilding => handle_done_building(state, actor),
        Action::PowerCities { plant_numbers } => handle_power_cities(state, actor, plant_numbers),
        Action::DiscardPlant { plant_number } => handle_discard_plant(state, actor, plant_number),
        Action::DiscardResource { gas, oil } => handle_discard_resource(state, actor, gas, oil),
        Action::PowerCitiesFuel { gas, oil } => handle_power_cities_fuel(state, actor, gas, oil),
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

    state.end_game_cities = match player_count {
        2 => 21,
        3 => 17,
        4 => 17,
        5 => 15,
        _ => 14, // 6 players
    };

    // Select active regions based on player count.
    let region_count = match player_count {
        2 | 3 => 3,
        4 => 4,
        _ => 5, // 5 or 6 players
    };
    let mut all_regions = state.map.regions.clone();
    all_regions.shuffle(&mut rng);
    state.active_regions = all_regions.into_iter().take(region_count).collect();
    state.log(format!(
        "Active regions: {}",
        state.active_regions.join(", ")
    ));

    state.market.setup_deck(&mut rng, player_count);

    begin_auction(state);
    state.log("Game started!".to_string());
    Ok(())
}

// ---------------------------------------------------------------------------
// Auction phase helpers
// ---------------------------------------------------------------------------

/// Minimum bid for a plant: 1 if it holds the discount token, else its printed number.
pub fn effective_min_bid(market: &crate::types::PlantMarket, plant_number: u8) -> u32 {
    if market.discount_token == Some(plant_number) {
        1
    } else {
        plant_number as u32
    }
}

/// Clears the discount token if the tokened plant is no longer in `actual`, or if a
/// plant smaller than the tokened plant has appeared in `actual`. Safe to call any time.
fn clear_discount_if_invalidated(state: &mut GameState) {
    if let Some(tok) = state.market.discount_token {
        let still_present = state.market.actual.iter().any(|p| p.number == tok);
        let undercut = state.market.actual.first().is_some_and(|p| p.number < tok);
        if !still_present || undercut {
            state.market.discount_token = None;
            state.log(format!("Discount token removed from plant {}.", tok));
        }
    }
}

fn begin_auction(state: &mut GameState) {
    apply_pending_step3(state, crate::state::Step3Pending::NextRound);
    state.market.discount_token = state.market.actual.first().map(|p| p.number);
    if let Some(n) = state.market.discount_token {
        state.log(format!(
            "Discount token placed on plant {} (min bid: 1 Elektro).",
            n
        ));
    }
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
    let min_bid = effective_min_bid(&state.market, plant_number);
    if min_bid > player_money {
        return Err(ActionError::CannotAfford);
    }

    // Start bid at the plant's effective minimum bid (1 if discounted, else plant number).
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
        return award_plant(state, actor, plant_number, min_bid, bought, passed);
    }

    state.phase = Phase::Auction {
        current_bidder_idx,
        active_bid: Some(ActiveBid {
            plant_number,
            highest_bidder: actor,
            amount: min_bid,
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

        if state.round == 1 {
            return Err(ActionError::MustBuyPlantInRoundOne);
        }

        passed.push(actor);
        advance_auction(state, bought, passed);
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

    // Clear token if: bought plant was the tokened one, or refill drew a smaller plant.
    clear_discount_if_invalidated(state);

    let player = state.player_mut(winner).ok_or(ActionError::UnknownPlayer)?;
    player.money -= cost;

    if player.plants.len() >= 3 {
        // Player already has 3 plants — pause and ask them which to discard.
        bought.push(winner);
        state.log(format!(
            "{} bought plant {} for {} — choose a plant to discard",
            state.player(winner).map(|p| p.name.as_str()).unwrap_or("?"),
            plant_number,
            cost
        ));
        // Check if purchasing this plant triggered the Step 3 card.
        note_step3_trigger(state);
        state.phase = Phase::DiscardPlant {
            player: winner,
            new_plant: plant,
            bought,
            passed,
        };
        return Ok(());
    }

    // Normal path: player has fewer than 3 plants.
    player.plants.push(plant.clone());
    player.plants.sort_by_key(|p| p.number);

    state.log(format!(
        "{} bought plant {} for {}",
        state.player(winner).map(|p| p.name.as_str()).unwrap_or("?"),
        plant_number,
        cost
    ));

    // Check if purchasing this plant triggered the Step 3 card.
    note_step3_trigger(state);

    bought.push(winner);

    advance_auction(state, bought, passed);
    Ok(())
}

fn handle_discard_plant(
    state: &mut GameState,
    actor: PlayerId,
    plant_number: u8,
) -> Result<(), ActionError> {
    let (player_id, new_plant, bought, passed) = match &state.phase {
        Phase::DiscardPlant {
            player,
            new_plant,
            bought,
            passed,
        } => (*player, new_plant.clone(), bought.clone(), passed.clone()),
        _ => return Err(ActionError::WrongPhase),
    };

    if actor != player_id {
        return Err(ActionError::NotYourTurn);
    }

    if plant_number == new_plant.number {
        return Err(ActionError::CannotDiscardNewPlant);
    }

    let player = state.player_mut(actor).ok_or(ActionError::UnknownPlayer)?;

    if !player.plants.iter().any(|p| p.number == plant_number) {
        return Err(ActionError::PlantNotOwned(plant_number));
    }

    // Remove the chosen plant and add the new one.
    player.plants.retain(|p| p.number != plant_number);
    player.plants.push(new_plant);
    player.plants.sort_by_key(|p| p.number);

    // Per-resource clamp: drop any resource that exceeds its individual cap (always
    // unambiguous — only one resource type can be responsible for each overflow).
    let excesses: Vec<(Resource, u8)> = {
        let player = state.player_mut(actor).ok_or(ActionError::UnknownPlayer)?;
        [
            Resource::Coal,
            Resource::Oil,
            Resource::Gas,
            Resource::Uranium,
        ]
        .into_iter()
        .filter_map(|r| {
            let cap = player.resource_capacity(r);
            let held = player.resources.get(r);
            if held > cap {
                Some((r, held - cap))
            } else {
                None
            }
        })
        .collect()
    };
    for (resource, excess) in excesses {
        {
            let player = state.player_mut(actor).ok_or(ActionError::UnknownPlayer)?;
            player.resources.remove(resource, excess);
        }
        state.resources.replenish(resource, excess);
    }

    // Shared-slot check: coal+oil may jointly exceed hybrid capacity even though
    // neither overflows its individual cap (only possible when both coal and oil
    // are nonzero after the per-resource clamp above).
    let drop_total = state
        .player(actor)
        .ok_or(ActionError::UnknownPlayer)?
        .shared_slot_overflow();

    if drop_total > 0 {
        // Ambiguous: pause and ask the player to choose the coal/oil split.
        // After per-resource clamping it is guaranteed that both coal > 0 and oil > 0
        // here; otherwise per-resource overflow would already have been resolved above.
        state.log(format!(
            "{} discarded plant {} — choose which fuel to discard",
            state.player(actor).map(|p| p.name.as_str()).unwrap_or("?"),
            plant_number
        ));
        state.phase = Phase::DiscardResource {
            player: actor,
            drop_total,
            bought,
            passed,
        };
        return Ok(());
    }

    state.log(format!(
        "{} discarded plant {}",
        state.player(actor).map(|p| p.name.as_str()).unwrap_or("?"),
        plant_number
    ));

    advance_auction(state, bought, passed);
    Ok(())
}

fn handle_discard_resource(
    state: &mut GameState,
    actor: PlayerId,
    gas: u8,
    oil: u8,
) -> Result<(), ActionError> {
    let (player_id, drop_total, bought, passed) = match &state.phase {
        Phase::DiscardResource {
            player,
            drop_total,
            bought,
            passed,
        } => (*player, *drop_total, bought.clone(), passed.clone()),
        _ => return Err(ActionError::WrongPhase),
    };

    if actor != player_id {
        return Err(ActionError::NotYourTurn);
    }

    let player = state.player(actor).ok_or(ActionError::UnknownPlayer)?;

    // Validate the split: sum must match, neither may exceed held.
    if gas + oil != drop_total || gas > player.resources.gas || oil > player.resources.oil {
        return Err(ActionError::InvalidDiscardSplit);
    }

    // Validate that this split actually resolves the overflow (no over-dropping).
    // After removing gas and oil, shared_slot_overflow must be 0.
    {
        let mut sim = player.clone();
        sim.resources.remove(Resource::Gas, gas);
        sim.resources.remove(Resource::Oil, oil);
        if sim.shared_slot_overflow() != 0 {
            return Err(ActionError::InvalidDiscardSplit);
        }
    }

    // Apply the drops.
    if gas > 0 {
        state
            .player_mut(actor)
            .ok_or(ActionError::UnknownPlayer)?
            .resources
            .remove(Resource::Gas, gas);
        state.resources.replenish(Resource::Gas, gas);
    }
    if oil > 0 {
        state
            .player_mut(actor)
            .ok_or(ActionError::UnknownPlayer)?
            .resources
            .remove(Resource::Oil, oil);
        state.resources.replenish(Resource::Oil, oil);
    }

    state.log(format!(
        "{} discarded {} gas and {} oil",
        state.player(actor).map(|p| p.name.as_str()).unwrap_or("?"),
        gas,
        oil,
    ));

    advance_auction(state, bought, passed);
    Ok(())
}

fn advance_auction(state: &mut GameState, bought: Vec<PlayerId>, passed: Vec<PlayerId>) {
    let total = state.player_order.len();
    let all_done: Vec<PlayerId> = bought.iter().chain(passed.iter()).cloned().collect();

    // Check if everyone has acted.
    if all_done.len() >= total {
        // End of auction. Discount rule: if the tokened plant was never bought, remove
        // it from the game entirely and replace via a new draw.
        let removed_discount = if let Some(tok) = state.market.discount_token {
            if let Some(pos) = state.market.actual.iter().position(|p| p.number == tok) {
                state.market.actual.remove(pos);
                state.market.refill();
                state.log(format!(
                    "Discount plant {} was not bought — removed from the game.",
                    tok
                ));
                true
            } else {
                false
            }
        } else {
            false
        };
        state.market.discount_token = None;
        // Original rule: if no one bought *any* plant, also remove the new lowest plant —
        // but skip if we already removed via the discount path to avoid double-removal.
        if bought.is_empty() && !removed_discount {
            state.market.remove_lowest();
        }
        note_step3_trigger(state);
        begin_buy_resources(state);
        return;
    }

    // Find the earliest player in turn order who hasn't bought or passed.
    let mut next_idx = 0;
    let mut iterations = 0;
    while all_done.contains(&state.player_order[next_idx]) {
        next_idx = (next_idx + 1) % total;
        iterations += 1;
        if iterations > total {
            let removed_discount = if let Some(tok) = state.market.discount_token {
                if let Some(pos) = state.market.actual.iter().position(|p| p.number == tok) {
                    state.market.actual.remove(pos);
                    state.market.refill();
                    state.log(format!(
                        "Discount plant {} was not bought — removed from the game.",
                        tok
                    ));
                    true
                } else {
                    false
                }
            } else {
                false
            };
            state.market.discount_token = None;
            if bought.is_empty() && !removed_discount {
                state.market.remove_lowest();
            }
            note_step3_trigger(state);
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
    // Defensive: ensure the discount token is always None outside Phase 2.
    // This covers the deferred Step 3 transition path.
    state.market.discount_token = None;
    apply_pending_step3(state, crate::state::Step3Pending::AfterAuction);
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

    let name = state
        .player(actor)
        .map(|p| p.name.clone())
        .unwrap_or_default();
    state.log(format!(
        "{name} bought {amount} {:?} for ${cost}",
        format!("{resource:?}").to_lowercase()
    ));

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

fn handle_buy_resource_batch(
    state: &mut GameState,
    actor: PlayerId,
    purchases: Vec<(Resource, u8)>,
) -> Result<(), ActionError> {
    let mut remaining = match &state.phase {
        Phase::BuyResources { remaining } => remaining.clone(),
        _ => return Err(ActionError::WrongPhase),
    };

    if remaining.first().copied() != Some(actor) {
        return Err(ActionError::NotYourTurn);
    }

    // Empty purchases = skip buying (equivalent to DoneBuying).
    if !purchases.is_empty() {
        // Validate and apply on a scratch copy for atomicity.
        let mut scratch = state.clone();
        let mut total_cost = 0u32;
        for (resource, amount) in &purchases {
            let cost = scratch
                .resources
                .price(*resource, *amount)
                .ok_or(ActionError::ResourceUnavailable)?;

            let player = scratch.player(actor).ok_or(ActionError::UnknownPlayer)?;
            if player.money < cost {
                return Err(ActionError::CannotAfford);
            }
            if !player.can_add_resource(*resource, *amount) {
                return Err(ActionError::OverCapacity);
            }

            scratch.resources.take(*resource, *amount);
            let player = scratch
                .player_mut(actor)
                .ok_or(ActionError::UnknownPlayer)?;
            player.money -= cost;
            player.resources.add(*resource, *amount);
            total_cost += cost;
        }
        // All succeeded — commit.
        *state = scratch;

        let name = state
            .player(actor)
            .map(|p| p.name.clone())
            .unwrap_or_default();
        let summary: Vec<String> = purchases
            .iter()
            .map(|(r, a)| format!("{a} {}", format!("{r:?}").to_lowercase()))
            .collect();
        state.log(format!(
            "{name} bought {} for ${total_cost}",
            summary.join(", ")
        ));
    }

    // Advance turn.
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

    if !state.is_city_active(city_id) {
        return Err(ActionError::CityRegionInactive(city_id.to_string()));
    }

    let max_per_city = state.step as usize;
    if city.owners.len() >= max_per_city {
        return Err(ActionError::CityFull(city_id.to_string()));
    }
    if city.owners.contains(&actor) {
        return Err(ActionError::AlreadyBuiltThere);
    }

    let owned_cities = state.player_cities(actor);
    let route_cost = state
        .map
        .connection_cost_to(&owned_cities, city_id)
        .unwrap_or(0);
    let city_slot_cost = connection_cost(state.map.cities[city_id].owners.len());
    let total_cost = route_cost + city_slot_cost;

    let player = state.player(actor).ok_or(ActionError::UnknownPlayer)?;
    if player.money < total_cost {
        return Err(ActionError::CannotAffordCity);
    }

    let player = state.player_mut(actor).ok_or(ActionError::UnknownPlayer)?;
    player.money -= total_cost;

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
        check_step2_trigger(state);
        begin_bureaucracy(state);
    } else {
        state.phase = Phase::BuildCities { remaining };
    }
}

fn check_end_game_trigger(state: &mut GameState) {
    let ids: Vec<_> = state.players.iter().map(|p| p.id).collect();
    let max_cities = ids
        .iter()
        .map(|&id| state.player_city_count(id))
        .max()
        .unwrap_or(0);
    if max_cities >= state.end_game_cities as usize {
        state.log("End-game triggered! Finish the round.".to_string());
    }
}

fn check_step2_trigger(state: &mut GameState) {
    if state.step != 1 {
        return;
    }
    let ids: Vec<_> = state.players.iter().map(|p| p.id).collect();
    let max_cities = ids
        .iter()
        .map(|&id| state.player_city_count(id))
        .max()
        .unwrap_or(0);
    if max_cities >= 7 {
        state.step = 2;
        state.market.remove_lowest();
        note_step3_trigger(state);
        state.log("Step 2 begins!".to_string());
    }
}

/// Called after any market refill that might have drawn the Step 3 card.
/// Records a deferred trigger on `state.step3_pending` and, for Phase 4/5,
/// immediately reshapes the market. Does NOT set `state.step = 3` — that is
/// deferred to the next phase boundary via `apply_pending_step3`.
fn note_step3_trigger(state: &mut GameState) {
    if !state.market.step3_triggered {
        return;
    }
    state.market.step3_triggered = false;

    match &state.phase {
        Phase::Auction { .. } | Phase::DiscardPlant { .. } => {
            // Phase 2: defer everything (market transition + step) to begin_buy_resources.
            state.step3_pending = Some(crate::state::Step3Pending::AfterAuction);
            state
                .log("Step 3 card drawn — takes effect at the start of Buy Resources.".to_string());
        }
        Phase::BuildCities { .. } => {
            // Phase 4: reshape market now; defer state.step to begin_bureaucracy.
            apply_step3_market_transition(state);
            state.step3_pending = Some(crate::state::Step3Pending::AfterBuilding);
            state.log("Step 3 card drawn — takes effect at the start of Bureaucracy.".to_string());
        }
        Phase::Bureaucracy { .. } => {
            // Phase 5: reshape market now; defer state.step to next begin_auction.
            // Current Phase 5 still uses Step 2 resupply (state.step not changed yet).
            apply_step3_market_transition(state);
            state.step3_pending = Some(crate::state::Step3Pending::NextRound);
            state.log("Step 3 card drawn — takes effect at the start of next round.".to_string());
        }
        _ => {
            apply_step3_market_transition(state);
            state.step3_pending = Some(crate::state::Step3Pending::NextRound);
        }
    }
}

/// Applies the market-side Step 3 transition: remove lowest plant, move
/// `below_step3` into the deck, shuffle, set `in_step3`, refill to 6.
/// Does NOT change `state.step`.
fn apply_step3_market_transition(state: &mut GameState) {
    if !state.market.actual.is_empty() {
        state.market.actual.remove(0);
    }
    if let Some(below) = state.market.below_step3.take() {
        state.market.deck = below;
    }
    let mut rng = match state.rng_seed {
        Some(seed) => rand::rngs::SmallRng::seed_from_u64(seed),
        None => rand::rngs::SmallRng::from_entropy(),
    };
    state.market.deck.shuffle(&mut rng);
    state.market.in_step3 = true;
    state.market.refill();
}

/// Called at each phase boundary. If `step3_pending` matches `expect`, applies
/// the deferred portion: `state.step = 3`, plus the market transition for the
/// Phase-2 case (which was fully deferred).
fn apply_pending_step3(state: &mut GameState, expect: crate::state::Step3Pending) {
    if state.step3_pending.as_ref() != Some(&expect) {
        return;
    }
    state.step3_pending = None;
    if expect == crate::state::Step3Pending::AfterAuction {
        apply_step3_market_transition(state);
    }
    state.step = 3;
    state.log("Step 3 begins!".to_string());
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
    let ids: Vec<_> = state.players.iter().map(|p| p.id).collect();
    let max_cities = ids
        .iter()
        .map(|&id| state.player_city_count(id))
        .max()
        .unwrap_or(0);
    state.market.remove_obsolete(max_cities);
    note_step3_trigger(state);

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
    let ids: Vec<_> = state.players.iter().map(|p| p.id).collect();
    let max_cities = ids
        .iter()
        .map(|&id| state.player_city_count(id))
        .max()
        .unwrap_or(0);
    state.market.remove_obsolete(max_cities);
    note_step3_trigger(state);
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
    apply_pending_step3(state, crate::state::Step3Pending::AfterBuilding);
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

    if !remaining.contains(&actor) {
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

    // Fire exactly the submitted plants; the player is responsible for the selection.
    let (best_subset_numbers, best_powered, best_resources) = {
        let player = state.player(actor).ok_or(ActionError::UnknownPlayer)?;
        let cities_owned = state.player_city_count(actor) as u8;
        let subset: Vec<&PowerPlant> = plant_numbers
            .iter()
            .map(|&num| player.plants.iter().find(|p| p.number == num).unwrap())
            .collect();
        let (powered, res) = check_plant_feasibility(&subset, &player.resources)
            .ok_or(ActionError::InfeasiblePlantSelection)?;
        (plant_numbers.clone(), powered.min(cities_owned), res)
    };

    // Check whether the hybrid fuel split is ambiguous for the chosen subset.
    let hybrid_cost = {
        let player = state.player(actor).ok_or(ActionError::UnknownPlayer)?;
        best_subset_numbers
            .iter()
            .filter_map(|&num| player.plants.iter().find(|p| p.number == num))
            .filter(|p| p.kind == PlantKind::GasOrOil)
            .map(|p| p.cost)
            .sum::<u8>()
    };

    if hybrid_cost > 0 {
        let player = state.player(actor).ok_or(ActionError::UnknownPlayer)?;
        let pure_gas: u8 = best_subset_numbers
            .iter()
            .filter_map(|&num| player.plants.iter().find(|p| p.number == num))
            .filter(|p| p.kind == PlantKind::Gas)
            .map(|p| p.cost)
            .sum();
        let pure_oil: u8 = best_subset_numbers
            .iter()
            .filter_map(|&num| player.plants.iter().find(|p| p.number == num))
            .filter(|p| p.kind == PlantKind::Oil)
            .map(|p| p.cost)
            .sum();
        let gas_after_pure = player.resources.gas.saturating_sub(pure_gas);
        let oil_after_pure = player.resources.oil.saturating_sub(pure_oil);
        let min_gas = hybrid_cost.saturating_sub(oil_after_pure);
        let max_gas = hybrid_cost.min(gas_after_pure);
        if min_gas < max_gas {
            // Split is genuinely ambiguous — pause and ask the player.
            state.phase = Phase::PowerCitiesFuel {
                player: actor,
                plant_numbers: best_subset_numbers,
                hybrid_cost,
                remaining,
            };
            return Ok(());
        }
    }

    // Unambiguous: apply the deterministic allocation.
    apply_power_result(state, actor, best_powered, best_resources, remaining)
}

fn apply_power_result(
    state: &mut GameState,
    actor: PlayerId,
    powered: u8,
    best_resources: PlayerResources,
    remaining: Vec<PlayerId>,
) -> Result<(), ActionError> {
    {
        let player = state.player_mut(actor).ok_or(ActionError::UnknownPlayer)?;
        player.resources = best_resources;
    }

    let income = income_for(powered);
    let player = state.player_mut(actor).ok_or(ActionError::UnknownPlayer)?;
    player.last_cities_powered = powered;
    player.money += income;

    state.log(format!(
        "{} powered {} cities, earned {}",
        state.player(actor).map(|p| p.name.as_str()).unwrap_or("?"),
        powered,
        income
    ));

    let mut remaining = remaining;
    if let Some(pos) = remaining.iter().position(|&id| id == actor) {
        remaining.remove(pos);
    }

    if remaining.is_empty() {
        end_of_round(state);
    } else {
        state.phase = Phase::Bureaucracy { remaining };
    }
    Ok(())
}

fn handle_power_cities_fuel(
    state: &mut GameState,
    actor: PlayerId,
    gas: u8,
    oil: u8,
) -> Result<(), ActionError> {
    let (player_id, plant_numbers, hybrid_cost, remaining) = match &state.phase {
        Phase::PowerCitiesFuel {
            player,
            plant_numbers,
            hybrid_cost,
            remaining,
        } => (
            *player,
            plant_numbers.clone(),
            *hybrid_cost,
            remaining.clone(),
        ),
        _ => return Err(ActionError::WrongPhase),
    };

    if actor != player_id {
        return Err(ActionError::NotYourTurn);
    }

    if gas + oil != hybrid_cost {
        return Err(ActionError::InvalidFuelSplit);
    }

    let player = state.player(actor).ok_or(ActionError::UnknownPlayer)?;

    // Compute per-type costs from the chosen subset.
    let mut pure_coal_cost: u8 = 0;
    let mut pure_oil_cost: u8 = 0;
    let mut pure_gas_cost: u8 = 0;
    let mut pure_uranium_cost: u8 = 0;
    let mut powered: u8 = 0;

    for &num in &plant_numbers {
        let plant = player
            .plants
            .iter()
            .find(|p| p.number == num)
            .ok_or(ActionError::PlantNotOwned(num))?;
        match plant.kind {
            PlantKind::Coal => pure_coal_cost += plant.cost,
            PlantKind::Oil => pure_oil_cost += plant.cost,
            PlantKind::Gas => pure_gas_cost += plant.cost,
            PlantKind::Uranium => pure_uranium_cost += plant.cost,
            PlantKind::GasOrOil | PlantKind::Wind => {}
        }
        powered += plant.cities;
    }

    let cities_owned = state.player_city_count(actor) as u8;
    let powered = powered.min(cities_owned);

    // Validate the split is feasible.
    if pure_coal_cost > player.resources.coal
        || pure_oil_cost + oil > player.resources.oil
        || pure_gas_cost + gas > player.resources.gas
        || pure_uranium_cost > player.resources.uranium
    {
        return Err(ActionError::InvalidFuelSplit);
    }

    // Build the resulting resource state and apply it.
    let new_resources = PlayerResources {
        coal: player.resources.coal - pure_coal_cost,
        oil: player.resources.oil - pure_oil_cost - oil,
        gas: player.resources.gas - pure_gas_cost - gas,
        uranium: player.resources.uranium - pure_uranium_cost,
    };

    apply_power_result(state, actor, powered, new_resources, remaining)
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

    // End-of-round market update.
    if state.step >= 3 {
        // Step 3: remove the lowest plant from the market entirely.
        state.market.remove_lowest();
    } else {
        // Steps 1 & 2: cycle the highest future-market plant to the bottom of the draw deck.
        state.market.cycle_highest_to_bottom();
        note_step3_trigger(state);
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
        .any(|p| state.player_city_count(p.id) >= state.end_game_cities as usize);
    if !triggered {
        return None;
    }

    // Winner: most cities actually powered; tie: most money; tie: most cities in network.
    state
        .players
        .iter()
        .max_by_key(|p| {
            (
                p.last_cities_powered,
                p.money,
                state.player_city_count(p.id),
            )
        })
        .map(|p| p.id)
}

fn recalculate_player_order(state: &mut GameState) {
    // Most cities → first. Tie: highest plant number → first.
    let mut order: Vec<PlayerId> = state.players.iter().map(|p| p.id).collect();
    order.sort_by(|&a, &b| {
        let ca = state.player_city_count(a);
        let cb = state.player_city_count(b);
        if ca != cb {
            return cb.cmp(&ca); // more cities = earlier
        }
        let plant_a = state
            .player(a)
            .map(|p| p.plants.iter().map(|pl| pl.number).max().unwrap_or(0))
            .unwrap_or(0);
        let plant_b = state
            .player(b)
            .map(|p| p.plants.iter().map(|pl| pl.number).max().unwrap_or(0))
            .unwrap_or(0);
        plant_b.cmp(&plant_a)
    });
    state.player_order = order;
}

/// Returns (coal, oil, gas, uranium) replenishment amounts for the given step and player count.
pub fn replenishment_amounts(step: u8, n_players: usize) -> (u8, u8, u8, u8) {
    match step {
        1 => match n_players {
            2 => (3, 2, 1, 1),
            3 => (4, 2, 1, 1),
            4 => (5, 3, 2, 1),
            5 => (5, 4, 3, 2),
            _ => (7, 5, 3, 2),
        },
        2 => match n_players {
            2 => (4, 2, 1, 1),
            3 => (5, 3, 2, 1),
            4 => (6, 4, 3, 2),
            5 => (7, 5, 3, 3),
            _ => (9, 6, 5, 3),
        },
        _ => match n_players {
            2 => (3, 4, 3, 1),
            3 => (3, 4, 3, 1),
            4 => (4, 5, 4, 2),
            5 => (5, 6, 5, 3),
            _ => (7, 7, 6, 3),
        },
    }
}

fn replenish_resources(state: &mut GameState) {
    let (coal, oil, gas, uranium) = replenishment_amounts(state.step, state.players.len());
    let before = state.resources.clone();
    state.resources.replenish(Resource::Coal, coal);
    state.resources.replenish(Resource::Oil, oil);
    state.resources.replenish(Resource::Gas, gas);
    state.resources.replenish(Resource::Uranium, uranium);
    let dc = state.resources.coal - before.coal;
    let do_ = state.resources.oil - before.oil;
    let dg = state.resources.gas - before.gas;
    let du = state.resources.uranium - before.uranium;
    if dc + do_ + dg + du > 0 {
        state.log(format!(
            "Resources replenished: +{dc} coal, +{do_} oil, +{dg} gas, +{du} uranium"
        ));
    } else {
        state.log("Resources: market already at capacity, nothing added");
    }
}

// ---------------------------------------------------------------------------
// Plant deck builder
// ---------------------------------------------------------------------------

/// Build the Power Grid Deluxe plant deck (42 plants).
/// All plants are placed in `deck`; `actual` and `future` are empty.
/// Call `PlantMarket::setup_deck` after players join to populate the market.
pub fn build_plant_deck() -> PlantMarket {
    // number, kind, resource cost, cities powered
    let mut deck: Vec<PowerPlant> = vec![
        PowerPlant {
            number: 3,
            kind: PlantKind::Coal,
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
            kind: PlantKind::Gas,
            cost: 2,
            cities: 1,
        },
        PowerPlant {
            number: 6,
            kind: PlantKind::Oil,
            cost: 1,
            cities: 1,
        },
        PowerPlant {
            number: 7,
            kind: PlantKind::Coal,
            cost: 1,
            cities: 1,
        },
        PowerPlant {
            number: 8,
            kind: PlantKind::GasOrOil,
            cost: 3,
            cities: 2,
        },
        PowerPlant {
            number: 9,
            kind: PlantKind::Coal,
            cost: 3,
            cities: 2,
        },
        PowerPlant {
            number: 10,
            kind: PlantKind::Oil,
            cost: 2,
            cities: 2,
        },
        PowerPlant {
            number: 11,
            kind: PlantKind::Wind,
            cost: 0,
            cities: 1,
        },
        PowerPlant {
            number: 12,
            kind: PlantKind::Coal,
            cost: 2,
            cities: 2,
        },
        PowerPlant {
            number: 13,
            kind: PlantKind::Uranium,
            cost: 1,
            cities: 2,
        },
        PowerPlant {
            number: 14,
            kind: PlantKind::Gas,
            cost: 1,
            cities: 2,
        },
        PowerPlant {
            number: 15,
            kind: PlantKind::Coal,
            cost: 1,
            cities: 2,
        },
        PowerPlant {
            number: 16,
            kind: PlantKind::Gas,
            cost: 2,
            cities: 3,
        },
        PowerPlant {
            number: 17,
            kind: PlantKind::Wind,
            cost: 0,
            cities: 2,
        },
        PowerPlant {
            number: 18,
            kind: PlantKind::Oil,
            cost: 2,
            cities: 3,
        },
        PowerPlant {
            number: 19,
            kind: PlantKind::Gas,
            cost: 1,
            cities: 3,
        },
        PowerPlant {
            number: 20,
            kind: PlantKind::Coal,
            cost: 3,
            cities: 4,
        },
        PowerPlant {
            number: 21,
            kind: PlantKind::Uranium,
            cost: 1,
            cities: 3,
        },
        PowerPlant {
            number: 22,
            kind: PlantKind::GasOrOil,
            cost: 3,
            cities: 5,
        },
        PowerPlant {
            number: 23,
            kind: PlantKind::Oil,
            cost: 2,
            cities: 4,
        },
        PowerPlant {
            number: 24,
            kind: PlantKind::Wind,
            cost: 0,
            cities: 3,
        },
        PowerPlant {
            number: 25,
            kind: PlantKind::Coal,
            cost: 2,
            cities: 5,
        },
        PowerPlant {
            number: 26,
            kind: PlantKind::Gas,
            cost: 1,
            cities: 4,
        },
        PowerPlant {
            number: 27,
            kind: PlantKind::Coal,
            cost: 1,
            cities: 4,
        },
        PowerPlant {
            number: 28,
            kind: PlantKind::Wind,
            cost: 0,
            cities: 3,
        },
        PowerPlant {
            number: 29,
            kind: PlantKind::Coal,
            cost: 2,
            cities: 5,
        },
        PowerPlant {
            number: 30,
            kind: PlantKind::Oil,
            cost: 2,
            cities: 5,
        },
        PowerPlant {
            number: 31,
            kind: PlantKind::Wind,
            cost: 0,
            cities: 4,
        },
        PowerPlant {
            number: 32,
            kind: PlantKind::Uranium,
            cost: 2,
            cities: 5,
        },
        PowerPlant {
            number: 33,
            kind: PlantKind::Coal,
            cost: 3,
            cities: 6,
        },
        PowerPlant {
            number: 34,
            kind: PlantKind::Gas,
            cost: 3,
            cities: 6,
        },
        PowerPlant {
            number: 35,
            kind: PlantKind::GasOrOil,
            cost: 2,
            cities: 5,
        },
        PowerPlant {
            number: 36,
            kind: PlantKind::Wind,
            cost: 0,
            cities: 5,
        },
        PowerPlant {
            number: 37,
            kind: PlantKind::Uranium,
            cost: 2,
            cities: 6,
        },
        PowerPlant {
            number: 38,
            kind: PlantKind::Oil,
            cost: 3,
            cities: 6,
        },
        PowerPlant {
            number: 39,
            kind: PlantKind::Gas,
            cost: 2,
            cities: 6,
        },
        PowerPlant {
            number: 40,
            kind: PlantKind::Coal,
            cost: 2,
            cities: 6,
        },
        PowerPlant {
            number: 42,
            kind: PlantKind::Oil,
            cost: 2,
            cities: 6,
        },
        PowerPlant {
            number: 44,
            kind: PlantKind::Wind,
            cost: 0,
            cities: 6,
        },
        PowerPlant {
            number: 46,
            kind: PlantKind::Gas,
            cost: 2,
            cities: 7,
        },
        PowerPlant {
            number: 50,
            kind: PlantKind::Uranium,
            cost: 2,
            cities: 7,
        },
    ];

    deck.sort_by_key(|p| p.number);

    PlantMarket {
        actual: Vec::new(),
        future: Vec::new(),
        deck,
        below_step3: None,
        step3_triggered: false,
        in_step3: false,
        discount_token: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::map::{CityData, ConnectionData, Map, MapData};
    use crate::state::give_player_cities;

    fn test_map() -> Map {
        Map::from_data(MapData {
            name: "Test".into(),
            width: 1.0,
            height: 1.0,
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

        // First player selects the second-cheapest plant in the market; second passes → first wins.
        // (Round 1 forbids passing a selection turn, but passing an active bid is always allowed.)
        let first_plant = state.market.actual[1].number;
        apply_action(
            &mut state,
            first,
            Action::SelectPlant {
                plant_number: first_plant,
            },
        )
        .unwrap();
        apply_action(&mut state, second, Action::PassAuction).unwrap();

        // Now only the second player remains for selection. Selecting a plant auto-awards it at minimum.
        // Capture the discount-aware expected cost BEFORE the purchase clears the token.
        let second_plant = state.market.actual[0].number;
        let expected_cost = effective_min_bid(&state.market, second_plant);
        apply_action(
            &mut state,
            second,
            Action::SelectPlant {
                plant_number: second_plant,
            },
        )
        .unwrap();

        // Should have advanced past auction into BuyResources.
        assert!(
            matches!(state.phase, Phase::BuyResources { .. }),
            "expected BuyResources after last player auto-wins plant, got {:?}",
            state.phase
        );

        // Second player should own the plant they selected, charged its effective minimum bid.
        // actual[0] carries the discount token so the cost is 1 Elektro, not the printed number.
        let second_player = state.player(second).unwrap();
        assert!(second_player
            .plants
            .iter()
            .any(|p| p.number == second_plant));
        assert_eq!(second_player.money, 50 - expected_cost);

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
        assert!(state_batch.player_owns_city(first_batch, "a"));
        assert!(state_single.player_owns_city(first_single, "a"));
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
        give_player_cities(&mut state, first, &["a"]);

        apply_action(
            &mut state,
            first,
            Action::BuildCities {
                city_ids: vec!["b".into(), "c".into()],
            },
        )
        .unwrap();

        assert!(state.player_owns_city(first, "b"));
        assert!(state.player_owns_city(first, "c"));
        // route costs: 5 + 3 = 8; slot costs: 10 + 10 = 20; total = 28.
        assert_eq!(state.player(first).unwrap().money, starting_money - 28);
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
        assert!(!state.player_owns_city(first, "a"));
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

        // A GasOrOil plant with cost=2 holds 4 total resources (gas + oil combined).
        let mut player = crate::types::Player::new("Test".into(), PlayerColor::Red);
        player.plants.push(PowerPlant {
            number: 5,
            kind: PlantKind::GasOrOil,
            cost: 2,
            cities: 1,
        });

        // Can buy up to 4 gas with 0 oil stored.
        assert!(player.can_add_resource(Resource::Gas, 4));
        // Cannot buy 5 gas — exceeds total slots.
        assert!(!player.can_add_resource(Resource::Gas, 5));

        // After storing 4 gas, cannot buy any oil.
        player.resources.gas = 4;
        assert!(!player.can_add_resource(Resource::Oil, 1));

        // After storing 2 gas, can only buy 2 more oil.
        player.resources.gas = 2;
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
        assert_eq!(coal_after_buy, 19, "expected 4 coal bought from initial 23");

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

    /// Set up a two-player game in the BuyResources phase.
    /// Both players have a coal plant (cost 2, capacity 4 coal) and plenty of money.
    fn two_player_buy_phase() -> (GameState, PlayerId, PlayerId) {
        use crate::types::{PlantKind, PowerPlant};
        let (mut state, p1, p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();
        state.phase = Phase::BuyResources {
            remaining: state.player_order.iter().rev().cloned().collect(),
        };
        for player in &mut state.players {
            player.money = 500;
            player.plants.push(PowerPlant {
                number: 4,
                kind: PlantKind::Coal,
                cost: 2,
                cities: 1,
            });
        }
        (state, p1, p2)
    }

    #[test]
    fn test_buy_resource_batch_valid() {
        let (mut state, _p1, _p2) = two_player_buy_phase();
        let first = match &state.phase {
            Phase::BuyResources { remaining } => *remaining.first().unwrap(),
            _ => unreachable!(),
        };
        let starting_money = state.player(first).unwrap().money;
        let expected_cost = state.resources.price(Resource::Coal, 2).unwrap();

        apply_action(
            &mut state,
            first,
            Action::BuyResourceBatch {
                purchases: vec![(Resource::Coal, 2)],
            },
        )
        .unwrap();

        let player = state.player(first).unwrap();
        assert_eq!(player.resources.coal, 2);
        assert_eq!(player.money, starting_money - expected_cost);
        // Turn should have advanced.
        match &state.phase {
            Phase::BuyResources { remaining } => {
                assert_ne!(remaining.first().copied(), Some(first));
            }
            Phase::BuildCities { .. } => {}
            other => panic!("unexpected phase: {other:?}"),
        }
    }

    #[test]
    fn test_buy_resource_batch_empty_advances_turn() {
        let (mut state, _p1, _p2) = two_player_buy_phase();
        let first = match &state.phase {
            Phase::BuyResources { remaining } => *remaining.first().unwrap(),
            _ => unreachable!(),
        };

        apply_action(
            &mut state,
            first,
            Action::BuyResourceBatch { purchases: vec![] },
        )
        .unwrap();

        // Turn advanced, first player no longer acting.
        match &state.phase {
            Phase::BuyResources { remaining } => {
                assert_ne!(remaining.first().copied(), Some(first));
            }
            Phase::BuildCities { .. } => {}
            other => panic!("unexpected phase: {other:?}"),
        }
    }

    #[test]
    fn test_buy_resource_batch_atomic_rollback() {
        let (mut state, _p1, _p2) = two_player_buy_phase();
        let first = match &state.phase {
            Phase::BuyResources { remaining } => *remaining.first().unwrap(),
            _ => unreachable!(),
        };
        // Give very little money — enough for some coal but not oil as well.
        state.player_mut(first).unwrap().money = 1;
        let coal_before = state.resources.coal;

        let err = apply_action(
            &mut state,
            first,
            Action::BuyResourceBatch {
                purchases: vec![(Resource::Coal, 1), (Resource::Coal, 1)],
            },
        )
        .unwrap_err();

        assert!(matches!(err, ActionError::CannotAfford), "got {err:?}");
        // State unchanged — market and player untouched.
        assert_eq!(state.resources.coal, coal_before);
        assert_eq!(state.player(first).unwrap().money, 1);
    }

    #[test]
    fn test_buy_resource_batch_over_capacity_rejected() {
        let (mut state, _p1, _p2) = two_player_buy_phase();
        let first = match &state.phase {
            Phase::BuyResources { remaining } => *remaining.first().unwrap(),
            _ => unreachable!(),
        };
        // Plant cost=2, capacity=4 coal. Trying to buy 5 should fail.
        let err = apply_action(
            &mut state,
            first,
            Action::BuyResourceBatch {
                purchases: vec![(Resource::Coal, 5)],
            },
        )
        .unwrap_err();
        assert!(matches!(err, ActionError::OverCapacity), "got {err:?}");
    }

    /// Player owns plants 8 (Coal/3/2), 10 (Coal/2/2), and 21 (CoalOrOil/2/4) with only
    /// 2 coal.  The player (or bot via `optimal_firing_subset`) selects plant 21 alone, which
    /// fires on 2 coal and powers 4 cities.  Submitting all three would be infeasible.
    #[test]
    fn test_bureaucracy_optimal_gas_or_oil_plant() {
        use crate::types::{PlantKind, PowerPlant};

        let (mut state, p1, _p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        // Force bureaucracy phase with p1 acting first.
        state.phase = Phase::Bureaucracy {
            remaining: vec![p1],
        };

        // Give 4 cities (test map only has a/b/c; insert d directly).
        give_player_cities(&mut state, p1, &["a", "b", "c"]);
        state.map.cities.insert(
            "d".into(),
            crate::map::City {
                id: "d".into(),
                name: "D".into(),
                region: "r1".into(),
                owners: vec![p1],
                x: None,
                y: None,
            },
        );
        let player = state.player_mut(p1).unwrap();
        // Plants: 8 (Coal, cost 3, 2 cities), 10 (Coal, cost 2, 2 cities), 21 (GasOrOil, cost 2, 4 cities).
        player.plants = vec![
            PowerPlant {
                number: 8,
                kind: PlantKind::Coal,
                cost: 3,
                cities: 2,
            },
            PowerPlant {
                number: 10,
                kind: PlantKind::Coal,
                cost: 2,
                cities: 2,
            },
            PowerPlant {
                number: 21,
                kind: PlantKind::GasOrOil,
                cost: 2,
                cities: 4,
            },
        ];
        // 2 gas — enough for plant 21 (GasOrOil, cost 2), but coal plants can't fire (0 coal).
        player.resources = PlayerResources {
            coal: 0,
            oil: 0,
            gas: 2,
            uranium: 0,
        };

        // Player selects only plant 21 (fires on gas).
        apply_action(
            &mut state,
            p1,
            Action::PowerCities {
                plant_numbers: vec![21],
            },
        )
        .unwrap();

        // Plant 21 fired: 4 cities powered.
        let found = state
            .event_log
            .iter()
            .any(|e| e.contains("powered 4 cities"));
        assert!(
            found,
            "expected a log entry with 'powered 4 cities'; log: {:?}",
            state.event_log
        );
        // Resources: plant 21 consumed 2 gas.
        let player = state.player(p1).unwrap();
        assert_eq!(player.resources.gas, 0, "2 gas should have been consumed");
    }

    /// Player explicitly selects only the wind plant to conserve coal even though
    /// the coal plant could also fire.  Income is capped at cities owned.
    #[test]
    fn test_bureaucracy_caps_at_cities_owned() {
        use crate::types::{PlantKind, PowerPlant};

        let (mut state, p1, _p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        state.phase = Phase::Bureaucracy {
            remaining: vec![p1],
        };

        // Only 2 cities owned.
        give_player_cities(&mut state, p1, &["a", "b"]);
        let player = state.player_mut(p1).unwrap();
        // Wind plant (free, 2 cities) + Coal plant (cost 2, 2 cities).
        player.plants = vec![
            PowerPlant {
                number: 13,
                kind: PlantKind::Wind,
                cost: 0,
                cities: 2,
            },
            PowerPlant {
                number: 10,
                kind: PlantKind::Coal,
                cost: 2,
                cities: 2,
            },
        ];
        player.resources = PlayerResources {
            coal: 2,
            oil: 0,
            gas: 0,
            uranium: 0,
        };

        // Player selects only the wind plant — coal plant is excluded to conserve resources.
        apply_action(
            &mut state,
            p1,
            Action::PowerCities {
                plant_numbers: vec![13],
            },
        )
        .unwrap();

        // Wind powers 2 cities (= cities owned); coal was not submitted so it stays.
        let player = state.player(p1).unwrap();
        assert_eq!(
            player.resources.coal, 2,
            "coal should be conserved when wind covers all cities; got {}",
            player.resources.coal
        );
    }

    /// Submitting a plant combination that exceeds available resources returns InfeasiblePlantSelection.
    #[test]
    fn test_bureaucracy_rejects_infeasible_selection() {
        use crate::types::{PlantKind, PowerPlant};

        let (mut state, p1, _p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        state.phase = Phase::Bureaucracy {
            remaining: vec![p1],
        };

        give_player_cities(&mut state, p1, &["a"]);
        let player = state.player_mut(p1).unwrap();
        player.plants = vec![PowerPlant {
            number: 10,
            kind: PlantKind::Coal,
            cost: 2,
            cities: 1,
        }];
        player.resources = PlayerResources {
            coal: 1, // only 1 coal but plant costs 2
            oil: 0,
            gas: 0,
            uranium: 0,
        };

        let err = apply_action(
            &mut state,
            p1,
            Action::PowerCities {
                plant_numbers: vec![10],
            },
        )
        .unwrap_err();
        assert!(
            matches!(err, ActionError::InfeasiblePlantSelection),
            "expected InfeasiblePlantSelection, got {err:?}"
        );
    }

    /// Submitting more plant capacity than cities owned burns all selected fuel but
    /// income is capped at the number of cities owned.
    #[test]
    fn test_bureaucracy_overfire_burns_fuel() {
        use crate::types::{PlantKind, PowerPlant};

        let (mut state, p1, _p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        state.phase = Phase::Bureaucracy {
            remaining: vec![p1],
        };

        // Only 2 cities owned.
        give_player_cities(&mut state, p1, &["a", "b"]);
        let player = state.player_mut(p1).unwrap();
        // Three plants each powering 1 city.
        player.plants = vec![
            PowerPlant {
                number: 5,
                kind: PlantKind::Coal,
                cost: 1,
                cities: 1,
            },
            PowerPlant {
                number: 6,
                kind: PlantKind::Coal,
                cost: 1,
                cities: 1,
            },
            PowerPlant {
                number: 7,
                kind: PlantKind::Coal,
                cost: 1,
                cities: 1,
            },
        ];
        player.resources = PlayerResources {
            coal: 3,
            oil: 0,
            gas: 0,
            uranium: 0,
        };

        // Player fires all three (over-fire: 3 capacity for 2 cities).
        apply_action(
            &mut state,
            p1,
            Action::PowerCities {
                plant_numbers: vec![5, 6, 7],
            },
        )
        .unwrap();

        let player = state.player(p1).unwrap();
        // All 3 coal are burned (over-fire).
        assert_eq!(player.resources.coal, 0, "all 3 coal should be consumed");
        // Income capped at 2 cities owned.
        assert_eq!(
            player.last_cities_powered, 2,
            "cities powered should be capped at 2"
        );
    }

    /// Player explicitly skips a powerable plant to conserve its fuel.
    #[test]
    fn test_bureaucracy_skip_plant_for_fuel_conservation() {
        use crate::types::{PlantKind, PowerPlant};

        let (mut state, p1, _p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        state.phase = Phase::Bureaucracy {
            remaining: vec![p1],
        };

        give_player_cities(&mut state, p1, &["a"]);
        let player = state.player_mut(p1).unwrap();
        // Wind (free, 1 city) and Coal (cost 2, 1 city).
        player.plants = vec![
            PowerPlant {
                number: 13,
                kind: PlantKind::Wind,
                cost: 0,
                cities: 1,
            },
            PowerPlant {
                number: 10,
                kind: PlantKind::Coal,
                cost: 2,
                cities: 1,
            },
        ];
        player.resources = PlayerResources {
            coal: 2,
            oil: 0,
            gas: 0,
            uranium: 0,
        };

        // Player submits only the wind plant — skipping coal to save it for later.
        apply_action(
            &mut state,
            p1,
            Action::PowerCities {
                plant_numbers: vec![13],
            },
        )
        .unwrap();

        let player = state.player(p1).unwrap();
        assert_eq!(player.resources.coal, 2, "coal should be untouched");
        assert_eq!(player.last_cities_powered, 1, "1 city powered via wind");
    }

    /// Players can submit PowerCities in any order during Bureaucracy.
    #[test]
    fn test_bureaucracy_out_of_order_submission() {
        use crate::types::{PlantKind, PowerPlant};

        let (mut state, p1, p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        // Force bureaucracy with p1 listed first, but have p2 act first.
        state.phase = Phase::Bureaucracy {
            remaining: vec![p1, p2],
        };

        // Give both players a wind plant and one city.
        for &pid in &[p1, p2] {
            give_player_cities(&mut state, pid, &["a"]);
            let player = state.player_mut(pid).unwrap();
            player.plants = vec![PowerPlant {
                number: 13,
                kind: PlantKind::Wind,
                cost: 0,
                cities: 2,
            }];
            player.resources = PlayerResources {
                coal: 0,
                oil: 0,
                gas: 0,
                uranium: 0,
            };
        }

        // p2 acts before p1 — should succeed.
        apply_action(
            &mut state,
            p2,
            Action::PowerCities {
                plant_numbers: vec![13],
            },
        )
        .unwrap();

        // Phase still has p1 remaining.
        assert!(matches!(&state.phase, Phase::Bureaucracy { remaining } if remaining == &vec![p1]));

        // p1 now acts.
        apply_action(
            &mut state,
            p1,
            Action::PowerCities {
                plant_numbers: vec![13],
            },
        )
        .unwrap();

        // Both players submitted — round should have ended.
        assert!(
            !matches!(state.phase, Phase::Bureaucracy { .. }),
            "expected phase to advance after both players submitted"
        );
    }

    /// Regression: buying a 4th plant (which discards the lowest) must not leave orphaned
    /// resources from the discarded plant blocking future purchases.
    ///
    /// Setup: player has 3 plants — a CoalOrOil hybrid (cost 2, cap 4) with 4 oil stored,
    /// a Coal plant (cost 2, cap 4), and a Coal plant (cost 3, cap 6).
    /// They then win a Coal plant (cost 4, cap 8) as their 4th plant.
    /// The lowest-numbered plant (the hybrid) is discarded; its 4 oil must be returned to
    /// the market so that `can_add_resource(Coal, 1)` succeeds on the remaining plants.
    #[test]
    fn test_fourth_plant_orphaned_resources_returned_to_market() {
        use crate::types::{PlantKind, PowerPlant};

        let (mut state, p1, _p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        // Give the player 3 plants. Plant 5 (hybrid, lowest number) holds 4 oil.
        let player = state.player_mut(p1).unwrap();
        player.money = 1000;
        player.plants = vec![
            PowerPlant {
                number: 5,
                kind: PlantKind::GasOrOil,
                cost: 2,
                cities: 1,
            }, // cap 4
            PowerPlant {
                number: 10,
                kind: PlantKind::Coal,
                cost: 2,
                cities: 1,
            }, // cap 4
            PowerPlant {
                number: 14,
                kind: PlantKind::Coal,
                cost: 3,
                cities: 2,
            }, // cap 6
        ];
        player.resources = PlayerResources {
            coal: 0,
            oil: 4,
            gas: 0,
            uranium: 0,
        };

        // Add a coal plant to the actual market (plant 24 doesn't normally appear in round 1,
        // so inject it directly).
        let new_plant = PowerPlant {
            number: 24,
            kind: PlantKind::Coal,
            cost: 4,
            cities: 3,
        };
        state.market.actual.push(new_plant);

        let oil_before = state.resources.oil;

        // Award the 4th plant — should enter DiscardPlant phase.
        award_plant(&mut state, p1, 24, 24, vec![], vec![]).unwrap();

        // Verify we're in DiscardPlant phase waiting for the player.
        assert!(
            matches!(state.phase, Phase::DiscardPlant { player, .. } if player == p1),
            "should be in DiscardPlant phase"
        );

        // Player chooses to discard plant 5 (the hybrid).
        apply_action(&mut state, p1, Action::DiscardPlant { plant_number: 5 }).unwrap();

        let player = state.player(p1).unwrap();

        // Plant 5 (hybrid) should have been discarded; player should now have plants 10, 14, 24.
        let plant_numbers: Vec<u8> = player.plants.iter().map(|p| p.number).collect();
        assert_eq!(
            plant_numbers,
            vec![10, 14, 24],
            "plant 5 should have been discarded"
        );

        // The 4 oil that lived on the hybrid must have been returned to the market.
        assert_eq!(
            state.resources.oil,
            oil_before + 4,
            "orphaned oil should be returned to market"
        );
        assert_eq!(
            player.resources.oil, 0,
            "player should hold no oil after discard"
        );

        // Critical: player must be able to add coal to their remaining coal plants.
        assert!(
            player.can_add_resource(Resource::Coal, 1),
            "can_add_resource(Coal) must succeed after hybrid plant is discarded"
        );
    }

    #[test]
    fn test_discard_plant_choice_non_lowest() {
        use crate::types::{PlantKind, PowerPlant};

        let (mut state, p1, _p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        let player = state.player_mut(p1).unwrap();
        player.money = 1000;
        player.plants = vec![
            PowerPlant {
                number: 5,
                kind: PlantKind::Coal,
                cost: 2,
                cities: 1,
            },
            PowerPlant {
                number: 10,
                kind: PlantKind::Coal,
                cost: 2,
                cities: 1,
            },
            PowerPlant {
                number: 14,
                kind: PlantKind::Coal,
                cost: 3,
                cities: 2,
            },
        ];

        let new_plant = PowerPlant {
            number: 24,
            kind: PlantKind::Coal,
            cost: 4,
            cities: 3,
        };
        state.market.actual.push(new_plant);

        award_plant(&mut state, p1, 24, 24, vec![], vec![]).unwrap();

        assert!(
            matches!(state.phase, Phase::DiscardPlant { player, .. } if player == p1),
            "should be in DiscardPlant phase"
        );

        // Player chooses to discard plant 10 (not the lowest).
        apply_action(&mut state, p1, Action::DiscardPlant { plant_number: 10 }).unwrap();

        let plant_numbers: Vec<u8> = state
            .player(p1)
            .unwrap()
            .plants
            .iter()
            .map(|p| p.number)
            .collect();
        assert_eq!(
            plant_numbers,
            vec![5, 14, 24],
            "plant 10 should have been discarded"
        );
    }

    #[test]
    fn test_discard_plant_wrong_player_rejected() {
        use crate::types::{PlantKind, PowerPlant};

        let (mut state, p1, p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        let player = state.player_mut(p1).unwrap();
        player.money = 1000;
        player.plants = vec![
            PowerPlant {
                number: 5,
                kind: PlantKind::Coal,
                cost: 2,
                cities: 1,
            },
            PowerPlant {
                number: 10,
                kind: PlantKind::Coal,
                cost: 2,
                cities: 1,
            },
            PowerPlant {
                number: 14,
                kind: PlantKind::Coal,
                cost: 3,
                cities: 2,
            },
        ];
        let new_plant = PowerPlant {
            number: 24,
            kind: PlantKind::Coal,
            cost: 4,
            cities: 3,
        };
        state.market.actual.push(new_plant);

        award_plant(&mut state, p1, 24, 24, vec![], vec![]).unwrap();

        let result = apply_action(&mut state, p2, Action::DiscardPlant { plant_number: 5 });
        assert!(matches!(result, Err(ActionError::NotYourTurn)));
    }

    #[test]
    fn test_discard_new_plant_rejected() {
        use crate::types::{PlantKind, PowerPlant};

        let (mut state, p1, _p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        let player = state.player_mut(p1).unwrap();
        player.money = 1000;
        player.plants = vec![
            PowerPlant {
                number: 5,
                kind: PlantKind::Coal,
                cost: 2,
                cities: 1,
            },
            PowerPlant {
                number: 10,
                kind: PlantKind::Coal,
                cost: 2,
                cities: 1,
            },
            PowerPlant {
                number: 14,
                kind: PlantKind::Coal,
                cost: 3,
                cities: 2,
            },
        ];
        let new_plant = PowerPlant {
            number: 24,
            kind: PlantKind::Coal,
            cost: 4,
            cities: 3,
        };
        state.market.actual.push(new_plant);

        award_plant(&mut state, p1, 24, 24, vec![], vec![]).unwrap();

        // Try to discard the newly won plant — should be rejected.
        let result = apply_action(&mut state, p1, Action::DiscardPlant { plant_number: 24 });
        assert!(matches!(result, Err(ActionError::CannotDiscardNewPlant)));
    }

    #[test]
    fn test_discard_unowned_plant_rejected() {
        use crate::types::{PlantKind, PowerPlant};

        let (mut state, p1, _p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        let player = state.player_mut(p1).unwrap();
        player.money = 1000;
        player.plants = vec![
            PowerPlant {
                number: 5,
                kind: PlantKind::Coal,
                cost: 2,
                cities: 1,
            },
            PowerPlant {
                number: 10,
                kind: PlantKind::Coal,
                cost: 2,
                cities: 1,
            },
            PowerPlant {
                number: 14,
                kind: PlantKind::Coal,
                cost: 3,
                cities: 2,
            },
        ];
        let new_plant = PowerPlant {
            number: 24,
            kind: PlantKind::Coal,
            cost: 4,
            cities: 3,
        };
        state.market.actual.push(new_plant);

        award_plant(&mut state, p1, 24, 24, vec![], vec![]).unwrap();

        // Try to discard a plant the player doesn't own.
        let result = apply_action(&mut state, p1, Action::DiscardPlant { plant_number: 99 });
        assert!(matches!(result, Err(ActionError::PlantNotOwned(99))));
    }

    // -----------------------------------------------------------------------
    // Region selection tests
    // -----------------------------------------------------------------------

    /// Build a map with multiple regions where cities connect across region boundaries.
    fn multi_region_map() -> Map {
        Map::from_data(MapData {
            name: "MultiRegion".into(),
            width: 1.0,
            height: 1.0,
            regions: vec![
                "r1".into(),
                "r2".into(),
                "r3".into(),
                "r4".into(),
                "r5".into(),
                "r6".into(),
            ],
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
                    region: "r2".into(),
                    x: None,
                    y: None,
                },
                CityData {
                    id: "c".into(),
                    name: "C".into(),
                    region: "r3".into(),
                    x: None,
                    y: None,
                },
                CityData {
                    id: "d".into(),
                    name: "D".into(),
                    region: "r4".into(),
                    x: None,
                    y: None,
                },
                CityData {
                    id: "e".into(),
                    name: "E".into(),
                    region: "r5".into(),
                    x: None,
                    y: None,
                },
                CityData {
                    id: "f".into(),
                    name: "F".into(),
                    region: "r6".into(),
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
                ConnectionData {
                    from: "c".into(),
                    to: "d".into(),
                    cost: 4,
                },
                ConnectionData {
                    from: "d".into(),
                    to: "e".into(),
                    cost: 2,
                },
                ConnectionData {
                    from: "e".into(),
                    to: "f".into(),
                    cost: 6,
                },
            ],
        })
    }

    fn start_multi_region_game(player_count: usize, seed: u64) -> (GameState, Vec<PlayerId>) {
        let colors = [
            PlayerColor::Red,
            PlayerColor::Blue,
            PlayerColor::Yellow,
            PlayerColor::Green,
            PlayerColor::Purple,
            PlayerColor::White,
        ];
        let names = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"];
        let mut state = GameState::new_with_seed(multi_region_map(), player_count, seed);
        let mut ids = Vec::new();
        for i in 0..player_count {
            let id = uuid::Uuid::new_v4();
            ids.push(id);
            apply_action(
                &mut state,
                id,
                Action::JoinGame {
                    name: names[i].into(),
                    color: colors[i],
                },
            )
            .unwrap();
        }
        apply_action(&mut state, ids[0], Action::StartGame).unwrap();
        (state, ids)
    }

    #[test]
    fn test_region_count_by_player_count() {
        let cases = [(2, 3), (3, 3), (4, 4), (5, 5), (6, 5)];
        for (player_count, expected_regions) in cases {
            let (state, _) = start_multi_region_game(player_count, 42);
            assert_eq!(
                state.active_regions.len(),
                expected_regions,
                "expected {} active regions for {} players",
                expected_regions,
                player_count
            );
        }
    }

    #[test]
    fn test_region_selection_deterministic_with_seed() {
        let (state1, _) = start_multi_region_game(4, 99);
        let (state2, _) = start_multi_region_game(4, 99);
        assert_eq!(
            state1.active_regions, state2.active_regions,
            "same seed should produce same active regions"
        );
    }

    #[test]
    fn test_build_in_inactive_region_rejected() {
        let (mut state, _ids) = start_multi_region_game(2, 42);
        // Find a city in an inactive region.
        let inactive_city = state
            .map
            .cities
            .values()
            .find(|c| !state.active_regions.contains(&c.region))
            .map(|c| c.id.clone())
            .expect("there should be inactive cities with 2 players (3 of 6 regions inactive)");

        // Give the player money and force into BuildCities.
        for player in &mut state.players {
            player.money = 500;
        }
        let build_order: Vec<PlayerId> = state.player_order.iter().rev().cloned().collect();
        state.phase = Phase::BuildCities {
            remaining: build_order,
        };
        let actor = match &state.phase {
            Phase::BuildCities { remaining } => remaining[0],
            _ => unreachable!(),
        };

        let result = apply_action(
            &mut state,
            actor,
            Action::BuildCities {
                city_ids: vec![inactive_city.clone()],
            },
        );
        assert!(
            matches!(result, Err(ActionError::CityRegionInactive(_))),
            "building in inactive region should return CityRegionInactive, got {:?}",
            result
        );
    }

    #[test]
    fn test_routing_through_inactive_region_still_works() {
        // Map: a(r1) --5-- b(r2) --3-- c(r3)
        // If r2 is inactive, routing from a to c should still cost 8 (through b).
        let mut state = GameState::new_with_seed(multi_region_map(), 2, 1);
        // Force active_regions to r1 and r3 only (making r2 inactive).
        state.active_regions = vec!["r1".into(), "r3".into(), "r4".into()];

        // Cost from "a" (r1) to "c" (r3) should traverse through "b" (r2, inactive).
        let cost = state.map.connection_cost_to(&["a".to_string()], "c");
        assert_eq!(
            cost,
            Some(8),
            "routing cost through inactive city should still be computed (a->b=5, b->c=3)"
        );
    }

    /// A CoalOrOil hybrid plant must not steal coal from a downstream pure-Coal
    /// plant.  With plants 20 (Coal/3/5), 29 (CoalOrOil/1/4), 42 (Coal/2/6)
    /// and resources 5 coal + 2 oil, all three should fire (15 cities).
    #[test]
    fn test_hybrid_plant_does_not_starve_pure_coal() {
        use crate::types::{PlantKind, PowerPlant};

        let (mut state, p1, _p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        state.phase = Phase::Bureaucracy {
            remaining: vec![p1],
        };

        // Give p1 fifteen cities so PowerCities can cap at 15.
        for i in 0..15 {
            let city_id = format!("city{i}");
            state.map.cities.insert(
                city_id.clone(),
                crate::map::City {
                    id: city_id.clone(),
                    name: city_id.clone(),
                    region: "r1".into(),
                    owners: vec![p1],
                    x: None,
                    y: None,
                },
            );
        }
        let player = state.player_mut(p1).unwrap();
        player.plants = vec![
            PowerPlant {
                number: 20,
                kind: PlantKind::Coal,
                cost: 3,
                cities: 5,
            },
            PowerPlant {
                number: 29,
                kind: PlantKind::GasOrOil,
                cost: 1,
                cities: 4,
            },
            PowerPlant {
                number: 42,
                kind: PlantKind::Coal,
                cost: 2,
                cities: 6,
            },
        ];
        player.resources = PlayerResources {
            coal: 5,
            oil: 2,
            gas: 0,
            uranium: 0,
        };

        apply_action(
            &mut state,
            p1,
            Action::PowerCities {
                plant_numbers: vec![20, 29, 42],
            },
        )
        .unwrap();

        let player = state.player(p1).unwrap();
        assert_eq!(
            player.last_cities_powered, 15,
            "all 15 cities should be powered; got {}",
            player.last_cities_powered
        );
        // Plant 20 (3 coal) + Plant 42 (2 coal) + Plant 29 (1 oil): 0 coal, 1 oil left.
        assert_eq!(player.resources.coal, 0, "expected 0 coal remaining");
        assert_eq!(player.resources.oil, 1, "expected 1 oil remaining");
    }

    /// Hybrids should prefer oil over coal so that coal is preserved for
    /// pure-Coal plants when resources are tight.
    #[test]
    fn test_hybrid_prefers_oil_to_conserve_gas() {
        use crate::types::{PlantKind, PowerPlant};

        let (mut state, p1, _p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        state.phase = Phase::Bureaucracy {
            remaining: vec![p1],
        };

        // Give p1 six cities so all three plants can fire at capacity.
        for i in 0..6 {
            let city_id = format!("c{i}");
            state.map.cities.insert(
                city_id.clone(),
                crate::map::City {
                    id: city_id.clone(),
                    name: city_id.clone(),
                    region: "r1".into(),
                    owners: vec![p1],
                    x: None,
                    y: None,
                },
            );
        }
        let player = state.player_mut(p1).unwrap();
        // GasOrOil hybrid (cost 2) + GasOrOil hybrid (cost 1) + Coal (cost 2).
        // Resources: coal=2, gas=3, oil=2.
        // Coal plant needs 2 coal. Hybrid total cost=3; oil=2, gas=3 → split is ambiguous
        // (min_gas=1, max_gas=3). Player submits gas=1,oil=2 to preserve gas.
        player.plants = vec![
            PowerPlant {
                number: 5,
                kind: PlantKind::GasOrOil,
                cost: 2,
                cities: 2,
            },
            PowerPlant {
                number: 8,
                kind: PlantKind::GasOrOil,
                cost: 1,
                cities: 2,
            },
            PowerPlant {
                number: 10,
                kind: PlantKind::Coal,
                cost: 2,
                cities: 2,
            },
        ];
        player.resources = PlayerResources {
            coal: 2,
            oil: 2,
            gas: 3,
            uranium: 0,
        };

        // Submit plant selection — enters PowerCitiesFuel because split is ambiguous.
        apply_action(
            &mut state,
            p1,
            Action::PowerCities {
                plant_numbers: vec![5, 8, 10],
            },
        )
        .unwrap();
        assert!(
            matches!(state.phase, Phase::PowerCitiesFuel { .. }),
            "expected PowerCitiesFuel after ambiguous hybrid selection"
        );

        // Choose the split that preserves gas: use minimum gas (1), maximum oil (2).
        apply_action(&mut state, p1, Action::PowerCitiesFuel { gas: 1, oil: 2 }).unwrap();

        let player = state.player(p1).unwrap();
        assert_eq!(
            player.last_cities_powered, 6,
            "all 6 cities should be powered; got {}",
            player.last_cities_powered
        );
        assert_eq!(player.resources.coal, 0, "expected 0 coal remaining");
        assert_eq!(player.resources.oil, 0, "expected 0 oil remaining");
        assert_eq!(
            player.resources.gas, 2,
            "expected 2 gas remaining (gas conserved)"
        );
    }

    // -----------------------------------------------------------------------
    // DiscardResource phase tests
    // -----------------------------------------------------------------------

    /// Discarding a pure-gas plant while two GasOrOil hybrids remain triggers Phase::DiscardResource
    /// when the player holds both gas and oil that jointly overflow the hybrid slots.
    ///
    /// Setup: hybrid #5 (cap 4) + pure-gas #3 (cap 4) + hybrid #7 (cap 6).
    /// Held: gas=6, oil=8 (fits: gas 4 in #3, 2+8=10 in hybrid ≤ 10).
    /// Win Uranium #24 (adds no gas/oil cap).  Discard pure-gas #3.
    /// After: plants #5, #7, #24.  gas_only=0, oil_only=0, hybrid=10.
    /// Per-resource OK (6≤10, 8≤10) but shared: 6+8=14 > 10.  drop_total = 4.
    #[test]
    fn test_discard_plant_triggers_resource_prompt_on_shared_slot_overflow() {
        use crate::types::{PlantKind, PowerPlant};

        let (mut state, p1, _p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        let player = state.player_mut(p1).unwrap();
        player.money = 1000;
        player.plants = vec![
            PowerPlant {
                number: 3,
                kind: PlantKind::Gas,
                cost: 2,
                cities: 1,
            }, // pure-gas cap 4
            PowerPlant {
                number: 5,
                kind: PlantKind::GasOrOil,
                cost: 2,
                cities: 1,
            }, // hybrid cap 4
            PowerPlant {
                number: 7,
                kind: PlantKind::GasOrOil,
                cost: 3,
                cities: 2,
            }, // hybrid cap 6
        ];
        player.resources = PlayerResources {
            coal: 0,
            oil: 8,
            gas: 6,
            uranium: 0,
        };

        // Use a Uranium plant so no gas/oil capacity is added — overflow stays ambiguous.
        let new_plant = PowerPlant {
            number: 24,
            kind: PlantKind::Uranium,
            cost: 1,
            cities: 3,
        };
        state.market.actual.push(new_plant);

        award_plant(&mut state, p1, 24, 24, vec![], vec![]).unwrap();
        assert!(
            matches!(state.phase, Phase::DiscardPlant { player, .. } if player == p1),
            "should be in DiscardPlant phase"
        );

        // Discard the pure-gas plant — triggers shared-slot overflow.
        apply_action(&mut state, p1, Action::DiscardPlant { plant_number: 3 }).unwrap();

        assert!(
            matches!(state.phase, Phase::DiscardResource { player, drop_total, .. } if player == p1 && drop_total == 4),
            "expected Phase::DiscardResource with drop_total=4, got {:?}",
            state.phase
        );

        // Resources must be untouched at this point.
        let player = state.player(p1).unwrap();
        assert_eq!(player.resources.gas, 6);
        assert_eq!(player.resources.oil, 8);
    }

    /// After Phase::DiscardResource is set, a valid split is accepted and the phase advances.
    #[test]
    fn test_discard_resource_applies_split() {
        use crate::types::{PlantKind, PowerPlant};

        let (mut state, p1, _p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        let player = state.player_mut(p1).unwrap();
        player.money = 1000;
        player.plants = vec![
            PowerPlant {
                number: 3,
                kind: PlantKind::Gas,
                cost: 2,
                cities: 1,
            },
            PowerPlant {
                number: 5,
                kind: PlantKind::GasOrOil,
                cost: 2,
                cities: 1,
            },
            PowerPlant {
                number: 7,
                kind: PlantKind::GasOrOil,
                cost: 3,
                cities: 2,
            },
        ];
        player.resources = PlayerResources {
            coal: 0,
            oil: 8,
            gas: 6,
            uranium: 0,
        };

        let new_plant = PowerPlant {
            number: 24,
            kind: PlantKind::Uranium,
            cost: 1,
            cities: 3,
        };
        state.market.actual.push(new_plant);

        // Pre-deplete market so replenish has room to add resources.
        state.resources.gas = 2;
        state.resources.oil = 14;

        award_plant(&mut state, p1, 24, 24, vec![], vec![]).unwrap();
        apply_action(&mut state, p1, Action::DiscardPlant { plant_number: 3 }).unwrap();

        assert!(matches!(state.phase, Phase::DiscardResource { .. }));

        let gas_market_before = state.resources.gas;
        let oil_market_before = state.resources.oil;

        // Drop 2 gas + 2 oil (sum = drop_total 4).
        apply_action(&mut state, p1, Action::DiscardResource { gas: 2, oil: 2 }).unwrap();

        // Phase must have advanced out of DiscardResource.
        assert!(
            !matches!(state.phase, Phase::DiscardResource { .. }),
            "phase should have advanced; got {:?}",
            state.phase
        );

        let player = state.player(p1).unwrap();
        assert_eq!(player.resources.gas, 4, "expected 4 gas remaining");
        assert_eq!(player.resources.oil, 6, "expected 6 oil remaining");
        assert_eq!(state.resources.gas, gas_market_before + 2);
        assert_eq!(state.resources.oil, oil_market_before + 2);
    }

    /// Sending an incorrect sum is rejected with InvalidDiscardSplit.
    #[test]
    fn test_discard_resource_rejects_wrong_total() {
        use crate::types::{PlantKind, PowerPlant};

        let (mut state, p1, _p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        let player = state.player_mut(p1).unwrap();
        player.money = 1000;
        player.plants = vec![
            PowerPlant {
                number: 3,
                kind: PlantKind::Gas,
                cost: 2,
                cities: 1,
            },
            PowerPlant {
                number: 5,
                kind: PlantKind::GasOrOil,
                cost: 2,
                cities: 1,
            },
            PowerPlant {
                number: 7,
                kind: PlantKind::GasOrOil,
                cost: 3,
                cities: 2,
            },
        ];
        player.resources = PlayerResources {
            coal: 0,
            oil: 8,
            gas: 6,
            uranium: 0,
        };

        let new_plant = PowerPlant {
            number: 24,
            kind: PlantKind::Uranium,
            cost: 1,
            cities: 3,
        };
        state.market.actual.push(new_plant);
        award_plant(&mut state, p1, 24, 24, vec![], vec![]).unwrap();
        apply_action(&mut state, p1, Action::DiscardPlant { plant_number: 3 }).unwrap();
        assert!(matches!(state.phase, Phase::DiscardResource { .. }));

        // gas(1) + oil(2) = 3, but drop_total == 4 → rejected.
        let result = apply_action(&mut state, p1, Action::DiscardResource { gas: 1, oil: 2 });
        assert!(
            matches!(result, Err(ActionError::InvalidDiscardSplit)),
            "expected InvalidDiscardSplit"
        );
        // Phase and resources unchanged.
        assert!(matches!(state.phase, Phase::DiscardResource { .. }));
        assert_eq!(state.player(p1).unwrap().resources.gas, 6);
    }

    /// Trying to drop more of a resource than the player holds is rejected.
    #[test]
    fn test_discard_resource_rejects_more_than_held() {
        use crate::types::{PlantKind, PowerPlant};

        let (mut state, p1, _p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        let player = state.player_mut(p1).unwrap();
        player.money = 1000;
        player.plants = vec![
            PowerPlant {
                number: 3,
                kind: PlantKind::Gas,
                cost: 2,
                cities: 1,
            },
            PowerPlant {
                number: 5,
                kind: PlantKind::GasOrOil,
                cost: 2,
                cities: 1,
            },
            PowerPlant {
                number: 7,
                kind: PlantKind::GasOrOil,
                cost: 3,
                cities: 2,
            },
        ];
        player.resources = PlayerResources {
            coal: 0,
            oil: 8,
            gas: 6,
            uranium: 0,
        };

        let new_plant = PowerPlant {
            number: 24,
            kind: PlantKind::Uranium,
            cost: 1,
            cities: 3,
        };
        state.market.actual.push(new_plant);
        award_plant(&mut state, p1, 24, 24, vec![], vec![]).unwrap();
        apply_action(&mut state, p1, Action::DiscardPlant { plant_number: 3 }).unwrap();
        assert!(matches!(state.phase, Phase::DiscardResource { .. }));

        // Asking for 7 gas but only holds 6 — rejected even though sum == 4 (7 > 6).
        let result = apply_action(&mut state, p1, Action::DiscardResource { gas: 7, oil: 0 });
        assert!(matches!(result, Err(ActionError::InvalidDiscardSplit)));
    }

    /// When discarding a plant leaves only one type of fuel over capacity, the per-resource
    /// clamp handles it automatically without entering Phase::DiscardResource.
    ///
    /// Setup: Oil #3 (cap 6) + hybrid #5 (cap 4) + hybrid #7 (cap 6). Hold 0 coal + 12 oil.
    /// Win Garbage #24. Discard Oil #3.
    /// After: plants #5, #7, #24. oil_cap = 10. Per-resource: drop 2 oil.
    /// Shared-slot: coal=0, oil=10, overflow=0. Phase advances directly.
    #[test]
    fn test_discard_plant_per_resource_overflow_auto_clamped() {
        use crate::types::{PlantKind, PowerPlant};

        let (mut state, p1, _p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        let player = state.player_mut(p1).unwrap();
        player.money = 1000;
        player.plants = vec![
            PowerPlant {
                number: 3,
                kind: PlantKind::Oil,
                cost: 3,
                cities: 1,
            }, // pure-oil cap 6
            PowerPlant {
                number: 5,
                kind: PlantKind::GasOrOil,
                cost: 2,
                cities: 1,
            }, // hybrid cap 4
            PowerPlant {
                number: 7,
                kind: PlantKind::GasOrOil,
                cost: 3,
                cities: 2,
            }, // hybrid cap 6
        ];
        // 12 oil, 0 coal. Fits: oil_only=6, hybrid=10, oil_cap=16. Shared: 0+6=6 ≤ 10 ✓.
        player.resources = PlayerResources {
            coal: 0,
            oil: 12,
            gas: 0,
            uranium: 0,
        };

        let new_plant = PowerPlant {
            number: 24,
            kind: PlantKind::Gas,
            cost: 4,
            cities: 3,
        };
        state.market.actual.push(new_plant);
        award_plant(&mut state, p1, 24, 24, vec![], vec![]).unwrap();
        apply_action(&mut state, p1, Action::DiscardPlant { plant_number: 3 }).unwrap();

        // Per-resource clamp drops 2 oil (oil_cap=10). No shared-slot overflow remains.
        // Phase should advance without pausing in DiscardResource.
        assert!(
            !matches!(state.phase, Phase::DiscardResource { .. }),
            "expected phase to advance without DiscardResource prompt; got {:?}",
            state.phase
        );

        let player = state.player(p1).unwrap();
        assert_eq!(
            player.resources.oil, 10,
            "expected 10 oil after per-resource clamp"
        );
        assert_eq!(player.resources.coal, 0);
    }

    // -----------------------------------------------------------------------
    // PowerCitiesFuel phase tests
    // -----------------------------------------------------------------------

    /// When a player fires a hybrid plant and has both gas and oil available
    /// beyond what pure plants need, the server enters Phase::PowerCitiesFuel.
    ///
    /// Setup: Hybrid #5 (GasOrOil, cost 2, 2 cities), pure-Coal #10 (cost 2, 2 cities).
    /// Resources: coal=4, gas=2, oil=4.  Pure coal needs 2. Hybrid cost 2:
    /// gas_after_pure=2, oil_after_pure=4; min_gas=0, max_gas=2 → ambiguous.
    #[test]
    fn test_power_cities_triggers_fuel_prompt_on_hybrid_ambiguity() {
        use crate::types::{PlantKind, PowerPlant};

        let (mut state, p1, _p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        state.phase = Phase::Bureaucracy {
            remaining: vec![p1],
        };

        give_player_cities(&mut state, p1, &["a", "b", "c"]);
        // Add one more city since test map only has a/b/c — insert d.
        state.map.cities.insert(
            "d".into(),
            crate::map::City {
                id: "d".into(),
                name: "D".into(),
                region: "r1".into(),
                owners: vec![p1],
                x: None,
                y: None,
            },
        );
        let player = state.player_mut(p1).unwrap();
        player.plants = vec![
            PowerPlant {
                number: 5,
                kind: PlantKind::GasOrOil,
                cost: 2,
                cities: 2,
            },
            PowerPlant {
                number: 10,
                kind: PlantKind::Coal,
                cost: 2,
                cities: 2,
            },
        ];
        player.resources = PlayerResources {
            coal: 4,
            oil: 4,
            gas: 2,
            uranium: 0,
        };

        apply_action(
            &mut state,
            p1,
            Action::PowerCities {
                plant_numbers: vec![5, 10],
            },
        )
        .unwrap();

        assert!(
            matches!(state.phase, Phase::PowerCitiesFuel { player, hybrid_cost, .. } if player == p1 && hybrid_cost == 2),
            "expected Phase::PowerCitiesFuel with hybrid_cost=2, got {:?}",
            state.phase
        );
        // Resources untouched until the player submits the split.
        let player = state.player(p1).unwrap();
        assert_eq!(player.resources.gas, 2);
        assert_eq!(player.resources.oil, 4);
    }

    /// A valid fuel split is applied: resources consumed, income awarded, phase advances.
    #[test]
    fn test_power_cities_fuel_applies_split() {
        use crate::types::{PlantKind, PowerPlant};

        let (mut state, p1, _p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        state.phase = Phase::Bureaucracy {
            remaining: vec![p1],
        };

        give_player_cities(&mut state, p1, &["a", "b", "c"]);
        state.map.cities.insert(
            "d".into(),
            crate::map::City {
                id: "d".into(),
                name: "D".into(),
                region: "r1".into(),
                owners: vec![p1],
                x: None,
                y: None,
            },
        );
        let player = state.player_mut(p1).unwrap();
        player.plants = vec![
            PowerPlant {
                number: 5,
                kind: PlantKind::GasOrOil,
                cost: 2,
                cities: 2,
            },
            PowerPlant {
                number: 10,
                kind: PlantKind::Coal,
                cost: 2,
                cities: 2,
            },
        ];
        player.resources = PlayerResources {
            coal: 4,
            oil: 4,
            gas: 2,
            uranium: 0,
        };
        let money_before = player.money;

        apply_action(
            &mut state,
            p1,
            Action::PowerCities {
                plant_numbers: vec![5, 10],
            },
        )
        .unwrap();

        assert!(matches!(state.phase, Phase::PowerCitiesFuel { .. }));

        // Player spends 0 gas + 2 oil for hybrid (preserves gas).
        apply_action(&mut state, p1, Action::PowerCitiesFuel { gas: 0, oil: 2 }).unwrap();

        // Phase should have advanced.
        assert!(
            !matches!(state.phase, Phase::PowerCitiesFuel { .. }),
            "expected phase to advance; got {:?}",
            state.phase
        );

        let player = state.player(p1).unwrap();
        // 4 coal − 2 (pure-coal #10) = 2
        assert_eq!(player.resources.coal, 2, "expected 2 coal remaining");
        // 2 gas − 0 (hybrid used oil instead) = 2
        assert_eq!(player.resources.gas, 2, "expected 2 gas remaining");
        // 4 oil − 2 (hybrid) = 2
        assert_eq!(player.resources.oil, 2, "expected 2 oil remaining");
        // Both plants fire (4 cities), income for 4 cities.
        assert_eq!(player.last_cities_powered, 4);
        let expected_income = income_for(4);
        assert_eq!(player.money, money_before + expected_income);
    }

    /// Sending an incorrect split total is rejected.
    #[test]
    fn test_power_cities_fuel_rejects_wrong_total() {
        use crate::types::{PlantKind, PowerPlant};

        let (mut state, p1, _p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        state.phase = Phase::Bureaucracy {
            remaining: vec![p1],
        };

        give_player_cities(&mut state, p1, &["a", "b"]);
        let player = state.player_mut(p1).unwrap();
        player.plants = vec![PowerPlant {
            number: 5,
            kind: PlantKind::GasOrOil,
            cost: 2,
            cities: 2,
        }];
        player.resources = PlayerResources {
            coal: 0,
            oil: 3,
            gas: 3,
            uranium: 0,
        };

        apply_action(
            &mut state,
            p1,
            Action::PowerCities {
                plant_numbers: vec![5],
            },
        )
        .unwrap();

        assert!(matches!(state.phase, Phase::PowerCitiesFuel { .. }));

        // gas(1) + oil(0) = 1 but hybrid_cost == 2 → rejected.
        let result = apply_action(&mut state, p1, Action::PowerCitiesFuel { gas: 1, oil: 0 });
        assert!(
            matches!(result, Err(ActionError::InvalidFuelSplit)),
            "expected InvalidFuelSplit"
        );
        // Phase and resources unchanged.
        assert!(matches!(state.phase, Phase::PowerCitiesFuel { .. }));
        assert_eq!(state.player(p1).unwrap().resources.gas, 3);
    }

    /// When only gas is available for hybrids the choice is forced and no prompt appears.
    #[test]
    fn test_power_cities_forced_split_no_prompt() {
        use crate::types::{PlantKind, PowerPlant};

        let (mut state, p1, _p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        state.phase = Phase::Bureaucracy {
            remaining: vec![p1],
        };

        give_player_cities(&mut state, p1, &["a", "b"]);
        let player = state.player_mut(p1).unwrap();
        player.plants = vec![PowerPlant {
            number: 5,
            kind: PlantKind::GasOrOil,
            cost: 2,
            cities: 2,
        }];
        // Only gas available → split is forced (oil=0, min_gas==max_gas).
        player.resources = PlayerResources {
            coal: 0,
            oil: 0,
            gas: 3,
            uranium: 0,
        };

        apply_action(
            &mut state,
            p1,
            Action::PowerCities {
                plant_numbers: vec![5],
            },
        )
        .unwrap();

        // No prompt — phase should advance without PowerCitiesFuel.
        assert!(
            !matches!(state.phase, Phase::PowerCitiesFuel { .. }),
            "expected phase to advance directly; got {:?}",
            state.phase
        );
        let player = state.player(p1).unwrap();
        assert_eq!(player.resources.gas, 1, "expected 1 gas remaining");
        assert_eq!(player.resources.oil, 0);
    }

    // -----------------------------------------------------------------------
    // Step 3 timing tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_step3_trigger_in_auction_defers_to_buy_resources() {
        // When the Step 3 card is drawn during Phase 2 (Auction), state.step must
        // not change until the start of Phase 3 (BuyResources).
        let (mut state, p1, p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();
        assert!(matches!(state.phase, Phase::Auction { .. }));

        // Drain the draw deck so the next refill draws the Step 3 card.
        // setup_deck already set below_step3 = Some(vec![]).
        state.market.deck.clear();

        let first = state.player_order[0];
        let second = state.player_order[1];
        let step_before = state.step;

        // First player selects a plant; second passes the bid → first wins.
        let first_plant = state.market.actual[0].number;
        apply_action(
            &mut state,
            first,
            Action::SelectPlant {
                plant_number: first_plant,
            },
        )
        .unwrap();
        apply_action(&mut state, second, Action::PassAuction).unwrap();
        // The refill after the purchase drew the Step 3 card.

        // Step must stay the same; transition deferred to BuyResources.
        assert_eq!(
            state.step, step_before,
            "state.step must not change mid-auction when Step 3 card is drawn"
        );
        assert_eq!(
            state.step3_pending,
            Some(crate::state::Step3Pending::AfterAuction),
            "step3_pending must be AfterAuction after drawing in Phase 2"
        );
        assert!(
            !state.market.in_step3,
            "market must still be in Step 2 layout during the auction"
        );

        // Second player selects the lowest available plant (auto-wins) → auction ends
        // → BuyResources starts → pending applied.
        let lowest = state.market.actual[0].number;
        apply_action(
            &mut state,
            second,
            Action::SelectPlant {
                plant_number: lowest,
            },
        )
        .unwrap();
        assert!(
            matches!(state.phase, Phase::BuyResources { .. }),
            "expected BuyResources; got {:?}",
            state.phase
        );
        assert_eq!(state.step, 3, "Step 3 must begin at start of BuyResources");
        assert!(state.market.in_step3, "market must be in Step 3 layout");
        assert_eq!(state.step3_pending, None, "pending must be cleared");

        let _ = (p1, p2);
    }

    #[test]
    fn test_step3_trigger_in_building_defers_to_bureaucracy() {
        // When the Step 3 card is drawn during Phase 4 (BuildCities), the market
        // transition runs immediately but state.step must not change until
        // the start of Phase 5 (Bureaucracy).
        let (mut state, p1, p2) = two_player_build_phase();
        state.step = 2;
        // Mark the Step 3 card as "already drawn" on the next note_step3_trigger call.
        state.market.below_step3 = Some(vec![]);
        state.market.step3_triggered = true;

        let first = match &state.phase {
            Phase::BuildCities { remaining } => *remaining.first().unwrap(),
            _ => unreachable!(),
        };
        let second = match &state.phase {
            Phase::BuildCities { remaining } => *remaining.last().unwrap(),
            _ => unreachable!(),
        };

        // First player builds a city — note_step3_trigger fires inside the handler.
        apply_action(
            &mut state,
            first,
            Action::BuildCities {
                city_ids: vec!["a".into()],
            },
        )
        .unwrap();

        // Market transition runs immediately for Phase 4 draws.
        assert!(
            state.market.in_step3,
            "market must switch to Step 3 layout immediately when card is drawn in Phase 4"
        );
        // But state.step must NOT yet be 3.
        assert_eq!(
            state.step, 2,
            "state.step must remain 2 until Bureaucracy starts"
        );
        assert_eq!(
            state.step3_pending,
            Some(crate::state::Step3Pending::AfterBuilding),
            "step3_pending must be AfterBuilding while still in Phase 4"
        );

        // Second player skips → phase ends → Bureaucracy starts → step becomes 3.
        apply_action(&mut state, second, Action::DoneBuilding).unwrap();
        assert!(
            matches!(state.phase, Phase::Bureaucracy { .. }),
            "expected Bureaucracy; got {:?}",
            state.phase
        );
        assert_eq!(state.step, 3, "Step 3 must begin at start of Bureaucracy");
        assert_eq!(state.step3_pending, None, "pending must be cleared");

        let _ = (p1, p2);
    }

    #[test]
    fn test_step3_trigger_in_bureaucracy_defers_to_next_round() {
        // When the Step 3 card is drawn during Phase 5 (Bureaucracy), state.step
        // must not change until the start of the next round. Critically, the
        // resource replenishment in that same bureaucracy must use Step 2 rates.
        let (mut state, p1, p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        state.phase = Phase::Bureaucracy {
            remaining: state.player_order.clone(),
        };
        state.step = 2;
        state.resources.coal = 0; // measure exact replenishment

        // Deck empty so the end-of-round market cycle hits the Step 3 card.
        state.market.deck.clear();
        state.market.below_step3 = Some(vec![]); // Step 3 card "in play"
                                                 // future must be non-empty for cycle_highest_to_bottom to pop.
        if state.market.future.is_empty() {
            use crate::types::{PlantKind, PowerPlant};
            state.market.future.push(PowerPlant {
                number: 50,
                kind: PlantKind::Uranium,
                cost: 2,
                cities: 7,
            });
        }

        // Both players power cities (no cities, no resources required).
        let power_order: Vec<PlayerId> = match &state.phase {
            Phase::Bureaucracy { remaining } => remaining.clone(),
            _ => unreachable!(),
        };
        for actor in &power_order {
            apply_action(
                &mut state,
                *actor,
                Action::PowerCities {
                    plant_numbers: vec![],
                },
            )
            .unwrap();
        }
        // end_of_round has fired; begin_auction started the next round.

        assert!(
            matches!(state.phase, Phase::Auction { .. }),
            "expected Auction for next round; got {:?}",
            state.phase
        );
        assert_eq!(
            state.step, 3,
            "Step 3 must begin at start of the next round"
        );
        assert_eq!(state.step3_pending, None, "pending must be cleared");
        // 2-player Step 2 replenishment rate = 4 coal; Step 3 rate = 3 coal.
        assert_eq!(
            state.resources.coal, 4,
            "replenishment must use Step 2 rates (4 coal) even though Step 3 was drawn this round"
        );

        let _ = (p1, p2);
    }

    #[test]
    fn test_step3_third_house_not_allowed_before_phase_boundary() {
        // After the Step 3 card is noted (step3_pending = AfterBuilding), state.step
        // is still 2. A city with 2 owners must reject a third builder — the Step 3
        // 3-house rule must not take effect until Bureaucracy starts.
        let (mut state, p1, p2) = two_player_build_phase();
        state.step = 2;

        // Populate city "a" with two fake owners to hit the Step-2 max.
        let fake1 = uuid::Uuid::new_v4();
        let fake2 = uuid::Uuid::new_v4();
        state
            .map
            .cities
            .get_mut("a")
            .unwrap()
            .owners
            .extend([fake1, fake2]);

        // Mark Step 3 card as drawn so the next build action notes it.
        state.market.below_step3 = Some(vec![]);
        state.market.step3_triggered = true;

        let first = match &state.phase {
            Phase::BuildCities { remaining } => *remaining.first().unwrap(),
            _ => unreachable!(),
        };
        let second = if first == p1 { p2 } else { p1 };

        // first builds city "b" — triggers the Step 3 note.
        apply_action(
            &mut state,
            first,
            Action::BuildCities {
                city_ids: vec!["b".into()],
            },
        )
        .unwrap();

        assert_eq!(
            state.step, 2,
            "step must remain 2 after noting the trigger (pending = AfterBuilding)"
        );
        assert_eq!(
            state.step3_pending,
            Some(crate::state::Step3Pending::AfterBuilding)
        );

        // second tries to build in city "a" which already has 2 owners.
        // state.step == 2 → max_per_city == 2 → must be rejected with CityFull.
        let err = apply_action(
            &mut state,
            second,
            Action::BuildCities {
                city_ids: vec!["a".into()],
            },
        )
        .unwrap_err();
        assert!(
            matches!(err, ActionError::CityFull(_)),
            "expected CityFull while step is still 2; got {err:?}"
        );

        let _ = (p1, p2);
    }

    #[test]
    fn test_handle_build_city_single_runs_step3_check() {
        // The single-city BuildCity action (Bug D fix) must call note_step3_trigger,
        // matching the behaviour of the batch BuildCities path.
        let (mut state, p1, p2) = two_player_build_phase();

        state.market.below_step3 = Some(vec![]);
        state.market.step3_triggered = true;

        let first = match &state.phase {
            Phase::BuildCities { remaining } => *remaining.first().unwrap(),
            _ => unreachable!(),
        };

        // Single-city Action::BuildCity (not the batch variant).
        apply_action(
            &mut state,
            first,
            Action::BuildCity {
                city_id: "a".into(),
            },
        )
        .unwrap();

        assert!(
            state.market.in_step3,
            "market must switch to Step 3 layout even via single BuildCity action"
        );
        assert_eq!(
            state.step3_pending,
            Some(crate::state::Step3Pending::AfterBuilding),
            "step3_pending must be set by the single-city BuildCity action"
        );

        let _ = (p1, p2);
    }

    // ── Discount token tests ──────────────────────────────────────────────────

    #[test]
    fn test_discount_token_placed_at_auction_start() {
        let (mut state, p1, _p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();
        assert!(matches!(state.phase, Phase::Auction { .. }));
        let smallest = state.market.actual[0].number;
        assert_eq!(
            state.market.discount_token,
            Some(smallest),
            "discount token must be on the smallest actual market plant at Phase 2 start"
        );
    }

    #[test]
    fn test_discount_token_min_bid_is_one() {
        let (mut state, p1, p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        let first = state.player_order[0];
        let tok_plant = state.market.discount_token.unwrap();

        // Give the first player only 1 Elektro — enough only if the token applies.
        state.player_mut(first).unwrap().money = 1;

        apply_action(
            &mut state,
            first,
            Action::SelectPlant {
                plant_number: tok_plant,
            },
        )
        .unwrap();

        // Bid should have been opened at 1, not at the plant's printed number.
        if let Phase::Auction { active_bid, .. } = &state.phase {
            let bid = active_bid.as_ref().expect("active bid after SelectPlant");
            assert_eq!(bid.amount, 1, "initial bid for discount plant must be 1");
        }
        let _ = (p1, p2);
    }

    #[test]
    fn test_buying_discount_plant_clears_token() {
        let (mut state, p1, p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        let first = state.player_order[0];
        let second = state.player_order[1];
        let tok_plant = state.market.discount_token.unwrap();

        // first selects the discount plant; second passes → first wins at min bid 1.
        apply_action(
            &mut state,
            first,
            Action::SelectPlant {
                plant_number: tok_plant,
            },
        )
        .unwrap();
        apply_action(&mut state, second, Action::PassAuction).unwrap();

        // Token must be cleared once the plant leaves the market.
        assert_eq!(
            state.market.discount_token, None,
            "discount token must be cleared after the tokened plant is purchased"
        );
        let _ = (p1, p2);
    }

    #[test]
    fn test_smaller_plant_drawn_clears_token() {
        let (mut state, p1, p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        let first = state.player_order[0];
        let second = state.player_order[1];
        let tok_plant = state.market.discount_token.unwrap();

        // Inject a plant with a smaller number than the tokened plant at the top of
        // the draw deck so the next `take_from_actual` refill surfaces it.
        let tiny = powergrid_core_types_test_plant(tok_plant.saturating_sub(1));
        state.market.deck.push(tiny);

        // Buy a non-tokened plant (actual[1]) so the refill draws our injected smaller plant.
        let non_tok = state.market.actual[1].number;
        apply_action(
            &mut state,
            first,
            Action::SelectPlant {
                plant_number: non_tok,
            },
        )
        .unwrap();
        apply_action(&mut state, second, Action::PassAuction).unwrap();

        // The smaller plant from the deck should have appeared in actual and cleared the token.
        assert_eq!(
            state.market.discount_token, None,
            "discount token must be cleared when a smaller plant is drawn into the market"
        );
        let _ = (p1, p2);
    }

    fn powergrid_core_types_test_plant(number: u8) -> crate::types::PowerPlant {
        use crate::types::{PlantKind, PowerPlant};
        PowerPlant {
            number,
            kind: PlantKind::Wind,
            cost: 0,
            cities: 1,
        }
    }

    #[test]
    fn test_unsold_discount_plant_removed_at_phase_end() {
        // Use round 2+ to allow players to pass nomination without hitting MustBuyPlantInRoundOne.
        // Set up a 2-player game in Auction phase for round 2.
        let (mut state, p1, p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        // Advance to round 2 by setting state.round and re-entering auction manually.
        // This is simpler than replaying all of round 1.
        state.round = 2;
        let first = state.player_order[0];
        let second = state.player_order[1];

        // Reset the discount token as begin_auction() would normally do.
        state.market.discount_token = state.market.actual.first().map(|p| p.number);
        let tok_plant = state.market.discount_token.unwrap();

        // Clear the deck so refill draws nothing (prevents a smaller plant clearing the token).
        state.market.deck.clear();
        state.market.below_step3 = None;

        // first buys actual[1] (non-discount); second passes the bid → first auto-wins.
        let non_tok = state.market.actual[1].number;
        apply_action(
            &mut state,
            first,
            Action::SelectPlant {
                plant_number: non_tok,
            },
        )
        .unwrap();
        apply_action(&mut state, second, Action::PassAuction).unwrap();

        // second passes their nomination turn (allowed since round > 1).
        apply_action(&mut state, second, Action::PassAuction).unwrap();

        // Phase should have advanced past auction.
        assert!(
            !matches!(state.phase, Phase::Auction { .. }),
            "should have moved past auction; got {:?}",
            state.phase
        );

        // The tokened plant must no longer be in actual or the deck.
        assert!(
            !state.market.actual.iter().any(|p| p.number == tok_plant),
            "unsold discount plant must be removed from actual market"
        );
        assert!(
            !state.market.deck.iter().any(|p| p.number == tok_plant),
            "unsold discount plant must not be returned to the deck"
        );

        // Token cleared.
        assert_eq!(state.market.discount_token, None);
        let _ = (p1, p2);
    }

    #[test]
    fn test_no_discount_outside_phase_two() {
        let (mut state, p1, p2) = two_player_game();
        apply_action(&mut state, p1, Action::StartGame).unwrap();

        let first = state.player_order[0];
        let second = state.player_order[1];

        // Both players buy a plant (round 1 requires it). After both have bought, the
        // auction ends automatically and the game moves to BuyResources.
        let plant1 = state.market.actual[0].number;
        let plant2 = state.market.actual[1].number;

        // first selects plant1; second passes the bid → first auto-wins.
        apply_action(
            &mut state,
            first,
            Action::SelectPlant {
                plant_number: plant1,
            },
        )
        .unwrap();
        apply_action(&mut state, second, Action::PassAuction).unwrap();

        // second selects plant2; first is already in `bought` so second auto-wins immediately.
        apply_action(
            &mut state,
            second,
            Action::SelectPlant {
                plant_number: plant2,
            },
        )
        .unwrap();

        // Should now be in BuyResources.
        assert!(
            matches!(state.phase, Phase::BuyResources { .. }),
            "expected BuyResources after both bought; got {:?}",
            state.phase
        );
        assert_eq!(
            state.market.discount_token, None,
            "discount token must be None outside Phase 2"
        );
        let _ = (p1, p2);
    }
}
