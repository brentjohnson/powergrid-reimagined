use numpy::{IntoPyArray, PyArray1};
use powergrid_bot_strategy::{default_registry, Bot};
use powergrid_core::{
    actions::Action,
    map::default_map,
    rules::{apply_action, effective_min_bid},
    state::GameState,
    types::{
        connection_cost, BotDifficulty, Phase, PlantKind, PlayerColor, PlayerId, PlayerResources,
        PowerPlant, Resource,
    },
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde::Serialize;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Observation / action-space constants — must stay in sync with constants.py
// ---------------------------------------------------------------------------

/// Sorted city ids of the default map (assets/maps/usa.toml).
const CITY_IDS: [&str; 49] = [
    "albuquerque",
    "atlanta",
    "boston",
    "calgary",
    "charlotte",
    "chicago",
    "chihuahua",
    "columbus",
    "dallas",
    "denver",
    "detroit",
    "edmonton",
    "guadalajara",
    "houston",
    "indianapolis",
    "jacksonville",
    "juarez",
    "kansascity",
    "lasvegas",
    "losangeles",
    "memphis",
    "mexicocityn",
    "mexicocitys",
    "miami",
    "milwaukee",
    "minneapolis",
    "monterrey",
    "montreal",
    "nashville",
    "neworleans",
    "newyorkn",
    "newyorks",
    "oklahomacity",
    "ottawa",
    "philadelphia",
    "pittsburgh",
    "portland",
    "quebec",
    "regina",
    "saltlakecity",
    "sanantonio",
    "sandiego",
    "sanfrancisco",
    "seattle",
    "stlouis",
    "toronto",
    "vancouver",
    "washington",
    "winnipeg",
];

const REGION_NAMES: [&str; 7] = [
    "central",
    "east",
    "northeast",
    "northwest",
    "south",
    "southwest",
    "west",
];

const N_CITIES: usize = CITY_IDS.len();
const N_REGIONS: usize = REGION_NAMES.len();

// Observation layout: money + resources + self plants + self cities +
// opponent summary + opponent cities + city slot counts + active regions +
// actual market + future market + market meta + resource market +
// phase/step/round/end-game/turn-order scalars + phase scratch.
const OBS_SIZE: usize =
    1 + 4 + 15 + N_CITIES + 20 + 5 * N_CITIES + N_CITIES + N_REGIONS + 24 + 20 + 3 + 4 + 5 + 8;

// Action base indices.
const PASS_AUCTION_IDX: usize = 0;
const DONE_BUYING_IDX: usize = 1;
const DONE_BUILDING_IDX: usize = 2;
const SELECT_PLANT_BASE: usize = 3;
const PLACE_BID_BASE: usize = 11;
const DISCARD_PLANT_BASE: usize = 61;
const BUILD_CITY_BASE: usize = 64;
const BUY_RESOURCE_BASE: usize = BUILD_CITY_BASE + N_CITIES;
const POWER_CITIES_BASE: usize = BUY_RESOURCE_BASE + 4;
const DISCARD_RESOURCE_BASE: usize = POWER_CITIES_BASE + 8;
const POWER_FUEL_BASE: usize = DISCARD_RESOURCE_BASE + 9;
const N_ACTIONS: usize = POWER_FUEL_BASE + 9;

// ---------------------------------------------------------------------------
// Legal-move info
// ---------------------------------------------------------------------------

#[derive(Default, Serialize)]
struct LegalMoveInfo {
    pass_auction: bool,
    done_buying: bool,
    done_building: bool,
    select_plant_slots: Vec<usize>,
    /// Minimum legal bid amount (= active_bid.amount + 1).
    bid_min: Option<u32>,
    /// Maximum legal bid amount (= actor's money).
    bid_max: Option<u32>,
    discard_plant_slots: Vec<usize>,
    buildable_city_ids: Vec<String>,
    /// Resource indices: coal=0, oil=1, garbage=2, uranium=3
    buyable_resources: Vec<u8>,
    /// Bitmasks 0..7 over the actor's first 3 plants (sorted by number).
    power_subsets: Vec<u8>,
    /// Valid gas amounts to drop in DiscardResource (oil = drop_total - gas).
    discard_resource_gas: Vec<u8>,
    /// Valid gas amounts to use in PowerCitiesFuel (oil = hybrid_cost - gas).
    fuel_gas: Vec<u8>,
}

fn is_subset_feasible(plants: &[PowerPlant], resources: &PlayerResources, mask: u8) -> bool {
    let mut coal = resources.coal;
    let mut oil = resources.oil;
    let mut gas = resources.gas;
    let mut uranium = resources.uranium;

    // Pass 1: satisfy pure-fuel plants.
    for (i, plant) in plants.iter().enumerate().take(3) {
        if mask & (1 << i) == 0 {
            continue;
        }
        match plant.kind {
            PlantKind::Coal => {
                if coal < plant.cost {
                    return false;
                }
                coal -= plant.cost;
            }
            PlantKind::Oil => {
                if oil < plant.cost {
                    return false;
                }
                oil -= plant.cost;
            }
            PlantKind::Gas => {
                if gas < plant.cost {
                    return false;
                }
                gas -= plant.cost;
            }
            PlantKind::Uranium => {
                if uranium < plant.cost {
                    return false;
                }
                uranium -= plant.cost;
            }
            PlantKind::Wind | PlantKind::GasOrOil => {}
        }
    }

    // Pass 2: satisfy GasOrOil hybrid plants with remaining fuel.
    for (i, plant) in plants.iter().enumerate().take(3) {
        if mask & (1 << i) == 0 || plant.kind != PlantKind::GasOrOil {
            continue;
        }
        let available = gas + oil;
        if available < plant.cost {
            return false;
        }
        let use_oil = plant.cost.min(oil);
        oil -= use_oil;
        gas -= plant.cost - use_oil;
    }

    true
}

fn current_actor_id(state: &GameState) -> Option<PlayerId> {
    match &state.phase {
        Phase::Lobby | Phase::PlayerOrder | Phase::GameOver { .. } => None,
        Phase::Auction {
            current_bidder_idx,
            active_bid,
            ..
        } => {
            if let Some(bid) = active_bid {
                bid.remaining_bidders.first().copied()
            } else {
                state.player_order.get(*current_bidder_idx).copied()
            }
        }
        Phase::DiscardPlant { player, .. } => Some(*player),
        Phase::DiscardResource { player, .. } => Some(*player),
        Phase::BuyResources { remaining } => remaining.first().copied(),
        Phase::BuildCities { remaining } => remaining.first().copied(),
        Phase::Bureaucracy { remaining } => remaining.first().copied(),
        Phase::PowerCitiesFuel { player, .. } => Some(*player),
    }
}

fn compute_legal_move_info(state: &GameState, actor_id: PlayerId) -> LegalMoveInfo {
    let mut info = LegalMoveInfo::default();

    let Some(player) = state.players.iter().find(|p| p.id == actor_id) else {
        return info;
    };

    match &state.phase {
        Phase::Lobby | Phase::PlayerOrder | Phase::GameOver { .. } => {}

        Phase::Auction {
            current_bidder_idx,
            active_bid,
            bought,
            passed,
        } => {
            if let Some(bid) = active_bid {
                if bid.remaining_bidders.first() == Some(&actor_id) {
                    info.pass_auction = true;
                    let min_bid = bid.amount + 1;
                    if player.money >= min_bid {
                        info.bid_min = Some(min_bid);
                        info.bid_max = Some(player.money);
                    }
                }
            } else if state.player_order.get(*current_bidder_idx) == Some(&actor_id)
                && !bought.contains(&actor_id)
                && !passed.contains(&actor_id)
            {
                if state.round > 1 || bought.contains(&actor_id) {
                    info.pass_auction = true;
                }
                // Only actual-market plants can be selected; future market is read-only.
                for (slot, plant) in state.market.actual.iter().enumerate() {
                    if player.money >= effective_min_bid(&state.market, plant.number) {
                        info.select_plant_slots.push(slot);
                    }
                }
            }
        }

        Phase::DiscardPlant {
            player: discard_player,
            ..
        } => {
            if *discard_player == actor_id {
                for slot in 0..player.plants.len() {
                    info.discard_plant_slots.push(slot);
                }
            }
        }

        Phase::DiscardResource {
            player: res_player,
            drop_total,
            ..
        } => {
            if *res_player == actor_id {
                for gas in 0..=*drop_total {
                    let oil = drop_total - gas;
                    if gas <= player.resources.gas && oil <= player.resources.oil {
                        info.discard_resource_gas.push(gas);
                    }
                }
            }
        }

        Phase::BuyResources { remaining } => {
            if remaining.first() == Some(&actor_id) {
                info.done_buying = true;
                for (ri, &resource) in [
                    Resource::Coal,
                    Resource::Oil,
                    Resource::Gas,
                    Resource::Uranium,
                ]
                .iter()
                .enumerate()
                {
                    if player.can_add_resource(resource, 1) {
                        if let Some(cost) = state.resources.price(resource, 1) {
                            if cost <= player.money {
                                info.buyable_resources.push(ri as u8);
                            }
                        }
                    }
                }
            }
        }

        Phase::BuildCities { remaining } => {
            if remaining.first() == Some(&actor_id) {
                info.done_building = true;
                let actor_cities = state.player_cities(actor_id);
                for (city_id, city) in &state.map.cities {
                    if !state.is_city_active(city_id) {
                        continue;
                    }
                    if city.owners.len() >= state.step as usize {
                        continue;
                    }
                    if city.owners.contains(&actor_id) {
                        continue;
                    }
                    if let Some(routing) = state.map.connection_cost_to(&actor_cities, city_id) {
                        if routing + connection_cost(city.owners.len()) <= player.money {
                            info.buildable_city_ids.push(city_id.clone());
                        }
                    }
                }
            }
        }

        Phase::Bureaucracy { remaining } => {
            if remaining.contains(&actor_id) {
                let n = player.plants.len().min(3) as u8;
                for mask in 0u8..(1u8 << n) {
                    if is_subset_feasible(&player.plants, &player.resources, mask) {
                        info.power_subsets.push(mask);
                    }
                }
            }
        }

        Phase::PowerCitiesFuel {
            player: fuel_player,
            hybrid_cost,
            ..
        } => {
            if *fuel_player == actor_id {
                for gas in 0..=*hybrid_cost {
                    let oil = hybrid_cost - gas;
                    if gas <= player.resources.gas && oil <= player.resources.oil {
                        info.fuel_gas.push(gas);
                    }
                }
            }
        }
    }

    info
}

// ---------------------------------------------------------------------------
// Fast native obs / mask / action encoding (no JSON round-trip)
// ---------------------------------------------------------------------------

fn city_index(city_id: &str) -> Option<usize> {
    CITY_IDS.iter().position(|&id| id == city_id)
}

fn plant_kind_id(kind: PlantKind) -> f32 {
    match kind {
        PlantKind::Coal => 1.0,
        PlantKind::Oil => 2.0,
        PlantKind::GasOrOil => 3.0,
        PlantKind::Gas => 4.0,
        PlantKind::Uranium => 5.0,
        PlantKind::Wind => 6.0,
    }
}

fn phase_id_f32(phase: &Phase) -> f32 {
    match phase {
        Phase::Lobby => 0.0,
        Phase::PlayerOrder => 1.0,
        Phase::Auction { .. } => 2.0,
        Phase::DiscardPlant { .. } => 3.0,
        Phase::DiscardResource { .. } => 4.0,
        Phase::BuyResources { .. } => 5.0,
        Phase::BuildCities { .. } => 6.0,
        Phase::Bureaucracy { .. } => 7.0,
        Phase::PowerCitiesFuel { .. } => 8.0,
        Phase::GameOver { .. } => 9.0,
    }
}

/// Port of `encoding.py::encode_observation` — builds obs vector directly from GameState.
fn build_observation(state: &GameState, actor_id: PlayerId) -> Vec<f32> {
    let mut obs = vec![0.0f32; OBS_SIZE];
    let mut idx = 0usize;

    let Some(me) = state.players.iter().find(|p| p.id == actor_id) else {
        return obs;
    };

    let opponents: Vec<_> = state.players.iter().filter(|p| p.id != actor_id).collect();

    // 1. Self money (1)
    obs[idx] = me.money as f32 / 500.0;
    idx += 1;

    // 2. Self resources (4): coal, oil, gas, uranium.
    // Denominators = market price-track capacities (coal 27, oil 20, gas 24, uranium 12).
    obs[idx] = me.resources.coal as f32 / 27.0;
    obs[idx + 1] = me.resources.oil as f32 / 20.0;
    obs[idx + 2] = me.resources.gas as f32 / 24.0;
    obs[idx + 3] = me.resources.uranium as f32 / 12.0;
    idx += 4;

    // 3. Self plants (3 × 5 = 15): padded to 3 slots
    for (i, plant) in me.plants.iter().take(3).enumerate() {
        let base = idx + i * 5;
        let cap = if matches!(plant.kind, PlantKind::Wind) {
            0.0
        } else {
            plant.cost as f32 * 2.0
        };
        obs[base] = plant.number as f32 / 60.0;
        obs[base + 1] = plant_kind_id(plant.kind) / 6.0;
        obs[base + 2] = plant.cost as f32 / 5.0;
        obs[base + 3] = plant.cities as f32 / 8.0;
        obs[base + 4] = cap / 10.0;
    }
    idx += 15;

    // 4. Self cities (N_CITIES)
    for city_id in state.player_cities(actor_id) {
        if let Some(ci) = city_index(&city_id) {
            obs[idx + ci] = 1.0;
        }
    }
    idx += N_CITIES;

    // 5. Opponents (5 × 4 = 20): plants, cities, cap, last_powered (money hidden)
    for (i, opp) in opponents.iter().take(5).enumerate() {
        let base = idx + i * 4;
        let cap: f32 = opp
            .plants
            .iter()
            .filter(|p| !matches!(p.kind, PlantKind::Wind))
            .map(|p| p.cost as f32 * 2.0)
            .sum();
        obs[base] = opp.plants.len() as f32 / 3.0;
        obs[base + 1] = state.player_city_count(opp.id) as f32 / N_CITIES as f32;
        obs[base + 2] = cap / 30.0;
        obs[base + 3] = opp.last_cities_powered as f32 / 21.0;
    }
    idx += 20;

    // 6. Opponent cities (5 × N_CITIES)
    for (i, opp) in opponents.iter().take(5).enumerate() {
        for city_id in state.player_cities(opp.id) {
            if let Some(ci) = city_index(&city_id) {
                obs[idx + i * N_CITIES + ci] = 1.0;
            }
        }
    }
    idx += 5 * N_CITIES;

    // 7. City slot counts (N_CITIES)
    for (ci, &city_id) in CITY_IDS.iter().enumerate() {
        if let Some(city) = state.map.cities.get(city_id) {
            obs[idx + ci] = city.owners.len() as f32 / 3.0;
        }
    }
    idx += N_CITIES;

    // 8. Active regions (N_REGIONS)
    for (i, &region) in REGION_NAMES.iter().enumerate() {
        if state.active_regions.iter().any(|r| r == region) {
            obs[idx + i] = 1.0;
        }
    }
    idx += N_REGIONS;

    // 9. Plant market actual (4 × 6 = 24): number, kind, cost, cities, present, discount
    for (i, plant) in state.market.actual.iter().take(4).enumerate() {
        let base = idx + i * 6;
        obs[base] = plant.number as f32 / 60.0;
        obs[base + 1] = plant_kind_id(plant.kind) / 6.0;
        obs[base + 2] = plant.cost as f32 / 5.0;
        obs[base + 3] = plant.cities as f32 / 8.0;
        obs[base + 4] = 1.0;
        obs[base + 5] = if state.market.discount_token == Some(plant.number) {
            1.0
        } else {
            0.0
        };
    }
    idx += 24;

    // 10. Plant market future (4 × 5 = 20): number, kind, cost, cities, present
    for (i, plant) in state.market.future.iter().take(4).enumerate() {
        let base = idx + i * 5;
        obs[base] = plant.number as f32 / 60.0;
        obs[base + 1] = plant_kind_id(plant.kind) / 6.0;
        obs[base + 2] = plant.cost as f32 / 5.0;
        obs[base + 3] = plant.cities as f32 / 8.0;
        obs[base + 4] = 1.0;
    }
    idx += 20;

    // 11. Market meta (3)
    obs[idx] = if state.market.step3_triggered {
        1.0
    } else {
        0.0
    };
    obs[idx + 1] = if state.market.in_step3 { 1.0 } else { 0.0 };
    obs[idx + 2] = state.market.deck.len() as f32 / 50.0;
    idx += 3;

    // 12. Resource market (4) — denominators = price-track capacities.
    obs[idx] = state.resources.coal as f32 / 27.0;
    obs[idx + 1] = state.resources.oil as f32 / 20.0;
    obs[idx + 2] = state.resources.gas as f32 / 24.0;
    obs[idx + 3] = state.resources.uranium as f32 / 12.0;
    idx += 4;

    // 13. Phase id (1)
    obs[idx] = phase_id_f32(&state.phase) / 9.0;
    idx += 1;

    // 14. Step (1)
    obs[idx] = state.step as f32 / 3.0;
    idx += 1;

    // 15. Round (1)
    obs[idx] = state.round as f32 / 50.0;
    idx += 1;

    // 16. End-game cities threshold (1)
    obs[idx] = state.end_game_cities as f32 / 25.0;
    idx += 1;

    // 17. Turn-order position of this actor (1)
    if let Some(pos) = state.player_order.iter().position(|&id| id == actor_id) {
        let n = (state.player_order.len() as f32 - 1.0).max(1.0);
        obs[idx] = pos as f32 / n;
    }
    idx += 1;

    // 18. Phase-specific scratch features (8)
    match &state.phase {
        Phase::Auction {
            current_bidder_idx,
            active_bid,
            bought,
            passed,
        } => {
            obs[idx] = *current_bidder_idx as f32 / 5.0;
            if let Some(bid) = active_bid {
                obs[idx + 1] = bid.amount as f32 / 200.0;
                obs[idx + 2] = bid.plant_number as f32 / 60.0;
                obs[idx + 3] = bid.remaining_bidders.len() as f32 / 5.0;
                obs[idx + 4] = 1.0;
            }
            obs[idx + 5] = bought.len() as f32 / 6.0;
            obs[idx + 6] = passed.len() as f32 / 6.0;
        }
        Phase::DiscardPlant { .. } => {
            obs[idx] = 1.0;
        }
        Phase::DiscardResource { drop_total, .. } => {
            obs[idx] = *drop_total as f32 / 8.0;
        }
        Phase::BuyResources { remaining } => {
            obs[idx] = remaining.len() as f32 / 6.0;
        }
        Phase::BuildCities { remaining } => {
            obs[idx] = remaining.len() as f32 / 6.0;
        }
        Phase::Bureaucracy { remaining } => {
            obs[idx] = remaining.len() as f32 / 6.0;
        }
        Phase::PowerCitiesFuel { hybrid_cost, .. } => {
            obs[idx] = *hybrid_cost as f32 / 20.0;
        }
        _ => {}
    }
    idx += 8;

    debug_assert_eq!(idx, OBS_SIZE, "observation size mismatch");
    // Clamp into the Box bounds: a few features (e.g. player stockpiles, late
    // rounds) can exceed their nominal denominator in extreme games.
    for v in &mut obs {
        *v = v.clamp(0.0, 1.0);
    }
    obs
}

/// Port of `encoding.py::mask_from_info` — builds action mask directly from GameState.
fn build_action_mask(state: &GameState, actor_id: PlayerId) -> Vec<u8> {
    let info = compute_legal_move_info(state, actor_id);
    let mut mask = vec![0u8; N_ACTIONS];

    if info.pass_auction {
        mask[PASS_AUCTION_IDX] = 1;
    }
    if info.done_buying {
        mask[DONE_BUYING_IDX] = 1;
    }
    if info.done_building {
        mask[DONE_BUILDING_IDX] = 1;
    }

    for &slot in &info.select_plant_slots {
        if slot < 8 {
            mask[SELECT_PLANT_BASE + slot] = 1;
        }
    }

    if let (Some(bid_min), Some(bid_max)) = (info.bid_min, info.bid_max) {
        for offset in 0u32..50 {
            if bid_min + offset <= bid_max {
                mask[PLACE_BID_BASE + offset as usize] = 1;
            } else {
                break;
            }
        }
    }

    for &slot in &info.discard_plant_slots {
        if slot < 3 {
            mask[DISCARD_PLANT_BASE + slot] = 1;
        }
    }

    for city_id in &info.buildable_city_ids {
        if let Some(ci) = city_index(city_id) {
            mask[BUILD_CITY_BASE + ci] = 1;
        }
    }

    for &ri in &info.buyable_resources {
        if (ri as usize) < 4 {
            mask[BUY_RESOURCE_BASE + ri as usize] = 1;
        }
    }

    for &bm in &info.power_subsets {
        if (bm as usize) < 8 {
            mask[POWER_CITIES_BASE + bm as usize] = 1;
        }
    }

    for &gas in &info.discard_resource_gas {
        if (gas as usize) < 9 {
            mask[DISCARD_RESOURCE_BASE + gas as usize] = 1;
        }
    }

    for &gas in &info.fuel_gas {
        if (gas as usize) < 9 {
            mask[POWER_FUEL_BASE + gas as usize] = 1;
        }
    }

    mask
}

/// Port of `encoding.py::id_to_action_json` — converts flat integer to Action directly.
fn action_id_to_action(action_id: u16, state: &GameState, actor_id: PlayerId) -> Action {
    let aid = action_id as usize;

    match aid {
        0 => return Action::PassAuction,
        1 => return Action::DoneBuying,
        2 => return Action::DoneBuilding,
        _ => {}
    }

    if (SELECT_PLANT_BASE..PLACE_BID_BASE).contains(&aid) {
        let slot = aid - SELECT_PLANT_BASE;
        let all: Vec<_> = state
            .market
            .actual
            .iter()
            .chain(state.market.future.iter())
            .collect();
        return if slot < all.len() {
            Action::SelectPlant {
                plant_number: all[slot].number,
            }
        } else {
            Action::PassAuction
        };
    }

    if (PLACE_BID_BASE..DISCARD_PLANT_BASE).contains(&aid) {
        let offset = (aid - PLACE_BID_BASE) as u32;
        if let Phase::Auction {
            active_bid: Some(bid),
            ..
        } = &state.phase
        {
            return Action::PlaceBid {
                amount: bid.amount + 1 + offset,
            };
        }
        return Action::PassAuction;
    }

    if (DISCARD_PLANT_BASE..BUILD_CITY_BASE).contains(&aid) {
        let slot = aid - DISCARD_PLANT_BASE;
        if let Some(player) = state.players.iter().find(|p| p.id == actor_id) {
            let mut plants = player.plants.clone();
            plants.sort_by_key(|p| p.number);
            if slot < plants.len() {
                return Action::DiscardPlant {
                    plant_number: plants[slot].number,
                };
            }
        }
        return Action::PassAuction;
    }

    if (BUILD_CITY_BASE..BUY_RESOURCE_BASE).contains(&aid) {
        let ci = aid - BUILD_CITY_BASE;
        return if ci < CITY_IDS.len() {
            Action::BuildCity {
                city_id: CITY_IDS[ci].to_string(),
            }
        } else {
            Action::DoneBuilding
        };
    }

    if (BUY_RESOURCE_BASE..POWER_CITIES_BASE).contains(&aid) {
        let ri = aid - BUY_RESOURCE_BASE;
        let resource = [
            Resource::Coal,
            Resource::Oil,
            Resource::Gas,
            Resource::Uranium,
        ][ri];
        return Action::BuyResourceBatch {
            purchases: vec![(resource, 1)],
        };
    }

    if (POWER_CITIES_BASE..DISCARD_RESOURCE_BASE).contains(&aid) {
        let bitmask = (aid - POWER_CITIES_BASE) as u8;
        if let Some(player) = state.players.iter().find(|p| p.id == actor_id) {
            let mut plants = player.plants.clone();
            plants.sort_by_key(|p| p.number);
            let plant_numbers: Vec<u8> = plants
                .iter()
                .take(3)
                .enumerate()
                .filter(|(i, _)| bitmask & (1 << i) != 0)
                .map(|(_, p)| p.number)
                .collect();
            return Action::PowerCities { plant_numbers };
        }
        return Action::PowerCities {
            plant_numbers: vec![],
        };
    }

    if (DISCARD_RESOURCE_BASE..POWER_FUEL_BASE).contains(&aid) {
        let gas = (aid - DISCARD_RESOURCE_BASE) as u8;
        let drop_total = if let Phase::DiscardResource { drop_total, .. } = &state.phase {
            *drop_total
        } else {
            0
        };
        let oil = drop_total.saturating_sub(gas);
        return Action::DiscardResource { gas, oil };
    }

    if (POWER_FUEL_BASE..N_ACTIONS).contains(&aid) {
        let gas = (aid - POWER_FUEL_BASE) as u8;
        let hybrid_cost = if let Phase::PowerCitiesFuel { hybrid_cost, .. } = &state.phase {
            *hybrid_cost
        } else {
            0
        };
        let oil = hybrid_cost.saturating_sub(gas);
        return Action::PowerCitiesFuel { gas, oil };
    }

    Action::PassAuction
}

// ---------------------------------------------------------------------------
// Game Python class
// ---------------------------------------------------------------------------

#[pyclass]
struct Game {
    state: GameState,
}

#[pymethods]
impl Game {
    #[new]
    fn new(num_players: usize, seed: Option<u64>) -> PyResult<Self> {
        if !(2..=6).contains(&num_players) {
            return Err(PyValueError::new_err("num_players must be 2–6"));
        }
        let map = default_map();
        let state = match seed {
            Some(s) => GameState::new_with_seed(map, num_players, s),
            None => GameState::new(map, num_players),
        };
        Ok(Game { state })
    }

    /// Join all players and start the game.
    /// `colors` must be snake_case strings: "red", "blue", "green", "yellow", "purple", "white".
    fn start(&mut self, player_names: Vec<String>, colors: Vec<String>) -> PyResult<()> {
        if player_names.len() != colors.len() {
            return Err(PyValueError::new_err(
                "player_names and colors must have the same length",
            ));
        }
        let mut host_id: Option<Uuid> = None;
        let base_seed = self.state.rng_seed.unwrap_or(0);
        for (i, (name, color_str)) in player_names.iter().zip(colors.iter()).enumerate() {
            let color: PlayerColor = serde_json::from_value(serde_json::Value::String(
                color_str.clone(),
            ))
            .map_err(|e| PyValueError::new_err(format!("invalid color '{}': {}", color_str, e)))?;
            // Deterministic UUID derived from seed+index so reset() with the same seed
            // produces identical agent IDs (required for reproducibility).
            let id = if base_seed != 0 {
                let lo = base_seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(i as u64);
                let hi = base_seed
                    .wrapping_mul(1442695040888963407)
                    .wrapping_add(i as u64 + 1);
                Uuid::from_u128((hi as u128) << 64 | lo as u128)
            } else {
                Uuid::new_v4()
            };
            apply_action(
                &mut self.state,
                id,
                Action::JoinGame {
                    name: name.clone(),
                    color,
                },
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if host_id.is_none() {
                host_id = Some(id);
            }
        }
        if let Some(hid) = host_id {
            apply_action(&mut self.state, hid, Action::StartGame)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        Ok(())
    }

    /// Serialized `GameStateView` as a JSON string. When `viewer` is given,
    /// that player's own money is included (opponent money is always zeroed,
    /// matching what a seated player may see).
    #[pyo3(signature = (viewer=None))]
    fn state_json(&self, viewer: Option<&str>) -> PyResult<String> {
        let viewer_id = match viewer {
            Some(v) => Some(Uuid::parse_str(v).map_err(|e| PyValueError::new_err(e.to_string()))?),
            None => None,
        };
        Ok(
            serde_json::to_string(&self.state.view_for(viewer_id))
                .expect("serialize GameStateView"),
        )
    }

    /// Player IDs in join order (same as `player_order` after `start()`).
    fn player_ids(&self) -> Vec<String> {
        self.state
            .players
            .iter()
            .map(|p| p.id.to_string())
            .collect()
    }

    /// UUID string of the player whose turn it is, or None if no single actor (Lobby, GameOver).
    fn current_actor(&self) -> Option<String> {
        current_actor_id(&self.state).map(|id| id.to_string())
    }

    fn is_terminal(&self) -> bool {
        matches!(self.state.phase, Phase::GameOver { .. })
    }

    fn winner(&self) -> Option<String> {
        if let Phase::GameOver { winner } = &self.state.phase {
            Some(winner.to_string())
        } else {
            None
        }
    }

    /// Apply an action. Raises `ValueError` on invalid actions (including wrong-phase, not-your-turn, etc.).
    fn apply(&mut self, actor: &str, action_json: &str) -> PyResult<()> {
        let actor_id = Uuid::parse_str(actor).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let action: Action = serde_json::from_str(action_json)
            .map_err(|e| PyValueError::new_err(format!("invalid action JSON: {}", e)))?;
        apply_action(&mut self.state, actor_id, action)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Ask the Rust strategy bot to decide an action for `actor`.
    /// Returns the action as a JSON string, or None if the bot has no move.
    fn bot_decide(&self, actor: &str, difficulty: &str) -> PyResult<Option<String>> {
        let actor_id = Uuid::parse_str(actor).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let player = self
            .state
            .players
            .iter()
            .find(|p| p.id == actor_id)
            .ok_or_else(|| PyValueError::new_err("actor not found in game"))?;
        let diff = parse_difficulty(difficulty);
        let registry = default_registry();
        let profile = registry.profile_for(diff).clone();
        let seed = actor_id.as_u128() as u64;
        let mut bot = Bot::new(actor_id, player.name.clone(), player.color, profile, seed);
        Ok(bot
            .decide(&self.state)
            .map(|a| serde_json::to_string(&a).expect("serialize action")))
    }

    /// Sorted list of all city IDs in the map (stable across calls — use to build the city index).
    fn city_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.state.map.cities.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// JSON describing which moves are legal for `actor` right now.
    /// Python uses this to build the action mask without re-implementing game rules.
    fn legal_move_info(&self, actor: &str) -> PyResult<String> {
        let actor_id = Uuid::parse_str(actor).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let info = compute_legal_move_info(&self.state, actor_id);
        Ok(serde_json::to_string(&info).expect("serialize LegalMoveInfo"))
    }

    // -----------------------------------------------------------------------
    // Fast native methods — no JSON, direct numpy output
    // -----------------------------------------------------------------------

    /// Observation vector for `actor` as a float32 numpy array of length OBS_SIZE.
    /// Bypasses JSON serialisation; ~10× faster than state_json() + encode_observation().
    fn observation<'py>(
        &self,
        py: Python<'py>,
        actor: &str,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let actor_id = Uuid::parse_str(actor).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(build_observation(&self.state, actor_id).into_pyarray(py))
    }

    /// Action mask for `actor` as a uint8 numpy array of length N_ACTIONS.
    /// Bypasses JSON serialisation; ~10× faster than legal_move_info() + mask_from_info().
    fn action_mask<'py>(&self, py: Python<'py>, actor: &str) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let actor_id = Uuid::parse_str(actor).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(build_action_mask(&self.state, actor_id).into_pyarray(py))
    }

    /// Apply action by integer id (0..N_ACTIONS). Bypasses JSON encoding.
    fn apply_action_id(&mut self, actor: &str, action_id: u16) -> PyResult<()> {
        let actor_id = Uuid::parse_str(actor).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let action = action_id_to_action(action_id, &self.state, actor_id);
        apply_action(&mut self.state, actor_id, action)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Fused self-play step: apply `action_id` for the current actor and return
    /// `(obs, mask, reward, terminal)` for the **next** actor in a single PyO3
    /// round-trip.  Both `obs` and `mask` are zero arrays when `terminal` is True.
    /// Reward is +1 if the acting player won, -1 if they lost, 0 otherwise.
    #[allow(clippy::type_complexity)]
    fn step_self_play<'py>(
        &mut self,
        py: Python<'py>,
        action_id: u16,
    ) -> PyResult<(
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<u8>>,
        f32,
        bool,
    )> {
        let actor_id = current_actor_id(&self.state).ok_or_else(|| {
            PyValueError::new_err("no current actor (game may be terminal or in lobby)")
        })?;

        let action = action_id_to_action(action_id, &self.state, actor_id);
        apply_action(&mut self.state, actor_id, action)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        let (reward, terminal) = match &self.state.phase {
            Phase::GameOver { winner } => {
                let r = if *winner == actor_id {
                    1.0_f32
                } else {
                    -1.0_f32
                };
                (r, true)
            }
            _ => (0.0_f32, false),
        };

        let (obs, mask) = if terminal {
            (vec![0.0f32; OBS_SIZE], vec![0u8; N_ACTIONS])
        } else {
            let next_actor = current_actor_id(&self.state)
                .ok_or_else(|| PyValueError::new_err("no actor after non-terminal step"))?;
            (
                build_observation(&self.state, next_actor),
                build_action_mask(&self.state, next_actor),
            )
        };

        Ok((
            obs.into_pyarray(py),
            mask.into_pyarray(py),
            reward,
            terminal,
        ))
    }

    /// Advance all non-learner seats with the strategy bot until it's the
    /// learner's turn or the game is terminal. Returns True if terminal.
    fn advance_bots(&mut self, learner: &str, difficulty: &str) -> PyResult<bool> {
        let learner_id =
            Uuid::parse_str(learner).map_err(|e| PyValueError::new_err(e.to_string()))?;
        drive_bots(&mut self.state, learner_id, parse_difficulty(difficulty))?;
        Ok(matches!(self.state.phase, Phase::GameOver { .. }))
    }

    /// Fused vs-bots step: apply `action_id` for the learner, drive all bot
    /// seats until the learner acts again or the game ends, and return
    /// `(obs, mask, reward, terminal, learner_cities, powered_now)` in one
    /// PyO3 round-trip.
    /// Reward is +1 if the learner won, -1 if anyone else did, 0 otherwise.
    /// `obs` and `mask` are zero arrays when `terminal` is True.
    /// `learner_cities` is the learner's current city count.
    /// `powered_now` is the number of cities the learner just got paid for, if
    /// this action resolved their powering for the round (PowerCities, or the
    /// PowerCitiesFuel split that completes it); 0 otherwise. Used for
    /// income-analogous reward shaping.
    #[allow(clippy::type_complexity)]
    fn step_vs_bots<'py>(
        &mut self,
        py: Python<'py>,
        learner: &str,
        action_id: u16,
        difficulty: &str,
    ) -> PyResult<(
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<u8>>,
        f32,
        bool,
        u32,
        u32,
    )> {
        let learner_id =
            Uuid::parse_str(learner).map_err(|e| PyValueError::new_err(e.to_string()))?;

        let aid = action_id as usize;
        let was_power_action = (POWER_CITIES_BASE..DISCARD_RESOURCE_BASE).contains(&aid)
            || (POWER_FUEL_BASE..N_ACTIONS).contains(&aid);

        let action = action_id_to_action(action_id, &self.state, learner_id);
        apply_action(&mut self.state, learner_id, action)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        // Powering resolves immediately unless the action paused on an
        // ambiguous hybrid fuel split for the learner.
        let power_pending = matches!(
            &self.state.phase,
            Phase::PowerCitiesFuel { player, .. } if *player == learner_id
        );
        let powered_now = if was_power_action && !power_pending {
            self.state
                .player(learner_id)
                .map(|p| p.last_cities_powered as u32)
                .unwrap_or(0)
        } else {
            0
        };

        if !matches!(self.state.phase, Phase::GameOver { .. }) {
            drive_bots(&mut self.state, learner_id, parse_difficulty(difficulty))?;
        }

        let (reward, terminal) = match &self.state.phase {
            Phase::GameOver { winner } => {
                let r = if *winner == learner_id {
                    1.0_f32
                } else {
                    -1.0_f32
                };
                (r, true)
            }
            _ => (0.0_f32, false),
        };

        let (obs, mask) = if terminal {
            (vec![0.0f32; OBS_SIZE], vec![0u8; N_ACTIONS])
        } else {
            (
                build_observation(&self.state, learner_id),
                build_action_mask(&self.state, learner_id),
            )
        };
        let learner_cities = self.state.player_cities(learner_id).len() as u32;

        Ok((
            obs.into_pyarray(py),
            mask.into_pyarray(py),
            reward,
            terminal,
            learner_cities,
            powered_now,
        ))
    }
}

fn parse_difficulty(s: &str) -> BotDifficulty {
    match s {
        "easy" => BotDifficulty::Easy,
        "hard" => BotDifficulty::Hard,
        _ => BotDifficulty::Normal,
    }
}

/// Drive every non-learner seat with the strategy bot until the learner is the
/// current actor, the game ends, or there is no single actor.
fn drive_bots(state: &mut GameState, learner: Uuid, diff: BotDifficulty) -> PyResult<()> {
    let registry = default_registry();
    for _ in 0..500 {
        if matches!(state.phase, Phase::GameOver { .. }) {
            return Ok(());
        }
        let Some(actor_id) = current_actor_id(state) else {
            return Ok(());
        };
        if actor_id == learner {
            return Ok(());
        }
        let (name, color) = {
            let player = state
                .players
                .iter()
                .find(|p| p.id == actor_id)
                .ok_or_else(|| PyValueError::new_err("bot actor not found in game"))?;
            (player.name.clone(), player.color)
        };
        let profile = registry.profile_for(diff).clone();
        let seed = actor_id.as_u128() as u64;
        let mut bot = Bot::new(actor_id, name, color, profile, seed);
        let Some(action) = bot.decide(state) else {
            return Err(PyValueError::new_err("bot has no move on its own turn"));
        };
        apply_action(state, actor_id, action)
            .map_err(|e| PyValueError::new_err(format!("bot move rejected: {}", e)))?;
    }
    Err(PyValueError::new_err("bot loop exceeded 500 iterations"))
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn powergrid_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Game>()?;
    Ok(())
}
