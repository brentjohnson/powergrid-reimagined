//! Observation / action-space encoding for the RL policy.
//!
//! This is the single Rust home of the encoding shared by the Python RL
//! environment (via `powergrid-py`) and the native Expert bot. Constants must
//! stay in sync with `python/src/powergrid_env/constants.py`; parity tests in
//! `python/tests/test_native_bridge.py` catch drift. The layout is compiled
//! against the **default (USA) map** — see [`map_matches_default`].

use powergrid_core::{
    actions::Action,
    map::Map,
    rules::effective_min_bid,
    state::GameState,
    types::{connection_cost, Phase, PlantKind, PlayerId, PlayerResources, PowerPlant, Resource},
};
use serde::Serialize;

// ---------------------------------------------------------------------------
// Observation / action-space constants — must stay in sync with constants.py
// ---------------------------------------------------------------------------

/// Sorted city ids of the default map (assets/maps/usa.toml).
pub const CITY_IDS: [&str; 49] = [
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

pub const REGION_NAMES: [&str; 7] = [
    "central",
    "east",
    "northeast",
    "northwest",
    "south",
    "southwest",
    "west",
];

pub const N_CITIES: usize = CITY_IDS.len();
pub const N_REGIONS: usize = REGION_NAMES.len();

// Observation layout: money + resources + self plants + self cities +
// opponent summary + opponent cities + city slot counts + active regions +
// actual market + future market + market meta + resource market +
// phase/step/round/end-game/turn-order scalars + phase scratch +
// per-city connection cost + opponent per-resource fuel demand + opponent plants.
pub const OBS_SIZE: usize = 1
    + 4
    + 15
    + N_CITIES
    + 20
    + 5 * N_CITIES
    + N_CITIES
    + N_REGIONS
    + 24
    + 20
    + 3
    + 4
    + 5
    + 8
    + N_CITIES // 19. connection cost from the actor's network to each city
    + 4 // 20. opponent per-resource fuel demand (coal, oil, gas, uranium)
    + 5 * 3 * 5; // 21. opponent plants (5 opp × 3 slots × 5 feats)

// Action base indices.
pub const PASS_AUCTION_IDX: usize = 0;
pub const DONE_BUYING_IDX: usize = 1;
pub const DONE_BUILDING_IDX: usize = 2;
pub const SELECT_PLANT_BASE: usize = 3;
pub const PLACE_BID_BASE: usize = 11;
/// Auction raises are English-auction style: the only representable raise is
/// +1 over the standing bid (`PassAuction` covers dropping out). This makes
/// over-bidding unrepresentable and turns each price level into a single
/// raise-vs-pass decision. See `assets/policies/expert.bin` notes in
/// `policy.rs` — this constant must match the embedded policy's output dim.
pub const N_BID_ACTIONS: usize = 1;
pub const DISCARD_PLANT_BASE: usize = PLACE_BID_BASE + N_BID_ACTIONS;
pub const BUILD_CITY_BASE: usize = DISCARD_PLANT_BASE + 3;
pub const BUY_RESOURCE_BASE: usize = BUILD_CITY_BASE + N_CITIES;
pub const POWER_CITIES_BASE: usize = BUY_RESOURCE_BASE + 4;
pub const DISCARD_RESOURCE_BASE: usize = POWER_CITIES_BASE + 8;
pub const POWER_FUEL_BASE: usize = DISCARD_RESOURCE_BASE + 9;
pub const N_ACTIONS: usize = POWER_FUEL_BASE + 9;

/// True when `map` is the default USA map the encoding was compiled against.
/// The Expert bot must fall back to the heuristic on any other map.
pub fn map_matches_default(map: &Map) -> bool {
    map.cities.len() == N_CITIES && CITY_IDS.iter().all(|id| map.cities.contains_key(*id))
}

// ---------------------------------------------------------------------------
// Legal-move info
// ---------------------------------------------------------------------------

#[derive(Default, Serialize)]
pub struct LegalMoveInfo {
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

/// The single player expected to act right now, if any. In Bureaucracy this is
/// `remaining.first()` (strict ordering), unlike the heuristic bots which act
/// whenever they are still in `remaining` — the RL policy was trained against
/// these semantics.
pub fn current_actor_id(state: &GameState) -> Option<PlayerId> {
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

pub fn compute_legal_move_info(state: &GameState, actor_id: PlayerId) -> LegalMoveInfo {
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
            plant_numbers,
            hybrid_cost,
            ..
        } => {
            if *fuel_player == actor_id {
                // Pure gas/oil plants in the same selection are paid from the
                // same pools; the hybrid split may only use what's left after
                // those — mirrors handle_power_cities_fuel's validation.
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
                let gas_after_pure = player.resources.gas.saturating_sub(pure_gas);
                let oil_after_pure = player.resources.oil.saturating_sub(pure_oil);
                for gas in 0..=*hybrid_cost {
                    let oil = hybrid_cost - gas;
                    if gas <= gas_after_pure && oil <= oil_after_pure {
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

/// Per-round fuel demand a plant places on `resource`. Mirrors
/// `features::per_round_demand` (kept local so the encoding layer doesn't depend
/// on the strategy layer). Hybrids split their firing cost across gas and oil.
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

/// Port of `encoding.py::encode_observation` — builds obs vector directly from GameState.
pub fn build_observation(state: &GameState, actor_id: PlayerId) -> Vec<f32> {
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

    // 9+10. Plant market (8 cards): chain `actual` then `future`, take 8.
    // Cards 0-3 (24 = 4 × 6): number, kind, cost, cities, present, discount.
    // Cards 4-7 (20 = 4 × 5): number, kind, cost, cities, present (no discount).
    // In steps 1/2, `actual` has exactly 4 and `future` has exactly 4, so this
    // reproduces the old per-section encoding exactly. In step 3, `future` is
    // empty and `actual` holds all 6 plants, so the 5th/6th actual plants land
    // in cards 4/5 instead of being dropped.
    let actual_base = idx;
    let future_base = idx + 24;
    for (i, plant) in state
        .market
        .actual
        .iter()
        .chain(state.market.future.iter())
        .take(8)
        .enumerate()
    {
        if i < 4 {
            let base = actual_base + i * 6;
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
        } else {
            let base = future_base + (i - 4) * 5;
            obs[base] = plant.number as f32 / 60.0;
            obs[base + 1] = plant_kind_id(plant.kind) / 6.0;
            obs[base + 2] = plant.cost as f32 / 5.0;
            obs[base + 3] = plant.cities as f32 / 8.0;
            obs[base + 4] = 1.0;
        }
    }
    idx += 24;
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

    // 19. Connection cost from the actor's network to each city (N_CITIES).
    // The Dijkstra routing cost `decide_build_cities` sorts candidates on — the
    // primary driver of build decisions and the one input no other obs section
    // exposes (the map graph is otherwise invisible to the net). 0 for cities
    // the actor already owns and, by convention, an empty network.
    let my_cities = state.player_cities(actor_id);
    let costs = state.map.connection_costs_from(&my_cities);
    for (ci, &city_id) in CITY_IDS.iter().enumerate() {
        let cost = costs.get(city_id).copied().unwrap_or(30);
        obs[idx + ci] = cost as f32 / 30.0;
    }
    idx += N_CITIES;

    // 20. Opponent per-resource fuel demand (4): total per-round fuel the
    // opponents' racks draw on each resource — the market-contention signal the
    // bot's fuel model uses (expected firing cost, feasibility, denial) but that
    // section 5 (opponent summary) omits.
    for (ri, resource) in [
        Resource::Coal,
        Resource::Oil,
        Resource::Gas,
        Resource::Uranium,
    ]
    .into_iter()
    .enumerate()
    {
        let demand: f32 = opponents
            .iter()
            .flat_map(|opp| opp.plants.iter())
            .map(|p| per_round_demand(p, resource))
            .sum();
        let denom = match resource {
            Resource::Coal => 27.0,
            Resource::Oil => 20.0,
            Resource::Gas => 24.0,
            Resource::Uranium => 12.0,
        };
        obs[idx + ri] = demand / denom;
    }
    idx += 4;

    // 21. Opponent plants (5 × 3 × 5 = 75): each opponent's rack encoded exactly
    // like section 3 (self plants) — number, kind, cost, cities, capacity.
    // Surfaces opponents' highest plant number (the turn-order tiebreaker, and
    // the sole determinant of order in round 1) and the per-plant kind/cost/
    // cities the bot's denial and fuel models read. Opponents are the same
    // players in the same order as section 5.
    for (i, opp) in opponents.iter().take(5).enumerate() {
        for (j, plant) in opp.plants.iter().take(3).enumerate() {
            let base = idx + (i * 3 + j) * 5;
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
    }
    idx += 5 * 3 * 5;

    debug_assert_eq!(idx, OBS_SIZE, "observation size mismatch");
    // Clamp into the Box bounds: a few features (e.g. player stockpiles, late
    // rounds) can exceed their nominal denominator in extreme games.
    for v in &mut obs {
        *v = v.clamp(0.0, 1.0);
    }
    obs
}

/// Port of `encoding.py::mask_from_info` — builds action mask directly from GameState.
pub fn build_action_mask(state: &GameState, actor_id: PlayerId) -> Vec<u8> {
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
        // Only raise representable is +1 over the standing bid.
        if bid_min <= bid_max {
            mask[PLACE_BID_BASE] = 1;
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
pub fn action_id_to_action(action_id: u16, state: &GameState, actor_id: PlayerId) -> Action {
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
        // Only one bid action exists: raise by +1 over the standing bid.
        if let Phase::Auction {
            active_bid: Some(bid),
            ..
        } = &state.phase
        {
            return Action::PlaceBid {
                amount: bid.amount + 1,
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
        // Additive single-unit buy: this does NOT end the buy phase (see
        // rules::handle_buy_resources), so a policy buys one unit at a time and
        // sequences as many as it wants before choosing DoneBuying. The per-step
        // mask (buyable_resources in compute_legal_move_info) already enforces
        // capacity (hybrid-aware), market availability, and affordability for
        // each +1, and the intra-turn coupling resolves automatically because
        // every unit commits before the next mask is computed.
        return Action::BuyResources {
            resource,
            amount: 1,
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

#[cfg(test)]
mod tests {
    use super::*;
    use powergrid_core::actions::ActionError;
    use powergrid_core::map::default_map;
    use powergrid_core::rules::apply_action;
    use powergrid_core::types::PlayerColor;

    /// Start a 2-player game and return (state, player ids in join order).
    fn start_game(seed: u64) -> (GameState, Vec<PlayerId>) {
        let mut state = GameState::new_with_seed(default_map(), 2, seed);
        let ids: Vec<PlayerId> = (0..2)
            .map(|i| PlayerId::from_u128(((seed as u128) << 8) | (i + 1) as u128))
            .collect();
        for (i, id) in ids.iter().enumerate() {
            apply_action(
                &mut state,
                *id,
                Action::JoinGame {
                    name: format!("P{i}"),
                    color: [PlayerColor::Red, PlayerColor::Blue][i],
                },
            )
            .expect("join");
        }
        apply_action(&mut state, ids[0], Action::StartGame).expect("start");
        (state, ids)
    }

    fn plant(number: u8, kind: PlantKind, cost: u8, cities: u8) -> PowerPlant {
        PowerPlant {
            number,
            kind,
            cost,
            cities,
        }
    }

    /// In Step 3, `market.actual` holds all 6 plants and `future` is empty.
    /// The observation must encode all 6 — not just the first 4 — by spilling
    /// the 5th/6th plants into the "future" card slots (cards 4/5).
    fn pad(mut plants: Vec<PowerPlant>) -> Vec<PowerPlant> {
        while plants.len() < 6 {
            plants.push(plant(60, PlantKind::Wind, 0, 1));
        }
        plants
    }

    #[test]
    fn step3_market_encodes_all_six_actual_plants() {
        let (mut state, ids) = start_game(42);

        state.market.in_step3 = true;
        state.market.actual = pad(vec![
            plant(3, PlantKind::Coal, 2, 1),
            plant(4, PlantKind::Oil, 2, 1),
            plant(5, PlantKind::Coal, 2, 1),
            plant(6, PlantKind::Oil, 1, 1),
            plant(7, PlantKind::GasOrOil, 3, 2),
            plant(8, PlantKind::Coal, 3, 2),
        ]);
        state.market.future = Vec::new();
        state.market.discount_token = None;

        let obs = build_observation(&state, ids[0]);

        // Cards 0-3 (indices 390..414) carry actual[0..4] as before.
        for (i, p) in state.market.actual[..4].iter().enumerate() {
            let base = 390 + i * 6;
            assert_eq!(obs[base], p.number as f32 / 60.0, "card {i} number");
            assert_eq!(obs[base + 4], 1.0, "card {i} present");
        }

        // Cards 4-5 (indices 414..424) now carry the 5th/6th actual plants,
        // using the 5-feature "future" layout (no discount feature).
        for (i, p) in state.market.actual[4..6].iter().enumerate() {
            let base = 414 + i * 5;
            assert_eq!(obs[base], p.number as f32 / 60.0, "card {} number", i + 4);
            assert_eq!(
                obs[base + 1],
                plant_kind_id(p.kind) / 6.0,
                "card {} kind",
                i + 4
            );
            assert_eq!(obs[base + 4], 1.0, "card {} present", i + 4);
        }

        // Cards 6-7 (indices 424..434) stay empty.
        for v in &obs[424..434] {
            assert_eq!(*v, 0.0, "unused card slots stay zero");
        }
    }

    #[test]
    fn steps_1_2_market_encoding_unchanged() {
        let (state, ids) = start_game(7);

        // Fresh game: 4 actual + 4 future plants, not in step 3.
        assert!(!state.market.in_step3);
        assert_eq!(state.market.actual.len(), 4);
        assert_eq!(state.market.future.len(), 4);

        let obs = build_observation(&state, ids[0]);

        for (i, p) in state.market.actual.iter().enumerate() {
            let base = 390 + i * 6;
            assert_eq!(obs[base], p.number as f32 / 60.0, "actual card {i} number");
            assert_eq!(obs[base + 4], 1.0, "actual card {i} present");
        }
        for (i, p) in state.market.future.iter().enumerate() {
            let base = 414 + i * 5;
            assert_eq!(obs[base], p.number as f32 / 60.0, "future card {i} number");
            assert_eq!(obs[base + 4], 1.0, "future card {i} present");
        }
    }

    /// Regression test: `fuel_gas` must only offer splits that
    /// `handle_power_cities_fuel` will actually accept. A pure-Oil plant
    /// fired alongside a hybrid plant consumes oil from the same pool the
    /// hybrid split draws from — `fuel_gas` must subtract that pure usage
    /// before checking feasibility, not just compare against full resources.
    ///
    /// Setup: pure-Oil #10 (cost 1), hybrid #5 (GasOrOil, cost 3).
    /// Resources: oil=3, gas=3. oil_after_pure = 3-1 = 2, gas_after_pure = 3.
    /// min_gas = 3-2 = 1, max_gas = 3 → ambiguous, gas=0 is NOT feasible
    /// (pure_oil_cost(1) + oil(3) = 4 > resources.oil(3)).
    #[test]
    fn fuel_gas_excludes_splits_invalid_after_pure_plant_usage() {
        use powergrid_core::types::Phase;

        let (mut state, ids) = start_game(99);
        let p1 = ids[0];

        state.phase = Phase::Bureaucracy {
            remaining: vec![p1],
        };
        let player = state.player_mut(p1).unwrap();
        player.plants = vec![
            plant(5, PlantKind::GasOrOil, 3, 2),
            plant(10, PlantKind::Oil, 1, 1),
        ];
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
                plant_numbers: vec![5, 10],
            },
        )
        .expect("power cities");
        assert!(
            matches!(state.phase, Phase::PowerCitiesFuel { hybrid_cost, .. } if hybrid_cost == 3),
            "expected ambiguous PowerCitiesFuel with hybrid_cost=3, got {:?}",
            state.phase
        );

        let info = compute_legal_move_info(&state, p1);
        assert_eq!(
            info.fuel_gas,
            vec![1, 2, 3],
            "gas=0 (oil=3) double-spends oil already committed to the pure-Oil plant"
        );

        // Every offered split must actually be accepted by the real handler.
        for &gas in &info.fuel_gas {
            let mut trial = state.clone();
            let oil = 3 - gas;
            apply_action(&mut trial, p1, Action::PowerCitiesFuel { gas, oil })
                .unwrap_or_else(|e| panic!("offered split gas={gas} oil={oil} rejected: {e:?}"));
        }

        // The excluded split must indeed be rejected, confirming this isn't
        // a false-negative regression.
        let mut rejected_trial = state.clone();
        let result = apply_action(
            &mut rejected_trial,
            p1,
            Action::PowerCitiesFuel { gas: 0, oil: 3 },
        );
        assert!(
            matches!(result, Err(ActionError::InvalidFuelSplit)),
            "expected gas=0 to be rejected, got {:?}",
            result
        );
    }

    /// `powergrid-py`'s `bot_decide_id` matches a heuristic bot's chosen
    /// `Action` against `action_id_to_action` over the legal ids in
    /// `build_action_mask`, comparing by JSON string. This pins the
    /// invariant that makes that match-up actually find something for most
    /// moves of a real game — the imitation-learning pipeline
    /// (`alphazero/imitation.py`) depends on it to harvest training pairs.
    #[test]
    fn hard_bot_decisions_round_trip_through_action_mask() {
        use crate::bot::Bot;
        use crate::profile::default_registry;
        use powergrid_core::types::BotDifficulty;

        let (mut state, _ids) = start_game(2024);
        let registry = default_registry();
        let mut round_tripped = 0;

        for _ in 0..500 {
            let Some(actor_id) = current_actor_id(&state) else {
                break;
            };
            let player = state
                .players
                .iter()
                .find(|p| p.id == actor_id)
                .expect("actor is a player");
            let profile = registry.profile_for(BotDifficulty::Hard).clone();
            let mut bot = Bot::new(
                actor_id,
                player.name.clone(),
                player.color,
                profile,
                actor_id.as_u128() as u64,
            );
            let Some(action) = bot.decide(&state) else {
                break;
            };

            let mask = build_action_mask(&state, actor_id);
            let chosen_json = serde_json::to_string(&action).expect("serialize action");
            let matched_id = (0..N_ACTIONS as u16).find(|&id| {
                mask[id as usize] == 1
                    && serde_json::to_string(&action_id_to_action(id, &state, actor_id))
                        .expect("serialize action")
                        == chosen_json
            });
            if let Some(id) = matched_id {
                round_tripped += 1;
                let candidate = action_id_to_action(id, &state, actor_id);
                apply_action(&mut state, actor_id, candidate)
                    .expect("round-tripped action must be legal");
            } else {
                // Not every heuristic choice is representable in the
                // action encoding (e.g. jump bids, since only +1 raises are
                // representable); apply the bot's real action directly so
                // the game still progresses.
                apply_action(&mut state, actor_id, action).expect("bot move should be legal");
            }
            if matches!(state.phase, powergrid_core::types::Phase::GameOver { .. }) {
                break;
            }
        }

        assert!(
            round_tripped > 20,
            "expected most moves of a real game to round-trip, got {round_tripped}"
        );
    }

    /// Regression test for the additive single-unit buy-resource encoding:
    /// each `BUY_RESOURCE_BASE + ri` id must apply `Action::BuyResources
    /// {resource, amount: 1}` (via `handle_buy_resources`, which does *not*
    /// pop `remaining`), so a policy can sequence several +1 buys — across
    /// several resources — in the same buy phase before ending it with
    /// `DoneBuying`. This is the fix for the structural cap where the old
    /// decode produced a single-unit `BuyResourceBatch`, which
    /// `handle_buy_resource_batch` ended the turn after unconditionally.
    #[test]
    fn buy_resource_ids_are_additive_and_dont_end_the_turn() {
        use powergrid_core::types::Phase;

        let (mut state, ids) = start_game(123);
        let p1 = ids[0];

        state.phase = Phase::BuyResources {
            remaining: vec![p1, ids[1]],
        };
        {
            let player = state.player_mut(p1).unwrap();
            // Coal plant with cost 5 -> capacity = cost * 2 = 10, plenty of
            // headroom for several +1 buys. Give ample money too.
            player.plants = vec![plant(3, PlantKind::Coal, 5, 1)];
            player.money = 200;
        }

        let coal_id = (BUY_RESOURCE_BASE) as u16; // ri=0 -> Coal
        let oil_id = (BUY_RESOURCE_BASE + 1) as u16; // ri=1 -> Oil

        // Coal: buy 3 units one id-application at a time. Each must be the
        // additive, non-turn-ending variant and must NOT advance `remaining`.
        for expected_coal in 1..=3u8 {
            let action = action_id_to_action(coal_id, &state, p1);
            assert!(
                matches!(
                    action,
                    Action::BuyResources {
                        resource: Resource::Coal,
                        amount: 1
                    }
                ),
                "expected additive 1-unit BuyResources, got {action:?}"
            );
            apply_action(&mut state, p1, action).expect("buy 1 coal");
            let player = state.player(p1).unwrap();
            assert_eq!(player.resources.coal, expected_coal);
            assert!(
                matches!(&state.phase, Phase::BuyResources { remaining } if remaining == &vec![p1, ids[1]]),
                "a single-unit buy must not advance the buy-phase turn order"
            );
        }

        // Oil plant capacity is zero (player only owns the Coal plant above),
        // so attempting to also buy oil now should be illegal via the mask —
        // confirming capacity is still enforced per-unit, hybrid-aware logic
        // notwithstanding (no GasOrOil plant in this setup).
        let mask = build_action_mask(&state, p1);
        assert_eq!(
            mask[oil_id as usize], 0,
            "no oil capacity, must be masked out"
        );

        // DoneBuying still ends this player's turn in the buy phase and
        // advances to the next player (only DoneBuying pops `remaining`).
        let action = action_id_to_action(DONE_BUYING_IDX as u16, &state, p1);
        assert!(matches!(action, Action::DoneBuying));
        apply_action(&mut state, p1, action).expect("done buying");
        assert!(
            matches!(&state.phase, Phase::BuyResources { remaining } if remaining == &vec![ids[1]]),
            "DoneBuying must advance turn order, unlike a single-unit buy"
        );
    }
}
