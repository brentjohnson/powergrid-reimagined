use crate::map::Map;
use crate::types::{Phase, PlantMarket, Player, PlayerId, PowerPlant, ResourceMarket};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameState {
    pub phase: Phase,
    pub players: Vec<Player>,
    /// Turn order (indices into `players`), recalculated each round.
    pub player_order: Vec<PlayerId>,
    pub market: PlantMarket,
    pub resources: ResourceMarket,
    pub map: Map,
    pub round: u32,
    /// Current game step (1, 2, or 3). Starts at 1; Step 2 begins when any player builds their 7th city.
    #[serde(default = "default_step")]
    pub step: u8,
    /// Number of cities needed to trigger end-game (depends on player count).
    pub end_game_cities: u8,
    /// Log of recent events for display.
    pub event_log: Vec<String>,
    /// RNG seed — never sent over the wire.
    #[serde(skip)]
    pub rng_seed: Option<u64>,
    /// Regions active for this game (subset of map.regions, chosen at game start).
    /// Empty during Lobby (before region selection).
    #[serde(default)]
    pub active_regions: Vec<String>,
}

// ---------------------------------------------------------------------------
// Wire-safe view types (no hidden information)
// ---------------------------------------------------------------------------

/// The plant market as seen by clients: deck contents are hidden, only the
/// count of remaining cards is sent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlantMarketView {
    pub actual: Vec<PowerPlant>,
    pub future: Vec<PowerPlant>,
    pub deck_remaining: usize,
    pub step3_triggered: bool,
    pub in_step3: bool,
}

/// A wire-safe projection of `GameState`. Strips hidden information:
/// - `rng_seed` (never sent)
/// - `PlantMarket.deck`, `plant_13`, `below_step3` (face-down cards)
/// - `map` (sent once on `RoomJoined`; only mutable `city_owners` is included)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameStateView {
    pub phase: Phase,
    pub players: Vec<Player>,
    pub player_order: Vec<PlayerId>,
    pub market: PlantMarketView,
    pub resources: ResourceMarket,
    /// City owners keyed by city ID. Only cities with at least one owner are included.
    pub city_owners: HashMap<String, Vec<PlayerId>>,
    pub round: u32,
    #[serde(default = "default_step")]
    pub step: u8,
    pub end_game_cities: u8,
    pub event_log: Vec<String>,
    #[serde(default)]
    pub active_regions: Vec<String>,
}

impl GameStateView {
    pub fn player(&self, id: PlayerId) -> Option<&Player> {
        self.players.iter().find(|p| p.id == id)
    }

    pub fn host_id(&self) -> Option<PlayerId> {
        self.players.first().map(|p| p.id)
    }

    pub fn is_city_active(&self, city_id: &str, map: &Map) -> bool {
        if self.active_regions.is_empty() {
            return true;
        }
        map.cities
            .get(city_id)
            .map(|c| self.active_regions.contains(&c.region))
            .unwrap_or(false)
    }

    /// Reconstruct a full `GameState` from this view using a static map.
    /// The deck is empty (not transmitted); `rng_seed` is set to `None`.
    pub fn into_game_state(self, map: &Map) -> GameState {
        let mut game_map = map.clone();
        for city in game_map.cities.values_mut() {
            city.owners = self.city_owners.get(&city.id).cloned().unwrap_or_default();
        }
        GameState {
            phase: self.phase,
            players: self.players,
            player_order: self.player_order,
            market: PlantMarket {
                actual: self.market.actual,
                future: self.market.future,
                deck: Vec::new(),
                plant_13: None,
                below_step3: None,
                step3_triggered: self.market.step3_triggered,
                in_step3: self.market.in_step3,
            },
            resources: self.resources,
            map: game_map,
            round: self.round,
            step: self.step,
            end_game_cities: self.end_game_cities,
            event_log: self.event_log,
            rng_seed: None,
            active_regions: self.active_regions,
        }
    }
}

fn default_step() -> u8 {
    1
}

impl GameState {
    pub fn new(map: Map, player_count: usize) -> Self {
        Self::new_inner(map, player_count, None)
    }

    pub fn new_with_seed(map: Map, player_count: usize, seed: u64) -> Self {
        Self::new_inner(map, player_count, Some(seed))
    }

    fn new_inner(map: Map, player_count: usize, rng_seed: Option<u64>) -> Self {
        let end_game_cities = match player_count {
            2 => 21,
            3 => 17,
            4 => 17,
            5 => 15,
            _ => 14, // 6 players
        };

        Self {
            phase: Phase::Lobby,
            players: Vec::new(),
            player_order: Vec::new(),
            market: crate::rules::build_plant_deck(),
            resources: ResourceMarket::initial(),
            map,
            round: 0,
            step: 1,
            end_game_cities,
            event_log: Vec::new(),
            rng_seed,
            active_regions: Vec::new(),
        }
    }

    pub fn player(&self, id: PlayerId) -> Option<&Player> {
        self.players.iter().find(|p| p.id == id)
    }

    pub fn player_mut(&mut self, id: PlayerId) -> Option<&mut Player> {
        self.players.iter_mut().find(|p| p.id == id)
    }

    /// Returns true if the city's region is active (or if no regions have been selected yet).
    pub fn is_city_active(&self, city_id: &str) -> bool {
        if self.active_regions.is_empty() {
            return true;
        }
        self.map
            .cities
            .get(city_id)
            .map(|c| self.active_regions.contains(&c.region))
            .unwrap_or(false)
    }

    pub fn host_id(&self) -> Option<PlayerId> {
        self.players.first().map(|p| p.id)
    }

    pub fn log(&mut self, msg: impl Into<String>) {
        let msg = msg.into();
        self.event_log.push(msg);
        if self.event_log.len() > 50 {
            self.event_log.remove(0);
        }
    }

    /// Produce a wire-safe view: strips hidden information (deck, seed, full map).
    pub fn view(&self) -> GameStateView {
        GameStateView {
            phase: self.phase.clone(),
            players: self.players.clone(),
            player_order: self.player_order.clone(),
            market: PlantMarketView {
                actual: self.market.actual.clone(),
                future: self.market.future.clone(),
                deck_remaining: self.market.deck.len(),
                step3_triggered: self.market.step3_triggered,
                in_step3: self.market.in_step3,
            },
            resources: self.resources.clone(),
            city_owners: self
                .map
                .cities
                .iter()
                .filter(|(_, c)| !c.owners.is_empty())
                .map(|(id, c)| (id.clone(), c.owners.clone()))
                .collect(),
            round: self.round,
            step: self.step,
            end_game_cities: self.end_game_cities,
            event_log: self.event_log.clone(),
            active_regions: self.active_regions.clone(),
        }
    }
}
