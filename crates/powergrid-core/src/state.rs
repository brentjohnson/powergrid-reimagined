use crate::map::Map;
use crate::types::{Phase, PlantMarket, Player, PlayerId, ResourceMarket};
use serde::{Deserialize, Serialize};

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
    /// Number of cities needed to trigger end-game (depends on player count).
    pub end_game_cities: u8,
    /// Log of recent events for display.
    pub event_log: Vec<String>,
    /// Optional RNG seed for deterministic play (used in tests; `None` = entropy).
    #[serde(default)]
    pub rng_seed: Option<u64>,
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
            end_game_cities,
            event_log: Vec::new(),
            rng_seed,
        }
    }

    pub fn player(&self, id: PlayerId) -> Option<&Player> {
        self.players.iter().find(|p| p.id == id)
    }

    pub fn player_mut(&mut self, id: PlayerId) -> Option<&mut Player> {
        self.players.iter_mut().find(|p| p.id == id)
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
}
