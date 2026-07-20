//! Serializable end-of-game report, extracted from a finished `Session`.
//!
//! This is the shared shape used to persist metrics. The lobby builds it from
//! an in-memory room; the client builds it from a local (in-process) session
//! and POSTs it to the lobby so local play is recorded too. Keeping the
//! extraction in one place means both paths capture identical fields.

use crate::Session;
use powergrid_core::rules::finish_ranks;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// A finished game and its final standings, ready to persist.
///
/// `room_name` and `started_at` are contextual and supplied by the caller when
/// turning this into a DB record — they aren't intrinsic to the session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameReport {
    pub map_name: String,
    pub rounds: i32,
    pub seats: Vec<SeatReport>,
}

/// One seat's final state in a finished game.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeatReport {
    /// The seat's player id (`Some` for humans, `None` for bots). For games
    /// recorded via the lobby's rooms this is the human's `users.id`. For local
    /// play submitted over `/games/local` it's an in-process id the server
    /// discards and re-attributes from the bearer token.
    pub user_id: Option<Uuid>,
    pub player_name: String,
    pub color: String,
    pub is_bot: bool,
    /// Difficulty for bot seats ("easy"|"normal"|"hard"|"expert"); None for humans.
    pub bot_difficulty: Option<String>,
    /// 1-based seat/turn order at game end.
    pub turn_order: i16,
    pub finish_position: i16,
    pub cities: i16,
    pub money: i32,
    pub powered: i16,
    pub plants: i16,
    // Cumulative economic activity over the whole game (from `Player::stats`).
    pub plants_bought: i32,
    pub spent_on_plants: i32,
    pub resources_bought: i32,
    pub spent_on_resources: i32,
    pub cities_bought: i32,
    pub spent_on_cities: i32,
    /// Every plant this seat still owned at game end.
    pub plant_details: Vec<PlantReport>,
}

/// One power plant held by a seat at game end.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlantReport {
    pub number: i16,
    pub kind: String,
    pub capacity: i16,
    pub resource_cost: i16,
}

/// Extract the final standings of a finished session into a `GameReport`.
///
/// Bots are identified via `session.bots`; every other seat is a human
/// (`user_id = Some(player_id)`, `is_bot = false`).
pub fn build_report(session: &Session) -> GameReport {
    let game = &session.game;
    let seats = finish_ranks(game)
        .into_iter()
        .filter_map(|(pid, position)| {
            let seat_index = game.players.iter().position(|p| p.id == pid)?;
            let player = &game.players[seat_index];
            let bot = session.bots.iter().find(|b| b.id == pid);
            let is_bot = bot.is_some();
            let plant_details = player
                .plants
                .iter()
                .map(|p| PlantReport {
                    number: p.number as i16,
                    kind: format!("{:?}", p.kind).to_lowercase(),
                    capacity: p.cities as i16,
                    resource_cost: p.cost as i16,
                })
                .collect();
            Some(SeatReport {
                user_id: (!is_bot).then_some(pid),
                player_name: player.name.clone(),
                color: format!("{:?}", player.color).to_lowercase(),
                is_bot,
                bot_difficulty: bot.map(|b| format!("{:?}", b.difficulty).to_lowercase()),
                turn_order: (seat_index + 1) as i16,
                finish_position: position as i16,
                cities: game.player_city_count(pid) as i16,
                money: player.money as i32,
                powered: player.last_cities_powered as i16,
                plants: player.plants.len() as i16,
                plants_bought: player.stats.plants_bought as i32,
                spent_on_plants: player.stats.spent_on_plants as i32,
                resources_bought: player.stats.resources_bought as i32,
                spent_on_resources: player.stats.spent_on_resources as i32,
                cities_bought: player.stats.cities_bought as i32,
                spent_on_cities: player.stats.spent_on_cities as i32,
                plant_details,
            })
        })
        .collect();

    GameReport {
        map_name: game.map.name.clone(),
        rounds: game.round as i32,
        seats,
    }
}
