use crate::{
    db::{Db, GameRecord, PlantRecord, SeatRecord},
    driver::run_bot_pump,
    rooms::{Room, RoomManager},
    ws::ConnState,
};
use powergrid_core::{
    actions::{LobbyError, ServerMessage},
    rules::finish_ranks,
    types::Phase,
    Action,
};
use std::{sync::Arc, time::Duration};
use tokio::sync::Mutex;
use tracing::{error, info, warn};

pub async fn handle_room_action(
    room_name: String,
    action: Action,
    conn: &ConnState,
    manager: &Arc<RoomManager>,
    bot_delay: Duration,
    db: &Db,
) {
    let room_arc = match manager.get(&room_name).await {
        None => {
            conn.send_msg(&ServerMessage::LobbyError {
                error: LobbyError::RoomNotFound {
                    name: room_name.clone(),
                },
            });
            return;
        }
        Some(r) => r,
    };

    // Verify membership.
    {
        let room = room_arc.lock().await;
        if !room.humans.iter().any(|(id, _)| *id == conn.user_id) {
            conn.send_msg(&ServerMessage::LobbyError {
                error: LobbyError::NotMember,
            });
            return;
        }
    }

    // Apply via session (broadcasts StateUpdate on success).
    let result = {
        let mut room = room_arc.lock().await;
        let was_lobby = matches!(room.session.game.phase, Phase::Lobby);
        let res = room.session.apply(conn.user_id, action);
        if res.is_ok() {
            // Record the moment the game leaves the Lobby phase.
            if was_lobby && !matches!(room.session.game.phase, Phase::Lobby) {
                room.started_at = Some(chrono::Utc::now());
            }
            info!(
                "Action from {} accepted in room '{}'",
                conn.user_id, room.name
            );
        } else if let Err(ref e) = res {
            warn!(
                "Action from {} rejected in room '{}': {}",
                conn.user_id, room.name, e
            );
        }
        res
    };

    if let Err(e) = result {
        conn.send_msg(&ServerMessage::ActionError { error: e });
        return;
    }

    run_bot_pump(Arc::clone(&room_arc), bot_delay).await;

    // Persist standings if this action (human or a subsequent bot move) ended
    // the game. Fires at most once per game via the room's guard flag.
    maybe_record_result(&room_arc, db).await;
}

/// Record a finished game's standings to the database exactly once.
async fn maybe_record_result(room_arc: &Arc<Mutex<Room>>, db: &Db) {
    let record = {
        let mut room = room_arc.lock().await;
        if !room.is_game_over() || room.results_recorded {
            return;
        }
        room.results_recorded = true;
        build_game_record(&room)
    };

    if let Err(e) = db.record_game(&record).await {
        error!("Failed to record finished game '{}': {e}", record.room_name);
    } else {
        info!("Recorded finished game '{}'", record.room_name);
    }
}

/// Build a `GameRecord` from a finished room's game state.
fn build_game_record(room: &Room) -> GameRecord {
    let game = &room.session.game;
    let seats = finish_ranks(game)
        .into_iter()
        .filter_map(|(pid, position)| {
            let seat_index = game.players.iter().position(|p| p.id == pid)?;
            let player = &game.players[seat_index];
            let bot = room.session.bots.iter().find(|b| b.id == pid);
            let is_bot = bot.is_some();
            let plant_details = player
                .plants
                .iter()
                .map(|p| PlantRecord {
                    number: p.number as i16,
                    kind: format!("{:?}", p.kind).to_lowercase(),
                    capacity: p.cities as i16,
                    resource_cost: p.cost as i16,
                })
                .collect();
            Some(SeatRecord {
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

    GameRecord {
        room_name: room.name.clone(),
        map_name: game.map.name.clone(),
        started_at: room.started_at,
        rounds: game.round as i32,
        seats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rooms::Room;
    use powergrid_core::{
        map::default_map,
        rules::apply_action,
        types::{BotDifficulty, PlayerColor},
    };
    use powergrid_session::Session;

    /// Drive an all-bot game to completion, then verify `build_game_record`
    /// extracts full standings — including the human-vs-bot distinction (we pop
    /// one bot so its seat is treated as a human before recording).
    #[test]
    fn build_game_record_from_finished_game() {
        let mut session = Session::new(default_map(), 6);
        let colors = [
            PlayerColor::Red,
            PlayerColor::Blue,
            PlayerColor::Green,
            PlayerColor::Yellow,
        ];
        for (i, color) in colors.iter().enumerate() {
            session
                .add_bot(format!("Bot{i}"), *color, BotDifficulty::Normal)
                .expect("add bot");
        }
        let starter = session.bots[0].id;
        apply_action(&mut session.game, starter, Action::StartGame).expect("start");

        // Drive every seat via its bot until the game ends (cap guards against a
        // stall; a heuristic game ends well under this).
        for _ in 0..8000 {
            if matches!(session.game.phase, Phase::GameOver { .. }) {
                break;
            }
            let mut decision = None;
            for i in 0..session.bots.len() {
                let id = session.bots[i].id;
                if let Some(action) = session.bots[i].decide(&session.game) {
                    decision = Some((id, action));
                    break;
                }
            }
            match decision {
                Some((id, action)) => apply_action(&mut session.game, id, action).expect("move"),
                None => panic!("no bot could move before game over"),
            }
        }
        assert!(
            matches!(session.game.phase, Phase::GameOver { .. }),
            "game did not finish within the cap"
        );

        // Treat the last seat as a human: remove its bot so `build_game_record`
        // exercises the human branch (user_id = Some(pid), is_bot = false).
        let human = session.bots.pop().expect("a bot to pop").id;

        let mut room = Room::new("test-room".into(), default_map(), starter);
        room.session = session;
        room.started_at = Some(chrono::Utc::now());

        let rec = build_game_record(&room);

        assert_eq!(rec.room_name, "test-room");
        assert_eq!(rec.map_name, "USA");
        assert_eq!(rec.seats.len(), 4);
        assert!(rec.rounds >= 1);

        // Positions are exactly 1..=4, contiguous, no duplicates.
        let mut positions: Vec<i16> = rec.seats.iter().map(|s| s.finish_position).collect();
        positions.sort_unstable();
        assert_eq!(positions, vec![1, 2, 3, 4]);

        // The popped seat records as a human with its id; the rest as bots.
        let human_seat = rec
            .seats
            .iter()
            .find(|s| s.user_id == Some(human))
            .expect("human seat present");
        assert!(!human_seat.is_bot);
        assert!(human_seat.bot_difficulty.is_none());
        assert_eq!(rec.seats.iter().filter(|s| s.is_bot).count(), 3);
        assert!(rec
            .seats
            .iter()
            .filter(|s| s.is_bot)
            .all(|s| s.user_id.is_none()));

        // Bot seats carry their difficulty ("normal", as added above).
        assert!(rec
            .seats
            .iter()
            .filter(|s| s.is_bot)
            .all(|s| s.bot_difficulty.as_deref() == Some("normal")));

        // turn_order is a permutation of 1..=4 (one per stable seat index).
        let mut orders: Vec<i16> = rec.seats.iter().map(|s| s.turn_order).collect();
        orders.sort_unstable();
        assert_eq!(orders, vec![1, 2, 3, 4]);

        // Plant details are captured: a seat's `plants` count matches its
        // recorded plant list, and a finished game leaves plants on the board.
        assert!(rec
            .seats
            .iter()
            .all(|s| s.plant_details.len() == s.plants as usize));
        assert!(rec.seats.iter().any(|s| !s.plant_details.is_empty()));

        // Cumulative economy stats accrue over a full game: every seat bought
        // plants and built cities (both cost money), and totals are positive.
        for s in &rec.seats {
            assert!(s.plants_bought > 0, "seat bought no plants");
            assert!(s.spent_on_plants > 0, "seat spent nothing on plants");
            assert!(s.cities_bought > 0, "seat built no cities");
            assert!(s.spent_on_cities > 0, "seat spent nothing on cities");
            // A seat holds at most its cumulative purchases (some may be discarded).
            assert!(s.plants as i32 <= s.plants_bought);
            assert!(s.cities as i32 <= s.cities_bought);
        }
    }
}
