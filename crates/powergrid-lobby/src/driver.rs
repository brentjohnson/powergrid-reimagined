use crate::rooms::Room;
use powergrid_bot::strategy;
use powergrid_core::{actions::ServerMessage, rules::apply_action};
use std::{sync::Arc, time::Duration};
use tokio::sync::Mutex;
use tracing::{info, warn};

const MAX_BOT_ITERATIONS: usize = 50;

/// Drive all in-process bots in `room_arc` until none has a move or the cap is hit.
/// The lock is released during the delay so other rooms aren't blocked.
pub async fn run_bot_pump(room_arc: Arc<Mutex<Room>>, delay: Duration) {
    for iter in 0..MAX_BOT_ITERATIONS {
        // Find the next bot that wants to act (read-only under lock).
        let next = {
            let room = room_arc.lock().await;
            room.bots
                .iter()
                .find_map(|b| strategy::decide(&room.game, b.id).map(|a| (b.id, a)))
        };

        let Some((bot_id, action)) = next else {
            return; // no bot has a move; pump is idle
        };

        // Release the lock during the delay so humans can still receive state updates.
        tokio::time::sleep(delay).await;

        let mut room = room_arc.lock().await;
        match apply_action(&mut room.game, bot_id, action) {
            Ok(()) => {
                info!(
                    "Bot {} acted in room '{}' (iter {})",
                    bot_id, room.name, iter
                );
                let msg = ServerMessage::StateUpdate(Box::new(room.game.clone()));
                room.broadcast_msg(&msg);
            }
            Err(e) => {
                warn!(
                    "Bot {} in room '{}' produced invalid action: {}",
                    bot_id, room.name, e
                );
                // Don't break the pump — try the next bot on the next iteration.
            }
        }
    }

    let room = room_arc.lock().await;
    warn!(
        "Bot pump hit MAX_BOT_ITERATIONS ({}) in room '{}'",
        MAX_BOT_ITERATIONS, room.name
    );
}
