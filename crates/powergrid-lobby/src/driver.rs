use crate::rooms::Room;
use std::{collections::HashSet, sync::Arc, time::Duration};
use tokio::sync::Mutex;
use tracing::{info, warn};

const MAX_BOT_ITERATIONS: usize = 500;

/// Drive all in-process bots in `room_arc` until none has a move or the cap is hit.
/// The lock is released during each delay so humans can still receive state updates.
/// Bots that produce an invalid action are blocked for the remainder of this pump
/// invocation so a strategy bug cannot stall the game.
pub async fn run_bot_pump(room_arc: Arc<Mutex<Room>>, delay: Duration) {
    let mut failed = HashSet::new();
    for iter in 0..MAX_BOT_ITERATIONS {
        let next = {
            let mut room = room_arc.lock().await;
            room.session.next_bot_action(&failed)
        };

        let Some((bot_id, action)) = next else {
            return;
        };

        tokio::time::sleep(delay).await;

        let mut room = room_arc.lock().await;
        match room.session.apply(bot_id, action) {
            Ok(()) => {
                info!(
                    "Bot {} acted in room '{}' (iter {})",
                    bot_id, room.name, iter
                );
            }
            Err(e) => {
                warn!(
                    "Bot {} in room '{}' produced invalid action: {}",
                    bot_id, room.name, e
                );
                failed.insert(bot_id);
            }
        }
    }

    let room = room_arc.lock().await;
    warn!(
        "Bot pump hit MAX_BOT_ITERATIONS ({}) in room '{}'",
        MAX_BOT_ITERATIONS, room.name
    );
}
