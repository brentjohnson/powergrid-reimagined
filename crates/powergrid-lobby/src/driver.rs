use crate::rooms::Room;
use std::{collections::HashSet, sync::Arc, time::Duration};
use tokio::sync::Mutex;
use tracing::{info, warn};

const MAX_BOT_ITERATIONS: usize = 500;

/// Minimum pause after each bot move — see the mirrored `MIN_PACE` in
/// `powergrid_session::run_bot_pump`.
const MIN_PACE: Duration = Duration::from_millis(50);

/// Drive all in-process bots in `room_arc` until none has a move or the cap is hit.
/// The lock is released while pacing so humans can still receive state updates.
/// Bots that produce an invalid action are blocked for the remainder of this pump
/// invocation so a strategy bug cannot stall the game.
pub async fn run_bot_pump(room_arc: Arc<Mutex<Room>>, delay: Duration) {
    let mut failed = HashSet::new();
    for iter in 0..MAX_BOT_ITERATIONS {
        let decide_start = std::time::Instant::now();
        let next = {
            let mut room = room_arc.lock().await;
            room.session.next_bot_action(&failed)
        };

        let Some((bot_id, action)) = next else {
            return;
        };

        // Apply first (broadcasts the move), then pace — mirrors
        // `powergrid_session::run_bot_pump`.
        {
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

        // Pace as a floor, not an addend: think time counts toward `delay`, but
        // always pause at least MIN_PACE so consecutive bot moves don't batch.
        let pace = delay
            .checked_sub(decide_start.elapsed())
            .unwrap_or(Duration::ZERO)
            .max(MIN_PACE);
        tokio::time::sleep(pace).await;
    }

    let room = room_arc.lock().await;
    warn!(
        "Bot pump hit MAX_BOT_ITERATIONS ({}) in room '{}'",
        MAX_BOT_ITERATIONS, room.name
    );
}
