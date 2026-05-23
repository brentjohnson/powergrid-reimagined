use crate::{driver::run_bot_pump, rooms::RoomManager, ws::ConnState};
use powergrid_core::{
    actions::{LobbyError, ServerMessage},
    Action,
};
use std::{sync::Arc, time::Duration};
use tracing::{info, warn};

pub async fn handle_room_action(
    room_name: String,
    action: Action,
    conn: &ConnState,
    manager: &Arc<RoomManager>,
    bot_delay: Duration,
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
        let res = room.session.apply(conn.user_id, action);
        if res.is_ok() {
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
}
