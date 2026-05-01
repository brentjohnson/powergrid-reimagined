use crate::{driver::run_bot_pump, rooms::RoomManager, ws::ConnState};
use powergrid_core::{actions::ServerMessage, rules::apply_action, Action};
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
                message: format!("room '{}' not found", room_name),
            });
            return;
        }
        Some(r) => r,
    };

    // Verify the socket is actually a member of this room.
    {
        let room = room_arc.lock().await;
        if !room.humans.iter().any(|(id, _)| *id == conn.socket_id) {
            conn.send_msg(&ServerMessage::LobbyError {
                message: format!("you are not in room '{}'", room_name),
            });
            return;
        }
    }

    // Apply the human's action.
    let apply_result = {
        let mut room = room_arc.lock().await;
        let result = apply_action(&mut room.game, conn.socket_id, action);
        if result.is_ok() {
            info!(
                "Action from {} accepted in room '{}'",
                conn.socket_id, room.name
            );
            let msg = ServerMessage::StateUpdate(Box::new(room.game.clone()));
            room.broadcast_msg(&msg);
        } else if let Err(ref e) = result {
            warn!(
                "Action from {} rejected in room '{}': {}",
                conn.socket_id, room.name, e
            );
        }
        result
    };

    if let Err(e) = apply_result {
        conn.send_msg(&ServerMessage::ActionError {
            message: e.to_string(),
        });
        return;
    }

    // Drive bots after every successful human action.
    run_bot_pump(Arc::clone(&room_arc), bot_delay).await;
}
