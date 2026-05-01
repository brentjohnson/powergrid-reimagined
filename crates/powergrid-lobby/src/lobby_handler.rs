use crate::{rooms::RoomManager, ws::ConnState};
use powergrid_core::{
    actions::{LobbyAction, ServerMessage},
    types::Phase,
};
use std::sync::Arc;
use tracing::info;

pub async fn handle_lobby_action(
    action: LobbyAction,
    conn: &mut ConnState,
    manager: &Arc<RoomManager>,
) {
    match action {
        LobbyAction::ListRooms => {
            let rooms = manager.list().await;
            let msg = ServerMessage::RoomList { rooms };
            conn.send_msg(&msg);
        }

        LobbyAction::CreateRoom { name } => {
            match manager.create(name.clone(), conn.socket_id).await {
                Err(e) => {
                    conn.send_msg(&ServerMessage::LobbyError { message: e });
                }
                Ok(room_arc) => {
                    conn.current_room = Some(name.to_lowercase());
                    let mut room = room_arc.lock().await;
                    room.humans.push((conn.socket_id, conn.tx.clone()));
                    let state_json = serde_json::to_string(&ServerMessage::StateUpdate(Box::new(
                        room.game.clone(),
                    )))
                    .unwrap();
                    drop(room);
                    conn.send_msg(&ServerMessage::RoomJoined {
                        room: name.clone(),
                        your_id: conn.socket_id,
                    });
                    conn.send_raw(&state_json);
                    info!(
                        "Socket {} created and joined room '{}'",
                        conn.socket_id, name
                    );
                }
            }
        }

        LobbyAction::JoinRoom { name } => {
            let room_arc = match manager.get(&name).await {
                None => {
                    conn.send_msg(&ServerMessage::LobbyError {
                        message: format!("room '{}' not found", name),
                    });
                    return;
                }
                Some(r) => r,
            };
            if conn.current_room.is_some() {
                conn.send_msg(&ServerMessage::LobbyError {
                    message: "leave your current room before joining another".to_string(),
                });
                return;
            }
            let mut room = room_arc.lock().await;
            room.humans.push((conn.socket_id, conn.tx.clone()));
            conn.current_room = Some(name.to_lowercase());
            let state_json =
                serde_json::to_string(&ServerMessage::StateUpdate(Box::new(room.game.clone())))
                    .unwrap();
            drop(room);
            conn.send_msg(&ServerMessage::RoomJoined {
                room: name.clone(),
                your_id: conn.socket_id,
            });
            conn.send_raw(&state_json);
            info!("Socket {} joined room '{}'", conn.socket_id, name);
        }

        LobbyAction::LeaveRoom => {
            leave_room(conn, manager).await;
        }

        LobbyAction::AddBot { bot_name, color } => {
            let room_name = match &conn.current_room {
                None => {
                    conn.send_msg(&ServerMessage::LobbyError {
                        message: "not in any room".to_string(),
                    });
                    return;
                }
                Some(r) => r.clone(),
            };
            let room_arc = match manager.get(&room_name).await {
                None => {
                    conn.send_msg(&ServerMessage::LobbyError {
                        message: "room no longer exists".to_string(),
                    });
                    return;
                }
                Some(r) => r,
            };
            let mut room = room_arc.lock().await;
            // Only the room creator (host) may add bots, and only in the lobby phase.
            if room.creator_socket != conn.socket_id {
                conn.send_msg(&ServerMessage::LobbyError {
                    message: "only the room host can add bots".to_string(),
                });
                return;
            }
            if !matches!(room.game.phase, Phase::Lobby) {
                conn.send_msg(&ServerMessage::LobbyError {
                    message: "cannot add bots after game has started".to_string(),
                });
                return;
            }
            match room.add_bot(bot_name, color) {
                Err(e) => {
                    conn.send_msg(&ServerMessage::LobbyError { message: e });
                }
                Ok(_) => {
                    let msg = ServerMessage::StateUpdate(Box::new(room.game.clone()));
                    room.broadcast_msg(&msg);
                }
            }
        }

        LobbyAction::RemoveBot { bot_id } => {
            let room_name = match &conn.current_room {
                None => {
                    conn.send_msg(&ServerMessage::LobbyError {
                        message: "not in any room".to_string(),
                    });
                    return;
                }
                Some(r) => r.clone(),
            };
            let room_arc = match manager.get(&room_name).await {
                None => {
                    conn.send_msg(&ServerMessage::LobbyError {
                        message: "room no longer exists".to_string(),
                    });
                    return;
                }
                Some(r) => r,
            };
            let mut room = room_arc.lock().await;
            if room.creator_socket != conn.socket_id {
                conn.send_msg(&ServerMessage::LobbyError {
                    message: "only the room host can remove bots".to_string(),
                });
                return;
            }
            match room.remove_bot(bot_id) {
                Err(e) => {
                    conn.send_msg(&ServerMessage::LobbyError { message: e });
                }
                Ok(()) => {
                    let msg = ServerMessage::StateUpdate(Box::new(room.game.clone()));
                    room.broadcast_msg(&msg);
                }
            }
        }
    }
}

/// Remove a socket from its current room. If the socket had joined as a player
/// and the game is in `Phase::Lobby`, remove the player record so their slot is freed.
/// Outside lobby, keep the record so they can reconnect.
pub async fn leave_room(conn: &mut ConnState, manager: &Arc<RoomManager>) {
    let room_name = match conn.current_room.take() {
        None => return,
        Some(r) => r,
    };
    let room_arc = match manager.get(&room_name).await {
        None => return,
        Some(r) => r,
    };
    let mut room = room_arc.lock().await;
    let socket_id = conn.socket_id;

    // Remove from human sender list.
    room.humans.retain(|(id, _)| *id != socket_id);

    // If in lobby phase, remove the player record so the color/name slot is freed.
    if matches!(room.game.phase, Phase::Lobby) {
        room.game.players.retain(|p| p.id != socket_id);
        room.game.player_order.retain(|id| *id != socket_id);
    }

    // If the creator left, assign a new creator (first remaining human, if any).
    if room.creator_socket == socket_id {
        if let Some((new_host, _)) = room.humans.first() {
            room.creator_socket = *new_host;
        }
    }

    let left_msg = serde_json::to_string(&ServerMessage::RoomLeft {
        room: room.name.clone(),
    })
    .unwrap();
    conn.send_raw(&left_msg);

    // Broadcast updated state to remaining players.
    if !room.humans.is_empty() {
        let msg = ServerMessage::StateUpdate(Box::new(room.game.clone()));
        room.broadcast_msg(&msg);
    }

    info!("Socket {} left room '{}'", socket_id, room_name);
    drop(room);
    manager.drop_if_finished(&room_name).await;
}
