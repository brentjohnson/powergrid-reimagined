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
            conn.send_msg(&ServerMessage::RoomList { rooms });
        }

        LobbyAction::CreateRoom { name } => {
            match manager.create(name.clone(), conn.user_id).await {
                Err(e) => {
                    conn.send_msg(&ServerMessage::LobbyError { message: e });
                }
                Ok(room_arc) => {
                    conn.current_room = Some(name.to_lowercase());
                    let mut room = room_arc.lock().await;
                    room.add_human(conn.user_id, conn.tx.clone());
                    let map = Box::new(room.session.game.map.clone());
                    let state_json = serde_json::to_string(&ServerMessage::StateUpdate(Box::new(
                        room.session.game.view(),
                    )))
                    .unwrap();
                    drop(room);
                    conn.send_msg(&ServerMessage::RoomJoined {
                        room: name.clone(),
                        your_id: conn.user_id,
                        map,
                    });
                    conn.send_raw(&state_json);
                    info!(
                        "User {} ({}) created and joined room '{}'",
                        conn.user_id, conn.username, name
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

            // Reconnect: if the user already has a seat, replace their sender.
            if room.humans.iter().any(|(id, _)| *id == conn.user_id) {
                room.replace_human(conn.user_id, conn.tx.clone());
                conn.current_room = Some(name.to_lowercase());
                let map = Box::new(room.session.game.map.clone());
                let state_json = serde_json::to_string(&ServerMessage::StateUpdate(Box::new(
                    room.session.game.view(),
                )))
                .unwrap();
                drop(room);
                conn.send_msg(&ServerMessage::RoomJoined {
                    room: name.clone(),
                    your_id: conn.user_id,
                    map,
                });
                conn.send_raw(&state_json);
                info!(
                    "User {} ({}) reconnected to room '{}'",
                    conn.user_id, conn.username, name
                );
                return;
            }

            room.add_human(conn.user_id, conn.tx.clone());
            conn.current_room = Some(name.to_lowercase());
            let map = Box::new(room.session.game.map.clone());
            let state_json = serde_json::to_string(&ServerMessage::StateUpdate(Box::new(
                room.session.game.view(),
            )))
            .unwrap();
            drop(room);
            conn.send_msg(&ServerMessage::RoomJoined {
                room: name.clone(),
                your_id: conn.user_id,
                map,
            });
            conn.send_raw(&state_json);
            info!(
                "User {} ({}) joined room '{}'",
                conn.user_id, conn.username, name
            );
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
            if room.creator_user_id != conn.user_id {
                conn.send_msg(&ServerMessage::LobbyError {
                    message: "only the room host can add bots".to_string(),
                });
                return;
            }
            if !matches!(room.session.game.phase, Phase::Lobby) {
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
                    let msg = ServerMessage::StateUpdate(Box::new(room.session.game.view()));
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
            if room.creator_user_id != conn.user_id {
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
                    let msg = ServerMessage::StateUpdate(Box::new(room.session.game.view()));
                    room.broadcast_msg(&msg);
                }
            }
        }
    }
}

/// Remove a user from their current room.
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
    let user_id = conn.user_id;

    room.humans.retain(|(id, _)| *id != user_id);

    if matches!(room.session.game.phase, Phase::Lobby) {
        room.session.game.players.retain(|p| p.id != user_id);
        room.session.game.player_order.retain(|id| *id != user_id);
    }

    if room.creator_user_id == user_id {
        if let Some((new_host, _)) = room.humans.first() {
            room.creator_user_id = *new_host;
        }
    }

    conn.send_msg(&ServerMessage::RoomLeft {
        room: room.name.clone(),
    });

    if !room.humans.is_empty() {
        let msg = ServerMessage::StateUpdate(Box::new(room.session.game.view()));
        room.broadcast_msg(&msg);
    }

    info!(
        "User {} ({}) left room '{}'",
        user_id, conn.username, room_name
    );
    drop(room);
    manager.drop_if_finished(&room_name).await;
}
