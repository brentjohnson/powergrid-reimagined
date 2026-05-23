use crate::{rooms::RoomManager, ws::ConnState};
use powergrid_core::{
    actions::{LobbyAction, LobbyError, ServerMessage},
    limits::MAX_PLAYER_NAME,
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
                    conn.send_msg(&ServerMessage::LobbyError { error: e });
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
                        error: LobbyError::RoomNotFound { name },
                    });
                    return;
                }
                Some(r) => r,
            };
            if conn.current_room.is_some() {
                conn.send_msg(&ServerMessage::LobbyError {
                    error: LobbyError::AlreadyInRoom,
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

        LobbyAction::AddBot {
            bot_name,
            color,
            difficulty,
        } => {
            let room_name = match &conn.current_room {
                None => {
                    conn.send_msg(&ServerMessage::LobbyError {
                        error: LobbyError::NotInRoom,
                    });
                    return;
                }
                Some(r) => r.clone(),
            };
            let room_arc = match manager.get(&room_name).await {
                None => {
                    conn.send_msg(&ServerMessage::LobbyError {
                        error: LobbyError::RoomNotFound { name: room_name },
                    });
                    return;
                }
                Some(r) => r,
            };
            let mut room = room_arc.lock().await;
            if room.creator_user_id != conn.user_id {
                conn.send_msg(&ServerMessage::LobbyError {
                    error: LobbyError::NotHost,
                });
                return;
            }
            if !matches!(room.session.game.phase, Phase::Lobby) {
                conn.send_msg(&ServerMessage::LobbyError {
                    error: LobbyError::GameAlreadyStarted,
                });
                return;
            }
            let trimmed = bot_name.trim();
            if trimmed.is_empty() {
                conn.send_msg(&ServerMessage::LobbyError {
                    error: LobbyError::BotNameEmpty,
                });
                return;
            }
            if trimmed.chars().count() > MAX_PLAYER_NAME {
                conn.send_msg(&ServerMessage::LobbyError {
                    error: LobbyError::BotNameTooLong,
                });
                return;
            }
            let bot_name = trimmed.to_string();
            match room.add_bot(bot_name, color, difficulty) {
                Err(e) => {
                    conn.send_msg(&ServerMessage::LobbyError { error: e });
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
                        error: LobbyError::NotInRoom,
                    });
                    return;
                }
                Some(r) => r.clone(),
            };
            let room_arc = match manager.get(&room_name).await {
                None => {
                    conn.send_msg(&ServerMessage::LobbyError {
                        error: LobbyError::RoomNotFound { name: room_name },
                    });
                    return;
                }
                Some(r) => r,
            };
            let mut room = room_arc.lock().await;
            if room.creator_user_id != conn.user_id {
                conn.send_msg(&ServerMessage::LobbyError {
                    error: LobbyError::NotHost,
                });
                return;
            }
            match room.remove_bot(bot_id) {
                Err(e) => {
                    conn.send_msg(&ServerMessage::LobbyError { error: e });
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
