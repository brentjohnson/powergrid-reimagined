use crate::{
    db::Db,
    hint_handler::handle_room_hint,
    lobby_handler::{handle_lobby_action, leave_room},
    room_handler::handle_room_action,
    rooms::RoomManager,
};
use axum::extract::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt};
use powergrid_core::{
    actions::{AuthError, ClientMessage, LobbyError, ServerMessage},
    types::PlayerId,
    PROTOCOL_VERSION,
};
use std::{sync::Arc, time::Duration};
use tokio::sync::mpsc;
use tracing::{info, warn};

/// Per-connection state, populated after authentication.
pub struct ConnState {
    pub user_id: PlayerId,
    pub username: String,
    pub current_room: Option<String>,
    pub tx: mpsc::UnboundedSender<String>,
}

impl ConnState {
    pub fn send_msg(&self, msg: &ServerMessage) {
        let json = serde_json::to_string(msg).unwrap();
        let _ = self.tx.send(json);
    }

    pub fn send_raw(&self, json: &str) {
        let _ = self.tx.send(json.to_string());
    }
}

pub async fn handle_socket(
    socket: WebSocket,
    manager: Arc<RoomManager>,
    bot_delay: Duration,
    db: Db,
) {
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();
    let (mut sink, mut stream) = socket.split();

    let send_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if sink.send(Message::Text(msg)).await.is_err() {
                break;
            }
        }
    });

    // Pre-auth: expect Authenticate as the first message within 10 seconds.
    let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(10);
    let auth_result = loop {
        match tokio::time::timeout_at(deadline, stream.next()).await {
            Err(_) => {
                let _ = tx.send(
                    serde_json::to_string(&ServerMessage::AuthError {
                        error: AuthError::Timeout,
                    })
                    .unwrap(),
                );
                break None;
            }
            Ok(None) | Ok(Some(Err(_))) => break None,
            Ok(Some(Ok(Message::Close(_)))) => break None,
            Ok(Some(Ok(Message::Text(text)))) => {
                match serde_json::from_str::<ClientMessage>(&text) {
                    Ok(ClientMessage::Authenticate {
                        token,
                        protocol_version,
                    }) => {
                        if protocol_version != PROTOCOL_VERSION {
                            let _ = tx.send(
                                serde_json::to_string(&ServerMessage::AuthError {
                                    error: AuthError::VersionMismatch {
                                        server_version: PROTOCOL_VERSION,
                                        client_version: protocol_version,
                                    },
                                })
                                .unwrap(),
                            );
                            break None;
                        }
                        match db.validate_token(&token).await {
                            Ok((user_id, username)) => break Some((user_id, username)),
                            Err(_) => {
                                let _ = tx.send(
                                    serde_json::to_string(&ServerMessage::AuthError {
                                        error: AuthError::InvalidToken,
                                    })
                                    .unwrap(),
                                );
                                break None;
                            }
                        }
                    }
                    _ => {
                        let _ = tx.send(
                            serde_json::to_string(&ServerMessage::AuthError {
                                error: AuthError::AuthRequired,
                            })
                            .unwrap(),
                        );
                        break None;
                    }
                }
            }
            Ok(Some(Ok(_))) => continue,
        }
    };

    let (user_id, username) = match auth_result {
        Some(u) => u,
        None => {
            // Auth was rejected and an AuthError was queued on `tx`. Drop our
            // sender so the send task drains the channel (flushing that error to
            // the client) and exits on its own, instead of aborting mid-flush —
            // otherwise the client sees a bare connection reset with no reason
            // and silently reconnect-loops forever. Bounded so a client that
            // stopped reading can't wedge the task.
            drop(tx);
            let _ = tokio::time::timeout(Duration::from_secs(5), send_task).await;
            return;
        }
    };

    let mut conn = ConnState {
        user_id,
        username: username.clone(),
        current_room: None,
        tx: tx.clone(),
    };

    conn.send_msg(&ServerMessage::Authenticated {
        user_id,
        username: username.clone(),
        server_version: env!("CARGO_PKG_VERSION").to_string(),
        server_protocol: PROTOCOL_VERSION,
    });
    info!("Client authenticated: {user_id} ({username})");

    while let Some(Ok(msg)) = stream.next().await {
        let text = match msg {
            Message::Text(t) => t.to_string(),
            Message::Close(_) => break,
            _ => continue,
        };

        let client_msg: ClientMessage = match serde_json::from_str(&text) {
            Ok(m) => m,
            Err(e) => {
                warn!("Malformed message from {user_id}: {e}");
                conn.send_msg(&ServerMessage::LobbyError {
                    error: LobbyError::InvalidMessage {
                        message: format!("{e}"),
                    },
                });
                continue;
            }
        };

        match client_msg {
            ClientMessage::Authenticate { .. } => {
                // Already authenticated; ignore duplicate.
            }
            ClientMessage::Lobby { action } => {
                handle_lobby_action(action, &mut conn, &manager).await;
            }
            ClientMessage::Room { room, action } => {
                handle_room_action(room, action, &conn, &manager, bot_delay, &db).await;
            }
            ClientMessage::RoomHint { room, hint } => {
                handle_room_hint(room, hint, &conn, &manager).await;
            }
        }
    }

    leave_room(&mut conn, &manager).await;
    info!("Client disconnected: {user_id} ({username})");
    send_task.abort();
}
