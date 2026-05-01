use crate::{
    lobby_handler::{handle_lobby_action, leave_room},
    room_handler::handle_room_action,
    rooms::RoomManager,
};
use axum::extract::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt};
use powergrid_core::{
    actions::{ClientMessage, ServerMessage},
    types::PlayerId,
};
use std::{sync::Arc, time::Duration};
use tokio::sync::mpsc;
use tracing::{info, warn};
use uuid::Uuid;

/// Per-connection state.
pub struct ConnState {
    pub socket_id: PlayerId,
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

pub async fn handle_socket(socket: WebSocket, manager: Arc<RoomManager>, bot_delay: Duration) {
    let socket_id: PlayerId = Uuid::new_v4();
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();

    let mut conn = ConnState {
        socket_id,
        current_room: None,
        tx: tx.clone(),
    };

    let welcome = serde_json::to_string(&ServerMessage::Welcome { your_id: socket_id }).unwrap();
    let _ = tx.send(welcome);
    info!("Client connected: {socket_id}");

    let (mut sink, mut stream) = socket.split();

    let send_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if sink.send(Message::Text(msg)).await.is_err() {
                break;
            }
        }
    });

    while let Some(Ok(msg)) = stream.next().await {
        let text = match msg {
            Message::Text(t) => t.to_string(),
            Message::Close(_) => break,
            _ => continue,
        };

        let client_msg: ClientMessage = match serde_json::from_str(&text) {
            Ok(m) => m,
            Err(e) => {
                warn!("Malformed message from {socket_id}: {e}");
                conn.send_msg(&ServerMessage::LobbyError {
                    message: format!("invalid message: {e}"),
                });
                continue;
            }
        };

        match client_msg {
            ClientMessage::Lobby(action) => {
                handle_lobby_action(action, &mut conn, &manager).await;
            }
            ClientMessage::Room { room, action } => {
                handle_room_action(room, action, &conn, &manager, bot_delay).await;
            }
        }
    }

    // Disconnect cleanup.
    leave_room(&mut conn, &manager).await;
    info!("Client disconnected: {socket_id}");
    send_task.abort();
}
