use axum::extract::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tracing::{info, warn};
use uuid::Uuid;
use powergrid_core::{
    actions::{Action, ServerMessage},
    rules::apply_action,
};
use crate::SharedState;

pub async fn handle_socket(socket: WebSocket, state: SharedState) {
    let player_id = Uuid::new_v4();
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();

    // Register the client and send Welcome so the client knows its own ID.
    {
        let mut s = state.lock().await;
        s.clients.push((player_id, tx.clone()));
        info!("Client connected: {player_id}");
    }
    let welcome = serde_json::to_string(&ServerMessage::Welcome { your_id: player_id }).unwrap();
    let _ = tx.send(welcome);

    let (mut sink, mut stream) = socket.split();

    // Spawn a task to forward outbound messages to the WebSocket.
    let send_task = tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            if sink.send(Message::Text(msg.into())).await.is_err() {
                break;
            }
        }
    });

    // Handle inbound messages.
    while let Some(Ok(msg)) = stream.next().await {
        let text = match msg {
            Message::Text(t) => t.to_string(),
            Message::Close(_) => break,
            _ => continue,
        };

        let action: Action = match serde_json::from_str(&text) {
            Ok(a) => a,
            Err(e) => {
                warn!("Malformed action from {player_id}: {e}");
                send_error(&state, player_id, format!("Invalid message: {e}")).await;
                continue;
            }
        };

        info!("Action from {player_id}: {action:?}");

        let mut s = state.lock().await;
        match apply_action(&mut s.game, player_id, action) {
            Ok(()) => {
                info!("Action from {player_id} succeeded; broadcasting state to {} client(s)", s.clients.len());
                // Broadcast full state to all clients.
                let msg = serde_json::to_string(
                    &ServerMessage::StateUpdate(s.game.clone())
                ).unwrap();
                s.clients.retain(|(_, tx): &(Uuid, _)| tx.send(msg.clone()).is_ok());
            }
            Err(e) => {
                warn!("Action from {player_id} rejected: {e}");
                let err_msg = serde_json::to_string(
                    &ServerMessage::ActionError { message: e.to_string() }
                ).unwrap();
                if let Some((_, tx)) = s.clients.iter().find(|(id, _): &&(Uuid, _)| *id == player_id) {
                    let _ = tx.send(err_msg);
                }
            }
        }
    }

    // Disconnect.
    {
        let mut s = state.lock().await;
        s.clients.retain(|(id, _)| *id != player_id);
        info!("Client disconnected: {player_id}");
    }

    send_task.abort();
}

async fn send_error(state: &SharedState, player_id: Uuid, msg: String) {
    let err_msg = serde_json::to_string(&ServerMessage::ActionError { message: msg }).unwrap();
    let s = state.lock().await;
    if let Some((_, tx)) = s.clients.iter().find(|(id, _): &&(Uuid, _)| *id == player_id) {
        let _ = tx.send(err_msg);
    }
}
