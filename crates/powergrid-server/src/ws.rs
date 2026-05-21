use crate::SharedSession;
use axum::extract::ws::{Message, WebSocket};
use futures::{SinkExt, StreamExt};
use powergrid_core::actions::{Action, ServerMessage};
use powergrid_session::Subscriber;
use tokio::sync::mpsc;
use tracing::{info, warn};
use uuid::Uuid;

pub async fn handle_socket(socket: WebSocket, session: SharedSession) {
    let player_id = Uuid::new_v4();
    let (tx, mut rx) = mpsc::unbounded_channel::<String>();

    {
        let mut s = session.lock().await;
        s.add_subscriber(Subscriber::mpsc(Some(player_id), tx.clone()));
        info!("Client connected: {player_id}");
    }

    let welcome = serde_json::to_string(&ServerMessage::Welcome { your_id: player_id }).unwrap();
    let _ = tx.send(welcome);

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

        let action: Action = match serde_json::from_str(&text) {
            Ok(a) => a,
            Err(e) => {
                warn!("Malformed action from {player_id}: {e}");
                let err = serde_json::to_string(&ServerMessage::ActionError {
                    message: format!("Invalid message: {e}"),
                })
                .unwrap();
                let _ = tx.send(err);
                continue;
            }
        };

        info!("Action from {player_id}: {action:?}");

        let mut s = session.lock().await;
        if let Err(e) = s.apply(player_id, action) {
            warn!("Action from {player_id} rejected: {e}");
            let err_msg = serde_json::to_string(&ServerMessage::ActionError {
                message: e.to_string(),
            })
            .unwrap();
            // Find the subscriber index for this player and send directly.
            // Since we can't easily find by identity, send via our tx channel directly.
            let _ = tx.send(err_msg);
        } else {
            info!(
                "Action from {player_id} accepted; state broadcast to {} subscriber(s)",
                s.subscriber_count()
            );
        }
    }

    // Remove our subscriber on disconnect by retaining only live senders.
    // The mpsc tx will be dropped when this function returns, and the next
    // broadcast will prune the dead entry via the retain in Session::broadcast.
    info!("Client disconnected: {player_id}");
    send_task.abort();
}
