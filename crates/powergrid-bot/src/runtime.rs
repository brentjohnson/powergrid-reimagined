use futures_util::{SinkExt, StreamExt};
use powergrid_core::{
    actions::{Action, ServerMessage},
    types::{PlayerColor, PlayerId},
};
use tokio::net::TcpStream;
use tokio_tungstenite::{
    connect_async, tungstenite::Message as WsMessage, MaybeTlsStream, WebSocketStream,
};
use tracing::{error, info, warn};

use crate::strategy;

pub async fn run_bot(url: String, name: String, color: PlayerColor) {
    loop {
        match connect_async(&url).await {
            Ok((stream, _)) => {
                info!("Connected to {url}");
                match bot_session(stream, &name, color).await {
                    SessionResult::GameOver => {
                        info!("Game over — exiting");
                        return;
                    }
                    SessionResult::Disconnected => {
                        warn!("Disconnected — reconnecting in 3s…");
                    }
                }
            }
            Err(e) => {
                warn!("Connection failed: {e} — retrying in 3s…");
            }
        }
        tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
    }
}

enum SessionResult {
    GameOver,
    Disconnected,
}

async fn bot_session(
    stream: WebSocketStream<MaybeTlsStream<TcpStream>>,
    name: &str,
    color: PlayerColor,
) -> SessionResult {
    let (mut write, mut read) = stream.split();
    let mut my_id: Option<PlayerId> = None;

    while let Some(msg) = read.next().await {
        let text = match msg {
            Ok(WsMessage::Text(t)) => t,
            Ok(WsMessage::Ping(_) | WsMessage::Pong(_)) => continue,
            Ok(WsMessage::Close(_)) | Err(_) => return SessionResult::Disconnected,
            Ok(_) => continue,
        };

        let server_msg = match serde_json::from_str::<ServerMessage>(&text) {
            Ok(m) => m,
            Err(e) => {
                warn!("Deserialize error: {e}");
                continue;
            }
        };

        match server_msg {
            ServerMessage::Welcome { your_id } => {
                my_id = Some(your_id);
                info!("Received Welcome as {your_id}; sending JoinGame");
                let action = Action::JoinGame {
                    name: name.to_string(),
                    color,
                };
                if write
                    .send(WsMessage::Text(
                        serde_json::to_string(&action).expect("serialize"),
                    ))
                    .await
                    .is_err()
                {
                    return SessionResult::Disconnected;
                }
            }

            ServerMessage::StateUpdate(gs) => {
                let Some(id) = my_id else { continue };

                if let powergrid_core::types::Phase::GameOver { winner } = gs.phase {
                    if let Some(winner_player) = gs.player(winner) {
                        info!(
                            "Game over! Winner: {} ({:?})",
                            winner_player.name, winner_player.color
                        );
                    }
                    return SessionResult::GameOver;
                }

                tokio::time::sleep(tokio::time::Duration::from_millis(800)).await;

                if let Some(action) = strategy::decide(&gs, id) {
                    info!("Sending action: {:?}", action);
                    if write
                        .send(WsMessage::Text(
                            serde_json::to_string(&action).expect("serialize"),
                        ))
                        .await
                        .is_err()
                    {
                        return SessionResult::Disconnected;
                    }
                }
            }

            ServerMessage::ActionError { message } => {
                error!("Action rejected by server: {message}");
            }

            ServerMessage::Event { message } => {
                info!("Game event: {message}");
            }

            _ => {}
        }
    }

    SessionResult::Disconnected
}
