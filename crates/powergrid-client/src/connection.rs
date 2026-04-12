use futures::{SinkExt, StreamExt};
use iced::futures::channel::mpsc;
use tokio_tungstenite::{connect_async, tungstenite::Message};
use powergrid_core::actions::{Action, ServerMessage};
use tracing::{debug, warn};

#[derive(Debug, Clone)]
pub enum WsEvent {
    Connected(mpsc::Sender<Action>),
    MessageReceived(ServerMessage),
    Disconnected,
}

pub fn connect(url: String) -> iced::Subscription<WsEvent> {
    iced::Subscription::run_with_id(
        url.clone(),
        // Wrap in `once(...).flatten()` so the worker is spawned lazily — only
        // when iced actually starts this subscription instance.  Without this,
        // the spawn fires on every render (each call to `subscription()`), and
        // any worker whose receiver was discarded by iced's dedup logic
        // immediately connects then disconnects (phantom pairs in the log).
        futures::stream::once(async move {
            let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel::<WsEvent>();
            tokio::spawn(ws_worker(url, event_tx));
            tokio_stream::wrappers::UnboundedReceiverStream::new(event_rx)
        })
        .flatten(),
    )
}

async fn ws_worker(
    url: String,
    event_tx: tokio::sync::mpsc::UnboundedSender<WsEvent>,
) {
    loop {
        let ws_stream = match connect_async(&url).await {
            Ok((s, _)) => s,
            Err(_) => {
                let _ = event_tx.send(WsEvent::Disconnected);
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                continue;
            }
        };

        let (mut write, mut read) = ws_stream.split();
        let (action_tx, mut action_rx) = mpsc::channel::<Action>(32);

        let _ = event_tx.send(WsEvent::Connected(action_tx));

        loop {
            tokio::select! {
                Some(action) = action_rx.next() => {
                    let json = serde_json::to_string(&action).unwrap();
                    if write.send(Message::Text(json.into())).await.is_err() {
                        break;
                    }
                }
                msg = read.next() => {
                    match msg {
                        Some(Ok(Message::Text(text))) => {
                            if let Ok(server_msg) = serde_json::from_str::<ServerMessage>(&text) {
                                if event_tx.send(WsEvent::MessageReceived(server_msg)).is_err() {
                                    return;
                                }
                            }
                        }
                        // Pings are auto-replied by tungstenite; pongs are no-ops.
                        Some(Ok(Message::Ping(_))) | Some(Ok(Message::Pong(_))) => {}
                        Some(Ok(Message::Close(frame))) => {
                            debug!("Server closed connection: {frame:?}");
                            break;
                        }
                        Some(Ok(_)) => {} // Binary / Frame — ignore
                        Some(Err(e)) => {
                            warn!("WebSocket error: {e}");
                            break;
                        }
                        None => break, // stream ended
                    }
                }
            }
        }

        let _ = event_tx.send(WsEvent::Disconnected);
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    }
}
