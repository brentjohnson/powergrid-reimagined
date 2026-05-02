use bevy::prelude::*;
use crossbeam_channel::{Receiver, Sender};
use futures_util::{SinkExt, StreamExt};
use powergrid_core::actions::{ClientMessage, LobbyAction, ServerMessage};
use tokio_tungstenite::{connect_async, tungstenite::Message as WsMessage};
use tracing::{debug, warn};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

pub enum WsEvent {
    Connected,
    MessageReceived(ServerMessage),
    Disconnected,
}

#[derive(Resource)]
pub struct WsChannels {
    pub event_rx: Receiver<WsEvent>,
    /// Send any ClientMessage (lobby or room-scoped game action) to the server.
    pub action_tx: Sender<ClientMessage>,
}

impl WsChannels {
    pub fn send_lobby(&self, action: LobbyAction) {
        self.action_tx.send(ClientMessage::Lobby(action)).ok();
    }

    pub fn send_room(&self, room: &str, action: powergrid_core::Action) {
        self.action_tx
            .send(ClientMessage::Room {
                room: room.to_string(),
                action,
            })
            .ok();
    }
}

// ---------------------------------------------------------------------------
// Public: spawn the WS worker thread and return channel handles
// ---------------------------------------------------------------------------

pub fn spawn_ws(url: String) -> WsChannels {
    let (event_tx, event_rx) = crossbeam_channel::unbounded::<WsEvent>();
    let (action_tx, action_rx) = crossbeam_channel::unbounded::<ClientMessage>();

    std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime")
            .block_on(ws_worker(url, event_tx, action_rx));
    });

    WsChannels {
        event_rx,
        action_tx,
    }
}

// ---------------------------------------------------------------------------
// Async worker — reconnects forever
// ---------------------------------------------------------------------------

async fn ws_worker(url: String, event_tx: Sender<WsEvent>, action_rx: Receiver<ClientMessage>) {
    loop {
        let ws_stream = match connect_async(&url).await {
            Ok((s, _)) => s,
            Err(e) => {
                warn!("WS connect failed ({url}): {e}");
                let _ = event_tx.send(WsEvent::Disconnected);
                tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
                continue;
            }
        };

        debug!("WS connected to {url}");
        let _ = event_tx.send(WsEvent::Connected);
        let (mut write, mut read) = ws_stream.split();

        'inner: loop {
            tokio::select! {
                msg = read.next() => {
                    match msg {
                        Some(Ok(WsMessage::Text(text))) => {
                            match serde_json::from_str::<ServerMessage>(&text) {
                                Ok(m) => {
                                    if event_tx.send(WsEvent::MessageReceived(m)).is_err() {
                                        return; // receiver dropped — app exiting
                                    }
                                }
                                Err(e) => warn!("WS deserialize error: {e}"),
                            }
                        }
                        Some(Ok(WsMessage::Ping(_) | WsMessage::Pong(_))) => {}
                        Some(Ok(WsMessage::Close(frame))) => {
                            debug!("WS close: {frame:?}");
                            break 'inner;
                        }
                        Some(Ok(_)) => {}
                        Some(Err(e)) => {
                            warn!("WS error: {e}");
                            break 'inner;
                        }
                        None => break 'inner,
                    }
                }
                // Poll outbound action queue every 16 ms
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(16)) => {
                    while let Ok(msg) = action_rx.try_recv() {
                        let json = serde_json::to_string(&msg).expect("serialize message");
                        if write.send(WsMessage::Text(json)).await.is_err() {
                            break 'inner;
                        }
                    }
                }
            }
        }

        debug!("WS disconnected, reconnecting in 2s…");
        let _ = event_tx.send(WsEvent::Disconnected);
        tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
    }
}

// ---------------------------------------------------------------------------
// Bevy system: drain the channel each frame and update AppState
// ---------------------------------------------------------------------------

pub fn process_ws_events(
    mut state: ResMut<crate::state::AppState>,
    channels: Option<Res<WsChannels>>,
) {
    let Some(channels) = channels else { return };

    while let Ok(event) = channels.event_rx.try_recv() {
        match event {
            WsEvent::Connected => {
                state.connected = true;
                // Send authentication token as the very first message.
                if let Some(token) = state.auth_token.clone() {
                    channels
                        .action_tx
                        .send(ClientMessage::Authenticate { token })
                        .ok();
                }
            }
            WsEvent::MessageReceived(msg) => match msg {
                ServerMessage::Authenticated { user_id, username } => {
                    state.my_id = Some(user_id);
                    state.auth_username = Some(username);
                    state.pending_connect = false;
                    state.screen = crate::state::Screen::RoomBrowser;
                    channels.send_lobby(LobbyAction::ListRooms);
                    if let Some(room_name) = state.auto_room.clone() {
                        channels.send_lobby(LobbyAction::CreateRoom { name: room_name });
                    }
                }
                ServerMessage::AuthError { message } => {
                    state.auth_error = Some(message);
                    state.connected = false;
                    // Saved token is invalid; clear it and send back to login.
                    state.logout();
                }
                ServerMessage::Welcome { .. } => {
                    // No-op: Authenticated supersedes Welcome for the auth flow.
                }
                ServerMessage::RoomJoined { room, your_id } => {
                    state.my_id = Some(your_id);
                    state.current_room = Some(room.clone());
                    state.error_message = None;
                    // Color is picked in the in-room lobby; no auto-join here.
                }
                ServerMessage::RoomLeft { .. } => {
                    state.current_room = None;
                    state.game_state = None;
                    state.screen = crate::state::Screen::RoomBrowser;
                    // Refresh the room list.
                    channels.send_lobby(LobbyAction::ListRooms);
                }
                ServerMessage::RoomList { rooms } => {
                    state.room_list = rooms;
                }
                ServerMessage::StateUpdate(gs) => {
                    state.handle_state_update(*gs);
                }
                ServerMessage::ActionError { message } => {
                    state.error_message = Some(message);
                }
                ServerMessage::LobbyError { message } => {
                    state.error_message = Some(message);
                }
                ServerMessage::Event { .. } => {}
            },
            WsEvent::Disconnected => {
                state.connected = false;
                state.current_room = None;
                state.game_state = None;
            }
        }
    }
}
