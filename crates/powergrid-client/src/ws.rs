use crossbeam_channel::{Receiver, Sender};
use futures_util::{SinkExt, StreamExt};
use powergrid_core::actions::{Action, ClientMessage, HintPayload, LobbyAction, ServerMessage};
use std::sync::Arc;
use tokio::sync::oneshot;
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

pub struct WsChannels {
    pub event_rx: Receiver<WsEvent>,
    action_tx: Sender<ClientMessage>,
    shutdown_tx: Option<oneshot::Sender<()>>,
}

impl WsChannels {
    pub fn send_lobby(&self, action: LobbyAction) {
        self.action_tx.send(ClientMessage::Lobby { action }).ok();
    }

    pub fn send_action(&self, room: Option<&str>, action: Action) {
        if let Some(r) = room {
            self.action_tx
                .send(ClientMessage::Room {
                    room: r.to_string(),
                    action,
                })
                .ok();
        }
    }

    /// Send an ephemeral selection hint. In local play the channel still accepts it;
    /// the local driver silently ignores non-Room messages.
    pub fn send_hint(&self, room: String, hint: HintPayload) {
        self.action_tx
            .send(ClientMessage::RoomHint { room, hint })
            .ok();
    }
}

impl WsChannels {
    /// Construct channels backed by an already-running local session driver.
    pub(crate) fn new_local(
        event_rx: Receiver<WsEvent>,
        action_tx: Sender<ClientMessage>,
        shutdown_tx: oneshot::Sender<()>,
    ) -> Self {
        Self {
            event_rx,
            action_tx,
            shutdown_tx: Some(shutdown_tx),
        }
    }
}

impl Drop for WsChannels {
    fn drop(&mut self) {
        drop(self.shutdown_tx.take());
    }
}

// ---------------------------------------------------------------------------
// Online: spawn the WS worker thread
// ---------------------------------------------------------------------------

pub fn spawn_ws(url: String) -> WsChannels {
    let (event_tx, event_rx) = crossbeam_channel::unbounded::<WsEvent>();
    let (action_tx, action_rx) = crossbeam_channel::unbounded::<ClientMessage>();
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("tokio runtime")
            .block_on(ws_worker(url, event_tx, action_rx, shutdown_rx));
    });

    WsChannels {
        event_rx,
        action_tx,
        shutdown_tx: Some(shutdown_tx),
    }
}

// ---------------------------------------------------------------------------
// Async worker — reconnects until shutdown signal
// ---------------------------------------------------------------------------

async fn ws_worker(
    url: String,
    event_tx: Sender<WsEvent>,
    action_rx: Receiver<ClientMessage>,
    mut shutdown_rx: oneshot::Receiver<()>,
) {
    loop {
        let ws_stream = tokio::select! {
            _ = &mut shutdown_rx => return,
            result = connect_async(&url) => match result {
                Ok((s, _)) => s,
                Err(e) => {
                    warn!("WS connect failed ({url}): {e}");
                    let _ = event_tx.send(WsEvent::Disconnected);
                    tokio::select! {
                        _ = &mut shutdown_rx => return,
                        _ = tokio::time::sleep(tokio::time::Duration::from_secs(2)) => {}
                    }
                    continue;
                }
            }
        };

        debug!("WS connected to {url}");
        let _ = event_tx.send(WsEvent::Connected);
        let (mut write, mut read) = ws_stream.split();

        'inner: loop {
            tokio::select! {
                _ = &mut shutdown_rx => return,
                msg = read.next() => {
                    match msg {
                        Some(Ok(WsMessage::Text(text))) => {
                            match serde_json::from_str::<ServerMessage>(&text) {
                                Ok(m) => {
                                    if event_tx.send(WsEvent::MessageReceived(m)).is_err() {
                                        return;
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
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(16)) => {
                    while let Ok(msg) = action_rx.try_recv() {
                        let json = serde_json::to_string(&msg).expect("serialize ClientMessage");
                        if write.send(WsMessage::Text(json)).await.is_err() {
                            break 'inner;
                        }
                    }
                }
            }
        }

        debug!("WS disconnected, reconnecting in 2s…");
        let _ = event_tx.send(WsEvent::Disconnected);
        tokio::select! {
            _ = &mut shutdown_rx => return,
            _ = tokio::time::sleep(tokio::time::Duration::from_secs(2)) => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Drain the WS channel each frame and update AppState
// ---------------------------------------------------------------------------

pub fn process_ws_events(state: &mut crate::state::AppState, channels: Option<&WsChannels>) {
    let Some(channels) = channels else { return };

    while let Ok(event) = channels.event_rx.try_recv() {
        match event {
            WsEvent::Connected => {
                state.connected = true;
                if let Some(token) = state.auth_token.clone() {
                    channels
                        .action_tx
                        .send(ClientMessage::Authenticate {
                            token,
                            protocol_version: powergrid_core::PROTOCOL_VERSION,
                        })
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
                ServerMessage::AuthError { error } => {
                    state.auth_error = Some(error.to_string());
                    state.connected = false;
                    state.logout();
                }
                ServerMessage::RoomJoined { room, your_id, map } => {
                    state.my_id = Some(your_id);
                    state.current_room = Some(room.clone());
                    state.map = Some(Arc::new(*map));
                    state.error_message = None;
                    state.peer_hints.clear();
                    state.hint_tracker.reset();
                }
                ServerMessage::RoomLeft { .. } => {
                    state.current_room = None;
                    state.game_state = None;
                    state.screen = crate::state::Screen::RoomBrowser;
                    state.peer_hints.clear();
                    state.hint_tracker.reset();
                    channels.send_lobby(LobbyAction::ListRooms);
                }
                ServerMessage::RoomList { rooms } => {
                    state.room_list = rooms;
                }
                ServerMessage::StateUpdate(gs) => {
                    state.handle_state_update(*gs);
                }
                ServerMessage::ActionError { error } => {
                    state.error_message = Some(error.to_string());
                }
                ServerMessage::LobbyError { error } => {
                    state.error_message = Some(error.to_string());
                }
                ServerMessage::Event { .. } => {
                    // event log is populated from gs.event_log in StateUpdate; no client-side dispatch needed
                }
                ServerMessage::PeerHint { player_id, hint } => {
                    if state.my_id != Some(player_id) {
                        state.peer_hints.set(player_id, hint);
                    }
                }
            },
            WsEvent::Disconnected => {
                state.connected = false;
                state.current_room = None;
                state.game_state = None;
                state.map = None;
                state.peer_hints.clear();
                state.hint_tracker.reset();
            }
        }
    }
}
