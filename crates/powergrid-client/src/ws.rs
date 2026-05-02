use bevy::prelude::*;
use crossbeam_channel::{Receiver, Sender};
use futures_util::{SinkExt, StreamExt};
use powergrid_core::{
    actions::{Action, ClientMessage, LobbyAction, ServerMessage},
    types::PlayerColor,
};
use tokio_tungstenite::{connect_async, tungstenite::Message as WsMessage};
use tracing::{debug, warn};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

pub enum WsMode {
    /// Production lobby protocol: Authenticate → RoomBrowser → Room actions.
    Lobby,
    /// Embedded `powergrid-server`: bare `Action::JoinGame` on connect, bare `Action` thereafter.
    Legacy { name: String, color: PlayerColor },
}

pub enum WsEvent {
    Connected,
    MessageReceived(ServerMessage),
    Disconnected,
}

enum OutboundMessage {
    Lobby(ClientMessage),
    Legacy(Action),
}

#[derive(Resource)]
pub struct WsChannels {
    pub event_rx: Receiver<WsEvent>,
    action_tx: Sender<OutboundMessage>,
    pub mode: WsMode,
}

impl WsChannels {
    pub fn send_lobby(&self, action: LobbyAction) {
        if matches!(self.mode, WsMode::Lobby) {
            self.action_tx
                .send(OutboundMessage::Lobby(ClientMessage::Lobby(action)))
                .ok();
        }
    }

    /// Send an in-game action: uses room-scoped `ClientMessage` in Lobby mode,
    /// or a bare `Action` in Legacy mode (ignored room arg).
    pub fn send_action(&self, room: Option<&str>, action: powergrid_core::Action) {
        match &self.mode {
            WsMode::Lobby => {
                if let Some(r) = room {
                    self.action_tx
                        .send(OutboundMessage::Lobby(ClientMessage::Room {
                            room: r.to_string(),
                            action,
                        }))
                        .ok();
                }
            }
            WsMode::Legacy { .. } => {
                self.action_tx.send(OutboundMessage::Legacy(action)).ok();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Public: spawn the WS worker thread and return channel handles
// ---------------------------------------------------------------------------

pub fn spawn_ws(url: String, mode: WsMode) -> WsChannels {
    let (event_tx, event_rx) = crossbeam_channel::unbounded::<WsEvent>();
    let (action_tx, action_rx) = crossbeam_channel::unbounded::<OutboundMessage>();

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
        mode,
    }
}

// ---------------------------------------------------------------------------
// Async worker — reconnects forever
// ---------------------------------------------------------------------------

async fn ws_worker(url: String, event_tx: Sender<WsEvent>, action_rx: Receiver<OutboundMessage>) {
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
                        let json = match msg {
                            OutboundMessage::Lobby(m) => {
                                serde_json::to_string(&m).expect("serialize ClientMessage")
                            }
                            OutboundMessage::Legacy(a) => {
                                serde_json::to_string(&a).expect("serialize Action")
                            }
                        };
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
                match &channels.mode {
                    WsMode::Lobby => {
                        if let Some(token) = state.auth_token.clone() {
                            channels
                                .action_tx
                                .send(OutboundMessage::Lobby(ClientMessage::Authenticate {
                                    token,
                                }))
                                .ok();
                        }
                    }
                    WsMode::Legacy { name, color } => {
                        let action = Action::JoinGame {
                            name: name.clone(),
                            color: *color,
                        };
                        channels
                            .action_tx
                            .send(OutboundMessage::Legacy(action))
                            .ok();
                    }
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
                    state.logout();
                }
                ServerMessage::Welcome { your_id } => {
                    if let WsMode::Legacy { .. } = &channels.mode {
                        state.my_id = Some(your_id);
                        state.pending_connect = false;
                        state.current_room = Some("local".into());
                    }
                }
                ServerMessage::RoomJoined { room, your_id } => {
                    state.my_id = Some(your_id);
                    state.current_room = Some(room.clone());
                    state.error_message = None;
                }
                ServerMessage::RoomLeft { .. } => {
                    state.current_room = None;
                    state.game_state = None;
                    state.screen = crate::state::Screen::RoomBrowser;
                    channels.send_lobby(LobbyAction::ListRooms);
                }
                ServerMessage::RoomList { rooms } => {
                    state.room_list = rooms;
                }
                ServerMessage::StateUpdate(gs) => {
                    // In local mode: auto-start once all expected players have joined.
                    if let WsMode::Legacy { .. } = &channels.mode {
                        let expected = state.local_expected_players as usize;
                        if expected > 0
                            && matches!(gs.phase, powergrid_core::types::Phase::Lobby)
                            && gs.players.len() >= expected
                            && gs.host_id() == state.my_id
                        {
                            channels
                                .action_tx
                                .send(OutboundMessage::Legacy(Action::StartGame))
                                .ok();
                        }
                    }
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
