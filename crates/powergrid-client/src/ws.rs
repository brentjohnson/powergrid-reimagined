use crossbeam_channel::{Receiver, Sender};
use futures_util::{SinkExt, StreamExt};
use powergrid_core::actions::{
    Action, AuthError, ClientMessage, HintPayload, LobbyAction, ServerMessage,
};
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

        // Keepalive: send a ping on an interval so an otherwise-idle lobby
        // connection keeps carrying traffic. Without this, an idle socket is
        // reaped by proxies/NAT after ~2 minutes and we churn through the
        // reconnect loop. Sending also flushes any pong tungstenite queued in
        // response to a server ping (writes only happen when we send).
        let mut keepalive = tokio::time::interval(tokio::time::Duration::from_secs(30));
        keepalive.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        'inner: loop {
            tokio::select! {
                _ = &mut shutdown_rx => return,
                _ = keepalive.tick() => {
                    if write.send(WsMessage::Ping(Vec::new())).await.is_err() {
                        break 'inner;
                    }
                }
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
                state.ws_got_msg_this_conn = false;
                state.ws_auth_sent_this_conn = false;
                if let Some(token) = state.auth_token.clone() {
                    channels
                        .action_tx
                        .send(ClientMessage::Authenticate {
                            token,
                            protocol_version: powergrid_core::PROTOCOL_VERSION,
                        })
                        .ok();
                    state.ws_auth_sent_this_conn = true;
                }
            }
            WsEvent::MessageReceived(msg) => {
                state.ws_got_msg_this_conn = true;
                state.ws_silent_auth_drops = 0;
                match msg {
                    ServerMessage::Authenticated {
                        user_id,
                        username,
                        server_version,
                        server_protocol,
                    } => {
                        state.my_id = Some(user_id);
                        state.auth_username = Some(username);
                        state.server_version =
                            (!server_version.is_empty()).then_some(server_version);
                        state.server_protocol = Some(server_protocol);
                        state.pending_connect = false;
                        state.screen = crate::state::Screen::RoomBrowser;
                        channels.send_lobby(LobbyAction::ListRooms);
                        if let Some(room_name) = state.auto_room.clone() {
                            channels.send_lobby(LobbyAction::CreateRoom { name: room_name });
                        }
                    }
                    ServerMessage::AuthError { error } => {
                        // Friendlier copy for the common cases; fall back to the
                        // error's own text otherwise. All cases drop the connection
                        // and return to the login screen. Only `InvalidToken` means
                        // the saved token is genuinely bad — forget it so the user
                        // logs in fresh. Every other case (version mismatch, etc.)
                        // leaves a valid token that we must keep on disk, so the
                        // next launch can still auto-login once the mismatch is
                        // resolved; nuking it here is what forced a re-login every
                        // time.
                        let msg = match &error {
                            AuthError::InvalidToken => {
                                "Your saved session has expired. Please log in again.".to_string()
                            }
                            AuthError::VersionMismatch {
                                server_version,
                                client_version,
                            } => format!(
                                "Version mismatch: this client speaks protocol {client_version}, \
                             but the server requires {server_version}. Update the client (or \
                             server) so they match."
                            ),
                            other => other.to_string(),
                        };
                        state.connected = false;
                        // logout()/end_session clear auth_error, so set the message after.
                        if matches!(error, AuthError::InvalidToken) {
                            state.logout();
                        } else {
                            state.end_session_keep_credentials();
                        }
                        state.auth_error = Some(msg);
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
                }
            }
            WsEvent::Disconnected => {
                // A connection that died right after `Authenticate` without a
                // single server message is an ambiguous failure: a rejected
                // handshake on a server that predates the auth-error flush fix
                // (it resets instead of reporting why), a transient network
                // blip, or the server briefly being down. After three in a row,
                // stop the reconnect loop and return to the login screen — but
                // *keep* the saved token on disk. We can't tell a bad token from
                // a transient drop here, and deleting a still-valid token is
                // exactly what forced a manual re-login on every launch. The
                // next launch will auto-retry with the same token.
                let silent_auth_drop = state.ws_auth_sent_this_conn && !state.ws_got_msg_this_conn;
                state.ws_auth_sent_this_conn = false;
                if silent_auth_drop {
                    state.ws_silent_auth_drops += 1;
                    if state.ws_silent_auth_drops >= 3 {
                        state.ws_silent_auth_drops = 0;
                        state.end_session_keep_credentials();
                        state.auth_error = Some(
                            "The server dropped the connection during sign-in without \
                             giving a reason (it may be down or running an older \
                             version). Your saved login was kept — try again, or it \
                             will reconnect automatically next time."
                                .to_string(),
                        );
                    }
                }
                state.connected = false;
                state.server_version = None;
                state.server_protocol = None;
                state.current_room = None;
                state.game_state = None;
                state.map = None;
                state.peer_hints.clear();
                state.hint_tracker.reset();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{AppState, CliArgs, Screen};

    fn test_state_with_token() -> AppState {
        let mut state = AppState::new(CliArgs {
            color: None,
            server: None,
            port: None,
            room: None,
            windowed: false,
            no_preferences: true, // never touch saved credentials on disk
        });
        state.auth_token = Some("stale-token".to_string());
        state
    }

    fn test_channels() -> (Sender<WsEvent>, WsChannels) {
        let (event_tx, event_rx) = crossbeam_channel::unbounded::<WsEvent>();
        let (action_tx, _action_rx) = crossbeam_channel::unbounded::<ClientMessage>();
        let (shutdown_tx, _shutdown_rx) = oneshot::channel::<()>();
        (
            event_tx,
            WsChannels::new_local(event_rx, action_tx, shutdown_tx),
        )
    }

    /// A server predating the auth-error flush fix resets a rejected handshake
    /// with no reply. Three such connect→silent-drop cycles must end the
    /// reconnect loop: in-memory session cleared, error shown, back to Login.
    /// The saved token on disk is NOT forgotten here (the drop is ambiguous —
    /// could be transient — so the next launch retries with the same token).
    #[test]
    fn silent_auth_drops_end_reconnect_loop() {
        let mut state = test_state_with_token();
        let (event_tx, channels) = test_channels();

        for _ in 0..2 {
            event_tx.send(WsEvent::Connected).unwrap();
            event_tx.send(WsEvent::Disconnected).unwrap();
        }
        process_ws_events(&mut state, Some(&channels));
        assert_eq!(state.ws_silent_auth_drops, 2);
        assert!(state.auth_token.is_some(), "two drops must not log out yet");

        event_tx.send(WsEvent::Connected).unwrap();
        event_tx.send(WsEvent::Disconnected).unwrap();
        process_ws_events(&mut state, Some(&channels));

        assert!(
            state.auth_token.is_none(),
            "in-memory session must be cleared"
        );
        assert!(state.disconnect_requested, "worker must be torn down");
        assert!(!state.pending_connect);
        assert_eq!(state.screen, Screen::Login);
        let err = state.auth_error.as_deref().expect("error must be shown");
        assert!(
            err.contains("saved login was kept"),
            "unexpected copy: {err}"
        );
    }

    /// A silently-dropped handshake must NOT delete the persisted token — that
    /// bug forced a manual re-login on every launch. Here we drive the real
    /// disk path (no_preferences = false) and confirm the credential file
    /// survives the teardown.
    #[test]
    fn silent_auth_drops_keep_saved_credentials_on_disk() {
        use crate::auth::{
            clear_credentials, load_credentials, save_credentials, SavedCredentials,
        };
        let server = "silentdrop.test.zzz";
        let port = 65003u16;
        let _ = clear_credentials(server, port);
        save_credentials(&SavedCredentials {
            token: "keep-me".into(),
            user_id: uuid::Uuid::nil(),
            username: "carol".into(),
            server: server.into(),
            port,
        })
        .unwrap();

        let mut state = AppState::new(CliArgs {
            color: None,
            server: Some(server.to_string()),
            port: Some(port),
            room: None,
            windowed: false,
            no_preferences: false,
        });
        assert_eq!(state.auth_token.as_deref(), Some("keep-me"));
        let (event_tx, channels) = test_channels();

        for _ in 0..3 {
            event_tx.send(WsEvent::Connected).unwrap();
            event_tx.send(WsEvent::Disconnected).unwrap();
        }
        process_ws_events(&mut state, Some(&channels));

        let survived = load_credentials(server, port);
        let _ = clear_credentials(server, port);
        assert_eq!(
            survived.map(|c| c.token).as_deref(),
            Some("keep-me"),
            "silent drop must not delete the saved token"
        );
    }

    /// Any server message on a connection proves the drop wasn't an auth
    /// rejection, so the streak resets and no logout happens.
    #[test]
    fn server_reply_resets_silent_drop_streak() {
        let mut state = test_state_with_token();
        let (event_tx, channels) = test_channels();

        for _ in 0..5 {
            event_tx.send(WsEvent::Connected).unwrap();
            event_tx
                .send(WsEvent::MessageReceived(ServerMessage::RoomList {
                    rooms: Vec::new(),
                }))
                .unwrap();
            event_tx.send(WsEvent::Disconnected).unwrap();
        }
        process_ws_events(&mut state, Some(&channels));

        assert_eq!(state.ws_silent_auth_drops, 0);
        assert!(state.auth_token.is_some());
        assert!(!state.disconnect_requested);
    }

    /// Failed connects (no `Connected` event, e.g. server down) never count
    /// toward the silent-drop streak — that path should keep retrying.
    #[test]
    fn failed_connects_do_not_count_as_auth_drops() {
        let mut state = test_state_with_token();
        let (event_tx, channels) = test_channels();

        for _ in 0..5 {
            event_tx.send(WsEvent::Disconnected).unwrap();
        }
        process_ws_events(&mut state, Some(&channels));

        assert_eq!(state.ws_silent_auth_drops, 0);
        assert!(state.auth_token.is_some());
    }
}
