use bevy::prelude::*;
use crossbeam_channel::Sender;
use powergrid_core::{
    actions::{Action, ClientMessage, ServerMessage},
    types::PlayerColor,
};
use powergrid_session::{run_bot_pump, Session, Subscriber, MAX_PLAYERS};
use std::{sync::Arc, time::Duration};
use tokio::sync::{oneshot, Mutex};
use tracing::info;
use uuid::Uuid;

use crate::ws::{WsChannels, WsEvent};

pub struct LocalConfig {
    pub human_name: String,
    pub human_color: PlayerColor,
    pub bot_count: u8,
}

/// Holds the background runtime thread for a local play session.
/// Dropping this resource blocks until the runtime fully shuts down.
/// Shutdown is triggered by dropping `WsChannels` (which holds the oneshot sender).
#[derive(Resource)]
pub struct LocalHandle {
    runtime_thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for LocalHandle {
    fn drop(&mut self) {
        if let Some(t) = self.runtime_thread.take() {
            t.join().ok();
        }
    }
}

pub fn start_local_session(cfg: LocalConfig) -> (WsChannels, LocalHandle) {
    let map = powergrid_core::default_map();

    let human_id = Uuid::new_v4();
    let human_name = cfg.human_name.clone();
    let human_color = cfg.human_color;

    let all_colors = [
        PlayerColor::Red,
        PlayerColor::Blue,
        PlayerColor::Green,
        PlayerColor::Yellow,
        PlayerColor::Purple,
        PlayerColor::White,
    ];
    let bot_colors: Vec<PlayerColor> = all_colors
        .iter()
        .copied()
        .filter(|&c| c != human_color)
        .take(cfg.bot_count as usize)
        .collect();

    let (event_tx, event_rx) = crossbeam_channel::unbounded::<WsEvent>();
    let (action_tx, action_rx) = crossbeam_channel::unbounded::<ClientMessage>();
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    // Build the session synchronously before spawning so errors surface early.
    let (state_tx, state_rx) = crossbeam_channel::unbounded::<ServerMessage>();
    let session = {
        let mut s = Session::new(map, MAX_PLAYERS);
        s.add_subscriber(Subscriber::Local(state_tx));
        s.apply(
            human_id,
            Action::JoinGame {
                name: human_name.clone(),
                color: human_color,
            },
        )
        .expect("human JoinGame must succeed");
        for (i, color) in bot_colors.into_iter().enumerate() {
            s.add_bot(format!("Bot {}", i + 1), color)
                .expect("add_bot must succeed in Lobby");
        }
        s.apply(human_id, Action::StartGame)
            .expect("StartGame must succeed with enough players");
        s
    };

    // Drain all initial StateUpdates from session setup.
    let initial_msgs: Vec<ServerMessage> = state_rx.try_iter().collect();

    // Pre-queue the full connection + auth + room handshake so the client
    // sees them all on the first Bevy frame and lands on the Game screen.
    let _ = event_tx.send(WsEvent::Connected);
    let _ = event_tx.send(WsEvent::MessageReceived(ServerMessage::Authenticated {
        user_id: human_id,
        username: human_name.clone(),
    }));
    let _ = event_tx.send(WsEvent::MessageReceived(ServerMessage::RoomJoined {
        room: "local".to_string(),
        your_id: human_id,
    }));
    for msg in initial_msgs {
        let _ = event_tx.send(WsEvent::MessageReceived(msg));
    }

    let session_arc = Arc::new(Mutex::new(session));

    let runtime_thread = {
        let session_arc = Arc::clone(&session_arc);
        let event_tx = event_tx.clone();
        std::thread::spawn(move || {
            tracing_subscriber::fmt()
                .with_env_filter(
                    tracing_subscriber::EnvFilter::try_from_default_env()
                        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
                )
                .try_init()
                .ok();

            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("tokio runtime")
                .block_on(local_session_driver(
                    session_arc,
                    state_rx,
                    event_tx,
                    action_rx,
                    human_id,
                    shutdown_rx,
                ));
        })
    };

    let channels = WsChannels::new_local(event_rx, action_tx, shutdown_tx);

    (
        channels,
        LocalHandle {
            runtime_thread: Some(runtime_thread),
        },
    )
}

async fn local_session_driver(
    session_arc: Arc<Mutex<Session>>,
    state_rx: crossbeam_channel::Receiver<ServerMessage>,
    event_tx: Sender<WsEvent>,
    action_rx: crossbeam_channel::Receiver<ClientMessage>,
    human_id: uuid::Uuid,
    mut shutdown_rx: oneshot::Receiver<()>,
) {
    let bot_delay = Duration::from_millis(
        std::env::var("BOT_DELAY_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(400),
    );

    info!("Local session driver started");

    // StartGame was applied synchronously in start_local_session; drive the
    // initial bot turns now so the game isn't stuck waiting on a bot before
    // the first human action arrives.
    run_bot_pump(Arc::clone(&session_arc), bot_delay).await;

    loop {
        // Forward any pending state updates from the session subscriber.
        for msg in state_rx.try_iter() {
            let _ = event_tx.send(WsEvent::MessageReceived(msg));
        }

        // Check for shutdown or wait 16ms.
        tokio::select! {
            _ = &mut shutdown_rx => break,
            _ = tokio::time::sleep(Duration::from_millis(16)) => {}
        }

        // Process pending client actions.
        let mut acted = false;
        while let Ok(msg) = action_rx.try_recv() {
            if let ClientMessage::Room { action, .. } = msg {
                let result = {
                    let mut s = session_arc.lock().await;
                    s.apply(human_id, action)
                };
                // Forward state updates triggered by the action.
                for msg in state_rx.try_iter() {
                    let _ = event_tx.send(WsEvent::MessageReceived(msg));
                }
                if let Err(e) = result {
                    let _ = event_tx.send(WsEvent::MessageReceived(ServerMessage::ActionError {
                        message: e.to_string(),
                    }));
                } else {
                    acted = true;
                }
                // Authenticate and Lobby actions are ignored — local mode handles them internally.
            }
        }

        // Drive bots after any human action.
        if acted {
            run_bot_pump(Arc::clone(&session_arc), bot_delay).await;
            // Forward state updates from bot turns.
            for msg in state_rx.try_iter() {
                let _ = event_tx.send(WsEvent::MessageReceived(msg));
            }
        }
    }

    let _ = event_tx.send(WsEvent::Disconnected);
    info!("Local session driver stopped");
}
