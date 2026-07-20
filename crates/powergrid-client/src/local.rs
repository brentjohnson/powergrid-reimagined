use crossbeam_channel::Sender;
use powergrid_core::{
    actions::{Action, ClientMessage, ServerMessage},
    types::{BotDifficulty, Phase, PlayerColor},
};
use powergrid_session::{build_report, run_bot_pump, GameReport, Session, Subscriber, MAX_PLAYERS};
use std::{sync::Arc, time::Duration};
use tokio::sync::{oneshot, Mutex};
use tracing::{info, warn};
use uuid::Uuid;

use crate::ws::{WsChannels, WsEvent};

pub struct LocalConfig {
    pub human_name: String,
    pub human_color: PlayerColor,
    /// One entry per bot; length determines bot count (1–5).
    pub bots: Vec<BotDifficulty>,
}

/// Where (and as whom) to report a finished local game for metrics.
///
/// Submission is best-effort: the server attributes the result to the account
/// behind `token`, or to `anonymous` if it's `None` or fails to validate.
#[derive(Clone)]
pub struct MetricsConfig {
    pub server: String,
    pub port: u16,
    pub token: Option<String>,
}

/// Holds the background runtime thread for a local play session.
/// Dropping this blocks until the runtime fully shuts down.
/// Shutdown is triggered by dropping `WsChannels` (which holds the oneshot sender).
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

pub fn start_local_session(cfg: LocalConfig, metrics: MetricsConfig) -> (WsChannels, LocalHandle) {
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
        .take(cfg.bots.len())
        .collect();

    let (event_tx, event_rx) = crossbeam_channel::unbounded::<WsEvent>();
    let (action_tx, action_rx) = crossbeam_channel::unbounded::<ClientMessage>();
    let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

    // Build the session synchronously before spawning so errors surface early.
    let map_for_join = map.clone();
    let (state_tx, state_rx) = crossbeam_channel::unbounded::<ServerMessage>();
    let session = {
        let mut s = Session::new(map, MAX_PLAYERS);
        s.add_subscriber(Subscriber::local(Some(human_id), state_tx));
        s.apply(
            human_id,
            Action::JoinGame {
                name: human_name.clone(),
                color: human_color,
            },
        )
        .expect("human JoinGame must succeed");
        for (i, (color, difficulty)) in bot_colors.into_iter().zip(cfg.bots.iter()).enumerate() {
            s.add_bot(format!("Bot {}", i + 1), color, *difficulty)
                .expect("add_bot must succeed in Lobby");
        }
        s.apply(human_id, Action::StartGame)
            .expect("StartGame must succeed with enough players");
        s
    };

    // Drain all initial StateUpdates from session setup.
    let initial_msgs: Vec<ServerMessage> = state_rx.try_iter().collect();

    // Pre-queue the full connection + auth + room handshake so the client
    // sees them all on the first frame and lands on the Game screen.
    let _ = event_tx.send(WsEvent::Connected);
    let _ = event_tx.send(WsEvent::MessageReceived(ServerMessage::Authenticated {
        user_id: human_id,
        username: human_name.clone(),
        // Local play: the "server" is this same binary.
        server_version: env!("CARGO_PKG_VERSION").to_string(),
        server_protocol: powergrid_core::PROTOCOL_VERSION,
    }));
    let _ = event_tx.send(WsEvent::MessageReceived(ServerMessage::RoomJoined {
        room: "local".to_string(),
        your_id: human_id,
        map: Box::new(map_for_join),
    }));
    for msg in initial_msgs {
        let _ = event_tx.send(WsEvent::MessageReceived(msg));
    }

    let session_arc = Arc::new(Mutex::new(session));

    let runtime_thread = {
        let session_arc = Arc::clone(&session_arc);
        let event_tx = event_tx.clone();
        std::thread::spawn(move || {
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
                    metrics,
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
    metrics: MetricsConfig,
    mut shutdown_rx: oneshot::Receiver<()>,
) {
    let bot_delay = Duration::from_millis(
        std::env::var("BOT_DELAY_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(400),
    );

    info!("Local session driver started");

    // Guards the one-shot metrics submission when the game ends.
    let mut submitted = false;

    // StartGame was applied synchronously in start_local_session; drive the
    // initial bot turns now so the game isn't stuck waiting on a bot before
    // the first human action arrives.
    drive_bots_with_forwarding(&session_arc, &state_rx, &event_tx, bot_delay).await;
    maybe_submit_metrics(&session_arc, &metrics, &mut submitted).await;

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
                        error: e,
                    }));
                } else {
                    acted = true;
                }
                // Authenticate and Lobby actions are ignored — local mode handles them internally.
            }
        }

        // Drive bots after any human action.
        if acted {
            drive_bots_with_forwarding(&session_arc, &state_rx, &event_tx, bot_delay).await;
            maybe_submit_metrics(&session_arc, &metrics, &mut submitted).await;
        }
    }

    let _ = event_tx.send(WsEvent::Disconnected);
    info!("Local session driver stopped");
}

/// Run the bot pump while concurrently forwarding state updates to the UI.
///
/// Without this, awaiting `run_bot_pump` blocks the driver task and all
/// `StateUpdate` messages pile up in `state_rx` until the pump finishes,
/// making every bot turn invisible until the whole batch lands at once.
async fn drive_bots_with_forwarding(
    session_arc: &Arc<Mutex<Session>>,
    state_rx: &crossbeam_channel::Receiver<ServerMessage>,
    event_tx: &Sender<WsEvent>,
    delay: Duration,
) {
    let pump = run_bot_pump(Arc::clone(session_arc), delay);
    tokio::pin!(pump);
    loop {
        tokio::select! {
            _ = &mut pump => break,
            _ = tokio::time::sleep(Duration::from_millis(16)) => {
                for msg in state_rx.try_iter() {
                    let _ = event_tx.send(WsEvent::MessageReceived(msg));
                }
            }
        }
    }
    for msg in state_rx.try_iter() {
        let _ = event_tx.send(WsEvent::MessageReceived(msg));
    }
}

/// If the game has ended and we haven't reported it yet, build the standings
/// report and hand it to a background thread for submission. Sets `submitted`
/// so this fires exactly once per game.
async fn maybe_submit_metrics(
    session_arc: &Arc<Mutex<Session>>,
    metrics: &MetricsConfig,
    submitted: &mut bool,
) {
    if *submitted {
        return;
    }
    let report = {
        let session = session_arc.lock().await;
        if !matches!(session.game.phase, Phase::GameOver { .. }) {
            return;
        }
        build_report(&session)
    };
    *submitted = true;
    submit_report(metrics.clone(), report);
}

/// Fire-and-forget POST of a finished local game to the lobby's metrics
/// endpoint. Runs on a detached thread with a blocking client so it never
/// stalls the game loop; failures are logged and otherwise ignored.
fn submit_report(metrics: MetricsConfig, report: GameReport) {
    let url = format!("http://{}:{}/games/local", metrics.server, metrics.port);
    std::thread::spawn(move || {
        let client = match reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
        {
            Ok(c) => c,
            Err(e) => {
                warn!("Failed to build metrics client: {e}");
                return;
            }
        };
        let mut req = client.post(&url).json(&report);
        if let Some(token) = metrics.token {
            req = req.header("Authorization", format!("Bearer {token}"));
        }
        match req.send() {
            Ok(resp) => info!("Local game metrics submitted ({})", resp.status()),
            Err(e) => warn!("Failed to submit local game metrics: {e}"),
        }
    });
}
