use powergrid_bot_strategy::strategy;
use powergrid_core::{
    actions::{Action, ActionError, ServerMessage},
    rules::apply_action,
    types::{Phase, PlayerColor, PlayerId},
    GameState,
};
use std::{sync::Arc, time::Duration};
use tokio::sync::Mutex;
use tracing::{info, warn};

pub use powergrid_core::map::Map;

/// Maximum players allowed per session.
pub const MAX_PLAYERS: u8 = 6;

// ---------------------------------------------------------------------------
// Subscriber
// ---------------------------------------------------------------------------

/// A destination for broadcasted `ServerMessage`s.
pub enum Subscriber {
    /// Serializes to JSON and forwards over a tokio mpsc channel (WS use).
    Mpsc(tokio::sync::mpsc::UnboundedSender<String>),
    /// Sends the typed message directly over a crossbeam channel (in-process use).
    Local(crossbeam_channel::Sender<ServerMessage>),
}

impl Subscriber {
    fn send(&self, msg: &ServerMessage) -> bool {
        match self {
            Subscriber::Mpsc(tx) => tx
                .send(serde_json::to_string(msg).expect("serialize ServerMessage"))
                .is_ok(),
            Subscriber::Local(tx) => tx.send(msg.clone()).is_ok(),
        }
    }
}

// ---------------------------------------------------------------------------
// BotSlot
// ---------------------------------------------------------------------------

pub struct BotSlot {
    pub id: PlayerId,
    pub name: String,
    pub color: PlayerColor,
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

pub struct Session {
    pub game: GameState,
    subscribers: Vec<Subscriber>,
    pub bots: Vec<BotSlot>,
}

impl Session {
    pub fn new(map: Map, max_players: u8) -> Self {
        Self {
            game: GameState::new(map, max_players.into()),
            subscribers: Vec::new(),
            bots: Vec::new(),
        }
    }

    pub fn add_subscriber(&mut self, sub: Subscriber) {
        self.subscribers.push(sub);
    }

    pub fn subscriber_count(&self) -> usize {
        self.subscribers.len()
    }

    /// Apply `action` from `actor`. On success, broadcasts `StateUpdate` to all subscribers
    /// and prunes disconnected ones. Returns the error without broadcasting on failure.
    pub fn apply(&mut self, actor: PlayerId, action: Action) -> Result<(), ActionError> {
        apply_action(&mut self.game, actor, action)?;
        let msg = ServerMessage::StateUpdate(Box::new(self.game.view()));
        self.broadcast(&msg);
        Ok(())
    }

    pub fn broadcast(&mut self, msg: &ServerMessage) {
        self.subscribers.retain(|s| s.send(msg));
    }

    pub fn broadcast_json(&mut self, json: &str) {
        self.subscribers.retain(|s| match s {
            Subscriber::Mpsc(tx) => tx.send(json.to_string()).is_ok(),
            Subscriber::Local(tx) => {
                if let Ok(msg) = serde_json::from_str(json) {
                    tx.send(msg).is_ok()
                } else {
                    true
                }
            }
        });
    }

    /// Add an in-process bot (Lobby phase only).
    pub fn add_bot(&mut self, bot_name: String, color: PlayerColor) -> Result<PlayerId, String> {
        let bot_id = uuid::Uuid::new_v4();
        apply_action(
            &mut self.game,
            bot_id,
            Action::JoinGame {
                name: bot_name.clone(),
                color,
            },
        )
        .map_err(|e| e.to_string())?;
        info!("Bot '{}' ({:?}) added to session", bot_name, color);
        self.bots.push(BotSlot {
            id: bot_id,
            name: bot_name,
            color,
        });
        Ok(bot_id)
    }

    /// Remove a bot (Lobby phase only).
    pub fn remove_bot(&mut self, bot_id: PlayerId) -> Result<(), String> {
        if !matches!(self.game.phase, Phase::Lobby) {
            return Err("cannot remove bot after game has started".to_string());
        }
        let idx = self
            .bots
            .iter()
            .position(|b| b.id == bot_id)
            .ok_or_else(|| "bot not found".to_string())?;
        self.bots.remove(idx);
        self.game.players.retain(|p| p.id != bot_id);
        self.game.player_order.retain(|id| *id != bot_id);
        info!("Bot {} removed from session", bot_id);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// BotPump
// ---------------------------------------------------------------------------

const MAX_BOT_ITERATIONS: usize = 50;

/// Drive all in-process bots in `session_arc` until none has a move or the cap is hit.
/// The lock is released during `delay` so other work can proceed.
pub async fn run_bot_pump(session_arc: Arc<Mutex<Session>>, delay: Duration) {
    for iter in 0..MAX_BOT_ITERATIONS {
        let next = {
            let session = session_arc.lock().await;
            session
                .bots
                .iter()
                .find_map(|b| strategy::decide(&session.game, b.id).map(|a| (b.id, a)))
        };

        let Some((bot_id, action)) = next else {
            return;
        };

        tokio::time::sleep(delay).await;

        let mut session = session_arc.lock().await;
        match session.apply(bot_id, action) {
            Ok(()) => {
                info!("Bot {} acted (iter {})", bot_id, iter);
            }
            Err(e) => {
                warn!("Bot {} produced invalid action: {}", bot_id, e);
            }
        }
    }

    let session = session_arc.lock().await;
    warn!(
        "Bot pump hit MAX_BOT_ITERATIONS ({}); game phase: {:?}",
        MAX_BOT_ITERATIONS, session.game.phase
    );
}
