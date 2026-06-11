use powergrid_bot_strategy::{default_registry, Bot};
use powergrid_core::{
    actions::{Action, ActionError, ServerMessage},
    rules::apply_action,
    types::{BotDifficulty, Phase, PlayerColor, PlayerId},
    GameState,
};
use std::{collections::HashSet, sync::Arc, time::Duration};
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
    Mpsc {
        viewer: Option<PlayerId>,
        tx: tokio::sync::mpsc::UnboundedSender<String>,
    },
    /// Sends the typed message directly over a crossbeam channel (in-process use).
    Local {
        viewer: Option<PlayerId>,
        tx: crossbeam_channel::Sender<ServerMessage>,
    },
}

impl Subscriber {
    pub fn mpsc(viewer: Option<PlayerId>, tx: tokio::sync::mpsc::UnboundedSender<String>) -> Self {
        Subscriber::Mpsc { viewer, tx }
    }

    pub fn local(viewer: Option<PlayerId>, tx: crossbeam_channel::Sender<ServerMessage>) -> Self {
        Subscriber::Local { viewer, tx }
    }

    fn viewer(&self) -> Option<PlayerId> {
        match self {
            Subscriber::Mpsc { viewer, .. } | Subscriber::Local { viewer, .. } => *viewer,
        }
    }

    fn send(&self, msg: &ServerMessage) -> bool {
        match self {
            Subscriber::Mpsc { tx, .. } => tx
                .send(serde_json::to_string(msg).expect("serialize ServerMessage"))
                .is_ok(),
            Subscriber::Local { tx, .. } => tx.send(msg.clone()).is_ok(),
        }
    }

    fn send_json(&self, json: &str) -> bool {
        match self {
            Subscriber::Mpsc { tx, .. } => tx.send(json.to_string()).is_ok(),
            Subscriber::Local { tx, .. } => {
                if let Ok(msg) = serde_json::from_str(json) {
                    tx.send(msg).is_ok()
                } else {
                    true
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

pub struct Session {
    pub game: GameState,
    subscribers: Vec<Subscriber>,
    pub bots: Vec<Bot>,
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

    /// Apply `action` from `actor`. On success, broadcasts per-recipient `StateUpdate` to all
    /// subscribers (each sees their own money but opponents' money is zeroed).
    /// Returns the error without broadcasting on failure.
    pub fn apply(&mut self, actor: PlayerId, action: Action) -> Result<(), ActionError> {
        apply_action(&mut self.game, actor, action)?;
        self.broadcast_state_update();
        Ok(())
    }

    /// Broadcast the current game state. Each subscriber receives a view with opponent
    /// money zeroed; only their own money is included.
    pub fn broadcast_state_update(&mut self) {
        let game = &self.game;
        self.subscribers.retain(|s| {
            let view = game.view_for(s.viewer());
            let msg = ServerMessage::StateUpdate(Box::new(view));
            s.send(&msg)
        });
    }

    pub fn broadcast(&mut self, msg: &ServerMessage) {
        self.subscribers.retain(|s| s.send(msg));
    }

    pub fn broadcast_json(&mut self, json: &str) {
        self.subscribers.retain(|s| s.send_json(json));
    }

    /// Add an in-process bot (Lobby phase only).
    pub fn add_bot(
        &mut self,
        bot_name: String,
        color: PlayerColor,
        difficulty: BotDifficulty,
    ) -> Result<PlayerId, ActionError> {
        let bot_id = uuid::Uuid::new_v4();
        apply_action(
            &mut self.game,
            bot_id,
            Action::JoinGame {
                name: bot_name.clone(),
                color,
            },
        )?;
        info!(
            "Bot '{}' ({:?}) added to session (difficulty: {:?})",
            bot_name, color, difficulty
        );

        let registry = default_registry();
        let profile = registry.profile_for(difficulty).clone();
        let seed = bot_id.as_u128() as u64;
        let mut bot = Bot::new(bot_id, bot_name, color, profile, seed);
        if difficulty == BotDifficulty::Expert {
            match powergrid_bot_strategy::policy::default_policy() {
                Some(policy) => bot = bot.with_policy(policy),
                None => warn!("expert RL policy unavailable; bot will use the hard heuristic"),
            }
        }
        self.bots.push(bot);
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

    /// Find the first non-skipped bot that has a move and return its id + action.
    /// Uses disjoint field borrows so game can be read while bots are iterated mutably.
    pub fn next_bot_action(&mut self, skip: &HashSet<PlayerId>) -> Option<(PlayerId, Action)> {
        let game = &self.game;
        self.bots
            .iter_mut()
            .filter(|b| !skip.contains(&b.id))
            .find_map(|b| b.decide(game).map(|a| (b.id, a)))
    }
}

// ---------------------------------------------------------------------------
// BotPump
// ---------------------------------------------------------------------------

const MAX_BOT_ITERATIONS: usize = 500;

/// Drive all in-process bots in `session_arc` until none has a move or the cap is hit.
/// The lock is released during `delay` so other work can proceed.
/// Bots that produce an invalid action are blocked for the remainder of this pump
/// invocation so a strategy bug cannot stall the game.
pub async fn run_bot_pump(session_arc: Arc<Mutex<Session>>, delay: Duration) {
    let mut failed: HashSet<PlayerId> = HashSet::new();
    for iter in 0..MAX_BOT_ITERATIONS {
        let next = {
            let mut session = session_arc.lock().await;
            session.next_bot_action(&failed)
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
                failed.insert(bot_id);
            }
        }
    }

    let session = session_arc.lock().await;
    warn!(
        "Bot pump hit MAX_BOT_ITERATIONS ({}); game phase: {:?}",
        MAX_BOT_ITERATIONS, session.game.phase
    );
}
