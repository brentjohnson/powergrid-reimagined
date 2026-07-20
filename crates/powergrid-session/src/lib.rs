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

mod report;
pub use report::{build_report, GameReport, PlantReport, SeatReport};

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
        let mut bot = Bot::new(bot_id, bot_name, color, profile, seed).with_difficulty(difficulty);
        if difficulty == BotDifficulty::Expert {
            match powergrid_bot_strategy::policy::default_policy() {
                // Play the macro policy with play-time MCTS search (policy as
                // prior, value net for leaf eval). `determinize` reshuffles the
                // unseen deck each move so the search can't exploit the true deck
                // order — the fair mode vs a human. Config tuned by held-out
                // sweep (vs 3 normal): 400 sims robustly beats 100 by +3-5pp on
                // two seed blocks (~80%/73% vs ~77%/68%); 800 adds ~nothing, and
                // more determinization worlds don't help — so 400 sims, 1 world.
                //
                // `time_budget_ms` is the real cap on think time; `num_sims` is a
                // high upper bound so the clock (not the sim count) is what binds.
                // At ~1.2ms/sim a 1000ms budget runs ~800 sims, past the ~400-sim
                // strength plateau, so play is at full strength while each move
                // stays ~1s. Override with `EXPERT_SEARCH_MS` (no rebuild): e.g.
                // 300 = snappier, 150 = snappiest/weaker. Falls back to rollout
                // leaves if the value net is unavailable, heuristic if no policy.
                Some(policy) => {
                    let value = powergrid_bot_strategy::policy::default_value_net();
                    let budget_ms = std::env::var("EXPERT_SEARCH_MS")
                        .ok()
                        .and_then(|v| v.parse::<u64>().ok())
                        .unwrap_or(1000);
                    let cfg = powergrid_bot_strategy::search::SearchConfig {
                        num_sims: 2000,
                        time_budget_ms: Some(budget_ms),
                        determinize: true,
                        ..Default::default()
                    };
                    bot = bot.with_policy(policy).with_search(cfg, value);
                }
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

/// Minimum pause after each bot move, even when the bot's think time already
/// exceeded `delay`. The driver only forwards queued `StateUpdate`s to the UI
/// while the pump is parked at an `await`; without a guaranteed pause here,
/// consecutive bot moves (e.g. three Expert bots each searching ~1s with no
/// slack left in `delay`) run back-to-back and the display jumps several actions
/// at once instead of one at a time. 50ms comfortably exceeds the driver's 16ms
/// forward tick, so each move renders before the next bot starts thinking.
const MIN_PACE: Duration = Duration::from_millis(50);

/// Drive all in-process bots in `session_arc` until none has a move or the cap is hit.
/// The lock is released while pacing so other work can proceed.
/// Bots that produce an invalid action are blocked for the remainder of this pump
/// invocation so a strategy bug cannot stall the game.
pub async fn run_bot_pump(session_arc: Arc<Mutex<Session>>, delay: Duration) {
    let mut failed: HashSet<PlayerId> = HashSet::new();
    for iter in 0..MAX_BOT_ITERATIONS {
        let decide_start = std::time::Instant::now();
        let next = {
            let mut session = session_arc.lock().await;
            session.next_bot_action(&failed)
        };

        let Some((bot_id, action)) = next else {
            return;
        };

        // Apply first (broadcasts the move), then pace — so the pause below is
        // the window in which the driver forwards *this* move to the UI before
        // the next bot's search begins.
        {
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

        // Pace as a FLOOR, not an addend: think time counts toward `delay` (so a
        // slow-thinking Expert bot isn't doubly slow — an Expert move is
        // ~max(search, delay), not search + delay), but always pause at least
        // MIN_PACE so consecutive moves render one at a time (see MIN_PACE).
        let pace = delay
            .checked_sub(decide_start.elapsed())
            .unwrap_or(Duration::ZERO)
            .max(MIN_PACE);
        tokio::time::sleep(pace).await;
    }

    let session = session_arc.lock().await;
    warn!(
        "Bot pump hit MAX_BOT_ITERATIONS ({}); game phase: {:?}",
        MAX_BOT_ITERATIONS, session.game.phase
    );
}
