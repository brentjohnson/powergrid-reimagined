use std::sync::Arc;

use powergrid_core::{
    actions::Action,
    state::GameState,
    types::{BotDifficulty, PlayerColor, PlayerId},
};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::policy::{MlpPolicy, ValueNet};
use crate::profile::BotProfile;
use crate::search::SearchConfig;
use crate::strategy::RlDecision;

/// A stateful bot: holds its identity, decision profile, and a seeded RNG.
/// The RNG must persist across `decide` calls so sampling is stable within a game.
pub struct Bot {
    pub id: PlayerId,
    pub name: String,
    pub color: PlayerColor,
    /// The difficulty this bot was created at. Not used by decision logic (the
    /// `profile`/`policy` already encode strength) — retained so callers (e.g.
    /// the lobby's game-result recorder) can report which kind of bot played.
    pub difficulty: BotDifficulty,
    pub profile: BotProfile,
    pub(crate) rng: SmallRng,
    /// RL policy (Expert difficulty). When set, `decide` plays the policy and
    /// only falls back to the heuristic if the policy is unusable (non-default map).
    pub(crate) policy: Option<Arc<MlpPolicy>>,
    /// When true, the policy plays greedily (argmax over legal) instead of
    /// sampling from the masked softmax. Stronger for behavior-cloned policies;
    /// used by the held-out evaluation harness.
    pub(crate) greedy: bool,
    /// Play-time MCTS search config. When set (with a policy), `decide` searches
    /// with the policy as prior instead of playing the raw policy.
    pub(crate) search_config: Option<SearchConfig>,
    /// Value net for search leaf evaluation (falls back to rollouts if absent).
    pub(crate) value: Option<Arc<ValueNet>>,
}

impl Bot {
    pub fn new(
        id: PlayerId,
        name: String,
        color: PlayerColor,
        profile: BotProfile,
        seed: u64,
    ) -> Self {
        Self {
            id,
            name,
            color,
            difficulty: BotDifficulty::default(),
            profile,
            rng: SmallRng::seed_from_u64(seed),
            policy: None,
            greedy: false,
            search_config: None,
            value: None,
        }
    }

    /// Tag this bot with the difficulty it was created at (reporting only).
    pub fn with_difficulty(mut self, difficulty: BotDifficulty) -> Self {
        self.difficulty = difficulty;
        self
    }

    /// Make the policy play greedily (argmax) rather than sampling.
    pub fn with_greedy(mut self, greedy: bool) -> Self {
        self.greedy = greedy;
        self
    }

    /// Play with MCTS search (policy as prior, `value` net for leaf eval — falls
    /// back to rollouts when `value` is `None`). Requires a policy (`with_policy`).
    pub fn with_search(mut self, cfg: SearchConfig, value: Option<Arc<ValueNet>>) -> Self {
        self.search_config = Some(cfg);
        self.value = value;
        self
    }

    pub fn with_policy(mut self, policy: Arc<MlpPolicy>) -> Self {
        self.policy = Some(policy);
        self
    }

    pub fn decide(&mut self, state: &GameState) -> Option<Action> {
        if self.policy.is_some() {
            match crate::strategy::decide_rl(state, self) {
                RlDecision::Action(action) => return Some(action),
                RlDecision::NotMyTurn => return None,
                RlDecision::Unavailable => {} // fall back to the heuristic
            }
        }
        crate::strategy::decide_with_bot(state, self)
    }

    /// Boltzmann / softmax selection over scored candidates.
    /// `temperature == 0.0` → pure argmax (deterministic).
    pub fn sample_softmax<C: Clone>(&mut self, scored: &[(C, f32)]) -> Option<C> {
        if scored.is_empty() {
            return None;
        }

        let temperature = self.profile.temperature;

        if temperature == 0.0 {
            return scored
                .iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(c, _)| c.clone());
        }

        // Shift by max for numerical stability before exponentiation.
        let max_score = scored
            .iter()
            .map(|(_, s)| *s)
            .fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<f32> = scored
            .iter()
            .map(|(_, s)| ((s - max_score) / temperature).exp())
            .collect();
        let total: f32 = weights.iter().sum();
        let mut threshold = self.rng.gen::<f32>() * total;
        for (i, w) in weights.iter().enumerate() {
            threshold -= w;
            if threshold <= 0.0 {
                return Some(scored[i].0.clone());
            }
        }
        scored.last().map(|(c, _)| c.clone())
    }

    /// Apply bid jitter with probability `profile.jitter`, adding 1..=max_jitter elektro.
    pub fn maybe_jitter(&mut self, base: u32, max_add: u8) -> u32 {
        if self.profile.jitter > 0.0 && max_add > 0 && self.rng.gen::<f32>() < self.profile.jitter {
            let add = self.rng.gen_range(1..=max_add) as u32;
            base.saturating_add(add)
        } else {
            base
        }
    }
}
