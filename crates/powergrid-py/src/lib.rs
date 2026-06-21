use std::collections::HashMap;
use std::sync::Arc;

use numpy::{IntoPyArray, PyArray1};
use powergrid_bot_strategy::encoding::{
    action_id_to_action, build_action_mask, build_observation, compute_legal_move_info,
    current_actor_id, DISCARD_RESOURCE_BASE, N_ACTIONS, OBS_SIZE, POWER_CITIES_BASE,
    POWER_FUEL_BASE,
};
use powergrid_bot_strategy::policy::MlpPolicy;
use powergrid_bot_strategy::{default_registry, Bot};
use powergrid_core::{
    actions::Action,
    map::default_map,
    rules::apply_action,
    state::GameState,
    types::{BotDifficulty, Phase, PlayerColor},
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use uuid::Uuid;

/// How non-learner seats are driven by `step_vs_bots` / `advance_bots`.
enum OpponentMode {
    Heuristic(BotDifficulty),
    /// Frozen policy snapshot loaded via `load_opponent_policy` (self-play).
    Policy,
}

// ---------------------------------------------------------------------------
// Game Python class
// ---------------------------------------------------------------------------

#[pyclass]
struct Game {
    state: GameState,
    /// Frozen policy snapshot driving opponent seats in `"policy"` mode.
    opponent_policy: Option<Arc<MlpPolicy>>,
    /// Persistent per-seat policy bots, so each bot's sampling RNG keeps its
    /// state across decisions within a game (mirrors `Session`-held bots).
    opponent_bots: HashMap<Uuid, Bot>,
}

#[pymethods]
impl Game {
    #[new]
    fn new(num_players: usize, seed: Option<u64>) -> PyResult<Self> {
        if !(2..=6).contains(&num_players) {
            return Err(PyValueError::new_err("num_players must be 2–6"));
        }
        let map = default_map();
        let state = match seed {
            Some(s) => GameState::new_with_seed(map, num_players, s),
            None => GameState::new(map, num_players),
        };
        Ok(Game {
            state,
            opponent_policy: None,
            opponent_bots: HashMap::new(),
        })
    }

    /// Load a frozen policy snapshot (PGRLPOL1 bytes, as written by
    /// `export_policy.py` / `policy_state_dict_to_bytes`) to drive opponent
    /// seats when `step_vs_bots`/`advance_bots` are called with
    /// difficulty `"policy"`.
    fn load_opponent_policy(&mut self, bytes: Vec<u8>) -> PyResult<()> {
        let policy = MlpPolicy::from_bytes(&bytes)
            .map_err(|e| PyValueError::new_err(format!("invalid policy weights: {:?}", e)))?;
        self.opponent_policy = Some(Arc::new(policy));
        self.opponent_bots.clear();
        Ok(())
    }

    /// Join all players and start the game.
    /// `colors` must be snake_case strings: "red", "blue", "green", "yellow", "purple", "white".
    fn start(&mut self, player_names: Vec<String>, colors: Vec<String>) -> PyResult<()> {
        if player_names.len() != colors.len() {
            return Err(PyValueError::new_err(
                "player_names and colors must have the same length",
            ));
        }
        let mut host_id: Option<Uuid> = None;
        let base_seed = self.state.rng_seed.unwrap_or(0);
        for (i, (name, color_str)) in player_names.iter().zip(colors.iter()).enumerate() {
            let color: PlayerColor = serde_json::from_value(serde_json::Value::String(
                color_str.clone(),
            ))
            .map_err(|e| PyValueError::new_err(format!("invalid color '{}': {}", color_str, e)))?;
            // Deterministic UUID derived from seed+index so reset() with the same seed
            // produces identical agent IDs (required for reproducibility).
            let id = if base_seed != 0 {
                let lo = base_seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(i as u64);
                let hi = base_seed
                    .wrapping_mul(1442695040888963407)
                    .wrapping_add(i as u64 + 1);
                Uuid::from_u128((hi as u128) << 64 | lo as u128)
            } else {
                Uuid::new_v4()
            };
            apply_action(
                &mut self.state,
                id,
                Action::JoinGame {
                    name: name.clone(),
                    color,
                },
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
            if host_id.is_none() {
                host_id = Some(id);
            }
        }
        if let Some(hid) = host_id {
            apply_action(&mut self.state, hid, Action::StartGame)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        Ok(())
    }

    /// Override the end-game city trigger (curriculum learning). Must be
    /// called after `start()`, which sets the rulebook default from player
    /// count; this replaces it for the rest of the game. The trigger is part
    /// of the observation, so a policy can condition on the current value.
    fn set_end_game_cities(&mut self, n: u8) -> PyResult<()> {
        if matches!(self.state.phase, Phase::Lobby) {
            return Err(PyValueError::new_err(
                "set_end_game_cities must be called after start()",
            ));
        }
        let max = self.state.map.cities.len();
        if n == 0 || n as usize > max {
            return Err(PyValueError::new_err(format!(
                "end_game_cities must be in 1..={max}"
            )));
        }
        self.state.end_game_cities = n;
        Ok(())
    }

    /// Serialized `GameStateView` as a JSON string. When `viewer` is given,
    /// that player's own money is included (opponent money is always zeroed,
    /// matching what a seated player may see).
    #[pyo3(signature = (viewer=None))]
    fn state_json(&self, viewer: Option<&str>) -> PyResult<String> {
        let viewer_id = match viewer {
            Some(v) => Some(Uuid::parse_str(v).map_err(|e| PyValueError::new_err(e.to_string()))?),
            None => None,
        };
        Ok(
            serde_json::to_string(&self.state.view_for(viewer_id))
                .expect("serialize GameStateView"),
        )
    }

    /// Player IDs in join order (same as `player_order` after `start()`).
    fn player_ids(&self) -> Vec<String> {
        self.state
            .players
            .iter()
            .map(|p| p.id.to_string())
            .collect()
    }

    /// UUID string of the player whose turn it is, or None if no single actor (Lobby, GameOver).
    fn current_actor(&self) -> Option<String> {
        current_actor_id(&self.state).map(|id| id.to_string())
    }

    fn is_terminal(&self) -> bool {
        matches!(self.state.phase, Phase::GameOver { .. })
    }

    fn winner(&self) -> Option<String> {
        if let Phase::GameOver { winner } = &self.state.phase {
            Some(winner.to_string())
        } else {
            None
        }
    }

    /// Apply an action. Raises `ValueError` on invalid actions (including wrong-phase, not-your-turn, etc.).
    fn apply(&mut self, actor: &str, action_json: &str) -> PyResult<()> {
        let actor_id = Uuid::parse_str(actor).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let action: Action = serde_json::from_str(action_json)
            .map_err(|e| PyValueError::new_err(format!("invalid action JSON: {}", e)))?;
        apply_action(&mut self.state, actor_id, action)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Ask the Rust strategy bot to decide an action for `actor`.
    /// Returns the action as a JSON string, or None if the bot has no move.
    fn bot_decide(&self, actor: &str, difficulty: &str) -> PyResult<Option<String>> {
        let actor_id = Uuid::parse_str(actor).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let player = self
            .state
            .players
            .iter()
            .find(|p| p.id == actor_id)
            .ok_or_else(|| PyValueError::new_err("actor not found in game"))?;
        let diff = parse_difficulty(difficulty);
        let registry = default_registry();
        let profile = registry.profile_for(diff).clone();
        let seed = actor_id.as_u128() as u64;
        let mut bot = Bot::new(actor_id, player.name.clone(), player.color, profile, seed);
        Ok(bot
            .decide(&self.state)
            .map(|a| serde_json::to_string(&a).expect("serialize action")))
    }

    /// Sorted list of all city IDs in the map (stable across calls — use to build the city index).
    fn city_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.state.map.cities.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// JSON describing which moves are legal for `actor` right now.
    /// Python uses this to build the action mask without re-implementing game rules.
    fn legal_move_info(&self, actor: &str) -> PyResult<String> {
        let actor_id = Uuid::parse_str(actor).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let info = compute_legal_move_info(&self.state, actor_id);
        Ok(serde_json::to_string(&info).expect("serialize LegalMoveInfo"))
    }

    // -----------------------------------------------------------------------
    // Fast native methods — no JSON, direct numpy output
    // -----------------------------------------------------------------------

    /// Observation vector for `actor` as a float32 numpy array of length OBS_SIZE.
    /// Bypasses JSON serialisation; ~10× faster than state_json() + encode_observation().
    fn observation<'py>(
        &self,
        py: Python<'py>,
        actor: &str,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        let actor_id = Uuid::parse_str(actor).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(build_observation(&self.state, actor_id).into_pyarray(py))
    }

    /// Action mask for `actor` as a uint8 numpy array of length N_ACTIONS.
    /// Bypasses JSON serialisation; ~10× faster than legal_move_info() + mask_from_info().
    fn action_mask<'py>(&self, py: Python<'py>, actor: &str) -> PyResult<Bound<'py, PyArray1<u8>>> {
        let actor_id = Uuid::parse_str(actor).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(build_action_mask(&self.state, actor_id).into_pyarray(py))
    }

    /// Apply action by integer id (0..N_ACTIONS). Bypasses JSON encoding.
    fn apply_action_id(&mut self, actor: &str, action_id: u16) -> PyResult<()> {
        let actor_id = Uuid::parse_str(actor).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let action = action_id_to_action(action_id, &self.state, actor_id);
        apply_action(&mut self.state, actor_id, action)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Advance all non-learner seats until it's the learner's turn or the game
    /// is terminal. Returns True if terminal. `difficulty` is
    /// `"easy" | "normal" | "hard" | "expert"` (heuristic bots) or `"policy"`
    /// (the snapshot loaded via `load_opponent_policy`).
    fn advance_bots(&mut self, learner: &str, difficulty: &str) -> PyResult<bool> {
        let learner_id =
            Uuid::parse_str(learner).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mode = self.opponent_mode(difficulty)?;
        self.drive_bots(learner_id, &mode)?;
        Ok(matches!(self.state.phase, Phase::GameOver { .. }))
    }

    /// Fused vs-bots step: apply `action_id` for the learner, drive all bot
    /// seats until the learner acts again or the game ends, and return
    /// `(obs, mask, reward, terminal, learner_cities, powered_now, opp_powered_max)`
    /// in one PyO3 round-trip.
    /// Reward is +1 if the learner won, -1 if anyone else did, 0 otherwise.
    /// `obs` and `mask` are zero arrays when `terminal` is True.
    /// `learner_cities` is the learner's current city count.
    /// `powered_now` is the number of cities the learner just got paid for, if
    /// this action resolved their powering for the round (PowerCities, or the
    /// PowerCitiesFuel split that completes it); 0 otherwise.
    /// `opp_powered_max` is the most cities any opponent powered in the round
    /// the learner just resolved (read after the bots have acted), or 0 when
    /// this step did not resolve the learner's powering. The pair lets Python
    /// shape rewards on `powered_now - opp_powered_max` — the learner's powered
    /// lead over the field — without a second round-trip; both are 0 on
    /// non-powering steps so the relative term is 0 there too.
    #[allow(clippy::type_complexity)]
    fn step_vs_bots<'py>(
        &mut self,
        py: Python<'py>,
        learner: &str,
        action_id: u16,
        difficulty: &str,
    ) -> PyResult<(
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<u8>>,
        f32,
        bool,
        u32,
        u32,
        u32,
    )> {
        let learner_id =
            Uuid::parse_str(learner).map_err(|e| PyValueError::new_err(e.to_string()))?;
        let mode = self.opponent_mode(difficulty)?;

        let aid = action_id as usize;
        let was_power_action = (POWER_CITIES_BASE..DISCARD_RESOURCE_BASE).contains(&aid)
            || (POWER_FUEL_BASE..N_ACTIONS).contains(&aid);

        let action = action_id_to_action(action_id, &self.state, learner_id);
        apply_action(&mut self.state, learner_id, action)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        // Powering resolves immediately unless the action paused on an
        // ambiguous hybrid fuel split for the learner.
        let power_pending = matches!(
            &self.state.phase,
            Phase::PowerCitiesFuel { player, .. } if *player == learner_id
        );
        let powered_now = if was_power_action && !power_pending {
            self.state
                .player(learner_id)
                .map(|p| p.last_cities_powered as u32)
                .unwrap_or(0)
        } else {
            0
        };

        if !matches!(self.state.phase, Phase::GameOver { .. }) {
            self.drive_bots(learner_id, &mode)?;
        }

        // Best opponent powering this round, read after the bots have acted so
        // it reflects the same round the learner just resolved. Gated to the
        // same steps as `powered_now` so the relative term is 0 off-round.
        let opp_powered_max = if was_power_action && !power_pending {
            self.state
                .players
                .iter()
                .filter(|p| p.id != learner_id)
                .map(|p| p.last_cities_powered as u32)
                .max()
                .unwrap_or(0)
        } else {
            0
        };

        let (reward, terminal) = match &self.state.phase {
            Phase::GameOver { winner } => {
                let r = if *winner == learner_id {
                    1.0_f32
                } else {
                    -1.0_f32
                };
                (r, true)
            }
            _ => (0.0_f32, false),
        };

        let (obs, mask) = if terminal {
            (vec![0.0f32; OBS_SIZE], vec![0u8; N_ACTIONS])
        } else {
            (
                build_observation(&self.state, learner_id),
                build_action_mask(&self.state, learner_id),
            )
        };
        let learner_cities = self.state.player_cities(learner_id).len() as u32;

        Ok((
            obs.into_pyarray(py),
            mask.into_pyarray(py),
            reward,
            terminal,
            learner_cities,
            powered_now,
            opp_powered_max,
        ))
    }
}

fn parse_difficulty(s: &str) -> BotDifficulty {
    match s {
        "easy" => BotDifficulty::Easy,
        "hard" => BotDifficulty::Hard,
        // Python-driven bots are rebuilt per decision and never get the RL
        // policy attached, so "expert" plays as the hard-style heuristic here.
        "expert" => BotDifficulty::Expert,
        _ => BotDifficulty::Normal,
    }
}

impl Game {
    /// Resolve a difficulty string into an opponent-driving mode.
    fn opponent_mode(&self, difficulty: &str) -> PyResult<OpponentMode> {
        if difficulty == "policy" {
            if self.opponent_policy.is_none() {
                return Err(PyValueError::new_err(
                    "difficulty \"policy\" requires load_opponent_policy() first",
                ));
            }
            Ok(OpponentMode::Policy)
        } else {
            Ok(OpponentMode::Heuristic(parse_difficulty(difficulty)))
        }
    }

    /// Drive every non-learner seat until the learner is the current actor,
    /// the game ends, or there is no single actor.
    fn drive_bots(&mut self, learner: Uuid, mode: &OpponentMode) -> PyResult<()> {
        let registry = default_registry();
        for _ in 0..500 {
            if matches!(self.state.phase, Phase::GameOver { .. }) {
                return Ok(());
            }
            let Some(actor_id) = current_actor_id(&self.state) else {
                return Ok(());
            };
            if actor_id == learner {
                return Ok(());
            }
            let (name, color) = {
                let player = self
                    .state
                    .players
                    .iter()
                    .find(|p| p.id == actor_id)
                    .ok_or_else(|| PyValueError::new_err("bot actor not found in game"))?;
                (player.name.clone(), player.color)
            };
            let seed = actor_id.as_u128() as u64;
            let action = match mode {
                OpponentMode::Heuristic(diff) => {
                    let profile = registry.profile_for(*diff).clone();
                    let mut bot = Bot::new(actor_id, name, color, profile, seed);
                    bot.decide(&self.state)
                }
                OpponentMode::Policy => {
                    let policy = self
                        .opponent_policy
                        .clone()
                        .expect("opponent_mode verified the policy is loaded");
                    let state = &self.state;
                    let bot = self.opponent_bots.entry(actor_id).or_insert_with(|| {
                        // Expert profile = the heuristic fallback used if the
                        // snapshot is unusable (non-default map).
                        let profile = registry.profile_for(BotDifficulty::Expert).clone();
                        Bot::new(actor_id, name, color, profile, seed).with_policy(policy)
                    });
                    bot.decide(state)
                }
            };
            let Some(action) = action else {
                return Err(PyValueError::new_err("bot has no move on its own turn"));
            };
            apply_action(&mut self.state, actor_id, action)
                .map_err(|e| PyValueError::new_err(format!("bot move rejected: {}", e)))?;
        }
        Err(PyValueError::new_err("bot loop exceeded 500 iterations"))
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn powergrid_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Game>()?;
    Ok(())
}
