//! Drives a real local game (one inspected seat + heuristic bot opponents) so
//! the inspector can be fed genuine observations and action masks. Mirrors
//! the synchronous `Game::start`/`drive_bots` pattern in `powergrid-py`.

use std::collections::HashMap;

use powergrid_bot_strategy::encoding::{build_action_mask, build_observation, current_actor_id};
use powergrid_bot_strategy::{default_registry, Bot};
use powergrid_core::rules::apply_action;
use powergrid_core::{
    default_map, Action, ActionError, BotDifficulty, GameState, Phase, PlayerColor, PlayerId,
};
use uuid::Uuid;

const SEAT_COLORS: [PlayerColor; 6] = [
    PlayerColor::Red,
    PlayerColor::Blue,
    PlayerColor::Green,
    PlayerColor::Yellow,
    PlayerColor::Purple,
    PlayerColor::White,
];

/// Setup for a new [`GameDriver`]. Seat 0 (the host) is the inspected player;
/// the remaining seats are heuristic bots of `difficulty`.
pub struct GameConfig {
    pub players: usize,
    pub difficulty: BotDifficulty,
    pub seed: u64,
    pub end_game_cities: Option<u8>,
}

impl Default for GameConfig {
    fn default() -> Self {
        Self {
            players: 4,
            difficulty: BotDifficulty::Normal,
            seed: 1,
            end_game_cities: None,
        }
    }
}

/// Deterministic per-seat player id, derived from the config seed so a given
/// seed always reproduces the same game.
fn seat_id(base_seed: u64, index: usize) -> PlayerId {
    let lo = base_seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(index as u64);
    let hi = base_seed
        .wrapping_mul(1442695040888963407)
        .wrapping_add(index as u64 + 1);
    Uuid::from_u128(((hi as u128) << 64) | lo as u128)
}

/// A live local game with one inspected seat and persistent heuristic bots
/// for the rest.
pub struct GameDriver {
    state: GameState,
    inspected: PlayerId,
    bots: HashMap<PlayerId, Bot>,
    note: Option<String>,
}

impl GameDriver {
    /// Sets up a new game per `cfg` and advances it to the first inspected
    /// turn (or terminal state).
    pub fn new(cfg: &GameConfig) -> Result<Self, ActionError> {
        let players = cfg.players.clamp(2, SEAT_COLORS.len());
        let mut state = GameState::new_with_seed(default_map(), players, cfg.seed);

        let registry = default_registry();
        let profile = registry.profile_for(cfg.difficulty).clone();

        let mut inspected = None;
        let mut bots = HashMap::new();
        for (i, &color) in SEAT_COLORS.iter().take(players).enumerate() {
            let id = seat_id(cfg.seed, i);
            let name = format!("Seat {}", i + 1);
            apply_action(
                &mut state,
                id,
                Action::JoinGame {
                    name: name.clone(),
                    color,
                },
            )?;
            if i == 0 {
                inspected = Some(id);
            } else {
                bots.insert(
                    id,
                    Bot::new(
                        id,
                        name,
                        color,
                        profile.clone(),
                        cfg.seed.wrapping_add(i as u64),
                    ),
                );
            }
        }
        let inspected = inspected.expect("players >= 2, so seat 0 exists");
        apply_action(&mut state, inspected, Action::StartGame)?;

        if let Some(n) = cfg.end_game_cities {
            state.end_game_cities = n;
        }

        let mut driver = Self {
            state,
            inspected,
            bots,
            note: None,
        };
        driver.advance();
        Ok(driver)
    }

    /// Drives bot seats until the inspected seat is to act, the game ends, or
    /// no single actor remains (capped to avoid infinite loops on a stalled
    /// heuristic).
    pub fn advance(&mut self) {
        self.note = None;
        for _ in 0..500 {
            if matches!(self.state.phase, Phase::GameOver { .. }) {
                return;
            }
            let Some(actor_id) = current_actor_id(&self.state) else {
                return;
            };
            if actor_id == self.inspected {
                return;
            }
            let Some(bot) = self.bots.get_mut(&actor_id) else {
                self.note = Some(format!("no bot registered for seat {actor_id}"));
                return;
            };
            let Some(action) = bot.decide(&self.state) else {
                self.note = Some("bot had no legal move; stalled".to_string());
                return;
            };
            if let Err(e) = apply_action(&mut self.state, actor_id, action) {
                self.note = Some(format!("bot action error: {e}"));
                return;
            }
        }
        self.note = Some("hit iteration cap while advancing bots".to_string());
    }

    /// Applies `action` for the inspected seat, then [`advance`](Self::advance)s.
    pub fn step_inspected(&mut self, action: Action) -> Result<(), ActionError> {
        apply_action(&mut self.state, self.inspected, action)?;
        self.advance();
        Ok(())
    }

    /// The inspected seat's observation vector (`OBS_SIZE` long).
    pub fn observation(&self) -> Vec<f32> {
        build_observation(&self.state, self.inspected)
    }

    /// The inspected seat's legal-action mask (`N_ACTIONS` long).
    pub fn action_mask(&self) -> Vec<u8> {
        build_action_mask(&self.state, self.inspected)
    }

    pub fn is_inspected_turn(&self) -> bool {
        current_actor_id(&self.state) == Some(self.inspected)
    }

    pub fn winner_name(&self) -> Option<String> {
        match &self.state.phase {
            Phase::GameOver { winner } => self.state.player(*winner).map(|p| p.name.clone()),
            _ => None,
        }
    }

    pub fn state(&self) -> &GameState {
        &self.state
    }

    pub fn inspected_id(&self) -> PlayerId {
        self.inspected
    }

    /// Short human-readable summary of the current phase, round, and actor —
    /// for a status banner.
    pub fn status(&self) -> String {
        let phase_name = match &self.state.phase {
            Phase::Lobby => "Lobby",
            Phase::PlayerOrder => "PlayerOrder",
            Phase::Auction { .. } => "Auction",
            Phase::DiscardPlant { .. } => "DiscardPlant",
            Phase::DiscardResource { .. } => "DiscardResource",
            Phase::BuyResources { .. } => "BuyResources",
            Phase::BuildCities { .. } => "BuildCities",
            Phase::Bureaucracy { .. } => "Bureaucracy",
            Phase::PowerCitiesFuel { .. } => "PowerCitiesFuel",
            Phase::GameOver { .. } => "GameOver",
        };

        let mut summary = if let Phase::GameOver { .. } = &self.state.phase {
            match self.winner_name() {
                Some(name) => format!("GameOver — winner: {name}"),
                None => "GameOver".to_string(),
            }
        } else {
            let actor = current_actor_id(&self.state)
                .and_then(|id| self.state.player(id))
                .map(|p| p.name.as_str())
                .unwrap_or("-");
            let turn = if self.is_inspected_turn() {
                "your turn"
            } else {
                "bot turn"
            };
            format!(
                "Round {} — {phase_name} — actor: {actor} ({turn})",
                self.state.round
            )
        };

        if let Some(note) = &self.note {
            summary.push_str(" — ");
            summary.push_str(note);
        }
        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use powergrid_bot_strategy::encoding::{action_id_to_action, OBS_SIZE};
    use powergrid_bot_strategy::policy::sample_masked;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn new_game_reaches_inspected_turn_or_terminal_with_valid_obs_and_mask() {
        let cfg = GameConfig {
            players: 4,
            difficulty: BotDifficulty::Normal,
            seed: 42,
            end_game_cities: None,
        };
        let driver = GameDriver::new(&cfg).expect("game starts");

        let is_terminal = matches!(driver.state().phase, Phase::GameOver { .. });
        assert!(driver.is_inspected_turn() || is_terminal);

        let obs = driver.observation();
        assert_eq!(obs.len(), OBS_SIZE);

        if driver.is_inspected_turn() {
            let mask = driver.action_mask();
            assert!(
                mask.iter().any(|&m| m != 0),
                "expected at least one legal move"
            );
        }
    }

    #[test]
    fn step_inspected_with_legal_action_advances() {
        let cfg = GameConfig {
            players: 4,
            difficulty: BotDifficulty::Normal,
            seed: 7,
            end_game_cities: None,
        };
        let mut driver = GameDriver::new(&cfg).expect("game starts");
        let mut rng = SmallRng::seed_from_u64(99);

        // Step a handful of inspected turns to exercise step_inspected end-to-end.
        for _ in 0..10 {
            if !driver.is_inspected_turn() {
                break;
            }
            let mask = driver.action_mask();
            let logits: Vec<f32> = mask
                .iter()
                .map(|&m| if m != 0 { 1.0 } else { -1e9 })
                .collect();
            let action_id = sample_masked(&logits, &mask, &mut rng).expect("legal action exists");
            let action =
                action_id_to_action(action_id as u16, driver.state(), driver.inspected_id());
            driver
                .step_inspected(action)
                .expect("legal action applies cleanly");
        }
    }
}
