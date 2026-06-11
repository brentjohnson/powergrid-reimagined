//! Regression guard: full games between heuristic bots must terminate.
//!
//! Power Grid has no forced progress — the game only ends when someone builds
//! `end_game_cities`. The heuristics used to deadlock short of that (no plant
//! clears the upgrade bar, no city beyond powering headroom gets built), so
//! all-bot games could loop forever. The endgame overbuild rule plus the
//! late-game urgency scaling break that fixed point; these games assert it
//! stays broken.

use powergrid_bot_strategy::{default_registry, Bot};
use powergrid_core::{
    actions::Action,
    map::default_map,
    rules::apply_action,
    state::GameState,
    types::{BotDifficulty, Phase, PlayerColor, PlayerId},
};

const COLORS: [PlayerColor; 4] = [
    PlayerColor::Red,
    PlayerColor::Blue,
    PlayerColor::Green,
    PlayerColor::Yellow,
];

/// Generous per-game action cap: a normal game ends well under 1000 actions,
/// and even a long stall-then-overbuild recovery fits with lots of margin.
const STEP_CAP: usize = 8000;

fn make_bot(id: PlayerId, difficulty: BotDifficulty) -> Bot {
    let registry = default_registry();
    let profile = registry.profile_for(difficulty).clone();
    Bot::new(
        id,
        format!("{difficulty:?}"),
        PlayerColor::Red,
        profile,
        id.as_u128() as u64,
    )
}

/// Start a seeded 4-player game, drive it with the given bots until GameOver
/// or the step cap, and return the number of actions played on success.
fn play_to_completion(seed: u64, difficulties: [BotDifficulty; 4]) -> Result<usize, String> {
    let mut state = GameState::new_with_seed(default_map(), 4, seed);
    let ids: Vec<PlayerId> = (0..4)
        .map(|i| PlayerId::from_u128(((seed as u128) << 8) | (i + 1) as u128))
        .collect();
    for (i, id) in ids.iter().enumerate() {
        apply_action(
            &mut state,
            *id,
            Action::JoinGame {
                name: format!("P{i}"),
                color: COLORS[i],
            },
        )
        .expect("join");
    }
    apply_action(&mut state, ids[0], Action::StartGame).expect("start");

    let mut bots: Vec<Bot> = ids
        .iter()
        .zip(difficulties)
        .map(|(&id, difficulty)| make_bot(id, difficulty))
        .collect();

    for step in 0..STEP_CAP {
        if matches!(state.phase, Phase::GameOver { .. }) {
            return Ok(step);
        }
        // Mirror run_bot_pump: first bot with a move acts.
        let mut acted = false;
        for bot in bots.iter_mut() {
            if let Some(action) = bot.decide(&state) {
                apply_action(&mut state, bot.id, action).expect("bot move must be legal");
                acted = true;
                break;
            }
        }
        if !acted {
            return Err(format!("no bot has a move in phase {:?}", state.phase));
        }
    }
    Err(format!(
        "game (seed {seed}) did not finish within {STEP_CAP} actions; round {}, phase {:?}",
        state.round, state.phase
    ))
}

#[test]
fn normal_bot_games_terminate() {
    for seed in 0..5u64 {
        let steps = play_to_completion(2000 + seed, [BotDifficulty::Normal; 4])
            .unwrap_or_else(|e| panic!("{e}"));
        println!("normal game seed {seed} finished in {steps} actions");
    }
}

#[test]
fn mixed_difficulty_games_terminate() {
    let mix = [
        BotDifficulty::Easy,
        BotDifficulty::Normal,
        BotDifficulty::Hard,
        BotDifficulty::Hard,
    ];
    for seed in 0..3u64 {
        let steps = play_to_completion(3000 + seed, mix).unwrap_or_else(|e| panic!("{e}"));
        println!("mixed game seed {seed} finished in {steps} actions");
    }
}
