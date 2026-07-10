//! Headless heuristic game playing + paired fitness evaluation.
//!
//! Modeled on `powergrid-bot-strategy/tests/heuristic_termination.rs`: a seeded
//! N-player game driven entirely by `Bot::decide` until `GameOver`. With every
//! bot's noise silenced (see [`crate::genome::silence_noise`]) a fixed game seed
//! yields a fully deterministic result, so two profiles compared on the same
//! seed block are truly paired — the property the training journal found
//! essential (jittered A/B has a ±5pp noise floor that inverts small effects).

use std::sync::atomic::{AtomicUsize, Ordering};

use powergrid_bot_strategy::{Bot, BotProfile};
use powergrid_core::{
    actions::Action,
    map::default_map,
    rules::{apply_action, finish_ranks},
    state::GameState,
    types::{Phase, PlayerColor, PlayerId},
};

const COLORS: [PlayerColor; 6] = [
    PlayerColor::Red,
    PlayerColor::Blue,
    PlayerColor::Green,
    PlayerColor::Yellow,
    PlayerColor::Purple,
    PlayerColor::White,
];

/// Per-game action cap (a normal game ends well under 1000 actions).
const STEP_CAP: usize = 8000;

/// One scheduled game: which base seed, which seat the candidate sits in, and
/// which opponent profile fills each *other* seat (indices into an opponent
/// pool held by the caller). Fixed for a whole generation so all candidates are
/// compared on identical games (common random numbers).
#[derive(Clone)]
pub struct Match {
    pub seed: u64,
    pub candidate_seat: usize,
    pub opponent_pick: Vec<usize>,
}

#[derive(Clone, Copy, Default)]
pub struct Fitness {
    /// Mean rank value in [-1, 1] (1st = +1, last = -1); the CMA objective.
    pub mean_rank_value: f64,
    /// Fraction of games the candidate finished 1st.
    pub win_rate: f64,
    /// Games that failed to terminate within the cap (counted as worst place).
    pub aborted: usize,
    pub games: usize,
}

/// Deterministic player id for a seat, mixing the seed so different games use
/// different (but reproducible) uuids — matching the bridge's seat-uuid scheme
/// closely enough that bot RNG is seeded per (seed, seat).
fn player_id(seed: u64, seat: usize) -> PlayerId {
    PlayerId::from_u128(((seed as u128) << 8) | (seat as u128 + 1))
}

/// Play a single deterministic game and return the candidate's rank value.
///
/// `candidate` sits in `m.candidate_seat`; the remaining seats are filled from
/// `opponents` per `m.opponent_pick`. All profiles must already have noise
/// silenced. Rank value: `1st → +1`, `last → -1`, linearly spaced.
pub fn play_one(
    m: &Match,
    candidate: &BotProfile,
    opponents: &[BotProfile],
    num_players: usize,
) -> f64 {
    let mut state = GameState::new_with_seed(default_map(), num_players, m.seed);
    let ids: Vec<PlayerId> = (0..num_players).map(|s| player_id(m.seed, s)).collect();
    for (seat, id) in ids.iter().enumerate() {
        apply_action(
            &mut state,
            *id,
            Action::JoinGame {
                name: format!("P{seat}"),
                color: COLORS[seat],
            },
        )
        .expect("join");
    }
    apply_action(&mut state, ids[0], Action::StartGame).expect("start");

    // Build one persistent Bot per seat (persistent RNG, though noise is off).
    let mut opp_cursor = 0;
    let mut bots: Vec<Bot> = Vec::with_capacity(num_players);
    for (seat, id) in ids.iter().enumerate() {
        let profile = if seat == m.candidate_seat {
            candidate.clone()
        } else {
            let pick = m.opponent_pick[opp_cursor % m.opponent_pick.len()];
            opp_cursor += 1;
            opponents[pick % opponents.len()].clone()
        };
        bots.push(Bot::new(
            *id,
            format!("P{seat}"),
            COLORS[seat],
            profile,
            id.as_u128() as u64,
        ));
    }

    for _ in 0..STEP_CAP {
        if matches!(state.phase, Phase::GameOver { .. }) {
            break;
        }
        let mut acted = false;
        for bot in bots.iter_mut() {
            if let Some(action) = bot.decide(&state) {
                apply_action(&mut state, bot.id, action).expect("bot move must be legal");
                acted = true;
                break;
            }
        }
        if !acted {
            // No bot can move — should not happen (bots guarantee termination).
            return f64::NAN;
        }
    }

    if !matches!(state.phase, Phase::GameOver { .. }) {
        return f64::NAN; // aborted (cap hit)
    }

    let ranks = finish_ranks(&state);
    let candidate_id = ids[m.candidate_seat];
    let pos = ranks
        .iter()
        .find(|(id, _)| *id == candidate_id)
        .map(|(_, r)| *r)
        .expect("candidate must be ranked");
    rank_value(pos, num_players)
}

/// Linearly-spaced finish value: 1st → +1, last → -1.
pub fn rank_value(pos: usize, num_players: usize) -> f64 {
    if num_players <= 1 {
        return 1.0;
    }
    1.0 - 2.0 * (pos as f64 - 1.0) / (num_players as f64 - 1.0)
}

/// Evaluate a candidate over the whole schedule, parallelized across `threads`.
///
/// The schedule (and thus the games) is identical for every candidate in a
/// generation, so returned fitnesses are directly comparable (paired).
pub fn evaluate(
    candidate: &BotProfile,
    opponents: &[BotProfile],
    schedule: &[Match],
    num_players: usize,
    threads: usize,
) -> Fitness {
    let next = AtomicUsize::new(0);
    let n = schedule.len();
    let threads = threads.max(1).min(n.max(1));

    let mut per_thread: Vec<(f64, usize, usize)> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| {
                    let mut sum = 0.0;
                    let mut wins = 0usize;
                    let mut aborts = 0usize;
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break;
                        }
                        let v = play_one(&schedule[i], candidate, opponents, num_players);
                        if v.is_nan() {
                            // Aborted game → treat as worst finish.
                            aborts += 1;
                            sum += -1.0;
                        } else {
                            sum += v;
                            if v >= 1.0 - 1e-9 {
                                wins += 1;
                            }
                        }
                    }
                    (sum, wins, aborts)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let (mut sum, mut wins, mut aborts) = (0.0, 0usize, 0usize);
    for (s, w, a) in per_thread.drain(..) {
        sum += s;
        wins += w;
        aborts += a;
    }
    let games = n.max(1);
    Fitness {
        mean_rank_value: sum / games as f64,
        win_rate: wins as f64 / games as f64,
        aborted: aborts,
        games: n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use powergrid_bot_strategy::{embedded_registry, BotProfile};

    fn silenced(mut p: BotProfile) -> BotProfile {
        crate::genome::silence_noise(&mut p);
        p
    }

    #[test]
    fn game_is_deterministic_when_noise_off() {
        let reg = embedded_registry();
        let hard = silenced(reg.hard.clone());
        let normal = silenced(reg.normal.clone());
        // Several seeds, several repeats each, and a rotated candidate seat:
        // any residual HashMap-iteration-order dependence in decision logic
        // would surface as a divergent replay here.
        for seed in [12345u64, 7, 99, 2024, 555] {
            for seat in 0..4 {
                let m = Match {
                    seed,
                    candidate_seat: seat,
                    opponent_pick: vec![0, 0, 0],
                };
                let first = play_one(&m, &hard, std::slice::from_ref(&normal), 4);
                assert!(first.is_finite(), "game should terminate (seed {seed})");
                for _ in 0..3 {
                    let again = play_one(&m, &hard, std::slice::from_ref(&normal), 4);
                    assert_eq!(first, again, "replay diverged (seed {seed}, seat {seat})");
                }
            }
        }
    }

    #[test]
    fn rank_value_endpoints() {
        assert!((rank_value(1, 4) - 1.0).abs() < 1e-9);
        assert!((rank_value(4, 4) + 1.0).abs() < 1e-9);
    }
}
