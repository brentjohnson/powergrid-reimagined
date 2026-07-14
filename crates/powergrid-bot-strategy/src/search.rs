//! Play-time search: PUCT MCTS over **macro** actions, guided by the RL policy.
//!
//! Phase 3 of the beat-humans plan. The trained macro policy is strong on its
//! own (~62% vs normal), but a policy is a one-shot guess; search turns "thinking
//! longer" into strength by looking ahead. The macro action space makes this
//! tractable — a game is ~50 macro decisions deep, not ~600 primitive ones.
//!
//! Design (a faithful port of `alphazero/mcts.py`, adapted to macros):
//! - Every tree node is a game state at some actor's macro decision. Each actor
//!   selects a child by PUCT using the policy's softmax as the prior.
//! - Leaf value: the exported PPO **value net** (`policy::ValueNet`), one forward
//!   pass per seat from that seat's observation perspective — microseconds, and
//!   what makes search fast enough to use. Without a value net it falls back to a
//!   policy **rollout** to terminal (correct but ~1000× slower). Terminal nodes
//!   use the engine's exact finish order (`rules::finish_ranks`). All values are
//!   per-seat rank/return in ~[-1, 1].
//! - Single-actor-per-turn, so a plain per-seat value map backs up the tree.
//! - **Determinization:** the search only ever sees a *reshuffled* copy of the
//!   unseen plant deck, so it cannot exploit the true deck order (which would be
//!   an unfair information advantage vs a human). Multiple determinized worlds
//!   are searched and their root visit counts summed.

use std::collections::HashMap;

use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use powergrid_core::state::GameState;
use powergrid_core::types::{Phase, PlayerId};

use crate::encoding::build_observation;
use crate::macro_actions::{apply_macro, legal_macros, macro_current_actor, resolve_auto_phases};
use crate::policy::{MlpPolicy, ValueNet};

/// Per-seat value map (rank value in [-1, 1], 1st = +1, last = -1).
type Values = HashMap<PlayerId, f64>;

#[derive(Clone)]
pub struct SearchConfig {
    /// MCTS simulations per determinized world.
    pub num_sims: usize,
    /// Number of reshuffled-deck worlds to search and sum (1 = search a single
    /// determinized world). Higher averages out the hidden deck order.
    pub determinizations: usize,
    /// PUCT exploration constant.
    pub cpuct: f64,
    /// First-play-urgency reduction: unvisited children are scored at
    /// (parent mean − fpu_reduction) rather than 0.
    pub fpu_reduction: f64,
    /// Hard cap on rollout length (macros) — a safety net; games end well under.
    pub rollout_cap: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            num_sims: 100,
            determinizations: 1,
            cpuct: 1.5,
            fpu_reduction: 0.2,
            rollout_cap: 400,
        }
    }
}

struct Edge {
    macro_id: u16,
    prior: f64,
    n: u32,
    w: f64,
    child: Option<Box<Node>>,
}

struct Node {
    state: GameState,
    actor: PlayerId,
    edges: Vec<Edge>,
    expanded: bool,
}

/// Choose a macro for `actor` by searching from `state` with the `policy` as
/// prior. Returns the most-visited root macro (summed over determinizations), or
/// `None` if there is no legal macro (caller should fall back to the heuristic).
///
/// `seed` makes the search reproducible (determinization reshuffles + rollout
/// tie-breaks are seeded from it), which the eval harness relies on.
pub fn choose_macro(
    state: &GameState,
    actor: PlayerId,
    policy: &MlpPolicy,
    value: Option<&ValueNet>,
    cfg: &SearchConfig,
    seed: u64,
) -> Option<u16> {
    // A single legal macro needs no search.
    let legal = legal_macros(state, actor);
    let legal_ids: Vec<u16> = (0..legal.len() as u16)
        .filter(|&i| legal[i as usize])
        .collect();
    match legal_ids.len() {
        0 => return None,
        1 => return Some(legal_ids[0]),
        _ => {}
    }

    let mut visits: HashMap<u16, u32> = HashMap::new();
    for d in 0..cfg.determinizations.max(1) {
        let mut rng = SmallRng::seed_from_u64(seed ^ ((d as u64).wrapping_mul(0x9E3779B97F4A7C15)));
        let mut root_state = state.clone();
        determinize(&mut root_state, &mut rng);
        let mut root = Node {
            state: root_state,
            actor,
            edges: Vec::new(),
            expanded: false,
        };
        for _ in 0..cfg.num_sims {
            simulate(&mut root, policy, value, cfg, &mut rng);
        }
        for e in &root.edges {
            *visits.entry(e.macro_id).or_insert(0) += e.n;
        }
    }

    visits.into_iter().max_by_key(|&(_, n)| n).map(|(m, _)| m)
}

/// Reshuffle the unseen plant deck (and the step-3 hold-back pile) in place. The
/// *set* of cards is known (they live in the state); only their order is hidden,
/// so reshuffling is a sound information-set model of the real uncertainty.
fn determinize(state: &mut GameState, rng: &mut SmallRng) {
    state.market.deck.shuffle(rng);
    if let Some(below) = state.market.below_step3.as_mut() {
        below.shuffle(rng);
    }
}

/// One MCTS simulation from `node`, returning the per-seat value backed up.
fn simulate(
    node: &mut Node,
    policy: &MlpPolicy,
    value: Option<&ValueNet>,
    cfg: &SearchConfig,
    rng: &mut SmallRng,
) -> Values {
    if matches!(node.state.phase, Phase::GameOver { .. }) {
        return terminal_values(&node.state);
    }
    if !node.expanded {
        expand(node, policy);
        // Leaf value: the value net (one forward pass per seat) when available,
        // else a policy rollout to terminal (correct but far slower).
        return leaf_values(&node.state, policy, value, cfg, rng);
    }

    // Select the edge maximizing PUCT for this node's actor.
    let total_n: u32 = node.edges.iter().map(|e| e.n).sum();
    let sqrt_total = (total_n.max(1) as f64).sqrt();
    let parent_q = if total_n > 0 {
        node.edges.iter().map(|e| e.w).sum::<f64>() / total_n as f64
    } else {
        0.0
    };
    let best = (0..node.edges.len())
        .max_by(|&a, &b| {
            puct(&node.edges[a], sqrt_total, parent_q, cfg)
                .partial_cmp(&puct(&node.edges[b], sqrt_total, parent_q, cfg))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("expanded node has edges");

    // Descend (lazily create the child state by applying the macro).
    let values = {
        let edge = &mut node.edges[best];
        if edge.child.is_none() {
            let mut child_state = node.state.clone();
            // If the macro no longer applies (shouldn't happen — the mask was
            // computed for this state), treat as a terminal-ish dead end.
            if apply_macro(&mut child_state, node.actor, edge.macro_id).is_err() {
                let v = leaf_values(&node.state, policy, value, cfg, rng);
                edge.n += 1;
                edge.w += v.get(&node.actor).copied().unwrap_or(0.0);
                return v;
            }
            let child_actor = next_actor(&child_state);
            edge.child = Some(Box::new(Node {
                state: child_state,
                actor: child_actor.unwrap_or(node.actor),
                edges: Vec::new(),
                expanded: false,
            }));
        }
        simulate(edge.child.as_mut().unwrap(), policy, value, cfg, rng)
    };

    let edge = &mut node.edges[best];
    edge.n += 1;
    edge.w += values.get(&node.actor).copied().unwrap_or(0.0);
    values
}

fn puct(edge: &Edge, sqrt_total: f64, parent_q: f64, cfg: &SearchConfig) -> f64 {
    let q = if edge.n == 0 {
        parent_q - cfg.fpu_reduction
    } else {
        edge.w / edge.n as f64
    };
    let u = cfg.cpuct * edge.prior * sqrt_total / (1.0 + edge.n as f64);
    q + u
}

/// Populate a node's edges with the policy's softmax priors over legal macros.
fn expand(node: &mut Node, policy: &MlpPolicy) {
    let legal = legal_macros(&node.state, node.actor);
    let obs = build_observation(&node.state, node.actor);
    let logits = policy.logits(&obs);
    let legal_ids: Vec<u16> = (0..legal.len() as u16)
        .filter(|&i| legal[i as usize])
        .collect();

    let max_logit = legal_ids
        .iter()
        .map(|&i| logits[i as usize])
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = legal_ids
        .iter()
        .map(|&i| ((logits[i as usize] - max_logit) as f64).exp())
        .collect();
    let sum: f64 = exps.iter().sum::<f64>().max(1e-12);

    node.edges = legal_ids
        .iter()
        .zip(exps)
        .map(|(&macro_id, e)| Edge {
            macro_id,
            prior: e / sum,
            n: 0,
            w: 0.0,
            child: None,
        })
        .collect();
    node.expanded = true;
}

/// Value estimate for a (non-terminal) leaf as a per-seat map.
///
/// With a value net: one forward pass per seat, each from that seat's own
/// observation perspective (`build_observation` places the viewer first, so the
/// net — trained single-perspective — evaluates each seat consistently). This is
/// microseconds vs a full rollout, and is what makes search fast enough to use.
/// Without a value net: fall back to a policy rollout to terminal.
fn leaf_values(
    state: &GameState,
    policy: &MlpPolicy,
    value: Option<&ValueNet>,
    cfg: &SearchConfig,
    rng: &mut SmallRng,
) -> Values {
    match value {
        Some(vnet) => state
            .players
            .iter()
            .map(|p| {
                let obs = build_observation(state, p.id);
                (p.id, vnet.value(&obs) as f64)
            })
            .collect(),
        None => rollout(state, policy, cfg, rng),
    }
}

/// Play the policy greedily for every seat from `state` to terminal, then score
/// the finish order into per-seat rank values.
///
/// Hot path: instead of computing the full (26-clone) legal-macro mask each step,
/// try macros in descending policy-logit order and apply the first that succeeds
/// directly on the rollout state. The policy's top choice is legal the vast
/// majority of the time, so this is ~1 attempt/step and does no cloning — the
/// difference between search being usable and not (the mask-per-step version was
/// ~1000× slower). A failed `apply_macro` is atomic (it errors before mutating),
/// so trying in order is safe.
fn rollout(
    state: &GameState,
    policy: &MlpPolicy,
    cfg: &SearchConfig,
    _rng: &mut SmallRng,
) -> Values {
    let mut s = state.clone();
    let n = crate::macro_actions::N_MACROS;
    for _ in 0..cfg.rollout_cap {
        resolve_auto_phases(&mut s);
        if matches!(s.phase, Phase::GameOver { .. }) {
            break;
        }
        let Some(actor) = macro_current_actor(&s) else {
            break;
        };
        let obs = build_observation(&s, actor);
        let logits = policy.logits(&obs);
        // Macro ids sorted by logit, highest first.
        let mut order: Vec<u16> = (0..n as u16).collect();
        order.sort_by(|&a, &b| {
            logits[b as usize]
                .partial_cmp(&logits[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut applied = false;
        for mid in order {
            if apply_macro(&mut s, actor, mid).is_ok() {
                applied = true;
                break;
            }
        }
        if !applied {
            break;
        }
    }
    terminal_values(&s)
}

/// Per-seat rank value from the current standings (meaningful at terminal;
/// used as the rollout's return).
fn terminal_values(state: &GameState) -> Values {
    let ranks = powergrid_core::rules::finish_ranks(state);
    let n = ranks.len();
    ranks
        .into_iter()
        .map(|(id, pos)| (id, rank_value(pos, n)))
        .collect()
}

fn rank_value(pos: usize, n: usize) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    1.0 - 2.0 * (pos as f64 - 1.0) / (n as f64 - 1.0)
}

/// The actor who owns the next macro decision, resolving trailing auto-phases
/// on a scratch clone is unnecessary here — `macro_current_actor` already
/// returns `None` for auto-phases, and callers apply macros that auto-resolve.
fn next_actor(state: &GameState) -> Option<PlayerId> {
    macro_current_actor(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{default_policy, default_value_net};
    use powergrid_core::actions::Action;
    use powergrid_core::map::default_map;
    use powergrid_core::rules::apply_action;
    use powergrid_core::types::PlayerColor;

    fn started_game(seed: u64) -> (GameState, Vec<PlayerId>) {
        let mut state = GameState::new_with_seed(default_map(), 4, seed);
        let ids: Vec<PlayerId> = (0..4)
            .map(|i| PlayerId::from_u128(((seed as u128) << 8) | (i + 1) as u128))
            .collect();
        let colors = [
            PlayerColor::Red,
            PlayerColor::Blue,
            PlayerColor::Green,
            PlayerColor::Yellow,
        ];
        for (i, id) in ids.iter().enumerate() {
            apply_action(
                &mut state,
                *id,
                Action::JoinGame {
                    name: format!("P{i}"),
                    color: colors[i],
                },
            )
            .unwrap();
        }
        apply_action(&mut state, ids[0], Action::StartGame).unwrap();
        (state, ids)
    }

    #[test]
    fn search_returns_a_legal_macro_and_is_deterministic() {
        let Some(policy) = default_policy() else {
            eprintln!("no embedded policy; skipping search test");
            return;
        };
        let (state, _) = started_game(42);
        let actor = macro_current_actor(&state).expect("a macro decision at game start");
        let cfg = SearchConfig {
            num_sims: 12,
            determinizations: 1,
            ..Default::default()
        };
        let value = default_value_net();
        let a = choose_macro(&state, actor, &policy, value.as_deref(), &cfg, 7);
        let b = choose_macro(&state, actor, &policy, value.as_deref(), &cfg, 7);
        assert_eq!(a, b, "same seed must give the same macro");
        let m = a.expect("a legal macro exists");
        let legal = legal_macros(&state, actor);
        assert!(legal[m as usize], "chosen macro must be legal");
    }

    #[test]
    #[ignore = "timing benchmark; run with --ignored --nocapture"]
    fn bench_search_timing() {
        let Some(policy) = default_policy() else {
            return;
        };
        let value = default_value_net();
        let (state, _) = started_game(42);
        let actor = macro_current_actor(&state).unwrap();
        for &sims in &[50usize, 100, 200] {
            let cfg = SearchConfig {
                num_sims: sims,
                determinizations: 1,
                ..Default::default()
            };
            let t = std::time::Instant::now();
            let _ = choose_macro(&state, actor, &policy, value.as_deref(), &cfg, 1);
            println!(
                "choose_macro {sims} sims (value-net leaf): {:?}",
                t.elapsed()
            );
        }
    }

    #[test]
    #[ignore = "slow (rollout MCTS ~1-2s/move until a value net replaces rollouts); \
                run with --ignored"]
    fn search_drives_a_full_game_to_completion() {
        let Some(policy) = default_policy() else {
            return;
        };
        let value = default_value_net();
        let (mut state, _) = started_game(7);
        let cfg = SearchConfig {
            num_sims: 16,
            determinizations: 1,
            ..Default::default()
        };
        for step in 0..8000 {
            resolve_auto_phases(&mut state);
            if matches!(state.phase, Phase::GameOver { .. }) {
                break;
            }
            let actor = macro_current_actor(&state).expect("a decision");
            let m = choose_macro(&state, actor, &policy, value.as_deref(), &cfg, step as u64)
                .expect("a macro");
            apply_macro(&mut state, actor, m).expect("search macro applies");
        }
        assert!(
            matches!(state.phase, Phase::GameOver { .. }),
            "game completed"
        );
    }
}
