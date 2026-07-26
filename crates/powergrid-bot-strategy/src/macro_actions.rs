//! Macro-action layer for the RL policy.
//!
//! Instead of the ~600 primitive micro-decisions per game that capped every
//! prior learning attempt (compounding error over `BuildCity`/`BuyResources`
//! unit sequences — see RL-TRAINING-JOURNAL.md), the policy chooses **one
//! complete phase-plan per turn** from a small fixed menu (`N_MACROS`). Each
//! macro expands to a short sequence of primitive [`Action`]s that the engine
//! already supports as whole-turn batches (`BuildCities`, `BuyResourceBatch`).
//!
//! **Design invariants that make this safe:**
//! - Only the *learner* plays macros; heuristic opponents keep the normal
//!   `strategy::decide_with_bot` path. So macro expansion always uses one
//!   canonical profile — the shipped champion `hard`, noise silenced.
//! - The `*_DEFAULT` macros delegate to `decide_with_bot` unchanged, so
//!   "play the heuristic" is always representable *bit-exactly* (Gate 0). The
//!   other menu items are new, isolated alternative plans; a bug in one only
//!   makes that option worse, it can't corrupt the heuristic path.
//! - Powering (`Bureaucracy`), fuel splits (`PowerCitiesFuel`) and resource
//!   discards (`DiscardResource`) carry no strategic content the menu could
//!   express — they are auto-resolved with the heuristic, so they never consume
//!   a policy decision.

use std::sync::OnceLock;

use powergrid_core::{
    actions::{Action, ActionError},
    state::GameState,
    types::{connection_cost, Phase, PlayerColor, PlayerId, Resource},
};

use crate::{embedded_registry, strategy, Bot, BotProfile};

// ---------------------------------------------------------------------------
// Macro id layout (flat, masked per phase)
// ---------------------------------------------------------------------------

/// Auction (no standing bid): nominate market `actual` slot 0..=5.
pub const NOMINATE_BASE: u16 = 0;
pub const N_NOMINATE: u16 = 6;
/// Drop out of / decline the auction (used in both auction sub-phases).
pub const AUCTION_PASS: u16 = 6;
/// Auction (standing bid): raise by +1 (English-auction convention).
pub const AUCTION_RAISE: u16 = 7;

/// Build the `n` cheapest reachable cities, `n = id - BUILD_COUNT_BASE` in
/// `0..=6`. `n = 0` is "build nothing" (`DoneBuilding`).
///
/// **How many** is the whole build decision, so it is the whole build menu. The
/// count is what trades income against turn order (more cities = earlier in
/// `player_order` = worse, since buying and building both run in reverse), and
/// what expresses the end-game push; *which* cities is a near-forced greedy
/// cheapest-first walk once the count is fixed. The previous menu
/// (`CHEAPEST_1/2/3` + `MAX` + `BLOCK` + `RACE`) spanned the same axis with
/// gaps, plus two plans that measured dead: `MAX` was 100% deduped against
/// the heuristic itself (greedy-cheapest-to-the-cap *is* what the heuristic
/// computes) and `RACE`'s all-or-nothing reach-the-trigger-this-turn condition
/// is essentially never affordable. `BLOCK` was the only *which*-lever and is
/// dropped with it: CMA-ES independently zeroed the heuristic's `block_weight`,
/// i.e. chasing contested cities did not pay.
pub const BUILD_COUNT_BASE: u16 = 8;
pub const N_BUILD_COUNT: u16 = 7;
// There is deliberately NO build-default macro. One existed as a safety valve for
// plans no count could express, and measured completely dead: over 1504 build
// decisions it was legal 0 times, was the teacher's label 0 times, and the
// heuristic never built more than 6 cities. A count always reproduces it, so it
// was an output unit that could never receive gradient. If a future expansion
// profile ever orders candidates by something other than cheapest-first,
// `teacher_macro_id` returns `None` for the build phase and Gate 0 fails loudly
// in test — which is the right place to find out, and better insurance than a
// permanently-masked id.

// --- Buy: which plants to fuel (a subset of the rack) ----------------------
//
// The decision, in the shape a player actually makes it: **choose which plants
// you intend to fire, then buy enough to top those up**, counting what you
// already hold. `BUY_SUBSET_BASE + mask`, where bit `i` selects plant slot `i`.
// With a 3-plant rack cap that is 8 ids covering every subset, including the
// empty one (buy nothing).
//
// Declaring the subset is what makes this well defined on a shared fuel pool.
// "Top plant A up" is ambiguous when plant B also burns coal — there is no fact
// about which of your 6 coal "belongs" to A. But "these plants will fire" fixes
// the pool requirement as the sum over the declared set, and the purchase is the
// deficit against current stock. That is exactly `plan_essential_buys` with the
// walk restricted to the selected plants.
//
// **Top-up, not additive**, which is why no heuristic escape hatch is needed
// here any more: the full-rack mask reproduces the champion's essential buy
// bit-for-bit (Gate 0), *including* its carry-over handling. The additive
// per-plant *presses* this replaces could not — being additive they bought a
// full set regardless of stock, matching the heuristic's total in only 71.6% of
// decisions, which forced a separate `BUY_DEFAULT` id to keep the heuristic
// playable at all and made buy the one multi-decision phase.
//
// **No stockpiling level.** Buying beyond one firing is deliberately not
// representable: `powergrid-evolve` had `buy.stockpile_rounds` in its CMA-ES
// genome over [1.0, 5.0] and the champion converged to 1.0, the floor — 200
// generations of paired evaluation say pre-buying does not pay. If that is ever
// revisited, the natural extension is a second mask (fire-set vs stock-set)
// rather than a level on this one.
//
/// Buy enough to top up the plants named by `id - BUY_SUBSET_BASE`, read as a
/// bitmask over plant slots (slot `i` = the `i`-th plant by number, matching the
/// observation and `DISCARD_PLANT`). Mask `0` buys nothing.
pub const BUY_SUBSET_BASE: u16 = 15;
pub const N_BUY_SUBSETS: u16 = 8;

/// Discard one owned plant (when a 4th was just bought): slot 0..=2 by number.
pub const DISCARD_PLANT_BASE: u16 = 23;
pub const N_DISCARD_PLANT: u16 = 3;

pub const N_MACROS: usize = 26;

// ---------------------------------------------------------------------------
// Canonical expansion profile
// ---------------------------------------------------------------------------

/// The shipped champion `hard`, with noise silenced (argmax, no bid jitter) so
/// macro expansion and teacher labelling are deterministic. Cached once.
fn expansion_profile() -> &'static BotProfile {
    static PROFILE: OnceLock<BotProfile> = OnceLock::new();
    PROFILE.get_or_init(|| {
        let mut p = embedded_registry().hard;
        p.temperature = 0.0;
        p.jitter = 0.0;
        p.max_jitter = 0;
        p
    })
}

fn expansion_bot(actor: PlayerId) -> Bot {
    let profile = expansion_profile().clone();
    Bot::new(actor, "expand".into(), PlayerColor::Red, profile, 0)
}

// ---------------------------------------------------------------------------
// Decision points
// ---------------------------------------------------------------------------

/// True for phases that require a *macro* decision from the policy. Powering and
/// the fuel/discard-resource sub-phases are auto-resolved (see
/// [`resolve_auto_phases`]) and are not decision points.
fn is_macro_phase(phase: &Phase) -> bool {
    matches!(
        phase,
        Phase::Auction { .. }
            | Phase::BuyResources { .. }
            | Phase::BuildCities { .. }
            | Phase::DiscardPlant { .. }
    )
}

/// The actor who must make a macro decision now, or `None` if the game is over,
/// in a pre-game phase, or resting in an auto-phase (call [`resolve_auto_phases`]
/// first in that case).
pub fn macro_current_actor(state: &GameState) -> Option<PlayerId> {
    if !is_macro_phase(&state.phase) {
        return None;
    }
    crate::encoding::current_actor_id(state)
}

/// Auto-resolve every `Bureaucracy` / `PowerCitiesFuel` / `DiscardResource`
/// phase with the heuristic (no strategic choice). Safe to call anytime; a no-op
/// unless the game is currently in one of those phases. Bounded iteration guards
/// against a pathological loop.
pub fn resolve_auto_phases(state: &mut GameState) {
    for _ in 0..64 {
        let actor = match &state.phase {
            Phase::PowerCitiesFuel { player, .. } | Phase::DiscardResource { player, .. } => {
                *player
            }
            // Powering carries no strategic choice the menu could express: the
            // teacher fires the optimal subset in 100% of decisions, and the only
            // alternative the macro layer ever offered (power nothing) was legal
            // everywhere and correct nowhere — a pure trap that also cost ~9 of a
            // seat's ~52 decisions per game. Resolved with the heuristic instead.
            Phase::Bureaucracy { remaining } => match remaining.first() {
                Some(id) => *id,
                None => return,
            },
            _ => return,
        };
        let mut bot = expansion_bot(actor);
        let Some(action) = strategy::decide_with_bot(state, &mut bot) else {
            return;
        };
        if powergrid_core::rules::apply_action(state, actor, action).is_err() {
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// Expansion: macro id -> primitive action sequence
// ---------------------------------------------------------------------------

/// The primitive action sequence a macro expands to in the current state, or
/// `None` if the macro is not applicable in this phase / for this actor. Does
/// not include the auto-resolution of trailing fuel/discard phases (that happens
/// at apply time).
pub fn expand_macro(state: &GameState, actor: PlayerId, macro_id: u16) -> Option<Vec<Action>> {
    match &state.phase {
        Phase::Auction { active_bid, .. } => expand_auction(state, actor, macro_id, active_bid),
        Phase::BuildCities { .. } => expand_build(state, actor, macro_id),
        Phase::BuyResources { .. } => expand_buy(state, actor, macro_id),
        Phase::DiscardPlant { .. } => expand_discard(state, actor, macro_id),
        _ => None,
    }
}

fn heuristic_action(state: &GameState, actor: PlayerId) -> Option<Action> {
    let mut bot = expansion_bot(actor);
    strategy::decide_with_bot(state, &mut bot)
}

fn expand_auction(
    state: &GameState,
    _actor: PlayerId,
    macro_id: u16,
    active_bid: &Option<powergrid_core::types::ActiveBid>,
) -> Option<Vec<Action>> {
    match (active_bid, macro_id) {
        // Responding to a standing bid: raise +1 or drop out.
        (Some(bid), AUCTION_RAISE) => Some(vec![Action::PlaceBid {
            amount: bid.amount + 1,
        }]),
        (Some(_), AUCTION_PASS) => Some(vec![Action::PassAuction]),
        // Opening: nominate a market slot or pass.
        (None, AUCTION_PASS) => Some(vec![Action::PassAuction]),
        (None, id) if (NOMINATE_BASE..NOMINATE_BASE + N_NOMINATE).contains(&id) => {
            let slot = (id - NOMINATE_BASE) as usize;
            let plant = state.market.actual.get(slot)?;
            Some(vec![Action::SelectPlant {
                plant_number: plant.number,
            }])
        }
        _ => None,
    }
}

fn expand_build(state: &GameState, actor: PlayerId, macro_id: u16) -> Option<Vec<Action>> {
    if (BUILD_COUNT_BASE..BUILD_COUNT_BASE + N_BUILD_COUNT).contains(&macro_id) {
        let n = (macro_id - BUILD_COUNT_BASE) as usize;
        return build_from_ids(cheapest_cities(state, actor, n));
    }
    None
}

fn build_from_ids(ids: Vec<String>) -> Option<Vec<Action>> {
    if ids.is_empty() {
        Some(vec![Action::DoneBuilding])
    } else {
        Some(vec![Action::BuildCities { city_ids: ids }])
    }
}

fn expand_buy(state: &GameState, actor: PlayerId, macro_id: u16) -> Option<Vec<Action>> {
    if !(BUY_SUBSET_BASE..BUY_SUBSET_BASE + N_BUY_SUBSETS).contains(&macro_id) {
        return None;
    }
    let slots = (macro_id - BUY_SUBSET_BASE) as u8;
    let player = state.player(actor)?;

    let mut purchases: Vec<(Resource, u8)> = Vec::new();
    let mut sim_market = state.resources.clone();
    let mut sim_player = player.clone();
    let mut budget = player.money;
    strategy::plan_essential_buys(
        &mut sim_market,
        &mut sim_player,
        &mut budget,
        &mut purchases,
        slots,
    );

    // Always a batch, never `DoneBuying`: an empty batch is what the engine
    // treats as "skip buying" and is the exact primitive the heuristic emits when
    // it buys nothing, so the empty mask dedups cleanly against it.
    Some(vec![Action::BuyResourceBatch { purchases }])
}

fn expand_discard(state: &GameState, actor: PlayerId, macro_id: u16) -> Option<Vec<Action>> {
    if !(DISCARD_PLANT_BASE..DISCARD_PLANT_BASE + N_DISCARD_PLANT).contains(&macro_id) {
        return None;
    }
    let slot = (macro_id - DISCARD_PLANT_BASE) as usize;
    let player = state.player(actor)?;
    let mut plants: Vec<u8> = player.plants.iter().map(|p| p.number).collect();
    plants.sort_unstable();
    let number = *plants.get(slot)?;
    Some(vec![Action::DiscardPlant {
        plant_number: number,
    }])
}

// ---------------------------------------------------------------------------
// Alternative build plans (isolated — never affect the heuristic/default path)
// ---------------------------------------------------------------------------

/// Buildable cities for `actor` as `(city_id, total_cost)` where cost is the
/// Dijkstra route cost from the actor's network plus the slot fee. Sorted by
/// cost with a city-id tiebreak (deterministic). Mirrors the enumeration in
/// `strategy::decide_build_cities` without touching it.
fn buildable(state: &GameState, actor: PlayerId) -> Vec<(String, u32)> {
    let owned = state.player_cities(actor);
    let mut out: Vec<(String, u32)> = state
        .map
        .cities
        .values()
        .filter(|c| {
            state.active_regions.contains(&c.region)
                && !c.owners.contains(&actor)
                && c.owners.len() < state.step as usize
        })
        .filter_map(|c| {
            let route = state.map.connection_cost_to(&owned, &c.id)?;
            Some((c.id.clone(), route + connection_cost(c.owners.len())))
        })
        .collect();
    out.sort_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    out
}

/// Greedily take the cheapest `limit` cities from `sorted` that the actor can
/// pay for, re-simulating route cost as the network grows. Returns owned ids.
///
/// **The only constraint is cash.** No fuel/auction reserve is withheld: the
/// count *is* the policy's decision, so refusing a requested city to protect a
/// heuristic reserve would silently turn `BUILD_n` into `BUILD_m`, which is the
/// failure this menu exists to remove. Two facts make spending down defensible:
/// building is the last spend of the round (`BuyResources` runs before it and
/// Bureaucracy income lands right after, so cash held back cannot buy fuel this
/// round), and over-spending is a real strategic error the policy should be able
/// to *make* and be punished for, not one the action space hides. A policy that
/// wants a reserve picks a smaller `n`.
///
/// The heuristic's own conservatism lives in `strategy::decide_build_cities`
/// (full money up to powering headroom, reserve-gated overbuild past it), which
/// the count ladder deliberately does not inherit.
fn greedy_pick(
    state: &GameState,
    actor: PlayerId,
    sorted: &[(String, u32)],
    budget: u32,
    limit: usize,
) -> Vec<String> {
    let mut owned = state.player_cities(actor);
    let mut chosen = Vec::new();
    let mut cash = budget;
    for (id, _) in sorted {
        if chosen.len() >= limit {
            break;
        }
        // Recompute route cost against the growing network.
        let Some(route) = state.map.connection_cost_to(&owned, id) else {
            continue;
        };
        let slot = state
            .map
            .cities
            .get(id)
            .map(|c| connection_cost(c.owners.len()))
            .unwrap_or(0);
        let cost = route + slot;
        if cost > cash {
            continue;
        }
        cash -= cost;
        owned.push(id.clone());
        chosen.push(id.clone());
    }
    chosen
}

/// The `n` cheapest cities the actor can afford, capped by the end-game trigger
/// (cities past it are pure waste — the game ends the moment it is reached).
fn cheapest_cities(state: &GameState, actor: PlayerId, n: usize) -> Vec<String> {
    if n == 0 {
        return Vec::new();
    }
    let money = state.player(actor).map(|p| p.money).unwrap_or(0);
    let cap = state
        .end_game_cities
        .saturating_sub(state.player_city_count(actor) as u8) as usize;
    let sorted = buildable(state, actor);
    greedy_pick(state, actor, &sorted, money, n.min(cap))
}

// ---------------------------------------------------------------------------
// Buy quantity ladder
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Legal mask + apply
// ---------------------------------------------------------------------------

/// Per-macro legality mask (length [`N_MACROS`]). A macro is legal iff it applies
/// cleanly in the current state (validated by trial application on a clone) and
/// its primitive expansion is not a duplicate of a lower-id macro's expansion
/// (dedup keeps the lowest id — e.g. when `cheapest_3` can only afford 2 cities,
/// it collapses onto `cheapest_2`).
pub fn legal_macros(state: &GameState, actor: PlayerId) -> Vec<bool> {
    let mut legal = vec![false; N_MACROS];
    let mut seen: Vec<Vec<Action>> = Vec::new();
    for id in 0..N_MACROS as u16 {
        let Some(expansion) = expand_macro(state, actor, id) else {
            continue;
        };
        if seen.contains(&expansion) {
            continue; // duplicate of an earlier (lower-id) macro
        }
        // Validate legality by trial application on a clone.
        let mut probe = state.clone();
        if apply_expansion(&mut probe, actor, &expansion).is_ok() {
            legal[id as usize] = true;
            seen.push(expansion);
        }
    }
    legal
}

fn apply_expansion(
    state: &mut GameState,
    actor: PlayerId,
    expansion: &[Action],
) -> Result<(), ActionError> {
    for action in expansion {
        powergrid_core::rules::apply_action(state, actor, action.clone())?;
    }
    Ok(())
}

/// Apply a macro: expand, apply the primitive sequence, then auto-resolve any
/// trailing fuel/discard-resource phase for `actor`. Errors if the macro is not
/// legal in the current state.
pub fn apply_macro(
    state: &mut GameState,
    actor: PlayerId,
    macro_id: u16,
) -> Result<(), ActionError> {
    let expansion = expand_macro(state, actor, macro_id).ok_or(ActionError::WrongPhase)?;
    apply_expansion(state, actor, &expansion)?;
    resolve_auto_phases(state);
    Ok(())
}

// ---------------------------------------------------------------------------
// Teacher labelling (imitation target)
// ---------------------------------------------------------------------------

/// The macro id the champion heuristic would pick now — the imitation label for
/// behavior cloning / DAgger. Always maps to a `*_DEFAULT`/matching macro whose
/// expansion is bit-exactly the heuristic's action, so a policy that copies the
/// teacher reproduces the heuristic exactly (Gate 0).
pub fn teacher_macro_id(state: &GameState, actor: PlayerId) -> Option<u16> {
    let action = heuristic_action(state, actor)?;
    let id = match &state.phase {
        Phase::Auction { .. } => match &action {
            Action::PlaceBid { .. } => AUCTION_RAISE,
            Action::PassAuction => AUCTION_PASS,
            Action::SelectPlant { plant_number } => {
                let slot = state
                    .market
                    .actual
                    .iter()
                    .position(|p| p.number == *plant_number)?;
                NOMINATE_BASE + slot as u16
            }
            _ => return None,
        },
        // Build: the label is the count whose expansion equals the heuristic's
        // action. With the champion profile one always does — same candidate
        // ordering, same greedy walk — so there is no default to fall back on.
        // `None` here means a count could NOT reproduce the heuristic, which is
        // a real regression: Gate 0 asserts on it rather than papering over it.
        Phase::BuildCities { .. } => {
            let expected = vec![action.clone()];
            (BUILD_COUNT_BASE..BUILD_COUNT_BASE + N_BUILD_COUNT)
                .find(|&id| expand_macro(state, actor, id).as_ref() == Some(&expected))?
        }
        // Buy: the subset whose top-up equals the heuristic's batch. The
        // full-rack mask always does (same walk, same order), but a smaller mask
        // may produce the identical purchase when a plant needed nothing — and
        // that is the id that survives dedup, so scan in id order.
        Phase::BuyResources { .. } => {
            let expected = vec![action.clone()];
            (BUY_SUBSET_BASE..BUY_SUBSET_BASE + N_BUY_SUBSETS)
                .find(|&id| expand_macro(state, actor, id).as_ref() == Some(&expected))?
        }
        Phase::DiscardPlant { .. } => match &action {
            Action::DiscardPlant { plant_number } => {
                let player = state.player(actor)?;
                let mut plants: Vec<u8> = player.plants.iter().map(|p| p.number).collect();
                plants.sort_unstable();
                let slot = plants.iter().position(|n| n == plant_number)?;
                DISCARD_PLANT_BASE + slot as u16
            }
            _ => return None,
        },
        _ => return None,
    };
    Some(id)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use powergrid_core::map::default_map;
    use powergrid_core::rules::apply_action;

    fn start_game(seed: u64) -> (GameState, Vec<PlayerId>) {
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
            .expect("join");
        }
        apply_action(&mut state, ids[0], Action::StartGame).expect("start");
        (state, ids)
    }

    /// Reference game: every player driven by the silenced champion heuristic
    /// (the exact profile macro expansion uses). Records every applied action.
    fn play_heuristic(seed: u64) -> Vec<(PlayerId, Action)> {
        let (mut state, _ids) = start_game(seed);
        let mut log = Vec::new();
        for _ in 0..8000 {
            if matches!(state.phase, Phase::GameOver { .. }) {
                break;
            }
            let Some(actor) = crate::encoding::current_actor_id(&state) else {
                break;
            };
            let mut bot = expansion_bot(actor);
            let Some(action) = strategy::decide_with_bot(&state, &mut bot) else {
                break;
            };
            log.push((actor, action.clone()));
            apply_action(&mut state, actor, action).expect("heuristic move legal");
        }
        log
    }

    /// Same game driven through the macro layer: at each decision point pick the
    /// teacher macro, expand it, apply + record every primitive, then record the
    /// auto-resolved trailing fuel/discard actions.
    fn play_macro(seed: u64) -> Vec<(PlayerId, Action)> {
        let (mut state, _ids) = start_game(seed);
        let mut log = Vec::new();
        for _ in 0..8000 {
            if matches!(state.phase, Phase::GameOver { .. }) {
                break;
            }
            let Some(actor) = macro_current_actor(&state) else {
                // Should not happen: apply below resolves trailing auto-phases.
                resolve_auto_phases(&mut state);
                continue;
            };
            let macro_id = teacher_macro_id(&state, actor).expect("teacher has a macro");
            let expansion = expand_macro(&state, actor, macro_id).expect("teacher macro expands");
            for action in expansion {
                log.push((actor, action.clone()));
                apply_action(&mut state, actor, action).expect("macro move legal");
            }
            // Record the auto-resolved trailing powering/fuel/discard actions in
            // order. Mirrors `resolve_auto_phases`' phase set exactly, so Gate 0
            // also proves auto-resolved *powering* matches the heuristic's.
            loop {
                let auto_actor = match &state.phase {
                    Phase::PowerCitiesFuel { player, .. }
                    | Phase::DiscardResource { player, .. } => *player,
                    Phase::Bureaucracy { remaining } => match remaining.first() {
                        Some(id) => *id,
                        None => break,
                    },
                    _ => break,
                };
                let mut bot = expansion_bot(auto_actor);
                let Some(action) = strategy::decide_with_bot(&state, &mut bot) else {
                    break;
                };
                log.push((auto_actor, action.clone()));
                apply_action(&mut state, auto_actor, action).expect("auto move legal");
            }
        }
        log
    }

    #[test]
    fn gate0_teacher_macro_reproduces_heuristic_bit_exactly() {
        // The macro round-trip (teacher_macro_id -> expand -> apply, with
        // auto-resolved fuel/discard) must produce the *identical* action
        // sequence as the pure heuristic — proving the macro layer imposes zero
        // compounding-error tax. This is the load-bearing test for all of Phase 2.
        for seed in [1u64, 7, 42, 99, 256, 1000, 2024, 55555] {
            let heuristic = play_heuristic(seed);
            let macros = play_macro(seed);
            assert_eq!(
                heuristic.len(),
                macros.len(),
                "seed {seed}: action count differs ({} heuristic vs {} macro)",
                heuristic.len(),
                macros.len()
            );
            for (i, (h, m)) in heuristic.iter().zip(macros.iter()).enumerate() {
                assert_eq!(h, m, "seed {seed}: action {i} differs: {h:?} vs {m:?}");
            }
            // And the game actually completed.
            assert!(!heuristic.is_empty(), "seed {seed}: empty game");
        }
    }

    #[test]
    fn legal_macros_never_empty_in_macro_phases() {
        // At every macro decision point there must be at least one legal macro
        // (else the policy would be stuck).
        let (mut state, _ids) = start_game(42);
        let mut steps = 0;
        while !matches!(state.phase, Phase::GameOver { .. }) && steps < 8000 {
            if let Some(actor) = macro_current_actor(&state) {
                let legal = legal_macros(&state, actor);
                assert!(
                    legal.iter().any(|&b| b),
                    "no legal macro at phase {:?}",
                    state.phase
                );
                // The teacher's macro must itself be legal.
                let t = teacher_macro_id(&state, actor).expect("teacher macro");
                assert!(legal[t as usize], "teacher macro {t} not in legal set");
                apply_macro(&mut state, actor, t).expect("apply teacher macro");
            } else {
                resolve_auto_phases(&mut state);
            }
            steps += 1;
        }
        assert!(
            matches!(state.phase, Phase::GameOver { .. }),
            "game finished"
        );
    }
}
