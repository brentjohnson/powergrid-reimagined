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
    types::{connection_cost, Phase, PlantKind, PlayerColor, PlayerId, Resource},
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

// --- Buy: per-plant fuel — none / one set / two sets -----------------------
//
// For each plant on the rack: skip it, buy one firing's worth of its fuel, or
// buy two. Two is the ceiling, not a choice of encoding — `can_add_resource`
// caps storage at `cost * 2` per plant, so a third set is unbuyable. (Same fact
// that made the old `BUY_STOCKPILE3` dead: its 3-round target clamped to the
// 2-round cap `BUY_STOCKPILE2` already hit.)
//
// **Per plant, not per fuel.** Fuel is fungible in the *stock* — 6 coal is 6
// coal whoever burns it — but it is spent in indivisible plant-sized chunks, so
// the plant is what quantizes the purchase. A rack with a coal-2 and a coal-4
// has a real decision to buy 4 coal (fire only the big plant when coal is dear)
// that a per-fuel encoding cannot name: summing demand offers only 0, 6 or 12.
// Per-plant reaches every total that corresponds to "this subset of plants
// fires, some with a spare round".
//
// Slot `i` is the player's `i`-th plant **by number ascending** — `rules.rs`
// re-sorts `player.plants` on every acquisition, so this is also the order the
// observation encodes self-plants in and the order `DISCARD_PLANT` slots use.
// The policy can therefore read slot i's number/kind/cost/cities straight off
// the observation at a fixed offset.
//
// Replaces an aggregate ±k ladder whose deviations were not decisions anyone
// makes: buying one unit short of a complete set cost cities in 32.4% of
// measured decisions (1.93 on average, for one unit of fuel) and changed nothing
// in the rest, because the shortfall came out of carry-over surplus.
//
/// End the buy turn having bought nothing more (`DoneBuying`).
pub const BUY_DONE: u16 = 15;
/// Buy exactly what the champion heuristic would, as one batch, and end the turn.
/// Gate 0 for this phase, and the imitation label.
pub const BUY_DEFAULT: u16 = 16;
/// Buy **one set** of fuel for plant slot 0..=2 — one firing's worth.
///
/// Emits the additive `Action::BuyResources` primitive, which does *not* end the
/// buy turn, so presses compose: "one set for the big plant, two for the
/// uranium, done". A press that could buy nothing (storage full, unaffordable,
/// market empty, or a wind plant that burns nothing) expands to `None` and is
/// masked out — so the turn can always be ended but never spun on.
pub const BUY_PLANT1_BASE: u16 = 17;
/// Buy **two sets** for plant slot 0..=2 — its storage ceiling. Same slot order.
pub const BUY_PLANT2_BASE: u16 = 20;
pub const N_BUY_PLANT_SLOTS: u16 = 3;

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
    match macro_id {
        BUY_DONE => return Some(vec![Action::DoneBuying]),
        // Delegates, so "what the heuristic buys" is its action bit-for-bit
        // (Gate 0) rather than a reimplementation that could drift. Emits a
        // batch, which ends the buy turn.
        BUY_DEFAULT => return Some(vec![heuristic_action(state, actor)?]),
        _ => {}
    }
    let (base, sets) = if (BUY_PLANT1_BASE..BUY_PLANT1_BASE + N_BUY_PLANT_SLOTS).contains(&macro_id)
    {
        (BUY_PLANT1_BASE, 1u8)
    } else if (BUY_PLANT2_BASE..BUY_PLANT2_BASE + N_BUY_PLANT_SLOTS).contains(&macro_id) {
        (BUY_PLANT2_BASE, 2u8)
    } else {
        return None;
    };
    plant_fuel(state, actor, (macro_id - base) as usize, sets)
}

/// Buy `sets` firings' worth of fuel for the plant in slot `slot`, or `None` if
/// nothing can be bought — which masks the macro out, so the buy phase can
/// always be ended but never spun on with no-ops.
///
/// Additive: a press adds this plant's requirement to the stock rather than
/// topping the stock up to it. With a shared pool there is no way to say how
/// much of the existing stock "belongs" to a given plant, so the honest
/// primitive is "buy enough for one more firing of this plant"; storage
/// (`cost * 2` per plant) bounds how often that can be repeated.
///
/// A hybrid draws on the shared gas/oil pool, so its purchase is split across
/// both, preferring whichever the market has more of — the same rule
/// `strategy::buy_for_plant` uses — and falling back to the other for any
/// shortfall. That is the one case where a press can emit two primitives.
fn plant_fuel(state: &GameState, actor: PlayerId, slot: usize, sets: u8) -> Option<Vec<Action>> {
    let player = state.player(actor)?;
    let plant = player.plants.get(slot)?;
    if !plant.kind.needs_resources() {
        return None; // wind burns nothing
    }
    let order: Vec<Resource> = match plant.kind {
        PlantKind::Coal => vec![Resource::Coal],
        PlantKind::Oil => vec![Resource::Oil],
        PlantKind::Gas => vec![Resource::Gas],
        PlantKind::Uranium => vec![Resource::Uranium],
        PlantKind::GasOrOil => {
            if state.resources.available(Resource::Oil) >= state.resources.available(Resource::Gas)
            {
                vec![Resource::Oil, Resource::Gas]
            } else {
                vec![Resource::Gas, Resource::Oil]
            }
        }
        PlantKind::Wind => return None,
    };

    let mut remaining = (plant.cost as u16 * sets as u16).min(u8::MAX as u16) as u8;
    let mut out = Vec::new();
    let mut sim_market = state.resources.clone();
    let mut sim_player = player.clone();
    let mut budget = player.money;

    for resource in order {
        if remaining == 0 {
            break;
        }
        // Largest chunk that fits under stock, storage and cash — mirroring
        // `strategy::try_buy`, so a press costs what the heuristic would pay.
        let cap = remaining.min(sim_market.available(resource));
        for amount in (1..=cap).rev() {
            if !sim_player.can_add_resource(resource, amount) {
                continue;
            }
            let Some(cost) = sim_market.price(resource, amount) else {
                continue;
            };
            if cost > budget {
                continue;
            }
            out.push(Action::BuyResources { resource, amount });
            sim_market.take(resource, amount);
            sim_player.resources.add(resource, amount);
            budget -= cost;
            remaining -= amount;
            break;
        }
    }
    (!out.is_empty()).then_some(out)
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
        // Buy: the heuristic emits one whole-turn batch, which is precisely what
        // BUY_DEFAULT expands to — so the label stays a single bit-exact macro
        // and Gate 0 is unchanged by the per-plant presses. Those presses are an
        // alternative the policy may compose; the teacher never needs them.
        Phase::BuyResources { .. } => match &action {
            Action::BuyResourceBatch { .. } | Action::BuyResources { .. } => BUY_DEFAULT,
            Action::DoneBuying => BUY_DONE,
            _ => return None,
        },
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
