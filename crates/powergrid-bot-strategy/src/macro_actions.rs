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
//! - Fuel splits (`PowerCitiesFuel`) and resource discards (`DiscardResource`)
//!   are minor tactical steps with no strategic content — they are auto-resolved
//!   with the heuristic, so they never consume a policy decision.

use std::sync::OnceLock;

use powergrid_core::{
    actions::{Action, ActionError},
    state::GameState,
    types::{connection_cost, Phase, PlantKind, Player, PlayerColor, PlayerId, Resource},
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
/// `BUILD_DEFAULT` (greedy-cheapest-to-the-cap *is* what the heuristic
/// computes) and `RACE`'s all-or-nothing reach-the-trigger-this-turn condition
/// is essentially never affordable. `BLOCK` was the only *which*-lever and is
/// dropped with it: CMA-ES independently zeroed the heuristic's `block_weight`,
/// i.e. chasing contested cities did not pay.
pub const BUILD_COUNT_BASE: u16 = 8;
pub const N_BUILD_COUNT: u16 = 7;
/// Build exactly what the champion heuristic would. Kept as the last build id so
/// dedup prefers the explicit count (stable id semantics for the net), and as the
/// safety valve for plans no `BUILD_COUNT_*` can express (more than 6 cities, or
/// a future profile whose ordering isn't plain cheapest-first).
pub const BUILD_DEFAULT: u16 = 15;

/// Buy `n` units of fuel, `n = id - BUY_COUNT_BASE` in `0..=6`. `n = 0` is an
/// empty batch (the same primitive the heuristic emits when it buys nothing).
///
/// The quantity ladder, mirroring [`BUILD_COUNT_BASE`]. *Which* units to buy is
/// near-forced — the rack's plants dictate the fuel types, and within that the
/// heuristic's canonical fill order applies — so **how many** is the decision.
/// It is also where the round's cash split is actually made: `BuyResources` runs
/// *before* `BuildCities`, so every Elektro spent on fuel is one unavailable for
/// cities, and the old menu had no way to express that tradeoff.
///
/// Replaces `BUY_NOTHING`/`BUY_STOCKPILE2`/`BUY_STOCKPILE3`/`BUY_DENIAL`, which
/// measured near-dead: the teacher chose `BUY_DEFAULT` in **100%** of 748 sampled
/// decisions. `STOCKPILE2/3` could rarely differ from the default at all (2.1% /
/// 0.1%) because plant storage caps at 2× firing cost and the heuristic's
/// stockpile pass sits behind a 140-Elektro reserve; `DENIAL` targeted the
/// priciest fuel — uranium 59.4% of the time — which the actor could not store at
/// all in 61.8% of decisions, so it collapsed to a no-op and was deduped away.
pub const BUY_COUNT_BASE: u16 = 16;
pub const N_BUY_COUNT: u16 = 7;
/// Buy exactly what the champion heuristic would. Last buy id for the same
/// reason as [`BUILD_DEFAULT`]: dedup then prefers the explicit count.
pub const BUY_DEFAULT: u16 = 23;

/// Discard one owned plant (when a 4th was just bought): slot 0..=2 by number.
pub const DISCARD_PLANT_BASE: u16 = 24;
pub const N_DISCARD_PLANT: u16 = 3;

pub const POWER_OPTIMAL: u16 = 27;
pub const POWER_NOTHING: u16 = 28;

pub const N_MACROS: usize = 29;

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

/// True for phases that require a *macro* decision from the policy. Fuel/discard
/// -resource phases are auto-resolved (see [`resolve_auto_phases`]) and are not
/// decision points.
fn is_macro_phase(phase: &Phase) -> bool {
    matches!(
        phase,
        Phase::Auction { .. }
            | Phase::BuyResources { .. }
            | Phase::BuildCities { .. }
            | Phase::Bureaucracy { .. }
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

/// Auto-resolve every `PowerCitiesFuel` / `DiscardResource` sub-phase with the
/// heuristic (these carry no strategic choice). Safe to call anytime; a no-op
/// unless the game is currently in one of those phases. Bounded iteration guards
/// against a pathological loop.
pub fn resolve_auto_phases(state: &mut GameState) {
    for _ in 0..64 {
        let actor = match &state.phase {
            Phase::PowerCitiesFuel { player, .. } | Phase::DiscardResource { player, .. } => {
                *player
            }
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
        Phase::Bureaucracy { .. } => expand_power(state, actor, macro_id),
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
    if macro_id == BUILD_DEFAULT {
        return Some(vec![heuristic_action(state, actor)?]);
    }
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
    if macro_id == BUY_DEFAULT {
        return Some(vec![heuristic_action(state, actor)?]);
    }
    if (BUY_COUNT_BASE..BUY_COUNT_BASE + N_BUY_COUNT).contains(&macro_id) {
        let n = (macro_id - BUY_COUNT_BASE) as usize;
        // Always a batch, never `DoneBuying`: the engine treats an empty batch as
        // "skip buying", and it is the exact primitive the heuristic emits when it
        // buys nothing — so `BUY_0` dedups cleanly against that instead of sitting
        // beside it as a second, bit-different way to say the same thing.
        return Some(vec![Action::BuyResourceBatch {
            purchases: buy_units(state, actor, n),
        }]);
    }
    None
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

fn expand_power(state: &GameState, actor: PlayerId, macro_id: u16) -> Option<Vec<Action>> {
    match macro_id {
        POWER_OPTIMAL => Some(vec![heuristic_action(state, actor)?]),
        POWER_NOTHING => Some(vec![Action::PowerCities {
            plant_numbers: vec![],
        }]),
        _ => None,
    }
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
/// The heuristic's own conservatism still exists — inside [`BUILD_DEFAULT`],
/// which delegates to `strategy::decide_build_cities` (full money up to powering
/// headroom, reserve-gated overbuild past it) unchanged.
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

/// True if the rack can actually burn `r` — hybrids burn both gas and oil.
fn burnable(player: &Player, r: Resource) -> bool {
    player.plants.iter().any(|p| {
        matches!(
            (p.kind, r),
            (PlantKind::Coal, Resource::Coal)
                | (PlantKind::Oil, Resource::Oil)
                | (PlantKind::Gas, Resource::Gas)
                | (PlantKind::Uranium, Resource::Uranium)
                | (PlantKind::GasOrOil, Resource::Gas | Resource::Oil)
        )
    })
}

/// The `n` fuel units to buy, walking the heuristic's own ordering.
///
/// Two segments, in order:
/// 1. **Essential** — the rack's needs for the coming firing, via
///    `strategy::plan_essential_buys` with a unit cap. Shared with the heuristic
///    rather than re-derived, so the ladder's low rungs cannot drift from the
///    teacher's cumulative hybrid-pool bookkeeping.
/// 2. **Stockpile** — if `n` exceeds the essential need, keep buying the cheapest
///    single unit the rack can *burn* and store, one at a time so the rising
///    price track is respected.
///
/// Cash is the only limit, exactly as in [`greedy_pick`]: no reserve is withheld,
/// because the count is the policy's decision. (The heuristic's own stockpile pass
/// is gated behind `city_reserve + safety_buffer`, 140 Elektro for the champion
/// profile — which is why `BUY_STOCKPILE2/3` could almost never differ from the
/// default and had to go.) Fuel the rack cannot burn is never bought: it can never
/// be fired, so it is strictly wasted money, and hoarding to deny opponents is
/// structurally weak here because storage caps at 2x firing cost.
fn buy_units(state: &GameState, actor: PlayerId, n: usize) -> Vec<(Resource, u8)> {
    let Some(player) = state.player(actor) else {
        return Vec::new();
    };
    if n == 0 {
        return Vec::new();
    }
    let n = n.min(u8::MAX as usize) as u8;

    let mut purchases: Vec<(Resource, u8)> = Vec::new();
    let mut sim_market = state.resources.clone();
    let mut sim_player = player.clone();
    let mut budget = player.money;

    strategy::plan_essential_buys(
        &mut sim_market,
        &mut sim_player,
        &mut budget,
        &mut purchases,
        Some(n),
    );

    let bought: u8 = purchases.iter().map(|(_, c)| *c).sum();
    let mut remaining = n.saturating_sub(bought);

    while remaining > 0 {
        // Cheapest next unit among burnable, storable, affordable fuels. Fixed
        // resource order + strict `<` makes ties resolve deterministically.
        let mut best: Option<(Resource, u32)> = None;
        for r in [
            Resource::Coal,
            Resource::Oil,
            Resource::Gas,
            Resource::Uranium,
        ] {
            if !burnable(&sim_player, r)
                || sim_market.available(r) == 0
                || !sim_player.can_add_resource(r, 1)
            {
                continue;
            }
            let Some(price) = sim_market.price(r, 1) else {
                continue;
            };
            if price > budget {
                continue;
            }
            if best.is_none_or(|(_, p)| price < p) {
                best = Some((r, price));
            }
        }
        let Some((r, price)) = best else { break };
        sim_market.take(r, 1);
        sim_player.resources.add(r, 1);
        budget -= price;
        purchases.push((r, 1));
        remaining -= 1;
    }

    purchases
}

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
        // Build: the label must be the id that SURVIVES dedup, or it would name an
        // illegal action. `BUILD_COUNT_*` sit below `BUILD_DEFAULT`, so whenever a
        // count reproduces the heuristic's plan exactly it shadows `BUILD_DEFAULT`
        // and becomes the label; `BUILD_DEFAULT` remains the answer only for plans
        // no count can express (>6 cities). With the champion profile the two
        // agree almost always — same candidate ordering, same greedy walk — so the
        // teacher usually speaks in counts, which is what we want the net to learn.
        Phase::BuildCities { .. } => {
            let expected = vec![action.clone()];
            (BUILD_COUNT_BASE..BUILD_COUNT_BASE + N_BUILD_COUNT)
                .find(|&id| expand_macro(state, actor, id).as_ref() == Some(&expected))
                .unwrap_or(BUILD_DEFAULT)
        }
        // Buy: same dedup-aware scan as the build ladder. The count rungs walk the
        // heuristic's own fill order (`plan_essential_buys`), so whenever the
        // teacher's batch is `n` units a rung reproduces it exactly and shadows
        // `BUY_DEFAULT`; `BUY_DEFAULT` remains the label only when the heuristic
        // buys more than 6 units or reaches its stockpile pass.
        Phase::BuyResources { .. } => {
            let expected = vec![action.clone()];
            (BUY_COUNT_BASE..BUY_COUNT_BASE + N_BUY_COUNT)
                .find(|&id| expand_macro(state, actor, id).as_ref() == Some(&expected))
                .unwrap_or(BUY_DEFAULT)
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
        // POWER_OPTIMAL expands to the heuristic's exact PowerCities action (empty
        // or not) and, having the lower id, survives dedup against POWER_NOTHING.
        Phase::Bureaucracy { .. } => match &action {
            Action::PowerCities { .. } => POWER_OPTIMAL,
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
            // Record the auto-resolved trailing fuel/discard actions in order.
            while let Phase::PowerCitiesFuel { player, .. }
            | Phase::DiscardResource { player, .. } = &state.phase
            {
                let auto_actor = *player;
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
