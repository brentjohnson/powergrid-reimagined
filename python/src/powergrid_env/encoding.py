"""
Observation encoding for the PettingZoo env (Python reference mirror).

The ACTION side (mask / apply / decode) lives natively in Rust
(`crate::macro_actions`, exposed via `powergrid_py.action_mask` /
`apply_action_id` / `bot_decide_id`) and is the single source of truth — the
old Python action-encoding mirror (`mask_from_info` / `id_to_action_json` /
`action_json_to_id`) was removed in the Phase-2 macro rebuild.

`encode_observation` remains a Python mirror of Rust's `build_observation`, kept
because the observation layout is nontrivial (it reconstructs the default-map
Dijkstra graph, which is not in the wire-safe view). The native parity test
`test_native_bridge.test_observation_matches_python` guards it against drift.
"""

import heapq
import tomllib
from pathlib import Path

import numpy as np
from .constants import (
    OBS_SIZE, CITY_IDS, CITY_INDEX, REGION_NAMES,
    KIND_IDS, PHASE_IDS, RESOURCE_IDX, MAX_CITIES,
)

# ---------------------------------------------------------------------------
# Default-map graph, for the per-city connection-cost observation feature.
#
# The observation is compiled against the default USA map (the map graph is
# NOT in the wire-safe GameStateView), so we load it once from the same asset
# Rust embeds and mirror `Map::from_data` (bidirectional edges) +
# `Map::connection_costs_from` (multi-source Dijkstra). The native parity test
# `test_native_bridge.test_observation_matches_python` guards against drift.
# ---------------------------------------------------------------------------

_USA_TOML = Path(__file__).resolve().parents[3] / "assets" / "maps" / "usa.toml"
_ADJ: dict[str, list[tuple[str, int]]] | None = None


def _adjacency() -> dict[str, list[tuple[str, int]]]:
    global _ADJ
    if _ADJ is None:
        data = tomllib.loads(_USA_TOML.read_text())
        adj: dict[str, list[tuple[str, int]]] = {c["id"]: [] for c in data["cities"]}
        for conn in data.get("connections", []):
            adj.setdefault(conn["from"], []).append((conn["to"], conn["cost"]))
            adj.setdefault(conn["to"], []).append((conn["from"], conn["cost"]))
        _ADJ = adj
    return _ADJ


def _connection_costs_from(owned: list[str]) -> dict[str, int]:
    """Cheapest connection cost from `owned` to every city (multi-source
    Dijkstra), mirroring Rust `Map::connection_costs_from`. Empty owned set →
    every city 0 (the first city is free of routing); owned cities → 0."""
    adj = _adjacency()
    if not owned:
        return {city: 0 for city in adj}
    dist: dict[str, int] = {}
    heap: list[tuple[int, str]] = []
    for start in owned:
        dist[start] = 0
        heapq.heappush(heap, (0, start))
    while heap:
        cost, node = heapq.heappop(heap)
        if dist.get(node, 1 << 30) < cost:
            continue
        for neighbor, edge_cost in adj.get(node, []):
            nc = cost + edge_cost
            if nc < dist.get(neighbor, 1 << 30):
                dist[neighbor] = nc
                heapq.heappush(heap, (nc, neighbor))
    return dist


def _plant_demand(plant: dict) -> dict[str, float]:
    """Per-round fuel demand a plant places on each resource — mirrors Rust
    `per_round_demand`. Hybrids (`gas_or_oil`) split their cost across gas/oil."""
    kind = plant["kind"]
    cost = float(plant["cost"])
    if kind == "gas_or_oil":
        return {"gas": cost / 2.0, "oil": cost / 2.0}
    if kind in ("coal", "oil", "gas", "uranium"):
        return {kind: cost}
    return {}  # wind


def encode_observation(state: dict, actor_id: str) -> np.ndarray:
    """
    Encode a GameStateView dict into a flat float32 observation vector of length OBS_SIZE.
    All values are normalized to approximately [0, 1].

    Fields NOT encoded: event_log (display-only text, no decision relevance).
    City ownership is read from state["city_owners"] (the canonical source);
    Player.cities was removed from the wire format as a redundant duplicate.
    """
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    idx = 0

    players = state["players"]
    me = _find_player(state, actor_id)
    if me is None:
        return obs

    opponents = [p for p in players if p["id"] != actor_id]

    # Build per-player city list from city_owners (single source of truth).
    cities_by_player: dict[str, list[str]] = {}
    for city_id, owners in state.get("city_owners", {}).items():
        for owner_id in owners:
            cities_by_player.setdefault(owner_id, []).append(city_id)

    # 1. Self money (1)
    obs[idx] = me["money"] / 500.0
    idx += 1

    # 2. Self resources (4): coal, oil, gas, uranium
    r = me["resources"]
    # Denominators = market price-track capacities (coal 27, oil 20, gas 24, uranium 12).
    obs[idx:idx+4] = [r["coal"] / 27, r["oil"] / 20, r["gas"] / 24, r["uranium"] / 12]
    idx += 4

    # 3. Self plants (3 × 5 = 15): padded to 3 slots
    for i, plant in enumerate((me.get("plants") or [])[:3]):
        base = idx + i * 5
        obs[base]   = plant["number"] / 60
        obs[base+1] = KIND_IDS.get(plant["kind"], 0) / 6
        obs[base+2] = plant["cost"] / 5          # max resource cost ≈ 3
        obs[base+3] = plant["cities"] / 8        # max cities per plant = 7 in base game
        cap = plant["cost"] * 2 if plant["kind"] not in ("wind",) else 0
        obs[base+4] = cap / 10                   # max cap = 6 (cost 3 × 2)
    idx += 15

    # 4. Self cities (MAX_CITIES)
    for city_id in cities_by_player.get(actor_id, []):
        ci = CITY_INDEX.get(city_id)
        if ci is not None:
            obs[idx + ci] = 1.0
    idx += MAX_CITIES

    # 5. Opponents (5 × 4 = 20): plants, cities, cap, last_powered (money hidden)
    for i, opp in enumerate(opponents[:5]):
        base = idx + i * 4
        obs[base]   = len(opp.get("plants", [])) / 3
        obs[base+1] = len(cities_by_player.get(opp["id"], [])) / MAX_CITIES
        cap = sum(p["cost"] * 2 for p in opp.get("plants", []) if p["kind"] not in ("wind",))
        obs[base+2] = cap / 30
        obs[base+3] = opp.get("last_cities_powered", 0) / 21
    idx += 20

    # 6. Opponent cities (5 × MAX_CITIES)
    for i, opp in enumerate(opponents[:5]):
        for city_id in cities_by_player.get(opp["id"], []):
            ci = CITY_INDEX.get(city_id)
            if ci is not None:
                obs[idx + i * MAX_CITIES + ci] = 1.0
    idx += 5 * MAX_CITIES

    # 7. City slot count (MAX_CITIES)
    city_owners = state.get("city_owners", {})
    for city_id, ci in CITY_INDEX.items():
        obs[idx + ci] = len(city_owners.get(city_id, [])) / 3
    idx += MAX_CITIES

    # 8. Active regions (N_REGIONS)
    for i, region in enumerate(REGION_NAMES):
        if region in state.get("active_regions", []):
            obs[idx + i] = 1.0
    idx += len(REGION_NAMES)

    # 9+10. Plant market (8 cards): chain `actual` then `future`, take 8.
    # Cards 0-3 (24 = 4 × 6): number, kind, cost, cities, present, discount.
    # Cards 4-7 (20 = 4 × 5): number, kind, cost, cities, present (no discount).
    # In steps 1/2, `actual` has exactly 4 and `future` has exactly 4, so this
    # reproduces the old per-section encoding exactly. In step 3, `future` is
    # empty and `actual` holds all 6 plants, so the 5th/6th actual plants land
    # in cards 4/5 instead of being dropped.
    mkt = state["market"]
    discount_tok = mkt.get("discount_token")
    actual_base = idx
    future_base = idx + 24
    chained = list(mkt.get("actual", [])) + list(mkt.get("future", []))
    for i, plant in enumerate(chained[:8]):
        if i < 4:
            base = actual_base + i * 6
            obs[base]   = plant["number"] / 60
            obs[base+1] = KIND_IDS.get(plant["kind"], 0) / 6
            obs[base+2] = plant["cost"] / 5
            obs[base+3] = plant["cities"] / 8
            obs[base+4] = 1.0
            obs[base+5] = 1.0 if plant["number"] == discount_tok else 0.0
        else:
            base = future_base + (i - 4) * 5
            obs[base]   = plant["number"] / 60
            obs[base+1] = KIND_IDS.get(plant["kind"], 0) / 6
            obs[base+2] = plant["cost"] / 5
            obs[base+3] = plant["cities"] / 8
            obs[base+4] = 1.0
    idx += 24
    idx += 20

    # 11. Plant market meta (3)
    mkt = state["market"]
    obs[idx]   = 1.0 if mkt.get("step3_triggered") else 0.0
    obs[idx+1] = 1.0 if mkt.get("in_step3") else 0.0
    obs[idx+2] = mkt.get("deck_remaining", 0) / 50
    idx += 3

    # 12. Resource market (4) — denominators = price-track capacities.
    rm = state["resources"]
    obs[idx:idx+4] = [rm["coal"]/27, rm["oil"]/20, rm["gas"]/24, rm["uranium"]/12]
    idx += 4

    # 13. Phase id (1)
    phase = state["phase"]
    phase_key = list(phase.keys())[0] if isinstance(phase, dict) else phase
    obs[idx] = PHASE_IDS.get(phase_key, 0) / 9
    idx += 1

    # 14. Step (1)
    obs[idx] = state.get("step", 1) / 3
    idx += 1

    # 15. Round (1)
    obs[idx] = state.get("round", 0) / 50  # games can run past round 30 with random play
    idx += 1

    # 16. End-game cities threshold (1)
    obs[idx] = state.get("end_game_cities", 17) / 25
    idx += 1

    # 17. Turn-order position of this actor (1)
    try:
        pos = state["player_order"].index(actor_id)
        n = max(len(state["player_order"]) - 1, 1)
        obs[idx] = pos / n
    except (ValueError, KeyError):
        obs[idx] = 0.0
    idx += 1

    # 18. Phase-specific scratch features (8)
    ps = np.zeros(8, dtype=np.float32)
    if isinstance(phase, dict):
        if "auction" in phase:
            a = phase["auction"]
            ps[0] = a.get("current_bidder_idx", 0) / 5
            ab = a.get("active_bid")
            if ab:
                ps[1] = ab["amount"] / 200
                ps[2] = ab["plant_number"] / 60
                ps[3] = len(ab.get("remaining_bidders", [])) / 5
                ps[4] = 1.0
            ps[5] = len(a.get("bought", [])) / 6
            ps[6] = len(a.get("passed", [])) / 6
        elif "discard_plant" in phase:
            ps[0] = 1.0
        elif "discard_resource" in phase:
            ps[0] = phase["discard_resource"]["drop_total"] / 8
        elif "buy_resources" in phase:
            ps[0] = len(phase["buy_resources"]["remaining"]) / 6
        elif "build_cities" in phase:
            ps[0] = len(phase["build_cities"]["remaining"]) / 6
        elif "bureaucracy" in phase:
            ps[0] = len(phase["bureaucracy"]["remaining"]) / 6
        elif "power_cities_fuel" in phase:
            ps[0] = phase["power_cities_fuel"]["hybrid_cost"] / 20
    obs[idx:idx+8] = ps
    idx += 8

    # 19. Connection cost from the actor's network to each city (MAX_CITIES).
    owned = cities_by_player.get(actor_id, [])
    costs = _connection_costs_from(owned)
    for city_id, ci in CITY_INDEX.items():
        obs[idx + ci] = costs.get(city_id, 30) / 30.0
    idx += MAX_CITIES

    # 20. Opponent per-resource fuel demand (4)
    demand = {"coal": 0.0, "oil": 0.0, "gas": 0.0, "uranium": 0.0}
    for opp in opponents:
        for p in opp.get("plants", []):
            for res, d in _plant_demand(p).items():
                demand[res] += d
    obs[idx:idx+4] = [demand["coal"]/27, demand["oil"]/20, demand["gas"]/24, demand["uranium"]/12]
    idx += 4

    # 21. Opponent plants (5 × 3 × 5 = 75): mirror section 3 for each opponent.
    for i, opp in enumerate(opponents[:5]):
        for j, plant in enumerate((opp.get("plants") or [])[:3]):
            base = idx + (i * 3 + j) * 5
            obs[base]   = plant["number"] / 60
            obs[base+1] = KIND_IDS.get(plant["kind"], 0) / 6
            obs[base+2] = plant["cost"] / 5
            obs[base+3] = plant["cities"] / 8
            cap = plant["cost"] * 2 if plant["kind"] not in ("wind",) else 0
            obs[base+4] = cap / 10
    idx += 5 * 3 * 5

    assert idx == OBS_SIZE, f"Observation size mismatch: expected {OBS_SIZE}, got {idx}"
    # Clamp into the Box bounds: a few features (e.g. player stockpiles, late
    # rounds) can exceed their nominal denominator in extreme games.
    np.clip(obs, 0.0, 1.0, out=obs)
    return obs


def _find_player(state: dict, actor_id: str) -> dict | None:
    for p in state.get("players", []):
        if p["id"] == actor_id:
            return p
    return None
