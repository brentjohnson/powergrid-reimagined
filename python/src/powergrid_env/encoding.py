"""
Action and observation encoding/decoding for the PettingZoo env.

Action space layout (N_ACTIONS = 136):
  0          PassAuction
  1          DoneBuying
  2          DoneBuilding
  3..10      SelectPlant  slot 0..7   (actual[0..5]; only actual plants are selectable)
  11..60     PlaceBid     offset 0..49  amount = active_bid.amount+1 + offset
  61..63     DiscardPlant slot 0..2   (index into player.plants sorted by number)
  64..105    BuildCity    city index 0..41 in CITY_IDS order
  106..109   BuyResources resource index 0..3 (coal/oil/gas/uranium), 1 unit
  110..117   PowerCities  bitmask 0..7 over first 3 plants (sorted by number)
  118..126   DiscardResource  coal_drop 0..8  (oil = drop_total - coal)
  127..135   PowerCitiesFuel  coal 0..8       (oil = hybrid_cost - coal)
"""

import json
import numpy as np
from .constants import (
    N_ACTIONS, OBS_SIZE, CITY_IDS, CITY_INDEX, REGION_NAMES,
    KIND_IDS, PHASE_IDS, RESOURCE_IDX, MAX_PLAYERS,
    PASS_AUCTION, DONE_BUYING, DONE_BUILDING,
    SELECT_PLANT_BASE, PLACE_BID_BASE, DISCARD_PLANT_BASE,
    BUILD_CITY_BASE, BUY_RESOURCE_BASE, POWER_CITIES_BASE,
    DISCARD_RESOURCE_BASE, POWER_FUEL_BASE,
)


def mask_from_info(move_info: dict, state: dict, actor_id: str) -> np.ndarray:
    """Convert a LegalMoveInfo dict (from game.legal_move_info) to an action mask."""
    mask = np.zeros(N_ACTIONS, dtype=np.int8)

    if move_info.get("pass_auction"):
        mask[PASS_AUCTION] = 1

    if move_info.get("done_buying"):
        mask[DONE_BUYING] = 1

    if move_info.get("done_building"):
        mask[DONE_BUILDING] = 1

    for slot in move_info.get("select_plant_slots", []):
        if 0 <= slot < 8:
            mask[SELECT_PLANT_BASE + slot] = 1

    bid_min = move_info.get("bid_min")
    bid_max = move_info.get("bid_max")
    if bid_min is not None and bid_max is not None:
        for offset in range(50):
            if bid_min + offset <= bid_max:
                mask[PLACE_BID_BASE + offset] = 1
            else:
                break

    for slot in move_info.get("discard_plant_slots", []):
        if 0 <= slot < 3:
            mask[DISCARD_PLANT_BASE + slot] = 1

    for city_id in move_info.get("buildable_city_ids", []):
        ci = CITY_INDEX.get(city_id)
        if ci is not None:
            mask[BUILD_CITY_BASE + ci] = 1

    for ri in move_info.get("buyable_resources", []):
        if 0 <= ri < 4:
            mask[BUY_RESOURCE_BASE + ri] = 1

    for bm in move_info.get("power_subsets", []):
        if 0 <= bm < 8:
            mask[POWER_CITIES_BASE + bm] = 1

    for coal in move_info.get("discard_resource_coal", []):
        if 0 <= coal < 9:
            mask[DISCARD_RESOURCE_BASE + coal] = 1

    for coal in move_info.get("fuel_coal", []):
        if 0 <= coal < 9:
            mask[POWER_FUEL_BASE + coal] = 1

    return mask


def id_to_action_json(action_id: int, state: dict, actor_id: str) -> str:
    """Convert a flat action integer to the JSON string accepted by game.apply()."""
    if action_id == PASS_AUCTION:
        return '{"type":"pass_auction"}'

    if action_id == DONE_BUYING:
        return '{"type":"done_buying"}'

    if action_id == DONE_BUILDING:
        return '{"type":"done_building"}'

    if SELECT_PLANT_BASE <= action_id < PLACE_BID_BASE:
        slot = action_id - SELECT_PLANT_BASE
        mkt = state["market"]
        slots = mkt["actual"] + mkt["future"]
        if slot < len(slots):
            return json.dumps({"type": "select_plant", "plant_number": slots[slot]["number"]})
        return '{"type":"pass_auction"}'

    if PLACE_BID_BASE <= action_id < DISCARD_PLANT_BASE:
        offset = action_id - PLACE_BID_BASE
        phase = state["phase"]
        if isinstance(phase, dict) and "auction" in phase:
            ab = phase["auction"].get("active_bid")
            if ab:
                amount = ab["amount"] + 1 + offset
                return json.dumps({"type": "place_bid", "amount": amount})
        return '{"type":"pass_auction"}'

    if DISCARD_PLANT_BASE <= action_id < BUILD_CITY_BASE:
        slot = action_id - DISCARD_PLANT_BASE
        me = _find_player(state, actor_id)
        if me:
            plants = sorted(me["plants"], key=lambda p: p["number"])
            if slot < len(plants):
                return json.dumps({"type": "discard_plant", "plant_number": plants[slot]["number"]})
        return '{"type":"pass_auction"}'

    if BUILD_CITY_BASE <= action_id < BUY_RESOURCE_BASE:
        ci = action_id - BUILD_CITY_BASE
        if ci < len(CITY_IDS):
            return json.dumps({"type": "build_city", "city_id": CITY_IDS[ci]})
        return '{"type":"done_building"}'

    if BUY_RESOURCE_BASE <= action_id < POWER_CITIES_BASE:
        ri = action_id - BUY_RESOURCE_BASE
        resource = ["coal", "oil", "gas", "uranium"][ri]
        return json.dumps({"type": "buy_resource_batch", "purchases": [[resource, 1]]})

    if POWER_CITIES_BASE <= action_id < DISCARD_RESOURCE_BASE:
        bitmask = action_id - POWER_CITIES_BASE
        me = _find_player(state, actor_id)
        numbers: list[int] = []
        if me:
            plants = sorted(me["plants"], key=lambda p: p["number"])
            numbers = [plants[i]["number"] for i in range(min(len(plants), 3)) if bitmask & (1 << i)]
        return json.dumps({"type": "power_cities", "plant_numbers": numbers})

    if DISCARD_RESOURCE_BASE <= action_id < POWER_FUEL_BASE:
        gas = action_id - DISCARD_RESOURCE_BASE
        drop_total = 0
        phase = state["phase"]
        if isinstance(phase, dict) and "discard_resource" in phase:
            drop_total = phase["discard_resource"]["drop_total"]
        oil = max(0, drop_total - gas)
        return json.dumps({"type": "discard_resource", "gas": gas, "oil": oil})

    if POWER_FUEL_BASE <= action_id < N_ACTIONS:
        gas = action_id - POWER_FUEL_BASE
        hybrid_cost = 0
        phase = state["phase"]
        if isinstance(phase, dict) and "power_cities_fuel" in phase:
            hybrid_cost = phase["power_cities_fuel"]["hybrid_cost"]
        oil = max(0, hybrid_cost - gas)
        return json.dumps({"type": "power_cities_fuel", "gas": gas, "oil": oil})

    return '{"type":"pass_auction"}'


def action_json_to_id(action_json: str, state: dict, actor_id: str) -> int:
    """Reverse-encode an action JSON string to a flat integer id (for bot policy)."""
    try:
        action = json.loads(action_json)
    except json.JSONDecodeError:
        return PASS_AUCTION

    t = action.get("type", "")

    if t == "pass_auction":
        return PASS_AUCTION
    if t == "done_buying":
        return DONE_BUYING
    if t == "done_building":
        return DONE_BUILDING

    if t == "select_plant":
        plant_num = action["plant_number"]
        mkt = state["market"]
        for i, p in enumerate(mkt["actual"] + mkt["future"]):
            if p["number"] == plant_num:
                return SELECT_PLANT_BASE + i
        return PASS_AUCTION

    if t == "place_bid":
        amount = action["amount"]
        phase = state["phase"]
        if isinstance(phase, dict) and "auction" in phase:
            ab = phase["auction"].get("active_bid")
            if ab:
                base_min = ab["amount"] + 1
                offset = max(0, min(49, amount - base_min))
                return PLACE_BID_BASE + offset
        return PASS_AUCTION

    if t == "discard_plant":
        plant_num = action["plant_number"]
        me = _find_player(state, actor_id)
        if me:
            plants = sorted(me["plants"], key=lambda p: p["number"])
            for i, p in enumerate(plants):
                if p["number"] == plant_num:
                    return DISCARD_PLANT_BASE + i
        return DISCARD_PLANT_BASE

    if t == "build_city":
        ci = CITY_INDEX.get(action["city_id"])
        return BUILD_CITY_BASE + ci if ci is not None else DONE_BUILDING

    if t == "build_cities":
        city_ids_list = action.get("city_ids", [])
        if city_ids_list:
            ci = CITY_INDEX.get(city_ids_list[0])
            return BUILD_CITY_BASE + ci if ci is not None else DONE_BUILDING
        return DONE_BUILDING

    if t in ("buy_resource_batch", "buy_resources"):
        if t == "buy_resource_batch":
            purchases = action.get("purchases", [])
            if not purchases:
                return DONE_BUYING
            resource = purchases[0][0]
        else:
            resource = action["resource"]
        ri = RESOURCE_IDX.get(resource, 0)
        return BUY_RESOURCE_BASE + ri

    if t == "power_cities":
        plant_numbers = set(action.get("plant_numbers", []))
        me = _find_player(state, actor_id)
        bitmask = 0
        if me:
            plants = sorted(me["plants"], key=lambda p: p["number"])
            for i, p in enumerate(plants[:3]):
                if p["number"] in plant_numbers:
                    bitmask |= 1 << i
        return POWER_CITIES_BASE + bitmask

    if t == "discard_resource":
        return DISCARD_RESOURCE_BASE + min(action.get("coal", 0), 8)

    if t == "power_cities_fuel":
        return POWER_FUEL_BASE + min(action.get("coal", 0), 8)

    return PASS_AUCTION


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
    obs[idx:idx+4] = [r["coal"] / 24, r["oil"] / 24, r["gas"] / 24, r["uranium"] / 12]
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

    # 4. Self cities (42)
    for city_id in cities_by_player.get(actor_id, []):
        ci = CITY_INDEX.get(city_id)
        if ci is not None:
            obs[idx + ci] = 1.0
    idx += 42

    # 5. Opponents (5 × 5 = 25)
    for i, opp in enumerate(opponents[:5]):
        base = idx + i * 5
        obs[base]   = opp["money"] / 500
        obs[base+1] = len(opp.get("plants", [])) / 3
        obs[base+2] = len(cities_by_player.get(opp["id"], [])) / 42
        cap = sum(p["cost"] * 2 for p in opp.get("plants", []) if p["kind"] not in ("wind",))
        obs[base+3] = cap / 30
        obs[base+4] = opp.get("last_cities_powered", 0) / 21
    idx += 25

    # 6. Opponent cities (5 × 42 = 210)
    for i, opp in enumerate(opponents[:5]):
        for city_id in cities_by_player.get(opp["id"], []):
            ci = CITY_INDEX.get(city_id)
            if ci is not None:
                obs[idx + i * 42 + ci] = 1.0
    idx += 210

    # 7. City slot count (42)
    city_owners = state.get("city_owners", {})
    for city_id, ci in CITY_INDEX.items():
        obs[idx + ci] = len(city_owners.get(city_id, [])) / 3
    idx += 42

    # 8. Active regions (6)
    for i, region in enumerate(REGION_NAMES):
        if region in state.get("active_regions", []):
            obs[idx + i] = 1.0
    idx += 6

    # 9. Plant market actual (4 × 6 = 24): number, kind, cost, cities, present, discount
    mkt = state["market"]
    discount_tok = mkt.get("discount_token")
    for i, plant in enumerate(mkt.get("actual", [])[:4]):
        base = idx + i * 6
        obs[base]   = plant["number"] / 60
        obs[base+1] = KIND_IDS.get(plant["kind"], 0) / 6
        obs[base+2] = plant["cost"] / 5
        obs[base+3] = plant["cities"] / 8
        obs[base+4] = 1.0
        obs[base+5] = 1.0 if plant["number"] == discount_tok else 0.0
    idx += 24

    # 10. Plant market future (4 × 5 = 20): number, kind, cost, cities, present
    for i, plant in enumerate(mkt.get("future", [])[:4]):
        base = idx + i * 5
        obs[base]   = plant["number"] / 60
        obs[base+1] = KIND_IDS.get(plant["kind"], 0) / 6
        obs[base+2] = plant["cost"] / 5
        obs[base+3] = plant["cities"] / 8
        obs[base+4] = 1.0
    idx += 20

    # 11. Plant market meta (3)
    mkt = state["market"]
    obs[idx]   = 1.0 if mkt.get("step3_triggered") else 0.0
    obs[idx+1] = 1.0 if mkt.get("in_step3") else 0.0
    obs[idx+2] = mkt.get("deck_remaining", 0) / 50
    idx += 3

    # 12. Resource market (4)
    rm = state["resources"]
    obs[idx:idx+4] = [rm["coal"]/24, rm["oil"]/24, rm["gas"]/24, rm["uranium"]/12]
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

    assert idx == OBS_SIZE, f"Observation size mismatch: expected {OBS_SIZE}, got {idx}"
    return obs


def _find_player(state: dict, actor_id: str) -> dict | None:
    for p in state.get("players", []):
        if p["id"] == actor_id:
            return p
    return None
