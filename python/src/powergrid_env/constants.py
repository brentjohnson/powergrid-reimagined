MAX_PLAYERS = 6
MAX_CITIES = 49
MAX_PLANTS_PER_PLAYER = 3

COLORS = ["red", "blue", "green", "yellow", "purple", "white"]

# Stable sorted city IDs for the USA map (assets/maps/usa.toml) — matches
# game.city_ids(). Must stay in sync with CITY_IDS in powergrid-py/src/lib.rs.
CITY_IDS = [
    "albuquerque", "atlanta", "boston", "calgary", "charlotte",
    "chicago", "chihuahua", "columbus", "dallas", "denver",
    "detroit", "edmonton", "guadalajara", "houston", "indianapolis",
    "jacksonville", "juarez", "kansascity", "lasvegas", "losangeles",
    "memphis", "mexicocityn", "mexicocitys", "miami", "milwaukee",
    "minneapolis", "monterrey", "montreal", "nashville", "neworleans",
    "newyorkn", "newyorks", "oklahomacity", "ottawa", "philadelphia",
    "pittsburgh", "portland", "quebec", "regina", "saltlakecity",
    "sanantonio", "sandiego", "sanfrancisco", "seattle", "stlouis",
    "toronto", "vancouver", "washington", "winnipeg",
]
assert len(CITY_IDS) == MAX_CITIES

CITY_INDEX: dict[str, int] = {c: i for i, c in enumerate(CITY_IDS)}

REGION_NAMES = ["central", "east", "northeast", "northwest", "south", "southwest", "west"]

N_REGIONS = len(REGION_NAMES)

KIND_IDS = {
    "coal": 1,
    "oil": 2,
    "gas_or_oil": 3,
    "gas": 4,
    "uranium": 5,
    "wind": 6,
}

PHASE_IDS = {
    "lobby": 0,
    "player_order": 1,
    "auction": 2,
    "discard_plant": 3,
    "discard_resource": 4,
    "buy_resources": 5,
    "build_cities": 6,
    "bureaucracy": 7,
    "power_cities_fuel": 8,
    "game_over": 9,
}

RESOURCE_IDX = {"coal": 0, "oil": 1, "gas": 2, "uranium": 3}

# Reward shaping: bonus per city powered, granted once per round when the
# player's powering resolves (analogous to income). A ~12-round game powering
# ~7 cities/round totals ≈ 0.8, below the ±1 terminal reward.
POWER_SHAPING_COEF = 0.01

# ---------------------------------------------------------------------------
# Macro action space layout — mirrors crate::macro_actions (Phase-2 rebuild).
#
# The policy chooses one complete phase-plan per turn (~50 decisions/game)
# instead of a primitive micro-action (~600/game). The old 94-id primitive
# layout — which shredded BuildCities/BuyResourceBatch into per-unit steps and
# imposed the compounding-error tax that capped every learner — was removed.
# Mask/apply/label all live natively (powergrid_py `action_mask`,
# `apply_action_id`, `bot_decide_id`); Python does NOT re-derive them.
# ---------------------------------------------------------------------------
NOMINATE_BASE   = 0   # 0..5: nominate market actual slot
N_NOMINATE      = 6
AUCTION_PASS    = 6
AUCTION_RAISE   = 7
# Build: the whole menu is "how many of the cheapest reachable cities".
# BUILD_COUNT_BASE + n builds exactly n (n = 0 is DoneBuilding); BUILD_DEFAULT
# is the heuristic's own plan, last so dedup prefers the explicit count.
BUILD_COUNT_BASE = 8
N_BUILD_COUNT   = 7   # n in 0..=6
BUILD_NOTHING   = BUILD_COUNT_BASE      # alias: n = 0
BUILD_DEFAULT   = 15
# Buy: the same quantity-ladder shape as build — "how many units of fuel".
# BUY_COUNT_BASE + n buys exactly n units, walking the heuristic's own fill
# order (rack needs first, then cheapest burnable stockpile); n = 0 is an empty
# batch. BUY_DEFAULT is the heuristic's own plan, last so dedup prefers the count.
BUY_COUNT_BASE  = 16
N_BUY_COUNT     = 7   # n in 0..=6
BUY_NOTHING     = BUY_COUNT_BASE        # alias: n = 0
BUY_DEFAULT     = 23
DISCARD_PLANT_BASE = 24  # 24..26: discard owned plant slot 0..2 (by number)
N_DISCARD_PLANT = 3
POWER_OPTIMAL   = 27
POWER_NOTHING   = 28
N_ACTIONS       = 29

# Observation vector size (flat float32): money + resources + self plants +
# self cities + opponent summary + opponent cities + city slot counts +
# active regions + actual market + future market + market meta +
# resource market + phase/step/round/end-game/turn-order scalars + scratch +
# per-city connection cost + opponent per-resource fuel demand + opponent plants.
OBS_SIZE = (
    1 + 4 + 15 + MAX_CITIES + 20 + 5 * MAX_CITIES + MAX_CITIES + N_REGIONS
    + 24 + 20 + 3 + 4 + 5 + 8
    + MAX_CITIES  # 19. connection cost from the actor's network to each city
    + 4           # 20. opponent per-resource fuel demand (coal, oil, gas, uranium)
    + 5 * 3 * 5   # 21. opponent plants (5 opp × 3 slots × 5 feats)
)
