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
# Action space layout
# ---------------------------------------------------------------------------
PASS_AUCTION         = 0          # 1 action
DONE_BUYING          = 1          # 1 action
DONE_BUILDING        = 2          # 1 action
SELECT_PLANT_BASE    = 3          # 8 actions: actual[0..7] (only 0..5 used; future not selectable)
PLACE_BID_BASE       = 11         # 1 action: raise +1 over the standing bid (English-auction
                                   # style; PassAuction covers dropping out). Mirrors the Rust
                                   # N_BID_ACTIONS constant in encoding.rs.
N_BID_ACTIONS        = 1
DISCARD_PLANT_BASE   = PLACE_BID_BASE + N_BID_ACTIONS  # 3 actions: discard player.plants[0..2]
BUILD_CITY_BASE      = DISCARD_PLANT_BASE + 3          # MAX_CITIES actions: one per city in CITY_IDS order
BUY_RESOURCE_BASE    = BUILD_CITY_BASE + MAX_CITIES   # 4 actions: coal/oil/gas/uranium (1 unit)
POWER_CITIES_BASE    = BUY_RESOURCE_BASE + 4          # 8 actions: bitmask 0..7 over first 3 plants
DISCARD_RESOURCE_BASE = POWER_CITIES_BASE + 8         # 9 actions: gas_drop 0..8 (oil = total - gas)
POWER_FUEL_BASE      = DISCARD_RESOURCE_BASE + 9      # 9 actions: gas 0..8 (oil = hybrid_cost - gas)
N_ACTIONS            = POWER_FUEL_BASE + 9

# Observation vector size (flat float32): money + resources + self plants +
# self cities + opponent summary + opponent cities + city slot counts +
# active regions + actual market + future market + market meta +
# resource market + phase/step/round/end-game/turn-order scalars + scratch.
OBS_SIZE = (
    1 + 4 + 15 + MAX_CITIES + 20 + 5 * MAX_CITIES + MAX_CITIES + N_REGIONS
    + 24 + 20 + 3 + 4 + 5 + 8
)
