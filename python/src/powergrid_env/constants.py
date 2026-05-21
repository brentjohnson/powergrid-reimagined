MAX_PLAYERS = 6
MAX_CITIES = 42
MAX_PLANTS_PER_PLAYER = 3

COLORS = ["red", "blue", "green", "yellow", "purple", "white"]

# Stable sorted city IDs for the Germany map — matches game.city_ids().
CITY_IDS = [
    "aachen", "augsburg", "berlin", "bremen", "cuxhaven",
    "dortmund", "dresden", "duesseldorf", "duisburg", "erfurt",
    "essen", "flensburg", "frankfurt", "frankfurt_oder", "freiburg",
    "fulda", "halle", "hamburg", "hannover", "kassel",
    "kiel", "koeln", "konstanz", "leipzig", "luebeck",
    "magdeburg", "mannheim", "muenchen", "muenster", "nuernberg",
    "osnabrueck", "passau", "regensburg", "rostock", "saarbruecken",
    "schwerin", "stuttgart", "torgelow", "trier", "wiesbaden",
    "wilhelmshaven", "wuerzburg",
]
assert len(CITY_IDS) == MAX_CITIES

CITY_INDEX: dict[str, int] = {c: i for i, c in enumerate(CITY_IDS)}

REGION_NAMES = ["northwest", "northeast", "west", "east", "southwest", "southeast"]

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

# ---------------------------------------------------------------------------
# Action space layout
# ---------------------------------------------------------------------------
PASS_AUCTION         = 0          # 1 action
DONE_BUYING          = 1          # 1 action
DONE_BUILDING        = 2          # 1 action
SELECT_PLANT_BASE    = 3          # 8 actions: actual[0..7] (only 0..5 used; future not selectable)
PLACE_BID_BASE       = 11         # 50 actions: bid at min+0, min+1, ..., min+49
DISCARD_PLANT_BASE   = 61         # 3 actions: discard player.plants[0..2]
BUILD_CITY_BASE      = 64         # 42 actions: one per city in CITY_IDS order
BUY_RESOURCE_BASE    = 106        # 4 actions: coal/oil/gas/uranium (1 unit)
POWER_CITIES_BASE    = 110        # 8 actions: bitmask 0..7 over first 3 plants
DISCARD_RESOURCE_BASE = 118       # 9 actions: gas_drop 0..8 (oil = total - gas)
POWER_FUEL_BASE      = 127        # 9 actions: gas 0..8 (oil = hybrid_cost - gas)
N_ACTIONS            = 136

# Observation vector size (flat float32).
OBS_SIZE = 404
