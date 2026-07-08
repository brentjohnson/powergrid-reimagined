"""Strategic game-metrics computed from a terminal `PowerGridGame`.

Pure helpers — no torch, no I/O — so `coach.py`/`arena.py`/`selfplay.py` can
call them at the point a game reaches `GameOver` (when `state()` unmasks every
player's money, see `game.py::PowerGridGame.state`) and fold the results into
the per-iteration metrics dict alongside the neural-net losses. Kept separate
from those modules so the computation is independently testable.
"""

from __future__ import annotations

import math

# `finish_positions`/`_city_count` moved to game.py (outcome() needs them for
# rank-based value targets); re-exported here so `metrics.finish_positions`
# keeps working for callers and tests.
from .game import PowerGridGame, _city_count, finish_positions

__all__ = [
    "plant_efficiency",
    "finish_positions",
    "game_stats",
    "agent_elo",
    "BENCHMARK_DIFFICULTIES",
    "BOT_ELO_ANCHORS",
]

# Fixed Elo anchors for the heuristic bot difficulties used by
# `arena.benchmark_suite`. These are arbitrary but fixed reference points —
# what matters is that they stay constant across runs so the resulting
# `agent_elo` curve is comparable iteration to iteration.
BENCHMARK_DIFFICULTIES = ("easy", "normal", "hard")
BOT_ELO_ANCHORS = {"easy": 800.0, "normal": 1000.0, "hard": 1200.0}


def plant_efficiency(plants: list[dict]) -> float:
    """Mean cities-powered-per-fuel-cost across a player's plants. Wind plants
    (cost == 0) need no fuel, so they count as maximally efficient (cities /
    1) rather than dividing by zero. 0.0 for a player with no plants."""
    if not plants:
        return 0.0
    ratios = [p["cities"] / p["cost"] if p["cost"] > 0 else float(p["cities"]) for p in plants]
    return sum(ratios) / len(ratios)


def game_stats(game: PowerGridGame, seat_id: str) -> dict:
    """Terminal-state strategic stats for `seat_id`. `game` must be terminal
    (`game.is_terminal()`), so `state()` returns every player's real money."""
    state = game.state()
    positions = finish_positions(state)
    player = next(p for p in state["players"] if p["id"] == seat_id)
    return {
        "won": 1.0 if positions[seat_id] == 1 else 0.0,
        "finish_position": float(positions[seat_id]),
        "end_money": float(player["money"]),
        "final_cities": float(_city_count(state, seat_id)),
        "plants_owned": float(len(player["plants"])),
        "plant_efficiency": plant_efficiency(player["plants"]),
        "game_len": float(state["round"]),
    }


def agent_elo(win_rates: dict[str, float]) -> float:
    """Fixed-anchor Elo estimate: invert each bot-difficulty win rate against
    that bot's fixed Elo anchor (`E = R + 400*log10(p/(1-p))`), then average
    across bots. `win_rates` is keyed by difficulty name; only entries that
    also appear in `BOT_ELO_ANCHORS` contribute. `p` is clamped away from 0/1
    so a shutout doesn't blow up to +/-inf."""
    estimates = []
    for difficulty, anchor in BOT_ELO_ANCHORS.items():
        if difficulty not in win_rates:
            continue
        p = min(max(win_rates[difficulty], 0.01), 0.99)
        estimates.append(anchor + 400.0 * math.log10(p / (1.0 - p)))
    if not estimates:
        return 0.0
    return sum(estimates) / len(estimates)
