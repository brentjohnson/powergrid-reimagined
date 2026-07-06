"""Tests for `alphazero.metrics`: terminal-state strategic stats and the
fixed-anchor Elo estimate."""

import math

from alphazero import metrics
from alphazero.game import PowerGridGame


def _play_to_terminal(game: PowerGridGame, difficulty: str = "easy") -> None:
    steps = 0
    while not game.is_terminal() and steps < 5000:
        assert game.bot_apply(difficulty), "heuristic bot should always have a move pre-terminal"
        steps += 1
    assert game.is_terminal(), "game should finish quickly with a tiny end_game_cities"


def test_plant_efficiency_empty():
    assert metrics.plant_efficiency([]) == 0.0


def test_plant_efficiency_wind_is_free():
    plants = [{"number": 1, "kind": "wind", "cost": 0, "cities": 3}]
    assert metrics.plant_efficiency(plants) == 3.0


def test_plant_efficiency_mixed():
    plants = [
        {"number": 1, "kind": "coal", "cost": 2, "cities": 2},  # 1.0
        {"number": 2, "kind": "wind", "cost": 0, "cities": 4},  # 4.0
    ]
    assert math.isclose(metrics.plant_efficiency(plants), 2.5)


def test_game_stats_sane_ranges():
    game = PowerGridGame(seed=42, end_game_cities=4)
    _play_to_terminal(game)

    for seat_id in game.player_ids():
        stats = metrics.game_stats(game, seat_id)
        assert stats["finish_position"] >= 1
        assert stats["finish_position"] <= len(game.player_ids())
        assert stats["final_cities"] >= 0
        assert stats["plants_owned"] >= 0
        assert stats["plant_efficiency"] >= 0
        assert stats["game_len"] >= 1
        assert stats["won"] in (0.0, 1.0)

    winner = game.winner()
    winner_stats = metrics.game_stats(game, winner)
    assert winner_stats["won"] == 1.0
    assert winner_stats["finish_position"] == 1.0


def test_finish_positions_is_a_permutation():
    game = PowerGridGame(seed=7, end_game_cities=4)
    _play_to_terminal(game)
    state = game.state()
    positions = metrics.finish_positions(state)
    assert sorted(positions.values()) == list(range(1, len(game.player_ids()) + 1))


def test_agent_elo_matches_anchor_at_50_percent():
    win_rates = {d: 0.5 for d in metrics.BENCHMARK_DIFFICULTIES}
    expected = sum(metrics.BOT_ELO_ANCHORS[d] for d in metrics.BENCHMARK_DIFFICULTIES) / len(
        metrics.BENCHMARK_DIFFICULTIES
    )
    assert math.isclose(metrics.agent_elo(win_rates), expected, abs_tol=1e-6)


def test_agent_elo_monotonic_in_win_rate():
    low = metrics.agent_elo({"normal": 0.2})
    mid = metrics.agent_elo({"normal": 0.5})
    high = metrics.agent_elo({"normal": 0.8})
    assert low < mid < high


def test_agent_elo_empty_win_rates_is_zero():
    assert metrics.agent_elo({}) == 0.0
