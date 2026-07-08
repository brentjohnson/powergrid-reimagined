"""Tests for the PowerGridGame AlphaZero adapter: mask/legal-action parity,
fork isolation, and terminal outcome correctness."""

import numpy as np
import pytest

from alphazero.game import PowerGridGame, to_absolute_dict, to_relative_vector


def test_legal_action_ids_matches_mask():
    game = PowerGridGame(seed=1)
    mask = game.action_mask()
    legal = game.legal_action_ids()
    assert legal == [int(i) for i in np.flatnonzero(mask)]
    assert len(legal) > 0


def test_fork_is_isolated():
    game = PowerGridGame(seed=2)
    fork = game.fork()
    before = game.observation().copy()

    action = fork.legal_action_ids()[0]
    fork.apply(action)

    after = game.observation()
    assert np.array_equal(before, after), "mutating the fork must not affect the original"


def test_relative_vector_round_trip():
    player_ids = ["a", "b", "c", "d"]
    absolute = {"a": 1.0, "b": -1.0, "c": -1.0, "d": -1.0}
    for to_move in player_ids:
        rel = to_relative_vector(player_ids, to_move, absolute)
        assert rel.shape == (4,)
        assert rel[0] == absolute[to_move]
        assert to_absolute_dict(player_ids, to_move, rel) == absolute


def test_outcome_is_rank_based():
    # A tiny end_game_cities trigger keeps this test fast; the Rust heuristic
    # bot has anti-stall guarantees (see CLAUDE.md / heuristic_termination
    # test), so driving every seat with it is a safe way to reach a real
    # terminal state quickly.
    game = PowerGridGame(seed=3, end_game_cities=4)
    steps = 0
    while not game.is_terminal() and steps < 5000:
        assert game.bot_apply("easy"), "heuristic bot should always have a move pre-terminal"
        steps += 1

    assert game.is_terminal(), "game should finish quickly with end_game_cities=4"
    outcome = game.outcome()
    # 4 players → values are a permutation of [+1, +1/3, -1/3, -1], one per seat.
    assert sorted(outcome.values()) == pytest.approx([-1.0, -1.0 / 3, 1.0 / 3, 1.0])
    # The engine winner is the +1 (top-ranked) seat.
    assert outcome[game.winner()] == 1.0
    assert max(outcome, key=outcome.get) == game.winner()
