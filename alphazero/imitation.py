"""Behavior-cloning data generation: play heuristic games and record
(observation, mask, one-hot teacher-macro, perspective-relative outcome)
examples from every seat.

This is the dense-supervision counterpart to `selfplay.py`'s MCTS-derived
`pi` — instead of a visit-count distribution, `target_pi` is a one-hot on the
**macro** the Rust teacher heuristic played. Feeding that into the same
`NNetWrapper.train` (masked cross-entropy + value MSE, unchanged) is exact
cloning of the teacher policy.

Since the Phase-2 macro rebuild, every phase maps one teacher decision to
exactly one macro id (`game.bot_decide_id` = `macro_actions::teacher_macro_id`),
so there is no build/buy batch-unrolling any more — one example per turn, ~50
per game instead of ~600. (The teacher is canonically the shipped champion
`hard`; the `difficulty` arg is accepted but does not change the label.)
"""

from __future__ import annotations

import numpy as np
from powergrid_env.constants import N_ACTIONS

from .config import AZConfig
from .game import PowerGridGame, to_relative_vector
from .selfplay import Example


def _one_hot(action_id: int) -> np.ndarray:
    pi = np.zeros(N_ACTIONS, dtype=np.float32)
    pi[action_id] = 1.0
    return pi


def _record(
    history: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
    game: PowerGridGame,
    to_move: str,
    action_id: int,
) -> None:
    history.append((game.observation(), game.action_mask(), _one_hot(action_id), to_move))


def generate_examples(
    n_games: int,
    seed: int,
    cfg: AZConfig,
    difficulty: str = "hard",
    end_game_cities: int | None = None,
) -> tuple[list[Example], int]:
    """Play `n_games` full games with every seat driven by the Rust teacher
    heuristic, collecting one labeled macro example per turn from every seat.
    Returns `(examples, skipped)`, where `skipped` counts turns where the
    teacher had no macro (should be rare/never)."""
    examples: list[Example] = []
    skipped = 0

    for game_idx in range(n_games):
        game = PowerGridGame(
            seed=seed + game_idx,
            num_players=cfg.num_players,
            end_game_cities=end_game_cities,
        )
        history: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
        moves = 0
        aborted = False

        while not game.is_terminal():
            if moves >= cfg.max_moves:
                aborted = True
                break
            to_move = game.current_player()
            assert to_move is not None

            macro_id = game.bot_decide_id(difficulty)
            if macro_id is None:
                skipped += 1
                break
            _record(history, game, to_move, macro_id)
            game.apply(macro_id)
            moves += 1

        if aborted or not history or not game.is_terminal():
            continue

        outcome = game.outcome()
        player_ids = game.player_ids()
        examples.extend(
            (obs, mask, pi, to_relative_vector(player_ids, to_move, outcome))
            for obs, mask, pi, to_move in history
        )

    return examples, skipped
