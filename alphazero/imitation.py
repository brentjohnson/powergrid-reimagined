"""Behavior-cloning data generation: play hard-vs-hard heuristic games and
record (observation, mask, one-hot bot-action, perspective-relative outcome)
examples from every seat.

This is the dense-supervision counterpart to `selfplay.py`'s MCTS-derived
`pi` — instead of a visit-count distribution, `target_pi` is a one-hot on the
move the Rust hard bot actually played. Feeding that into the same
`NNetWrapper.train` (masked cross-entropy + value MSE, unchanged) is exact
cloning of the teacher policy.

Most phases (auction, bureaucracy, discard_plant, discard_resource,
power_cities_fuel) map one bot decision to exactly one of the 94 action
ids, so `bot_decide_id` is a direct match. Two phases decide a whole-turn
*batch* (`Action::BuildCities{city_ids}`, `Action::BuyResourceBatch
{purchases}`) where the encoding only has single-unit ids — but both decode
to handlers that do *not* end the turn (`handle_build_city`,
`handle_buy_resources`; only `DoneBuilding`/`DoneBuying` do), so both
decompose losslessly into one example per unit (each city; each individual
resource unit, in the bot's priority order) plus a final Done* example.
Replaying that exact sequence reproduces the same end state as the bot's
batch action in one shot — no information is lost and no move is skipped.
"""

from __future__ import annotations

import json

import numpy as np
from powergrid_env.constants import (
    BUILD_CITY_BASE,
    BUY_RESOURCE_BASE,
    CITY_INDEX,
    DONE_BUILDING,
    DONE_BUYING,
    N_ACTIONS,
    RESOURCE_IDX,
)

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
    """Play `n_games` full games with every seat driven by the Rust
    `difficulty` heuristic bot, collecting labeled examples from every seat.
    Returns `(examples, skipped)`, where `skipped` counts moves dropped
    because the bot had no move at all (should be rare/never) — `build_cities`
    and `buy_resources` decompose losslessly, as described in this module's
    docstring, and are never skipped."""
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

            action_json = game.bot_decide_json(difficulty)
            if action_json is None:
                break
            action = json.loads(action_json)
            kind = action["type"]

            if kind == "build_cities":
                for city_id in action["city_ids"]:
                    aid = BUILD_CITY_BASE + CITY_INDEX[city_id]
                    _record(history, game, to_move, aid)
                    game.apply(aid)
                _record(history, game, to_move, DONE_BUILDING)
                game.apply(DONE_BUILDING)
            elif kind == "build_city":
                aid = BUILD_CITY_BASE + CITY_INDEX[action["city_id"]]
                _record(history, game, to_move, aid)
                game.apply(aid)
                _record(history, game, to_move, DONE_BUILDING)
                game.apply(DONE_BUILDING)
            elif kind == "done_building":
                _record(history, game, to_move, DONE_BUILDING)
                game.apply(DONE_BUILDING)
            elif kind in ("buy_resources", "buy_resource_batch"):
                purchases = (
                    action["purchases"]
                    if kind == "buy_resource_batch"
                    else [(action["resource"], action["amount"])]
                )
                for resource, amount in purchases:
                    aid = BUY_RESOURCE_BASE + RESOURCE_IDX[resource]
                    for _ in range(amount):
                        _record(history, game, to_move, aid)
                        game.apply(aid)
                _record(history, game, to_move, DONE_BUYING)
                game.apply(DONE_BUYING)
            elif kind == "done_buying":
                _record(history, game, to_move, DONE_BUYING)
                game.apply(DONE_BUYING)
            else:
                action_id = game.bot_decide_id(difficulty)
                if action_id is None:
                    skipped += 1
                    game.apply_json(action_json)
                else:
                    _record(history, game, to_move, action_id)
                    game.apply(action_id)
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
