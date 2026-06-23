"""PowerGridGame: thin AlphaZero adapter over the native `powergrid_py.Game`.

Wraps the Rust engine so MCTS can treat a position as a search node: fork
(clone) it, read the to-move player's masked observation, apply an action id,
and check terminal/outcome. Hidden information (deck order, opponent money)
lives inside the wrapped `powergrid_py.Game` and search is free to fork it
arbitrarily (perfect-information MCTS on the full seeded state) — but the
*network* is only ever shown `observation()`/`action_mask()`, which already
strip that hidden info (see `encoding.rs`), so the learned policy never
cheats, only the search does.
"""

from __future__ import annotations

import numpy as np
import powergrid_py as pg
from powergrid_env.constants import COLORS


class PowerGridGame:
    """One position's worth of state, plus the player-id bookkeeping AlphaZero
    needs (a fixed `num_players`-length value vector keyed by join order)."""

    def __init__(
        self,
        seed: int,
        num_players: int = 4,
        end_game_cities: int | None = None,
    ):
        self._game = pg.Game(num_players, seed)
        names = [f"p{i}" for i in range(num_players)]
        colors = COLORS[:num_players]
        self._game.start(names, colors)
        if end_game_cities is not None:
            self._game.set_end_game_cities(end_game_cities)
        self._player_ids = self._game.player_ids()

    # -- identity -------------------------------------------------------------
    def player_ids(self) -> list[str]:
        return self._player_ids

    def current_player(self) -> str | None:
        return self._game.current_actor()

    # -- AlphaZero primitives --------------------------------------------------
    def observation(self) -> np.ndarray:
        actor = self.current_player()
        assert actor is not None, "observation() requires a current actor"
        return self._game.observation(actor)

    def action_mask(self) -> np.ndarray:
        actor = self.current_player()
        assert actor is not None, "action_mask() requires a current actor"
        return self._game.action_mask(actor)

    def legal_action_ids(self) -> list[int]:
        return [int(i) for i in np.flatnonzero(self.action_mask())]

    def apply(self, action_id: int) -> None:
        actor = self.current_player()
        assert actor is not None, "apply() requires a current actor"
        self._game.apply_action_id(actor, int(action_id))

    def fork(self) -> PowerGridGame:
        """Independent copy for MCTS exploration — mutating the fork never
        affects `self` (backed by `Game::clone` in Rust, including the seeded
        RNG, so deck order stays reproducible from the fork point)."""
        clone = PowerGridGame.__new__(PowerGridGame)
        clone._game = self._game.copy()
        clone._player_ids = self._player_ids
        return clone

    def is_terminal(self) -> bool:
        return self._game.is_terminal()

    def winner(self) -> str | None:
        return self._game.winner()

    def outcome(self) -> dict[str, float]:
        """+1 for the winner, -1 for everyone else. Only valid when terminal."""
        winner = self.winner()
        assert winner is not None, "outcome() requires a terminal game"
        return {pid: (1.0 if pid == winner else -1.0) for pid in self._player_ids}

    # -- heuristic-bot driving (eval / fallback) -------------------------------
    def advance_bots(self, learner: str, difficulty: str) -> bool:
        """Drive every non-learner seat with the Rust heuristic bot until it's
        the learner's turn (or the game ends). Returns True if now terminal."""
        return self._game.advance_bots(learner, difficulty)

    def bot_apply(self, difficulty: str = "easy") -> bool:
        """Drive the current actor with the Rust heuristic bot once. Returns
        False if there is no current actor or the bot has no move."""
        actor = self.current_player()
        if actor is None:
            return False
        action_json = self._game.bot_decide(actor, difficulty)
        if action_json is None:
            return False
        self._game.apply(actor, action_json)
        return True

    def bot_decide_id(self, difficulty: str) -> int | None:
        """The current actor's heuristic-bot move as an action id (for the
        encoding's 143-action space), or `None` if there's no current actor,
        the bot has no move, or its choice isn't representable as an id."""
        actor = self.current_player()
        if actor is None:
            return None
        return self._game.bot_decide_id(actor, difficulty)

    def bot_decide_json(self, difficulty: str) -> str | None:
        """The current actor's heuristic-bot move as the full, raw action
        JSON — used by `imitation.py` for `build_cities`/`buy_resources`,
        where the bot's real decision is a multi-unit batch that
        `bot_decide_id` can't match against any single id (see that
        module's module docstring for why)."""
        actor = self.current_player()
        if actor is None:
            return None
        return self._game.bot_decide(actor, difficulty)

    def apply_json(self, action_json: str) -> None:
        """Apply a raw action JSON for the current actor, bypassing the
        action-id encoding — used to advance the game with the bot's real
        (possibly multi-unit) decision when that decision isn't representable
        as a single id."""
        actor = self.current_player()
        assert actor is not None, "apply_json() requires a current actor"
        self._game.apply(actor, action_json)


# ---------------------------------------------------------------------------
# Perspective-relative value vectors
#
# `build_observation` (encoding.rs) always writes self first, then opponents
# in `players` (join) order excluding self. The value head mirrors that
# layout so its output lines up with what the network's input already
# encodes — these two helpers convert between that per-to-move-player
# "relative" vector and an absolute {player_id: value} dict.
# ---------------------------------------------------------------------------


def relative_order(player_ids: list[str], to_move: str) -> list[str]:
    return [to_move] + [p for p in player_ids if p != to_move]


def to_relative_vector(
    player_ids: list[str], to_move: str, absolute: dict[str, float]
) -> np.ndarray:
    order = relative_order(player_ids, to_move)
    return np.array([absolute[p] for p in order], dtype=np.float32)


def to_absolute_dict(
    player_ids: list[str], to_move: str, relative: np.ndarray
) -> dict[str, float]:
    order = relative_order(player_ids, to_move)
    return {pid: float(v) for pid, v in zip(order, relative)}
