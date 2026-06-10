"""Wraps the Rust strategy bot as a Python policy callable."""

import numpy as np

import powergrid_py  # type: ignore[import]

from ..constants import PASS_AUCTION
from ..encoding import action_json_to_id


class RustBotPolicy:
    """
    Delegates decisions to the Rust strategy bot via `game.bot_decide()`.

    Use via `act(game, agent_id)` which returns a flat action integer.
    The `observation` / `action_mask` arguments are accepted for API
    compatibility but ignored — the bot uses the live game state directly.
    """

    def __init__(self, difficulty: str = "normal"):
        self.difficulty = difficulty

    def act(
        self,
        game: powergrid_py.Game,
        agent_id: str,
        state: dict,
        observation: np.ndarray | None = None,
        action_mask: np.ndarray | None = None,
    ) -> int:
        action_json = game.bot_decide(agent_id, self.difficulty)
        if action_json is None:
            return PASS_AUCTION
        return action_json_to_id(action_json, state, agent_id)
