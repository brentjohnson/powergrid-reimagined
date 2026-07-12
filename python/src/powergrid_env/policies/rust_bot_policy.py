"""Wraps the Rust strategy bot as a Python policy callable."""

import numpy as np

import powergrid_py  # type: ignore[import]

from ..constants import BUILD_NOTHING


class RustBotPolicy:
    """
    Delegates decisions to the Rust strategy bot via `game.bot_decide_id()`,
    which returns the **macro** id the heuristic would play (the imitation
    label). Use via `act(game, agent_id)` which returns a flat macro integer.
    The `observation` / `state` / `action_mask` arguments are accepted for API
    compatibility but ignored — the bot uses the live game state directly.
    """

    def __init__(self, difficulty: str = "normal"):
        self.difficulty = difficulty

    def act(
        self,
        game: powergrid_py.Game,
        agent_id: str,
        state: dict | None = None,
        observation: np.ndarray | None = None,
        action_mask: np.ndarray | None = None,
    ) -> int:
        macro_id = game.bot_decide_id(agent_id, self.difficulty)
        if macro_id is None:
            return BUILD_NOTHING  # harmless no-op fallback; shouldn't occur mid-game
        return int(macro_id)
