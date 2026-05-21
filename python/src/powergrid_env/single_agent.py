"""
PowerGridSingleAgentEnv: wraps PowerGridAECEnv for single-agent RL.

All seats except the learner's are filled by RustBotPolicy (the existing
Rust strategy bot). Only yields control to the learner when it is their turn,
advancing all bot turns automatically between steps.
"""

import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import powergrid_py  # type: ignore[import]

from .constants import COLORS, MAX_PLAYERS, N_ACTIONS, OBS_SIZE, PASS_AUCTION
from .encoding import encode_observation, id_to_action_json, mask_from_info


class PowerGridSingleAgentEnv(gym.Env):
    """
    Single-agent Gymnasium env.

    The learner occupies seat `learner_seat` (0-based). All other seats are
    controlled by the Rust strategy bot at `bot_difficulty`.

    Observation: flat float32 vector of length OBS_SIZE.
    Action:      Discrete(N_ACTIONS) with action_mask in info dict.
    Reward:      +1 on win, -1 on loss, 0 each step.
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        num_players: int = 4,
        learner_seat: int = 0,
        bot_difficulty: str = "normal",
        seed: int | None = None,
        reward_shaping: bool = False,
        render_mode: str | None = None,
    ):
        super().__init__()
        if not (2 <= num_players <= MAX_PLAYERS):
            raise ValueError(f"num_players must be 2–{MAX_PLAYERS}")
        if not (0 <= learner_seat < num_players):
            raise ValueError("learner_seat must be in range [0, num_players)")

        self.num_players = num_players
        self.learner_seat = learner_seat
        self.bot_difficulty = bot_difficulty
        self._init_seed = seed
        self.reward_shaping = reward_shaping
        self.render_mode = render_mode

        self.observation_space = spaces.Box(0.0, 1.0, (OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

        self.game: powergrid_py.Game | None = None
        self._learner_id: str | None = None
        self._state_cache: dict | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        effective_seed = seed if seed is not None else self._init_seed
        self.game = powergrid_py.Game(self.num_players, effective_seed)
        names = [f"agent_{i}" for i in range(self.num_players)]
        colors = COLORS[:self.num_players]
        self.game.start(names, colors)

        player_ids = self.game.player_ids()
        self._learner_id = player_ids[self.learner_seat]
        self._state_cache = json.loads(self.game.state_json())

        # Advance bots until it's the learner's turn (or game over).
        self._advance_bots()
        self._state_cache = json.loads(self.game.state_json())

        obs = encode_observation(self._state_cache, self._learner_id)
        info = {"action_mask": self._build_mask()}
        return obs, info

    def step(self, action: int):
        assert self.game is not None and self._learner_id is not None

        state = self._state_cache
        action_json = id_to_action_json(int(action), state, self._learner_id)

        try:
            self.game.apply(self._learner_id, action_json)
        except ValueError:
            # Invalid action: end episode with penalty.
            obs = encode_observation(self._state_cache, self._learner_id)
            return obs, -1.0, True, False, {"action_mask": self._build_mask()}

        self._state_cache = json.loads(self.game.state_json())

        if self.game.is_terminal():
            winner = self.game.winner()
            reward = 1.0 if winner == self._learner_id else -1.0
            obs = encode_observation(self._state_cache, self._learner_id)
            return obs, reward, True, False, {}

        # Let bots play until learner's turn or game over.
        self._advance_bots()
        self._state_cache = json.loads(self.game.state_json())

        if self.game.is_terminal():
            winner = self.game.winner()
            reward = 1.0 if winner == self._learner_id else -1.0
            obs = encode_observation(self._state_cache, self._learner_id)
            return obs, reward, True, False, {}

        reward = 0.0
        if self.reward_shaping:
            for p in self._state_cache.get("players", []):
                if p["id"] == self._learner_id:
                    city_count = sum(1 for owners in self._state_cache.get("city_owners", {}).values() if p["id"] in owners)
                    reward += city_count * 0.001
                    break

        obs = encode_observation(self._state_cache, self._learner_id)
        info = {"action_mask": self._build_mask()}
        return obs, reward, False, False, info

    def render(self) -> str | None:
        if self._state_cache is None:
            return None
        from .env import _render_ansi
        text = _render_ansi(self._state_cache)
        if self.render_mode == "human":
            print(text)
        return text

    def close(self) -> None:
        self.game = None
        self._state_cache = None

    def _advance_bots(self) -> None:
        """Drive all non-learner seats with the Rust bot until it's the learner's turn or terminal."""
        if self.game is None or self._learner_id is None:
            return
        for _ in range(200):  # safety cap
            if self.game.is_terminal():
                break
            actor = self.game.current_actor()
            if actor is None or actor == self._learner_id:
                break
            action_json = self.game.bot_decide(actor, self.bot_difficulty)
            if action_json is None:
                break
            try:
                self.game.apply(actor, action_json)
            except ValueError:
                break

    def action_masks(self) -> np.ndarray:
        """Called by MaskablePPO via env_method('action_masks')."""
        return self._build_mask()

    def _build_mask(self) -> np.ndarray:
        if self.game is None or self._learner_id is None:
            return np.zeros(N_ACTIONS, dtype=np.int8)
        info_json = self.game.legal_move_info(self._learner_id)
        move_info = json.loads(info_json)
        return mask_from_info(move_info, self._state_cache or {}, self._learner_id)
