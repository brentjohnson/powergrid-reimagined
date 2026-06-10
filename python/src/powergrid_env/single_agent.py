"""
PowerGridSingleAgentEnv: single-agent Gymnasium env vs Rust strategy bots.

All seats except the learner's are filled by the Rust strategy bot. Each step()
applies the learner's action and advances every bot turn inside Rust via
`step_vs_bots` — observation, mask, reward, and the bot moves all happen in a
single PyO3 round-trip with no JSON serialisation.
"""

import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import powergrid_py  # type: ignore[import]

from .constants import COLORS, MAX_PLAYERS, N_ACTIONS, OBS_SIZE


class PowerGridSingleAgentEnv(gym.Env):
    """
    Single-agent Gymnasium env.

    The learner occupies seat `learner_seat` (0-based). All other seats are
    controlled by the Rust strategy bot at `bot_difficulty`.

    Observation: flat float32 vector of length OBS_SIZE.
    Action:      Discrete(N_ACTIONS) with action_mask in info dict.
    Reward:      +1 on win, -1 on loss, 0 each step (plus optional shaping).
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
        # Seed stream: one generator seeded once, drawing a fresh game seed per
        # episode. Same constructor seed → same reproducible *sequence* of games;
        # reusing one fixed seed every reset would replay the identical game.
        self._seed_rng = np.random.default_rng(seed)
        self.reward_shaping = reward_shaping
        self.render_mode = render_mode

        self.observation_space = spaces.Box(0.0, 1.0, (OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

        self.game: powergrid_py.Game | None = None
        self._learner_id: str | None = None
        self._current_mask: np.ndarray = np.zeros(N_ACTIONS, dtype=np.uint8)
        self.learner_cities: int = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._seed_rng = np.random.default_rng(seed)
        game_seed = int(self._seed_rng.integers(1, 2**63))
        self.game = powergrid_py.Game(self.num_players, game_seed)
        names = [f"agent_{i}" for i in range(self.num_players)]
        colors = COLORS[:self.num_players]
        self.game.start(names, colors)

        player_ids = self.game.player_ids()
        self._learner_id = player_ids[self.learner_seat]
        self.learner_cities = 0

        # Advance bots until it's the learner's turn (or game over).
        self.game.advance_bots(self._learner_id, self.bot_difficulty)

        obs = np.asarray(self.game.observation(self._learner_id), dtype=np.float32)
        mask = np.asarray(self.game.action_mask(self._learner_id), dtype=np.uint8)
        self._current_mask = mask
        return obs, {"action_mask": mask}

    def step(self, action: int):
        assert self.game is not None and self._learner_id is not None

        try:
            obs_arr, mask_arr, reward, terminal, cities = self.game.step_vs_bots(
                self._learner_id, int(action), self.bot_difficulty
            )
        except ValueError:
            # Invalid action (out-of-mask move by the policy). End the episode
            # with a penalty so training can continue.
            obs = np.zeros(OBS_SIZE, dtype=np.float32)
            mask = np.zeros(N_ACTIONS, dtype=np.uint8)
            self._current_mask = mask
            return obs, -1.0, True, False, {"action_mask": mask}

        obs = np.asarray(obs_arr, dtype=np.float32)
        mask = np.asarray(mask_arr, dtype=np.uint8)
        self._current_mask = mask
        self.learner_cities = int(cities)

        reward = float(reward)
        if self.reward_shaping and not terminal:
            reward += self.learner_cities * 0.001

        return obs, reward, terminal, False, {"action_mask": mask}

    def render(self) -> str | None:
        if self.game is None:
            return None
        from .env import _render_ansi
        text = _render_ansi(json.loads(self.game.state_json()))
        if self.render_mode == "human":
            print(text)
        return text

    def close(self) -> None:
        self.game = None

    def action_masks(self) -> np.ndarray:
        """Called by MaskablePPO via env_method('action_masks')."""
        return self._current_mask
