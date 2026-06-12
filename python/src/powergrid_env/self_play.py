"""
PowerGridSelfPlayEnv: single-agent Gymnasium env for shared-policy self-play.

All seats use the same policy. On each step() the env applies the action for the
current actor and returns the *next* actor's obs + mask in one Rust call — no JSON
round-trips, no per-agent padding waste from the turn_based wrapper chain.

Reward: +1 to the player who made the final move and won, -1 if they lost, 0 otherwise.
The value function learns to credit earlier moves via GAE. With `reward_shaping`,
a per-round bonus of POWER_SHAPING_COEF × cities powered is added to the acting
seat's step when its powering resolves — the same income-analogous signal
PowerGridSingleAgentEnv uses.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import powergrid_py  # type: ignore[import]

from .constants import COLORS, MAX_PLAYERS, N_ACTIONS, OBS_SIZE, POWER_SHAPING_COEF


class PowerGridSelfPlayEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        num_players: int = 4,
        seed: int | None = None,
        reward_shaping: bool = False,
        end_game_cities: int | None = None,
    ):
        super().__init__()
        if not (2 <= num_players <= MAX_PLAYERS):
            raise ValueError(f"num_players must be 2–{MAX_PLAYERS}")
        self.num_players = num_players
        self.reward_shaping = reward_shaping
        # Curriculum override of the end-game city trigger. None = rulebook
        # default. Applied at reset, so a mid-episode change (via
        # set_end_game_cities) takes effect from the next episode.
        self.end_game_cities = end_game_cities
        # Seed stream: one generator seeded once, drawing a fresh game seed per
        # episode. Same constructor seed → same reproducible *sequence* of games;
        # reusing one fixed seed every reset would replay the identical game.
        self._seed_rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(0.0, 1.0, (OBS_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)

        self.game: powergrid_py.Game | None = None
        self._current_mask: np.ndarray = np.zeros(N_ACTIONS, dtype=np.uint8)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._seed_rng = np.random.default_rng(seed)
        game_seed = int(self._seed_rng.integers(1, 2**63))
        self.game = powergrid_py.Game(self.num_players, game_seed)
        names = [f"agent_{i}" for i in range(self.num_players)]
        colors = COLORS[:self.num_players]
        self.game.start(names, colors)
        if self.end_game_cities is not None:
            self.game.set_end_game_cities(self.end_game_cities)

        actor = self.game.current_actor()
        obs = np.asarray(self.game.observation(actor), dtype=np.float32)
        mask = np.asarray(self.game.action_mask(actor), dtype=np.uint8)
        self._current_mask = mask
        return obs, {"action_mask": mask}

    def step(self, action: int):
        assert self.game is not None
        try:
            obs_arr, mask_arr, reward, terminal, powered_now = self.game.step_self_play(int(action))
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

        reward = float(reward)
        if self.reward_shaping and not terminal:
            reward += int(powered_now) * POWER_SHAPING_COEF

        return obs, reward, terminal, False, {"action_mask": mask}

    def action_masks(self) -> np.ndarray:
        """Called by MaskablePPO via env_method('action_masks')."""
        return self._current_mask

    def set_end_game_cities(self, n: int | None) -> None:
        """Curriculum hook (called via VecEnv.env_method). Applies from the
        next reset; the episode in progress keeps its current trigger."""
        self.end_game_cities = n

    def close(self) -> None:
        self.game = None
