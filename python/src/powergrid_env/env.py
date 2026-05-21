"""
PowerGridAECEnv: PettingZoo AEC environment wrapping the Rust Power Grid game engine.

Usage:
    from powergrid_env import PowerGridAECEnv
    env = PowerGridAECEnv(num_players=4, seed=42)
    env.reset()
    for agent in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()
        if terminated or truncated:
            action = None
        else:
            action = env.action_space(agent).sample(mask=info["action_mask"])
        env.step(action)
"""

import json
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers

import powergrid_py  # type: ignore[import]  # built by maturin

from .constants import (
    COLORS, MAX_PLAYERS, N_ACTIONS, OBS_SIZE,
)
from .encoding import encode_observation, id_to_action_json, mask_from_info


def env(**kwargs) -> AECEnv:
    """Convenience factory that wraps the raw env in PettingZoo's recommended wrappers."""
    raw = PowerGridAECEnv(**kwargs)
    raw = wrappers.AssertOutOfBoundsWrapper(raw)
    raw = wrappers.OrderEnforcingWrapper(raw)
    return raw


class PowerGridAECEnv(AECEnv):
    metadata = {
        "name": "powergrid_v1",
        "render_modes": ["human", "ansi"],
        "is_parallelizable": False,
    }

    def __init__(
        self,
        num_players: int = 4,
        seed: int | None = None,
        reward_shaping: bool = False,
        render_mode: str | None = None,
    ):
        super().__init__()
        if not (2 <= num_players <= MAX_PLAYERS):
            raise ValueError(f"num_players must be 2–{MAX_PLAYERS}")
        self.num_players = num_players
        self._init_seed = seed
        self.reward_shaping = reward_shaping
        self.render_mode = render_mode

        # Spaces are constant regardless of game state.
        obs_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32)
        act_space = spaces.Discrete(N_ACTIONS)

        # PettingZoo requires these to be populated before reset().
        # We use placeholder agent ids; real ids are set in reset().
        self.possible_agents = [f"player_{i}" for i in range(num_players)]
        self._obs_spaces = {a: obs_space for a in self.possible_agents}
        self._act_spaces = {a: act_space for a in self.possible_agents}

        # Game and per-agent state (initialised in reset).
        self.game: powergrid_py.Game | None = None
        self._state_cache: dict | None = None
        # Stable-ID ↔ game-UUID mappings, populated in reset().
        self._id_to_uuid: dict[str, str] = {}
        self._uuid_to_id: dict[str, str] = {}

    # ------------------------------------------------------------------
    # gymnasium.spaces accessors
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> spaces.Space:
        return self._obs_spaces.get(agent, next(iter(self._obs_spaces.values())))

    def action_space(self, agent: str) -> spaces.Space:
        return self._act_spaces.get(agent, next(iter(self._act_spaces.values())))

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None, options: dict | None = None) -> None:
        effective_seed = seed if seed is not None else self._init_seed
        self.game = powergrid_py.Game(self.num_players, effective_seed)
        names = [f"agent_{i}" for i in range(self.num_players)]
        colors = COLORS[:self.num_players]
        self.game.start(names, colors)

        # Build stable-ID ↔ UUID mappings. possible_agents stays as the
        # fixed placeholder list set in __init__ so wrappers that capture
        # possible_agents at construction time see a consistent value.
        uuids = self.game.player_ids()
        self._id_to_uuid = {pid: uuid for pid, uuid in zip(self.possible_agents, uuids)}
        self._uuid_to_id = {uuid: pid for pid, uuid in self._id_to_uuid.items()}

        self.agents = list(self.possible_agents)

        self._state_cache = json.loads(self.game.state_json())

        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {"action_mask": self._build_mask(a)} for a in self.agents}

        self.agent_selection = self._next_agent()

    def step(self, action: int | None) -> None:
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        uuid = self._id_to_uuid.get(agent, agent)

        # Reset instantaneous rewards each step.
        self.rewards = {a: 0.0 for a in self.agents}

        if action is None:
            action = PASS_AUCTION  # shouldn't happen with wrappers

        state = self._state_cache
        action_json = id_to_action_json(int(action), state, uuid)

        try:
            self.game.apply(uuid, action_json)
        except ValueError as e:
            # Invalid action: penalise and terminate.
            self.rewards[agent] = -1.0
            for a in self.agents:
                self.terminations[a] = True
            self._accumulate_rewards()
            return

        self._state_cache = json.loads(self.game.state_json())

        if self.game.is_terminal():
            winner_uuid = self.game.winner()
            winner = self._uuid_to_id.get(winner_uuid, winner_uuid)
            for a in self.agents:
                self.rewards[a] = 1.0 if a == winner else -1.0
                self.terminations[a] = True
        elif self.reward_shaping:
            self._shape_rewards(agent)

        self._accumulate_rewards()
        self.agent_selection = self._next_agent()

        # Update mask for the new current agent.
        if not all(self.terminations.values()):
            cur = self.agent_selection
            self.infos[cur] = {"action_mask": self._build_mask(cur)}

    def observe(self, agent: str) -> np.ndarray:
        if self._state_cache is None or self.game is None:
            return np.zeros(OBS_SIZE, dtype=np.float32)
        uuid = self._id_to_uuid.get(agent, agent)
        return encode_observation(self._state_cache, uuid)

    def render(self) -> str | None:
        if self._state_cache is None:
            return None
        if self.render_mode == "ansi":
            return _render_ansi(self._state_cache)
        if self.render_mode == "human":
            text = _render_ansi(self._state_cache)
            print(text)
        return None

    def close(self) -> None:
        self.game = None
        self._state_cache = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_agent(self) -> str:
        actor_uuid = self.game.current_actor() if self.game else None
        actor = self._uuid_to_id.get(actor_uuid, actor_uuid) if actor_uuid else None
        if actor and actor in self.agents:
            return actor
        # Fallback: first non-terminated agent.
        for a in self.agents:
            if not self.terminations.get(a, False):
                return a
        return self.agents[0]

    def _build_mask(self, agent: str) -> np.ndarray:
        if self.game is None:
            return np.zeros(N_ACTIONS, dtype=np.int8)
        uuid = self._id_to_uuid.get(agent, agent)
        move_info_json = self.game.legal_move_info(uuid)
        move_info = json.loads(move_info_json)
        state = self._state_cache or {}
        return mask_from_info(move_info, state, uuid)

    def _shape_rewards(self, agent: str) -> None:
        """Optional per-step reward shaping (delta cities × small bonus)."""
        state = self._state_cache
        if state is None:
            return
        city_owners = state.get("city_owners", {})
        cities_by_player: dict[str, int] = {}
        for owners in city_owners.values():
            for pid in owners:
                cities_by_player[pid] = cities_by_player.get(pid, 0) + 1
        for p in state.get("players", []):
            a = p["id"]
            if a in self.rewards:
                self.rewards[a] += cities_by_player.get(a, 0) * 0.001


def _render_ansi(state: dict) -> str:
    lines = []
    phase = state["phase"]
    phase_key = list(phase.keys())[0] if isinstance(phase, dict) else phase
    lines.append(f"Round {state.get('round', 0)}  Step {state.get('step', 1)}  Phase: {phase_key}")
    lines.append(f"Active regions: {', '.join(state.get('active_regions', []))}")
    lines.append("")

    for p in state.get("players", []):
        plants_str = ", ".join(f"{pl['number']}({pl['kind'][0]})" for pl in p.get("plants", []))
        r = p.get("resources", {})
        res_str = f"C{r.get('coal',0)} O{r.get('oil',0)} G{r.get('gas',0)} U{r.get('uranium',0)}"
        lines.append(
            f"  {p['name']:12s}  ${p['money']:4d}  "
            f"cities={sum(1 for owners in state.get('city_owners', {}).values() if p['id'] in owners):2d}  plants=[{plants_str}]  res={res_str}"
        )

    lines.append("")
    mkt = state.get("market", {})
    actual_str = " ".join(str(p["number"]) for p in mkt.get("actual", []))
    future_str = " ".join(str(p["number"]) for p in mkt.get("future", []))
    lines.append(f"Market actual=[{actual_str}] future=[{future_str}] deck={mkt.get('deck_remaining', 0)}")

    rm = state.get("resources", {})
    lines.append(
        f"Resources  coal={rm.get('coal',0)}  oil={rm.get('oil',0)}  "
        f"gas={rm.get('gas',0)}  uranium={rm.get('uranium',0)}"
    )

    if state.get("event_log"):
        lines.append("")
        for msg in state["event_log"][-5:]:
            lines.append(f"  » {msg}")

    return "\n".join(lines)


# Import at bottom to avoid circular reference from constants.
from .constants import PASS_AUCTION  # noqa: E402
