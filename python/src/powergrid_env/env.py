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
    POWER_OPTIMAL, POWER_NOTHING,
    POWER_SHAPING_COEF,
)
from .encoding import encode_observation


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
        shaping_mode: str = "absolute",
        render_mode: str | None = None,
        end_game_cities: int | None = None,
    ):
        super().__init__()
        if not (2 <= num_players <= MAX_PLAYERS):
            raise ValueError(f"num_players must be 2–{MAX_PLAYERS}")
        if shaping_mode not in ("absolute", "relative"):
            raise ValueError("shaping_mode must be 'absolute' or 'relative'")
        # Curriculum override of the end-game city trigger. None = rulebook
        # default. Applied at reset.
        self.end_game_cities = end_game_cities
        self.num_players = num_players
        # Seed stream: one generator seeded once, drawing a fresh game seed per
        # episode. Same constructor seed → same reproducible *sequence* of games;
        # reusing one fixed seed every reset would replay the identical game.
        self._seed_rng = np.random.default_rng(seed)
        self.reward_shaping = reward_shaping
        self.shaping_mode = shaping_mode
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
        if seed is not None:
            self._seed_rng = np.random.default_rng(seed)
        game_seed = int(self._seed_rng.integers(1, 2**63))
        self.game = powergrid_py.Game(self.num_players, game_seed)
        names = [f"agent_{i}" for i in range(self.num_players)]
        colors = COLORS[:self.num_players]
        self.game.start(names, colors)
        if self.end_game_cities is not None:
            self.game.set_end_game_cities(self.end_game_cities)

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
            # Shouldn't happen with action-mask wrappers; fall back to any legal
            # macro so the game can proceed.
            legal = np.flatnonzero(self.game.action_mask(uuid))
            action = int(legal[0]) if len(legal) else 0

        try:
            # Apply the chosen MACRO natively (expands to its primitive action and
            # auto-resolves any trailing fuel/discard split).
            self.game.apply_action_id(uuid, int(action))
        except ValueError:
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
            self._shape_rewards(agent, uuid, int(action))

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
        # Per-viewer state so the agent's own money is visible (the shared
        # spectator cache zeroes all money).
        state = json.loads(self.game.state_json(uuid))
        return encode_observation(state, uuid)

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
        # Native macro mask (length N_ACTIONS) — the single source of truth.
        return self.game.action_mask(uuid).astype(np.int8)

    def _shape_rewards(self, agent: str, uuid: str, action: int) -> None:
        """Per-round powered-cities bonus, granted when the acting agent's
        powering resolves. `shaping_mode="absolute"` adds its own powered count
        (always ≥ 0); `"relative"` adds its lead over the best opponent (rewards
        out-powering the field, the win condition, and can go negative)."""
        # A POWER macro resolves the agent's powering for the round. `apply_macro`
        # auto-resolves the trailing hybrid fuel split in the same call, so
        # powering is fully settled here (no pending-fuel step to wait for).
        was_power_action = action in (POWER_OPTIMAL, POWER_NOTHING)
        if not was_power_action:
            return
        state = self._state_cache
        if state is None:
            return
        mine = 0
        opp_max = 0
        for p in state.get("players", []):
            powered = p.get("last_cities_powered", 0)
            if p["id"] == uuid:
                mine = powered
            else:
                opp_max = max(opp_max, powered)
        shaped = mine - opp_max if self.shaping_mode == "relative" else mine
        self.rewards[agent] += shaped * POWER_SHAPING_COEF


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
            f"  {p['name']:12s}  "
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
