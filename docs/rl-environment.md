# Reinforcement Learning Environment

A [PettingZoo 1.26.1](https://pettingzoo.farama.org/) multi-agent environment wraps the Rust game engine via a PyO3 extension module. Use it to train neural-network agents with standard RL libraries.

## Quick start

```bash
cd python

# First time: create venv, build the Rust extension, install Python package.
make develop

# Run all tests (29 tests, ~12 s).
make test

# Roll out one game with the Rust strategy bots and print the event log.
.venv/bin/python scripts/play_game.py --all-bots --render

# Train a single-agent PPO policy vs Normal bots (CPU-friendly).
.venv/bin/python scripts/train_vs_bots.py --total-timesteps 500_000

# Train self-play (all seats share one policy).
.venv/bin/python scripts/train_selfplay.py --num-players 4 --total-timesteps 1_000_000
```

Training checkpoints are written to `python/runs/`.

---

## Architecture

```
Python (powergrid_env)          Rust (powergrid-py PyO3 crate)
──────────────────────          ──────────────────────────────
PowerGridAECEnv.step()   ──►   Game.apply(actor, action_json)
PowerGridAECEnv.observe()──►   Game.state_json()   →  GameStateView JSON
PowerGridAECEnv._mask()  ──►   Game.legal_move_info(actor) → JSON
RustBotPolicy.act()      ──►   Game.bot_decide(actor, difficulty)
```

The PyO3 crate (`crates/powergrid-py`) depends only on `powergrid-core` and `powergrid-bot-strategy`. There is no network, lobby, or server involved — every game step is a direct Rust function call.

---

## PettingZoo API

### `PowerGridAECEnv`

```python
from powergrid_env import PowerGridAECEnv

env = PowerGridAECEnv(num_players=4, seed=42, reward_shaping=False)
env.reset()

for agent in env.agent_iter():
    obs, reward, terminated, truncated, info = env.last()
    if terminated or truncated:
        action = None
    else:
        mask = info["action_mask"]          # np.ndarray of shape (136,), dtype int8
        action = env.action_space(agent).sample(mask)
    env.step(action)
```

**Parameters:**
- `num_players` — 2–6 players (default 4)
- `seed` — integer seed for reproducible games; `None` for random (default)
- `reward_shaping` — if `True`, adds a small per-step bonus proportional to cities owned
- `render_mode` — `"ansi"` or `"human"` for text rendering

**Spaces:**
- `observation_space(agent)` → `Box(0.0, 1.0, (405,), float32)` — flat normalised feature vector
- `action_space(agent)` → `Discrete(136)` — see action encoding table below

**Rewards:** sparse — `+1.0` to winner, `-1.0` to all others at game end; `0.0` every other step.

**Agents:** each agent's id is a UUID string (stable within an episode; deterministic when `seed` is set).

### `PowerGridSingleAgentEnv`

A `gymnasium.Env` that exposes a single learner seat while the remaining seats are driven by the Rust strategy bot.

```python
from powergrid_env import PowerGridSingleAgentEnv

env = PowerGridSingleAgentEnv(
    num_players=4,
    learner_seat=0,
    bot_difficulty="normal",   # "easy" | "normal" | "hard"
    seed=0,
    reward_shaping=True,
)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

---

## Action encoding (N = 136)

Each integer maps to one game action. The mask in `info["action_mask"]` is `1` only for legal actions in the current state.

| Range | Action | Notes |
|---|---|---|
| 0 | `PassAuction` | Forbidden in round 1 before buying a plant |
| 1 | `DoneBuying` | Always legal during BuyResources |
| 2 | `DoneBuilding` | Always legal during BuildCities |
| 3–10 | `SelectPlant` slot 0–7 | Only `actual` market plants (up to 6 in Step 3); future market not selectable |
| 11–60 | `PlaceBid` offset 0–49 | Bid amount = `active_bid.amount + 1 + offset`; masked above player's money |
| 61–63 | `DiscardPlant` slot 0–2 | Index into player's plants sorted by number; forced when winning a 4th plant |
| 64–105 | `BuildCity` city 0–41 | Sorted alphabetically; see constants.py for order |
| 106–109 | `BuyResources` coal/oil/gas/uranium | Buys 1 unit; masked if market empty, player over capacity, or unaffordable |
| 110–117 | `PowerCities` bitmask 0–7 | Bitmask over player's first 3 plants sorted by number; 0 = power nothing |
| 118–126 | `DiscardResource` gas\_drop 0–8 | `oil_drop = drop_total − gas_drop`; forced on hybrid-slot overflow |
| 127–135 | `PowerCitiesFuel` gas 0–8 | `oil = hybrid_cost − gas`; forced when hybrid fuel split is ambiguous |

---

## Observation encoding (dim = 405)

All values are normalised to `[0, 1]`. Segments in order:

| Segment | Size | Content |
|---|---|---|
| Self money | 1 | `money / 500` |
| Self resources | 4 | coal/24, oil/24, gas/24, uranium/12 |
| Self plants | 15 | 3 plant slots × (number/60, kind/6, cost/5, cities/8, capacity/10) |
| Self cities | 42 | Binary ownership vector (Germany cities in sorted order) |
| Opponents | 25 | 5 opponents × (money/500, n\_plants/3, n\_cities/42, total\_cap/30, last\_powered/21) |
| Opponent cities | 210 | 5 opponents × 42-city binary ownership |
| City slot count | 42 | `owner_count / 3` per city |
| Active regions | 6 | Binary region-active flags |
| Plant market actual | 20 | 4 slots × (number/60, kind/6, cost/5, cities/8, present=1) |
| Plant market future | 20 | Same layout; empty in Step 3 |
| Market meta | 3 | step3\_triggered, in\_step3, deck\_remaining/50 |
| Resource market | 4 | coal/24, oil/24, gas/24, uranium/12 |
| Phase id | 1 | 0–9 encoding of phase variant |
| Step | 1 | step/3 |
| Round | 1 | round/50 |
| End-game threshold | 1 | end\_game\_cities/25 |
| Turn-order position | 1 | actor's index in player\_order / (n\_players − 1) |
| Phase scratch | 8 | Phase-specific features (bid amount, bidder index, remaining queue length, etc.) |

**City ownership source:** `Player.cities` was removed from the wire format as a redundant duplicate of `city_owners`. All city-ownership slots in the observation (self cities, opponent cities, city slot count) are derived from `state["city_owners"]`, which maps `city_id → [player_id, ...]`.

**Fields NOT encoded:** `event_log` (a `Vec<String>` of human-readable game events) is present in the wire payload but intentionally excluded from the observation tensor — it contains no decision-relevant signal beyond what the structured state already encodes.

---

## Policies

Two reference policies are provided in `python/src/powergrid_env/policies/`:

**`RandomPolicy`** — samples uniformly from the legal action mask. Useful as a baseline and in random-rollout tests.

```python
from powergrid_env import RandomPolicy
policy = RandomPolicy(rng=np.random.default_rng(0))
action = policy.act(observation, action_mask)
```

**`RustBotPolicy`** — delegates to the Rust strategy bot at a chosen difficulty. Uses `game.bot_decide()` directly (does not decode the action id; for eval only).

```python
from powergrid_env import RustBotPolicy
policy = RustBotPolicy(difficulty="hard")
action = policy.act(game, agent_id, state_dict)
```

---

## Training with Stable-Baselines3

The training scripts use [sb3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)'s `MaskablePPO`, which natively consumes `info["action_mask"]`.

### vs. Bots

```bash
python scripts/train_vs_bots.py \
    --num-players 4 \
    --bot-difficulty normal \
    --total-timesteps 500_000 \
    --run-dir runs/vs_bots
```

Uses `PowerGridSingleAgentEnv`. The three bot-controlled seats are fixed opponents; only the learner seat is optimised. Converges faster than self-play for early training.

### Self-play

```bash
python scripts/train_selfplay.py \
    --num-players 4 \
    --total-timesteps 2_000_000 \
    --run-dir runs/selfplay
```

Uses `PowerGridAECEnv` converted to a vectorised SB3 env via SuperSuit's `pettingzoo_env_to_vec_env_v1`. All seats share a single policy network (parameter sharing / MAPPO). The agent must learn to reason from the perspective of whichever player it happens to control.

> **SuperSuit note:** SuperSuit (`pip install supersuit`) requires Python dev headers to build its `tinyscaler` C extension. If `pip install supersuit` fails, install `python3.14-devel` (or your distro's equivalent) first.

### Loading a checkpoint

```python
from sb3_contrib import MaskablePPO
model = MaskablePPO.load("runs/vs_bots/final_model")
obs, info = env.reset()
action, _ = model.predict(obs, action_masks=info["action_mask"], deterministic=True)
```

Or use the rollout script:

```bash
python scripts/play_game.py --model runs/vs_bots/final_model --render
```

---

## Tests

```bash
make test
# or
.venv/bin/pytest tests/ -v
```

| Test file | What it checks |
|---|---|
| `test_encoding.py` | Action roundtrip (id ↔ JSON), observation shape/range, city id ordering |
| `test_env.py` | `pettingzoo.test.api_test` conformance, seed determinism, mask non-empty at every step |
| `test_random_play.py` | 15 random games complete (reach `game_over`), no invalid actions slip through the mask |

---

## Code map

| Path | Purpose |
|---|---|
| `crates/powergrid-py/Cargo.toml` | PyO3 crate manifest (pyo3 0.28, cdylib) |
| `crates/powergrid-py/src/lib.rs` | `Game` class: `apply`, `bot_decide`, `legal_move_info`, `current_actor`, `city_ids`, etc. |
| `python/pyproject.toml` | Python package metadata (hatchling build backend) |
| `python/Makefile` | `make develop` = build Rust + install Python |
| `python/src/powergrid_env/constants.py` | Action layout constants, CITY_IDS, normalisation denominators |
| `python/src/powergrid_env/encoding.py` | `mask_from_info`, `id_to_action_json`, `action_json_to_id`, `encode_observation` |
| `python/src/powergrid_env/env.py` | `PowerGridAECEnv` (PettingZoo AEC) |
| `python/src/powergrid_env/single_agent.py` | `PowerGridSingleAgentEnv` (Gymnasium, vs Rust bots) |
| `python/src/powergrid_env/policies/` | `RandomPolicy`, `RustBotPolicy` |
| `python/scripts/train_selfplay.py` | Self-play MaskablePPO training |
| `python/scripts/train_vs_bots.py` | Single-agent MaskablePPO vs Rust bots |
| `python/scripts/play_game.py` | Rollout viewer / evaluation |
| `python/tests/` | Encoding, API conformance, and random-play tests |
