# Reinforcement Learning Environment

A [PettingZoo 1.26.1](https://pettingzoo.farama.org/) multi-agent environment wraps the Rust game engine via a PyO3 extension module. Use it to train neural-network agents with standard RL libraries.

For a step-by-step training runbook (start, resume, monitor, evaluate), see [python/TRAINING.md](../python/TRAINING.md).

## Quick start

```bash
cd python

# First time: create venv, build the Rust extension, install Python package.
make develop

# Run all tests.
make test

# Roll out one game with the Rust strategy bots and print the event log.
.venv/bin/python scripts/play_game.py --all-bots --render

# Train a single-agent PPO policy vs Normal bots (CPU-friendly).
.venv/bin/python scripts/train_vs_bots.py --total-timesteps 500_000

# Train self-play (all seats share one policy).
.venv/bin/python scripts/train_selfplay.py --num-players 4 --total-timesteps 1_000_000

# Measure a checkpoint's win rate vs the bots.
.venv/bin/python scripts/evaluate.py --model runs/vs_bots/best_model --games 100
```

Training checkpoints are written to `python/runs/`.

> **Important:** the extension is compiled from the Rust crates at `make develop` time. After *any* change to `powergrid-core`, `powergrid-bot-strategy`, or `powergrid-py`, re-run `make develop` — a stale `.so` silently plays by old rules.

---

## Architecture

```
Python (powergrid_env)            Rust (powergrid-py PyO3 crate)
──────────────────────            ──────────────────────────────
SelfPlayEnv.step()         ──►   Game.step_self_play(action_id)        (no JSON)
SingleAgentEnv.step()      ──►   Game.step_vs_bots(learner, id, diff)  (no JSON)
SingleAgentEnv.reset()     ──►   Game.advance_bots(learner, diff)      (no JSON)
AECEnv.step()              ──►   Game.apply(actor, action_json)
AECEnv.observe()           ──►   Game.state_json(viewer) → GameStateView JSON
AECEnv._mask()             ──►   Game.legal_move_info(actor) → JSON
RustBotPolicy.act()        ──►   Game.bot_decide(actor, difficulty)
```

The PyO3 crate (`crates/powergrid-py`) depends only on `powergrid-core` and `powergrid-bot-strategy`. There is no network, lobby, or server involved — every game step is a direct Rust function call.

The two training envs (`PowerGridSelfPlayEnv`, `PowerGridSingleAgentEnv`) use fused native methods that apply the action and return the next observation, mask, and reward in a single PyO3 round-trip — roughly an order of magnitude faster than the JSON path. The PettingZoo `PowerGridAECEnv` keeps the JSON path for API conformance and debugging.

---

## Environments

### `PowerGridSelfPlayEnv` (training, fastest)

Single `gymnasium.Env` in which every seat is played by the same policy; `step()` applies the current actor's action and returns the *next* actor's observation and mask. Reward is +1/−1 to the player who made the final move; the value function learns credit assignment via GAE.

### `PowerGridSingleAgentEnv` (training vs bots)

A `gymnasium.Env` exposing one learner seat; all other seats are driven inside Rust by the strategy bot via `step_vs_bots`.

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

### `PowerGridAECEnv` (PettingZoo)

```python
from powergrid_env import PowerGridAECEnv

env = PowerGridAECEnv(num_players=4, seed=42, reward_shaping=False)
env.reset()

for agent in env.agent_iter():
    obs, reward, terminated, truncated, info = env.last()
    if terminated or truncated:
        action = None
    else:
        mask = info["action_mask"]          # np.ndarray of shape (143,), dtype int8
        action = env.action_space(agent).sample(mask)
    env.step(action)
```

**Common parameters:**
- `num_players` — 2–6 players (default 4)
- `seed` — seeds a per-env generator that draws a *fresh game seed each episode*, so consecutive resets play different games while the overall sequence stays reproducible. `None` for nondeterministic.
- `reward_shaping` — if `True`, adds a small per-step bonus proportional to cities owned
- `render_mode` — `"ansi"` or `"human"` for text rendering

**Spaces:**
- `observation_space` → `Box(0.0, 1.0, (454,), float32)` — flat normalised feature vector
- `action_space` → `Discrete(143)` — see action encoding table below

**Rewards:** sparse — `+1.0` to winner, `-1.0` to all others at game end; `0.0` every other step.

---

## Action encoding (N = 143)

Each integer maps to one game action. The mask in `info["action_mask"]` is `1` only for legal actions in the current state. City actions cover the 49 cities of the default USA map (`assets/maps/usa.toml`), sorted alphabetically.

| Range | Action | Notes |
|---|---|---|
| 0 | `PassAuction` | Forbidden in round 1 before buying a plant |
| 1 | `DoneBuying` | Always legal during BuyResources |
| 2 | `DoneBuilding` | Always legal during BuildCities |
| 3–10 | `SelectPlant` slot 0–7 | Only `actual` market plants (up to 6 in Step 3); future market not selectable |
| 11–60 | `PlaceBid` offset 0–49 | Bid amount = `active_bid.amount + 1 + offset`; masked above player's money |
| 61–63 | `DiscardPlant` slot 0–2 | Index into player's plants sorted by number; forced when winning a 4th plant |
| 64–112 | `BuildCity` city 0–48 | Sorted alphabetically; see constants.py for order |
| 113–116 | `BuyResources` coal/oil/gas/uranium | Buys 1 unit; masked if market empty, player over capacity, or unaffordable |
| 117–124 | `PowerCities` bitmask 0–7 | Bitmask over player's first 3 plants sorted by number; 0 = power nothing |
| 125–133 | `DiscardResource` gas\_drop 0–8 | `oil_drop = drop_total − gas_drop`; forced on hybrid-slot overflow |
| 134–142 | `PowerCitiesFuel` gas 0–8 | `oil = hybrid_cost − gas`; forced when hybrid fuel split is ambiguous |

---

## Observation encoding (dim = 454)

All values are normalised and clamped to `[0, 1]`. Segments in order:

| Segment | Size | Content |
|---|---|---|
| Self money | 1 | `money / 500` |
| Self resources | 4 | coal/27, oil/20, gas/24, uranium/12 (market price-track capacities) |
| Self plants | 15 | 3 plant slots × (number/60, kind/6, cost/5, cities/8, capacity/10) |
| Self cities | 49 | Binary ownership vector (USA cities in sorted order) |
| Opponents | 20 | 5 opponents × (n\_plants/3, n\_cities/49, total\_cap/30, last\_powered/21) — money is hidden |
| Opponent cities | 245 | 5 opponents × 49-city binary ownership |
| City slot count | 49 | `owner_count / 3` per city |
| Active regions | 7 | Binary region-active flags |
| Plant market actual | 24 | 4 slots × (number/60, kind/6, cost/5, cities/8, present, discount\_token) |
| Plant market future | 20 | 4 slots × (number/60, kind/6, cost/5, cities/8, present); empty in Step 3 |
| Market meta | 3 | step3\_triggered, in\_step3, deck\_remaining/50 |
| Resource market | 4 | coal/27, oil/20, gas/24, uranium/12 |
| Phase id | 1 | 0–9 encoding of phase variant |
| Step | 1 | step/3 |
| Round | 1 | round/50 |
| End-game threshold | 1 | end\_game\_cities/25 |
| Turn-order position | 1 | actor's index in player\_order / (n\_players − 1) |
| Phase scratch | 8 | Phase-specific features (bid amount, bidder index, remaining queue length, etc.) |

The layout is defined twice and kept in sync by parity tests (`tests/test_native_bridge.py`): natively in `crates/powergrid-py/src/lib.rs` (`build_observation`) and in Python (`encoding.py::encode_observation`). The same applies to `CITY_IDS` and the action layout in `constants.py`. **If the default map or the layout changes, both sides must be regenerated and old checkpoints become incompatible.**

**City ownership source:** all city-ownership segments are derived from `city_owners` (`city_id → [player_id, ...]`); players carry no redundant city list.

**Hidden information:** opponent money, the deck's card faces, and the RNG seed are never exposed. The deck *size* (`deck_remaining`) is public, as in the physical game.

---

## Policies

Two reference policies live in `python/src/powergrid_env/policies/`:

**`RandomPolicy`** — samples uniformly from the legal action mask. Useful as a baseline and in random-rollout tests.

**`RustBotPolicy`** — delegates to the Rust strategy bot at a chosen difficulty via `game.bot_decide()`. Note that batch actions (multi-city builds, multi-resource buys) are lossily mapped onto the single-action id space, so bots are slightly weaker through this bridge than in the lobby.

---

## Training with Stable-Baselines3

The training scripts use [sb3-contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)'s `MaskablePPO`, which consumes the mask via the env's `action_masks()` method. See [python/TRAINING.md](../python/TRAINING.md) for the full workflow including resuming and monitoring.

```bash
python scripts/train_vs_bots.py  --total-timesteps 500_000  --run-dir runs/vs_bots
python scripts/train_selfplay.py --total-timesteps 2_000_000 --run-dir runs/selfplay
python scripts/evaluate.py --model runs/vs_bots/best_model --games 100
python scripts/play_game.py --model runs/vs_bots/best_model --render
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
| `test_random_play.py` | Random games complete (reach `game_over`), no invalid actions slip through the mask |
| `test_native_bridge.py` | Rust-native obs/mask/step parity vs the Python reference implementations |
| `test_reseeding.py` | Consecutive resets play different games; same seed reproduces the same sequence |

---

## Code map

| Path | Purpose |
|---|---|
| `crates/powergrid-py/Cargo.toml` | PyO3 crate manifest (pyo3 0.28, cdylib) |
| `crates/powergrid-py/src/lib.rs` | `Game` class: native obs/mask/step methods, `apply`, `bot_decide`, `legal_move_info`, etc. |
| `python/pyproject.toml` | Python package metadata (hatchling build backend) |
| `python/Makefile` | `make develop` = build Rust + install Python |
| `python/TRAINING.md` | Step-by-step training runbook |
| `python/src/powergrid_env/constants.py` | Action layout constants, CITY_IDS, normalisation denominators |
| `python/src/powergrid_env/encoding.py` | `mask_from_info`, `id_to_action_json`, `action_json_to_id`, `encode_observation` |
| `python/src/powergrid_env/env.py` | `PowerGridAECEnv` (PettingZoo AEC) |
| `python/src/powergrid_env/single_agent.py` | `PowerGridSingleAgentEnv` (Gymnasium, vs Rust bots, native fast path) |
| `python/src/powergrid_env/self_play.py` | `PowerGridSelfPlayEnv` (Gymnasium, shared-policy self-play, native fast path) |
| `python/src/powergrid_env/policies/` | `RandomPolicy`, `RustBotPolicy` |
| `python/scripts/train_selfplay.py` | Self-play MaskablePPO training |
| `python/scripts/train_vs_bots.py` | Single-agent MaskablePPO vs Rust bots |
| `python/scripts/evaluate.py` | Win-rate evaluation of a checkpoint vs bots |
| `python/scripts/play_game.py` | Rollout viewer |
| `python/tests/` | Encoding, API conformance, parity, reseeding, and random-play tests |
