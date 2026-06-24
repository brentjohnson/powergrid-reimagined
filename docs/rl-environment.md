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

# Train self-play (opponents = frozen snapshots of the learner's policy).
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
SingleAgentEnv.step()      ──►   Game.step_vs_bots(learner, id, diff)  (no JSON)
SingleAgentEnv.reset()     ──►   Game.advance_bots(learner, diff)      (no JSON)
                                 Game.load_opponent_policy(bytes)      ("policy" mode)
AECEnv.step()              ──►   Game.apply(actor, action_json)
AECEnv.observe()           ──►   Game.state_json(viewer) → GameStateView JSON
AECEnv._mask()             ──►   Game.legal_move_info(actor) → JSON
RustBotPolicy.act()        ──►   Game.bot_decide(actor, difficulty)
```

The PyO3 crate (`crates/powergrid-py`) depends only on `powergrid-core` and `powergrid-bot-strategy`. There is no network, lobby, or server involved — every game step is a direct Rust function call.

The training env (`PowerGridSingleAgentEnv`) uses fused native methods that apply the action, drive all opponent seats, and return the next observation, mask, and reward in a single PyO3 round-trip — roughly an order of magnitude faster than the JSON path. The PettingZoo `PowerGridAECEnv` keeps the JSON path for API conformance and debugging.

---

## Environments

### `PowerGridSingleAgentEnv` (training)

A `gymnasium.Env` exposing one learner seat; all other seats are driven inside Rust via `step_vs_bots`, either by the heuristic strategy bot or — with `bot_difficulty="policy"` — by a frozen snapshot of the learner's own network (frozen-opponent self-play; see below). Reward is learner-centric: +1/−1 on the learner's final transition.

```python
from powergrid_env import PowerGridSingleAgentEnv

env = PowerGridSingleAgentEnv(
    num_players=4,
    learner_seat=0,
    bot_difficulty="normal",   # "easy" | "normal" | "hard" | "expert" | "policy"
    seed=0,
    reward_shaping=True,
)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

**Frozen-opponent self-play** (`scripts/train_selfplay.py`): the env is created with `bot_difficulty="policy"`; `OpponentSnapshotCallback` periodically serializes the current policy network (`powergrid_env.export.policy_state_dict_to_bytes`, the same `PGRLPOL1` format the Rust Expert bot consumes) and pushes it to the envs via `set_opponent_policy(bytes)`. Each env loads the snapshot into Rust at its next reset, so opponents improve alongside the learner while rewards stay correctly attributed to the learner's own moves. Until the first snapshot arrives (and, with `bot_mix=p`, for a random share of episodes) the env falls back to `"normal"` heuristic bots.

> Historical note: an earlier `PowerGridSelfPlayEnv` had all seats share one policy in a single transition stream. Its terminal reward went to whichever seat happened to make the round's last bureaucracy move (almost always the trailing player, since `GameOver` is only set at end-of-round), so the winner essentially never saw +1 — it was removed in favour of the frozen-opponent design.

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
        mask = info["action_mask"]          # np.ndarray of shape (94,), dtype int8
        action = env.action_space(agent).sample(mask)
    env.step(action)
```

**Common parameters:**
- `num_players` — 2–6 players (default 4)
- `seed` — seeds a per-env generator that draws a *fresh game seed each episode*, so consecutive resets play different games while the overall sequence stays reproducible. `None` for nondeterministic.
- `reward_shaping` — if `True`, adds a per-round powered-cities bonus (× `POWER_SHAPING_COEF`, in constants.py), granted when the player's powering resolves. `shaping_mode` selects the quantity:
  - `"absolute"` (default) — the player's **own** powered count (always ≥ 0). A clean "build more = more reward" signal; the better *teacher* for from-scratch runs.
  - `"relative"` — the player's **lead over the best opponent** (`own_powered − max_opponent_powered`, can go negative). Better *aligned* with the win condition (out-power the field), but a poor cold-start teacher: a from-scratch agent at full end-game-cities trails the bots every round, so the signal is always-negative and dominated by the opponent term, giving no usable gradient toward building.

  Recommended flow: **bootstrap with `absolute` (+ the egc curriculum)** to learn the build→power→win loop, then `--resume-from` with `--shaping-mode relative` (or `--no-reward-shaping`) to fine-tune for positional play. Eval is always unshaped, so `eval/mean_reward` stays a clean ±1 yardstick across phases.
- `render_mode` — `"ansi"` or `"human"` for text rendering

**Spaces:**
- `observation_space` → `Box(0.0, 1.0, (454,), float32)` — flat normalised feature vector
- `action_space` → `Discrete(94)` — see action encoding table below

**Rewards:** sparse — `+1.0` to winner, `-1.0` to all others at game end; `0.0` every other step.

---

## Action encoding (N = 94)

Each integer maps to one game action. The mask in `info["action_mask"]` is `1` only for legal actions in the current state. City actions cover the 49 cities of the default USA map (`assets/maps/usa.toml`), sorted alphabetically.

| Range | Action | Notes |
|---|---|---|
| 0 | `PassAuction` | Forbidden in round 1 before buying a plant |
| 1 | `DoneBuying` | Always legal during BuyResources |
| 2 | `DoneBuilding` | Always legal during BuildCities |
| 3–10 | `SelectPlant` slot 0–7 | Only `actual` market plants (up to 6 in Step 3); future market not selectable |
| 11 | `PlaceBid` | English-auction style: the only raise is +1 over the standing bid (`active_bid.amount + 1`); masked out once the player can't afford it. No jump bids — `PassAuction` (0) covers dropping out. (Before 2026-06-24 this was a 50-action range, offset 0–49, allowing arbitrary jump bids up to the player's money; self-play exploited it with large, non-strategic raises, so it was collapsed to a single +1 action.) |
| 12–14 | `DiscardPlant` slot 0–2 | Index into player's plants sorted by number; forced when winning a 4th plant |
| 15–63 | `BuildCity` city 0–48 | Sorted alphabetically; see constants.py for order |
| 64–67 | `BuyResources` coal/oil/gas/uranium | Additive: buys +1 unit and does *not* end the buy phase (mirrors `BuildCity`/`DoneBuilding`) — sequence several before `DoneBuying`. Masked per-unit if market empty, player over capacity (hybrid-aware), or unaffordable |
| 68–75 | `PowerCities` bitmask 0–7 | Bitmask over player's first 3 plants sorted by number; 0 = power nothing |
| 76–84 | `DiscardResource` gas\_drop 0–8 | `oil_drop = drop_total − gas_drop`; forced on hybrid-slot overflow |
| 85–93 | `PowerCitiesFuel` gas 0–8 | `oil = hybrid_cost − gas`; forced when hybrid fuel split is ambiguous |

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

**`RustBotPolicy`** — delegates to the Rust strategy bot at a chosen difficulty via `game.bot_decide()`. The bot is re-consulted every step, so its batch decisions (multi-city builds, multi-resource buys) aren't lost — they're realized incrementally as repeated single-unit/single-city ids (`BuildCity`×N + `DoneBuilding`; `BuyResources`×N + `DoneBuying`) until the bot itself returns the `Done*` action, exactly mirroring a real turn.

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
| `python/src/powergrid_env/single_agent.py` | `PowerGridSingleAgentEnv` (Gymnasium, vs Rust bots or frozen policy snapshots, native fast path) |
| `python/src/powergrid_env/export.py` | `policy_state_dict_to_bytes` — PGRLPOL1 policy serialization (export script + self-play snapshots) |
| `python/src/powergrid_env/policies/` | `RandomPolicy`, `RustBotPolicy` |
| `python/scripts/train_selfplay.py` | Frozen-opponent self-play MaskablePPO training |
| `python/scripts/train_vs_bots.py` | Single-agent MaskablePPO vs Rust bots |
| `python/scripts/evaluate.py` | Win-rate evaluation of a checkpoint vs bots |
| `python/scripts/run_report.py` | Status report for a run dir: checkpoints, live process, TB metrics, health flags |
| `python/scripts/play_game.py` | Rollout viewer |
| `python/tests/` | Encoding, API conformance, parity, reseeding, and random-play tests |
