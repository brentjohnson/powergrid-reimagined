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
AECEnv.step()              ──►   Game.apply_action_id(actor, macro_id)
AECEnv.observe()           ──►   Game.state_json(viewer) → GameStateView JSON
AECEnv._mask()             ──►   Game.action_mask(actor) → np.uint8[26]
RustBotPolicy.act()        ──►   Game.bot_decide_id(actor, difficulty)
```

Macro legality, expansion, and teacher labelling all live natively in `crates/powergrid-bot-strategy/src/macro_actions.rs` — Python never re-derives game rules. (`Game.apply(actor, action_json)` and `Game.legal_move_info(actor)` still exist for primitive-level debugging, but no env uses them.)

The PyO3 crate (`crates/powergrid-py`) depends only on `powergrid-core` and `powergrid-bot-strategy`. There is no network, lobby, or server involved — every game step is a direct Rust function call.

The training env (`PowerGridSingleAgentEnv`) uses fused native methods that apply the action, drive all opponent seats, and return the next observation, mask, and reward in a single PyO3 round-trip — roughly an order of magnitude faster than the JSON path. The PettingZoo `PowerGridAECEnv` keeps the JSON path for API conformance and debugging.

---

## Environments

### `PowerGridSingleAgentEnv` (training)

A `gymnasium.Env` exposing one learner seat; all other seats are driven inside Rust via `step_vs_bots`, either by the heuristic strategy bot or — with `bot_difficulty="policy"` — by a frozen snapshot of the learner's own network (frozen-opponent self-play; see below). Reward is learner-centric: +1/−1 on the learner's final transition, or, with `terminal_reward="placement"`, the learner's final rank mapped linearly onto [−1, +1] (computed from the terminal state via `powergrid_env.stats.learner_stats`).

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

**Frozen-opponent self-play** (`scripts/train_selfplay.py`): the env is created with `bot_difficulty="policy"`; `OpponentSnapshotCallback` periodically serializes the current policy network (`powergrid_env.export.policy_state_dict_to_bytes`, the same `PGRLPOL1` format the Rust Expert bot consumes) and pushes it to the envs via `set_opponent_policy(bytes)`. Each env loads the snapshot into Rust at its next reset, so opponents improve alongside the learner while rewards stay correctly attributed to the learner's own moves. Until the first snapshot arrives (and, with `bot_mix=p`, for a random share of episodes) the env falls back to `"hard"` heuristic bots.

**League self-play** (the default for `train_selfplay.py`): instead of a single snapshot, `LeagueSnapshotCallback` persists every snapshot to `<run-dir>/league/snap_<steps>.bin` and pushes a weighted pool via `set_opponent_pool([(kind, payload, weight), ...])` — `("policy", pgrlpol1_bytes, w)` or `("bots", difficulty, w)` entries, sampled independently at each reset. The pool overrides `set_opponent_policy`/`bot_mix` while set. Two more `env_method` hooks support training schedules: `set_shaping_scale(f)` (multiplier on the powered-cities bonus, driven by `ShapingAnnealCallback` to anneal shaping away) and `set_end_game_cities(n)` (curriculum). See [python/TRAINING.md](../python/TRAINING.md) for the flags and `scripts/orchestrate.py` for the hands-off train → evaluate → adapt loop built on top.

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
        mask = info["action_mask"]          # np.ndarray of shape (26,), dtype int8
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
- `observation_space` → `Box(0.0, 1.0, (582,), float32)` — flat normalised feature vector
- `action_space` → `Discrete(26)` — macro actions, see table below

**Rewards:** sparse — `+1.0` to winner, `-1.0` to all others at game end; `0.0` every other step.

---

## Action encoding — macro actions (N = 26)

The policy does **not** choose primitive game moves. It chooses one complete **phase-plan per turn** from a fixed menu of 26 macros; each expands natively into a short primitive sequence the engine already accepts as a whole-turn batch (`BuildCities`, `BuyResourceBatch`). A game is ~50 macro decisions instead of ~600 primitive ones, which removes the compounding-error tax that capped every earlier learner.

> **History:** until the Phase-2 rebuild this was a 94-id *primitive* space (`BuildCity` per city, `BuyResources` per unit, etc.). It was removed, not extended. Any policy trained against it is incompatible.

| Id | Macro | Notes |
|---|---|---|
| 0–5 | `NOMINATE` market slot 0–5 | Auction with no standing bid: nominate an `actual` market plant |
| 6 | `AUCTION_PASS` | Drop out of / decline the auction (both auction sub-phases) |
| 7 | `AUCTION_RAISE` | Raise +1 over the standing bid (English-auction convention). No jump bids — self-play with a ±50 raise range learned large non-strategic jumps |
| 8–14 | `BUILD_COUNT_BASE + n`, n = 0…6 | Build the **n cheapest** reachable cities you can pay for. `n = 0` is `DoneBuilding`. Cash is the only limit — no reserve is withheld |
| 15 | `BUILD_DEFAULT` | Whatever the champion `hard` heuristic would build, **bit-exactly** (Gate 0). Last id so dedup prefers the explicit count; in practice a count always reproduces it, so this is dead weight kept as a safety valve for >6-city plans |
| 16 | `BUY_NOTHING` | `DoneBuying` |
| 17 | `BUY_DEFAULT` | The heuristic's resource batch, bit-exactly |
| 18–19 | `BUY_STOCKPILE2/3` | The heuristic with `stockpile_rounds` forced to 2 / 3 rounds of fuel |
| 20 | `BUY_DENIAL` | Buy out the resource with the highest forward price, to deny it to opponents |
| 21–23 | `DISCARD_PLANT` slot 0–2 | Index into the player's plants sorted by number; forced when winning a 4th plant |
| 24 | `POWER_OPTIMAL` | The heuristic's optimal firing subset |
| 25 | `POWER_NOTHING` | Power no cities (earn minimum income) |

**Auto-resolved phases.** `PowerCitiesFuel` (hybrid gas/oil split) and `DiscardResource` (hybrid-slot overflow) are minor tactical steps with no strategic content, so `macro_actions::resolve_auto_phases` settles them with the heuristic. They never consume a policy decision and have no macro ids.

**Masking and dedup.** `info["action_mask"]` comes from `macro_actions::legal_macros`, which validates each macro by trial application on a cloned state **and drops duplicates** — a macro whose primitive expansion equals a lower-id macro's is marked illegal. So `BUILD_3` collapses onto `BUILD_2` when only two cities are affordable. Dedup is why id ordering matters: counts sit *below* `BUILD_DEFAULT` so that "build 2" is always id 10, never sometimes-10-sometimes-15 depending on whether the heuristic happened to agree.

**Imitation labels.** `Game.bot_decide_id(actor, difficulty)` returns the macro id the champion heuristic would play (`macro_actions::teacher_macro_id`) — always the id that *survives dedup*, so the label is never an illegal action. This is the behavior-cloning / DAgger target.

---

## Observation encoding (dim = 582)

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
| Connection cost to city | 49 | `route_cost / 30` per city — cheapest Dijkstra cost to connect each city to the actor's network (`Map::connection_costs_from`); `0` for cities already owned and for an empty network |
| Opponent fuel demand | 4 | coal/27, oil/20, gas/24, uranium/12 — total per-round fuel demand summed across every opponent's plants (hybrids split cost across gas/oil) |
| Opponent plants | 75 | 5 opponents × 3 plant slots × (number/60, kind/6, cost/5, cities/8, capacity/10) — each opponent's rack encoded like *Self plants*. Surfaces opponents' highest plant number (the turn-order tiebreaker, and the sole determinant of order in round 1) and the per-plant kind/cost/cities the bot's denial/fuel models read |

The last three segments were added 2026-07-08 (obs 454 → 507 → 582) to close a capacity-independent information ceiling: the heuristic bot leans on all three — connection/routing cost drives its build decisions, opponent fuel demand drives its resource-market contention model, and opponent plant numbers/kinds/costs drive turn-order and denial/fuel reasoning — but the net previously couldn't see any of them (the map graph is not in `GameStateView`, and the opponent summary was a coarse count/capacity rollup with no per-plant detail). See `crates/powergrid-bot-strategy/src/encoding.rs` for the rationale.

The layout is defined twice and kept in sync by parity tests (`tests/test_native_bridge.py`): natively in `crates/powergrid-bot-strategy/src/encoding.rs` (`build_observation`, wrapped by `powergrid-py`) and in Python (`encoding.py::encode_observation`, which loads `assets/maps/usa.toml` and replicates the routing Dijkstra). The same applies to `CITY_IDS` and the action layout in `constants.py`. **If the default map or the layout changes, both sides must be regenerated and old checkpoints become incompatible.**

**City ownership source:** all city-ownership segments are derived from `city_owners` (`city_id → [player_id, ...]`); players carry no redundant city list.

**Hidden information:** opponent money, the deck's card faces, and the RNG seed are never exposed. The deck *size* (`deck_remaining`) is public, as in the physical game. Opponent *plants* are public (the fuel-demand segment is derived from them), matching the physical game where a rival's power plants sit face-up.

---

## Policies

Two reference policies live in `python/src/powergrid_env/policies/`:

**`RandomPolicy`** — samples uniformly from the legal action mask. Useful as a baseline and in random-rollout tests.

**`RustBotPolicy`** — delegates to the Rust strategy bot via `game.bot_decide_id()`, which returns the heuristic's move as a single macro id. Because a macro *is* a whole-turn plan, the bot's batch decisions (multi-city builds, multi-resource buys) survive intact — no incremental replay is needed, unlike the primitive encoding this replaced.

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
| `test_encoding.py` | Observation shape/range, city id ordering, macro mask shape + non-empty at start, teacher macro is always legal |
| `test_env.py` | `pettingzoo.test.api_test` conformance, seed determinism, mask non-empty at every step |
| `test_random_play.py` | Random games complete (reach `game_over`), no invalid actions slip through the mask |
| `test_native_bridge.py` | Rust-native obs/mask/step parity vs the Python reference implementations |
| `test_reseeding.py` | Consecutive resets play different games; same seed reproduces the same sequence |
| `test_league_and_rewards.py` | Opponent-pool validation/sampling/weights, shaping scale + anneal schedule, placement reward matches finish rank, league pool construction |

---

## Code map

| Path | Purpose |
|---|---|
| `crates/powergrid-py/Cargo.toml` | PyO3 crate manifest (pyo3 0.28, cdylib) |
| `crates/powergrid-py/src/lib.rs` | `Game` class: native obs/mask/step methods (`observation`, `action_mask`, `apply_action_id`, `step_vs_bots`), `bot_decide_id`, `copy`, etc. |
| `crates/powergrid-bot-strategy/src/macro_actions.rs` | The 26-macro action space: expansion, legality + dedup, teacher labels |
| `python/pyproject.toml` | Python package metadata (hatchling build backend) |
| `python/Makefile` | `make develop` = build Rust + install Python |
| `python/TRAINING.md` | Step-by-step training runbook |
| `python/src/powergrid_env/constants.py` | Macro id constants (mirrors `macro_actions`), CITY_IDS, normalisation denominators |
| `python/src/powergrid_env/encoding.py` | `encode_observation` — the Python reference obs implementation used by the parity tests (includes the routing Dijkstra) |
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
