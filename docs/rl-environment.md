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
AECEnv._mask()             ──►   Game.action_mask(actor) → np.uint8[29]
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

**Frozen-opponent self-play** (`scripts/train_selfplay.py`): the env is created with `bot_difficulty="policy"`; `OpponentSnapshotCallback` periodically serializes the current policy network (`powergrid_env.export.policy_state_dict_to_bytes`, the same `PGRLPOL3` format the Rust Expert bot consumes) and pushes it to the envs via `set_opponent_policy(bytes)`. Each env loads the snapshot into Rust at its next reset, so opponents improve alongside the learner while rewards stay correctly attributed to the learner's own moves. Until the first snapshot arrives (and, with `bot_mix=p`, for a random share of episodes) the env falls back to `"hard"` heuristic bots.

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
        mask = info["action_mask"]          # np.ndarray of shape (29,), dtype int8
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
- `action_space` → `Discrete(29)` — macro actions, see table below

**Rewards:** sparse — `+1.0` to winner, `-1.0` to all others at game end; `0.0` every other step.

---

## Action encoding — macro actions (N = 29)

The policy does **not** choose primitive game moves. It chooses one complete **phase-plan per turn** from a fixed menu of 29 macros; each expands natively into a short primitive sequence the engine already accepts as a whole-turn batch (`BuildCities`, `BuyResourceBatch`). A game is ~50 macro decisions instead of ~600 primitive ones, which removes the compounding-error tax that capped every earlier learner.

> **History:** until the Phase-2 rebuild this was a 94-id *primitive* space (`BuildCity` per city, `BuyResources` per unit, etc.). It was removed, not extended. Any policy trained against it is incompatible.

| Id | Macro | Notes |
|---|---|---|
| 0–5 | `NOMINATE` market slot 0–5 | Auction with no standing bid: nominate an `actual` market plant |
| 6 | `AUCTION_PASS` | Drop out of / decline the auction (both auction sub-phases) |
| 7 | `AUCTION_RAISE` | Raise +1 over the standing bid (English-auction convention). No jump bids — self-play with a ±50 raise range learned large non-strategic jumps |
| 8–14 | `BUILD_COUNT_BASE + n`, n = 0…6 | Build the **n cheapest** reachable cities you can pay for. `n = 0` is `DoneBuilding`. Cash is the only limit — no reserve is withheld |
| 15 | `BUILD_DEFAULT` | Whatever the champion `hard` heuristic would build, **bit-exactly** (Gate 0). Last id so dedup prefers the explicit count; in practice a count always reproduces it, so this is dead weight kept as a safety valve for >6-city plans |
| 16 | `BUY_DONE` | End the buy turn having bought nothing more |
| 17 | `BUY_DEFAULT` | The heuristic's whole batch in one shot, bit-exactly (Gate 0), ending the turn |
| 18–21 | `BUY_FUEL1_BASE + r` | Top **one fuel** up to **one round** of the rack's demand: coal / oil / gas / uranium |
| 22–25 | `BUY_FUEL2_BASE + r` | Same fuel order, up to **two rounds** — the storage ceiling |
| 26–28 | `DISCARD_PLANT` slot 0–2 | Index into the player's plants sorted by number; forced when winning a 4th plant |

**Buy is the one multi-decision phase.** `BUY_FUEL*` presses emit the additive `Action::BuyResources` primitive, which does *not* end the buy turn, so they compose — "a round of coal, two rounds of uranium, done". A press that could buy nothing (already at target, unaffordable, out of stock, or the rack does not burn that fuel) is masked out, so the turn can always be ended but never spun on; the worst case measured under a press-spamming policy is 4 presses. `BUY_DEFAULT` remains a single-decision escape hatch, which is why Gate 0 is unaffected. Everything else in the space is still exactly one decision per turn.

**The two menus are shaped differently on purpose.** Cities are interchangeable — one more city is one more income step wherever it is — so the build menu is an absolute *count*. Fuel is not interchangeable across types but *is* fungible within a type, so the buy menu is **per fuel**: two coal plants draw on one coal stock and which plant burns which unit is settled at firing time, so "fuel plant A but not plant B" is not a distinct game state. Per-fuel is the finest granularity that names a real decision, and two rounds is the ceiling because `can_add_resource` caps storage at `cost * 2`.

**Auto-resolved phases.** Powering (`Bureaucracy`), the hybrid gas/oil split (`PowerCitiesFuel`) and hybrid-slot overflow (`DiscardResource`) carry no strategic content the menu could express, so `macro_actions::resolve_auto_phases` settles them with the heuristic. They never consume a policy decision and have no macro ids. Powering was measured out: the teacher fired the optimal subset in **100%** of decisions and the only alternative the menu ever offered ("power nothing") was legal everywhere and correct nowhere — a trap that also cost ~9 of a seat's ~52 decisions per game.

**Masking and dedup.** `info["action_mask"]` comes from `macro_actions::legal_macros`, which validates each macro by trial application on a cloned state **and drops duplicates** — a macro whose primitive expansion equals a lower-id macro's is marked illegal. So `BUILD_3` collapses onto `BUILD_2` when only two cities are affordable. Dedup is why id ordering matters: counts sit *below* `BUILD_DEFAULT` so that "build 2" is always id 10, never sometimes-10-sometimes-15 depending on whether the heuristic happened to agree.

**Imitation labels.** `Game.bot_decide_id(actor, difficulty)` returns the macro id the champion heuristic would play (`macro_actions::teacher_macro_id`) — always the id that *survives dedup*, so the label is never an illegal action. This is the behavior-cloning / DAgger target.

---

## Phase-by-phase decision analysis

What the policy actually gets to decide, per phase, and how much of the menu is live in real play.

> **Method.** All figures below are measured over 20 seeded 4-player games between champion `hard` heuristic bots (3,402 macro decisions), sampling `legal_macros` and `teacher_macro_id` at every decision point. Measured 2026-07-25, on the final action space. Two caveats: (1) "legal %" is a property of the *state distribution the heuristic generates* — a differently-behaved policy visits different states and would see different numbers; (2) "teacher %" describes the heuristic, which is the behavior-cloning target, not a ceiling on good play.

### Decision budget

| Phase | Decisions / game (4 seats) | Live options (avg) | Menu size |
|---|---:|---:|---:|
| Auction — nominate | 37.4 | 5.01 | 7 |
| Auction — bidding | 45.1 | 2.00 | 2 |
| BuyResources | 37.4 | **5.89** | 7 |
| BuildCities | 37.4 | 2.99 | 8 |
| DiscardPlant | 12.8 | 3.00 | 3 |

**≈170 macro decisions per game, ≈43 per seat** when the buy phase is settled in one decision — which is what the teacher does, so a behavior clone starts at that length. A policy that composes per-fuel presses pays more: up to 4 extra decisions per buy turn (the worst case measured under a press-spamming policy), so ~43 to ~80 per seat depending on how much it uses them. The cost is opt-in, which is the point. (~600 was the old primitive count, per seat.) Powering used to add ~9 more steps per seat with one real option; auto-resolving it shortened every episode by ~18%.

### Auction — the richest decision, split awkwardly in two

The auction is the only phase with a genuinely wide menu, and the only one where the policy makes *two different kinds* of decision.

**Nominating** (7 ids: 6 market slots + pass) is where plant choice happens. The four `actual` slots are legal ~92–99% of the time; slots 4–5 only exist in Step 3, when the market widens to six, hence their ~13–15%.

| | NOM_0 | NOM_1 | NOM_2 | NOM_3 | NOM_4 | NOM_5 | PASS |
|---|---:|---:|---:|---:|---:|---:|---:|
| legal | 98.7% | 97.7% | 95.6% | 92.1% | 14.6% | 12.7% | 89.3% |
| teacher picks | 4.1% | 7.0% | 15.2% | 30.5% | 4.0% | 5.3% | 33.8% |

The teacher's spread here is healthy — it declines a third of the time and otherwise favours the higher-numbered (more capable) plants. This is the one phase where behavior cloning transfers real strategy.

**Bidding** (2 ids: `+1` or pass) is where the menu stops carrying the decision:

- The teacher **passes 99.7%** of bidding decisions and raises 0.3%.
- Only **30.1%** of the 279 auctions see even one raise; the average auction sees **0.39** raises total.
- Of the 110 raises played, 102 exceeded the plant's base price — by **1.1 Elektro on average**. Plants essentially sell at their printed number.

Two consequences worth knowing before training on this:

1. **Behavior cloning teaches "never raise."** With a constant label on 99.7% of bidding states, a clone learns pass as its bidding policy, and self-play must discover contested bidding entirely on its own — starting from a maximally biased prior. This is a plausible contributor to the end-game weakness: a policy that cannot win a plant fight cannot build the capacity to close a game.
2. **It is the one phase that is not one-decision-per-turn.** Bidding 10 over base costs 10 sequential policy calls. That is a faithful English auction — each call really is "am I still willing at this price?" — and the alternative was measured worse (a ±50 jump-bid range taught large, non-strategic raises), so it is kept deliberately. But the cost is real, and it compounds with the constant teacher above.

There is also a structural split: *which plant to nominate* and *how much it is worth* are separate decisions, made at different times, with other players' turns in between. The policy cannot express "nominate slot 3, willing to go to 25."

### Resources — per-fuel levels

| | DONE | DEFAULT | coal ×1 | oil ×1 | gas ×1 | uran ×1 | coal ×2 | oil ×2 | gas ×2 | uran ×2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| legal | 100% | 100% | 56.3% | 40.4% | 49.2% | 15.8% | 64.0% | 45.9% | 52.1% | 17.2% |
| teacher | 0% | **100%** | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

**5.41 live options.** Legality tracks whether the rack burns that fuel and can still store and afford it — uranium is lowest at ~16% because few racks burn it, coal highest at ~64%. The teacher always plays `BUY_DEFAULT`, which is the whole point of keeping it: the heuristic's plan stays a single bit-exact macro, so **Gate 0 is unchanged** by the finer menu, and the presses are an alternative the policy may compose rather than something the teacher has to speak in.

The decision this encodes is, for each fuel the rack burns: **skip it, buy a round, or buy two**. That is the complete enumeration — `can_add_resource` caps storage at `cost * 2` per plant, so a third round is unbuyable. It also enables the two motives that justify deviating from the heuristic's plan, without encoding *when* to do either: buy two rounds of a fuel whose price track is climbing or that opponents are about to strip, and skip a fuel whose cash is worth more as cities.

> **Two earlier attempts, and why they failed.** The original menu (`BUY_NOTHING`, `BUY_DEFAULT`, `BUY_STOCKPILE2/3`, `BUY_DENIAL`) offered 2.39 live options. `STOCKPILE3` was dead by construction — its 3-round target clamps to the same 2-round storage cap `STOCKPILE2` hits — and `STOCKPILE2` itself could differ from the default in only 2.1% of decisions because the heuristic's stockpile pass sits behind a 140-Elektro reserve. `BUY_DENIAL` was legal 36.4% but targeted the priciest fuel, uranium 59.4% of the time, which the actor could not store *at all* in 61.8% of decisions.
>
> It was then replaced by an aggregate ±k ladder around "one complete set", which fixed reachability (5.89 live options) but encoded a decision nobody makes. Buying one unit short of a complete set **cost cities in 32.4% of measured decisions — 1.93 cities on average, in exchange for one unit of fuel** — and changed nothing in the other 68%, where the shortfall came out of carry-over surplus. Fuel only pays in whole plant-sized chunks, so a unit-granular offset is either a blunder or a rounding difference.

### Cities — an absolute count ladder

| | BUILD_0 | BUILD_1 | BUILD_2 | BUILD_3 | BUILD_4 | BUILD_5 | BUILD_6 | DEFAULT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| legal | 100% | 89.0% | 66.7% | 29.7% | 9.2% | 3.2% | 1.1% | 0.0% |
| teacher picks | 12.3% | 26.1% | **42.8%** | 16.0% | 2.7% | 0.1% | 0.0% | 0.0% |

The menu now spans the decision monotonically, the teacher's label is genuinely varied, and the ladder tops out well above what anyone reaches (the teacher never exceeds 5). `BUILD_DEFAULT` is never legal — a count always reproduces the heuristic exactly — so it costs one id as insurance against a future profile whose ordering isn't cheapest-first.

The count is the strategic axis for two reasons the observation can support: more cities means more income but an **earlier** turn-order position, which is a disadvantage (both buying and building run in reverse player order); and building past powering headroom is how the end-game trigger is reached. Because `BUILD_n` is limited only by cash, the end-game push is now expressible — the previous menu could not express it at all.

**Accepted limitation:** *which* cities is not a decision. Greedy cheapest-first can expand in a strategically poor direction (into a corner, or away from a region that opens up in Step 2), and blocking is unavailable. `BUILD_BLOCK` was dropped because CMA-ES independently drove the heuristic's `block_weight` to zero; if a trained policy later shows it is losing games to positional network mistakes, this is the axis to re-add.

### Powering and the auto-resolved phases — no decision at all

Powering used to be a macro pair, and measured as a pure trap: `POWER_OPTIMAL` was the teacher's choice in **100%** of decisions while `POWER_NOTHING` was legal in 100% and correct in none. It is now auto-resolved with the heuristic, along with the hybrid fuel split and resource discard.

The genuine decision the engine supports but the macro layer never exposed is powering a **subset**: `Action::PowerCities { plant_numbers }` accepts any set of plants, so declining to fire an expensive uranium plant when the income gain is below the fuel's value is a legal, occasionally correct play. All-or-nothing could not express it. If a trained policy later shows it is losing games to fuel mismanagement, a proper partial-power ladder is the thing to add — not the old binary.

### DiscardPlant — small but real

| | slot 0 | slot 1 | slot 2 |
|---|---:|---:|---:|
| legal | 100% | 100% | 100% |
| teacher picks | 87.5% | 11.4% | 1.2% |

Forced when a 4th plant is won (12.8 decisions per game). All three options are always available and the teacher's choice varies, so this small menu is already well formed: drop the lowest-numbered plant most of the time, but not always.

### Cross-cutting: where the teacher still can't teach

| Decision | Teacher's most common label | Share |
|---|---|---:|
| Auction — nominate | (spread across 7) | 33.8% |
| **Auction — bidding** | `AUCTION_PASS` | **99.7%** |
| BuyResources | `BUY_DEFAULT` (a complete set) | 94.3% |
| BuildCities | (spread across 6) | 42.8% |
| DiscardPlant | slot 0 | 87.5% |

Every menu now has reachable, meaningful alternatives — which was the real problem, more than label entropy. A behavior clone learns a strong prior everywhere and has somewhere sane to explore from in every phase.

The one place that is still structurally awkward is **bidding**. It is a binary "stay in at +1 or drop out", so a contested auction costs one policy call per Elektro — bidding 10 over base is 10 sequential decisions. That is a faithful model of an English auction (each call really is "am I still willing at this price?"), and the alternative was measured worse: a ±50 jump-bid range was removed for teaching large, non-strategic raises. But combined with a teacher that passes 99.7% of the time, it means contested bidding is the one skill neither imitation nor a short rollout is likely to produce, and a policy that cannot win a plant fight cannot build the capacity to close a game.

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
| `crates/powergrid-bot-strategy/src/macro_actions.rs` | The 29-macro action space: expansion, legality + dedup, teacher labels |
| `python/pyproject.toml` | Python package metadata (hatchling build backend) |
| `python/Makefile` | `make develop` = build Rust + install Python |
| `python/TRAINING.md` | Step-by-step training runbook |
| `python/src/powergrid_env/constants.py` | Macro id constants (mirrors `macro_actions`), CITY_IDS, normalisation denominators |
| `python/src/powergrid_env/encoding.py` | `encode_observation` — the Python reference obs implementation used by the parity tests (includes the routing Dijkstra) |
| `python/src/powergrid_env/env.py` | `PowerGridAECEnv` (PettingZoo AEC) |
| `python/src/powergrid_env/single_agent.py` | `PowerGridSingleAgentEnv` (Gymnasium, vs Rust bots or frozen policy snapshots, native fast path) |
| `python/src/powergrid_env/export.py` | `policy_state_dict_to_bytes` / `policy_bytes_to_state_dict` — PGRLPOL3 policy (de)serialization: export script, self-play snapshots, and the behavior-clone warm start (`train_selfplay.py --init-policy-from`) |
| `python/src/powergrid_env/policies/` | `RandomPolicy`, `RustBotPolicy` |
| `python/scripts/train_selfplay.py` | Frozen-opponent self-play MaskablePPO training |
| `python/scripts/train_vs_bots.py` | Single-agent MaskablePPO vs Rust bots |
| `python/scripts/evaluate.py` | Win-rate evaluation of a checkpoint vs bots |
| `python/scripts/run_report.py` | Status report for a run dir: checkpoints, live process, TB metrics, health flags |
| `python/scripts/play_game.py` | Rollout viewer |
| `python/tests/` | Encoding, API conformance, parity, reseeding, and random-play tests |
