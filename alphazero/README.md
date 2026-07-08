# AlphaZero for Power Grid

MCTS-guided self-play training, as an alternative to the PettingZoo+PPO stack
in `python/` (which repeatedly struggled with entropy collapse and brittle
reward shaping). Fixed at **4 players**. Reuses the Rust game engine and the
454-dim observation / 94-action encoding via `powergrid_py` and
`powergrid_env.constants` — no duplicated game logic.

Structured like [alpha-zero-general](https://github.com/suragnair/alpha-zero-general)
(Game adapter / NNet wrapper / MCTS / Coach), adapted for:
- **Action masking** (94-action space, mostly illegal at any given state).
- **4-player value** (a value *vector*, not a single zero-sum scalar).
- **Perfect-information search on the full seeded engine state.** MCTS forks
  the real `GameState` (including the hidden deck order and opponent money)
  to explore the tree — but the *network* is only ever shown the masked
  `observation()`, the same thing a real seated player would see. Search
  cheats; the trained policy doesn't.

## Setup

Uses the existing `python/` venv (already has `powergrid_py`, `powergrid_env`,
torch via the `train` extras):

```bash
cd python
make develop   # builds the Rust extension + installs powergrid_env, if not already done
cd ..
```

Then activate the venv so `python` resolves to that interpreter. Every command
below is run from the **repo root** with the venv active, as a module (the
package isn't pip-installed):

```bash
. python/.venv/bin/activate
python -m alphazero.train --iters 1 --episodes 2 --sims 10 --end-game-cities 5
```

## Training

The recommended pipeline is two phases: behavior-clone the hard heuristic bot
for a warm start, then AlphaZero-finetune on top. Everything runs at the
**rulebook** end-game trigger (17 cities for 4p) — the short-game curriculum
was measured to be *counterproductive* (bots are harder to beat in short games,
and self-play there never learns competitive tempo), so it's off by default.

```bash
# Smoke test: tiny, fast, just exercises the full loop (2 workers, both
# opponent mixes on).
python -m alphazero.train --iters 2 --episodes 4 --sims 10 --eval-games 4 \
    --workers 2 --vs-bot-fraction 0.25 --vs-past-fraction 0.25 \
    --run-dir alphazero/runs/smoke

# --- Phase 1: behavior-clone the hard bot (cheap: minutes) -------------------
# DIAGNOSTIC GATE: expect win-vs-normal well above 0% (the hard bot itself
# wins ~33% as seat 0). If the clone is ~0%, STOP — that points at a
# capacity/observation problem to fix before spending AZ compute.
python -m alphazero.pretrain --games 600 --epochs 30 --eval-games 100 \
    --run-dir alphazero/runs/clone2

# --- Phase 2: AlphaZero finetune from the clone -----------------------------
# Rulebook, 200 sims, anchor (vs hard bot) + league (vs past checkpoints)
# episodes, low finetune LR, parallel workers. ~5-10 min/iter with 8 workers.
python -m alphazero.train --iters 100 --episodes 16 --sims 200 --workers 8 \
    --vs-bot-fraction 0.3 --vs-bot-difficulty hard --vs-past-fraction 0.2 \
    --lr 1e-4 --eval-games 100 \
    --resume alphazero/runs/clone2/cloned.pt \
    --run-dir alphazero/runs/az-ft1

# Resume/continue az-ft1 later (runs 50 MORE iterations, continuing numbering):
python -m alphazero.train --iters 50 --episodes 16 --sims 200 --workers 8 \
    --vs-bot-fraction 0.3 --vs-past-fraction 0.2 --lr 1e-4 --eval-games 100 \
    --resume alphazero/runs/az-ft1/best.pt --run-dir alphazero/runs/az-ft1
```

Each iteration: self-play `--episodes` games with MCTS (`--sims` sims/move),
farmed across `--workers` processes; a mix of pure self-play, vs-hard-bot
anchor episodes (`--vs-bot-fraction`), and vs-past-checkpoint league episodes
(`--vs-past-fraction`). Then train `--train-batches` minibatches sampled from a
replay **window** of the last `--buffer-iters` iterations, evaluate win rate vs
`--eval-bot-difficulty` bots (net-only greedy — the exported artifact — unless
`--eval-num-sims > 0`), checkpoint to `--run-dir/iter_NNNN.pt`, and update
`best.pt` on a new best.

`--iters` is the number of iterations to run *this invocation*. Run
bookkeeping (`last_iter`, `best_win_rate`, curriculum state) is saved to
`--run-dir/coach_state.json`; pointing `--run-dir` at an existing run continues
its iteration numbering and won't clobber a better `best.pt`. Starting a
*fresh* run into a dir that already has checkpoints (but no `coach_state.json`)
errors out rather than overwriting. The replay buffer is not persisted — a
resumed run refills it over the first `--buffer-iters` iterations.

Checkpoints/metrics live under `alphazero/runs/` (gitignored — same convention
as `python/runs/`).

### Reading the metrics

**`eval/win_rate` is the KPI.** The reference bar: the hard heuristic bot in
seat 0 wins ~33% vs three normal bots (measured), so a net beating that is
genuinely strong. Policy/value **loss are *not* progress indicators** — the
targets are non-stationary (self-play distribution and value labels shift every
iteration), so loss drifting up while win rate rises is normal and fine. The
failure signature to watch for is *rising loss with flat/declining win rate*.

## Monitoring (TensorBoard)

Every iteration's metrics (`win_rate`, `best_win_rate`, `policy_loss`,
`value_loss`, `end_game_cities`, buffer/example counts, `elapsed_s`) are written
both to `<run-dir>/metrics.csv` and to TensorBoard event files under
`<run-dir>/tb/`. Point TensorBoard at the top-level `runs/` dir to see every run
side by side:

```bash
tensorboard --logdir alphazero/runs
```

Then open http://localhost:6006. Each run appears as its own series (named by
its run directory).

## Export to the Rust Expert bot

The policy path (`PGNet.policy_state_dict()`) is laid out under the same key
names sb3's MaskablePPO uses, so it serializes to the existing PGRLPOL1 binary
format with **no Rust changes**:

```bash
python -m alphazero.export \
    --checkpoint alphazero/runs/curriculum1/best.pt \
    --out assets/policies/expert.bin \
    --golden assets/policies/expert.golden.json
```

Then run the Rust golden-logits parity test
(`crates/powergrid-bot-strategy/src/policy.rs`) to confirm the embedded
weights match. The value head is training-only and is never exported.

## Tests

```bash
python -m pytest alphazero/tests -v
```

## Module map

- `config.py` — `AZConfig`: MCTS/network/training/curriculum hyperparameters.
- `game.py` — `PowerGridGame`: AlphaZero adapter over `powergrid_py.Game`
  (fork/apply/observation/mask/outcome), plus the perspective-relative
  value-vector helpers (`relative_order`, `to_relative_vector`, `to_absolute_dict`).
- `network.py` — `PGNet` (trunk + policy head + value head) and `NNetWrapper`
  (predict/train/save/load).
- `mcts.py` — `MCTS`/`Node`: node-based (forked-engine) multiplayer masked PUCT
  search, with FPU reduction for unvisited children and a forced-move shortcut.
- `selfplay.py` — `play_episode` (pure self-play), `play_episode_vs_bots`
  (learner vs heuristic bots), `play_episode_vs_net` (learner vs a past
  checkpoint); all share one MCTS loop and skip forced moves. Also the
  `multiprocessing` worker entry points used by the coach.
- `coach.py` — `Coach`: the self-play -> train -> eval -> checkpoint loop.
  Windowed replay (`buffer_iters`), fixed per-iteration training budget
  (`train_batches`), parallel self-play (`num_workers`), league opponents, and
  `coach_state.json`-based resume.
- `arena.py` — win-rate evaluation: net vs. Rust heuristic bots, or net vs. net.
- `train.py` — CLI entry point.
- `export.py` — checkpoint -> PGRLPOL1 binary for the Rust Expert bot.
