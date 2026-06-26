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
```

All commands below are run from the **repo root**, using that venv's
interpreter, as a module (the package isn't pip-installed):

```bash
python/.venv/bin/python -m alphazero.train --iters 1 --episodes 2 --sims 10 --end-game-cities 5
```

## Training

```bash
# Smoke test: tiny, fast, just exercises the full loop.
python/.venv/bin/python -m alphazero.train --iters 1 --episodes 2 --sims 10 --end-game-cities 5

# Real run: curriculum from a short trigger up to the rulebook 17, ramping
# every 5 iterations.
python/.venv/bin/python -m alphazero.train \
    --iters 200 --episodes 25 --sims 50 \
    --curriculum-start 5 --curriculum-step 2 --curriculum-every 5 \
    --run-dir alphazero/runs/curriculum1

# Resume from a checkpoint.
python/.venv/bin/python -m alphazero.train --resume alphazero/runs/curriculum1/best.pt \
    --run-dir alphazero/runs/curriculum1 ...
```

Each iteration: self-play `--episodes` games with MCTS (`--sims` simulations
per move), train on the replay buffer, evaluate win rate vs `--eval-bot-difficulty`
Rust heuristic bots (`python/scripts/evaluate.py`'s methodology), checkpoint to
`--run-dir/iter_NNNN.pt`, and update `--run-dir/best.pt` on a new best win rate.

Checkpoints/metrics live under `alphazero/runs/` (gitignored — same convention
as `python/runs/`).

## Export to the Rust Expert bot

The policy path (`PGNet.policy_state_dict()`) is laid out under the same key
names sb3's MaskablePPO uses, so it serializes to the existing PGRLPOL1 binary
format with **no Rust changes**:

```bash
python/.venv/bin/python -m alphazero.export \
    --checkpoint alphazero/runs/curriculum1/best.pt \
    --out assets/policies/expert.bin \
    --golden assets/policies/expert.golden.json
```

Then run the Rust golden-logits parity test
(`crates/powergrid-bot-strategy/src/policy.rs`) to confirm the embedded
weights match. The value head is training-only and is never exported.

## Tests

```bash
python/.venv/bin/python -m pytest alphazero/tests -v
```

## Module map

- `config.py` — `AZConfig`: MCTS/network/training/curriculum hyperparameters.
- `game.py` — `PowerGridGame`: AlphaZero adapter over `powergrid_py.Game`
  (fork/apply/observation/mask/outcome), plus the perspective-relative
  value-vector helpers (`relative_order`, `to_relative_vector`, `to_absolute_dict`).
- `network.py` — `PGNet` (trunk + policy head + value head) and `NNetWrapper`
  (predict/train/save/load).
- `mcts.py` — `MCTS`/`Node`: node-based (forked-engine) multiplayer masked PUCT search.
- `selfplay.py` — `play_episode`: one self-play game -> labeled training examples.
- `coach.py` — `Coach`: the self-play -> train -> eval -> checkpoint loop.
- `arena.py` — win-rate evaluation: net vs. Rust heuristic bots, or net vs. net.
- `train.py` — CLI entry point.
- `export.py` — checkpoint -> PGRLPOL1 binary for the Rust Expert bot.
