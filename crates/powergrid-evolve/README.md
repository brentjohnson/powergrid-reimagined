# powergrid-evolve

Offline **evolutionary tuner** (CMA-ES) for the heuristic bot's `BotProfile`
weights. This is Phase 1 of the "beat humans" plan (see `RL-TRAINING-JOURNAL.md`):
the hand-tuned `hard` profile is the strongest agent in the project (~34.5% seat-0
vs 3 normal bots), but its ~14 strategy weights were never actually searched.
This crate searches them.

It is a **training tool**, not part of the shipped server/client. It plays
thousands of fully-deterministic headless games per generation and optimizes the
weights to maximize the candidate seat's finish position.

## Why it can trust its measurements

Fitness is **paired**: every candidate in a generation plays the *same* set of
seeded games (common random numbers), with the candidate seat rotated across all
seats to remove position bias. Crucially, all bots play with **noise silenced**
(`temperature = 0`, `jitter = 0`), so a fixed game seed yields a bit-identical
game — the property the training journal found essential (jittered A/B has a
±5pp noise floor that inverts small effects). A determinism regression test
(`games::tests`) guards this, and it drove a real engine fix: the heuristic's
build-city sort now has a city-id tiebreak so tied cities no longer resolve by
`HashMap` iteration order.

## Run

```bash
cargo build --release -p powergrid-evolve

# Stage 1 — evolve vs the fixed `normal` lineup (the eval yardstick)
./target/release/powergrid-evolve \
  --opponents normal --pop 20 --games-per-eval 600 --seat-rotations 4 \
  --generations 200 --seed-block-rotate 10 --out-dir runs/evolve1

# Stage 2 — co-evolve vs a pool of earlier champions (avoids overfitting normal)
./target/release/powergrid-evolve \
  --resume runs/evolve1/checkpoint.json --opponents pool \
  --pool-dir runs/pool --generations 100 --out-dir runs/evolve2
```

`--help` lists all flags.

## Outputs (in `--out-dir`)

- `history.csv` — per generation: `sigma`, the distribution-mean win rate / rank
  value, the population-best win rate, elapsed seconds, and the 14 mean weights.
  Gen 0's mean row is exactly the shipped `hard` profile, so it reproduces the
  known baseline and anchors every later row.
- `best.toml` — the best distribution-mean so far, as a full `ProfileRegistry`
  (easy/normal kept from defaults so the yardstick is untouched; `hard` = `expert`
  = champion, with `hard`'s shipped jitter restored for in-game variety).
- `checkpoint.json` — full CMA-ES state; `--resume` continues from it.

## Deploying a champion

`best.toml` is a drop-in profile file. Point any binary at it with the
`BOT_PROFILES_FILE` env var (lobby, client, and the Python eval via
`powergrid_py` all honor it — read once at process start):

```bash
BOT_PROFILES_FILE=runs/evolve1/best.toml \
  python/.venv/bin/python python/scripts/evaluate_lineup.py \
  --player hard --player normal --player normal --player normal \
  --games 600 --seed 90001 --quiet
```

Use held-out seeds (≥90000, never seen in training) for an honest gate. To ship
permanently, paste the champion `hard`/`expert` weights into
`assets/bots/default.toml` and rebuild.

## Design notes

- `genome.rs` — the 14-weight ⇄ normalized-vector mapping (`x = 0` is `hard`).
- `games.rs` — headless deterministic game + parallel paired fitness.
- `cmaes.rs` — a self-contained `(μ/μ_w, λ)`-CMA-ES (hand-rolled Jacobi
  eigensolver; no `nalgebra`/BLAS).
- `main.rs` — CLI, schedule building, the generation loop, output writing.
