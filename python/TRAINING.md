# Training Guide

Step-by-step instructions for training Power Grid RL agents, resuming interrupted
runs, monitoring progress, and evaluating results. All commands run from the
`python/` directory. For environment internals (observation/action encoding,
architecture), see [docs/rl-environment.md](../docs/rl-environment.md).

---

## 1. One-time setup

```bash
cd python
make develop
```

This creates `.venv/`, compiles the Rust extension (`maturin develop --release`),
and installs the `powergrid_env` package plus training dependencies
(stable-baselines3, sb3-contrib, torch, tensorboard).

Verify the install:

```bash
make test          # full test suite, should pass in a few seconds
```

> **⚠ Rebuild after Rust changes.** The extension is a compiled snapshot of the
> game rules. After *any* change to `powergrid-core`, `powergrid-bot-strategy`,
> or `powergrid-py`, run `make develop` again. A stale build silently plays by
> old rules — symptoms include `KeyError`s in encoding (renamed fields) or
> mysteriously failing parity tests. When in doubt, rebuild and run `make test`.

All commands below assume the venv's Python:

```bash
.venv/bin/python scripts/...
# or: source .venv/bin/activate
```

---

## 2. Training vs bots (recommended starting point)

The learner occupies one seat; the Rust strategy bots fill the rest. This gives
a stable, non-moving opponent and converges much faster than self-play early on.

```bash
.venv/bin/python scripts/train_vs_bots.py \
    --num-players 4 \
    --bot-difficulty normal \
    --num-envs 8 \
    --total-timesteps 2_000_000 \
    --run-dir runs/vs_bots
```

Key arguments (defaults in parentheses):

| Flag | Meaning |
|---|---|
| `--bot-difficulty` (normal) | `easy` / `normal` / `hard` opponents |
| `--num-envs` (8) | Parallel envs in a `DummyVecEnv`. 4–8 is the sweet spot; the Rust steps are too fast for subprocess vectorisation to pay off |
| `--total-timesteps` (500 000) | Learner steps for this invocation |
| `--reward-shaping` (on) | Small per-step bonus ∝ cities owned; disable with `--no-reward-shaping` for pure win/loss reward |
| `--save-freq` (50 000) | Checkpoint every N steps *per env*; `0` disables |
| `--eval-freq` (25 000) | Win-rate eval vs bots every N steps *per env*; `0` disables |
| `--device` (auto) | `cpu` / `cuda`; `auto` picks the GPU if available. For this MLP-sized policy, CPU is often as fast |
| `--seed` (0) | Reproducible env seed streams and PPO init |

While running, SB3 prints a table every iteration. The numbers that matter:
`rollout/ep_rew_mean` (should rise toward +1), `fps`, and after each eval pass
`eval/mean_reward`.

### Artifacts in `--run-dir`

```
runs/vs_bots/
├── ckpt_50000_steps.zip     periodic checkpoints (CheckpointCallback)
├── ckpt_100000_steps.zip
├── best_model.zip           best eval mean reward so far (MaskableEvalCallback)
├── final_model.zip          written when the run completes normally
└── tb/                      TensorBoard event files
```

`best_model.zip` is usually the one you want — `final_model.zip` is just the
last state, which may be worse than the best intermediate policy.

---

## 3. Self-play training

All seats share one policy; the env returns each successive actor's observation
in a single fast Rust call. Use this after a vs-bots policy plateaus, or to
train without bot bias.

```bash
.venv/bin/python scripts/train_selfplay.py \
    --num-players 4 \
    --num-envs 8 \
    --total-timesteps 5_000_000 \
    --run-dir runs/selfplay
```

Self-play rewards are sparse (+1/−1 only on the final move of a game), so it
needs more timesteps than vs-bots training. Progress is measured externally:
the eval callback plays the policy *against normal bots* every `--eval-freq`
steps, so `eval/mean_reward` is comparable between the two scripts.

PPO hyperparameters are CPU-tuned in both scripts (`n_steps=512`,
`batch_size=512`, `n_epochs=4`) — fewer, larger mini-batch updates per rollout
than SB3 defaults, which is ~6× faster per iteration at no practical quality
loss for this task.

---

## 4. Resuming an interrupted run

Both training scripts accept `--resume-from` with a checkpoint path **without
the `.zip` suffix**:

```bash
# Find the newest checkpoint:
ls -t runs/vs_bots/ckpt_*.zip | head -1

# Resume from it, adding another 1M steps:
.venv/bin/python scripts/train_vs_bots.py \
    --resume-from runs/vs_bots/ckpt_450000_steps \
    --total-timesteps 1_000_000 \
    --run-dir runs/vs_bots
```

Notes:

- The script prints `Resumed from ... at N timesteps`. The internal step counter
  continues (`reset_num_timesteps=False`), so TensorBoard curves stay continuous
  and new checkpoints are numbered from where the old run stopped.
- `--total-timesteps` counts steps for *this invocation*, on top of the restored
  counter.
- Model hyperparameters (network size, n_steps, batch size) are restored from
  the checkpoint; command-line hyperparameter flags only affect *fresh* runs.
- Keep the same `--run-dir` so checkpoints and TensorBoard logs accumulate in
  one place. Keep the same `--num-envs`; PPO buffers are sized per env.
- You can resume from `best_model`, `final_model`, or any `ckpt_*` file — they
  are all complete `MaskablePPO` saves.

---

## 5. Monitoring progress

### TensorBoard

```bash
.venv/bin/tensorboard --logdir runs
```

Open http://localhost:6006. Curves worth watching:

| Metric | Healthy signal |
|---|---|
| `rollout/ep_rew_mean` | Rising. With shaping on it can exceed +1; without shaping it approaches +1 as the win rate climbs |
| `eval/mean_reward` | The real score: shaping-free reward vs bots in [−1, 1]. Win rate ≈ `(mean_reward + 1) / 2` |
| `train/explained_variance` | Should climb toward ~0.5+; near 0 long-term means the value function is learning nothing |
| `train/approx_kl`, `clip_fraction` | Spikes indicate too-aggressive updates |
| `rollout/ep_len_mean` | Power Grid games are roughly 70–250 learner steps; sustained growth can mean stalling play |

### Win-rate evaluation

For a precise, repeatable measurement (the eval callback only plays ~20 games):

```bash
.venv/bin/python scripts/evaluate.py \
    --model runs/vs_bots/best_model \
    --games 100 \
    --bot-difficulty normal
```

Prints per-game results then a summary: win rate, average cities, average game
length. Random play wins ≈0% against normal bots; meaningful progress shows up
as a rising win rate across checkpoints. Evaluate against `--bot-difficulty
hard` once the normal bots are beaten.

By default actions are sampled stochastically. `--deterministic` plays the
greedy policy, but beware: an undertrained greedy policy can pass forever and
stall games (they count as losses via `--max-steps`).

### Watching a game

```bash
.venv/bin/python scripts/play_game.py --model runs/vs_bots/best_model --render
```

Renders each state to the terminal and prints the event log at the end.

---

## 6. Typical workflow

```bash
cd python
make develop                                          # once, and after Rust changes
.venv/bin/python scripts/train_vs_bots.py \
    --total-timesteps 2_000_000 --run-dir runs/vs_bots &
.venv/bin/tensorboard --logdir runs                   # watch eval/mean_reward
# ...interrupted? resume:
.venv/bin/python scripts/train_vs_bots.py \
    --resume-from "$(ls -t runs/vs_bots/ckpt_*.zip | head -1 | sed 's/\.zip$//')" \
    --total-timesteps 1_000_000 --run-dir runs/vs_bots
# measure:
.venv/bin/python scripts/evaluate.py --model runs/vs_bots/best_model --games 100
# then graduate to self-play:
.venv/bin/python scripts/train_selfplay.py \
    --total-timesteps 5_000_000 --run-dir runs/selfplay
```

---

## 7. Troubleshooting

**`KeyError: 'gas'` (or similar) in encoding, or parity test failures** —
the compiled extension is stale. Run `make develop`, then `make test`.

**`ValueError: Observation spaces do not match` when resuming or evaluating** —
the checkpoint was trained with a different observation/action layout (e.g.
before a map or encoding change). Old checkpoints cannot be migrated; retrain.
Current layout: obs 454, actions 143 (USA map, 49 cities).

**Eval pass seems to hang** — eval episodes are stochastic and time-limited
(2000 steps) precisely because deterministic pass-everything policies stall
games. If you changed that, change it back.

**Training is slow** — check the printed `fps`. Expect thousands of steps/s
with `--num-envs 8` on CPU. If far below: make sure the extension was built
with `--release` (the Makefile does), and don't use `SubprocVecEnv` — IPC costs
more than the Rust step.

**`game did not finish` in custom rollouts** — random/weak policies can
legally stall a game (everyone passes forever; nothing in the rules forces
progress). Always cap custom rollout loops with a step limit.

**Old checkpoints in `runs/`** — checkpoints from before 2026-06 used the
Germany map (obs 404/actions 136) and won't load against the current build.
