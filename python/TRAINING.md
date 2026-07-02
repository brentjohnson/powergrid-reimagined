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
| `--num-players` (4) | Total seats (learner + bots) |
| `--learner-seat` (0) | Which seat index the learner occupies |
| `--bot-difficulty` (normal) | `easy` / `normal` / `hard` opponents |
| `--num-envs` (8) | Parallel envs in a `DummyVecEnv`. 4–8 is the sweet spot; the Rust steps are too fast for subprocess vectorisation to pay off |
| `--total-timesteps` (500 000) | Learner steps for this invocation |
| `--seed` (0) | Reproducible env seed streams and PPO init |
| `--device` (auto) | `cpu` / `cuda`; `auto` picks the GPU if available. For this MLP-sized policy, CPU is often as fast |
| `--run-dir` (runs/vs_bots) | Directory for checkpoints, `best_model.zip`, and TensorBoard logs |
| `--resume-from` | Path to a checkpoint `.zip` (without the suffix) to continue from; omit for a fresh run |
| `--net-width` (128) | Hidden width of the two equal-width MLP layers, fresh runs only (ignored with `--resume-from`). 128 is the default; 64 = the old SB3 default / pre-2026-06 checkpoints. A new width can't resume an old-width checkpoint |
| `--save-freq` (50 000) | Checkpoint every N steps *per env*; `0` disables |
| `--eval-freq` (25 000) | Win-rate eval vs bots every N steps *per env*; `0` disables |
| `--eval-episodes` (20) | Games played per eval pass |
| `--reward-shaping` (on) | Per-round bonus ∝ cities powered, granted when the learner's powering resolves — analogous to income, so it values plants, resources, and cities in the game's own balance. Disable with `--no-reward-shaping` for pure win/loss reward |
| `--anneal-shaping-steps` (0) | Linearly anneal the shaping bonus to zero over this many timesteps — shaping should bootstrap, not steer the final policy. 0 = constant shaping. Resume-safe (scale derives from the step counter) |
| `--terminal-reward` (winloss) | `winloss` = +1/−1. `placement` = final rank mapped onto [−1, +1] (4p: +1 / +⅓ / −⅓ / −1) — denser signal that values 2nd place over last. Eval always scores winloss, so `eval/mean_reward` stays comparable |
| `--end-game-cities` | Pin the end-game city trigger to a fixed value for training and eval (omit to use the rulebook number) |
| `--ent-coef` (0.03) | PPO entropy bonus coefficient; re-applied on `--resume-from` to prevent policy collapse |

While running, SB3 prints a table every iteration. The numbers that matter:
`rollout/ep_rew_mean` (should rise toward +1), `fps`, and after each eval pass
`eval/mean_reward`.

### Artifacts in `--run-dir`

```
runs/vs_bots/
├── ckpt_50000_steps.zip     periodic checkpoints (CheckpointCallback)
├── ckpt_100000_steps.zip
├── best_model.zip           best eval mean reward so far (PersistentBestEvalCallback)
├── best_mean_reward.json    the eval score best_model.zip achieved
├── final_model.zip          written when the run completes normally
└── tb/                      TensorBoard event files
```

`best_model.zip` is usually the one you want — `final_model.zip` is just the
last state, which may be worse than the best intermediate policy.

---

## 3. Self-play training

League (population-based) self-play: the learner plays one seat; the other
seats are driven by an opponent sampled *per episode* from a league of frozen
snapshots of its own policy (run natively in Rust) plus heuristic hard bots.
Every `--snapshot-every` timesteps the current weights are frozen into
`<run-dir>/league/snap_<steps>.bin` and the envs' pool is rebuilt: the latest
snapshot, a few random older snapshots, and hard bots, mixed by
`--league-mix`. Sampling opponents from history instead of only the latest
snapshot avoids the self-play echo chamber (two near-identical policies
reinforcing each other's habits). Snapshots persist on disk, so a resumed run
reloads its league. `--no-league` restores plain frozen-opponent self-play
(latest snapshot only, optionally mixed with bots via `--bot-mix`).

```bash
.venv/bin/python scripts/train_selfplay.py \
    --num-players 4 \
    --num-envs 8 \
    --total-timesteps 5_000_000 \
    --snapshot-every 100_000 \
    --run-dir runs/selfplay
```

The terminal reward is sparse (+1 win / −1 loss on the learner's final
transition), so self-play needs more timesteps than vs-bots training. The
same per-round "cities powered" shaping bonus as vs-bots is on by default
(`--reward-shaping` / `--no-reward-shaping`). `--bot-mix 0.2` makes ~20% of
episodes face normal heuristic bots instead of the snapshot — useful
grounding against self-play degeneracies. Progress is measured externally:
the eval callback plays the policy *against normal bots* every `--eval-freq`
steps, always unshaped, so `eval/mean_reward` is comparable between the two
scripts.

Key self-play-only arguments (other flags match vs-bots — `--learner-seat` and
`--bot-difficulty` are absent since all non-learner seats use the frozen snapshot):

| Flag | Meaning |
|---|---|
| `--snapshot-every` (100 000) | Freeze the current policy into the league (and refresh the envs' pool) every N total timesteps |
| `--league` (on) | Sample opponents from the snapshot league + hard bots; `--no-league` = latest snapshot only |
| `--league-past-k` (4) | How many older snapshots are in the pool at a time (resampled at every snapshot) |
| `--league-mix` (0.5,0.3,0.2) | Pool weights: latest snapshot, past snapshots (shared), heuristic hard bots |
| `--bot-mix` (0.0) | `--no-league` only: per-episode probability of hard heuristic bots instead of the snapshot |
| `--eval-episodes` (20) | Games per eval pass (eval is always vs normal bots, unshaped) |
| `--end-game-cities` | Pin the end-game trigger (mutually exclusive with `--curriculum-start`) |

PPO hyperparameters are CPU-tuned in both scripts (`n_steps=512`,
`batch_size=512`, `n_epochs=4`) — fewer, larger mini-batch updates per rollout
than SB3 defaults, which is ~6× faster per iteration at no practical quality
loss for this task.

### End-game-cities curriculum

If the policy never wins (eval reward pinned at −1.0), the win signal is too
sparse to bootstrap from. The curriculum shortens games by lowering the
end-game city trigger, then ratchets it back up:

```bash
.venv/bin/python scripts/train_selfplay.py \
    --num-players 4 \
    --total-timesteps 50_000_000 \
    --curriculum-start 3 \
    --curriculum-step 2 \
    --curriculum-every 5_000_000 \
    --run-dir runs/selfplay_curriculum
```

Games start ending when someone builds 3 cities (wins every few dozen moves),
and the trigger rises by `--curriculum-step` every `--curriculum-every` total
timesteps until it reaches the rulebook value (17 for 4 players). The trigger
is part of the observation, so the policy conditions on it rather than being
surprised by the moving goalpost. The stage is derived from `num_timesteps`,
so `--resume-from` lands on the right stage automatically; the current value
is logged to TensorBoard as `curriculum/end_game_cities`.

Caveats:

- Eval games use the *current* trigger, so `eval/mean_reward` is only
  comparable within a stage. Each bump resets the persistent best bar
  (`best_mean_reward.json`), and `best_model.zip` means "best at the current
  stage" — only final-stage bests reflect the real game.
- If a bump craters performance, resume from the last checkpoint with a
  larger `--curriculum-every` (or smaller `--curriculum-step`).

To train and evaluate at **one fixed trigger** instead of ramping, use
`--end-game-cities X` (mutually exclusive with `--curriculum-start`). Both
training scripts accept it, and it applies to training *and* eval games
alike; `evaluate.py` accepts it too, so offline win-rate measurement can
match the trained condition. As with the curriculum, eval scores at
different triggers aren't comparable — delete `best_mean_reward.json` when
changing the trigger between runs in the same `--run-dir`.

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
  Exception: `--ent-coef` (default 0.01) is re-applied on every resume — an
  entropy bonus of 0 let long runs collapse to a near-deterministic policy,
  so the flag always wins over the checkpoint's stored value.
- Keep the same `--run-dir` so checkpoints and TensorBoard logs accumulate in
  one place. Keep the same `--num-envs`; PPO buffers are sized per env.
- You can resume from `best_model`, `final_model`, or any `ckpt_*` file — they
  are all complete `MaskablePPO` saves.
- The best-eval bar persists across resumes: the eval callback reads
  `best_mean_reward.json` from `--run-dir` on startup, so `best_model.zip` is
  only overwritten when an eval beats the all-time best for that run dir.
  Delete the json to reset the bar — do this whenever old scores stop being
  comparable (changed reward shaping, eval opponents, or `--eval-episodes`).

---

## 5. Monitoring progress

### One-shot status report

```bash
.venv/bin/python scripts/run_report.py runs/selfplay
```

Prints everything below without opening TensorBoard: checkpoint inventory and
best-eval bar, whether the training process is still running, the recent eval
history (with derived win rate), entropy/explained-variance/value-loss/fps
trends, curriculum/snapshot tags, and health flags for the known failure
patterns (eval pinned at −1, entropy collapse, critic converged to a constant,
eval never firing because `--eval-freq` counts per-env steps).

| Flag | Meaning |
|---|---|
| `run_dir` (positional) | Training run directory, e.g. `runs/selfplay` |
| `--last` (10) | How many recent eval points to list |
| `--all-tags` | Also dump the last value of every TensorBoard scalar tag |

### TensorBoard

```bash
.venv/bin/tensorboard --logdir runs
```

Open http://localhost:6006. Curves worth watching:

| Metric | Healthy signal |
|---|---|
| `rollout/ep_rew_mean` | Rising. With shaping on it includes the powering bonuses (≈ +0.5–1 extra over a strong game); without shaping it approaches +1 as the win rate climbs |
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

| Flag | Meaning |
|---|---|
| `--model` **(required)** | Path to a saved checkpoint (without `.zip`) |
| `--games` (100) | Number of games to play |
| `--num-players` (4) | Total seats |
| `--learner-seat` (0) | Which seat index the learner occupies |
| `--bot-difficulty` (normal) | `easy` / `normal` / `hard` opponents |
| `--seed` (0) | RNG seed |
| `--device` (auto) | PyTorch device |
| `--deterministic` | Greedy action selection — an undertrained greedy policy can pass forever and stall games (counted as losses via `--max-steps`) |
| `--max-steps` (2000) | Per-game step cap; a game hitting it counts as a loss |
| `--end-game-cities` | Match this to the trigger the model was trained at |

### Mixed-field evaluation

To benchmark the expert against easy / normal / hard bots all in the same game:

```bash
.venv/bin/python scripts/evaluate_mixed.py \
    --model runs/vs_bots/best_model \
    --games 100
```

Runs 4-player games (easy + normal + hard + expert) and reports each bot's 1st–4th
placement distribution plus average cities and plant capacity.

| Flag | Meaning |
|---|---|
| `--model` **(required)** | Path to a saved checkpoint (without `.zip`) |
| `--games` (100) | Number of games to play |
| `--seed` (0) | RNG seed |
| `--device` (auto) | PyTorch device |
| `--deterministic` | Greedy action selection for the expert (see caveat above) |
| `--max-steps` (5000) | Per-game step cap; stalled games are dropped from rankings |

### Watching a game

```bash
.venv/bin/python scripts/play_game.py \
    --model runs/vs_bots/best_model \
    --render
```

Renders each state to the terminal and prints the event log at the end.

| Flag | Meaning |
|---|---|
| `--model` | Path to a saved checkpoint; omit to run all bots |
| `--render` | Print board state each step |
| `--all-bots` | All seats use Rust heuristic bots (useful for sanity-checking the env) |
| `--num-players` (4) | Total seats |
| `--seed` | RNG seed (random if omitted) |
| `--bot-difficulty` (normal) | `easy` / `normal` / `hard` |
| `--max-steps` (5000) | Per-game step cap |
| `--end-game-cities` | Match the trigger the model was trained at |

---

## 6. Exporting the policy to the game

Once you have a checkpoint you're happy with, export it so the in-game Expert
bot uses it. The policy weights are embedded in the Rust binary at compile time,
so you need to export first, then rebuild.

```bash
.venv/bin/python scripts/export_policy.py \
    --model runs/vs_bots/best_model \
    --out ../assets/policies/expert.bin \
    --golden ../assets/policies/expert.golden.json
```

| Flag | Meaning |
|---|---|
| `--model` (runs/vs_bots/best_model) | Source checkpoint (without `.zip`) |
| `--out` (../assets/policies/expert.bin) | Destination for the flat `PGRLPOL1` binary weights file |
| `--golden` (../assets/policies/expert.golden.json) | Destination for a torch reference logit file used by the Rust parity test |

After exporting:

1. **Rebuild** the Rust workspace so `expert.bin` is re-embedded:
   ```bash
   cd ..
   cargo build -p powergrid-bot-strategy
   ```
2. **Run the parity test** to verify the Rust forward pass matches torch to
   floating-point tolerance:
   ```bash
   cargo test -p powergrid-bot-strategy
   ```

The Expert bot in `powergrid-client` and `powergrid-lobby` picks up the new
weights after the rebuild. If the parity test fails, the export probably
targeted a different architecture — retrain from a compatible checkpoint.

---

## 7. Typical workflow

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
# export the best checkpoint into the game:
.venv/bin/python scripts/export_policy.py --model runs/selfplay/best_model
cd ..
cargo build -p powergrid-bot-strategy                 # re-embeds expert.bin
cargo test -p powergrid-bot-strategy                  # parity check
```

---

## 8. Orchestrated forever-training (orchestrate.py)

For hands-off training, the orchestrator runs the train → evaluate → adapt
loop indefinitely:

```bash
.venv/bin/python scripts/orchestrate.py                # run forever (Ctrl-C safe)
.venv/bin/python scripts/orchestrate.py --once         # one segment+eval cycle
```

It manages a small population of runs ("lineages" — by default `a-league`,
width 128, and `b-wide`, width 256 with a bots-heavier league mix), each
advanced one *segment* (`--segment-steps`, default 15M) at a time as a
`train_selfplay.py` subprocess. After each segment the newest model plays
`--eval-games` (200) offline games vs normal bots, and the next segment's
hyperparameters are chosen by the triage rules from
[TRAINING-NEXT-STEPS.md](TRAINING-NEXT-STEPS.md):

1. **Never won a game** → archive the run and restart the lineage with the
   end-game-cities curriculum (`--curriculum-start 3`).
2. **Entropy collapse** (< 0.2 nats) → step ent-coef up 0.03 → 0.1 → 0.15
   (and back down once entropy recovers past 0.9).
3. **Eval flat for 3 segments** → flip the league bots weight (raise it on a
   snapshot-heavy mix, drop it on a bot-heavy mix).
4. **Regression** (eval > 0.2 reward below the lineage best) → resume from
   `best_model`; if still regressed, cross-pollinate from the best *other*
   lineage's best_model.
5. **Flat for 100M+ steps after 200M lived** → retire the lineage and spawn a
   mutated replacement (flipped width, new seed, reversed mix).

Everything lives under `--orch-dir` (default `runs/orch/`):

```
runs/orch/
├── state.json          machine state — lineages, knobs, history, pids
├── journal.md          every decision with its reasoning + eval results
├── best/<lineage>/     copy of each lineage's record-setting best_model.zip
└── <lineage>/          a normal training run dir (run_report.py works on it)
    └── logs/           per-segment train/eval logs
```

Console output is deliberately quiet: one line per decision/result plus a
live per-lineage progress bar. The reasoning record is `journal.md`.

**Resumability:** kill it (Ctrl-C/SIGTERM) any time — training subprocesses
keep running, state is saved. On restart it re-attaches to live subprocesses
and otherwise resumes each lineage from its newest checkpoint. Old
checkpoints and league snapshots are pruned automatically (recent + sparse
history kept).

CPU allocation is dynamic: each training subprocess gets
`os.cpu_count() / active lineages` OMP threads. `--envs-per-lineage`
(default 8) controls env parallelism per subprocess.

### TRAINING-SUGGESTION.md status

Where each suggestion from [TRAINING-SUGGESTION.md](../TRAINING-SUGGESTION.md)
landed:

- **Self-play PPO, shared net, masking, flat encoding, scripted bots** — long
  since implemented (this document).
- **League / population-based opponents** — `--league` (default on) and the
  orchestrator's multi-lineage population + cross-pollination.
- **Shaping annealed away** — `--anneal-shaping-steps`.
- **Placement-based reward** — `--terminal-reward placement`.
- **Factored / auto-regressive action heads** — already satisfied by the
  action encoding: cities are built one action at a time under legality
  masking, and auction bidding is a single +1/pass action (since 2026-06-24),
  so there is no composite action left to factor.
- **GNN map encoder** — consciously skipped: the in-game Expert bot's Rust
  inference (`PGRLPOL1`) only runs a fixed two-hidden-layer MLP, and a policy
  that cannot be exported into the game is not useful here.

---

## 9. Troubleshooting

**`KeyError: 'gas'` (or similar) in encoding, or parity test failures** —
the compiled extension is stale. Run `make develop`, then `make test`.

**`ValueError: Observation spaces do not match` when resuming or evaluating** —
the checkpoint was trained with a different observation/action layout (e.g.
before a map or encoding change). Old checkpoints cannot be migrated; retrain.
Current layout: obs 454, actions 94 (USA map, 49 cities). (Before 2026-06-24
this was 143 actions — auction bidding allowed raises of +1..+50 over the
standing bid; it's now collapsed to a single +1 raise, see CLAUDE.md.)

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
