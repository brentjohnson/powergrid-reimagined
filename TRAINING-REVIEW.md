# Training Environment Review — 2026-08-06

Scope: observation space (`encoding.rs` / `encoding.py`), macro action space
(`macro_actions.rs`), reward & shaping (`single_agent.py`, `constants.py`),
curriculum/callbacks, end-game rules flow (`rules.rs`). Focus: what helps or
hinders the policy **discovering the end-game push**.

## Ground truth about the end-game (from rules.rs)

- The trigger is checked at **end of round** (`end_of_round` →
  `determine_winner` → `end_game_triggered`): the round in which someone builds
  `end_game_cities` finishes normally — everyone still powers in Bureaucracy —
  then ranks are `last_cities_powered` → money → cities.
- So the decisive quantity at the moment of triggering is **"how many cities can
  each player power right now"**, and the push decision is: *build to the
  trigger exactly when my powerable count beats everyone else's*.

## Findings, ranked

### 1. (BIGGEST — being fixed now) The observation hides the end-game race

The net must *derive* every quantity the push decision needs, through two tanh
layers, from raw parts on **mismatched scales**:

- No self city **count** anywhere — only the 49-dim ownership bitmask
  (section 4). Summing 49 inputs is learnable but expensive and noisy.
- Opponent city counts exist but normalized `/49` (section 5), while the trigger
  is normalized `/25` (section 16). "Opponent is 2 cities from ending the game"
  is a cross-scale nonlinear combination the net must invent.
- **Self `last_cities_powered` is absent** (opponents' is in section 5!).
- **Nobody's "powerable right now" is in the obs** — the quantity that decides
  the winner at trigger time. Opponent *capacity* (2× plant cost) is there, but
  capacity ≠ powerable: fuel stock is what binds, and opponent fuel is public
  information (view_for hides only money/stats) yet not encoded.
- No affordability signal: "can I reach the trigger this build phase, and what
  would it cost" requires the net to run a greedy knapsack over section 19 +
  slot fees + money. The action *mask* knows (BUILD_n legality), but MaskablePPO
  masks logits only — the policy cannot condition on the mask, only on obs.

**Fix (in progress below): append an 18-feature "end-game race" section 22.**
Append-only + the PGRLPOL/PGRLVAL headers carrying `obs_size` means old
policies can be migrated by zero-padding l1 weight rows — **bit-identical
logits, no checkpoint invalidation**, embedded expert stays the wave-7
champion, and new-format runs can warm-start from a padded champion via
`--init-policy-from`.

New features (all mirrored in Python for the parity test):

| idx | feature | norm |
|-----|---------|------|
| 0 | self cities / trigger (progress; 1.0 = game ends this round) | /trigger |
| 1 | self deficit-to-trigger, saturating | min(d,6)/6 |
| 2–6 | per-opponent progress (5 slots, section-5 order) | /trigger |
| 7 | min opponent deficit, saturating | min(d,6)/6 |
| 8 | self powerable-now = min(best feasible plant subset cities, own cities) | /21 |
| 9–13 | per-opponent powerable-now (public info: plants + fuel stock) | /21 |
| 14 | self last_cities_powered (was missing; opponents already had it) | /21 |
| 15 | powered margin: (self powerable − max opp powerable + 21) | /42 |
| 16 | can-finish-now: greedy cheapest `deficit` cities affordable (exact, same walk as BUILD_n) | 0/1 |
| 17 | money left after finishing (0 if can't finish) | /500 |

OBS_SIZE 582 → 600. Feature 15/16/17 are the push trigger condition stated
directly: *"I can reach the trigger and I out-power the field."*

### 2. Build ladder caps at 6 — the one-turn "explosive finish" can be unrepresentable

`N_BUILD_COUNT` = 0..6. With trigger 17, a player at ≤10 cities cannot end the
game in one build phase even with unlimited cash — it must telegraph with
BUILD_6 and give every opponent a full round to react. The classic human
cash-hoard-then-burst line is thus structurally weakened. The heuristic never
built >6 (1504 decisions), but that describes the *heuristic's* style, not the
policy's reachable optimum. **Recommendation (not done now):** append a
`BUILD_TO_TRIGGER` macro at id 26 (build the cheapest `deficit` cities iff
affordable; dedups onto BUILD_n when deficit ≤ 6, so it is only distinctly
legal in exactly the burst case). Appending at the end doesn't renumber ids;
old policies can be migrated by adding one output row with a strongly negative
bias (~never chosen) — near-identity warm start. Deferred: action-space change
mid-wave-8 isn't worth it until the obs change proves out.

### 3. Shaping never rewards the decisive final powering

`single_agent.step`: shaping applies only `if not terminal`, and the trigger
round's Bureaucracy resolves *on* the terminal step — the one powering that
decides the game gets zero shaped credit. Harmless under winloss (±1 dominates)
but worth knowing; not changed.

### 4. Reward defaults

`--terminal-reward` defaults to `winloss`; `placement` is strictly denser and
values 2nd over 4th (the tiebreak ladder the push is aimed at). Cheap sweep arm
worth trying if not already covered. `POWER_SHAPING_COEF` totals ≈0.8/game vs
±1 terminal — sane.

### 5. Minor notes (no action)

- `round /50` and trigger `/25` live in narrow subranges; fine for an MLP.
- Invalid-action episode-abort path returns a zero obs + all-zero mask with
  −1: fine (mask makes it near-unreachable).
- `eval-difficulty` defaults to `normal` while sweeps select vs hard/frozen
  champion — script callers already override; leave.
- `greedy_pick` walks candidates in *static* cost order while recomputing
  routes against the grown network — mildly suboptimal city sets, but it is
  the teacher's exact behavior (Gate 0) and both sides mirror it; leave.

## Migration / operational notes (IMPORTANT for wave-8)

- **Do not `make develop` into a venv that is driving an in-flight 582-format
  run** — its league snapshots (582) would be rejected by the new native
  `load_opponent_policy` at the next reset and crash the run. Let wave-8 finish
  on its current build.
- `python/scripts/migrate_policy_obs.py` (new) zero-pads any PGRLPOL6/PGRLVAL1
  `.bin` (and golden JSONs) from 582 → 600. Applied to
  `assets/policies/expert.bin` / `expert.value.bin` / both goldens.
  Logits/values are bit-identical by construction (new weights = 0).
- Old sb3 checkpoints cannot be resumed under the new constants directly
  (l1 shape mismatch); warm-start fresh runs from a migrated `.bin` via
  `--init-policy-from` instead.

## Progress log

- [x] Review complete; findings ranked above.
- [x] Rust: section 22 in `encoding.rs` (`max_powerable_now`, race features) +
      `macro_actions::cheapest_cities_with_cost` (greedy walk now also returns
      the spend; `cheapest_cities` delegates, expansions unchanged — Gate 0
      still passes bit-exactly). OBS_SIZE 582 → 600.
- [x] Python: `constants.py` OBS_SIZE + `encoding.py` mirror
      (`_is_subset_feasible`, `_max_powerable_now`, `_cheapest_cities_cost`,
      section 22).
- [x] netviz `obs_layout.rs`: "End-game race" section with per-feature labels.
- [x] `python/scripts/migrate_policy_obs.py` written; embedded
      `expert.bin`/`expert.value.bin` + both goldens migrated 582 → 600.
- [x] `cargo fmt` / `check` / `clippy -D warnings` clean. Rust tests green
      (65 bot-strategy incl. Gate 0, golden-logits vs the migrated expert,
      new section-22 unit test; netviz layout tests).
- [x] `make develop` + full Python suite: 67 passed, including the native
      parity test (Rust `build_observation` == Python mirror across games) —
      confirmed no training process was running before rebuilding the venv.
- [x] CLAUDE.md updated (OBS_SIZE 600, section-22 + migration note).

Perf note: the exact affordability walk only runs when `1 ≤ deficit ≤ 6`
(late game); all other new features are O(plants·2³) per seat. alphazero
checkpoints (`.pt`) are also 582-wide — fresh runs pick up 600 automatically;
padding a `.pt` trunk is analogous if a warm start is ever wanted there.

## Wave-8 evaluation (2026-08-06, from the synced runs/sweep3)

The wave was stopped at ~453–497M of the 550M target — only ~46–70M steps in.
No games can be replayed locally (582-format checkpoints cannot run under the
600-format env), so the ranking below is the recorded frozen-champion eval
(vs 3× q4-y3-finish, 200 episodes/pass, par ≈ −0.50) — the metric that
correctly picked the winner in wave 7:

| arm | best (≈ win) | read |
|-----|-------------|------|
| r4-y3-finish | −0.34 (33%) | **leader** — cross-lineage decay from y3, again |
| y3-batch | −0.36 (32%) | donor still compounding at 496M, entropy healthy |
| r1-main / r2-finish / r5-sharp | −0.38 (31%) | pack |
| r7-exploit | −0.40 | as designed (pool hardener) |
| r3-small-finish | −0.44 | small batch trailing |
| r6-q3-finish | −0.49 (26%) | **at par — the q3 cross-decay failed** |

Conclusions: cross-lineage decay led a second straight wave *from the y3
donor specifically* (r6 shows it does not transfer to other donors); the
ent-0.015 probe (r5) eased entropy 0.25→0.21 with no collapse and no lead —
inconclusive, worth re-arming; no full tiebreak ran, so r4 is "wave-8 leader"
on this single metric, to be confirmed by wave 9's own `--h2h`.

## Wave 9 (sweep script rewritten, runs/sweep4)

`sweep_selfplay.sh` is now the format-reset wave. All migration is automatic
and idempotent (`--prepare` stages everything without launching; artifacts
already staged locally in `runs/sweep4/`, verified against the native
engine): the wave-8 leader and the y3 donor are converted to 600-wide `.bin`
clones (`--from-ckpt`, zero-padded, play-identical), the frozen eval opponent
is the champion clone, and an `--h2h` baseline checkpoint is materialised
from it (`--bin-to-ckpt`) since the 582 original can't run. Every arm is a
fresh run (`--init-policy-from`, fresh value head, fresh 150M budget); the
script refuses to run under a non-600 venv or into the old sweep dir, and
never peers into 582-format league dirs.

Arms: s1-main (control) · s2-finish (decay, 6th re-arm) · s3-gentle
(lr 3e-5→0 — fresh-value-head guard) · s4-y3 (donor lineage reconstituted,
never decays) · s5-placement (decay + placement terminal reward — denser rank
gradient for the new race features; eval envs stay winloss so `best=` stays
comparable) · s6-explore (ent 0.045 — the race features enter with zero
weights; exploration must surface the push lines that use them) ·
s7-exploit (pool hardener) · s8-sharp (ent 0.015 re-arm).

## Recommended next steps (in order)

1. Retrain/fine-tune on the 600-wide obs: migrate the wave-7/8 champion `.bin`
   (`migrate_policy_obs.py runs/<arm>/league/…` or the exported champion) and
   warm-start with `--init-policy-from` — behavior starts bit-identical to the
   champion; gradient can now flow from the race features.
2. If the new features move win rate, revisit finding #2 (`BUILD_TO_TRIGGER`
   macro appended at id 26, near-identity migration via a −10-bias output row).
3. Cheap sweep arm: `--terminal-reward placement` (finding #4).
