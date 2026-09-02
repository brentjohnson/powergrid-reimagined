# RL Training Journal

A consolidated history of the effort to train a neural network that plays Power Grid well
enough to beat humans, what actually happened at each stage, the lessons that survived,
and (at the end) three high-level proposals for where to go next.

Written 2026-07-10, reconstructed from run records, project memory, and git history.

---

## 1. Goal and constraints

- **Goal:** an AI that beats humans. The in-game "Expert" bot is the deployment target.
- **Deployment constraint (hard):** whatever we train must run inside the Rust Expert bot.
  Today that means the PGRLPOL1 format — a fixed two-equal-width-hidden-layer MLP
  (`OBS → H → tanh → H → tanh → N_ACTIONS`) with hand-rolled forward pass in
  `powergrid-bot-strategy/src/policy.rs`. Anything else (GNN, autoregressive heads, extra
  layers) *cannot run in the game at all* without new Rust inference code.
- **The yardstick:** seat 0 vs 3 `normal` heuristic bots, 4 players, USA map, rulebook
  trigger (17 cities). Equal-strength baseline is 25%. The `hard` heuristic bot scores
  **~34.5%** on this benchmark (with jitter=0 paired measurement) — it is, to date, the
  strongest agent in the project.

## 2. Timeline

### Phase 0 — Infrastructure (through early June 2026)

Built the PyO3 bridge (`powergrid-py`), the PettingZoo/Gymnasium env, the shared Rust/Python
observation-action encoding, fused native step methods (opponents driven inside Rust), and
the export path: sb3 MaskablePPO checkpoint → `export_policy.py` → `expert.bin` → native
Rust inference with a golden-logits parity test. Early work targeted the Germany map
(obs 404 / 136 actions); switched to the USA map (obs 454 / 143 actions) on 2026-06-10.

This plumbing has been repeatedly validated (parity tests, golden logits, fair-seat eval
checks) and has essentially never been the problem.

### Phase 1 — PPO (MaskablePPO), June 2026

**vs-bots training (≤2026-06-10).** 50M steps vs heuristic bots: `eval/mean_reward` pinned
at −1.0. The win signal was unreachable from random play; the critic learned "I always
lose" (explained_variance ~0.96), entropy collapsed, policy converged to loss-minimizing
play. Win rate ~0%.

**Curriculum (2026-06-11).** Added `set_end_game_cities` so games can be made short
(trigger 3 → ~80-move games) and grown toward the rulebook 17. Also made the heuristic
bots guarantee game termination (anti-stall overbuild + urgency-scaled auction thresholds).

**Self-play reward bug (diagnosed 2026-06-12).** The original single-stream self-play env
gave the terminal ±1 to the actor of the move that landed on `GameOver` — which, because
bureaucracy runs leader-first, is always the *trailing* player. The winner ~never saw +1.
Also, single-stream GAE mixes all seats' transitions, making per-seat credit assignment
impossible. Fix: frozen-opponent self-play (learner-centric env; opponents run a frozen
snapshot of the policy natively in Rust; `--bot-mix` grounds some episodes vs heuristic
bots). All earlier self-play checkpoints were junk.

**Curriculum grind (2026-06-12 → 06-18).** Chained runs 3 → 12 cities, hundreds of
millions of steps. Findings:
- At trigger 12: ~3.5% win vs normal bots. Pure self-play continuation *regressed*;
  `--bot-mix 0.5` grounding broke the plateau and roughly doubled it to ~6.5%.
- `eval/mean_reward` (even at 100 episodes) could not resolve real 3.5%→6.5% gains;
  `best_model.zip` repeatedly picked lucky eval batches. Only 1000-game offline
  `evaluate.py` sweeps were trustworthy. At rulebook 17, the best checkpoint won **0.7%**.

**Width + shaping experiments (2026-06-19 → 06-21).** Net width made configurable
(64 → 128 default). A 128-wide curriculum run peaked at 21% at trigger 3, then declined
to 0% as the curriculum advanced — entropy collapse (fixed by raising ent_coef) plus a
misaligned absolute powered-cities shaping proxy. Switching to *relative* shaping
(own − best opponent) fixed alignment but destroyed the bootstrap: the agent settled into
a passive money-hoarding optimum, finishing dead last ~100%. Lesson: absolute shaping is
the better cold-start *teacher*, relative is only right for an already-competent agent.

### Phase 2 — AlphaZero, late June 2026

**Run 1 (2026-06-22).** 200 iterations, 50 sims, curriculum 5→17: **0% vs even easy bots
at every checkpoint.** Diagnostics showed the eval harness was fair, the net wasn't
degenerate — it had learned coherent *self-play* that simply didn't transfer to playing
competent opponents. Root causes: closed-world self-play distribution shift, and 50 sims
over ~20–40 legal actions in a 100–300-move game is far too weak a policy-improvement
operator.

### The bug that invalidated everything (2026-06-23/24)

**Buy-resource encoding bug.** Every buy-resource action id decoded to a *batch* purchase
of one unit, and the batch handler unconditionally ended the player's buy turn. Any policy
confined to the action encoding — PPO, AZ, behavior clones — could buy **at most one fuel
unit per round**, while heuristic opponents bought full batches. A player who can't fuel
plants can't power cities and always loses. This single bug explains the ~0% win rates
across *every* algorithm to that point. A behavior clone with 73% per-step agreement with
its teacher had still won 0% — because of this.

Fixed additively (buy ids now map to non-turn-ending single-unit `BuyResources`).
Consequences:
- **All pre-fix training was deleted.** Every negative result before 2026-06-24 is
  uninformative about algorithm choice.
- The auction bid space was collapsed from 50 raise sizes to +1/pass (N_ACTIONS 143 → 94),
  killing a learned jump-bidding artifact at the encoding level.
- The first post-fix PPO run immediately hit **30% vs normal bots** — by far the best PPO
  result ever, confirming the bug (not the algorithm) had been the binding constraint.
  (That checkpoint became stale the same day when the bid collapse changed the action
  space; the embedded `expert.bin` has been rejected at load ever since, so the in-game
  Expert bot currently plays the hard heuristic.)

### Phase 2b — Post-fix PPO and the orchestrator (late June → July)

Long chained self-play runs (600M+ steps through `runs/selfplay`, various shaping modes,
plus the `orchestrate.py` forever-loop). No run produced a policy that beats the hard
heuristic; the durable ranking stayed "heuristic on top."

### Phase 2c — AlphaZero overhaul, BC, DAgger (2026-07-08)

The AZ loop was rebuilt: windowed replay, fixed per-iteration training budget, FPU
reduction + 200 sims, rank-based outcomes (+1…−1 by finish position), parallel self-play
workers, league opponents, proper resume. Also *measured* that the curriculum was
counterproductive: short games are **harder** vs bots (hard bot wins only ~18% at trigger
3–5 vs ~33% at 17), so the win-gate could never advance. Curriculum retired.

Pipeline results (all width 128–256, 4p, rulebook trigger):
- **Behavior cloning the hard bot:** plateaus at ~8–11% vs normal. More epochs don't help
  — it's the BC compounding-error ceiling, not under-training.
- **AZ finetune from the clone:** *regressed* 10.7% → 2.0% (apples-to-apples). Even a
  pure expert-anchored finetune (no self-play episodes at all) regressed. Diagnosis: as a
  ~90% underdog, the value head sees ~all positions as losing, so MCTS visit-count targets
  carry no move-quality signal and (with Dirichlet noise) actively flatten the clone's
  sharp policy.
- **DAgger** (net rolls out vs hard bots, every learner state labeled with the hard bot's
  move): does not collapse; 60 iterations lifted the clone to ~15% vs normal / ~10% vs
  hard. First stable learning-based improvement post-reset.

**Observation enrichment (2026-07-08).** Audit found the obs was blind to three inputs the
teacher uses: per-city connection costs (the Dijkstra term driving builds), opponent fuel
demand, and per-opponent plant detail (turn-order tiebreaker + denial/fuel model). Obs grew
454 → 507 → 582. Then two clean rule-outs:
- **Capacity:** width 256 fits the teacher much better (policy loss 0.287 → 0.168) but wins
  *identically* to width 128. Not capacity.
- **Observation:** clone/DAgger on the full 582 obs score *identically* to the 454 runs
  (clone ~9%, DAgger ~15%). Not observability either.

Conclusion: the plateau is **structural compounding error**. The net agrees with the
teacher on 81.5% of moves on its own rollouts; the 1-in-5 disagreements compound over
~600 decisions per game. Inference-time search recovers some of it — with careful
measurement, dagger582 is ~23% net-only and ~26–27% with MCTS-800 — **still below the
~34.5% teacher.** Imitation + search recovers imitation loss but cannot exceed the teacher.

### Phase 2d — Final self-play test (2026-07-10)

Hypothesis: AZ self-play failed before because it started from an underdog; from a
competent base it should bootstrap. Ran `az582-ft1`: AZ finetune from the best DAgger net,
low noise, 50% self-play / 25% hard-anchor / 25% league, 60 iterations. **Failed.** Best
win rate hit at iteration 18 and never beaten; five independent metrics (bench win rate,
finish position, final cities, end money) all drifted *worse* while policy loss fell —
confidently fitting self-play targets that don't transfer.

**Self-play is now 0-for-4 in this project** (PPO single-stream, PPO frozen-opponent
at scale, AZ from scratch, AZ from a competent base). Verdict recorded: stop re-testing
self-play bootstrapping in the current formulation.

### Phase 3 — Track (A): strengthen the heuristic (2026-07-09/10)

If nothing learned beats the teacher, raise the teacher. Audit found four gaps; three were
implemented and measured — and the measurement itself produced the most important lesson:

- **Broken methodology discovered:** `evaluate_lineup.py --seed` fixes the deck but bot
  RNG (bid jitter) reseeds per game via random UUIDs → ±5pp noise *at the same seed*. The
  early "endgame grab = +6pp" result was pure noise. True paired A/B requires `jitter=0`
  plus a fixed seed and ≥600 games.
- Clean paired results: baseline **32.7%**. Fuel stockpiling (#1): **+1.8pp, kept** →
  ~34.5%. Endgame winning-grab (#3): **−2.4pp, reverted** (its optimistic opponent bound
  made it fire only on fragile money tiebreaks). Turn-order penalty (#2): **−2.8pp,
  reverted**.

## 3. Current strength ladder (2026-07-13)

| Agent | Win rate seat-0 vs 3 normal (held-out, jitter=0) |
|---|---|
| **macro PPO policy (SHIPPED as expert.bin, greedy) — beats the champion** | **~60% greedy / ~54% sampled** |
| evolved hard (powergrid-evolve champion) | ~50% |
| old hand-tuned hard (with stockpiling) | ~31–34.5% |
| macro DAgger / macro BC clone | ~32% / ~31% |
| dagger582 + MCTS-800 (old primitive-encoding Python search) | ~26% |
| **old primitive-encoding BC clone (the ceiling that stalled the project)** | ~9% |
| equal-player baseline | 25% |

### THE MILESTONE (2026-07-13): a learned agent finally beats the heuristic — Phase 2 succeeded

For the first time in the project's history a *learned* policy is the strongest agent.
**MaskablePPO trained on the macro action space wins ~60% greedy (54% sampled) vs 3 normal
bots on held-out seeds — decisively beating the ~50% evolved champion — and ~47% vs three
copies of the champion itself (25% = equal footing).** Stable across held-out seed blocks
(62.5% @ seed 90000, 58.3% @ 95000, 1200 games each) and the full 100M-step run.

This is the payoff of the whole diagnosis→redesign arc. PPO failed for months on the
~600-primitive-decision encoding (sparse-reward credit assignment over a huge horizon);
the macro rebuild gave it a ~50-decision horizon, and it now wins. **Gate 1 already proved
the mechanism** — behavior cloning jumped from the old ~9% primitive-BC ceiling to ~31%
with *only the action representation changed* (same teacher, same algorithm). Gate 2 then
showed PPO exceeds the teacher outright.

Notable: DAgger barely moved the clone (~31%→~32%) — imitation is capped below its teacher,
as always — whereas **PPO, which can discover play the teacher never demonstrates, blew past
it.** Deployment finding: greedy (argmax) play is safe on the macro space (explicit terminal
macros, no stalls) and stronger than sampling, so the Expert bot plays greedy.

Shipped: `assets/policies/expert.bin` is now the PPO policy (582/256/26); the golden-logits
parity test (Rust inference == torch, un-`#[ignore]`d) passes, so the in-game Expert bot is
a bit-faithful copy. `session::add_bot` attaches it with `.with_greedy(true)`.

**Open frontier — beating humans.** The policy beats every *bot* we have, including the
champion. Whether it beats strong *humans* is untested (no automated proxy exists).

### Phase 3 result (2026-07-14): play-time search adds ~+6pp — SHIPPED on the Expert bot

MCTS-over-macros (`search.rs`) with the policy as prior and the exported **PPO value
head** for leaf values (one forward pass, ~40× faster than rollouts → ~120ms/move at 100
sims). Held-out gate (600 games, seed 90000, vs 3 normal): bare greedy **67%** → search-50
**68.5%** → **search-100 73.2%** (+6.2pp); the fair *determinized* mode (reshuffles the
unseen deck so it can't exploit true deck order) also helps (search-50 det **71%**). Vs 3
champions the gain is smaller (bare 54% → search-100 55.5%). **Shipped**: the Expert bot
now plays search-100 determinized (`session::add_bot` → `with_search`); the value net is
embedded (`expert.value.bin`, PGRLVAL1) with a golden parity test. This is the first time
the deployed bot *thinks* rather than one-shotting a policy.

### Follow-up runs (2026-07-13/14): PPO plateauing (~62-68%); self-play regresses (0-for-5)

- **Extended PPO to 200M steps** (resumed the 100M run). Held-out greedy, averaged over 3
  seed blocks: 200M-best ~**62%** (64.5/60.3/62.0) vs the 100M-best's ~60% (62.5/58.3/58.8)
  — a **robust +2–3pp** on every block. Diminishing but real; the steep early climb
  (BC 31% → 63% at 100M) has flattened to ~+2pp per 100M. Re-shipped the 200M-best as
  `expert.bin` (golden test passes). **Caveat learned:** the in-run `best_mean_reward`
  jumped 0.26→0.44 (implying 72%), but held-out greedy is only ~64% — the in-run vs-jittered
  small-N eval materially *overstates* strength; trust the held-out harness, not TB reward.
- **Extended PPO to 300M steps.** 300M-best held-out greedy: 68.1% @90000 / 58.4% @95000 —
  **~neutral vs 200M-best** (better on one block, worse on the other; ~+1pp avg, within
  noise). In-run `best_mean_reward` 0.44→0.48 overstates again. PPO has **plateaued** at
  ~62–68% (block-dependent); further raw-PPO gains are marginal. Kept the 200M-best shipped
  (the search gate was validated on it; 300M isn't clearly better).
- **Self-play from the competent PPO base FAILED — now 0-for-5.** Resumed self-play from the
  ~63% base (frozen-opponent + league + `--bot-mix 0.3` grounding, no shaping); ran to 200M.
  Held-out greedy: 63% base → self-play best **51%** / final **58%** — it *regressed* and
  never recovered the base's competence. Heavy grounding kept it from full collapse but
  couldn't make it improve. This falsifies the last standing pro-self-play hypothesis ("it'll
  bootstrap once we start from a non-underdog") — self-play does not bootstrap in this
  project even from a base that *beats* the champion. Stop pursuing it.

**Net: the shipped Expert bot is PPO-200M-best (~62% vs normal, beats the ~50% champion).**
Remaining levers toward beating humans, in order: inference-time search (Phase 3) on this
net (the macro horizon is now shallow enough for MCTS), still more PPO / a wider net (gains
are shrinking but positive), and human playtesting.

The in-game Expert bot falls back to the hard heuristic (embedded policy stale since the
action-space change) — now the *evolved* hard, which is the strongest agent in the project.

### Phase 1 result (2026-07-10) — evolutionary search shipped a big win

`powergrid-evolve` (CMA-ES, 200 generations, paired jitter=0 games vs normal bots) found a
`BotProfile` that, on **held-out seeds**, beats the old hand-tuned hard by **+19–28pp**:
vs 3 normal 30.8% → **50.2%**, vs 3 hard 25.0% → **53.4%** (i.e. it beats three copies of
the old hard bot), vs 3 easy 28.2% → 48.7%. It wins *legitimately* — builds more cities
(16.2 vs 15.9), powers more (16.2 vs 15.8), ends sooner (9.3 vs 9.7 rounds). The profile
is "extreme-frugal": reserve cash for cities+fuel, high bar to bid on plants, keep plants
fed, opponent-interference (denial/block) and stockpiling turned **off**. That reads as
**auction discipline**, exactly how strong humans play — a much better Phase-2 teacher.
Shipped into `assets/bots/default.toml` (`hard` = `expert`). Confirmed near-optimal:
pushing the bounds-hit weights further *hurts* (widening bounds not warranted; diverging
CMA `sigma` is just a benign clamped-plateau artifact). This also fixed a latent
`InvalidFuelSplit` heuristic bug that profile search exposed (gas-preferring fuel split
could exceed available gas).

**Stage 2 (co-evolution, 2026-07-11) — champion confirmed robust; Phase 1 complete.**
Two 200-gen searches starting from the champion: (A) evolve vs 3 copies of the champion,
(B) co-evolve vs a mixed pool {normal, old-hard, champion}. Neither beat the champion on
held-out seeds — the best "counter" reached only 27.5% vs 3 champions (vs the 25% symmetric
floor), and both Stage-2 results were slightly *worse* than the champion vs normal
(49.1–49.6% vs 50.2%). All three independent searches converged on the same strategic spine
(high reserves/thresholds, denial off, high fuel-risk), differing only in low-sensitivity
weights — a **broad plateau of equivalent frugal strategies, not an exploitable knife-edge**.
Verdict: the heuristic parameterization is maxed out (~50% vs normal, unexploitable vs
itself). Robustness is proven only *within* the heuristic family — a fundamentally different
strategy (a human, or a Phase-2/3 learned agent) could still find a blind spot. To go
further we need a richer strategy space (Phase 2 macro-actions, with this champion as the
DAgger teacher) or play-time search (Phase 3).

---

## 3.5 The macro-action rebuild and the Great Reset (2026-07-25/26)

*Everything in Section 3 above ends here as a self-consistent record — but the numbers in
it (including the "60% macro PPO" milestone) were measured on a macro action space that was
about to be torn down and a training loop that was about to be declared unsound. Read them
as history, not as the current baseline.*

**The macro menus were rebuilt around quantity ladders (2026-07-25), `N_ACTIONS = 26`.**
The first macro space (the one that produced the 2026-07-13 milestone) was replaced by a
cleaner design that settled the shapes of the two menus on principle:

- **Build is an absolute count** — "build the *n* cheapest cities" for n = 0..6 (ids 8–14).
  Cities are interchangeable; one more is one more income step wherever it sits.
- **Buy is a bitmask over plant slots** (ids 15–22): choose *which plants you intend to
  fire*, then top each up to a firing's worth counting current stock. Declaring the subset
  is what makes a purchase well-defined on a *shared* fuel pool ("top plant A up" is
  ambiguous when B also burns coal; "these plants fire" fixes the requirement as the sum
  over the declared set). Because it tops up rather than adds, the full-rack mask reproduces
  the champion's buy bit-for-bit — so no `BUY_DEFAULT` escape hatch is needed, and this is
  the one phase where the rebuild turned a *constant* imitation label into a *varied* one
  (the teacher spreads across all 8 masks).
- **Powering has no macro at all** — `Bureaucracy`, fuel splits, and resource discards are
  auto-resolved with the heuristic (the teacher fired the optimal subset 100% of the time;
  "power nothing" was legal everywhere and correct nowhere). Removing it cut episodes ~18%
  (52 → 43 macro-decisions per seat).
- **No `*_DEFAULT` ids remain**; both ladders reproduce the champion bit-exactly on their
  own (Gate 0), and `legal_macros` dedups so a given intent always maps to the same id.
  Stockpiling is deliberately unrepresentable (CMA-ES had pinned `stockpile_rounds` at its
  floor). New format epoch: **PGRLPOL6** (the magic is a *layout* guard — bump it whenever
  ids are renumbered even at unchanged `N_ACTIONS`, so a dims-only check can't silently load
  a scrambled action map).

**The Great Reset (2026-07-26): all prior RL *findings* are void, not just the checkpoints.**
An audit concluded the earlier sweeps (waves 1–2) had run against broken rules, a broken
environment, and a mis-mapped action space, so their conclusions describe a system that no
longer exists. Explicitly *not* evidence any more: "big batch beats base 39% vs 26%",
"constant lr helps", "entropy 0.10 collapses a converged policy", "a mostly-historical
league regresses", "the embedded expert wins ~30% vs normal", and every pre-reset AlphaZero
/ curriculum / shaping conclusion. The danger was operational: those numbers had been baked
into the sweep script's `COMMON` block, so an unvalidated recipe was being applied silently
to *every* arm of a supposedly controlled comparison. What survived the reset is
methodology, not results: measure jitter-0, fixed-seed, ≥600 games; `--init-policy-from`
leaves the value head random; a clone of `hard` saturates an eval against `normal`. Training
restarts from a fresh **behavior clone** of the evolved-hard champion.

## 3.6 The self-play sweep grind (waves 3–15, 2026-07-28 → 2026-08-26)

With the reset, the project settled into a disciplined loop: `sweep_selfplay.sh` runs 8
parallel arms, each forked from the *pinned* checkpoint of the reigning champion, trains a
growing step budget (50M → ~1.05B cumulative over the waves), and the wave winner becomes
the next fork point *and* the next embedded Expert. This has now run **thirteen waves** and
is the source of every current result.

**How arms are ranked (this took several waves to get right).** The `--compare` yardstick
(arm in seat 0 vs 3× `hard`) **saturates** once arms pass ~68% and can even *anti-correlate*
with true strength — so from wave 7 on, checkpoint selection and ranking use a **frozen
copy of the reigning champion** as the eval opponent (`--eval-opponent`), plus a head-to-head
`--h2h` (arm vs 3× the frozen champion). When the two boards disagree at the top, the wave is
decided by a **full tiebreak**: direct arm-vs-arm matches in both seat orders plus a
fresh-seed 800-game h2h. Every champion since has had to win that tiebreak, not just top one
board.

**The recipe that keeps winning: cross-lineage decay.** Fork the *never-annealed* "y3" donor
lineage (a constant-lr arm that is deliberately never decayed and kept alive as pure fork
material) and finetune it with **lr decay → 0**. This "cross-lineage decay" factory won
waves 7, 8, 10, and 13, and *decay in some form* has produced ~11 straight champions. The
mechanism: the donor banks raw exploration at constant lr; a fresh fork then anneals it to a
sharp, converged policy whose approx-KL falls to ~0. The eval peak reliably lands in the
**back third** of each 150M budget, exactly where the lr finishes annealing — which is why
the budget was kept at 150M/wave rather than cut to 100M.

**The gamma breakthrough (wave 12).** Reward is terminal-only (win/loss at game end) over
~50 macro-decisions. At the sweep-long `gamma 0.99`, that finish is discounted to
0.99^50 ≈ 0.61 by turn 1 — the early game barely feels the outcome. Raising gamma
propagates the finish backward: 0.997 → ~0.86, **0.999 → ~0.95**. Sweeping it found a clean
inverted-U with the **peak at 0.999** (1.0 undiscounted hurts). This was the single biggest
lever of the whole grind, and — crucially — **gamma is training-only; it never touches the
exported net**, so gamma-varied arms stay bit-faithfully exportable. Wave 14 added a subtle
finding: at the peak gamma the **lineage advantage reversed** (champion-line beat
cross-lineage), best explained as *value-head gamma-continuity* — re-arming at a new gamma
forces the critic to re-learn the return scale, so fork the checkpoint whose training gamma
is *closest* to the target.

**Other levers, settled:**
- **Population play pays (wave 6).** `--league-peers` lets the 8 arms train against each
  other's evolving snapshots. The clean control (identical recipe, no peers) never left its
  fork point; the peered twin beat it on both metrics. Peers on by default since.
- **The 0.20 bot anchor is load-bearing.** Every attempt to drop bots from the opponent mix
  (BOTS=0, or trading half the anchor for more peers) landed at h2h par and below 50% vs
  hard — across *four* waves. The 20% heuristic share is what keeps self-play grounded.
- **Low entropy (ent-coef 0.015)** was validated as a champion-line lever (wave 11), after
  three waves of "safe but never winning".
- **Dead knobs (do not re-test):** entropy-UP (collapses a good policy, every time),
  placement reward (neutral 4×), gentle-lr (turned out to be only a fresh-value-head
  migration guard, not a real lever), a **wide net (192-wide)** (failed at 300M — the
  plateau is not capacity-bound), target-KL, and tight clip.

**Observation growth without invalidation (2026-08-06).** Obs grew 582 → 600 with
**section 22, "the end-game race"** (18 features: per-seat trigger progress and saturating
deficits, powerable-right-now for every seat — the exact quantity `finish_ranks` ranks by —
self last-cities-powered, powered margin, and can-finish-now cost from the same greedy walk
`BUILD_n` uses). The key infrastructure lesson: **append-only obs growth is checkpoint-safe.**
The policy header carries `obs_size`; `migrate_policy_obs.py` zero-pads the first-layer rows
(and the golden `obs`) to the new width for bit-identical outputs, so the embedded champion
and every fork warm-started right through the change. Only a *reordering* of obs features
would invalidate checkpoints.

**The succession of embedded Experts** (each the wave winner, PGRLPOL6, net-width 128, with
its matching value net re-exported so Phase-3 search leaf values track the policy):

| Date | Embedded | Wave | Recipe headline | Native vs 3× hard |
|---|---|---|---|---|
| 2026-07-28 | w3-low-lr | 3 | freeze policy until random value head means something | ~52% |
| 2026-07-31 | y4-lr-decay | 4 | lr decay → 0 (the finisher) | 76% |
| 2026-08-01 | z3-batch-decay | 5 | batch 1024 + decay | 68% |
| 2026-08-03 | p2-finish | 6 | population play + decay | 74% |
| 2026-08-05 | q4-y3-finish | 7 | **cross-lineage decay** (donor fork + decay) | 78% |
| 2026-08-09 | s3-gentle | 9 | first obs-600 champion (gentle warm-start) | 80% |
| 2026-08-12 | t3-y3-finish | 10 | cross-lineage decay, post-reset | 86% |
| 2026-08-15 | u4-sharp-finish | 11 | champion-line decay + low entropy | 80% |
| 2026-08-18 | v7-gamma | 12 | **gamma 0.997** | 84% |
| 2026-08-23 | w5-y3-gamma | 13 | cross-lineage decay + gamma 0.997 | 88% |
| 2026-08-26 | x5-champ-g999 | 14 | champion-line decay + gamma 0.999 | 84% |
| **2026-09-01** | **a3-nsteps4096** | **16** | **rollout length (n-steps 4096)** | *(pending user test)* |

(The native 50-game harness carries ±~6–7pp noise and is a sanity check only; wave winners
are decided by the torch `--compare` + full tiebreak, not this number. x5-champ-g999's 84%
is *below* the outgoing w5's 88%, but x5 won the meaningful boards — it led the wave-14 h2h
vs frozen w5 and won the fresh-seed decider — the native dip being the expected
champion-line-vs-hard trade-off. Committed after the user tested it in-game.)

**Wave 15 (2026-08-29) — a NULL WAVE; the champion-continuation line has plateaued.**
Gamma 0.999 locked; the arms were the champion-line anchor (z1), a cross-lineage hedge (z2),
four one-knob levers (rollout length z3, deep league z4, value-emphasis z5, exploiter mix
z7), and a weight-average "soup" (z6). Seven arms reached 1.05B; z6-soup stalled at 190M.
**Nothing beat the incumbent x5-champ-g999.** The two arms that flickered above h2h par —
z3-nsteps (n-steps 2048, eval leader) and z7-exploiter (mix 0.70/0.10/0.20, h2h+compare
leader) — both by <1.5pp, inside the ±5pp noise. A fresh-seed decider (88888, 400 games/dir)
was decisive: **x5 dominates both** (x5→z3 29.8 vs z3→x5 25.5; x5→z7 26.5 vs z7→x5 22.0;
best offense 28.15 AND best defense 23.75). The most telling arm was z1-champ-cont, the
presumptive winner: a pure value-continuous fork of x5 that landed *below* par. **Forking x5
and annealing at lr 1e-4 converges back to x5, not past it.** The Expert is unchanged.

**Wave 16 (2026-09-01) — the plateau BREAKS: rollout length is real.** Bigger perturbations:
1 control (a1) + 5 distinct larger levers (a2 lr restart 3e-4, a3 rollout 4096, a4 relative
shaping, a5 n-epochs 8 + target-kl, a6 gae-lambda 0.98) + 2 soups (z6 continued, z8 new).
Six a* arms hit 1.05B; z6 finished (350M); z8 stalled at 212M. **a3-nsteps4096 WINS** — the
first successor to x5 in three waves. It led **all three boards** (compare 87.5%, h2h 30.0%,
eval −0.28) and a two-seed decider vs the incumbent confirmed a *sign-stable* edge (seed
161616: a3 23.2 vs x5 21.8, +1.4; seed 24242: a3 27.0 vs x5 24.2, +2.8; combined 800 games/dir
a3 offense 25.1 vs x5 23.0, **+2.1pp**). Unlike wave 15's z3 (led reporting, then the decider
flipped to x5), nothing flips for a3. **Key finding: ROLLOUT LENGTH IS A REAL LEVER** — the
identical idea at n-steps 2048 (z3) flickered-then-lost last wave, but 4096 wins; at gamma
0.999 the return horizon is long and GAE truncates at n-steps, so longer rollouts cut
truncation bias. n-steps 4096 is now the champion default. gae-lambda 0.98 (a6) also edged x5
(the same advantage-horizon idea from the other side). **Dead levers** (do not re-test): lr
restart 3e-4 (a2 worst everywhere — x5 is a real basin, not an escapable local optimum),
relative shaping (a4 below par), n-epochs+kl (a5 below par). Soups sat at par. a3-nsteps4096
is the new embedded Expert (export staged, pending user in-game test).

**Observation growth #2 — obs 600 → 624 (2026-09-01).** Section 23 adds four **derived
per-actual-slot market decision features** (affordable, effective min-bid, powering headroom,
is-upgrade) for the 6 Nominate slots — facts the heuristic's `evaluate_plant` uses but the net
had to reconstruct through two tanh layers from the raw market attributes (sections 9/10) plus
its own money/plants. Motivated by the analysis question "is the slot-positional market
encoding inefficient?": the layout is fine (the market is canonically sorted by number, so
slots are stable roles and identity is already encoded), but the raw-attributes-vs-decisions
gap is real, and this is the same append-only pattern that worked for section 22. Wired into
Rust `build_observation` (authoritative), the Python `encode_observation` mirror, `constants.py`,
and the netviz layout; parity + section tests green. **This forces wave 17 into a wave-9-style
format reset** (below): a 600-wide `.zip` can't be resumed under the 624 env, so every arm
becomes a fresh clone of the *migrated* champion (`migrate_policy_obs.py` zero-pads l1 → plays
a3 bit-for-bit; the embedded expert was migrated the same way and stays behavior-identical).

**Wave 17 (2026-09-01) — OBS-624 MIGRATION + exploit the rollout win + two weight-space bets,
configured.** Because of the obs growth above, every arm is a **fresh clone**
(`--init-policy-from`) of the migrated a3 @624 (`wave17-champion.bin`), fresh value head, step 0,
WAVE_STEPS 250M (up from the 150M fork increments — a fresh value head must refit before the
lr-decay peak, and there are 24 new inputs to learn). The eval opponent is the migrated a3; the
`--h2h` baseline is the same weights as a 624 sb3 ckpt (`--bin-to-ckpt`). 6 progress arms: b1
control (re-establish a3 under obs-624, n-steps 4096); **b2 n-steps 8192** and **b3 n-steps
2048** bracket the rollout peak (2048/4096/8192 → one curve on shared peers); b4 gae-lambda 0.98
stacked on the champion; b5 ent 0.015 sharpen; **b6-gentle** (lr 3e-5→0) — the fresh-value-head
migration guard (b6 vs b1 says whether the standard-lr clones are healthy). **2 crazy
weight-space arms** (replacing the stale x5-basin soups z6/z8): **c1-extrapolate** — clone from
merge `−0.5·x5 + 1.5·a3`, i.e. step 1.5× *past* a3 along the demonstrated x5→a3 improvement
direction (task arithmetic / extrapolation, since interpolating soups only reach par), gentle lr
as the safety net; **c2-swa-wave16** — SWA soup mean(a1,a3,a6). New tool `scripts/make_merge.py`
does the weighted/extrapolated merge (and, with make_soup, now reads migrated `.bin` inputs so
`--prepare` rebuilds the merges at 624). Donor/cross-lineage stays retired (gamma-continuity
predicts a 0.999 fork of the gamma-0.99 donor lands at par — wave 15's z2 confirmed).

**Interpretability tooling (2026-08-26).** `analyze_policy.py` reads the Expert `.bin` and,
numpy-only, reports what a net computes in *game terms*: an exact input→macro attribution
Jacobian grouped by the 18 obs sections ("Build keys off self cities/plants/resources;
Nominate off the plant market + opponent cities; Buy-Nothing is strongly −by money"), plus a
`--compare` behavioral/attribution diff between two champions (valid because the lineage is a
warm-started, index-aligned chain). First run showed wave 14's champion mostly re-tuned
**opponent-cities** sensitivity of Build/Nominate — i.e. opponent-aware end-game build timing.

## 3.7 Current state (2026-09-01)

The deployed Expert bot is **a3-nsteps4096** (the wave-16 winner; export staged, pending the
user's in-game test — the outgoing x5-champ-g999 held from wave 14 through the wave-15/16
plateau), playing **Phase-3
determinized MCTS-100 over macros** with this policy as prior and its exported value net for
leaf evaluation. It is a *learned* agent, decisively the strongest in the project — it beats
the ~50% evolved-hard champion head-to-head and every heuristic bot. Across sixteen sweep
waves the single throughline is that the wins came from **training-loop hygiene, not new
algorithms**: cross-lineage decay, the right gamma, longer rollouts at that gamma, a
load-bearing bot anchor, population play, rigorous frozen-champion evaluation, and
append-only obs growth. The self-play-doesn't-
bootstrap verdict of Section 3 is not contradicted — this loop is *frozen-opponent /
league / bot-anchored* finetuning of a competent clone, not open-ended self-play, and the
0.20 bot anchor is exactly what keeps it grounded. The open frontier is unchanged and
untested: whether any of this beats strong **humans** (no automated proxy exists).

## 4. Lessons that survived everything

1. **Check the action interface before the algorithm.** One encoding bug (1 fuel/turn)
   silently capped *every* algorithm at ~0% for weeks. Uninformative negatives are worse
   than no experiments.
2. **Self-play does not bootstrap here** (0-for-4), regardless of base competence. The
   trained policy gets confidently good at a closed world that doesn't transfer.
3. **Imitation has a hard ceiling below the teacher** — ~60–70% of teacher strength —
   caused by compounding error over ~600 primitive decisions per game, not by capacity or
   observability (both cleanly ruled out).
4. **Evaluation noise has repeatedly manufactured false conclusions.** Small-N evals,
   `best_model.zip`, unpaired A/Bs, and jittered bots each produced results that later
   inverted. Only large-N, paired, jitter-0 measurement is trustworthy.
5. **The sparse relative win signal is brutal.** Every shaping proxy either taught the
   wrong thing (absolute → mid-pack complacency, hoarding) or was too weak to bootstrap
   (relative). The curriculum designed to soften it turned out to make the game *harder*.
6. **Long-horizon primitive-action credit assignment is the recurring villain.** ~600
   decisions, one win/loss bit, four seats. PPO, AZ, and BC all broke against this same
   wall in different ways. (The macro rebuild — ~50 decisions/game — is what finally let
   PPO learn; every post-reset gain lives on that shorter horizon.)
7. **The eval opponent must not saturate.** `--compare` vs `hard` tops out and can *invert*
   the true ranking once arms pass ~68%; checkpoint selection then tracks eval noise. The
   fix that stuck: eval against a *frozen copy of the reigning champion*, and decide close
   waves by a full tiebreak (direct matches in both seat orders + a fresh-seed 800-game h2h),
   never by one 200-game board.
8. **The improvements came from training-loop hygiene, not new algorithms.** Cross-lineage
   decay (fork a never-annealed donor, anneal lr → 0), the right discount (gamma 0.999 to
   propagate a terminal reward across ~50 macros), population play (arms in a shared league),
   and a load-bearing 0.20 bot anchor each moved the needle; entropy-up, placement reward,
   gentle-lr, a wider net, and dropping the bot anchor each did not. Gamma is training-only,
   so it never costs deployability.
9. **Append-only observation growth is checkpoint-safe.** Zero-padding the first layer to a
   wider obs gives bit-identical outputs; only *reordering* features invalidates a policy.

---

## 5. Where to go from here — top three ideas

The pattern across every failure is the same: **the learning problem as formulated —
~600 primitive decisions per game, one relative win/loss bit, four seats — is the enemy**,
not the algorithm and not the network. All three proposals restructure the problem instead
of tuning within it. They are complementary and sequenced cheapest-first; #1 and #2 in
particular are designed to feed each other.

### Idea 1 — Directly optimize the heuristic's parameters with evolutionary search (CMA-ES / population-based)

**The observation:** the strongest agent we have is the hard heuristic, and its ~30
`BotProfile` weights are *hand-tuned guesses*. Nobody has ever searched that space. The
one clean strengthening win (#1, +1.8pp) came from a hand-designed change measured with
the new paired harness — but hand-designing one change at a time through a ±0.5pp-noise
harness is slow, and interactions between weights are invisible to it.

**The proposal:** treat the full weight vector (auction weights, buy weights, build
weights, urgency scalars — everything in `default.toml`) as a genome and run CMA-ES or a
small population-based search. Fitness = paired, jitter-0, fixed-seed win rate over a
few hundred games — exactly the harness we just built and validated. Rust games are fast
and embarrassingly parallel; a 40-member population evaluated at 600 games each is very
tractable. Two stages: first vs the fixed normal-bot lineup (fitness is well-defined),
then round-robin within the population (co-evolution) so the result doesn't overfit to
normal-bot quirks.

**Why this fits the failure history:** it has *none* of the diseases that killed the other
approaches — no credit assignment (fitness is whole-game win rate), no distribution shift
(it plays the real game the whole time), no deployment gap (the artifact **is** the
Expert bot — no export, no Rust port, ships immediately). And it raises the imitation
ceiling for everything else: DAgger tops out at ~60–70% of *whatever the teacher is*.

**Honest limits:** it can only find the best agent expressible in the current heuristic's
functional form. It won't invent new strategy concepts — that's what Ideas 2 and 3 are
for. Expected payoff is a few points, maybe more if the hand-tuning left interactions on
the table; cost is low and the infrastructure mostly exists.

### Idea 2 — Rebuild the action space around macro-plans (options), scored by the network

*(This adopts and sharpens the options/macro-actions suggestion.)*

**The observation:** the decisive diagnosis of the imitation plateau was compounding error
over ~600 primitive decisions. The same horizon is what starved PPO's credit assignment
and what made 200-sim MCTS anemic. Meanwhile the game's *strategic* content is maybe 40–80
real decisions: which plant to want and how high to go, which fuel posture to take, which
expansion to commit to, when to race the trigger.

**The proposal:** the policy never picks a city, a fuel unit, or a bid increment again.
Each phase, a deterministic planner enumerates a small set of complete, legal macro-plans,
and the network scores/chooses among them:

- *Build:* cheapest single city / max-affordable expansion / cheapest k-city expansion /
  expand into cheapest-future region / block opponent's cheapest slot / build nothing.
- *Buy:* fuel exactly this round's firing / stockpile n rounds / starve a contested
  resource / buy nothing.
- *Auction:* value each market plant with a max-bid (the heuristic's `evaluate_plant`
  already computes exactly this); the "policy" nominates a plant+ceiling or passes, and
  scripted bidding executes to the ceiling.

Critically, **the planners already exist** — they are the decomposed hard heuristic
(`decide_build_cities`, `decide_buy_resources`, `evaluate_plant`, the Dijkstra routing).
This isn't new game AI, it's re-exposing existing code as an action menu. The heuristic
itself becomes one particular fixed chooser over this menu, which gives a perfect
diagnostic: a policy that merely *matches* the heuristic's choices reproduces ~34.5%
exactly (no compounding-error tax, because each choice executes a complete correct plan),
and every learned improvement is strictly additive on top.

**Why this fits the failure history:** episode length drops ~600 → ~50 macro-decisions
(a >10× improvement in credit assignment for *any* algorithm — PPO's sparse-reward
problem, MCTS's depth problem, and BC's compounding error all shrink together). Action
space drops 94 → ~15 semantically meaningful choices. A one-step wrong choice costs one
plan, not a cascade of ruined micro-moves. It's the single change most likely to make the
*previously failed* algorithms start working, and it's the only proposal that could let a
learned policy genuinely exceed the teacher at deployable (no-search) speed.

**Cost and constraints:** this is the big rebuild — new encoding (new obs is fine, new
small N_ACTIONS), plan-generator extraction in Rust, mirrored in Python, all current
checkpoints invalidated (they're below the heuristic anyway; nothing of value is lost).
The network stays a PGRLPOL1-compatible MLP, and the plan executors are Rust code the
Expert bot calls — so it **can** run in-game, but the Expert bot gains a plan-execution
layer. Start training with DAgger over macro-plans (proven stable), then — only from that
competent base — the macro-level game tree is finally shallow enough that search/AZ gets
a real chance.

### Idea 3 — Stop trying to train a superhuman *policy*; build a superhuman *engine* (play-time search + a supervised value net)

**The observation:** every strong game AI in games like this is *search plus evaluation*,
not a bare policy. Our one unambiguous positive learning result is that inference-time
MCTS improves the net (~23% → ~26%). And the piece search actually needs — a good **value
function** — is trainable by plain supervised regression, which sidesteps every
instability we've hit: generate millions of positions from fast Rust heuristic-vs-heuristic
games (with profile diversity from Idea 1's population), label each with the actual
finish-rank outcome, and fit `obs → rank-value`. No self-play loop, no moving targets, no
underdog collapse — it's just supervised learning on a stationary dataset, and unlike the
policy-imitation ceiling, a value net has no compounding-error problem (it's consulted,
not rolled out).

**The proposal:** port determinized/information-set MCTS to Rust inside the Expert bot:
sample plausible hidden states (deck order, opponent money) rather than peeking — the
Python MCTS's true-state forking is an info advantage that can't fairly ship against
humans — and search over macro-plans (Idea 2) so the tree is shallow. Use the supervised
value net for leaf evaluation and the best available policy (DAgger net or the heuristic
itself) as the prior. Thinking time is a strength dial: even 200ms of Rust search per
move is thousands of simulations.

**Why this fits the failure history:** it reuses only components with *positive* evidence
(heuristic rollouts, imitation priors, search-as-improvement-operator, supervised
learning) and none with negative evidence (no self-play training loop, no
sparse-reward RL). It also changes what "beating humans" requires: the net no longer has
to be superhuman greedy — it only has to be a decent prior and evaluator, and search
supplies the superhuman part at play time. This is the Stockfish/KataGo-at-inference
framing rather than the train-a-god-policy framing.

**Cost and constraints:** the value net is a new small MLP head — trivially runnable in
Rust with a PGRLPOL1-style format extension (explicitly: today's Expert bot has *no*
search; this is a real Rust engineering project, the largest of the three). Determinization
costs some strength vs the cheating searcher, but `Game::copy()` is already cheap and the
hidden information in Power Grid is mild (deck order; opponent money is trackable).

### Recommended sequencing

1. **Now:** Idea 1 (evolutionary search over `BotProfile`) — cheapest, zero new concepts,
   ships directly as the Expert bot, and raises the teacher every other idea depends on.
2. **Next:** Idea 2 (macro-action rebuild) — the structural fix for the diagnosed root
   cause; re-run DAgger (proven) on the new action space and expect it to *match* the
   teacher instead of plateauing at 60% of it.
3. **Then:** Idea 3 (Rust IS-MCTS + supervised value net) over the macro-action space —
   converts whatever policy/value quality exists into maximum playing strength, and is
   the most plausible route to actually-beats-humans.
