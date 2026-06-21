<!--
MAINTENANCE NOTE TO CLAUDE (future self): This is a living document.
Every time we run a new RL training experiment or change the training setup,
append a new dated entry to the "Experiment log" section below (newest at the
bottom), and fold any durable lesson into "The cheat sheet I wish I'd had".
Keep the voice: honest, second-person, written for Brent-at-the-start. Record
failures as plainly as successes — the failures are where the learning is.
Update the "Where we are now" section to point at the latest state.
-->

# Training an RL bot to play Power Grid: a field journal

*Written for you, back at the start — before the first 0% win rate, before you
learned to read an entropy curve like a heart monitor. Here's what's coming.*

## What you're trying to do

You want an "Expert" bot that actually plays Power Grid well — well enough to
beat the hand-written heuristic bots. The plan is reinforcement learning:
train a policy network in Python (MaskablePPO, via Stable-Baselines3), export
the weights to a tiny format (`PGRLPOL1`), and run inference natively in Rust
so the bot needs no Python at game time. The whole RL stack lives in `python/`;
the encoding (454-dim observation, 143 actions, USA map) is shared with the
Rust Expert bot so what you train is exactly what ships.

That part — the plumbing — works. This journal is about the hard part: getting
the thing to *learn*.

## The thing nobody tells you up front

Power Grid is a long game. A full match is ~70 decision-steps, and the only
"true" reward is +1 if you win, −1 if you don't. That signal arrives once, at
the very end. Asking a network to discover good play from that alone is like
learning to cook from a single bite of the final dish. **Almost everything in
this journal is a fight against that sparsity** — reward shaping, curricula,
self-play, entropy — they're all different ways of manufacturing a usable
gradient when the real one is too faint and too late.

Keep one principle nailed to the wall: **the eval win-rate vs the bots is the
only honest yardstick.** Training reward can climb beautifully while the agent
never wins a single game. You will be fooled by this more than once.

---

## Experiment log

### Episode 0 — vs-bots, the hopeful beginning

The first approach: train directly against the heuristic bots
(`train_vs_bots.py`). It produced a checkpoint, it exported, the Rust bot
loaded it. Victory? No — the resulting Expert was weak (~0/30 vs the *normal*
bots). But it proved the pipeline end-to-end, and that's worth something.

**Learning:** "it trains and exports" is not "it plays well." Build the
evaluation harness early. (`scripts/evaluate.py` — win rate, placement,
cities/powered/money — became the most important tool you have.)

### Episode 1 — self-play, and a bug that poisoned everything

Self-play seemed like the obvious next move: let the agent learn against itself.
The first design had all seats share one policy in a single transition stream.
It pinned eval reward at −1.0 and we couldn't figure out why for a while.

The cause was subtle and brutal: `GameOver` is only set at the *end of a round*,
so the terminal +1/−1 reward landed on whichever seat happened to make the
round's last bureaucracy move — almost always the *trailing* player. The winner
essentially never saw the +1. The agent was being told it lost every time it
won. Every checkpoint from that era is junk.

We redesigned it (2026-06-12) into **frozen-opponent self-play**: the learner
occupies one seat, opponents are driven by periodic frozen *snapshots* of the
learner's own network, and reward is cleanly attributed to the learner's moves.

**Learning:** in a multi-agent game, reward *attribution* is a correctness
problem, not a tuning problem. A shaping/credit bug looks exactly like "the
algorithm isn't working." When eval is pinned at a suspicious constant, suspect
the plumbing before the hyperparameters.

### Episode 2 — the curriculum, because −1.0 wouldn't budge

Eval reward was still stuck at −1.0: from scratch, the agent never finished a
game in a winning position, so it never saw a +1 to reinforce. So we added an
**end-game-cities curriculum** (`EndGameCurriculumCallback`,
`--curriculum-start`): start with very short games (end the game at 3 cities
instead of the rulebook ~17), which are faster and produce wins more often, then
ramp the trigger up over training toward the real value.

We also made the network width configurable (`--net-width`, default 128) around
this time.

**Learning (provisional):** if the reward is unreachable, change the *task*
until it's reachable, then make it harder. Good instinct. The execution had
problems — see the next episode.

### Episode 3 — `selfplay_w128_curriculum`: a confident, total failure

Setup: 128-wide net, curriculum egc 3→17, `--bot-mix 0.5` (half the opponents
real bots, half frozen self-snapshots), `ent_coef=0.01`, 121M steps.

Result: it **got worse the longer it trained.** Eval peaked at −0.58 (~21% win)
very early, at egc=3 around 14M steps — and then declined *monotonically* to
−1.0 (0% win) as the curriculum advanced. The final policy lost every game.

Two root causes, and both are lessons you'll reuse:

1. **Entropy collapse.** Policy entropy fell to ~0.15 nats by 20M steps and to
   *0.0* by the end. The policy had gone fully deterministic — it stopped
   exploring before it had found good play, then mechanically repeated a losing
   line. `ent_coef=0.01` was too low (SB3's default of 0.0 would've been worse).
2. **A misaligned proxy.** The reward shaping at the time was *absolute* cities
   powered (`own_powered × 0.01` per round). That rewards "power lots of your
   own cities," which a comfortable 2nd-place finisher does perfectly well.
   Winning Power Grid is *relative*.

And a humbling detail: ~21% win at egc=3 in a **4-player** game is *below the
25% you'd get from random play.* The best this run ever managed was worse than a
coin-flip-with-four-sides. The curriculum also marched on by a fixed step
schedule regardless of whether the agent had mastered the current stage — so it
dragged an incompetent policy into ever-harder games.

**Learnings:**
- **Watch entropy like a vital sign.** Collapse to ~0 = exploration is dead.
- **Watch `explained_variance` + `value_loss` together.** When EV→1, value_loss
  →0, *and* eval is flat, the critic has perfectly learned a *constant* outcome
  ("I always lose"). That's not convergence, it's a dead gradient.
- **Anchor win-rate to the random baseline (1/num_players), not to zero.**
- **Fixed-step curricula hide failure.** A mastery-gate ("advance only when you
  can win the current stage") would have stalled at egc=3 and screamed that
  stage 1 was never solved — which is the information you wanted.

### Interlude — fixing the two causes

We made two changes:
- Raised `--ent-coef` default 0.01 → 0.03 (and used 0.05 in the next run) to
  keep exploration alive.
- Switched shaping from absolute to **relative**:
  `(own_powered − best_opponent_powered) × 0.01`. The intent: reward
  *out-powering the field*, which is what winning actually is. (Implemented in
  Rust's `step_vs_bots`, which now returns the opponents' best powered count too,
  so Python can compute the difference with no extra round-trip.)

There was a good discussion here about whether relative shaping would bias the
agent toward *taking the lead too early* — which is a real Power Grid mistake,
since the leader gets punished by turn order (goes first in the auction, last in
buying/building). The plan was to anneal shaping away once the agent was
competent, so the final policy could discover positional play like deliberately
hanging back for cheap resources. Hold that thought — reality had other ideas.

### Episode 4 — `selfplay_w128_egc17_relshape`: the right fix to the wrong problem

Setup: relative shaping, **flat egc=17** (no curriculum — we wanted to test
whether the better-aligned reward could carry it), `ent_coef=0.05`, 32 envs,
120M steps.

The good news: **the entropy fix worked perfectly.** Entropy held ~0.85 nats
for the *entire* run. No collapse. `explained_variance` climbed healthily to
0.85. The training dynamics looked great.

The bad news: **0% win for all 150 evals.** Never once above −1.0.

So we stopped staring at metrics and looked at *behavior* (`evaluate.py`). The
agent finished **dead last in ~100% of games — even against the *easy* bots** —
and the stats told the whole story: it bought ~3 plants (enough capacity to
power ~14 cities) but built only ~5 cities, powered ~2.5, and sat on a pile of
**40–80 elektro**. A passive money-hoarder. It had never learned to *build*. And
checking checkpoints across the run, it was like this from 8M steps onward — it
never even tried.

Why? The very change we were proud of:

- **Relative shaping is a terrible *teacher* from a cold start.** At flat egc=17,
  a fresh agent is hopelessly behind the build-racing bots every single round,
  so `own − best_opp` is always strongly negative and dominated by the opponent
  term it can't control. The faint "+0.01 for powering one more city" is buried
  in that noise. The *old absolute* shaping, for all its flaws, gave a clean,
  always-positive "build more → more reward" signal — and it had demonstrably
  taught building (that 21% at egc=3 was a *building* agent).
- **Flat egc=17 offers no winnable games to learn from.** vs-bots episodes are a
  constant −1 (no gradient); self-play episodes are against equally-passive
  copies of itself (no pressure). Nothing anywhere said "build."

The irony writes itself: we'd worried relative shaping would make the agent lead
*too much*. Instead it removed the signal to compete *at all*.

**The big learning — write this one on your hand:** *a proxy reward that is
better **aligned** with the goal can be a worse **teacher** for it.* Absolute
powered-cities is the teacher (clean, dense, always-positive: "build!").
Relative powered-cities is the finisher (aligned with winning, but only useful
once you already know how to build). These are two different jobs and they want
**two different phases**, not one compromise.

### Interlude — the shaping-mode switch

We added `--shaping-mode {absolute,relative}` (default `absolute`) so we can run
the two phases explicitly. It's Python-only — the Rust step already returns both
the learner's powered count and the opponents' best, so the env just chooses
whether to subtract. Tests cover both modes (absolute is always ≥0; relative can
go negative). Next run is designed but not yet run.

---

## The cheat sheet I wish I'd had

If you read nothing else, read this.

| Signal | Where | What it means |
|---|---|---|
| **eval/mean_reward** | tensorboard / `run_report.py` | The only truth. `win_rate = (reward+1)/2`. Pinned at −1.0 = never wins. |
| **policy entropy** (`entropy_loss`) | tensorboard | Exploration's pulse. Drifting toward ~0 = collapse → raise `ent_coef`. |
| **explained_variance + value_loss** | tensorboard | EV→1 & loss→0 *with flat eval* = critic learned a constant ("I lose"). Dead gradient, not success. |
| **random baseline** | arithmetic | `1/num_players`. With 4 players, anything ≤25% win is ~noise, not skill. |
| **behavioral stats** | `scripts/evaluate.py` | cities/powered/money/placement. *Look at these.* "44 money, 5 cities" diagnosed Episode 4 in one line. |

Principles, earned the hard way:

1. **Suspect the plumbing first.** A pinned constant eval is more often a
   correctness bug (reward attribution) than a bad hyperparameter.
2. **Metrics describe, behavior explains.** When the numbers are baffling, run a
   game and watch what the agent actually *does*.
3. **A proxy is not the objective.** And: alignment and teachability are
   different axes. Bootstrap with the teachable proxy; fine-tune toward the
   aligned one.
4. **Make the task reachable, then make it hard** — but *gate the "make it hard"
   on actual mastery*, or you'll drag an incompetent policy into deep water.
5. **Change one variable at a time.** Episode 4 isolated the entropy fix
   cleanly, which is exactly why we could see that building was the *next*
   problem. Slow is fast.
6. **Self-play can chase its own tail.** Keep real bots in the mix
   (`--bot-mix`) and keep an external eval, or it'll get good at beating a bad
   version of itself.

Your tools: `run_report.py` (run health at a glance), `evaluate.py` (behavior +
win rate), tensorboard (curves), and `powergrid-netviz` (stare at the actual
forward pass when you really need to understand a decision).

---

## Where we are now (2026-06-21)

- ✅ Pipeline works (train → export `PGRLPOL1` → native Rust inference).
- ✅ Frozen-opponent self-play (reward attribution fixed).
- ✅ Entropy collapse fixed (`ent_coef` 0.05 holds ~0.85 nats).
- ✅ `--shaping-mode {absolute,relative}` implemented and tested.
- ❌ No agent yet beats the heuristic bots. Current best ever: ~21% at the
  easiest setting (still below random). The flagship runs sit at 0%.
- 🔎 Current best hypothesis: the agent never learned the **build→power→win**
  loop because we never gave it both a *teachable* reward and *winnable* games at
  the same time.

### Next steps (designed, not yet run)

A **two-phase** run, combining everything that worked:

- **Phase 1 — bootstrap.** `--shaping-mode absolute` + `--curriculum-start 3`
  (curriculum is safe again now that entropy won't collapse) + `--ent-coef 0.05`
  + `--bot-mix 0.5`. Goal: teach it to build and power near the bots. Gate
  success on eval win-rate clearing ~35–40% (well above the 25% baseline), and
  on `evaluate.py` showing real cities built and the money-hoard shrinking — not
  just a rising shaped-reward curve.
- **Phase 2 — fine-tune.** `--resume-from` the phase-1 model with
  `--shaping-mode relative --end-game-cities 17`, to teach positional, win-
  oriented play. Possibly a phase 3 with `--no-reward-shaping` for pure win/loss.

Open questions still circling:
- Is egc=3 actually "easy," or is a 3-city race so luck-dominated that no policy
  can beat ~random there? ("Fewer cities = easier" is an assumption, not a fact.)
- Should the curriculum advance on a **mastery gate** instead of a fixed step
  schedule? (Almost certainly yes — it would turn silent failure into a loud
  signal.)
- Does relative shaping really bias toward over-leading once the agent *is*
  competent — and is annealing it away the right cure?

We'll find out. Append the next episode here when we do.
