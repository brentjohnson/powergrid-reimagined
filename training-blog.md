<!--
MAINTENANCE NOTE TO CLAUDE (future self): This is a living document.
Every time we run a new RL training experiment or change the training setup,
append a new dated entry to the "Experiment log" section below (newest at the
bottom), and fold any durable lesson into "The cheat sheet". Keep the voice:
a typical explanatory blog post for a reader who is curious but new to
reinforcement learning — define jargon in plain language the first time it
appears, use "we" for what the project did. Record failures as plainly as
successes — the failures are where the learning is. Update the "Where we are
now" section to point at the latest state.
-->

# Teaching a neural network to play Power Grid (and mostly failing, instructively)

This is the running story of an attempt to train a *reinforcement learning* bot
to play the board game **Power Grid** well enough to beat the hand-written
opponents already in the game. It's written for someone who finds this
interesting but doesn't live and breathe machine learning — so I'll explain the
jargon as we go. The short version so far: the easy parts were easy, the hard
part is genuinely hard, and the failures have taught us more than any success
would have.

A quick orientation on the terms, since we'll lean on them:

- **Reinforcement learning (RL)** — instead of showing the computer labeled
  examples, you let it *play*, give it a reward when it does well, and let it
  adjust itself to earn more reward over time. Think training a dog with treats,
  except the dog is a neural network and the treats are numbers.
- **Policy** — the network that, given the current game state, decides what move
  to make. Training = improving the policy.
- **PPO / MaskablePPO** — the specific, popular RL algorithm we use (via the
  Stable-Baselines3 library). The "Maskable" part just means it respects the
  game's legal-move rules so it never tries an illegal action.

## What we're actually building

The goal is an "Expert" bot that plays Power Grid well. The approach: train a
policy network in Python, then export its weights to a tiny custom format and
run the finished bot natively in the game's Rust engine — so at game time there's
no Python, no heavyweight ML library, just fast matrix math. The bot sees the
game as a 454-number summary (money, plants, cities, the market, etc.) and
chooses from 143 possible actions.

That whole pipeline — train, export, load, play — **works.** This blog isn't
about the pipeline. It's about the much harder problem of getting the network to
actually *learn good play*.

## The problem nobody warns you about: sparse reward

Here's the crux of the difficulty. A full game of Power Grid is about **70
decisions long**, and the only truly meaningful reward is at the very end:
**+1 if you win, −1 if you lose.** One number, delivered once, after seventy
moves.

Imagine learning to cook by making an entire multi-course meal and being told
only "good" or "bad" at the end — no feedback on any individual step. Which of
the seventy things you did mattered? You have almost no way to know. This is
called the **sparse reward** problem, and it turns out that *almost every
technique in this story is a different trick for manufacturing more feedback*
when the real feedback is too rare and arrives too late.

And one principle worth tattooing somewhere visible: **the only honest measure
of progress is how often the bot actually wins against the real opponents.**
There are lots of intermediate numbers that can look great while the bot still
loses every single game. We got fooled by this repeatedly.

---

## The experiment log

### Episode 0 — Training against the bots: the hopeful beginning

The first attempt was the obvious one: have the network play directly against
the game's existing hand-written ("heuristic") bots and learn from the
win/loss reward. It ran, it produced a trained network, the network exported and
loaded into the Rust engine correctly. Success?

Not really — the resulting bot was weak, losing essentially every game against
the *normal*-difficulty opponents. But it proved the end-to-end machinery
worked, which matters.

**Lesson:** "it trains and exports without errors" is not the same as "it plays
well." We built an evaluation tool early (`scripts/evaluate.py`, which reports
win rate plus details like cities built, cities powered, and money held) and it
became the single most valuable thing we have.

### Episode 1 — Self-play, and a bug that quietly ruined everything

A natural next idea is **self-play**: instead of training against the fixed
bots, let the network learn by playing *against copies of itself*. As it
improves, its opponents improve too, in principle pulling it ever higher.

Our first version of this had every seat at the table controlled by the same
network at once. And it got stuck: the evaluation reward was pinned at −1.0
(meaning: loses 100% of the time) and for a while we couldn't see why.

The cause was subtle. The game only officially registers "game over" at the end
of a full round. Because of that timing quirk, the final +1/−1 reward was being
handed to whichever player happened to make the last little move of the round —
which was almost always the *losing* player, not the winner. In effect, **the
network was being told it lost on the very turns where it actually won.** No
amount of training can survive that; every model from this period was garbage.

We rebuilt self-play into a cleaner design (a "frozen-opponent" setup): the
network controls one seat, its opponents are *periodic frozen snapshots* of the
network from earlier in training, and crucially the reward is now attributed to
the right player.

**Lesson:** in a multiplayer game, *getting the reward to the right player* is a
correctness problem, not a tuning problem — and a bug there looks exactly like
"the algorithm just isn't working." When a metric is pinned at a suspiciously
perfect constant, suspect the plumbing before you start fiddling with settings.

### Episode 2 — A curriculum, because the reward wouldn't budge

Even with the bug fixed, the reward stayed stuck at −1.0. The reason is the
sparse-reward problem in its purest form: a beginner network playing a full-
length game basically never stumbles into a *win*, so it never receives a single
+1 to learn from. It's all stick, no carrot, ever.

So we tried **curriculum learning** — the RL version of "start with easy
homework." Power Grid normally ends when someone builds ~17 cities; we made the
game end at just **3** cities instead. Short games finish fast and produce wins
much more often, giving the network actual successes to learn from. The plan was
to start easy and gradually raise that target back toward the real number as the
network improved.

**Lesson (tentative at this point):** if the goal is unreachable, change the
task until it's reachable, *then* make it harder. The instinct was right; the
execution had problems, as the next episode shows.

### Episode 3 — A confident, total failure

Setup: a medium-sized network, the curriculum ramping the end-game target from
3 up to 17 cities, a 50/50 mix of real-bot and self-play opponents, over 121
million steps of training.

The result was almost poetic: **it got steadily worse the more it trained.** It
hit its best score early — about a 21% win rate while games were still at the
easy 3-city setting — and then declined all the way back down to 0% as the
curriculum made games longer. The final network lost every game.

Two root causes, both of which became permanent lessons:

1. **Exploration collapsed.** RL networks need to keep *trying varied moves*
   ("exploration") long enough to discover good ones, rather than prematurely
   committing to one rigid strategy. There's a number that measures this called
   **entropy** — high entropy means "still experimenting," near-zero means "doing
   the same thing every time." Ours fell almost to zero early on. The network
   locked in a strategy *before* it had found a good one, then mechanically
   repeated that losing strategy forever. The setting that controls how much
   exploration is encouraged (`ent_coef`) was simply too low.

2. **We were rewarding the wrong thing.** To fight sparse reward we'd been giving
   a small bonus each round for the number of cities the network *powered* — a
   technique called **reward shaping** (adding helpful intermediate rewards). But
   we used the *absolute* count, which rewards "power lots of your own cities" —
   something a comfortable second-place finisher does just fine. Winning Power
   Grid is **relative**: you have to out-power the others, not just do well in a
   vacuum.

A humbling footnote: that 21% peak was in a **four-player** game, where blindly
playing at random already wins about **25%** of the time. Our best result was
*worse than random.* On top of that, the curriculum advanced on a fixed timer
regardless of whether the network had actually mastered the easy stage — so it
kept shoving an incompetent player into harder and harder games.

**Lessons:**
- **Watch the entropy number like a heart-rate monitor.** Drifting toward zero
  means exploration is dying.
- **Two more numbers, read together, can fake you out:** "explained variance"
  and "value loss" describe how well the network's internal *predictor of its own
  fate* (the **critic**) is doing. They looked great here — but only because the
  critic had correctly learned to predict "I always lose." Confidently predicting
  a constant defeat is not progress.
- **Compare the win rate to random (1 ÷ number of players), not to zero.** Below
  ~25% in a 4-player game isn't skill, it's noise.
- **A fixed-timer curriculum hides failure.** Advancing only after the network
  proves it can win the current stage would have stalled at the easy stage and
  loudly signaled "stage one was never solved" — which is exactly what we needed
  to know.

### Interlude — fixing the two causes

We made two changes. First, we turned up the exploration incentive (`ent_coef`)
so the network would keep experimenting. Second, we switched reward shaping from
*absolute* to **relative**: instead of "+a bit for each city you power," it
became "+a bit for each city you power *beyond what your best opponent powered*."
The intent was to reward out-competing the field, which is what winning truly is.

There was a good debate here, worth recording: would this *relative* reward push
the network to grab the lead too aggressively? In real Power Grid, leading is
often a trap — the leader is penalized by turn order (acts first in the auction,
last when buying resources and building), so skilled players deliberately hang
back for cheap resources and surge at the end. The plan was to fade out the
shaping once the network was competent, so it could rediscover that subtle
positional play on its own. Reality, as usual, had other plans.

### Episode 4 — The right fix to the wrong problem

Setup: relative reward shaping, the higher exploration setting, games fixed at
the *full* 17-city length (no curriculum — we wanted to see if the better-aligned
reward could carry it alone), 120 million steps.

**The good news: the exploration fix worked beautifully.** Entropy stayed
healthy the entire run — no collapse. The critic's prediction quality improved
steadily. By every internal measure, training looked great.

**The bad news: a 0% win rate, for the entire run.** Not a single evaluation
above rock bottom.

So we stopped staring at graphs and did something we should reach for sooner:
we *watched the bot play*. The picture was damning and immediately clarifying.
The network finished **dead last in essentially every game — even against the
*easiest* bots** — and the breakdown explained itself: it bought a few power
plants (enough to power ~14 cities), but then built only ~5 cities, powered
barely 2, and sat on a giant **pile of unspent money**. It had become a passive
hoarder. **It never learned to actually build.**

Why? Ironically, the very change we were proud of:

- **The relative reward is a terrible *teacher* for a beginner.** In a full-
  length game, a fresh network is hopelessly behind the build-racing bots every
  single round — so "your cities minus the leader's cities" is always a big
  negative number dominated by something the network can't control. The tiny
  "+a bit for one more city" signal is drowned out. The *old absolute* reward,
  for all its flaws, gave a clear, always-positive "build more → get more reward"
  nudge — and it had genuinely taught the network to build (that earlier 21% was
  a *building* network).
- **Full-length games gave it nothing winnable to learn from.** Against the real
  bots it always lost (no useful signal); against frozen copies of itself, the
  copies were equally passive (no pressure to do better). Nothing, anywhere, was
  telling it to build.

The irony is almost too neat: we'd feared the relative reward would make the bot
*compete too hard for the lead.* Instead it erased the incentive to compete *at
all.*

**The biggest lesson of the whole project:** *a reward that is better **aligned**
with your true goal can be a worse **teacher** of it.* The absolute "build more"
reward is the teacher — dense, clear, encouraging. The relative "out-power the
field" reward is the finisher — it captures what winning means, but it's only
useful *after* the network already knows how to build. Those are two different
jobs, and they call for **two separate phases** of training, not one reward
trying to do both.

### Interlude — a switch to run both phases

We added a setting (`--shaping-mode`, choosing `absolute` or `relative`) so we
can deliberately teach with one reward and then fine-tune with the other. It's a
small change because the underlying engine already provides both numbers. This is
the setup for the next run, which we haven't done yet.

---

## The cheat sheet

If you skim one section, skim this. These are the dials and gauges that matter,
in plain terms.

| What to watch | What it tells you |
|---|---|
| **Win rate vs. the bots** | The only real measure of success. A reward "score" can climb while this stays at zero — don't be fooled. |
| **Entropy** | Whether the network is still exploring. Trending to zero = it's frozen into one strategy too early; turn up the exploration incentive. |
| **Critic accuracy** (explained variance + value loss) | How well it predicts its own outcome. Looks great even when the predicted outcome is "I always lose" — good prediction ≠ good play. |
| **The random baseline** (1 ÷ players) | The bar to beat. In a 4-player game, anything under ~25% is essentially luck, not skill. |
| **Just watching it play** | The fastest diagnosis there is. "Tons of money, almost no cities" explained an entire failed run in one glance. |

Hard-won principles:

1. **Suspect the plumbing first.** A metric stuck at a perfect constant is more
   often a bug (like reward going to the wrong player) than a bad setting.
2. **Numbers describe; watching explains.** When the metrics are baffling, go
   watch a game.
3. **A helper reward is not the real goal.** And being *aligned* with the goal is
   different from being good at *teaching* it — start with the teachable reward,
   finish with the aligned one.
4. **Make the task easy enough to learn, then harder — but only advance once it's
   actually mastered the easy version.**
5. **Change one thing at a time.** Isolating the exploration fix is exactly what
   let us clearly see that "doesn't build" was the *next* problem. Slow is fast.
6. **Self-play can chase its own tail** — it can get good at beating a weak
   version of itself while still losing to everyone else. Keep some real
   opponents in the mix and always measure against an outside benchmark.

## Where we are now

- ✅ The full pipeline works: train in Python → export → play natively in Rust.
- ✅ Self-play is sound (the reward-attribution bug is fixed).
- ✅ The exploration-collapse problem is fixed.
- ✅ We can now switch between the two reward styles at will.
- ❌ **No version has yet beaten the hand-written bots.** The best result ever
  was ~21% on the easiest setting — still below random — and the most recent
  full-scale runs sit at 0%.
- 🔎 Best current theory: the network never learned the core **build → power →
  win** loop, because we never managed to give it a *teachable* reward and
  *winnable* games at the same time.

### What we're trying next

A **two-phase** run that combines everything that worked:

- **Phase 1 — teach it to build.** Use the *absolute* reward (clear "build more"
  signal) plus the easy-games curriculum (now safe to use again, since
  exploration no longer collapses). Success means the win rate clearly clears the
  ~25% random bar, *and* watching it play shows real cities going up and the money
  hoard going down — not just a prettier reward graph.
- **Phase 2 — teach it to win.** Take that competent network and fine-tune it
  with the *relative* reward on full-length games, to learn the subtler
  positional play. Possibly a third phase with no shaping at all, optimizing
  purely for winning.

Open questions still circling:
- Is the 3-city game actually "easy," or is it so short and luck-driven that no
  amount of skill can beat random there?
- Should the curriculum advance based on *proven mastery* rather than a fixed
  timer? (Almost certainly yes.)
- Does the relative reward really over-encourage leading once the network is
  good — and is fading it out the right remedy?

We'll find out. The next episode gets written here when we run it.
