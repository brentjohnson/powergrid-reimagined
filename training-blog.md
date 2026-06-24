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
chooses from 94 possible actions. (This was 143 for most of the events below —
see Episode 8 for why it shrank.)

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
small change because the underlying engine already provides both numbers.

### A fork in the road: trying a completely different algorithm

That two-phase plan was the obvious next move. Instead, we made a bigger bet
first. PPO had now failed four times in a row, each time in a *different* way
(weak baseline, a reward-plumbing bug, exploration collapse, a reward that
taught the wrong lesson) — and all four failures trace back to the same root
problem: a single +1/−1 number, seventy moves late, is a very thin trickle of
feedback to learn from. Rather than keep patching the same algorithm's
relationship with that thin trickle, we tried something built differently from
the ground up: **AlphaZero**.

AlphaZero pairs a neural network with an explicit **search** procedure called
**Monte Carlo Tree Search (MCTS)**. Before committing to a move, it mentally
plays out many possible continuations ("simulations" — typically dozens to
hundreds per move), using the network to judge which branches are worth
exploring and how good a resulting position looks, without playing any of them
all the way to the end. The move it actually plays is whatever search liked
best — not just whatever the raw network would have guessed on its own. PPO,
by contrast, has the network output a move directly from its own judgment, no
lookahead at all. The hope: that explicit "think a few moves ahead before
committing" step would inject far denser, more immediate signal than waiting
for one win/loss number per entire game.

It was also, usefully, an experiment in itself: if a completely different
algorithm — different training loop, different update rule, the only shared
code being the Rust game engine and the 143-action move encoding — *also*
collapsed to 0%, that would be a meaningful clue that the problem wasn't
specific to PPO at all. That clue is exactly what happened, though not for the
reason we expected at the time.

### Episode 5 — First AlphaZero run: 0%, again

We built a separate implementation (following an established public AlphaZero
pattern) reusing only the Rust engine and the action encoding — no other code
shared with the PPO work. First real run: 200 training iterations, 25
self-play games per iteration, 50 search simulations per move, with the same
kind of curriculum as before ramping the end-game city trigger from 5 up to
the full 17.

Result: **0% win rate against even the easiest heuristic bots**, at every
single checkpoint we saved — iteration 1, 50, 100, 150, 200, and the run's
self-reported "best." In fact "best" turned out to just be iteration 1 — the
run never beat its own first attempt the entire time.

Same instinct as Episode 0: don't trust 0% at face value, go check the
plumbing first. A few read-only diagnostics, no code changes:

- The evaluation harness itself was fair — a heuristic bot dropped into the
  same seat against equal-strength opponents wins 17–37% of games (close to
  the 25% baseline four equal players would each get by chance), so the seat
  isn't rigged to lose.
- A **uniformly random** policy in that seat *also* won 0%. That told us our
  trained network wasn't merely weak — it had landed at the absolute floor,
  no better than blind guessing, against the weakest bots we have.
- The network itself wasn't broken or degenerate: its move probabilities
  stayed healthily varied (no entropy collapse this time), and its value
  head — the part that predicts who wins — had clearly learned to recognize
  the eventual *self-play* winner late in a game.

That last point pinned the diagnosis. The network had genuinely learned
something — just the wrong thing. It got good at reading the patterns of
games against earlier, weaker versions of itself, but that skill never once
had to face a truly competent opponent, so it had nothing useful to do
against the real bots. Left alone, self-play can wander off into a closed
little world of its own making rather than climbing toward real skill. We'd
also used only 50 search simulations per move — a thin amount of lookahead
for a 143-action game lasting 100+ moves, not nearly enough for the search
itself to meaningfully out-think a still-bad underlying network.

**Lesson:** self-play is not automatically self-correcting. Without something
actively pulling it toward genuinely good play — real opponents mixed in, or
far deeper search — it can become a confident expert at beating an
out-of-shape version of itself. This is Episode 1's self-play danger again in
a new costume, and the sharpest version of it yet: training that looks
*completely healthy* by every internal gauge (steady entropy, a value head
doing its job) can still be teaching exactly the wrong lesson, if the
opponents it's training against aren't the ones that actually matter.

### Episode 6 — Trying to shortcut the problem: cloning the hard bot directly

The fixes Episode 5 pointed to (deeper search, mixing in real opponents, and
more) were each expensive to test properly, and we didn't yet know which one
actually mattered. Before burning another full run finding out, we tried a
much cheaper, more direct diagnostic: **behavior cloning**.

Behavior cloning skips reinforcement learning's "try things and see what
scores well" loop entirely. You record a *competent player's* actual
decisions across many games — here, our own hand-written "hard" bot — and
train the network with ordinary supervised learning to imitate those exact
moves. It's "watch the expert and copy them," not reinforcement learning at
all — a great fast warm-start if it works, and a great diagnostic if it
doesn't.

We generated 400 games of the hard bot playing itself (four hard-bot seats,
~156,000 individual decisions in total) and trained the network to match each
one. After 20 passes over that data, the network correctly predicted the hard
bot's actual move **73% of the time** on moves outside its training data — a
genuinely strong score for this kind of task.

Then we let it actually play real games. **0% win rate.** Not "low" —
zero, the exact same wall as every previous attempt, despite faithfully
imitating a genuinely good player nearly three times out of four.

We re-checked the obvious suspects before accepting that: always playing the
single most-likely move, sampling randomly in proportion to the network's
confidence, even bolting real MCTS search on top of the cloned network with
up to 200 simulations per move. All 0%.

**Why a 73%-accurate copy can still lose every game:** this is the
**compounding error** problem, and it's a genuinely different failure mode
from sparse reward. Imagine copying a chef's recipe with 73% per-step
fidelity: get one early step slightly wrong — a bit too much salt — and every
later step the chef intended no longer quite applies; you're now improvising
in a situation the chef never actually faced. One misstep early in a
~70-move game can push play into a state the hard bot itself would never have
created, and from there the clone has no idea what a good "hard bot" move
even looks like, because it never saw anything resembling that state during
training. Small per-step errors compound, multiplicatively, over a long game
into near-certain derailment by the end — exactly why long-horizon tasks are
unusually punishing for plain imitation, even imitation that looks excellent
move by move.

**Lesson:** per-step accuracy and full-game competence are *different things
that can disagree wildly.* A 73%-accurate clone that loses literally every
game is the cleanest demonstration of that gap this whole project has
produced.

### Episode 7 — The actual culprit, finally: a one-unit-per-turn bug

Three genuinely different approaches — direct PPO, self-play AlphaZero, and
now behavior cloning — had each independently bottomed out at essentially 0%.
Three different algorithms hitting the *exact same wall* is a strong hint the
wall isn't about any one algorithm. So we asked a different question: is
there something **structural** about the game's 143-action move encoding
itself that makes it hard for *any* learned policy to play well, no matter
how it's trained?

We went looking, and found a real bug — not in any learning code, but in the
interface between policy and game. Buying fuel resources (coal, oil, gas,
uranium) is supposed to let a player buy *several units in one turn* — e.g.
"3 coal and 2 oil" in one trip to the market, exactly what the hand-written
bots do. But the 143-action encoding had only **one action id per resource
type**, and that single action was wired to a game-engine function that
quietly **ended the player's entire turn** after buying just one unit, as a
side effect of how it had been written.

The consequence: every policy we trained could buy **at most one unit of
fuel per round**, while every heuristic opponent it played against bought
normal multi-unit batches. A power plant starved of more than half its
needed fuel can't power its cities; a player who can't power cities can't
win. This wasn't a subtle disadvantage — it's closer to playing with one
hand tied behind your back — and it applied identically whether the "hand"
belonged to PPO, AlphaZero, or a behavior-cloned copy of our own best bot. It
explains, in one stroke, every single 0% result this entire project has
produced.

Our diagnostic instincts along the way (the eval harness, the random/heuristic
floor, the entropy and value-head checks) were genuinely useful and correctly
ruled out plenty of other explanations — but they could only ever tell us
"this isn't working," never "the move encoding itself is broken," because
none of the algorithms we'd tried were actually the problem.

### The fix

We changed the encoding so each "buy resource" action adds exactly **one
unit and lets the turn continue** — the same action can be chosen again,
buying another unit, as many times as wanted, before an explicit "done
buying" action ends the turn. This mirrors how *building cities* already
worked in the encoding (one action per city, then a separate "done building"
action) — so the fix brings buying resources in line with a pattern that was
already correct elsewhere. Nothing needed to change about which moves are
*legal* at each moment (the existing checks already correctly accounted for
plant capacity, market stock, and the game's hybrid gas/oil plants) — only
what a single action *does*.

We made the same change in both places the move encoding lives (the Rust game
engine, and the separate Python copy used for PPO training) and added an
automated test that plays a multi-unit purchase across several resources in
one turn, proving the turn no longer ends early.

**Important consequence:** every checkpoint trained so far — every PPO run,
the AlphaZero run, the behavior clone — was trained against the *broken*,
one-unit encoding. None of them are valid evidence about whether their
underlying algorithm can actually work on this game. We have, in effect, been
grading three different students on an exam with a typo that made every
answer come out wrong. The fix doesn't tell us PPO or AlphaZero will now
succeed — it means we finally get to ask that question for the first time.

### Episode 8 — The fix actually works, and a new, smaller bug: the agent over-bids

It worked. A self-play run trained after the buy-resource fix produced the
first checkpoint that plays *real* games: **30% win rate** against three
Normal bots in a 4-player game (previous best, pre-fix, was under 1%), with
realistic city and money counts and no stalled games. The bug really had been
the whole story for Episodes 0 through 7.

With a checkpoint that finally played competently, a new and much subtler
problem showed up on inspection: watching it play, the agent would jump a
plant auction's price up by 20 or 30 Elektro in a single move, where a human
(and every hand-written bot) raises a standing bid by the smallest amount that
keeps them in the auction — "+1 and see what happens" — because winning the
same plant for less leaves more money for everything else later in the round.

First question, as always: bug or learned behavior? It turned out to be
neither a rules bug nor an encoding bug exactly — both the game rules and the
move encoding *intentionally* allow a bid to jump by any amount up to the
player's cash (the encoding offered 50 different raise sizes, +1 through +50
over the standing bid, as separate action choices). Nothing was broken; the
agent was just freely picking from a menu of jump sizes that humans, by
convention rather than rule, never use.

Self-play had no particular reason to discover the "+1" convention on its own.
The convention exists for humans because overpaying is a real cost — but in
self-play, "overpaying" was a soft, delayed signal (less cash, sometime later,
maybe), while winning the auction was an immediate, obvious win. With 50
near-equivalent jump-size buttons and only a faint reward gradient telling them
apart, the policy had little pressure to ever prefer the small one. This was
also the second time in this project a single large degree of freedom in the
*action space* — not the algorithm, not the reward — quietly made good
strategy harder to find than it needed to be (Episode 7's one-unit fuel-buy
bug was the first, though that one really was a bug; this one is intentional
design that simply gave self-play more rope than it needed).

**The fix:** rather than hope more training or better shaping would eventually
teach frugality, we removed the temptation structurally. The 50 bid-raise
actions collapsed into a single one — raise by exactly +1 over the standing
bid (pass still covers dropping out) — turning every price level into the one
decision that actually matters: *is this plant worth one more Elektro, yes or
no?* Any final price still reachable by jumping is equally reachable by
repeated +1s, since the bidding queue revisits every remaining player at each
price step; the only thing removed is the jump itself, which a sequential
agent never needed anyway. Action count dropped from 143 to 94 as a result —
every checkpoint trained before this point, including the 30%-win-rate one
above, is now incompatible with the encoding and must be retrained.

**Lesson:** when an action space offers more freedom than the real decision
actually requires, don't assume training will discover the "obviously sane"
restriction on its own — self-play has no instinct for human convention, only
for whatever the reward gradient happens to favor. If a degree of freedom only
exists to be thrown away by every competent player anyway, removing it from
the encoding is cheaper and more reliable than teaching an agent not to use it.

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
7. **If wildly different algorithms all fail the same way, suspect the shared
   interface, not the algorithms.** PPO, AlphaZero, and plain supervised
   behavior cloning are about as different from each other as three
   approaches get. When all three landed at the same 0%, the far more likely
   explanation was something they all depended on in common — here, the move
   encoding — not three unrelated algorithmic failures.
8. **A high-accuracy clone that still loses every game is itself a strong,
   specific signal** — it points at compounding error / a broken interface,
   not just "imitation didn't work." Behavior cloning is worth doing as a
   cheap diagnostic even when it isn't the intended final approach.
9. **An action that's supposed to represent "buy a few of these" can hide a
   bug if it secretly ends the turn after the first one** — and that kind of
   bug is invisible to every metric that only watches the *learning*, because
   the hand-written bots never go through that same encoding at all.
10. **Don't expect self-play to discover a human convention the reward doesn't
    require.** If a degree of freedom in the action space (e.g. "jump the bid
    by any amount, not just +1") only exists because the rules permit it, not
    because a good player would ever use it, an agent has no particular reason
    to avoid it on its own — the cost of using it shows up too late and too
    faintly. When you spot this, constrain the encoding to match the real
    decision instead of hoping training learns the restriction.

## Where we are now

- ✅ The full pipeline works: train in Python → export → play natively in Rust.
- ✅ Self-play is sound (the reward-attribution bug is fixed).
- ✅ The exploration-collapse problem is fixed.
- ✅ We can now switch between the two reward styles at will.
- ✅ Tried a structurally different algorithm (AlphaZero/MCTS, Episode 5) —
  it also failed at 0%, but the diagnosis (a closed self-play loop, too
  little search) was genuinely informative and ruled out "PPO specifically is
  broken" as the only explanation.
- ✅ Tried behavior cloning the hard bot as a cheap diagnostic (Episode 6) —
  73% per-move accuracy, still 0% win rate — which is what sent us looking
  for something structural rather than algorithmic.
- ✅ **Found and fixed a real bug** (Episode 7): the move encoding could buy
  at most one unit of fuel per turn no matter which algorithm was driving it,
  while every opponent bought normal multi-unit batches. Fixed to be
  additive (buy one unit, turn continues) across both the Rust engine and the
  Python training copy of the encoding.
- ✅ **First real, non-degenerate checkpoint:** a self-play run trained after
  the buy-resource fix hit 30% win rate vs three Normal bots (4-player game) —
  full games, no stalls, realistic money/cities. Every prior 0% result really
  did share the one explanation above.
- ✅ **Found and fixed a second, smaller bug-shaped problem** (Episode 8): that
  checkpoint had learned to jump auction bids by large, non-strategic amounts
  instead of the human "+1 and see" convention. Not a rules or encoding bug —
  both intentionally allowed any jump up to the player's cash — but self-play
  had no reason to discover the human convention on its own. Collapsed the
  50-action bid-raise range to a single +1 action; action count is now 94
  (was 143). Every checkpoint trained before this point is invalid and must be
  retrained.
- ❌ **No version has yet been retrained against the cleaned-up 94-action
  encoding.** That's the next run.

### What we're trying next

The buy-resource fix (Episode 7) worked — the first real, non-degenerate
checkpoint came out of it. But that checkpoint was trained against the
50-action bid-raise range Episode 8 just collapsed to a single +1 action, so
it's no longer valid evidence about anything either. The plan is the simplest
possible next step: retrain the same self-play setup, unchanged except for the
now-94-action encoding, and see whether win rate holds at ~30% or improves now
that jump-bidding isn't an option. If it holds or improves, that's confirmation
the bid fix was a net positive and not just a cosmetic change. If it regresses,
that's itself informative — it would mean the agent had been relying on
jump-bidding for something beyond what it looked like from the outside.

Open questions still circling from earlier episodes:
- Is the 3-city curriculum game actually "easy," or is it so short and
  luck-driven that no amount of skill can beat random there?
- Should the curriculum advance based on *proven mastery* rather than a fixed
  timer? (Almost certainly yes.)
- Does the relative reward really over-encourage leading once the network is
  good — and is fading it out the right remedy?

We'll find out. The next episode gets written here when we run it.
