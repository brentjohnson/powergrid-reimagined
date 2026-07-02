# Training Next Steps — Plateau / Regression Recovery

Without seeing the numbers I can't be specific about which run is which, but there are only a few levers available and two root causes dominate plateaus in self-play:

---

## Root cause 1: Entropy collapse

The policy became near-deterministic and stopped exploring. Check TensorBoard's `train/entropy_loss` — if it's near zero, this is the culprit.

Fix: Resume with `--ent-coef 0.1` (or even `0.15`). The jump is deliberate — `0.05` wasn't enough in run 8, and you need to actually destabilize the overconfident policy.

---

## Root cause 2: Self-play echo chamber

The frozen snapshot is too close to the current policy; they reinforce each other's strategies rather than challenging them.

Fix: Resume with `--bot-mix 0.5` or higher. If a run already has high bot-mix and still plateaued, try going the other way — drop to `0.0` so the snapshot *is* the only opponent (sometimes a very stale snapshot is actually a better teacher than heuristic bots at this stage).

---

## For runs that have actually regressed

The model drifted past its best_model. The most useful thing is cross-pollination: take the **best_model from your best-performing run** and apply the regressed run's hyperparameters to it. This tests whether the strategy was sound but the starting point was degraded, vs. the strategy itself being the problem.

---

## Triage order

1. Check entropy first (free information from TensorBoard)
2. If collapsed → `--ent-coef 0.1` on each stalled run
3. If not collapsed → raise `--bot-mix` on low-mix runs, lower it on high-mix runs (flip the knob)
4. If regressed → cross-pollinate from the best run's `best_model`
5. Kill any run that's been flat for 200M+ steps with all of the above tried — it found a local optimum that isn't escapable from that trajectory
