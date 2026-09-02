#!/usr/bin/env bash
#
# sweep_selfplay.sh — wave 17: OBS-624 MIGRATION + exploit the rollout-length win
# + two weight-space bets. TWO things happen this wave. (1) The observation grew
# 600 -> 624: section 23 adds four DERIVED per-actual-slot market decision
# features (affordable, min-bid, powering gain, is-upgrade) that sections 9/10
# left the net to reconstruct from raw attributes. Append-only, so it is
# checkpoint-safe — but a 600-wide sb3 .zip cannot be --resume-from'd under the
# 624 env, so (like wave 9's 582->600 reset) EVERY arm is a FRESH clone
# (--init-policy-from) of the MIGRATED champion .bin: migrate_policy_obs.py
# zero-pads the champion's l1 rows so it plays a3 bit-for-bit, and training then
# learns to use the 24 new inputs. (2) Wave 16 broke the two-wave plateau:
# a3-nsteps4096 (n-steps 4096) beat x5 and is the new champion. Wave 17 maps the
# rollout-length curve around it (2048 / 4096 / 8192), tests gae-lambda and an
# entropy sharpen, keeps a gentle-lr migration guard, and spends its two crazy
# slots on weight-space moves training cannot make: an EXTRAPOLATION past the
# champion and an SWA soup of this wave's survivors.
#
# WAVE 17 (2026-09-01). Fresh value heads everywhere (the migration cost), so
# the wave both RE-ESTABLISHES the champion under obs-624 and probes the levers.
#
# WAVE 16 (plateau-break: 1 control + 5 larger perturbations + 2 soups) is
# settled. All six a* arms hit 1050M; z6-soup finished (350M); z8-champ-soup
# stalled at 212M. Result: a3-nsteps4096 WINS — the first successor to x5 in
# three waves.
#
#   Frozen-champion eval (best mean_reward vs 3x x5-champ-g999, par ~-0.50):
#     a5-epochs-kl -0.26  a1-champ-cont -0.27  a3-nsteps4096 -0.28  z6-soup -0.29
#     a6-gae98 -0.30  z8-champ-soup -0.34(@212M)  a4-relative-shaping -0.36
#     a2-hi-lr-restart -0.43
#   Compare vs 3x hard (saturated, reporting-only, seat-0 all-bots par ~21.5%):
#     a3-nsteps4096 87.5  z8-champ-soup 85.5(@212M)  a4/a6 85.0
#     a1/a5 83.0  a2 81.0  z6-soup 80.5
#   Primary h2h (seat 0 vs 3x the frozen champion x5; mirror par 24.5%):
#     a3-nsteps4096 30.0  a6-gae98 26.0  z8-champ-soup 26.0(@212M)  z6-soup 25.0
#     a1-champ-cont 23.5  a4 23.5  a5 23.0  a2-hi-lr-restart 19.5
#   a3-nsteps4096 LED ALL THREE boards. A two-seed DECIDER vs the incumbent x5
#   (400 games/dir each, 4-way par 25%) confirmed it — a3's edge is sign-stable:
#     seed 161616:  a3 vs 3x x5 = 23.2  |  x5 vs 3x a3 = 21.8   -> a3 +1.4
#     seed 24242 :  a3 vs 3x x5 = 27.0  |  x5 vs 3x a3 = 24.2   -> a3 +2.8
#     combined (800g/dir): a3 offense 25.1 vs x5 offense 23.0   -> a3 +2.1
#   Unlike wave 15's z3 (led reporting, then the decider FLIPPED to x5), nothing
#   flips for a3. a3-nsteps4096 is the champion and the new embedded Expert.
#
# What wave 16 SETTLED / OPENED:
#   * *** ROLLOUT LENGTH IS A REAL LEVER. *** n-steps 1024->2048 (z3) flickered
#     and lost its decider; 1024->4096 (a3) wins. At gamma 0.999 the return
#     horizon is long and GAE truncates at n-steps, so longer rollouts cut
#     truncation bias / gradient variance. n-steps 4096 is now the champion
#     default; wave 17 brackets the PEAK (2048 / 4096 / 8192).
#   * gae-lambda 0.98 (a6) also edged x5 (h2h 26.0, decider a6 24.5 > x5 23.5) —
#     the SAME advantage-horizon idea from the other side. Kept and STACKED onto
#     the new champion (b4). Both live levers are horizon knobs, consistent with
#     the gamma-0.999 story.
#   * DEAD LEVERS (wave 16 losers, do not re-test): lr restart 3e-4 (a2, WORST
#     everywhere — x5 is a real basin, not an escapable local optimum), relative
#     shaping annealed off (a4, below par — terminal-only holds), n-epochs 8 +
#     target-kl (a5, below par). GAMMA STAYS 0.999. The 0.20 bot anchor stays.
#   * SOUPS sat at par (z6 25.0; z8 26.0 but only @212M). The soup family is
#     alive but not a breakthrough. Wave 17 REPLACES the stale x5-basin soups
#     (z6 mean(x2,x3,x4), z8 mean(x5,z3,z7)) with soups/merges in the NEW
#     champion's basin (c1 extrapolation, c2 wave-16 SWA). z6/z8 stay on disk,
#     not resumed.
#   * DONOR / CROSS-LINEAGE stays retired: s4-y3 (gamma 0.99) sits at 1050M and
#     the value-head gamma-continuity finding (wave 14) predicts a gamma-0.999
#     fork of it must re-learn the return scale — wave 15's z2 confirmed it lands
#     at par. No wave-17 arm forks the donor. It stays on disk as fork material.
#
# Budget: every arm is a FRESH clone from step 0 (fresh value head), so all share
# one WAVE_STEPS = 250M budget (up from the 150M fork increments — a fresh value
# head needs to refit before the lr-decay convergence peak in the back third, and
# there are 24 new inputs to learn). No fork/resume arms this wave, so
# TOTAL_TIMESTEPS is unused.
#
# THE STRUCTURE — six progress arms (1 control + rollout-curve + 2 levers + guard)
# + two crazies, ALL fresh clones of the migrated 624 champion:
#   * CHAMPION_BIN = the migrated a3 @624 (wave17-champion.bin, from
#     migrate_policy_obs.py --from-ckpt a3/best_model). It plays a3 bit-for-bit
#     (zero weights on the 24 new inputs) and is BOTH the warm start for b1-b6 AND
#     the frozen --eval opponent. The frozen --h2h yardstick is wave17-baseline
#     (the same weights as a 624 sb3 ckpt via --bin-to-ckpt). Neither trains.
#   * Every b-arm changes ONE axis from the b1 control so its effect is readable
#     on the shared peers/seeds. The champion default is n-steps/batch 4096.
#   * runs/sweep4 is this epoch's home. Inert history NOT peered: wave-9 (s*),
#     wave-10 (t*), wave-11 (u*), wave-12 (v*), wave-13 (w*), wave-14 (x*/y*),
#     wave-15 (z1-z7), wave-16 (a1-a6, z8) — all now the wrong (600) obs width.
#     runs/sweep3 is inert 582-format. Never resume, never peer any of them. Only
#     wave-17 arms (b1-b6 + the crazies c1/c2) peer.
#
# Arm roles (all gamma 0.999, mix 0.40/0.40/0.20, past_k 8, ent 0.03, fresh clone
# of the migrated 624 champion, lr 1e-4->0, n-steps/batch 4096, unless noted):
#   b1-champ-cont     CONTROL: re-establish the champion a3 under obs-624 (clone
#                     the migrated bin, fresh value head). The incumbent-in-the-
#                     field baseline every other arm is read against, and the
#                     reference for how much the 24 new market features buy.
#   b2-nsteps8192     Rollout curve (upper): b1 but n-steps/batch 8192 (2x). Push
#                     the winning lever one more notch — is longer still better,
#                     or is 4096 the peak? Higher gradient variance; watch it.
#   b3-nsteps2048     Rollout curve (lower): b1 but n-steps/batch 2048 (0.5x).
#                     Bracket the peak from below on the SAME peers as b1/b2, so
#                     the three points 2048/4096/8192 map one clean curve.
#   b4-gae98          Lever stack (advantage horizon): b1 but gae-lambda 0.98
#                     (from 0.95). Wave 16's a6 edged x5 with this; stack it onto
#                     the n-steps-4096 champion — do the two horizon levers add?
#   b5-low-ent        Lever (entropy sharpen): b1 but ent-coef 0.015 (validated
#                     champion-line lever, wave 11). Sharpen the converged policy.
#   b6-gentle         MIGRATION GUARD: b1 but gentle lr 3e-5 -> 0 (the wave-9
#                     recipe that won the last format reset). Every arm has a
#                     fresh value head over a warm policy; b6 vs b1 tells us
#                     whether the standard-lr clones are healthy (b6 ~ b1) or the
#                     fresh value head is damaging them (b6 > b1 -> go gentle).
#   c1-extrapolate    *** CRAZY #1: weight-space EXTRAPOLATION past the champion.
#                     *** Fresh clone from merge -0.5*x5 + 1.5*a3 (step 1.5x along
#                     the demonstrated x5->a3 improvement direction; a3 forks x5
#                     so the direction is well-defined), migrated to 624.
#                     Interpolating soups sit at par; extrapolation OVERSHOOTS
#                     along the improvement vector — a non-gradient move to weights
#                     beyond a3 that forking+annealing cannot reach (task
#                     arithmetic, Ilharco et al.). GENTLE lr 3e-5 -> 0 pulls it
#                     back into a good basin if it overshoots. If it collapses,
#                     alpha=1.5 is too aggressive; next wave retries a smaller step.
#   c2-swa-wave16     *** CRAZY #2: wave-16 survivors SWA soup. *** Fresh clone
#                     from mean(a1, a3, a6) — the three best-behaved wave-16 forks
#                     (all fork x5 @894M -> shared basin), migrated to 624, gentle
#                     lr 3e-5 -> 0.
#                     Stochastic-weight-averaging bet: does averaging this wave's
#                     survivors land in a flatter minimum than the champion a3
#                     alone? The current-basin replacement for the retired z8.
#
# Sized for a 28-core machine: 8 variants x THREADS=3 = 24 cores, leaving
# headroom for the eval passes and the OS.
#
# Re-running is idempotent and self-healing: launching is the same command as
# resuming. For each selected variant the script inspects its run dir and
# resumes from the furthest-along readable checkpoint; if there is none, a fork
# arm starts from its (pinned) fork source and a resume arm hard-errors rather
# than silently restarting. The running-check verifies the recorded PID is
# still a train_selfplay.py process for THIS run dir, so a stale pidfile can't
# block a resume or let two trainers write the same dir. Operational loop:
# run it; if the box reboots or an arm crashes, run it again.
#
# Resume-lr note: MaskablePPO.load is passed custom_objects built from THIS
# launch's flags, so an arm's lr schedule is its own across resumes (a fork's
# fresh lr 1e-4 overrides the decayed lr stored in the champion checkpoint).
#
# Usage:
#   ./scripts/sweep_selfplay.sh              # launch/resume all 8 in the background
#   ./scripts/sweep_selfplay.sh 3 5          # launch/resume only variants 3 and 5
#   ./scripts/sweep_selfplay.sh --prepare    # format guard + export the eval opponent, launch nothing
#   ./scripts/sweep_selfplay.sh --list       # show the variant table, launch nothing
#   ./scripts/sweep_selfplay.sh --status     # per-variant progress / best eval
#   ./scripts/sweep_selfplay.sh --compare    # ABSOLUTE: each variant vs 3x hard bots
#   ./scripts/sweep_selfplay.sh --h2h        # RELATIVE: each variant vs 3x the frozen champion (migrated a3 @624)
#   ./scripts/sweep_selfplay.sh --stop       # stop every running variant
#
# Env overrides (all optional):
#   WAVE_STEPS=250000000       per-arm budget; EVERY arm is a fresh clone from 0
#   CHAMPION_CKPT=runs/sweep4/a3-nsteps4096/best_model   600-wide source, migrated in --prepare
#   CHAMPION_BIN=runs/sweep4/wave17-champion.bin  migrated a3 @624: b1-b6 warm start + eval opponent
#   EVAL_OPPONENT=$CHAMPION_BIN   frozen champion @624 (par ~ -0.50)
#   BASELINE=wave17-baseline   synthetic run dir holding the migrated a3 @624 sb3 ckpt (--h2h yardstick)
#   EXTRAP=runs/sweep4/extrap-x5a3.bin   merge -0.5*x5 + 1.5*a3 @624 (c1 warm start), built in prepare
#   SOUP16=runs/sweep4/soup-a1a3a6.bin   averaged a1/a3/a6 @624 (c2 warm start), built in prepare
#   TOTAL_TIMESTEPS            unused this wave (no fork/resume arms)
#   BASELINE=a3-nsteps4096     run-dir name of the frozen --h2h opponent
#   SWEEP_DIR=runs/sweep4      root for the per-variant run dirs
#   NET_WIDTH=128              must match the fork checkpoints' hidden width
#   NUM_ENVS=8                 parallel envs per variant (keep equal across variants)
#   THREADS=3                  torch/OMP threads per variant (8 x 3 = 24 of 28 cores)
#   COMPARE_GAMES=200  COMPARE_SEED=12345
#   COMPARE_DETERMINISTIC=1    rank with argmax instead of sampling. Training is
#                              stochastic, so the sampled numbers remain primary.
#   NICE=10  STAGGER=15  DRY_RUN=1
#
set -euo pipefail

cd "$(dirname "$0")/.."          # python/

PY=${PY:-.venv/bin/python}
SWEEP_DIR=${SWEEP_DIR:-runs/sweep4}
# OBS-624 MIGRATION WAVE (like wave 9). The observation grew 600 -> 624 (section
# 23, per-actual-slot market decision features), so a 600-wide sb3 .zip cannot be
# --resume-from'd under the 624 env. Every arm is therefore a FRESH clone
# (--init-policy-from) of a MIGRATED policy .bin (fresh value head, step 0), the
# proven append-only path: migrate_policy_obs.py zero-pads the champion's l1 rows
# so the migrated net plays a3 bit-for-bit, then training learns to use the 24
# new inputs.
CHAMPION_CKPT=${CHAMPION_CKPT:-$SWEEP_DIR/a3-nsteps4096/best_model}   # 600-wide source (migrated in --prepare)
CHAMPION_BIN=${CHAMPION_BIN:-$SWEEP_DIR/wave17-champion.bin}          # migrated a3 @624: warm start for b1-b6 AND the eval opponent
FORK_FROM=${FORK_FROM:-$CHAMPION_CKPT}   # generic fork source for the launch loop; UNUSED this wave (every arm is a clone)
EVAL_OPPONENT=${EVAL_OPPONENT:-$CHAMPION_BIN}
                             # Frozen champion at 624 = the migrated a3 (plays a3
                             # exactly). Selects best_model; par ~ -0.50.
WAVE_STEPS=${WAVE_STEPS:-250000000}   # every arm is a fresh clone from step 0, so all share this per-wave budget
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-$WAVE_STEPS}   # unused this wave (no fork/resume arms); kept for the generic launch loop
EXTRAP=${EXTRAP:-$SWEEP_DIR/extrap-x5a3.bin}   # extrapolation -0.5*x5 + 1.5*a3 @624 (c1 warm start), built in prepare
SOUP16=${SOUP16:-$SWEEP_DIR/soup-a1a3a6.bin}   # wave-16 SWA soup mean(a1,a3,a6) @624 (c2 warm start), built in prepare
NET_WIDTH=${NET_WIDTH:-128}
NUM_ENVS=${NUM_ENVS:-8}
THREADS=${THREADS:-3}
NICE=${NICE:-10}
STAGGER=${STAGGER:-15}
DRY_RUN=${DRY_RUN:-0}

# The frozen --h2h opponent: the champion a3-nsteps4096, MIGRATED to obs-624.
# a3's own best_model.zip is 600-wide and cannot play under the 624 env, so
# --prepare rebuilds it as a 624 checkpoint (wave17-baseline/best_model, via
# migrate_policy_obs.py --bin-to-ckpt — policy loaded, value head fresh, which is
# irrelevant to --h2h/--compare since only the policy logits pick actions). It
# plays a3 bit-for-bit and never trains, so this yardstick never moves. It is the
# embedded Expert (also migrated to 624).
BASELINE=${BASELINE:-wave17-baseline}

# Shared across all variants — held constant so the comparison is clean.
# A variant's own flags come after these on the command line, so repeating a
# flag there overrides the value set here.
COMMON=(
    --num-players 4
    --num-envs "$NUM_ENVS"
    --net-width "$NET_WIDTH"
    --no-reward-shaping         # terminal reward is the objective (wave 3's
                                # shaped arm was neutral-to-negative)
    --eval-opponent "$EVAL_OPPONENT"
                                # selects best_model vs the frozen champion.
                                # Par ~25% == mean_reward ~ -0.50, so best= in
                                # --status is NEGATIVE by design.
    --save-freq 250000          # ~2M timesteps per checkpoint at 8 envs
    --eval-freq 50000           # ~400k timesteps per eval pass
    --eval-episodes 200         # 20 (the trainer default) is too noisy to rank
)

# name|seed|init|pop|hypothesis|extra flags
#
# `init` is "resume" (own dir; hard-error if it has no checkpoint), "fork"
# (first launch --resume-from $FORK_FROM into a fresh dir), "fork=<stem>"
# (first launch --resume-from that checkpoint stem instead), or "clone" (first
# launch is a FRESH run warm-started from a policy .bin via --init-policy-from
# — the crazy model-soup arms z6/z8, whose --init-policy-from lives in their
# extra flags and is ignored by the trainer on any later --resume-from; the
# soups are 128-wide so --net-width comes from COMMON). After the first launch
# every arm resumes its OWN checkpoints (so z6 continues its wave-15 190M dir).
#
# `pop` is "peers" (the launch loop appends --league-peers with every OTHER
# variant's league dir) or "solo" (no peers).
#
# League mix order is LATEST,PAST,BOTS; the trainer default is 0.5,0.3,0.2.
# The 0.20 bot share is load-bearing (wave-7 q6 faceplant) — do not cut it.
VARIANTS=(
"b1-champ-cont|1701|clone|peers|CONTROL / anchor: re-establish the champion a3 under obs-624. A FRESH clone (--init-policy-from the migrated champion bin -> plays a3 bit-for-bit, fresh value head) at the champion recipe: lr decay 1e-4 -> 0 + gamma 0.999 + ent 0.03 + n-steps/batch 4096. The incumbent-in-the-field baseline every other arm is read against on the same peers/seeds; also the reference for how much the 24 new market features buy once the value head refits. Collapse guard: entropy diving toward ~0.1 nats = kill the arm.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --init-policy-from $CHAMPION_BIN --n-steps 4096 --batch-size 4096 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"b2-nsteps8192|1702|clone|peers|ROLLOUT CURVE (upper): b1 but n-steps 8192 / batch 8192 (2x the champion's 4096). n-steps 1024->2048 flickered (z3), 1024->4096 WON (a3); push the winning lever one more notch. Longer rollouts cut GAE truncation bias at the gamma-0.999 horizon but raise gradient variance and halve the update count per budget — the direct test of whether 4096 is the peak or a waypoint. Clone of the migrated champion, gamma 0.999.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --init-policy-from $CHAMPION_BIN --n-steps 8192 --batch-size 8192 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"b3-nsteps2048|1703|clone|peers|ROLLOUT CURVE (lower): b1 but n-steps 2048 / batch 2048 (0.5x). Bracket the rollout-length peak from BELOW on the same peers/seeds as b1 (4096) and b2 (8192), so 2048/4096/8192 map one clean curve. z3's 2048 lost its decider off x5; re-measure it here off the new champion a3 to place the curve's left shoulder. Clone of the migrated champion, gamma 0.999.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --init-policy-from $CHAMPION_BIN --n-steps 2048 --batch-size 2048 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"b4-gae98|1704|clone|peers|LEVER STACK (advantage horizon): b1 but gae-lambda 0.98 (default 0.95). Wave 16's a6-gae98 edged x5 (h2h 26.0; decider a6 24.5 > x5 23.5) — the same long-horizon idea from the advantage side. Stack it onto the n-steps-4096 champion: at gamma 0.999 lambda 0.95 still discounts advantages on a ~20-step scale, so lifting to 0.98 lengthens advantage estimation toward the discount horizon. Do the two horizon levers (rollout length + GAE lambda) add? Clone of the migrated champion, gamma 0.999, n-steps 4096.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --gae-lambda 0.98 --init-policy-from $CHAMPION_BIN --n-steps 4096 --batch-size 4096 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"b5-low-ent|1705|clone|peers|LEVER (entropy sharpen): b1 but ent-coef 0.015 (half the 0.03 default). Validated as a champion-line lever in wave 11 (u4-sharp beat plain-decay both directions); re-arm it on the new n-steps-4096 champion to sharpen the converged policy. Watch entropy: a dive toward ~0.1 nats = kill. Clone of the migrated champion, gamma 0.999, n-steps 4096.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.015 --gamma 0.999 --init-policy-from $CHAMPION_BIN --n-steps 4096 --batch-size 4096 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"b6-gentle|1706|clone|peers|MIGRATION GUARD (fresh-value-head control): b1 but GENTLE lr 3e-5 -> 0 (the wave-9 recipe that won the last format-reset). Every arm this wave has a fresh value head over a warm (migrated) policy, so early updates ride on meaningless advantages that can wreck the clone; a gentler peak lr protects it while the value head refits. b6 vs b1 answers whether the standard lr 1e-4 clones are healthy (b6 ~ b1) or the fresh value head is damaging them (b6 > b1) -> then the whole wave should go gentle. Clone of the migrated champion, gamma 0.999, n-steps 4096.|--learning-rate 3e-5 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --init-policy-from $CHAMPION_BIN --n-steps 4096 --batch-size 4096 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"c1-extrapolate|1707|clone|peers|CRAZY #1 — weight-space EXTRAPOLATION past the champion. A FRESH clone warm-started (--init-policy-from) from extrap-x5a3.bin = merge(-0.5*x5-champ-g999 + 1.5*a3-nsteps4096), i.e. step 1.5x from the old champion x5 along the DEMONSTRATED x5->a3 improvement direction (a3 forks x5 @894M so the two share a coordinate system and the delta is meaningful). Interpolating soups sit at par (z6/z8); extrapolation OVERSHOOTS along the improvement vector to weights BEYOND a3 that forking+annealing cannot reach (task arithmetic, Ilharco et al.). GENTLE lr 3e-5 -> 0 + gamma 0.999 pulls it back into a good basin if the overshoot degrades it. If it collapses, alpha=1.5 is too aggressive; next wave retries a smaller step. n-steps 4096. WAVE_STEPS (250M) from 0. scripts/make_merge.py builds extrap-x5a3.bin in --prepare.|--learning-rate 3e-5 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --init-policy-from $EXTRAP --n-steps 4096 --batch-size 4096 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"c2-swa-wave16|1708|clone|peers|CRAZY #2 — the wave-16 survivors SWA soup. A FRESH clone warm-started from soup-a1a3a6.bin, the UNIFORM WEIGHT-SPACE AVERAGE of the three best-behaved wave-16 forks a1-champ-cont / a3-nsteps4096 (champion) / a6-gae98 — all fork x5 @894M so they share ONE loss basin and are a valid soup (Wortsman et al.). Replaces the retired stale-basin z8. Stochastic-weight-averaging bet: does averaging this wave's survivors land in a flatter/better minimum than the champion a3 alone? GENTLE lr 3e-5 -> 0 + gamma 0.999 + n-steps 4096. WAVE_STEPS (250M) from 0. scripts/make_soup.py builds soup-a1a3a6.bin in --prepare.|--learning-rate 3e-5 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --init-policy-from $SOUP16 --n-steps 4096 --batch-size 4096 --league-mix 0.40,0.40,0.20 --league-past-k 8"
)


variant_field() { echo "${VARIANTS[$1]}" | cut -d'|' -f"$2"; }

# Comma-separated league dirs of every variant EXCEPT $1 (by name) — the
# --league-peers value for a "peers" arm. Dirs may not exist yet; the trainer
# tolerates that (empty pool slice until the peer launches). Only wave-17 arms
# (the b* progress arms + the two crazy clones c1/c2) are peered — never
# an inert wave-9-through-16 dir (the x*/y*/z*/a* arms, kept for reference
# only), and never a 582-format runs/sweep3. SOLO arms are excluded BOTH ways:
# no one peers a solo arm (none this wave — every arm is 128-wide, so no width
# mismatch can leak into a 128-wide arm's PAST pool).
peer_league_dirs() {
    local self=$1 i name out=""
    for i in "${!VARIANTS[@]}"; do
        name=$(variant_field "$i" 1)
        [[ $name == "$self" ]] && continue
        [[ $(variant_field "$i" 4) == solo ]] && continue
        out+="${out:+,}$SWEEP_DIR/$name/league"
    done
    echo "$out"
}

# Echo the live trainer PID for a run dir, or nothing. Guards against a stale
# pidfile whose PID has been recycled by an unrelated process (a reboot resets
# the PID space): the recorded PID must still be alive AND be a
# train_selfplay.py process whose command line names THIS run dir. Without the
# command-line check we could either wrongly skip a resume (a recycled PID that
# happens to be alive) or, if we launched anyway, end up with two trainers
# writing the same dir.
running_pid() {
    local dir=$1 pidfile=$1/train.pid pid cmdline
    [[ -f $pidfile ]] || return 0
    pid=$(cat "$pidfile" 2>/dev/null) || return 0
    [[ $pid =~ ^[0-9]+$ ]] || return 0
    kill -0 "$pid" 2>/dev/null || return 0
    if [[ -r /proc/$pid/cmdline ]]; then
        cmdline=$(tr '\0' ' ' < "/proc/$pid/cmdline")
    else
        cmdline=$(ps -o args= -p "$pid" 2>/dev/null || true)
    fi
    if [[ $cmdline == *train_selfplay.py* && $cmdline == *"--run-dir $dir"* ]]; then
        echo "$pid"
    fi
    return 0
}

# Echo the num_timesteps recorded inside an sb3 checkpoint zip, or fail (also
# fails on a zip truncated by a kill mid-write, which is what makes the resume
# scan below skip unreadable checkpoints).
zip_steps() {
    "$PY" - "$1" <<'EOF' 2>/dev/null
import json, sys, zipfile
try:
    with zipfile.ZipFile(sys.argv[1]) as z:
        print(json.loads(z.read("data"))["num_timesteps"])
except Exception:
    sys.exit(1)
EOF
}

# Echo "<checkpoint-path-without-.zip> <num_timesteps>" for the furthest-along
# *readable* checkpoint in a run dir, or nothing if there is none. Candidates
# are tried highest-step-first; a truncated checkpoint fails the zip read and
# is skipped in favour of the previous one, so an interrupted run always
# resumes from a clean point.
latest_checkpoint() {
    local dir=$1 zip steps
    while IFS= read -r zip; do
        [[ -n $zip ]] || continue
        steps=$(zip_steps "$zip") || continue
        echo "${zip%.zip} $steps"
        return 0
    done < <(ls "$dir"/ckpt_*_steps.zip 2>/dev/null \
                | sed -E 's/.*ckpt_([0-9]+)_steps\.zip/\1 &/' \
                | sort -k1,1nr | cut -d' ' -f2-)
    return 0
}

# --- format guard ----------------------------------------------------------

# Hard guard: the venv must be built for the 624-format encoding (obs 600 -> 624,
# section 23 per-slot market features). Launching a stale 600-format trainer into
# this epoch's dirs wastes a run; refuse instead. Run 'make develop' after the
# encoding change to rebuild powergrid_py.
check_format() {
    "$PY" - <<'EOF' || { echo "venv is not built for the obs-624 encoding (run 'make develop')" >&2; exit 1; }
from powergrid_env.constants import OBS_SIZE
import powergrid_py, numpy as np
assert OBS_SIZE == 624, f"constants OBS_SIZE={OBS_SIZE}, expected 624"
g = powergrid_py.Game(4, 1)
g.start(["a","b","c","d"], ["red","blue","green","yellow"])
n = np.asarray(g.observation(g.player_ids()[0])).shape[0]
assert n == 624, f"native obs is {n}-wide, expected 624 (stale powergrid_py build)"
EOF
}

# All pre-launch setup for the obs-624 MIGRATION wave. Idempotent (every step
# skips if its output exists); exposed as --prepare so it can be staged on a dev
# box and synced to the training machine. Builds, all at obs-624:
#   * CHAMPION_BIN            migrated a3 (--from-ckpt zero-pads l1 -> plays a3)
#   * wave17-baseline ckpt    the same weights as a runnable sb3 ckpt (--h2h)
#   * EXTRAP / SOUP16         the two crazy-arm merges, rebuilt from migrated 624
#                             source .bins (the 600-wide .zip checkpoints can no
#                             longer be re-serialized under 624 constants)
prepare() {
    check_format
    mkdir -p "$SWEEP_DIR"

    # Migrate the champion checkpoint (600) to a 624 policy .bin: the warm start
    # for b1-b6 AND the frozen eval opponent. Zero-padded l1 rows -> plays a3
    # bit-for-bit; training then learns the 24 new market inputs.
    if [[ ! -f $CHAMPION_BIN ]]; then
        if [[ ! -f ${CHAMPION_CKPT}.zip ]]; then
            echo "cannot migrate champion: no checkpoint at ${CHAMPION_CKPT}.zip" >&2
            echo "(set CHAMPION_CKPT/CHAMPION_BIN, or sync the champion's run dir)" >&2
            exit 1
        fi
        echo "migrating champion -> $CHAMPION_BIN (obs 600 -> 624)"
        "$PY" scripts/migrate_policy_obs.py --from-ckpt "$CHAMPION_CKPT" --out "$CHAMPION_BIN"
    fi

    # The --h2h yardstick: the migrated champion as a runnable 624 sb3 ckpt
    # (policy loaded, value head fresh — irrelevant to h2h/compare, which use
    # only the policy logits). Lives in a synthetic run dir named by $BASELINE.
    local base_ckpt="$SWEEP_DIR/$BASELINE/best_model"
    if [[ ! -f ${base_ckpt}.zip ]]; then
        echo "building --h2h baseline ckpt: $CHAMPION_BIN -> ${base_ckpt}.zip"
        mkdir -p "$SWEEP_DIR/$BASELINE"
        "$PY" scripts/migrate_policy_obs.py --bin-to-ckpt "$CHAMPION_BIN" --out "$base_ckpt"
    fi

    # The two crazy-arm warm starts — weight-space merges of fine-tunes of a
    # shared checkpoint (x5 @894M), so the inputs sit in one loss basin and a
    # merge is meaningful (Wortsman et al. for the average; Ilharco et al. for the
    # extrapolation). Built from the migrated 624 source .bins (make_merge /
    # make_soup accept a PGRLPOL6 .bin, so the merge happens at 624 directly).
    #   EXTRAP (c1): -0.5*x5 + 1.5*a3 — step 1.5x past a3 along the x5->a3
    #                improvement direction (coeffs sum to 1 -> activation scale
    #                preserved). SOUP16 (c2): mean(a1,a3,a6), the wave-16 SWA soup.
    # A build is skipped (with a warning) if a source is missing — its arm then
    # hard-errors at launch on the absent --init-policy-from path.
    migrate_src() {  # run-dir name -> $SWEEP_DIR/<name>-624.bin (echoed), or empty
        local name=$1 out="$SWEEP_DIR/$name-624.bin"
        if [[ -f $out ]]; then echo "$out"; return 0; fi
        [[ -f $SWEEP_DIR/$name/best_model.zip ]] || return 1
        "$PY" scripts/migrate_policy_obs.py --from-ckpt "$SWEEP_DIR/$name/best_model" \
            --out "$out" >&2
        echo "$out"
    }
    if [[ ! -f $EXTRAP ]]; then
        local x5b; x5b=$(migrate_src x5-champ-g999) \
            && echo "building weight-space extrapolation -> $EXTRAP" \
            && "$PY" scripts/make_merge.py --out "$EXTRAP" \
                 --model "$x5b" --coeff -0.5 --model "$CHAMPION_BIN" --coeff 1.5 \
            || echo "skipping extrap build (missing x5 source); c1 will error at launch" >&2
    fi
    if [[ ! -f $SOUP16 ]]; then
        local a1b a6b; a1b=$(migrate_src a1-champ-cont) && a6b=$(migrate_src a6-gae98) \
            && echo "building model soup: mean(a1,a3,a6) -> $SOUP16" \
            && "$PY" scripts/make_soup.py --out "$SOUP16" \
                 --model "$a1b" --model "$CHAMPION_BIN" --model "$a6b" \
            || echo "skipping soup build (missing a1/a6 source); c2 will error at launch" >&2
    fi
}

list_variants() {
    printf '%-20s %-6s %-14s %-6s %s\n' NAME SEED INIT POP FLAGS
    for i in "${!VARIANTS[@]}"; do
        printf '%-20s %-6s %-14s %-6s %s\n' \
            "$(( i + 1 )). $(variant_field "$i" 1)" \
            "$(variant_field "$i" 2)" \
            "$(variant_field "$i" 3)" \
            "$(variant_field "$i" 4)" \
            "$(variant_field "$i" 6)"
    done
}

status() {
    # NOTE: best= is eval/mean_reward vs 3x the frozen champion (a3-nsteps4096);
    # win_rate = (best+1)/2, par ~ -0.50. Negative values are normal.
    for i in "${!VARIANTS[@]}"; do
        local name dir pid state ckpt best when
        name=$(variant_field "$i" 1); dir="$SWEEP_DIR/$name"
        [[ -d $dir ]] || continue
        pid=$(running_pid "$dir")
        if [[ -n $pid ]]; then
            state="running (pid $pid)"
        elif [[ -f $dir/train.pid ]]; then
            state="stopped"      # pidfile present but no live trainer
        else
            state="idle"
        fi
        ckpt=$(ls "$dir"/ckpt_*_steps.zip 2>/dev/null \
                 | sed -E 's/.*ckpt_([0-9]+)_steps\.zip/\1/' | sort -nr | head -1 || true)
        # The bar file is rewritten only when a new best is hit, so its mtime is
        # when that best happened.
        if [[ -f $dir/best_mean_reward.json ]]; then
            best=$(cat "$dir/best_mean_reward.json")
            when=$(date -r "$dir/best_mean_reward.json" '+%Y-%m-%d %H:%M' 2>/dev/null || echo '-')
        else
            best='-'; when='-'
        fi
        printf '%-20s %-20s steps=%-13s best=%-28s best@=%s\n' \
            "$name" "$state" "${ckpt:--}" "$best" "$when"
    done
}

# ABSOLUTE yardstick: each variant's best_model in seat 0 against three `hard`
# bots — the bar the whole project aims at. The all-bots row gives seat 0's
# structural share (~25% plus seat bias), so read a variant against that, not
# against 0. Saturated for ranking since wave 6 (the field crowds 68-79%);
# kept for wave-end reporting and cross-wave comparability.
compare() {
    local games=${COMPARE_GAMES:-200} seed=${COMPARE_SEED:-12345}
    local det=(); [[ ${COMPARE_DETERMINISTIC:-0} == 1 ]] && det=(--deterministic)
    echo "=== baseline: 4x hard bots ($games games) — seat 0's structural share ==="
    "$PY" scripts/evaluate_lineup.py --games "$games" --seed "$seed" --quiet "${det[@]}" \
        --player hard --player hard --player hard --player hard
    for i in "${!VARIANTS[@]}"; do
        local name model
        name=$(variant_field "$i" 1); model="$SWEEP_DIR/$name/best_model"
        [[ -f ${model}.zip ]] || continue
        echo
        echo "=== $name (seat 0) vs 3x hard ($games games) ==="
        "$PY" scripts/evaluate_lineup.py --games "$games" --seed "$seed" --quiet "${det[@]}" \
            --player "$model" --player hard --player hard --player hard
    done
}

# RELATIVE ranking: each variant in seat 0 against three copies of the frozen
# champion (a3-nsteps4096 best_model) — the primary ranking. Above-par here ==
# genuinely past the champion. Measure the mirror par (the self-baseline row)
# before reading small edges; expect ~25 +/- 2%. At wave end, run the FULL
# tiebreak: direct matches between the leaders + a fresh-seed 800-game h2h
# (wave 7's one-shot 400-game samples flipped ranks, and wave 9's seed-12345
# and seed-99999 h2h disagreed at the top).
h2h() {
    local games=${COMPARE_GAMES:-200} seed=${COMPARE_SEED:-12345}
    local det=(); [[ ${COMPARE_DETERMINISTIC:-0} == 1 ]] && det=(--deterministic)
    local base="$SWEEP_DIR/$BASELINE/best_model"
    [[ -f ${base}.zip ]] || { echo "baseline $BASELINE has no best_model (sync $SWEEP_DIR/$BASELINE)" >&2; exit 1; }
    echo "=== self-baseline: 4x $BASELINE ($games games) — mirror par ==="
    "$PY" scripts/evaluate_lineup.py --games "$games" --seed "$seed" --quiet "${det[@]}" \
        --player "$base" --player "$base" --player "$base" --player "$base"
    for i in "${!VARIANTS[@]}"; do
        local name model
        name=$(variant_field "$i" 1); model="$SWEEP_DIR/$name/best_model"
        [[ $name == "$BASELINE" ]] && continue
        [[ -f ${model}.zip ]] || continue
        echo
        echo "=== $name (seat 0) vs 3x $BASELINE ($games games) ==="
        "$PY" scripts/evaluate_lineup.py --games "$games" --seed "$seed" --quiet "${det[@]}" \
            --player "$model" --player "$base" --player "$base" --player "$base"
    done
}

stop_all() {
    for i in "${!VARIANTS[@]}"; do
        local name dir pid
        name=$(variant_field "$i" 1); dir="$SWEEP_DIR/$name"
        pid=$(running_pid "$dir")
        if [[ -n $pid ]]; then
            echo "stopping $name (pid $pid)"
            kill "$pid"
        fi
    done
}

case "${1:-}" in
    --list)    list_variants; exit 0 ;;
    --status)  status;        exit 0 ;;
    --stop)    stop_all;      exit 0 ;;
    --compare) compare;       exit 0 ;;
    --h2h)     h2h;           exit 0 ;;
    --prepare) prepare; echo "wave-16 eval opponent + soups ready in $SWEEP_DIR"; exit 0 ;;
esac

# Which variants to launch (1-based indices; default all).
if (( $# )); then
    SELECTED=("$@")
else
    SELECTED=($(seq 1 ${#VARIANTS[@]}))
fi

[[ -x $PY ]] || { echo "no interpreter at $PY (run 'make develop' first)" >&2; exit 1; }

[[ $DRY_RUN == 1 ]] || prepare

echo "champion (624) : $CHAMPION_BIN (b1-b6 warm start = migrated a3)"
echo "extrapolation  : $EXTRAP (c1-extrapolate warm start)"
echo "wave-16 SWA soup: $SOUP16 (c2-swa-wave16 warm start)"
echo "eval opponent  : $EVAL_OPPONENT (frozen; selects best_model)"
echo "h2h baseline   : $SWEEP_DIR/$BASELINE/best_model (migrated a3 @624)"
echo "target         : $WAVE_STEPS timesteps per variant (all fresh clones from 0)"
echo "sweep dir      : $SWEEP_DIR"
echo "launching      : ${SELECTED[*]}"
echo

mkdir -p "$SWEEP_DIR"

for n in "${SELECTED[@]}"; do
    i=$(( n - 1 ))
    (( i >= 0 && i < ${#VARIANTS[@]} )) || { echo "no variant $n" >&2; exit 1; }

    name=$(variant_field "$i" 1)
    seed=$(variant_field "$i" 2)
    init=$(variant_field "$i" 3)
    pop=$(variant_field "$i" 4)
    why=$(variant_field "$i" 5)
    read -r -a extra <<< "$(variant_field "$i" 6)"
    dir="$SWEEP_DIR/$name"

    # Per-arm fork source: "fork" uses $FORK_FROM, "fork=<stem>" its own.
    fork_src=$FORK_FROM
    if [[ $init == fork=* ]]; then
        fork_src=${init#fork=}
        init=fork
    fi

    # Already running? Never start a second writer on the same run dir. The
    # check confirms the pidfile's PID is genuinely this variant's trainer, so
    # a stale/recycled PID neither blocks a needed resume nor risks a duplicate.
    live=$(running_pid "$dir")
    if [[ -n $live ]]; then
        echo "skip $name: already running (pid $live)"
        continue
    fi

    # No eval-metric migration this wave: wave 16 REUSES the wave-15 frozen eval
    # opponent (the x5 export) because x5 survived wave 15's decider unbeaten, so
    # every continuing arm's stored best bar (z6-soup's -0.35) was earned against
    # the exact same opponent and stays comparable. (The wave-15 migration block
    # that reset bars on an eval-opponent change lived here; it is unneeded when
    # the opponent is unchanged and was removed.)

    # Per-arm cumulative target. Fork/resume arms aim at TOTAL_TIMESTEPS (the
    # wave's cumulative budget). A fresh "clone" arm starts at 0 steps, so its
    # target is a single per-wave budget (WAVE_STEPS) — otherwise it would run
    # ~4x longer than the forks and gate the whole sweep. Using a target (not a
    # hardcoded TOTAL_TIMESTEPS) keeps idempotent relaunch correct: a
    # half-trained clone arm resumes to WAVE_STEPS, not to 600M.
    if [[ $init == clone ]]; then target=$WAVE_STEPS; else target=$TOTAL_TIMESTEPS; fi

    # Auto-resume: continue from the arm's own furthest readable checkpoint.
    # Only a fork/clone arm's very first launch uses its fork source; a
    # continuation arm with no checkpoint is an error, not a fresh start.
    start_args=()
    ckpt_stem=""; done_steps=""
    read -r ckpt_stem done_steps < <(latest_checkpoint "$dir") || true
    if [[ -n $ckpt_stem ]]; then
        steps=$(( target - done_steps ))
        if (( steps <= 0 )); then
            echo "skip $name: already at $done_steps >= target $target timesteps"
            continue
        fi
        start_args=(--resume-from "$ckpt_stem")
        echo "resuming $name from $(basename "$ckpt_stem").zip @ $done_steps (+$steps)"
    elif [[ $init == fork ]]; then
        if [[ ! -f ${fork_src}.zip ]]; then
            echo "cannot fork $name: no checkpoint at ${fork_src}.zip" >&2
            echo "(set FORK_FROM, or sync the fork source's run dir first)" >&2
            exit 1
        fi
        fork_steps=$(zip_steps "${fork_src}.zip") || {
            echo "cannot fork $name: ${fork_src}.zip is unreadable" >&2; exit 1; }
        steps=$(( target - fork_steps ))
        if (( steps <= 0 )); then
            echo "skip $name: fork point already at $fork_steps >= target $target" >&2
            continue
        fi
        start_args=(--resume-from "$fork_src")
        echo "forking $name from $(basename "$fork_src").zip @ $fork_steps (+$steps)"
    elif [[ $init == clone ]]; then
        # Fresh warm-start from a behavior clone (--init-policy-from lives in the
        # arm's extra flags). No --resume-from, so the trainer builds a fresh
        # model at the extra's --net-width and resets the step counter. Gets a
        # fixed per-wave budget (not the cumulative TOTAL_TIMESTEPS) because it
        # starts at 0 and would otherwise run ~4x longer than the forks.
        clone_path=""; prev=""
        for e in "${extra[@]}"; do
            [[ $prev == --init-policy-from ]] && clone_path=$e
            prev=$e
        done
        if [[ -z $clone_path || ! -f $clone_path ]]; then
            echo "cannot clone-init $name: --init-policy-from path missing/unreadable ($clone_path)" >&2
            echo "(run: python -m alphazero.pretrain --net-width <W> --export $clone_path)" >&2
            exit 1
        fi
        steps=$target   # WAVE_STEPS; fresh from 0
        start_args=()   # fresh run
        echo "clone-init $name (fresh) from $clone_path (+$steps)"
    else
        echo "refusing to start $name: it continues an earlier run but $dir has no" >&2
        echo "readable checkpoint. Sync the earlier run dir (or fix SWEEP_DIR)." >&2
        exit 1
    fi

    # Population wiring: peers arms sample every other variant's league dir.
    if [[ $pop == peers ]]; then
        extra+=(--league-peers "$(peer_league_dirs "$name")")
    fi

    cmd=("$PY" scripts/train_selfplay.py
         "${COMMON[@]}"
         --run-dir "$dir"
         --total-timesteps "$steps"
         --seed "$seed"
         "${start_args[@]}"
         "${extra[@]}")

    if [[ $DRY_RUN == 1 ]]; then
        echo "[dry-run] $name:"; printf '  %q' "${cmd[@]}"; echo; continue
    fi

    mkdir -p "$dir"
    # Append (not overwrite): keep every launch's provenance across resumes.
    {
        echo
        echo "# $name — $why"
        echo "# launched: $(date -Is)  init: $init  pop: $pop  target: $TOTAL_TIMESTEPS cumulative"
        printf '%q ' "${cmd[@]}"; echo
    } >> "$dir/variant.txt"

    OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS \
    OPENBLAS_NUM_THREADS=$THREADS NUMEXPR_NUM_THREADS=$THREADS \
        nohup nice -n "$NICE" "${cmd[@]}" >> "$dir/train.log" 2>&1 &
    echo $! > "$dir/train.pid"
    echo "started $name (pid $!, seed $seed) -> $dir"
    echo "  $why"

    # Stagger startup: eight processes importing torch and building envs at the
    # same instant just thrashes.
    (( n == ${SELECTED[-1]} )) || sleep "$STAGGER"
done

cat <<EOF

Monitor:
  ./scripts/sweep_selfplay.sh --status     # best= is vs the frozen champion: win = (best+1)/2, par ~ -0.50
  tail -f $SWEEP_DIR/b1-champ-cont/train.log
  $PY -m tensorboard.main --logdir $SWEEP_DIR      # league/peer_size, eval/mean_reward
  $PY scripts/run_report.py $SWEEP_DIR/b1-champ-cont
Rank the variants:
  ./scripts/sweep_selfplay.sh --compare    # absolute: vs 3x hard bots (reporting; saturated)
  ./scripts/sweep_selfplay.sh --h2h        # relative: vs 3x the frozen $BASELINE best (primary ranking)
Stop everything:
  ./scripts/sweep_selfplay.sh --stop
EOF
