#!/usr/bin/env bash
#
# sweep_selfplay.sh — wave 15: lock in gamma 0.999 on the CHAMPION line and
# sweep the levers a high-gamma regime actually needs (rollout length, league
# depth, value emphasis) + two CRAZY probes (a weight-space model soup, and a
# targeted exploiter). Six "progress" arms RESUME their own 600-format
# checkpoints or FORK (--resume-from, warm policy+value head) from a pinned
# 600-format checkpoint — the proven waves-4-8/10-14 structure. One crazy arm
# goes off-script as a fresh clone warm-started from an averaged policy.
#
# WAVE 15 (2026-08-26).
#
# WAVE 14 (triangulate the gamma peak inside the cross-lineage recipe + two
# crazy probes) is settled. All 8 arms reached their targets (forks/resumes
# 865M-900M). Results:
#
#   Frozen-champion eval (best mean_reward vs 3x w5-y3-gamma, par ~-0.50):
#     x3-y3-g999 -0.24  x5-champ-g999 -0.26  x4-y3-g9995 -0.28
#     x1-champ -0.30  x2-y3-g997 -0.30  y7-placement -0.31
#     y6-mc-return -0.38  s4-y3 -0.41
#   Compare vs 3x hard (saturated, reporting-only, seat-0 all-bots par ~21.5%):
#     x1-champ 85.5  x4-y3-g9995 84.0  x2-y3-g997 82.5  x3-y3-g999 82.0
#     s4-y3 80.5  y7-placement 80.5  x5-champ-g999 79.5  y6-mc-return 75.0
#   Primary h2h (seat 0 vs 3x the frozen champion w5-y3-gamma; seat-0 par ~23.5%):
#     x5-champ-g999 31.0  x4-y3-g9995 28.5  x1-champ 26.5  x3-y3-g999 26.5
#     x2-y3-g997 24.0  y7-placement 23.0  s4-y3 21.0  y6-mc-return 20.5
#   The three boards DISAGREED at the top (x3 led eval, x5 led h2h, x4 was the
#   all-round runner-up), so a DIRECT decider ran between the top three (seed
#   77777, 400 games each way, 4-way par 25%):
#     x5 vs 3x x3 = 27.8  |  x3 vs 3x x5 = 26.8   -> x5 edges x3
#     x5 vs 3x x4 = 27.0  |  x4 vs 3x x5 = 22.5   -> x5 DOMINATES x4 (x4 < par)
#     x3 vs 3x x4 = 26.2  |  x4 vs 3x x3 = 27.5   -> x4 edges x3
#   x5-champ-g999 has the best offense (avg 27.4 as challenger) AND the best
#   defense (opponents held to avg 24.65, below par). x5-champ-g999 is the
#   wave-14 champion and the new embedded Expert.
#
# What wave 14 SETTLED / OPENED:
#   * *** GAMMA PEAK IS 0.999. *** The cross-lineage eval curve is a clean
#     inverted-U: 0.997 (-0.30) -> 0.999 (-0.24) -> 0.9995 (-0.28), and wave 13
#     already showed 1.0 hurts. 0.999 is the peak; every wave-15 continuation
#     and lever arm runs gamma 0.999. (Gamma is now a SETTLED constant, no
#     longer swept.)
#   * *** LINEAGE REVERSED AT THE PEAK GAMMA. *** Wave 13 found cross-lineage
#     (fork the never-annealed donor) beat champion-line at gamma 0.997. At the
#     BETTER gamma 0.999 it flips: champion-line x5 beat cross-lineage x3 on
#     h2h (31.0 vs 26.5) and won the decider; x3 led only the (noisier) eval
#     proxy. The likely mechanism is VALUE-HEAD GAMMA-CONTINUITY: re-arming at a
#     new gamma forces the critic to re-learn the return scale, and the champion
#     w5 was trained at 0.997 (a small 0.997->0.999 shift) while the donor was
#     trained at 0.99 (a large 0.99->0.999 shift). Fork the checkpoint whose
#     gamma is CLOSEST to the target. So wave 15's presumptive champion is the
#     CHAMPION-LINE continuation (fork x5, already @0.999 — zero gamma shift);
#     cross-lineage is demoted to a control/hedge, not the presumptive winner.
#   * BOTH CRAZIES UNDERPERFORMED. y6-mc-return (gamma 1.0 + gae-lambda 1.0, pure
#     Monte-Carlo return, no bootstrap) was worst-but-one on every board — no
#     bootstrap horizon clearly hurts, confirming the value net matters. That
#     motivates wave 15's z5-vf (STRENGTHEN the critic, don't remove it).
#     y7-placement (finish-rank terminal reward) was middling (-0.31 eval, h2h
#     23.0 ~ par) — no edge over winloss; retired.
#   * DECAY held (11th straight champion is a decay arm). The constant-lr donor
#     s4-y3 stays weak-but-useful as cross-lineage fork material (eval -0.41,
#     h2h 21.0). KEEP THE DONOR ALIVE.
#   * WIDE NET (192) stays retired (failed twice; 128 field is not
#     capacity-bound).
#
# 150M per-arm convergence budget still holds — eval peaks land in the back
# third as lr-decay anneals to 0. Forks from x5-champ-g999/best_model @~894M
# gain ~+156M to 1050M cumulative; the donor s4-y3 resumes 900M -> 1050M.
#
# THE STRUCTURE — six progress arms + two crazy arms:
#   * FORK_FROM = the new champion x5-champ-g999/best_model (@~894M). RETIRED (lr
#     annealed to ~0), so its best_model is a stable fork point AND the frozen
#     --h2h/--eval yardstick AND the embedded Expert — it never trains. It was
#     trained at gamma 0.999, so a gamma-0.999 fork of it is value-continuous.
#   * The DONOR lineage lives on: s4-y3 (never-annealed, constant lr, gamma 0.99)
#     RESUMES 900M -> 1050M. The cross-lineage control z2 forks its PINNED @900M
#     checkpoint + decay + gamma 0.999 — the hedge in case champion-line's edge
#     is data-limited, not permanent.
#   * runs/sweep4 is this epoch's home. Inert history NOT peered: wave-9 (s*),
#     wave-10 (t*), wave-11 (u*), wave-12 (v1-v7), wave-13 (w*), wave-14 losers
#     kept for reference only. runs/sweep3 is inert 582-format — never resume,
#     never peer. Only wave-15 arms + the resuming donor s4-y3 peer.
#
# Arm roles (all lr 1e-4->0, gamma 0.999, n-steps/batch 1024, mix 0.40/0.40/0.20,
# past_k 8, ent 0.03, all peers, all fork the champion x5 @~894M, unless noted):
#   s4-y3          RESUME the donor: constant lr, never decayed, never forked,
#                  gamma 0.99. The fork material for the cross-lineage control.
#   z1-champ-cont  THE presumptive champion: fork x5 (already @0.999) + fresh lr
#                  decay 1e-4 -> 0 + gamma 0.999. A pure, value-continuous
#                  continuation of the winning line — zero gamma shift. The
#                  main bet and the champion-line anchor for the lever arms.
#   z2-y3-g999     Cross-lineage control: fork the donor's pinned @900M + decay +
#                  gamma 0.999. Re-tests lineage at the settled gamma from an
#                  advanced donor state — does cross-lineage ever catch up, or is
#                  champion-line's value-continuity edge permanent? The hedge.
#   z3-nsteps      Lever: z1 but n-steps 2048 / batch 2048. Longer rollouts cut
#                  GAE truncation bias and gradient variance — the fix for the
#                  harder value fitting at gamma 0.999. Champion line, @0.999.
#   z4-deep-league Lever: z1 but league-past-k 16 and mix 0.35/0.45/0.20 (deeper,
#                  more PAST-weighted pool). More diverse opponents -> a more
#                  robust champion. Champion line, @0.999.
#   z5-vf          Lever: z1 but vf-coef 1.0 (default 0.5). High gamma stresses
#                  the critic (y6 showed removing the bootstrap hurts); give the
#                  value loss more weight so explained_variance stays high and
#                  advantages are better estimated. Champion line, @0.999.
#   z6-soup        *** CRAZY #1: the model soup. *** A fresh clone warm-started
#                  (--init-policy-from) from the UNIFORM WEIGHT-SPACE AVERAGE of
#                  the three shared-init cross-lineage arms x2/x3/x4 (all fork
#                  the donor @750M, spanning the gamma peak 0.997/0.999/0.9995),
#                  then trained with GENTLE lr 3e-5 -> 0 (protect the soup while
#                  the fresh value head re-learns) + gamma 0.999. Souping only
#                  works among fine-tunes of a shared checkpoint (they share a
#                  loss basin); x2/x3/x4 are exactly that. Does the average of
#                  the gamma sweep sit in a flatter/better minimum than any
#                  single gamma? scripts/make_soup.py builds the .bin in prepare.
#   z7-exploiter   *** CRAZY #2: the targeted exploiter. *** fork x5 + gamma
#                  0.999 + league-mix 0.70/0.10/0.20 — train almost entirely
#                  against LATEST (the current champion/self), à la AlphaStar
#                  exploiters, keeping the load-bearing 0.20 bot share and
#                  cutting PAST to 0.10. Deliberately overfits the h2h opponent.
#                  If it wins the h2h decisively AND holds vs hard bots it is a
#                  real champion; if it wins h2h but tanks --compare it overfit,
#                  which the decider will catch. gamma 0.999, champion line.
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
#   ./scripts/sweep_selfplay.sh --h2h        # RELATIVE: each variant vs 3x the frozen champion (x5-champ-g999)
#   ./scripts/sweep_selfplay.sh --stop       # stop every running variant
#
# Env overrides (all optional):
#   TOTAL_TIMESTEPS=1050000000 CUMULATIVE per-arm target (forks from ~894M
#                              gain ~+156M; resumes never overshoot)
#   FORK_FROM=runs/sweep4/x5-champ-g999/best_model   champion fork point + baseline
#   Y3_FORK=runs/sweep4/s4-y3/ckpt_900000000_steps   pinned donor fork stem
#   EVAL_OPPONENT=runs/sweep4/wave15-eval-opponent.bin
#                              frozen eval opponent, exported from FORK_FROM on
#                              first launch (selects best_model; par ~ -0.50)
#   SOUP=runs/sweep4/soup-x234.bin   averaged x2/x3/x4 policy, built in prepare
#   BASELINE=x5-champ-g999     run-dir name of the frozen --h2h opponent
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
FORK_FROM=${FORK_FROM:-$SWEEP_DIR/x5-champ-g999/best_model}
Y3_FORK=${Y3_FORK:-$SWEEP_DIR/s4-y3/ckpt_900000000_steps}
EVAL_OPPONENT=${EVAL_OPPONENT:-$SWEEP_DIR/wave15-eval-opponent.bin}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-1050000000}
WAVE_STEPS=${WAVE_STEPS:-300000000}   # fresh-clone arms' cumulative target (the z6-soup crazy arm)
SOUP=${SOUP:-$SWEEP_DIR/soup-x234.bin}   # model soup: averaged x2/x3/x4 policy, built in prepare (z6-soup warm start)
NET_WIDTH=${NET_WIDTH:-128}
NUM_ENVS=${NUM_ENVS:-8}
THREADS=${THREADS:-3}
NICE=${NICE:-10}
STAGGER=${STAGGER:-15}
DRY_RUN=${DRY_RUN:-0}

# The frozen --h2h opponent: the wave-14 champion's best_model — the exact
# weights the fork arms started from and the eval opponent was exported from.
# x5-champ-g999 no longer trains (its lr annealed to 0), so this yardstick never
# moves. It is also the new embedded Expert.
BASELINE=${BASELINE:-x5-champ-g999}

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
# — the crazy model-soup arm z6, whose --init-policy-from lives in its extra
# flags and is ignored by the trainer on any later --resume-from; the soup is
# 128-wide so --net-width comes from COMMON). After the first launch every arm
# resumes its OWN checkpoints.
#
# `pop` is "peers" (the launch loop appends --league-peers with every OTHER
# variant's league dir) or "solo" (no peers).
#
# League mix order is LATEST,PAST,BOTS; the trainer default is 0.5,0.3,0.2.
# The 0.20 bot share is load-bearing (wave-7 q6 faceplant) — do not cut it.
VARIANTS=(
"s4-y3|804|resume|peers|The donor lineage: constant lr 1e-4 + batch 1024, gamma 0.99, never decayed, never forked — compounding since the obs-600 reset (now @900M, on to 1050M). The cross-lineage control z2 forks a fresh copy of its pinned @900M state and adds decay + gamma 0.999. Keeping it alive is what gives every wave an independent, never-annealed history to cross-decay.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"z1-champ-cont|1501|fork|peers|THE presumptive champion: fork x5-champ-g999 (already trained at gamma 0.999) + fresh lr decay 1e-4 -> 0 + gamma 0.999 + normal ent (0.03). A pure, value-CONTINUOUS continuation of the winning line — zero gamma shift, so the forked critic already speaks the target return scale. The main bet and the champion-line anchor the lever arms (z3/z4/z5) vary one knob from. Collapse guard: entropy diving toward ~0.1 nats = kill the arm.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"z2-y3-g999|1502|fork=$SWEEP_DIR/s4-y3/ckpt_900000000_steps|peers|Cross-lineage control / hedge: fork the never-annealed donor's pinned @900M state + lr decay -> 0 + gamma 0.999 + normal ent. Wave 14 showed champion-line beat cross-lineage at gamma 0.999 (x5 > x3), likely on value-head gamma-continuity (the donor was trained at 0.99, a big shift to 0.999). This re-tests it from a MORE-advanced donor state: does cross-lineage catch up with more donor data, or is champion-line's edge permanent?|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"z3-nsteps|1503|fork|peers|LEVER (rollout length): z1 but n-steps 2048 / batch 2048 (2x). At gamma 0.999 the effective horizon is long and GAE truncates the return at n-steps; doubling the rollout cuts truncation bias and halves gradient variance per update — the direct fix for the harder value fitting the high gamma demands. Champion line, fork x5, gamma 0.999. If it clears z1 on both boards, longer rollouts are the free lunch at high gamma.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --n-steps 2048 --batch-size 2048 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"z4-deep-league|1504|fork|peers|LEVER (opponent diversity): z1 but league-past-k 16 (deeper PAST pool) and mix 0.35,0.45,0.20 (more PAST-weighted, load-bearing 0.20 bots kept). A more diverse set of past selves should build a champion that is robust rather than tuned to the few latest snapshots — potentially a higher, less brittle h2h. Champion line, fork x5, gamma 0.999.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --n-steps 1024 --batch-size 1024 --league-mix 0.35,0.45,0.20 --league-past-k 16"
"z5-vf|1505|fork|peers|LEVER (value emphasis): z1 but vf-coef 1.0 (default 0.5). Gamma 0.999 stresses the critic and y6-mc-return showed that REMOVING the bootstrap hurts, so the value net matters — give the value loss 2x weight so explained_variance stays high and the advantages PPO clips on are better estimated. Champion line, fork x5, gamma 0.999. Watch that policy entropy is not crowded out.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --vf-coef 1.0 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"z6-soup|1506|clone|peers|CRAZY #1 — the model soup. A FRESH clone warm-started (--init-policy-from) from the UNIFORM WEIGHT-SPACE AVERAGE of the three shared-init cross-lineage arms x2/x3/x4 (all fork the donor @750M, differing only in gamma 0.997/0.999/0.9995 — exactly the fine-tunes-of-a-shared-checkpoint setting where model soups sit in one loss basin). Trained with GENTLE lr 3e-5 -> 0 (protect the averaged policy while the fresh value head re-learns, the s3-gentle recipe) + gamma 0.999. Does the average across the gamma sweep land in a flatter/better minimum than any single gamma? scripts/make_soup.py builds soup-x234.bin in --prepare. Fresh run: 300M (WAVE_STEPS) budget, not the 1050M cumulative target.|--learning-rate 3e-5 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --init-policy-from $SOUP --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"z7-exploiter|1507|fork|peers|CRAZY #2 — the targeted exploiter. fork x5 + gamma 0.999 + league-mix 0.70,0.10,0.20: train almost entirely against LATEST (the current champion/self), à la AlphaStar exploiters, keeping the load-bearing 0.20 bot share and cutting PAST to 0.10. Deliberately OVERFITS the h2h opponent to squeeze the promotion metric. If it wins the h2h decisively AND holds vs hard bots (--compare) it is a real champion; if it wins h2h but tanks --compare it overfit, which the wave-end decider catches. Champion line.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --n-steps 1024 --batch-size 1024 --league-mix 0.70,0.10,0.20 --league-past-k 8"
)


variant_field() { echo "${VARIANTS[$1]}" | cut -d'|' -f"$2"; }

# Comma-separated league dirs of every variant EXCEPT $1 (by name) — the
# --league-peers value for a "peers" arm. Dirs may not exist yet; the trainer
# tolerates that (empty pool slice until the peer launches). Only wave-14 arms
# (the x*/y* arms + the resuming donor s4-y3) are peered — never an inert wave-9
# through -13 dir, and never a 582-format runs/sweep3. SOLO arms are excluded
# BOTH ways: no one peers a solo arm (none this wave — the 192-wide v6-wide is
# retired, so no width mismatch can leak into a 128-wide arm's PAST pool).
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

# Hard guard: the venv must be built for the 600-format encoding. Launching a
# stale 582-format trainer into this epoch's dirs corrupts nothing but wastes a
# run; refuse instead.
check_format() {
    "$PY" - <<'EOF' || { echo "venv is not built for the obs-600 encoding (run 'make develop')" >&2; exit 1; }
from powergrid_env.constants import OBS_SIZE
import powergrid_py, numpy as np
assert OBS_SIZE == 600, f"constants OBS_SIZE={OBS_SIZE}, expected 600"
g = powergrid_py.Game(4, 1)
g.start(["a","b","c","d"], ["red","blue","green","yellow"])
n = np.asarray(g.observation(g.player_ids()[0])).shape[0]
assert n == 600, f"native obs is {n}-wide, expected 600 (stale powergrid_py build)"
EOF
}

# All pre-launch setup: format guard + export the frozen eval opponent from the
# champion fork point. Idempotent; exposed as --prepare so it can be staged
# (e.g. on a dev box, then synced to the training machine) without launching.
prepare() {
    check_format
    mkdir -p "$SWEEP_DIR"

    # Export the frozen eval opponent from the champion. Once exported it is
    # never touched again; the golden sidecar goes next to it, NOT into assets/
    # (export_policy.py's default).
    if [[ ! -f $EVAL_OPPONENT ]]; then
        if [[ ! -f ${FORK_FROM}.zip ]]; then
            echo "cannot export eval opponent: no checkpoint at ${FORK_FROM}.zip" >&2
            echo "(set FORK_FROM/EVAL_OPPONENT, or sync the champion's run dir)" >&2
            exit 1
        fi
        echo "exporting frozen eval opponent: ${FORK_FROM}.zip -> $EVAL_OPPONENT"
        "$PY" scripts/export_policy.py --model "$FORK_FROM" \
            --out "$EVAL_OPPONENT" --golden "${EVAL_OPPONENT}.golden.json"
    fi

    # Build the z6-soup warm start: the uniform weight-space average of the three
    # shared-init cross-lineage arms x2/x3/x4 (all fork the donor @750M, spanning
    # the gamma peak). Idempotent; only their best_model exports are read, so it
    # is safe to rebuild. Skipped (with a warning) if an input is missing —
    # z6-soup then hard-errors at launch on the absent --init-policy-from path.
    if [[ ! -f $SOUP ]]; then
        local soup_in=(x2-y3-g997 x3-y3-g999 x4-y3-g9995) missing=0 m
        for m in "${soup_in[@]}"; do
            [[ -f $SWEEP_DIR/$m/best_model.zip ]] || { echo "soup input missing: $SWEEP_DIR/$m/best_model.zip" >&2; missing=1; }
        done
        if [[ $missing == 0 ]]; then
            echo "building model soup: mean(x2,x3,x4) -> $SOUP"
            "$PY" scripts/make_soup.py --out "$SOUP" \
                --model "$SWEEP_DIR/x2-y3-g997/best_model" \
                --model "$SWEEP_DIR/x3-y3-g999/best_model" \
                --model "$SWEEP_DIR/x4-y3-g9995/best_model"
        else
            echo "skipping soup build (missing inputs); z6-soup will error at launch" >&2
        fi
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
    # NOTE: best= is eval/mean_reward vs 3x the frozen champion (x5-champ-g999);
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
# champion (x5-champ-g999 best_model) — the primary ranking. Above-par here ==
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
    --prepare) prepare; echo "wave-15 eval opponent ready in $SWEEP_DIR"; exit 0 ;;
esac

# Which variants to launch (1-based indices; default all).
if (( $# )); then
    SELECTED=("$@")
else
    SELECTED=($(seq 1 ${#VARIANTS[@]}))
fi

[[ -x $PY ]] || { echo "no interpreter at $PY (run 'make develop' first)" >&2; exit 1; }

[[ $DRY_RUN == 1 ]] || prepare

echo "fork point     : $FORK_FROM (fork arms' first launch only)"
echo "donor fork stem: $Y3_FORK (z2-y3-g999's first launch only)"
echo "eval opponent  : $EVAL_OPPONENT (frozen; selects best_model)"
echo "h2h baseline   : $SWEEP_DIR/$BASELINE/best_model"
echo "target         : $TOTAL_TIMESTEPS cumulative timesteps per variant"
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

    # One-time wave-15 migration for any arm continuing from an earlier wave
    # (the resuming donor s4-y3): its stored best bar was earned vs the wave-14
    # eval opponent (the frozen w5-y3-gamma) and is not comparable with the new
    # frozen champion (x5-champ-g999). Set the old best aside, drop the bar, and
    # let the new metric re-earn best_model.zip. Fresh fork arms have no best bar
    # yet, so they skip this. The marker makes re-runs a no-op.
    if [[ $DRY_RUN != 1 && -f $dir/best_mean_reward.json \
          && ! -f $dir/.wave15-eval-opponent ]]; then
        [[ -f $dir/best_model.zip && ! -f $dir/best_model.wave14-vs-champion.zip ]] \
            && cp "$dir/best_model.zip" "$dir/best_model.wave14-vs-champion.zip"
        rm "$dir/best_mean_reward.json"
        touch "$dir/.wave15-eval-opponent"
        echo "migrated $name to the wave-15 eval metric (old best kept as best_model.wave14-vs-champion.zip)"
    fi

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
  tail -f $SWEEP_DIR/z1-champ-cont/train.log
  $PY -m tensorboard.main --logdir $SWEEP_DIR      # league/peer_size, eval/mean_reward
  $PY scripts/run_report.py $SWEEP_DIR/z1-champ-cont
Rank the variants:
  ./scripts/sweep_selfplay.sh --compare    # absolute: vs 3x hard bots (reporting; saturated)
  ./scripts/sweep_selfplay.sh --h2h        # relative: vs 3x the frozen $BASELINE best (primary ranking)
Stop everything:
  ./scripts/sweep_selfplay.sh --stop
EOF
