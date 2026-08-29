#!/usr/bin/env bash
#
# sweep_selfplay.sh — wave 16: PLATEAU-BREAK. Wave 15 produced no successor to
# the champion x5-champ-g999 (its decider beat every arm), so timid one-knob
# continuations at gamma 0.999 are exhausted — they relax straight back into the
# x5 minimum. Wave 16 keeps the proven fork-and-anneal machinery but swings
# BIGGER: six arms are one CONTROL (the champion continuation) + five DISTINCT
# perturbations, each on a different axis (lr restart, rollout length, objective,
# optimization intensity, advantage horizon), plus two CRAZY weight-space soups —
# the one config-only lever that reaches weights x5 cannot train its way to.
#
# WAVE 16 (2026-08-29).
#
# WAVE 15 (lock gamma 0.999 on the champion line + sweep rollout/league/value
# levers + two crazies) is settled. Seven of eight arms hit the 1050M target
# (z6-soup, a fresh clone, reached only 190M — unfinished, CONTINUED this wave).
# Result: a NULL WAVE — nothing beat the incumbent x5-champ-g999.
#
#   Frozen-champion eval (best mean_reward vs 3x x5-champ-g999, par ~-0.50):
#     z3-nsteps -0.29  z4-deep-league -0.30  z1-champ-cont -0.31
#     z5-vf -0.33  z7-exploiter -0.33  z2-y3-g999 -0.34  z6-soup -0.35(@190M)
#     s4-y3 -0.46
#   Compare vs 3x hard (saturated, reporting-only, seat-0 all-bots par ~21.5%):
#     z7-exploiter 86.5  z3-nsteps 86.0  z1-champ-cont 85.0  z6-soup 84.5
#     z2-y3-g999 82.5  s4-y3 81.0  z5-vf 81.0  z4-deep-league 80.5
#   Primary h2h (seat 0 vs 3x the frozen champion x5-champ-g999; mirror par 24.5%):
#     z7-exploiter 26.0  z3-nsteps 25.5  z2/z4 23.5  z1-champ-cont 21.5(BELOW par)
#     z6-soup 21.5(@190M)  z5-vf 20.0  s4-y3 19.5
#   Only z3-nsteps and z7-exploiter cleared h2h par, and by <1.5pp (inside the
#   +-5pp noise). A DECIDER between them and the incumbent x5 (seed 88888, 400
#   games/dir, 4-way par 25%) settled it:
#     x5 vs 3x z3 = 29.8  |  z3 vs 3x x5 = 25.5   -> x5 DOMINATES z3
#     x5 vs 3x z7 = 26.5  |  z7 vs 3x x5 = 22.0   -> x5 DOMINATES z7
#     z3 vs 3x z7 = 25.8  |  z7 vs 3x z3 = 23.8   -> z3 edges z7
#   x5-champ-g999 has the best offense (avg 28.15 as challenger) AND the best
#   defense (opponents held to avg 23.75). The champion HOLDS; the Expert is
#   unchanged. z3/z7's per-arm edges were eval/board noise, not real gains.
#
# What wave 15 SETTLED / OPENED:
#   * *** THE CHAMPION-CONTINUATION LINE HAS PLATEAUED. *** z1-champ-cont, the
#     presumptive winner (pure value-continuous fork of x5), landed BELOW par on
#     h2h. Forking x5 and annealing at lr 1e-4 converges back to x5, not past it.
#     Beating x5 now needs a BIGGER perturbation, a DIFFERENT objective, or a
#     move x5 cannot reach by training against itself (a soup). That is wave 16.
#   * TWO LEVERS FLICKERED (then died in the decider): longer rollouts (z3,
#     n-steps 2048 -> eval leader) and the exploiter mix (z7, 0.70/0.10/0.20 ->
#     h2h+compare leader). Both are DATA/optimization-regime changes, not value
#     or gamma knobs — the axis worth pushing. Wave 16 pushes rollout length
#     further (a3, 4096) and keeps the exploiter idea alive inside the soup z8.
#   * DEAD LEVERS: deep-league (z4, worst compare), vf-coef 1.0 (z5, worst h2h
#     but the donor), cross-lineage at 0.999 (z2, at par). All retired.
#   * GAMMA STAYS 0.999 (settled wave 14; every arm runs it). WIDE NET stays
#     retired. The 0.20 bot share stays load-bearing.
#   * DONOR: s4-y3 sits at 1050M, weak (eval -0.46, h2h 19.5). No wave-16 arm
#     forks it (cross-lineage keeps losing at the peak gamma), so it is NOT
#     resumed — it stays on disk at 1050M as future fork material, not a slot.
#
# 150M per-arm convergence budget holds — eval peaks land in the back third as
# lr-decay anneals to 0. The six fork arms gain ~+156M from x5/best_model @~894M
# to 1050M cumulative; the two soup clones start fresh (z6 resumes its 190M) and
# get a WAVE_STEPS 350M budget.
#
# THE STRUCTURE — six progress arms (1 control + 5 perturbations) + two crazies:
#   * FORK_FROM = the champion x5-champ-g999/best_model (@~894M). RETIRED (lr ~0),
#     so its best_model is a stable fork point AND the frozen --h2h/--eval
#     yardstick AND the embedded Expert — it never trains. Trained at gamma
#     0.999, so a gamma-0.999 fork of it is value-continuous.
#   * Every arm changes ONE axis from the a1 control so its effect is readable on
#     the shared peers/seeds. The perturbations are deliberately LARGER than
#     wave 15's (3x lr, 4x rollout, a different objective) because the small ones
#     provably collapse back to x5.
#   * runs/sweep4 is this epoch's home. Inert history NOT peered: wave-9 (s*),
#     wave-10 (t*), wave-11 (u*), wave-12 (v*), wave-13 (w*), wave-14 (x*/y*),
#     wave-15 losers (z1-z5, z7). runs/sweep3 is inert 582-format — never resume,
#     never peer. Only wave-16 arms (a1-a6 + the continuing soups z6/z8) peer.
#
# Arm roles (all gamma 0.999, n-steps/batch 1024, mix 0.40/0.40/0.20, past_k 8,
# ent 0.03, fork the champion x5 @~894M, lr 1e-4->0, unless noted):
#   a1-champ-cont     CONTROL: pure value-continuous continuation of x5 (zero
#                     gamma shift). Not expected to win (wave 15's z1 proved the
#                     line plateaus); the incumbent-in-the-field baseline every
#                     perturbation is read against.
#   a2-hi-lr-restart  Perturbation (basin escape): a1 but lr 3e-4 -> 0 (3x). An
#                     SGDR-style restart to kick out of the x5 minimum, then
#                     re-anneal. Is x5 an escapable local optimum?
#   a3-nsteps4096     Perturbation (rollout length): a1 but n-steps/batch 4096
#                     (4x). Push wave 15's longest-lived lever further; cut GAE
#                     truncation bias at the long gamma-0.999 horizon.
#   a4-relative-shaping  Perturbation (objective): a1 but relative reward shaping
#                     (own - best_opp powered) annealed off over 100M, then pure
#                     terminal. Dense early credit while the horizon is hard.
#   a5-epochs-kl      Perturbation (optimization intensity): a1 but n-epochs 8
#                     (2x) + target-kl 0.03 cap. More passes per sample without
#                     the over-update cliff.
#   a6-gae98          Perturbation (advantage horizon): a1 but gae-lambda 0.98
#                     (from 0.95) to lengthen advantage estimation toward the
#                     gamma-0.999 reward horizon.
#   z6-soup           *** CRAZY #1: gamma-sweep soup, CONTINUED. *** Resume the
#                     wave-15 clone (@190M) warm-started from mean(x2,x3,x4) (the
#                     gamma-peak sweep, shared basin), gentle lr 3e-5 -> 0. Does
#                     the average across the gamma sweep sit in a flatter minimum
#                     than any single gamma? Finish the experiment.
#   z8-champ-soup     *** CRAZY #2: champion-survivors soup. *** Fresh clone from
#                     mean(x5, z3-nsteps, z7-exploiter) — the incumbent and its
#                     two most-different wave-15 children (all fork x5 -> shared
#                     basin), gentle lr 3e-5 -> 0. The boldest plateau bet:
#                     weight-averaging reaches weights forking+annealing x5
#                     cannot, a move to the basin's flatter interior, not a
#                     gradient step out of it.
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
#   TOTAL_TIMESTEPS=1050000000 CUMULATIVE per-arm target for the fork arms (from
#                              ~894M gain ~+156M; resumes never overshoot)
#   WAVE_STEPS=350000000       cumulative target for the fresh-clone soup arms
#                              (z6 resumes its 190M; z8 starts at 0)
#   FORK_FROM=runs/sweep4/x5-champ-g999/best_model   champion fork point + baseline
#   EVAL_OPPONENT=runs/sweep4/wave15-eval-opponent.bin
#                              frozen eval opponent = the x5 export, REUSED from
#                              wave 15 (champion unchanged; par ~ -0.50)
#   SOUP=runs/sweep4/soup-x234.bin      averaged x2/x3/x4 (z6-soup warm start), built in prepare
#   SOUP2=runs/sweep4/soup-x5z3z7.bin   averaged x5/z3/z7 (z8-champ-soup warm start), built in prepare
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
EVAL_OPPONENT=${EVAL_OPPONENT:-$SWEEP_DIR/wave15-eval-opponent.bin}
                             # REUSED from wave 15: it is the frozen x5 export,
                             # and x5 survived wave 15's decider unbeaten, so the
                             # champion (hence the eval opponent) is unchanged.
                             # Reusing it keeps every continued arm's best bar
                             # (z6-soup, ...) directly comparable — no migration.
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-1050000000}
WAVE_STEPS=${WAVE_STEPS:-350000000}   # fresh-clone arms' cumulative target (the two soup crazy arms z6/z8)
SOUP=${SOUP:-$SWEEP_DIR/soup-x234.bin}       # gamma-sweep soup: averaged x2/x3/x4 policy (z6-soup warm start), built in prepare
SOUP2=${SOUP2:-$SWEEP_DIR/soup-x5z3z7.bin}   # champion-survivors soup: averaged x5/z3/z7 policy (z8-champ-soup warm start), built in prepare
NET_WIDTH=${NET_WIDTH:-128}
NUM_ENVS=${NUM_ENVS:-8}
THREADS=${THREADS:-3}
NICE=${NICE:-10}
STAGGER=${STAGGER:-15}
DRY_RUN=${DRY_RUN:-0}

# The frozen --h2h opponent: x5-champ-g999's best_model — the champion that
# survived wave 15's decider unbeaten, the weights every fork arm starts from,
# and the source of the frozen eval opponent. It no longer trains (its lr
# annealed to 0), so this yardstick never moves. It is the embedded Expert.
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
"a1-champ-cont|1601|fork|peers|CONTROL / anchor: a pure, value-continuous continuation of the champion x5-champ-g999 (fork its best_model @~894M, already trained at gamma 0.999 -> zero gamma shift, the forked critic already speaks the target return scale) + fresh lr decay 1e-4 -> 0 + gamma 0.999 + ent 0.03. Wave 15 proved this line relaxes straight back into the x5 minimum (its continuation z1 landed BELOW par), so a1 is NOT expected to win — it is the incumbent-in-the-field baseline every perturbation arm (a2-a6) is read against, on the same peers/seeds. Collapse guard: entropy diving toward ~0.1 nats = kill the arm.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"a2-hi-lr-restart|1602|fork|peers|PERTURBATION (basin escape): a1 but a HIGH lr warm-restart 3e-4 -> 0 (3x a1's 1e-4). Wave 15 showed timid 1e-4 continuations relax straight back into the x5 basin; an SGDR-style restart kicks the policy out of that minimum, then anneals to 0 to re-settle — the direct test of whether x5 is an ESCAPABLE local optimum or a true plateau. Champion line, gamma 0.999. Expect a mid-run entropy spike then recovery; a permanent collapse toward ~0.1 nats = kill.|--learning-rate 3e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"a3-nsteps4096|1603|fork|peers|PERTURBATION (rollout length): a1 but n-steps 4096 / batch 4096 (4x). n-steps 2048 (wave-15 z3) led the eval board of a null wave and was the longest-lived signal — push the lever further. At gamma 0.999 GAE truncates the return at n-steps, so longer rollouts cut truncation bias and gradient variance, the fix for the harder value fitting the long horizon demands. Champion line, gamma 0.999.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --n-steps 4096 --batch-size 4096 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"a4-relative-shaping|1604|fork|peers|PERTURBATION (objective): a1 but reward shaping back ON in RELATIVE mode (own - best_opp powered cities), annealed to 0 over the first 100M steps so the back half is pure terminal reward. The champion line trains on terminal reward only; a dense relative signal early gives per-step credit while the long horizon is still hard to bridge, then hands off to the true objective. --reward-shaping here overrides COMMON's --no-reward-shaping (later flag wins). Champion line, gamma 0.999.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --reward-shaping --shaping-mode relative --anneal-shaping-steps 100000000 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"a5-epochs-kl|1605|fork|peers|PERTURBATION (optimization intensity): a1 but n-epochs 8 (2x) with a target-kl 0.03 trust-region cap. More gradient passes per rollout extract more signal per sample near a plateau; the KL cap early-stops the epoch loop before those extra passes push the policy off a cliff (the classic PPO over-update failure). Champion line, gamma 0.999.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --n-epochs 8 --target-kl 0.03 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"a6-gae98|1606|fork|peers|PERTURBATION (advantage horizon): a1 but gae-lambda 0.98 (default 0.95). At gamma 0.999 the reward horizon is long, but GAE at lambda 0.95 still discounts advantages on a ~20-step scale; lifting lambda to 0.98 lengthens the advantage-estimation horizon toward the discount horizon, trading a little variance for less bias on the long macro-level credit assignment. Champion line, gamma 0.999.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --gae-lambda 0.98 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"z6-soup|1607|clone|peers|CRAZY #1 — the gamma-sweep soup (CONTINUED from wave 15). Wave 15 left this fresh clone unfinished at 190M, already competitive for its steps (compare 84.5, eval -0.35). Warm-started (--init-policy-from) from soup-x234.bin, the UNIFORM WEIGHT-SPACE AVERAGE of the shared-init cross-lineage arms x2/x3/x4 (all fork the donor @750M, spanning the gamma peak 0.997/0.999/0.9995 — the fine-tunes-of-a-shared-checkpoint setting where model soups sit in one loss basin). GENTLE lr 3e-5 -> 0 (protect the average while the fresh value head re-learns) + gamma 0.999. RESUMES its own 190M checkpoint to WAVE_STEPS (350M). Does the average across the gamma sweep land in a flatter/better minimum than any single gamma?|--learning-rate 3e-5 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --init-policy-from $SOUP --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"z8-champ-soup|1608|clone|peers|CRAZY #2 — the champion-survivors soup. A FRESH clone warm-started from soup-x5z3z7.bin, the UNIFORM WEIGHT-SPACE AVERAGE of the INCUMBENT x5-champ-g999 and its two most-different wave-15 children z3-nsteps (eval leader) and z7-exploiter (h2h/compare leader) — all three fork x5 @894M so they share ONE loss basin and are a valid soup. This is the boldest plateau bet: weight-averaging can reach a model that forking+annealing x5 CANNOT, because it is not a gradient step trying to escape the basin but a direct move to its flatter interior — the one config-only lever that produces weights unreachable by training x5 against itself. GENTLE lr 3e-5 -> 0 + gamma 0.999. WAVE_STEPS (350M) budget from 0. scripts/make_soup.py builds soup-x5z3z7.bin in --prepare.|--learning-rate 3e-5 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --init-policy-from $SOUP2 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
)


variant_field() { echo "${VARIANTS[$1]}" | cut -d'|' -f"$2"; }

# Comma-separated league dirs of every variant EXCEPT $1 (by name) — the
# --league-peers value for a "peers" arm. Dirs may not exist yet; the trainer
# tolerates that (empty pool slice until the peer launches). Only wave-16 arms
# (the a* perturbation arms + the two continuing soups z6/z8) are peered — never
# an inert wave-9-through-15 dir (the x*/y*/z1-z5/z7 arms, kept for reference
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

    # Build the two model soups (uniform weight-space averages of fine-tunes of a
    # shared checkpoint, so each set sits in one loss basin — Wortsman et al.).
    # Idempotent; only best_model exports are read, so rebuilding is safe. A soup
    # is skipped (with a warning) if an input is missing — its arm then
    # hard-errors at launch on the absent --init-policy-from path.
    #
    #   SOUP  (z6-soup): mean(x2,x3,x4), the gamma-sweep soup (all fork the donor
    #                    @750M, spanning gamma 0.997/0.999/0.9995). Built in wave
    #                    15; present on disk, so this normally no-ops.
    #   SOUP2 (z8-champ-soup): mean(x5,z3,z7), the champion-survivors soup — the
    #                    incumbent x5 and its two most-different wave-15 children
    #                    z3-nsteps / z7-exploiter (both fork x5 @894M -> shared
    #                    basin). NEW this wave.
    build_soup() {  # $1 out-path; $2.. run-dir names whose best_model to average
        local out=$1; shift
        [[ -f $out ]] && return 0
        local missing=0 m args=()
        for m in "$@"; do
            if [[ -f $SWEEP_DIR/$m/best_model.zip ]]; then
                args+=(--model "$SWEEP_DIR/$m/best_model")
            else
                echo "soup input missing: $SWEEP_DIR/$m/best_model.zip" >&2; missing=1
            fi
        done
        if [[ $missing == 0 ]]; then
            echo "building model soup: mean($*) -> $out"
            "$PY" scripts/make_soup.py --out "$out" "${args[@]}"
        else
            echo "skipping soup build for $out (missing inputs); its arm will error at launch" >&2
        fi
    }
    build_soup "$SOUP"  x2-y3-g997 x3-y3-g999 x4-y3-g9995
    build_soup "$SOUP2" x5-champ-g999 z3-nsteps z7-exploiter
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

echo "fork point     : $FORK_FROM (fork arms' first launch only)"
echo "gamma-sweep soup   : $SOUP (z6-soup warm start)"
echo "champ-survivors soup: $SOUP2 (z8-champ-soup warm start)"
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
  tail -f $SWEEP_DIR/z1-champ-cont/train.log
  $PY -m tensorboard.main --logdir $SWEEP_DIR      # league/peer_size, eval/mean_reward
  $PY scripts/run_report.py $SWEEP_DIR/z1-champ-cont
Rank the variants:
  ./scripts/sweep_selfplay.sh --compare    # absolute: vs 3x hard bots (reporting; saturated)
  ./scripts/sweep_selfplay.sh --h2h        # relative: vs 3x the frozen $BASELINE best (primary ranking)
Stop everything:
  ./scripts/sweep_selfplay.sh --stop
EOF
