#!/usr/bin/env bash
#
# sweep_selfplay.sh — wave 14: triangulate the gamma peak INSIDE the winning
# cross-lineage recipe + two CRAZY probes. Six "progress" arms RESUME their own
# 600-format checkpoints or FORK (--resume-from, warm policy+value head) from a
# pinned 600-format checkpoint — the proven waves-4-8/10-13 structure. Two arms
# go off-script (pure Monte-Carlo return, and a placement terminal reward).
#
# WAVE 14 (2026-08-23).
#
# WAVE 13 (map the gamma curve + two crazy probes) is settled. All 8 arms
# reached their targets (forks/resumes 750M-900M; the wide clone 300M). Results:
#
#   Frozen-champion eval (best mean_reward vs 3x v7-gamma, par ~-0.50):
#     w5-y3-gamma -0.26  w4-gamma-nent -0.29  w1-champ-gamma -0.33
#     w6-gamma-max -0.34  w2-gamma999 -0.35  w3-gamma995 -0.37
#     s4-y3 -0.40  v6-wide -0.65
#   Compare vs 3x hard (saturated, reporting-only, seat-0 all-bots par ~21.5%):
#     w6-gamma-max 85.0  w1-champ-gamma 83.5  w2-gamma999 81.5  w5-y3-gamma 81.0
#     w3-gamma995 80.5  w4-gamma-nent 80.5  s4-y3 80.0  v6-wide 66.0
#   Primary h2h (seat 0 vs 3x the frozen champion v7-gamma; mirror par ~26.5%):
#     w5-y3-gamma 33.0  w2-gamma999 28.5  w3-gamma995 25.5  w4-gamma-nent 25.0
#     s4-y3 24.0  w6-gamma-max 24.0  w1-champ-gamma 23.0  v6-wide 10.0
#   w5-y3-gamma LED BOTH meaningful boards (eval -0.26 and h2h 33.0, +6.5pp over
#   par — the largest margin). The DIRECT decider (seed 77777, 400 games each
#   way, 4-way par 25%) confirmed it both directions of both matches:
#     w5 @seat0 vs 3x v7 = 29.0%  BEAT  v7 @seat0 vs 3x w5 = 21.2%
#     w5 @seat0 vs 3x w2 = 26.2%  BEAT  w2 @seat0 vs 3x w5 = 18.5%
#   w5-y3-gamma is the wave-13 champion and the new embedded Expert; it beat the
#   OUTGOING champion v7-gamma decisively both ways (29.0% vs par 25% as
#   challenger; v7 sank to 21.2% as challenger against it).
#
# What wave 13 SETTLED / OPENED:
#   * *** CROSS-LINEAGE DECAY WON. *** w5-y3-gamma forks the never-annealed
#     DONOR (s4-y3, constant lr, gamma 0.99) at @600M, then adds decay -> 0 +
#     gamma 0.997 + NORMAL entropy. It beat the champion-line re-arm
#     w1-champ-gamma (the exact wave-12 recipe, fork v7-gamma) on every board
#     (h2h 33 vs 23, eval -0.26 vs -0.33). Second straight wave the cross-
#     lineage line led the pack (v4-y3 was 3rd in wave 12); now it is #1. The
#     donor's independent, never-annealed history is the best fork material we
#     have. KEEP THE DONOR ALIVE and keep cross-lineage forks central.
#   * GAMMA still climbs, but the peak and the lineage are CONFOUNDED. On the
#     champion-line re-arms the h2h ordering was w3-995 (25.5) / w1-997 (23.0,
#     a noisy dip) / w2-999 (28.5), and w6-1.0 (24.0) < w2-999 — so gamma wants
#     to be HIGHER than 0.997 (~0.999) but 1.0 starts to hurt. BUT the decider
#     showed the gamma-0.997 cross-lineage w5 beats the gamma-0.999 champion-
#     line w2 both directions, so we cannot tell whether w5 won on lineage or on
#     gamma. Wave 14's core job: sweep gamma (0.997 / 0.999 / 0.9995) INSIDE the
#     winning cross-lineage recipe to find the true peak, un-confounded, with
#     champion-line controls at the same gammas.
#   * NORMAL ENTROPY >= LOW ENTROPY now. w4-gamma-nent (ent 0.03) >= w1 (ent
#     0.015) on h2h (25.0 vs 23.0) and the winner w5 used normal entropy. Low
#     entropy no longer earns its place in the high-gamma regime; every wave-14
#     arm runs the trainer-default ent 0.03. (Low-entropy line retired.)
#   * DECAY held (10th straight champion is a decay arm). The constant-lr donor
#     s4-y3 is useful ONLY as fork material, never a contender (eval -0.40, h2h
#     24.0). Kept alive as the cross-lineage supply.
#   * *** WIDE NET (192) FAILED. *** v6-wide, the capacity-ceiling probe, was
#     dead last both waves it ran (h2h 10.0, compare 66.0) even at 300M. The 128
#     field is NOT capacity-bound; the 2.25x net just needs far more data than a
#     150-300M clone wave and never caught up. RETIRED — no wide arm in wave 14
#     (its 192-wide snapshots stay on disk but never train or peer again).
#
# 150M per-arm convergence budget still holds — eval peaks land in the back
# third as lr-decay anneals to 0. Forks from w5-y3-gamma @~748M gain ~+150M to
# 900M cumulative; the donor s4-y3 resumes 750M -> 900M.
#
# THE STRUCTURE — six progress arms + two crazy arms:
#   * FORK_FROM = the new champion w5-y3-gamma/best_model (@~748M). RETIRED (lr
#     annealed to ~0), so its best_model is a stable fork point AND the frozen
#     --h2h/--eval yardstick AND the embedded Expert — it never trains.
#   * The DONOR lineage lives on: s4-y3 (never-annealed, constant lr, gamma 0.99)
#     RESUMES 750M -> 900M. The three cross-lineage arms (x2/x3/x4) fork its
#     PINNED @750M checkpoint — the freshest donor state — and add decay + a
#     gamma from the swept curve. This is the recipe that produced w5.
#   * runs/sweep4 is this epoch's home. Inert history NOT peered: wave-9 (s*),
#     wave-10 (t*), wave-11 (u*), wave-12 (v1-v7), wave-13 losers kept for
#     reference only. runs/sweep3 is inert 582-format — never resume, never peer.
#     Only wave-14 arms + the resuming donor s4-y3 peer.
#
# Arm roles (all lr 1e-4, n-steps/batch 1024, mix 0.40/0.40/0.20, past_k 8,
# ent 0.03, all peers, all fork the champion w5-y3-gamma @~748M, unless noted):
#   s4-y3          RESUME the donor: constant lr, never decayed, never forked,
#                  gamma 0.99. The fork material for the cross-lineage arms.
#   x1-champ       THE wave-13 winning recipe re-armed on the CHAMPION's own
#                  line: fork the champion (w5-y3-gamma) + decay -> 0 + gamma
#                  0.997. Control against x2 (same recipe, cross-lineage) and
#                  the champion-line anchor of the gamma sweep.
#   x2-y3-g997     Cross-lineage @ gamma 0.997: fork the donor's pinned @750M +
#                  decay -> 0 + gamma 0.997. Reproduces w5's EXACT recipe from
#                  the advanced donor state; the factory continuity + center of
#                  the cross-lineage gamma sweep.
#   x3-y3-g999     Cross-lineage @ gamma 0.999: x2 but gamma 0.999. THE
#                  combination bet — the winning lineage carried to the higher
#                  gamma the champion-line board pointed at. Presumptive next
#                  champion.
#   x4-y3-g9995    Cross-lineage @ gamma 0.9995: x2 but gamma 0.9995, bracketing
#                  the peak just below the 1.0 that hurt. x2/x3/x4 triangulate
#                  the gamma curve (0.997/0.999/0.9995) INSIDE the winning
#                  cross-lineage, un-confounded from lineage.
#   x5-champ-g999  Champion line @ gamma 0.999: fork the champion + decay +
#                  gamma 0.999. The clean lineage control against x3 (same gamma
#                  0.999, champion vs cross-lineage) — re-confirms cross-
#                  lineage's edge at the new gamma.
#   y6-mc-return   *** CRAZY #1: the pure Monte-Carlo return. *** fork champion +
#                  decay + gamma 1.0 AND gae-lambda 1.0 together. w6-gamma-max
#                  tried gamma 1.0 with lambda 0.95 (GAE still bootstraps through
#                  the value net) and it slightly hurt; setting lambda 1.0 too
#                  makes the advantage the ACTUAL full-game return with zero
#                  bootstrapping — the true limit of the gamma breakthrough.
#                  Value learning may destabilize with no bootstrap horizon —
#                  watch value_loss/explained_variance/approx-kl; entropy toward
#                  ~0.1 nats = kill it. gamma/lambda never touch the exported net.
#   y7-placement   *** CRAZY #2: the placement terminal reward. *** fork champion
#                  + decay + gamma 0.999 + --terminal-reward placement. The whole
#                  sweep has used winloss (+1/-1); placement maps finish rank
#                  onto [+1, +1/3, -1/3, -1], a denser gradient that values 2nd
#                  over last. Combined with high gamma it propagates a finer
#                  finish-position signal through the whole game (does teaching
#                  "climb from 4th to 2nd" build better positioning than pure
#                  win/lose?). EVAL STAYS WINLOSS, so eval/mean_reward and the
#                  exported artifact stay comparable to every other arm.
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
#   ./scripts/sweep_selfplay.sh --h2h        # RELATIVE: each variant vs 3x the frozen champion (w5-y3-gamma)
#   ./scripts/sweep_selfplay.sh --stop       # stop every running variant
#
# Env overrides (all optional):
#   TOTAL_TIMESTEPS=900000000  CUMULATIVE per-arm target (forks from ~748-750M
#                              gain ~+150M; resumes never overshoot)
#   FORK_FROM=runs/sweep4/w5-y3-gamma/best_model   champion fork point + baseline
#   Y3_FORK=runs/sweep4/s4-y3/ckpt_750000000_steps   pinned donor fork stem
#   EVAL_OPPONENT=runs/sweep4/wave14-eval-opponent.bin
#                              frozen eval opponent, exported from FORK_FROM on
#                              first launch (selects best_model; par ~ -0.50)
#   BASELINE=w5-y3-gamma       run-dir name of the frozen --h2h opponent
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
FORK_FROM=${FORK_FROM:-$SWEEP_DIR/w5-y3-gamma/best_model}
Y3_FORK=${Y3_FORK:-$SWEEP_DIR/s4-y3/ckpt_750000000_steps}
EVAL_OPPONENT=${EVAL_OPPONENT:-$SWEEP_DIR/wave14-eval-opponent.bin}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-900000000}
WAVE_STEPS=${WAVE_STEPS:-300000000}   # fresh-clone arms' cumulative target (none this wave; generic machinery kept)
CLONE_192=${CLONE_192:-$SWEEP_DIR/clone-192.bin}   # retired wide-net warm start (no clone arm in wave 14)
NET_WIDTH=${NET_WIDTH:-128}
NUM_ENVS=${NUM_ENVS:-8}
THREADS=${THREADS:-3}
NICE=${NICE:-10}
STAGGER=${STAGGER:-15}
DRY_RUN=${DRY_RUN:-0}

# The frozen --h2h opponent: the wave-13 champion's best_model — the exact
# weights the fork arms started from and the eval opponent was exported from.
# w5-y3-gamma no longer trains (its lr annealed to 0), so this yardstick never
# moves. It is also the new embedded Expert.
BASELINE=${BASELINE:-w5-y3-gamma}

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
# launch is a FRESH run warm-started from a behavior clone via --init-policy-from
# — the crazy wide-net arm, whose --net-width/--init-policy-from live in its
# extra flags and are ignored by the trainer on any later --resume-from). After
# the first launch every arm resumes its OWN checkpoints.
#
# `pop` is "peers" (the launch loop appends --league-peers with every OTHER
# variant's league dir) or "solo" (no peers).
#
# League mix order is LATEST,PAST,BOTS; the trainer default is 0.5,0.3,0.2.
# The 0.20 bot share is load-bearing (wave-7 q6 faceplant) — do not cut it.
VARIANTS=(
"s4-y3|804|resume|peers|The donor lineage: constant lr 1e-4 + batch 1024, gamma 0.99, never decayed, never forked — compounding since the obs-600 reset (now @750M, on to 900M). The three cross-lineage arms (x2/x3/x4) fork a fresh copy of its pinned @750M state and add decay + a swept gamma. Keeping it alive is what gives every wave an independent, never-annealed history to cross-decay — and it produced wave 13's champion (w5-y3-gamma).|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"x1-champ|1401|fork|peers|THE wave-13 winning recipe re-armed on the CHAMPION's own line: fork the champion (w5-y3-gamma) + lr decay 1e-4 -> 0 + gamma 0.997 + normal ent (0.03). Control against x2 (same recipe, cross-lineage) and the champion-line anchor of the gamma sweep. Collapse guard: entropy diving toward ~0.1 nats = kill the arm.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.997 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"x2-y3-g997|1402|fork=$SWEEP_DIR/s4-y3/ckpt_750000000_steps|peers|Cross-lineage @ gamma 0.997: fork the never-annealed donor's pinned @750M state + lr decay -> 0 + gamma 0.997 + normal ent. Reproduces w5's EXACT winning recipe from the advanced donor state — the factory continuity AND the center of the cross-lineage gamma sweep (x2/x3/x4 = 0.997/0.999/0.9995).|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.997 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"x3-y3-g999|1403|fork=$SWEEP_DIR/s4-y3/ckpt_750000000_steps|peers|Cross-lineage @ gamma 0.999: x2 but gamma 0.999 (0.999^50 ~= 0.95 vs 0.997^50 ~= 0.86). THE combination bet — the winning lineage carried to the higher gamma the wave-13 champion-line board pointed at (w2-gamma999 was 2nd there). Presumptive next champion. Watch value_loss/approx-kl — less discounting is harder to fit.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"x4-y3-g9995|1404|fork=$SWEEP_DIR/s4-y3/ckpt_750000000_steps|peers|Cross-lineage @ gamma 0.9995: x2 but gamma 0.9995, bracketing the peak just below the gamma 1.0 that HURT in wave 13 (w6-gamma-max h2h 24.0 < w2-gamma999 28.5). x2/x3/x4 triangulate the gamma curve INSIDE the winning cross-lineage so the peak is chosen from a measured curve, un-confounded from lineage.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.9995 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"x5-champ-g999|1405|fork|peers|Champion line @ gamma 0.999: fork the champion (w5-y3-gamma) + decay -> 0 + gamma 0.999 + normal ent. The clean LINEAGE control against x3 (same gamma 0.999, champion vs cross-lineage) — the wave-13 decider could not separate lineage from gamma, so this pins the lineage effect at the new gamma. If x3 > x5, cross-lineage's edge holds at 0.999.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"y6-mc-return|1406|fork|peers|CRAZY #1 — the pure Monte-Carlo return. fork the champion + decay + normal ent + gamma 1.0 AND gae-lambda 1.0 together. w6-gamma-max tried gamma 1.0 with lambda 0.95 (GAE still bootstraps through the value net) and it slightly hurt; setting lambda 1.0 too makes the advantage the ACTUAL full-game return with zero bootstrapping — the true limit of the gamma breakthrough. x3 is the discounted control. Value learning may destabilize with no bootstrap horizon — watch value_loss/explained_variance/approx-kl; entropy toward ~0.1 nats = kill it. gamma/lambda never touch the exported net.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 1.0 --gae-lambda 1.0 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"y7-placement|1407|fork|peers|CRAZY #2 — the placement terminal reward. fork the champion + decay + normal ent + gamma 0.999 + --terminal-reward placement. The whole sweep has used winloss (+1/-1); placement maps finish rank onto [+1, +1/3, -1/3, -1], a denser gradient that values 2nd over last. Combined with high gamma it propagates a finer finish-position signal through the whole game — does teaching 'climb from 4th to 2nd' build better positioning than pure win/lose? x3 is the winloss control at the same gamma. EVAL STAYS WINLOSS, so eval/mean_reward and the exported artifact stay comparable to every other arm.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --terminal-reward placement --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
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
    # NOTE: best= is eval/mean_reward vs 3x the frozen champion (v7-gamma);
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
# champion (v7-gamma best_model) — the primary ranking. Above-par here ==
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
    --prepare) prepare; echo "wave-14 eval opponent ready in $SWEEP_DIR"; exit 0 ;;
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
echo "donor fork stem: $Y3_FORK (w5-y3-gamma's first launch only)"
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

    # One-time wave-14 migration for any arm continuing from an earlier wave
    # (the resuming donor s4-y3): its stored best bar was earned vs the wave-13
    # eval opponent (the frozen v7-gamma) and is not comparable with the new
    # frozen champion (w5-y3-gamma). Set the old best aside, drop the bar, and
    # let the new metric re-earn best_model.zip. Fresh fork arms have no best bar
    # yet, so they skip this. The marker makes re-runs a no-op.
    if [[ $DRY_RUN != 1 && -f $dir/best_mean_reward.json \
          && ! -f $dir/.wave14-eval-opponent ]]; then
        [[ -f $dir/best_model.zip && ! -f $dir/best_model.wave13-vs-champion.zip ]] \
            && cp "$dir/best_model.zip" "$dir/best_model.wave13-vs-champion.zip"
        rm "$dir/best_mean_reward.json"
        touch "$dir/.wave14-eval-opponent"
        echo "migrated $name to the wave-14 eval metric (old best kept as best_model.wave13-vs-champion.zip)"
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
  tail -f $SWEEP_DIR/w1-champ-gamma/train.log
  $PY -m tensorboard.main --logdir $SWEEP_DIR      # league/peer_size, eval/mean_reward
  $PY scripts/run_report.py $SWEEP_DIR/w1-champ-gamma
Rank the variants:
  ./scripts/sweep_selfplay.sh --compare    # absolute: vs 3x hard bots (reporting; saturated)
  ./scripts/sweep_selfplay.sh --h2h        # relative: vs 3x the frozen $BASELINE best (primary ranking)
Stop everything:
  ./scripts/sweep_selfplay.sh --stop
EOF
