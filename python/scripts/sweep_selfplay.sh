#!/usr/bin/env bash
#
# sweep_selfplay.sh — wave 10: the first WITHIN-format wave since the obs-600
# reset. Every arm either RESUMES its own 600-format checkpoints or FORKS
# (--resume-from, warm value head) from a pinned 600-format checkpoint. No
# .bin migration anywhere — that machinery was wave 9's format-reset artifact
# and is gone.
#
# WAVE 10 (2026-08-09).
#
# Wave 9 (the obs-600 reset, all 8 arms to a fresh 150M budget) is settled.
# Every arm was a fresh run warm-started via --init-policy-from from a MIGRATED
# clone of the wave-8 leader, so every arm carried a FRESH (random) value head.
# What that measured:
#
#   Frozen-champion eval (best mean_reward vs 3x the wave-8 leader, par ~-0.50):
#     s3-gentle -0.19  s8-sharp -0.21  s2-finish -0.26  s5-placement -0.31
#     s7-exploit -0.32  s4-y3 -0.33  s1-main -0.34  s6-explore -0.43
#   Primary h2h (seat 0 vs 3x the frozen champion; combined 600 games over
#   seeds 12345+99999, mirror par ~25%):
#     s2-finish ~34.2   s3-gentle ~32.5   s8-sharp ~31.9   s1-main ~29.6 (par)
#   The two boards disagreed at the top (eval liked s3, h2h liked s2), so a
#   DIRECT head-to-head broke it (seed 77777, 400 games each direction):
#     s3-gentle @seat0 vs 3x s2 = 25.5%   s2-finish @seat0 vs 3x s3 = 21.2%
#   → s3-gentle wins the direct match both ways (its pool also suppressed lone
#   s2 harder). s3-gentle is the wave-9 champion and the new embedded Expert
#   (80% native vs 3x hard, ~+9pp above the frozen wave-8 champion).
#
# What wave 9 SETTLED (knobs closed / opened):
#   * GENTLE lr (3e-5 -> 0) WON. But every wave-9 arm had a fresh value head,
#     so the win may be nothing more than "a small lr protects the clone while
#     the random critic warms up" — exactly the fresh-value-head guard it was
#     armed to test. Wave 10 forks a WARM value head, so t5-gentle-finish
#     re-runs the recipe with that confound removed: if gentle still beats
#     standard decay from a warm head, lower peak lr is a real lever; if it
#     ties/loses, gentle was format-migration-only. RESOLVE THIS.
#   * DECAY held (s2-finish, s8-sharp both healthy, top of h2h/eval besides
#     s3). Re-armed a SEVENTH straight wave as t2-finish.
#   * ENTROPY UP is dead: s6-explore (ent 0.045) was worst on BOTH boards.
#     Knob closed for good — do not re-arm high entropy.
#   * ENTROPY DOWN (ent 0.015) is safe and mildly good, never a champion:
#     s8-sharp was healthy (0.25 -> 0.19 nats, no collapse) and 3rd. Third and
#     final test as t6-sharp-finish: beat t2 this wave or the knob closes.
#   * PLACEMENT reward did nothing again (s5 ~ s2's parent). Closed a 4th time.
#   * CONSTANT lr is the weakest of the update recipes (s1-main last-but-one on
#     eval); kept only as the t1-main control.
#
# THE STRUCTURE (why wave 10 looks like waves 4-8 again):
#   * The obs-600 format is stable now. Wave-9 checkpoints are 600-wide sb3
#     zips, so forks use --resume-from (full checkpoint: warm policy AND value
#     head, step counter continues) instead of --init-policy-from. No fresh
#     value head, no early eval dip, no gentle-lr guard needed by default.
#   * FORK_FROM = the new champion s3-gentle/best_model (@135.2M). It is
#     RETIRED (its lr annealed to ~0), so its best_model is a stable fork
#     point and the frozen --h2h/--eval yardstick — it never trains again.
#   * The DONOR lineage lives on: s4-y3 (wave 9's reconstituted never-annealed
#     y3 clone, constant lr) RESUMES and keeps compounding, exactly as y3-batch
#     did through waves 4-8. t3-y3-finish forks its PINNED @150M checkpoint and
#     decays — the cross-lineage-decay recipe that won waves 7 (q4) and led
#     wave 8 (r4). It is young again (the format reset restarted y3's clock at
#     150M), so this wave rebuilds the lineage as much as it races it.
#   * runs/sweep3 remains INERT 582-format history + donor archaeology — never
#     resume it, never peer into it (its snapshots would crash a 600 trainer).
#     runs/sweep4 is this epoch's home; wave-9 dirs s1/s2/s5/s6/s7/s8 stay as
#     inert history and are NOT peered (only the wave-10 arms peer each other).
#
# Arm roles (all lr 1e-4, n-steps/batch 1024, mix 0.40/0.40/0.20, past_k 8,
# all peers, unless noted):
#   s4-y3          RESUME the donor: constant lr, never decayed, never forked.
#                  The fork material for this and future cross-lineage decays.
#   t1-main        fork champion; constant lr — the decay/gentle control.
#   t2-finish      t1 + lr decay -> 0. The champion recipe, 7th re-arm.
#   t3-y3-finish   fork=s4-y3 @150M + decay -> 0. Cross-lineage decay from the
#                  never-annealed donor (the q4/r4 champion factory).
#   t4-small-finish  fork champion, batch 512 + decay. Small-batch+decay
#                  co-won wave 6 (p3); keep both batch sizes and a distinct
#                  default-lineage pool opponent.
#   t5-gentle-finish  fork champion, lr 3e-5 -> 0 (warm value head). The wave-9
#                  winner's recipe with the fresh-value-head confound removed.
#                  vs t2-finish: is lower peak lr a real lever or was it a
#                  migration guard? This is the wave's key question.
#   t6-sharp-finish  t2 + ent-coef 0.015. Low-entropy's third and final test;
#                  promote (beats t2) or retire the knob. Collapse guard:
#                  entropy diving toward ~0.1 nats = kill the arm.
#   t7-exploit     mix 0.10/0.70/0.20, past_k 12: the pool hardener, kept a
#                  fifth wave. Hardens everyone's opponents, not a candidate.
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
#   ./scripts/sweep_selfplay.sh --h2h        # RELATIVE: each variant vs 3x the frozen champion (s3-gentle)
#   ./scripts/sweep_selfplay.sh --stop       # stop every running variant
#
# Env overrides (all optional):
#   TOTAL_TIMESTEPS=300000000  CUMULATIVE per-arm target (forks from ~135-150M
#                              gain ~+150-165M; resumes never overshoot)
#   FORK_FROM=runs/sweep4/s3-gentle/best_model   champion fork point + baseline
#   Y3_FORK=runs/sweep4/s4-y3/ckpt_150000000_steps   pinned donor fork stem
#   EVAL_OPPONENT=runs/sweep4/wave10-eval-opponent.bin
#                              frozen eval opponent, exported from FORK_FROM on
#                              first launch (selects best_model; par ~ -0.50)
#   BASELINE=s3-gentle         run-dir name of the frozen --h2h opponent
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
FORK_FROM=${FORK_FROM:-$SWEEP_DIR/s3-gentle/best_model}
Y3_FORK=${Y3_FORK:-$SWEEP_DIR/s4-y3/ckpt_150000000_steps}
EVAL_OPPONENT=${EVAL_OPPONENT:-$SWEEP_DIR/wave10-eval-opponent.bin}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-300000000}
NET_WIDTH=${NET_WIDTH:-128}
NUM_ENVS=${NUM_ENVS:-8}
THREADS=${THREADS:-3}
NICE=${NICE:-10}
STAGGER=${STAGGER:-15}
DRY_RUN=${DRY_RUN:-0}

# The frozen --h2h opponent: the wave-9 champion's best_model — the exact
# weights the fork arms started from and the eval opponent was exported from.
# s3-gentle no longer trains (its lr annealed to 0), so this yardstick never
# moves.
BASELINE=${BASELINE:-s3-gentle}

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
# (first launch --resume-from $FORK_FROM into a fresh dir), or "fork=<stem>"
# (first launch --resume-from that checkpoint stem instead). After the first
# launch every arm resumes its OWN checkpoints.
#
# `pop` is "peers" (the launch loop appends --league-peers with every OTHER
# variant's league dir) or "solo" (no peers).
#
# League mix order is LATEST,PAST,BOTS; the trainer default is 0.5,0.3,0.2.
# The 0.20 bot share is load-bearing (wave-7 q6 faceplant) — do not cut it.
VARIANTS=(
"s4-y3|804|resume|peers|The donor lineage: constant lr 1e-4 + batch 1024, never decayed, never forked — reconstituted at the obs-600 reset and now compounding again. Joins the wave-10 pool unchanged; t3 separately decays a fresh copy of its pinned @150M state. Keeping it alive is what gives future waves an independent history to cross-decay.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"t1-main|1001|fork|peers|The population default and this wave's decay/gentle control: champion weights (warm value head), batch 1024, constant lr, mix 0.40/0.40/0.20. Every finisher reads against this — same food, same fork, no anneal.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"t2-finish|1002|fork|peers|The champion-line recipe re-armed a seventh time: t1-main + lr decay 1e-4 -> 0. Decay has finished on top of six straight champions (y4, z3, p2, q4, r4, and s3's gentle variant); until it loses, every wave re-arms it from the new champion.|--learning-rate 1e-4 --lr-final 0 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"t3-y3-finish|1003|fork=$SWEEP_DIR/s4-y3/ckpt_150000000_steps|peers|Cross-lineage decay: fork the never-annealed donor's pinned @150M state + lr decay -> 0. This is the exact recipe that won wave 7 (q4) and led wave 8 (r4). The donor was reconstituted only this epoch, so its history is young; a top finish means the factory survived the format reset, a mid finish means it just needs more donor steps (which s4-y3 is banking).|--learning-rate 1e-4 --lr-final 0 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"t4-small-finish|1004|fork|peers|Batch 512 + decay: the small-batch recipe stays in play — it co-won wave 6 (p3) and fought to the wire in wave 7 (q3). Half the batch is different update noise and doubles as the pool's most distinct default-lineage opponent.|--learning-rate 1e-4 --lr-final 0 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"t5-gentle-finish|1005|fork|peers|THE key question: fork the champion (WARM value head) with lr 3e-5 -> 0 — the wave-9 winner's recipe minus the fresh-value-head confound. If t5 beats t2-finish from a warm head, lower peak lr is a genuine lever and the next wave pushes it further; if it ties/loses, s3-gentle's win was a migration guard and gentle retires. Watch t5 vs t2 head to head at wave end.|--learning-rate 3e-5 --lr-final 0 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"t6-sharp-finish|1006|fork|peers|t2-finish + ent-coef 0.015 (default 0.03). Low-entropy's third and final test — safe and mildly good in waves 8 (r5) and 9 (s8) but never a champion. Beat t2 this wave or the entropy knob closes. Collapse guard: if entropy dives toward ~0.1 nats early, kill the arm.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.015 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"t7-exploit|1007|fork|peers|The exploiter, kept a fifth wave: mix 0.10/0.70/0.20, past_k 12 — barely plays itself, lives on the population. Below-par h2h as a candidate; its job is hardening everyone else's opponents, not winning the wave.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.10,0.70,0.20 --league-past-k 12"
)


variant_field() { echo "${VARIANTS[$1]}" | cut -d'|' -f"$2"; }

# Comma-separated league dirs of every variant EXCEPT $1 (by name) — the
# --league-peers value for a "peers" arm. Dirs may not exist yet; the trainer
# tolerates that (empty pool slice until the peer launches). Only wave-10 arms
# are peered — never an inert wave-9 dir, and never a 582-format runs/sweep3.
peer_league_dirs() {
    local self=$1 i name out=""
    for i in "${!VARIANTS[@]}"; do
        name=$(variant_field "$i" 1)
        [[ $name == "$self" ]] && continue
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
    # NOTE: best= is eval/mean_reward vs 3x the frozen champion (s3-gentle);
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
# champion (s3-gentle best_model) — the primary ranking. Above-par here ==
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
    --prepare) prepare; echo "wave-10 eval opponent ready in $SWEEP_DIR"; exit 0 ;;
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
echo "donor fork stem: $Y3_FORK (t3's first launch only)"
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

    # One-time wave-10 migration for the continued donor: s4-y3's stored best
    # bar was earned vs the wave-9 eval opponent (the frozen wave-8 leader) and
    # is not comparable with the new frozen champion (s3-gentle). Set the old
    # best aside, drop the bar, and let the new metric re-earn best_model.zip.
    # The marker file makes re-runs a no-op.
    if [[ $init == resume && $DRY_RUN != 1 && -f $dir/best_mean_reward.json \
          && ! -f $dir/.wave10-eval-opponent ]]; then
        [[ -f $dir/best_model.zip && ! -f $dir/best_model.wave9-vs-champion.zip ]] \
            && cp "$dir/best_model.zip" "$dir/best_model.wave9-vs-champion.zip"
        rm "$dir/best_mean_reward.json"
        touch "$dir/.wave10-eval-opponent"
        echo "migrated $name to the wave-10 eval metric (old best kept as best_model.wave9-vs-champion.zip)"
    fi

    # Auto-resume: continue from the arm's own furthest readable checkpoint.
    # Only a fork arm's very first launch uses its fork source; a continuation
    # arm with no checkpoint is an error, not a fresh start.
    start_args=()
    ckpt_stem=""; done_steps=""
    read -r ckpt_stem done_steps < <(latest_checkpoint "$dir") || true
    if [[ -n $ckpt_stem ]]; then
        steps=$(( TOTAL_TIMESTEPS - done_steps ))
        if (( steps <= 0 )); then
            echo "skip $name: already at $done_steps >= target $TOTAL_TIMESTEPS timesteps"
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
        steps=$(( TOTAL_TIMESTEPS - fork_steps ))
        if (( steps <= 0 )); then
            echo "skip $name: fork point already at $fork_steps >= target $TOTAL_TIMESTEPS" >&2
            continue
        fi
        start_args=(--resume-from "$fork_src")
        echo "forking $name from $(basename "$fork_src").zip @ $fork_steps (+$steps)"
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
  tail -f $SWEEP_DIR/t2-finish/train.log
  $PY -m tensorboard.main --logdir $SWEEP_DIR      # league/peer_size, eval/mean_reward
  $PY scripts/run_report.py $SWEEP_DIR/t2-finish
Rank the variants:
  ./scripts/sweep_selfplay.sh --compare    # absolute: vs 3x hard bots (reporting; saturated)
  ./scripts/sweep_selfplay.sh --h2h        # relative: vs 3x the frozen $BASELINE best (primary ranking)
Stop everything:
  ./scripts/sweep_selfplay.sh --stop
EOF
