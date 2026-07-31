#!/usr/bin/env bash
#
# sweep_selfplay.sh — wave 5: fork the wave-4 winner, self-play with a bot anchor.
#
# WAVE 5 (2026-07-31).
#
# Wave 4 finished at 150M steps per arm. Both rankings, 200 games, seed 12345
# (evaluate_lineup rotates seats; seat 0's par is ~25%):
#
#   ABSOLUTE (--compare, vs 3x hard):        RELATIVE (--h2h, vs 3x w3 best):
#     y4-lr-decay      58.5%   <- champion     y3-batch         35.5%
#     w3-low-lr        57.0%                   y4-lr-decay      32.5%
#     w4-big-batch     54.5%                   y2-mirror        29.0%
#     y2-mirror        50.5%                   y5-placement     29.0%
#     y3-batch         49.0%                   y1-nobots        28.5%
#     y6-sp-batch      47.5%                   y6-sp-batch      28.5%
#     y1-nobots        44.0%                   w4-big-batch     17.5%
#     y5-placement     42.0%
#
#   Tiebreak, y3 best (seat 0) vs 3x y4 best: 20.5% — y4 wins the direct
#   match, leads vs hard, and beats the old champion. y4-lr-decay's
#   best_model (144.4M steps, eval +0.43) is the wave-5 fork point AND the
#   frozen --h2h yardstick.
#
# What wave 4 settled:
#
#   * BOTS=0 HURTS. All four no-bots arms (y1, y2, y5, y6) landed at h2h par
#     (~28-29% vs the thing they forked from — i.e. 100M steps of training
#     bought nothing relative to the champion) and BELOW 50% vs hard. The
#     arms that kept the default 20% bot share (y3, y4) beat par on both
#     boards. The 20% grounding share is load-bearing; the wave-4 thesis is
#     rejected in its strong form. Wave 5 still leans into self-play, but
#     with a nonzero bot anchor (0.10) instead of zero.
#
#   * LR DECAY IS A REAL FINISHER. y4 (1e-4 -> 0) converted w3's plateau
#     into the new champion, and its approx-KL hit ~0 as the lr annealed
#     out — it converged. That also means y4 itself CANNOT continue (its lr
#     is 0); its value is as the fork point, and being retired makes its
#     best_model a *stable* h2h opponent, unlike wave 4's moving target.
#
#   * BIG BATCH helps at lr 1e-4 (y3: best h2h) but not at 3e-4 (w4: worst
#     h2h). y3's recipe is still live (constant lr, best_model at 135.6M),
#     so y3 is the one wave-4 arm that KEEPS TRAINING.
#
#   * Placement reward: neutral again, now in the setting it was designed
#     for (y5 ~= y1). Closed for good, along with entropy-up (collapses),
#     shaped start, scratch, and vf-warmup from earlier waves.
#
# The wave-5 menu: y3 continues; seven z-arms fork from the pinned y4 best.
# Every fork carries lr 1e-4; the axes are (a) decay vs constant, (b) batch
# 512 vs 1024, (c) league mix — default 0.5/0.3/0.2 vs self-play-lean
# 0.65/0.25/0.10 vs past-heavy 0.50/0.40/0.10. Four of the seven forks are
# self-play-lean: that is the "lean into self-play" bet, re-run with the
# measured-necessary bot anchor kept.
#
# EVAL CAVEAT: --eval-difficulty hard still has headroom (champion 58.5%)
# but it saturates as arms improve, and a saturated eval selects best_model
# on noise. Once several arms clear ~70% on --compare, rank with --h2h
# (vs 3x the frozen y4 best — exactly the weights every fork started from)
# and consider raising COMPARE_GAMES.
#
# Sized for a 28-core machine: 8 variants x THREADS=3 = 24 cores, leaving
# headroom for the eval passes and the OS.
#
# Re-running is idempotent and self-healing: launching is the same command as
# resuming. For each selected variant the script inspects its run dir and
# picks up where it left off — resume from the furthest-along readable
# checkpoint; if there is none, a fork arm starts from $FORK_FROM and a
# continuation arm refuses loudly (silently restarting a wave-4 run from
# scratch would be wrong). The fork source is only ever used for a variant's
# FIRST launch; after that it resumes its own checkpoints. The running-check
# verifies the recorded PID is still a train_selfplay.py process for THIS run
# dir, so a stale pidfile (e.g. a PID recycled across a reboot) can't block a
# resume or, worse, let two trainers write the same dir. The intended
# operational loop is simply: run it, and if the box reboots or a variant
# crashes, run it again.
#
# TOTAL_TIMESTEPS is CUMULATIVE: sb3 counts timesteps across resumes, and each
# launch is passed only the remaining budget, so re-running never overshoots.
# y3 sits at 150M and the fork point at 144.4M, so the default 250M target
# buys every arm +100-106M — the same increment wave 4 ran.
#
# Resume-lr note: MaskablePPO.load is passed custom_objects built from THIS
# launch's flags, so a fork's lr schedule is its own — y4's decayed-to-zero
# lr does not leak into the forks.
#
# League note for forks: snapshots live in each run dir's own league/ subdir.
# A fork's league starts empty, but the trainer pushes a snapshot of the
# just-loaded weights at training start, so a nonzero PAST share never sees an
# empty pool — the early "past" opponent is simply the fork point itself.
#
# The retired arms (wave 3's w1/w2/w5-w8 and wave 4's w3-low-lr,
# w4-big-batch, y1, y2, y4-lr-decay, y5, y6) were finished or stopped before
# this rewrite and their dirs remain in $SWEEP_DIR as inert history —
# y4-lr-decay's dir must stay, it IS the fork point and h2h opponent. --stop
# only knows the variants in the table below; if a retired arm is somehow
# still running, kill its $SWEEP_DIR/<name>/train.pid by hand.
#
# Usage:
#   ./scripts/sweep_selfplay.sh              # launch/resume all 8 in the background
#   ./scripts/sweep_selfplay.sh 3 5          # launch/resume only variants 3 and 5
#   ./scripts/sweep_selfplay.sh --list       # show the variant table, launch nothing
#   ./scripts/sweep_selfplay.sh --status     # per-variant progress / best eval
#   ./scripts/sweep_selfplay.sh --compare    # ABSOLUTE: each variant vs 3x hard bots
#   ./scripts/sweep_selfplay.sh --h2h        # RELATIVE: each variant vs 3x the frozen champion
#   ./scripts/sweep_selfplay.sh --stop       # stop every running variant
#
# Env overrides (all optional):
#   TOTAL_TIMESTEPS=250000000  CUMULATIVE timesteps per variant (see above)
#   FORK_FROM=runs/sweep3/y4-lr-decay/best_model
#                              pinned fork point for the z-arms (stem, no .zip)
#   SWEEP_DIR=runs/sweep3      root for the per-variant run dirs
#   NET_WIDTH=128              policy width for fresh runs (resumes carry their own)
#   NUM_ENVS=8                 parallel envs per variant (keep equal across variants)
#   THREADS=3                  torch/OMP threads per variant (8 x 3 = 24 of 28 cores)
#   COMPARE_GAMES=200  COMPARE_SEED=12345
#   COMPARE_DETERMINISTIC=1    rank with argmax instead of sampling. Training is
#                              stochastic, so the sampled numbers remain the
#                              primary ranking.
#   NICE=10  STAGGER=15  DRY_RUN=1
#
set -euo pipefail

cd "$(dirname "$0")/.."          # python/

PY=${PY:-.venv/bin/python}
SWEEP_DIR=${SWEEP_DIR:-runs/sweep3}
FORK_FROM=${FORK_FROM:-$SWEEP_DIR/y4-lr-decay/best_model}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-250000000}
NET_WIDTH=${NET_WIDTH:-128}
NUM_ENVS=${NUM_ENVS:-8}
THREADS=${THREADS:-3}
NICE=${NICE:-10}
STAGGER=${STAGGER:-15}
DRY_RUN=${DRY_RUN:-0}

# The frozen --h2h opponent: the retired wave-4 champion's best_model — the
# exact weights every z-arm forked from. Because y4-lr-decay no longer trains
# (its lr annealed to 0), this yardstick never moves, unlike wave 4's.
BASELINE=y4-lr-decay

# Shared across all variants — held constant so the comparison is clean.
# A variant's own flags come after these on the command line, so repeating a
# flag there overrides the value set here.
COMMON=(
    --num-players 4
    --num-envs "$NUM_ENVS"
    --net-width "$NET_WIDTH"
    --no-reward-shaping         # terminal reward is the objective (wave 3's
                                # shaped arm was neutral-to-negative)
    --eval-difficulty hard      # NOT normal: saturated instantly in wave 3,
                                # and this metric selects best_model
    --save-freq 250000          # ~2M timesteps per checkpoint at 8 envs
    --eval-freq 50000           # ~400k timesteps per eval pass
    --eval-episodes 200         # 20 (the trainer default) is too noisy to rank
)

# name|seed|init|hypothesis|extra flags
#
# `init` is "resume" (a continuing wave-4 arm; must already have checkpoints)
# or "fork" (first launch resumes from $FORK_FROM into a fresh dir).
#
# League mix order is LATEST,PAST,BOTS; the trainer default is 0.5,0.3,0.2.
VARIANTS=(
"y3-batch|303|resume|The one continuing wave-4 arm: best h2h vs the old champion (35.5%) on constant lr 1e-4 + batch 1024, and still setting bests at 135.6M — its recipe has no built-in stopping point. Continues unchanged to see if it catches the y4 lineage.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024"
"z1-cont|401|fork|Control: the champion's weights on w3's recipe (constant lr 1e-4, default league). y4 cannot continue at lr 0, so this is its plain 'unfrozen' successor — if the decay's win was convergence-to-a-local-optimum, restarting at 1e-4 should walk away from it and score BELOW the frozen yardstick early.|--learning-rate 1e-4"
"z2-redecay|402|fork|Repeat the proven finisher from the new start: lr 1e-4 -> 0 across the remaining budget, default league. Wave 4's single biggest win re-armed; if decay's benefit was one-shot (you can only converge once), this lands at par and closes the question.|--learning-rate 1e-4 --lr-final 0"
"z3-batch-decay|403|fork|Stack the two arms that beat h2h par: batch 1024 (y3's axis) + lr decay (y4's axis). They were only ever tested separately; if even partly independent this arm should lead the decay group.|--learning-rate 1e-4 --lr-final 0 --n-steps 1024 --batch-size 1024"
"z4-sp-lean|404|fork|Self-play-lean league 0.65/0.25/0.10 at constant lr 1e-4: the wave-4 bet re-run with the measured-necessary bot anchor kept. Wave 4 only tested 0.20 bots vs zero; 0.10 is the untested middle where the anchor holds but self-play pressure rises.|--learning-rate 1e-4 --league-mix 0.65,0.25,0.10"
"z5-sp-decay|405|fork|Self-play-lean 0.65/0.25/0.10 + the decay finisher. If z4 beats z1 (mix effect) and z2 beats z1 (decay effect), this is the combination candidate for the next fork point.|--learning-rate 1e-4 --lr-final 0 --league-mix 0.65,0.25,0.10"
"z6-past-heavy|406|fork|Lean into self-play via the PAST pool instead of LATEST: 0.50/0.40/0.10. y2's 0.85-LATEST mirror was the weakest self-play arm, so raw mirror pressure isn't the driver; a deeper frozen-snapshot pool buys opponent diversity without the nonstationarity.|--learning-rate 1e-4 --league-mix 0.50,0.40,0.10"
"z7-sp-all|407|fork|Everything at once: self-play-lean 0.65/0.25/0.10 + batch 1024 + decay. The kitchen-sink bet — mirror games are the noisiest reward source, the big batch is aimed at that variance, and the decay finishes it. Reads against z3/z5 to attribute any win.|--learning-rate 1e-4 --lr-final 0 --n-steps 1024 --batch-size 1024 --league-mix 0.65,0.25,0.10"
)


variant_field() { echo "${VARIANTS[$1]}" | cut -d'|' -f"$2"; }

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

list_variants() {
    printf '%-20s %-6s %-8s %s\n' NAME SEED INIT FLAGS
    for i in "${!VARIANTS[@]}"; do
        printf '%-20s %-6s %-8s %s\n' \
            "$(( i + 1 )). $(variant_field "$i" 1)" \
            "$(variant_field "$i" 2)" \
            "$(variant_field "$i" 3)" \
            "$(variant_field "$i" 5)"
    done
}

status() {
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
# against 0. Watch for saturation: past ~70-80% this stops separating arms and
# --h2h becomes the ranking that matters.
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
# wave-4 champion — the exact weights the z-arms forked from, and no longer a
# moving target since y4-lr-decay is retired. This is the primary ranking once
# the vs-hard eval saturates. Above-par here == genuinely past the fork point.
h2h() {
    local games=${COMPARE_GAMES:-200} seed=${COMPARE_SEED:-12345}
    local det=(); [[ ${COMPARE_DETERMINISTIC:-0} == 1 ]] && det=(--deterministic)
    local base="$SWEEP_DIR/$BASELINE/best_model"
    [[ -f ${base}.zip ]] || { echo "baseline $BASELINE has no best_model (sync $SWEEP_DIR/$BASELINE)" >&2; exit 1; }
    echo "=== self-baseline: 4x $BASELINE ($games games) ==="
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
esac

# Which variants to launch (1-based indices; default all).
if (( $# )); then
    SELECTED=("$@")
else
    SELECTED=($(seq 1 ${#VARIANTS[@]}))
fi

[[ -x $PY ]] || { echo "no interpreter at $PY (run 'make develop' first)" >&2; exit 1; }

echo "fork point     : $FORK_FROM (z-arms' first launch only)"
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
    why=$(variant_field "$i" 4)
    read -r -a extra <<< "$(variant_field "$i" 5)"
    dir="$SWEEP_DIR/$name"

    # Already running? Never start a second writer on the same run dir. The
    # check confirms the pidfile's PID is genuinely this variant's trainer, so
    # a stale/recycled PID neither blocks a needed resume nor risks a duplicate.
    live=$(running_pid "$dir")
    if [[ -n $live ]]; then
        echo "skip $name: already running (pid $live)"
        continue
    fi

    # Auto-resume: continue from the arm's own furthest readable checkpoint.
    # Only a fork arm's very first launch uses $FORK_FROM; a continuation arm
    # with no checkpoint is an error, not a fresh start.
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
        if [[ ! -f ${FORK_FROM}.zip ]]; then
            echo "cannot fork $name: no checkpoint at ${FORK_FROM}.zip" >&2
            echo "(set FORK_FROM, or sync the wave-4 champion's run dir first)" >&2
            exit 1
        fi
        fork_steps=$(zip_steps "${FORK_FROM}.zip") || {
            echo "cannot fork $name: ${FORK_FROM}.zip is unreadable" >&2; exit 1; }
        steps=$(( TOTAL_TIMESTEPS - fork_steps ))
        if (( steps <= 0 )); then
            echo "skip $name: fork point already at $fork_steps >= target $TOTAL_TIMESTEPS" >&2
            continue
        fi
        start_args=(--resume-from "$FORK_FROM")
        echo "forking $name from $(basename "$FORK_FROM").zip @ $fork_steps (+$steps)"
    else
        echo "refusing to start $name: it continues a wave-4 run but $dir has no" >&2
        echo "readable checkpoint. Sync the wave-4 run dir (or fix SWEEP_DIR)." >&2
        exit 1
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
        echo "# launched: $(date -Is)  init: $init  target: $TOTAL_TIMESTEPS cumulative"
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
  ./scripts/sweep_selfplay.sh --status
  tail -f $SWEEP_DIR/z2-redecay/train.log
  $PY -m tensorboard.main --logdir $SWEEP_DIR      # rollout/entropy_loss, eval/mean_reward
  $PY scripts/run_report.py $SWEEP_DIR/z2-redecay
Rank the variants:
  ./scripts/sweep_selfplay.sh --compare    # absolute: vs 3x hard bots (watch for saturation)
  ./scripts/sweep_selfplay.sh --h2h        # relative: vs 3x the frozen $BASELINE best (primary once saturated)
Stop everything:
  ./scripts/sweep_selfplay.sh --stop
EOF
