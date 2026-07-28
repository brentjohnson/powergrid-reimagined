#!/usr/bin/env bash
#
# sweep_selfplay.sh — wave 4: finetune the wave-3 winner, leaning into self-play.
#
# WAVE 4 (2026-07-28).
#
# Wave 3 finished at 50M steps per arm and — unlike waves 1-2 — was measured
# against the current rules/env/action space, so its numbers are real priors:
#
#   vs 3x hard, 200 games (seat 0's structural fair share ~25%):
#     w3-low-lr        51.5%   <- champion; exported as the embedded Expert policy
#     w4-big-batch     42.0%
#     w2-vf-warmup     40.0%
#     w8-scratch       37.0%   (control: matched the clone-start baseline)
#     w1-base          36.5%
#     w7-shaped-start  35.5%
#     w6-placement     30.5%
#     w5-ent-up         8.0%   (entropy-up collapse, reproduced on a WORKING env)
#
# What wave 4 does with that:
#
#   * KEEP TRAINING the two arms that earned it: w3-low-lr (the champion, now
#     also the --h2h opponent) and w4-big-batch (the strongest arm on an
#     independent axis). Same dirs, same flags, target raised to
#     TOTAL_TIMESTEPS cumulative.
#
#   * REPLACE the other six with FORKS of the champion: every y-arm resumes
#     from the same pinned w3 checkpoint ($FORK_FROM, 50M steps) into its own
#     fresh dir, carrying w3's recipe (constant lr 1e-4) plus one further idea.
#     Pinning matters: w3 itself keeps training, so forking "w3's latest"
#     would give later-launched arms a head start; the pin keeps every fork's
#     starting weights identical.
#
#   * LEAN INTO SELF-PLAY. The heuristic bots cannot teach the end-game push —
#     they never fight the closing race the way a strong opponent does — so
#     past ~50% vs hard the marginal training signal is in playing yourself,
#     not them. Four of the six forks remove bots from the opponent pool
#     entirely (--league-mix LATEST,PAST,BOTS with BOTS=0; the trainer
#     supports a zero component). Two guards against forgetting how to beat
#     bots remain: the frozen-past snapshot pool anchors the league, and the
#     vs-hard eval still selects best_model.
#
# Closed questions, deliberately NOT re-tested: entropy up (collapses — now
# confirmed on a working env, twice); shaped start and placement from a fresh
# clone (neutral-to-negative; placement returns below, but in the mirror-match
# setting it was designed for); scratch vs clone (washed out by 50M steps —
# the recipe dominates the init); vf warmup (mild, and moot now the critic is
# trained).
#
# EVAL CAVEAT: --eval-difficulty hard still has headroom (champion ~52%,
# ceiling 100%) but it saturates as arms improve, and a saturated eval selects
# best_model on noise. Once several arms clear ~70% on --compare, rank them
# with --h2h (vs 3x the continuing champion) and consider raising
# COMPARE_GAMES.
#
# Sized for a 28-core machine: 8 variants x THREADS=3 = 24 cores, leaving
# headroom for the eval passes and the OS.
#
# Re-running is idempotent and self-healing: launching is the same command as
# resuming. For each selected variant the script inspects its run dir and
# picks up where it left off — resume from the furthest-along readable
# checkpoint; if there is none, a fork arm starts from $FORK_FROM and a
# continuation arm refuses loudly (silently restarting a wave-3 run from
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
# The wave-3 dirs sit at 50M, so the default 150M target buys every arm +100M.
#
# League note for forks: snapshots live in each run dir's own league/ subdir.
# A fork's league starts empty, but the trainer pushes a snapshot of the
# just-loaded weights at training start, so a nonzero PAST share never sees an
# empty pool — the early "past" opponent is simply the fork point itself.
#
# The retired wave-3 arms (w1, w2, w5, w6, w7, w8) were stopped before this
# rewrite and their dirs remain in $SWEEP_DIR as inert history. --stop only
# knows the variants in the table below; if a retired arm is somehow still
# running, kill its $SWEEP_DIR/<name>/train.pid by hand.
#
# Usage:
#   ./scripts/sweep_selfplay.sh              # launch/resume all 8 in the background
#   ./scripts/sweep_selfplay.sh 3 5          # launch/resume only variants 3 and 5
#   ./scripts/sweep_selfplay.sh --list       # show the variant table, launch nothing
#   ./scripts/sweep_selfplay.sh --status     # per-variant progress / best eval
#   ./scripts/sweep_selfplay.sh --compare    # ABSOLUTE: each variant vs 3x hard bots
#   ./scripts/sweep_selfplay.sh --h2h        # RELATIVE: each variant vs 3x the champion
#   ./scripts/sweep_selfplay.sh --stop       # stop every running variant
#
# Env overrides (all optional):
#   TOTAL_TIMESTEPS=150000000  CUMULATIVE timesteps per variant (see above)
#   FORK_FROM=runs/sweep3/w3-low-lr/ckpt_50000000_steps
#                              pinned fork point for the y-arms (stem, no .zip)
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
FORK_FROM=${FORK_FROM:-$SWEEP_DIR/w3-low-lr/ckpt_50000000_steps}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-150000000}
NET_WIDTH=${NET_WIDTH:-128}
NUM_ENVS=${NUM_ENVS:-8}
THREADS=${THREADS:-3}
NICE=${NICE:-10}
STAGGER=${STAGGER:-15}
DRY_RUN=${DRY_RUN:-0}

# The champion and continuing reference arm; the opponent for --h2h.
BASELINE=w3-low-lr

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
# `init` is "resume" (a continuing wave-3 arm; must already have checkpoints)
# or "fork" (first launch resumes from $FORK_FROM into a fresh dir).
#
# League mix order is LATEST,PAST,BOTS; the trainer default is 0.5,0.3,0.2.
VARIANTS=(
"w3-low-lr|203|resume|WAVE-3 CHAMPION (51.5% vs hard), continuing unchanged: constant lr 1e-4 on the trainer's own defaults. Doubles as the --h2h opponent, so every fork is measured against the thing it forked from, trained equally long.|--learning-rate 1e-4"
"w4-big-batch|204|resume|Runner-up (42.0%), continuing unchanged: bigger rollout/minibatch (512 -> 1024) at the default lr 3e-4. The only strong arm on a different axis than 'move less' -- kept to see whether it converges on, or past, the champion with more steps.|--n-steps 1024 --batch-size 1024"
"y1-nobots|301|fork|Champion's recipe with bots removed from the pool (league 0.60/0.40/0.00). The direct test of the wave-4 thesis: bots cannot teach the end-game push, so their 20% of rollouts is dead weight now. The past pool keeps the anchor.|--learning-rate 1e-4 --league-mix 0.60,0.40,0.00"
"y2-mirror|302|fork|Nearly pure mirror-match (league 0.85/0.15/0.00): maximum pressure from the strongest available opponent -- the current self -- accepting nonstationarity risk; the past share is kept small but nonzero to damp cycles. If y1 beats w3 and y2 beats y1, opponent strength is the driver.|--learning-rate 1e-4 --league-mix 0.85,0.15,0.00"
"y3-batch|303|fork|Stack the two measured winners: lr 1e-4 AND n-steps/batch 1024. Wave 3 tested them as separate single knobs off the same base; if the effects are even partly independent this arm should lead early.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024"
"y4-lr-decay|304|fork|The champion holds lr constant; this fork decays it (1e-4 -> 0 across the remaining budget). Classic finishing schedule: if w3's late progress is noise around an optimum, annealing converts it into convergence -- and if it stalls below w3, constant lr was doing real work.|--learning-rate 1e-4 --lr-final 0"
"y5-placement|305|fork|Placement reward + no bots. Rank-shaped terminal reward lost from a fresh clone (w6: random critic, vs bots); this is the setting it was designed for -- near-equal mirror games, where win/loss collapses toward a coin flip and the between-losers gradient is exactly the end-game-push signal.|--learning-rate 1e-4 --league-mix 0.60,0.40,0.00 --terminal-reward placement"
"y6-sp-batch|306|fork|The wave-4 bet, combined: no bots AND big batch on the champion's lr. Mirror games between near-equal policies are the noisiest reward source available; the bigger batch is aimed at exactly that variance.|--learning-rate 1e-4 --league-mix 0.60,0.40,0.00 --n-steps 1024 --batch-size 1024"
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

# RELATIVE ranking: each variant in seat 0 against three copies of the
# champion. This is the wave-4 primary once the vs-hard eval saturates, and it
# is exactly fair: every fork started from the champion's own weights.
h2h() {
    local games=${COMPARE_GAMES:-200} seed=${COMPARE_SEED:-12345}
    local det=(); [[ ${COMPARE_DETERMINISTIC:-0} == 1 ]] && det=(--deterministic)
    local base="$SWEEP_DIR/$BASELINE/best_model"
    [[ -f ${base}.zip ]] || { echo "baseline $BASELINE has no best_model yet" >&2; exit 1; }
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

echo "fork point     : $FORK_FROM (y-arms' first launch only)"
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
            echo "(set FORK_FROM, or sync the wave-3 champion's run dir first)" >&2
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
        echo "refusing to start $name: it continues a wave-3 run but $dir has no" >&2
        echo "readable checkpoint. Sync the wave-3 run dir (or fix SWEEP_DIR)." >&2
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
  tail -f $SWEEP_DIR/$BASELINE/train.log
  $PY -m tensorboard.main --logdir $SWEEP_DIR      # rollout/entropy_loss, eval/mean_reward
  $PY scripts/run_report.py $SWEEP_DIR/$BASELINE
Rank the variants:
  ./scripts/sweep_selfplay.sh --compare    # absolute: vs 3x hard bots (watch for saturation)
  ./scripts/sweep_selfplay.sh --h2h        # relative: vs 3x $BASELINE (primary once saturated)
Stop everything:
  ./scripts/sweep_selfplay.sh --stop
EOF
