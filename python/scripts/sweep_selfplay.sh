#!/usr/bin/env bash
#
# sweep_selfplay.sh — launch 8 parallel self-play variants from one base model.
#
# Every variant resumes from the SAME checkpoint (runs/macro-selfplay2/best_model
# by default) and trains for the same number of *additional* timesteps, so the
# only thing that differs between runs is the knob under test. Each variant gets
# its own run dir, league, TensorBoard log and best_model.zip; the base run dir
# is never written to.
#
# Two things are fixed across every variant, by design:
#
#   * NO reward shaping (--no-reward-shaping). The powered-cities bonus is a
#     proxy for winning, not winning; the terminal reward is the whole signal.
#   * Minimal bot contact. The base model already wins ~75% vs normal bots, so
#     bots are no longer a training target — the league keeps at most a 5%
#     heuristic weight as an anchor against drifting into a degenerate
#     equilibrium, and one variant removes even that. Bots remain only as the
#     *eval* yardstick (eval never feeds back into training).
#
# The knobs explored span four axes: exploration (--ent-coef), optimisation
# (--learning-rate / --lr-final), terminal reward (--terminal-reward), and the
# self-play opponent distribution (--league-mix / --league-past-k /
# --snapshot-every / --no-league).
# The only thing that CANNOT be varied here is net width: --net-width is ignored
# on --resume-from, since the architecture comes from the checkpoint. Every other
# PPO hyperparameter (clip_range, gamma, gae_lambda, n_steps, batch_size,
# n_epochs, vf_coef, target_kl) has a flag if you want to add a variant.
#
# Sized for a 28-core machine: 8 variants x THREADS=3 = 24 cores, leaving
# headroom for the eval passes and the OS.
#
# Usage:
#   ./scripts/sweep_selfplay.sh              # launch all 8 in the background
#   ./scripts/sweep_selfplay.sh 2 5          # launch only variants 2 and 5
#   ./scripts/sweep_selfplay.sh --list       # show the variant table, launch nothing
#   ./scripts/sweep_selfplay.sh --status     # per-variant progress / best eval
#   ./scripts/sweep_selfplay.sh --compare    # head-to-head: each variant vs the base model
#   ./scripts/sweep_selfplay.sh --stop       # stop every running variant
#
# Env overrides (all optional):
#   TOTAL_TIMESTEPS=200000000  additional timesteps per variant
#   BASE_MODEL=runs/macro-selfplay2/best_model   checkpoint every variant starts from
#   SWEEP_DIR=runs/sweep1      root for the per-variant run dirs
#   NUM_ENVS=8                 parallel envs per variant (keep equal across variants)
#   THREADS=3                  torch/OMP threads per variant (8 x 3 = 24 of 28 cores)
#   LEAGUE_SEED=120            past snapshots copied from the base run's league
#                              (0 = start each league empty)
#   NICE=10  STAGGER=15  DRY_RUN=1  RESUME=1
#
set -euo pipefail

cd "$(dirname "$0")/.."          # python/

PY=${PY:-.venv/bin/python}
BASE_MODEL=${BASE_MODEL:-runs/macro-selfplay2/best_model}
SWEEP_DIR=${SWEEP_DIR:-runs/sweep1}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-200000000}
NUM_ENVS=${NUM_ENVS:-8}
THREADS=${THREADS:-3}
LEAGUE_SEED=${LEAGUE_SEED:-120}
NICE=${NICE:-10}
STAGGER=${STAGGER:-15}
DRY_RUN=${DRY_RUN:-0}
RESUME=${RESUME:-0}

# Shared across all variants — held constant so the comparison is clean.
# A variant's own flags come after these on the command line, so repeating a
# flag there (e.g. --league-mix) overrides the value set here.
COMMON=(
    --num-players 4
    --num-envs "$NUM_ENVS"
    --no-reward-shaping         # terminal reward only, for every variant
    --league-mix 0.45,0.50,0.05 # 5% heuristic anchor; the rest is self-play
    --save-freq 250000          # ~2M timesteps per checkpoint at 8 envs
    --eval-freq 50000           # ~400k timesteps per eval pass
    --eval-episodes 40          # 20 (the default) is too noisy to rank variants
)

# name|seed|hypothesis|extra flags
VARIANTS=(
"v1-control|101|Reference point: unshaped self-play at the current hyperparameters.|"
"v2-entropy|102|Plateau is entropy collapse — re-open exploration.|--ent-coef 0.10"
"v3-placement|103|With shaping gone the terminal reward is the only signal; rank-shaped gives gradient between losing seats instead of one bit per game.|--terminal-reward placement"
"v4-no-bots|104|Pure self-play: does even the 5% heuristic anchor hold the policy back?|--league-mix 0.5,0.5,0.0"
"v5-lr-decay|105|A 1B-step policy is past the point where 3e-4 refines anything; decay the step size to settle it.|--learning-rate 1e-4 --lr-final 0"
"v6-deep-league|106|Long-memory league: mostly-historical opponents, rarely refreshed, so the policy stops chasing its own latest quirks.|--league-past-k 24 --league-mix 0.20,0.75,0.05 --snapshot-every 250000"
"v7-fast-latest|107|Opposite opponent regime from v6: fast-churning latest-snapshot-only self-play, no bots at all.|--no-league --bot-mix 0.0 --snapshot-every 25000"
"v8-combo|108|Swing arm: the two changes most likely to help, together, to see whether they compose.|--ent-coef 0.10 --terminal-reward placement"
# Extras — append to the array (they become variants 9, 10, ...) if there are
# cores to spare. The 8 above were picked to cover one axis each.
# "v9-entropy-high|109|Harder push on exploration if 0.10 is not enough to re-expand the policy.|--ent-coef 0.20"
# "v10-target-kl|110|Cap destructive updates instead of shrinking every step.|--target-kl 0.02 --n-epochs 8"
)

variant_field() { echo "${VARIANTS[$1]}" | cut -d'|' -f"$2"; }

list_variants() {
    printf '%-20s %-6s %s\n' NAME SEED FLAGS
    for i in "${!VARIANTS[@]}"; do
        printf '%-20s %-6s %s\n' \
            "$(( i + 1 )). $(variant_field "$i" 1)" \
            "$(variant_field "$i" 2)" \
            "$(variant_field "$i" 4)"
    done
}

status() {
    for i in "${!VARIANTS[@]}"; do
        local name dir pid state ckpt best
        name=$(variant_field "$i" 1); dir="$SWEEP_DIR/$name"
        [[ -d $dir ]] || continue
        state="not running"
        if [[ -f $dir/train.pid ]]; then
            pid=$(cat "$dir/train.pid")
            kill -0 "$pid" 2>/dev/null && state="running (pid $pid)" || state="exited"
        fi
        ckpt=$(ls -t "$dir"/ckpt_*.zip 2>/dev/null | head -1 || true)
        ckpt=${ckpt:+$(basename "$ckpt")}
        best=$( [[ -f $dir/best_mean_reward.json ]] && cat "$dir/best_mean_reward.json" || echo '-' )
        printf '%-20s %-22s ckpt=%-28s best=%s\n' "$name" "$state" "${ckpt:--}" "$best"
    done
}

# Head-to-head vs the common ancestor. eval/mean_reward measures play against
# *normal bots*, which the base model already beats ~75% of the time — it will
# saturate long before the interesting differences appear. This is the yardstick
# that actually tracks "got better at beating strong opponents": each variant's
# best_model takes seat 0 against three copies of the base model. The base-vs-
# base row gives seat 0's own baseline, so read every variant against that, not
# against 25%.
compare() {
    local games=${COMPARE_GAMES:-200} seed=${COMPARE_SEED:-12345}
    [[ -f ${BASE_MODEL}.zip ]] || { echo "base model ${BASE_MODEL}.zip not found" >&2; exit 1; }
    echo "=== baseline: base model in all four seats ($games games) ==="
    "$PY" scripts/evaluate_lineup.py --games "$games" --seed "$seed" --quiet \
        --player "$BASE_MODEL" --player "$BASE_MODEL" \
        --player "$BASE_MODEL" --player "$BASE_MODEL"
    for i in "${!VARIANTS[@]}"; do
        local name model
        name=$(variant_field "$i" 1); model="$SWEEP_DIR/$name/best_model"
        [[ -f ${model}.zip ]] || continue
        echo
        echo "=== $name (seat 0) vs 3x base ($games games) ==="
        "$PY" scripts/evaluate_lineup.py --games "$games" --seed "$seed" --quiet \
            --player "$model" --player "$BASE_MODEL" \
            --player "$BASE_MODEL" --player "$BASE_MODEL"
    done
}

stop_all() {
    for i in "${!VARIANTS[@]}"; do
        local name pidfile pid
        name=$(variant_field "$i" 1); pidfile="$SWEEP_DIR/$name/train.pid"
        [[ -f $pidfile ]] || continue
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
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
esac

# Which variants to launch (1-based indices; default all).
if (( $# )); then
    SELECTED=("$@")
else
    SELECTED=($(seq 1 ${#VARIANTS[@]}))
fi

[[ -x $PY ]]                || { echo "no interpreter at $PY (run 'make develop' first)" >&2; exit 1; }
[[ -f ${BASE_MODEL}.zip ]]  || { echo "base model ${BASE_MODEL}.zip not found" >&2; exit 1; }

# Timesteps are cumulative in SB3 when resuming: learn() adds the checkpoint's
# num_timesteps to --total-timesteps. Read the base count so a RESUME=1 restart
# can ask for only the *remaining* steps and still stop at the same target.
BASE_STEPS=$("$PY" - "$BASE_MODEL" <<'EOF'
import json, sys, zipfile
with zipfile.ZipFile(sys.argv[1] + ".zip") as z:
    print(json.loads(z.read("data"))["num_timesteps"])
EOF
)
TARGET_STEPS=$(( BASE_STEPS + TOTAL_TIMESTEPS ))
BASE_LEAGUE=$(dirname "$BASE_MODEL")/league

echo "base model     : $BASE_MODEL (at $BASE_STEPS timesteps)"
echo "target         : +$TOTAL_TIMESTEPS -> $TARGET_STEPS timesteps per variant"
echo "sweep dir      : $SWEEP_DIR"
echo "launching      : ${SELECTED[*]}"
echo

mkdir -p "$SWEEP_DIR"

for n in "${SELECTED[@]}"; do
    i=$(( n - 1 ))
    (( i >= 0 && i < ${#VARIANTS[@]} )) || { echo "no variant $n" >&2; exit 1; }

    name=$(variant_field "$i" 1)
    seed=$(variant_field "$i" 2)
    why=$(variant_field "$i" 3)
    read -r -a extra <<< "$(variant_field "$i" 4)"
    dir="$SWEEP_DIR/$name"

    # Already running? Never start a second writer on the same run dir.
    if [[ -f $dir/train.pid ]] && kill -0 "$(cat "$dir/train.pid")" 2>/dev/null; then
        echo "skip $name: already running (pid $(cat "$dir/train.pid"))"
        continue
    fi

    resume_from="$BASE_MODEL"
    steps="$TOTAL_TIMESTEPS"
    newest=$(ls -t "$dir"/ckpt_*.zip 2>/dev/null | head -1 || true)
    if [[ -n $newest ]]; then
        if [[ $RESUME != 1 ]]; then
            echo "skip $name: $dir already has checkpoints (set RESUME=1 to continue it)"
            continue
        fi
        resume_from=${newest%.zip}
        done_steps=$(basename "$newest" | sed -E 's/ckpt_([0-9]+)_steps\.zip/\1/')
        steps=$(( TARGET_STEPS - done_steps ))
        (( steps > 0 )) || { echo "skip $name: already at $done_steps >= $TARGET_STEPS"; continue; }
        echo "resuming $name from $(basename "$newest") (+$steps remaining)"
    fi

    mkdir -p "$dir"

    # Seed a fresh league with a spread of the base run's snapshots, so
    # --league-past-k has real history to sample from at step 0 instead of
    # four copies of the model it just resumed from.
    if [[ $LEAGUE_SEED -gt 0 && ! -d $dir/league && -d $BASE_LEAGUE ]]; then
        mkdir -p "$dir/league"
        mapfile -t snaps < <(ls "$BASE_LEAGUE"/snap_*.bin 2>/dev/null | sort -V)
        if (( ${#snaps[@]} )); then
            stride=$(( (${#snaps[@]} + LEAGUE_SEED - 1) / LEAGUE_SEED ))
            (( stride > 0 )) || stride=1
            for ((s = 0; s < ${#snaps[@]}; s += stride)); do
                cp "${snaps[$s]}" "$dir/league/"
            done
            echo "  seeded league with $(ls "$dir/league" | wc -l) snapshots"
        fi
    fi

    cmd=("$PY" scripts/train_selfplay.py
         "${COMMON[@]}"
         --run-dir "$dir"
         --resume-from "$resume_from"
         --total-timesteps "$steps"
         --seed "$seed"
         "${extra[@]}")

    {
        echo "# $name — $why"
        echo "# launched: $(date -Is)"
        echo "# base: $BASE_MODEL @ $BASE_STEPS  target: $TARGET_STEPS"
        printf '%q ' "${cmd[@]}"; echo
    } > "$dir/variant.txt"

    if [[ $DRY_RUN == 1 ]]; then
        echo "[dry-run] $name"; sed -n '4p' "$dir/variant.txt"; continue
    fi

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
  tail -f $SWEEP_DIR/v1-control/train.log
  $PY -m tensorboard.main --logdir $SWEEP_DIR      # rollout/entropy_loss, eval/mean_reward
  $PY scripts/run_report.py $SWEEP_DIR/v2-entropy
Rank the variants (eval-vs-bots saturates; this is the real yardstick):
  ./scripts/sweep_selfplay.sh --compare
Stop everything:
  ./scripts/sweep_selfplay.sh --stop
EOF
