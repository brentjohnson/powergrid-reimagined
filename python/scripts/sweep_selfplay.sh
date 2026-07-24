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
# WAVE 2 (2026-07-24). Wave 1 (the original v1..v8) found that this converged
# ~1B-step base policy only improves under *gentler* optimisation and is wrecked
# by anything that adds churn: --ent-coef 0.10 (old v2/v8) collapsed it to ~9-13%
# vs base, and a mostly-historical league (old v6) regressed it to ~15%. The only
# two levers that didn't drop below base head-to-head were lr-decay (v5, kept) and
# placement reward (v3, kept) — and note the *vs-bots* eval was actively
# misleading (it ranked v5 a star and hid v6's collapse), so rank by --compare
# only. Wave 2 keeps v3 + v5 and replaces the other six with a study of *which
# form of gentle/stable update* helps (lr magnitude, decay-vs-constant, KL cap,
# clip size, gradient variance), plus the two disproven directions flipped to
# their opposites (entropy DOWN). The opponent-distribution axis is dropped: it
# gave the only significant regression and nothing positive.
#
# The only thing that CANNOT be varied here is net width: --net-width is ignored
# on --resume-from, since the architecture comes from the checkpoint. Every other
# PPO hyperparameter (clip_range, gamma, gae_lambda, n_steps, batch_size,
# n_epochs, vf_coef, target_kl) has a flag if you want to add a variant.
#
# Sized for a 28-core machine: 8 variants x THREADS=3 = 24 cores, leaving
# headroom for the eval passes and the OS.
#
# Re-running is idempotent and self-healing: launching is the same command as
# resuming. For each selected variant the script inspects its run dir and picks
# up where it left off — resume from the furthest-along readable checkpoint (or
# start fresh from the base if there is none) — but ONLY after confirming no
# trainer is already running for that dir. The running-check verifies the
# recorded PID is still a train_selfplay.py process for THIS run dir, so a stale
# pidfile (e.g. a PID recycled across a reboot) can't block a resume or, worse,
# let two trainers write the same dir. So the intended operational loop is
# simply: run it, and if the box reboots or a variant crashes, run it again.
#
# Usage:
#   ./scripts/sweep_selfplay.sh              # launch/resume all 8 in the background
#   ./scripts/sweep_selfplay.sh 2 5          # launch/resume only variants 2 and 5
#   ./scripts/sweep_selfplay.sh --list       # show the variant table, launch nothing
#   ./scripts/sweep_selfplay.sh --status     # per-variant progress / best eval
#   ./scripts/sweep_selfplay.sh --compare    # head-to-head: each variant vs the base model
#   ./scripts/sweep_selfplay.sh --stop       # stop every running variant
#
# Env overrides (all optional):
#   TOTAL_TIMESTEPS=200000000  target additional timesteps beyond the base model
#   BASE_MODEL=runs/macro-selfplay2/best_model   checkpoint every variant starts from
#   SWEEP_DIR=runs/sweep1      root for the per-variant run dirs
#   NUM_ENVS=8                 parallel envs per variant (keep equal across variants)
#   THREADS=3                  torch/OMP threads per variant (8 x 3 = 24 of 28 cores)
#   LEAGUE_SEED=120            past snapshots copied from the base run's league
#                              (0 = start each league empty)
#   NICE=10  STAGGER=15  DRY_RUN=1
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
    --eval-episodes 200          # 20 (the default) is too noisy to rank variants
)

# name|seed|hypothesis|extra flags
#
# v3-placement and v5-lr-decay are UNCHANGED from wave 1 (same name -> same run
# dir -> auto-resume from their own furthest checkpoint). The other six are wave-2
# variants with NEW names, so they get fresh run dirs and start from the base.
VARIANTS=(
"w1-combo|109|The two surviving levers together: gentle lr-decay + placement reward. Old v8 combined the two DISPROVEN levers (ent-up + placement) and collapsed; this is the good-levers combo.|--learning-rate 1e-4 --lr-final 0 --terminal-reward placement"
"w2-lr-const|110|v5 decays lr to 0 (freezes on a checkpoint). Test whether a CONSTANT small lr keeps genuinely refining instead of just settling.|--learning-rate 1e-4"
"v3-placement|103|With shaping gone the terminal reward is the only signal; rank-shaped gives gradient between losing seats instead of one bit per game. KEPT from wave 1.|--terminal-reward placement"
"w4-target-kl|111|Gentle updates by CAPPING KL per update instead of shrinking every step; extra epochs under the cap. Different mechanism from lr for the same 'settle the converged policy' goal.|--target-kl 0.02 --n-epochs 8"
"v5-lr-decay|105|A 1B-step policy is past the point where 3e-4 refines anything; decay the step size to settle it. KEPT from wave 1.|--learning-rate 1e-4 --lr-final 0"
"w6-tight-clip|112|Same gentle-update theme via a smaller PPO trust region (clip 0.2 -> 0.1): every update takes a smaller step.|--clip-range 0.1"
"w7-ent-low|113|Entropy UP collapsed the policy (old v2/v8). Test the opposite: sharpen exploitation on the converged policy (default 0.03 -> 0.005).|--ent-coef 0.005"
"w8-big-batch|114|Lower-variance gradients as a different route to stable updates: double the rollout and minibatch (512 -> 1024).|--n-steps 1024 --batch-size 1024"
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

# Echo "<checkpoint-path-without-.zip> <num_timesteps>" for the furthest-along
# *readable* checkpoint in a run dir, or nothing if there is none. Candidates
# are tried highest-step-first; a checkpoint truncated by a kill mid-write fails
# the zip read and is skipped in favour of the previous one, so an interrupted
# run always resumes from a clean point. The step count comes from inside the
# zip (num_timesteps), which the filename mirrors.
latest_checkpoint() {
    local dir=$1 zip stem steps
    while IFS= read -r zip; do
        [[ -n $zip ]] || continue
        steps=$("$PY" - "$zip" <<'EOF' 2>/dev/null
import json, sys, zipfile
try:
    with zipfile.ZipFile(sys.argv[1]) as z:
        print(json.loads(z.read("data"))["num_timesteps"])
except Exception:
    sys.exit(1)
EOF
        ) || continue
        stem=${zip%.zip}
        echo "$stem $steps"
        return 0
    done < <(ls "$dir"/ckpt_*_steps.zip 2>/dev/null \
                | sed -E 's/.*ckpt_([0-9]+)_steps\.zip/\1 &/' \
                | sort -k1,1nr | cut -d' ' -f2-)
    return 0
}

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

    # Already running? Never start a second writer on the same run dir. The
    # check confirms the pidfile's PID is genuinely this variant's trainer, so
    # a stale/recycled PID neither blocks a needed resume nor risks a duplicate.
    live=$(running_pid "$dir")
    if [[ -n $live ]]; then
        echo "skip $name: already running (pid $live)"
        continue
    fi

    # Auto-resume: inspect the run dir and continue from where it left off.
    # SB3 counts timesteps cumulatively across resumes, so we ask each launch
    # for only the steps still needed to reach TARGET_STEPS from the resumed
    # checkpoint — re-running never overshoots the target.
    resume_from="$BASE_MODEL"
    steps="$TOTAL_TIMESTEPS"
    ckpt_stem=""; done_steps=""
    read -r ckpt_stem done_steps < <(latest_checkpoint "$dir") || true
    if [[ -n $ckpt_stem ]]; then
        resume_from="$ckpt_stem"
        steps=$(( TARGET_STEPS - done_steps ))
        if (( steps <= 0 )); then
            echo "skip $name: already at $done_steps >= target $TARGET_STEPS timesteps"
            continue
        fi
        echo "resuming $name from $(basename "$ckpt_stem").zip @ $done_steps (+$steps to $TARGET_STEPS)"
    else
        echo "starting $name fresh from base (+$TOTAL_TIMESTEPS)"
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
  tail -f $SWEEP_DIR/w1-combo/train.log
  $PY -m tensorboard.main --logdir $SWEEP_DIR      # rollout/entropy_loss, eval/mean_reward
  $PY scripts/run_report.py $SWEEP_DIR/w1-combo
Rank the variants (eval-vs-bots saturates; this is the real yardstick):
  ./scripts/sweep_selfplay.sh --compare
Stop everything:
  ./scripts/sweep_selfplay.sh --stop
EOF
