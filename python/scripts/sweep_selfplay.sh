#!/usr/bin/env bash
#
# sweep_selfplay.sh — launch 8 parallel self-play variants from one behavior clone.
#
# WAVE 3 (2026-07-26) — FULL RESET, INCLUDING THE PRIORS.
#
# Two resets happened at once and it matters that they are separate.
#
#   1. The ARTIFACTS are dead. The macro action space was rebuilt: build is a
#      count ladder, buy is a bitmask of which plants to fuel, powering is
#      auto-resolved, every phase is one decision per turn, policy format is
#      PGRLPOL6. Every prior checkpoint and clone is invalid.
#
#   2. The FINDINGS are dead too. Waves 1 and 2 ran against broken rules, a
#      broken environment and a mis-mapped action space (see
#      RL-TRAINING-JOURNAL.md). Their conclusions -- "big batch beats base 39% vs
#      26%", "constant 1e-4 helps", "entropy 0.10 collapses a policy", "a
#      historical league regresses" -- are measurements of a system that no
#      longer exists. They are NOT carried forward, and in particular they are no
#      longer baked into COMMON, where they silently applied to every arm.
#
# So this sweep does not refine a known-good recipe; there is no known-good
# recipe. w1 is the trainer's own defaults plus the clone, and every other arm
# moves exactly ONE knob off it. The point is to find out which knobs matter in
# THIS environment, from scratch.
#
# What legitimately informs the design, because it was measured against the
# current code in this repo rather than inherited:
#
#   * --init-policy-from loads only the three policy layers, so the VALUE HEAD
#     starts random and the first updates ride on meaningless advantages. That is
#     a fact about the code, not a training result, and it is the single most
#     likely way to wreck a good clone. w2 and w7 attack it from two directions.
#   * A clone of the `hard` bot saturates an eval against `normal` immediately,
#     and that metric selects best_model -- so eval runs against `hard`.
#   * The rebuilt menus are wider than the old ones (buy 3.84 live options,
#     build 2.99) and three of five decision types now teach a varied imitation
#     label (nominate, buy, build). Exploration therefore has real room, which is
#     why entropy is probed UPWARD here: the old prior that forbade it is void.
#
# Deliberately untested this wave, for lack of slots rather than lack of
# interest: opponent-pool composition, clip range, target-KL, net width. Add them
# once the first-order knobs are pinned down.
#
# PREREQUISITE — build the clone first (this script does not create it):
#
#   python/.venv/bin/python -m alphazero.pretrain \
#       --games 400 --epochs 20 --net-width 128 \
#       --run-dir alphazero/runs/clone_w3 \
#       --export alphazero/runs/clone_w3/clone.bin
#
# `--net-width` there MUST match NET_WIDTH here; the loader refuses a mismatch.
#
# WORTH DOING FIRST — score the clone, since it is the reference point every arm
# is measured against and a weak one invalidates the whole sweep:
#
#   cargo run -p powergrid-evolve --release -- \
#       --policy-file alphazero/runs/clone_w3/clone.bin --greedy \
#       --opponent-toml assets/bots/default.toml
#
# `--greedy` matters: a behavior clone plays materially stronger with argmax than
# with sampling (sampling picks the teacher's non-top move a fraction of the
# time), and with the macro action space greedy no longer risks the stalls the
# primitive encoding had. If the clone is far under the hard bot's own seat-0
# share, fix the clone before burning 8 x 50M timesteps on finetuning it.
#
# Sized for a 28-core machine: 8 variants x THREADS=3 = 24 cores, leaving
# headroom for the eval passes and the OS.
#
# Re-running is idempotent and self-healing: launching is the same command as
# resuming. For each selected variant the script inspects its run dir and picks
# up where it left off — resume from the furthest-along readable checkpoint (or
# start from the clone if there is none) — but ONLY after confirming no trainer
# is already running for that dir. The running-check verifies the recorded PID is
# still a train_selfplay.py process for THIS run dir, so a stale pidfile (e.g. a
# PID recycled across a reboot) can't block a resume or, worse, let two trainers
# write the same dir. So the intended operational loop is simply: run it, and if
# the box reboots or a variant crashes, run it again.
#
# Usage:
#   ./scripts/sweep_selfplay.sh              # launch/resume all 8 in the background
#   ./scripts/sweep_selfplay.sh 2 5          # launch/resume only variants 2 and 5
#   ./scripts/sweep_selfplay.sh --list       # show the variant table, launch nothing
#   ./scripts/sweep_selfplay.sh --status     # per-variant progress / best eval
#   ./scripts/sweep_selfplay.sh --compare    # ABSOLUTE: each variant vs 3x hard bots
#   ./scripts/sweep_selfplay.sh --h2h        # RELATIVE: each variant vs 3x the baseline arm
#   ./scripts/sweep_selfplay.sh --stop       # stop every running variant
#
# Env overrides (all optional):
#   TOTAL_TIMESTEPS=50000000   timesteps per variant
#   CLONE=../alphazero/runs/clone_w3/clone.bin   PGRLPOL6 warm-start weights
#   SWEEP_DIR=runs/sweep3      root for the per-variant run dirs
#   NET_WIDTH=128              must match the clone's width
#   NUM_ENVS=8                 parallel envs per variant (keep equal across variants)
#   THREADS=3                  torch/OMP threads per variant (8 x 3 = 24 of 28 cores)
#   COMPARE_GAMES=200  COMPARE_SEED=12345
#   COMPARE_DETERMINISTIC=1    rank with argmax instead of sampling. Worth a
#                              second look for clone-derived policies, which play
#                              stronger greedily; training is stochastic, so the
#                              sampled numbers remain the primary ranking.
#   NICE=10  STAGGER=15  DRY_RUN=1
#
set -euo pipefail

cd "$(dirname "$0")/.."          # python/

PY=${PY:-.venv/bin/python}
CLONE=${CLONE:-../alphazero/runs/clone_w3/clone.bin}
SWEEP_DIR=${SWEEP_DIR:-runs/sweep3}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-50000000}
NET_WIDTH=${NET_WIDTH:-128}
NUM_ENVS=${NUM_ENVS:-8}
THREADS=${THREADS:-3}
NICE=${NICE:-10}
STAGGER=${STAGGER:-15}
DRY_RUN=${DRY_RUN:-0}

# The reference arm every other variant is a single-knob deviation from, and the
# opponent for --h2h.
BASELINE=w1-base

# Shared across all variants — held constant so the comparison is clean.
# A variant's own flags come after these on the command line, so repeating a
# flag there (e.g. --ent-coef) overrides the value set here.
COMMON=(
    --num-players 4
    --num-envs "$NUM_ENVS"
    --net-width "$NET_WIDTH"
    --no-reward-shaping         # terminal reward is the objective; shaping is a
                                # proxy for it (w7 turns it back on, annealed)
    --eval-difficulty hard      # NOT normal: a clone saturates normal instantly,
                                # and this metric selects best_model
    --save-freq 250000          # ~2M timesteps per checkpoint at 8 envs
    --eval-freq 50000           # ~400k timesteps per eval pass
    --eval-episodes 200         # 20 (the trainer default) is too noisy to rank
)
# NOTE: no PPO hyperparameters here. Wave 2's "winning recipe" (n-steps 1024,
# batch 1024, lr 1e-4, league-mix 0.45/0.50/0.05) used to live in this block and
# so applied to every arm unexamined. It was measured on a broken environment, so
# it is now an arm to be tested (w3, w4), not an assumption to build on.


# name|seed|init|hypothesis|extra flags
#
# `init` is "clone" (warm-start from $CLONE) or "scratch" (no warm start).
VARIANTS=(
"w1-base|201|clone|BASELINE: the clone plus the trainer's own defaults (lr 3e-4, n-steps/batch 512, n-epochs 4, vf-coef 0.5, ent-coef 0.03, league 0.5/0.3/0.2, win/loss reward). Deliberately unopinionated -- with every prior wave invalidated there is no recipe to inherit, so the reference is the default and each arm is one knob off it.|"
"w2-vf-warmup|202|clone|The warm start leaves the CRITIC RANDOM, so early advantages are noise that can wreck a good clone. Let the critic catch up fast (vf-coef 0.5 -> 1.0) while the policy moves less (n-epochs 4 -> 2). This is the structural risk, not an inherited hunch.|--vf-coef 1.0 --n-epochs 2"
"w3-low-lr|203|clone|Same risk, blunter answer: make the policy nearly immovable (lr 3e-4 -> 1e-4) until the critic means something. Wave 2 claimed a constant small lr helps, on a broken environment; this tests the claim rather than assuming it.|--learning-rate 1e-4"
"w4-big-batch|204|clone|Lower-variance gradients via a bigger rollout and minibatch (512 -> 1024). Wave 2's single largest reported effect (39% vs 26%) and therefore the one most worth re-measuring honestly, as an arm rather than as a COMMON default.|--n-steps 1024 --batch-size 1024"
"w5-ent-up|205|clone|Entropy UP (0.03 -> 0.10). The old prior said this collapses a policy; that measurement is void, and the rebuilt menus are wider (buy 3.84 live options, build 2.99) so there is more to explore than there was. If the clone needs to discover the new buy subsets and build counts, this is how.|--ent-coef 0.10"
"w6-placement|206|clone|Rank-shaped terminal reward instead of +1/-1. A 4-player game gives one bit per episode under win/loss; placement gives gradient between the losing seats, which should matter most while the critic is still forming. Untested on a working environment.|--terminal-reward placement"
"w7-shaped-start|207|clone|The other answer to the random critic: give it a DENSE early signal. Powered-cities shaping, annealed to zero over the first fifth so it bootstraps the value head without steering the final policy. A critic-bootstrap argument, not the policy-teaching one that failed before.|--reward-shaping --shaping-mode absolute --anneal-shaping-steps 10000000"
"w8-scratch|208|scratch|CONTROL: w1 with NO warm start. Answers 'is the behavior clone load-bearing?' and would catch a silently broken --init-policy-from. With three of five phases now teaching a varied imitation label the clone should be markedly better than scratch; if it is not, suspect the clone before the recipe.|"
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
# against 0. Unlike wave 2's vs-normal eval this does not saturate: a behavior
# clone starts at roughly the bots' own level, so there is headroom in both
# directions and the metric stays informative for the whole run.
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

# RELATIVE ranking: each variant in seat 0 against three copies of the baseline
# arm. Once several arms clear the bot bar, beating strong opposition is the
# finer signal — this is wave 2's --compare, retargeted now that the common
# ancestor is a clone rather than a checkpoint.
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
if [[ ! -f $CLONE ]]; then
    cat >&2 <<EOF
behavior clone not found at: $CLONE

Build it first (see the header of this script):
  python/.venv/bin/python -m alphazero.pretrain \\
      --games 400 --epochs 20 --net-width $NET_WIDTH \\
      --run-dir alphazero/runs/clone_w3 \\
      --export alphazero/runs/clone_w3/clone.bin

Run that from the repo root. Or set CLONE=/path/to/clone.bin.
Only variant 8 (w8-scratch) can run without it.
EOF
    exit 1
fi

echo "clone          : $CLONE (net-width $NET_WIDTH)"
echo "target         : $TOTAL_TIMESTEPS timesteps per variant"
echo "sweep dir      : $SWEEP_DIR"
echo "launching      : ${SELECTED[*]}"
echo
echo "NOTE: no league seeding. Snapshots from earlier waves are from dead layout"
echo "      epochs (PGRLPOL1..5) and are rejected on load, by design."
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

    # Auto-resume: inspect the run dir and continue from where it left off.
    # SB3 counts timesteps cumulatively across resumes, so we ask each launch for
    # only the steps still needed to reach TOTAL_TIMESTEPS — re-running never
    # overshoots. A resumed checkpoint already carries trained weights, so the
    # warm start applies only to the very first launch.
    start_args=()
    steps="$TOTAL_TIMESTEPS"
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
    elif [[ $init == clone ]]; then
        start_args=(--init-policy-from "$CLONE")
        echo "starting $name from the clone (+$TOTAL_TIMESTEPS)"
    else
        echo "starting $name from scratch, no warm start (+$TOTAL_TIMESTEPS)"
    fi

    mkdir -p "$dir"

    cmd=("$PY" scripts/train_selfplay.py
         "${COMMON[@]}"
         --run-dir "$dir"
         --total-timesteps "$steps"
         --seed "$seed"
         "${start_args[@]}"
         "${extra[@]}")

    {
        echo "# $name — $why"
        echo "# launched: $(date -Is)"
        echo "# init: $init  clone: $CLONE  target: $TOTAL_TIMESTEPS"
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
  tail -f $SWEEP_DIR/$BASELINE/train.log
  $PY -m tensorboard.main --logdir $SWEEP_DIR      # rollout/entropy_loss, eval/mean_reward
  $PY scripts/run_report.py $SWEEP_DIR/$BASELINE
Rank the variants:
  ./scripts/sweep_selfplay.sh --compare    # absolute: vs 3x hard bots (the project bar)
  ./scripts/sweep_selfplay.sh --h2h        # relative: vs 3x $BASELINE
Stop everything:
  ./scripts/sweep_selfplay.sh --stop
EOF
