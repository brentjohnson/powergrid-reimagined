#!/usr/bin/env bash
#
# sweep_selfplay.sh — launch 8 parallel self-play variants from one behavior clone.
#
# WAVE 3 (2026-07-26) — FULL RESET. The macro action space was rebuilt (build is a
# count ladder, buy is per-plant none/1-set/2-sets, powering is auto-resolved,
# PGRLPOL1 -> PGRLPOL5),
# so *every* prior checkpoint is invalid, including the wave-2 winners. There is
# no common ancestor to resume from any more. The sweep now starts from a
# BEHAVIOR CLONE of the champion heuristic and asks a different question:
#
#     Given a policy that already plays like the `hard` bot, what finetuning
#     recipe improves it instead of destroying it?
#
# That is a genuinely different problem from wave 2, which was refining a
# converged 1B-step policy. Some wave-2 findings carry over; several do not.
#
#   CARRIES OVER:
#     * Low-variance updates win. Big batch (1024) beat base 39% vs 26%,
#       constant small lr (1e-4) 34%, the two combined 34.5%. w1 uses both, and
#       every arm inherits them so the sweep tests deviations from a known-good
#       recipe rather than from SB3 defaults.
#     * Entropy UP collapses a good policy (0.10 -> ~9-13%). No arm raises it
#       above 0.01; w4 probes only a small amount, for a specific new reason.
#     * A mostly-historical league regresses (15%). The opponent-distribution
#       axis is probed once (w6), toward MORE bot contact, not less.
#
#   DOES NOT CARRY OVER:
#     * "Bots are no longer a training target." False now: the clone starts at
#       roughly heuristic strength, so the hard bot is exactly the bar.
#     * Eval vs *normal* bots. A clone of the hard bot saturates that instantly
#       and best_model selection degenerates into tracking eval noise — every
#       variant now evaluates against `hard` (--eval-difficulty).
#     * "Shaping is dead." Worth one arm (w7) for a NEW reason: the warm start
#       loads only the policy, so the critic is random and shaping gives it a
#       dense early signal. That is a critic-bootstrap argument, not the
#       policy-teaching argument that failed before.
#
# THE MAIN RISK this sweep is designed around: --init-policy-from loads the three
# policy layers and leaves the VALUE HEAD randomly initialised. The first updates
# therefore ride on meaningless advantages and can wreck a good clone before the
# critic catches up. w1/w2/w3 are three different answers to that (gentle updates
# / fast critic / near-frozen policy); w8 is the control that says whether the
# warm start was load-bearing at all.
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
#   CLONE=../alphazero/runs/clone_w3/clone.bin   PGRLPOL5 warm-start weights
#   SWEEP_DIR=runs/sweep3      root for the per-variant run dirs
#   NET_WIDTH=128              must match the clone's width
#   NUM_ENVS=8                 parallel envs per variant (keep equal across variants)
#   THREADS=3                  torch/OMP threads per variant (8 x 3 = 24 of 28 cores)
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
BASELINE=w1-clone-anchor

# Shared across all variants — held constant so the comparison is clean.
# A variant's own flags come after these on the command line, so repeating a
# flag there (e.g. --ent-coef) overrides the value set here.
COMMON=(
    --num-players 4
    --num-envs "$NUM_ENVS"
    --net-width "$NET_WIDTH"
    --no-reward-shaping         # terminal reward only (w7 turns it back on)
    --league-mix 0.45,0.50,0.05 # 5% heuristic anchor; the rest is self-play
    --eval-difficulty hard      # NOT normal: a clone saturates normal instantly
    --save-freq 250000          # ~2M timesteps per checkpoint at 8 envs
    --eval-freq 50000           # ~400k timesteps per eval pass
    --eval-episodes 200         # 20 (the default) is too noisy to rank variants
    # Wave-2's winning low-variance recipe, inherited by every arm.
    --n-steps 1024
    --batch-size 1024
    --learning-rate 1e-4
)

# name|seed|init|hypothesis|extra flags
#
# `init` is "clone" (warm-start from $CLONE) or "scratch" (no warm start).
VARIANTS=(
"w1-clone-anchor|201|clone|BASELINE. The clone plus everything wave 2 proved: big batch (1024), constant small lr (1e-4), no shaping, terminal reward only. Every other arm is this with ONE knob moved, so any difference is attributable.|--terminal-reward placement"
"w2-vf-warmup|202|clone|The warm start leaves the CRITIC RANDOM, so early advantages are noise that can wreck a good clone. Let the critic catch up fast (vf-coef 0.5 -> 1.0) while the policy barely moves (n-epochs 4 -> 2).|--terminal-reward placement --vf-coef 1.0 --n-epochs 2"
"w3-tiny-lr|203|clone|Same risk, blunter answer: make the policy nearly immovable (lr 1e-4 -> 3e-5) until the critic means something. If w3 >> w1 the clone is being damaged at 1e-4; if w3 ~= w1 it is not, and w1's larger steps are free.|--terminal-reward placement --learning-rate 3e-5"
"w4-explore|204|clone|A clone of a deterministic teacher is peaked, so it may NEVER sample the ladder rungs the action-space rebuild added (per-plant 1/2-set presses, BUILD_n). Entropy 0.03 -> 0.01: enough to try them, far below the 0.10 that collapsed wave 2.|--terminal-reward placement --ent-coef 0.01"
"w5-winloss|205|clone|Isolates the reward shape. Wave 2 could not separate placement from win/loss on a converged policy (29% vs 26% base, inside noise). With a fresh critic the denser rank signal should matter more — this is the control that proves or kills it.|--terminal-reward winloss"
"w6-bot-anchor|206|clone|A clone self-playing is mostly playing the teacher, so early self-play may add little. Weight the league toward the heuristic (0.45/0.50/0.05 -> 0.30/0.30/0.40) to keep the learner honest against the bar it is scored on.|--terminal-reward placement --league-mix 0.30,0.30,0.40"
"w7-shaped-start|207|clone|The one arm that reintroduces shaping, for a NEW reason: the critic is random at step 0 and the powered-cities bonus is a dense signal to bootstrap it. Annealed to 0 over the first fifth so it cannot distort the converged policy.|--terminal-reward placement --reward-shaping --shaping-mode absolute --anneal-shaping-steps 10000000"
"w8-scratch|208|scratch|CONTROL: identical to w1 but with NO warm start. Answers 'is the behavior clone load-bearing?' and would catch a silently broken --init-policy-from. History says from-scratch PPO fails on this game; if w8 keeps up, suspect the clone.|--terminal-reward placement"
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
    echo "=== baseline: 4x hard bots ($games games) — seat 0's structural share ==="
    "$PY" scripts/evaluate_lineup.py --games "$games" --seed "$seed" --quiet \
        --player hard --player hard --player hard --player hard
    for i in "${!VARIANTS[@]}"; do
        local name model
        name=$(variant_field "$i" 1); model="$SWEEP_DIR/$name/best_model"
        [[ -f ${model}.zip ]] || continue
        echo
        echo "=== $name (seat 0) vs 3x hard ($games games) ==="
        "$PY" scripts/evaluate_lineup.py --games "$games" --seed "$seed" --quiet \
            --player "$model" --player hard --player hard --player hard
    done
}

# RELATIVE ranking: each variant in seat 0 against three copies of the baseline
# arm. Once several arms clear the bot bar, beating strong opposition is the
# finer signal — this is wave 2's --compare, retargeted now that the common
# ancestor is a clone rather than a checkpoint.
h2h() {
    local games=${COMPARE_GAMES:-200} seed=${COMPARE_SEED:-12345}
    local base="$SWEEP_DIR/$BASELINE/best_model"
    [[ -f ${base}.zip ]] || { echo "baseline $BASELINE has no best_model yet" >&2; exit 1; }
    echo "=== self-baseline: 4x $BASELINE ($games games) ==="
    "$PY" scripts/evaluate_lineup.py --games "$games" --seed "$seed" --quiet \
        --player "$base" --player "$base" --player "$base" --player "$base"
    for i in "${!VARIANTS[@]}"; do
        local name model
        name=$(variant_field "$i" 1); model="$SWEEP_DIR/$name/best_model"
        [[ $name == "$BASELINE" ]] && continue
        [[ -f ${model}.zip ]] || continue
        echo
        echo "=== $name (seat 0) vs 3x $BASELINE ($games games) ==="
        "$PY" scripts/evaluate_lineup.py --games "$games" --seed "$seed" --quiet \
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
echo "NOTE: no league seeding. Snapshots from earlier waves are PGRLPOL1 files"
echo "      from a dead layout epoch and are rejected on load, by design."
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
