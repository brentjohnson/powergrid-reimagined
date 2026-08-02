#!/usr/bin/env bash
#
# sweep_selfplay.sh — wave 6: population play — the arms train against each other.
#
# WAVE 6 (2026-08-01).
#
# Wave 5 finished at ~250M steps per arm. Both rankings, 200 games, seed 12345
# (seat-0 par ~25%):
#
#   ABSOLUTE (--compare, vs 3x hard):        RELATIVE (--h2h, vs 3x y4 best):
#     z3-batch-decay   66.5%   <- champion     z3-batch-decay   30.0%
#     z2-redecay       62.5%                   y3-batch         29.0%
#     y3-batch         62.0%                   z5-sp-decay      29.0%
#     z5-sp-decay      58.5%                   z2-redecay       27.5%
#     z1-cont          56.0%                   z1-cont          24.5%
#     z7-sp-all        55.0%                   z7-sp-all        24.0%
#     z4-sp-lean       54.5%                   z6-past-heavy    23.0%
#     z6-past-heavy    52.0%                   z4-sp-lean       20.5%
#
#   Direct matches sealed it: z3 best (seat 0) took 29.0% vs 3x y3 best and
#   30.5% vs 3x z2 best. z3-batch-decay's best_model (246M steps) is the
#   wave-6 fork point AND the frozen --h2h yardstick.
#
# What wave 5 settled — this closes the recipe era:
#
#   * DECAY IS REPEATABLE: z2-redecay (62.5%) beat z1-cont (56.0%) from the
#     same weights. You don't converge once; re-arming the anneal from a new
#     start works every time.
#   * BATCH + DECAY STACK: z3 (66.5%) beat both single-knob parents. The two
#     winning axes are at least partly independent.
#   * SELF-PLAY-LEAN LOST AGAIN, third wave in a row: every 0.10-bots arm
#     trailed its 0.20-bots twin (z4<z1, z5<z2, z7<z3, z6 worst). The default
#     0.5/0.3/0.2 league mix stays. The way to get more from self-play is not
#     a bigger self share — it is BETTER OPPONENTS, which is what this wave
#     buys.
#   * y3-batch's constant-lr lineage keeps compounding (49% -> 62% vs hard
#     without ever decaying) — kept alive as the second lineage.
#   * The frozen-y4 h2h barely separates arms any more (its own mirror
#     measured 29.5% for seat order alone); expect the same of the z3
#     yardstick late in this wave — read trends, not single numbers.
#
# WHAT'S NEW: cross-arm population play (trainer `--league-peers`). Every
# arm's league PAST share now also samples the OTHER arms' snapshot dirs,
# rescanned at every snapshot refresh (~every 100k steps), so the eight
# trainers evolve against each other instead of only their own history.
# Rationale: the echo chamber is self-play's failure mode, the bot share is
# already maxed out as a teacher (arms are at 55-66% vs hard), and eight
# diverging recipes are the strongest, most varied opponents available.
# Mechanics: snapshots are written atomically (tmp+rename) and validated on
# read, so an arm can never crash another by being mid-write; a peer dir
# that doesn't exist yet (staggered launch) is just an empty pool slice.
#
# The population forfeits per-knob attribution BY DESIGN (every arm's
# opponents now depend on every other arm), with one exception kept on
# purpose: p4-solo runs the exact p1-main recipe with NO peers — the p1 vs
# p4 gap is the cleanest read on whether population play itself pays.
#
# Arm roles (all lr 1e-4 unless noted, all forked from z3 except y3):
#   y3-batch   second lineage (never-decayed, never-forked), joins the pool
#   p1-main    population default: batch 1024, const lr, mix 0.40/0.40/0.20
#   p2-finish  champion recipe on the population: p1 + lr decay -> 0
#   p3-small-finish  batch 512 + decay: a stylistically distinct member
#   p4-solo    p1-main WITHOUT peers — the population-effect control
#   p5-peer-heavy    mix 0.30/0.50/0.20, past_k 12: deepest population diet
#   p6-gentle  lr 5e-5 const: slow-moving, stable target for the others
#   p7-exploit mix 0.10/0.70/0.20, past_k 12: barely plays itself, hunts the
#              population — the closest cheap approximation of an AlphaStar
#              exploiter without win-rate-targeted matchmaking
#
# EVAL CAVEAT, now acute: --eval-difficulty hard selects best_model and the
# champion is at 66.5% — approaching the band where selection tracks noise.
# Rank arms with --h2h (vs 3x the frozen z3 best) once --compare crowds
# above ~70%, raise COMPARE_GAMES for the final ranking, and treat wave 6's
# winner as a candidate for a HARDER eval opponent in wave 7 (e.g. an
# embedded-policy eval) rather than pushing vs-hard further.
#
# Sized for a 28-core machine: 8 variants x THREADS=3 = 24 cores, leaving
# headroom for the eval passes and the OS.
#
# Re-running is idempotent and self-healing: launching is the same command as
# resuming. For each selected variant the script inspects its run dir and
# picks up where it left off — resume from the furthest-along readable
# checkpoint; if there is none, a fork arm starts from $FORK_FROM and a
# continuation arm refuses loudly (silently restarting a finished run from
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
# y3 sits at 250M and the fork point at 246M, so the default 350M target buys
# every arm +100-104M — the same increment waves 4 and 5 ran.
#
# Resume-lr note: MaskablePPO.load is passed custom_objects built from THIS
# launch's flags, so a fork's lr schedule is its own — z3's decayed-to-zero
# lr does not leak into the forks.
#
# League note for forks: snapshots live in each run dir's own league/ subdir.
# A fork's league starts empty, but the trainer pushes a snapshot of the
# just-loaded weights at training start, so a nonzero PAST share never sees an
# empty pool — the early "past" opponent is simply the fork point itself (and,
# this wave, whatever the peers have published).
#
# The retired arms (waves 3-5: w1-w8, y1/y2/y4/y5/y6, z1/z2/z4-z7) stay in
# $SWEEP_DIR as inert history. z3-batch-decay's dir MUST stay — it is the
# fork point and the frozen h2h yardstick (y4-lr-decay was the wave-5 one).
# --stop only knows the variants in the table below; if a retired arm is
# somehow still running, kill its $SWEEP_DIR/<name>/train.pid by hand.
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
#   TOTAL_TIMESTEPS=350000000  CUMULATIVE timesteps per variant (see above)
#   FORK_FROM=runs/sweep3/z3-batch-decay/best_model
#                              pinned fork point for the p-arms (stem, no .zip)
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
FORK_FROM=${FORK_FROM:-$SWEEP_DIR/z3-batch-decay/best_model}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-350000000}
NET_WIDTH=${NET_WIDTH:-128}
NUM_ENVS=${NUM_ENVS:-8}
THREADS=${THREADS:-3}
NICE=${NICE:-10}
STAGGER=${STAGGER:-15}
DRY_RUN=${DRY_RUN:-0}

# The frozen --h2h opponent: the retired wave-5 champion's best_model — the
# exact weights every p-arm forked from. z3-batch-decay no longer trains (its
# lr annealed to 0), so this yardstick never moves.
BASELINE=z3-batch-decay

# Shared across all variants — held constant so the comparison is clean.
# A variant's own flags come after these on the command line, so repeating a
# flag there overrides the value set here.
COMMON=(
    --num-players 4
    --num-envs "$NUM_ENVS"
    --net-width "$NET_WIDTH"
    --no-reward-shaping         # terminal reward is the objective (wave 3's
                                # shaped arm was neutral-to-negative)
    --eval-difficulty hard      # selects best_model; nearing saturation — see
                                # the eval caveat in the header
    --save-freq 250000          # ~2M timesteps per checkpoint at 8 envs
    --eval-freq 50000           # ~400k timesteps per eval pass
    --eval-episodes 200         # 20 (the trainer default) is too noisy to rank
)

# name|seed|init|pop|hypothesis|extra flags
#
# `init` is "resume" (a continuing arm; must already have checkpoints) or
# "fork" (first launch resumes from $FORK_FROM into a fresh dir).
#
# `pop` is "peers" (the launch loop appends --league-peers with every OTHER
# variant's league dir) or "solo" (no peers — the population-effect control).
#
# League mix order is LATEST,PAST,BOTS; the trainer default is 0.5,0.3,0.2.
# With peers, the PAST share samples own history AND the peers' snapshots.
VARIANTS=(
"y3-batch|303|resume|peers|The second lineage: constant lr 1e-4 + batch 1024, never decayed, never forked, still compounding (49% -> 62% vs hard across wave 5). Joins the population — its independent history is the most stylistically distinct opponent the pool has.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"p1-main|501|fork|peers|The population default: champion weights, batch 1024, constant lr, mix 0.40/0.40/0.20 with the PAST share fed by all seven siblings. Reads directly against p4-solo: same recipe, population on vs off.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"p2-finish|502|fork|peers|The champion recipe re-armed on the population: p1-main + lr decay 1e-4 -> 0. Decay has finished on top twice in a row (y4, z3); if opponent quality is what gates it, this is the wave's likeliest champion.|--learning-rate 1e-4 --lr-final 0 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"p3-small-finish|503|fork|peers|Batch 512 + decay: the strongest small-batch recipe (z2's) inside the population. Partly a performance arm, partly a diversity donor — different update noise makes a stylistically different opponent for everyone else.|--learning-rate 1e-4 --lr-final 0 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"p4-solo|504|fork|solo|CONTROL: bit-for-bit p1-main's recipe with NO peers — its league is its own history, exactly like wave 5. The p1 vs p4 gap (and each one's h2h vs the frozen z3) is the clean measurement of whether population play itself pays.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"p5-peer-heavy|505|fork|peers|Deepest population diet: mix 0.30/0.50/0.20, past_k 12. Half of every rollout is against the pool. If p1 beats it, a taste of population suffices; if it beats p1, opponent variety is the binding resource.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.30,0.50,0.20 --league-past-k 12"
"p6-gentle|506|fork|peers|Slow-moving member: constant lr 5e-5, otherwise p1. A near-stationary strong player stabilizes the population (others can learn against it without it shifting underfoot) and tests whether the champion keeps improving at half the step size.|--learning-rate 5e-5 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"p7-exploit|507|fork|peers|The exploiter: mix 0.10/0.70/0.20, past_k 12 — barely plays itself, lives on the population. Uniform sampling stands in for AlphaStar's win-rate matchmaking; its job is to find and punish the pool's shared habits, which feeds harder opponents back to everyone.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.10,0.70,0.20 --league-past-k 12"
)


variant_field() { echo "${VARIANTS[$1]}" | cut -d'|' -f"$2"; }

# Comma-separated league dirs of every variant EXCEPT $1 (by name) — the
# --league-peers value for a "peers" arm. Dirs may not exist yet; the trainer
# tolerates that (empty pool slice until the peer launches).
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

list_variants() {
    printf '%-20s %-6s %-8s %-6s %s\n' NAME SEED INIT POP FLAGS
    for i in "${!VARIANTS[@]}"; do
        printf '%-20s %-6s %-8s %-6s %s\n' \
            "$(( i + 1 )). $(variant_field "$i" 1)" \
            "$(variant_field "$i" 2)" \
            "$(variant_field "$i" 3)" \
            "$(variant_field "$i" 4)" \
            "$(variant_field "$i" 6)"
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
# wave-5 champion — the exact weights the p-arms forked from, and not a moving
# target since z3-batch-decay is retired. This is the primary ranking once the
# vs-hard eval saturates. Above-par here == genuinely past the fork point.
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

echo "fork point     : $FORK_FROM (p-arms' first launch only)"
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
            echo "(set FORK_FROM, or sync the wave-5 champion's run dir first)" >&2
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
  ./scripts/sweep_selfplay.sh --status
  tail -f $SWEEP_DIR/p2-finish/train.log
  $PY -m tensorboard.main --logdir $SWEEP_DIR      # league/peer_size, eval/mean_reward
  $PY scripts/run_report.py $SWEEP_DIR/p2-finish
Rank the variants:
  ./scripts/sweep_selfplay.sh --compare    # absolute: vs 3x hard bots (watch for saturation)
  ./scripts/sweep_selfplay.sh --h2h        # relative: vs 3x the frozen $BASELINE best (primary once saturated)
Stop everything:
  ./scripts/sweep_selfplay.sh --stop
EOF
