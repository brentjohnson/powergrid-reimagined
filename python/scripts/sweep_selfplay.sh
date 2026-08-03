#!/usr/bin/env bash
#
# sweep_selfplay.sh — wave 7: the population converges — decay everything that
# won, and select checkpoints against the champion instead of the hard bot.
#
# WAVE 7 (2026-08-03).
#
# Wave 6 finished at 350M steps per arm. Both rankings, 400 games, seed 12345:
#
#   ABSOLUTE (--compare, vs 3x hard;         RELATIVE (--h2h, vs 3x z3 best;
#             4x-hard seat-0 par 21.0%):               mirror par 24.5%):
#     p2-finish        68.5%   <- champion     p3-small-finish  29.0%
#     y3-batch         66.2%                   p2-finish        28.7%
#     p3-small-finish  65.5%                   y3-batch         28.2%
#     p1-main          65.0%                   p1-main          26.5%
#     p6-gentle        64.8%                   p4-solo          25.0%
#     p7-exploit       64.8%                   p5-peer-heavy    24.5%
#     p5-peer-heavy    63.2%                   p7-exploit       23.5%
#     p4-solo          60.5%                   p6-gentle        20.8%
#
#   Direct matches p2 vs p3 were a dead heat (p2 in seat 0 took 26.8% vs 3x
#   p3; p3 took 26.2% vs 3x p2; both ~par 24.5%), so the vs-hard edge and the
#   recipe's track record decided it: p2-finish is the wave-6 champion. Its
#   best_model (337.2M steps) is the wave-7 fork point, the frozen --h2h
#   yardstick, AND the new --eval-opponent (see below). It is also the
#   embedded Expert as of 2026-08-03 (74% on the noisy 50-game native
#   harness, vs 68% for the z3 export it replaced).
#
# What wave 6 settled:
#
#   * POPULATION PLAY PAYS: p1-main (65.0% / 26.5%) beat p4-solo (60.5% /
#     25.0%) on the identical recipe, and p4's best_model never moved past
#     the fork point (captured at 246.8M ≈ the 246M fork). Every arm keeps
#     peers from here on; the solo control has answered its question.
#   * DECAY WINS A THIRD STRAIGHT WAVE (y4, z3, now p2). Re-arming the
#     anneal from each new start remains the single most reliable move.
#   * SMALL-BATCH IS BACK IN THE RACE: p3 (batch 512 + decay) tied p2 on
#     h2h and the direct matches. Keep both batch sizes in play.
#   * A MODERATE PEER DIET IS THE SWEET SPOT: the deeper diets did nothing
#     for the arm itself (p5 0.30/0.50 at par, p7 0.10/0.70 below par), and
#     p6-gentle (const 5e-5) was the h2h WORST at 20.8% — a slow-moving arm
#     gets farmed by an evolving pool. Don't field a deliberately slow arm
#     again; 0.40/0.40/0.20 stays the default.
#   * y3-batch's constant-lr lineage still compounds (62.0% -> 66.2% vs
#     hard) without ever decaying — kept alive, and this wave its lineage
#     finally gets a decay fork (q4) to see what it has been leaving on the
#     table.
#
# WHAT'S NEW: checkpoint selection vs the frozen champion. The vs-hard eval
# has saturated (the field crowds 63-68% and best_model selection tracks
# noise — exactly the failure the wave-6 header predicted). The trainer grew
# an `--eval-opponent POLICY.bin` flag (2026-08-03): eval episodes now seat
# the learner against three copies of the frozen wave-6 champion, so
# eval/mean_reward — and therefore best_model — measures progress PAST the
# fork point instead of noise against a beaten yardstick. Reading the new
# numbers: win_rate = (mean_reward + 1) / 2, and the champion mirror par is
# ~24.5%, so mean_reward ≈ -0.51 is "as good as the fork point" and anything
# above that is genuine progress. best= in --status is NEGATIVE now — that
# is expected, not a regression. The eval opponent file is auto-exported
# from $FORK_FROM on first launch (to $EVAL_OPPONENT) and then never moves.
# --compare (vs hard) remains the absolute yardstick for wave-end reporting.
#
# Continued-arm migration: y3-batch's stored best bar (vs hard) is not
# comparable with the new metric, so its FIRST wave-7 launch sets the old
# best_model aside as best_model.wave6-vs-hard.zip and deletes
# best_mean_reward.json (marker file .wave7-eval-opponent makes this
# one-time). Its best_model.zip is then re-earned under the new metric.
#
# Arm roles (all lr 1e-4, all peers, all forked from p2 except y3/q4):
#   y3-batch   the second lineage: const lr + batch 1024, never decayed
#   q1-main    population default, const lr, batch 1024 — the decay control
#   q2-finish  champion recipe re-armed: q1 + lr decay -> 0 (likeliest champion)
#   q3-small-finish  batch 512 + decay — wave 6's co-winner recipe
#   q4-y3-finish     forked from y3's best (297.2M) + decay -> 0: the
#              never-annealed lineage finally converges; also injects y3's
#              distinct style into the candidate set, not just the pool
#   q5-clip-finish   q2 + clip-range 0.1: does a tighter trust region stack
#              with decay the way batch did? (clip has never been varied)
#   q6-anchor-lean   mix 0.40/0.50/0.10, past_k 12: trades HALF THE BOT
#              ANCHOR for peers. Every earlier sp-lean cut bots for SELF
#              (echo chamber, lost 3 waves); cutting bots for the population
#              is the untested version of the same question
#   q7-exploit mix 0.10/0.70/0.20, past_k 12: the pool hardener, kept from
#              wave 6 — mid-field itself but the only adversarial-pressure
#              member, and cheap to keep
#
# EVAL CAVEAT, one wave early: once the field crowds ~30%+ vs the frozen
# champion the same saturation logic applies again — rank with --h2h trends,
# raise COMPARE_GAMES for the final ranking (wave 6 used 400), and expect to
# re-freeze wave 7's winner as wave 8's eval opponent.
#
# Sized for a 28-core machine: 8 variants x THREADS=3 = 24 cores, leaving
# headroom for the eval passes and the OS.
#
# Re-running is idempotent and self-healing: launching is the same command as
# resuming. For each selected variant the script inspects its run dir and
# picks up where it left off — resume from the furthest-along readable
# checkpoint; if there is none, a fork arm starts from its fork source and a
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
# y3 sits at 350M and the p2 fork point at 337.2M, so the default 450M target
# buys most arms +100-113M — the same increment waves 4-6 ran. The exception
# is q4-y3-finish, forking from y3's best at 297.2M: it gets +153M and simply
# runs longer than its siblings.
#
# Resume-lr note: MaskablePPO.load is passed custom_objects built from THIS
# launch's flags, so a fork's lr schedule is its own — p2's decayed-to-zero
# lr does not leak into the forks.
#
# League note for forks: snapshots live in each run dir's own league/ subdir.
# A fork's league starts empty, but the trainer pushes a snapshot of the
# just-loaded weights at training start, so a nonzero PAST share never sees an
# empty pool — the early "past" opponent is simply the fork point itself (and
# whatever the peers have published).
#
# The retired arms (waves 3-6: w1-w8, y1/y2/y4/y5/y6, z1-z7, p1/p3-p7) stay
# in $SWEEP_DIR as inert history. p2-finish's dir MUST stay — it is the fork
# point, the frozen h2h yardstick, and the --eval-opponent source
# (z3-batch-decay was the wave-6 yardstick; its dir is now history only).
# y3-batch's dir must obviously stay too (it both continues and is q4's fork
# source). --stop only knows the variants in the table below; if a retired
# arm is somehow still running, kill its $SWEEP_DIR/<name>/train.pid by hand.
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
#   TOTAL_TIMESTEPS=450000000  CUMULATIVE timesteps per variant (see above)
#   FORK_FROM=runs/sweep3/p2-finish/best_model
#                              pinned fork point for the q-arms (stem, no .zip)
#   EVAL_OPPONENT=runs/sweep3/wave7-eval-opponent.bin
#                              frozen PGRLPOL6 eval opponent; auto-exported
#                              from $FORK_FROM if missing
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
FORK_FROM=${FORK_FROM:-$SWEEP_DIR/p2-finish/best_model}
EVAL_OPPONENT=${EVAL_OPPONENT:-$SWEEP_DIR/wave7-eval-opponent.bin}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-450000000}
NET_WIDTH=${NET_WIDTH:-128}
NUM_ENVS=${NUM_ENVS:-8}
THREADS=${THREADS:-3}
NICE=${NICE:-10}
STAGGER=${STAGGER:-15}
DRY_RUN=${DRY_RUN:-0}

# The frozen --h2h opponent: the retired wave-6 champion's best_model — the
# exact weights the q-arms forked from and the eval opponent was exported
# from. p2-finish no longer trains (its lr annealed to 0), so this yardstick
# never moves.
BASELINE=p2-finish

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
                                # selects best_model vs the frozen wave-6
                                # champion; vs-hard saturated at 63-68%.
                                # Par is ~24.5% == mean_reward ~ -0.51, so
                                # best= in --status goes NEGATIVE by design.
    --save-freq 250000          # ~2M timesteps per checkpoint at 8 envs
    --eval-freq 50000           # ~400k timesteps per eval pass
    --eval-episodes 200         # 20 (the trainer default) is too noisy to rank
)

# name|seed|init|pop|hypothesis|extra flags
#
# `init` is "resume" (a continuing arm; must already have checkpoints),
# "fork" (first launch resumes from $FORK_FROM into a fresh dir), or
# "fork=<stem>" (first launch resumes from that checkpoint stem instead).
#
# `pop` is "peers" (the launch loop appends --league-peers with every OTHER
# variant's league dir) or "solo" (no peers — wave 6's control; answered).
#
# League mix order is LATEST,PAST,BOTS; the trainer default is 0.5,0.3,0.2.
# With peers, the PAST share samples own history AND the peers' snapshots.
VARIANTS=(
"y3-batch|303|resume|peers|The second lineage: constant lr 1e-4 + batch 1024, never decayed, never forked, compounding for a third straight wave (62.0% -> 66.2% vs hard across wave 6). Joins the wave-7 pool unchanged; q4 separately decays a copy of it.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"q1-main|601|fork|peers|The population default and this wave's decay control: champion weights, batch 1024, constant lr, mix 0.40/0.40/0.20. Every decay arm reads against this — same food, same fork, no anneal.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"q2-finish|602|fork|peers|The champion recipe re-armed a fourth time: q1-main + lr decay 1e-4 -> 0. Decay has finished on top three waves running (y4, z3, p2); until it loses, it is the presumptive champion.|--learning-rate 1e-4 --lr-final 0 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"q3-small-finish|603|fork|peers|Batch 512 + decay: wave 6's co-winner (29.0% h2h, dead heat with p2 in direct matches). Half the batch means different update noise — a real contender that is also the pool's most distinct opponent.|--learning-rate 1e-4 --lr-final 0 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"q4-y3-finish|604|fork=$SWEEP_DIR/y3-batch/best_model|peers|The never-decayed lineage finally gets its anneal: fork y3's best (297.2M) + lr decay -> 0. Re-arming the anneal has worked from every start so far; if it works from this independent history too, decay is unconditional — and the pool gains a champion-strength player that never saw p2's weights.|--learning-rate 1e-4 --lr-final 0 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"q5-clip-finish|605|fork|peers|q2-finish + clip-range 0.1. Batch stacked with decay in wave 5; clip is the other update-size knob and has never been varied. If a tighter trust region stacks too, wave 8 inherits a three-knob recipe; if it drags, clip 0.2 is vindicated.|--learning-rate 1e-4 --lr-final 0 --clip-range 0.1 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"q6-anchor-lean|606|fork|peers|Mix 0.40/0.50/0.10, past_k 12: trades half the BOT anchor for peers. Every earlier sp-lean arm cut bots to feed SELF and lost (echo chamber); cutting bots to feed the population is the untested variant — at 65%+ vs hard the heuristic teacher may finally be dead weight.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.50,0.10 --league-past-k 12"
"q7-exploit|607|fork|peers|The exploiter, kept from wave 6: mix 0.10/0.70/0.20, past_k 12 — barely plays itself, lives on the population. Mid-field as a candidate but the pool's only adversarial-pressure member; its job is hardening everyone else's opponents, not winning.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.10,0.70,0.20 --league-past-k 12"
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
    # NOTE: best= is eval/mean_reward vs 3x the frozen wave-6 champion
    # (win_rate = (best+1)/2; par ~ -0.51). Negative values are normal.
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
# against 0. Saturated for ranking since wave 6 (the field crowds 63-68%);
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
# wave-6 champion — the exact weights the q-arms forked from, and not a moving
# target since p2-finish is retired. This is the primary ranking (the vs-hard
# eval is saturated). Above-par here == genuinely past the fork point; the
# mirror measured par at 24.5% for seat order alone in wave 6.
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

# The frozen eval opponent every arm's best_model selection runs against.
# Exported once from the fork point and never touched again — the golden
# sidecar goes next to it, NOT into assets/ (export_policy.py's default).
if [[ ! -f $EVAL_OPPONENT && $DRY_RUN != 1 ]]; then
    if [[ ! -f ${FORK_FROM}.zip ]]; then
        echo "cannot export eval opponent: no checkpoint at ${FORK_FROM}.zip" >&2
        echo "(set FORK_FROM/EVAL_OPPONENT, or sync the wave-6 champion's run dir)" >&2
        exit 1
    fi
    echo "exporting frozen eval opponent: ${FORK_FROM}.zip -> $EVAL_OPPONENT"
    "$PY" scripts/export_policy.py --model "$FORK_FROM" \
        --out "$EVAL_OPPONENT" --golden "${EVAL_OPPONENT%.bin}.golden.json"
fi

echo "fork point     : $FORK_FROM (q-arms' first launch only)"
echo "eval opponent  : $EVAL_OPPONENT (frozen; selects best_model)"
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

    # One-time wave-7 migration for continued arms: the stored best bar was
    # earned vs hard bots and is not comparable with the new frozen-champion
    # eval. Set the old best aside, drop the bar, and let the new metric
    # re-earn best_model.zip. The marker file makes re-runs a no-op.
    if [[ $init == resume && $DRY_RUN != 1 && -f $dir/best_mean_reward.json \
          && ! -f $dir/.wave7-eval-opponent ]]; then
        [[ -f $dir/best_model.zip && ! -f $dir/best_model.wave6-vs-hard.zip ]] \
            && cp "$dir/best_model.zip" "$dir/best_model.wave6-vs-hard.zip"
        rm "$dir/best_mean_reward.json"
        touch "$dir/.wave7-eval-opponent"
        echo "migrated $name to the wave-7 eval metric (old best kept as best_model.wave6-vs-hard.zip)"
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
  ./scripts/sweep_selfplay.sh --status     # best= is vs the frozen champion: win = (best+1)/2, par ~ -0.51
  tail -f $SWEEP_DIR/q2-finish/train.log
  $PY -m tensorboard.main --logdir $SWEEP_DIR      # league/peer_size, eval/mean_reward
  $PY scripts/run_report.py $SWEEP_DIR/q2-finish
Rank the variants:
  ./scripts/sweep_selfplay.sh --compare    # absolute: vs 3x hard bots (reporting; saturated for ranking)
  ./scripts/sweep_selfplay.sh --h2h        # relative: vs 3x the frozen $BASELINE best (primary ranking)
Stop everything:
  ./scripts/sweep_selfplay.sh --stop
EOF
