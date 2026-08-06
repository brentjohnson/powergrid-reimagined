#!/usr/bin/env bash
#
# sweep_selfplay.sh — wave 8: cross-lineage decay forks are the champion
# recipe — field two of them, and probe the one update knob never varied.
#
# WAVE 8 (2026-08-05).
#
# Wave 7 finished at 450M steps per arm (q4 at ~449M from its 297M fork).
# Both rankings, 400 games, seed 12345:
#
#   ABSOLUTE (--compare, vs 3x hard;         RELATIVE (--h2h, vs 3x p2 best;
#             4x-hard seat-0 par 21.0%):               mirror par 22.8%):
#     q7-exploit       72.5%                  q3-small-finish  32.5%
#     q4-y3-finish     72.0%   <- champion    q5-clip-finish   32.2%
#     q2-finish        69.2%                  y3-batch         29.5%
#     q5-clip-finish   68.8%                  q4-y3-finish     29.0%
#     q3-small-finish  68.5%                  q1-main          25.5%
#     y3-batch         68.0%                  q7-exploit       25.2%
#     q1-main          65.5%                  q2-finish        24.0%
#     q6-anchor-lean   58.2%                  q6-anchor-lean   22.8%
#
#   The two rankings disagreed at the top, so the tiebreak ran deep: direct
#   matches (400 games) had q4 beating q3 from BOTH sides (26.8% seat-0 vs
#   3x q3 against q3's 22.8% vs 3x q4) and crushing q5 (31.2% vs 20.8%),
#   and a fresh-seed 800-game h2h (seed 54321, mirror par 26.9%) had q4 on
#   top again (29.5% vs q3 28.6%, q5 26.6%). Add the best vs-hard score and
#   the best training eval (-0.21, ~40% vs the frozen p2 — the whole field
#   sat at -0.28..-0.35) and q4-y3-finish is the wave-7 champion. Its
#   best_model (407.2M steps) is the wave-8 fork point, the frozen --h2h
#   yardstick, and the new --eval-opponent. It is also the embedded Expert
#   as of 2026-08-05 (native harness 39/50 = 78%, up from 74% for p2).
#
# What wave 7 settled:
#
#   * DECAY IS UNCONDITIONAL (4 waves: y4, z3, p2, q4) — and the biggest
#     win came from decaying an INDEPENDENT lineage: q4 re-armed y3's
#     never-annealed history and never saw p2's weights. Cross-lineage
#     decay forks are now the champion recipe; wave 8 fields two (r4 from
#     y3 again, r6 from q3's small-batch line).
#   * THE BOT ANCHOR IS LOAD-BEARING: q6 traded half of it for peers and
#     finished last on BOTH boards (58.2% vs hard — a 7pp faceplant). The
#     0.20 bot share stays; don't cut it again in any form.
#   * CLIP 0.1 IS NEUTRAL: q5 tied q2 everywhere it mattered and lost both
#     direct matches to q4. Clip 0.2 stays the default; knob closed.
#   * SINGLE 400-GAME H2H SAMPLES FLIP RANKS (q3/q5 topped the seed-12345
#     h2h, then lost the direct matches). Final rankings now always get the
#     full tiebreak: direct matches between leaders + a fresh-seed 800-game
#     h2h. Budget for it at wave end.
#   * THE FROZEN-CHAMPION EVAL WORKED: best_model selection tracked real
#     progress (q4's -0.21 stood out for hundreds of eval passes while the
#     one-shot h2h samples wobbled). Re-freeze on each wave's winner.
#   * y3-batch STILL COMPOUNDS (68.0% vs hard, 29.5% h2h — above par at
#     450M without ever decaying) and is now the proven donor lineage.
#     Kept alive unchanged; r4 decays a fresh copy of it.
#
# Arm roles (all lr 1e-4 unless noted, all peers, forks from q4 except
# where noted):
#   y3-batch   the donor lineage: const lr + batch 1024, never decayed —
#              its wave-7 fork (q4) just won the wave. Keeps compounding.
#   r1-main    population default, const lr, batch 1024 — the decay control
#   r2-finish  champion recipe re-armed a fifth time: r1 + lr decay -> 0
#   r3-small-finish  batch 512 + decay — the small-batch recipe stays in
#              play (q3 was the seed-12345 h2h leader and wave-6 co-winner)
#   r4-y3-finish     fork y3's CURRENT best (411.6M) + decay -> 0: re-run
#              the exact recipe that just won, from the donor's newer state
#   r5-sharp-finish  r2 + --ent-coef 0.015 (default 0.03): the entropy
#              bonus has never been varied. At a converged 72%-vs-hard
#              start, entropy ~0.25 nats may be leaving sharpness on the
#              table; half the bonus + decay-to-zero is the safest probe.
#              WATCH IT: if policy entropy dives toward ~0.1 nats early,
#              that is the old collapse signature — kill the arm.
#   r6-q3-finish     fork q3-small-finish's best (433.6M) + decay -> 0: the
#              second cross-lineage candidate — injects the co-leader's
#              distinct small-batch style into the candidate set
#   r7-exploit mix 0.10/0.70/0.20, past_k 12: the pool hardener, kept a
#              third wave. Below-par h2h as a candidate but top of the
#              vs-hard board (72.5%) — its adversarial diet generalizes;
#              its job is hardening everyone else's opponents.
#
# EVAL NOTE: best= in --status is eval/mean_reward vs 3x the frozen wave-7
# champion (win_rate = (best+1)/2). The q4 mirror par isn't measured yet;
# expect ~25% seat-0 share == mean_reward ~ -0.50, same as every frozen
# yardstick so far. Negative best= is normal. Once the field crowds ~30%+,
# the same saturation logic applies: rank with --h2h trends, run the full
# tiebreak at wave end, and re-freeze wave 8's winner for wave 9.
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
# The q4 fork point sits at 407.2M, y3 at 450M, and the cross-lineage fork
# sources at 411.6M (y3's best) and 433.6M (q3's best), so the 550M target
# buys the q4 forks +143M, y3 +100M, r4 +138M, and r6 +116M — the arms just
# run different lengths, as q4 itself did in wave 7 (it got +153M and won).
#
# Resume-lr note: MaskablePPO.load is passed custom_objects built from THIS
# launch's flags, so a fork's lr schedule is its own — a decayed-to-zero
# source lr does not leak into the forks.
#
# League note for forks: snapshots live in each run dir's own league/ subdir.
# A fork's league starts empty, but the trainer pushes a snapshot of the
# just-loaded weights at training start, so a nonzero PAST share never sees an
# empty pool — the early "past" opponent is simply the fork point itself (and
# whatever the peers have published).
#
# The retired arms (waves 3-7: w*, y1/y2/y4/y5/y6, z*, p*, q1/q2/q5/q6/q7)
# stay in $SWEEP_DIR as inert history. These dirs MUST stay:
#   q4-y3-finish     wave-8 fork point, frozen h2h yardstick, --eval-opponent
#                    source
#   y3-batch         continues training AND is r4's fork source
#   q3-small-finish  r6's fork source
# (p2-finish, the wave-7 yardstick, is now history only.) --stop only knows
# the variants in the table below; if a retired arm is somehow still running,
# kill its $SWEEP_DIR/<name>/train.pid by hand.
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
#   TOTAL_TIMESTEPS=550000000  CUMULATIVE timesteps per variant (see above)
#   FORK_FROM=runs/sweep3/q4-y3-finish/best_model
#                              pinned fork point for the r-arms (stem, no .zip)
#   EVAL_OPPONENT=runs/sweep3/wave8-eval-opponent.bin
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
FORK_FROM=${FORK_FROM:-$SWEEP_DIR/q4-y3-finish/best_model}
EVAL_OPPONENT=${EVAL_OPPONENT:-$SWEEP_DIR/wave8-eval-opponent.bin}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-550000000}
NET_WIDTH=${NET_WIDTH:-128}
NUM_ENVS=${NUM_ENVS:-8}
THREADS=${THREADS:-3}
NICE=${NICE:-10}
STAGGER=${STAGGER:-15}
DRY_RUN=${DRY_RUN:-0}

# The frozen --h2h opponent: the retired wave-7 champion's best_model — the
# exact weights the r-arms forked from and the eval opponent was exported
# from. q4-y3-finish no longer trains (its lr annealed to 0), so this
# yardstick never moves.
BASELINE=q4-y3-finish

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
                                # selects best_model vs the frozen wave-7
                                # champion. Par ~25% == mean_reward ~ -0.50,
                                # so best= in --status is NEGATIVE by design.
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
"y3-batch|303|resume|peers|The donor lineage: constant lr 1e-4 + batch 1024, never decayed, never forked, compounding for a fourth straight wave — and its wave-7 decay fork (q4) just won the wave. Joins the wave-8 pool unchanged; r4 separately decays a fresh copy of it.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"r1-main|701|fork|peers|The population default and this wave's decay control: champion weights, batch 1024, constant lr, mix 0.40/0.40/0.20. Every decay arm reads against this — same food, same fork, no anneal.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"r2-finish|702|fork|peers|The champion-line recipe re-armed a fifth time: r1-main + lr decay 1e-4 -> 0. Decay has finished on top of four straight waves (y4, z3, p2, q4); until it loses, every wave re-arms it from the new champion.|--learning-rate 1e-4 --lr-final 0 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"r3-small-finish|703|fork|peers|Batch 512 + decay: the small-batch recipe stays in play — q3 topped the seed-12345 h2h sample and was wave 6's co-winner. Half the batch means different update noise, and it doubles as the pool's most distinct default-lineage opponent.|--learning-rate 1e-4 --lr-final 0 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"r4-y3-finish|704|fork=$SWEEP_DIR/y3-batch/best_model|peers|Re-run the exact recipe that just won: fork y3's current best (411.6M) + lr decay -> 0. The donor has kept evolving since q4's fork, so this is a NEW start from the proven independent lineage — if it lands at the top again, cross-lineage decay is a repeatable champion factory, not a one-off.|--learning-rate 1e-4 --lr-final 0 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"r5-sharp-finish|705|fork|peers|r2-finish + ent-coef 0.015 (default 0.03, never varied). The field sits at entropy ~0.25 nats and 70%+ vs hard; halving the entropy bonus while the lr anneals to zero asks whether a converged policy is leaving sharpness on the table. Collapse guard: if entropy dives toward ~0.1 nats early, kill the arm — that is the old collapse signature.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.015 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"r6-q3-finish|706|fork=$SWEEP_DIR/q3-small-finish/best_model|peers|The second cross-lineage candidate: fork q3-small-finish's best (433.6M) + lr decay -> 0. q3's small-batch line fought q4 to the wire in wave 7; re-arming ITS history tests whether the cross-lineage recipe works from any strong donor, and injects the co-leader's style into the candidate set, not just the pool.|--learning-rate 1e-4 --lr-final 0 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"r7-exploit|707|fork|peers|The exploiter, kept a third wave: mix 0.10/0.70/0.20, past_k 12 — barely plays itself, lives on the population. Below-par h2h as a candidate but top of the wave-7 vs-hard board (72.5%); its job is hardening everyone else's opponents, not winning.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.10,0.70,0.20 --league-past-k 12"
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
    # NOTE: best= is eval/mean_reward vs 3x the frozen wave-7 champion
    # (win_rate = (best+1)/2; par ~ -0.50). Negative values are normal.
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
# against 0. Saturated for ranking since wave 6 (the field crowds 63-72%);
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
# wave-7 champion — the exact weights most r-arms forked from, and not a
# moving target since q4-y3-finish is retired. This is the primary ranking
# (the vs-hard eval is saturated). Above-par here == genuinely past the fork
# point. Wave-7 mirrors measured seat-0 par at 22.8% (seed 12345, 400g) and
# 26.9% (seed 54321, 800g) — call it ~25 +/- 2%; measure the q4 mirror before
# reading small edges. At wave end, run the FULL tiebreak: direct matches
# between the leaders + a fresh-seed 800-game h2h (wave 7's one-shot 400-game
# samples flipped ranks).
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
        echo "(set FORK_FROM/EVAL_OPPONENT, or sync the wave-7 champion's run dir)" >&2
        exit 1
    fi
    echo "exporting frozen eval opponent: ${FORK_FROM}.zip -> $EVAL_OPPONENT"
    "$PY" scripts/export_policy.py --model "$FORK_FROM" \
        --out "$EVAL_OPPONENT" --golden "${EVAL_OPPONENT%.bin}.golden.json"
fi

echo "fork point     : $FORK_FROM (r-arms' first launch only)"
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

    # One-time wave-8 migration for continued arms: the stored best bar was
    # earned vs the wave-7 eval opponent (p2) and is not comparable with the
    # new frozen champion (q4). Set the old best aside, drop the bar, and let
    # the new metric re-earn best_model.zip. The marker file makes re-runs a
    # no-op.
    if [[ $init == resume && $DRY_RUN != 1 && -f $dir/best_mean_reward.json \
          && ! -f $dir/.wave8-eval-opponent ]]; then
        [[ -f $dir/best_model.zip && ! -f $dir/best_model.wave7-vs-p2.zip ]] \
            && cp "$dir/best_model.zip" "$dir/best_model.wave7-vs-p2.zip"
        rm "$dir/best_mean_reward.json"
        touch "$dir/.wave8-eval-opponent"
        echo "migrated $name to the wave-8 eval metric (old best kept as best_model.wave7-vs-p2.zip)"
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
  tail -f $SWEEP_DIR/r2-finish/train.log
  $PY -m tensorboard.main --logdir $SWEEP_DIR      # league/peer_size, eval/mean_reward
  $PY scripts/run_report.py $SWEEP_DIR/r2-finish
Rank the variants:
  ./scripts/sweep_selfplay.sh --compare    # absolute: vs 3x hard bots (reporting; saturated for ranking)
  ./scripts/sweep_selfplay.sh --h2h        # relative: vs 3x the frozen $BASELINE best (primary ranking)
Stop everything:
  ./scripts/sweep_selfplay.sh --stop
EOF
