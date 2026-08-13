#!/usr/bin/env bash
#
# sweep_selfplay.sh — wave 11: the LINEAGE x ENTROPY 2x2. Every arm RESUMES its
# own 600-format checkpoints or FORKS (--resume-from, warm policy+value head)
# from a pinned 600-format checkpoint — the proven waves-4-8/10 structure.
#
# WAVE 11 (2026-08-12).
#
# Wave 10 (the first within-format wave, all 8 arms to a 300M cumulative budget)
# is settled. Every fork carried a WARM value head (--resume-from), so the wave
# measured recipes, not format-migration guards. What it measured:
#
#   Frozen-champion eval (best mean_reward vs 3x s3-gentle, par ~-0.50):
#     t3-y3-finish -0.20  t6-sharp -0.27  t5-gentle -0.30  t4-small -0.31
#     t2-finish -0.32  s4-y3 -0.33  t7-exploit -0.40  t1-main -0.42
#   Primary h2h (seat 0 vs 3x the frozen champion; combined 600 games over
#   seeds 12345+99999, mirror par ~24.5%):
#     t6-sharp ~32.0  t3-y3-finish ~31.5  t5-gentle ~29.0  t4-small ~28.5
#     t2-finish ~28.0  t1-main ~26.0  t7-exploit ~25.5  s4-y3 ~25.0 (par)
#   The boards disagreed at the top (eval liked t3 clearly; h2h had t6 ahead of
#   t3 by ~0.5pp = 1 game), so the full tiebreak decided it. DIRECT head-to-head
#   (seed 77777, 400 games each direction, 4-way par 25%):
#     t3-y3-finish @seat0 vs 3x t6 = 24.8%  BEAT  t6-sharp @seat0 vs 3x t3 = 22.5%
#   → t3 challenges to ~par and suppresses t6 below par; t6 challenges only to
#   22.5%. t3-y3-finish wins the direct match both ways. Combined with its
#   decisive eval (-0.20) and compare (80.5% vs hard) leads, t3-y3-finish is the
#   wave-10 champion and the new embedded Expert (86% native vs 3x hard, 43/50,
#   up from s3-gentle's 80%).
#
# What wave 10 SETTLED (knobs closed / opened):
#   * CROSS-LINEAGE DECAY is confirmed THE champion factory — fork the
#     never-annealed y3 donor + lr decay -> 0. It won wave 7 (q4), led wave 8
#     (r4), and now won wave 10 (t3), surviving the obs-600 format reset. The
#     donor s4-y3 is THE fork material; re-arm the recipe from its freshest
#     pinned state every wave.
#   * LOW ENTROPY (ent 0.015) is finally VALIDATED. t6-sharp beat its parent
#     t2-finish on ALL THREE boards (h2h 31.5>28.0, eval -0.27>-0.32, compare
#     76.5>73.0) and tied t3 for h2h #1 — losing only the champion tiebreak.
#     After three waves of "safe but never a champion", it is now a positive
#     lever. Wave 11 STACKS it onto the winning recipes (u4, u5) rather than
#     retesting it alone. Collapse guard still stands: entropy diving toward
#     ~0.1 nats = kill the arm.
#   * GENTLE lr (3e-5 -> 0) RETIRES. From a WARM value head t5-gentle only tied
#     t2-finish (h2h 29.0~28.0, eval -0.30~-0.32) — so wave-9's gentle win was a
#     fresh-value-head migration guard, not a genuine lower-peak-lr lever. Knob
#     closed; do not re-arm gentle lr.
#   * DECAY held (t2-finish healthy mid-pack; the champion t3 IS decay, on the
#     donor line). Re-armed an EIGHTH straight wave as u2-finish.
#   * CONSTANT lr is the weakest recipe (t1-main last on eval and compare, ~par
#     h2h); kept only as the u1-main control.
#   * PLACEMENT / ENTROPY-UP remain dead — never re-armed.
#
# THE STRUCTURE — wave 11 is a clean LINEAGE x ENTROPY 2x2 plus scaffolding:
#   * The obs-600 format is stable. Forks use --resume-from (warm policy AND
#     value head, step counter continues) from a pinned 600-format checkpoint.
#   * FORK_FROM = the new champion t3-y3-finish/best_model (@~285M). It is
#     RETIRED (lr annealed to ~0), so its best_model is a stable fork point and
#     the frozen --h2h/--eval yardstick — it never trains again.
#   * The DONOR lineage lives on: s4-y3 (never-annealed, constant lr) RESUMES
#     and compounds to 450M, exactly as y3-batch did through waves 4-8. The
#     cross-lineage-decay arms fork its PINNED @300M checkpoint — the freshest
#     donor state, one wave newer than the @150M that produced t3.
#   * The 2x2: {champion line = u2, donor cross-decay = u3} x {normal entropy,
#     low entropy = u4, u5}. u2/u3 are the normal-entropy cells (both decay);
#     u4/u5 add ent 0.015. Both entropy main effects and the lineage x entropy
#     interaction are estimable, with u3-y3-finish (donor, normal ent) the
#     control for the stacked u5-y3-sharp-finish (donor, low ent) — the arm that
#     combines the two winning recipes and is the wave's headline experiment.
#   * runs/sweep3 remains INERT 582-format history — never resume, never peer.
#     runs/sweep4 is this epoch's home; the wave-9 dirs (s1-s3,s5-s8) and the
#     wave-10 dirs (t1-t7) stay as inert history and are NOT peered (only the
#     wave-11 arms + the resuming donor s4-y3 peer each other).
#
# Arm roles (all lr 1e-4, n-steps/batch 1024, mix 0.40/0.40/0.20, past_k 8,
# all peers, unless noted):
#   s4-y3          RESUME the donor: constant lr, never decayed, never forked.
#                  The fork material for this and future cross-lineage decays.
#   u1-main        fork champion; constant lr — the decay/entropy control.
#   u2-finish      u1 + lr decay -> 0. The champion-line recipe, 8th re-arm;
#                  the champion x normal-entropy cell of the 2x2.
#   u3-y3-finish   fork=s4-y3 @300M + decay -> 0. Cross-lineage decay from the
#                  never-annealed donor (the q4/r4/t3 champion factory); the
#                  donor x normal-entropy cell and u5's control.
#   u4-sharp-finish  u2 + ent-coef 0.015. Low entropy on the champion line —
#                  re-arm of t6's winning recipe from the new champion; the
#                  champion x low-entropy cell.
#   u5-y3-sharp-finish  fork=s4-y3 @300M + decay + ent 0.015. THE headline:
#                  the two winning recipes stacked (cross-lineage decay x low
#                  entropy); the donor x low-entropy cell.
#   u6-small-finish  fork champion, batch 512 + decay. Small-batch+decay co-won
#                  wave 6 (p3), healthy in wave 10 (t4); keep both batch sizes
#                  and a distinct default-lineage pool opponent.
#   u7-exploit     mix 0.10/0.70/0.20, past_k 12: the pool hardener, kept a
#                  sixth wave. Hardens everyone's opponents, not a candidate.
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
#   ./scripts/sweep_selfplay.sh --h2h        # RELATIVE: each variant vs 3x the frozen champion (t3-y3-finish)
#   ./scripts/sweep_selfplay.sh --stop       # stop every running variant
#
# Env overrides (all optional):
#   TOTAL_TIMESTEPS=450000000  CUMULATIVE per-arm target (forks from ~285-300M
#                              gain ~+150-165M; resumes never overshoot)
#   FORK_FROM=runs/sweep4/t3-y3-finish/best_model   champion fork point + baseline
#   Y3_FORK=runs/sweep4/s4-y3/ckpt_300000000_steps   pinned donor fork stem
#   EVAL_OPPONENT=runs/sweep4/wave11-eval-opponent.bin
#                              frozen eval opponent, exported from FORK_FROM on
#                              first launch (selects best_model; par ~ -0.50)
#   BASELINE=t3-y3-finish      run-dir name of the frozen --h2h opponent
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
FORK_FROM=${FORK_FROM:-$SWEEP_DIR/t3-y3-finish/best_model}
Y3_FORK=${Y3_FORK:-$SWEEP_DIR/s4-y3/ckpt_300000000_steps}
EVAL_OPPONENT=${EVAL_OPPONENT:-$SWEEP_DIR/wave11-eval-opponent.bin}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-450000000}
NET_WIDTH=${NET_WIDTH:-128}
NUM_ENVS=${NUM_ENVS:-8}
THREADS=${THREADS:-3}
NICE=${NICE:-10}
STAGGER=${STAGGER:-15}
DRY_RUN=${DRY_RUN:-0}

# The frozen --h2h opponent: the wave-10 champion's best_model — the exact
# weights the fork arms started from and the eval opponent was exported from.
# t3-y3-finish no longer trains (its lr annealed to 0), so this yardstick never
# moves.
BASELINE=${BASELINE:-t3-y3-finish}

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
"s4-y3|804|resume|peers|The donor lineage: constant lr 1e-4 + batch 1024, never decayed, never forked — compounding since the obs-600 reset (now @300M, on to 450M). Joins the wave-11 pool unchanged; u3 and u5 separately decay a fresh copy of its pinned @300M state. Keeping it alive is what gives every wave an independent, never-annealed history to cross-decay.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"u1-main|1101|fork|peers|The population default and this wave's decay/entropy control: champion weights (warm value head), batch 1024, constant lr, mix 0.40/0.40/0.20. Every finisher reads against this — same food, same fork, no anneal.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"u2-finish|1102|fork|peers|The champion-line recipe re-armed an eighth time: u1-main + lr decay 1e-4 -> 0. Decay has finished on top of seven straight champions (y4, z3, p2, q4, r4, s3, t3); until it loses, every wave re-arms it from the new champion. The champion x normal-entropy cell of the 2x2 (u4 adds low entropy to it).|--learning-rate 1e-4 --lr-final 0 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"u3-y3-finish|1103|fork=$SWEEP_DIR/s4-y3/ckpt_300000000_steps|peers|Cross-lineage decay: fork the never-annealed donor's pinned @300M state + lr decay -> 0. This is the exact recipe that won waves 7 (q4), 8 (r4), and 10 (t3 -> current Expert). Forks a donor one wave newer than the @150M that produced t3, so a top finish is the factory compounding, not just repeating. The donor x normal-entropy cell of the 2x2 and u5's control.|--learning-rate 1e-4 --lr-final 0 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"u4-sharp-finish|1104|fork|peers|Low entropy on the champion line: u2-finish + ent-coef 0.015 (default 0.03). Re-arm of wave-10's t6-sharp, which beat its parent t2 on all three boards and tied t3 for h2h #1 — now validated from the new champion. The champion x low-entropy cell. Collapse guard: entropy diving toward ~0.1 nats = kill the arm.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.015 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"u5-y3-sharp-finish|1105|fork=$SWEEP_DIR/s4-y3/ckpt_300000000_steps|peers|THE headline: the two winning recipes stacked — cross-lineage decay (fork donor @300M + decay -> 0) x low entropy (ent 0.015). The donor x low-entropy cell of the 2x2; beats u3-y3-finish (its normal-entropy control) means the champion factory and the entropy lever compound. Same collapse guard as u4.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.015 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"u6-small-finish|1106|fork|peers|Batch 512 + decay: the small-batch recipe stays in play — it co-won wave 6 (p3) and was healthy in wave 10 (t4, 28.5 h2h). Half the batch is different update noise and doubles as the pool's most distinct default-lineage opponent.|--learning-rate 1e-4 --lr-final 0 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"u7-exploit|1107|fork|peers|The exploiter, kept a sixth wave: mix 0.10/0.70/0.20, past_k 12 — barely plays itself, lives on the population. Below-par h2h as a candidate; its job is hardening everyone else's opponents, not winning the wave.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.10,0.70,0.20 --league-past-k 12"
)


variant_field() { echo "${VARIANTS[$1]}" | cut -d'|' -f"$2"; }

# Comma-separated league dirs of every variant EXCEPT $1 (by name) — the
# --league-peers value for a "peers" arm. Dirs may not exist yet; the trainer
# tolerates that (empty pool slice until the peer launches). Only wave-11 arms
# (the u* arms + the resuming donor s4-y3) are peered — never an inert wave-9 or
# wave-10 dir, and never a 582-format runs/sweep3.
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
    # NOTE: best= is eval/mean_reward vs 3x the frozen champion (t3-y3-finish);
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
# champion (t3-y3-finish best_model) — the primary ranking. Above-par here ==
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
    --prepare) prepare; echo "wave-11 eval opponent ready in $SWEEP_DIR"; exit 0 ;;
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
echo "donor fork stem: $Y3_FORK (u3/u5's first launch only)"
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

    # One-time wave-11 migration for the continued donor: s4-y3's stored best
    # bar was earned vs the wave-10 eval opponent (the frozen s3-gentle) and is
    # not comparable with the new frozen champion (t3-y3-finish). Set the old
    # best aside, drop the bar, and let the new metric re-earn best_model.zip.
    # The marker file makes re-runs a no-op.
    if [[ $init == resume && $DRY_RUN != 1 && -f $dir/best_mean_reward.json \
          && ! -f $dir/.wave11-eval-opponent ]]; then
        [[ -f $dir/best_model.zip && ! -f $dir/best_model.wave10-vs-champion.zip ]] \
            && cp "$dir/best_model.zip" "$dir/best_model.wave10-vs-champion.zip"
        rm "$dir/best_mean_reward.json"
        touch "$dir/.wave11-eval-opponent"
        echo "migrated $name to the wave-11 eval metric (old best kept as best_model.wave10-vs-champion.zip)"
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
  tail -f $SWEEP_DIR/u2-finish/train.log
  $PY -m tensorboard.main --logdir $SWEEP_DIR      # league/peer_size, eval/mean_reward
  $PY scripts/run_report.py $SWEEP_DIR/u2-finish
Rank the variants:
  ./scripts/sweep_selfplay.sh --compare    # absolute: vs 3x hard bots (reporting; saturated)
  ./scripts/sweep_selfplay.sh --h2h        # relative: vs 3x the frozen $BASELINE best (primary ranking)
Stop everything:
  ./scripts/sweep_selfplay.sh --stop
EOF
