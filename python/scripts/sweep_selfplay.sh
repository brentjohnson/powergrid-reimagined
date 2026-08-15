#!/usr/bin/env bash
#
# sweep_selfplay.sh — wave 12: re-arm the new champion recipe + two CRAZY probes.
# Six "progress" arms RESUME their own 600-format checkpoints or FORK
# (--resume-from, warm policy+value head) from a pinned 600-format checkpoint —
# the proven waves-4-8/10-11 structure. Two arms go off-script (a wide net and
# a long-horizon gamma bump).
#
# WAVE 12 (2026-08-15).
#
# WAVE 11 (the LINEAGE x ENTROPY 2x2) is settled. All 8 arms to 450M. Results:
#
#   Frozen-champion eval (best mean_reward vs 3x t3-y3-finish, par ~-0.50):
#     u3-y3-finish -0.17  u4-sharp -0.20  u5-y3-sharp -0.21  u2-finish -0.24
#     u6-small -0.27  s4-y3 -0.33  u1-main -0.36  u7-exploit -0.38
#   Primary h2h (seat 0 vs 3x the frozen champion; combined 600 games over
#   seeds 12345+99999, mirror par ~23.2%):
#     u4-sharp 32.7  u2-finish 30.2  u5-y3-sharp 29.5  u3-y3-finish 29.3
#     u6-small 27.8  s4-y3 26.2  u7-exploit 23.5  u1-main 22.2 (par)
#   Compare vs 3x hard (saturated, reporting-only): u4 82.5 > u5 79.0 > u2 77.5
#     > u3 75.5 > u1 75.0 > u6 74.5 > u7 74.0 > s4 72.5.
#   The boards DISAGREED at the top (eval -> u3, h2h+compare -> u4), so the full
#   tiebreak decided it. DIRECT head-to-head (seed 77777, 400 games each way,
#   4-way par 25%):
#     u4 @seat0 vs 3x u3 = 29.0%  BEAT  u3 @seat0 vs 3x u4 = 23.0%
#     u4 @seat0 vs 3x u2 = 28.0%  BEAT  u2 @seat0 vs 3x u4 = 20.8%
#   u4-sharp-finish wins BOTH directions of BOTH matches. Confirmed vs the
#   OUTGOING Expert: u4 @seat0 vs 3x t3 = 31.0% (t3 only 21.8% the other way).
#   u4-sharp-finish is the wave-11 champion and the new embedded Expert
#   (native 40/50 = 80% vs 3x hard; h2h +9.5pp over t3's mirror par).
#
# What wave 11 SETTLED (knobs closed / opened):
#   * LOW ENTROPY (ent 0.015) on the CHAMPION LINE is now a CHAMPION recipe, not
#     just "validated": u4-sharp (champion + decay + ent 0.015) beat plain decay
#     u2 in both directions of the direct match and won the wave. Re-armed as
#     the presumptive champion v3-sharp-finish. Collapse guard stands (entropy
#     -> ~0.1 nats = kill the arm).
#   * LOW ENTROPY x DONOR LINE is DEAD: u5 (cross-decay + low ent) LOST to u3
#     (cross-decay, normal ent) on eval (-0.21<-0.17) and the direct decider.
#     The entropy lever helps the champion line and HURTS the donor line — that
#     cell is retired. Cross-lineage decay stays NORMAL entropy (v4).
#   * CROSS-LINEAGE DECAY (u3) LED eval a fourth time (-0.17, best of the field)
#     but LOST the head-to-head decider to u4. Still the strongest independent
#     lineage; kept as v4-y3-finish (the factory, from the donor's fresh @450M).
#   * DECAY held (8th straight champion is a decay arm). CONSTANT lr weakest
#     again (u1 last on eval, par h2h) — kept only as the v1-main control.
#   * PLACEMENT / ENTROPY-UP / GENTLE-lr remain dead — never re-armed.
#
# 150M vs 100M — is the wave's per-arm budget worth 150M over 100M? YES, keep
# 150M. Eval peaks land in the BACK THIRD, not early: the decay arms peaked at
# +100-145M into their 150M budgets and gained +0.06 to +0.10 mean_reward
# (~+3-5pp win) in the +100M -> +150M window — u2 -0.33 -> -0.24, u3 -0.28 ->
# -0.17, u4 -0.30 -> -0.20 — because that is where lr-decay anneals to 0 and
# converges. Cutting to 100M truncates the convergence window of the exact
# recipe that produces champions. The plateau arms (u1/u7, and the low-ent
# donor u5) don't benefit, but they don't win either. TOTAL stays 600M
# cumulative (forks from u4 @429M gain ~+171M; the donor resumes 450M -> 600M).
#
# THE STRUCTURE — six progress arms + two crazy arms:
#   * FORK_FROM = the new champion u4-sharp-finish/best_model (@429M). RETIRED
#     (lr annealed to ~0), so its best_model is a stable fork point AND the
#     frozen --h2h/--eval yardstick AND the embedded Expert — it never trains.
#   * The DONOR lineage lives on: s4-y3 (never-annealed, constant lr) RESUMES
#     450M -> 600M. v4-y3-finish forks its PINNED @450M checkpoint — the
#     freshest donor state, one wave newer than the @300M that made u3.
#   * runs/sweep4 is this epoch's home. Inert history NOT peered: wave-9 (s1-s3,
#     s5-s8), wave-10 (t1-t7), wave-11 (u1-u7). runs/sweep3 is inert 582-format
#     — never resume, never peer. Only wave-12 arms + the resuming donor peer.
#   * SOLO arms are peered by no one (v6-wide's 192-wide snapshots must never
#     enter the 128-wide PAST pool).
#
# Arm roles (all lr 1e-4, n-steps/batch 1024, mix 0.40/0.40/0.20, past_k 8,
# all peers, all fork the champion @429M, unless noted):
#   s4-y3          RESUME the donor: constant lr, never decayed, never forked.
#                  The fork material for this and future cross-lineage decays.
#   v1-main        constant lr — the decay/entropy control (re-verifies
#                  decay>constant from the new champion).
#   v2-finish      v1 + lr decay -> 0. Plain champion-line decay, 9th re-arm;
#                  the control for whether low entropy still adds on top (v3).
#   v3-sharp-finish  v2 + ent-coef 0.015 = the EXACT wave-11 champion recipe.
#                  Presumptive champion.
#   v4-y3-finish   fork=s4-y3 @450M + decay -> 0. Cross-lineage decay, normal
#                  entropy (the low-ent donor cell is retired). The factory,
#                  kept alive from the donor's freshest state.
#   v5-small-finish  batch 512 + decay. Small-batch+decay co-won wave 6 (p3),
#                  healthy in 10-11; keep both batch sizes + a distinct opponent.
#   v6-wide        *** CRAZY #1: capacity ceiling. *** FRESH net-width 192
#                  (2.25x params) warm-started from a 192-wide behavior clone
#                  (--init-policy-from clone-192.bin; value head fresh), constant
#                  lr, SOLO, gets a fixed 150M budget. Every sweep to date is
#                  128-wide; this asks if the plateau is capacity-bound. Won't
#                  win in one wave; the signal is its win%-vs-hard slope vs the
#                  historical 128 clone (wave 3: 51.5% @50M). Exportable.
#   v7-gamma       *** CRAZY #2: long-horizon credit. *** fork champion, the
#                  champion recipe (decay + ent 0.015) + gamma 0.997 (default
#                  0.99, never varied in the whole sweep). Terminal-only reward
#                  over a ~50-macro game: at 0.99 the win/loss is 0.99^50~=0.61
#                  by turn 1; 0.997 lifts that to ~0.86, propagating the finish
#                  to early decisions. v3-sharp-finish is the clean control
#                  (same recipe, gamma 0.99). gamma never touches the exported
#                  net. Watch value_loss/approx-kl for instability.
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
#   ./scripts/sweep_selfplay.sh --h2h        # RELATIVE: each variant vs 3x the frozen champion (u4-sharp-finish)
#   ./scripts/sweep_selfplay.sh --stop       # stop every running variant
#
# Env overrides (all optional):
#   TOTAL_TIMESTEPS=450000000  CUMULATIVE per-arm target (forks from ~285-300M
#                              gain ~+150-165M; resumes never overshoot)
#   FORK_FROM=runs/sweep4/u4-sharp-finish/best_model   champion fork point + baseline
#   Y3_FORK=runs/sweep4/s4-y3/ckpt_300000000_steps   pinned donor fork stem
#   EVAL_OPPONENT=runs/sweep4/wave11-eval-opponent.bin
#                              frozen eval opponent, exported from FORK_FROM on
#                              first launch (selects best_model; par ~ -0.50)
#   BASELINE=u4-sharp-finish      run-dir name of the frozen --h2h opponent
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
FORK_FROM=${FORK_FROM:-$SWEEP_DIR/u4-sharp-finish/best_model}
Y3_FORK=${Y3_FORK:-$SWEEP_DIR/s4-y3/ckpt_450000000_steps}
EVAL_OPPONENT=${EVAL_OPPONENT:-$SWEEP_DIR/wave12-eval-opponent.bin}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-600000000}
WAVE_STEPS=${WAVE_STEPS:-150000000}   # fresh (clone) arms get this per-wave budget
CLONE_192=${CLONE_192:-$SWEEP_DIR/clone-192.bin}   # the crazy wide-net warm start
NET_WIDTH=${NET_WIDTH:-128}
NUM_ENVS=${NUM_ENVS:-8}
THREADS=${THREADS:-3}
NICE=${NICE:-10}
STAGGER=${STAGGER:-15}
DRY_RUN=${DRY_RUN:-0}

# The frozen --h2h opponent: the wave-11 champion's best_model — the exact
# weights the fork arms started from and the eval opponent was exported from.
# u4-sharp-finish no longer trains (its lr annealed to 0), so this yardstick
# never moves. It is also the new embedded Expert.
BASELINE=${BASELINE:-u4-sharp-finish}

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
# (first launch --resume-from $FORK_FROM into a fresh dir), "fork=<stem>"
# (first launch --resume-from that checkpoint stem instead), or "clone" (first
# launch is a FRESH run warm-started from a behavior clone via --init-policy-from
# — the crazy wide-net arm, whose --net-width/--init-policy-from live in its
# extra flags and are ignored by the trainer on any later --resume-from). After
# the first launch every arm resumes its OWN checkpoints.
#
# `pop` is "peers" (the launch loop appends --league-peers with every OTHER
# variant's league dir) or "solo" (no peers).
#
# League mix order is LATEST,PAST,BOTS; the trainer default is 0.5,0.3,0.2.
# The 0.20 bot share is load-bearing (wave-7 q6 faceplant) — do not cut it.
VARIANTS=(
"s4-y3|804|resume|peers|The donor lineage: constant lr 1e-4 + batch 1024, never decayed, never forked — compounding since the obs-600 reset (now @450M, on to 600M). v4-y3-finish decays a fresh copy of its pinned @450M state. Keeping it alive is what gives every wave an independent, never-annealed history to cross-decay.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"v1-main|1201|fork|peers|The population default and this wave's decay/entropy control: champion weights (warm value head), batch 1024, constant lr, mix 0.40/0.40/0.20. Re-verifies decay>constant from the NEW champion; every finisher reads against this — same food, same fork, no anneal.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"v2-finish|1202|fork|peers|The champion-line recipe re-armed a ninth time: v1-main + lr decay 1e-4 -> 0. Decay has finished on top of eight straight champions (y4, z3, p2, q4, r4, s3, t3, u4); it is the control for whether low entropy (v3) still adds on top of plain decay.|--learning-rate 1e-4 --lr-final 0 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"v3-sharp-finish|1203|fork|peers|THE champion recipe re-armed: fork the champion + lr decay 1e-4 -> 0 + ent-coef 0.015 (default 0.03) — the exact wave-11 winner (u4-sharp-finish -> current Expert). Low entropy on the champion line beat plain decay (u4 beat u2 both directions of the direct match) and won the wave over cross-decay. Presumptive champion; v2-finish is its normal-entropy control. Collapse guard: entropy diving toward ~0.1 nats = kill the arm.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.015 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"v4-y3-finish|1204|fork=$SWEEP_DIR/s4-y3/ckpt_450000000_steps|peers|Cross-lineage decay, the factory kept alive: fork the never-annealed donor's pinned @450M state + lr decay -> 0. This recipe won waves 7 (q4), 8 (r4), 10 (t3), and LED wave-11 eval (u3 -0.17, best of the field) though it lost the wave-11 head-to-head decider to u4. Forks a donor one wave newer than the @300M that made u3. Normal entropy only — wave 11 showed low entropy HURTS the donor line (u5<u3), so that cell is retired.|--learning-rate 1e-4 --lr-final 0 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"v5-small-finish|1205|fork|peers|Batch 512 + decay: the small-batch recipe stays in play — it co-won wave 6 (p3) and was healthy in waves 10-11 (t4/u6). Half the batch is different update noise and doubles as the pool's most distinct default-lineage opponent.|--learning-rate 1e-4 --lr-final 0 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"v6-wide|1206|clone|solo|CRAZY #1 — the capacity-ceiling probe. Every sweep to date has been net-width 128; is the plateau capacity-bound? Fresh net-width 192 (2.25x params) warm-started from a 192-wide behavior clone (value head fresh), constant lr 1e-4, 20% bot anchor kept. SOLO (its 192-wide league snapshots are never peered into the 128 pool). Won't beat the 128 champion in one 150M wave; the signal is whether its win%-vs-hard trajectory sits ABOVE the historical 128 clone curve (wave 3: 51.5% @50M) at matched steps. Exportable (PGRLPOL6 header carries the width).|--init-policy-from $SWEEP_DIR/clone-192.bin --net-width 192 --learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"v7-gamma|1207|fork|peers|CRAZY #2 — the long-horizon probe. gamma has been 0.99 for the ENTIRE sweep; the objective is terminal-only (no shaping) and a game is ~50 macros, so at 0.99 the win/loss signal is discounted to 0.99^50 ~= 0.61 by turn 1. Raise gamma to 0.997 (0.997^50 ~= 0.86) to propagate the finish reward far more strongly to early/mid-game decisions. Built on the exact champion recipe (decay + ent 0.015) so v3-sharp-finish is its clean control: beats v3 => longer credit assignment sharpens end-game-aware play. gamma re-applies on resume and never touches the exported net; watch value_loss/approx-kl for instability.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.015 --gamma 0.997 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
)


variant_field() { echo "${VARIANTS[$1]}" | cut -d'|' -f"$2"; }

# Comma-separated league dirs of every variant EXCEPT $1 (by name) — the
# --league-peers value for a "peers" arm. Dirs may not exist yet; the trainer
# tolerates that (empty pool slice until the peer launches). Only wave-12 arms
# (the v* arms + the resuming donor s4-y3) are peered — never an inert wave-9,
# -10, or -11 dir, and never a 582-format runs/sweep3. SOLO arms are excluded
# BOTH ways: no one peers a solo arm, so the 192-wide v6-wide snapshots never
# get loaded into a 128-wide arm's PAST pool.
peer_league_dirs() {
    local self=$1 i name out=""
    for i in "${!VARIANTS[@]}"; do
        name=$(variant_field "$i" 1)
        [[ $name == "$self" ]] && continue
        [[ $(variant_field "$i" 4) == solo ]] && continue
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
    # NOTE: best= is eval/mean_reward vs 3x the frozen champion (u4-sharp-finish);
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
# champion (u4-sharp-finish best_model) — the primary ranking. Above-par here ==
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
    --prepare) prepare; echo "wave-12 eval opponent ready in $SWEEP_DIR"; exit 0 ;;
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
echo "donor fork stem: $Y3_FORK (v4-y3-finish's first launch only)"
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

    # One-time wave-12 migration for the continued donor: s4-y3's stored best
    # bar was earned vs the wave-11 eval opponent (the frozen t3-y3-finish) and
    # is not comparable with the new frozen champion (u4-sharp-finish). Set the
    # old best aside, drop the bar, and let the new metric re-earn best_model.zip.
    # The marker file makes re-runs a no-op.
    if [[ $init == resume && $DRY_RUN != 1 && -f $dir/best_mean_reward.json \
          && ! -f $dir/.wave12-eval-opponent ]]; then
        [[ -f $dir/best_model.zip && ! -f $dir/best_model.wave11-vs-champion.zip ]] \
            && cp "$dir/best_model.zip" "$dir/best_model.wave11-vs-champion.zip"
        rm "$dir/best_mean_reward.json"
        touch "$dir/.wave12-eval-opponent"
        echo "migrated $name to the wave-12 eval metric (old best kept as best_model.wave11-vs-champion.zip)"
    fi

    # Per-arm cumulative target. Fork/resume arms aim at TOTAL_TIMESTEPS (the
    # wave's cumulative budget). A fresh "clone" arm starts at 0 steps, so its
    # target is a single per-wave budget (WAVE_STEPS) — otherwise it would run
    # ~4x longer than the forks and gate the whole sweep. Using a target (not a
    # hardcoded TOTAL_TIMESTEPS) keeps idempotent relaunch correct: a
    # half-trained clone arm resumes to WAVE_STEPS, not to 600M.
    if [[ $init == clone ]]; then target=$WAVE_STEPS; else target=$TOTAL_TIMESTEPS; fi

    # Auto-resume: continue from the arm's own furthest readable checkpoint.
    # Only a fork/clone arm's very first launch uses its fork source; a
    # continuation arm with no checkpoint is an error, not a fresh start.
    start_args=()
    ckpt_stem=""; done_steps=""
    read -r ckpt_stem done_steps < <(latest_checkpoint "$dir") || true
    if [[ -n $ckpt_stem ]]; then
        steps=$(( target - done_steps ))
        if (( steps <= 0 )); then
            echo "skip $name: already at $done_steps >= target $target timesteps"
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
        steps=$(( target - fork_steps ))
        if (( steps <= 0 )); then
            echo "skip $name: fork point already at $fork_steps >= target $target" >&2
            continue
        fi
        start_args=(--resume-from "$fork_src")
        echo "forking $name from $(basename "$fork_src").zip @ $fork_steps (+$steps)"
    elif [[ $init == clone ]]; then
        # Fresh warm-start from a behavior clone (--init-policy-from lives in the
        # arm's extra flags). No --resume-from, so the trainer builds a fresh
        # model at the extra's --net-width and resets the step counter. Gets a
        # fixed per-wave budget (not the cumulative TOTAL_TIMESTEPS) because it
        # starts at 0 and would otherwise run ~4x longer than the forks.
        clone_path=""; prev=""
        for e in "${extra[@]}"; do
            [[ $prev == --init-policy-from ]] && clone_path=$e
            prev=$e
        done
        if [[ -z $clone_path || ! -f $clone_path ]]; then
            echo "cannot clone-init $name: --init-policy-from path missing/unreadable ($clone_path)" >&2
            echo "(run: python -m alphazero.pretrain --net-width <W> --export $clone_path)" >&2
            exit 1
        fi
        steps=$target   # WAVE_STEPS; fresh from 0
        start_args=()   # fresh run
        echo "clone-init $name (fresh) from $clone_path (+$steps)"
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
