#!/usr/bin/env bash
#
# sweep_selfplay.sh — wave 13: map the gamma curve around the new champion +
# two CRAZY probes. Six "progress" arms RESUME their own 600-format checkpoints
# or FORK (--resume-from, warm policy+value head) from a pinned 600-format
# checkpoint — the proven waves-4-8/10-12 structure. Two arms go off-script (an
# undiscounted gamma extreme and the continued wide net).
#
# WAVE 13 (2026-08-18).
#
# WAVE 12 (re-arm champion recipe + two crazy probes) is settled. All 8 arms
# reached their targets (forks/resumes 600M, the wide clone 150M). Results:
#
#   Frozen-champion eval (best mean_reward vs 3x u4-sharp-finish, par ~-0.50):
#     v3-sharp -0.25  v4-y3 -0.25  v7-gamma -0.25  v2-finish -0.31
#     v5-small -0.34  s4-y3 -0.35  v1-main -0.41  v6-wide -0.67
#   Compare vs 3x hard (saturated, reporting-only, seat-0 par ~21.5%):
#     v7-gamma 84.0  v3-sharp 82.0  v5-small 82.0  v4-y3 81.0  v2-finish 78.5
#     v1-main 74.0  s4-y3 73.5  v6-wide 65.0
#   Primary h2h (seat 0 vs 3x the frozen champion u4-sharp; mirror par ~22-25%):
#     v7-gamma 36.0  v2-finish 30.0  v4-y3 28.5  v5-small 25.5  v3-sharp 24.0
#     s4-y3 22.0  v1-main 21.5  v6-wide 10.5
#   v7-gamma LED ALL THREE boards (unlike wave 11, where they disagreed). The
#   DIRECT decider (seed 77777, 400 games each way, 4-way par 25%) confirmed it
#   both directions of both matches:
#     v7-gamma @seat0 vs 3x v2 = 26.8%  BEAT  v2 @seat0 vs 3x v7 = 24.2%
#     v7-gamma @seat0 vs 3x v4 = 25.5%  BEAT  v4 @seat0 vs 3x v7 = 23.5%
#   Margins are modest (the 128-wide field is saturated and near-mirror), but
#   the direction is consistent everywhere. v7-gamma is the wave-12 champion and
#   the new embedded Expert; it beat the OUTGOING champion u4-sharp by +11-14pp
#   on the h2h yardstick (36.0% vs par ~22-25%).
#
# What wave 12 SETTLED / OPENED:
#   * *** GAMMA IS THE BREAKTHROUGH. *** The crazy #2 probe won. gamma had been
#     0.99 for the ENTIRE sweep; raising it to 0.997 (0.997^50~=0.86 vs
#     0.99^50~=0.61) propagates the terminal win/loss far more strongly to
#     early/mid-game decisions, and v7-gamma beat its own clean control
#     v3-sharp-finish (identical recipe, gamma 0.99) on every board — h2h 36 vs
#     24, compare 84 vs 82. Longer credit assignment sharpens end-game-aware
#     play. gamma NEVER touches the exported net (training-only), so every
#     gamma-varied arm is exportable. Wave 13's job: locate the peak of the
#     gamma curve (0.995 / 0.997 / 0.999 / 1.0) and re-settle the entropy
#     question in the new gamma regime.
#   * ENTROPY re-opened. Wave 11 found low ent (0.015) helps the champion line;
#     wave 12 muddied it — v2-finish (normal ent, gamma 0.99) beat v3-sharp
#     (low ent, gamma 0.99) on h2h (30 vs 24), yet the winner v7 was low-ent +
#     high-gamma. So low ent's value may be gamma-dependent. w4-gamma-nent is
#     the clean control: same as the champion but ent 0.03, to decide whether
#     low ent still earns its place now that gamma is high.
#   * DECAY held (9th straight champion is a decay arm). CONSTANT lr weakest
#     again (v1-main last-but-wide on eval). Kept only as the donor's regime.
#   * CROSS-LINEAGE DECAY (v4-y3) again a strong independent lineage (h2h 28.5,
#     3rd). Kept as w5-y3-gamma — now forking the donor AND carrying the new
#     gamma, so the factory tracks the champion recipe.
#   * WIDE NET (v6-wide, crazy #1) inconclusive after one 150M clone wave:
#     65.0% vs hard at 150M, below the 128 field but a fresh-from-clone start.
#     Not thrown away — CONTINUED another 150M (crazy again) to read a real
#     slope before judging the capacity ceiling.
#
# 150M per-arm convergence budget still holds — eval peaks land in the back
# third as lr-decay anneals to 0. Forks from v7-gamma @~598M gain ~+150M to
# 750M cumulative; the donor resumes 600M -> 750M; the wide net 150M -> 300M.
#
# THE STRUCTURE — six progress arms + two crazy arms:
#   * FORK_FROM = the new champion v7-gamma/best_model (@~598M). RETIRED (lr
#     annealed to ~0), so its best_model is a stable fork point AND the frozen
#     --h2h/--eval yardstick AND the embedded Expert — it never trains.
#   * The DONOR lineage lives on: s4-y3 (never-annealed, constant lr, gamma 0.99)
#     RESUMES 600M -> 750M. w5-y3-gamma forks its PINNED @600M checkpoint — the
#     freshest donor state — and adds decay + gamma 0.997.
#   * runs/sweep4 is this epoch's home. Inert history NOT peered: wave-9 (s*),
#     wave-10 (t*), wave-11 (u*), wave-12 (v1-v5, v7). runs/sweep3 is inert
#     582-format — never resume, never peer. Only wave-13 arms + the resuming
#     donor s4-y3 peer.
#   * SOLO arms are peered by no one (v6-wide's 192-wide snapshots must never
#     enter the 128-wide PAST pool).
#
# Arm roles (all lr 1e-4, n-steps/batch 1024, mix 0.40/0.40/0.20, past_k 8,
# all peers, all fork the champion v7-gamma @~598M, unless noted):
#   s4-y3          RESUME the donor: constant lr, never decayed, never forked,
#                  gamma 0.99. The fork material for cross-lineage decays.
#   w1-champ-gamma The EXACT wave-12 champion recipe re-armed: decay -> 0 +
#                  ent 0.015 + gamma 0.997. Presumptive champion; the control
#                  every gamma/entropy variant reads against.
#   w2-gamma999    w1 but gamma 0.999 (0.999^50~=0.95). Pushes the horizon
#                  further — does more gamma keep helping, or did 0.997 peak?
#   w3-gamma995    w1 but gamma 0.995. Brackets the peak below 0.997, so w3/w1/w2
#                  triangulate the gamma curve (0.995/0.997/0.999).
#   w4-gamma-nent  w1 but ent-coef 0.03 (normal). The clean entropy control in
#                  the new gamma regime — decides if low ent still earns its
#                  place now that gamma carries the finish signal.
#   w5-y3-gamma    fork=s4-y3 @600M + decay -> 0 + gamma 0.997, normal entropy.
#                  Cross-lineage decay with the new gamma — the factory, kept
#                  alive from the donor's freshest state (low ent stays retired
#                  on the donor line, wave 11).
#   v6-wide        *** CRAZY #1: capacity ceiling, continued. *** RESUME the
#                  net-width 192 clone (2.25x params) another 150M -> 300M,
#                  constant lr, SOLO (its 192-wide snapshots never peer into the
#                  128 pool). One 150M wave was inconclusive (65% vs hard); this
#                  reads the win%-vs-hard slope at 300M to judge whether the
#                  plateau is capacity-bound. Exportable (width in the header).
#   w6-gamma-max   *** CRAZY #2: the undiscounted extreme. *** fork champion,
#                  the champion recipe (decay + ent 0.015) + gamma 1.0 — NO
#                  discounting, the terminal win/loss weighted equally across
#                  every decision in the game. gamma 0.997 beat 0.99 decisively;
#                  1.0 is the limit of that dimension. w1 is the clean control.
#                  Value learning may destabilize with no discount horizon —
#                  watch value_loss/explained_variance/approx-kl; entropy toward
#                  ~0.1 nats = kill it. gamma never touches the exported net.
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
#   ./scripts/sweep_selfplay.sh --h2h        # RELATIVE: each variant vs 3x the frozen champion (v7-gamma)
#   ./scripts/sweep_selfplay.sh --stop       # stop every running variant
#
# Env overrides (all optional):
#   TOTAL_TIMESTEPS=750000000  CUMULATIVE per-arm target (forks from ~598-600M
#                              gain ~+150M; resumes never overshoot)
#   FORK_FROM=runs/sweep4/v7-gamma/best_model   champion fork point + baseline
#   Y3_FORK=runs/sweep4/s4-y3/ckpt_600000000_steps   pinned donor fork stem
#   EVAL_OPPONENT=runs/sweep4/wave13-eval-opponent.bin
#                              frozen eval opponent, exported from FORK_FROM on
#                              first launch (selects best_model; par ~ -0.50)
#   BASELINE=v7-gamma          run-dir name of the frozen --h2h opponent
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
FORK_FROM=${FORK_FROM:-$SWEEP_DIR/v7-gamma/best_model}
Y3_FORK=${Y3_FORK:-$SWEEP_DIR/s4-y3/ckpt_600000000_steps}
EVAL_OPPONENT=${EVAL_OPPONENT:-$SWEEP_DIR/wave13-eval-opponent.bin}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-750000000}
WAVE_STEPS=${WAVE_STEPS:-300000000}   # clone/continued-solo arms' cumulative target (v6-wide 150M -> 300M)
CLONE_192=${CLONE_192:-$SWEEP_DIR/clone-192.bin}   # the crazy wide-net warm start
NET_WIDTH=${NET_WIDTH:-128}
NUM_ENVS=${NUM_ENVS:-8}
THREADS=${THREADS:-3}
NICE=${NICE:-10}
STAGGER=${STAGGER:-15}
DRY_RUN=${DRY_RUN:-0}

# The frozen --h2h opponent: the wave-12 champion's best_model — the exact
# weights the fork arms started from and the eval opponent was exported from.
# v7-gamma no longer trains (its lr annealed to 0), so this yardstick never
# moves. It is also the new embedded Expert.
BASELINE=${BASELINE:-v7-gamma}

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
"s4-y3|804|resume|peers|The donor lineage: constant lr 1e-4 + batch 1024, gamma 0.99, never decayed, never forked — compounding since the obs-600 reset (now @600M, on to 750M). w5-y3-gamma decays a fresh copy of its pinned @600M state and adds gamma 0.997. Keeping it alive is what gives every wave an independent, never-annealed history to cross-decay.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"w1-champ-gamma|1301|fork|peers|THE champion recipe re-armed: fork the champion (v7-gamma) + lr decay 1e-4 -> 0 + ent-coef 0.015 + gamma 0.997 — the exact wave-12 winner (v7-gamma -> current Expert). Presumptive champion and the control every gamma/entropy variant reads against. Collapse guard: entropy diving toward ~0.1 nats = kill the arm.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.015 --gamma 0.997 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"w2-gamma999|1302|fork|peers|Push the horizon further: w1 but gamma 0.999 (0.999^50 ~= 0.95 vs 0.997^50 ~= 0.86). gamma 0.997 beat 0.99 decisively in wave 12; does even longer credit assignment keep helping, or did 0.997 already sit at the peak? Beats w1 => climb further; loses => 0.997 is near-optimal. Watch value_loss/approx-kl — less discounting is harder to fit.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.015 --gamma 0.999 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"w3-gamma995|1303|fork|peers|Bracket the peak below: w1 but gamma 0.995 (0.995^50 ~= 0.78). With w2 (0.999) above and w1 (0.997) at center, w3/w1/w2 triangulate the gamma curve so the champion recipe's gamma is chosen from a measured peak, not a lucky first hit.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.015 --gamma 0.995 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"w4-gamma-nent|1304|fork|peers|The entropy control in the new gamma regime: w1 but ent-coef 0.03 (normal, the trainer default). Wave 11 said low ent (0.015) helps the champion line; wave 12 muddied it (normal-ent v2 beat low-ent v3 on h2h, yet the winner v7 was low-ent + high-gamma). This isolates entropy at fixed gamma 0.997 — if w4 >= w1, low ent no longer earns its place and the champion recipe drops back to normal entropy.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.997 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"w5-y3-gamma|1305|fork=$SWEEP_DIR/s4-y3/ckpt_600000000_steps|peers|Cross-lineage decay with the new gamma, the factory kept alive: fork the never-annealed donor's pinned @600M state + lr decay -> 0 + gamma 0.997. Cross-lineage decay was a strong independent lineage again in wave 12 (v4-y3 h2h 28.5, 3rd); now it also carries the winning gamma so the factory tracks the champion recipe. Normal entropy only — wave 11 showed low entropy HURTS the donor line, so that cell stays retired.|--learning-rate 1e-4 --lr-final 0 --gamma 0.997 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"v6-wide|1206|clone|solo|CRAZY #1 continued — the capacity-ceiling probe, second wave. RESUME the net-width 192 clone (2.25x params) another 150M -> 300M, constant lr 1e-4, 20% bot anchor, SOLO (its 192-wide league snapshots are never peered into the 128 pool). One 150M clone wave was inconclusive (65% vs 3x hard); a second reads the win%-vs-hard slope at 300M to judge whether the 128 plateau is capacity-bound before committing a full run. --net-width/--init-policy-from are ignored on resume (width read from the checkpoint). Exportable (PGRLPOL6 header carries the width).|--init-policy-from $SWEEP_DIR/clone-192.bin --net-width 192 --learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"w6-gamma-max|1306|fork|peers|CRAZY #2 — the undiscounted extreme. fork the champion + the champion recipe (decay + ent 0.015) + gamma 1.0 — NO discounting at all, so the terminal win/loss is weighted equally across every decision in the ~50-macro game (the limit of the dimension gamma 0.997 won on). w1-champ-gamma is the clean control (identical but gamma 0.997). Value learning may destabilize with no discount horizon — watch value_loss/explained_variance/approx-kl closely; entropy toward ~0.1 nats = kill it. gamma re-applies on resume and never touches the exported net.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.015 --gamma 1.0 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
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
    # NOTE: best= is eval/mean_reward vs 3x the frozen champion (v7-gamma);
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
# champion (v7-gamma best_model) — the primary ranking. Above-par here ==
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
echo "donor fork stem: $Y3_FORK (w5-y3-gamma's first launch only)"
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

    # One-time wave-13 migration for any arm continuing from an earlier wave
    # (the resuming donor s4-y3 and the continued wide net v6-wide): their stored
    # best bar was earned vs the wave-12 eval opponent (the frozen u4-sharp) and
    # is not comparable with the new frozen champion (v7-gamma). Set the old best
    # aside, drop the bar, and let the new metric re-earn best_model.zip. Fresh
    # fork arms have no best bar yet, so they skip this. The marker makes re-runs
    # a no-op.
    if [[ $DRY_RUN != 1 && -f $dir/best_mean_reward.json \
          && ! -f $dir/.wave13-eval-opponent ]]; then
        [[ -f $dir/best_model.zip && ! -f $dir/best_model.wave12-vs-champion.zip ]] \
            && cp "$dir/best_model.zip" "$dir/best_model.wave12-vs-champion.zip"
        rm "$dir/best_mean_reward.json"
        touch "$dir/.wave13-eval-opponent"
        echo "migrated $name to the wave-13 eval metric (old best kept as best_model.wave12-vs-champion.zip)"
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
  tail -f $SWEEP_DIR/w1-champ-gamma/train.log
  $PY -m tensorboard.main --logdir $SWEEP_DIR      # league/peer_size, eval/mean_reward
  $PY scripts/run_report.py $SWEEP_DIR/w1-champ-gamma
Rank the variants:
  ./scripts/sweep_selfplay.sh --compare    # absolute: vs 3x hard bots (reporting; saturated)
  ./scripts/sweep_selfplay.sh --h2h        # relative: vs 3x the frozen $BASELINE best (primary ranking)
Stop everything:
  ./scripts/sweep_selfplay.sh --stop
EOF
