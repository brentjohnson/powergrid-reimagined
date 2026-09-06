#!/usr/bin/env bash
#
# sweep_selfplay.sh — wave 18: exploit the obs-624 refit + hunt a lever past the
# plateau + two real swings. NOT a migration wave: the observation stays 624, so
# every arm can FORK the wave-17 winner's own 624 checkpoint directly (no clone /
# migrate). Wave 17 re-established the champion under obs-624 and the plain
# champion recipe (b1) topped every lever it was raced against — the same
# near-plateau shape as wave 15, but one rung higher: b1 = a3 + the learned
# section-23 market features, a genuine successor to the embedded a3. Wave 18
# forks b1, re-tests the two levers that TIED it (gae 0.98, ent 0.015) as warm
# continuations + stacks them, probes two fresh axes (value emphasis, n-epochs),
# and spends its two crazy slots on moves the 17-wave warm-start lineage has
# never allowed: a FROM-SCRATCH run and a wide-trust-region basin escape.
#
# WAVE 18 (2026-09-06). Fork source + eval opponent + h2h yardstick + new
# embedded Expert are all b1-champ-cont (the wave-17 winner). Its dir is frozen
# (no wave-18 arm writes to it), so it is a stable reference.
#
# WAVE 17 (obs-624 migration + rollout-curve + 2 weight-space bets) is settled.
# All 8 arms reached 250M (fresh clones of the migrated a3 @624). Result:
# b1-champ-cont WINS — the plain champion recipe, refit to the 24 new section-23
# market inputs, beat every lever.
#
#   Frozen-champion eval (best mean_reward vs 3x the migrated a3 @624, par ~-0.50):
#     b1 -0.31  b2/b4/c2 -0.32  b5/b6/c1 -0.33  b3 -0.36
#   Primary h2h (seat 0 vs 3x the frozen a3 @624; mirror par 23.0%):
#     b1 25.0  b4 25.0  b5 24.0  c2 23.0  b3/c1 22.5  b6 19.5  b2 18.5
#   Compare vs 3x hard (saturated, reporting-only, seat-0 all-bots par ~21.5%):
#     c2 87.0  b1/b5 86.5  c1 85.5  b2/b6 84.0  b4 82.5  b3 79.5
#   b1 LED the two boards that count (eval #1, h2h tied #1). A fresh-seed decider
#   (seed 88888, 400 games/dir) confirmed b1 over the incumbent a3 and over its
#   only h2h co-leader b4 (see the journal). b1 is the champion and new Expert.
#
# What wave 17 SETTLED / OPENED:
#   * *** 4096 IS THE ROLLOUT PEAK. *** The bracket resolved cleanly: n-steps
#     2048 (b3, h2h 22.5) < 4096 (b1, 25.0) > 8192 (b2, 18.5). The right shoulder
#     falls off HARD (8192 is worst on h2h), so wave 18 does NOT re-map the curve
#     — 4096 is locked as the champion default. gamma STAYS 0.999.
#   * THE TWO HORIZON LEVERS TIED, DIDN'T WIN. gae-lambda 0.98 (b4) tied b1 on
#     h2h (25.0) but trailed on eval and was weakest on compare; ent 0.015 (b5)
#     was a hair back (24.0). Both were tested as FRESH CLONES (value head
#     refitting from zero over 24 new inputs) — a lever's effect is muddied while
#     the whole net stabilizes. Wave 18 re-tests both as WARM continuations of
#     the converged b1 (d3 gae, d2 ent) where a lever operates cleanly, and
#     STACKS them (d4) to see if the two near-ties combine.
#   * MIGRATION GUARD not needed: b6-gentle (lr 3e-5) landed BELOW b1 (h2h 19.5),
#     so the standard lr 1e-4 clones were healthy — the fresh value head did NOT
#     damage them. Wave 18 uses standard lr 1e-4->0 on all forks (gentle retired).
#   * WEIGHT-SPACE MOVES sat at/below par again: c1-extrapolate (-0.5*x5+1.5*a3)
#     h2h 22.5 (extrapolation did not overshoot to a win), c2-swa mean(a1,a3,a6)
#     h2h 23.0 = par. Three waves of soups/merges now cluster at par. The family
#     is RETIRED as a champion source; wave 18 spends the crazy slots elsewhere.
#   * DEAD LEVERS (do not re-test): lr restart 3e-4, relative shaping, n-epochs 8
#     + target-kl, entropy-UP, placement reward, wide net (192-warm), target-KL,
#     tight clip, gamma 0.99, cross-lineage fork at a new gamma, gentle-lr as a
#     lever, soups/extrapolation. The 0.20 bot anchor is load-bearing; keep it.
#
# Budget: fork arms (d1-d6, e2) resume b1's best_model @250M and run to
# TOTAL_TIMESTEPS = 550M (a +300M increment = a full fresh lr 1e-4->0 anneal
# cycle, "the finisher", on the converged champion). The one from-scratch arm
# (e1) is a clone from step 0 with WAVE_STEPS = 300M, so all eight arms run ~300M
# of new steps and finish together.
#
# THE STRUCTURE — six progress arms (1 control + 2 warm lever re-tests + 1 stack
# + 2 fresh axes), all 128-wide forks of b1, peered; + two crazies:
#   * FORK_FROM = b1-champ-cont/best_model (the wave-17 winner, 624 format).
#     d1-d6/e2 all --resume-from it; d1 is the pure continuation control.
#   * EVAL_OPPONENT = wave18-champion.bin = b1 exported to a PGRLPOL6 .bin
#     (--prepare runs export_policy.py). BASELINE = b1-champ-cont's own dir (its
#     best_model.zip is the frozen --h2h yardstick; nothing trains it this wave).
#   * runs/sweep4 stays this epoch's home. Only wave-18 arms peer each other; the
#     from-scratch arm e1 is SOLO (its weak early snapshots stay out of the
#     progress arms' PAST pool, and it stays a clean "no lineage" test). Inert
#     history (wave-9..17 s*/t*/u*/v*/w*/x*/y*/z*/a*/b*/c*) is never peered.
#
# Arm roles (all gamma 0.999, mix 0.40/0.40/0.20, past_k 8, ent 0.03, lr 1e-4->0,
# n-steps/batch 4096, vf-coef 0.5, n-epochs 4, clip 0.2, gae 0.95, fork of b1,
# unless noted):
#   d1-champ-cont     CONTROL / anchor: fork b1, champion recipe unchanged, +300M
#                     (a second full lr anneal cycle). Tests whether the champion
#                     line keeps climbing past 250M, and is the reference every
#                     other arm is read against on the shared peers/seeds.
#   d2-low-ent        WARM lever re-test: d1 but ent-coef 0.015 (validated
#                     champion-line sharpen, wave 11). b5 tested it as a cold
#                     clone (24.0); re-test on the WARM converged champion where
#                     sharpening should actually bite. Kill guard: entropy diving
#                     toward ~0.1 nats = kill the arm.
#   d3-gae98          WARM lever re-test: d1 but gae-lambda 0.98. b4 tied b1 on
#                     h2h as a cold clone (25.0); re-test warm to see if the
#                     advantage-horizon lever separates once the value head is
#                     converged rather than refitting.
#   d4-sharp-gae      LEVER STACK: d1 + ent 0.015 + gae 0.98 — the two arms that
#                     tied/near-tied b1 in wave 17, combined. Do the two
#                     independent near-ties add to a real edge, or does the field
#                     just sit at the same converged strength?
#   d5-vf-emphasis    FRESH AXIS (critic weight): d1 but vf-coef 1.0 (from 0.5).
#                     The value head was just refit over the 24 new section-23
#                     market inputs; a stronger critic term may exploit them
#                     better. Value-emphasis lost in wave 15, but on a long-
#                     converged critic — here the critic is freshly informative.
#   d6-epochs6        FRESH AXIS (optimization passes): d1 but n-epochs 6 (from
#                     4). More reuse of each big 4096 rollout. a5's n-epochs 8
#                     lost, but only WITH target-kl (the dead part); this is a
#                     milder plain increase with no kl cap. Watch policy drift.
#   e1-scratch        *** CRAZY #1 — FROM SCRATCH. *** No warm start: a fresh
#                     128-wide net at the champion recipe, but a bot-heavy league
#                     mix 0.30/0.10/0.60 so macro-space self-play can bootstrap
#                     off the heuristic instead of random-vs-random. SOLO, 300M
#                     from step 0. The one foundational assumption 17 waves never
#                     tested: can macro-space RL reach the warm-start lineage
#                     champion FROM ZERO? A win reframes the whole grind; a loss
#                     definitively shows the lineage/warm-start is load-bearing.
#   e2-wide-clip      *** CRAZY #2 — basin escape via a wide trust region. ***
#                     Fork b1 + clip-range 0.3 (from 0.2), champion recipe
#                     otherwise, +300M. lr-restart (wave 16 a2) failed to leave
#                     the x5 basin because its steps stayed clip-capped at 0.2;
#                     a wider clip is the untested way to take LARGE policy steps
#                     and reach weights a normal anneal cannot. lr decays to 0 so
#                     it re-settles. If it just reconverges to b1, the basin is
#                     clip-robust too; if it lands somewhere new, follow it.
#
# Sized for a 28-core machine: 8 variants x THREADS=3 = 24 cores, leaving
# headroom for the eval passes and the OS.
#
# Re-running is idempotent and self-healing: launching is the same command as
# resuming. For each selected variant the script inspects its run dir and
# resumes from the furthest-along readable checkpoint; if there is none, a fork
# arm starts from its (pinned) fork source, a scratch arm starts fresh, and a
# resume arm hard-errors rather than silently restarting. The running-check
# verifies the recorded PID is still a train_selfplay.py process for THIS run
# dir, so a stale pidfile can't block a resume or let two trainers write the
# same dir. Operational loop: run it; if the box reboots or an arm crashes, run
# it again.
#
# Resume-lr note: MaskablePPO.load is passed custom_objects built from THIS
# launch's flags, so an arm's lr schedule is its own across resumes (a fork's
# fresh lr 1e-4 overrides the decayed lr stored in b1's checkpoint).
#
# Usage:
#   ./scripts/sweep_selfplay.sh              # launch/resume all 8 in the background
#   ./scripts/sweep_selfplay.sh 3 5          # launch/resume only variants 3 and 5
#   ./scripts/sweep_selfplay.sh --prepare    # format guard + export the eval opponent, launch nothing
#   ./scripts/sweep_selfplay.sh --list       # show the variant table, launch nothing
#   ./scripts/sweep_selfplay.sh --status     # per-variant progress / best eval
#   ./scripts/sweep_selfplay.sh --compare    # ABSOLUTE: each variant vs 3x hard bots
#   ./scripts/sweep_selfplay.sh --h2h        # RELATIVE: each variant vs 3x the frozen champion (b1 @624)
#   ./scripts/sweep_selfplay.sh --stop       # stop every running variant
#
# Env overrides (all optional):
#   TOTAL_TIMESTEPS=550000000  fork-arm cumulative target (b1 @250M + 300M)
#   WAVE_STEPS=300000000       from-scratch arm budget (e1, from step 0)
#   FORK_FROM=runs/sweep4/b1-champ-cont/best_model   wave-17 winner (fork source)
#   CHAMPION_BIN=runs/sweep4/wave18-champion.bin  b1 exported to .bin (eval opponent), built in --prepare
#   EVAL_OPPONENT=$CHAMPION_BIN   frozen champion b1 (par ~ -0.50)
#   BASELINE=b1-champ-cont     run-dir whose best_model is the frozen --h2h yardstick
#   SWEEP_DIR=runs/sweep4      root for the per-variant run dirs
#   NET_WIDTH=128              hidden width of b1 and all forks
#   NUM_ENVS=8                 parallel envs per variant (keep equal across variants)
#   THREADS=3                  torch/OMP threads per variant (8 x 3 = 24 of 28 cores)
#   COMPARE_GAMES=200  COMPARE_SEED=12345
#   COMPARE_DETERMINISTIC=1    rank with argmax instead of sampling. Training is
#                              stochastic, so the sampled numbers remain primary.
#   NICE=10  STAGGER=15  DRY_RUN=1

set -euo pipefail

cd "$(dirname "$0")/.."          # python/

PY=${PY:-.venv/bin/python}
SWEEP_DIR=${SWEEP_DIR:-runs/sweep4}
# NOT a migration wave: the observation stays 624, so fork arms --resume-from
# the wave-17 winner's own 624 checkpoint directly. FORK_FROM is b1's best_model;
# CHAMPION_BIN is b1 exported to a PGRLPOL6 .bin (the frozen --eval opponent),
# built in --prepare via export_policy.py.
FORK_FROM=${FORK_FROM:-$SWEEP_DIR/b1-champ-cont/best_model}   # wave-17 winner (624 ckpt); d1-d6/e2 fork it
CHAMPION_BIN=${CHAMPION_BIN:-$SWEEP_DIR/wave18-champion.bin}  # b1 exported to .bin: the frozen --eval opponent
EVAL_OPPONENT=${EVAL_OPPONENT:-$CHAMPION_BIN}
                             # Frozen champion b1 @624. Selects best_model; par ~ -0.50.
WAVE_STEPS=${WAVE_STEPS:-300000000}          # from-scratch arm e1 budget (from step 0)
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-550000000}   # fork-arm cumulative target = b1 @250M + 300M
NET_WIDTH=${NET_WIDTH:-128}
NUM_ENVS=${NUM_ENVS:-8}
THREADS=${THREADS:-3}
NICE=${NICE:-10}
STAGGER=${STAGGER:-15}
DRY_RUN=${DRY_RUN:-0}

# The frozen --h2h opponent: the wave-17 winner b1-champ-cont. Its own
# best_model.zip is already 624-format and runnable, and no wave-18 arm writes to
# its dir (the continuation control is d1-champ-cont), so this yardstick never
# moves. It is the new embedded Expert (pending the user's in-game test).
BASELINE=${BASELINE:-b1-champ-cont}

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
# (first launch --resume-from that checkpoint stem instead), "clone" (first
# launch is a FRESH run warm-started from a policy .bin via --init-policy-from in
# the arm's extra flags), or "scratch" (a FRESH run from a randomly-initialised
# net — no --resume-from, no --init-policy-from; --net-width comes from COMMON or
# an extra override). clone/scratch arms start at step 0 and get WAVE_STEPS; fork
# arms aim at TOTAL_TIMESTEPS. After the first launch every arm resumes its OWN
# checkpoints.
#
# `pop` is "peers" (the launch loop appends --league-peers with every OTHER
# variant's league dir) or "solo" (no peers).
#
# League mix order is LATEST,PAST,BOTS; the trainer default is 0.5,0.3,0.2.
# The 0.20 bot share is load-bearing (wave-7 q6 faceplant) — do not cut it.
VARIANTS=(
"d1-champ-cont|1801|fork|peers|CONTROL / anchor: fork the wave-17 winner b1 @250M and continue the champion recipe unchanged for +300M (a second full lr 1e-4 -> 0 anneal cycle — 'the finisher' re-armed on the converged champion). Tests whether the champion line keeps climbing past 250M; the reference every other arm is read against on the same peers/seeds. Collapse guard: entropy diving toward ~0.1 nats = kill the arm.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --n-steps 4096 --batch-size 4096 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"d2-low-ent|1802|fork|peers|WARM lever re-test (entropy sharpen): d1 but ent-coef 0.015 (half the 0.03 default). Validated as a champion-line lever in wave 11; b5 tested it in wave 17 as a COLD clone (value head refitting) and it landed a hair back (h2h 24.0). Re-test on the WARM, converged champion where sharpening should actually bite. Watch entropy: a dive toward ~0.1 nats = kill.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.015 --gamma 0.999 --n-steps 4096 --batch-size 4096 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"d3-gae98|1803|fork|peers|WARM lever re-test (advantage horizon): d1 but gae-lambda 0.98 (from 0.95). b4 TIED b1 on h2h in wave 17 (25.0) as a cold clone; re-test warm to see if the advantage-horizon lever separates once the value head is converged rather than refitting. At gamma 0.999, lambda 0.95 still discounts advantages on a ~20-step scale, so 0.98 lengthens advantage estimation toward the discount horizon.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --gae-lambda 0.98 --n-steps 4096 --batch-size 4096 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"d4-sharp-gae|1804|fork|peers|LEVER STACK: d1 + ent-coef 0.015 + gae-lambda 0.98 — the two arms that tied/near-tied b1 in wave 17 (b4 gae, b5 ent), combined on the warm champion. Do the two independent near-ties ADD to a real edge over d1, or does the field just sit at the same converged strength (the plateau signature)?|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.015 --gamma 0.999 --gae-lambda 0.98 --n-steps 4096 --batch-size 4096 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"d5-vf-emphasis|1805|fork|peers|FRESH AXIS (critic weight): d1 but vf-coef 1.0 (from 0.5). The value head was just refit over the 24 new section-23 market inputs, so a stronger critic term may exploit them better and sharpen advantage estimates. Value-emphasis lost in wave 15, but on a long-converged critic; here the critic is freshly informative, a genuinely different context.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --vf-coef 1.0 --n-steps 4096 --batch-size 4096 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"d6-epochs6|1806|fork|peers|FRESH AXIS (optimization passes): d1 but n-epochs 6 (from 4) — more reuse of each big 4096 rollout per collection. a5's n-epochs 8 lost in wave 16, but ONLY paired with target-kl (the dead part); this is a milder plain increase with no kl cap. More passes can improve sample efficiency but raise per-update policy drift — watch approx_kl.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --n-epochs 6 --n-steps 4096 --batch-size 4096 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"e1-scratch|1807|scratch|solo|CRAZY #1 — FROM SCRATCH. No warm start: a randomly-initialised 128-wide net at the champion recipe, but a bot-heavy league mix 0.30,0.10,0.60 so macro-space self-play can bootstrap off the heuristic instead of random-vs-random. SOLO (its weak early snapshots stay out of the progress arms' PAST pool, and it stays a clean 'no lineage' test). 300M from step 0. The one foundational assumption 17 waves never tested: can macro-space RL reach the warm-start lineage champion FROM ZERO? A win reframes the whole grind; a loss definitively shows the lineage is load-bearing.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --n-steps 4096 --batch-size 4096 --league-mix 0.30,0.10,0.60 --league-past-k 8"
"e2-wide-clip|1808|fork|peers|CRAZY #2 — basin escape via a wide trust region. Fork b1 + clip-range 0.3 (from 0.2), champion recipe otherwise, +300M. lr-restart (wave 16 a2) failed to leave the x5 basin because its steps stayed clip-capped at 0.2; a wider clip is the untested way to take LARGE policy steps and reach weights a normal anneal cannot. lr still decays to 0 so it re-settles. If it just reconverges to d1, the basin is clip-robust too; if it lands somewhere new, follow it next wave. Watch approx_kl / entropy for instability.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.03 --gamma 0.999 --clip-range 0.3 --n-steps 4096 --batch-size 4096 --league-mix 0.40,0.40,0.20 --league-past-k 8"
)


variant_field() { echo "${VARIANTS[$1]}" | cut -d'|' -f"$2"; }

# Comma-separated league dirs of every variant EXCEPT $1 (by name) — the
# --league-peers value for a "peers" arm. Dirs may not exist yet; the trainer
# tolerates that (empty pool slice until the peer launches). Only wave-18 arms
# (the d* progress arms + the crazy fork e2-wide-clip) are peered — never an
# inert wave-9-through-17 dir (the x*/y*/z*/a*/b*/c* arms, kept for reference
# only), and never a 582-format runs/sweep3. SOLO arms are excluded BOTH ways:
# no one peers a solo arm, and a solo arm peers no one — this wave that is
# e1-scratch, kept out of the progress arms' PAST pool so its weak early
# from-scratch snapshots don't dilute their opponent mix.
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

# Hard guard: the venv must be built for the 624-format encoding (obs 600 -> 624,
# section 23 per-slot market features). Launching a stale 600-format trainer into
# this epoch's dirs wastes a run; refuse instead. Run 'make develop' after the
# encoding change to rebuild powergrid_py.
check_format() {
    "$PY" - <<'EOF' || { echo "venv is not built for the obs-624 encoding (run 'make develop')" >&2; exit 1; }
from powergrid_env.constants import OBS_SIZE
import powergrid_py, numpy as np
assert OBS_SIZE == 624, f"constants OBS_SIZE={OBS_SIZE}, expected 624"
g = powergrid_py.Game(4, 1)
g.start(["a","b","c","d"], ["red","blue","green","yellow"])
n = np.asarray(g.observation(g.player_ids()[0])).shape[0]
assert n == 624, f"native obs is {n}-wide, expected 624 (stale powergrid_py build)"
EOF
}

# All pre-launch setup for wave 18. Idempotent (skips if its output exists);
# exposed as --prepare so it can be staged on a dev box and synced to the
# training machine. NOT a migration wave — obs stays 624, so the only build is:
#   * CHAMPION_BIN   b1's best_model exported to a PGRLPOL6 .bin (the frozen
#                    --eval opponent). The fork source (FORK_FROM) and the --h2h
#                    yardstick ($BASELINE) are b1's own 624 checkpoint directly,
#                    so nothing else needs building.
prepare() {
    check_format
    mkdir -p "$SWEEP_DIR"

    # Export the wave-17 winner b1 -> a PGRLPOL6 .bin for --eval-opponent. b1's
    # best_model.zip is already 624-format, so this is a plain export (no migrate).
    if [[ ! -f $CHAMPION_BIN ]]; then
        if [[ ! -f ${FORK_FROM}.zip ]]; then
            echo "cannot export eval opponent: no checkpoint at ${FORK_FROM}.zip" >&2
            echo "(set FORK_FROM/CHAMPION_BIN, or sync the b1-champ-cont run dir)" >&2
            exit 1
        fi
        echo "exporting eval opponent (b1) -> $CHAMPION_BIN"
        # Pin --golden next to the .bin: export_policy.py defaults --golden to
        # assets/policies/expert.golden.json, which would clobber the embedded
        # expert's golden. The eval opponent only needs the .bin.
        "$PY" scripts/export_policy.py --model "$FORK_FROM" --out "$CHAMPION_BIN" \
            --golden "${CHAMPION_BIN%.bin}.golden.json"
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
    # NOTE: best= is eval/mean_reward vs 3x the frozen champion (b1-champ-cont);
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
# champion (b1-champ-cont best_model) — the primary ranking. Above-par here ==
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
    --prepare) prepare; echo "wave-18 eval opponent ready in $SWEEP_DIR"; exit 0 ;;
esac

# Which variants to launch (1-based indices; default all).
if (( $# )); then
    SELECTED=("$@")
else
    SELECTED=($(seq 1 ${#VARIANTS[@]}))
fi

[[ -x $PY ]] || { echo "no interpreter at $PY (run 'make develop' first)" >&2; exit 1; }

[[ $DRY_RUN == 1 ]] || prepare

echo "fork source    : $FORK_FROM (wave-17 winner b1 @250M; d1-d6/e2 --resume-from it)"
echo "eval opponent  : $EVAL_OPPONENT (b1 exported; frozen; selects best_model)"
echo "h2h baseline   : $SWEEP_DIR/$BASELINE/best_model (frozen b1 @624)"
echo "fork target    : $TOTAL_TIMESTEPS cumulative (b1 @250M + 300M); scratch budget: $WAVE_STEPS"
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

    # No eval-metric migration needed: the eval opponent changed (wave-17 frozen
    # a3 -> the new champion b1), but every wave-18 arm is a fresh dir (d*/e*)
    # with no stored best bar, so nothing carries a bar earned against the old
    # opponent. All arms' best bars are earned against b1 from their first eval.

    # Per-arm cumulative target. Fork/resume arms aim at TOTAL_TIMESTEPS (the
    # wave's cumulative budget). A fresh "clone" or "scratch" arm starts at 0
    # steps, so its target is a single per-wave budget (WAVE_STEPS) — otherwise
    # it would run longer than the forks' increment and gate the whole sweep.
    # Using a target (not a hardcoded TOTAL_TIMESTEPS) keeps idempotent relaunch
    # correct: a half-trained clone/scratch arm resumes to WAVE_STEPS.
    if [[ $init == clone || $init == scratch ]]; then target=$WAVE_STEPS; else target=$TOTAL_TIMESTEPS; fi

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
    elif [[ $init == scratch ]]; then
        # From-scratch run: a randomly-initialised net (no --resume-from, no
        # --init-policy-from). The trainer builds a fresh model at --net-width
        # (COMMON's, or an arm override) and resets the step counter. Gets the
        # per-wave budget (WAVE_STEPS) like a clone arm.
        steps=$target   # WAVE_STEPS; fresh from 0
        start_args=()   # fresh run, random init
        echo "scratch-init $name (random init, fresh) (+$steps)"
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
  tail -f $SWEEP_DIR/b1-champ-cont/train.log
  $PY -m tensorboard.main --logdir $SWEEP_DIR      # league/peer_size, eval/mean_reward
  $PY scripts/run_report.py $SWEEP_DIR/b1-champ-cont
Rank the variants:
  ./scripts/sweep_selfplay.sh --compare    # absolute: vs 3x hard bots (reporting; saturated)
  ./scripts/sweep_selfplay.sh --h2h        # relative: vs 3x the frozen $BASELINE best (primary ranking)
Stop everything:
  ./scripts/sweep_selfplay.sh --stop
EOF
