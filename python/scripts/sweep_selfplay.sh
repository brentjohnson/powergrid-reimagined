#!/usr/bin/env bash
#
# sweep_selfplay.sh — wave 9: the obs-600 format reset. Every arm is a fresh
# run warm-started from a MIGRATED clone of a wave-8 donor; the script
# auto-migrates every policy it touches.
#
# WAVE 9 (2026-08-06).
#
# Wave 8 was STOPPED EARLY (~453-497M of the 550M target, i.e. only ~46-70M
# into the wave) for the observation format change: obs 582 -> 600 with the
# end-game-race section 22 (see TRAINING-REVIEW.md). What the frozen-champion
# eval (vs 3x q4-y3-finish, par ~ -0.50) recorded before the stop:
#
#   r4-y3-finish     -0.34 (~33%)  <- wave-8 leader
#   y3-batch         -0.36 (~32%)  (496M, still compounding, never decayed)
#   r1-main          -0.38         r2-finish  -0.38    r5-sharp  -0.38
#   r7-exploit       -0.40
#   r3-small-finish  -0.44
#   r6-q3-finish     -0.49         <- at par: the q3 cross-decay FAILED
#
# What the partial wave settled:
#   * CROSS-LINEAGE DECAY FROM y3 LED AGAIN (r4, second straight wave) — but
#     r6 (same recipe from the q3 donor) finished at par. The recipe is
#     donor-specific: y3's never-annealed history is the champion factory,
#     not "any strong donor". Knob closed: y3 is THE donor.
#   * r5 (ent 0.015): entropy eased 0.25 -> 0.21 with no collapse and no
#     lead — inconclusive at the stop. Re-armed this wave as s8-sharp.
#   * No full tiebreak ran (the wave was cut short, and the 582-format
#     checkpoints cannot play under the 600-format env on this machine), so
#     r4 is the wave-8 leader on the frozen-champion eval ONLY. Wave 9's own
#     --h2h/--compare confirm or overturn it as the arms train.
#
# THE FORMAT RESET (why this wave looks different):
#   * Old sb3 checkpoints CANNOT be resumed (582-wide l1). Every arm is a
#     FRESH run (--init-policy-from) warm-started from a migrated .bin:
#     zero-padded l1 rows make the clone play the donor's policy exactly
#     (Rust-forward bit-identical), while gradient can now reach the 18 new
#     end-game-race inputs.
#   * Value heads start FRESH (a .bin carries only the policy path). Expect
#     an early eval dip while the critic re-learns; s3-gentle exists to
#     measure how much a small-lr start protects the clone (the
#     --init-policy-from docs recommend one).
#   * Step counters start at 0, so TOTAL_TIMESTEPS is a per-arm FRESH
#     budget (default 150M), not the old cumulative-across-waves number.
#   * runs/sweep3 is INERT HISTORY plus donor source — never resume, never
#     peer into it (its league snapshots are 582 and would crash a
#     new-format trainer at reset). This wave lives in runs/sweep4.
#
# AUTO-MIGRATION (all idempotent, all before any launch):
#   * wave9-champion.bin  <- migrate_policy_obs.py --from-ckpt on the wave-8
#     leader's best_model (r4-y3-finish @ -0.34)
#   * wave9-y3.bin        <- same, from y3-batch's best_model (the donor)
#   * wave9-eval-opponent.bin <- the champion bin (frozen; selects
#     best_model; par ~25% == mean_reward ~ -0.50, negative best= is normal)
#   * wave9-baseline/best_model.zip <- --bin-to-ckpt on the champion bin, so
#     --h2h can seat the frozen champion under the new format
#   * any pre-existing .bin named via EVAL_OPPONENT/CHAMP_BIN/Y3_BIN is
#     width-checked and zero-padded in place if it is still 582
#
# Arm roles (all clones of wave9-champion.bin except s4; all peers; lr 1e-4,
# n-steps/batch 1024, mix 0.40/0.40/0.20, past_k 8 unless noted):
#   s1-main       the population default and decay control: constant lr
#   s2-finish     champion recipe re-armed a sixth time: s1 + lr decay -> 0
#   s3-gentle     lr 3e-5 -> 0: the fresh-value-head guard. If it holds the
#                 clone better than s1/s2 early, gentle starts become the
#                 format-migration default
#   s4-y3         clone of wave9-y3.bin, constant lr, never decayed: the
#                 donor lineage reconstituted in the new format so future
#                 waves have a fresh independent history to cross-decay
#   s5-placement  s2 + --terminal-reward placement: rank-ladder terminal
#                 reward (1st..4th mapped to +1..-1). The new section-22
#                 features expose exactly the powered->money->cities race
#                 that decides ranks; placement pays partial credit for
#                 winning that race from behind. Eval envs stay winloss, so
#                 best= remains comparable across arms
#   s6-explore    s1 + --ent-coef 0.045 (1.5x default): the new race inputs
#                 enter with ZERO weights — only exploration can surface
#                 push behaviors that exploit them; this arm pays extra
#                 entropy to look. If it just adds noise, kill the knob
#   s7-exploit    mix 0.10/0.70/0.20, past_k 12: the pool hardener, kept a
#                 fourth wave — its job is hardening everyone's opponents
#   s8-sharp      s2 + --ent-coef 0.015: r5's inconclusive probe re-armed.
#                 Collapse guard: entropy diving toward ~0.1 nats = kill
#
# Sized for a 28-core machine: 8 variants x THREADS=3 = 24 cores, leaving
# headroom for the eval passes and the OS.
#
# Re-running is idempotent and self-healing: launching is the same command as
# resuming. For each selected variant the script inspects its run dir and
# picks up where it left off — resume from the furthest-along readable
# checkpoint; if there is none, the arm starts fresh from its clone source
# (--init-policy-from is only ever used for a variant's FIRST launch; after
# that it resumes its own 600-format checkpoints). The running-check verifies
# the recorded PID is still a train_selfplay.py process for THIS run dir, so
# a stale pidfile (e.g. a PID recycled across a reboot) can't block a resume
# or, worse, let two trainers write the same dir. The intended operational
# loop is simply: run it, and if the box reboots or a variant crashes, run it
# again.
#
# Resume-lr note: MaskablePPO.load is passed custom_objects built from THIS
# launch's flags, so an arm's lr schedule is its own across resumes.
#
# League note: snapshots live in each run dir's own league/ subdir, all
# 600-format from birth. A fresh arm's league is empty, but the trainer
# pushes a snapshot of the just-loaded clone at training start, so a nonzero
# PAST share never sees an empty pool.
#
# Usage:
#   ./scripts/sweep_selfplay.sh              # launch/resume all 8 in the background
#   ./scripts/sweep_selfplay.sh 3 5          # launch/resume only variants 3 and 5
#   ./scripts/sweep_selfplay.sh --prepare    # run the format guard + all migrations, launch nothing
#   ./scripts/sweep_selfplay.sh --list       # show the variant table, launch nothing
#   ./scripts/sweep_selfplay.sh --status     # per-variant progress / best eval
#   ./scripts/sweep_selfplay.sh --compare    # ABSOLUTE: each variant vs 3x hard bots
#   ./scripts/sweep_selfplay.sh --h2h        # RELATIVE: each variant vs 3x the frozen wave-8 leader
#   ./scripts/sweep_selfplay.sh --stop       # stop every running variant
#
# Env overrides (all optional):
#   TOTAL_TIMESTEPS=150000000  FRESH timesteps per arm (counters reset at 0
#                              this wave; resumes still never overshoot)
#   WAVE8_WINNER=../runs-or-wherever/r4-y3-finish/best_model
#                              582-format checkpoint stem the champion bin is
#                              migrated from
#   Y3_SOURCE=runs/sweep3/y3-batch/best_model   likewise for the donor bin
#   CHAMP_BIN=runs/sweep4/wave9-champion.bin    migrated champion (built if
#                              missing, width-fixed in place if 582)
#   Y3_BIN=runs/sweep4/wave9-y3.bin             migrated donor clone source
#   EVAL_OPPONENT=runs/sweep4/wave9-eval-opponent.bin
#                              frozen eval opponent (defaults to a copy of
#                              $CHAMP_BIN; width-fixed in place if 582)
#   SWEEP_DIR=runs/sweep4      root for the per-variant run dirs
#   NET_WIDTH=128              must match the clone bins' hidden width
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
SWEEP_DIR=${SWEEP_DIR:-runs/sweep4}
OLD_SWEEP_DIR=${OLD_SWEEP_DIR:-runs/sweep3}
WAVE8_WINNER=${WAVE8_WINNER:-$OLD_SWEEP_DIR/r4-y3-finish/best_model}
Y3_SOURCE=${Y3_SOURCE:-$OLD_SWEEP_DIR/y3-batch/best_model}
CHAMP_BIN=${CHAMP_BIN:-$SWEEP_DIR/wave9-champion.bin}
Y3_BIN=${Y3_BIN:-$SWEEP_DIR/wave9-y3.bin}
EVAL_OPPONENT=${EVAL_OPPONENT:-$SWEEP_DIR/wave9-eval-opponent.bin}
BASELINE_STEM=${BASELINE_STEM:-$SWEEP_DIR/wave9-baseline/best_model}
TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS:-150000000}
NET_WIDTH=${NET_WIDTH:-128}
NUM_ENVS=${NUM_ENVS:-8}
THREADS=${THREADS:-3}
NICE=${NICE:-10}
STAGGER=${STAGGER:-15}
DRY_RUN=${DRY_RUN:-0}

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
                                # selects best_model vs the frozen wave-8
                                # leader. Par ~25% == mean_reward ~ -0.50,
                                # so best= in --status is NEGATIVE by design.
    --save-freq 250000          # ~2M timesteps per checkpoint at 8 envs
    --eval-freq 50000           # ~400k timesteps per eval pass
    --eval-episodes 200         # 20 (the trainer default) is too noisy to rank
)

# name|seed|init|pop|hypothesis|extra flags
#
# `init` is "clone" (first launch warm-starts from $CHAMP_BIN via
# --init-policy-from) or "clone=<bin>" (warm-start from that migrated .bin
# instead). There are no resume/fork arms this wave: 582-format checkpoints
# cannot be resumed, only cloned through migration.
#
# `pop` is "peers" (the launch loop appends --league-peers with every OTHER
# variant's league dir) or "solo" (no peers).
#
# League mix order is LATEST,PAST,BOTS; the trainer default is 0.5,0.3,0.2.
# The 0.20 bot share is load-bearing (wave-7 q6 faceplant) — do not cut it.
VARIANTS=(
"s1-main|801|clone|peers|The population default and decay control: champion clone, constant lr 1e-4, batch 1024, mix 0.40/0.40/0.20. Every other arm reads against this — same food, same clone, no anneal.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"s2-finish|802|clone|peers|The champion-line recipe re-armed a sixth time: s1-main + lr decay 1e-4 -> 0. Decay has finished on top of five straight waves (y4, z3, p2, q4, r4); until it loses, every wave re-arms it from the new champion.|--learning-rate 1e-4 --lr-final 0 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"s3-gentle|803|clone|peers|The fresh-value-head guard: lr 3e-5 -> 0. Every arm starts with a random critic this wave; the --init-policy-from docs recommend a small lr so the first noisy-advantage updates cannot wreck the clone. If s3 holds the clone visibly better than s1/s2 early, gentle starts become the format-migration default.|--learning-rate 3e-5 --lr-final 0 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"s4-y3|804|clone=Y3_BIN|peers|The donor lineage reconstituted: clone of y3-batch's best (-0.36 at 496M), constant lr, never decayed, never forked. y3's independent never-annealed history is the proven champion factory (q4, then r4 — and r6 showed OTHER donors do not work); this arm rebuilds that resource in the new format for future cross-decays.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"s5-placement|805|clone|peers|s2-finish + --terminal-reward placement: terminal reward = final rank on +1..-1 instead of win/loss. The new section-22 features expose the powered/money/cities race that decides ranks; placement pays partial credit for climbing it, giving the race features a denser gradient than the 25%-of-episodes win signal. Eval envs stay winloss, so best= reads on the same scale as every other arm.|--learning-rate 1e-4 --lr-final 0 --terminal-reward placement --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"s6-explore|806|clone|peers|s1-main + ent-coef 0.045 (1.5x default): the 18 race inputs enter with zero weights, so only behavioral exploration can surface the end-game-push lines that make them pay. This arm buys extra entropy to look; if it lags s1 at equal steps the knob just adds noise — close it.|--learning-rate 1e-4 --ent-coef 0.045 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
"s7-exploit|807|clone|peers|The exploiter, kept a fourth wave: mix 0.10/0.70/0.20, past_k 12 — barely plays itself, lives on the population. Its job is hardening everyone else's opponents, not winning the wave.|--learning-rate 1e-4 --n-steps 1024 --batch-size 1024 --league-mix 0.10,0.70,0.20 --league-past-k 12"
"s8-sharp|808|clone|peers|s2-finish + ent-coef 0.015: wave 8's r5 probe re-armed — entropy eased 0.25 -> 0.21 nats with no collapse before the stop, verdict pending. Collapse guard: if entropy dives toward ~0.1 nats early, kill the arm — that is the old collapse signature.|--learning-rate 1e-4 --lr-final 0 --ent-coef 0.015 --n-steps 1024 --batch-size 1024 --league-mix 0.40,0.40,0.20 --league-past-k 8"
)


variant_field() { echo "${VARIANTS[$1]}" | cut -d'|' -f"$2"; }

# Comma-separated league dirs of every variant EXCEPT $1 (by name) — the
# --league-peers value for a "peers" arm. Dirs may not exist yet; the trainer
# tolerates that (empty pool slice until the peer launches). NEVER add an old
# 582-format league dir here — its snapshots would crash a new-format trainer.
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

# --- format / migration helpers -------------------------------------------

# Hard guard: the venv must be built for the 600-format encoding. Launching a
# 582-format trainer into this wave's dirs (or vice versa) corrupts nothing
# but wastes a run; refuse instead.
check_format() {
    "$PY" - <<'EOF' || { echo "venv is not built for the current encoding (run 'make develop')" >&2; exit 1; }
from powergrid_env.constants import OBS_SIZE
import powergrid_py, numpy as np
assert OBS_SIZE == 600, f"constants OBS_SIZE={OBS_SIZE}, expected 600"
g = powergrid_py.Game(4, 1)
g.start(["a","b","c","d"], ["red","blue","green","yellow"])
n = np.asarray(g.observation(g.player_ids()[0])).shape[0]
assert n == 600, f"native obs is {n}-wide, expected 600 (stale powergrid_py build)"
EOF
}

# Width-fix an existing .bin in place (no-op if already current-width).
migrate_bin_inplace() {
    "$PY" scripts/migrate_policy_obs.py "$1"
}

# Ensure a migrated clone .bin exists at $2: width-fix it if present, else
# build it from the 582-format checkpoint stem $1.
ensure_clone_bin() {
    local src=$1 out=$2
    if [[ -f $out ]]; then
        migrate_bin_inplace "$out"
        return 0
    fi
    if [[ ! -f ${src}.zip ]]; then
        echo "cannot build $out: no checkpoint at ${src}.zip" >&2
        echo "(sync the wave-8 run dirs, or point WAVE8_WINNER/Y3_SOURCE elsewhere)" >&2
        exit 1
    fi
    echo "migrating $(basename "$src").zip -> $out"
    "$PY" scripts/migrate_policy_obs.py --from-ckpt "$src" --out "$out"
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
    # NOTE: best= is eval/mean_reward vs 3x the frozen wave-8 leader
    # (win_rate = (best+1)/2; par ~ -0.50). Negative values are normal, and
    # early values will sag while each arm's fresh value head warms up.
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
# against 0. Saturated for ranking since wave 6 (the 582-format field crowded
# 63-72%); kept for wave-end reporting and cross-wave comparability — and to
# confirm the migrated clones actually re-attain the wave-8 numbers.
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
# wave-8 leader — materialised as a real checkpoint (wave9-baseline) because
# the 582-format original cannot run under the new env. This is the primary
# ranking. Above-par here == genuinely past the clone point. Measure the
# mirror par (the self-baseline row) before reading small edges; expect
# ~25 +/- 2%. At wave end, run the FULL tiebreak: direct matches between the
# leaders + a fresh-seed 800-game h2h (wave 7's one-shot 400-game samples
# flipped ranks).
h2h() {
    local games=${COMPARE_GAMES:-200} seed=${COMPARE_SEED:-12345}
    local det=(); [[ ${COMPARE_DETERMINISTIC:-0} == 1 ]] && det=(--deterministic)
    local base=$BASELINE_STEM
    [[ -f ${base}.zip ]] || { echo "no baseline at ${base}.zip — run the script once (it materialises the frozen champion)" >&2; exit 1; }
    echo "=== self-baseline: 4x wave-8 leader clone ($games games) — mirror par ==="
    "$PY" scripts/evaluate_lineup.py --games "$games" --seed "$seed" --quiet "${det[@]}" \
        --player "$base" --player "$base" --player "$base" --player "$base"
    for i in "${!VARIANTS[@]}"; do
        local name model
        name=$(variant_field "$i" 1); model="$SWEEP_DIR/$name/best_model"
        [[ -f ${model}.zip ]] || continue
        echo
        echo "=== $name (seat 0) vs 3x wave-8 leader ($games games) ==="
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

# All pre-launch setup: format guard + every auto-migration. Idempotent; also
# exposed as --prepare so the wave-9 artifacts can be staged (e.g. on a dev
# box, then synced to the training machine) without launching anything.
prepare() {
    check_format
    mkdir -p "$SWEEP_DIR"

    # Donor clones (auto-migrated; idempotent).
    ensure_clone_bin "$WAVE8_WINNER" "$CHAMP_BIN"
    ensure_clone_bin "$Y3_SOURCE" "$Y3_BIN"

    # The frozen eval opponent every arm's best_model selection runs against:
    # the wave-8 leader, i.e. exactly the champion clone bin.
    if [[ -f $EVAL_OPPONENT ]]; then
        migrate_bin_inplace "$EVAL_OPPONENT"
    else
        echo "freezing eval opponent: $CHAMP_BIN -> $EVAL_OPPONENT"
        cp "$CHAMP_BIN" "$EVAL_OPPONENT"
    fi

    # Materialise the frozen champion as a runnable checkpoint for --h2h.
    if [[ ! -f ${BASELINE_STEM}.zip ]]; then
        echo "materialising --h2h baseline: $CHAMP_BIN -> ${BASELINE_STEM}.zip"
        "$PY" scripts/migrate_policy_obs.py --bin-to-ckpt "$CHAMP_BIN" --out "$BASELINE_STEM"
    fi
}

case "${1:-}" in
    --list)    list_variants; exit 0 ;;
    --status)  status;        exit 0 ;;
    --stop)    stop_all;      exit 0 ;;
    --compare) compare;       exit 0 ;;
    --h2h)     h2h;           exit 0 ;;
    --prepare) prepare; echo "wave-9 artifacts ready in $SWEEP_DIR"; exit 0 ;;
esac

# Which variants to launch (1-based indices; default all).
if (( $# )); then
    SELECTED=("$@")
else
    SELECTED=($(seq 1 ${#VARIANTS[@]}))
fi

[[ -x $PY ]] || { echo "no interpreter at $PY (run 'make develop' first)" >&2; exit 1; }

# Refuse to write the new-format wave into the old sweep root.
if [[ $(realpath -m "$SWEEP_DIR") == $(realpath -m "$OLD_SWEEP_DIR") ]]; then
    echo "SWEEP_DIR must not be the 582-format $OLD_SWEEP_DIR (it is history + donor source)" >&2
    exit 1
fi

[[ $DRY_RUN == 1 ]] || prepare

echo "champion clone : $CHAMP_BIN (from $WAVE8_WINNER)"
echo "donor clone    : $Y3_BIN (from $Y3_SOURCE)"
echo "eval opponent  : $EVAL_OPPONENT (frozen; selects best_model)"
echo "h2h baseline   : ${BASELINE_STEM}.zip"
echo "target         : $TOTAL_TIMESTEPS fresh timesteps per variant"
echo "sweep dir      : $SWEEP_DIR"
echo "launching      : ${SELECTED[*]}"
echo

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

    # Per-arm clone source: "clone" uses $CHAMP_BIN, "clone=Y3_BIN" the donor
    # bin (symbolic name, resolved here so the table stays override-friendly).
    clone_src=$CHAMP_BIN
    if [[ $init == clone=* ]]; then
        case ${init#clone=} in
            Y3_BIN) clone_src=$Y3_BIN ;;
            *)      clone_src=${init#clone=} ;;
        esac
        init=clone
    fi

    # Already running? Never start a second writer on the same run dir. The
    # check confirms the pidfile's PID is genuinely this variant's trainer, so
    # a stale/recycled PID neither blocks a needed resume nor risks a duplicate.
    live=$(running_pid "$dir")
    if [[ -n $live ]]; then
        echo "skip $name: already running (pid $live)"
        continue
    fi

    # Auto-resume: continue from the arm's own furthest readable checkpoint.
    # Only the very first launch clones from the migrated .bin.
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
    else
        if [[ $DRY_RUN != 1 && ! -f $clone_src ]]; then
            echo "cannot clone $name: no migrated bin at $clone_src" >&2
            exit 1
        fi
        steps=$TOTAL_TIMESTEPS
        start_args=(--init-policy-from "$clone_src")
        echo "cloning $name from $(basename "$clone_src") (+$steps, fresh value head)"
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
        echo "# launched: $(date -Is)  init: $init  pop: $pop  target: $TOTAL_TIMESTEPS fresh"
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
  ./scripts/sweep_selfplay.sh --status     # best= is vs the frozen wave-8 leader: win = (best+1)/2, par ~ -0.50
  tail -f $SWEEP_DIR/s2-finish/train.log
  $PY -m tensorboard.main --logdir $SWEEP_DIR      # league/peer_size, eval/mean_reward
  $PY scripts/run_report.py $SWEEP_DIR/s2-finish
Rank the variants:
  ./scripts/sweep_selfplay.sh --compare    # absolute: vs 3x hard bots (reporting; watch clones re-attain ~70%)
  ./scripts/sweep_selfplay.sh --h2h        # relative: vs 3x the frozen wave-8 leader (primary ranking)
Stop everything:
  ./scripts/sweep_selfplay.sh --stop
EOF
