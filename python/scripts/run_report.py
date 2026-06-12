"""
Status report for a training run directory: checkpoints, live process,
TensorBoard metrics, and health flags. Reads only the run dir and /proc —
no model is loaded, so it's safe to point at a run that is still training.

Usage:
    python scripts/run_report.py runs/selfplay_frozen
    python scripts/run_report.py runs/vs_bots --last 20 --all-tags
"""

import argparse
import datetime
import glob
import json
import os
import re
import sys

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

# Tags shown in the training-health trend table: (tag, display name, transform).
TREND_TAGS = [
    ("train/entropy_loss", "policy entropy (nats)", lambda v: -v),
    ("train/explained_variance", "explained variance", None),
    ("train/value_loss", "value loss", None),
    ("train/approx_kl", "approx KL", None),
    ("time/fps", "fps", None),
]

# Optional run-specific tags: shown as "first → last" when present.
EXTRA_TAGS = [
    "curriculum/end_game_cities",
    "selfplay/snapshot_timesteps",
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
]

ENTROPY_COLLAPSE_NATS = 0.2


def fmt_ts(unix: float) -> str:
    return datetime.datetime.fromtimestamp(unix).strftime("%Y-%m-%d %H:%M")


def report_inventory(run_dir: str) -> None:
    print("== Inventory ==")
    ckpts = sorted(
        (int(m.group(1)), p)
        for p in glob.glob(os.path.join(run_dir, "ckpt_*_steps.zip"))
        if (m := re.search(r"ckpt_(\d+)_steps", p))
    )
    if ckpts:
        latest_step, latest_path = ckpts[-1]
        print(f"checkpoints:   {len(ckpts)} "
              f"(steps {ckpts[0][0]:,} – {latest_step:,}, "
              f"latest written {fmt_ts(os.path.getmtime(latest_path))})")
    else:
        print("checkpoints:   none")

    best = os.path.join(run_dir, "best_model.zip")
    if os.path.exists(best):
        line = f"best_model:    saved {fmt_ts(os.path.getmtime(best))}"
        bar_path = os.path.join(run_dir, "best_mean_reward.json")
        try:
            with open(bar_path) as f:
                bar = json.load(f)["best_mean_reward"]
            line += (f", best eval mean_reward {bar:+.3f} "
                     f"(win rate ≈ {(bar + 1) / 2:.0%})")
        except FileNotFoundError:
            line += ", no best_mean_reward.json"
        print(line)
    else:
        print("best_model:    none (eval disabled, or no eval has run yet)")

    final = os.path.join(run_dir, "final_model.zip")
    print(f"final_model:   {'present (run completed)' if os.path.exists(final) else 'absent'}")


def report_process(run_dir: str) -> None:
    """Find a live python process whose command line mentions this run dir."""
    print("\n== Process ==")
    needle = os.path.basename(os.path.normpath(run_dir))
    for pid_dir in glob.glob("/proc/[0-9]*"):
        try:
            with open(os.path.join(pid_dir, "cmdline"), "rb") as f:
                argv = f.read().decode(errors="replace").split("\0")
        except OSError:
            continue
        if not any("python" in a for a in argv[:2]):
            continue
        if not any(needle in a for a in argv[1:]):
            continue
        started = fmt_ts(os.path.getmtime(pid_dir))
        print(f"RUNNING — pid {os.path.basename(pid_dir)}, started {started}")
        print(f"  {' '.join(a for a in argv if a)}")
        return
    print("not running")


def newest_tb_dir(run_dir: str) -> str | None:
    subdirs = [d for d in glob.glob(os.path.join(run_dir, "tb", "*")) if os.path.isdir(d)]
    if not subdirs:
        return None
    subdirs.sort(key=os.path.getmtime)
    if len(subdirs) > 1:
        print(f"(multiple tb runs: {', '.join(os.path.basename(d) for d in subdirs)}; "
              f"reporting the newest)")
    return subdirs[-1]


def trend_points(values: list) -> list:
    """First / 25% / 50% / 75% / last samples of a scalar series."""
    n = len(values)
    idxs = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
    return [values[i] for i in idxs]


def report_metrics(run_dir: str, last_n: int, all_tags: bool) -> dict:
    """Print metric tables; return the series needed by the health flags."""
    print("\n== Metrics ==")
    tb_dir = newest_tb_dir(run_dir)
    if tb_dir is None:
        print("no tb/ event files")
        return {}
    ea = event_accumulator.EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    series = {t: ea.Scalars(t) for t in tags}

    evals = series.get("eval/mean_reward", [])
    if evals:
        lengths = {e.step: e.value for e in series.get("eval/mean_ep_length", [])}
        shown = evals[-last_n:]
        print(f"eval vs bots ({len(evals)} evals; last {len(shown)}):")
        print("       step   mean_reward   win_rate   ep_length")
        for e in shown:
            line = f"  {e.step:>9,}        {e.value:+.3f}      {(e.value + 1) / 2:5.0%}"
            ep_len = lengths.get(e.step)
            if ep_len is not None:
                line += f"   {ep_len:9.1f}"
            print(line)
    else:
        print("eval vs bots: no eval points logged")

    print("\ntraining health (first / 25% / 50% / 75% / last):")
    for tag, name, transform in TREND_TAGS:
        if tag not in tags or not series[tag]:
            continue
        pts = trend_points(series[tag])
        vals = [transform(p.value) if transform else p.value for p in pts]
        samples = ", ".join(f"{p.step // 1000}k: {v:.3f}" for p, v in zip(pts, vals))
        print(f"  {name:<22} {samples}")

    extras = [t for t in EXTRA_TAGS if t in tags and series[t]]
    if extras:
        print()
        for tag in extras:
            first, last = series[tag][0], series[tag][-1]
            print(f"  {tag:<30} {first.value:g} (at {first.step:,}) "
                  f"→ {last.value:g} (at {last.step:,})")

    if all_tags:
        print("\nall scalar tags (last value):")
        for tag in sorted(tags):
            if series[tag]:
                last = series[tag][-1]
                print(f"  {tag:<34} {last.value:g} (at {last.step:,})")

    return series


def report_health(series: dict, run_dir: str) -> None:
    print("\n== Health flags ==")
    flags = []

    evals = series.get("eval/mean_reward", [])
    if not evals:
        flags.append("no eval points: eval never ran — note --eval-freq counts "
                     "per-ENV steps (total_timesteps / num_envs must exceed it)")
    elif all(e.value <= -1.0 for e in evals):
        flags.append(f"eval pinned at -1.0 for all {len(evals)} evals: the policy "
                     "has never beaten the eval bots — no win signal yet")

    ent = series.get("train/entropy_loss", [])
    if ent:
        entropy = -ent[-1].value
        if entropy < ENTROPY_COLLAPSE_NATS:
            flags.append(f"policy entropy {entropy:.3f} nats (< {ENTROPY_COLLAPSE_NATS}): "
                         "near-deterministic policy, exploration has collapsed")

    ev = series.get("train/explained_variance", [])
    vl = series.get("train/value_loss", [])
    eval_flat = len(evals) >= 3 and len({round(e.value, 3) for e in evals[-10:]}) == 1
    if (ev and ev[-1].value > 0.99 and vl and vl[-1].value < 1e-3 and eval_flat):
        flags.append("critic converged to a constant outcome (explained_variance ≈ 1, "
                     "value_loss ≈ 0, eval flat): no learning gradient left — "
                     "the reward signal is unreachable or degenerate")

    bar_path = os.path.join(run_dir, "best_mean_reward.json")
    if evals and os.path.exists(bar_path):
        with open(bar_path) as f:
            bar = json.load(f)["best_mean_reward"]
        latest = evals[-1].value
        if latest < bar - 0.2:
            flags.append(f"latest eval ({latest:+.3f}) is well below the best bar "
                         f"({bar:+.3f}): recent training has regressed vs best_model")

    if flags:
        for f in flags:
            print(f"  ⚠ {f}")
    else:
        print("  none — no known failure pattern detected")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Training run directory, e.g. runs/selfplay")
    parser.add_argument("--last", type=int, default=10,
                        help="How many recent eval points to list.")
    parser.add_argument("--all-tags", action="store_true",
                        help="Also dump the last value of every scalar tag.")
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        sys.exit(f"not a directory: {args.run_dir}")

    print(f"Run report: {args.run_dir}")
    report_inventory(args.run_dir)
    report_process(args.run_dir)
    series = report_metrics(args.run_dir, args.last, args.all_tags)
    report_health(series, args.run_dir)


if __name__ == "__main__":
    main()
