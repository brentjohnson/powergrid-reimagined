"""
Forever-training orchestrator: train → evaluate → adapt → repeat.

Manages a small population of self-play runs ("lineages"), each advanced one
*segment* of timesteps at a time as a subprocess of scripts/train_selfplay.py.
After each segment the resulting model is evaluated offline vs heuristic bots
(scripts/evaluate.py) and the next segment's hyperparameters are chosen by the
triage rules from TRAINING-NEXT-STEPS.md:

  - never won a game        → restart the lineage with the end-game curriculum
  - entropy collapse        → step up ent-coef (0.03 → 0.1 → 0.15)
  - eval plateau            → flip the opponent-mix knob (league bots weight)
  - eval regression         → resume from best_model, then cross-pollinate
                              from the best other lineage
  - dead lineage (200M+ flat) → retire it and spawn a mutated replacement

Every decision and its reasoning is appended to runs/orch/journal.md; machine
state lives in runs/orch/state.json. The orchestrator can be killed and
restarted at any time: it re-attaches to still-running training subprocesses
(they are not killed with it) and otherwise resumes each lineage from its
newest checkpoint. Console output is intentionally sparse — one line per
decision/result plus a live progress bar; details go to the journal and to
per-segment logs in runs/orch/<lineage>/logs/.

Usage (from python/):
    .venv/bin/python scripts/orchestrate.py                    # run forever
    .venv/bin/python scripts/orchestrate.py --once             # one full cycle
    .venv/bin/python scripts/orchestrate.py --segment-steps 15_000_000
"""

import argparse
import datetime
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

from powergrid_env.run_metrics import load_tb_series

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # python/
PY = sys.executable

# Triage thresholds (TRAINING-NEXT-STEPS.md / run_report.py).
ENTROPY_COLLAPSE_NATS = 0.2
ENTROPY_RECOVERED_NATS = 0.9
ENT_COEF_LADDER = [0.03, 0.1, 0.15]
PLATEAU_SEGMENTS = 3          # this many flat segment evals → flip the mix knob
PLATEAU_EPS = 0.04            # "flat" = reward spread below this (2% win rate)
REGRESSION_DELTA = 0.2        # reward drop below lineage best → regressed
RETIRE_STEPS = 200_000_000    # min lifetime before a lineage can be retired
RETIRE_STALE_STEPS = 100_000_000  # no new best for this long → retired
HARD_EVAL_AT = 0.2            # best reward vs normal (60% win) → also eval vs hard
NEAR_SOLVED = 0.9             # vs-normal reward where that metric stops informing

# Starting knob presets. Every lineage trains 4-player league self-play with
# annealed absolute shaping and placement terminal reward; they differ in
# network width, seed, and league mix (the population's diversity).
DEFAULT_LINEAGES = {
    "a-league": dict(net_width=128, seed=0, league_mix=[0.5, 0.3, 0.2]),
    "b-wide": dict(net_width=256, seed=1000, league_mix=[0.4, 0.2, 0.4]),
}
BASE_KNOBS = dict(
    ent_coef=0.03,
    league_past_k=4,
    anneal_shaping_steps=40_000_000,
    terminal_reward="placement",
    snapshot_every=250_000,
    curriculum_start=None,      # set by the pinned-at-zero-wins rule
    curriculum_every=5_000_000,
)


def now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fmt_steps(n: float) -> str:
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def atomic_write_json(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)


def pid_alive(pid: int, needle: str) -> bool:
    """Is `pid` a live process whose command line mentions `needle`?"""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            argv = f.read().decode(errors="replace")
    except OSError:
        return False
    return needle in argv


def ckpt_steps(path: str) -> int:
    m = re.search(r"ckpt_(\d+)_steps", path)
    return int(m.group(1)) if m else 0


def newest_model(run_dir: str) -> str | None:
    """Newest complete save (final_model or ckpt), path without .zip."""
    cands = glob.glob(os.path.join(run_dir, "ckpt_*_steps.zip"))
    final = os.path.join(run_dir, "final_model.zip")
    if os.path.exists(final):
        cands.append(final)
    if not cands:
        return None
    return max(cands, key=os.path.getmtime)[: -len(".zip")]


def total_steps(run_dir: str) -> int:
    """Lineage lifetime timesteps, from checkpoint names (cheap, no TB read)."""
    return max((ckpt_steps(p) for p in glob.glob(os.path.join(run_dir, "ckpt_*_steps.zip"))),
               default=0)


def tail(path: str, nbytes: int = 20_000) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - nbytes))
            return f.read().decode(errors="replace")
    except OSError:
        return ""


class Orchestrator:
    def __init__(self, args):
        self.args = args
        self.orch_dir = os.path.join(BASE, args.orch_dir)
        self.state_path = os.path.join(self.orch_dir, "state.json")
        self.journal_path = os.path.join(self.orch_dir, "journal.md")
        os.makedirs(self.orch_dir, exist_ok=True)
        self.procs: dict[str, subprocess.Popen] = {}  # our own children only
        self.state = self._load_state()
        self._bar_len = 0

    # ------------------------------------------------------------ state

    def _load_state(self) -> dict:
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                return json.load(f)
        lineages = {}
        for name in self.args.lineages.split(","):
            preset = DEFAULT_LINEAGES.get(name, {})
            lineages[name] = self._fresh_lineage(name, {**BASE_KNOBS, **preset})
        return {"created": now(), "lineages": lineages}

    def _fresh_lineage(self, name: str, knobs: dict) -> dict:
        return {
            "name": name,
            "run_dir": os.path.join(self.args.orch_dir, name),
            "knobs": knobs,
            "phase": "idle",          # idle | train | eval | halted | retired
            "pid": None,
            "segment": 0,
            "segment_start_steps": 0,
            "segment_target_steps": 0,
            "pending_evals": [],
            "history": [],            # one entry per finished segment
            "best_reward": -1.0,      # best offline eval (vs normal) so far
            "best_reward_at_steps": 0,
            "failures": 0,
            "best_fallback_used": False,
            "last_mix_flip_seg": 0,
        }

    def save(self) -> None:
        atomic_write_json(self.state_path, self.state)

    def lineages(self, active_only: bool = True):
        for lin in self.state["lineages"].values():
            if not active_only or lin["phase"] not in ("retired", "halted"):
                yield lin

    def run_dir(self, lin) -> str:
        return os.path.join(BASE, lin["run_dir"])

    # ------------------------------------------------------------ output

    def say(self, msg: str) -> None:
        """Console line (clears the progress bar first)."""
        sys.stdout.write("\r" + " " * self._bar_len + "\r")
        print(f"[{now()}] {msg}")
        self._bar_len = 0

    def journal(self, title: str, lines: list[str]) -> None:
        with open(self.journal_path, "a") as f:
            f.write(f"\n## {now()} — {title}\n")
            for line in lines:
                f.write(f"- {line}\n")
        self.say(title)

    def progress_bar(self) -> None:
        parts = []
        for lin in self.state["lineages"].values():
            name, phase = lin["name"], lin["phase"]
            if phase == "train":
                cur = self._train_progress(lin)
                start, target = lin["segment_start_steps"], lin["segment_target_steps"]
                span = max(1, target - start)
                frac = min(1.0, max(0.0, (cur - start) / span))
                bar = "#" * int(frac * 10)
                parts.append(f"{name} s{lin['segment']} [{bar:<10}] "
                             f"{fmt_steps(cur)}/{fmt_steps(target)}")
            elif phase == "eval":
                which = lin["pending_evals"][0] if lin["pending_evals"] else "?"
                parts.append(f"{name} s{lin['segment']} eval vs {which} "
                             f"({self.args.eval_games} games)…")
            else:
                parts.append(f"{name} {phase}")
        line = " | ".join(parts)
        pad = max(0, self._bar_len - len(line))
        sys.stdout.write("\r" + line + " " * pad)
        sys.stdout.flush()
        self._bar_len = len(line)

    # ------------------------------------------------------------ logs & metrics

    def seg_log(self, lin, kind: str) -> str:
        d = os.path.join(self.run_dir(lin), "logs")
        os.makedirs(d, exist_ok=True)
        return os.path.join(d, f"seg{lin['segment']:03d}_{kind}.log")

    def _train_progress(self, lin) -> int:
        text = tail(self.seg_log(lin, "train"), 8_000)
        vals = re.findall(r"total_timesteps\s*\|\s*(\d+)", text)
        return int(vals[-1]) if vals else lin["segment_start_steps"]

    def last_entropy(self, lin) -> float | None:
        series = load_tb_series(self.run_dir(lin))
        ent = series.get("train/entropy_loss")
        return -ent[-1].value if ent else None

    # ------------------------------------------------------------ decision engine

    def decide(self, lin) -> tuple[str | None, list[str]]:
        """Choose the next segment's resume point and knob changes.

        Returns (resume_path or None for a fresh net, reasons). May mutate
        lin["knobs"] and archive a failed run dir.
        """
        knobs, hist, reasons = lin["knobs"], lin["history"], []
        resume = newest_model(self.run_dir(lin))

        if resume is None:
            reasons.append(
                "fresh lineage: league self-play "
                f"(mix {'/'.join(map(str, knobs['league_mix']))}), absolute shaping "
                f"annealed over {fmt_steps(knobs['anneal_shaping_steps'])}, "
                f"placement terminal reward, ent-coef {knobs['ent_coef']}, "
                f"net width {knobs['net_width']}"
                + (f", curriculum from {knobs['curriculum_start']} cities"
                   if knobs["curriculum_start"] else ""))
            return None, reasons

        # Rule: no win signal at all → restart with the end-game curriculum.
        if (hist and hist[-1]["games"] >= 50 and hist[-1]["wins"] == 0
                and not knobs["curriculum_start"]):
            knobs["curriculum_start"] = 3
            attic = os.path.join(self.orch_dir, "attic",
                                 f"{lin['name']}_seg{lin['segment']}_"
                                 f"{datetime.datetime.now():%Y%m%d_%H%M%S}")
            os.makedirs(os.path.dirname(attic), exist_ok=True)
            shutil.move(self.run_dir(lin), attic)
            lin.update(history=[], best_reward=-1.0, best_reward_at_steps=0,
                       segment=0, best_fallback_used=False)
            reasons.append(
                f"eval never won a game ({hist[-1]['games']} games): the win signal is "
                f"too sparse to bootstrap from. Restarting fresh with the end-game-cities "
                f"curriculum (start 3, +2 every "
                f"{fmt_steps(knobs['curriculum_every'])}); old run archived to {attic}")
            return None, reasons

        # Rule: entropy triage (check first — free information).
        entropy = self.last_entropy(lin)
        rung = ENT_COEF_LADDER.index(knobs["ent_coef"]) \
            if knobs["ent_coef"] in ENT_COEF_LADDER else 0
        if entropy is not None and entropy < ENTROPY_COLLAPSE_NATS:
            if rung + 1 < len(ENT_COEF_LADDER):
                knobs["ent_coef"] = ENT_COEF_LADDER[rung + 1]
                reasons.append(
                    f"entropy collapsed to {entropy:.3f} nats (< {ENTROPY_COLLAPSE_NATS}): "
                    f"raising ent-coef to {knobs['ent_coef']} to destabilize the "
                    f"overconfident policy")
            else:
                reasons.append(
                    f"entropy {entropy:.3f} nats but ent-coef already at the top of the "
                    f"ladder ({knobs['ent_coef']}); leaving it")
        elif entropy is not None and entropy > ENTROPY_RECOVERED_NATS and rung > 0:
            knobs["ent_coef"] = ENT_COEF_LADDER[rung - 1]
            reasons.append(
                f"entropy recovered to {entropy:.3f} nats: stepping ent-coef back "
                f"down to {knobs['ent_coef']}")

        # Rule: plateau → flip the opponent-mix knob (with a cooldown so one
        # long plateau doesn't flip the knob back and forth every segment).
        rewards = [h["reward"] for h in hist]
        if (len(rewards) >= PLATEAU_SEGMENTS
                and max(rewards[-PLATEAU_SEGMENTS:]) - min(rewards[-PLATEAU_SEGMENTS:])
                < PLATEAU_EPS
                and rewards[-1] < NEAR_SOLVED
                and lin["segment"] - lin["last_mix_flip_seg"] >= PLATEAU_SEGMENTS):
            lin["last_mix_flip_seg"] = lin["segment"]
            mix = knobs["league_mix"]
            old = list(mix)
            if mix[2] <= 0.25:
                mix[2], mix[0] = 0.5, max(0.1, mix[0] - 0.3)
                reasons.append(
                    f"eval flat for {PLATEAU_SEGMENTS} segments (spread < {PLATEAU_EPS}): "
                    f"self-play echo chamber suspected — raising heuristic-bot share, "
                    f"league mix {old} → {mix}")
            else:
                mix[2], mix[0] = 0.1, min(0.7, mix[0] + 0.3)
                reasons.append(
                    f"eval flat for {PLATEAU_SEGMENTS} segments with a bot-heavy mix: "
                    f"flipping the knob the other way (snapshots as teacher), "
                    f"league mix {old} → {mix}")

        # Rule: regression → best_model fallback, then cross-pollination.
        if hist and hist[-1]["reward"] < lin["best_reward"] - REGRESSION_DELTA:
            best_model = os.path.join(self.run_dir(lin), "best_model")
            if not lin["best_fallback_used"] and os.path.exists(best_model + ".zip"):
                lin["best_fallback_used"] = True
                resume = best_model
                reasons.append(
                    f"regressed: last eval {hist[-1]['reward']:+.2f} is more than "
                    f"{REGRESSION_DELTA} below the lineage best {lin['best_reward']:+.2f} — "
                    f"resuming from best_model instead of the newest checkpoint")
            else:
                donor = max((o for o in self.lineages() if o["name"] != lin["name"]),
                            key=lambda o: o["best_reward"], default=None)
                if donor and donor["best_reward"] > lin["best_reward"] + 0.05:
                    donor_best = os.path.join(self.run_dir(donor), "best_model")
                    if os.path.exists(donor_best + ".zip"):
                        resume = donor_best
                        lin["best_fallback_used"] = False
                        reasons.append(
                            f"still regressed after a best_model fallback: "
                            f"cross-pollinating from {donor['name']}'s best_model "
                            f"({donor['best_reward']:+.2f}) with this lineage's knobs "
                            f"(adopts the donor's net architecture)")
        else:
            lin["best_fallback_used"] = False

        if not reasons:
            trend = (f"{rewards[-2]:+.2f} → {rewards[-1]:+.2f}" if len(rewards) >= 2
                     else f"{rewards[-1]:+.2f}" if rewards else "no evals yet")
            ent_txt = f"{entropy:.2f}" if entropy is not None else "n/a"
            reasons.append(f"healthy: eval {trend}, entropy {ent_txt} nats — "
                           f"continuing the current trajectory unchanged")
        return resume, reasons

    def check_retirement(self, lin) -> bool:
        steps = total_steps(self.run_dir(lin))
        if steps < RETIRE_STEPS:
            return False
        if steps - lin["best_reward_at_steps"] < RETIRE_STALE_STEPS:
            return False
        lin["phase"] = "retired"
        lin["pid"] = None
        old_knobs = lin["knobs"]
        name = f"{lin['name']}-r{sum(1 for n in self.state['lineages'] if n.startswith(lin['name']))}"
        knobs = {**BASE_KNOBS,
                 "net_width": 256 if old_knobs["net_width"] == 128 else 128,
                 "seed": old_knobs["seed"] + 7919,
                 "league_mix": list(reversed(old_knobs["league_mix"]))}
        self.state["lineages"][name] = self._fresh_lineage(name, knobs)
        self.journal(
            f"{lin['name']} retired, {name} spawned",
            [f"post-mortem: {fmt_steps(steps)} steps lived, best eval "
             f"{lin['best_reward']:+.2f} last improved at "
             f"{fmt_steps(lin['best_reward_at_steps'])} — flat for "
             f"{fmt_steps(steps - lin['best_reward_at_steps'])} with the full triage "
             f"already applied; local optimum judged inescapable from this trajectory",
             f"replacement {name}: net width {knobs['net_width']}, seed {knobs['seed']}, "
             f"league mix {knobs['league_mix']}"])
        return True

    # ------------------------------------------------------------ subprocesses

    def child_env(self) -> dict:
        env = dict(os.environ)
        active = max(1, sum(1 for _ in self.lineages()))
        cores = os.cpu_count() or 4
        env["OMP_NUM_THREADS"] = str(max(1, cores // active))
        return env

    def launch_train(self, lin) -> None:
        if self.check_retirement(lin):
            return
        resume, reasons = self.decide(lin)
        knobs = lin["knobs"]
        lin["segment"] += 1
        start = ckpt_steps(resume or "") or total_steps(self.run_dir(lin))
        cmd = [
            PY, "scripts/train_selfplay.py",
            "--num-players", "4",
            "--num-envs", str(self.args.envs_per_lineage),
            "--total-timesteps", str(self.args.segment_steps),
            "--run-dir", lin["run_dir"],
            "--seed", str(knobs["seed"]),
            "--net-width", str(knobs["net_width"]),
            "--ent-coef", str(knobs["ent_coef"]),
            "--league",
            "--league-past-k", str(knobs["league_past_k"]),
            "--league-mix", ",".join(str(w) for w in knobs["league_mix"]),
            "--snapshot-every", str(knobs["snapshot_every"]),
            "--anneal-shaping-steps", str(knobs["anneal_shaping_steps"]),
            "--terminal-reward", knobs["terminal_reward"],
            "--save-freq", str(self.args.save_freq),
            "--eval-freq", str(self.args.eval_freq),
            "--eval-episodes", "100",
        ]
        if resume:
            cmd += ["--resume-from", resume]
        if knobs["curriculum_start"]:
            cmd += ["--curriculum-start", str(knobs["curriculum_start"]),
                    "--curriculum-every", str(knobs["curriculum_every"])]

        log = open(self.seg_log(lin, "train"), "w")
        proc = subprocess.Popen(cmd, cwd=BASE, stdout=log, stderr=subprocess.STDOUT,
                                env=self.child_env())
        log.close()
        self.procs[lin["name"]] = proc
        lin.update(phase="train", pid=proc.pid, segment_start_steps=start,
                   segment_target_steps=start + self.args.segment_steps)
        self.journal(
            f"{lin['name']} segment {lin['segment']} launched "
            f"({fmt_steps(start)} → {fmt_steps(lin['segment_target_steps'])})",
            [f"resume: {resume or 'fresh model'}"]
            + [f"reasoning: {r}" for r in reasons]
            + [f"command: {' '.join(cmd)}"])
        self.save()

    def launch_eval(self, lin) -> None:
        difficulty = lin["pending_evals"][0]
        model = newest_model(self.run_dir(lin))
        cmd = [
            PY, "scripts/evaluate.py",
            "--model", model,
            "--games", str(self.args.eval_games),
            "--bot-difficulty", difficulty,
            "--quiet",
            "--seed", str(1_000_000 + lin["segment"]),
        ]
        log = open(self.seg_log(lin, f"eval_{difficulty}"), "w")
        proc = subprocess.Popen(cmd, cwd=BASE, stdout=log, stderr=subprocess.STDOUT,
                                env=self.child_env())
        log.close()
        self.procs[lin["name"]] = proc
        lin.update(phase="eval", pid=proc.pid)
        self.save()

    def parse_eval(self, lin, difficulty: str) -> dict | None:
        text = tail(self.seg_log(lin, f"eval_{difficulty}"), 4_000)
        m = re.search(r"win rate:\s+(\d+)/(\d+)", text)
        if not m:
            return None
        wins, games = int(m.group(1)), int(m.group(2))
        cities = re.search(r"avg cities:\s+([\d.]+)", text)
        return {
            "wins": wins,
            "games": games,
            "win_rate": wins / games,
            "reward": 2 * wins / games - 1,
            "avg_cities": float(cities.group(1)) if cities else None,
        }

    # ------------------------------------------------------------ lifecycle

    def poll_child(self, lin) -> int | None:
        """Return the exit code if the lineage's process has ended, else None."""
        proc = self.procs.get(lin["name"])
        if proc is not None and proc.pid == lin["pid"]:
            return proc.poll()
        # Not our child (previous orchestrator run): watch /proc.
        needle = "train_selfplay.py" if lin["phase"] == "train" else "evaluate.py"
        if lin["pid"] and pid_alive(lin["pid"], needle) and pid_alive(lin["pid"], lin["name"]):
            return None
        return 0  # gone; exit code unknown — treat as finished and let eval judge

    def on_train_exit(self, lin, code: int) -> None:
        lin["pid"] = None
        self.procs.pop(lin["name"], None)
        if code != 0:
            lin["failures"] += 1
            self.journal(
                f"{lin['name']} segment {lin['segment']} FAILED (exit {code}, "
                f"failure {lin['failures']}/3)",
                [f"log tail: …{tail(self.seg_log(lin, 'train'), 600).strip()[-500:]}"])
            if lin["failures"] >= 3:
                lin["phase"] = "halted"
                self.journal(f"{lin['name']} halted",
                             ["3 consecutive segment failures; needs human attention. "
                              "Fix the cause and delete the 'halted' phase in state.json "
                              "to revive."])
            else:
                lin["phase"] = "idle"
            self.save()
            return
        lin["failures"] = 0
        self.prune(lin)
        lin["pending_evals"] = ["normal"]
        if lin["best_reward"] >= HARD_EVAL_AT:
            lin["pending_evals"].append("hard")
        self.launch_eval(lin)

    def on_eval_exit(self, lin, code: int) -> None:
        lin["pid"] = None
        self.procs.pop(lin["name"], None)
        difficulty = lin["pending_evals"].pop(0)
        result = self.parse_eval(lin, difficulty)
        if result is None:
            # Crashed or was killed mid-eval (e.g. orchestrator restart): don't
            # poison the history with a fake result, just move on.
            self.journal(f"{lin['name']} segment {lin['segment']} eval vs {difficulty} "
                         f"incomplete (exit {code}); result skipped",
                         [f"log tail: …{tail(self.seg_log(lin, f'eval_{difficulty}'), 400)}"])
            if lin["pending_evals"]:
                self.launch_eval(lin)
            else:
                lin["phase"] = "idle"
                self.save()
            return
        if difficulty == "normal":
            steps = total_steps(self.run_dir(lin))
            entry = {"segment": lin["segment"], "steps": steps,
                     "difficulty": difficulty, **result}
            lin["history"].append(entry)
            if result["reward"] > lin["best_reward"]:
                lin["best_reward"] = result["reward"]
                lin["best_reward_at_steps"] = steps
                self.archive_best(lin)
        lines = [f"eval({os.path.basename(newest_model(self.run_dir(lin)) or '?')}, "
                 f"{result['games']} games vs {difficulty}): "
                 f"win rate {result['win_rate']:.1%}, mean reward {result['reward']:+.3f}"
                 + (f", avg cities {result['avg_cities']:.1f}"
                    if result["avg_cities"] else "")]
        ent = self.last_entropy(lin)
        if ent is not None:
            lines.append(f"entropy {ent:.2f} nats; lineage best {lin['best_reward']:+.3f} "
                         f"(win rate ≈ {(lin['best_reward'] + 1) / 2:.0%})")
        self.journal(f"{lin['name']} segment {lin['segment']} eval vs {difficulty} done",
                     lines)
        if lin["pending_evals"]:
            self.launch_eval(lin)
        else:
            lin["phase"] = "idle"
            self.save()

    def archive_best(self, lin) -> None:
        src = os.path.join(self.run_dir(lin), "best_model.zip")
        if not os.path.exists(src):
            src = newest_model(self.run_dir(lin)) + ".zip"
        dst_dir = os.path.join(self.orch_dir, "best", lin["name"])
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy2(src, os.path.join(dst_dir, "best_model.zip"))
        atomic_write_json(os.path.join(dst_dir, "orch_record.json"),
                          {"reward": lin["best_reward"],
                           "win_rate_vs_normal": (lin["best_reward"] + 1) / 2,
                           "steps": lin["best_reward_at_steps"],
                           "segment": lin["segment"], "recorded": now()})

    def prune(self, lin) -> None:
        """Cap disk growth: keep recent checkpoints + sparse history, thin the league."""
        run_dir = self.run_dir(lin)
        ckpts = sorted(glob.glob(os.path.join(run_dir, "ckpt_*_steps.zip")),
                       key=ckpt_steps)
        keep = set(ckpts[-4:])
        keep.update(p for p in ckpts if ckpt_steps(p) % 10_000_000 == 0)
        for p in ckpts:
            if p not in keep:
                os.remove(p)
        snaps = sorted(glob.glob(os.path.join(run_dir, "league", "snap_*.bin")),
                       key=ckpt_steps)
        if len(snaps) > 120:
            stride = len(snaps) / 100
            keep = {snaps[int(i * stride)] for i in range(100)} | {snaps[-1]}
            for p in snaps:
                if p not in keep:
                    os.remove(p)

    # ------------------------------------------------------------ main loop

    def reattach(self) -> None:
        """Reconcile state.json with reality after a restart."""
        for lin in list(self.state["lineages"].values()):
            if lin["phase"] in ("train", "eval") and lin["pid"]:
                needle = "train_selfplay.py" if lin["phase"] == "train" else "evaluate.py"
                if pid_alive(lin["pid"], needle) and pid_alive(lin["pid"], lin["name"]):
                    self.say(f"re-attached to {lin['name']} {lin['phase']} "
                             f"(pid {lin['pid']})")
                    continue
                if lin["phase"] == "eval" and lin["pending_evals"]:
                    # Let the poll loop replay/skip the interrupted eval: the
                    # log may already hold a complete result to salvage.
                    lin["pid"] = None
                    self.say(f"{lin['name']} eval process is gone; salvaging its log")
                else:
                    self.say(f"{lin['name']} {lin['phase']} process (pid {lin['pid']}) "
                             f"is gone; will resume from the newest checkpoint")
                    lin["phase"] = "idle"
                    lin["pid"] = None
                    lin["pending_evals"] = []
        self.save()

    def run(self) -> None:
        # SIGTERM behaves like Ctrl-C: state is saved, children keep running.
        def _sigterm(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGTERM, _sigterm)
        self.journal(
            "orchestrator started" + (" (--once)" if self.args.once else ""),
            [f"lineages: {', '.join(x['name'] for x in self.lineages())}",
             f"segment: {fmt_steps(self.args.segment_steps)} steps, "
             f"{self.args.envs_per_lineage} envs/lineage, "
             f"{self.args.eval_games}-game offline evals, "
             f"{os.cpu_count()} CPUs available"])
        self.reattach()
        cycles_done = {lin["name"]: False for lin in self.lineages()}
        try:
            while True:
                for lin in list(self.state["lineages"].values()):
                    if lin["phase"] == "idle":
                        if self.args.once and cycles_done.get(lin["name"]):
                            continue
                        self.launch_train(lin)
                    elif lin["phase"] in ("train", "eval"):
                        code = self.poll_child(lin)
                        if code is None:
                            continue
                        if lin["phase"] == "train":
                            self.on_train_exit(lin, code)
                        else:
                            self.on_eval_exit(lin, code)
                            if lin["phase"] == "idle":
                                cycles_done[lin["name"]] = True
                if self.args.once and all(
                        cycles_done.get(lin["name"], True) or lin["phase"] == "halted"
                        for lin in self.lineages()):
                    self.say("--once cycle complete for every lineage; exiting")
                    break
                self.progress_bar()
                time.sleep(self.args.poll_seconds)
        except KeyboardInterrupt:
            self.say("interrupted — training subprocesses keep running; "
                     "restart orchestrate.py to re-attach")
            self.journal("orchestrator stopped (SIGINT)",
                         [f"{lin['name']}: {lin['phase']}"
                          + (f" pid {lin['pid']}" if lin["pid"] else "")
                          for lin in self.lineages()])
        finally:
            self.save()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lineages", default=",".join(DEFAULT_LINEAGES),
                        help="Comma-separated lineage names to create on first run "
                             "(ignored once state.json exists). Names with presets: "
                             + ", ".join(DEFAULT_LINEAGES))
    parser.add_argument("--orch-dir", default="runs/orch",
                        help="Base directory (relative to python/) for state.json, "
                             "journal.md, and per-lineage run dirs.")
    parser.add_argument("--segment-steps", type=int, default=15_000_000,
                        help="Timesteps per training segment between adaptation points.")
    parser.add_argument("--envs-per-lineage", type=int, default=8,
                        help="Parallel envs per training subprocess (DummyVecEnv).")
    parser.add_argument("--eval-games", type=int, default=200,
                        help="Offline evaluation games per segment.")
    parser.add_argument("--save-freq", type=int, default=125_000,
                        help="Checkpoint every N steps per env (train_selfplay.py).")
    parser.add_argument("--eval-freq", type=int, default=50_000,
                        help="In-run eval every N steps per env (keeps best_model.zip).")
    parser.add_argument("--poll-seconds", type=float, default=5.0,
                        help="Progress/monitor poll interval.")
    parser.add_argument("--once", action="store_true",
                        help="Run a single segment+eval cycle per lineage, then exit "
                             "(the state remains resumable).")
    args = parser.parse_args()
    Orchestrator(args).run()


if __name__ == "__main__":
    main()
