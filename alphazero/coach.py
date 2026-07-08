"""AlphaZero training loop: self-play -> train -> eval -> checkpoint, AZG-style.

Self-play episodes are farmed out to a `multiprocessing.Pool` of
`cfg.num_workers` workers (or run in-process when `num_workers <= 1`). The
replay buffer is a *window* of the most recent `cfg.buffer_iters` iterations'
examples (one block per iteration); each iteration trains a fixed budget of
`cfg.train_batches` minibatches sampled uniformly from that window, so training
stays roughly on-policy instead of churning on stale targets.

Run bookkeeping (`last_iter`, `best_win_rate`, `current_egc`) is persisted to
`<run-dir>/coach_state.json` after every iteration so a resumed run continues
its iteration numbering and can't clobber a better `best.pt` with a fresh 0%.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import random
import time
from collections import deque
from multiprocessing import get_context

from torch.utils.tensorboard import SummaryWriter

from . import arena, selfplay
from .config import AZConfig
from .network import NNetWrapper
from .selfplay import Example, play_episode, play_episode_vs_bots, play_episode_vs_net

METRICS_FIELDS = [
    "iter",
    "end_game_cities",
    "new_examples",
    "aborted_episodes",
    "buffer_size",
    "policy_loss",
    "value_loss",
    "win_rate",
    "best_win_rate",
    "is_best",
    "elapsed_s",
    # Self-play winner stats (mean over the iteration's completed episodes) —
    # describe how efficiently self-play games are ending, independent of
    # the eval/benchmark opponents.
    "sp_winner_money",
    "sp_winner_cities",
    "sp_winner_plants",
    "sp_winner_plant_eff",
    "sp_game_len",
    # Benchmark suite (arena.benchmark_suite) — only populated every
    # `cfg.benchmark_every` iterations; `None` (blank/skipped) otherwise.
    "bench_win_rate_easy",
    "bench_win_rate_normal",
    "bench_win_rate_hard",
    "agent_elo",
    "eval_finish_position",
    "eval_end_money",
    "eval_final_cities",
    "eval_plants_owned",
    "eval_plant_efficiency",
    "eval_game_len",
]

# TensorBoard tag overrides for a handful of fields, grouping related scalars
# under `eval/` and `selfplay/` in the UI. Fields not listed here keep their
# bare name as the tag (unchanged behavior for the original fields).
TB_TAGS = {
    "win_rate": "eval/win_rate",
    "best_win_rate": "eval/best_win_rate",
    "bench_win_rate_easy": "eval/win_rate_easy",
    "bench_win_rate_normal": "eval/win_rate_normal",
    "bench_win_rate_hard": "eval/win_rate_hard",
    "agent_elo": "eval/elo",
    "eval_finish_position": "eval/finish_position",
    "eval_end_money": "eval/end_money",
    "eval_final_cities": "eval/final_cities",
    "eval_plants_owned": "eval/plants_owned",
    "eval_plant_efficiency": "eval/plant_efficiency",
    "eval_game_len": "eval/game_len",
    "sp_winner_money": "selfplay/winner_money",
    "sp_winner_cities": "selfplay/winner_cities",
    "sp_winner_plants": "selfplay/winner_plants",
    "sp_winner_plant_eff": "selfplay/winner_plant_efficiency",
    "sp_game_len": "selfplay/game_len",
    "policy_loss": "loss/policy",
    "value_loss": "loss/value",
}

STATE_FILE = "coach_state.json"


class Coach:
    def __init__(self, cfg: AZConfig, resume_path: str | None = None):
        self.cfg = cfg
        os.makedirs(cfg.run_dir, exist_ok=True)
        self.metrics_path = os.path.join(cfg.run_dir, "metrics.csv")
        self.state_path = os.path.join(cfg.run_dir, STATE_FILE)
        # One deque block per iteration; the window keeps `buffer_iters` of them.
        self.buffer: deque[list[Example]] = deque(maxlen=cfg.buffer_iters)

        # Resume/continuation bookkeeping is driven by coach_state.json in the
        # run dir (independent of `resume_path`, which only controls *weight*
        # initialization and may point at a checkpoint in a different dir — e.g.
        # a behavior-cloning warm start).
        prev = self._load_state()
        if prev is not None:
            continuation = True
            self.start_iter = prev["last_iter"] + 1
            self.best_win_rate = prev["best_win_rate"]
            self.current_egc = prev["current_egc"]
        else:
            self._guard_nonempty_run_dir()
            continuation = False
            self.start_iter = 1
            self.best_win_rate = -1.0
            self.current_egc = cfg.end_game_cities_start

        # Build the network: explicit resume path wins; otherwise a continuation
        # loads the run's latest checkpoint; otherwise a fresh net.
        if resume_path:
            self.nnet = NNetWrapper.load(resume_path, device=cfg.device, cfg=cfg)
        elif continuation:
            latest = os.path.join(cfg.run_dir, f"iter_{prev['last_iter']:04d}.pt")
            if not os.path.exists(latest):
                latest = os.path.join(cfg.run_dir, "best.pt")
            self.nnet = NNetWrapper.load(latest, device=cfg.device, cfg=cfg)
        else:
            self.nnet = NNetWrapper(cfg)

        if not continuation and not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=METRICS_FIELDS).writeheader()
        # TensorBoard event files live alongside metrics.csv in the run dir, so
        # `tensorboard --logdir alphazero/runs` picks up every run as a series.
        self.tb = SummaryWriter(log_dir=os.path.join(cfg.run_dir, "tb"))

    # -- resume / run-dir hygiene ------------------------------------------------

    def _load_state(self) -> dict | None:
        if not os.path.exists(self.state_path):
            return None
        with open(self.state_path) as f:
            return json.load(f)

    def _save_state(self, last_iter: int) -> None:
        with open(self.state_path, "w") as f:
            json.dump(
                {
                    "last_iter": last_iter,
                    "best_win_rate": self.best_win_rate,
                    "current_egc": self.current_egc,
                },
                f,
            )

    def _guard_nonempty_run_dir(self) -> None:
        """A fresh run (no coach_state.json) must not scribble over a previous
        run's checkpoints/metrics — the exact accident that interleaved the old
        `runs/curriculum` segments. Refuse and point the user elsewhere."""
        clutter = glob.glob(os.path.join(self.cfg.run_dir, "iter_*.pt"))
        if clutter or os.path.exists(self.metrics_path):
            raise SystemExit(
                f"Run dir {self.cfg.run_dir!r} already contains checkpoints/metrics "
                f"but no {STATE_FILE} to continue from. Pick a new --run-dir, or "
                f"delete the directory to start over. (To continue a run started "
                f"with this version, just point --run-dir at it and it resumes.)"
            )

    # -- episode dispatch --------------------------------------------------------

    def _league_checkpoints(self) -> list[str]:
        """Past checkpoints of this run usable as league opponents: the most
        recent ~20 `iter_*.pt` plus `best.pt`. Empty on a fresh run (league
        episodes then fall back to pure self-play)."""
        paths = sorted(glob.glob(os.path.join(self.cfg.run_dir, "iter_*.pt")))[-20:]
        best = os.path.join(self.cfg.run_dir, "best.pt")
        if os.path.exists(best):
            paths.append(best)
        return paths

    def _build_tasks(self, it: int, egc: int | None) -> list[tuple]:
        """One (seed, egc, mode, opp) task per episode this iteration, mixing
        vs-bot anchor episodes, past-checkpoint league episodes, and pure
        self-play per `cfg.vs_bot_fraction` / `cfg.vs_past_fraction`."""
        n = self.cfg.episodes_per_iter
        n_vs_bot = round(n * self.cfg.vs_bot_fraction)
        n_vs_past = round(n * self.cfg.vs_past_fraction)
        n_vs_bot = min(n_vs_bot, n)
        n_vs_past = min(n_vs_past, n - n_vs_bot)

        league = self._league_checkpoints()
        rng = random.Random(self.cfg.seed + it)
        tasks: list[tuple] = []
        for ep in range(n):
            seed = self.cfg.seed + it * 10_000 + ep
            if ep < n_vs_bot:
                tasks.append((seed, egc, "vs_bots", self.cfg.vs_bot_difficulty))
            elif ep < n_vs_bot + n_vs_past and league:
                tasks.append((seed, egc, "vs_net", rng.choice(league)))
            else:
                tasks.append((seed, egc, "selfplay", None))
        return tasks

    def _run_episode_inproc(self, task: tuple, opp_cache: dict) -> tuple:
        import numpy as np

        seed, egc, mode, opp = task
        np.random.seed(seed & 0xFFFFFFFF)
        if mode == "vs_bots":
            return play_episode_vs_bots(self.nnet, self.cfg, seed, egc, opp)
        if mode == "vs_net":
            opp_nnet = opp_cache.get(opp)
            if opp_nnet is None:
                opp_nnet = NNetWrapper.load(opp, device=self.cfg.device)
                opp_nnet.net.eval()
                opp_cache[opp] = opp_nnet
            return play_episode_vs_net(self.nnet, opp_nnet, self.cfg, seed, egc)
        return play_episode(self.nnet, self.cfg, seed, egc)

    def _run_episodes(self, tasks: list[tuple]) -> list[tuple]:
        if self.cfg.num_workers <= 1:
            opp_cache: dict = {}
            return [self._run_episode_inproc(t, opp_cache) for t in tasks]
        # Snapshot the current learner weights (CPU) once; the pool initializer
        # hands them to each worker, which rebuilds the net locally.
        state_dict = {k: v.cpu() for k, v in self.nnet.net.state_dict().items()}
        ctx = get_context("spawn")
        with ctx.Pool(
            processes=self.cfg.num_workers,
            initializer=selfplay._worker_init,
            initargs=(self.cfg, state_dict),
        ) as pool:
            return pool.map(selfplay._worker_run, tasks)

    # -- one iteration -----------------------------------------------------------

    def run_iteration(self, it: int) -> dict:
        """Run one self-play -> train -> eval -> checkpoint iteration
        (global iteration number `it`). Returns a metrics dict for logging."""
        t0 = time.time()
        egc = self.current_egc

        results = self._run_episodes(self._build_tasks(it, egc))

        block: list[Example] = []
        aborted = 0
        sp_stat_totals: dict[str, float] = {}
        sp_stat_count = 0
        for examples, outcome, stats in results:
            if outcome is None:
                aborted += 1
                continue
            block.extend(examples)
            if stats is not None:
                for key, value in stats.items():
                    sp_stat_totals[key] = sp_stat_totals.get(key, 0.0) + value
                sp_stat_count += 1
        if block:
            self.buffer.append(block)

        flat = [ex for blk in self.buffer for ex in blk]
        losses = (
            self.nnet.train(flat, num_batches=self.cfg.train_batches)
            if flat
            else {"policy_loss": 0.0, "value_loss": 0.0}
        )

        win_rate = arena.net_vs_bots(
            self.nnet,
            self.cfg,
            n_games=self.cfg.eval_games,
            difficulty=self.cfg.eval_bot_difficulty,
            seed_base=self.cfg.seed + it * 99_991,
            end_game_cities=egc,
            num_sims=self.cfg.eval_num_sims,
        )

        self._maybe_advance_curriculum(win_rate, it)

        ckpt_path = os.path.join(self.cfg.run_dir, f"iter_{it:04d}.pt")
        self.nnet.save(ckpt_path)
        is_best = win_rate > self.best_win_rate
        if is_best:
            self.best_win_rate = win_rate
            self.nnet.save(os.path.join(self.cfg.run_dir, "best.pt"))
        self._save_state(it)

        # Benchmark suite (per-difficulty win rate + fixed-anchor Elo +
        # strategic eval stats) is expensive — n_games * len(difficulties)
        # extra games — so it only runs periodically. Fields stay `None`
        # (skipped by `_log_tb`, blank in the CSV) on other iterations.
        bench: dict = {}
        if it == 1 or it % self.cfg.benchmark_every == 0:
            bench = arena.benchmark_suite(
                self.nnet,
                self.cfg,
                n_games=self.cfg.eval_games,
                seed_base=self.cfg.seed + it * 7_777_777,
                num_sims=self.cfg.eval_num_sims,
                end_game_cities=egc,
            )

        # `sp_stat_totals` is keyed by the raw `metrics.game_stats` fields
        # (won, finish_position, end_money, final_cities, plants_owned,
        # plant_efficiency, game_len) for the winner of each completed
        # self-play episode this iteration.
        sp_means = (
            {k: v / sp_stat_count for k, v in sp_stat_totals.items()} if sp_stat_count else {}
        )

        return {
            "iter": it,
            "end_game_cities": egc,
            "new_examples": len(block),
            "aborted_episodes": aborted,
            "buffer_size": len(flat),
            "policy_loss": losses["policy_loss"],
            "value_loss": losses["value_loss"],
            "win_rate": win_rate,
            "best_win_rate": self.best_win_rate,
            "is_best": is_best,
            "elapsed_s": time.time() - t0,
            "sp_winner_money": sp_means.get("end_money"),
            "sp_winner_cities": sp_means.get("final_cities"),
            "sp_winner_plants": sp_means.get("plants_owned"),
            "sp_winner_plant_eff": sp_means.get("plant_efficiency"),
            "sp_game_len": sp_means.get("game_len"),
            "bench_win_rate_easy": bench.get("bench_win_rate_easy"),
            "bench_win_rate_normal": bench.get("bench_win_rate_normal"),
            "bench_win_rate_hard": bench.get("bench_win_rate_hard"),
            "agent_elo": bench.get("agent_elo"),
            "eval_finish_position": bench.get("eval_finish_position"),
            "eval_end_money": bench.get("eval_end_money"),
            "eval_final_cities": bench.get("eval_final_cities"),
            "eval_plants_owned": bench.get("eval_plants_owned"),
            "eval_plant_efficiency": bench.get("eval_plant_efficiency"),
            "eval_game_len": bench.get("eval_game_len"),
        }

    def _maybe_advance_curriculum(self, win_rate: float, it: int) -> None:
        """Advance current_egc one step when the advancement condition is met.

        Win-gated (curriculum_win_threshold > 0): advance when win_rate reaches
        the threshold at the current end_game_cities setting.
        Iter-based (threshold == 0): advance every curriculum_every iterations,
        matching the original schedule (iters curriculum_every, 2*curriculum_every, ...).
        """
        if self.current_egc is None or self.current_egc >= self.cfg.end_game_cities_target:
            return
        if self.cfg.curriculum_win_threshold > 0.0:
            should_advance = win_rate >= self.cfg.curriculum_win_threshold
        else:
            should_advance = it % self.cfg.curriculum_every == 0
        if should_advance:
            prev = self.current_egc
            self.current_egc = min(
                self.current_egc + self.cfg.end_game_cities_step,
                self.cfg.end_game_cities_target,
            )
            print(f"  curriculum: end_game_cities {prev} → {self.current_egc}")

    def run(self) -> None:
        end_iter = self.start_iter + self.cfg.num_iters
        for it in range(self.start_iter, end_iter):
            m = self.run_iteration(it)
            with open(self.metrics_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=METRICS_FIELDS).writerow(m)
            self._log_tb(m)
            print(
                f"[iter {m['iter']:4d}] end_game_cities={m['end_game_cities']!s:>4}  "
                f"examples+={m['new_examples']:5d} (buf={m['buffer_size']}, "
                f"aborted={m['aborted_episodes']})  "
                f"policy_loss={m['policy_loss']:.4f}  value_loss={m['value_loss']:.4f}  "
                f"win_rate={m['win_rate']:.1%} (best={m['best_win_rate']:.1%})  "
                f"{m['elapsed_s']:.1f}s"
            )
            if m["agent_elo"] is not None:
                print(
                    f"  benchmark: elo={m['agent_elo']:.0f}  "
                    f"win_rate easy={m['bench_win_rate_easy']:.1%} "
                    f"normal={m['bench_win_rate_normal']:.1%} "
                    f"hard={m['bench_win_rate_hard']:.1%}"
                )
        self.tb.close()

    def _log_tb(self, m: dict) -> None:
        """Mirror the numeric metrics row to TensorBoard scalars, keyed by
        iter. Most fields use their bare name as the tag; a handful are
        grouped under `eval/`, `selfplay/`, or `loss/` via `TB_TAGS`."""
        it = m["iter"]
        for field in METRICS_FIELDS:
            if field == "iter":
                continue
            value = m[field]
            if value is None:
                continue
            tag = TB_TAGS.get(field, field)
            self.tb.add_scalar(tag, float(value), it)
