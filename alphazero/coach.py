"""AlphaZero training loop: self-play -> train -> eval -> checkpoint, AZG-style."""

from __future__ import annotations

import csv
import os
import time
from collections import deque

from torch.utils.tensorboard import SummaryWriter

from . import arena
from .config import AZConfig
from .network import NNetWrapper
from .selfplay import Example, play_episode, play_episode_vs_bots

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


class Coach:
    def __init__(self, cfg: AZConfig):
        self.cfg = cfg
        self.nnet = NNetWrapper(cfg)
        self.buffer: deque[Example] = deque(maxlen=cfg.buffer_size)
        self.best_win_rate = -1.0
        # Curriculum state: current end_game_cities trigger (None = no curriculum).
        self.current_egc: int | None = cfg.end_game_cities_start
        os.makedirs(cfg.run_dir, exist_ok=True)
        self.metrics_path = os.path.join(cfg.run_dir, "metrics.csv")
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=METRICS_FIELDS).writeheader()
        # TensorBoard event files live alongside metrics.csv in the run dir, so
        # `tensorboard --logdir alphazero/runs` picks up every run as a series.
        self.tb = SummaryWriter(log_dir=os.path.join(cfg.run_dir, "tb"))

    def run_iteration(self, it: int) -> dict:
        """Run one self-play -> train -> eval -> checkpoint iteration
        (1-indexed `it`). Returns a metrics dict for logging."""
        t0 = time.time()
        egc = self.current_egc

        n_vs_bot = round(self.cfg.episodes_per_iter * self.cfg.vs_bot_fraction)

        new_examples = 0
        aborted = 0
        sp_stat_totals: dict[str, float] = {}
        sp_stat_count = 0
        for ep in range(self.cfg.episodes_per_iter):
            seed = self.cfg.seed + it * 10_000 + ep
            if ep < n_vs_bot:
                examples, outcome, stats = play_episode_vs_bots(
                    self.nnet, self.cfg, seed, egc, self.cfg.vs_bot_difficulty
                )
            else:
                examples, outcome, stats = play_episode(self.nnet, self.cfg, seed, egc)
            if outcome is None:
                aborted += 1
                continue
            self.buffer.extend(examples)
            new_examples += len(examples)
            for key, value in stats.items():
                sp_stat_totals[key] = sp_stat_totals.get(key, 0.0) + value
            sp_stat_count += 1

        losses = (
            self.nnet.train(list(self.buffer))
            if self.buffer
            else {"policy_loss": 0.0, "value_loss": 0.0}
        )

        win_rate = arena.net_vs_bots(
            self.nnet,
            self.cfg,
            n_games=self.cfg.eval_games,
            difficulty=self.cfg.eval_bot_difficulty,
            seed_base=self.cfg.seed + it * 99_991,
            end_game_cities=egc,
            num_sims=self.cfg.num_sims,
        )

        self._maybe_advance_curriculum(win_rate, it)

        ckpt_path = os.path.join(self.cfg.run_dir, f"iter_{it:04d}.pt")
        self.nnet.save(ckpt_path)
        is_best = win_rate > self.best_win_rate
        if is_best:
            self.best_win_rate = win_rate
            self.nnet.save(os.path.join(self.cfg.run_dir, "best.pt"))

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
                num_sims=self.cfg.num_sims,
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
            "new_examples": new_examples,
            "aborted_episodes": aborted,
            "buffer_size": len(self.buffer),
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
        for it in range(1, self.cfg.num_iters + 1):
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
