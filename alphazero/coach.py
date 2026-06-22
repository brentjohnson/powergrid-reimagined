"""AlphaZero training loop: self-play -> train -> eval -> checkpoint, AZG-style."""

from __future__ import annotations

import os
import time
from collections import deque

from . import arena
from .config import AZConfig
from .network import NNetWrapper
from .selfplay import Example, curriculum_end_game_cities, play_episode


class Coach:
    def __init__(self, cfg: AZConfig):
        self.cfg = cfg
        self.nnet = NNetWrapper(cfg)
        self.buffer: deque[Example] = deque(maxlen=cfg.buffer_size)
        self.best_win_rate = -1.0
        os.makedirs(cfg.run_dir, exist_ok=True)

    def run_iteration(self, it: int) -> dict:
        """Run one self-play -> train -> eval -> checkpoint iteration
        (1-indexed `it`). Returns a metrics dict for logging."""
        t0 = time.time()
        egc = curriculum_end_game_cities(self.cfg, it)

        new_examples = 0
        aborted = 0
        for ep in range(self.cfg.episodes_per_iter):
            seed = self.cfg.seed + it * 10_000 + ep
            examples, outcome = play_episode(self.nnet, self.cfg, seed, egc)
            if outcome is None:
                aborted += 1
                continue
            self.buffer.extend(examples)
            new_examples += len(examples)

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
        )

        ckpt_path = os.path.join(self.cfg.run_dir, f"iter_{it:04d}.pt")
        self.nnet.save(ckpt_path)
        is_best = win_rate > self.best_win_rate
        if is_best:
            self.best_win_rate = win_rate
            self.nnet.save(os.path.join(self.cfg.run_dir, "best.pt"))

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
        }

    def run(self) -> None:
        for it in range(1, self.cfg.num_iters + 1):
            m = self.run_iteration(it)
            print(
                f"[iter {m['iter']:4d}] end_game_cities={m['end_game_cities']!s:>4}  "
                f"examples+={m['new_examples']:5d} (buf={m['buffer_size']}, "
                f"aborted={m['aborted_episodes']})  "
                f"policy_loss={m['policy_loss']:.4f}  value_loss={m['value_loss']:.4f}  "
                f"win_rate={m['win_rate']:.1%} (best={m['best_win_rate']:.1%})  "
                f"{m['elapsed_s']:.1f}s"
            )
