"""Sweep a checkpoint's win rate vs heuristic bots across MCTS simulation counts.

`0` sims = net-only greedy (the bare-policy deployment). `>0` = greedy MCTS at
that sim count (temp=0, no Dirichlet noise — the deployment-style search, same
as `arena.net_vs_bots`). Games are parallelized across worker processes, and
each game seed is reused across every (difficulty, sims) cell so the comparison
across sim counts is paired on identical games.

    python -m alphazero.eval_search \
        --checkpoint alphazero/runs/dagger582/iter_0060.pt \
        --games 120 --sims 0,200,400,800 --difficulty normal,hard --workers 12

Run from the repo root with the `python/` venv active (see alphazero/README.md).
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from multiprocessing import get_context

import numpy as np

from .config import AZConfig
from .game import PowerGridGame
from .mcts import MCTS
from .network import NNetWrapper

_WORKER: dict = {}


def _init(checkpoint: str, device: str) -> None:
    import torch

    torch.set_num_threads(1)  # parallelism comes from the process pool
    net = NNetWrapper.load(checkpoint, device=device)
    net.net.eval()
    _WORKER["net"] = net


def _play(task: tuple[str, int, int]) -> tuple[str, int, bool]:
    """Play one game: learner (seat 0) = the net, other seats = `difficulty`
    bots. Returns (difficulty, num_sims, learner_won)."""
    difficulty, num_sims, seed = task
    net = _WORKER["net"]
    cfg = AZConfig(num_players=4, num_sims=max(num_sims, 1))
    np.random.seed(seed & 0xFFFFFFFF)
    game = PowerGridGame(seed=seed, num_players=4)
    learner = game.player_ids()[0]
    terminal = game.advance_bots(learner, difficulty)
    while not terminal:
        if num_sims > 0:
            pi = MCTS(net, cfg).get_action_probs(game, temp=0.0, add_noise=False)
            action = int(np.argmax(pi))
        else:
            probs, _ = net.predict(game.observation(), game.action_mask())
            action = int(np.argmax(probs))
        game.apply(action)
        terminal = game.advance_bots(learner, difficulty)
    return difficulty, num_sims, game.winner() == learner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--games", type=int, default=120, help="Games per (difficulty, sims) cell.")
    parser.add_argument("--sims", default="0,200,400,800", help="Comma-separated sim counts (0 = net-only).")
    parser.add_argument("--difficulty", default="normal,hard", help="Comma-separated bot difficulties.")
    parser.add_argument("--seed-base", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    sims = [int(s) for s in args.sims.split(",")]
    diffs = [d.strip() for d in args.difficulty.split(",")]
    tasks = [
        (d, s, args.seed_base + g)
        for d in diffs
        for s in sims
        for g in range(args.games)
    ]

    wins: dict[tuple[str, int], int] = defaultdict(int)
    ctx = get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init, initargs=(args.checkpoint, args.device)) as pool:
        for d, s, won in pool.imap_unordered(_play, tasks, chunksize=1):
            wins[(d, s)] += int(won)

    print(f"checkpoint: {args.checkpoint}   games/cell: {args.games}   workers: {args.workers}")
    for d in diffs:
        for s in sims:
            w = wins[(d, s)]
            wr = w / args.games
            se = (wr * (1 - wr) / args.games) ** 0.5
            tag = "net-only" if s == 0 else f"MCTS-{s}"
            print(f"  vs {d:>6}  {tag:>10}: {wr:5.1%}  (95%CI ±{1.96 * se:.1%})")


if __name__ == "__main__":
    main()
