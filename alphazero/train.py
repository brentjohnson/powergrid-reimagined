"""CLI entry point for AlphaZero training.

Must be run as a module (relative imports inside the package) from the repo
root, using the `python/` venv that has `powergrid_py`/`powergrid_env`/torch
installed:

    cd /path/to/powergrid-reimagined
    python/.venv/bin/python -m alphazero.train --iters 100 --episodes 20 --sims 50

See `alphazero/README.md` for the full runbook.
"""

from __future__ import annotations

import argparse
import os

from .coach import Coach
from .config import AZConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=100, help="Iterations to run THIS invocation (a resume continues numbering).")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Self-play worker processes (1 = in-process; default cpu_count//2).",
    )
    parser.add_argument("--net-width", type=int, default=128)
    parser.add_argument("--value-hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--train-batches",
        type=int,
        default=800,
        help="Minibatches trained per iteration, sampled from the replay window.",
    )
    parser.add_argument(
        "--buffer-iters",
        type=int,
        default=16,
        help="Replay window size in iterations (older examples are dropped).",
    )
    parser.add_argument(
        "--fpu",
        type=float,
        default=0.2,
        help="First-play-urgency reduction for unvisited MCTS children.",
    )
    parser.add_argument(
        "--temp-threshold",
        type=int,
        default=120,
        help="Move index after which self-play switches from sampling to greedy.",
    )
    parser.add_argument(
        "--dirichlet-eps",
        type=float,
        default=0.25,
        help="Weight of root Dirichlet exploration noise mixed into the priors "
        "during self-play. Lower it (e.g. 0.1) when finetuning a competent net "
        "so search doesn't flatten an already-sharp policy.",
    )
    parser.add_argument(
        "--dirichlet-alpha",
        type=float,
        default=0.3,
        help="Concentration of the root Dirichlet noise.",
    )
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument(
        "--eval-bot-difficulty", default="normal", choices=["easy", "normal", "hard"]
    )
    parser.add_argument(
        "--eval-num-sims",
        type=int,
        default=0,
        help="MCTS sims for eval. 0 = net-only greedy (the exported artifact).",
    )
    parser.add_argument(
        "--benchmark-every",
        type=int,
        default=5,
        help="Run the full benchmark suite (win rate vs easy/normal/hard + "
        "fixed-anchor Elo + strategic eval stats) every N iterations "
        "(always runs on iter 1). Costs eval_games*3 extra games per run.",
    )
    parser.add_argument(
        "--end-game-cities",
        type=int,
        default=None,
        help="Fixed end-game trigger for every game (overrides the curriculum).",
    )
    parser.add_argument(
        "--curriculum-start",
        type=int,
        default=None,
        help="Start end_game_cities low and ramp to the rulebook trigger (17 for 4p).",
    )
    parser.add_argument("--curriculum-every", type=int, default=5)
    parser.add_argument("--curriculum-step", type=int, default=2)
    parser.add_argument(
        "--curriculum-win-threshold",
        type=float,
        default=0.0,
        help="Min win rate against eval bots required to advance the curriculum. "
        "0 uses the original iter-based schedule.",
    )
    parser.add_argument(
        "--vs-bot-fraction",
        type=float,
        default=0.0,
        help="Fraction of each iteration's episodes played as MCTS-learner vs "
        "--vs-bot-difficulty heuristic bots instead of pure self-play. Raise "
        "this if self-play win rate vs bots stalls or regresses.",
    )
    parser.add_argument(
        "--vs-bot-difficulty", default="hard", choices=["easy", "normal", "hard"]
    )
    parser.add_argument(
        "--vs-past-fraction",
        type=float,
        default=0.2,
        help="Fraction of each iteration's episodes played as MCTS-learner vs "
        "three seats driven by a past checkpoint of this run (net-only "
        "sampling). Falls back to self-play until checkpoints exist.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-dir", default="alphazero/runs/default")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
    args = parser.parse_args()

    if args.vs_bot_fraction + args.vs_past_fraction > 1.0 + 1e-9:
        parser.error("--vs-bot-fraction + --vs-past-fraction must be <= 1.0")

    cfg = AZConfig(
        num_sims=args.sims,
        fpu_reduction=args.fpu,
        temp_threshold=args.temp_threshold,
        dirichlet_eps=args.dirichlet_eps,
        dirichlet_alpha=args.dirichlet_alpha,
        num_workers=args.workers,
        net_width=args.net_width,
        value_hidden=args.value_hidden,
        lr=args.lr,
        batch_size=args.batch_size,
        train_batches=args.train_batches,
        buffer_iters=args.buffer_iters,
        num_iters=args.iters,
        episodes_per_iter=args.episodes,
        eval_games=args.eval_games,
        eval_bot_difficulty=args.eval_bot_difficulty,
        eval_num_sims=args.eval_num_sims,
        benchmark_every=args.benchmark_every,
        end_game_cities_start=args.curriculum_start,
        end_game_cities_target=args.end_game_cities or 17,
        end_game_cities_step=args.curriculum_step,
        curriculum_every=args.curriculum_every,
        curriculum_win_threshold=args.curriculum_win_threshold,
        vs_bot_fraction=args.vs_bot_fraction,
        vs_bot_difficulty=args.vs_bot_difficulty,
        vs_past_fraction=args.vs_past_fraction,
        seed=args.seed,
        device=args.device,
        run_dir=args.run_dir,
    )
    if args.end_game_cities is not None and args.curriculum_start is None:
        # No ramp requested: play every game at this fixed trigger.
        cfg.end_game_cities_start = args.end_game_cities

    # `resume_path` only initializes weights (it may point at a checkpoint in a
    # different dir, e.g. a behavior-cloning warm start). Whether this run
    # *continues* an existing run dir's iteration numbering is decided by the
    # presence of coach_state.json in --run-dir (see Coach.__init__).
    Coach(cfg, resume_path=args.resume).run()


if __name__ == "__main__":
    main()
