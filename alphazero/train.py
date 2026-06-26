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

from .coach import Coach
from .config import AZConfig
from .network import NNetWrapper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--sims", type=int, default=50)
    parser.add_argument("--net-width", type=int, default=128)
    parser.add_argument("--value-hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-epochs", type=int, default=4)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument(
        "--eval-bot-difficulty", default="normal", choices=["easy", "normal", "hard"]
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-dir", default="alphazero/runs/default")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from.")
    args = parser.parse_args()

    cfg = AZConfig(
        num_sims=args.sims,
        net_width=args.net_width,
        value_hidden=args.value_hidden,
        lr=args.lr,
        batch_size=args.batch_size,
        train_epochs=args.train_epochs,
        buffer_size=args.buffer_size,
        num_iters=args.iters,
        episodes_per_iter=args.episodes,
        eval_games=args.eval_games,
        eval_bot_difficulty=args.eval_bot_difficulty,
        end_game_cities_start=args.curriculum_start,
        end_game_cities_target=args.end_game_cities or 17,
        end_game_cities_step=args.curriculum_step,
        curriculum_every=args.curriculum_every,
        curriculum_win_threshold=args.curriculum_win_threshold,
        vs_bot_fraction=args.vs_bot_fraction,
        vs_bot_difficulty=args.vs_bot_difficulty,
        seed=args.seed,
        device=args.device,
        run_dir=args.run_dir,
    )
    if args.end_game_cities is not None and args.curriculum_start is None:
        # No ramp requested: play every game at this fixed trigger.
        cfg.end_game_cities_start = args.end_game_cities

    coach = Coach(cfg)
    if args.resume:
        # Pass `cfg` through so resuming keeps this run's hyperparameters
        # (lr, batch_size, ...) — only the checkpoint's architecture fields
        # (num_players/net_width/value_hidden) are taken from the file.
        coach.nnet = NNetWrapper.load(args.resume, device=cfg.device, cfg=cfg)
    coach.run()


if __name__ == "__main__":
    main()
