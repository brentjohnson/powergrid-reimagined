"""Configuration for the AlphaZero training loop."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AZConfig:
    # --- Game ---------------------------------------------------------------
    num_players: int = 4
    seed: int = 0
    # End-game-cities curriculum: start low (fast, frequent terminal signal)
    # and ramp toward the rulebook trigger (17 for 4p) every `curriculum_every`
    # iterations. `end_game_cities_start = None` disables the curriculum and
    # plays every game at the rulebook trigger from the start.
    end_game_cities_start: int | None = None
    end_game_cities_target: int = 17
    end_game_cities_step: int = 2
    curriculum_every: int = 5
    # Min win rate (vs eval bots) required to advance the curriculum one step.
    # 0.0 disables win-gating and falls back to the original iter-based schedule.
    curriculum_win_threshold: float = 0.0

    # --- MCTS -----------------------------------------------------------------
    num_sims: int = 50
    cpuct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    # Move index (within an episode) after which action selection switches
    # from temperature-1 sampling to greedy (temp=0).
    temp_threshold: int = 30

    # --- Network --------------------------------------------------------------
    net_width: int = 128
    value_hidden: int = 64
    lr: float = 1e-3
    device: str = "cpu"

    # --- Training loop --------------------------------------------------------
    num_iters: int = 100
    episodes_per_iter: int = 20
    train_epochs: int = 4
    batch_size: int = 256
    buffer_size: int = 200_000
    # Safety cap on moves per self-play game; an episode that exceeds this is
    # aborted (no training examples emitted) rather than mislabeled.
    max_moves: int = 4000

    # --- Eval / checkpoints -----------------------------------------------------
    eval_games: int = 20
    eval_bot_difficulty: str = "normal"
    run_dir: str = "alphazero/runs/default"
    # How often (in iterations) to run the full benchmark suite (win rate vs
    # each of easy/normal/hard + fixed-anchor Elo + strategic eval stats —
    # see `arena.benchmark_suite`). Always runs on iter 1. Costs
    # `eval_games * 3` extra games per run, so this doesn't run every
    # iteration by default.
    benchmark_every: int = 5

    # --- Anchor episodes --------------------------------------------------------
    # Fraction of each iteration's episodes played as MCTS-learner vs
    # `vs_bot_difficulty` heuristic bots (see `selfplay.play_episode_vs_bots`)
    # instead of pure self-play. 0 = pure self-play (the default). Raise this
    # if self-play win rate vs bots stalls or regresses — it keeps some
    # training data grounded in the competent-opponent state distribution the
    # net is actually evaluated on.
    vs_bot_fraction: float = 0.0
    vs_bot_difficulty: str = "hard"
