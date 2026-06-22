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
