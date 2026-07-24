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
    num_sims: int = 200
    cpuct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    # First-play-urgency reduction: an unvisited child is scored as its
    # parent's mean value minus this, instead of the old Q=0. With Q=0 and
    # real values averaging ~-0.5, every child looked better than any visited
    # one, so `num_sims` was spent visiting each child exactly once (near-
    # uniform visit counts, no policy signal). See plan §2.
    fpu_reduction: float = 0.2
    # Move index (within an episode) after which action selection switches
    # from temperature-1 sampling to greedy (temp=0). Rulebook games run
    # ~600+ moves, so keep exploration on for a large early window.
    temp_threshold: int = 120

    # --- Network --------------------------------------------------------------
    net_width: int = 128
    value_hidden: int = 64
    lr: float = 3e-4
    device: str = "cpu"

    # --- Training loop --------------------------------------------------------
    num_iters: int = 100
    episodes_per_iter: int = 20
    # Fixed per-iteration training budget: this many minibatches sampled
    # uniformly from the replay window (replaces whole-buffer epoch training,
    # which churned on up-to-`buffer_iters`-old stale targets and made loss
    # climb over time). `train_epochs` is retained only for `pretrain.py`,
    # which trains epoch-style over a fixed cloning dataset.
    train_batches: int = 800
    train_epochs: int = 4
    batch_size: int = 256
    # Replay window measured in *iterations*: the buffer keeps examples from
    # the most recent `buffer_iters` self-play iterations (one deque block per
    # iteration) and drops older ones, so training stays near-on-policy.
    buffer_iters: int = 16
    # Safety cap on moves per self-play game; an episode that exceeds this is
    # aborted (no training examples emitted) rather than mislabeled.
    max_moves: int = 4000

    # --- Parallelism ----------------------------------------------------------
    # Self-play episodes per iteration are farmed out to this many worker
    # processes. 1 keeps the in-process path (used by tests/smoke). The CLI
    # default is `max(1, cpu_count()//2)`.
    num_workers: int = 1

    # --- Eval / checkpoints -----------------------------------------------------
    eval_games: int = 100
    eval_bot_difficulty: str = "normal"
    # Sims used for the win-rate eval. 0 = net-only greedy play, which is the
    # artifact actually exported to the Rust Expert bot (no search at deploy
    # time) and ~100x cheaper than searched eval. Raise only to measure
    # searched strength.
    eval_num_sims: int = 0
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
    # Fraction of each iteration's episodes played as MCTS-learner vs three
    # seats driven by a *past checkpoint* of this same run (net-only,
    # masked-softmax sampling — see `selfplay.play_episode_vs_net`). Guards
    # against overfitting to heuristic-bot quirks. Falls back to pure
    # self-play while no checkpoints exist yet. `vs_bot_fraction +
    # vs_past_fraction` must be <= 1; the remainder is pure self-play.
    vs_past_fraction: float = 0.2
