"""Self-play episode generation.

Every seat in a self-play game is driven by the *same* shared network through
MCTS — observations are already perspective-relative (self first, then
opponents in join order; see `encoding.rs::build_observation`), so one net
can legitimately play every seat without seeing anything seat-specific.
"""

from __future__ import annotations

import numpy as np

from .config import AZConfig
from .game import PowerGridGame, to_relative_vector
from .mcts import MCTS
from .network import NNetWrapper

Example = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]  # obs, mask, pi, value


def curriculum_end_game_cities(cfg: AZConfig, iteration: int) -> int | None:
    """End-game-cities trigger for self-play games at this training
    iteration (1-indexed), per `cfg`'s curriculum schedule. `None` means
    "use the rulebook default" (no override)."""
    if cfg.end_game_cities_start is None:
        return None
    bumps = (iteration - 1) // cfg.curriculum_every
    value = cfg.end_game_cities_start + bumps * cfg.end_game_cities_step
    return min(value, cfg.end_game_cities_target)


def play_episode(
    nnet: NNetWrapper, cfg: AZConfig, seed: int, end_game_cities: int | None
) -> tuple[list[Example], dict[str, float] | None]:
    """Play one full self-play game. Returns `(examples, outcome)`. If the
    game exceeds `cfg.max_moves` without finishing, the episode is aborted —
    `examples` is empty and `outcome` is `None` — rather than mislabeling
    unterminated training data."""
    game = PowerGridGame(
        seed=seed, num_players=cfg.num_players, end_game_cities=end_game_cities
    )
    mcts = MCTS(nnet, cfg)
    history: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []

    move_idx = 0
    while not game.is_terminal():
        if move_idx >= cfg.max_moves:
            return [], None
        temp = 1.0 if move_idx < cfg.temp_threshold else 0.0
        pi = mcts.get_action_probs(game, temp=temp, add_noise=True)
        history.append((game.observation(), game.action_mask(), pi, game.current_player()))

        pi64 = pi.astype(np.float64)
        pi64 /= pi64.sum()
        action = int(np.random.choice(len(pi64), p=pi64))
        game.apply(action)
        move_idx += 1

    outcome = game.outcome()
    player_ids = game.player_ids()
    examples = [
        (obs, mask, pi, to_relative_vector(player_ids, to_move, outcome))
        for obs, mask, pi, to_move in history
    ]
    return examples, outcome


def play_episode_vs_bots(
    nnet: NNetWrapper,
    cfg: AZConfig,
    seed: int,
    end_game_cities: int | None,
    difficulty: str,
) -> tuple[list[Example], dict[str, float] | None]:
    """Like `play_episode`, but only one seat (the learner) is driven by
    MCTS; the rest are driven by the Rust `difficulty` heuristic bot via
    `advance_bots`. Only the learner's turns are recorded as examples.

    Pure self-play can drift: the net only ever sees states its own (possibly
    still-weak) play reaches, which can diverge from the competent-opponent
    states it's actually evaluated on (`arena.net_vs_bots`). Mixing in some
    of these anchor episodes (see `--vs-bot-fraction` in `train.py`) keeps a
    fraction of training data grounded in that distribution.
    """
    game = PowerGridGame(seed=seed, num_players=cfg.num_players, end_game_cities=end_game_cities)
    learner = game.player_ids()[0]
    mcts = MCTS(nnet, cfg)
    history: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []

    if game.advance_bots(learner, difficulty):
        return [], None

    move_idx = 0
    while not game.is_terminal():
        if move_idx >= cfg.max_moves:
            return [], None
        temp = 1.0 if move_idx < cfg.temp_threshold else 0.0
        pi = mcts.get_action_probs(game, temp=temp, add_noise=True)
        history.append((game.observation(), game.action_mask(), pi, game.current_player()))

        pi64 = pi.astype(np.float64)
        pi64 /= pi64.sum()
        action = int(np.random.choice(len(pi64), p=pi64))
        game.apply(action)
        move_idx += 1
        game.advance_bots(learner, difficulty)

    outcome = game.outcome()
    player_ids = game.player_ids()
    examples = [
        (obs, mask, pi, to_relative_vector(player_ids, to_move, outcome))
        for obs, mask, pi, to_move in history
    ]
    return examples, outcome
