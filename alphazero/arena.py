"""Evaluate a network's win rate against the Rust heuristic bots — the same
external yardstick `python/scripts/evaluate.py` uses for the PPO stack, so
AlphaZero progress is comparable to it.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .config import AZConfig
from .game import PowerGridGame
from .mcts import MCTS
from .network import NNetWrapper


def _net_move(nnet: NNetWrapper, cfg: AZConfig, game: PowerGridGame, num_sims: int) -> int:
    if num_sims > 0:
        search_cfg = cfg if num_sims == cfg.num_sims else dataclasses.replace(cfg, num_sims=num_sims)
        pi = MCTS(nnet, search_cfg).get_action_probs(game, temp=0.0, add_noise=False)
        return int(np.argmax(pi))
    probs, _ = nnet.predict(game.observation(), game.action_mask())
    return int(np.argmax(probs))


def net_vs_bots(
    nnet: NNetWrapper,
    cfg: AZConfig,
    n_games: int = 20,
    difficulty: str = "normal",
    seed_base: int = 0,
    num_sims: int = 0,
    end_game_cities: int | None = None,
) -> float:
    """Win rate of the network — greedy network-only play by default
    (`num_sims=0`); pass `num_sims>0` to play through MCTS instead — seated
    once per game (seat 0) against `num_players - 1` heuristic Rust bots."""
    wins = 0
    for g in range(n_games):
        game = PowerGridGame(
            seed=seed_base + g, num_players=cfg.num_players, end_game_cities=end_game_cities
        )
        learner = game.player_ids()[0]
        terminal = game.advance_bots(learner, difficulty)
        while not terminal:
            action = _net_move(nnet, cfg, game, num_sims)
            game.apply(action)
            terminal = game.advance_bots(learner, difficulty)
        if game.winner() == learner:
            wins += 1
    return wins / n_games


def net_vs_net(
    nnet_a: NNetWrapper,
    nnet_b: NNetWrapper,
    cfg: AZConfig,
    n_games: int = 20,
    seed_base: int = 0,
    num_sims: int = 0,
    end_game_cities: int | None = None,
) -> float:
    """Win rate of `nnet_a` playing seat 0 against `nnet_b` on every other
    seat (AZG-style accept/reject between checkpoints)."""
    wins = 0
    for g in range(n_games):
        game = PowerGridGame(
            seed=seed_base + g, num_players=cfg.num_players, end_game_cities=end_game_cities
        )
        a_id = game.player_ids()[0]
        while not game.is_terminal():
            nnet = nnet_a if game.current_player() == a_id else nnet_b
            action = _net_move(nnet, cfg, game, num_sims)
            game.apply(action)
        if game.winner() == a_id:
            wins += 1
    return wins / n_games
