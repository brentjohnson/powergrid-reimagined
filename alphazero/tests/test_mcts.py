"""Tests for the multiplayer masked PUCT MCTS: the mask is respected, greedy
(temp=0) selection is a clean one-hot, and search is reproducible given a
fixed network and no Dirichlet noise."""

import numpy as np

from alphazero.config import AZConfig
from alphazero.game import PowerGridGame
from alphazero.mcts import MCTS
from alphazero.network import NNetWrapper


def _cfg(**overrides) -> AZConfig:
    base = dict(num_sims=20, net_width=16, value_hidden=8, device="cpu")
    base.update(overrides)
    return AZConfig(**base)


def test_action_probs_only_on_legal_actions():
    cfg = _cfg()
    nnet = NNetWrapper(cfg)
    game = PowerGridGame(seed=10, end_game_cities=4)
    pi = MCTS(nnet, cfg).get_action_probs(game, temp=1.0)

    mask = game.action_mask()
    assert np.all(pi[mask == 0] == 0)
    assert np.isclose(pi.sum(), 1.0, atol=1e-4)


def test_temp_zero_is_greedy_onehot():
    cfg = _cfg()
    nnet = NNetWrapper(cfg)
    game = PowerGridGame(seed=30, end_game_cities=4)
    pi = MCTS(nnet, cfg).get_action_probs(game, temp=0.0, add_noise=False)

    assert np.isclose(pi.sum(), 1.0)
    assert np.count_nonzero(pi) == 1
    assert game.action_mask()[np.argmax(pi)] == 1


def test_deterministic_given_fixed_network_and_no_noise():
    cfg = _cfg(num_sims=30)
    nnet = NNetWrapper(cfg)  # fixed (random-init) weights, reused for both forks

    game_a = PowerGridGame(seed=20, end_game_cities=4)
    game_b = game_a.fork()

    pi_a = MCTS(nnet, cfg).get_action_probs(game_a, temp=1.0, add_noise=False)
    pi_b = MCTS(nnet, cfg).get_action_probs(game_b, temp=1.0, add_noise=False)
    assert np.allclose(pi_a, pi_b)
