"""Tests for behavior-cloning data generation (`imitation.py`).

Regression coverage for the `build_cities`/`buy_resources` decomposition:
those two phases used to make `bot_decide_id`'s whole-action JSON match
fail almost always (the bot's batch decision never equals any single-id
decoded action), silently dropping ~90% of those turns. `generate_examples`
must now decompose both losslessly into per-unit steps (one id per city;
one id per resource unit, in the bot's priority order) plus a final Done*
step — see `imitation.py`'s module docstring — and never skip a move.
"""

import numpy as np
from powergrid_env.constants import N_ACTIONS

from alphazero.config import AZConfig
from alphazero.imitation import generate_examples
from alphazero.network import NNetWrapper


def _cfg() -> AZConfig:
    return AZConfig(num_players=4, net_width=16, value_hidden=8, device="cpu")


def test_examples_are_legal_one_hot_targets():
    examples, skipped = generate_examples(
        n_games=4, seed=1, cfg=_cfg(), difficulty="hard", end_game_cities=4
    )
    assert skipped == 0, "no move should be unrepresentable after the build_cities/buy_resources fix"
    assert len(examples) > 0

    for obs, mask, pi, value in examples:
        assert obs.shape == (obs.shape[0],)
        assert pi.shape == (N_ACTIONS,)
        assert value.shape == (4,)
        nonzero = np.flatnonzero(pi)
        assert len(nonzero) == 1, "target_pi must be one-hot"
        assert pi[nonzero[0]] == 1.0
        assert mask[nonzero[0]] == 1, "the one-hot action must be legal per its own mask"
        assert np.all(np.abs(value) <= 1.0)


def test_cloning_reduces_policy_loss():
    cfg = _cfg()
    examples, _ = generate_examples(n_games=8, seed=2, cfg=cfg, difficulty="hard", end_game_cities=4)
    assert len(examples) > 50

    nnet = NNetWrapper(cfg)
    first = nnet.train(examples)["policy_loss"]
    for _ in range(9):
        last = nnet.train(examples)["policy_loss"]

    assert last < first, f"policy CE loss should drop with repeated training: {first} -> {last}"
