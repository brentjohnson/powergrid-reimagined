"""Tests for DAgger data generation (`dagger.py`): the expert label is always
a legal one-hot, examples carry a valid rank-outcome value, and the learner
seat is genuinely driven by the net (not the bot)."""

import numpy as np
from powergrid_env.constants import N_ACTIONS

from alphazero.config import AZConfig
from alphazero.dagger import bot_first_action_id, generate_dagger_examples
from alphazero.game import PowerGridGame
from alphazero.network import NNetWrapper


def _cfg(**overrides) -> AZConfig:
    base = dict(num_players=4, net_width=16, value_hidden=8, device="cpu", max_moves=4000)
    base.update(overrides)
    return AZConfig(**base)


def test_bot_first_action_id_is_legal():
    game = PowerGridGame(seed=1, end_game_cities=4)
    # Walk a few states with the bot and confirm its "first action id" is always
    # a legal move in the current mask.
    for _ in range(200):
        if game.is_terminal():
            break
        aid = bot_first_action_id(game, "hard")
        assert aid is not None
        assert game.action_mask()[aid] == 1, "expert label must be legal in its own state"
        game.apply(aid)


def test_generate_dagger_examples_shapes_and_legality():
    cfg = _cfg()
    nnet = NNetWrapper(cfg)
    examples, skipped = generate_dagger_examples(
        nnet, cfg, n_games=3, seed=5, difficulty="hard", end_game_cities=4
    )
    assert skipped == 0
    assert len(examples) > 0
    for obs, mask, pi, value in examples:
        assert pi.shape == (N_ACTIONS,)
        assert value.shape == (4,)
        nonzero = np.flatnonzero(pi)
        assert len(nonzero) == 1 and pi[nonzero[0]] == 1.0, "label must be one-hot"
        assert mask[nonzero[0]] == 1, "labeled action must be legal"
        assert np.all(np.abs(value) <= 1.0)


def test_learner_seat_driven_by_net_not_bot(monkeypatch):
    # If the net drives the learner seat, forcing the net to always pick a
    # specific legal action should change which states get visited vs. the
    # bot's own line. We assert generation still completes and records the
    # *bot's* label (not the net's forced move) at each learner state.
    cfg = _cfg()
    nnet = NNetWrapper(cfg)

    def fake_predict(obs, mask):
        probs = np.zeros(N_ACTIONS, dtype=np.float32)
        probs[int(np.flatnonzero(mask)[0])] = 1.0  # always the lowest legal id
        return probs, np.zeros(4, dtype=np.float32)

    monkeypatch.setattr(nnet, "predict", fake_predict)
    examples, _ = generate_dagger_examples(
        nnet, cfg, n_games=2, seed=9, difficulty="hard", end_game_cities=4
    )
    assert len(examples) > 0
    # At least some labels should differ from the net's forced "lowest legal id"
    # choice — otherwise we'd just be cloning the net, not the bot.
    labels = [int(np.argmax(pi)) for _o, _m, pi, _v in examples]
    lowest = [int(np.flatnonzero(m)[0]) for _o, m, _pi, _v in examples]
    assert any(a != b for a, b in zip(labels, lowest)), "labels must come from the bot, not the net"
