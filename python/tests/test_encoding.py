"""Observation-encoding + native macro-mask tests.

The action encoding (mask/apply/decode) is native-only since the Phase-2 macro
rebuild; the Python action mirror was removed. These tests cover the surviving
Python observation mirror and the native macro mask.
"""
import json
import numpy as np

import powergrid_py  # type: ignore[import]
from powergrid_env.constants import OBS_SIZE, CITY_IDS, N_ACTIONS
from powergrid_env.encoding import encode_observation


def _make_started_game(num_players=2, seed=0):
    g = powergrid_py.Game(num_players, seed)
    from powergrid_env.constants import COLORS
    g.start([f"p{i}" for i in range(num_players)], COLORS[:num_players])
    return g


def test_observation_shape():
    g = _make_started_game()
    state = json.loads(g.state_json())
    actor = g.current_actor()
    obs = encode_observation(state, actor)
    assert obs.shape == (OBS_SIZE,)
    assert obs.dtype == np.float32


def test_observation_range():
    g = _make_started_game(4, seed=1)
    state = json.loads(g.state_json())
    actor = g.current_actor()
    obs = encode_observation(state, actor)
    assert np.all(obs >= -0.1), f"min value {obs.min()}"
    assert np.all(obs <= 1.1), f"max value {obs.max()}"


def test_city_ids_sorted():
    g = _make_started_game()
    ids = g.city_ids()
    assert ids == sorted(ids)
    assert ids == CITY_IDS


def test_macro_mask_shape_and_nonzero_at_start():
    g = _make_started_game(2, seed=5)
    actor = g.current_actor()
    mask = g.action_mask(actor)
    assert mask.shape == (N_ACTIONS,)
    assert mask.sum() >= 1, "there must be at least one legal macro"


def test_teacher_macro_is_legal():
    """The heuristic's macro (bot_decide_id) is always in the legal mask."""
    g = _make_started_game(4, seed=3)
    actor = g.current_actor()
    macro = g.bot_decide_id(actor, "hard")
    assert macro is not None
    mask = g.action_mask(actor)
    assert mask[macro] == 1
