"""Encoding roundtrip tests."""
import json
import numpy as np
import pytest

import powergrid_py  # type: ignore[import]
from powergrid_env.constants import OBS_SIZE, CITY_IDS
from powergrid_env.encoding import action_json_to_id, id_to_action_json, encode_observation


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


def test_action_mask_nonzero_at_start():
    g = _make_started_game(2, seed=5)
    actor = g.current_actor()
    info_json = g.legal_move_info(actor)
    info = json.loads(info_json)
    assert info["select_plant_slots"] or info["pass_auction"]


def test_id_to_action_json_pass_auction():
    g = _make_started_game()
    state = json.loads(g.state_json())
    actor = g.current_actor()
    # id 0 = PassAuction
    j = id_to_action_json(0, state, actor)
    assert json.loads(j)["type"] == "pass_auction"


def test_select_plant_roundtrip():
    """SelectPlant action survives encoding roundtrip."""
    g = _make_started_game(2, seed=0)
    state = json.loads(g.state_json())
    actor = g.current_actor()
    info = json.loads(g.legal_move_info(actor))
    slots = info.get("select_plant_slots", [])
    if not slots:
        pytest.skip("No selectable plants (round 1 all passed)")
    slot = slots[0]
    action_id = 3 + slot  # SELECT_PLANT_BASE=3
    action_json = id_to_action_json(action_id, state, actor)
    action_dict = json.loads(action_json)
    assert action_dict["type"] == "select_plant"
    # Roundtrip back
    recovered_id = action_json_to_id(action_json, state, actor)
    assert recovered_id == action_id


def test_done_buying_roundtrip():
    from powergrid_env.constants import DONE_BUYING
    g = _make_started_game(2, seed=0)
    state = json.loads(g.state_json())
    actor = g.current_actor()
    j = id_to_action_json(DONE_BUYING, state, actor)
    assert json.loads(j)["type"] == "done_buying"
    assert action_json_to_id(j, state, actor) == DONE_BUYING
