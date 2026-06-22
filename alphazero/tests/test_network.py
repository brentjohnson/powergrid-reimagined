"""Tests for PGNet/NNetWrapper: output shapes, masked softmax, save/load
round-trip, and the exported state-dict key layout (must match sb3's
MaskablePPO naming so `powergrid_env.export` needs no changes)."""

import numpy as np
from powergrid_env.constants import N_ACTIONS, OBS_SIZE

from alphazero.config import AZConfig
from alphazero.network import NNetWrapper


def _cfg() -> AZConfig:
    return AZConfig(num_players=4, net_width=16, value_hidden=8, device="cpu")


def test_predict_shapes_and_masked_softmax():
    nnet = NNetWrapper(_cfg())
    obs = np.random.default_rng(0).uniform(0, 1, OBS_SIZE).astype(np.float32)
    mask = np.zeros(N_ACTIONS, dtype=np.uint8)
    mask[[0, 5, 10]] = 1

    probs, value = nnet.predict(obs, mask)
    assert probs.shape == (N_ACTIONS,)
    assert value.shape == (4,)
    assert np.all(probs[mask == 0] == 0)
    assert np.isclose(probs.sum(), 1.0, atol=1e-5)
    assert np.all(np.abs(value) <= 1.0)


def test_save_load_round_trip(tmp_path):
    cfg = _cfg()
    nnet = NNetWrapper(cfg)
    obs = np.random.default_rng(1).uniform(0, 1, OBS_SIZE).astype(np.float32)
    mask = np.ones(N_ACTIONS, dtype=np.uint8)
    before_probs, before_value = nnet.predict(obs, mask)

    path = tmp_path / "ckpt.pt"
    nnet.save(str(path))
    loaded = NNetWrapper.load(str(path), device="cpu")
    after_probs, after_value = loaded.predict(obs, mask)

    assert np.allclose(before_probs, after_probs)
    assert np.allclose(before_value, after_value)


def test_policy_state_dict_keys_match_sb3_layout():
    nnet = NNetWrapper(_cfg())
    sd = nnet.net.policy_state_dict()
    expected_keys = {
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.policy_net.0.bias",
        "mlp_extractor.policy_net.2.weight",
        "mlp_extractor.policy_net.2.bias",
        "action_net.weight",
        "action_net.bias",
    }
    assert set(sd.keys()) == expected_keys
    assert tuple(sd["mlp_extractor.policy_net.0.weight"].shape) == (16, OBS_SIZE)
    assert tuple(sd["action_net.weight"].shape) == (N_ACTIONS, 16)
