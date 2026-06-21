"""
Parity tests: Rust-native obs/mask/apply_action_id must produce the same results
as the existing Python reference implementations (encode_observation, mask_from_info,
id_to_action_json + game.apply).
"""

import json
import numpy as np
import pytest

import powergrid_py
from powergrid_env.encoding import encode_observation, mask_from_info, id_to_action_json
from powergrid_env.constants import COLORS, OBS_SIZE


def make_game(num_players: int = 4, seed: int = 42) -> tuple:
    """Return (game, player_ids, state_dict) after start()."""
    game = powergrid_py.Game(num_players, seed)
    names = [f"agent_{i}" for i in range(num_players)]
    colors = COLORS[:num_players]
    game.start(names, colors)
    state = json.loads(game.state_json())
    player_ids = game.player_ids()
    return game, player_ids, state


# ---------------------------------------------------------------------------
# Observation parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_players", [2, 3, 4])
@pytest.mark.parametrize("seat", [0, 1])
def test_observation_matches_python(num_players: int, seat: int):
    if seat >= num_players:
        pytest.skip("seat out of range")
    game, player_ids, state = make_game(num_players)
    actor = player_ids[seat]

    rust_obs = np.asarray(game.observation(actor), dtype=np.float32)
    py_obs = encode_observation(json.loads(game.state_json(actor)), actor)

    np.testing.assert_array_almost_equal(
        rust_obs, py_obs, decimal=5,
        err_msg=f"observation mismatch for seat={seat}, num_players={num_players}",
    )


# ---------------------------------------------------------------------------
# Action mask parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_action_mask_matches_python(num_players: int):
    game, player_ids, state = make_game(num_players)
    actor = player_ids[0]  # first bidder in auction

    rust_mask = np.asarray(game.action_mask(actor), dtype=np.int8)

    move_info = json.loads(game.legal_move_info(actor))
    py_mask = mask_from_info(move_info, state, actor)

    np.testing.assert_array_equal(
        rust_mask, py_mask,
        err_msg=f"action mask mismatch for num_players={num_players}",
    )


# ---------------------------------------------------------------------------
# apply_action_id parity: same result as apply(id_to_action_json(...))
# ---------------------------------------------------------------------------

def test_apply_action_id_select_plant():
    game, player_ids, state = make_game(4)
    actor = player_ids[0]

    # Find a legal select_plant action.
    move_info = json.loads(game.legal_move_info(actor))
    slots = move_info.get("select_plant_slots", [])
    if not slots:
        pytest.skip("no selectable plants in auction")

    from powergrid_env.constants import SELECT_PLANT_BASE
    action_id = SELECT_PLANT_BASE + slots[0]

    # Reference: apply via JSON
    game_ref, player_ids_ref, state_ref = make_game(4)
    action_json = id_to_action_json(action_id, state_ref, player_ids_ref[0])
    game_ref.apply(player_ids_ref[0], action_json)
    ref_state = json.loads(game_ref.state_json())

    # Fast: apply via action_id
    game.apply_action_id(actor, action_id)
    fast_state = json.loads(game.state_json())

    # Market and phase should agree.
    assert fast_state["phase"] == ref_state["phase"], "phase mismatch"
    assert fast_state["market"]["actual"] == ref_state["market"]["actual"], "market mismatch"


# ---------------------------------------------------------------------------
# Frozen-opponent ("policy" difficulty) integration
# ---------------------------------------------------------------------------

def test_load_opponent_policy_rejects_garbage(random_policy_bytes):
    game, player_ids, state = make_game(2)
    with pytest.raises(ValueError):
        game.load_opponent_policy(b"not a policy")
    with pytest.raises(ValueError):  # valid header, truncated payload
        game.load_opponent_policy(random_policy_bytes[:64])


def test_policy_difficulty_requires_loaded_policy():
    game, player_ids, state = make_game(2)
    with pytest.raises(ValueError, match="load_opponent_policy"):
        game.advance_bots(player_ids[0], "policy")


def test_step_vs_policy_runs_full_game(random_policy_bytes):
    """A full game against frozen-policy opponents terminates with reward ±1."""
    rng = np.random.default_rng(seed=17)
    game = powergrid_py.Game(2, 17)
    game.start(["a", "b"], COLORS[:2])
    game.load_opponent_policy(random_policy_bytes)
    learner = game.player_ids()[0]

    terminal = game.advance_bots(learner, "policy")
    assert not terminal
    current_mask = np.asarray(game.action_mask(learner), dtype=np.uint8)

    steps = 0
    reward = 0.0
    while not terminal and steps < 10_000:
        legal = np.where(current_mask)[0]
        assert len(legal) > 0, "empty mask at non-terminal step"
        action = int(rng.choice(legal))
        obs, mask, reward, terminal, cities, powered, opp_powered = game.step_vs_bots(
            learner, action, "policy"
        )
        obs = np.asarray(obs)
        current_mask = np.asarray(mask, dtype=np.uint8)
        steps += 1

    assert terminal, f"game did not finish within {steps} steps"
    assert reward in (1.0, -1.0), f"unexpected final reward {reward}"
    assert obs.shape == (OBS_SIZE,)


def test_exported_state_dict_round_trips_into_game():
    """Bytes from a fresh MaskablePPO state_dict load into the Rust engine."""
    from sb3_contrib import MaskablePPO

    from powergrid_env import PowerGridSingleAgentEnv
    from powergrid_env.export import policy_state_dict_to_bytes

    env = PowerGridSingleAgentEnv(num_players=2, seed=1)
    model = MaskablePPO("MlpPolicy", env, device="cpu", n_steps=8, batch_size=8)
    data = policy_state_dict_to_bytes(model.policy.state_dict())
    env.close()

    game, player_ids, state = make_game(2)
    game.load_opponent_policy(data)
    game.advance_bots(player_ids[0], "policy")  # must not raise


def test_env_policy_mode_falls_back_without_snapshot(random_policy_bytes):
    """bot_difficulty="policy" uses heuristic bots until a snapshot is set."""
    from powergrid_env import PowerGridSingleAgentEnv

    env = PowerGridSingleAgentEnv(num_players=2, bot_difficulty="policy", seed=2)
    env.reset()
    assert env._episode_difficulty == "normal"

    env.set_opponent_policy(random_policy_bytes)
    env.reset()
    assert env._episode_difficulty == "policy"

    rng = np.random.default_rng(2)
    terminated = False
    steps = 0
    while not terminated and steps < 10_000:
        action = int(rng.choice(np.where(env.action_masks())[0]))
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
    assert terminated and reward in (1.0, -1.0)
    env.close()


# ---------------------------------------------------------------------------
# step_vs_bots integration
# ---------------------------------------------------------------------------

def test_step_vs_bots_runs_full_game():
    rng = np.random.default_rng(seed=11)
    game = powergrid_py.Game(4, 11)
    game.start([f"agent_{i}" for i in range(4)], COLORS[:4])
    learner = game.player_ids()[0]

    terminal = game.advance_bots(learner, "normal")
    assert not terminal
    current_mask = np.asarray(game.action_mask(learner), dtype=np.uint8)

    steps = 0
    reward = 0.0
    while not terminal and steps < 10_000:
        legal = np.where(current_mask)[0]
        assert len(legal) > 0, "empty mask at non-terminal step"
        action = int(rng.choice(legal))
        obs, mask, reward, terminal, cities, powered, opp_powered = game.step_vs_bots(
            learner, action, "normal"
        )
        obs = np.asarray(obs)
        current_mask = np.asarray(mask, dtype=np.uint8)
        steps += 1

    assert terminal, f"game did not finish within {steps} steps"
    assert reward in (1.0, -1.0), f"unexpected final reward {reward}"
    assert obs.shape == (OBS_SIZE,)


def test_step_vs_bots_obs_matches_observation():
    """Obs/mask returned by step_vs_bots must match observation/action_mask(learner)."""
    game = powergrid_py.Game(4, 23)
    game.start([f"agent_{i}" for i in range(4)], COLORS[:4])
    learner = game.player_ids()[0]
    game.advance_bots(learner, "normal")

    mask = np.asarray(game.action_mask(learner), dtype=np.uint8)
    action = int(np.where(mask)[0][0])
    obs_from_step, mask_from_step, reward, terminal, cities, powered, opp_powered = (
        game.step_vs_bots(learner, action, "normal")
    )

    if not terminal:
        obs_ref = np.asarray(game.observation(learner), dtype=np.float32)
        mask_ref = np.asarray(game.action_mask(learner), dtype=np.uint8)
        np.testing.assert_array_almost_equal(
            np.asarray(obs_from_step), obs_ref, decimal=5,
            err_msg="obs from step_vs_bots doesn't match game.observation(learner)",
        )
        np.testing.assert_array_equal(
            np.asarray(mask_from_step), mask_ref,
            err_msg="mask from step_vs_bots doesn't match game.action_mask(learner)",
        )


# ---------------------------------------------------------------------------
# Reward shaping: bonus only when the learner's powering resolves
# ---------------------------------------------------------------------------

def test_reward_shaping_only_on_powering():
    from powergrid_env import PowerGridSingleAgentEnv
    from powergrid_env.constants import (
        POWER_CITIES_BASE, DISCARD_RESOURCE_BASE, POWER_FUEL_BASE,
        N_ACTIONS, POWER_SHAPING_COEF,
    )

    env = PowerGridSingleAgentEnv(num_players=4, seed=3, reward_shaping=True)
    rng = np.random.default_rng(3)
    shaped_steps = 0
    for _ in range(3):
        obs, info = env.reset()
        terminated = False
        steps = 0
        while not terminated and steps < 20_000:
            mask = env.action_masks()
            action = int(rng.choice(np.where(mask)[0]))
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or reward == 0.0:
                continue
            shaped_steps += 1
            is_power_action = (
                POWER_CITIES_BASE <= action < DISCARD_RESOURCE_BASE
                or POWER_FUEL_BASE <= action < N_ACTIONS
            )
            assert is_power_action, (
                f"nonzero non-terminal reward {reward} on non-power action {action}"
            )
            # Relative shaping: own_powered − max_opponent_powered, so the
            # reward is a whole multiple of the coefficient but may be negative.
            k = reward / POWER_SHAPING_COEF
            assert abs(k - round(k)) < 1e-6, (
                f"shaped reward {reward} is not a whole multiple of POWER_SHAPING_COEF"
            )
    assert shaped_steps > 0, "no shaped rewards observed in 3 games"
    env.close()


