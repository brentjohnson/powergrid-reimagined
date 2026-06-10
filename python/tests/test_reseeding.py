"""
Per-episode reseeding: consecutive resets must play different games (different
deck shuffle / regions), while the same constructor seed reproduces the same
sequence of games.
"""
import numpy as np

from powergrid_env import PowerGridSelfPlayEnv, PowerGridSingleAgentEnv


def test_selfplay_consecutive_resets_differ():
    env = PowerGridSelfPlayEnv(num_players=4, seed=3)
    obs_list = [env.reset()[0] for _ in range(4)]
    assert any(
        not np.array_equal(obs_list[0], o) for o in obs_list[1:]
    ), "every episode replayed the identical game setup"
    env.close()


def test_selfplay_seed_sequence_reproducible():
    env1 = PowerGridSelfPlayEnv(num_players=4, seed=3)
    env2 = PowerGridSelfPlayEnv(num_players=4, seed=3)
    for _ in range(3):
        o1, _ = env1.reset()
        o2, _ = env2.reset()
        np.testing.assert_array_equal(o1, o2)
    env1.close()
    env2.close()


def test_single_agent_consecutive_resets_differ():
    env = PowerGridSingleAgentEnv(num_players=4, seed=3)
    obs_list = [env.reset()[0] for _ in range(4)]
    assert any(
        not np.array_equal(obs_list[0], o) for o in obs_list[1:]
    ), "every episode replayed the identical game setup"
    env.close()


def test_single_agent_seed_sequence_reproducible():
    env1 = PowerGridSingleAgentEnv(num_players=4, seed=3)
    env2 = PowerGridSingleAgentEnv(num_players=4, seed=3)
    for _ in range(3):
        o1, _ = env1.reset()
        o2, _ = env2.reset()
        np.testing.assert_array_equal(o1, o2)
    env1.close()
    env2.close()
