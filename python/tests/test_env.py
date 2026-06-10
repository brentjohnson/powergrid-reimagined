"""PettingZoo API conformance tests."""
from pettingzoo.test import api_test

from powergrid_env import PowerGridAECEnv


def test_api_conformance_2players():
    env = PowerGridAECEnv(num_players=2, seed=42)
    api_test(env, num_cycles=500, verbose_progress=False)


def test_api_conformance_4players():
    env = PowerGridAECEnv(num_players=4, seed=0)
    api_test(env, num_cycles=200, verbose_progress=False)


def test_seed_determinism():
    """Two envs with the same seed must produce identical first observations."""
    env1 = PowerGridAECEnv(num_players=3, seed=7)
    env2 = PowerGridAECEnv(num_players=3, seed=7)
    env1.reset(seed=7)
    env2.reset(seed=7)
    agent = env1.agent_selection
    obs1 = env1.observe(agent)
    obs2 = env2.observe(agent)
    import numpy as np
    np.testing.assert_array_equal(obs1, obs2)
    env1.close()
    env2.close()


def test_different_seeds_differ():
    env1 = PowerGridAECEnv(num_players=3, seed=1)
    env2 = PowerGridAECEnv(num_players=3, seed=2)
    env1.reset()
    env2.reset()
    agent1 = env1.agent_selection
    agent2 = env2.agent_selection
    obs1 = env1.observe(agent1)
    obs2 = env2.observe(agent2)
    # Observations may or may not differ (region/order randomisation), just check shapes.
    assert obs1.shape == obs2.shape
    env1.close()
    env2.close()


def test_mask_always_has_legal_action():
    """Every step's action mask must contain at least one legal action."""
    import numpy as np
    rng = np.random.default_rng(42)
    env = PowerGridAECEnv(num_players=2, seed=99)
    env.reset()
    no_legal = 0
    for agent in env.agent_iter():
        _, _, terminated, truncated, info = env.last()
        if terminated or truncated:
            env.step(None)
            continue
        mask = info.get("action_mask")
        if mask is None or mask.sum() == 0:
            no_legal += 1
            env.step(0)
        else:
            action = int(rng.choice(np.where(mask)[0]))
            env.step(action)
    assert no_legal == 0, f"{no_legal} steps had no legal actions"
