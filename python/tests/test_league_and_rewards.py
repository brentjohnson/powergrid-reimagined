"""League opponent pool, shaping anneal, and placement-reward tests."""

import numpy as np
import pytest
from conftest import make_policy_bytes

from powergrid_env import PowerGridSingleAgentEnv
from powergrid_env.callbacks import LeagueSnapshotCallback, ShapingAnnealCallback
from powergrid_env.constants import POWER_SHAPING_COEF
from powergrid_env.stats import learner_stats


def selfplay_env(**kwargs) -> PowerGridSingleAgentEnv:
    return PowerGridSingleAgentEnv(num_players=4, bot_difficulty="policy",
                                   seed=kwargs.pop("seed", 0), **kwargs)


# ---------------------------------------------------------------- pool


def test_pool_validation():
    env = selfplay_env()
    with pytest.raises(ValueError, match="kind"):
        env.set_opponent_pool([("snapshots", b"x", 1.0)])
    with pytest.raises(ValueError, match="difficulty"):
        env.set_opponent_pool([("bots", "brutal", 1.0)])
    with pytest.raises(ValueError, match="bytes"):
        env.set_opponent_pool([("policy", "not-bytes", 1.0)])
    with pytest.raises(ValueError, match="weights"):
        env.set_opponent_pool([("bots", "hard", 0.0)])
    env.set_opponent_pool([("bots", "easy", 1.0)])
    assert env._opponent_pool
    env.set_opponent_pool([])  # clears back to single-snapshot behaviour
    assert env._opponent_pool is None


def test_pool_bots_entry_drives_episode():
    env = selfplay_env()
    env.set_opponent_pool([("bots", "easy", 1.0)])
    env.reset()
    assert env._episode_difficulty == "easy"


def test_pool_policy_entry_loads_snapshot(random_policy_bytes):
    env = selfplay_env()
    env.set_opponent_pool([("policy", random_policy_bytes, 1.0)])
    obs, info = env.reset()
    assert env._episode_difficulty == "policy"
    # The episode must actually be playable against the loaded snapshot.
    mask = info["action_mask"]
    env.step(int(np.flatnonzero(mask)[0]))


def test_pool_sampling_respects_weights():
    env = selfplay_env(seed=123)
    env.set_opponent_pool([("bots", "easy", 0.8), ("bots", "hard", 0.2)])
    picks = []
    for _ in range(200):
        env.reset()
        picks.append(env._episode_difficulty)
    easy_ratio = picks.count("easy") / len(picks)
    assert 0.65 < easy_ratio < 0.95, easy_ratio


def test_pool_overrides_single_snapshot_and_bot_mix(random_policy_bytes):
    # bot_mix=1.0 would force every episode to heuristic "hard"; the pool wins.
    env = selfplay_env(bot_mix=1.0)
    env.set_opponent_policy(random_policy_bytes)
    env.set_opponent_pool([("policy", random_policy_bytes, 1.0)])
    env.reset()
    assert env._episode_difficulty == "policy"


# ---------------------------------------------------------------- shaping


class _StubGame:
    """Fixed step_vs_bots result: a non-terminal step where the learner
    powered 4 cities and the best opponent powered 1."""

    def __init__(self, obs_size, n_actions):
        self._obs = np.zeros(obs_size, dtype=np.float32)
        self._mask = np.ones(n_actions, dtype=np.uint8)

    def step_vs_bots(self, learner, action, difficulty):
        return self._obs, self._mask, 0.0, False, 5, 4, 1


def _stubbed_step_reward(env) -> float:
    from powergrid_env.constants import N_ACTIONS, OBS_SIZE

    env.reset()
    env.game = _StubGame(OBS_SIZE, N_ACTIONS)
    _, reward, _, _, _ = env.step(0)
    return reward


@pytest.mark.parametrize("scale", [1.0, 0.5, 0.0])
def test_shaping_scale_applied_absolute(scale):
    env = selfplay_env(reward_shaping=True, shaping_mode="absolute")
    env.set_shaping_scale(scale)
    assert _stubbed_step_reward(env) == pytest.approx(4 * POWER_SHAPING_COEF * scale)


def test_shaping_scale_applied_relative():
    env = selfplay_env(reward_shaping=True, shaping_mode="relative")
    env.set_shaping_scale(0.5)
    assert _stubbed_step_reward(env) == pytest.approx((4 - 1) * POWER_SHAPING_COEF * 0.5)


def test_shaping_scale_validation():
    env = selfplay_env()
    with pytest.raises(ValueError):
        env.set_shaping_scale(1.5)


def test_anneal_schedule():
    cb = ShapingAnnealCallback(train_env=None, anneal_steps=1_000_000)
    assert cb._scale_for(0) == 1.0
    assert cb._scale_for(500_000) == pytest.approx(0.5)
    assert cb._scale_for(1_000_000) == 0.0
    assert cb._scale_for(9_999_999) == 0.0


# ---------------------------------------------------------------- placement


def _play_to_terminal(env, max_steps=4000):
    obs, info = env.reset()
    rng = np.random.default_rng(0)
    for _ in range(max_steps):
        legal = np.flatnonzero(env.action_masks())
        obs, reward, terminated, _, info = env.step(int(rng.choice(legal)))
        if terminated:
            return reward
    pytest.skip("game did not finish under random play")


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_placement_reward_matches_rank(seed):
    import json

    env = PowerGridSingleAgentEnv(num_players=4, bot_difficulty="normal", seed=seed,
                                  end_game_cities=3, terminal_reward="placement")
    reward = _play_to_terminal(env)
    state = json.loads(env.game.state_json(env._learner_id))
    rank = learner_stats(state, env._learner_id, env.game.winner())["rank"]
    assert reward == pytest.approx(1.0 - 2.0 * (rank - 1) / 3)
    if env.game.winner() == env._learner_id:
        assert reward == pytest.approx(1.0)


def test_winloss_default_unchanged():
    env = PowerGridSingleAgentEnv(num_players=4, bot_difficulty="normal", seed=5,
                                  end_game_cities=3)
    reward = _play_to_terminal(env)
    assert reward in (1.0, -1.0)


# ---------------------------------------------------------------- league callback


def test_league_build_pool(tmp_path):
    rng = np.random.default_rng(0)
    league = tmp_path / "league"
    league.mkdir()
    for steps in (100, 200, 300, 400, 500):
        (league / f"snap_{steps}.bin").write_bytes(make_policy_bytes(rng))

    cb = LeagueSnapshotCallback(train_env=None, snapshot_every=1000,
                                league_dir=str(league), past_k=2,
                                mix=(0.5, 0.3, 0.2), seed=0)
    cb._scan_league()
    assert [s for s, _ in cb._snapshots] == [100, 200, 300, 400, 500]

    pool = cb._build_pool()
    kinds = [k for k, _, _ in pool]
    assert kinds.count("policy") == 3  # latest + past_k
    assert kinds.count("bots") == 1
    assert sum(w for _, _, w in pool) == pytest.approx(1.0)
    latest = (league / "snap_500.bin").read_bytes()
    assert pool[0] == ("policy", latest, 0.5)
    # The env must accept exactly what the callback builds.
    env = selfplay_env()
    env.set_opponent_pool(pool)
    env.reset()


def test_league_build_pool_single_snapshot(tmp_path):
    league = tmp_path / "league"
    league.mkdir()
    (league / "snap_100.bin").write_bytes(make_policy_bytes(np.random.default_rng(1)))
    cb = LeagueSnapshotCallback(train_env=None, snapshot_every=1000,
                                league_dir=str(league), past_k=4, seed=0)
    cb._scan_league()
    pool = cb._build_pool()
    assert [k for k, _, _ in pool] == ["policy", "bots"]  # no past snapshots yet


def test_league_build_pool_skips_corrupt_latest(tmp_path):
    # A power outage left a zero-byte snapshot whose filename step (500) is
    # newer than every good one AND newer than the last checkpoint. LATEST must
    # fall back to the newest *readable* snapshot instead of handing the empty
    # blob to load_opponent_policy (which raises BadMagic). Regression for the
    # wave-13 resume failure of w3-gamma995 / w5-y3-gamma.
    rng = np.random.default_rng(0)
    league = tmp_path / "league"
    league.mkdir()
    good = {}
    for steps in (100, 200, 300, 400):
        good[steps] = make_policy_bytes(rng)
        (league / f"snap_{steps}.bin").write_bytes(good[steps])
    (league / "snap_500.bin").write_bytes(b"")  # truncated mid-write

    cb = LeagueSnapshotCallback(train_env=None, snapshot_every=1000,
                                league_dir=str(league), past_k=2,
                                mix=(0.5, 0.3, 0.2), seed=0)
    cb._scan_league()
    pool = cb._build_pool()
    # LATEST is the newest readable snapshot (400), not the corrupt 500.
    assert pool[0] == ("policy", good[400], 0.5)
    # The corrupt file never enters PAST, and every entry is loadable.
    env = selfplay_env()
    env.set_opponent_pool(pool)
    env.reset()


def test_league_peer_pool(tmp_path):
    rng = np.random.default_rng(0)
    league = tmp_path / "league"
    league.mkdir()
    (league / "snap_100.bin").write_bytes(make_policy_bytes(rng))
    peer = tmp_path / "peer" / "league"
    peer.mkdir(parents=True)
    peer_blob = make_policy_bytes(rng)
    (peer / "snap_900.bin").write_bytes(peer_blob)

    cb = LeagueSnapshotCallback(
        train_env=None, snapshot_every=1000, league_dir=str(league), past_k=4,
        mix=(0.5, 0.3, 0.2), seed=0,
        peer_dirs=[str(peer), str(tmp_path / "not-launched-yet" / "league"),
                   str(league)],  # own dir must be ignored, missing dir tolerated
    )
    assert cb.peer_dirs == [str(peer), str(tmp_path / "not-launched-yet" / "league")]
    cb._scan_league()
    assert cb._peer_snapshots == [str(peer / "snap_900.bin")]

    # No own past, but the peer snapshot fills the PAST share.
    pool = cb._build_pool()
    assert [k for k, _, _ in pool] == ["policy", "policy", "bots"]
    assert pool[1] == ("policy", peer_blob, 0.3)
    env = selfplay_env()
    env.set_opponent_pool(pool)
    env.reset()


def test_league_peer_pool_skips_invalid_snapshot(tmp_path):
    rng = np.random.default_rng(0)
    league = tmp_path / "league"
    league.mkdir()
    for steps in (100, 200):
        (league / f"snap_{steps}.bin").write_bytes(make_policy_bytes(rng))
    peer = tmp_path / "peer-league"
    peer.mkdir()
    (peer / "snap_500.bin").write_bytes(b"NOTAPOLICY-half-synced-garbage")

    cb = LeagueSnapshotCallback(train_env=None, snapshot_every=1000,
                                league_dir=str(league), past_k=8,
                                mix=(0.5, 0.3, 0.2), seed=0,
                                peer_dirs=[str(peer)])
    cb._scan_league()
    pool = cb._build_pool()
    # The bad peer file is dropped; the valid own-past snapshot carries the
    # full PAST weight and every entry is loadable.
    assert [k for k, _, _ in pool] == ["policy", "policy", "bots"]
    assert pool[1][2] == pytest.approx(0.3)
    env = selfplay_env()
    env.set_opponent_pool(pool)
    env.reset()
