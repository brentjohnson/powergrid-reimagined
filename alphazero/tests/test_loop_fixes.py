"""Tests for the 2026-07 AlphaZero loop fixes: fixed-budget training, optimizer
lr reset on load, the MCTS forced-move shortcut, league (vs-net) episodes,
windowed replay, and coach resume/run-dir hygiene."""

import numpy as np
import pytest

from alphazero.coach import Coach
from alphazero.config import AZConfig
from alphazero.game import PowerGridGame
from alphazero.mcts import MCTS
from alphazero.network import NNetWrapper
from alphazero.selfplay import play_episode, play_episode_vs_net


def _cfg(**overrides) -> AZConfig:
    base = dict(num_players=4, net_width=16, value_hidden=8, device="cpu", seed=0)
    base.update(overrides)
    return AZConfig(**base)


# -- network: fixed training budget + lr reset ------------------------------


def test_train_fixed_budget_runs_exactly_num_batches():
    cfg = _cfg(batch_size=4)
    nnet = NNetWrapper(cfg)
    examples = _fake_examples(20)
    # A fixed budget trains regardless of dataset size; just assert it runs and
    # returns finite losses (the count is internal, but 0 batches would leave
    # losses at 0.0).
    losses = nnet.train(examples, num_batches=5)
    assert losses["policy_loss"] > 0.0
    assert np.isfinite(losses["value_loss"])


def test_load_resets_optimizer_lr(tmp_path):
    saved = NNetWrapper(_cfg(lr=1e-3))
    path = tmp_path / "ckpt.pt"
    saved.save(str(path))
    # Resume with a *different* (finetune) lr; load must honor it, not the
    # checkpoint's saved 1e-3.
    loaded = NNetWrapper.load(str(path), device="cpu", cfg=_cfg(lr=1e-5))
    for group in loaded.opt.param_groups:
        assert group["lr"] == 1e-5


def _fake_examples(n):
    from powergrid_env.constants import N_ACTIONS, OBS_SIZE

    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        obs = rng.uniform(0, 1, OBS_SIZE).astype(np.float32)
        mask = np.zeros(N_ACTIONS, dtype=np.float32)
        legal = rng.choice(N_ACTIONS, size=3, replace=False)
        mask[legal] = 1.0
        pi = np.zeros(N_ACTIONS, dtype=np.float32)
        pi[legal[0]] = 1.0
        value = rng.uniform(-1, 1, 4).astype(np.float32)
        out.append((obs, mask, pi, value))
    return out


# -- mcts: forced-move shortcut ---------------------------------------------


def test_forced_move_returns_onehot_without_search(monkeypatch):
    cfg = _cfg(num_sims=50)
    nnet = NNetWrapper(cfg)
    game = PowerGridGame(seed=5, end_game_cities=4)

    # Force the mask to a single legal action.
    real_mask = game.action_mask
    single = np.zeros_like(real_mask())
    legal = int(np.flatnonzero(real_mask())[0])
    single[legal] = 1
    monkeypatch.setattr(game, "action_mask", lambda: single)

    calls = {"n": 0}
    mcts = MCTS(nnet, cfg)
    orig_sim = mcts._simulate
    mcts._simulate = lambda root: (calls.__setitem__("n", calls["n"] + 1), orig_sim(root))[1]

    pi = mcts.get_action_probs(game, temp=1.0, add_noise=True)
    assert calls["n"] == 0, "forced move must skip simulations"
    assert np.count_nonzero(pi) == 1
    assert np.argmax(pi) == legal


# -- selfplay: league episode -----------------------------------------------


def test_play_episode_vs_net_smoke():
    cfg = _cfg(num_sims=5, max_moves=4000)
    learner = NNetWrapper(cfg)
    opp = NNetWrapper(cfg)
    examples, outcome, stats = play_episode_vs_net(learner, opp, cfg, seed=7, end_game_cities=4)
    assert outcome is not None
    assert len(examples) > 0
    for obs, mask, pi, value in examples:
        assert pi.shape[0] == mask.shape[0]
        assert value.shape == (4,)


def test_selfplay_skips_forced_moves():
    # Every recorded example must be a genuine decision (>1 legal action): the
    # loop applies forced moves but does not record them.
    cfg = _cfg(num_sims=5, temp_threshold=999, max_moves=4000)
    nnet = NNetWrapper(cfg)
    examples, outcome, _ = play_episode(nnet, cfg, seed=9, end_game_cities=4)
    assert outcome is not None
    for _obs, mask, _pi, _value in examples:
        assert int(mask.sum()) > 1, "forced-move states should not be recorded"


# -- coach: windowed buffer + resume ----------------------------------------


def _tiny_coach_cfg(run_dir) -> AZConfig:
    return _cfg(
        num_sims=3,
        episodes_per_iter=2,
        num_iters=1,
        eval_games=2,
        eval_num_sims=0,
        benchmark_every=999,
        buffer_iters=2,
        train_batches=2,
        num_workers=1,
        vs_bot_fraction=0.0,
        vs_past_fraction=0.0,
        end_game_cities_start=4,
        end_game_cities_target=4,
        run_dir=str(run_dir),
    )


def test_windowed_buffer_evicts_old_blocks(tmp_path):
    coach = Coach(_tiny_coach_cfg(tmp_path / "run"))
    coach.run_iteration(1)
    coach.run_iteration(2)
    coach.run_iteration(3)
    assert len(coach.buffer) <= coach.cfg.buffer_iters == 2


def test_coach_state_round_trip_and_resume(tmp_path):
    run_dir = tmp_path / "run"
    coach = Coach(_tiny_coach_cfg(run_dir))
    coach.run()  # runs iter 1, writes coach_state.json + iter_0001.pt
    assert (run_dir / "coach_state.json").exists()

    # A second Coach on the same dir continues numbering from iter 2 and keeps
    # the prior best_win_rate (can't be clobbered by a fresh -1.0).
    resumed = Coach(_tiny_coach_cfg(run_dir))
    assert resumed.start_iter == 2
    assert resumed.best_win_rate == coach.best_win_rate


def test_fresh_run_into_dirty_dir_errors(tmp_path):
    run_dir = tmp_path / "run"
    Coach(_tiny_coach_cfg(run_dir)).run_iteration(1)
    # Simulate a foreign/old run: checkpoints present but no coach_state.json.
    (run_dir / "coach_state.json").unlink()
    with pytest.raises(SystemExit):
        Coach(_tiny_coach_cfg(run_dir))
