"""Self-play episode generation.

Three episode flavors share one MCTS loop (`_mcts_episode`), differing only in
how the non-learner seats are advanced between the learner's moves:

- `play_episode` — pure self-play: every seat is the *same* shared network
  through MCTS (observations are already perspective-relative — self first,
  then opponents in join order; see `encoding.rs::build_observation` — so one
  net can legitimately play every seat). Every non-forced move is recorded.
- `play_episode_vs_bots` — only the learner (seat 0) is MCTS; the rest are the
  Rust `difficulty` heuristic bot. Grounds training data in the competent-
  opponent distribution the net is evaluated against.
- `play_episode_vs_net` — only the learner is MCTS; the rest are a *past
  checkpoint* of this run playing net-only (masked-softmax sampling). Guards
  against overfitting to heuristic-bot quirks.

Forced moves (a single legal action) are applied but never recorded — search
can't change the decision, so they carry no policy signal.

Worker entry points (`_worker_init`/`_worker_run`) let `coach.py` farm episodes
out to a `multiprocessing.Pool`; they are module-level and picklable.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from . import metrics
from .config import AZConfig
from .game import PowerGridGame, to_relative_vector
from .mcts import MCTS
from .network import NNetWrapper

Example = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]  # obs, mask, pi, value
EpisodeResult = tuple[list[Example], dict[str, float] | None, dict | None]


def curriculum_end_game_cities(cfg: AZConfig, iteration: int) -> int | None:
    """End-game-cities trigger for self-play games at this training
    iteration (1-indexed), per `cfg`'s curriculum schedule. `None` means
    "use the rulebook default" (no override)."""
    if cfg.end_game_cities_start is None:
        return None
    bumps = (iteration - 1) // cfg.curriculum_every
    value = cfg.end_game_cities_start + bumps * cfg.end_game_cities_step
    return min(value, cfg.end_game_cities_target)


def _sample_action(pi: np.ndarray) -> int:
    """Sample an action index from a (masked) probability vector, renormalizing
    defensively."""
    p = pi.astype(np.float64)
    total = p.sum()
    if total <= 0:
        # Degenerate distribution (shouldn't happen for a legal position);
        # pick the argmax so we still make a move.
        return int(np.argmax(pi))
    return int(np.random.choice(len(p), p=p / total))


def _mcts_episode(
    game: PowerGridGame,
    mcts: MCTS,
    cfg: AZConfig,
    advance: Callable[[PowerGridGame], None],
) -> EpisodeResult:
    """Run one episode where MCTS plays whichever seats `advance` leaves for
    it. `advance(game)` drives the non-learner seats until it is an
    MCTS-played seat's turn (a no-op for pure self-play, where MCTS plays every
    seat). Returns `(examples, outcome, stats)`; an episode that exceeds
    `cfg.max_moves`, or in which MCTS never got a (recordable) move, is aborted
    (`[], None, None`) rather than mislabeled."""
    history: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []

    advance(game)
    move_idx = 0
    while not game.is_terminal():
        if move_idx >= cfg.max_moves:
            return [], None, None
        mask = game.action_mask()
        forced = int(mask.sum()) == 1
        temp = 1.0 if move_idx < cfg.temp_threshold else 0.0
        pi = mcts.get_action_probs(game, temp=temp, add_noise=not forced)
        if not forced:
            history.append((game.observation(), mask, pi, game.current_player()))
        game.apply(_sample_action(pi))
        move_idx += 1
        advance(game)

    if not history:
        # MCTS never made a recordable move (e.g. opponents finished the game
        # first, or every move was forced) — nothing to learn from.
        return [], None, None

    outcome = game.outcome()
    player_ids = game.player_ids()
    examples = [
        (obs, mask, pi, to_relative_vector(player_ids, to_move, outcome))
        for obs, mask, pi, to_move in history
    ]
    stats = metrics.game_stats(game, game.winner())
    return examples, outcome, stats


def play_episode(
    nnet: NNetWrapper, cfg: AZConfig, seed: int, end_game_cities: int | None
) -> EpisodeResult:
    """Pure self-play: every seat driven by `nnet` through MCTS. See
    `_mcts_episode`."""
    game = PowerGridGame(
        seed=seed, num_players=cfg.num_players, end_game_cities=end_game_cities
    )
    mcts = MCTS(nnet, cfg)
    return _mcts_episode(game, mcts, cfg, advance=lambda _game: None)


def play_episode_vs_bots(
    nnet: NNetWrapper,
    cfg: AZConfig,
    seed: int,
    end_game_cities: int | None,
    difficulty: str,
) -> EpisodeResult:
    """Learner (seat 0) driven by MCTS; the other seats by the Rust
    `difficulty` heuristic bot. Only the learner's turns are recorded."""
    game = PowerGridGame(
        seed=seed, num_players=cfg.num_players, end_game_cities=end_game_cities
    )
    learner = game.player_ids()[0]
    mcts = MCTS(nnet, cfg)

    def advance(g: PowerGridGame) -> None:
        g.advance_bots(learner, difficulty)

    return _mcts_episode(game, mcts, cfg, advance)


def _advance_opponents_with_net(
    game: PowerGridGame, learner: str, opp_nnet: NNetWrapper
) -> None:
    """Drive every non-learner seat with `opp_nnet` (net-only, masked-softmax
    sampling) until it is the learner's turn or the game ends. Sampling (not
    argmax) keeps the opponent games diverse — greedy self-opponents collapse
    to a single line of play."""
    while not game.is_terminal():
        if game.current_player() == learner:
            return
        probs, _ = opp_nnet.predict(game.observation(), game.action_mask())
        game.apply(_sample_action(probs))


def play_episode_vs_net(
    nnet: NNetWrapper,
    opp_nnet: NNetWrapper,
    cfg: AZConfig,
    seed: int,
    end_game_cities: int | None,
) -> EpisodeResult:
    """Learner (seat 0) driven by MCTS; the other seats by `opp_nnet` (a past
    checkpoint) playing net-only. Only the learner's turns are recorded."""
    game = PowerGridGame(
        seed=seed, num_players=cfg.num_players, end_game_cities=end_game_cities
    )
    learner = game.player_ids()[0]
    mcts = MCTS(nnet, cfg)

    def advance(g: PowerGridGame) -> None:
        _advance_opponents_with_net(g, learner, opp_nnet)

    return _mcts_episode(game, mcts, cfg, advance)


# ---------------------------------------------------------------------------
# Multiprocessing worker entry points
#
# `coach.py` creates a `multiprocessing.Pool` per iteration with `_worker_init`
# as the initializer (handed the current learner weights once), then maps
# `_worker_run` over lightweight per-episode task tuples. Both are module-level
# so they pickle cleanly. Per-process globals cache the learner net (set once
# per iteration in the initializer) and any past-checkpoint opponents (loaded
# lazily by path, cached across tasks the process handles).
# ---------------------------------------------------------------------------

_W_CFG: AZConfig | None = None
_W_NNET: NNetWrapper | None = None
_W_OPPONENTS: dict[str, NNetWrapper] = {}


def _worker_init(cfg: AZConfig, state_dict: dict) -> None:
    global _W_CFG, _W_NNET, _W_OPPONENTS
    import torch

    torch.set_num_threads(1)  # each episode is single-threaded; workers give parallelism
    _W_CFG = cfg
    _W_NNET = NNetWrapper(cfg)
    _W_NNET.net.load_state_dict(state_dict)
    _W_NNET.net.eval()
    _W_OPPONENTS = {}


def _worker_opponent(path: str) -> NNetWrapper:
    opp = _W_OPPONENTS.get(path)
    if opp is None:
        assert _W_CFG is not None
        opp = NNetWrapper.load(path, device=_W_CFG.device)
        opp.net.eval()
        _W_OPPONENTS[path] = opp
    return opp


def _worker_run(task: tuple) -> EpisodeResult:
    """Run one episode in a pool worker. `task` = (seed, egc, mode, opp), where
    `mode` is "selfplay" | "vs_bots" | "vs_net" and `opp` is the bot difficulty
    (vs_bots), the opponent checkpoint path (vs_net), or None (selfplay)."""
    seed, egc, mode, opp = task
    assert _W_CFG is not None and _W_NNET is not None
    np.random.seed(seed & 0xFFFFFFFF)
    if mode == "vs_bots":
        return play_episode_vs_bots(_W_NNET, _W_CFG, seed, egc, opp)
    if mode == "vs_net":
        return play_episode_vs_net(_W_NNET, _worker_opponent(opp), _W_CFG, seed, egc)
    return play_episode(_W_NNET, _W_CFG, seed, egc)
