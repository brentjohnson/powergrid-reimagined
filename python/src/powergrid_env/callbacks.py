"""Training callbacks shared by the training scripts."""

import glob
import json
import os
import re

import numpy as np
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import BaseCallback

# Rulebook end-game city trigger by player count (rules.rs::handle_start).
RULEBOOK_END_GAME_CITIES = {2: 21, 3: 17, 4: 17, 5: 15, 6: 14}


class PersistentBestEvalCallback(MaskableEvalCallback):
    """MaskableEvalCallback whose best-reward bar survives --resume-from.

    The stock callback keeps ``best_mean_reward`` only in memory, so a resumed
    run restarts at -inf and its first eval overwrites best_model.zip even if
    it scores worse than the old best. This subclass persists the bar in
    ``best_mean_reward.json`` next to ``best_model.zip`` and reloads it on
    startup.

    Delete the json to reset the bar — needed whenever stored scores stop
    being comparable, e.g. after changing reward shaping, eval opponents, or
    --eval-episodes.
    """

    def _bar_path(self) -> str:
        assert self.best_model_save_path is not None
        return os.path.join(self.best_model_save_path, "best_mean_reward.json")

    def _init_callback(self) -> None:
        super()._init_callback()
        if self.best_model_save_path is None:
            return
        try:
            with open(self._bar_path()) as f:
                self.best_mean_reward = float(json.load(f)["best_mean_reward"])
        except FileNotFoundError:
            return
        print(
            f"Inherited best_mean_reward={self.best_mean_reward:.4f} from "
            f"{self._bar_path()}; best_model.zip is kept until an eval beats it. "
            f"Delete that file to reset the bar."
        )

    def _on_step(self) -> bool:
        previous_best = self.best_mean_reward
        continue_training = super()._on_step()
        if self.best_mean_reward > previous_best and self.best_model_save_path is not None:
            with open(self._bar_path(), "w") as f:
                json.dump({"best_mean_reward": self.best_mean_reward}, f)
        return continue_training

    def reset_bar(self) -> None:
        """Drop the best-reward bar (in memory and on disk). Used when eval
        scores stop being comparable, e.g. on a curriculum stage change."""
        self.best_mean_reward = -np.inf
        if self.best_model_save_path is not None:
            try:
                os.remove(self._bar_path())
            except FileNotFoundError:
                pass


class OpponentSnapshotCallback(BaseCallback):
    """Frozen-opponent self-play: periodically freeze the current policy and
    hand it to the training envs as their opponent.

    Every ``snapshot_every`` timesteps (and at training start) the policy-path
    weights are serialized via ``powergrid_env.export.policy_state_dict_to_bytes``
    and broadcast with ``env_method("set_opponent_policy", ...)``; each env
    picks the new snapshot up at its next reset.
    """

    def __init__(self, train_env, *, snapshot_every: int, verbose: int = 0):
        super().__init__(verbose)
        if snapshot_every < 1:
            raise ValueError("snapshot_every must be >= 1")
        self.train_env = train_env
        self.snapshot_every = snapshot_every
        self._last_snapshot_at = 0

    def _push(self) -> None:
        from .export import policy_state_dict_to_bytes

        data = policy_state_dict_to_bytes(self.model.policy.state_dict())
        self.train_env.env_method("set_opponent_policy", data)
        self._last_snapshot_at = self.model.num_timesteps
        if self.verbose:
            print(f"Self-play: opponent snapshot at {self._last_snapshot_at:,} timesteps")

    def _on_training_start(self) -> None:
        self._push()

    def _on_step(self) -> bool:
        if self.model.num_timesteps - self._last_snapshot_at >= self.snapshot_every:
            self._push()
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("selfplay/snapshot_timesteps", self._last_snapshot_at)


class LeagueSnapshotCallback(BaseCallback):
    """Population-based self-play: keep a league of past policy snapshots and
    sample opponents from it (AlphaStar-style), instead of only the latest
    snapshot — pure latest-snapshot self-play can converge to echo-chamber
    equilibria (see TRAINING-SUGGESTION.md / TRAINING-NEXT-STEPS.md).

    Every ``snapshot_every`` timesteps the current policy is serialized to
    ``league_dir/snap_<timesteps>.bin`` (PGRLPOL6) and the envs' opponent pool
    is rebuilt via ``env_method("set_opponent_pool", ...)``:

      - the latest snapshot, weight ``mix[0]``
      - up to ``past_k`` uniformly sampled *older* snapshots sharing ``mix[1]``
      - heuristic ``bot_difficulty`` bots, weight ``mix[2]``

    Snapshots persist on disk, so a resumed run (--resume-from) reloads its
    league on training start and continues where it left off.

    ``peer_dirs`` (cross-arm population play) adds other concurrently-training
    runs' league dirs to the *past* pool: their snapshots are rescanned at
    every refresh, so each arm keeps sampling its siblings as they evolve.
    Peer snapshots share ``mix[1]`` with the arm's own history; the "latest"
    slot stays the arm's own. Peer files are validated on read (a sibling may
    be mid-write or its dir not created yet) — invalid picks are dropped from
    that refresh rather than crashing training.
    """

    def __init__(self, train_env, *, snapshot_every: int, league_dir: str,
                 past_k: int = 4, mix: tuple[float, float, float] = (0.5, 0.3, 0.2),
                 bot_difficulty: str = "hard", seed: int | None = None,
                 peer_dirs: list[str] | tuple[str, ...] = (),
                 verbose: int = 0):
        super().__init__(verbose)
        if snapshot_every < 1:
            raise ValueError("snapshot_every must be >= 1")
        if past_k < 0:
            raise ValueError("past_k must be >= 0")
        if len(mix) != 3 or any(w < 0 for w in mix) or sum(mix) <= 0:
            raise ValueError("mix must be three non-negative weights summing > 0")
        self.train_env = train_env
        self.snapshot_every = snapshot_every
        self.league_dir = league_dir
        self.past_k = past_k
        self.mix = tuple(float(w) for w in mix)
        self.bot_difficulty = bot_difficulty
        own = os.path.abspath(league_dir)
        self.peer_dirs = [d for d in peer_dirs if os.path.abspath(d) != own]
        self._rng = np.random.default_rng(seed)
        self._last_snapshot_at = 0
        self._snapshots: list[tuple[int, str]] = []  # (timesteps, path), sorted
        self._peer_snapshots: list[str] = []

    def _scan_league(self) -> None:
        self._snapshots = sorted(
            (int(m.group(1)), p)
            for p in glob.glob(os.path.join(self.league_dir, "snap_*.bin"))
            if (m := re.search(r"snap_(\d+)\.bin$", p))
        )
        self._peer_snapshots = [
            p for d in self.peer_dirs
            for p in glob.glob(os.path.join(d, "snap_*.bin"))
        ]

    @staticmethod
    def _read_snapshot(path: str) -> bytes | None:
        """Snapshot bytes, or None if unreadable/not a policy file. Own writes
        are atomic (tmp + rename), but a peer arm may have been synced half-way
        or use a different format epoch — never let that kill training."""
        from .export import MAGIC

        try:
            with open(path, "rb") as f:
                data = f.read()
        except OSError:
            return None
        return data if data.startswith(MAGIC) else None

    def _build_pool(self) -> list[tuple[str, bytes | str, float]]:
        p_latest, p_past, p_bots = self.mix
        pool: list[tuple[str, bytes | str, float]] = []
        latest_step, latest_path = self._snapshots[-1]
        with open(latest_path, "rb") as f:
            pool.append(("policy", f.read(), p_latest))
        past = [p for _, p in self._snapshots[:-1]] + self._peer_snapshots
        if past and self.past_k > 0 and p_past > 0:
            picks = self._rng.choice(len(past), size=min(self.past_k, len(past)),
                                     replace=False)
            blobs = [b for i in picks if (b := self._read_snapshot(past[int(i)]))]
            for b in blobs:
                pool.append(("policy", b, p_past / len(blobs)))
        if p_bots > 0:
            pool.append(("bots", self.bot_difficulty, p_bots))
        return pool

    def _push(self) -> None:
        from .export import policy_state_dict_to_bytes

        os.makedirs(self.league_dir, exist_ok=True)
        data = policy_state_dict_to_bytes(self.model.policy.state_dict())
        path = os.path.join(self.league_dir, f"snap_{self.model.num_timesteps}.bin")
        # Atomic write: peer arms scan this dir, so a snapshot must never be
        # observable half-written.
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
        self._scan_league()
        self.train_env.env_method("set_opponent_pool", self._build_pool())
        self._last_snapshot_at = self.model.num_timesteps
        if self.verbose:
            peers = f", {len(self._peer_snapshots)} peer" if self.peer_dirs else ""
            print(f"League: snapshot at {self._last_snapshot_at:,} timesteps "
                  f"({len(self._snapshots)} in league{peers})")

    def _on_training_start(self) -> None:
        self._push()

    def _on_step(self) -> bool:
        if self.model.num_timesteps - self._last_snapshot_at >= self.snapshot_every:
            self._push()
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("league/size", len(self._snapshots))
        if self.peer_dirs:
            self.logger.record("league/peer_size", len(self._peer_snapshots))
        self.logger.record("selfplay/snapshot_timesteps", self._last_snapshot_at)


class ShapingAnnealCallback(BaseCallback):
    """Anneal the powered-cities shaping bonus away over ``anneal_steps``
    timesteps (linear 1.0 → 0.0, then stays 0), per TRAINING-SUGGESTION.md:
    shaped rewards can teach the policy to optimize the wrong thing, so they
    should only bootstrap. Scale is derived from ``num_timesteps``, so
    --resume-from lands on the right value automatically.
    """

    def __init__(self, train_env, *, anneal_steps: int, verbose: int = 0):
        super().__init__(verbose)
        if anneal_steps < 1:
            raise ValueError("anneal_steps must be >= 1")
        self.train_env = train_env
        self.anneal_steps = anneal_steps
        self._current: float | None = None

    def _scale_for(self, num_timesteps: int) -> float:
        return max(0.0, 1.0 - num_timesteps / self.anneal_steps)

    def _apply(self) -> None:
        # Round so the env_method broadcast only happens ~100 times total.
        scale = round(self._scale_for(self.model.num_timesteps), 2)
        if scale != self._current:
            self.train_env.env_method("set_shaping_scale", scale)
            self._current = scale

    def _on_training_start(self) -> None:
        self._apply()
        print(f"Shaping anneal: scale={self._current} "
              f"(reaches 0 at {self.anneal_steps:,} timesteps)")

    def _on_step(self) -> bool:
        self._apply()
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("shaping/scale", self._current)


class EndGameCurriculumCallback(BaseCallback):
    """Fixed-schedule curriculum on the end-game city trigger.

    Games start with ``end_game_cities = start`` (short games, frequent wins,
    dense terminal signal) and the trigger is raised by ``step`` every
    ``bump_every`` timesteps until it reaches ``target`` (the rulebook value).
    The stage is derived from ``model.num_timesteps``, so --resume-from lands
    on the right stage automatically.

    On every stage change the train and eval envs are retargeted (taking
    effect at each env's next reset) and the eval callback's best-reward bar
    is reset — eval scores at different triggers are not comparable, and
    best_model.zip should mean "best at the current stage".
    """

    def __init__(
        self,
        train_env,
        eval_env=None,
        eval_callback: PersistentBestEvalCallback | None = None,
        *,
        start: int,
        step: int,
        bump_every: int,
        target: int,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        if start < 1 or step < 1 or bump_every < 1:
            raise ValueError("start, step, and bump_every must all be >= 1")
        self.train_env = train_env
        self.eval_env = eval_env
        self.eval_callback = eval_callback
        self.start = start
        self.step = step
        self.bump_every = bump_every
        self.target = target
        self.current: int | None = None

    def _stage_for(self, num_timesteps: int) -> int:
        return min(self.start + self.step * (num_timesteps // self.bump_every),
                   self.target)

    def _apply(self, value: int) -> None:
        self.train_env.env_method("set_end_game_cities", value)
        if self.eval_env is not None:
            self.eval_env.env_method("set_end_game_cities", value)
        self.current = value

    def _on_training_start(self) -> None:
        self._apply(self._stage_for(self.model.num_timesteps))
        print(f"Curriculum: end_game_cities={self.current} "
              f"(target {self.target}, +{self.step} every {self.bump_every:,} steps)")

    def _on_step(self) -> bool:
        value = self._stage_for(self.model.num_timesteps)
        if value != self.current:
            self._apply(value)
            if self.eval_callback is not None:
                self.eval_callback.reset_bar()
            print(f"Curriculum: end_game_cities raised to {value} "
                  f"at {self.model.num_timesteps:,} timesteps")
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record("curriculum/end_game_cities", self.current)
