"""Training callbacks shared by the training scripts."""

import json
import os

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
