"""Training callbacks shared by the training scripts."""

import json
import os

from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback


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
