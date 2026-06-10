import numpy as np


class RandomPolicy:
    """Uniform-random policy over the legal action mask."""

    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()

    def act(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        legal = np.where(action_mask)[0]
        if len(legal) == 0:
            return 0  # fallback: PassAuction
        return int(self.rng.choice(legal))
