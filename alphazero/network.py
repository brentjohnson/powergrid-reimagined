"""PGNet: the AlphaZero policy+value network for Power Grid.

The shared trunk and policy head exactly mirror the exportable Rust shape
(`crates/powergrid-bot-strategy/src/policy.rs::MlpPolicy`): two equal-width
tanh hidden layers feeding a 94-logit policy head. `policy_state_dict()`
exposes those three layers under the same key names sb3's MaskablePPO uses
(`mlp_extractor.policy_net.{0,2}` / `action_net`), so the existing
`powergrid_env.export.policy_state_dict_to_bytes` serializes this net to
PGRLPOL1 unchanged — no new export format, no Rust loader change needed.

The value head is a small side branch off the second hidden layer, used only
during training (search leaf evaluation + the value-target loss); it is
never exported.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from powergrid_env.constants import N_ACTIONS, OBS_SIZE

from .config import AZConfig


class PGNet(nn.Module):
    def __init__(self, num_players: int, hidden: int = 128, value_hidden: int = 64):
        super().__init__()
        self.l1 = nn.Linear(OBS_SIZE, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.policy_head = nn.Linear(hidden, N_ACTIONS)
        self.value_l1 = nn.Linear(hidden, value_hidden)
        self.value_head = nn.Linear(value_hidden, num_players)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.tanh(self.l1(obs))
        h = torch.tanh(self.l2(h))
        logits = self.policy_head(h)
        v = torch.tanh(self.value_l1(h))
        value = torch.tanh(self.value_head(v))
        return logits, value

    def policy_state_dict(self) -> dict[str, torch.Tensor]:
        """Policy-path weights under sb3 MaskablePPO key names, for reuse with
        `powergrid_env.export.policy_state_dict_to_bytes` (no format change)."""
        return {
            "mlp_extractor.policy_net.0.weight": self.l1.weight.detach(),
            "mlp_extractor.policy_net.0.bias": self.l1.bias.detach(),
            "mlp_extractor.policy_net.2.weight": self.l2.weight.detach(),
            "mlp_extractor.policy_net.2.bias": self.l2.bias.detach(),
            "action_net.weight": self.policy_head.weight.detach(),
            "action_net.bias": self.policy_head.bias.detach(),
        }


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Log-probabilities with illegal actions driven to ~0 probability."""
    neg_inf = torch.finfo(logits.dtype).min
    masked_logits = torch.where(mask > 0, logits, torch.full_like(logits, neg_inf))
    return F.log_softmax(masked_logits, dim=-1)


class NNetWrapper:
    """Train/predict/save/load wrapper around `PGNet`, playing the role of
    alpha-zero-general's `NeuralNet` interface."""

    def __init__(self, cfg: AZConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.net = PGNet(
            cfg.num_players, hidden=cfg.net_width, value_hidden=cfg.value_hidden
        ).to(self.device)
        # Created once and reused across `train()` calls (rather than
        # recreated per call) so Adam's running moments persist across
        # training iterations — recreating it every call effectively resets
        # momentum each time, which destabilizes finetuning.
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)

    def predict(self, obs: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """obs: (OBS_SIZE,) float array. mask: (N_ACTIONS,) 0/1 array.
        Returns (masked_softmax_probs[N_ACTIONS], value[num_players])."""
        self.net.eval()
        with torch.no_grad():
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(
                self.device
            )
            mask_t = torch.from_numpy(np.asarray(mask, dtype=np.float32)).unsqueeze(0).to(
                self.device
            )
            logits, value = self.net(obs_t)
            probs = torch.exp(masked_log_softmax(logits, mask_t))[0].cpu().numpy()
            value = value[0].cpu().numpy()
        return probs, value

    def _train_batch(
        self,
        examples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        batch: np.ndarray,
    ) -> tuple[float, float]:
        """One optimizer step over the examples indexed by `batch`. Returns
        (policy_loss, value_loss) as floats."""
        obs = torch.from_numpy(np.stack([examples[i][0] for i in batch])).float()
        mask = torch.from_numpy(np.stack([examples[i][1] for i in batch])).float()
        target_pi = torch.from_numpy(np.stack([examples[i][2] for i in batch])).float()
        target_v = torch.from_numpy(np.stack([examples[i][3] for i in batch])).float()
        obs, mask = obs.to(self.device), mask.to(self.device)
        target_pi, target_v = target_pi.to(self.device), target_v.to(self.device)

        logits, value = self.net(obs)
        log_probs = masked_log_softmax(logits, mask)
        pi_loss = -(target_pi * log_probs).sum(dim=1).mean()
        v_loss = F.mse_loss(value, target_v)
        loss = pi_loss + v_loss

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return pi_loss.item(), v_loss.item()

    def train(
        self,
        examples: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        num_batches: int | None = None,
    ) -> dict[str, float]:
        """`examples`: list of (obs, mask, target_pi, target_value).

        Two modes:
        - `num_batches=None` (default): epoch-style — `cfg.train_epochs` full
          shuffled passes over `examples`. Used by `pretrain.py` over its
          fixed behavior-cloning dataset.
        - `num_batches=K`: a fixed budget of K minibatches, each sampled
          uniformly at random (with replacement) from `examples`. Used by the
          coach's windowed replay so each iteration does a bounded, roughly
          on-policy amount of training regardless of buffer size.
        """
        self.net.train()
        n = len(examples)
        if n == 0:
            return {"policy_loss": 0.0, "value_loss": 0.0}
        pi_losses: list[float] = []
        v_losses: list[float] = []
        bs = self.cfg.batch_size
        if num_batches is None:
            idx = np.arange(n)
            for _epoch in range(self.cfg.train_epochs):
                np.random.shuffle(idx)
                for start in range(0, n, bs):
                    pl, vl = self._train_batch(examples, idx[start : start + bs])
                    pi_losses.append(pl)
                    v_losses.append(vl)
        else:
            for _ in range(num_batches):
                batch = np.random.randint(0, n, size=min(bs, n))
                pl, vl = self._train_batch(examples, batch)
                pi_losses.append(pl)
                v_losses.append(vl)
        return {
            "policy_loss": float(np.mean(pi_losses)) if pi_losses else 0.0,
            "value_loss": float(np.mean(v_losses)) if v_losses else 0.0,
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "model_state": self.net.state_dict(),
                "optimizer_state": self.opt.state_dict(),
                "num_players": self.cfg.num_players,
                "net_width": self.cfg.net_width,
                "value_hidden": self.cfg.value_hidden,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu", cfg: AZConfig | None = None) -> NNetWrapper:
        """Load a checkpoint's weights (and optimizer momentum, if present).

        `cfg`, if given, supplies the training hyperparameters to use going
        forward (lr, batch_size, train_epochs, ...) — e.g. a lower finetune
        LR. Only the *architecture* fields (num_players/net_width/
        value_hidden), which must match the saved weights, are taken from
        the checkpoint, overriding whatever `cfg` says. Without `cfg`, a
        fresh default `AZConfig` is used (e.g. for arena/eval-only loads).
        """
        ckpt = torch.load(path, map_location=device, weights_only=True)
        base = cfg if cfg is not None else AZConfig()
        cfg = dataclasses.replace(
            base,
            num_players=ckpt["num_players"],
            net_width=ckpt["net_width"],
            value_hidden=ckpt["value_hidden"],
            device=device,
        )
        wrapper = cls(cfg)
        wrapper.net.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            wrapper.opt.load_state_dict(ckpt["optimizer_state"])
            # load_state_dict restores the *saved* lr into every param group,
            # silently overriding cfg.lr — which would defeat a low-lr
            # finetune resume (the whole point of passing cfg here). Force the
            # requested lr back on.
            for group in wrapper.opt.param_groups:
                group["lr"] = cfg.lr
        return wrapper
