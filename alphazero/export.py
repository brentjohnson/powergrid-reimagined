"""Export an AlphaZero checkpoint's policy path to the PGRLPOL1 binary
consumed by the Rust Expert bot, reusing the existing serializer and
golden-logits parity-test format unchanged
(`powergrid_env.export`, `crates/powergrid-bot-strategy/src/policy.rs`).

The value head is training-only and is not exported — only the
trunk + policy head (`PGNet.policy_state_dict()`) is serialized.

Usage:
    python/.venv/bin/python -m alphazero.export --checkpoint alphazero/runs/default/best.pt \
        --out assets/policies/expert.bin \
        --golden assets/policies/expert.golden.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from powergrid_env.constants import OBS_SIZE
from powergrid_env.export import MAGIC, policy_state_dict_to_bytes, policy_tensors_from_state_dict

from .network import NNetWrapper


def _forward(tensors: list[torch.Tensor], obs: torch.Tensor) -> torch.Tensor:
    l1w, l1b, l2w, l2b, ow, ob = tensors
    h = torch.tanh(obs @ l1w.T + l1b)
    h = torch.tanh(h @ l2w.T + l2b)
    return h @ ow.T + ob


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default="assets/policies/expert.bin")
    parser.add_argument("--golden", default="assets/policies/expert.golden.json")
    args = parser.parse_args()

    wrapper = NNetWrapper.load(args.checkpoint, device="cpu")
    sd = wrapper.net.policy_state_dict()

    data = policy_state_dict_to_bytes(sd)
    with open(args.out, "wb") as f:
        f.write(data)

    tensors = policy_tensors_from_state_dict(sd)
    obs = torch.from_numpy(
        np.random.default_rng(12345).uniform(0.0, 1.0, OBS_SIZE).astype(np.float32)
    )
    zeros = torch.zeros(OBS_SIZE, dtype=torch.float32)
    golden = {
        "obs": obs.tolist(),
        "logits": _forward(tensors, obs).tolist(),
        "zeros_logits": _forward(tensors, zeros).tolist(),
    }
    with open(args.golden, "w") as f:
        json.dump(golden, f)

    n_params = (len(data) - len(MAGIC) - 12) // 4
    print(f"Wrote {args.out} ({n_params} params, {len(data)} bytes) and {args.golden}")
    print("First 5 logits:", [round(v, 4) for v in golden["logits"][:5]])


if __name__ == "__main__":
    main()
