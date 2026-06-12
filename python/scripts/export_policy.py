"""
Export a trained MaskablePPO policy network to the flat binary format consumed
by the Rust Expert bot (powergrid-bot-strategy/src/policy.rs).

The binary layout (PGRLPOL1) and the serialization live in
powergrid_env.export; this script adds checkpoint loading and writes a golden
JSON (deterministic observation + torch logits) used by the Rust parity test
in policy.rs.

Usage:
    python scripts/export_policy.py --model runs/vs_bots/best_model \
        --out ../assets/policies/expert.bin \
        --golden ../assets/policies/expert.golden.json
"""

import argparse
import io
import json
import zipfile

import numpy as np
import torch

from powergrid_env.constants import OBS_SIZE
from powergrid_env.export import (
    MAGIC,
    policy_state_dict_to_bytes,
    policy_tensors_from_state_dict,
)


def load_state_dict(model_path: str) -> dict:
    if not model_path.endswith(".zip"):
        model_path += ".zip"
    with zipfile.ZipFile(model_path) as zf:
        return torch.load(
            io.BytesIO(zf.read("policy.pth")), map_location="cpu", weights_only=True
        )


def forward(tensors: list[torch.Tensor], obs: torch.Tensor) -> torch.Tensor:
    l1w, l1b, l2w, l2b, ow, ob = tensors
    h = torch.tanh(obs @ l1w.T + l1b)
    h = torch.tanh(h @ l2w.T + l2b)
    return h @ ow.T + ob


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/vs_bots/best_model",
                        help="Path to a saved MaskablePPO .zip (without .zip suffix).")
    parser.add_argument("--out", default="../assets/policies/expert.bin")
    parser.add_argument("--golden", default="../assets/policies/expert.golden.json")
    args = parser.parse_args()

    sd = load_state_dict(args.model)
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
        "logits": forward(tensors, obs).tolist(),
        "zeros_logits": forward(tensors, zeros).tolist(),
    }
    with open(args.golden, "w") as f:
        json.dump(golden, f)

    n_params = (len(data) - len(MAGIC) - 12) // 4
    print(f"Wrote {args.out} ({n_params} params, {len(data)} bytes) and {args.golden}")
    print("First 5 logits:", [round(v, 4) for v in golden["logits"][:5]])


if __name__ == "__main__":
    main()
