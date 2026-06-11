"""
Export a trained MaskablePPO policy network to the flat binary format consumed
by the Rust Expert bot (powergrid-bot-strategy/src/policy.rs).

Only the policy path is exported (the value head is not needed for play):

    obs(OBS_SIZE) -> Linear -> tanh -> Linear -> tanh -> Linear -> logits(N_ACTIONS)

Binary layout (all little-endian):
    8 bytes   magic b"PGRLPOL1"
    3 * u32   obs_size, hidden, n_actions
    f32[]     l1.weight, l1.bias, l2.weight, l2.bias, out.weight, out.bias
              (torch row-major order: weight[out][in])

Also writes a golden JSON (deterministic observation + torch logits) used by
the Rust parity test in policy.rs.

Usage:
    python scripts/export_policy.py --model runs/vs_bots/best_model \
        --out ../assets/policies/expert.bin \
        --golden ../assets/policies/expert.golden.json
"""

import argparse
import io
import json
import struct
import zipfile

import numpy as np
import torch

from powergrid_env.constants import N_ACTIONS, OBS_SIZE

MAGIC = b"PGRLPOL1"
HIDDEN = 64

# (state-dict key prefix, expected shape) in file order.
LAYERS = [
    ("mlp_extractor.policy_net.0", (HIDDEN, OBS_SIZE)),
    ("mlp_extractor.policy_net.2", (HIDDEN, HIDDEN)),
    ("action_net", (N_ACTIONS, HIDDEN)),
]


def load_policy_tensors(model_path: str) -> list[torch.Tensor]:
    if not model_path.endswith(".zip"):
        model_path += ".zip"
    with zipfile.ZipFile(model_path) as zf:
        sd = torch.load(
            io.BytesIO(zf.read("policy.pth")), map_location="cpu", weights_only=True
        )
    tensors = []
    for key, shape in LAYERS:
        weight = sd[f"{key}.weight"]
        bias = sd[f"{key}.bias"]
        assert tuple(weight.shape) == shape, (
            f"{key}.weight has shape {tuple(weight.shape)}, expected {shape} — "
            "was the model trained with a custom net_arch?"
        )
        assert tuple(bias.shape) == (shape[0],)
        tensors.extend([weight, bias])
    return tensors


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

    tensors = load_policy_tensors(args.model)

    with open(args.out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<III", OBS_SIZE, HIDDEN, N_ACTIONS))
        for t in tensors:
            f.write(t.detach().numpy().astype("<f4").tobytes())

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

    n_params = sum(t.numel() for t in tensors)
    print(f"Wrote {args.out} ({n_params} params, "
          f"{len(MAGIC) + 12 + 4 * n_params} bytes) and {args.golden}")
    print("First 5 logits:", [round(v, 4) for v in golden["logits"][:5]])


if __name__ == "__main__":
    main()
