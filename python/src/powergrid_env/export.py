"""
Serialization of an sb3 MaskablePPO policy network to the flat PGRLPOL1 binary
consumed by Rust (powergrid-bot-strategy/src/policy.rs::MlpPolicy::from_bytes).

Only the policy path is exported (the value head is not needed for play):

    obs(OBS_SIZE) -> Linear -> tanh -> Linear -> tanh -> Linear -> logits(N_ACTIONS)

Binary layout (all little-endian):
    8 bytes   magic b"PGRLPOL1"
    3 * u32   obs_size, hidden, n_actions
    f32[]     l1.weight, l1.bias, l2.weight, l2.bias, out.weight, out.bias
              (torch row-major order: weight[out][in])

Used by scripts/export_policy.py (Rust Expert bot weights) and by
OpponentSnapshotCallback (frozen-opponent self-play).
"""

import struct

from .constants import N_ACTIONS, OBS_SIZE

MAGIC = b"PGRLPOL1"
HIDDEN = 64

# (state-dict key prefix, expected shape) in file order.
LAYERS = [
    ("mlp_extractor.policy_net.0", (HIDDEN, OBS_SIZE)),
    ("mlp_extractor.policy_net.2", (HIDDEN, HIDDEN)),
    ("action_net", (N_ACTIONS, HIDDEN)),
]


def policy_tensors_from_state_dict(state_dict) -> list:
    """Policy-path tensors in file order, shape-checked against LAYERS."""
    tensors = []
    for key, shape in LAYERS:
        weight = state_dict[f"{key}.weight"]
        bias = state_dict[f"{key}.bias"]
        assert tuple(weight.shape) == shape, (
            f"{key}.weight has shape {tuple(weight.shape)}, expected {shape} — "
            "was the model trained with a custom net_arch?"
        )
        assert tuple(bias.shape) == (shape[0],)
        tensors.extend([weight, bias])
    return tensors


def policy_state_dict_to_bytes(state_dict) -> bytes:
    """Serialize the policy path of a MaskablePPO ``policy.state_dict()``."""
    out = bytearray(MAGIC)
    out += struct.pack("<III", OBS_SIZE, HIDDEN, N_ACTIONS)
    for t in policy_tensors_from_state_dict(state_dict):
        out += t.detach().cpu().numpy().astype("<f4").tobytes()
    return bytes(out)
