"""
Serialization of an sb3 MaskablePPO policy network to the flat PGRLPOL2 binary
consumed by Rust (powergrid-bot-strategy/src/policy.rs::MlpPolicy::from_bytes).

Only the policy path is exported (the value head is not needed for play):

    obs(OBS_SIZE) -> Linear -> tanh -> Linear -> tanh -> Linear -> logits(N_ACTIONS)

Binary layout (all little-endian):
    8 bytes   magic b"PGRLPOL2"
    3 * u32   obs_size, hidden, n_actions
    f32[]     l1.weight, l1.bias, l2.weight, l2.bias, out.weight, out.bias
              (torch row-major order: weight[out][in])

Used by scripts/export_policy.py (Rust Expert bot weights) and by
OpponentSnapshotCallback (frozen-opponent self-play).
"""

import struct

from .constants import N_ACTIONS, OBS_SIZE

# Layout epoch, not just a format tag: bump whenever macro ids are renumbered,
# even if N_ACTIONS is unchanged, so a stale policy fails to load instead of
# silently playing a scrambled action map. See policy.rs.
MAGIC = b"PGRLPOL2"
# Value network: same MLP shape as the policy but a single scalar output (the
# acting seat's expected return / win-value). Used by the Rust play-time search
# (search.rs) as the MCTS leaf value — one forward pass instead of a rollout.
VALUE_MAGIC = b"PGRLVAL1"
VALUE_OUT_DIM = 1

# Net width (the single hidden size shared by both layers) is inferred from the
# state_dict rather than hard-coded — the format encodes it in the header, the
# Rust loader reads it back, and this serializer is used live during self-play
# to snapshot opponents, so it must track whatever width the model was built
# with. The architecture is fixed at two equal-width hidden layers.
def _hidden_size(state_dict) -> int:
    return int(state_dict["mlp_extractor.policy_net.0.weight"].shape[0])


def policy_tensors_from_state_dict(state_dict) -> list:
    """Policy-path tensors in file order, shape-checked for the
    ``OBS_SIZE -> hidden -> hidden -> N_ACTIONS`` MLP (hidden inferred)."""
    hidden = _hidden_size(state_dict)
    # (state-dict key prefix, expected weight shape) in file order.
    layers = [
        ("mlp_extractor.policy_net.0", (hidden, OBS_SIZE)),
        ("mlp_extractor.policy_net.2", (hidden, hidden)),
        ("action_net", (N_ACTIONS, hidden)),
    ]
    tensors = []
    for key, shape in layers:
        weight = state_dict[f"{key}.weight"]
        bias = state_dict[f"{key}.bias"]
        assert tuple(weight.shape) == shape, (
            f"{key}.weight has shape {tuple(weight.shape)}, expected {shape} — "
            "the net must be two equal-width hidden layers (net_arch=dict("
            "pi=[h, h], vf=[h, h]))."
        )
        assert tuple(bias.shape) == (shape[0],)
        tensors.extend([weight, bias])
    return tensors


def policy_state_dict_to_bytes(state_dict) -> bytes:
    """Serialize the policy path of a MaskablePPO ``policy.state_dict()``."""
    out = bytearray(MAGIC)
    out += struct.pack("<III", OBS_SIZE, _hidden_size(state_dict), N_ACTIONS)
    for t in policy_tensors_from_state_dict(state_dict):
        out += t.detach().cpu().numpy().astype("<f4").tobytes()
    return bytes(out)


def _value_hidden_size(state_dict) -> int:
    return int(state_dict["mlp_extractor.value_net.0.weight"].shape[0])


def value_tensors_from_state_dict(state_dict) -> list:
    """Value-path tensors in file order, shape-checked for the
    ``OBS_SIZE -> hidden -> hidden -> 1`` MLP (hidden inferred)."""
    hidden = _value_hidden_size(state_dict)
    layers = [
        ("mlp_extractor.value_net.0", (hidden, OBS_SIZE)),
        ("mlp_extractor.value_net.2", (hidden, hidden)),
        ("value_net", (VALUE_OUT_DIM, hidden)),
    ]
    tensors = []
    for key, shape in layers:
        weight = state_dict[f"{key}.weight"]
        bias = state_dict[f"{key}.bias"]
        assert tuple(weight.shape) == shape, (
            f"{key}.weight has shape {tuple(weight.shape)}, expected {shape} — "
            "the value net must be two equal-width hidden layers (vf=[h, h])."
        )
        assert tuple(bias.shape) == (shape[0],)
        tensors.extend([weight, bias])
    return tensors


def value_state_dict_to_bytes(state_dict) -> bytes:
    """Serialize the value path of a MaskablePPO ``policy.state_dict()`` to the
    flat PGRLVAL1 binary consumed by Rust (``policy.rs::ValueNet::from_bytes``).

    Layout mirrors PGRLPOL2: magic + (obs_size, hidden, out_dim=1) + the six
    f32 tensor blocks (l1/l2/out weight+bias, torch row-major)."""
    out = bytearray(VALUE_MAGIC)
    out += struct.pack("<III", OBS_SIZE, _value_hidden_size(state_dict), VALUE_OUT_DIM)
    for t in value_tensors_from_state_dict(state_dict):
        out += t.detach().cpu().numpy().astype("<f4").tobytes()
    return bytes(out)
