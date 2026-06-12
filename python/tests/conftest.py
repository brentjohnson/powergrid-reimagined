"""Shared test helpers."""

import numpy as np
import pytest


def make_policy_bytes(rng: np.random.Generator) -> bytes:
    """Valid PGRLPOL1 blob with small random weights (a 'random' policy)."""
    import struct

    from powergrid_env.constants import N_ACTIONS, OBS_SIZE
    from powergrid_env.export import HIDDEN, MAGIC

    out = bytearray(MAGIC)
    out += struct.pack("<III", OBS_SIZE, HIDDEN, N_ACTIONS)
    n_params = (
        HIDDEN * OBS_SIZE + HIDDEN + HIDDEN * HIDDEN + HIDDEN
        + N_ACTIONS * HIDDEN + N_ACTIONS
    )
    out += (rng.standard_normal(n_params) * 0.05).astype("<f4").tobytes()
    return bytes(out)


@pytest.fixture
def random_policy_bytes() -> bytes:
    return make_policy_bytes(np.random.default_rng(0))
