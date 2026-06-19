"""Shared test helpers."""

import numpy as np
import pytest


def make_policy_bytes(rng: np.random.Generator, hidden: int = 64) -> bytes:
    """Valid PGRLPOL1 blob with small random weights (a 'random' policy).

    ``hidden`` is the (equal-width) hidden size; the loader reads it from the
    header, so any value the Rust port accepts works here.
    """
    import struct

    from powergrid_env.constants import N_ACTIONS, OBS_SIZE
    from powergrid_env.export import MAGIC

    out = bytearray(MAGIC)
    out += struct.pack("<III", OBS_SIZE, hidden, N_ACTIONS)
    n_params = (
        hidden * OBS_SIZE + hidden + hidden * hidden + hidden
        + N_ACTIONS * hidden + N_ACTIONS
    )
    out += (rng.standard_normal(n_params) * 0.05).astype("<f4").tobytes()
    return bytes(out)


@pytest.fixture
def random_policy_bytes() -> bytes:
    return make_policy_bytes(np.random.default_rng(0))
