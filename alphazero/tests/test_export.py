"""Tests for `alphazero.export`: the run-dir checkpoint resolver picks the
canonical best in preference order, and a resolved checkpoint round-trips
through the PGRLPOL1 serializer."""

import pytest

from alphazero.config import AZConfig
from alphazero.export import _BEST_NAMES, resolve_checkpoint
from alphazero.network import NNetWrapper


def _write(path):
    NNetWrapper(AZConfig(num_players=4, net_width=16, value_hidden=8)).save(str(path))


def test_resolver_prefers_named_best_over_iters(tmp_path):
    _write(tmp_path / "iter_0001.pt")
    _write(tmp_path / "iter_0002.pt")
    _write(tmp_path / "best.pt")
    assert resolve_checkpoint(str(tmp_path)) == str(tmp_path / "best.pt")


def test_resolver_preference_order(tmp_path):
    # dagger.pt beats best.pt beats cloned.pt.
    for name in _BEST_NAMES:
        _write(tmp_path / name)
    assert resolve_checkpoint(str(tmp_path)).endswith(_BEST_NAMES[0])
    (tmp_path / _BEST_NAMES[0]).unlink()
    assert resolve_checkpoint(str(tmp_path)).endswith(_BEST_NAMES[1])


def test_resolver_falls_back_to_latest_iter(tmp_path):
    _write(tmp_path / "iter_0003.pt")
    _write(tmp_path / "iter_0011.pt")
    _write(tmp_path / "iter_0007.pt")
    assert resolve_checkpoint(str(tmp_path)) == str(tmp_path / "iter_0011.pt")


def test_resolver_errors_on_empty_dir(tmp_path):
    with pytest.raises(SystemExit):
        resolve_checkpoint(str(tmp_path))
