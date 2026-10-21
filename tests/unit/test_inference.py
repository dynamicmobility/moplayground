"""Checkpoint discovery and dispatch. Loading real checkpoints is covered by
tests/gpu/test_inference_roundtrip.py."""
import numpy as np
import pytest

import moplayground as mop


def test_get_all_models_sorted_numerically(tmp_path):
    for name in ['000000000200', '000000001000', '000000000030', 'plots', 'config.yaml']:
        (tmp_path / name).mkdir() if '.' not in name else (tmp_path / name).touch()
    (tmp_path / '999').touch()  # a file, not a checkpoint directory
    assert [p.name for p in mop.get_all_models(tmp_path)] == ['000000000030', '000000000200', '000000001000']
    assert mop.get_last_model(tmp_path).name == '000000001000'


def test_get_all_models_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        mop.get_all_models(tmp_path / 'nope')


def test_load_mo_policy_unknown_algorithm(tmp_path):
    with pytest.raises(ValueError, match='Unknown algorithm'):
        mop.load_mo_policy('ppo', {}, 2, tmp_path, np.array([0.5, 0.5]))
