"""Entry-point scripts: dispatch logic, plus a static check that every
``mop.<name>`` / ``cfg.<name>`` they use still exists (scripts run at import
time, so they cannot simply be imported)."""
import ast
import functools

import minimal_mjx as mm
import pytest

import moplayground as mop
from scripts.train import get_handle_params, get_progress
from tests.helpers import REPO_ROOT

SCRIPTS = [
    pytest.param('train.py'),
    pytest.param('tune.py'),
    pytest.param('rollout.py'),
    pytest.param('download_model.py'),
]


@pytest.mark.parametrize('algorithm, setup, progress, network_property', [
    ('morlax', mop.setup_morlax, mop.MOProgress, 'morlax_network_params'),
    ('amor', mop.setup_amor, mop.MOProgress, 'amor_network_params'),
    ('ppo', mm.setup_ppo, mm.Progress, 'ppo_network_params'),
])
def test_dispatch(cheetah_cfg, monkeypatch, algorithm, setup, progress, network_property):
    monkeypatch.setitem(cheetah_cfg.cfg, 'algorithm', algorithm)
    handle, train_params, network_params = get_handle_params(cheetah_cfg)
    assert handle is setup
    assert network_params == getattr(cheetah_cfg, network_property)
    has_sampling = set(cheetah_cfg.sampling_params) <= set(train_params)
    assert has_sampling == (algorithm != 'ppo')
    assert get_progress(cheetah_cfg) is progress


@pytest.mark.parametrize('setup', [mop.setup_morlax, mop.setup_amor])
def test_mo_setup(cheetah_cfg, setup):
    train_fn, network_factory = setup(cheetah_cfg.ppo_params, cheetah_cfg.amor_network_params, None)
    assert isinstance(train_fn, functools.partial)
    assert train_fn.keywords['wrap_env_fn'] is mop.mo_wrapper
    assert train_fn.keywords['network_factory'] is network_factory


def _used_names(tree, base):
    return {
        node.attr for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == base
    }


@pytest.mark.parametrize('script', SCRIPTS)
def test_script_references_exist(script):
    tree = ast.parse((REPO_ROOT / 'scripts' / script).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == 'moplayground':
            for alias in node.names:
                assert hasattr(mop, alias.name), f'from moplayground import {alias.name}'
    missing = {n for n in _used_names(tree, 'mop') if not hasattr(mop, n)}
    assert not missing, f'mop.{missing} do not exist'
    missing = {n for n in _used_names(tree, 'cfg') if not hasattr(mop.MOConfig, n) and n != 'cfg'}
    assert not missing, f'MOConfig has no {missing}'
