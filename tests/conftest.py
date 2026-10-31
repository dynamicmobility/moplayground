"""Shared fixtures for the moplayground test suite.

Two tiers:

* fast (default): ``pytest`` -- CPU-only unit tests. ``pyproject.toml`` adds
  ``-m "not gpu"`` so GPU tests are deselected.
* gpu: ``pytest -m gpu`` -- end-to-end training / inference / evaluation on a
  CUDA device. Skipped (not failed) when JAX cannot see a GPU.

Nothing that imports JAX may be imported at module level here: the fast tier
pins JAX to the CPU in ``pytest_configure``, which must run first.
"""
import os
from pathlib import Path

import pytest

from tests.helpers import REPO_ROOT, TINY_BASE_CONFIG


def pytest_addoption(parser):
    parser.addoption(
        '--config-dir',
        default=str(REPO_ROOT / 'config'),
        help='Directory of training configs that MOConfig must be able to load '
             '(sweep configs are read from its sweeps/ subdirectory).',
    )


def pytest_configure(config):
    if 'gpu' not in (config.option.markexpr or '').replace('not gpu', ''):
        # Fast tier: keep JAX off the GPU so unit tests are deterministic and
        # do not contend with training jobs for device memory.
        os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    else:
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('WANDB_MODE', 'disabled')
    os.environ.setdefault('MPLBACKEND', 'Agg')


@pytest.hookimpl(trylast=True)  # after -m has deselected items
def pytest_collection_modifyitems(config, items):
    gpu_items = [item for item in items if 'gpu' in item.keywords]
    if not gpu_items:
        return
    import jax
    if any(d.platform == 'gpu' for d in jax.devices()):
        return
    skip = pytest.mark.skip(reason='no GPU visible to JAX')
    for item in gpu_items:
        item.add_marker(skip)


@pytest.fixture(scope='session')
def config_dir(pytestconfig) -> Path:
    return Path(pytestconfig.getoption('--config-dir')).resolve()


@pytest.fixture(scope='session')
def cheetah_cfg():
    import moplayground as mop
    return mop.MOConfig(TINY_BASE_CONFIG)
