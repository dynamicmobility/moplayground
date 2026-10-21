import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from tests.helpers import REPO_ROOT, make_tiny_config

ALGORITHMS = ('morlax', 'amor', 'ppo')


@dataclass
class TrainedRun:
    algorithm: str
    config_path: Path
    returncode: int
    log: str

    @property
    def cfg(self):
        import moplayground as mop
        return mop.MOConfig(self.config_path)


@pytest.fixture(scope='session', autouse=True)
def _training_procs(tmp_path_factory):
    """Start a tiny MOCheetah training run per algorithm with ``scripts/train.py``.

    The runs are JIT-bound, so they go in parallel subprocesses (each one
    through the real CLI entry point, wandb disabled) that start with the
    session and overlap the in-process GPU tests; ``trained_runs`` waits.
    """
    env = {**os.environ, 'WANDB_MODE': 'disabled', 'XLA_PYTHON_CLIENT_PREALLOCATE': 'false'}
    env.pop('JAX_PLATFORMS', None)
    procs = {}
    for algorithm in ALGORITHMS:
        save_dir = tmp_path_factory.mktemp(algorithm)
        config_path = make_tiny_config(algorithm, save_dir)
        log = open(save_dir / 'train.log', 'w')
        proc = subprocess.Popen(
            [sys.executable, '-m', 'scripts.train', str(config_path)],
            cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        procs[algorithm] = (proc, config_path, log)
    yield procs
    for proc, _, _ in procs.values():
        if proc.poll() is None:
            proc.kill()


@pytest.fixture(scope='session')
def trained_runs(_training_procs):
    runs = {}
    for algorithm, (proc, config_path, log) in _training_procs.items():
        returncode = proc.wait(timeout=900)
        log.close()
        runs[algorithm] = TrainedRun(algorithm, config_path, returncode, Path(log.name).read_text())
    return runs


@pytest.fixture(params=ALGORITHMS)
def trained_run(request, trained_runs):
    run = trained_runs[request.param]
    if run.returncode != 0:
        pytest.fail(f'{run.algorithm} training failed:\n' + '\n'.join(run.log.splitlines()[-40:]))
    return run


@pytest.fixture(params=('morlax', 'amor'))
def trained_mo_run(request, trained_runs):
    run = trained_runs[request.param]
    if run.returncode != 0:
        pytest.fail(f'{run.algorithm} training failed:\n' + '\n'.join(run.log.splitlines()[-40:]))
    return run
