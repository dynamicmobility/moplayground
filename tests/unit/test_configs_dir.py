"""Every config in ``--config-dir`` (default ``config/``) must be loadable by
``MOConfig`` as it stands: each property the run's algorithm needs resolves,
every key in the YAML is one something reads, and the values are consistent
with each other and with the functions they are handed to.

Point it at another directory with ``pytest tests/unit/test_configs_dir.py
--config-dir path/to/configs``.
"""
import inspect
from pathlib import Path

import pytest
import yaml

import moplayground as mop
from moplayground.utils.config import MOConfig

ALGORITHMS = ('morlax', 'amor', 'ppo')
ENVS = ('MOCheetah', 'MOHopper', 'MOAnt', 'MOWalker', 'MOHumanoid', 'NaviGait')

# MOConfig properties every config must resolve, whatever its algorithm.
# ``network_params`` is inherited from mm.Config but this repo keeps one block
# per algorithm instead, so it is never read.
UNUSED_PROPERTIES = {'network_params'}
NETWORK_PROPERTY = {
    'morlax': 'morlax_network_params',
    'amor': 'amor_network_params',
    'ppo': 'ppo_network_params',
}
ALGORITHM_ONLY = set(NETWORK_PROPERTY.values()) | {'sampling_params'}

# Keys MOConfig/mm.Config read (or write, like git_hash). Anything else is
# stale. Keep in sync with config.py and the layout in CLAUDE.md.
TOP_LEVEL_KEYS = {
    'name', 'save_dir', 'description', 'algorithm', 'env', 'backend',
    'gaitlib_path', 'env_config', 'learning_params', 'git_hash',
}
LEARNING_PARAMS_KEYS = {
    'ppo_params', 'sampling_params', 'morlax_network_params',
    'amor_network_params', 'ppo_network_params', 'morlax_warmup_params',
}


def _config_paths(config_dir: Path):
    return sorted(p for p in config_dir.rglob('*.yaml') if 'sweeps' not in p.relative_to(config_dir).parts)


def _sweep_paths(config_dir: Path):
    return sorted((config_dir / 'sweeps').glob('*.yaml'))


def pytest_generate_tests(metafunc):
    config_dir = Path(metafunc.config.getoption('--config-dir')).resolve()
    if 'config_path' in metafunc.fixturenames:
        paths = _config_paths(config_dir)
        metafunc.parametrize('config_path', paths, ids=[str(p.relative_to(config_dir)) for p in paths])
    if 'sweep_path' in metafunc.fixturenames:
        paths = _sweep_paths(config_dir)
        metafunc.parametrize('sweep_path', paths, ids=[p.name for p in paths])


def _properties(cls):
    return {
        name for klass in cls.__mro__ for name, attr in vars(klass).items()
        if isinstance(attr, property)
    }


def _accepted_kwargs(fn):
    return set(inspect.signature(fn).parameters)


def _train_fn(algorithm):
    if algorithm == 'ppo':
        from brax.training.agents.ppo.train import train
        return train
    return {'morlax': mop.train_morlax, 'amor': mop.train_amor}[algorithm]


def _network_factory(algorithm):
    if algorithm == 'ppo':
        from brax.training.agents.ppo.networks import make_ppo_networks
        return make_ppo_networks
    return {'morlax': mop.make_morlax_networks, 'amor': mop.make_amor_networks}[algorithm]


def check_loadable(cfg: MOConfig, repo_root: Path):
    """Assert ``cfg`` can drive a run of its configured algorithm."""
    assert cfg.algorithm in ALGORITHMS, f'unknown algorithm {cfg.algorithm!r}'
    assert cfg.env_name in ENVS, f'unknown env {cfg.env_name!r}'

    # Every property resolves (except other algorithms' blocks).
    required = _properties(MOConfig) - UNUSED_PROPERTIES - ALGORITHM_ONLY
    required.add(NETWORK_PROPERTY[cfg.algorithm])
    if cfg.algorithm != 'ppo':
        required.add('sampling_params')
    for name in sorted(required):
        try:
            getattr(cfg, name)
        except Exception as e:  # noqa: BLE001 -- report which property broke
            pytest.fail(f'MOConfig.{name} failed: {type(e).__name__}: {e}')

    # No stale keys.
    raw = cfg.cfg.to_dict()
    assert set(raw) <= TOP_LEVEL_KEYS, f'unknown top-level keys {sorted(set(raw) - TOP_LEVEL_KEYS)}'
    lp = set(raw['learning_params'])
    assert lp <= LEARNING_PARAMS_KEYS, f'unknown learning_params keys {sorted(lp - LEARNING_PARAMS_KEYS)}'

    # Values line up with each other.
    n = cfg.num_objectives
    assert n >= 2, 'multi-objective configs need at least two objectives'
    assert len(cfg.default_scalarization) == n, 'default_scalarization length != number of objectives'
    if cfg.objective_labels is not None:
        assert len(cfg.objective_labels) == n, 'labels length != number of objectives'
    weights = set(raw['env_config']['reward']['weights'])
    opt = raw['env_config']['reward']['optimization']
    used = {k for objective in opt['objectives'] for k in objective} | set(opt.get('shared_objectives', []))
    assert used <= weights, f'objective reward keys missing from reward.weights: {sorted(used - weights)}'
    if cfg.env_name == 'NaviGait':
        assert cfg.gaitlib_path is not None, 'NaviGait needs gaitlib_path'
    if cfg.gaitlib_path is not None:
        assert (repo_root / cfg.gaitlib_path).is_dir(), f'gaitlib_path {cfg.gaitlib_path} does not exist'

    # Values are accepted by the functions scripts/train.py hands them to.
    train_kwargs = set(cfg.ppo_params) | (set(cfg.sampling_params) if cfg.algorithm != 'ppo' else set())
    bad = train_kwargs - _accepted_kwargs(_train_fn(cfg.algorithm))
    assert not bad, f'{cfg.algorithm} train fn does not accept {sorted(bad)}'
    net_kwargs = set(getattr(cfg, NETWORK_PROPERTY[cfg.algorithm]))
    bad = net_kwargs - _accepted_kwargs(_network_factory(cfg.algorithm))
    assert not bad, f'{cfg.algorithm} network factory does not accept {sorted(bad)}'
    if cfg.algorithm != 'ppo' and cfg.sampling_params['sampling'].startswith('sparse'):
        # sample_preferences tiles k directives across the envs
        assert cfg.ppo_params['num_envs'] % cfg.sampling_params['k'] == 0, \
            'num_envs must be a multiple of sampling_params.k'


def test_config_dir_has_configs(config_dir):
    assert config_dir.is_dir(), f'{config_dir} is not a directory'
    assert _config_paths(config_dir), f'no *.yaml configs under {config_dir}'


def test_config_is_loadable(config_path, config_dir):
    check_loadable(MOConfig(config_path), config_dir.parent)


def _leaves(parameters, prefix=()):
    """``(key_path, spec)`` for every swept value in a nested wandb ``parameters`` block."""
    for key, spec in parameters.items():
        if 'parameters' in spec:
            yield from _leaves(spec['parameters'], prefix + (key,))
        else:
            yield prefix + (key,), spec


def _choices(spec):
    if 'value' in spec:
        return [spec['value']]
    if 'values' in spec:
        return list(spec['values'])
    return [spec['min'], spec['max']]


def _sweep_overrides(parameters):
    """A baseline override (first choice everywhere) plus one variant per
    alternative choice, so each categorical value is checked at least once."""
    leaves = list(_leaves(parameters))
    baseline = {path: _choices(spec)[0] for path, spec in leaves}
    variants = [baseline]
    for path, spec in leaves:
        for choice in _choices(spec)[1:]:
            variants.append({**baseline, path: choice})
    for flat in variants:
        nested = {}
        for path, value in flat.items():
            node = nested
            for key in path[:-1]:
                node = node.setdefault(key, {})
            node[path[-1]] = value
        yield nested


def test_sweep_is_loadable(sweep_path, config_dir):
    from scripts.tune import apply_overrides

    with open(sweep_path) as f:
        sweep = yaml.safe_load(f)
    for key in ('base_config', 'num_trials', 'method', 'parameters'):
        assert key in sweep, f'sweep is missing {key!r}'
    repo_root = config_dir.parent
    base = repo_root / sweep['base_config']
    assert base.is_file(), f'base_config {base} does not exist'

    # Each swept key path must exist in the base config, and the result must
    # still be a loadable config (e.g. switching algorithm needs its block).
    for overrides in _sweep_overrides(sweep['parameters']):
        cfg = MOConfig(base)
        apply_overrides(cfg.cfg, overrides)
        check_loadable(cfg, repo_root)
