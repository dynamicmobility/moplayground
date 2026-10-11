"""Single access point between YAML config files and the ``moplayground`` package.

This is the only module that knows YAML key names. Scripts call :func:`load`
to read a config, then use the getter functions to pull plain values out of
it and pass those values to package functions. Package code never indexes a
config directly.

Canonical layout::

    name, save_dir, description, algorithm, env, backend, gaitlib_path
    env_config: {...}
    learning_params:
      ppo_params:           {learning_rate, num_envs, ...}
      sampling_params:      {alpha, k, sampling, warmup_frac}
      morlax_params:        {hypertype, hypersize, num_features,
                             policy_hidden_layer_sizes, value_hidden_layer_sizes}
      amor_params:          {policy_hidden_layer_sizes, value_hidden_layer_sizes}
      morlax_warmup_params: {enabled, policy}

:func:`load` renames keys from older layouts to the canonical layout, so saved
``config.yaml`` files from old runs still load.
"""
import copy
import os
from pathlib import Path

import yaml
from ml_collections import config_dict
import minimal_mjx as mm

ALGORITHMS     = ('morlax', 'amor')
ENVS           = ('MOCheetah', 'MOHopper', 'MOAnt', 'MOWalker', 'MOHumanoid', 'NaviGait')
SAMPLING_MODES = ('dense', 'sparse', 'sparse-heavytail', 'single-avg')
HYPERTYPES     = ('single', 'dual')

_SAMPLING_KEYS   = ('alpha', 'k', 'sampling', 'warmup_frac')
_MORLAX_NET_KEYS = ('hypertype', 'hypersize', 'num_features',
                    'policy_hidden_layer_sizes', 'value_hidden_layer_sizes')
_AMOR_NET_KEYS   = ('policy_hidden_layer_sizes', 'value_hidden_layer_sizes')


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------
import yaml
import sys
from ml_collections import config_dict
import copy

class FlowSeqDumper(yaml.Dumper):
    def represent_sequence(self, tag, sequence, flow_style=None):
        # Force all sequences (lists) to use flow style
        return super().represent_sequence(tag, sequence, flow_style=True)

def read_yaml(yaml_file):
    try:
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)
    except Exception as e:
        print(f"Error reading {yaml_file}: {e}")
        sys.exit(1) 
    return data

def read_config(path=None):
    """Reads the YAML config file"""
    if len(sys.argv) != 2 and path is None:
        print("Usage: python script.py <yaml_file>")
        sys.exit(1)
    
    yaml_file = sys.argv[1] if path is None else path
    data = read_yaml(yaml_file)
    
    return config_dict.ConfigDict(data)

class Config:
    
    def __init__(self, config: Path):
        self.cfg = read_config(config)
        
    def load(self, config):
        self.cfg = read_config(config)
        return self.cfg
    
    def save(self, path: Path, make_dir=True, rewrite_test=True, warn_github_changes=True):
        if make_dir:
            run_dir = Path(run_dir)
            os.makedirs(run_dir, exist_ok=self.cfg.name == 'test')
        
        if self.cfg.name != 'test' and rewrite_test:
            self.cfg.git_hash = mm.utils.config.get_commit_hash(warn=warn_github_changes)
    
        with open(path, 'w') as f:
            yaml.dump(self.cfg.to_dict(), f)
    
    @property
    def algorithm(self) -> str:
        """Learning algorithm, like ``MORLAX``"""
        return self.cfg.algorithm

    @property
    def env_name(self) -> str:
        """Environment class name, e.g. ``'MOCheetah'``."""
        return self.cfg.env

    @property
    def backend(self) -> str:
        """``'jnp'`` when building an env for training, else the configured backend."""
        return self.cfg.backend

    @property
    def env_params(self) -> config_dict.ConfigDict:
        """The ``env_config`` section, given to env classes as ``env_params``."""
        return mm.create_config_dict(self.cfg.env_config.to_dict())

    @property
    def ppo_params(self) -> dict:
        """PPO settings shared by both algorithms."""
        return self.cfg.learning_params.ppo_params.to_dict()

    @property
    def network_params(self) -> dict:
        """Network settings for the configured algorithm."""
        return self.cfg.learning_params.network_params.to_dict()

    @property
    def normalize_observations(self) -> bool:
        """Whether training keeps a running observation normalizer."""
        return self.cfg.learning_params.ppo_params.normalize_observations

    @property
    def name(self) -> str:
        """Run name."""
        return self.cfg.name

    @property
    def save_dir(self) -> Path:
        """Parent directory of all runs for this config."""
        return Path(self.cfg.save_dir)

    @property
    def run_dir(self) -> Path:
        """Run directory: ``save_dir / name``."""
        return self.save_dir / self.name
        

class MOConfig(Config):

    @property
    def sampling_params(self) -> dict:
        """Sampling settings: ``alpha``, ``k``, ``sampling``, ``warmup_frac``."""
        return self.cfg.learning_params.sampling_params.to_dict()
    
    @property
    def network_params(self) -> dict:
        """Network settings for the configured algorithm."""
        return self.cfg.learning_params[f'{self.algorithm}_params'].to_dict()
    
    @property
    def objectives(self) -> list:
        """Reward keys per objective, one list per reward dimension."""
        return list(self.cfg.env_config.reward.optimization.objectives)

    @property
    def num_objectives(self) -> int:
        """Number of objectives (length of the reward vector)."""
        return len(self.cfg.env_config.reward.optimization.objectives)

    @property
    def objective_labels(self):
        """Display names for the objectives, or ``None`` if not set."""
        labels = self.cfg.env_config.reward.optimization.get('labels')
        return None if labels is None else list(labels)



def check_same_env(cfgs):
    """Raise ``ValueError`` unless every config deploys the same environment.

    Compares ``env`` and the full ``env_config`` section. Use this before
    evaluating several runs together (e.g. ``eval.pareto.get_morlax_fronts``
    with a list of run directories).
    """
    cfgs = list(cfgs)
    if len(cfgs) <= 1:
        return
    ref = cfgs[0]
    for i, c in enumerate(cfgs[1:], start=1):
        if c.env != ref.env:
            raise ValueError(
                f"Config[{i}] env={c.env!r} does not match config[0] env={ref.env!r}; "
                f"all configs must deploy the same environment."
            )
        if c.env_config.to_dict() != ref.env_config.to_dict():
            raise ValueError(
                f"Config[{i}] env_config differs from config[0]; "
                f"all configs must deploy the same environment."
            )


# ---------------------------------------------------------------------------
# Setters (used by scripts/ablation.py and scripts/tune.py)
# ---------------------------------------------------------------------------

def set_name(cfg, value):
    cfg.name = value


def set_hypertype(cfg, value):
    cfg.learning_params.morlax_params.hypertype = value


def set_sampling(cfg, value):
    cfg.learning_params.sampling_params.sampling = value


def set_k(cfg, value):
    cfg.learning_params.sampling_params.k = value


def apply_overrides(cfg, overrides, where='overrides'):
    """Write each value in a nested dict into ``cfg`` at the same key path.

    ``overrides`` mirrors the config layout, e.g.
    ``{'learning_params': {'ppo_params': {'learning_rate': 3e-4}}}``. A wandb
    sweep with nested ``parameters`` gives its sampled values in this form.

    Args:
        cfg: Config returned by :func:`load`. Modified in place.
        overrides: Nested dict of values to write.
        where: Name for ``overrides`` in error messages.

    Raises:
        ValueError: If a key path in ``overrides`` does not exist in ``cfg``
            (e.g. a typo in a sweep file), or if the result fails validation.
    """
    errors = []
    _apply_overrides(cfg, overrides, '', errors)
    if errors:
        _raise(where, errors)
    _validate(cfg, where)


def _apply_overrides(section, overrides, prefix, errors):
    for key, value in overrides.items():
        path = f'{prefix}{key}'
        if key not in section:
            errors.append(f"key '{path}' does not exist in the config")
        elif isinstance(value, dict):
            _apply_overrides(section[key], value, f'{path}.', errors)
        else:
            section[key] = value


# ---------------------------------------------------------------------------
# Old layout -> canonical layout
# ---------------------------------------------------------------------------

def _migrate(raw: dict) -> dict:
    """Return a copy of ``raw`` in the canonical layout.

    Handles three older layouts:

    1. ``base_ppo_params``; ``morlax_params`` holds sampling keys; top-level
       ``network_params`` holds the MORLAX network; ``amor_params`` holds
       ``train_fn_params`` and ``network_params``.
    2. Like 1, but ``morlax_params`` holds ``train_fn_params``,
       ``network_params`` and ``warmup_params``.
    3. ``hypermorl_params`` (sampling), ``hypernetwork_params`` (hypertype,
       hypersize, num_features), ``network_params`` (MORLAX layer sizes),
       ``warmup_params``; ``algorithm`` is ``ppo`` or missing.

    Also renames the old hypertype ``ActorCritic`` to ``dual``.
    Sampling settings are taken from the block of the configured algorithm.
    """
    cfg = copy.deepcopy(raw)

    # Layout 3 used 'ppo' (or nothing) for the hypernetwork algorithm.
    if cfg.get('algorithm') in (None, 'ppo'):
        cfg['algorithm'] = 'morlax'
    algo = cfg['algorithm']

    lp = cfg.get('learning_params')
    if lp is None or 'sampling_params' in lp:
        return cfg

    lp = dict(lp)
    morlax_net = morlax_sampling = amor_net = amor_sampling = warmup = None

    ppo = lp.pop('ppo_params', None)
    if 'base_ppo_params' in lp:
        ppo = lp.pop('base_ppo_params')

    if 'hypernetwork_params' in lp or 'hypermorl_params' in lp:
        # Layout 3
        morlax_net      = {**lp.pop('hypernetwork_params', {}), **lp.pop('network_params', {})}
        morlax_sampling = lp.pop('hypermorl_params', None)
        warmup          = lp.pop('warmup_params', None)
    else:
        morlax = lp.pop('morlax_params', None) or {}
        if 'train_fn_params' in morlax:
            # Layout 2
            morlax_sampling = morlax.get('train_fn_params')
            morlax_net      = morlax.get('network_params')
            warmup          = morlax.get('warmup_params')
        else:
            # Layout 1
            morlax_sampling = morlax or None
            morlax_net      = lp.pop('network_params', None)
            warmup          = lp.pop('morlax_warmup_params', None)

    amor = lp.pop('amor_params', None)
    if amor is not None:
        amor_sampling = amor.get('train_fn_params')
        amor_net      = amor.get('network_params')

    # Commit 899b126 renamed hypertype 'ActorCritic' (DualA2CHypernet) to 'dual'.
    if morlax_net is not None and morlax_net.get('hypertype') == 'ActorCritic':
        morlax_net = {**morlax_net, 'hypertype': 'dual'}

    new_lp = {}
    if ppo is not None:
        new_lp['ppo_params'] = ppo
    sampling = amor_sampling if algo == 'amor' else morlax_sampling
    if sampling is not None:
        new_lp['sampling_params'] = sampling
    if morlax_net is not None:
        new_lp['morlax_params'] = morlax_net
    if amor_net is not None:
        new_lp['amor_params'] = amor_net
    if warmup is not None:
        # TODO: nothing reads morlax_warmup_params yet; implement policy warmup or remove it.
        new_lp['morlax_warmup_params'] = warmup
    new_lp.update(lp)  # keep any keys this function does not know about

    cfg['learning_params'] = new_lp
    return cfg


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate(cfg, path):
    errors = []

    for key in ('name', 'save_dir', 'env', 'backend', 'env_config', 'learning_params'):
        if key not in cfg:
            errors.append(f"missing top-level key '{key}'")
    if errors:
        _raise(path, errors)

    if cfg.algorithm not in ALGORITHMS:
        errors.append(f"algorithm '{cfg.algorithm}' is not one of {ALGORITHMS}")
    if cfg.env not in ENVS:
        errors.append(f"env '{cfg.env}' is not one of {ENVS}")
    if cfg.env == 'NaviGait' and not cfg.get('gaitlib_path'):
        errors.append("env 'NaviGait' needs 'gaitlib_path'")
    if cfg.backend not in ('np', 'jnp'):
        errors.append(f"backend '{cfg.backend}' is not 'np' or 'jnp'")

    lp = cfg.learning_params
    if 'ppo_params' not in lp:
        errors.append("missing 'learning_params.ppo_params'")

    if 'sampling_params' not in lp:
        errors.append("missing 'learning_params.sampling_params'")
    else:
        _check_keys(errors, lp.sampling_params, _SAMPLING_KEYS, 'learning_params.sampling_params')
        sampling = lp.sampling_params.get('sampling')
        if sampling is not None and sampling not in SAMPLING_MODES:
            errors.append(f"sampling '{sampling}' is not one of {SAMPLING_MODES}")

    if cfg.algorithm == 'morlax':
        if 'morlax_params' not in lp:
            errors.append("algorithm 'morlax' needs 'learning_params.morlax_params'")
        else:
            _check_keys(errors, lp.morlax_params, _MORLAX_NET_KEYS, 'learning_params.morlax_params')
            hypertype = lp.morlax_params.get('hypertype')
            if hypertype is not None and hypertype not in HYPERTYPES:
                errors.append(f"hypertype '{hypertype}' is not one of {HYPERTYPES}")
    elif cfg.algorithm == 'amor':
        if 'amor_params' not in lp:
            errors.append("algorithm 'amor' needs 'learning_params.amor_params'")
        else:
            _check_keys(errors, lp.amor_params, _AMOR_NET_KEYS, 'learning_params.amor_params')

    reward = cfg.env_config.get('reward')
    opt = reward.get('optimization') if reward is not None else None
    if opt is None or 'objectives' not in opt:
        errors.append("missing 'env_config.reward.optimization.objectives'")
    else:
        weights = reward.get('weights') or {}
        used = [k for group in opt.objectives for k in group]
        used += list(opt.get('shared_objectives') or [])
        for k in used:
            if k not in weights:
                errors.append(f"objective key '{k}' is not in 'env_config.reward.weights'")
        labels = opt.get('labels')
        if labels is not None and len(labels) != len(opt.objectives):
            errors.append(
                f"{len(labels)} labels for {len(opt.objectives)} objectives "
                "in 'env_config.reward.optimization'"
            )

    if errors:
        _raise(path, errors)


def _check_keys(errors, section, keys, where):
    for k in keys:
        if k not in section:
            errors.append(f"missing '{where}.{k}'")


def _raise(path, errors):
    lines = '\n  - '.join(errors)
    raise ValueError(f"Invalid config '{path}':\n  - {lines}")
