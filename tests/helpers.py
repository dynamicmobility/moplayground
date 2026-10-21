"""Constants and helpers shared by tests (importable, unlike conftest)."""
import copy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

# Base config the tiny training fixtures are derived from, so they track the
# real config layout instead of drifting as a hand-copied YAML.
TINY_BASE_CONFIG = REPO_ROOT / 'config' / 'mocheetah.yaml'

TINY_PPO_PARAMS = dict(
    num_envs=16,
    num_eval_envs=16,
    batch_size=8,
    num_minibatches=2,
    unroll_length=10,
    episode_length=50,
    num_updates_per_batch=1,
    num_evals=2,
    num_timesteps=4000,
)
TINY_SAMPLING_PARAMS = dict(k=8, sampling='sparse-heavytail', warmup_frac=0.0)
TINY_MLP = dict(policy_hidden_layer_sizes=[8, 8], value_hidden_layer_sizes=[8, 8])
TINY_MORLAX_PARAMS = dict(hypertype='dual', hypersize=[8], num_features=4, **TINY_MLP)


def make_tiny_config(algorithm: str, save_dir: Path) -> Path:
    """Write a tiny MOCheetah training config for ``algorithm`` into ``save_dir``."""
    with open(TINY_BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg['name'] = 'test'
    cfg['save_dir'] = str(save_dir)
    cfg['algorithm'] = algorithm
    lp = cfg['learning_params']
    lp['ppo_params'].update(TINY_PPO_PARAMS)
    lp['sampling_params'].update(TINY_SAMPLING_PARAMS)
    lp['morlax_network_params'].update(TINY_MORLAX_PARAMS)
    lp['amor_network_params'].update(TINY_MLP)
    lp['ppo_network_params'].update(TINY_MLP)
    path = save_dir / f'tiny_{algorithm}.yaml'
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f)
    return path
