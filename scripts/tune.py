"""Hyperparameter sweep driver (wandb sweeps, e.g. Bayesian optimisation).

Reads a sweep YAML: a wandb sweep config plus `base_config` (path to the base
training YAML) and `num_trials`. Sweep parameters are nested to match the
training config layout, so each sampled value has the same key path in the
wandb run config as in the training config (no duplicate keys). Each sweep
trial writes its sampled values into a fresh copy of the base config and
trains in-process. The run is named `{base_name}-{wandb run id}`.

Usage:
    python -m scripts.tune config/sweeps/mocheetah-morlax-bayes.yaml
"""

import matplotlib
matplotlib.use('Agg')

import argparse

import wandb
import yaml

import minimal_mjx as mm
import moplayground as mop
from scripts.train import get_handle_params, get_progress

ENTITY  = 'njanwani-gatech'
PROJECT = 'MO-Playground-2'


def define_sweep_metric(run, metric):
    """Registers the sweep objective when it names a summary aggregate, e.g.
    'pareto/hypervolume.max'. wandb otherwise summarizes a metric by its last
    logged value, which scores a fluctuating objective on wherever it happened to end.
    """
    if metric is None:
        return
    name, _, aggregate = metric['name'].rpartition('.')
    if aggregate in ('min', 'max', 'mean'):
        run.define_metric(name, summary=aggregate)


def apply_overrides(section, overrides, prefix='', errors=None):
    """Writes each value in a nested dict into ``section`` at the same key path.

    Raises ValueError if a key path does not exist (e.g. a typo in a sweep file).
    """
    top = errors is None
    errors = [] if top else errors
    for key, value in overrides.items():
        path = f'{prefix}{key}'
        if key not in section:
            errors.append(f"key '{path}' does not exist in the config")
        elif isinstance(value, dict):
            apply_overrides(section[key], value, f'{path}.', errors)
        else:
            section[key] = value
    if top and errors:
        raise ValueError("Invalid sweep parameters:\n  " + "\n  ".join(errors))


def train(cfg: mop.MOConfig, run):
    """Builds the environments for ``cfg`` and trains into ``cfg.run_dir``
    (mirrors scripts/train.py::main)."""
    TRAIN_KWARGS = {}
    EVAL_KWARGS  = {}

    # Environment-specific config handling...
    match cfg.env_name:
        case 'BRUCE':
            EVAL_KWARGS = {'manual_speed': [0.0, 0.0, 0.0], 'idealistic': True}

    env_kwargs = dict(
        env_name     = cfg.env_name,
        backend      = 'jnp',
    )
    env, _      = mop.create_environment(env_params=cfg.env_params, **env_kwargs, **TRAIN_KWARGS)
    eval_env, _ = mop.create_environment(env_params=cfg.env_params, **env_kwargs, **EVAL_KWARGS)
    if cfg.algorithm == 'ppo':
        env         = mop.Multi2SingleObjective(env, cfg.default_scalarization)
        eval_env    = mop.Multi2SingleObjective(eval_env, cfg.default_scalarization)

    handle_params, train_fn_params, network_factory_params = get_handle_params(cfg)

    mm.train(
        env                     = env,
        eval_env                = eval_env,
        run_dir                 = cfg.run_dir,
        handle_params           = handle_params,
        train_fn_params         = train_fn_params,
        network_factory_params  = network_factory_params,
        progress_cls            = get_progress(cfg),
        run                     = run,
        resume                  = False,
        overwrite               = cfg.name == 'test'
    )


def run_sweep(wandb_sweep_config, base_config_path, count=3):
    """Runs a hyperparameter sweep"""

    def edit_and_train():
        """Trains one trial of the sweep on a copy of the base config."""
        with wandb.init() as run:
            define_sweep_metric(run, wandb_sweep_config.get('metric'))

            cfg = mop.MOConfig(base_config_path)
            apply_overrides(cfg.cfg, dict(run.config))

            cfg.name = f"{cfg.name}-{run.id}"
            run.name = cfg.name

            config_path = cfg.run_dir / 'config.yaml'
            cfg.save(config_path, warn_github_changes=False)

            # Publish all of the train config, not just the swept params, so we
            # can rebuild the whole config from the run later.
            run.config.update(cfg.cfg.to_dict(), allow_val_change=True)
            run.log_artifact(str(config_path), name='config')

            train(cfg, run)

    sweep_id = wandb.sweep(sweep=wandb_sweep_config, entity=ENTITY, project=PROJECT)
    wandb.agent(sweep_id, edit_and_train, entity=ENTITY, project=PROJECT, count=count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_path", type=str, help="Sweep YAML to run")
    args = parser.parse_args()

    with open(args.sweep_path, 'r') as f:
        sweep = yaml.safe_load(f)
    # Keys for this script, not for wandb.
    base_config_path = sweep.pop('base_config')
    num_trials       = sweep.pop('num_trials')
    run_sweep(sweep, base_config_path, count=num_trials)
