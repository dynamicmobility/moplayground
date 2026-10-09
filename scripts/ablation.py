"""Sweep driver for the MORLAX ablation study.

Loads a base YAML config, then iterates over the cartesian product of
{hypertype, sampling, k}, overriding the corresponding fields in a
deep-copied config and running `train_policy` in-process for each combo.

Usage:
    python -m scripts.ablation --base config/mocheetah.yaml \
        --hypertypes single,dual \
        --samplings dense,sparse-heavytail \
        --ks 4,8,16
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import copy
import itertools
import os
import traceback

import wandb

import moplayground as mop
from moplayground import config
import minimal_mjx as mm


def parse_csv(s, cast=str):
    return [cast(x.strip()) for x in s.split(',') if x.strip()]


def build_combos(hypertypes, samplings, ks):
    combos = []
    seen = set()
    for h, s, k in itertools.product(hypertypes, samplings, ks):
        # `dense` ignores k — dedupe so we only run dense once per hypertype.
        key = (h, s, k if s != 'dense' else None)
        if key in seen:
            continue
        seen.add(key)
        combos.append((h, s, k))
    return combos


def apply_overrides(base_config, hypertype, sampling, k):
    cfg = copy.deepcopy(base_config)
    config.set_hypertype(cfg, hypertype)
    config.set_sampling(cfg, sampling)
    config.set_k(cfg, k)

    base_name = config.run_name(cfg)
    config.set_name(cfg, f"{base_name}-h={hypertype}-s={sampling}-k={k}")
    return cfg


def run_one(cfg):
    env_kwargs = dict(
        env_name     = config.env_name(cfg),
        backend      = config.backend(cfg, for_training=True),
        gaitlib_path = config.gaitlib_path(cfg),
    )
    env, _      = mop.create_environment(env_params=config.env_params(cfg), **env_kwargs)
    eval_env, _ = mop.create_environment(env_params=config.env_params(cfg), **env_kwargs)

    run_dir = config.run_dir(cfg)
    config_path = config.save(cfg, run_dir, warn_github_changes=False)
    run = mm.utils.logging.initialize_wandb(
        name    = str(run_dir).replace('/', ''),
        entity  = 'njanwani-gatech',
        project = 'PrefMORL',
        config  = cfg.to_dict(),
    )
    try:
        run.log_artifact(str(config_path), name='config')
        mop.train_policy(
            algorithm       = config.algorithm(cfg),
            ppo_params      = config.ppo_params(cfg),
            sampling_params = config.sampling_params(cfg),
            network_params  = config.network_params(cfg),
            run_dir         = run_dir,
            env             = env,
            eval_env        = eval_env,
            run             = run,
        )
    finally:
        try:
            wandb.finish()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, required=True,
                        help='Path to base YAML config.')
    parser.add_argument('--hypertypes', type=str, default='single,dual')
    parser.add_argument('--samplings',  type=str, default='dense,sparse-heavytail')
    parser.add_argument('--ks',         type=str, default='4,8,16')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip combo if save_dir/name already exists.')
    args = parser.parse_args()

    base_config = config.load(args.base)
    hypertypes  = parse_csv(args.hypertypes)
    samplings   = parse_csv(args.samplings)
    ks          = parse_csv(args.ks, cast=int)

    combos = build_combos(hypertypes, samplings, ks)
    print(f'Sweep: {len(combos)} combos')
    for c in combos:
        print(f'  hypertype={c[0]} sampling={c[1]} k={c[2]}')

    results = []
    for hypertype, sampling, k in combos:
        cfg = apply_overrides(base_config, hypertype, sampling, k)
        name = config.run_name(cfg)
        run_path = config.run_dir(cfg)
        if args.skip_existing and os.path.isdir(run_path) and os.listdir(run_path):
            print(f'[skip] {name} (exists at {run_path})')
            results.append((name, 'skipped'))
            continue

        print(f'\n===== Running {name} =====')
        try:
            run_one(cfg)
            results.append((name, 'ok'))
        except Exception as e:
            print(f'[FAIL] {name}: {e}')
            traceback.print_exc()
            results.append((name, f'fail: {e}'))

    print('\n===== Sweep summary =====')
    for name, status in results:
        print(f'  {status:<10} {name}')


if __name__ == '__main__':
    main()
