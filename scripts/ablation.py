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
    morlax = cfg.learning_params.morlax_params
    morlax.network_params.hypertype = hypertype
    morlax.train_fn_params.sampling = sampling
    morlax.train_fn_params.k = k

    base_name = cfg.name
    cfg.name = f"{base_name}-h={hypertype}-s={sampling}-k={k}"
    return cfg


def run_one(cfg):
    env, _      = mop.envs.create_environment(cfg, for_training=True)
    eval_env, _ = mop.envs.create_environment(cfg, for_training=True)
    name = cfg.save_dir + '/' + cfg.name
    run = mm.utils.logging.initialize_wandb(
        name    = name.replace('/', ''),
        entity  = 'njanwani-gatech',
        project = 'PrefMORL',
    )
    try:
        mop.learning.train_policy(cfg, env, eval_env, run, warn_github_changes=False)
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

    base_config = mop.utils.read_config(args.base)
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
        run_path = os.path.join(cfg.save_dir, cfg.name)
        if args.skip_existing and os.path.isdir(run_path) and os.listdir(run_path):
            print(f'[skip] {cfg.name} (exists at {run_path})')
            results.append((cfg.name, 'skipped'))
            continue

        print(f'\n===== Running {cfg.name} =====')
        try:
            run_one(cfg)
            results.append((cfg.name, 'ok'))
        except Exception as e:
            print(f'[FAIL] {cfg.name}: {e}')
            traceback.print_exc()
            results.append((cfg.name, f'fail: {e}'))

    print('\n===== Sweep summary =====')
    for name, status in results:
        print(f'  {status:<10} {name}')


if __name__ == '__main__':
    main()
