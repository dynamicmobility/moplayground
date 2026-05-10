"""Plot a MORLAX ablation sweep.

Given an ablation folder (e.g. `accepted-results/may/4/`) containing one
subdirectory per experiment (each with `config.yaml`, `progress.csv`, and
checkpoints), produce three figures:

  1. `pareto_overlay.svg`  — final Pareto fronts of every experiment, overlayed.
  2. `hv_by_iter.svg`      — hypervolume vs. training iteration.
  3. `hv_by_time.svg`      — hypervolume vs. wall-clock seconds (zeroed per run).

Hypervolume curves and the dense final fronts are derived from `obj*.txt`
files written by `mop.eval.pareto.compute_fronts(save_results=True)`. By
default this script regenerates those files via dense rollouts (1024
random directives × all checkpoints) — pass `--no-regenerate` to plot
from existing files only.

Regeneration groups runs by (algorithm, network architecture) and dispatches
one batched call to `get_*_fronts` per group, so the rollout JIT compiles
once per architecture instead of once per run.

Usage:
    python -m ral.ablation_plots accepted-results/may/4/
    python -m ral.ablation_plots accepted-results/may/4/ --no-regenerate
"""

import os
os.environ["MUJOCO_GL"] = "egl"

import argparse
from collections import defaultdict
from pathlib import Path

import jax
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from minimal_mjx.utils.setupGPU import run_setup
import moplayground as mop
import minimal_mjx as mm
from moplayground.eval import pareto as pareto_eval


N_STEPS_DEFAULT  = 500
NUM_ENVS_DEFAULT = 2 ** 10


def discover_experiments(folder: Path):
    """Return subdirs that look like training runs (have config.yaml + progress.csv)."""
    runs = []
    for d in sorted(folder.iterdir()):
        if d.is_dir() and (d / 'config.yaml').is_file() and (d / 'progress.csv').is_file():
            runs.append(d)
    return runs


def load_config(run_dir: Path, ablation_folder: Path):
    cfg = mm.utils.read_config(str(run_dir / 'config.yaml'))
    # On-disk paths supersede whatever's in the saved config (the folder may have moved).
    cfg.save_dir = str(ablation_folder)
    cfg.name     = run_dir.name
    return cfg


def arch_key(cfg):
    """Hashable key identifying network architecture compatibility for JIT sharing."""
    if cfg.algorithm == 'morlax':
        np_ = cfg.learning_params.morlax_params.network_params
        return (
            'morlax', cfg.env, np_.hypertype,
            tuple(np_.hypersize), int(np_.num_features),
            tuple(np_.policy_hidden_layer_sizes),
            tuple(np_.value_hidden_layer_sizes),
        )
    if cfg.algorithm == 'amor':
        np_ = cfg.learning_params.amor_params.network_params
        return (
            'amor', cfg.env,
            tuple(np_.policy_hidden_layer_sizes),
            tuple(np_.value_hidden_layer_sizes),
        )
    raise ValueError(f"Unknown algorithm '{cfg.algorithm}' in {cfg.name}")


def regenerate_group(group_cfgs, n_steps: int, num_envs: int):
    """Run dense rollouts for every checkpoint of every config in a group.

    All configs in ``group_cfgs`` must share env + network architecture (the
    grouping in main() guarantees this). The rollout JIT is built once and
    reused across every checkpoint of every config.
    """
    algo = group_cfgs[0].algorithm
    env, _ = mop.envs.create_environment(group_cfgs[0], for_training=True)
    fn = pareto_eval.get_morlax_fronts if algo == 'morlax' else pareto_eval.get_amor_fronts
    fn(
        config=group_cfgs, rng=jax.random.PRNGKey(0), env=env,
        N_STEPS=n_steps, NUM_ENVS=num_envs, save_results=True,
    )


def read_obj_files(run_dir: Path):
    obj_files = sorted(run_dir.glob('obj*.txt'),
                       key=lambda p: int(p.stem[3:]))
    return [pd.read_csv(f).iloc[:, 1:].values for f in obj_files]


def compute_hvs(fronts):
    return np.array([mop.utils.pareto.get_pareto_statistics(F)[0] for F in fronts])


def compute_sparsities(fronts):
    return np.array([mop.utils.pareto.get_pareto_statistics(F)[1] for F in fronts])


def load_plot_data(run_dir: Path, cfg):
    fronts = read_obj_files(run_dir)
    if not fronts:
        print(f'[skip] {run_dir.name}: no obj*.txt found')
        return None

    progress = pd.read_csv(run_dir / 'progress.csv')
    all_times = progress['times'].values
    all_iters = progress['iters'].values
    # progress.csv row 0 holds the run's start_time (iter 0); subsequent rows
    # are evals. Zero times so cross-run comparisons share a wall-clock origin.
    start_time = all_times[0]
    times = all_times[1:] - start_time
    iters = all_iters[1:]

    n = min(len(fronts), len(times))
    hypertype = None
    if cfg.algorithm == 'morlax':
        hypertype = cfg.learning_params.morlax_params.network_params.hypertype
    hvs = compute_hvs(fronts[:n])
    sps = compute_sparsities(fronts[:n])
    best_idx = int(np.argmax(hvs))
    return {
        'fronts':       fronts[:n],
        'best_front':   fronts[best_idx],
        'hvs':          hvs,
        'sparsities':   sps,
        'times':        times[:n],
        'iters':        iters[:n],
        'labels':       list(cfg.env_config.reward.optimization.labels),
        'hypertype':    hypertype,
    }


def plot_pareto_overlay(names, datas, save_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.tab10
    for i, (name, d) in enumerate(zip(names, datas)):
        F = d['best_front']
        nd = F[mop.utils.pareto.get_nondominated(F)]
        color = cmap(i % 10)
        ax.scatter(F[:, 0], F[:, 1], s=4, color=color, alpha=0.15)
        ax.scatter(nd[:, 0], nd[:, 1], s=20, color=color,
                   edgecolors='black', linewidths=0.5, label=name)
    labels = datas[0]['labels']
    if len(labels) >= 2:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
    ax.legend(fontsize=7, loc='best')
    fig.suptitle('Best-HV Pareto fronts (dense rollouts)')
    fig.tight_layout()
    fig.savefig(save_path)
    print(f'Saved {save_path}')


HYPERTYPE_STYLES = {
    'single': {'linestyle': ':',  'color': 'tab:orange'},
    'dual':   {'linestyle': '-',  'color': 'tab:blue'},
}


def _aggregate(group_datas, x_key, y_key='hvs'):
    """Return (x, mean, std) by stacking per-run series at common length.

    Trims every run to the shortest length in the group so we can take
    elementwise mean/std even when runs differ slightly in eval count.
    """
    n = min(len(d[y_key]) for d in group_datas)
    ys = np.stack([d[y_key][:n] for d in group_datas], axis=0)
    xs = np.stack([d[x_key][:n] for d in group_datas], axis=0)
    return xs.mean(axis=0), ys.mean(axis=0), ys.std(axis=0)


def plot_hv_curve(names, datas, x_key: str, xlabel: str, title: str,
                  save_path: Path):
    fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
        3, 1, figsize=(8, 12), sharex=True
    )
    cmap = plt.cm.tab10

    # Top: HV per-run, linestyle keyed by hypertype.
    for i, (name, d) in enumerate(zip(names, datas)):
        style = HYPERTYPE_STYLES.get(d.get('hypertype'), {'linestyle': '-'})
        ax_top.plot(
            d[x_key][1:], d['hvs'][1:],
            color=cmap(i % 10), label=name, marker='o', markersize=3,
            linestyle=style['linestyle'],
        )
    ax_top.set_ylabel('Hypervolume')
    ax_top.legend(fontsize=7, loc='best')
    ax_top.set_title('Per-run hypervolume')

    # Middle: HV mean ± std aggregated by hypertype.
    groups = defaultdict(list)
    for d in datas:
        ht = d.get('hypertype')
        if ht in HYPERTYPE_STYLES:
            groups[ht].append(d)

    for ht, group_datas in groups.items():
        if not group_datas:
            continue
        x, mean, std = _aggregate(group_datas, x_key, 'hvs')
        style = HYPERTYPE_STYLES[ht]
        ax_mid.plot(x[1:], mean[1:],
                    color=style['color'], linestyle=style['linestyle'],
                    label=f'{ht} (n={len(group_datas)})', marker='o', markersize=3)
        ax_mid.fill_between(x[1:], (mean - std)[1:], (mean + std)[1:],
                            color=style['color'], alpha=0.2, linewidth=0)
    ax_mid.set_ylabel('Hypervolume')
    ax_mid.legend(fontsize=8, loc='best')
    ax_mid.set_title('Aggregated hypervolume by hypertype (mean ± std)')

    # Bottom: sparsity per-run.
    for i, (name, d) in enumerate(zip(names, datas)):
        style = HYPERTYPE_STYLES.get(d.get('hypertype'), {'linestyle': '-'})
        ax_bot.plot(
            d[x_key][1:], d['sparsities'][1:],
            color=cmap(i % 10), label=name, marker='o', markersize=3,
            linestyle=style['linestyle'],
        )
    ax_bot.set_xlabel(xlabel)
    ax_bot.set_ylabel('Sparsity')
    ax_bot.legend(fontsize=7, loc='best')
    ax_bot.set_title('Per-run sparsity')

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path)
    print(f'Saved {save_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=str,
                        help='Ablation folder containing per-experiment subdirs.')
    parser.add_argument('--no-regenerate', action='store_true',
                        help='Skip dense-rollout obj*.txt regeneration; plot from existing files.')
    parser.add_argument('--out', type=str, default=None,
                        help='Output dir for plots (default: <folder>/plots).')
    parser.add_argument('--n-steps', type=int, default=N_STEPS_DEFAULT,
                        help='Env steps per rollout when regenerating (default 500).')
    parser.add_argument('--num-envs', type=int, default=NUM_ENVS_DEFAULT,
                        help='Number of parallel directive samples per checkpoint (default 1024).')
    args = parser.parse_args()

    folder  = Path(args.folder).resolve()
    out_dir = Path(args.out).resolve() if args.out else folder / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_experiments(folder)
    if not runs:
        raise SystemExit(f'No experiment subdirs (with config.yaml + progress.csv) found in {folder}')
    print(f'Found {len(runs)} experiments in {folder}:')
    for r in runs:
        print(f'  {r.name}')

    # Load configs (overriding save_dir/name to point at on-disk locations).
    cfgs = {r.name: load_config(r, folder) for r in runs}

    if not args.no_regenerate:
        run_setup()
        # Group runs by architecture so each group shares one JIT compile.
        groups = defaultdict(list)
        for r in runs:
            groups[arch_key(cfgs[r.name])].append(r)

        print(f'\nRegenerating {len(runs)} runs in {len(groups)} architecture group(s):')
        for key, group_runs in groups.items():
            label = ' / '.join(str(p) for p in key)
            print(f'  [{label}] -> {len(group_runs)} run(s): '
                  f'{", ".join(r.name for r in group_runs)}')

        for group_runs in groups.values():
            group_cfgs = [cfgs[r.name] for r in group_runs]
            regenerate_group(group_cfgs, n_steps=args.n_steps, num_envs=args.num_envs)

    names, datas = [], []
    for r in runs:
        d = load_plot_data(r, cfgs[r.name])
        if d is None:
            continue
        names.append(r.name)
        datas.append(d)

    if not datas:
        raise SystemExit('No runs had obj*.txt data. Re-run without --no-regenerate.')

    plot_pareto_overlay(names, datas, out_dir / 'pareto_overlay.svg')
    plot_hv_curve(names, datas, x_key='iters', xlabel='Iteration',
                  title='Hypervolume by iteration',
                  save_path=out_dir / 'hv_by_iter.svg')
    plot_hv_curve(names, datas, x_key='times', xlabel='Time (s, relative)',
                  title='Hypervolume by time',
                  save_path=out_dir / 'hv_by_time.svg')


if __name__ == '__main__':
    main()
