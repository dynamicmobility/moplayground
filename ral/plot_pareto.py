from minimal_mjx.utils.plotting import set_mpl_params
set_mpl_params()

import argparse
from pathlib import Path
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import yaml
from ral import FINAL_YAMLS

ENVS = ['cheetah', 'hopper', 'ant', 'humanoid', 'walker']
STROKE = [pe.withStroke(linewidth=3, foreground='white')]
TEXT_KW = dict(color='black', path_effects=STROKE)


def pareto_front(points):
    """Return indices of pareto-optimal points (maximizing all objectives)."""
    mask = np.ones(len(points), dtype=bool)
    for i, p in enumerate(points):
        if mask[i]:
            mask[mask] = np.any(points[mask] >= p, axis=1)
            mask[i] = True
    return mask


def style_axis(ax):
    """Apply white-bordered black text to ticks and offset labels, with scientific notation."""
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(plt.MaxNLocator(3))
        axis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        axis.get_offset_text().set_color('black')
        axis.get_offset_text().set_path_effects(STROKE)
    ax.ticklabel_format(style='scientific', axis='both', scilimits=(0, 0))
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color('black')
        label.set_path_effects(STROKE)


def plot_pareto(env_name):
    config_path = Path(FINAL_YAMLS[env_name])
    with open(config_path) as f:
        config = yaml.safe_load(f)
    labels = config['env_config']['reward']['optimization']['labels']

    # Find the largest obj#.txt
    path = Path(config['save_dir']) / config['name'] / 'config.yaml'
    path = path.resolve()
    obj_files = sorted(path.parent.glob('obj*.txt'), key=lambda p: int(p.stem[3:]))
    latest_obj = obj_files[-1]

    # Read data and compute pareto front
    points = pd.read_csv(latest_obj, index_col=0)[['obj0', 'obj1']].values
    front = points[pareto_front(points)]
    front = front[front[:, 0].argsort()]

    # Plot
    fig, ax = plt.subplots(figsize=(3.75, 3.75))
    ax.scatter(points[:, 0], points[:, 1], alpha=0.3, s=20, color='tab:blue')
    ax.plot(front[:, 0], front[:, 1], 'o-', color='tab:blue', markersize=5, markeredgecolor='black', markeredgewidth=1.5)
    ax.set_xlabel(labels[0], **TEXT_KW)
    ax.set_ylabel(labels[1], **TEXT_KW)
    ax.set_title(config['env'], **TEXT_KW)
    style_axis(ax)
    fig.tight_layout()

    out_dir = Path('ral/plots')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{env_name}_pareto.svg'
    fig.savefig(out_path, dpi=200, transparent=True)
    plt.close(fig)
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot Pareto frontiers')
    parser.add_argument('env', nargs='?', default='all', choices=ENVS + ['all'],
                        help='Environment to plot (default: all)')
    args = parser.parse_args()

    envs = ENVS if args.env == 'all' else [args.env]
    for env in envs:
        plot_pareto(env)
