"""Publication-ready hypervolume + spacing plot, keyed by sampling type.

Sibling of ``ral.hypernetwork_type``: same data, same layout, but the *linestyle*
encodes the sampling scheme (solid = sparse, dotted = dense) and *color* encodes
hypertype (so single & dual share a sampling line style but differ in color).

Usage:
    python -m ral.sampling_type accepted-results/may/8/
    python -m ral.sampling_type accepted-results/may/8/ --out paper/figs/sampling.pdf
"""

import os
os.environ["MUJOCO_GL"] = "egl"

import argparse
from pathlib import Path

import matplotlib as mpl

LABEL_SIZE = 8
TICK_SIZE  = 7
mpl.rcParams.update({
    "text.usetex":         True,
    "text.latex.preamble": r"\usepackage{amsmath,amssymb,times}",
    "font.family":         "serif",
    "font.serif":          ["Times", "Computer Modern Roman"],
    "font.size":           LABEL_SIZE,
    "axes.labelsize":      LABEL_SIZE,
    "axes.titlesize":      LABEL_SIZE,
    "legend.fontsize":     TICK_SIZE,
    "xtick.labelsize":     TICK_SIZE,
    "ytick.labelsize":     TICK_SIZE,
    "axes.linewidth":      0.8,
    "lines.linewidth":     1.2,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.major.size":    3,
    "ytick.major.size":    3,
    "xtick.major.width":   0.8,
    "ytick.major.width":   0.8,
    "legend.frameon":      False,
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
})

import matplotlib.pyplot as plt

from ral.ablation_plots import discover_experiments, load_config, load_plot_data
from ral.hypernetwork_type import pretty_label


FIGSIZE = (7.16, 2.6)


def sampling_linestyle(cfg):
    samp = cfg.learning_params.morlax_params.train_fn_params.sampling
    return ':' if samp == 'dense' else '-'


def hypertype_key(cfg):
    np_ = cfg.learning_params.morlax_params.network_params
    tp  = cfg.learning_params.morlax_params.train_fn_params
    # Color groups runs that share hypertype + K (since K varies the curve
    # within a hypertype too); dense runs ignore K.
    samp = tp.sampling
    k    = None if samp == 'dense' else int(tp.k)
    return (np_.hypertype, k)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=str)
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    out_path = Path(args.out).resolve() if args.out else folder / 'plots' / 'sampling_type.pdf'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    runs = discover_experiments(folder)
    if not runs:
        raise SystemExit(f'No experiment subdirs found in {folder}')

    cfgs = {r.name: load_config(r, folder) for r in runs}
    labels, datas, cfg_list = [], [], []
    for r in runs:
        d = load_plot_data(r, cfgs[r.name])
        if d is None:
            continue
        labels.append(pretty_label(cfgs[r.name]))
        datas.append(d)
        cfg_list.append(cfgs[r.name])

    if not datas:
        raise SystemExit('No runs had obj*.txt data. Run ral.ablation_plots first.')

    cmap = plt.cm.tab10
    color_map = {}
    for cfg in cfg_list:
        k = hypertype_key(cfg)
        if k not in color_map:
            color_map[k] = cmap(len(color_map) % 10)

    fig, (ax_hv, ax_sp) = plt.subplots(1, 2, figsize=FIGSIZE, sharex=True)
    for label, d, cfg in zip(labels, datas, cfg_list):
        color = color_map[hypertype_key(cfg)]
        linestyle = sampling_linestyle(cfg)
        ax_hv.plot(d['iters'][1:], d['hvs'][1:],
                   color=color, linestyle=linestyle,
                   label=label, linewidth=1.2)
        ax_sp.plot(d['iters'][1:], d['sparsities'][1:],
                   color=color, linestyle=linestyle,
                   label=label, linewidth=1.2)
    for ax in (ax_hv, ax_sp):
        ax.set_xlabel('Iteration')
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    ax_hv.set_ylabel('Hypervolume')
    ax_sp.set_ylabel('Spacing')
    ax_sp.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), ncol=1)

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
