"""Publication-ready hypervolume-by-iteration plot for the MORLAX ablation.

Reuses the data-loading helpers from ``ral.ablation_plots`` (so it expects the
same on-disk layout: an ablation folder with per-run subdirs containing
``config.yaml``, ``progress.csv``, and the ``obj*.txt`` files emitted by
``compute_fronts(save_results=True)``). It does *not* regenerate fronts —
run ``ral.ablation_plots`` first if the ``obj*.txt`` files are missing.

Usage:
    python -m ral.hypernetwork_type accepted-results/may/8/
    python -m ral.hypernetwork_type accepted-results/may/8/ --out paper/figs/hv.pdf
"""

import os
os.environ["MUJOCO_GL"] = "egl"

import argparse
from pathlib import Path

import matplotlib as mpl

# IEEE-style: Times serif via LaTeX, small label/tick sizes, thin-but-visible
# axis lines, no legend frame. Single-column figure width ~3.5in.
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
    "pdf.fonttype":        42,  # embed TrueType so reviewers can copy text
    "ps.fonttype":         42,
})

import matplotlib.pyplot as plt

from ral.ablation_plots import discover_experiments, load_config, load_plot_data


FIGSIZE = (7.16, 2.6)  # IEEE full text-width (two-column); HV + sparsity side-by-side

HYPERTYPE_LINESTYLES = {'single': ':', 'dual': '-'}

HYPERTYPE_LABELS = {'single': 'Single', 'dual': 'Dual'}
SAMPLING_LABELS  = {
    'dense':            'Dense',
    'sparse':           'Sparse',
    'sparse-heavytail': 'Sparse',  # collapse per user request
    'single-avg':       'Single-Avg',
}


def pretty_label(cfg):
    """Build a human-readable legend entry from a run's config.

    Examples:
      hypertype=dual,   sampling=dense                 -> "Dual + Dense"
      hypertype=dual,   sampling=sparse-heavytail, k=4 -> "Dual + Sparse, K=4"
      hypertype=single, sampling=sparse-heavytail, k=8 -> "Single + Sparse, K=8"
    """
    if cfg.algorithm != 'morlax':
        return cfg.name
    np_   = cfg.learning_params.morlax_params.network_params
    tp    = cfg.learning_params.morlax_params.train_fn_params
    ht    = HYPERTYPE_LABELS.get(np_.hypertype, np_.hypertype)
    samp  = SAMPLING_LABELS.get(tp.sampling, tp.sampling)
    label = f'{ht} + {samp}'
    if tp.sampling != 'dense':
        label += f', $K={int(tp.k)}$'
    return label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=str,
                        help='Ablation folder containing per-experiment subdirs.')
    parser.add_argument('--out', type=str, default=None,
                        help='Output PDF path (default: <folder>/plots/hypernetwork_type.pdf).')
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    out_path = Path(args.out).resolve() if args.out else folder / 'plots' / 'hypernetwork_type.pdf'
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

    # Color by sampling-config (sampling, K) — single & dual share a color when
    # the sampling matches; hypertype is encoded only via linestyle.
    def sampling_key(cfg):
        tp = cfg.learning_params.morlax_params.train_fn_params
        # 'sparse' and 'sparse-heavytail' are visually labeled the same; treat
        # them as the same color group too.
        samp = 'sparse' if tp.sampling.startswith('sparse') else tp.sampling
        return (samp, None if samp == 'dense' else int(tp.k))

    cmap = plt.cm.tab10
    sampling_colors = {}
    for cfg in cfg_list:
        k = sampling_key(cfg)
        if k not in sampling_colors:
            sampling_colors[k] = cmap(len(sampling_colors) % 10)

    fig, (ax_hv, ax_sp) = plt.subplots(1, 2, figsize=FIGSIZE, sharex=True)
    for label, d, cfg in zip(labels, datas, cfg_list):
        ht = cfg.learning_params.morlax_params.network_params.hypertype
        color = sampling_colors[sampling_key(cfg)]
        linestyle = HYPERTYPE_LINESTYLES.get(ht, '-')
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
