import matplotlib as mpl

LABEL_SIZE = 11
TICK_SIZE = 10
mpl.rcParams.update({
    "text.usetex"           : True,
    "text.latex.preamble"   : r"\usepackage{amsmath,amssymb}",
    "font.family"           : "serif",
    "font.serif"            : ["Computer Modern Roman"],
    "font.size"             : LABEL_SIZE,
    "axes.labelsize"        : LABEL_SIZE,
    "axes.titlesize"        : LABEL_SIZE,
    "legend.fontsize"       : TICK_SIZE,
    "xtick.labelsize"       : TICK_SIZE,
    "ytick.labelsize"       : TICK_SIZE,
    "axes.linewidth"        : 1.2,
    "lines.linewidth"       : 2.0,
    "xtick.direction"       : "in",
    "ytick.direction"       : "in",
    "xtick.major.size"      : 4,
    "ytick.major.size"      : 4,
    "xtick.major.width"     : 1.0,
    "ytick.major.width"     : 1.0,
    "legend.frameon"        : False,
})
import matplotlib.pyplot as plt
import pandas as pd
import moplayground as mop
import minimal_mjx as mm
from pathlib import Path
import argparse
from ral import FINAL_YAMLS, HYPER_PARETOS


FIGSIZE = (2, 2.0)


def read_final_front(config):
    obj_files = list(Path(config['save_dir'], config['name']).glob('obj*.txt'))
    obj_files.sort(key=lambda x: int(x.name[3:].split('.')[0]))
    F = pd.read_csv(obj_files[-1]).iloc[:, 1:].values
    return F, F[mop.utils.pareto.get_nondominated(F)]


def read_hyper_front(hyper_path):
    F = pd.read_csv(hyper_path).iloc[:, 1:].values
    return F, F[mop.utils.pareto.get_nondominated(F)]


def scatter_front(ax, F, F_nd, color, label):
    ax.scatter(F[:, 0], F[:, 1], s=3, color=color, alpha=0.15, label=label)
    ax.scatter(F_nd[:, 0], F_nd[:, 1], s=15, color=color, edgecolors='black',
               label=f'{label} non-dominated')


def make_plot(env, morlax_config, amor_config, hyper_path):
    morlax_F, morlax_F_nd = read_final_front(morlax_config)
    amor_F, amor_F_nd     = read_final_front(amor_config)
    hyper_F, hyper_F_nd   = read_hyper_front(hyper_path)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    scatter_front(ax, amor_F,   amor_F_nd,   [0, 0.6, 0], 'AMOR')
    scatter_front(ax, hyper_F,  hyper_F_nd,  [1, 0, 0], 'HYPER-MORL')
    scatter_front(ax, morlax_F, morlax_F_nd, [0, 0, 1], 'MORLaX')

    labels = morlax_config['env_config']['reward']['optimization']['labels']
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])

    fig.subplots_adjust(left=0.3, right=0.9, top=0.8, bottom=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    save_path = f'ral/plots/{env}_overlayed_pareto.svg'
    fig.savefig(save_path)
    print(f'Saving to {save_path}')


def run(env):
    morlax_config = mm.utils.read_config(FINAL_YAMLS['morlax'][env])
    amor_config   = mm.utils.read_config(FINAL_YAMLS['amor'][env])
    make_plot(env, morlax_config, amor_config, HYPER_PARETOS[env])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, help="Env to train on")
    args = parser.parse_args()
    if args.env == 'all':
        for env in HYPER_PARETOS:
            if FINAL_YAMLS['amor'].get(env, 'n/a') == 'n/a':
                continue
            run(env)
    else:
        run(args.env)
