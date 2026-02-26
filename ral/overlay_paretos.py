import matplotlib as mpl

LABEL_SIZE = 12
TICK_SIZE = 12
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
import numpy as np
import pandas as pd
from moplayground.learning.startup import read_config
from pathlib import Path
from moplayground.eval.pareto import get_nondominated
import argparse
from ral import FINAL_YAMLS, HYPER_PARETOS


FIGSIZE = (2, 2.5)

def make_plot(config, hyper_path):
    obj_files = list(Path(config['save_dir'], config['name']).glob('obj*.txt'))
    obj_files.sort(key=lambda x: int(x.name[3:].split('.')[0]))
    morlax_df       = pd.read_csv(obj_files[-1])
    morlax_F        = morlax_df.iloc[:, 1:].values
    morlax_F_max    = morlax_F[get_nondominated(morlax_F)]


    hyper_df      = pd.read_csv(hyper_path)
    hyper_F       = hyper_df.iloc[:, 1:].values
    hyper_F_max   = hyper_F[get_nondominated(hyper_F)]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter(morlax_df['obj0'], morlax_df['obj1'], s=3, color=[0, 0, 1], alpha=0.15, label='MORLaX')
    ax.scatter(*(morlax_F_max.T), s=15, color=[0, 0, 1], edgecolors='black', label='MORLaX non-dominated')
    ax.scatter(hyper_df['0'], hyper_df['1'], s=3, color=[1, 0, 0], alpha=0.15, label='HYPER-MORL')
    ax.scatter(*(hyper_F_max.T), s=15, color=[1, 0, 0], edgecolors='black', label='HYPER-MORL non-dominated')

    ax.set_xlabel(f'{config['env_config']['reward']['optimization']['labels'][0]}')
    ax.set_ylabel(f'{config['env_config']['reward']['optimization']['labels'][1]}')
    # ax.set_xlim((0, None))
    # ax.set_ylim((0, None))
    # ax.legend()
    # ax.set_aspect('equal')

    fig.subplots_adjust(
        left=0.3,    # 20% from left
        right=0.9,   # 90% from left
        top=0.8,     # 90% from bottom
        bottom=0.2   # 20% from bottom
    )
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    # fig.suptitle(f'{config['env']}')

    # fig.set_size_inches((5, 5))
    # ax.set_position([0.15, 0.15, 0.75, 0.75])
    # fig.tight_layout()
    save_path = f'ral/plots/{config['env']}_overlayed_pareto.svg'
    fig.savefig(save_path)
    print(f'Saving to {save_path}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, help="Env to train on")
    args = parser.parse_args()
    if args.env == 'all':
        for key in FINAL_YAMLS:
            if key == 'bruce': continue
            config = read_config(FINAL_YAMLS[key])
            make_plot(config, HYPER_PARETOS[key])
    else:
        print('here')
        config = read_config(FINAL_YAMLS[args.env])
        make_plot(config, HYPER_PARETOS[args.env])