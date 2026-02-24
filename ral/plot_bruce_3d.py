import matplotlib as mpl

# mpl.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"],
#     "axes.labelsize": 16,
#     "font.size": 16,
#     "legend.fontsize": 7,
#     "xtick.labelsize": 12,
#     "ytick.labelsize": 12,
# })
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

from moplayground.utils.plotting import plot_pareto
from moplayground.eval.pareto import get_nondominated, get_pareto_rollout
from moplayground.envs.create import create_environment
from moplayground.learning.startup import read_config
from moplayground.learning.inference import load_hypernetwork, get_last_model
from moplayground.utils.plotting import get_subplot_grid
import matplotlib.pyplot as plt
import numpy as np
from ral import FINAL_YAMLS
import jax
from pathlib import Path
import pickle
import pandas as pd
from tqdm import tqdm
from itertools import combinations

def main():
    config    = read_config(FINAL_YAMLS['bruce5D'])
    save_path = Path(config['save_dir']) / config['name']
    rng       = jax.random.PRNGKey(0)
    if (save_path / 'the-obj.txt').exists():
        # Load existing results instead of running experiments
        paretos = np.array([pd.read_csv(save_path / f'the-obj.txt').iloc[:, 1:].values])
        directives = np.array(pd.read_csv(save_path / f'the-trade-off.txt').iloc[:, 1:].values)
    else:
        raise Exception('Run 2D first')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    trio = [0, 2, 4]
    pareto = paretos[-1, :, trio].T  # Select the last iteration's Pareto front and only the 3 objectives of interest
    tradeoff = directives[:, trio]
    ax = plot_pareto(
        ax, 
        pareto, 
        tradeoff / np.sum(tradeoff, axis=1)[:,np.newaxis],
        objectives=[
            config['env_config']['reward']['optimization']['labels'][i] for i in trio
        ],
        nondominated=True
    )
    if trio == [0, 1, 4]:
        elev = 20
        azim = 45
    elif trio == [0, 2, 4]:
        elev = 20
        azim = 45
    ax.view_init(elev, azim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('white')
    ax.yaxis.pane.set_edgecolor('white')
    ax.zaxis.pane.set_edgecolor('white')
    ax.xaxis.labelpad = -10
    ax.yaxis.labelpad = -10
    ax.zaxis.labelpad = -10
    # ax.set_xlim((0, None))
    # ax.set_ylim((0, None))
    fig.set_size_inches((6,5))
    fig.tight_layout()
    fig.savefig('output/plots/bruce_3d_pareto.png', dpi=400)



if __name__ == '__main__':
    main()