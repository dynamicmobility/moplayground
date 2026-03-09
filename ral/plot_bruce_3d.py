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
LABEL_SIZE = 16
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
from ral import FINAL_YAMLS, BRUCE_TRADEOFFS
import jax
from pathlib import Path
import pickle
import pandas as pd
from tqdm import tqdm
from itertools import combinations


def find_point(desc, PARETO):
    def cut_voxel(trio):
        pareto = PARETO[:, trio]
        # Find indices where values are within the specified ranges
        mask = np.ones(pareto.shape[0], dtype=bool)

        # Apply XLIM constraint for trio[0] (x-axis)
        if XLIM[0] is not None:
            mask &= pareto[:, 0] >= XLIM[0]
        if XLIM[1] is not None:
            mask &= pareto[:, 0] <= XLIM[1]

        # Apply YLIM constraint for trio[1] (y-axis)
        if YLIM[0] is not None:
            mask &= pareto[:, 1] >= YLIM[0]
        if YLIM[1] is not None:
            mask &= pareto[:, 1] <= YLIM[1]

        # Apply ZLIM constraint for trio[2] (z-axis)
        if ZLIM[0] is not None:
            mask &= pareto[:, 2] >= ZLIM[0]
        if ZLIM[1] is not None:
            mask &= pareto[:, 2] <= ZLIM[1]

        indices = np.where(mask)[0]
        return indices

    match desc:
        case 'swing-arms':
            XLIM = (950, None)
            YLIM = (990, None)
            ZLIM = (None, None)
            trio = [2, 0, 4]
            return cut_voxel(trio)
        case 'smooth':
            XLIM = (None, None)
            YLIM = (None, None)
            ZLIM = (1272.7, None)
            trio = [2, 0, 4]
            return cut_voxel(trio)
        # case 'rigid-arms':
        #     XLIM = (990, None)
        #     YLIM = (900, None)
        #     ZLIM = (1480, None)
        #     trio = [3, 5, 1]
        #     return cut_voxel(trio)
        case 'rigid-arms':
            XLIM = (None, 563)
            YLIM = (990, None)
            ZLIM = (None, None)
            trio = [2, 0, 4]
            return cut_voxel(trio)
        
        
def plot_tradeoff(ax, pareto_allD, tradeoff_allD, tradeoffs, trio):
    for td in tradeoffs:
        idx = np.argmin(np.linalg.norm(tradeoff_allD - td[np.newaxis, :], axis=1))
        c = td[trio]
        print(td)
        print(*(pareto_allD[idx, trio].T))
        print(c)
        print()
        ax.scatter(
            *(pareto_allD[idx, trio].T),
            s=48,
            color=c,
            edgecolors='black', # Border color
            linewidths=1.0,   # Border width,
            zorder=10
        )


def main():
    config    = read_config(FINAL_YAMLS['bruce6D+DR'])
    save_path = Path(config['save_dir']) / config['name']
    rng       = jax.random.PRNGKey(0)
    if (save_path / 'the-obj.txt').exists():
        # Load existing results instead of running experiments
        paretos = np.array([pd.read_csv(save_path / f'the-obj.txt').iloc[:, 1:].values])
        directives = np.array(pd.read_csv(save_path / f'the-trade-off.txt').iloc[:, 1:].values)
    else:
        raise Exception('Run 2D first')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', computed_zorder=False)
    trio = [2, 0, 4]
    trio = [3, 5, 1]
    pareto = paretos[-1, :, trio].T  # Select the last iteration's Pareto front and only the 3 objectives of interest
    tradeoff = directives[:, trio]
    chosen_tradeoffs = np.array(list(BRUCE_TRADEOFFS.values()))
    plot_tradeoff(ax, paretos[-1], directives, chosen_tradeoffs, trio)

    if False:
        tradeoff_key = ['smooth', 'swing-arms', 'rigid-arms']
        idxs = find_point(tradeoff_key[2], paretos[-1])
        chosen = paretos[-1, idxs]
        nidxs = get_nondominated(chosen)
        ax.scatter(
            *(chosen[nidxs].T[trio]),
            color='black',
            # marker='s',
            s=12,
            alpha=1.0,
            zorder=10
        )
        print(len(nidxs))
        for ni in nidxs:
            print(repr(directives[idxs][ni]))
            print(repr(pareto[idxs][ni]))
            print('--')
        
    ax = plot_pareto(
        ax, 
        pareto, 
        tradeoff / np.sum(tradeoff, axis=1)[:,np.newaxis],
        # objectives=[
        #     config['env_config']['reward']['optimization']['labels'][i] for i in trio
        # ],
        nondominated=get_nondominated(paretos[-1])
    )
    if trio == [2, 0, 4]:
        ax.set_zlim((950, 1300))
    elif trio == [3, 1, 5]:
        ax.set_ylim((1100, 1600))
        ax.set_zlim((650, 1150))
    # elif trio == [3, 5, 1]:
        # ax.set_ylim((600, 1000))
        # ax.set_zlim((800, 1600))
    elev = 20
    azim = 55#45
    OBJS = [config['env_config']['reward']['optimization']['labels'][i] for i in trio]
    ax.view_init(elev, azim)

    ax.set_xlabel('X')

    ax.locator_params(axis='x', nbins=5)
    ax.locator_params(axis='y', nbins=5)
    ax.locator_params(axis='z', nbins=5)
    ax.set_box_aspect([1,1,1])
    ax.grid(False)
    fig.set_size_inches((6,5))
    fig.tight_layout()
    # plt.show()
    fig.savefig(f'ral/images/fronts/bruce_3d-{OBJS[0]}-{OBJS[1]}-{OBJS[2]}.png', dpi=400)



if __name__ == '__main__':
    main()