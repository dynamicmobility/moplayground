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


def find_point(desc, pareto):
    match desc:
        case 'swing-arms':
            XLIM = (700, 950)
            YLIM = (950, None)
            ZLIM = (1020, None)

    
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
    ax = fig.add_subplot(111, projection='3d')
    trio = [2, 0, 4]
    # trio = [3, 1, 5]
    pareto = paretos[-1, :, trio].T  # Select the last iteration's Pareto front and only the 3 objectives of interest
    tradeoff = directives[:, trio]


    idxs = find_point('swing-arms', pareto)
    chosen = pareto[idxs]
    nidxs = get_nondominated(chosen)
    ax.scatter(
        *(chosen[nidxs].T),
        color='black',
        marker='s',
        s=40,
        alpha=1.0,
        zorder=0
    )
    print(len(nidxs))
    for td in directives[idxs][nidxs]:
        print(repr(td))

    ax = plot_pareto(
        ax, 
        pareto, 
        tradeoff / np.sum(tradeoff, axis=1)[:,np.newaxis],
        objectives=[
            config['env_config']['reward']['optimization']['labels'][i] for i in trio
        ],
        nondominated=True
    )
    if trio == [2, 0, 4]:
        pass
    elif trio == [3, 1, 5]:
        ax.set_ylim((1100, 1600))
        ax.set_zlim((650, 1150))
    elev = 20
    azim = 45
    OBJS = [config['env_config']['reward']['optimization']['labels'][i] for i in trio]
    ax.view_init(elev, azim)



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