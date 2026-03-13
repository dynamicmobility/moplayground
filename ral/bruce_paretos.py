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
import moplayground as mop
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
    config    =mop.learning.read_config(FINAL_YAMLS['bruce6D+DR'])
    save_path = Path(config['save_dir']) / config['name']
    rng       = jax.random.PRNGKey(0)
    OBJS      = config['env_config']['reward']['optimization']['labels']
    if (save_path / 'the-obj.txt').exists():
        # Load existing results instead of running experiments
        paretos = np.array([pd.read_csv(save_path / f'the-obj.txt').iloc[:, 1:].values])
        directives = np.array(pd.read_csv(save_path / f'the-trade-off.txt').iloc[:, 1:].values)
    else:
        raise Exception('Run plot_bruce_2ds first')

    def plot(PAIR):
        match PAIR:
            case [0, 1]:
                XLIM = (650, 1050)
                YLIM = (1400, 1520)
            case [0, 2]:
                XLIM = (650, 1050)
                YLIM = (490, 1010)
            case [0, 4]:
                XLIM = (500, 1100)
                YLIM = (950, 1300)
            case [0, 5]:
                XLIM = (500, 1050)
                YLIM = (800, 1050)
            case [1, 4]:
                XLIM = (800, 1550)
                YLIM = (950, 1300)
            case [2, 3]:
                XLIM = (490, 1010)
                YLIM = (500, 1100)
            case [2,4]:
                XLIM = (490, 1010)
                YLIM = (950, 1300)
            case [4,5]:
                XLIM = (950, 1300)
                YLIM = (800, 1050)
            case _:
                return
                
        pareto = paretos[-1, :, PAIR].T  # Select the last iteration's Pareto front and only the 3 objectives of interest
        tradeoff = directives[:, PAIR]
        fig, ax = plt.subplots()
        ax = mop.utils.plotting.plot_pareto(
            ax, 
            pareto, 
            tradeoff / np.sum(tradeoff, axis=1)[:,np.newaxis],
            objectives=[
                OBJS[i] for i in PAIR
            ],
            nondominated=True
        )
        nd_idx = mop.eval.pareto.get_nondominated(pareto, epsilon=10)
        nd = pareto[nd_idx]
        nd_td = directives[nd_idx]
        max_x = np.argmax(nd[:, 0])
        max_y = np.argmax(nd[:, 1])
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5) # Set the thickness in points
        ax.locator_params(axis='x', nbins=5)
        ax.locator_params(axis='y', nbins=5)
        print('MAX X', max_x, repr(nd_td[max_x]))
        print('MAX Y', max_y, repr(nd_td[max_y]))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlim(XLIM)
        ax.set_ylim(YLIM)
        fig.set_size_inches((5, 4))
        path = f'ral/images/fronts/{OBJS[PAIR[0]]}-{OBJS[PAIR[1]]}.jpg'
        fig.savefig(path, dpi=1000)
        print(path)
        print()

    PAIR = [0, 2]
    PAIR = None
    if PAIR == None:
        for pair in combinations(range(6), 2):
            plot(pair)
    else:
        plot(PAIR)

if __name__ == '__main__':
    main()