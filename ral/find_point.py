from moplayground.learning.startup import read_config
import matplotlib.pyplot as plt
import numpy as np
from ral import FINAL_YAMLS
import jax
from pathlib import Path
import pickle
import pandas as pd
from tqdm import tqdm
from itertools import combinations
import argparse

def find_point():
    config    = read_config(FINAL_YAMLS['bruce6D+DR'])
    save_path = Path(config['save_dir']) / config['name']
    if (save_path / 'the-obj.txt').exists():
        # Load existing results instead of running experiments
        paretos = np.array([pd.read_csv(save_path / f'the-obj.txt').iloc[:, 1:].values])
        directives = np.array(pd.read_csv(save_path / f'the-trade-off.txt').iloc[:, 1:].values)
    else:
        raise Exception('Run 2D first')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # trio = [2, 0, 4]
    trio = [3, 1, 5]
    pareto = paretos[-1, :, trio].T 
    tradeoff = directives[:, trio]
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tradeoff', default='swing-arms', help='Tradeoff method')
    args = parser.parse_args()
    match args.tradeoff:
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



if __name__ == '__main__':
    main()