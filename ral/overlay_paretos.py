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
mpl.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath,amssymb}",
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "font.size": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "legend.frameon": False,
})
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from moplayground.learning.startup import read_config
from pathlib import Path
from moplayground.eval.pareto import get_nondominated

config    = read_config()
obj_files = list(Path(config['save_dir'], config['name']).glob('obj*.txt'))
obj_files.sort(key=lambda x: int(x.name[3:].split('.')[0]))

morlax_df       = pd.read_csv(obj_files[-1])
morlax_F        = morlax_df.iloc[:, 1:].values
morlax_F_max    = get_nondominated(morlax_F)


hyper_path    = input('Enter file path of hyper objs: ')
hyper_df      = pd.read_csv(hyper_path)
hyper_F       = hyper_df.iloc[:, 1:].values
hyper_F_max   = get_nondominated(hyper_F)

fig, ax = plt.subplots()
ax.scatter(morlax_df['obj0'], morlax_df['obj1'], s=3, color=[0, 0, 1], alpha=0.15, label='MORLaX')
ax.scatter(*(morlax_F_max.T), s=15, color=[0, 0, 1], edgecolors='black', label='MORLaX non-dominated')
ax.scatter(hyper_df['0'], hyper_df['1'], s=3, color=[1, 0, 0], alpha=0.15, label='HYPER-MORL')
ax.scatter(*(hyper_F_max.T), s=15, color=[1, 0, 0], edgecolors='black', label='HYPER-MORL non-dominated')

ax.set_xlabel(f'{config['env_config']['reward']['optimization']['labels'][0]}')
ax.set_ylabel(f'{config['env_config']['reward']['optimization']['labels'][1]}')
# ax.set_xlim((0, None))
# ax.set_ylim((0, None))
# ax.legend()
# fig.suptitle(f'Pareto Comparison: {config['env']}')

fig.set_size_inches((5, 5))
fig.tight_layout()
fig.savefig(f'ral/plots/{config['env']}_overlayed_pareto.svg')



