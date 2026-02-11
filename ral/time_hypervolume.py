import os
os.environ["MUJOCO_GL"] = "egl"
# os.environ['JAX_PLATFORMS']='cpu'
from minimal_mjx.utils.setupGPU import run_setup
from pathlib import Path
from minimal_mjx.learning.startup import read_config
from moplayground.envs.create import create_environment
from moplayground.learning.inference import load_hypernetwork
from minimal_mjx.learning.inference import get_all_models
import jax
from jax import numpy as jnp
import numpy as np
import functools
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from moplayground.eval.pareto import run_experiments, get_pareto_statistics, get_nondominated
from minimal_mjx.utils.plotting import get_subplot_grid

if __name__ == '__main__':
    run_setup()
    config = read_config()
    obj_files = list(Path(config['save_dir'], config['name']).glob('obj*.txt'))
    obj_files.sort(key=lambda x: int(x.name[3:].split('.')[0]))
    if obj_files:
        hvs = []
        nrows, ncols = get_subplot_grid(len(obj_files))
        fig, axs = plt.subplots(nrows, ncols)
        axs = axs.flatten()
        PLOT_PARETOS = True
        for idx, (file, ax) in enumerate(zip(obj_files, axs)):
            print(file)
            F = pd.read_csv(file).iloc[:, 1:].values
            hv, _ = get_pareto_statistics(F)
            hvs.append(hv)
            
            if PLOT_PARETOS:
                F_max = get_nondominated(F)
                ax.scatter(*(F.T), c='black', s=1)
                ax.scatter(*(F_max.T), c='r', s=3)
                ax.set_title(str(file.name))
        
        if PLOT_PARETOS:
            fig.set_size_inches((ncols*3, nrows*3))
            fig.tight_layout()
            fig.savefig(f'ral/plots/{config['env']}-pareto.pdf')
    else:
        rewards_over_iters, directives = run_experiments(
            config          = config,
            rng             = jax.random.PRNGKey(0),
            env             = create_environment(config, for_training=True)[0],
            N_STEPS         = 500,
            NUM_ENVS        = 2**10,
            save_results    = True
        )
        hvs = []
        for F in rewards_over_iters:
            hv, sp = get_pareto_statistics(np.array(F))
            hvs.append(hv)
    jax_progress = pd.read_csv(Path(config['save_dir']) / config['name'] / 'progress.csv')
    
    hyper_path = input('Enter hyper comparison path: ')
    hyperdf = pd.read_csv(hyper_path)
    
    fig, ax = plt.subplots()
    ax.plot(
        jax_progress['times'].values[1:] - jax_progress['times'].values[0],
        hvs,
        color='blue',
        label='MORLaX'
    )
    ax.plot(
        hyperdf['seconds'].values - hyperdf['seconds'].values[0],
        hyperdf['hypervolume'],
        color='red',
        label='HYPER-MORL'
    )
    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Hypervolume')
    fig.suptitle(f'Hypervolume by time: {config['env']}')
    fig.set_size_inches((10, 5))
    fig.tight_layout()
    fig.savefig(f'ral/plots/{config['env']}-hypervolume-speed.pdf')

    finish_line = hyperdf['hypervolume'].values[-1]
    hyper_idx_finish = np.argmax(hyperdf['hypervolume'].values >= finish_line)
    morlax_idx_finish = np.argmax(np.array(hvs) >= finish_line)
    MORLaX_duration = jax_progress['times'].iloc[morlax_idx_finish] - jax_progress['times'].iloc[0]
    HYPER_duration = hyperdf['seconds'].iloc[hyper_idx_finish] - hyperdf['seconds'].iloc[0]
    print('MORLaX finish time', MORLaX_duration)
    print('HYPER-MORL finish time', HYPER_duration)
    print('Speedup factor:', HYPER_duration / MORLaX_duration)