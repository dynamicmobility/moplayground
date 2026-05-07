import os
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
from minimal_mjx.utils.setupGPU import run_setup
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from ral import FINAL_YAMLS, HYPER_TIMES

import moplayground as mop
import minimal_mjx as mm


def read_obj_files(config):
    obj_files = list(Path(config['save_dir'], config['name']).glob('obj*.txt'))
    obj_files.sort(key=lambda x: int(x.name[3:].split('.')[0]))
    hvs, sps = [], []
    for file in obj_files:
        F = pd.read_csv(file).iloc[:, 1:].values
        hv, sp = mop.utils.pareto.get_pareto_statistics(F)
        hvs.append(hv)
        sps.append(sp)
    return hvs, sps
    

if __name__ == '__main__':
    run_setup()
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str, help="Env to train on")
    args = parser.parse_args()
    
    morlax_config = mm.utils.read_config(FINAL_YAMLS['morlax'][args.env])
    amor_config = mm.utils.read_config(FINAL_YAMLS['amor'][args.env])
    morlax_statistics = read_obj_files(morlax_config)
    amor_statistics = read_obj_files(amor_config)     
    
    morlax_progress = pd.read_csv(Path(morlax_config['save_dir']).resolve() / morlax_config['name'] / 'progress.csv')
    amor_progress   = pd.read_csv(Path(amor_config['save_dir']).resolve() / amor_config['name'] / 'progress.csv')
    
    hyper_path = HYPER_TIMES[args.env]
    hyperdf = pd.read_csv(hyper_path)
    
    fig, ax = plt.subplots()
    ax.plot(
        morlax_progress['times'].values[1:] - morlax_progress['times'].values[0],
        morlax_statistics[0],
        color='blue',
        label='MORLAX'
    )
    ax.plot(
        amor_progress['times'].values[1:] - amor_progress['times'].values[0],
        amor_statistics[0],
        color='green',
        label='AMOR'
    )
    ax.plot(
        hyperdf['seconds'].values, #- hyperdf['seconds'].values[0],
        hyperdf['hypervolume'],
        color='red',
        label='HYPER-MORL'
    )
    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Hypervolume')
    fig.suptitle(f'Hypervolume by time: {args.env}')
    fig.set_size_inches((10, 5))
    fig.tight_layout()
    fig.savefig(f'ral/plots/{args.env}-hypervolume-speed.pdf')

    max_hyper_hv        = np.max(hyperdf['hypervolume'].values)
    max_morlax_hv       = np.max(morlax_statistics[0])
    max_amor_hv         = np.max(amor_statistics[0])

    
    if max_amor_hv < max_hyper_hv:
        min_algo = 'amor'
        min_hv = max_amor_hv
    else:
        min_algo = 'hyper'
        min_hv = max_hyper_hv
    
    hyper_finish        = np.argmax(hyperdf['hypervolume'].values >= min_hv)
    hyper_idx_finish    = np.argmax(hyperdf['hypervolume'].values >= min_hv)
    morlax_idx_finish   = np.argmax(np.array(morlax_statistics[0]) >= min_hv)
    amor_idx_finish     = np.argmax(np.array(amor_statistics[0]) >= min_hv)

    MORLAX_duration     = morlax_progress['times'].iloc[morlax_idx_finish] - morlax_progress['times'].iloc[0]
    AMOR_duration       = amor_progress['times'].iloc[amor_idx_finish] - amor_progress['times'].iloc[0]
    HYPER_duration      = hyperdf['seconds'].iloc[hyper_idx_finish]
    
    print('---')
    print('Smallest Max HV Algo =', min_algo)
    print('MORLAX finish time',     round(MORLAX_duration, 1))
    print('AMOR finish time',       round(AMOR_duration, 1))
    print('HYPER-MORL finish time', round(HYPER_duration, 1))
    print('Speedup factor HYPER-MORL vs MORLAX:', HYPER_duration / MORLAX_duration)
    print('Speedup factor from AMOR vs MORLAX:', AMOR_duration / MORLAX_duration)
    print('---')
    print('Max hypervolume (MORLAX):', f'{max_morlax_hv:.2e}')
    print('Max hypervolume (AMOR):', f'{max_amor_hv:.2e}')
    print('Max hypervolume (HYPER-MORL):', f'{max_hyper_hv:.2e}')
    hypermorl_improvement_ratio = max_morlax_hv / max_hyper_hv
    amor_improvement_ratio = max_morlax_hv / max_amor_hv
    print('MORLAX improvement over HYPER-MORL:', f'{hypermorl_improvement_ratio:.2f}x')
    print('MORLAX improvement over AMOR:', f'{amor_improvement_ratio:.2f}x')
    print('---')
    best_morlax_hv = np.argmax(morlax_statistics[0])
    best_amor_hv = np.argmax(amor_statistics[0])
    total_morlax_time = morlax_progress['times'].iloc[best_morlax_hv] - morlax_progress['times'].iloc[0]
    total_amor_time = amor_progress['times'].iloc[best_amor_hv] - amor_progress['times'].iloc[0]
    total_hyper_time = hyperdf['seconds'].iloc[np.argmax(hyperdf['hypervolume'].values)]
    print('Total time (MORLAX):', total_morlax_time, best_morlax_hv)
    print('Total time (AMOR):', total_amor_time, best_amor_hv)
    print('Total time (HYPER-MORL):', total_hyper_time)
    print('---')
    print('Final sparsity (MORLAX):', f'{morlax_statistics[1][np.argmax(morlax_statistics[0])]:.3f}')
    print('Final sparsity (AMOR):', f'{amor_statistics[1][np.argmax(amor_statistics[0])]:.3f}')
    print('Final sparsity (HYPER-MORL):', f'{hyperdf["sparsity"].iloc[np.argmax(hyperdf["hypervolume"].values)]:.3f}')


