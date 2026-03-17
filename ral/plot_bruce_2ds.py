import moplayground as mop
import matplotlib.pyplot as plt
import numpy as np
from ral import FINAL_YAMLS
import jax
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from itertools import combinations

def main():
    # config    = read_config(FINAL_YAMLS['bruce6D+DR'])
    config    = mop.utils.read_config('results/MOCheetah-all3/test/config.yaml')
    save_path = Path(config['save_dir']) / config['name']
    rng       = jax.random.PRNGKey(0)
    N_OBJS    = len(config['env_config']['reward']['optimization']['objectives'])
    if (save_path / 'the-obj.txt').exists():
        # Load existing results instead of running experiments
        paretos = np.array([pd.read_csv(save_path / f'the-obj.txt').iloc[:, 1:].values])
        directives = np.array(pd.read_csv(save_path / f'the-trade-off.txt').iloc[:, 1:].values)
    else:
        N_STEPS       = 500
        N_ENVS        = 2**10
        N_ITERS       = 16

        env, _                      = mop.envs.create_environment(
            config, 
            for_training    = True,
            # manual_speed    = [0.12, 0.0, 0.0],
            # idealistic      = True
        )
        make_policy, hyper_params   = mop.learning.inference.load_hypernetwork(config)
        pareto_rollout              = mop.eval.pareto.get_pareto_rollout(env, N_STEPS, make_policy)

        rewards_over_iters = []
        directives_over_iters = []
        for _ in tqdm(range(N_ITERS)):
            _, rng            = jax.random.split(rng)
            keys              = jax.random.split(rng, N_ENVS)
            directives        = jax.random.dirichlet(
                rng, 
                alpha=np.ones(N_OBJS) * 0.5,
                shape=(N_ENVS,)
            )
            (_, rewards), _   = pareto_rollout(keys, directives, hyper_params)

            rewards_over_iters.append(rewards)
            directives_over_iters.append(directives)
        
        paretos       = np.concat(rewards_over_iters)
        directives    = np.concat(directives_over_iters)
        pd.DataFrame.from_dict(
            {f'obj{i}' : objs for i, objs in enumerate(paretos.T)}
        ).to_csv(Path(config['save_dir']) / config['name'] / f'the-obj.txt')
        pd.DataFrame.from_dict(
            {f'td{i}' : tds for i, tds in enumerate(directives.T)}
        ).to_csv(Path(config['save_dir']) / config['name'] / f'the-trade-off.txt')
    quit()
    pairs = list(combinations(range(N_OBJS), 2))
    nrows, ncols = get_subplot_grid(len(pairs))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    axs = axs.flatten()
    
    failure = np.argmin(np.sum(paretos[-1], axis=1))
    print(repr(directives[failure]))
    for ax, pair in tqdm(zip(axs, pairs), total=len(pairs)):
        pareto = paretos[-1, :, pair].T  # Select the last iteration's Pareto front and only the 3 objectives of interest
        tradeoff = directives[:, pair]
        ax = plot_pareto(
            ax, 
            pareto, 
            tradeoff / np.sum(tradeoff, axis=1)[:,np.newaxis],
            objectives=[
                config['env_config']['reward']['optimization']['labels'][i] for i in pair
            ],
            nondominated=True
        )
        # ax.set_xlim((0, None))
        # ax.set_ylim((0, None))
    fig.set_size_inches((ncols*5, nrows*4))
    fig.tight_layout()
    fig.savefig('output/plots/bruce_paretos.png', dpi=400)



if __name__ == '__main__':
    main()