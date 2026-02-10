import jax
import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jax import numpy as jnp
from tqdm import tqdm
from pathlib import Path

from pymoo.indicators.hv import HV
from pymoo.util.normalization import normalize
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from minimal_mjx.learning.startup import read_config
from minimal_mjx.learning.inference import get_all_models
from moplayground.learning.inference import load_hypernetwork

def run_experiments(config, rng, env, N_STEPS, NUM_ENVS, save_results=False):
    def step_fn(carry, x, policy):
        state, old_reward = carry
        action, _ = policy(state.obs, rng)
        new_state = env.step(state, action)
        return (new_state, old_reward + new_state.reward * (1 - new_state.done)), None
    
    def rollout(key, directive, params):
        policy = make_policy(
            params        = params,
            deterministic = True,
            directive     = directive,
            single_policy = True
        )
        scan_step_fn = functools.partial(
            step_fn,
            policy=policy
        )
        state = env.reset(key)
        carry = (state, state.reward)
        return jax.lax.scan(scan_step_fn, carry, (), N_STEPS)
    
    run_rollouts = jax.jit(jax.vmap(rollout, in_axes=(0, 0, None)))

    rewards_over_iters = []
    keys = jax.random.split(rng, NUM_ENVS)
    directives = jnp.linspace(0, 1, num=NUM_ENVS)
    directives = jnp.concatenate(
        [directives[:,jnp.newaxis],
         1 - directives[:,jnp.newaxis]],
        axis=1
    )
    model_files = [get_all_models(config)[-1]]
    for i, file in enumerate(tqdm(model_files)):
        make_policy, moppo_params = load_hypernetwork(config, path=file)
        (_, rewards), _ = run_rollouts(keys, directives, moppo_params)
        rewards_over_iters.append(rewards)
        if save_results:
            pd.DataFrame.from_dict(
                {f'obj{i}' : objs for i, objs in enumerate(rewards.T)}
            ).to_csv(Path(config['save_dir']) / config['name'] / f'obj{i}.txt')
    
    rewards_over_iters = np.array(rewards_over_iters)
    return rewards_over_iters, np.repeat(
        directives[np.newaxis, :, :], 
        rewards_over_iters.shape[0], 
        axis=0
    )


def get_nondominated(F):
    nds = NonDominatedSorting()
    front_indices = nds.do(-F, only_non_dominated_front=True)
    F_max = F[front_indices]
    return F_max

def hypervolume_from_nondominated(F_min):
    # Reference point must be worse in minimization space
    # i.e., larger than all points in F_min
    ref_point = np.zeros(F_min.shape[1])

    hv = HV(ref_point=ref_point)
    hypervolume = hv(F_min)
    return hypervolume

def sparsity_from_normalized_nondominated(F_min_norm):
    spacing = SpacingIndicator()
    sparsity = spacing(F_min_norm)
    return sparsity

def get_pareto_statistics(F):
    # Extract nondominated front to do calculations
    F_max = get_nondominated(F)
    F_norm = normalize(F_max.copy())

    # Convert to minimization
    F_min = -F_max.copy()
    F_min_norm = -F_norm.copy()

    print(F_min_norm.shape)
    if F_min_norm.shape[0] == 1:
        # Sparsity always needs 2 points to calculate
        F_min_norm = np.repeat(F_min_norm, 2, axis=0)
    print('After', F_min_norm.shape)
    return (
        hypervolume_from_nondominated(F_min), 
        sparsity_from_normalized_nondominated(F_min_norm)
    )

if __name__ == '__main__':
    config = read_config()
    F = pd.read_csv(f'eval/data/{config['env']}.csv')
    F = F.iloc[:, 1:].values
    hypervolume, sparsity = get_pareto_statistics(F)
    # print("Hypervolume:", hypervolume)
    print(f"Hypervolume: {hypervolume:.3e}")
    print(f"spacing", sparsity)

    fig, ax = plt.subplots()
    ax.scatter(*(F.T))
    ax.set_xlim((0, 1.1 * np.max(F[0, :])))
    ax.set_ylim((0, 1.1 * np.max(F[1, :])))
    fig.savefig('img.pdf')
    print(F.T.shape)