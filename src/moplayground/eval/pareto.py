import jax
import functools
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import moplayground as mop


def get_pareto_rollout(env, N_STEPS, make_policy):
    policy_rng = jax.random.PRNGKey(0)
    def step_fn(carry, x, policy):
        state, old_reward = carry
        action, _ = policy(state.obs, policy_rng)
        new_state = env.step(state, action)
        return (new_state, old_reward + new_state.reward * (1 - new_state.done)), None
    
    def rollout(key, directive, params):
        policy = make_policy(
            params        = params,
            deterministic = True,
            directive     = directive,
        )
        scan_step_fn = functools.partial(
            step_fn,
            policy=policy
        )
        state = env.reset(key)
        carry = (state, state.reward)
        return jax.lax.scan(scan_step_fn, carry, (), N_STEPS)
    
    run_rollouts = jax.jit(jax.vmap(rollout, in_axes=(0, 0, None)))
    return run_rollouts

def run_experiments(config, rng, env, N_STEPS, NUM_ENVS, save_results=False, only_final=False):
    rewards_over_iters = []
    keys = jax.random.split(rng, NUM_ENVS)
    NUM_OBJS = len(config['env_config']['reward']['optimization']['objectives'])
    directives = jax.random.dirichlet(rng, alpha=np.ones(NUM_OBJS), shape=(NUM_ENVS,))
    model_files = mop.learning.inference.get_all_models(config)
    if only_final:
        model_files = [model_files[-1]]
    make_policy, _ = mop.learning.inference.load_hypernetwork(config, path=model_files[0])
    run_rollouts = get_pareto_rollout(env, N_STEPS, make_policy)
    for i, file in enumerate(tqdm(model_files)):
        _, moppo_params = mop.learning.inference.load_hypernetwork(config, path=file)
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

def run_experiments(config, rng, env, N_STEPS, NUM_ENVS, save_results=False, only_final=False):
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
    NUM_OBJS = len(config['env_config']['reward']['optimization']['objectives'])
    directives = jax.random.dirichlet(rng, alpha=np.ones(NUM_OBJS), shape=(NUM_ENVS,))
    print(directives.shape)
    model_files = mop.learning.inference.get_all_models(config)
    if only_final:
        model_files = [model_files[-1]]
    for i, file in enumerate(tqdm(model_files)):
        make_policy, moppo_params = mop.learning.inference.load_hypernetwork(config, path=file)
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