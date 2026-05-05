import jax
import functools
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import moplayground as mop
import minimal_mjx as mm

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

def get_morlax_fronts(config, rng, env, N_STEPS, NUM_ENVS, save_results=False, only_final=False):
    rewards_over_iters = []
    keys = jax.random.split(rng, NUM_ENVS)
    NUM_OBJS = len(config['env_config']['reward']['optimization']['objectives'])
    tradeoffs = jax.random.dirichlet(rng, alpha=np.ones(NUM_OBJS), shape=(NUM_ENVS,))
    model_files = mm.learning.inference.get_all_models(config)
    if only_final:
        model_files = [model_files[-1]]
    make_policy, _ = mop.learning.inference.load_hypernetwork_inference_fn(config, path=model_files[0])
    run_rollouts = get_pareto_rollout(env, N_STEPS, make_policy)
    for i, file in enumerate(tqdm(model_files)):
        _, hyperparams = mop.learning.inference.load_hypernetworks(config, path=file)
        (_, rewards), _ = run_rollouts(keys, tradeoffs, hyperparams)
        rewards_over_iters.append(rewards)
        if save_results:
            pd.DataFrame.from_dict(
                {f'obj{i}' : objs for i, objs in enumerate(rewards.T)}
            ).to_csv(Path(config['save_dir']) / config['name'] / f'obj{i}.txt')
    
    rewards_over_iters = np.array(rewards_over_iters)
    return rewards_over_iters, np.repeat(
        tradeoffs[np.newaxis, :, :], 
        rewards_over_iters.shape[0], 
        axis=0
    )
    
def get_amor_fronts(config, rng, env, N_STEPS, NUM_ENVS, save_results=False, only_final=False):
    
    
    def make_amor_so_policy(_make_amor_inference_fn, params, deterministic, directive):
        # checkpoint stores (normalizer, policy, value); inference uses (normalizer, policy).
        normalizer_params, policy_params = params[0], params[1]
        amor_inference_fn = _make_amor_inference_fn(
            params        = (normalizer_params, policy_params),
            deterministic = deterministic,
        )
        tradeoff_jnp = jax.numpy.asarray(directive)
        def policy(obs, key):
            return amor_inference_fn(obs, tradeoff_jnp, key)
    
        return policy
    
    keys = jax.random.split(rng, NUM_ENVS)
    NUM_OBJS = len(config['env_config']['reward']['optimization']['objectives'])
    tradeoffs = jax.random.dirichlet(rng, alpha=np.ones(NUM_OBJS), shape=(NUM_ENVS,))
    model_files = mm.learning.inference.get_all_models(config)
    if only_final:
        model_files = [model_files[-1]]
        
    make_amor_inference_fn, _ = mop.learning.inference.load_make_amor_inference_fn(config, path=model_files[0])
    make_policy = functools.partial(make_amor_so_policy, make_amor_inference_fn)
    run_rollouts = get_pareto_rollout(env, N_STEPS, make_policy)
    
    rewards_over_iters = []
    for i, file in enumerate(tqdm(model_files)):
        _, params = mop.learning.inference.load_make_amor_inference_fn(config, path=file)
        (_, rewards), _ = run_rollouts(keys, tradeoffs, params)
        rewards_over_iters.append(rewards)
        if save_results:
            pd.DataFrame.from_dict(
                {f'obj{i}' : objs for i, objs in enumerate(rewards.T)}
            ).to_csv(Path(config['save_dir']) / config['name'] / f'obj{i}.txt')
    
    rewards_over_iters = np.array(rewards_over_iters)
    return rewards_over_iters, np.repeat(
        tradeoffs[np.newaxis, :, :], 
        rewards_over_iters.shape[0], 
        axis=0
    )