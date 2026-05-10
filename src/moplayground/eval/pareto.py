import jax
import functools
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import moplayground as mop
import minimal_mjx as mm


def get_pareto_rollout(env, N_STEPS, make_policy):
    """Build a jit/vmapped rollout fn that sweeps a batch of (key, directive, params).

    The same scaffold works for any algorithm — algorithm-specific details live
    inside ``make_policy``.

    Parameters
    ----------
    env : the brax/mjx environment to roll out in.
    N_STEPS : int, number of env steps per rollout.
    make_policy : callable with signature
        ``(params, deterministic, directive) -> policy(obs, key) -> (action, extras)``.

    Returns
    -------
    run_rollouts : jit-compiled fn ``(keys, directives, params) -> ((final_state, returns), None)``
        vmapped over the leading axis of ``keys`` and ``directives``. ``returns`` is
        the per-objective episodic return, shape ``(NUM_ENVS, num_objs)``.
    """
    policy_rng = jax.random.PRNGKey(0)

    def step_fn(carry, _, policy):
        state, old_reward = carry
        action, _ = policy(state.obs, policy_rng)
        new_state = env.step(state, action)
        # accumulate per-objective reward, masking out steps after termination
        return (new_state, old_reward + new_state.reward * (1 - new_state.done)), None

    def rollout(key, directive, params):
        policy = make_policy(
            params        = params,
            deterministic = True,
            directive     = directive,
        )
        scan_step_fn = functools.partial(step_fn, policy=policy)
        state = env.reset(key)
        carry = (state, state.reward)
        return jax.lax.scan(scan_step_fn, carry, (), N_STEPS)

    return jax.jit(jax.vmap(rollout, in_axes=(0, 0, None)))


def _to_plain(value):
    """Recursively convert ConfigDict/FrozenDict/etc to plain Python for equality checks."""
    if hasattr(value, 'to_dict'):
        return _to_plain(value.to_dict())
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    return value


def _normalize_configs(config):
    """Accept a single config or a list of configs and return a list.

    If a list is passed, validate that every config deploys the same
    environment (same ``env`` field and same ``env_config`` contents) so
    they can share a single rollout JIT.
    """
    configs = list(config) if isinstance(config, (list, tuple)) else [config]
    if len(configs) <= 1:
        return configs
    ref = configs[0]
    ref_env        = ref['env']
    ref_env_config = _to_plain(ref['env_config'])
    for i, c in enumerate(configs[1:], start=1):
        if c['env'] != ref_env:
            raise ValueError(
                f"Config[{i}] env={c['env']!r} does not match config[0] env={ref_env!r}; "
                f"all configs must deploy the same environment."
            )
        c_env_config = _to_plain(c['env_config'])
        if c_env_config != ref_env_config:
            raise ValueError(
                f"Config[{i}] env_config differs from config[0]; "
                f"all configs must deploy the same environment."
            )
    return configs


def compute_fronts(
    configs,
    rng,
    env,
    N_STEPS,
    NUM_ENVS,
    make_policy,
    load_params_fn,
    model_files_per_config,
    save_results,
):
    """Run rollouts for every checkpoint in every run over a fixed batch of
    randomly sampled directives, and return the resulting Pareto fronts.

    Algorithm-agnostic: callers supply ``make_policy`` and ``load_params_fn`` to
    plug in MORLAX, AMOR, or anything else with the right shapes. The rollout
    JIT is built once (from the first config's network) and reused across every
    config in ``configs`` — so multiple runs that share an environment and
    network architecture amortize the compile cost.

    Parameters
    ----------
    configs : list of training configs, all sharing the same environment. The
        first config is used to read the number of objectives; ``save_results``
        uses each file's owning config to choose the output directory.
    rng : PRNGKey used both to sample the (fixed) batch of directives and to
        seed the per-rollout env reset keys.
    env : environment to roll out in.
    N_STEPS : int, env steps per rollout.
    NUM_ENVS : int, number of directives sampled (== number of parallel rollouts
        per checkpoint).
    make_policy : callable with signature
        ``(params, deterministic, directive) -> policy(obs, key) -> (action, extras)``,
        passed through to :func:`get_pareto_rollout`.
    load_params_fn : callable ``file -> params`` mapping a checkpoint path to the
        params pytree expected by ``make_policy``.
    model_files_per_config : list of lists of checkpoint paths, aligned with
        ``configs``. ``model_files_per_config[i]`` are the checkpoints for
        ``configs[i]``.
    save_results : if True, dump per-checkpoint returns to
        ``{cfg.save_dir}/{cfg.name}/obj{j}.txt`` (where ``cfg`` is the file's
        owning config and ``j`` is its index within that config's checkpoints).

    Returns
    -------
    rewards_over_iters : ndarray, shape ``(total_checkpoints, NUM_ENVS, num_objs)``,
        per-objective episodic return. Files are concatenated in config order.
    tradeoffs_over_iters : ndarray, same shape, the directive used for each
        rollout. Fixed across checkpoints (broadcast along axis 0).
    """
    # one env-reset key per directive; same batch reused across checkpoints
    keys = jax.random.split(rng, NUM_ENVS)
    NUM_OBJS = len(configs[0]['env_config']['reward']['optimization']['objectives'])
    tradeoffs = jax.random.dirichlet(rng, alpha=np.ones(NUM_OBJS), shape=(NUM_ENVS,))

    run_rollouts = get_pareto_rollout(env, N_STEPS, make_policy)

    flat = []
    for cfg, files in zip(configs, model_files_per_config):
        for j, file in enumerate(files):
            flat.append((cfg, j, file))

    rewards_over_iters = []
    for cfg, j, file in tqdm(flat):
        params = load_params_fn(file)
        (_, rewards), _ = run_rollouts(keys, tradeoffs, params)
        rewards_over_iters.append(rewards)
        if save_results:
            pd.DataFrame.from_dict(
                {f'obj{k}': objs for k, objs in enumerate(rewards.T)}
            ).to_csv(Path(cfg['save_dir']) / cfg['name'] / f'obj{j}.txt')

    rewards_over_iters = np.array(rewards_over_iters)
    tradeoffs_over_iters = np.repeat(
        tradeoffs[np.newaxis, :, :], rewards_over_iters.shape[0], axis=0,
    )
    return rewards_over_iters, tradeoffs_over_iters


def get_morlax_fronts(config, rng, env, N_STEPS, NUM_ENVS, save_results=False,
                      only_final=False):
    """Compute Pareto fronts for one or more MORLAX (hypernetwork) runs.

    Parameters
    ----------
    config : training config dict, or list of training config dicts. When a
        list is passed, every config must deploy the same environment (same
        ``env`` field and ``env_config`` contents); the rollout JIT is built
        once from the first config and reused across all of them. ``ValueError``
        is raised if envs disagree.
    rng : PRNGKey used both to sample the (fixed) batch of directives and to
        seed the per-rollout env reset keys.
    env : environment to roll out in.
    N_STEPS : int, env steps per rollout.
    NUM_ENVS : int, number of directives sampled (== number of parallel rollouts
        per checkpoint).
    save_results : if True, dump per-checkpoint returns to
        ``{cfg.save_dir}/{cfg.name}/obj{j}.txt`` for each owning config.
    only_final : if True, evaluate only the final checkpoint of each config.

    Returns
    -------
    rewards_over_iters : ndarray, shape ``(total_checkpoints, NUM_ENVS, num_objs)``,
        per-objective episodic return. Files are concatenated in config order.
    tradeoffs_over_iters : ndarray, same shape, the directive used for each
        rollout. Fixed across checkpoints (broadcast along axis 0).
    """
    configs = _normalize_configs(config)

    model_files_per_config = []
    for cfg in configs:
        files = mm.learning.inference.get_all_models(cfg)
        if only_final:
            files = [files[-1]]
        model_files_per_config.append(files)

    # MORLAX: params == hypernet params; make_policy directly takes (params, deterministic, directive)
    first_file = model_files_per_config[0][0]
    make_policy, _ = mop.learning.inference.load_hypernetwork_inference_fn(configs[0], path=first_file)

    def load_params_fn(file):
        _, hyperparams = mop.learning.inference.load_hypernetworks(configs[0], path=file)
        return hyperparams

    return compute_fronts(
        configs, rng, env, N_STEPS, NUM_ENVS,
        make_policy, load_params_fn, model_files_per_config, save_results,
    )


def get_amor_fronts(config, rng, env, N_STEPS, NUM_ENVS, save_results=False,
                    only_final=False):
    """Compute Pareto fronts for one or more AMOR (tradeoff-conditioned) runs.

    Parameters
    ----------
    config : training config dict, or list of training config dicts. When a
        list is passed, every config must deploy the same environment (same
        ``env`` field and ``env_config`` contents); the rollout JIT is built
        once from the first config and reused across all of them. ``ValueError``
        is raised if envs disagree.
    rng : PRNGKey used both to sample the (fixed) batch of directives and to
        seed the per-rollout env reset keys.
    env : environment to roll out in.
    N_STEPS : int, env steps per rollout.
    NUM_ENVS : int, number of directives sampled (== number of parallel rollouts
        per checkpoint).
    save_results : if True, dump per-checkpoint returns to
        ``{cfg.save_dir}/{cfg.name}/obj{j}.txt`` for each owning config.
    only_final : if True, evaluate only the final checkpoint of each config.

    Returns
    -------
    rewards_over_iters : ndarray, shape ``(total_checkpoints, NUM_ENVS, num_objs)``,
        per-objective episodic return. Files are concatenated in config order.
    tradeoffs_over_iters : ndarray, same shape, the directive used for each
        rollout. Fixed across checkpoints (broadcast along axis 0).
    """
    configs = _normalize_configs(config)

    model_files_per_config = []
    for cfg in configs:
        files = mm.learning.inference.get_all_models(cfg)
        if only_final:
            files = [files[-1]]
        model_files_per_config.append(files)

    # AMOR's native inference fn takes the directive at call time: policy(obs, directive, key).
    # Wrap it to match the (params, deterministic, directive) -> policy(obs, key) shape used by
    # get_pareto_rollout.
    first_file = model_files_per_config[0][0]
    make_amor_inference_fn, _ = mop.learning.inference.load_make_amor_inference_fn(
        configs[0], path=first_file,
    )

    def make_policy(params, deterministic, directive):
        # checkpoint stores (normalizer, policy, value); inference only needs (normalizer, policy)
        normalizer_params, policy_params = params[0], params[1]
        amor_inference_fn = make_amor_inference_fn(
            params        = (normalizer_params, policy_params),
            deterministic = deterministic,
        )
        tradeoff_jnp = jax.numpy.asarray(directive)

        def policy(obs, key):
            return amor_inference_fn(obs, tradeoff_jnp, key)

        return policy

    def load_params_fn(file):
        _, params = mop.learning.inference.load_make_amor_inference_fn(configs[0], path=file)
        return params

    return compute_fronts(
        configs, rng, env, N_STEPS, NUM_ENVS,
        make_policy, load_params_fn, model_files_per_config, save_results,
    )
