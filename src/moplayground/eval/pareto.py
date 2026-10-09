import jax
import functools
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import moplayground as mop


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


def _as_list(run_dirs):
    """Accept one run directory or a list of them and return a list of Paths."""
    if isinstance(run_dirs, (list, tuple)):
        return [Path(d) for d in run_dirs]
    return [Path(run_dirs)]


def compute_fronts(
    run_dirs,
    num_objectives,
    rng,
    env,
    N_STEPS,
    NUM_ENVS,
    make_policy,
    load_params_fn,
    model_files_per_run,
    save_results,
):
    """Run rollouts for every checkpoint in every run over a fixed batch of
    randomly sampled directives, and return the resulting Pareto fronts.

    Algorithm-agnostic: callers supply ``make_policy`` and ``load_params_fn`` to
    plug in MORLAX, AMOR, or anything else with the right shapes. The rollout
    JIT is built once and reused across every run in ``run_dirs`` — so multiple
    runs that share an environment and network architecture amortize the
    compile cost.

    Parameters
    ----------
    run_dirs : list of run directories, all sharing the same environment.
        ``save_results`` writes into each file's owning run directory.
    num_objectives : int, number of objectives.
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
    model_files_per_run : list of lists of checkpoint paths, aligned with
        ``run_dirs``. ``model_files_per_run[i]`` are the checkpoints for
        ``run_dirs[i]``.
    save_results : if True, dump per-checkpoint returns to
        ``{run_dir}/obj{j}.txt`` (where ``run_dir`` is the file's owning run
        directory and ``j`` is its index within that run's checkpoints).

    Returns
    -------
    rewards_over_iters : ndarray, shape ``(total_checkpoints, NUM_ENVS, num_objs)``,
        per-objective episodic return. Files are concatenated in run order.
    tradeoffs_over_iters : ndarray, same shape, the directive used for each
        rollout. Fixed across checkpoints (broadcast along axis 0).
    """
    # one env-reset key per directive; same batch reused across checkpoints
    keys = jax.random.split(rng, NUM_ENVS)
    tradeoffs = jax.random.dirichlet(rng, alpha=np.ones(num_objectives), shape=(NUM_ENVS,))

    run_rollouts = get_pareto_rollout(env, N_STEPS, make_policy)

    flat = []
    for run_dir, files in zip(run_dirs, model_files_per_run):
        for j, file in enumerate(files):
            flat.append((run_dir, j, file))

    rewards_over_iters = []
    for run_dir, j, file in tqdm(flat):
        params = load_params_fn(file)
        (_, rewards), _ = run_rollouts(keys, tradeoffs, params)
        rewards_over_iters.append(rewards)
        if save_results:
            pd.DataFrame.from_dict(
                {f'obj{k}': objs for k, objs in enumerate(rewards.T)}
            ).to_csv(Path(run_dir) / f'obj{j}.txt')

    rewards_over_iters = np.array(rewards_over_iters)
    tradeoffs_over_iters = np.repeat(
        tradeoffs[np.newaxis, :, :], rewards_over_iters.shape[0], axis=0,
    )
    return rewards_over_iters, tradeoffs_over_iters


def get_morlax_fronts(run_dirs, network_params, num_objectives, rng, env, N_STEPS,
                      NUM_ENVS, save_results=False, only_final=False):
    """Compute Pareto fronts for one or more MORLAX (hypernetwork) runs.

    Parameters
    ----------
    run_dirs : run directory, or list of run directories. Every run must deploy
        the same environment and network architecture; the rollout JIT is
        built once and reused across all of them. Check this before calling
        (see ``moplayground.config.check_same_env``).
    network_params : dict, network-factory keyword arguments shared by all runs.
    num_objectives : int, number of objectives.
    rng : PRNGKey used both to sample the (fixed) batch of directives and to
        seed the per-rollout env reset keys.
    env : environment to roll out in.
    N_STEPS : int, env steps per rollout.
    NUM_ENVS : int, number of directives sampled (== number of parallel rollouts
        per checkpoint).
    save_results : if True, dump per-checkpoint returns to
        ``{run_dir}/obj{j}.txt`` for each owning run.
    only_final : if True, evaluate only the final checkpoint of each run.

    Returns
    -------
    rewards_over_iters : ndarray, shape ``(total_checkpoints, NUM_ENVS, num_objs)``,
        per-objective episodic return. Files are concatenated in run order.
    tradeoffs_over_iters : ndarray, same shape, the directive used for each
        rollout. Fixed across checkpoints (broadcast along axis 0).
    """
    run_dirs = _as_list(run_dirs)

    model_files_per_run = []
    for run_dir in run_dirs:
        files = mop.learning.inference.get_all_models(run_dir)
        if only_final:
            files = [files[-1]]
        model_files_per_run.append(files)

    # MORLAX: params == hypernet params; make_policy directly takes (params, deterministic, directive)
    first_file = model_files_per_run[0][0]
    make_policy, _ = mop.learning.inference.load_hypernetwork_inference_fn(
        network_params, num_objectives, path=first_file,
    )

    def load_params_fn(file):
        _, hyperparams = mop.learning.inference.load_hypernetworks(
            network_params, num_objectives, path=file,
        )
        return hyperparams

    return compute_fronts(
        run_dirs, num_objectives, rng, env, N_STEPS, NUM_ENVS,
        make_policy, load_params_fn, model_files_per_run, save_results,
    )


def get_amor_fronts(run_dirs, network_params, num_objectives, rng, env, N_STEPS,
                    NUM_ENVS, save_results=False, only_final=False):
    """Compute Pareto fronts for one or more AMOR (tradeoff-conditioned) runs.

    Parameters
    ----------
    run_dirs : run directory, or list of run directories. Every run must deploy
        the same environment and network architecture; the rollout JIT is
        built once and reused across all of them. Check this before calling
        (see ``moplayground.config.check_same_env``).
    network_params : dict, network-factory keyword arguments shared by all runs.
    num_objectives : int, number of objectives.
    rng : PRNGKey used both to sample the (fixed) batch of directives and to
        seed the per-rollout env reset keys.
    env : environment to roll out in.
    N_STEPS : int, env steps per rollout.
    NUM_ENVS : int, number of directives sampled (== number of parallel rollouts
        per checkpoint).
    save_results : if True, dump per-checkpoint returns to
        ``{run_dir}/obj{j}.txt`` for each owning run.
    only_final : if True, evaluate only the final checkpoint of each run.

    Returns
    -------
    rewards_over_iters : ndarray, shape ``(total_checkpoints, NUM_ENVS, num_objs)``,
        per-objective episodic return. Files are concatenated in run order.
    tradeoffs_over_iters : ndarray, same shape, the directive used for each
        rollout. Fixed across checkpoints (broadcast along axis 0).
    """
    run_dirs = _as_list(run_dirs)

    model_files_per_run = []
    for run_dir in run_dirs:
        files = mop.learning.inference.get_all_models(run_dir)
        if only_final:
            files = [files[-1]]
        model_files_per_run.append(files)

    # AMOR's native inference fn takes the directive at call time: policy(obs, directive, key).
    # Wrap it to match the (params, deterministic, directive) -> policy(obs, key) shape used by
    # get_pareto_rollout.
    first_file = model_files_per_run[0][0]
    make_amor_inference_fn, _ = mop.learning.inference.load_make_amor_inference_fn(
        network_params, num_objectives, path=first_file,
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
        _, params = mop.learning.inference.load_make_amor_inference_fn(
            network_params, num_objectives, path=file,
        )
        return params

    return compute_fronts(
        run_dirs, num_objectives, rng, env, N_STEPS, NUM_ENVS,
        make_policy, load_params_fn, model_files_per_run, save_results,
    )
