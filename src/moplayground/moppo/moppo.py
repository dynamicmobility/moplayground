# Copyright 2025 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Proximal policy optimization training.

See: https://arxiv.org/pdf/1707.06347.pdf
"""

import functools
import time
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
from brax.training.agents.ppo import networks as ppo_networks
from brax.training import gradients
from brax.training import logger as metric_logger
from brax.training import pmap
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.agents.ppo import checkpoint
from brax.training.types import Params
from brax.training.types import PRNGKey
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax

from moplayground.moppo import acting
from moplayground.moppo import factory
from moplayground.moppo import losses


InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class MOTrainingState:
    """Contains training state for the learner."""

    optimizer_state   : optax.OptState
    params            : losses.MOPPONetworkParams
    normalizer_params : running_statistics.RunningStatisticsState
    env_steps         : types.UInt64


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
    # brax user code is sometimes ambiguous about weak_type.  in order to
    # avoid extra jit recompilations we strip all weak types from user input
    def f(leaf):
        leaf = jnp.asarray(leaf)
        return jnp.astype(leaf, leaf.dtype)

    return jax.tree_util.tree_map(f, tree)


def _maybe_wrap_env(
    env: envs.Env,
    wrap_env: bool,
    num_envs: int,
    episode_length: Optional[int],
    action_repeat: int,
    device_count: int,
    key_env: PRNGKey,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
):
    """Wraps the environment for training/eval if wrap_env is True."""
    if not wrap_env:
        return env
    if episode_length is None:
        raise ValueError('episode_length must be specified in ppo.train')
    v_randomization_fn = None
    if randomization_fn is not None:
        randomization_batch_size = num_envs // device_count
        # all devices gets the same randomization rng
        randomization_rng = jax.random.split(key_env, randomization_batch_size)
        v_randomization_fn = functools.partial(
            randomization_fn, rng=randomization_rng
        )
    if wrap_env_fn is not None:
        wrap_for_training = wrap_env_fn
    else:
        wrap_for_training = envs.training.wrap
    env = wrap_for_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )  # pytype: disable=wrong-keyword-args
    return env


def sample_preferences(key, it, sampling, K, warmup_frac, alpha, num_evals, num_envs, num_objs):
    cond = it < np.round(warmup_frac * num_evals)
    def warmup_fn(_):
        print('WARMUP')
        w = jnp.ones((1, num_envs, num_objs)) / num_objs
        return w
    
    def morl_fn(_):
        if sampling == 'dense':
            _, w_key = jax.random.split(key)
            w = jax.random.dirichlet(
                key=w_key,
                alpha=jnp.ones(num_objs) * alpha,
                shape=(1, num_envs)
            )
        elif sampling == 'sparse':
            _, w_key = jax.random.split(key)
            w = jax.random.dirichlet(
                key=w_key,
                alpha=jnp.ones(num_objs) * alpha,
                shape=k
            )
            w = jnp.repeat(w, num_envs // (k + 2), axis=0)[jnp.newaxis, :]
        elif sampling == 'sparse-heavytail':
            k = K - num_objs
            _, w_key = jax.random.split(key)
            w = jax.random.dirichlet(
                key=w_key,
                alpha=jnp.ones(num_objs) * alpha,
                shape=k
            )
            w = jnp.concat(
                (w, jnp.eye(num_objs)),
                axis=0
            )
            w = jnp.repeat(w, num_envs // (k + num_objs), axis=0)[jnp.newaxis, :]
        elif sampling == 'single-avg':
            w = jnp.ones(1, num_envs, num_objs) / num_objs
        else:
            raise Exception(f'Sampling type {sampling} not implemented')
        return w
    w = jax.lax.cond(
        cond,
        warmup_fn,
        morl_fn,
        None
    )
    return w

def _get_process_count():
    process_count = jax.process_count()
    process_id = jax.process_index()
    return process_count, process_id

def _get_device_count(max_devices_per_host, process_count, process_id):
    """Write this when you understand it."""
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    logging.info(
        'Device count: %d, process count: %d (id %d), local device count: %d, '
        'devices to be used count: %d',
        jax.device_count(),
        process_count,
        process_id,
        local_device_count,
        local_devices_to_use,
    )
    device_count = local_devices_to_use * process_count
    return device_count, local_devices_to_use

def _get_random_keys(seed, process_id):
    key = jax.random.PRNGKey(seed)
    global_key, local_key = jax.random.split(key)
    local_key = jax.random.fold_in(local_key, process_id)
    local_key, key_env, eval_key = jax.random.split(local_key, 3)
    # key_networks should be global, so that networks are initialized the same
    # way for different processes.
    key_policy, key_value = jax.random.split(global_key)
    return local_key, key_env, eval_key, key_policy, key_value

def _make_reset_envs(local_devices_to_use, use_pmap_on_reset, env, key_env, num_envs, process_count):
    if local_devices_to_use > 1 or use_pmap_on_reset:
        reset_fn = jax.pmap(env.reset, axis_name=_PMAP_AXIS_NAME)
    else:
        reset_fn = jax.jit(jax.vmap(env.reset))

    key_envs = jax.random.split(key_env, num_envs // process_count)
    key_envs = jnp.reshape(
        key_envs, (local_devices_to_use, -1) + key_envs.shape[1:]
    )
    env_state = reset_fn(key_envs)
    return key_envs, env_state, reset_fn


def train(
    environment             : envs.Env,
    num_timesteps           : int,
    max_devices_per_host    : Optional[int] = None,
    # high-level control flow
    wrap_env                : bool = True,
    madrona_backend         : bool = False,
    augment_pixels          : bool = False,
    # environment wrapper
    num_envs                : int = 1,
    episode_length          : Optional[int] = None,
    action_repeat           : int = 1,
    wrap_env_fn             : Optional[Callable[[Any], Any]] = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ]                       = None,
    # ppo params
    learning_rate           : float = 1e-4,
    entropy_cost            : float = 1e-4,
    discounting             : float = 0.9,
    unroll_length           : int = 10,
    batch_size              : int = 32,
    num_minibatches         : int = 16,
    num_updates_per_batch   : int = 2,
    num_resets_per_eval     : int = 0,
    normalize_observations  : bool = False,
    reward_scaling          : float = 1.0,
    clipping_epsilon        : float = 0.3,
    gae_lambda              : float = 0.95,
    max_grad_norm           : Optional[float] = None,
    normalize_advantage     : bool = True,
    alpha                   : float = 1.0,
    warmup_frac             : float = 0.0,
    sampling                : str = 'dense',
    k                       : int = 4,
    network_factory: types.NetworkFactory[
        factory.MOPPONetworks
    ]                       = factory.make_moppo_networks,
    init_policy_params      : dict = None,
    init_normalizer_params  : dict = None,
    init_value_params       : dict = None,
    seed                    : int = 0,
    use_pmap_on_reset       : bool = True,
    # eval
    num_evals               : int = 1,
    eval_env                : Optional[envs.Env] = None,
    num_eval_envs           : int = 128,
    deterministic_eval      : bool = False,
    # training metrics
    log_training_metrics    : bool = False,
    training_metrics_steps  : Optional[int] = None,
    # callbacks
    progress_fn             : Callable[[int, Metrics], None] = lambda *args: None,
    policy_params_fn        : Callable[..., None] = lambda *args: None,
    # checkpointing
    save_checkpoint_path    : Optional[str] = None,
    restore_checkpoint_path : Optional[str] = None,
    restore_params          : Optional[Any] = None,
    restore_value_fn        : bool = True,
    run_evals               : bool = True,
):
    assert batch_size * num_minibatches % num_envs == 0
    xt = time.time()
    process_count, process_id = _get_process_count()
    device_count, local_devices_to_use = _get_device_count(max_devices_per_host, process_count, process_id)
    assert num_envs % device_count == 0

    # The number of environment steps executed for every training step.
    env_step_per_training_step = batch_size * unroll_length * num_minibatches * action_repeat

    # The number of training_step calls per training_epoch call.
    num_evals_after_init = max(num_evals - 1, 1)
    num_training_steps_per_epoch = num_timesteps / (num_evals_after_init * env_step_per_training_step * max(num_resets_per_eval, 1))
    num_training_steps_per_epoch = np.ceil(num_training_steps_per_epoch).astype(int)

    # Get random keys
    local_key, key_env, eval_key, key_policy, key_value = _get_random_keys(seed, process_id)

    env = _maybe_wrap_env(
        environment,
        wrap_env,
        num_envs,
        episode_length,
        action_repeat,
        device_count,
        key_env,
        wrap_env_fn,
        randomization_fn,
    )
    
    eval_env = _maybe_wrap_env(
        eval_env, #or environment,
        wrap_env,
        num_eval_envs,
        episode_length,
        action_repeat,
        device_count        = 1,  # eval on the host only
        key_env             = eval_key,
        wrap_env_fn         = wrap_env_fn,
    )

    # Reset the environment in a vectorized fashion
    key_envs, env_state, reset_fn = _make_reset_envs(
        local_devices_to_use,
        use_pmap_on_reset,
        env,
        key_env,
        num_envs,
        process_count
    )
    
    # Discard the batch axes over devices and envs.
    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)
    num_objectives = env_state.reward.shape[2]

    normalize = lambda x, y: x
    if normalize_observations:
        normalize = running_statistics.normalize
    moppo_network: factory.MOPPONetworks = network_factory(
        key                        = key_policy,
        observation_size           = obs_shape,
        action_size                = env.action_size,
        num_objectives             = num_objectives,
        preprocess_observations_fn = normalize,
        target_policy_params       = init_policy_params,
        target_value_params        = init_value_params
    )
    make_policy = factory.make_mo_inference_fn(moppo_network)

    optimizer = optax.adam(learning_rate=learning_rate)
    if max_grad_norm is not None:
        # TODO(btaba): Move gradient clipping to `training/gradients.py`.
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate),
        )

    loss_fn = functools.partial(
        losses.compute_mo_ppo_loss,
        moppo_network       = moppo_network,
        entropy_cost        = entropy_cost,
        discounting         = discounting,
        reward_scaling      = reward_scaling,
        gae_lambda          = gae_lambda,
        clipping_epsilon    = clipping_epsilon,
        normalize_advantage = normalize_advantage,
    )

    gradient_update_fn = gradients.gradient_update_fn(
        loss_fn, optimizer, pmap_axis_name=_PMAP_AXIS_NAME, has_aux=True
    )

    metrics_aggregator = metric_logger.EpisodeMetricsLogger(
        steps_between_logging=training_metrics_steps
        or env_step_per_training_step,
        progress_fn=progress_fn,
    )

    def minibatch_step(
        carry,
        data: acting.MultiObjectiveTransition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        (_, metrics), params, optimizer_state = gradient_update_fn(
            params,
            normalizer_params,
            data,
            key_loss,
            optimizer_state=optimizer_state,
        )

        return (optimizer_state, params, key), metrics

    def sgd_step(
        carry,
        unused_t,
        data: acting.MultiObjectiveTransition,
        normalizer_params: running_statistics.RunningStatisticsState,
    ):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)

        def convert_data(x: jnp.ndarray):
            x = jax.random.permutation(key_perm, x)
            x = jnp.reshape(x, (num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jax.tree_util.tree_map(convert_data, data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(minibatch_step, normalizer_params=normalizer_params),
            (optimizer_state, params, key_grad),
            shuffled_data,
            length=num_minibatches,
        )
        return (optimizer_state, params, key), metrics

    def training_step(
        carry: Tuple[MOTrainingState, envs.State, PRNGKey], unused_t, iteration
    ) -> Tuple[Tuple[MOTrainingState, envs.State, PRNGKey], Metrics]:
        training_state, state, key = carry
        key_sgd, key_generate_unroll, key_pref, new_key = jax.random.split(key, 4)
        directive = sample_preferences(
            key         = key_pref,
            it          = iteration,
            sampling    = sampling,
            K           = k,
            warmup_frac = warmup_frac,
            alpha       = alpha,
            num_evals   = num_evals,
            num_envs    = num_envs,
            num_objs   = state.reward.shape[1]
        )[0]

        policy = make_policy(
            params    = (
                training_state.normalizer_params,
                training_state.params.hypernetwork,
            ),
            directive = directive
        )

        def scan_unroll(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jax.random.split(current_key)
            next_state, data = acting.generate_unroll(
                env,
                current_state,
                policy,
                directive,
                current_key,
                unroll_length,
                extra_fields=('truncation', 'episode_metrics', 'episode_done'),
            )
            return (next_state, next_key), data
   
        (state, _), data = jax.lax.scan(
            scan_unroll,
            (state, key_generate_unroll),
            (),
            length=batch_size * num_minibatches // num_envs,
        )
        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data: acting.MultiObjectiveTransition = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
        )
        
        # Normalize rewards...
        # mins = data.reward.min(axis=(0, 1), keepdims=True)
        # maxs = data.reward.max(axis=(0, 1), keepdims=True)
        # data = data._replace(reward=(data.reward - mins) / (maxs - mins + 1e-8))
        
        if log_training_metrics:  # log unroll metrics
            jax.debug.callback(
                metrics_aggregator.update_episode_metrics,
                data.extras['state_extras']['episode_metrics'],
                data.extras['state_extras']['episode_done'],
            )
        # Update normalization params and normalize observations.
        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            data.observation,
            pmap_axis_name=_PMAP_AXIS_NAME,
        )
        (optimizer_state, params, _), metrics = jax.lax.scan(
            functools.partial(
                sgd_step, data=data, normalizer_params=normalizer_params
            ),
            (training_state.optimizer_state, training_state.params, key_sgd),
            (),
            length=num_updates_per_batch,
        )

        new_training_state = MOTrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_step_per_training_step,
        )
        return (new_training_state, state, new_key), metrics

    def training_epoch(
        training_state: MOTrainingState, state: envs.State, key: PRNGKey, it: int
    ) -> Tuple[MOTrainingState, envs.State, Metrics]:
        training_step_iter = functools.partial(
            training_step,
            iteration=it
        )
        (training_state, state, _), loss_metrics = jax.lax.scan(
            training_step_iter,
            (training_state, state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return training_state, state, loss_metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME, in_axes=(0, 0, 0, None))

    # Note that this is NOT a pure jittable method.
    def training_epoch_with_timing(
        training_state: MOTrainingState, env_state: envs.State, key: PRNGKey, it: int
    ) -> Tuple[MOTrainingState, envs.State, Metrics]:
        nonlocal training_walltime
        t = time.time()
        training_state, env_state = _strip_weak_type((training_state, env_state))
        result = training_epoch(training_state, env_state, key, it)
        training_state, env_state, metrics = _strip_weak_type(result)

        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

        epoch_training_time = time.time() - t
        training_walltime += epoch_training_time
        sps = (
            num_training_steps_per_epoch
            * env_step_per_training_step
            * max(num_resets_per_eval, 1)
        ) / epoch_training_time
        print(f'{num_training_steps_per_epoch=}, {env_step_per_training_step=}')
        metrics = {
            'training/sps': sps,
            'training/walltime': training_walltime,
            **{f'training/{name}': value for name, value in metrics.items()},
        }
        return training_state, env_state, metrics  # pytype: disable=bad-return-type  # py311-upgrade

    # Initialize model params and training state.
    init_params = losses.MOPPONetworkParams(
        hypernetwork = moppo_network.hypernetwork.init(key_policy),
    )
    # moppo_network: networks.MOPPONetworks = network_factory(
    #     key                        = key_policy,
    #     observation_size           = obs_shape,
    #     action_size                = env.action_size,
    #     num_objectives             = num_objectives,
    #     preprocess_observations_fn = normalize,
    #     target_policy_params       = init_policy_params
    # )
    # make_policy = networks.make_mo_inference_fn(moppo_network)

    obs_shape = jax.tree_util.tree_map(
        lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')), env_state.obs
    )
    if init_normalizer_params is None:
        init_normalizer_params = running_statistics.init_state(
            obs_shape
        )
    training_state = MOTrainingState(  # pytype: disable=wrong-arg-types  # jax-ndarray
        optimizer_state=optimizer.init(init_params),  # pytype: disable=wrong-arg-types  # numpy-scalars
        params=init_params,
        normalizer_params=init_normalizer_params,
        env_steps=types.UInt64(hi=0, lo=0),
    )

    if restore_checkpoint_path is not None:
        print('restoring checkpoint')
        params = checkpoint.load(restore_checkpoint_path)
        value_params = params[2] if restore_value_fn else init_params.value
        training_state = training_state.replace(
            normalizer_params=params[0],
            params=training_state.params.replace(
                policy=params[1], value=value_params
            ),
        )

    if restore_params is not None:
        print('restoring params')
        logging.info('Restoring TrainingState from `restore_params`.')
        value_params = restore_params[2] if restore_value_fn else init_params.value
        training_state = training_state.replace(
            normalizer_params=restore_params[0],
            params=training_state.params.replace(
                policy=restore_params[1], value=value_params
            ),
        )

    if num_timesteps == 0:
        return (
            make_policy,
            (
                training_state.normalizer_params,
                training_state.params.hypernetwork,
            ),
            {},
        )

    training_state: MOTrainingState = jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )

    # eval_env = _maybe_wrap_env(
    #     eval_env or environment,
    #     wrap_env,
    #     num_eval_envs,
    #     episode_length,
    #     action_repeat,
    #     device_count=1,  # eval on the host only
    #     key_env=eval_key,
    #     wrap_env_fn=wrap_env_fn,
    #     randomization_fn=randomization_fn,
    # )
    evaluator = acting.Evaluator(
        eval_env       = eval_env,
        eval_policy_fn = functools.partial(
            make_policy,
            deterministic = deterministic_eval,
            single_policy = False
        ),
        num_eval_envs  = num_eval_envs,
        episode_length = episode_length,
        action_repeat  = action_repeat,
        key            = eval_key,
        num_objs       = num_objectives
    )

    training_metrics = {}
    training_walltime = 0
    current_step = 0

    # Run initial eval
    metrics = {}
    if process_id == 0 and num_evals > 1 and run_evals:
        metrics = evaluator.run_evaluation(
            _unpmap((
                training_state.normalizer_params,
                training_state.params.hypernetwork,
            )),
            training_metrics={},
        )
        logging.info(metrics)
        progress_fn(0, metrics)

    # Run initial policy_params_fn.
    params = _unpmap((
        training_state.normalizer_params,
        training_state.params.hypernetwork,
    ))
    policy_params_fn(current_step, make_policy, params)

    for it in range(num_evals_after_init):
        logging.info('starting iteration %s %s', it, time.time() - xt)
        print(f'starting iteration {it} {time.time() - xt}')

        for _ in range(max(num_resets_per_eval, 1)):
            # optimization
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (training_state, env_state, training_metrics) = (
                training_epoch_with_timing(training_state, env_state, epoch_keys, it)
            )
            current_step = int(_unpmap(training_state.env_steps))

            key_envs = jax.vmap(
                lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)
            )(key_envs, key_envs.shape[1])
            # TODO(brax-team): move extra reset logic to the AutoResetWrapper.
            env_state = reset_fn(key_envs) if num_resets_per_eval > 0 else env_state

        if process_id != 0:
            continue

        # Process id == 0.
        params = _unpmap((
            training_state.normalizer_params,
            training_state.params.hypernetwork,
        ))

        policy_params_fn(current_step, make_policy, params)

        if save_checkpoint_path is not None:
            ckpt_config = checkpoint.network_config(
                observation_size=obs_shape,
                action_size=env.action_size,
                normalize_observations=normalize_observations,
                network_factory=network_factory,
            )
            checkpoint.save(
                save_checkpoint_path, current_step, params, ckpt_config
            )

        if num_evals > 0:
            metrics = training_metrics
            if run_evals:
                metrics = evaluator.run_evaluation(
                    params,
                    training_metrics,
                )
            logging.info(metrics)
            progress_fn(current_step, metrics)

    total_steps = current_step
    if not total_steps >= num_timesteps:
        raise AssertionError(
            f'Total steps {total_steps} is less than `num_timesteps`='
            f' {num_timesteps}.'
        )

    # If there was no mistakes the training_state should still be identical on all
    # devices.
    pmap.assert_is_replicated(training_state)
    params = _unpmap((
        training_state.normalizer_params,
        training_state.params.hypernetwork,
    ))
    logging.info('total steps: %s', total_steps)
    pmap.synchronize_hosts()
    return (make_policy, params, metrics)
