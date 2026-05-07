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

"""AMOR: tradeoff-conditioned PPO baseline (no hypernetwork).

The policy and value networks consume ``concat(normalized_obs, raw_directive)``.
Directives are sampled per-env per-iteration (same scheme as MORLAX) and held
constant within an unroll. Inference exposes a ``policy(obs, directive, key)``
closure so callers can change the tradeoff at any step without rebuilding the
policy.
"""

import functools
import time
from typing import Any, Callable, Mapping, Optional, Tuple, Union

from absl import logging
from brax import base
from brax import envs
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
from moplayground.moppo.morlax import sample_preferences


InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]
Metrics = types.Metrics

_PMAP_AXIS_NAME = 'i'


@flax.struct.dataclass
class AMORTrainingState:
    """Contains training state for the AMOR learner."""

    optimizer_state   : optax.OptState
    params            : losses.AMORNetworkParams
    normalizer_params : running_statistics.RunningStatisticsState
    env_steps         : types.UInt64


def _unpmap(v):
    return jax.tree_util.tree_map(lambda x: x[0], v)


def _strip_weak_type(tree):
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
        raise ValueError('episode_length must be specified in amor.train')
    v_randomization_fn = None
    if randomization_fn is not None:
        randomization_batch_size = num_envs // device_count
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
    )
    return env


def _get_process_count():
    process_count = jax.process_count()
    process_id = jax.process_index()
    return process_count, process_id


def _get_device_count(max_devices_per_host, process_count, process_id):
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


def _make_baked_inference_fn(amor_inference_fn):
    """Adapter that turns AMOR's call-time-directive inference fn into the
    ``(params, directive, deterministic) -> policy(obs, key)`` signature that
    ``acting.Evaluator`` and the unroll loop expect.
    """
    def baked_inference_fn(params, directive, deterministic=False):
        raw_policy = amor_inference_fn(params, deterministic=deterministic)
        def policy(obs, key):
            return raw_policy(obs, directive, key)
        return policy
    return baked_inference_fn


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
    # directive sampling (shared with MORLAX)
    alpha                   : float = 1.0,
    warmup_frac             : float = 0.0,
    sampling                : str = 'dense',
    k                       : int = 4,
    network_factory: types.NetworkFactory[
        factory.AMORNetworks
    ]                       = factory.make_amor_networks,
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

    env_step_per_training_step = batch_size * unroll_length * num_minibatches * action_repeat

    num_evals_after_init = max(num_evals - 1, 1)
    num_training_steps_per_epoch = num_timesteps / (num_evals_after_init * env_step_per_training_step * max(num_resets_per_eval, 1))
    num_training_steps_per_epoch = np.ceil(num_training_steps_per_epoch).astype(int)

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
        eval_env,
        wrap_env,
        num_eval_envs,
        episode_length,
        action_repeat,
        device_count        = 1,
        key_env             = eval_key,
        wrap_env_fn         = wrap_env_fn,
    )

    key_envs, env_state, reset_fn = _make_reset_envs(
        local_devices_to_use,
        use_pmap_on_reset,
        env,
        key_env,
        num_envs,
        process_count
    )

    obs_shape = jax.tree_util.tree_map(lambda x: x.shape[2:], env_state.obs)
    num_objectives = env_state.reward.shape[2]

    normalize = lambda x, y: x
    if normalize_observations:
        normalize = running_statistics.normalize
    amor_networks: factory.AMORNetworks = network_factory(
        key                        = key_policy,
        observation_size           = obs_shape,
        action_size                = env.action_size,
        num_objectives             = num_objectives,
        preprocess_observations_fn = normalize,
    )
    amor_inference_fn = factory.make_amor_inference_fn(amor_networks)
    baked_inference_fn = _make_baked_inference_fn(amor_inference_fn)

    optimizer = optax.adam(learning_rate=learning_rate)
    if max_grad_norm is not None:
        optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=learning_rate),
        )

    loss_fn = functools.partial(
        losses.compute_amor_loss,
        amor_networks       = amor_networks,
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
        carry: Tuple[AMORTrainingState, envs.State, PRNGKey], unused_t, iteration
    ) -> Tuple[Tuple[AMORTrainingState, envs.State, PRNGKey], Metrics]:
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
            num_objs    = state.reward.shape[1]
        )[0]  # strip leading 1 axis -> (num_envs, num_objs)

        # AMOR's flat policy: directive is concatenated to obs inside the network.
        # Within an unroll the directive is held constant per-env (matches MORLAX).
        policy = baked_inference_fn(
            params    = (training_state.normalizer_params, training_state.params.policy),
            directive = directive,
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
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data: acting.MultiObjectiveTransition = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (-1,) + x.shape[2:]), data
        )

        if log_training_metrics:
            jax.debug.callback(
                metrics_aggregator.update_episode_metrics,
                data.extras['state_extras']['episode_metrics'],
                data.extras['state_extras']['episode_done'],
            )

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

        new_training_state = AMORTrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_step_per_training_step,
        )
        return (new_training_state, state, new_key), metrics

    def training_epoch(
        training_state: AMORTrainingState, state: envs.State, key: PRNGKey, it: int
    ) -> Tuple[AMORTrainingState, envs.State, Metrics]:
        training_step_iter = functools.partial(training_step, iteration=it)
        (training_state, state, _), loss_metrics = jax.lax.scan(
            training_step_iter,
            (training_state, state, key),
            (),
            length=num_training_steps_per_epoch,
        )
        loss_metrics = jax.tree_util.tree_map(jnp.mean, loss_metrics)
        return training_state, state, loss_metrics

    training_epoch = jax.pmap(training_epoch, axis_name=_PMAP_AXIS_NAME, in_axes=(0, 0, 0, None))

    def training_epoch_with_timing(
        training_state: AMORTrainingState, env_state: envs.State, key: PRNGKey, it: int
    ) -> Tuple[AMORTrainingState, envs.State, Metrics]:
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
        return training_state, env_state, metrics

    # Initialize model params and training state.
    if init_policy_params is None:
        init_policy_params = amor_networks.policy_network.init(key_policy)
    if init_value_params is None:
        init_value_params = amor_networks.value_network.init(key_value)

    init_params = losses.AMORNetworkParams(
        policy = init_policy_params,
        value  = init_value_params,
    )

    obs_shape_specs = jax.tree_util.tree_map(
        lambda x: specs.Array(x.shape[-1:], jnp.dtype('float32')), env_state.obs
    )
    if init_normalizer_params is None:
        init_normalizer_params = running_statistics.init_state(obs_shape_specs)
    training_state = AMORTrainingState(
        optimizer_state=optimizer.init(init_params),
        params=init_params,
        normalizer_params=init_normalizer_params,
        env_steps=types.UInt64(hi=0, lo=0),
    )

    if restore_checkpoint_path is not None:
        print('restoring checkpoint')
        restored = checkpoint.load(restore_checkpoint_path)
        # restored = (normalizer, policy, value) — standard 3-tuple
        value_params = restored[2] if restore_value_fn else init_params.value
        training_state = training_state.replace(
            normalizer_params=restored[0],
            params=training_state.params.replace(
                policy=restored[1], value=value_params
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
            amor_inference_fn,
            (
                training_state.normalizer_params,
                training_state.params.policy,
                training_state.params.value,
            ),
            {},
        )

    training_state: AMORTrainingState = jax.device_put_replicated(
        training_state, jax.local_devices()[:local_devices_to_use]
    )

    evaluator = acting.Evaluator(
        eval_env       = eval_env,
        eval_policy_fn = functools.partial(
            baked_inference_fn,
            deterministic = deterministic_eval,
        ),
        num_eval_envs  = num_eval_envs,
        episode_length = episode_length,
        action_repeat  = action_repeat,
        key            = eval_key,
        num_objs       = num_objectives,
    )

    training_metrics = {}
    training_walltime = 0
    current_step = 0

    metrics = {}
    if process_id == 0 and num_evals > 1 and run_evals:
        metrics = evaluator.run_evaluation(
            _unpmap((
                training_state.normalizer_params,
                training_state.params.policy,
            )),
            training_metrics={},
        )
        logging.info(metrics)
        progress_fn(0, metrics)

    params = _unpmap((
        training_state.normalizer_params,
        training_state.params.policy,
        training_state.params.value,
    ))
    policy_params_fn(current_step, baked_inference_fn, params)

    for it in range(num_evals_after_init):
        logging.info('starting iteration %s %s', it, time.time() - xt)
        print(f'starting iteration {it} {time.time() - xt}')

        for _ in range(max(num_resets_per_eval, 1)):
            epoch_key, local_key = jax.random.split(local_key)
            epoch_keys = jax.random.split(epoch_key, local_devices_to_use)
            (training_state, env_state, training_metrics) = (
                training_epoch_with_timing(training_state, env_state, epoch_keys, it)
            )
            current_step = int(_unpmap(training_state.env_steps))

            key_envs = jax.vmap(
                lambda x, s: jax.random.split(x[0], s), in_axes=(0, None)
            )(key_envs, key_envs.shape[1])
            env_state = reset_fn(key_envs) if num_resets_per_eval > 0 else env_state

        if process_id != 0:
            continue

        params = _unpmap((
            training_state.normalizer_params,
            training_state.params.policy,
            training_state.params.value,
        ))

        policy_params_fn(current_step, baked_inference_fn, params)

        if save_checkpoint_path is not None:
            ckpt_config = checkpoint.network_config(
                observation_size=obs_shape,
                action_size=env.action_size,
                normalize_observations=normalize_observations,
                network_factory=network_factory,
            )
            checkpoint.save(save_checkpoint_path, current_step, params, ckpt_config)

        if num_evals > 0:
            metrics = training_metrics
            if run_evals:
                metrics = evaluator.run_evaluation(
                    _unpmap((
                        training_state.normalizer_params,
                        training_state.params.policy,
                    )),
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

    pmap.assert_is_replicated(training_state)
    params = _unpmap((
        training_state.normalizer_params,
        training_state.params.policy,
        training_state.params.value,
    ))
    logging.info('total steps: %s', total_steps)
    pmap.synchronize_hosts()
    return (amor_inference_fn, params, metrics)
