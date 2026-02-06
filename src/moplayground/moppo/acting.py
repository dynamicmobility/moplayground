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

"""Brax training acting functions."""

import time
from typing import Callable, Sequence, Tuple
from typing import Any, Tuple, NamedTuple
from brax.training.acme.types import NestedArray

from brax import envs
from brax.training.types import Metrics
from brax.training.types import Policy
from brax.training.types import PolicyParams
from brax.training.types import PRNGKey
from brax.training.types import Transition
from moplayground.learning.wrappers import MultiObjectiveEvalWrapper

import jax
import numpy as np
from jax import numpy as jnp
import functools

State = envs.State
Env = envs.Env

class MultiObjectiveTransition(NamedTuple):
    """Container for a transition."""

    observation: NestedArray
    action: NestedArray
    reward: NestedArray
    directive: NestedArray
    discount: NestedArray
    next_observation: NestedArray
    extras: NestedArray = ()  # pytype: disable=annotation-type-mismatch  # jax-ndarray

def actor_step(
    env: Env,
    env_state: State,
    policy: Policy,
    directive,
    key: PRNGKey,
    extra_fields: Sequence[str] = (),
) -> Tuple[State, MultiObjectiveTransition]:
    """Collect data."""
    actions, policy_extras = policy(env_state.obs, key)
    nstate = env.step(env_state, actions)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    return nstate, MultiObjectiveTransition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=env_state.obs,
        action=actions,
        reward=nstate.reward,
        directive=directive.copy(),
        discount=1 - nstate.done,
        next_observation=nstate.obs,
        extras={'policy_extras': policy_extras, 'state_extras': state_extras},
    )


def generate_unroll(
    env: Env,
    env_state: State,
    policy: Policy,
    directive: jax.Array,
    key: PRNGKey,
    unroll_length: int,
    extra_fields: Sequence[str] = (),
) -> Tuple[State, MultiObjectiveTransition]:
    """Collect trajectories of given unroll_length."""

    @jax.jit
    def f(carry, unused_t):
        state, current_key = carry
        current_key, next_key = jax.random.split(current_key)
        nstate, transition = actor_step(
            env, state, policy, directive, current_key, extra_fields=extra_fields
        )
        return (nstate, next_key), transition

    (final_state, _), data = jax.lax.scan(
        f, (env_state, key), (), length=unroll_length
    )
    return final_state, data


# TODO(eorsini): Consider moving this to its own file.
class Evaluator:
    """Class to run evaluations."""

    def __init__(
        self,
        eval_env: envs.Env,
        num_objs: int,
        eval_policy_fn,
        num_eval_envs: int,
        episode_length: int,
        action_repeat: int,
        key: PRNGKey,
    ):
        self._key = key
        self._eval_walltime = 0.0

        eval_env = MultiObjectiveEvalWrapper(eval_env)

        def generate_eval_unroll(
            policy_params, key: PRNGKey, directives
        ) -> State:
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)
            return generate_unroll(
                eval_env,
                eval_first_state,
                eval_policy_fn(policy_params, directives),
                directives,
                key,
                unroll_length=episode_length // action_repeat,
            )[0]

        self._generate_eval_unroll = jax.jit(generate_eval_unroll)
        self._steps_per_unroll = episode_length * num_eval_envs
        self.num_eval_envs = num_eval_envs
        self.num_objs = num_objs

    def run_evaluation(
        self,
        policy_params,
        training_metrics: Metrics,
        aggregate_episodes: bool = True,
    ) -> Metrics:
        """Run one epoch of evaluation."""
        self._key, unroll_key = jax.random.split(self._key)

        t = time.time()
        _, w_key = jax.random.split(self._key)
        directives = jax.random.dirichlet(key=w_key, alpha=jnp.ones(self.num_objs), shape=self.num_eval_envs)
        # directives = jnp.ones((self.num_eval_envs, self.num_objs)) / self.num_objs
        eval_state = self._generate_eval_unroll(policy_params, unroll_key, directives)
        eval_metrics = eval_state.info['eval_metrics']
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}
        # for fn in [np.mean, np.std]:
        #     suffix = '_std' if fn == np.std else ''
        #     metrics.update({
        #         f'eval/episode_{name}{suffix}': (
        #             fn(value, axis=0) if aggregate_episodes else value
        #         )
        #         for name, value in eval_metrics.episode_metrics.items()
        #     })
        # metrics['eval/avg_episode_length'] = np.mean(eval_metrics.episode_steps)
        # metrics['eval/std_episode_length'] = np.std(eval_metrics.episode_steps)
        # metrics['eval/epoch_eval_time'] = epoch_eval_time
        # metrics['eval/sps'] = self._steps_per_unroll / epoch_eval_time
        # self._eval_walltime = self._eval_walltime + epoch_eval_time
        # metrics = {
        #     'eval/walltime': self._eval_walltime,
        #     **training_metrics,
        #     **metrics,
        # }
        metrics = eval_metrics.episode_metrics
        metrics['directive'] = directives

        return metrics  # pytype: disable=bad-return-type  # jax-ndarray




# def step_fn(carry, x, policy, env, rng):
#     state, old_reward = carry
#     action, _ = policy(state.obs, rng)
#     new_state = env.step(state, action)
#     return (new_state, old_reward + new_state.reward * (1 - new_state.done)), None
    
# def rollout(key, directive, params, env, make_policy, N_STEPS):
#     policy = make_policy(
#         params        = params,
#         deterministic = True,
#         directive     = directive,
#         single_policy = True
#     )
#     scan_step_fn = functools.partial(
#         step_fn,
#         policy=policy,
#         env=env,
#         rng=key
#     )
#     state = env.reset(key)
#     carry = (state, state.reward)
#     return jax.lax.scan(scan_step_fn, carry, (), N_STEPS)

# class MOEvaluator:
#     """Class to run evaluations."""

#     def __init__(
#         self,
#         eval_env: envs.Env,
#         eval_policy_fn,
#         num_eval_envs: int,
#         episode_length: int,
#         action_repeat: int,
#         key: PRNGKey,
#     ):
#         self._key = key
#         self._eval_walltime = 0.0

#         # eval_env = MultiObjectiveEvalWrapper(eval_env)
#         self.num_eval_envs = num_eval_envs
#         run_rollout = functools.partial(
#             rollout,
#             make_policy=eval_policy_fn,
#             N_STEPS=episode_length,
#             env=eval_env
#         )
#         self.run_rollouts = jax.jit(jax.vmap(run_rollout, in_axes=(0, 0, None)))
        
#     def run_evaluation(
#         self,
#         moppo_params,
#         training_metrics: Metrics,
#         aggregate_episodes: bool = True,
#     ) -> Metrics:
#         """Run one epoch of evaluation."""
#         self._key, directive_key = jax.random.split(self._key)
#         env_keys = jax.random.split(self._key, self.num_eval_envs)

#         t = time.time()
#         directives = jax.random.dirichlet(key=directive_key, alpha=jnp.ones(2), shape=self.num_eval_envs)
#         (_, rewards), _ = self.run_rollouts(env_keys, directives, moppo_params)
#         print(rewards.shape)
#         quit()

#         metrics = {}
#         metrics['reward'] = rewards
#         return metrics  # pytype: disable=bad-return-type  # jax-ndarray
