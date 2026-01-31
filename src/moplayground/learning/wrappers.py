from brax.envs.base import Env, State, Wrapper
from brax.envs.wrappers.training import EvalMetrics
import jax
from jax import numpy as jnp

class MultiObjectiveEpisodeWrapper(Wrapper):
    """Maintains episode step count and sets done at episode end."""

    def __init__(self, env: Env, episode_length: int, action_repeat: int):
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info['steps'] = jnp.zeros(rng.shape[:-1])
        state.info['truncation'] = jnp.zeros(rng.shape[:-1])
        # Keep separate record of episode done as state.info['done'] can be erased
        # by AutoResetWrapper
        state.info['episode_done'] = jnp.zeros(rng.shape[:-1])
        episode_metrics = dict()
        episode_metrics['sum_reward'] = jnp.zeros_like(state.reward)
        episode_metrics['length'] = jnp.zeros(rng.shape[:-1])
        for metric_name in state.metrics.keys():
            episode_metrics[metric_name] = jnp.zeros(rng.shape[:-1])
        state.info['episode_metrics'] = episode_metrics
        return state

    def step(self, state: State, action: jax.Array) -> State:
        def f(state, _):
            nstate = self.env.step(state, action)
            return nstate, nstate.reward

        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jnp.sum(rewards, axis=0))
        steps = state.info['steps'] + self.action_repeat
        one = jnp.ones_like(state.done)
        zero = jnp.zeros_like(state.done)
        episode_length = jnp.array(self.episode_length, dtype=jnp.int32)
        done = jnp.where(steps >= episode_length, one, state.done)
        state.info['truncation'] = jnp.where(
            steps >= episode_length, 1 - state.done, zero
        )
        state.info['steps'] = steps

        # Aggregate state metrics into episode metrics
        prev_done = state.info['episode_done']
        state.info['episode_metrics']['sum_reward'] += jnp.sum(rewards, axis=0)
        state.info['episode_metrics']['sum_reward'] *= (1 - prev_done[:, jnp.newaxis])
        state.info['episode_metrics']['length'] += self.action_repeat
        state.info['episode_metrics']['length'] *= (1 - prev_done)
        for metric_name in state.metrics.keys():
            if metric_name != 'reward':
                state.info['episode_metrics'][metric_name] += state.metrics[metric_name]
                state.info['episode_metrics'][metric_name] *= (1 - prev_done)
        state.info['episode_done'] = done
        return state.replace(done=done)
    
class MultiObjectiveEvalWrapper(Wrapper):
    """Brax env with eval metrics."""

    def reset(self, rng: jax.Array) -> State:
        reset_state = self.env.reset(rng)
        reset_state.metrics['reward'] = reset_state.reward
        eval_metrics = EvalMetrics(
            episode_metrics=jax.tree_util.tree_map(
                jnp.zeros_like, reset_state.metrics
            ),
            active_episodes=jnp.ones(reset_state.reward.shape[0]),
            episode_steps=jnp.zeros(reset_state.reward.shape[0]),
        )
        reset_state.info['eval_metrics'] = eval_metrics
        return reset_state

    def step(self, state: State, action: jax.Array) -> State:
        state_metrics = state.info['eval_metrics']
        if not isinstance(state_metrics, EvalMetrics):
            raise ValueError(
                f'Incorrect type for state_metrics: {type(state_metrics)}'
            )
        del state.info['eval_metrics']
        nstate = self.env.step(state, action)
        episode_steps = jnp.where(
            state_metrics.active_episodes,
            nstate.info['steps'],
            state_metrics.episode_steps,
        )
        old_reward = state_metrics.episode_metrics['reward'].copy()
        new_reward = nstate.metrics['reward'].copy()
        del state_metrics.episode_metrics['reward']
        del nstate.metrics['reward']
        episode_metrics = jax.tree_util.tree_map(
            lambda a, b: a + b * state_metrics.active_episodes,
            state_metrics.episode_metrics,
            nstate.metrics,
        )
        nstate.metrics['reward'] = nstate.reward
        episode_metrics['reward'] = old_reward + new_reward * state_metrics.active_episodes[:, jnp.newaxis]
        active_episodes = state_metrics.active_episodes * (1 - nstate.done)

        eval_metrics = EvalMetrics(
            episode_metrics=episode_metrics,
            active_episodes=active_episodes,
            episode_steps=episode_steps,
        )
        nstate.info['eval_metrics'] = eval_metrics
        return nstate
