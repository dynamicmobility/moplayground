"""Wrapper and rollout logic, tested against a tiny analytic env so the
results can be checked exactly (and nothing has to JIT a MuJoCo model)."""
import jax
import jax.numpy as jnp
import numpy as np
from brax.envs.base import Env, State
from brax.envs.wrappers.training import VmapWrapper

import moplayground as mop

M, B = 2, 3


class VectorRewardEnv(Env):
    """Reward is the action itself (so a policy can choose its returns);
    terminates once the first action component goes negative."""

    def reset(self, rng):
        return State(pipeline_state=None, obs=jnp.zeros(M), reward=jnp.zeros(M),
                     done=jnp.zeros(()), metrics={'m': jnp.zeros(())}, info={})

    def step(self, state, action):
        done = jnp.where(action[0] < 0, 1.0, state.done)
        return state.replace(obs=state.obs + 1, reward=action, done=done, metrics={**state.metrics, 'm': jnp.ones(())})

    @property
    def observation_size(self):
        return M

    @property
    def action_size(self):
        return M

    @property
    def backend(self):
        return 'none'


def test_episode_wrapper_truncates_and_accumulates():
    env = mop.MultiObjectiveEpisodeWrapper(VmapWrapper(VectorRewardEnv()), episode_length=3, action_repeat=1)
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), B))
    action = jnp.tile(jnp.array([1.0, 2.0]), (B, 1))
    for t in range(3):
        assert not state.done.any()
        state = env.step(state, action)
    np.testing.assert_array_equal(state.done, 1)
    np.testing.assert_array_equal(state.info['truncation'], 1)
    np.testing.assert_allclose(state.info['episode_metrics']['sum_reward'], np.tile([3.0, 6.0], (B, 1)))
    np.testing.assert_allclose(state.info['episode_metrics']['length'], 3)
    np.testing.assert_allclose(state.info['episode_metrics']['m'], 3)


def test_episode_wrapper_action_repeat():
    env = mop.MultiObjectiveEpisodeWrapper(VmapWrapper(VectorRewardEnv()), episode_length=10, action_repeat=2)
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), B))
    state = env.step(state, jnp.ones((B, M)))
    np.testing.assert_allclose(state.reward, 2.0)
    np.testing.assert_allclose(state.info['steps'], 2)


def test_eval_wrapper_stops_counting_after_done():
    env = mop.MultiObjectiveEvalWrapper(
        mop.MultiObjectiveEpisodeWrapper(VmapWrapper(VectorRewardEnv()), episode_length=100, action_repeat=1)
    )
    state = env.reset(jax.random.split(jax.random.PRNGKey(0), B))
    good = jnp.tile(jnp.array([1.0, 2.0]), (B, 1))
    # env 0 terminates on the second step; the rest keep going
    bad = good.at[0, 0].set(-1.0)
    for action in (good, bad, good, good):
        state = env.step(state, action)
    rewards = np.asarray(state.info['eval_metrics'].episode_metrics['reward'])
    np.testing.assert_allclose(rewards[0], [1.0 - 1.0, 4.0])
    np.testing.assert_allclose(rewards[1:], np.tile([4.0, 8.0], (B - 1, 1)))
    np.testing.assert_array_equal(state.info['eval_metrics'].active_episodes, [0, 1, 1])


def test_pareto_rollout_returns_directive_scaled_rewards():
    env = VectorRewardEnv()

    def make_policy(params, deterministic, directive):
        return lambda obs, key: (directive * params, {})

    run = mop.get_pareto_rollout(env, N_STEPS=4, make_policy=make_policy)
    directives = jnp.array([[0.25, 0.75], [1.0, 0.0], [0.5, 0.5]])
    (_, returns), _ = run(jax.random.split(jax.random.PRNGKey(0), 3), directives, 2.0)
    np.testing.assert_allclose(returns, 4 * 2.0 * directives)


def test_compute_fronts_shapes_and_saving(tmp_path):
    runs = [tmp_path / 'a', tmp_path / 'b']
    for r in runs:
        r.mkdir()
    rewards, tradeoffs = mop.compute_fronts(
        run_dirs=runs, num_objectives=M, rng=jax.random.PRNGKey(0), env=VectorRewardEnv(),
        N_STEPS=2, NUM_ENVS=5,
        make_policy=lambda params, deterministic, directive: (lambda obs, key: (directive * params, {})),
        load_params_fn=lambda file: float(file),
        model_files_per_run=[['1', '2'], ['3']],
        save_results=True,
    )
    assert rewards.shape == tradeoffs.shape == (3, 5, M)
    np.testing.assert_allclose(tradeoffs.sum(-1), 1.0, atol=1e-5)
    np.testing.assert_allclose(rewards[2], 3 * 2 * tradeoffs[2], rtol=1e-5)
    assert sorted(p.name for p in runs[0].iterdir()) == ['obj0.txt', 'obj1.txt']
    assert sorted(p.name for p in runs[1].iterdir()) == ['obj0.txt']
