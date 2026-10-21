from types import SimpleNamespace

import jax
import numpy as np
import pytest
from ml_collections import config_dict

import moplayground as mop
from tests.helpers import REPO_ROOT

ENVS = [
    pytest.param('mocheetah', id='MOCheetah'),
    pytest.param('mowalker', id='MOWalker'),
    pytest.param('mohopper', id='MOHopper'),
    pytest.param('moant', id='MOAnt'),
    pytest.param('mohumanoid', id='MOHumanoid'),
]


def make_env(config_name, backend='np'):
    cfg = mop.MOConfig(REPO_ROOT / 'config' / f'{config_name}.yaml')
    env, _ = mop.create_environment(cfg.env_name, cfg.env_params, backend=backend)
    return cfg, env


@pytest.mark.parametrize('config_name', ENVS)
def test_env_steps(config_name):
    cfg, env = make_env(config_name)
    state = env.reset(jax.random.PRNGKey(0))
    assert np.shape(state.reward) == (cfg.num_objectives,)
    rng = np.random.default_rng(0)
    for _ in range(5):
        state = env.step(state, rng.uniform(-1, 1, env.action_size))
        assert np.shape(state.reward) == (cfg.num_objectives,)
        assert np.all(np.isfinite(state.reward))
        obs = state.obs['state'] if isinstance(state.obs, dict) else state.obs
        assert np.all(np.isfinite(obs))


def test_multi2single_objective():
    cfg, env = make_env('mocheetah')
    weights = np.array([0.2, 0.3, 0.5])
    single = mop.Multi2SingleObjective(env, weights)
    action = np.zeros(env.action_size)
    mo_state = env.step(env.reset(jax.random.PRNGKey(0)), action)
    so_state = single.step(single.reset(jax.random.PRNGKey(0)), action)
    assert np.ndim(so_state.reward) == 0
    np.testing.assert_allclose(so_state.reward, mo_state.reward @ weights, rtol=1e-6)
    assert single.action_size == env.action_size  # attribute delegation


def test_unknown_env_raises(cheetah_cfg):
    with pytest.raises(Exception, match='Unknown'):
        mop.create_environment('NotAnEnv', cheetah_cfg.env_params)


def test_reward_vector_combines_objectives():
    fake = SimpleNamespace(
        objectives=[['a', 'b'], ['c']],
        shared_objectives=['alive'],
        params=config_dict.ConfigDict({'reward': {'weights': {'a': 1.0, 'b': 2.0, 'c': 10.0, 'alive': 0.5}}}),
        _np=np,
        get_metrics=lambda metrics, rewards: metrics,
    )
    rewards = {'a': 1.0, 'b': 3.0, 'c': 0.1, 'alive': 2.0}
    reward, _ = mop.MultiObjectiveBase.get_reward_and_metrics(fake, rewards, {})
    np.testing.assert_allclose(reward, [1 + 6 + 1, 1 + 1])
