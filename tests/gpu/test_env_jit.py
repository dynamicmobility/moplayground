"""Training-backend (``jnp``) envs under jit + vmap, as the trainers use them."""
import jax
import numpy as np
import pytest

from tests.unit.test_envs import ENVS, make_env

pytestmark = pytest.mark.gpu

B = 4


@pytest.mark.parametrize('config_name', ENVS)
def test_env_jit_vmap(config_name):
    cfg, env = make_env(config_name, backend='jnp')
    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))
    state = reset(jax.random.split(jax.random.PRNGKey(0), B))
    actions = jax.random.uniform(jax.random.PRNGKey(1), (B, env.action_size), minval=-1, maxval=1)
    for _ in range(3):
        state = step(state, actions)
    assert state.reward.shape == (B, cfg.num_objectives)
    assert np.all(np.isfinite(state.reward))
