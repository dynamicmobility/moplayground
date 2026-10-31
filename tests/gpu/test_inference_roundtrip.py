import jax
import numpy as np
import pytest
from brax.training.agents.ppo import checkpoint
from etils import epath

import moplayground as mop

pytestmark = pytest.mark.gpu

NETWORK_PROPERTY = {'morlax': 'morlax_network_params', 'amor': 'amor_network_params'}
CHECKPOINT_LEN = {'morlax': 2, 'amor': 3}  # (normalizer, hypernet) / (normalizer, policy, value)


def test_checkpoint_layout(trained_mo_run):
    path = mop.get_last_model(trained_mo_run.cfg.run_dir)
    params = checkpoint.load(epath.Path(path.resolve()))
    assert len(params) == CHECKPOINT_LEN[trained_mo_run.algorithm]


def test_load_mo_policy_rollout(trained_mo_run):
    cfg = trained_mo_run.cfg
    env, _ = mop.create_environment(cfg.env_name, cfg.env_params, backend='jnp')
    tradeoff = np.ones(cfg.num_objectives) / cfg.num_objectives
    policy = jax.jit(mop.load_mo_policy(
        algorithm=cfg.algorithm,
        network_params=getattr(cfg, NETWORK_PROPERTY[cfg.algorithm]),
        num_objectives=cfg.num_objectives,
        run_dir=cfg.run_dir,
        tradeoff=tradeoff,
    ))
    step = jax.jit(env.step)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    key = jax.random.PRNGKey(1)
    for _ in range(5):
        action, _ = policy(state.obs, key)
        assert action.shape == (env.action_size,)
        assert np.all(np.abs(action) <= 1.0)
        state = step(state, action)
        assert np.all(np.isfinite(state.reward))
