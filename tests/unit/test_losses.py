import jax
import jax.numpy as jnp
import numpy as np
import pytest
from brax.training.agents.ppo.losses import compute_gae

import moplayground as mop
from tests.unit.test_factory import ACT, B, M, OBS, make_amor, make_morlax

T = 6


def test_mo_gae_matches_brax_per_objective():
    keys = jax.random.split(jax.random.PRNGKey(0), 4)
    truncation = (jax.random.uniform(keys[0], (T, B)) < 0.1).astype(jnp.float32)
    termination = (jax.random.uniform(keys[1], (T, B)) < 0.2).astype(jnp.float32)
    rewards = jax.random.normal(keys[2], (T, B, M))
    values = jax.random.normal(keys[3], (T, B, M))
    bootstrap = jnp.ones((B, M))

    vs, adv = mop.compute_mo_gae(truncation, termination, rewards, values, bootstrap, lambda_=0.9, discount=0.95)
    assert vs.shape == adv.shape == (T, B, M)
    for m in range(M):
        vs_m, adv_m = compute_gae(
            truncation, termination, rewards[..., m], values[..., m], bootstrap[:, m], lambda_=0.9, discount=0.95,
        )
        np.testing.assert_allclose(vs[..., m], vs_m, atol=1e-5)
        np.testing.assert_allclose(adv[..., m], adv_m, atol=1e-5)


def fake_data():
    keys = jax.random.split(jax.random.PRNGKey(0), 6)
    tradeoff = jax.random.dirichlet(keys[0], jnp.ones(M), (B,))
    return mop.MultiObjectiveTransition(
        observation=jax.random.normal(keys[1], (B, T, OBS)),
        action=jnp.zeros((B, T, ACT)),
        reward=jax.random.normal(keys[2], (B, T, M)),
        tradeoff=jnp.repeat(tradeoff[:, None], T, axis=1),
        discount=jnp.ones((B, T)),
        next_observation=jax.random.normal(keys[3], (B, T, OBS)),
        extras={
            'policy_extras': {
                'raw_action': jax.random.normal(keys[4], (B, T, ACT)),
                'log_prob': -jnp.ones((B, T)),
            },
            'state_extras': {'truncation': jnp.zeros((B, T))},
        },
    )


def _check(loss_fn, params):
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    assert np.isfinite(loss)
    assert set(metrics) == {'total_loss', 'policy_loss', 'v_loss', 'entropy_loss'}
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(np.all(np.isfinite(g)) for g in leaves)
    assert any(np.any(g != 0) for g in leaves)


@pytest.mark.parametrize('hypertype', ['single', 'dual'])
def test_morlax_loss(hypertype):
    nets, hparams = make_morlax(hypertype)
    data = fake_data()
    _check(
        lambda p: mop.compute_morlax_loss(p, None, data, jax.random.PRNGKey(5), nets),
        mop.MORLAXNetworkParams(hypernetwork=hparams),
    )


def test_amor_loss():
    nets, policy_params, value_params = make_amor()
    data = fake_data()
    _check(
        lambda p: mop.compute_amor_loss(p, None, data, jax.random.PRNGKey(5), nets),
        mop.AMORNetworkParams(policy=policy_params, value=value_params),
    )
