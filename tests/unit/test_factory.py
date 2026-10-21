import jax
import jax.numpy as jnp
import numpy as np
import pytest

import moplayground as mop

OBS, ACT, M, B = 5, 3, 2, 4
MLP_SIZES = dict(policy_hidden_layer_sizes=(8,), value_hidden_layer_sizes=(8,))


def make_morlax(hypertype):
    nets = mop.make_morlax_networks(
        observation_size=OBS, action_size=ACT, num_objectives=M, hypersize=(8,),
        key=jax.random.PRNGKey(0), hypertype=hypertype, num_features=4, **MLP_SIZES,
    )
    return nets, nets.hypernetwork.init(jax.random.PRNGKey(1))


def make_amor():
    nets = mop.make_amor_networks(
        observation_size=OBS, action_size=ACT, num_objectives=M, key=jax.random.PRNGKey(0), **MLP_SIZES,
    )
    return nets, nets.policy_network.init(jax.random.PRNGKey(1)), nets.value_network.init(jax.random.PRNGKey(2))


def test_morlax_bad_hypertype():
    with pytest.raises(Exception, match='hypertype'):
        make_morlax('MLP')


@pytest.mark.parametrize('hypertype', ['single', 'dual'])
def test_morlax_inference(hypertype):
    nets, hparams = make_morlax(hypertype)
    make_policy = mop.make_hypernetwork_inference_fn(nets)
    obs = jax.random.normal(jax.random.PRNGKey(2), (B, OBS))
    key = jax.random.PRNGKey(3)

    # One directive shared by the whole batch.
    policy = make_policy((None, hparams), jnp.array([0.5, 0.5]))
    action, extras = policy(obs, key)
    assert action.shape == (B, ACT)
    assert np.all(np.abs(action) <= 1.0)
    assert extras['log_prob'].shape == (B,)

    # One directive per env.
    directives = jax.random.dirichlet(jax.random.PRNGKey(4), jnp.ones(M), (B,))
    action, _ = make_policy((None, hparams), directives, deterministic=True)(obs, key)
    assert action.shape == (B, ACT)


@pytest.mark.parametrize('hypertype', ['single', 'dual'])
def test_morlax_hypernet_normalizes_directive(hypertype):
    nets, hparams = make_morlax(hypertype)
    a = nets.hypernetwork.get_flat_mlps(hparams, jnp.array([1.0, 3.0]))
    b = nets.hypernetwork.get_flat_mlps(hparams, jnp.array([0.25, 0.75]))
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, atol=1e-6)


def test_amor_networks():
    nets, policy_params, value_params = make_amor()
    obs = jax.random.normal(jax.random.PRNGKey(2), (B, OBS))
    directive = jnp.tile(jnp.array([0.2, 0.8]), (B, 1))
    assert nets.value_network.apply(None, value_params, obs, directive).shape == (B, M)

    policy = mop.make_amor_inference_fn(nets)((None, policy_params), deterministic=True)
    action, _ = policy(obs, directive, jax.random.PRNGKey(3))
    assert action.shape == (B, ACT)
    assert np.all(np.abs(action) <= 1.0)

    # The directive is a real input: changing it changes the action.
    other, _ = policy(obs, jnp.tile(jnp.array([0.9, 0.1]), (B, 1)), jax.random.PRNGKey(3))
    assert not np.allclose(action, other)
