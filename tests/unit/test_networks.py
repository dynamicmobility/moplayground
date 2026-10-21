import jax
import jax.numpy as jnp
import numpy as np
import pytest

import moplayground as mop

OBS, ACT, M = 5, 3, 2


def by_path(tree):
    """``{key path: leaf}``, ignoring dict vs FrozenDict containers."""
    return {
        tuple(str(getattr(k, 'key', k)) for k in path): leaf
        for path, leaf in jax.tree_util.tree_flatten_with_path(jax.tree_util.tree_map(np.asarray, tree))[0]
    }


@pytest.fixture(scope='module')
def targets():
    key = jax.random.PRNGKey(0)
    policy = mop.MLP(obs_dim=OBS, hidden_sizes=(4,), action_dim=2 * ACT)
    value = mop.MLP(obs_dim=OBS, hidden_sizes=(4,), action_dim=1)
    x = jnp.zeros((1, OBS))
    return policy, policy.init(key, x), value, value.init(key, x)


def test_mlp_shape():
    mlp = mop.MLP(obs_dim=OBS, hidden_sizes=(8, 8), action_dim=ACT)
    params = mlp.init(jax.random.PRNGKey(0), jnp.zeros((1, OBS)))
    assert mlp.apply(params, jnp.ones((7, OBS))).shape == (7, ACT)
    assert mop.count_params(params) == (OBS * 8 + 8) + (8 * 8 + 8) + (8 * ACT + ACT)


def test_flatten_model(targets):
    _, policy_params, _, _ = targets
    flat, n, info = mop.flatten_model(policy_params['params'])
    assert flat.shape == (n,)
    assert n == mop.count_params(policy_params)
    assert [name for name, _ in info] == sorted(name for name, _ in info)


@pytest.mark.parametrize('cls', ['ActorCriticHypernet', 'DualA2CHypernet'])
def test_actor_critic_hypernets(targets, cls):
    policy, policy_params, value, value_params = targets
    hyper = getattr(mop, cls)(
        target_policy_dict=policy_params, target_value_dict=value_params,
        num_objs=M, obs_dim=OBS, hypersize=(8,), num_features=4,
    )
    pref = jnp.array([0.3, 0.7])
    params = hyper.init(jax.random.PRNGKey(1), pref)

    # Single directive -> unbatched target params with the target's structure.
    (pol, val), (pol_flat, val_flat), _ = hyper.apply(params, pref)
    got, want = by_path(pol), by_path(policy_params)
    assert got.keys() == want.keys()
    assert pol_flat.size == mop.count_params(policy_params)
    assert val_flat.size == mop.count_params(value_params)
    assert policy.apply(pol, jnp.ones((1, OBS))).shape == (1, 2 * ACT)
    assert value.apply(val, jnp.ones((1, OBS))).shape == (1, 1)

    # W starts at zero (W_variance=0), so the initial hypernet reproduces the
    # target network's initial weights for every directive.
    for path in want:
        np.testing.assert_allclose(got[path], want[path], atol=1e-6)

    # Batched directives -> leading batch axis on every leaf.
    (pol_b, _), _, _ = hyper.apply(params, jnp.array([[0.3, 0.7], [0.9, 0.1], [0.5, 0.5]]))
    batched = by_path(pol_b)
    for path in want:
        assert batched[path].shape == (3,) + want[path].shape
