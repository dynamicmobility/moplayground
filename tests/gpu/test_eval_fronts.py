import jax
import numpy as np
import pytest

import moplayground as mop

pytestmark = pytest.mark.gpu

NUM_ENVS, N_STEPS = 8, 20


def test_fronts(trained_mo_run):
    cfg = trained_mo_run.cfg
    env, _ = mop.create_environment(cfg.env_name, cfg.env_params, backend='jnp')
    get_fronts, network_params = {
        'morlax': (mop.get_morlax_fronts, cfg.morlax_network_params),
        'amor': (mop.get_amor_fronts, cfg.amor_network_params),
    }[cfg.algorithm]
    rewards, tradeoffs = get_fronts(
        run_dirs=cfg.run_dir, network_params=network_params, num_objectives=cfg.num_objectives,
        rng=jax.random.PRNGKey(0), env=env, N_STEPS=N_STEPS, NUM_ENVS=NUM_ENVS, only_final=True,
    )
    assert rewards.shape == tradeoffs.shape == (1, NUM_ENVS, cfg.num_objectives)
    assert np.all(np.isfinite(rewards))
    hv, sparsity = mop.get_pareto_statistics(rewards[-1], ref_point=rewards[-1].min(0) - 1)
    assert hv > 0 and np.isfinite(sparsity)
