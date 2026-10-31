import jax
import numpy as np
import pytest

import moplayground as mop

NUM_ENVS, NUM_EVALS = 16, 10


def sample(sampling, M=3, K=8, warmup_frac=0.0, it=5):
    w = mop.sample_preferences(jax.random.PRNGKey(0), it, sampling, K, warmup_frac, 1.0, NUM_EVALS, NUM_ENVS, M)
    return np.asarray(w)


@pytest.mark.parametrize('sampling', ['dense', 'sparse-heavytail'])
@pytest.mark.parametrize('M', [2, 3])
def test_on_simplex(sampling, M):
    w = sample(sampling, M=M)
    assert w.shape == (1, NUM_ENVS, M)
    assert np.all(w >= 0)
    np.testing.assert_allclose(w.sum(-1), 1.0, atol=1e-5)


def test_heavytail_includes_corners_and_k_distinct():
    w = sample('sparse-heavytail', M=3, K=8)[0]
    distinct = np.unique(w.round(6), axis=0)
    assert len(distinct) == 8
    for corner in np.eye(3):
        assert any(np.allclose(row, corner) for row in distinct)


def test_warmup_is_uniform():
    w = sample('dense', M=3, warmup_frac=0.5, it=0)
    np.testing.assert_allclose(w, 1 / 3)
    assert not np.allclose(sample('dense', M=3, warmup_frac=0.5, it=NUM_EVALS), 1 / 3)


def test_unknown_sampling_raises():
    with pytest.raises(Exception, match='not implemented'):
        sample('nope')


def test_sparse_tiles_k_tradeoffs():
    w = sample('sparse', M=3, K=8)[0]
    assert w.shape == (NUM_ENVS, 3)
    np.testing.assert_allclose(w.sum(-1), 1.0, atol=1e-5)
    assert len(np.unique(w.round(6), axis=0)) == 8


def test_single_avg_is_uniform():
    np.testing.assert_allclose(sample('single-avg', M=3), np.full((1, NUM_ENVS, 3), 1 / 3))
