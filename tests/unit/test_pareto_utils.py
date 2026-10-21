import numpy as np
import pytest

import moplayground as mop


@pytest.mark.parametrize('v', [[0.2, 0.3, 0.5], [3.0, -1.0, 0.5], [-2.0, -2.0], [10.0]])
def test_project_to_simplex(v):
    p = mop.project_to_simplex(v)
    assert p.shape == (len(v),)
    assert np.all(p >= 0)
    assert np.isclose(p.sum(), 1.0)


def test_project_to_simplex_is_identity_on_simplex():
    v = np.array([0.1, 0.6, 0.3])
    np.testing.assert_allclose(mop.project_to_simplex(v), v)


def test_closest_points_and_corners():
    points = np.array([[0.9, 0.1], [0.5, 0.5], [0.05, 0.95]])
    np.testing.assert_array_equal(mop.closest_points(points, [[1, 0], [0, 1]]), [0, 2])
    np.testing.assert_array_equal(mop.closest_points(points, [0.5, 0.4]), [1])
    np.testing.assert_array_equal(mop.corner_tradeoffs(points), [0, 2])


def test_get_nondominated_maximizes():
    F = np.array([[1.0, 1.0], [2.0, 0.5], [0.5, 2.0], [0.4, 0.4]])
    assert set(mop.get_nondominated(F)) == {0, 1, 2}


def test_hypervolume_known_values():
    # minimization space, origin reference: a single point at (-1, -2) dominates a 1x2 box
    assert np.isclose(mop.hypervolume_from_nondominated(np.array([[-1.0, -2.0]])), 2.0)
    F_min = np.array([[-2.0, -1.0], [-1.0, -2.0]])
    assert np.isclose(mop.hypervolume_from_nondominated(F_min), 3.0)
    assert np.isclose(mop.hypervolume_from_nondominated(F_min, ref_point=[-0.5, -0.5]), 1.25)


def test_get_pareto_statistics():
    F = np.array([[2.0, 1.0], [1.0, 2.0], [0.5, 0.5]])  # last point is dominated
    hv, sparsity = mop.get_pareto_statistics(F)
    assert np.isclose(hv, 3.0)
    assert np.isfinite(sparsity)
    hv_ref, _ = mop.get_pareto_statistics(F, ref_point=[0.5, 0.5])
    assert np.isclose(hv_ref, 1.25)


def test_get_pareto_statistics_single_point():
    hv, sparsity = mop.get_pareto_statistics(np.array([[1.0, 2.0]]))
    assert np.isclose(hv, 2.0)
    assert np.isfinite(sparsity)
