from pymoo.indicators.hv import HV
from pymoo.util.normalization import normalize
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from scipy.spatial import cKDTree
import numpy as np


def project_to_simplex(v):
    """Euclidean projection of ``v`` onto the probability simplex.
    """
    v = np.asarray(v, dtype=float).reshape(-1)
    sorted_v = np.sort(v)[::-1]
    cumulative = np.cumsum(sorted_v)
    support = np.arange(1, v.size + 1)
    # Largest support size whose sorted entry still clears the running threshold.
    rho = support[sorted_v - (cumulative - 1.0) / support > 0][-1]
    theta = (cumulative[rho - 1] - 1.0) / rho
    return np.maximum(v - theta, 0.0)


def closest_points(points, targets):
    """Index into ``points`` of the nearest (Euclidean) neighbor of each target.

    Args:
        points: ``(n_points, dim)`` array searched over.
        targets: ``(dim,)`` or ``(n_targets, dim)`` query point(s).

    Returns:
        ``(n_targets,)`` integer indices into ``points``.
    """
    return cKDTree(np.asarray(points)).query(np.atleast_2d(targets))[1]


def corner_tradeoffs(tradeoffs):
    """Indices of the tradeoffs nearest each one-hot corner of the simplex.
    """
    tradeoffs = np.asarray(tradeoffs)
    return closest_points(tradeoffs, np.eye(tradeoffs.shape[1]))

def get_nondominated(F, epsilon=None):
    nds = NonDominatedSorting(epsilon=epsilon)
    front_indices = nds.do(-F, only_non_dominated_front=True)
    return front_indices

def hypervolume_from_nondominated(F_min, ref_point=None):
    """Compute the hypervolume of a non-dominated front in minimization space.

    Args:
        F_min: ``(n_points, n_objectives)`` array of points in
            minimization space (i.e. negated objective values).
        ref_point: ``(n_objectives,)`` upper bound in that same minimization
            space; a point not strictly below it on every axis contributes
            nothing. Defaults to the origin.

    Returns:
        Hypervolume as a float, computed by ``pymoo.indicators.hv.HV``.
    """
    if ref_point is None:
        ref_point = np.zeros(F_min.shape[1])

    hv = HV(ref_point=np.asarray(ref_point, dtype=float))
    hypervolume = hv(F_min)
    return hypervolume

def sparsity_from_normalized_nondominated(F_min_norm):
    spacing = SpacingIndicator()
    sparsity = spacing(F_min_norm)
    return sparsity

def get_pareto_statistics(F, ref_point=None):
    """Compute hypervolume and sparsity for a set of objective vectors.

    Args:
        F: ``(n_points, n_objectives)`` array of objective vectors
            (higher is better).
        ref_point: ``(n_objectives,)`` reference in the same maximization space
            as ``F``; an objective at or below it contributes no hypervolume.
            Defaults to the origin.

    Returns:
        Tuple ``(hypervolume, sparsity)``.
    """
    F_max = F[get_nondominated(F)]
    F_norm = normalize(F_max.copy())

    # Convert to minimization
    F_min = -F_max.copy()
    F_min_norm = -F_norm.copy()

    if F_min_norm.shape[0] == 1:
        # Sparsity always needs 2 points to calculate
        F_min_norm = np.repeat(F_min_norm, 2, axis=0)
    # The reference is negated alongside the front to stay in minimization space.
    return (
        hypervolume_from_nondominated(
            F_min, None if ref_point is None else -np.asarray(ref_point, dtype=float)
        ),
        sparsity_from_normalized_nondominated(F_min_norm)
    )