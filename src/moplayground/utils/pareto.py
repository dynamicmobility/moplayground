from pymoo.indicators.hv import HV
from pymoo.util.normalization import normalize
from pymoo.indicators.spacing import SpacingIndicator
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np

def get_nondominated(F, epsilon=None):
    nds = NonDominatedSorting(epsilon=epsilon)
    front_indices = nds.do(-F, only_non_dominated_front=True)
    return front_indices

def hypervolume_from_nondominated(F_min):
    # Reference point must be worse in minimization space
    # i.e., larger than all points in F_min
    ref_point = np.zeros(F_min.shape[1])

    hv = HV(ref_point=ref_point)
    hypervolume = hv(F_min)
    return hypervolume

def sparsity_from_normalized_nondominated(F_min_norm):
    spacing = SpacingIndicator()
    sparsity = spacing(F_min_norm)
    return sparsity

def get_pareto_statistics(F):
    # Extract nondominated front to do calculations
    F_max = F[get_nondominated(F)]
    F_norm = normalize(F_max.copy())

    # Convert to minimization
    F_min = -F_max.copy()
    F_min_norm = -F_norm.copy()

    if F_min_norm.shape[0] == 1:
        # Sparsity always needs 2 points to calculate
        F_min_norm = np.repeat(F_min_norm, 2, axis=0)
    return (
        hypervolume_from_nondominated(F_min), 
        sparsity_from_normalized_nondominated(F_min_norm)
    )