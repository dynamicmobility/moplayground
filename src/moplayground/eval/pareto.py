from moplayground.envs.generic.mobase import Multi2SingleObjective, MultiObjectiveBase
from minimal_mjx.learning.training import plot_progress
import numpy as np

def mo2so(env: MultiObjectiveBase, weighting: np.ndarray):
    """Converts a multi-objective environment into a single objective one with
    pre-determined scalarization defined by weighting."""
    return Multi2SingleObjective(
        env       = env,
        weighting = weighting
    )
    
def plot_progress(
    run, num_steps, metrics, times, x_data, y_data, directives, save_dir, labels
):
    pass

