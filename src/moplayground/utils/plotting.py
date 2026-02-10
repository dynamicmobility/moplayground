import matplotlib.pyplot as plt
import numpy as np
from minimal_mjx.utils.plotting import get_subplot_grid
from moplayground.eval.pareto import get_pareto_statistics

def plot_squential_paretos(
    ax_titles: list[str],
    paretos: np.ndarray,
    directives: np.ndarray = None,
    objectives: list[str] = None
):
    if len(ax_titles) != len(paretos) != len(directives):
        raise Exception('Incompatible lengths of input arrays')

    if directives is None: directives = np.zeros_like(paretos)
    if objectives is None: objectives = [''] * paretos[0].shape[1]
    
    nrows, ncols = get_subplot_grid(len(ax_titles))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    if type(axs) == np.ndarray: axs = axs.flatten()
    else: axs = [axs]
    
    paretos = np.array(paretos)
    directives = np.array(directives)
    xlim   = np.array((np.min(paretos[..., 0]), np.max(paretos[..., 0])))
    ylim   = np.array((np.min(paretos[..., 1]), np.max(paretos[..., 1])))
    border = np.array([-1., 1.])
    xlim   = xlim + border * np.abs(xlim[1] - xlim[0]) * 0.1
    ylim   = ylim + border * np.abs(ylim[1] - ylim[0]) * 0.1
    for ax, x, y, d in zip(axs, ax_titles, paretos, directives):
        c = np.zeros((paretos.shape[1], 3))
        c[:, 0] = d[:, 0]
        c[:, 2] = d[:, 1]
        ax.scatter(
            y[:, 0],
            y[:, 1],
            s=4,
            c=c
        )
        ax.set_title(x)
        ax.set_xlabel(objectives[0])
        ax.set_ylabel(objectives[1])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
    fig.set_size_inches((4 * ncols, 4 * nrows))
    fig.tight_layout()

    return fig, ax

def plot_sequential_hypervolume(
    iterations: list[int] | np.ndarray,
    paretos: np.ndarray
):
    fig, ax = plt.subplots()
    hvs = []
    sps = []
    for p in paretos:
        hv, sp = get_pareto_statistics(np.array(p.block_until_ready()))
        hvs.append(hv)
        sps.append(sp)
    print(sps)
    ax2 = ax.twinx()
    ax.plot(iterations, hvs, label='Hypervolume')
    ax2.plot(iterations, sps, 'r-', label='Sparsity')
    fig.set_size_inches((10, 7))
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Hypervolume')
    ax2.set_ylabel('Sparsity')
    ax.legend()
    ax2.legend()
    return fig, ax