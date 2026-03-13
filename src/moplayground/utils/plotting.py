import matplotlib.pyplot as plt
import numpy as np
from minimal_mjx.utils.plotting import get_subplot_grid
from moplayground.utils.pareto import get_pareto_statistics


def plot_pareto(
        ax, 
        pareto: np.ndarray, 
        directive: np.ndarray = None, 
        objectives: list[str] = None,
        nondominated=None,
        alpha=1.0
    ):
    if directive is None: directive = np.zeros_like(pareto)
    if objectives is None: objectives = [''] * pareto.shape[1]
    
    if pareto.shape[1] == 2:
        c = np.zeros((pareto.shape[0], 3))
        c[:, 0] = directive[:, 0]
        c[:, 2] = directive[:, 1]
        alpha = 0.05 if nondominated else 1
        ax.scatter(
            pareto[:, 0],
            pareto[:, 1],
            s       = 8,
            c       = c,
            alpha   = alpha
        )
        if nondominated is not None:
            nd_idx = nondominated
            ax.scatter(
                pareto[nd_idx, 0],
                pareto[nd_idx, 1],
                alpha=1.0,
                zorder=1,
                s = 12,
                edgecolors='black', # Border color
                linewidths=1.5,   # Border width,
                c = c[nd_idx,:],
            )
            ax.set_xlim((0.65 * np.min(pareto[nd_idx, 0]), 1.05 * np.max(pareto[nd_idx, 0])))
            ax.set_ylim((0.65 * np.min(pareto[nd_idx, 1]), 1.05 * np.max(pareto[nd_idx, 1])))
        ax.set_xlabel(objectives[0])
        ax.set_ylabel(objectives[1])
        return ax
    elif pareto.shape[1] == 3:
        c = np.zeros((pareto.shape[0], 3))
        # Convert directive weights to HSV-like color scheme
        # Use directive weights to create more distinct colors
        # c[:, 0] = 0.2 + 0.6 * directive[:, 0]  # Red channel: warm tones
        # c[:, 1] = 1.0 - 0.7 * directive[:, 1]  # Green channel: cool tones  
        # c[:, 2] = 0.3 + 0.7 * directive[:, 2]  # Blue channel: mid tones
        c[:, 0] = (1 - directive[:, 0]) * 0.9  # Red channel: warm tones
        c[:, 1] = (1 - directive[:, 1]) * 0.9  # Green channel: cool tones  
        c[:, 2] = (1 - directive[:, 2]) * 0.9  # Blue channel: mid tones
        ax.scatter(
            pareto[:, 0],
            pareto[:, 1],
            pareto[:, 2],
            s=24,
            c=c,
            alpha=0.01,
            axlim_clip=True
        )
        if nondominated is not None:
            nd_idx = nondominated
            ax.scatter(
                pareto[nd_idx, 0],
                pareto[nd_idx, 1],
                pareto[nd_idx, 2],
                s = 12,
                alpha=0.99,
                zorder=1,
                # edgecolors='black', # Border color
                # linewidths=0.5,   # Border width,
                c = c[nd_idx,:],
                axlim_clip=True
            )
            ax.set_xlim((0.95 * np.min(pareto[nd_idx, 0]), 1.05 * np.max(pareto[nd_idx, 0])))
            ax.set_ylim((0.95 * np.min(pareto[nd_idx, 1]), 1.05 * np.max(pareto[nd_idx, 1])))
            ax.set_zlim((1.00 * np.min(pareto[nd_idx, 2]), 1.05 * np.max(pareto[nd_idx, 2])))
        ax.set_xlabel(objectives[0])
        ax.set_ylabel(objectives[1])
        ax.set_zlabel(objectives[2])
        return ax
    else:
        raise NotImplementedError('Only 2D and 3D paretos are supported for plotting')

def plot_sequential_paretos(
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
        ax = plot_pareto(ax, y, d, objectives)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(x)
        
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