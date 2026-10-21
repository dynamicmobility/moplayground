import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import moplayground as mop


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def test_default_coloring_is_rgb():
    colors = np.asarray(mop.default_coloring(np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])))
    assert colors.shape[0] == 3
    assert np.all((colors >= 0) & (colors <= 1))


@pytest.mark.parametrize('M', [2, 3])
def test_plot_pareto(M):
    rng = np.random.default_rng(0)
    directives = rng.dirichlet(np.ones(M), 20)
    kwargs = {'subplot_kw': {'projection': '3d'}} if M == 3 else {}
    _, ax = plt.subplots(**kwargs)
    mop.plot_pareto(ax=ax, pareto=rng.normal(size=(20, M)), colors=mop.default_coloring(directives),
                    objective=[f'o{i}' for i in range(M)])


@pytest.mark.parametrize('M', [2, 3])
def test_mo_progress(tmp_path, M):
    progress = mop.MOProgress(tmp_path, total_timesteps=100, labels=[f'o{i}' for i in range(M)])
    rng = np.random.default_rng(0)
    for step in (10, 20):
        progress.plot_progress(step, {'reward': rng.normal(size=(8, M)), 'directive': rng.dirichlet(np.ones(M), 8)})
    df = pd.read_csv(tmp_path / 'progress.csv')
    assert list(df['iters']) == [0, 10, 20]
    assert np.all(np.isfinite(df['hypervolume'][1:]))
    assert (tmp_path / 'progress.svg').exists() == (M == 2)
    with pytest.raises(NotImplementedError):
        progress.load()
