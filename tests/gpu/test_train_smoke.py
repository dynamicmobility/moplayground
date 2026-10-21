import pandas as pd
import pytest

import moplayground as mop

pytestmark = pytest.mark.gpu


def test_training_writes_run(trained_run):
    cfg = trained_run.cfg
    run_dir = cfg.run_dir
    assert (run_dir / 'config.yaml').is_file()
    saved = mop.MOConfig(run_dir / 'config.yaml')
    assert saved.algorithm == trained_run.algorithm
    checkpoints = mop.get_all_models(run_dir)
    # one checkpoint per eval (num_evals=2): initial + final
    assert [int(p.name) for p in checkpoints] == [0, cfg.ppo_params['num_timesteps']]


def test_mo_progress_logged(trained_mo_run):
    df = pd.read_csv(trained_mo_run.cfg.run_dir / 'progress.csv')
    assert len(df) == trained_mo_run.cfg.ppo_params['num_evals'] + 1
    assert df['hypervolume'][1:].notna().all()
