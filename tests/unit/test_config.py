import numpy as np
import pytest
import yaml

import moplayground as mop
from tests.helpers import TINY_BASE_CONFIG


def test_properties(cheetah_cfg):
    cfg = cheetah_cfg
    assert cfg.env_name == 'MOCheetah'
    assert cfg.algorithm == 'morlax'
    assert cfg.objectives == [['run'], ['energy'], ['height']]
    assert cfg.num_objectives == 3
    assert cfg.objective_labels == ['Running Speed', 'Energy Efficiency', 'Height']
    np.testing.assert_array_equal(cfg.default_scalarization, [1, 1, 1])
    assert set(cfg.sampling_params) == {'alpha', 'k', 'sampling', 'warmup_frac'}
    assert cfg.morlax_network_params['hypertype'] in ('single', 'dual')
    assert 'policy_hidden_layer_sizes' in cfg.amor_network_params
    assert 'policy_hidden_layer_sizes' in cfg.ppo_network_params
    assert cfg.gaitlib_path is None
    assert cfg.run_dir == cfg.save_dir / cfg.name


def _write(tmp_path, fname, **overrides):
    with open(TINY_BASE_CONFIG) as f:
        raw = yaml.safe_load(f)
    raw.update(overrides)
    path = tmp_path / f'{fname}.yaml'
    with open(path, 'w') as f:
        yaml.safe_dump(raw, f)
    return mop.MOConfig(path)


def test_labels_optional(tmp_path):
    with open(TINY_BASE_CONFIG) as f:
        raw = yaml.safe_load(f)
    del raw['env_config']['reward']['optimization']['labels']
    path = tmp_path / 'nolabels.yaml'
    path.write_text(yaml.safe_dump(raw))
    assert mop.MOConfig(path).objective_labels is None


def test_check_same_env(tmp_path, cheetah_cfg):
    mop.check_same_env([])
    mop.check_same_env([cheetah_cfg])
    mop.check_same_env([cheetah_cfg.cfg, _write(tmp_path, 'same', name='other').cfg])
    with pytest.raises(ValueError, match='env='):
        mop.check_same_env([cheetah_cfg.cfg, _write(tmp_path, 'env', env='MOHopper').cfg])
    with open(TINY_BASE_CONFIG) as f:
        raw = yaml.safe_load(f)
    raw['env_config']['ctrl_dt'] = 0.02
    (tmp_path / 'dt.yaml').write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='env_config differs'):
        mop.check_same_env([cheetah_cfg.cfg, mop.MOConfig(tmp_path / 'dt.yaml').cfg])


@pytest.mark.parametrize('name, expect_hash', [('test', False), ('real-run', True)])
def test_save(tmp_path, name, expect_hash):
    cfg = _write(tmp_path, name, name=name, save_dir=str(tmp_path / 'runs'))
    cfg.save(warn_github_changes=False)
    saved = mop.MOConfig(cfg.run_dir / 'config.yaml')
    assert ('git_hash' in saved.cfg) == expect_hash
    assert saved.cfg.env_config.to_dict() == cfg.cfg.env_config.to_dict()
