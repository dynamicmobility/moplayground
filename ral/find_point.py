import minimal_mjx as mm
import numpy as np
from ral import FINAL_YAMLS
from pathlib import Path
import pandas as pd

config    = mm.utils.read_config(FINAL_YAMLS['bruce6D+DR'])
save_path = Path(config['save_dir']) / config['name']
if (save_path / 'the-obj.txt').exists():
    # Load existing results instead of running experiments
    paretos = np.array([pd.read_csv(save_path / f'the-obj.txt').iloc[:, 1:].values])
    directives = np.array(pd.read_csv(save_path / f'the-trade-off.txt').iloc[:, 1:].values)
else:
    raise Exception('Run 2D first')
print(config.env_config.reward.optimization.labels)
idx = np.argmax(paretos[0, :, 1])
print(directives[idx])