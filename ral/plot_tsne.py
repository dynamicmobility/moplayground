import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import minimal_mjx as mm
import pandas as pd
from ral import FINAL_YAMLS

# Load t-SNE results
log_path = Path('output/logs/paretos_spectral.npy')
paretos_2d = np.load(log_path)

config    = mm.utils.read_config(FINAL_YAMLS['bruce6D+DR'])
save_path = Path(config['save_dir']) / config['name']
directives = np.array(pd.read_csv(save_path / f'the-trade-off.txt').iloc[:, 1:].values)

plt.figure(figsize=(8, 6))
plt.scatter(paretos_2d[:, 0], paretos_2d[:, 1], c='blue', alpha=0.7, s=2)
plt.title('t-SNE Dimensionality Reduction of Pareto Vectors')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)

# Save plot to file
plot_path = Path('output/plots/paretos_tsne.png')
plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_path)
