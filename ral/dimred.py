import minimal_mjx as mm
import moplayground as mop
import numpy as np
from ral import FINAL_YAMLS
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import SpectralEmbedding, TSNE

config    = mm.utils.read_config(FINAL_YAMLS['bruce6D+DR'])
save_path = Path(config['save_dir']) / config['name']
if (save_path / 'the-obj.txt').exists():
    # Load existing results instead of running experiments
    paretos = np.array(pd.read_csv(save_path / f'the-obj.txt').iloc[:, 1:].values)
    directives = np.array(pd.read_csv(save_path / f'the-trade-off.txt').iloc[:, 1:].values)
else:
    raise Exception('Run 2D first')
# Spectral Embedding dimensionality reduction to 2D
D = 3
less_paretos = paretos[::D]
less_directives = directives[::D]
paretos_2d = TSNE(n_components=2, random_state=95, perplexity=50).fit_transform(less_paretos)

# Save Spectral Embedding results to file
log_path = Path('output/logs/paretos_spectral.npy')
log_path.parent.mkdir(parents=True, exist_ok=True)
np.save(log_path, paretos_2d)

low_x = np.argmin(paretos_2d[:,0])
low_y = np.argmin(paretos_2d[:,1])
high_x = np.argmax(paretos_2d[:,0])
high_y = np.argmax(paretos_2d[:,1])

less_frontier = paretos_2d[mop.utils.pareto.get_nondominated(less_paretos)]

print(f"Lowest X: Index {repr(less_directives[low_x])}")
print(f"Lowest Y: Index {repr(less_directives[low_y])}")
print(f"Highest X: Index {repr(less_directives[high_x])}")
print(f"Highest Y: Index {repr(less_directives[high_y])}")

plt.figure(figsize=(8, 6))
plt.scatter(paretos_2d[:, 0], paretos_2d[:, 1], c='blue', alpha=0.2, s=1)
plt.scatter(less_frontier[:, 0], less_frontier[:, 1], c='red', alpha=1.0, s=10)
plt.title('t-SNE Dimensionality Reduction of Pareto Vectors')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)

# Save plot to file
plot_path = Path('output/plots/paretos_tsne.pdf')
plot_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(plot_path)
