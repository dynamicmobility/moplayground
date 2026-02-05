import os
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np
from moplayground.eval.rollout import main
from minimal_mjx.learning.startup import read_config
from pathlib import Path

# main(Path('output/videos'), directive = np.array([0.0, 0.5]))
main(Path('output/videos'), directive = np.array([0.5, 0.15]))
# main(Path('output/videos'), directive = np.array([0.0, 0.5, 0.0]))