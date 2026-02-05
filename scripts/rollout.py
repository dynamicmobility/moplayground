import os
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np
from moplayground.eval.rollout import main
from moplayground.envs.create import create_environment
from minimal_mjx.learning.startup import read_config
from pathlib import Path

config            = read_config()
env, env_params   = create_environment(
    config,
    manual_speed    = True,
    idealistic      = True
)
main(env, Path('output/videos'), directive = np.array([0.5, 0.0]))


# main(Path('output/videos'), directive = np.array([0.0, 0.5]))
# main(Path('output/videos'), directive = np.array([0.0, 0.5, 0.0]))