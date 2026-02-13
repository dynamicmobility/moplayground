import os
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np
from moplayground.eval.rollout import rollout_policy
from moplayground.envs.create import create_environment
from minimal_mjx.learning.startup import read_config
from pathlib import Path

config            = read_config()
env, env_params   = create_environment(
    config,
    # manual_speed    = [0.2, 0.0, 0.0],
    # idealistic      = True
)

# camera = 'side_fixed'
camera = 'track'

directive = np.array([1.0, 1.0, 0.0, 1.0])
# directive = np.array([1.0, 1.0, 0.0])
# directive = np.array([1.0, 0.0])
# directive = np.array([0.2, 1.0])


rollout_policy(
    env         = env,
    save_dir    = Path('output/videos'),
    directive   = directive,
    T           = 5.0,
    camera      = camera,
    # width       = 2560,
    # height      = 1080
)