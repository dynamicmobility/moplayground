import os
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np
from moplayground.eval.rollout import rollout_policy
from moplayground.envs.create import create_environment
from minimal_mjx.learning.startup import read_config
from minimal_mjx.utils.plotting import save_metrics, save_video
from pathlib import Path

config            = read_config()
env, env_params   = create_environment(
    config,
    manual_speed    = [0.0, 0.0, 0.0],
    idealistic      = True
)

camera = 'side_fixed'
# camera = 'track'

# directive = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
# directive = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
# directive = np.array([1.0, 0.5, 0.2])
directive = np.array([1.0, 0.1])
# directive = np.array([0.0, 1.0])


frames, reward_plotter, _, _ = rollout_policy(
    env         = env,
    config      = config,
    directive   = directive,
    T           = 10.0,
    camera      = camera,
    # width       = 2560,
    # height      = 1080
)

save_video(
    frames,
    env.dt,
    Path(f'output/videos/{config['env']}-rollout.mp4')
)

save_metrics(
    reward_plotter,
    Path(f'output/videos/{config['env']}-reward.pdf')
)