import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np
import moplayground as mop
import minimal_mjx as mm
from pathlib import Path
from ral import BRUCE_TRADEOFFS


config = mm.utils.read_config()
env, env_params = mop.envs.create_environment(
    config,
    manual_speed    = [0.12, 0.0, 0.0],
    idealistic      = True,
    track_yaw       = False,
)

camera = 'track'
# directive = np.array([0.0, 1.0])
# directive = np.array([1.0, 0.0])
directive = np.array(BRUCE_TRADEOFFS['swing_arms'])
directive = np.array(BRUCE_TRADEOFFS['smooth'])
directive = np.array(BRUCE_TRADEOFFS['energy'])
directive = np.array(BRUCE_TRADEOFFS['rigid_arms'])
# directive = np.array(BRUCE_TRADEOFFS['base_tracking'])


frames, reward_plotter, _, _ = mop.eval.rollout_policy(
    env         = env,
    config      = config,
    directive   = directive,
    T           = 6.0,
    camera      = camera,
    width       = 2560,
    height      = 1440
)

mm.utils.plotting.save_video(
    frames,
    env.dt,
    Path(f'output/videos/{config['env']}-rollout.mp4')
)

mm.utils.plotting.save_metrics(
    reward_plotter,
    Path(f'output/videos/{config['env']}-reward.pdf')
)