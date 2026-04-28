import os
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np
import moplayground as mop
import minimal_mjx as mm
from pathlib import Path
from ral import BRUCE_TRADEOFFS, FINAL_YAMLS

config = mm.utils.read_config(FINAL_YAMLS['bruce6D+DR'])
env, env_params = mop.envs.create_environment(
    config,
    manual_speed    = [0.12, 0.0, 0.0],
    idealistic      = True,
)

camera = 'track'
directive = np.array([0.0, 1.0])
directive = np.array(BRUCE_TRADEOFFS['swing_arms'])

frames, reward_plotter, _, _ = mop.eval.rollout_policy(
    env         = env,
    config      = config,
    directive   = directive,
    T           = 3.0,
    camera      = camera,
    width       = 640,
    height      = 480
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