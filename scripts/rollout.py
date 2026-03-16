import os
# os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np
import moplayground as mop
import minimal_mjx as mm
from pathlib import Path


config = mm.utils.read_config()
env, env_params = mop.envs.create_environment(
    config,
    manual_speed    = [0.12, 0.0, 0.0],
    idealistic      = True
)

camera = 'track'
directive = np.array([0.0, 1.0])
directive = np.array([0.2812006 , 0.37578177, 0.03217732, 0.23709323, 0.02666323, 0.04708387]) # Low X
# directive = np.array([7.2073470e-05, 4.2898576e-03, 4.3646526e-01, 6.0297470e-02, 5.0404664e-02, 4.4847056e-01]) # Low Y
# directive = np.array([0.3207971 , 0.18218784, 0.0065451 , 0.370252  , 0.08222353, 0.03799446]) # High X
# directive = np.array([1.1459584e-01, 8.2459840e-02, 1.5060634e-02, 4.0510620e-01, 2.2142822e-05, 3.8275546e-01]) # High Y


frames, reward_plotter, _, _ = mop.eval.rollout_policy(
    env         = env,
    config      = config,
    directive   = directive,
    T           = 10.0,
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