import os
os.environ["MUJOCO_GL"] = "egl" # (comment out if not on Ubuntu SSH)
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np
import moplayground as mop
import minimal_mjx as mm
from pathlib import Path

# Read the config file and create the environment
config = mm.utils.read_config()
kwargs = {} if config['env'] != 'NaviGait' else {
    'manual_speed'    : [0.12, 0.0, 0.0],
    'track_yaw'       : False,
    'idealistic'      : True
}
env, env_params = mop.envs.create_environment(
    config,
    **kwargs
)

# Choose a tradeoff
camera    = 'track'
n_objs    = mop.learning.inference.get_num_objectives(config)
tradeoff  = np.random.dirichlet(alpha=np.ones(n_objs))
print(f'Chosen tradeoff {tradeoff} with {n_objs} objectives')

# Rollout the policy
frames, reward_plotter, _, _ = mop.learning.inference.rollout_policy(
    env         = env,
    config      = config,
    tradeoff    = tradeoff,
    T           = 6.0,
    camera      = camera,
    width       = 2560,
    height      = 1440
)

# Save video and metrics
mm.utils.plotting.save_video(
    frames,
    env.dt,
    Path(f'output/videos/{config['env']}-rollout.mp4')
)
mm.utils.plotting.save_metrics(
    reward_plotter,
    Path(f'output/videos/{config['env']}-reward.pdf')
)