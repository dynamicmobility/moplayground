import os
os.environ["MUJOCO_GL"] = "egl" # (comment out if not on Ubuntu SSH)
os.environ['JAX_PLATFORMS']='cpu'
import argparse
import jax
import numpy as np
import moplayground as mop
import minimal_mjx as mm
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=str, help="YAML config of the run to roll out")
args = parser.parse_args()

# Read the config file and create the environment
# cfg      = config.load(args.config_path)
cfg      = mop.config.MOConfig(args.config_path)
kwargs = {} if cfg.env_name != 'NaviGait' else {
    'manual_speed'    : [0.12, 0.0, 0.0],
    'track_yaw'       : False,
    'idealistic'      : True
}
env, env_params = mop.create_environment(
    env_name     = cfg.env_name,
    env_params   = cfg.env_params,
    backend      = cfg.backend,
    # gaitlib_path = cfg.gaitlib_path,
    **kwargs
)

# Choose a tradeoff
camera    = 'track'
n_objs    = cfg.num_objectives
tradeoff  = np.random.dirichlet(alpha=np.ones(n_objs))
print(f'Chosen tradeoff {tradeoff} with {n_objs} objectives')

# Build the policy manually
inference_fn = mop.load_mo_policy(
    algorithm       = cfg.algorithm,
    network_params  = cfg.network_params,
    num_objectives  = n_objs,
    run_dir         = cfg.run_dir,
    tradeoff        = tradeoff,
    deterministic   = True
)

inference_fn = jax.jit(inference_fn)

# Rollout the policy
frames, traj, reward_plotter, _, _ = mm.eval.rollout_policy(
    inference_fn = inference_fn,
    env          = env,
    T            = 6.0,
    camera       = camera,
    width        = 640,
    height       = 480
)

# Save video and metrics
mm.utils.plotting.save_video(
    frames,
    env.dt,
    Path(f'output/videos/{cfg.env_name}-rollout.mp4')
)
mm.utils.plotting.save_metrics(
    reward_plotter,
    Path(f'output/videos/{cfg.env_name}-reward.pdf')
)