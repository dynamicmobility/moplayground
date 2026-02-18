import os
import argparse
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np
from moplayground.eval.rollout import rollout_policy
from moplayground.envs.create import create_environment
from minimal_mjx.learning.startup import read_config
from minimal_mjx.utils.plotting import save_video
from pathlib import Path
from ral import FINAL_YAMLS

parser = argparse.ArgumentParser()
parser.add_argument("env", type=str, help="Env to train on")
args = parser.parse_args()

config = read_config(FINAL_YAMLS[args.env])
KWARGS = {}

match args.env.lower():
    case 'cheetah':
        camera = 'side_fixed'
        WIDTH  = 2560
        HEIGHT = 1440
        directives = {
            'run':    np.array([1.0, 0.0]),
            'energy': np.array([0.0, 1.0])
        }
        T = 2.0
    case 'hopper':
        camera = 'side_fixed'
        WIDTH  = 2560
        HEIGHT = 1080
        directives = {
            'run':    np.array([1.0, 0.0]),
            'height': np.array([0.2, 1.0])
        }
        T = 4.0
    case 'ant':
        camera = 'side_fixed'
        WIDTH  = 1920
        HEIGHT = 1080
        directives = {
            'vx': np.array([1.0, 0.0]),
            'vy': np.array([0.0, 1.0])
        }
        T = 4.0
    case 'walker':
        camera = 'side_fixed'
        WIDTH  = 1920
        HEIGHT = 1440
        directives = {
            'run':    np.array([1.0, 0.0]),
            'energy': np.array([0.0, 1.0])
        }
        T = 2.0
    case 'humanoid':
        camera = 'side_fixed'
        WIDTH  = 1920
        HEIGHT = 1080
        directives = {
            'run':    np.array([1.0, 0.1]),
            'energy': np.array([0.0, 1.0])
        }
        T = 4.0
env, env_params   = create_environment(
    config,
    **KWARGS
)    
for key in directives:
    frames, _, _, _ = rollout_policy(
        env         = env,
        config      = config,
        directive   = directives[key],
        T           = T,
        camera      = camera,
        width       = WIDTH,
        height      = HEIGHT
    )
    save_video(
        frames = frames,
        dt = env.dt,
        path = Path(f'ral/videos/{args.env.lower()}-{key}.mp4')
    )

