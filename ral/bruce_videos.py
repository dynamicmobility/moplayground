import os
import argparse
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np
import moplayground as mp
from minimal_mjx.utils.plotting import save_video
from pathlib import Path
from ral import FINAL_YAMLS, BRUCE_TRADEOFFS

# 'all_imitate'         : np.array([1.0, 1.0, 0.0, 0.0, 0.0]),
# 'smooth'              : np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
# 'swing_arms'          : np.array([1.0, 1.0, 1.0, 0.0, 0.0]),
# 'swing_arms_smooth'   : np.array([0.0, 0.0, 0.8, 0.0, 1.0]),

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, help="Env to train on", default="bruce5D")
parser.add_argument("--tradeoff", type=str, help="Trade-off to rollout", default="balanced")
args = parser.parse_args()

config = mp.learning.startup.read_config(FINAL_YAMLS[args.env])
KWARGS = {'idealistic': True}

match args.tradeoff.lower():
    case 'rigid_arms':
        camera = 'side_fixed'
        WIDTH  = 3840
        HEIGHT = 1080
        manual_speed = [0.12, 0.0, 0.0]
        T = 30.0
    case 'swing_arms':
        camera = 'side_fixed'
        WIDTH  = 3840
        HEIGHT = 1080
        manual_speed = [0.12, 0.0, 0.0]
        T = 30.0
    case 'smooth':
        camera = 'up_close'
        WIDTH  = 2560
        HEIGHT = 2560
        manual_speed = [0.12, 0.0, 0.0]
        T = 10.0
    case _:
        raise Exception('Unknown trade-off', args.tradeoff)

KWARGS['manual_speed'] = manual_speed
env, env_params = mp.envs.create_environment(config, **KWARGS)

frames, _, _, _ = mp.eval.rollout_policy(
    env       = env,
    config    = config,
    directive = np.array(BRUCE_TRADEOFFS[args.tradeoff.lower()]),
    T         = T,
    camera    = camera,
    width     = WIDTH,
    height    = HEIGHT
)

output_dir = Path('ral/videos')

video_path = output_dir / f"bruce_{args.tradeoff}.mp4"
save_video(frames, path=str(video_path), dt=env.dt)