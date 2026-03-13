import os
import argparse
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np
import moplayground as mop
from pathlib import Path
from ral import FINAL_YAMLS, BRUCE_TRADEOFFS

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, help="Env to train on", default="bruce5D")
parser.add_argument("--tradeoff", type=str, help="Trade-off to rollout", default="balanced")
args = parser.parse_args()

config = mop.utils.read_config(FINAL_YAMLS[args.env])
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
        WIDTH = 3840 # ral submission
        # WIDTH = 1440 # personal website
        HEIGHT = 1080
        manual_speed = [0.12, 0.0, 0.0]
        T = 30.0
    case 'swing_arms_banner':
        camera = 'side_fixed_forward'
        WIDTH = 3840 # ral submission
        HEIGHT = 1080
        manual_speed = [0.12, 0.0, 0.0]
        T = 20.5 #29.5
    case 'swing_arms_favicon':
        camera = 'track'
        WIDTH = 480
        HEIGHT = 480
        manual_speed = [0.12, 0.0, 0.0]
        T = 10.0
    case 'swing_arms_metatag':
        camera = 'side_fixed'
        WIDTH = 1200
        HEIGHT = 630
        manual_speed = [0.12, 0.0, 0.0]
        T = 15.0
    case 'smooth':
        camera = 'up_close'
        WIDTH  = 2560
        HEIGHT = 2560
        manual_speed = [0.12, 0.0, 0.0]
        T = 10.0
    case _:
        raise Exception('Unknown trade-off', args.tradeoff)

KWARGS['manual_speed'] = manual_speed
env, env_params = mop.envs.create_environment(config, **KWARGS)

frames, _, _, _ = mop.eval.rollout_policy(
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
mop.utils.save_video(frames, path=str(video_path), dt=env.dt)