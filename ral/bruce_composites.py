import os
import argparse
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np
import moplayground as mp
from minimal_mjx.utils.plotting import save_video
from pathlib import Path
from ral import FINAL_YAMLS
from dynamo_figures import CompositeImage, CompositeMode
from copy import deepcopy
import cv2

DEFAULT_KWARGS = {
    'mode'          : CompositeMode.MIN_VALUE,
    'start_t'       : 0.0,
    'end_t'         : 1.0,
    'skip_frame'    : 12,
    'alpha'         : 0.4
}
composite_kwargs = []

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, help="Env to train on", default="bruce5D")
parser.add_argument("--tradeoff", type=str, help="Trade-off to rollout", default="balanced")
args = parser.parse_args()

config = mp.learning.startup.read_config(FINAL_YAMLS[args.env])
KWARGS = {'idealistic': True}

match args.tradeoff.lower():
    case 'imitation':
        kwargs = deepcopy(DEFAULT_KWARGS)
        kwargs['end_t'] = 12.0
        kwargs['skip_frame'] = 100
        
        composite_kwargs = {
            'imitation'    : kwargs,
        }
    case 'swing_arms':
        kwargs = deepcopy(DEFAULT_KWARGS)
        kwargs['end_t'] = 12.0
        kwargs['skip_frame'] = 100
        
        composite_kwargs = {
            'swing_arms'    : kwargs,
        }
    case 'smooth':
        kwargs = deepcopy(DEFAULT_KWARGS)
        kwargs['mode'] = CompositeMode.MIN_VALUE,
        kwargs['alpha'] = 0.1
        kwargs['end_t'] = 10.0
        kwargs['skip_frame'] = 150
        
        composite_kwargs = {
            'smooth'    : kwargs,
        }
    case _:
        raise Exception('Unknown trade-off', args.tradeoff)

output_dir = Path('ral/images')
for key in composite_kwargs:
    merger = CompositeImage(
        video_path=f'ral/videos/bruce_{key}.mp4',
        **(composite_kwargs[key])
    )
    result = merger.merge_images()
    
    cv2.imwrite(output_dir / f'{args.env.lower()}-{key}.jpg', result)
