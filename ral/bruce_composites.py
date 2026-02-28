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
    'alpha'         : 0.7
}
composite_kwargs = []

parser = argparse.ArgumentParser()
parser.add_argument("--tradeoff", type=str, help="Trade-off to rollout", default="balanced")
args = parser.parse_args()

KWARGS = {'idealistic': True}

match args.tradeoff.lower():
    case 'swing_arms':
        kwargs = deepcopy(DEFAULT_KWARGS)
        kwargs['end_t'] = 20.0
        kwargs['skip_frame'] = 130 #110
        vid = 'swing_arms'
    case 'swing_arms_hero':
        kwargs = deepcopy(DEFAULT_KWARGS)
        kwargs['end_t'] = 9.0
        kwargs['skip_frame'] = 90 #110
        vid = 'swing_arms'
    case 'rigid_arms_hero':
        kwargs = deepcopy(DEFAULT_KWARGS)
        kwargs['end_t'] = 10.0
        kwargs['skip_frame'] = 130 #110
        vid = 'rigid_arms'
    case 'rigid_arms':
        kwargs = deepcopy(DEFAULT_KWARGS)
        kwargs['end_t'] = 20.0
        kwargs['skip_frame'] = 130 #110
        vid = 'rigid_arms'
    case 'smooth':
        kwargs = deepcopy(DEFAULT_KWARGS)
        kwargs['mode'] = CompositeMode.MIN_VALUE,
        kwargs['alpha'] = 0.1
        kwargs['end_t'] = 10.0
        kwargs['skip_frame'] = 150
        vid = 'smooth'
    case _:
        raise Exception('Unknown trade-off', args.tradeoff)

output_dir = Path('ral/images')
key = args.tradeoff.lower()
merger = CompositeImage(
    video_path=f'ral/videos/bruce_{vid}.mp4',
    **kwargs
)
result = merger.merge_images()

path = output_dir / f'bruce-{key}.jpg'
cv2.imwrite(path, result)
print(output_dir / f'bruce-{key}.jpg')
