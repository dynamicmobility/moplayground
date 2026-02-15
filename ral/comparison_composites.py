from dynamo_figures import CompositeImage, CompositeMode
import cv2
import argparse
from pathlib import Path
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("env", type=str, help="Env to train on")
args = parser.parse_args()
DEFAULT_KWARGS = {
    'mode'          : CompositeMode.MIN_VALUE,
    'start_t'       : 0.0,
    'end_t'         : 1.0,
    'skip_frame'    : 12,
    'alpha'         : 0.4
}
composite_kwargs = []

match args.env.lower():
    case 'cheetah':
        energy_kwargs = deepcopy(DEFAULT_KWARGS)
        energy_kwargs['end_t'] = 0.7
        energy_kwargs['skip_frame'] = 10
        
        run_kwargs = deepcopy(DEFAULT_KWARGS)
        run_kwargs['end_t'] = 0.9
        run_kwargs['skip_frame'] = 9
        
        composite_kwargs = {
            'energy'    : energy_kwargs,
            'run'       : run_kwargs
        }
    case 'hopper':
        camera = 'side_fixed'
    case 'ant':
        camera = None
    case 'walker':
        camera = 'side_fixed'
    case 'humanoid':
        camera = None


for key in composite_kwargs:
    merger = CompositeImage(
        video_path=f'ral/videos/{args.env.lower()}-{key}.mp4',
        **(composite_kwargs[key])
    )

    result = merger.merge_images()
    cv2.imwrite(f'images/{args.env.lower()}-{key}.jpg', result)