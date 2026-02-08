import numpy as np
import jax
from pathlib import Path
from minimal_mjx.learning.startup import read_config
from minimal_mjx.learning.inference import rollout, load_policy
from minimal_mjx.utils.plotting import save_video, save_metrics
from moplayground.envs.create import create_environment
from moplayground.learning.inference import load_mo_policy

def main(env, save_dir, directive = None, T=10.0):
    config            = read_config()
    if directive is None:
        directive = np.ones(len(config.env_config.reward.optimization.objectives))
    if config['mo2so']['enabled']:
        print('Loading single objective policy')
        inference_fn = load_policy(config, deterministic=True)
    else:
        print('Loading multi-objective policy')
        inference_fn      = load_mo_policy(
            config          = config,
            directive       = directive,
            deterministic   = True
        )
    inference_fn = jax.jit(inference_fn)
    
    frames, reward_plotter, data_plotter, info_plotter = rollout(
        inference_fn    = inference_fn,
        env             = env,
        T               = T,
        height          = 640,
        width           = 480
    )

    save_video(
        frames    = frames,
        dt        = env.dt,
        path      = save_dir / f'{config['env']}-rollout.mp4'
    )
    save_metrics(
        plotter   = reward_plotter,
        path      = save_dir / f'{config['env']}-reward.pdf'
    )

if __name__ == '__main__':
    save_dir = Path('output/videos')
    main(save_dir)