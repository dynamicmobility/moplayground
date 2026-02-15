import numpy as np
import jax
from pathlib import Path
from minimal_mjx.learning.startup import read_config
from minimal_mjx.learning.inference import rollout, load_policy
from minimal_mjx.utils.plotting import save_video, save_metrics
from moplayground.envs.create import create_environment
from moplayground.learning.inference import load_mo_policy

def rollout_policy(
    env, 
    config,
    directive = None,
    T=10.0,
    camera = 'track',
    width  = 1080,
    height = 720
):
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
    
    return rollout(
        inference_fn    = inference_fn,
        env             = env,
        T               = T,
        height          = height,
        width           = width,
        camera          = camera
    )