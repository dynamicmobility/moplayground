import numpy as np
import jax
import minimal_mjx as mm
import moplayground as mop

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
        inference_fn = mm.learning.inference.load_policy(config, deterministic=True)
    else:
        print('Loading multi-objective policy')
        inference_fn        = mop.learning.inference.load_mo_policy(
            config          = config,
            directive       = directive,
            deterministic   = True
        )
    inference_fn = jax.jit(inference_fn)
    
    return mm.eval.rollout_policy(
        inference_fn    = inference_fn,
        env             = env,
        T               = T,
        height          = height,
        width           = width,
        camera          = camera
    )