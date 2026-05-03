import numpy as np
import jax
from etils import epath
import functools
from brax.training.checkpoint import get_network
from brax.training.agents.ppo import checkpoint
# from minimal_mjx.learning.inference import *
import minimal_mjx as mm
from moplayground.moppo.factory import make_morlax_networks, make_hypernetwork_inference_fn
import moplayground as mop

def load_mo_policy(
    config,
    tradeoff: np.ndarray,
    network_factory = make_morlax_networks,
    deterministic: bool = True,
):
    hypernetwork_inference_fn, params = load_hypernetwork_inference_fn(
        config,
        network_factory,
    )
    return hypernetwork_inference_fn(
        params        = params,
        deterministic = deterministic,
        directive     = tradeoff,
        single_policy = True
    )

def load_hypernetworks(
    config,
    network_factory = make_morlax_networks,
    path = None
) -> tuple[mop.moppo.factory.MORLAXNetworks, dict]:
    """Loads the MOPPO object"""
    if path is None:
        path = mm.learning.inference.get_last_model(config)
    print(f'Loading model at {path.as_posix()}')
    fullpath = path.resolve()
    
    fullpath = epath.Path(fullpath)
    params_config = checkpoint.load_config(fullpath)
    hyperparams = checkpoint.load(fullpath)
    hyperconfig = config['learning_params']['hypernetwork_params']
    network_factory = functools.partial(
        network_factory, 
        key            = jax.random.PRNGKey(0),
        num_objectives = len(config['env_config']['reward']['optimization']['objectives']),
        **hyperconfig
    )
    hypernetworks = get_network(params_config, network_factory)
    return hypernetworks, hyperparams
    
def load_hypernetwork_inference_fn(
    config,
    network_factory = make_morlax_networks,
    path = None
):
    """Loads policy inference function from PPO checkpoint."""
    hypernetworks, hyperparams = load_hypernetworks(config, network_factory, path)
    make_inference_fn = make_hypernetwork_inference_fn(hypernetworks)
    return make_inference_fn, hyperparams

def rollout_policy(
    env, 
    config,
    tradeoff = None,
    T=10.0,
    camera = 'track',
    width  = 1080,
    height = 720
):
    if tradeoff is None:
        tradeoff = np.ones(len(config.env_config.reward.optimization.objectives))
    if config['mo2so']['enabled']:
        print('Loading single objective policy')
        inference_fn = mm.learning.inference.load_policy(config, deterministic=True)
    else:
        print('Loading multi-objective policy')
        inference_fn = mop.learning.inference.load_mo_policy(
            config          = config,
            tradeoff        = tradeoff,
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


def get_num_objectives(config):
    return len(config['env_config']['reward']['optimization']['objectives'])