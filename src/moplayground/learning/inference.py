import numpy as np
import jax
from etils import epath
import functools
from brax.training.checkpoint import get_network
from brax.training.agents.ppo import checkpoint
from minimal_mjx.learning.inference import *
from moplayground.moppo.factory import make_moppo_networks, make_mo_inference_fn
import moplayground as mop

def load_mo_policy(
    config,
    directive: np.ndarray,
    network_factory = make_moppo_networks,
    deterministic: bool = True,
):
    make_inference_fn, params = load_hypernetwork(
        config,
        network_factory,
    )
    return make_inference_fn(
        params        = params,
        deterministic = deterministic,
        directive     = directive,
    )

def load_moppo_network(
    config,
    network_factory = make_moppo_networks,
    path = None
) -> tuple[mop.moppo.factory.MOPPONetworks, dict]:
    """Loads the MOPPO object"""
    if path is None:
        path = get_last_model(config)
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
    moppo_network = get_network(params_config, network_factory)
    return moppo_network, hyperparams
    
def load_hypernetwork(
    config,
    network_factory = make_moppo_networks,
    path = None
):
    """Loads policy inference function from PPO checkpoint."""
    moppo_networks, hyperparams = load_moppo_network(config, network_factory, path)
    make_inference_fn = make_mo_inference_fn(moppo_networks)
    return make_inference_fn, hyperparams