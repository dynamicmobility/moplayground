import numpy as np
import jax
from etils import epath
import functools
from brax.training.checkpoint import get_network
from brax.training.agents.ppo import checkpoint
from minimal_mjx.learning.inference import get_last_model
from moplayground.moppo.factory import make_moppo_networks, make_mo_inference_fn

def load_mo_policy(
    config,
    directive: np.ndarray,
    network_factory = make_moppo_networks,
    deterministic: bool = True,
):
    """Loads policy inference function from PPO checkpoint."""
    path = get_last_model(config)
    print(f'Loading model at {path.as_posix()}')
    fullpath = path.resolve()
    
    fullpath = epath.Path(fullpath)
    params_config = checkpoint.load_config(fullpath)
    params = checkpoint.load(fullpath)
    hyperconfig = config['learning_params']['hypernetwork_params']
    network_factory = functools.partial(
        network_factory, 
        key            = jax.random.PRNGKey(0),
        num_objectives = directive.shape[0],
        **hyperconfig
    )
    moppo_network = get_network(params_config, network_factory)
    make_inference_fn = make_mo_inference_fn(moppo_network)

    return make_inference_fn(
        params        = params,
        deterministic = deterministic,
        directive     = directive,
        single_policy = True
    )