import numpy as np
import jax
from etils import epath
import functools
from brax.training.checkpoint import get_network
from brax.training.agents.ppo import checkpoint
import minimal_mjx as mm
from moplayground.moppo.factory import (
    make_morlax_networks,
    make_hypernetwork_inference_fn,
    make_amor_networks,
    make_amor_inference_fn,
)
import moplayground as mop


def load_mo_policy(
    config,
    tradeoff: np.ndarray,
    network_factory = None,
    deterministic: bool = True,
):
    """Load a multi-objective policy for the configured algorithm.

    Dispatches on ``config.algorithm``. Returns a 2-arg callable
    ``policy(obs, key) -> (action, extras)`` with ``tradeoff`` baked in, so it
    is compatible with ``mm.eval.rollout_policy`` and other consumers.

    For AMOR specifically, the underlying inference function natively accepts
    a directive at call time (so the tradeoff can change per step). To get
    that 3-arg form, call :func:`load_amor_inference_fn` directly instead of
    this function.
    """
    algo = config['algorithm'] if isinstance(config, dict) else config.algorithm
    if algo == 'morlax':
        if network_factory is None:
            network_factory = make_morlax_networks
        hypernetwork_inference_fn, params = load_hypernetwork_inference_fn(
            config,
            network_factory,
        )
        return hypernetwork_inference_fn(
            params        = params,
            deterministic = deterministic,
            directive     = tradeoff,
            single_policy = True,
        )
    elif algo == 'amor':
        if network_factory is None:
            network_factory = make_amor_networks
        make_amor_inference_fn, params = load_make_amor_inference_fn(
            config,
            network_factory,
        )
        # checkpoint stores (normalizer, policy, value); inference uses (normalizer, policy).
        normalizer_params, policy_params = params[0], params[1]
        amor_inference_fn = make_amor_inference_fn(
            params        = (normalizer_params, policy_params),
            deterministic = deterministic,
        )
        tradeoff_jnp = jax.numpy.asarray(tradeoff)
        def policy(obs, key):
            return amor_inference_fn(obs, tradeoff_jnp, key)
        return policy
    else:
        raise ValueError(f"Unknown algorithm '{algo}' in config.")


def load_hypernetworks(
    config,
    network_factory = make_morlax_networks,
    path = None,
    quiet = True
) -> tuple[mop.moppo.factory.MORLAXNetworks, dict]:
    """Loads the MOPPO object"""
    if path is None:
        path = mm.learning.inference.get_last_model(config)
    if not quiet: print(f'Loading model at {path.as_posix()}')
    fullpath = path.resolve()

    fullpath = epath.Path(fullpath)
    params_config = checkpoint.load_config(fullpath)
    hyperparams = checkpoint.load(fullpath)
    hyperconfig = config['learning_params']['network_params']
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


def load_amor_networks(
    config,
    network_factory = make_amor_networks,
    path = None,
    quiet = True
) -> tuple:
    """Load AMOR networks + saved (normalizer, policy, value) params."""
    if path is None:
        path = mm.learning.inference.get_last_model(config)
    if not quiet: print(f'Loading model at {path.as_posix()}')
    fullpath = epath.Path(path.resolve())
    params_config = checkpoint.load_config(fullpath)
    saved_params = checkpoint.load(fullpath)
    network_params = config['learning_params']['amor_params']['network_params']
    network_factory = functools.partial(
        network_factory,
        key            = jax.random.PRNGKey(0),
        num_objectives = len(config['env_config']['reward']['optimization']['objectives']),
        **network_params,
    )
    amor_networks = get_network(params_config, network_factory)
    return amor_networks, saved_params


def load_make_amor_inference_fn(
    config,
    network_factory = make_amor_networks,
    path = None,
):
    """Load the (call-time-directive) AMOR inference function from a checkpoint.

    Returns ``(amor_inference_fn, saved_params)`` where ``amor_inference_fn`` has
    signature ``(params, deterministic) -> policy(obs, directive, key)`` and
    ``saved_params`` is the ``(normalizer, policy, value)`` 3-tuple from the
    checkpoint.
    """
    amor_networks, saved_params = load_amor_networks(config, network_factory, path)
    make_inference_fn = make_amor_inference_fn(amor_networks)
    return make_inference_fn, saved_params

def rollout_policy(
    env, 
    config,
    tradeoff = None,
    T=10.0,
    camera = 'track',
    width  = 1080,
    height = 720,
    inference_fn = None,
):

    """Roll out a trained policy on ``env`` and return rendered frames.

    Loads the policy specified by ``config`` (single-objective if
    ``config.mo2so.enabled`` else multi-objective with the given
    ``tradeoff``), JIT-compiles the inference function, and delegates the
    actual stepping/rendering to ``minimal_mjx.eval.rollout_policy``.

    Args:
        env: The environment to roll out in (typically built with
            ``create_environment``).
        config: Run config (ConfigDict) for the trained policy.
        tradeoff: Per-objective weighting used to condition the
            multi-objective policy. Defaults to all-ones with length equal
            to the number of objectives in ``config``. Ignored when
            ``config.mo2so.enabled`` or when ``inference_fn`` is given.
        T: Rollout duration in seconds.
        camera: MuJoCo camera name to render from.
        width: Frame width in pixels.
        height: Frame height in pixels.
        inference_fn: Optional pre-built policy ``policy(obs, key)``, e.g.
            from ``load_mo_policy``. If given, it is used as-is instead of
            loading a policy from ``config``.

    Returns:
        Whatever ``minimal_mjx.eval.rollout_policy`` returns — at the time
        of writing a 4-tuple ``(frames, reward_plotter, ...)``.
    """
    if inference_fn is None:
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