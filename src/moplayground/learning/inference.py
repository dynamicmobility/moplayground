from pathlib import Path
import numpy as np
import jax
from etils import epath
import functools
from brax.training.checkpoint import get_network
from brax.training.agents.ppo import checkpoint
from moplayground.moppo.factory import (
    make_morlax_networks,
    make_hypernetwork_inference_fn,
    make_amor_networks,
    make_amor_inference_fn,
)
import moplayground as mop


def get_all_models(run_dir) -> list[Path]:
    """Checkpoint directories in ``run_dir``, sorted by training step.

    A checkpoint directory is a subdirectory whose name is an integer (the
    training step), e.g. ``000050032640``.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Model directory does not exist: {run_dir}")
    model_files = [p for p in run_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    model_files.sort(key=lambda p: int(p.name))
    return model_files


def get_last_model(run_dir) -> Path:
    """Most recent checkpoint directory in ``run_dir``."""
    return get_all_models(run_dir)[-1]


def load_mo_policy(
    algorithm: str,
    network_params: dict,
    num_objectives: int,
    run_dir,
    tradeoff: np.ndarray,
    network_factory = None,
    deterministic: bool = True,
):
    """Load the latest multi-objective policy in ``run_dir``.

    Dispatches on ``algorithm`` (``'morlax'`` or ``'amor'``). Returns a 2-arg callable
    ``policy(obs, key) -> (action, extras)`` with ``tradeoff`` baked in, so it
    is compatible with ``mm.eval.rollout_policy`` and other consumers.

    For AMOR specifically, the underlying inference function natively accepts
    a directive at call time (so the tradeoff can change per step). To get
    that 3-arg form, call :func:`load_amor_inference_fn` directly instead of
    this function.
    """
    if algorithm == 'morlax':
        if network_factory is None:
            network_factory = make_morlax_networks
        hypernetwork_inference_fn, params = load_hypernetwork_inference_fn(
            network_params,
            num_objectives,
            run_dir         = run_dir,
            network_factory = network_factory,
        )
        return hypernetwork_inference_fn(
            params        = params,
            deterministic = deterministic,
            directive     = tradeoff,
            single_policy = True,
        )
    elif algorithm == 'amor':
        if network_factory is None:
            network_factory = make_amor_networks
        make_amor_inference_fn, params = load_make_amor_inference_fn(
            network_params,
            num_objectives,
            run_dir         = run_dir,
            network_factory = network_factory,
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
        raise ValueError(f"Unknown algorithm '{algorithm}'.")


def load_hypernetworks(
    network_params: dict,
    num_objectives: int,
    run_dir = None,
    path = None,
    network_factory = make_morlax_networks,
    quiet = True
) -> tuple[mop.moppo.factory.MORLAXNetworks, dict]:
    """Load MORLAX networks and hypernetwork params from a checkpoint.

    Args:
        network_params: MORLAX network-factory keyword arguments.
        num_objectives: Number of objectives.
        run_dir: Run directory; the latest checkpoint in it is used when
            ``path`` is not given.
        path: Explicit checkpoint directory. Overrides ``run_dir``.
    """
    if path is None:
        path = get_last_model(run_dir)
    path = Path(path)
    if not quiet: print(f'Loading model at {path.as_posix()}')
    fullpath = epath.Path(path.resolve())
    params_config = checkpoint.load_config(fullpath)
    hyperparams = checkpoint.load(fullpath)
    network_factory = functools.partial(
        network_factory,
        key            = jax.random.PRNGKey(0),
        num_objectives = num_objectives,
        **network_params
    )
    hypernetworks = get_network(params_config, network_factory)
    return hypernetworks, hyperparams


def load_hypernetwork_inference_fn(
    network_params: dict,
    num_objectives: int,
    run_dir = None,
    path = None,
    network_factory = make_morlax_networks,
):
    """Loads policy inference function from PPO checkpoint."""
    hypernetworks, hyperparams = load_hypernetworks(
        network_params, num_objectives, run_dir=run_dir, path=path,
        network_factory=network_factory,
    )
    make_inference_fn = make_hypernetwork_inference_fn(hypernetworks)
    return make_inference_fn, hyperparams


def load_amor_networks(
    network_params: dict,
    num_objectives: int,
    run_dir = None,
    path = None,
    network_factory = make_amor_networks,
    quiet = True
) -> tuple:
    """Load AMOR networks + saved (normalizer, policy, value) params.

    Args:
        network_params: AMOR network-factory keyword arguments.
        num_objectives: Number of objectives.
        run_dir: Run directory; the latest checkpoint in it is used when
            ``path`` is not given.
        path: Explicit checkpoint directory. Overrides ``run_dir``.
    """
    if path is None:
        path = get_last_model(run_dir)
    path = Path(path)
    if not quiet: print(f'Loading model at {path.as_posix()}')
    fullpath = epath.Path(path.resolve())
    params_config = checkpoint.load_config(fullpath)
    saved_params = checkpoint.load(fullpath)
    network_factory = functools.partial(
        network_factory,
        key            = jax.random.PRNGKey(0),
        num_objectives = num_objectives,
        **network_params,
    )
    amor_networks = get_network(params_config, network_factory)
    return amor_networks, saved_params


def load_make_amor_inference_fn(
    network_params: dict,
    num_objectives: int,
    run_dir = None,
    path = None,
    network_factory = make_amor_networks,
):
    """Load the (call-time-directive) AMOR inference function from a checkpoint.

    Returns ``(amor_inference_fn, saved_params)`` where ``amor_inference_fn`` has
    signature ``(params, deterministic) -> policy(obs, directive, key)`` and
    ``saved_params`` is the ``(normalizer, policy, value)`` 3-tuple from the
    checkpoint.
    """
    amor_networks, saved_params = load_amor_networks(
        network_params, num_objectives, run_dir=run_dir, path=path,
        network_factory=network_factory,
    )
    make_inference_fn = make_amor_inference_fn(amor_networks)
    return make_inference_fn, saved_params