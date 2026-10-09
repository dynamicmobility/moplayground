from pathlib import Path
import time
import datetime
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

# RL imports
import functools
from brax.training.agents.ppo import checkpoint

import moplayground as mop
from moplayground.moppo import morlax
from moplayground.moppo import amor
from moplayground.moppo import factory
from moplayground.learning.wrappers import MultiObjectiveEpisodeWrapper
from brax.envs.wrappers.training import VmapWrapper

# jax and MJX imports
from mujoco_playground import wrapper
from mujoco_playground._src import mjx_env
import minimal_mjx as mm

# TODO: support privileged value observations. Envs already return a 'privileged_state'
# obs, but both factories default policy_obs_key and value_obs_key to 'state'. Add a
# config key + getter in config.py and pass value_obs_key='privileged_state' here.
def setup_morlax(ppo_params, sampling_params, network_params):
    train_fn_params = dict(ppo_params) | dict(sampling_params)
    
    network_factory = functools.partial(
        factory.make_morlax_networks,
        **network_params
    )

    train_fn = functools.partial(
        morlax.train, **dict(train_fn_params),
        network_factory=network_factory,
    )
        
    return train_fn, network_factory

def setup_amor(ppo_params, sampling_params, network_params):
    train_fn_params = dict(ppo_params) | dict(sampling_params)

    network_factory = functools.partial(
        factory.make_amor_networks,
        **network_params
    )

    train_fn = functools.partial(
        amor.train, **dict(train_fn_params),
        network_factory=network_factory,
    )

    return train_fn, network_factory


_ALGO_HANDLERS = {
    'morlax': setup_morlax,
    'amor':   setup_amor,
}


def train_policy(
    algorithm,
    ppo_params,
    sampling_params,
    network_params,
    run_dir,
    env,
    eval_env,
    run=None,
    handle_params=None,
    progress_fn=None,
):
    """Train a multi-objective policy on the given environment.

    Sets up the GPU, builds the train function and network factory for
    ``algorithm``, and runs training. Progress plots and checkpoints are
    written to ``run_dir``. The caller creates ``run_dir`` and writes the
    run's ``config.yaml`` before calling this function (see
    ``moplayground.config.save``).

    Args:
        algorithm: ``'morlax'`` or ``'amor'``.
        ppo_params: PPO keyword arguments shared by both algorithms.
        sampling_params: Preference-sampling keyword arguments
            (``alpha``, ``k``, ``sampling``, ``warmup_frac``).
        network_params: Keyword arguments for the algorithm's network factory.
        run_dir: Existing run directory for checkpoints and progress plots.
        env: Training environment.
        eval_env: Evaluation environment used for periodic rollouts.
        run: (optional) Experiment-tracking handle (e.g. a wandb run) used to
            log progress and checkpoints.
        handle_params: (optional) Callable
            ``(ppo_params, sampling_params, network_params) -> (train_fn, network_factory)``.
            Defaults to the handler registered for ``algorithm`` in
            ``_ALGO_HANDLERS``.
        progress_fn: (optional) Callback invoked each eval step as
            ``progress_fn(run, num_steps, metrics, save_dir, training_data)``
            to log/plot training progress. Defaults to
            ``mop.utils.plotting.plot_mo_progress``.

    Returns:
        Tuple ``(make_inference_fn, params)`` — a factory that builds an
        inference function and the trained policy parameters.
    """
    if progress_fn is None:
        progress_fn = mop.utils.plotting.plot_mo_progress
    mm.utils.setupGPU.run_setup()
    output_dir = Path(run_dir)

    # Load training and network structure
    if handle_params is None:
        print('Using default parameter handler')
        if algorithm not in _ALGO_HANDLERS:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. Expected one of {list(_ALGO_HANDLERS)}."
            )
        handle_params = _ALGO_HANDLERS[algorithm]
    train_fn, network_factory = handle_params(ppo_params, sampling_params, network_params)

    network_config = checkpoint.network_config(
        observation_size=eval_env.observation_size,
        action_size=eval_env.action_size,
        normalize_observations=ppo_params['normalize_observations'],
        network_factory=network_factory,
    )
    training_data = mop.utils.plotting.MOTrainingPlottingInfo(
        start_time = time.time(),
        labels = env.params.reward.optimization.objectives
    )
        
    train_fn = functools.partial(
        train_fn,
        progress_fn=lambda num_steps, metrics: progress_fn(
            run             = run,
            num_steps       = num_steps,
            metrics         = metrics,
            save_dir        = output_dir,
            training_data   = training_data
        ),
        policy_params_fn=functools.partial(
            mm.utils.logging.save_model,
            output_dir        = output_dir,
            run               = run,
            network_config    = network_config
        ),
    )
    
    # Start training
    print(
        'Started training at', 
        datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")
    )
    make_inference_fn, trained_params, metrics = train_fn(
        environment=env,
        wrap_env_fn=mo_wrapper,
        eval_env=eval_env
    )
    
    return make_inference_fn, trained_params, metrics    

def mo_wrapper(
    env: mjx_env.MjxEnv,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn = None,
) -> wrapper.Wrapper:
    """Multi-Objective Wrapper"""

    env = VmapWrapper(env)
    env = MultiObjectiveEpisodeWrapper(env, episode_length, action_repeat)
    env = wrapper.BraxAutoResetWrapper(env)
    return env
