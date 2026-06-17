from pathlib import Path
import time
import yaml
import os
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

def setup_morlax(config):
    general_ppo_params = config.learning_params.base_ppo_params
    morlax_algo_params = config.learning_params.morlax_params.train_fn_params
    network_params = config.learning_params.morlax_params.network_params
    
    train_fn_params = dict(general_ppo_params) | dict(morlax_algo_params)
    
    network_factory = functools.partial(
        factory.make_morlax_networks,
        **network_params
    )

    train_fn = functools.partial(
        morlax.train, **dict(train_fn_params),
        network_factory=network_factory,
    )
        
    return train_fn, network_factory

def setup_amor(config):
    general_ppo_params = config.learning_params.base_ppo_params
    amor_algo_params   = config.learning_params.amor_params.train_fn_params
    network_params     = config.learning_params.amor_params.network_params

    train_fn_params = dict(general_ppo_params) | dict(amor_algo_params)

    network_factory = functools.partial(
        factory.make_amor_networks,
        **network_params
    )

    train_fn = functools.partial(
        amor.train, **dict(train_fn_params),
        network_factory=network_factory,
    )

    return train_fn, network_factory


def create_training_directory(config, warn_github_changes=True):
    output_dir = Path(config['save_dir']) / config['name']
    os.makedirs(output_dir, exist_ok=config['name'] == 'test')
    
    # Save configuration
    config_save_path = Path(output_dir) / 'config.yaml'
    if config.name != 'test':
        git_hash = mm.utils.config.get_commit_hash(warn=warn_github_changes)
        config.git_hash = git_hash
    with open(config_save_path, 'w') as f:
        yaml.dump(config.to_dict(), f)

    return output_dir

_ALGO_HANDLERS = {
    'morlax': setup_morlax,
    'amor':   setup_amor,
}


def train_policy(
    config,
    env,
    eval_env,
    run=None,
    handle_params=None,
    warn_github_changes=False,
    progress_fn=mop.utils.plotting.plot_mo_progress
):
    """Train a policy on the given environment.

    Sets up the GPU, builds MOPPO network parameters from ``config``, saves
    the resolved config alongside the run, and dispatches to either the
    standard single-objective trainer (when ``config.mo2so.enabled`` is
    True — wrapping ``env``/``eval_env`` with ``Multi2SingleObjective``)
    or the multi-objective ``mo_train`` loop.

    Args:
        config: Training config (ConfigDict). Must include ``save_dir``,
            ``name``, ``mo2so`` (with ``enabled`` and, if enabled,
            ``weighting``), and ``learning_params``.
        env: Training environment.
        eval_env: Evaluation environment used for periodic rollouts.
        run: (optional) Experiment-tracking handle (e.g. a wandb run) forwarded to the
            multi-objective trainer; ignored on the single-objective path.
        handle_params: (optional) Callable ``config -> (train_fn, network_factory)``.
            Defaults to the handler registered for ``config.algorithm`` in
            ``_ALGO_HANDLERS``.
        warn_github_changes: (optional) If True, warn about uncommitted git
            changes when creating the training directory. Defaults to False.
        progress_fn: (optional) Callback invoked each eval step as
            ``progress_fn(run, num_steps, metrics, save_dir, training_data)``
            to log/plot training progress. Defaults to
            ``mop.utils.plotting.plot_mo_progress``.

    Returns:
        Tuple ``(make_inference_fn, params)`` — a factory that builds an
        inference function and the trained policy parameters.
    """
    mm.utils.setupGPU.run_setup()
    config = mm.utils.config.create_config_dict(config)
    output_dir = create_training_directory(config, warn_github_changes=warn_github_changes)

    # Load training and network structure
    if handle_params is None:
        print('Using default parameter handler')
        algo = config.algorithm
        if algo not in _ALGO_HANDLERS:
            raise ValueError(
                f"Unknown algorithm '{algo}'. Expected one of {list(_ALGO_HANDLERS)}."
            )
        handle_params = _ALGO_HANDLERS[algo]
    train_fn, network_factory = handle_params(config)

    network_config = checkpoint.network_config(
        observation_size=eval_env.observation_size,
        action_size=eval_env.action_size,
        normalize_observations=config.learning_params.base_ppo_params.normalize_observations,
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
    if run:
        run.log_artifact(str(output_dir / 'config.yaml'), name='config')
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
