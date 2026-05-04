from pathlib import Path
import time
import yaml
import os
import datetime
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo
from ml_collections import config_dict

# Graphics and plotting.
import wandb
import matplotlib.pyplot as plt

# RL imports
import functools
from brax.training.agents.ppo import checkpoint

import moplayground as mop
from moplayground.moppo import morlax
from moplayground.moppo import factory
from moplayground.envs.generic import mobase
from moplayground.learning.wrappers import MultiObjectiveEpisodeWrapper
from brax.envs.wrappers.training import VmapWrapper

# jax and MJX imports
from mujoco_playground import wrapper
from mujoco_playground._src import mjx_env
import numpy as np
import minimal_mjx as mm
from moplayground.utils.plotting import plot_sequential_paretos, plot_sequential_hypervolume
from dataclasses import dataclass, field

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
    pass
    
def create_training_directory(config):
    output_dir = Path(config['save_dir']) / config['name']
    os.makedirs(output_dir, exist_ok=config['name'] == 'test')
    
    # Save configuration
    config_save_path = Path(output_dir) / 'config.yaml'
    if config.name != 'test':
        git_hash = mm.utils.config.get_commit_hash()
        config.git_hash = git_hash
    with open(config_save_path, 'w') as f:
        yaml.dump(config.to_dict(), f)

    return output_dir

def train_policy(
    config, 
    env, 
    eval_env, 
    run=None, 
    handle_params=setup_morlax,
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

    Returns:
        Tuple ``(make_inference_fn, params)`` — a factory that builds an
        inference function and the trained policy parameters.
    """
    mm.utils.setupGPU.run_setup()
    config = mm.utils.config.create_config_dict(config)
    
    output_dir = create_training_directory(config)
    train_fn, network_factory = handle_params(config)

    if config.mo2so.enabled:
        weighting = np.array(config.mo2so.weighting)
        env         = mobase.Multi2SingleObjective(env, weighting=weighting)
        eval_env    = mobase.Multi2SingleObjective(eval_env, weighting=weighting)
        make_inference_fn, params, metrics = mm.learning.training.train(
            config, output_dir, env, eval_env, train_fn_params, network_factory_params
        )
    else:
        # if config.learning_params.warmup_params.enabled:
        #     policy_init_params = mm.learning.inference.get_params(
        #         mm.utils.config.read_yaml(config.learning_params.warmup_params.policy),
        #     )
        # else:
        policy_init_params = (None, None, None)
        
        network_config = checkpoint.network_config(
            observation_size=eval_env.observation_size,
            action_size=eval_env.action_size,
            normalize_observations=config.learning_params.base_ppo_params.normalize_observations,
            network_factory=network_factory,
        )
        training_data = MOTrainingInfo(
            start_time = time.time(),
            labels = env.params.reward.optimization.objectives
        )
        if run:
            run.log_artifact(str(output_dir / 'config.yaml'), name='config')
            
        train_fn = functools.partial(
            train_fn,
            progress_fn=lambda num_steps, metrics: plot_mo_progress(
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
        
        print(
            'Started training at', 
            datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S %Z")
        )
        make_inference_fn, trained_params, metrics = train_fn(
            environment=env,
            wrap_env_fn=mo_wrapper,
            init_policy_params=policy_init_params[1],
            init_normalizer_params=policy_init_params[0],
            init_value_params=policy_init_params[2],
            eval_env=eval_env
        )
        
        return make_inference_fn, trained_params, metrics
    
    return make_inference_fn, params

@dataclass(frozen=False)
class MOTrainingInfo:
    start_time    : float
    times         : list = field(default_factory=list)
    iterations    : list = field(default_factory=list)
    paretos       : list = field(default_factory=list)
    directives    : list = field(default_factory=list)
    labels        : list = field(default_factory=list)
    
    def save(self, save_dir, create_time=True):
        pd.DataFrame(
            {
                'times': [self.start_time] + self.times if create_time else self.times,
                'iters': [0] + self.iterations if create_time else self.iterations
            }
        ).to_csv(save_dir)

def plot_mo_progress(
    num_steps       : int,
    metrics         : dict,
    training_data   : MOTrainingInfo,
    save_dir        : Path,
    run             : wandb.Run = None
):
    # print current itme
    tz = ZoneInfo("America/New_York")
    now = datetime.now(tz)
    print(now.strftime("%Y-%m-%d %H:%M:%S %Z"))
    
    # save data from iteration
    training_data.iterations.append(num_steps)
    training_data.paretos.append(metrics['reward'])
    training_data.directives.append(metrics['directive'])
    training_data.times.append(time.time())
    training_data.save(save_dir / 'progress.csv')

    if np.array(training_data.directives).shape[2] == 2:
        # create the plot
        fig, axs = plot_sequential_paretos(
            ax_titles   = training_data.iterations,
            paretos     = training_data.paretos,
            directives  = training_data.directives,
            objectives  = training_data.labels
        )
    else:
        fig, axs = plot_sequential_hypervolume(
            iterations    = training_data.iterations,
            paretos       = training_data.paretos
        )
    
    # save and upload to wandb
    fig.savefig(save_dir / 'progress.svg')
    if run:
        with open(save_dir / 'progress.svg', "r") as f:
            svg = f.read()
        run.log(
            {"reward_plot": wandb.Html(svg)},
            step=num_steps,
        )
    

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
