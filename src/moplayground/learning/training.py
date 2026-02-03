# Basic imports
from pathlib import Path
import time
import datetime
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

# Graphics and plotting.
import wandb
import matplotlib.pyplot as plt

# RL imports
import functools
from brax.training.agents.ppo import checkpoint

from moplayground.moppo import moppo
from moplayground.moppo import factory
from moplayground.learning.wrappers import MultiObjectiveEpisodeWrapper
from brax.envs.wrappers.training import VmapWrapper

# jax and MJX imports
from mujoco_playground import wrapper
from mujoco_playground._src import mjx_env
import numpy as np
from minimal_mjx.utils.plotting import get_subplot_grid
from minimal_mjx.learning.training import initialize_wandb, save_model
from moplayground.utils.plotting import plot_squential_paretos, plot_sequential_hypervolume
from dataclasses import dataclass, field

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
        fig, axs = plot_squential_paretos(
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
    
def mo_train(
    config_yaml, output_dir: Path, env, eval_env, moppo_params, network_params, policy_init_params
):
    train_algo = moppo.train
    network_factory = functools.partial(
        factory.make_moppo_networks,
        **config_yaml['learning_params']['hypernetwork_params'],
        **network_params
    )
    network_config = checkpoint.network_config(
        observation_size=eval_env.observation_size,
        action_size=eval_env.action_size,
        normalize_observations=moppo_params.normalize_observations,
        network_factory=network_factory,
    )
    run = initialize_wandb(name=f'{config_yaml['save_dir']}/{config_yaml['name']}') # move this outside...
    training_data = MOTrainingInfo(
        start_time = time.time(),
        labels = env.params.reward.optimization.objectives
    )
    train_fn = functools.partial(
        train_algo, **dict(moppo_params),
        network_factory=network_factory,
        progress_fn=lambda num_steps, metrics: plot_mo_progress(
            run             = run,
            num_steps       = num_steps,
            metrics         = metrics,
            save_dir        = output_dir,
            training_data   = training_data
        ),
        policy_params_fn=functools.partial(
            save_model,
            output_dir        = output_dir,
            run               = run,
            network_config    = network_config
        ),
    )
    
    make_inference_fn, trained_params, metrics = train_fn(
        environment=env,
        # wrap_env_fn=wrapper.wrap_for_brax_training,
        wrap_env_fn=mo_wrapper,
        init_policy_params=policy_init_params[1],
        init_normalizer_params=policy_init_params[0],
        init_value_params=policy_init_params[2]
        # eval_env=eval_env,
    )
    # print(f"time to jit: {times[1] - times[0]}")
    # print(f"time to train: {times[-1] - times[1]}")
    
    return make_inference_fn, trained_params, metrics