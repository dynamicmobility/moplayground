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
from moplayground.moppo import networks
from brax.envs.wrappers.training import VmapWrapper

# jax and MJX imports
from mujoco_playground import wrapper
from mujoco_playground._src import mjx_env
import numpy as np
from minimal_mjx.utils.plotting import get_subplot_grid

def plot_mo_progress(
    run, num_steps, metrics, times, x_data, y_data, directives, save_dir, labels
):
    tz = ZoneInfo("America/New_York")
    now = datetime.now(tz)
    print(now.strftime("%Y-%m-%d %H:%M:%S %Z"))

    x_data.append(num_steps)
    y_data.append(metrics['reward'])
    directives.append(metrics['directive'])
    times.append(time.time())
    pd.DataFrame({'times': times, 'iters': [0] + x_data}).to_csv(save_dir / 'progress.csv')
    
    nrows, ncols = get_subplot_grid(len(x_data))
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    if type(axs) == np.ndarray:
        axs = axs.flatten()
    else:
        axs = [axs]
    
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    directives = np.array(directives)
    xlim   = np.array((np.min(y_data[..., 0]), np.max(y_data[..., 0])))
    ylim   = np.array((np.min(y_data[..., 1]), np.max(y_data[..., 1])))
    border = np.array([-1., 1.])
    xlim   = xlim + border * np.abs(xlim[1] - xlim[0]) * 0.1
    ylim   = ylim + border * np.abs(ylim[1] - ylim[0]) * 0.1
    for ax, x, y, d in zip(axs, x_data, y_data, directives):
        c = np.zeros((y_data.shape[1], 3))
        c[:, 0] = d[:, 0]
        c[:, 2] = d[:, 1]
        ax.scatter(
            y[:, 0],
            y[:, 1],
            s=4,
            c=c
        )
        ax.set_title(x)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        
    fig.set_size_inches((4 * ncols, 4 * nrows))
    fig.tight_layout()
    fig.savefig(save_dir / 'progress.svg')
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
        networks.make_moppo_networks,
        **config_yaml['learning_params']['hypernetwork_params'],
        **network_params
    )
    network_config = checkpoint.network_config(
        observation_size=eval_env.observation_size,
        action_size=eval_env.action_size,
        normalize_observations=moppo_params.normalize_observations,
        network_factory=network_factory,
    )
    run = initialize_wandb(name=f'{config_yaml['save_dir']}/{config_yaml['name']}')
    x_data, y_data, directives = [], [], []
    times = [time.time()]
    train_fn = functools.partial(
        train_algo, **dict(moppo_params),
        network_factory=network_factory,
        progress_fn=lambda num_steps, metrics: plot_mo_progress(
            run         = run,
            num_steps   = num_steps,
            metrics     = metrics,
            times       = times,
            x_data      = x_data,
            y_data      = y_data,
            directives  = directives,
            save_dir    = output_dir,
            labels      = env.params.reward.optimization.objectives
        ),
        policy_params_fn=lambda current_step, make_policy, params: checkpoint.save(
            path   = output_dir.resolve(),
            step   = current_step,
            params = params,
            config = network_config
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