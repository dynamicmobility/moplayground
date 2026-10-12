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
def setup_morlax(train_fn_params, network_factory_params, resumed):
    
    network_factory = functools.partial(
        factory.make_morlax_networks,
        **network_factory_params
    )

    train_fn = functools.partial(
        morlax.train, **dict(train_fn_params),
        network_factory=network_factory,
        wrap_env_fn=mo_wrapper
    )
        
    return train_fn, network_factory

def setup_amor(train_fn_params, network_factory_params, resumed):

    network_factory = functools.partial(
        factory.make_amor_networks,
        **network_factory_params
    )

    train_fn = functools.partial(
        amor.train, **dict(train_fn_params),
        network_factory=network_factory,
        wrap_env_fn=mo_wrapper
    )

    return train_fn, network_factory

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
