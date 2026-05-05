import os
os.environ["MUJOCO_GL"] = "egl"
os.environ['JAX_PLATFORMS']='cpu'
from minimal_mjx.utils.setupGPU import run_setup
from pathlib import Path
import jax
from jax import numpy as jnp
import numpy as np
import functools
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import argparse
from ral import FINAL_YAMLS, HYPER_TIMES

import moplayground as mop
import minimal_mjx as mm

if __name__ == '__main__':
    run_setup()
    parser = argparse.ArgumentParser()
    parser.add_argument('algo', type=str, help="Algo to use")
    parser.add_argument('env', type=str, help="Env to train on")
    args = parser.parse_args()
    config = mm.utils.read_config(FINAL_YAMLS[args.algo][args.env])
    
    if args.algo == 'morlax':
        rewards_over_iters, directives = mop.eval.pareto.get_morlax_fronts(
            config          = config,
            rng             = jax.random.PRNGKey(0),
            env             = mop.envs.create_environment(config, for_training=True)[0],
            N_STEPS         = 500,
            NUM_ENVS        = 2**4, #10,
            save_results    = True
        )
    elif args.algo == 'amor':
        rewards_over_iters, directives = mop.eval.pareto.get_amor_fronts(
            config          = config,
            rng             = jax.random.PRNGKey(0),
            env             = mop.envs.create_environment(config, for_training=True)[0],
            N_STEPS         = 500,
            NUM_ENVS        = 2**4, #10,
            save_results    = True
        )