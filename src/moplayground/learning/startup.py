# Internal imports
from minimal_mjx.utils.setupGPU import run_setup
from minimal_mjx.learning.startup import read_config, get_commit_hash, read_yaml
from minimal_mjx.learning.training import setup_training, train
from minimal_mjx.learning.inference import get_params

from moplayground.learning.training import mo_train
from moplayground.envs.generic.mobase import Multi2SingleObjective
from moplayground.envs.create import create_environment

# Basic imports
import yaml
import os
import numpy as np
from pathlib import Path
from ml_collections import config_dict

def mo2so(env, weighting):
    return Multi2SingleObjective(
        env       = env,
        weighting = weighting
    )
    
def train_policy(config, env, eval_env):
    run_setup()
    
    # Initialize stuff
    output_dir = Path(config['save_dir']) / config['name']
    os.makedirs(output_dir, exist_ok=config['name'] == 'test')
    ppo_params, network_params = setup_training(config.learning_params)

    if config.mo2so.enabled:
        weighting = np.array(config.mo2so.weighting)
        env = mo2so(env, weighting=weighting)
    else:
        ppo_params = config_dict.ConfigDict(
            dict(config.learning_params.hypermorl_params) | dict(ppo_params)
        )
    

    # Save configuration
    config_save_path = Path(output_dir) / 'config.yaml'
    if config.name != 'test':
        git_hash = get_commit_hash()
        config.git_hash = git_hash
    with open(config_save_path, 'w') as f:
        yaml.dump(config.to_dict(), f)
    # Train
    print('Training...')
    if config.mo2so.enabled:
        make_inference_fn, params, metrics = train(
            config, output_dir, env, eval_env, ppo_params, network_params
        )
    else:
        if config.learning_params.warmup_params.enabled:
            policy_init_params = get_params(
                read_yaml(config.learning_params.warmup_params.policy),
            )
        else:
            policy_init_params = (None, None, None)
        make_inference_fn, params, metrics = mo_train(
            config, output_dir, env, eval_env, ppo_params, network_params, policy_init_params
        )
    
    return make_inference_fn, params
    
    
if __name__ == "__main__":
    config = read_config()
    env, env_cfg = create_environment(config, for_training=True)
    eval_env, _ = create_environment(
        config, for_training=True, manual_speed=True
    )
    train_policy(config, env)