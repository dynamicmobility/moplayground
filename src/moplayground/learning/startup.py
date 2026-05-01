# Internal imports
import minimal_mjx as mm
import moplayground as mop

# Basic imports
import yaml
import os
import numpy as np
from pathlib import Path
from ml_collections import config_dict

def mo2so(env, weighting):
    return mop.envs.generic.mobase.Multi2SingleObjective(
        env       = env,
        weighting = weighting
    )
    
def train_policy(config, env, eval_env, run):
    """Train a policy on the given environment.

    Sets up the GPU, builds PPO/network parameters from ``config``, saves
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
        run: Experiment-tracking handle (e.g. a wandb run) forwarded to the
            multi-objective trainer; ignored on the single-objective path.

    Returns:
        Tuple ``(make_inference_fn, params)`` — a factory that builds an
        inference function and the trained policy parameters.
    """
    mm.utils.setupGPU.run_setup()
    
    # Initialize stuff
    output_dir = Path(config['save_dir']) / config['name']
    os.makedirs(output_dir, exist_ok=config['name'] == 'test')
    ppo_params, network_params = mm.learning.training.setup_training(config)

    if config.mo2so.enabled:
        weighting = np.array(config.mo2so.weighting)
        env         = mo2so(env, weighting=weighting)
        eval_env    = mo2so(eval_env, weighting=weighting)
    else:
        ppo_params = config_dict.ConfigDict(
            dict(config.learning_params.hypermorl_params) | dict(ppo_params)
        )
    

    # Save configuration
    config_save_path = Path(output_dir) / 'config.yaml'
    if config.name != 'test':
        git_hash = mm.utils.config.get_commit_hash()
        config.git_hash = git_hash
    with open(config_save_path, 'w') as f:
        yaml.dump(config.to_dict(), f)
    # Train
    print('Training...')
    if config.mo2so.enabled:
        make_inference_fn, params, metrics = mm.learning.training.train(
            config, output_dir, env, eval_env, ppo_params, network_params
        )
    else:
        if config.learning_params.warmup_params.enabled:
            policy_init_params = mm.learning.inference.get_params(
                mm.utils.config.read_yaml(config.learning_params.warmup_params.policy),
            )
        else:
            policy_init_params = (None, None, None)
        make_inference_fn, params, metrics = mop.learning.training.mo_train(
            config_yaml           = config,
            output_dir            = output_dir,
            env                   = env,
            eval_env              = eval_env,
            moppo_params          = ppo_params,
            network_params        = network_params,
            policy_init_params    = policy_init_params,
            run                   = run
        )
    
    return make_inference_fn, params