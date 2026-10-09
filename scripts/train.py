import matplotlib
matplotlib.use('Agg')
import moplayground as mop
from moplayground import config
import minimal_mjx as mm
import argparse

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=str, help="YAML config to train with")
args = parser.parse_args()
TRAIN_KWARGS = {}
EVAL_KWARGS  = {}

# Read in config
cfg = config.load(args.config_path)

# Environment-specific config handling...
match config.env_name(cfg):
    case 'BRUCE':
        EVAL_KWARGS = {'manual_speed': [0.0, 0.0, 0.0], 'idealistic': True}

# Create environments
print('Training', args.config_path)
env_kwargs = dict(
    env_name     = config.env_name(cfg),
    backend      = config.backend(cfg, for_training=True),
    gaitlib_path = config.gaitlib_path(cfg),
)
env, env_cfg = mop.create_environment(env_params=config.env_params(cfg), **env_kwargs, **TRAIN_KWARGS)
eval_env, _  = mop.create_environment(env_params=config.env_params(cfg), **env_kwargs, **EVAL_KWARGS)

# Create the run directory and save the config (adds git_hash unless name is 'test')
run_dir = config.run_dir(cfg)
config_path = config.save(cfg, run_dir, warn_github_changes=False)

run = mm.utils.logging.initialize_wandb(
    name    = str(run_dir).replace('/', ''),
    entity  = 'njanwani-gatech',
    project = 'MO-Playground-2',
    config  = cfg.to_dict()
)
run.log_artifact(str(config_path), name='config')

mop.train_policy(
    algorithm       = config.algorithm(cfg),
    ppo_params      = config.ppo_params(cfg),
    sampling_params = config.sampling_params(cfg),
    network_params  = config.network_params(cfg),
    run_dir         = run_dir,
    env             = env,
    eval_env        = eval_env,
    run             = run,
)
