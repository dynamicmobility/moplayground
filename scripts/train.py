import matplotlib
matplotlib.use('Agg')
import moplayground as mop
import minimal_mjx as mm
import argparse

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("env", type=str, help="Env to train on")
args = parser.parse_args()
TRAIN_KWARGS = {}
EVAL_KWARGS  = {}

# Read in configs
train_config = mop.utils.read_config(args.env)
eval_config  = mop.utils.read_config(args.env)

# Environment-specific config handling...
match train_config.env:
    case 'BRUCE':
        EVAL_KWARGS = {'manual_speed': [0.0, 0.0, 0.0], 'idealistic': True}

# Create environments
print('Training', args.env)
env, env_cfg = mop.envs.create_environment(train_config, for_training=True, **TRAIN_KWARGS)
eval_env, _  = mop.envs.create_environment(eval_config, for_training=True, **EVAL_KWARGS)

name = train_config['save_dir'] + '/' + train_config['name']
run = mm.utils.logging.initialize_wandb(
    name    = name.replace('/', ''),
    entity  = 'njanwani-gatech',
    project = 'PrefMORL',
    config  = dict(train_config)
)
mop.learning.train_policy(train_config, env, eval_env, run)