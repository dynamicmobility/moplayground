import matplotlib
matplotlib.use('Agg')
import moplayground as mop
import minimal_mjx as mm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("env", type=str, help="Env to train on")
args = parser.parse_args()
TRAIN_KWARGS = {}
EVAL_KWARGS  = {}

match args.env:
    case 'BRUCE':
        CONFIG_PATH = 'config/bruce-navigait.yaml'
        EVAL_KWARGS = {'manual_speed': [0.0, 0.0, 0.0], 'idealistic': True}
        train_config = mop.utils.read_config(CONFIG_PATH)
        eval_config  = mop.utils.read_config(CONFIG_PATH)
    case 'MOCheetah':
        CONFIG_PATH = 'config/mocheetah.yaml'
        train_config = mop.utils.read_config(CONFIG_PATH)
        eval_config  = mop.utils.read_config(CONFIG_PATH)
    case 'MOHopper':
        CONFIG_PATH = 'config/mohopper.yaml'
        train_config = mop.utils.read_config(CONFIG_PATH)
        eval_config  = mop.utils.read_config(CONFIG_PATH)
    case 'MOWalker':
        CONFIG_PATH = 'config/mowalker.yaml'
        train_config = mop.utils.read_config(CONFIG_PATH)
        eval_config  = mop.utils.read_config(CONFIG_PATH)
    case _:
        CONFIG_PATH = args.env
        train_config = mop.utils.read_config(CONFIG_PATH)
        eval_config  = mop.utils.read_config(CONFIG_PATH)
    
print('Training', CONFIG_PATH)
env, env_cfg = mop.envs.create_environment(train_config, for_training=True, **TRAIN_KWARGS)
eval_env, _  = mop.envs.create_environment(eval_config, for_training=True, **EVAL_KWARGS)
name = train_config['save_dir'] + '/' + train_config['name']
run = mm.utils.logging.initialize_wandb(
    name    = name.replace('/', ''),
    entity  = 'njanwani-gatech',
    project = 'PrefMORL'
)
mop.learning.train_policy(train_config, env, eval_env, run)