import matplotlib
matplotlib.use('Agg')
from moplayground.learning.startup import train_policy
from minimal_mjx.learning.startup import read_config
from moplayground.envs.create import create_environment
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("env", type=str, help="Env to train on")
args = parser.parse_args()
TRAIN_KWARGS = {}
EVAL_KWARGS  = {}

match args.env:
    case 'BRUCE':
        CONFIG_PATH = 'config/bruce-navigait.yaml'
        EVAL_KWARGS = {'manual_speed': True, 'idealistic': True}
        train_config = read_config(CONFIG_PATH)
        eval_config  = read_config(CONFIG_PATH)
    case 'MOCheetah':
        CONFIG_PATH = 'config/mocheetah.yaml'
        train_config = read_config(CONFIG_PATH)
        eval_config  = read_config(CONFIG_PATH)
    case 'MOHopper':
        CONFIG_PATH = 'config/mohopper.yaml'
        train_config = read_config(CONFIG_PATH)
        eval_config  = read_config(CONFIG_PATH)
    case _:
        CONFIG_PATH = args.env
        train_config = read_config(CONFIG_PATH)
        eval_config = read_config(CONFIG_PATH)
    
print('Training', CONFIG_PATH)
env, env_cfg = create_environment(train_config, for_training=True, **TRAIN_KWARGS)
eval_env, _  = create_environment(eval_config, for_training=True, **EVAL_KWARGS)
train_policy(train_config, env, eval_env)