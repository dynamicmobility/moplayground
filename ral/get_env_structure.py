import argparse
from moplayground.learning.startup import read_config
from moplayground.envs.create import create_environment
from ral import FINAL_YAMLS

parser = argparse.ArgumentParser()
parser.add_argument("env", type=str, help="Env to train on")
args = parser.parse_args()
path = FINAL_YAMLS[args.env]
config = read_config(path)
env, _ = create_environment(config)
print('Action Space:', env.action_size)
print()
print('Observation Space:', env.observation_size)
print()
print('Feature Count:', config.learning_params.hypernetwork_params.num_features)