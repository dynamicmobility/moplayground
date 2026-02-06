from moplayground.learning.startup import train_policy
from minimal_mjx.learning.startup import read_config
from moplayground.envs.create import create_environment
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("env", type=str, help="Env to train on")
args = parser.parse_args()

match args.env:
    case 'BRUCE':
        config = read_config(
            'src/moplayground/envs/locomotion/config/bruce-navigait.yaml'
        )
        env, env_cfg = create_environment(config, for_training=True)
        eval_env, _ = create_environment(
            config, for_training=True, manual_speed=True, idealistic=True
        )
    case 'MOCheetah':
        config = read_config()
        env, env_cfg = create_environment(config, for_training=True)
        eval_env, _ = create_environment(
            config, for_training=True
        )
    # case _:
    #     config = read_config(args.env)
    #     env, env_cfg = create_environment(config, for_training=True)
    #     eval_env, _ = create_environment(
    #         config, for_training=True
    #     )
    


train_policy(config, env, eval_env)