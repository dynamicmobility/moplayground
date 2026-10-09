import os
os.environ["MUJOCO_GL"] = "egl" # (comment out if not on Ubuntu SSH)
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np
import moplayground as mop
import minimal_mjx as mm
from pathlib import Path
import argparse
import functools

def main(config_path, tradeoff, T, height, width):
    # Read the config file and create the environment
    config = mm.read_config(config_path)
    kwargs = {} if config['env'] != 'NaviGait' else {
        'manual_speed'    : [0.12, 0.0, 0.0],
        'track_yaw'       : False,
        'idealistic'      : True
    }
    env, env_params = mop.create_environment(
        config,
        **kwargs
    )

    # Choose a tradeoff
    camera    = 'track'
    n_objs    = mop.get_num_objectives(config)
    tradeoff  = np.random.dirichlet(alpha=np.ones(n_objs))
    print(f'Chosen tradeoff {tradeoff} with {n_objs} objectives')

    # # Rollout the policy
    # frames, reward_plotter, _, _ = mop.learning.inference.rollout_policy(
    #     env         = env,
    #     config      = config,
    #     tradeoff    = tradeoff,
    #     T           = 6.0,
    #     camera      = camera,
    #     width       = 2560,
    #     height      = 1440
    # )
    inference_fn = mop.load_mo_policy(
        # config          = config,
        algo            = config['algorithm'],
        model_path      = mm.get_last_model(config),
        tradeoff        = tradeoff,
        deterministic   = True,
        network_factory = functools.partial(
            mop.make_morlax_networks,
            mop.get_num_objectives(config),
            **config.learning_params.morlax_params.network_params,
        )
    )
    frames, reward_plotter, _, _ = mm.rollout_policy(
        inference_fn    = inference_fn,
        env             = env,
        T               = T,
        height          = height,
        width           = width,
        camera          = camera
    )

    # Save video and metrics
    mm.save_video(
        frames,
        env.dt,
        Path(f'output/videos/{config['env']}-rollout.mp4')
    )
    mm.save_metrics(
        reward_plotter,
        Path(f'output/videos/{config['env']}-reward.pdf')
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Config to train on")
    parser.add_argument("--tradeoff", type=int, nargs='+', help="tradeoff (w) to evaluate")
    parser.add_argument("--height", type=int, default=720, help="height of video")
    parser.add_argument("--width", type=int, default=1080, help="width of video")
    parser.add_argument("--time", type=int, default=5.0, help="length (seconds) of video")
    args = parser.parse_args()
    main(
        config_path = args.config, 
        tradeoff = args.tradeoff,
        T = args.time,
        height = args.height,
        width = args.width
    )