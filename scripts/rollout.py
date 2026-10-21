import os
os.environ["MUJOCO_GL"] = "egl" # (comment out if not on Ubuntu SSH)
os.environ['JAX_PLATFORMS']='cpu'
import argparse
import jax
import numpy as np
import moplayground as mop
import minimal_mjx as mm
from pathlib import Path


def load_policy(cfg: mop.MOConfig, tradeoff: np.ndarray, deterministic: bool = True):
    """Load the latest checkpoint of the run described by ``cfg``."""
    match cfg.algorithm:
        case 'ppo':
            return mm.load_policy(cfg.run_dir, deterministic=deterministic)
        case 'morlax':
            network_params = cfg.morlax_network_params
        case 'amor':
            network_params = cfg.amor_network_params
        case _:
            raise ValueError(f'Unknown algorithm {cfg.algorithm}')
    return mop.load_mo_policy(
        algorithm       = cfg.algorithm,
        network_params  = network_params,
        num_objectives  = cfg.num_objectives,
        run_dir         = cfg.run_dir,
        tradeoff        = tradeoff,
        deterministic   = deterministic
    )


def main(
    config_path      : Path,
    inference_fn_type: str = 'policy',
    tradeoff         : list = None,
    T                : float = 5.0,
    camera           : str = 'track',
    seed             : int = None
):
    if seed is not None:
        np.random.seed(seed)

    # Read the config file and create the environment
    cfg    = mop.MOConfig(config_path)
    kwargs = {} if cfg.env_name != 'NaviGait' else {
        'manual_speed'    : [0.12, 0.0, 0.0],
        'track_yaw'       : False,
        'idealistic'      : True
    }
    env, env_params = mop.create_environment(
        env_name     = cfg.env_name,
        env_params   = cfg.env_params,
        backend      = cfg.backend,
        gaitlib_path = cfg.gaitlib_path,
        **kwargs
    )

    # Choose a tradeoff (a random point on the simplex unless one is given)
    n_objs = cfg.num_objectives
    if tradeoff is None:
        tradeoff = np.random.dirichlet(alpha=np.ones(n_objs))
    tradeoff = np.asarray(tradeoff, dtype=float)
    assert tradeoff.shape == (n_objs,), f'Expected {n_objs} tradeoff weights, got {tradeoff.shape[0]}'

    # Build the inference function
    if inference_fn_type == 'policy':
        print(f'Chosen tradeoff {tradeoff} with {n_objs} objectives')
        inference_fn = jax.jit(load_policy(cfg, tradeoff))
    else:
        inference_fn = mm.make_dummy_inference_fn(env=env, mode=inference_fn_type)

    # Rollout the policy
    frames, traj, reward_plotter, _, _ = mm.rollout_policy(
        inference_fn = inference_fn,
        env          = env,
        n_steps      = int(T / env.dt),
        camera       = camera,
        width        = 640,
        height       = 480
    )

    # Save video and metrics
    mm.save_video(
        frames,
        env.dt,
        Path(f'output/videos/{cfg.env_name}-rollout.mp4')
    )
    mm.save_metrics(
        reward_plotter,
        Path(f'output/videos/{cfg.env_name}-reward.pdf')
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Roll out a policy and save a video and reward plot.')
    parser.add_argument("config_path", type=str, help="YAML config of the run to roll out")
    parser.add_argument('--inference_fn', default='policy', choices=['policy', 'zero', 'random'],
                        help='Use the trained policy, or a dummy zero/random action function')
    parser.add_argument('--tradeoff', type=float, nargs='+', default=None,
                        help='Objective weights to condition the policy on (default: random)')
    parser.add_argument('--T', type=float, default=5.0, help='Rollout duration in seconds')
    parser.add_argument('--camera', default='track', help='Camera name to render from')
    parser.add_argument('--seed', type=int, default=None, help='Seed for the tradeoff and initial state')
    args = parser.parse_args()

    main(
        config_path       = args.config_path,
        inference_fn_type = args.inference_fn,
        tradeoff          = args.tradeoff,
        T                 = args.T,
        camera            = args.camera,
        seed              = args.seed
    )
