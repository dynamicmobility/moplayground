import os
os.environ["MUJOCO_GL"] = "egl" # (comment out if not on Ubuntu SSH)
os.environ['JAX_PLATFORMS']='cpu'
import argparse
import jax
import numpy as np
import moplayground as mop
import minimal_mjx as mm
from pathlib import Path
from scripts.rollout import load_policy


def default_tradeoffs(cfg: mop.MOConfig) -> dict:
    """One one-hot tradeoff per objective, keyed by the objective label."""
    n_objs = cfg.num_objectives
    labels = cfg.objective_labels or [f'obj{i}' for i in range(n_objs)]
    return {label: np.eye(n_objs)[i] for i, label in enumerate(labels)}


def main(
    config_path: Path,
    tradeoffs  : list = None,
    T          : float = 10.0,
    camera     : str = 'track',
    width      : int = 1280,
    height     : int = 720,
    out_dir    : Path = Path('output/videos')
):
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

    # Collect the tradeoffs (one one-hot per objective unless some are given)
    n_objs = cfg.num_objectives
    if cfg.algorithm == 'ppo':
        named = {'ppo': None}
    else:
        if tradeoffs is None:
            named = default_tradeoffs(cfg)
        else:
            named = {'-'.join(f'{w:g}' for w in t): np.asarray(t, dtype=float) for t in tradeoffs}
        for key, tradeoff in named.items():
            assert tradeoff.shape == (n_objs,), f'Tradeoff {key}: expected {n_objs} weights, got {tradeoff.shape[0]}'

    # Roll out the policy once per tradeoff and save one video per tradeoff
    for key, tradeoff in named.items():
        print(f'Rolling out tradeoff {key}: {tradeoff}')
        inference_fn = jax.jit(load_policy(cfg, tradeoff))
        frames, _, reward_plotter, _, _ = mm.rollout_policy(
            inference_fn = inference_fn,
            env          = env,
            n_steps      = int(T / env.dt),
            camera       = camera,
            width        = width,
            height       = height
        )
        mm.save_video(
            frames,
            env.dt,
            Path(out_dir) / f'{cfg.env_name}-{key}.mp4'
        )
        mm.save_metrics(
            reward_plotter,
            Path(out_dir) / f'{cfg.env_name}-{key}-reward.pdf'
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Roll out a trained MO policy at several tradeoffs and save one video per tradeoff.')
    parser.add_argument("config_path", type=str, help="YAML config of the run to roll out")
    parser.add_argument('--tradeoff', type=float, nargs='+', action='append', default=None,
                        help='Objective weights; repeat the flag for more tradeoffs (default: one one-hot per objective)')
    parser.add_argument('--T', type=float, default=10.0, help='Rollout duration in seconds')
    parser.add_argument('--camera', default='track', help='Camera name to render from')
    parser.add_argument('--width', type=int, default=1280, help='Video width in pixels')
    parser.add_argument('--height', type=int, default=720, help='Video height in pixels')
    parser.add_argument('--out_dir', type=str, default='output/videos', help='Directory for the videos and reward plots')
    args = parser.parse_args()

    main(
        config_path = args.config_path,
        tradeoffs   = args.tradeoff,
        T           = args.T,
        camera      = args.camera,
        width       = args.width,
        height      = args.height,
        out_dir     = args.out_dir
    )
