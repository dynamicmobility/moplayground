"""Train a plain (single-objective) PPO baseline on a MO-Playground environment.

Usage::

    python -m scripts.train_ppo config/mohumanoid.yaml                  # weights 0.5,0.5
    python -m scripts.train_ppo config/mohumanoid.yaml --weights 1.0,0.0
    python -m scripts.train_ppo config/mohumanoid.yaml --name my-run

How it works
------------
1. The YAML config supplies only the environment (``env`` and ``env_config``)
   and ``save_dir``. Its ``learning_params`` (MORLAX / AMOR settings) are
   ignored.
2. The multi-objective env returns a reward vector, one entry per objective.
   For MOHumanoid this is ``[run + alive, energy + alive]``.
   ``Multi2SingleObjective`` replaces it with ``reward · weights``, so standard
   PPO sees a scalar reward.
3. Brax's standard PPO trains with ``PPO_PARAMS`` and ``NETWORK_PARAMS``
   below. The training block copies ``mm.learning.train`` (minimal_mjx),
   because minimal_mjx 0.1.5 does not work with brax 0.14.0 (see the TODO).

Outputs go to ``<save_dir>/<name>/``: a ``config.yaml`` that records the
weights and PPO settings, ``progress.csv`` and plots, and brax checkpoints.
The default name is ``ppo-w=<weights>``. The run also logs to wandb.

Design decisions
----------------
- Scalarization weights come from ``--weights`` (default ``0.5,0.5``), so one
  config can produce several fixed-tradeoff baselines. The shared ``alive``
  reward is added to every objective before weighting. Thus weights that sum
  to 1 keep the alive bonus at its configured scale.
- PPO settings are constants in this file, not in the YAML config. This keeps
  ``moplayground.config`` free of a third algorithm. Note that
  ``algorithm: ppo`` in a YAML file is a legacy alias for ``morlax``, so the
  YAML must not be used to select this baseline.
- PPO settings are mujoco_playground's tuned dm_control defaults
  (``dm_control_suite_params.brax_ppo_config``; humanoid has no overrides).
  ``episode_length`` is 1000 (the dm_control default). MORLAX configs use 500.
- ``NETWORK_PARAMS`` is empty, so brax's default PPO networks are used:
  policy ``(32,) * 4``, value ``(256,) * 5``. This matches mujoco_playground.
- The saved ``config.yaml`` uses minimal_mjx's layout (``learning_params``
  holds ``ppo_params`` and ``network_params``). ``moplayground.config.load``
  cannot read it, so ``scripts/rollout.py`` and ``scripts/draw_frontier.py``
  do not support these runs.
"""
import matplotlib
matplotlib.use('Agg')
import argparse
import functools
from pathlib import Path

from ml_collections import config_dict
from brax.training.agents.ppo import checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo.train import train as train_ppo
from mujoco_playground import wrapper

import moplayground as mop
from moplayground import config
import minimal_mjx as mm

# mujoco_playground dm_control_suite_params.brax_ppo_config defaults
PPO_PARAMS = dict(
    num_timesteps          = 60_000_000,
    num_evals              = 10,
    reward_scaling         = 10.0,
    episode_length         = 1000,
    normalize_observations = True,
    action_repeat          = 1,
    unroll_length          = 30,
    num_minibatches        = 32,
    num_updates_per_batch  = 16,
    discounting            = 0.995,
    learning_rate          = 1e-3,
    entropy_cost           = 1e-2,
    num_envs               = 2048,
    batch_size             = 1024,
)

# Empty: use brax's default PPO network sizes
NETWORK_PARAMS = dict()

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("config_path", type=str, help="YAML config that defines the environment")
parser.add_argument("--weights", type=str, default="0.5,0.5",
                    help="Comma-separated scalarization weights, one per objective")
parser.add_argument("--name", type=str, default=None,
                    help="Run name (default: ppo-w=<weights>)")
args = parser.parse_args()

# Read in config
cfg = config.load(args.config_path)
weights = [float(w) for w in args.weights.split(',')]
if len(weights) != config.num_objectives(cfg):
    raise ValueError(
        f"--weights has {len(weights)} entries but the env has "
        f"{config.num_objectives(cfg)} objectives {config.objective_labels(cfg)}"
    )

# Create environments with a scalar reward
print('Training plain PPO on', config.env_name(cfg), 'with weights', weights)
env_kwargs = dict(
    env_name     = config.env_name(cfg),
    env_params   = config.env_params(cfg),
    backend      = config.backend(cfg, for_training=True),
    gaitlib_path = config.gaitlib_path(cfg),
)
env, _      = mop.create_environment(**env_kwargs)
eval_env, _ = mop.create_environment(**env_kwargs)
env      = mop.Multi2SingleObjective(env, weights)
eval_env = mop.Multi2SingleObjective(eval_env, weights)

# Build the run config in minimal_mjx's layout
run_cfg = config_dict.ConfigDict(dict(
    name            = args.name or f'ppo-w={args.weights}',
    save_dir        = str(config.save_dir(cfg)),
    description     = f'Plain PPO baseline, scalarized with weights {weights}',
    source_config   = args.config_path,
    env             = config.env_name(cfg),
    weights         = weights,
    env_config      = config.env_params(cfg).to_dict(),
    learning_params = dict(
        ppo_params     = PPO_PARAMS,
        network_params = NETWORK_PARAMS,
    ),
))

# Create the run directory and save the config (adds git_hash unless name is 'test')
run_dir = Path(run_cfg.save_dir) / run_cfg.name
run_dir.mkdir(parents=True, exist_ok=run_cfg.name == 'test')
if run_cfg.name != 'test':
    run_cfg.git_hash = mm.utils.config.get_commit_hash(warn=False)
config_path = run_dir / 'config.yaml'
mm.utils.config.save_config(run_cfg, config_path)

run = mm.utils.logging.initialize_wandb(
    name    = str(run_dir).replace('/', ''),
    entity  = 'njanwani-gatech',
    project = 'MO-Playground-2',
    config  = run_cfg.to_dict()
)
run.log_artifact(str(config_path), name='config')

mm.utils.setupGPU.run_setup()

# TODO: replace this block with mm.learning.train(run_cfg, env, eval_env, run)
# once minimal_mjx is fixed. minimal_mjx 0.1.5 passes mean_kernel_init_fn to
# make_ppo_networks, which brax 0.14.0 no longer accepts. This block copies
# mm.learning.train without that argument (brax already defaults to lecun_uniform).
network_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    **NETWORK_PARAMS
)
network_config = checkpoint.network_config(
    observation_size       = eval_env.observation_size,
    action_size            = eval_env.action_size,
    normalize_observations = PPO_PARAMS['normalize_observations'],
    network_factory        = network_factory,
)
times, x_data, y_data, y_dataerr = [], [], [], []
train_fn = functools.partial(
    train_ppo, **PPO_PARAMS,
    network_factory=network_factory,
    progress_fn=lambda num_steps, metrics: mm.utils.plotting.plot_progress(
        num_steps  = num_steps,
        metrics    = metrics,
        times      = times,
        x_data     = x_data,
        y_data     = y_data,
        y_dataerr  = y_dataerr,
        ppo_params = PPO_PARAMS,
        save_dir   = run_dir,
        run        = run
    ),
    policy_params_fn=functools.partial(
        mm.utils.logging.save_model,
        output_dir     = run_dir,
        run            = run,
        network_config = network_config
    ),
)
train_fn(
    environment = env,
    wrap_env_fn = wrapper.wrap_for_brax_training,
    eval_env    = eval_env,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")
