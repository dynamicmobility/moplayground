"""Roll out a trained MORLAX hypernetwork over a batch of tradeoffs and plot its Pareto front.

Usage:
    python -m scripts.morlax_fronts <save_dir>/<name>/config.yaml
    python -m scripts.morlax_fronts <save_dir>/<name>/config.yaml --all     # every checkpoint
    python -m scripts.morlax_fronts <save_dir>/<name>/config.yaml --show    # also open the plots

By default only the final checkpoint is evaluated. With ``--all``, every checkpoint is
rolled out and one front scatter is saved per checkpoint. Plots go to ``output/plots/``.
"""
import argparse
from pathlib import Path

import jax
from matplotlib import pyplot as plt

import moplayground as mop


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config_path", type=str, help="saved config.yaml of the MORLAX run to evaluate")
    parser.add_argument("--all", action="store_true", help="evaluate every checkpoint, not just the final one")
    parser.add_argument("--num_envs", type=int, default=1024, help="number of tradeoffs sampled (parallel rollouts)")
    parser.add_argument("--n_steps", type=int, default=None,
                        help="env steps per rollout (default: ppo_params.episode_length, else 500)")
    parser.add_argument("--seed", type=int, default=95, help="seed for tradeoff sampling and env resets")
    parser.add_argument("--save_results", action="store_true",
                        help="also write per-checkpoint returns to <run_dir>/obj{j}.txt")
    parser.add_argument("--out_dir", type=str, default="output/plots", help="where plots are saved")
    parser.add_argument("--show", action="store_true", help="open the plots after saving")
    return parser.parse_args()


def plot_front(returns, tradeoffs, labels, title):
    """Scatter one front (``returns``: (NUM_ENVS, num_objs)) colored by tradeoff."""
    num_objs = returns.shape[1]
    if num_objs == 3:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    elif num_objs == 2:
        fig, ax = plt.subplots()
    else:
        raise ValueError(f"Can only plot 2 or 3 objective Pareto fronts, got {num_objs}")

    mop.plot_pareto(
        ax        = ax,
        pareto    = returns,
        colors    = mop.default_coloring(tradeoffs),
        objective = labels,
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    cfg = mop.MOConfig(args.config_path)
    if cfg.algorithm != "morlax":
        raise ValueError(f"{args.config_path} is an '{cfg.algorithm}' run; this script only handles morlax")

    n_steps = args.n_steps or cfg.ppo_params.get("episode_length", 500)

    env, _ = mop.create_environment(
        env_name   = cfg.env_name,
        env_params = cfg.env_params,
        backend    = "jnp",
    )

    rewards_over_iters, tradeoffs_over_iters = mop.get_morlax_fronts(
        run_dirs       = cfg.run_dir,
        network_params = cfg.morlax_network_params,
        num_objectives = cfg.num_objectives,
        rng            = jax.random.PRNGKey(args.seed),
        env            = env,
        N_STEPS        = n_steps,
        NUM_ENVS       = args.num_envs,
        save_results   = args.save_results,
        only_final     = not args.all,
    )

    # label each front by its checkpoint's training step (checkpoint dir names are step counts)
    checkpoints = mop.learning.inference.get_all_models(cfg.run_dir)
    if not args.all:
        checkpoints = checkpoints[-1:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = Path(cfg.run_dir).name

    for ckpt, returns, tradeoffs in zip(checkpoints, rewards_over_iters, tradeoffs_over_iters):
        step = int(ckpt.name)
        fig = plot_front(returns, tradeoffs, cfg.objective_labels, f"{run_name} - step {step:,}")
        save_path = out_dir / f"{cfg.env_name}_{run_name}_front_{step}.pdf"
        fig.savefig(save_path)
        print(f"Saved {save_path}")
        if not args.show:
            plt.close(fig)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
