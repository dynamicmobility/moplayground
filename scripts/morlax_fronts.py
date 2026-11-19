"""Roll out a trained MORLAX hypernetwork over a batch of tradeoffs and plot its Pareto front.

Usage:
    python -m scripts.morlax_fronts <save_dir>/<name>/config.yaml
    python -m scripts.morlax_fronts <save_dir>/<name>/config.yaml --all     # every checkpoint
    python -m scripts.morlax_fronts <save_dir>/<name>/config.yaml --show    # also open the plots

By default only the final checkpoint is evaluated. With ``--all``, every checkpoint is
rolled out and one front scatter is saved per checkpoint. Plots go to ``output/plots/``.

The final front (interactive Plotly) and its hypervolume/sparsity are logged to the run's
wandb run, found via ``<run_dir>/wandb_run_path.txt`` (written by ``scripts.train``) or
``--wandb_path``. Pass ``--no_wandb`` to skip.
"""
import argparse
from pathlib import Path

import jax
import wandb
from matplotlib import pyplot as plt

import moplayground as mop

RUN_PATH_FILE = 'wandb_run_path.txt'


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
    parser.add_argument("--wandb_path", type=str, default=None,
                        help=f"entity/project/run_id to log to (default: read from <run_dir>/{RUN_PATH_FILE})")
    parser.add_argument("--no_wandb", action="store_true", help="do not log to wandb")
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


def log_final_front(run, returns, tradeoffs, labels, title):
    """Log the final front as a Plotly figure plus its hypervolume and sparsity."""
    hv, sp = mop.get_pareto_statistics(returns)
    print(f"Final hypervolume: {hv:.4e}  sparsity: {sp:.4e}")
    if run is None:
        return
    fig = mop.plot_pareto_plotly(returns, tradeoffs, labels)
    fig.update_layout(title=title)
    run.log({
        "pareto/final_front":       wandb.Plotly(fig),
        "pareto/final_hypervolume": hv,
        "pareto/final_sparsity":    sp,
    })
    run.summary["pareto/final_hypervolume"] = hv
    run.summary["pareto/final_sparsity"] = sp


def resume_run(run_dir, wandb_path=None):
    """Resume the run's wandb run, or return ``None`` if its path is unknown."""
    path_file = Path(run_dir) / RUN_PATH_FILE
    if wandb_path is None and path_file.exists():
        wandb_path = path_file.read_text().strip()
    if wandb_path is None:
        print(f"No {path_file} and no --wandb_path; skipping wandb logging")
        return None
    entity, project, run_id = wandb_path.split("/")
    return wandb.init(entity=entity, project=project, id=run_id, resume="must")


def main(
    config_path,
    all          : bool = False,
    num_envs     : int = 1024,
    n_steps      : int = None,
    seed         : int = 95,
    save_results : bool = False,
    out_dir      : str = "output/plots",
    show         : bool = False,
    run          : wandb.Run | None = None,
):
    cfg = mop.MOConfig(config_path)
    if cfg.algorithm != "morlax":
        raise ValueError(f"{config_path} is an '{cfg.algorithm}' run; this script only handles morlax")

    n_steps = n_steps or cfg.ppo_params.get("episode_length", 500)

    env, _ = mop.create_environment(
        env_name   = cfg.env_name,
        env_params = cfg.env_params,
        backend    = "jnp",
    )

    rewards_over_iters, tradeoffs_over_iters = mop.get_morlax_fronts(
        run_dirs       = cfg.run_dir,
        network_params = cfg.morlax_network_params,
        num_objectives = cfg.num_objectives,
        rng            = jax.random.PRNGKey(seed),
        env            = env,
        N_STEPS        = n_steps,
        NUM_ENVS       = num_envs,
        save_results   = save_results,
        only_final     = not all,
    )

    # label each front by its checkpoint's training step (checkpoint dir names are step counts)
    checkpoints = mop.learning.inference.get_all_models(cfg.run_dir)
    if not all:
        checkpoints = checkpoints[-1:]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = Path(cfg.run_dir).name

    for ckpt, returns, tradeoffs in zip(checkpoints, rewards_over_iters, tradeoffs_over_iters):
        step = int(ckpt.name)
        fig = plot_front(returns, tradeoffs, cfg.objective_labels, f"{run_name} - step {step:,}")
        save_path = out_dir / f"{cfg.env_name}_{run_name}_front_{step}.pdf"
        fig.savefig(save_path)
        print(f"Saved {save_path}")
        if not show:
            plt.close(fig)

    final_step = int(checkpoints[-1].name)
    log_final_front(
        run, rewards_over_iters[-1], tradeoffs_over_iters[-1], cfg.objective_labels,
        f"{run_name} - step {final_step:,}",
    )

    if show:
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    cfg = mop.MOConfig(args.config_path)
    run = None if args.no_wandb else resume_run(cfg.run_dir, args.wandb_path)
    try:
        main(
            config_path  = args.config_path,
            all          = args.all,
            num_envs     = args.num_envs,
            n_steps      = args.n_steps,
            seed         = args.seed,
            save_results = args.save_results,
            out_dir      = args.out_dir,
            show         = args.show,
            run          = run,
        )
    finally:
        if run is not None:
            run.finish()
